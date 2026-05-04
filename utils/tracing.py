# /utils/tracing.py

import os
from typing import Optional, Dict, Any, List
from datetime import datetime


class LangfuseTracer:
    """
    Integración con Langfuse para tracking y observabilidad del workflow.
    Proporciona: tracing completo, logging de decisiones, scoring de calidad.
    """

    def __init__(self, public_key: str = None, secret_key: str = None):
        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.host = os.getenv("LANGFUSE_HOST") or os.getenv(
            "LANGFUSE_BASE_URL", "https://cloud.langfuse.com"
        )
        self.client = None
        self._trace_id = None
        self._current_span = None

    def is_enabled(self) -> bool:
        return bool(self.public_key and self.secret_key)

    def setup(self):
        if not self.is_enabled():
            return False
        try:
            from langfuse import Langfuse

            self.client = Langfuse(
                public_key=self.public_key, secret_key=self.secret_key, host=self.host
            )
            return True
        except Exception as e:
            print(f"Warning: Error al conectar con Langfuse: {e}")
            return False

    def start_trace(self, query: str, metadata: Dict = None) -> str:
        if not self.client:
            return None
        try:
            span = self.client.start_observation(
                name="support_query",
                input={"query": query},
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    **(metadata or {}),
                },
            )
            self._trace_id = span.trace_id
            self._current_span = span
            return self._trace_id
        except Exception as e:
            print(f"Error al crear trace: {e}")
            return None

    def log_classification(self, categories: List[str], segments: List[Dict]):
        if not self._current_span:
            return
        try:
            self._current_span.start_observation(
                name="intent_classification",
                metadata={
                    "categories": categories,
                    "segments": segments,
                },
            )
        except Exception as e:
            print(f"Error logging classification: {e}")

    def log_routing_decision(self, from_node: str, to_node: str, reason: str = None):
        if not self._current_span:
            return
        try:
            self._current_span.start_observation(
                name="routing_decision",
                metadata={
                    "from": from_node,
                    "to": to_node,
                    "reason": reason,
                },
            )
        except Exception as e:
            print(f"Error logging routing: {e}")

    def score_response(
        self,
        agent: str,
        response: str,
        relevance: float = None,
        completeness: float = None,
        accuracy: float = None,
    ):
        if not self._current_span:
            return
        try:
            span = self._current_span.start_observation(
                name=f"agent_{agent}",
                metadata={
                    "response_length": len(response),
                },
            )
            if relevance is not None:
                span.score(name="relevance", value=relevance)
            if completeness is not None:
                span.score(name="completeness", value=completeness)
            if accuracy is not None:
                span.score(name="accuracy", value=accuracy)
            if (
                relevance is not None
                and completeness is not None
                and accuracy is not None
            ):
                overall = (relevance + completeness + accuracy) / 3
                span.score(name="overall", value=overall)
        except Exception as e:
            print(f"Error logging score: {e}")

    def log_generation(self, step_name: str, input_data: Any, output_data: Any):
        if not self._current_span:
            return
        try:
            gen_span = self._current_span.start_observation(
                name=step_name,
                input={"content": str(input_data)[:500]},
                output={"content": str(output_data)[:500]},
            )
            gen_span.end()
        except Exception as e:
            print(f"Error logging generation: {e}")

    def end_trace(self, final_response: str = None, success: bool = True):
        if not self._current_span:
            return
        try:
            self._current_span.update(
                metadata={
                    "final_response_length": len(final_response)
                    if final_response
                    else 0,
                    "success": success,
                    "completed_at": datetime.now().isoformat(),
                }
            )
            self._current_span.end()
            self._current_span = None
        except Exception as e:
            print(f"Error ending trace: {e}")


class ResponseEvaluator:
    """Evaluador automatizado de respuestas."""

    def __init__(self, llm=None):
        self.llm = llm

    def evaluate(self, query: str, context: str, response: str) -> Dict[str, float]:
        if not self.llm:
            return {
                "relevance": 0.5,
                "completeness": 0.5,
                "accuracy": 0.5,
                "reason": "No LLM disponible",
            }

        evaluation_prompt = f"""EVALÚA la siguiente respuesta del asistente.

CONSULTA: {query}

RESPUESTA: {response[:500]}...

Evalúa (responde solo con números 0-1):
relevance: X.X
completeness: X.X
accuracy: X.X
reason: breve
"""
        try:
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "Eres un evaluador de calidad."),
                    ("human", evaluation_prompt),
                ]
            )
            chain = prompt | self.llm
            result = chain.invoke({})

            scores = {
                "relevance": 0.5,
                "completeness": 0.5,
                "accuracy": 0.5,
                "reason": "",
            }
            for line in result.content.lower().split("\n"):
                if "relevance" in line:
                    try:
                        scores["relevance"] = float(line.split(":")[1].strip())
                    except:
                        pass
                elif "completeness" in line:
                    try:
                        scores["completeness"] = float(line.split(":")[1].strip())
                    except:
                        pass
                elif "accuracy" in line:
                    try:
                        scores["accuracy"] = float(line.split(":")[1].strip())
                    except:
                        pass
                elif "reason" in line:
                    scores["reason"] = line.split(":")[1].strip() if ":" in line else ""
            return scores
        except Exception as e:
            return {
                "relevance": 0.5,
                "completeness": 0.5,
                "accuracy": 0.5,
                "reason": f"Error: {str(e)[:50]}",
            }

    def evaluate_all_agents(
        self, query: str, agent_responses: Dict[str, str], contexts: Dict[str, str]
    ) -> Dict[str, Dict]:
        results = {}
        for agent, response in agent_responses.items():
            scores = self.evaluate(query, contexts.get(agent, ""), response)
            results[agent] = scores
            results[agent]["overall"] = (
                sum([scores["relevance"], scores["completeness"], scores["accuracy"]])
                / 3
            )
        return results

    def get_quality_report(self, evaluations: Dict[str, Dict]) -> str:
        lines = ["=== REPORTE DE CALIDAD DE RESPUESTAS ===\n"]
        for agent, scores in evaluations.items():
            overall = scores.get("overall", 0)
            status = (
                "[OK] BUENA"
                if overall >= 0.7
                else "[!] REGULAR"
                if overall >= 0.5
                else "[X] MALA"
            )
            lines.append(f"Agente: {agent}")
            lines.append(f"  Overall: {overall:.2f} {status}")
            lines.append(f"  Relevancia: {scores['relevance']:.2f}")
            lines.append(f"  Completitud: {scores['completeness']:.2f}")
            lines.append(f"  Exactitud: {scores['accuracy']:.2f}")
            if scores.get("reason"):
                lines.append(f"  Nota: {scores['reason'][:100]}")
            lines.append("")
        return "\n".join(lines)


_tracer = None


def get_tracer() -> LangfuseTracer:
    global _tracer
    if _tracer is None:
        _tracer = LangfuseTracer()
        _tracer.setup()
    return _tracer


def is_tracing_enabled() -> bool:
    tracer = get_tracer()
    return tracer.is_enabled()


if __name__ == "__main__":
    print("=== LANGFUSE CONFIG ===")
    tracer = LangfuseTracer()
    print(f"Habilitado: {tracer.is_enabled()}")
    print("""
Para habilitar: export LANGFUSE_PUBLIC_KEY=pk-...
              export LANGFUSE_SECRET_KEY=sk-...
""")
