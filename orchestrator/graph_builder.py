# /orchestrator/graph_builder.py

from langgraph.graph import StateGraph, END
from typing import List, Dict, Any, TypedDict

from .router import Router


class GraphState(TypedDict):
    query: str
    classification: Dict[str, Any]
    segments: List[Dict]
    agent_results: List[Dict]
    final_response: str


class SupportGraphBuilder:
    """
    Construye y gestiona el grafo de soporte multi-agente usando LangGraph.
    """

    def __init__(self, llm, rrh_agent=None, it_agent=None, finance_agent=None):
        self.llm = llm
        self.router = Router(llm)
        self.rrh_agent = rrh_agent
        self.it_agent = it_agent
        self.finance_agent = finance_agent

    def build_graph(self) -> StateGraph:
        """
        Define la estructura del grafo (Nodos y Aristas).
        """
        workflow = StateGraph(GraphState)

        # --- 1. Definición de Nodos ---

        def route_and_classify(state: Dict[str, Any]) -> Dict[str, Any]:
            """Nodo para clasificar la consulta y determinar el ruteo."""
            query = state["query"]
            classification = self.router.classify_query(query)
            return {
                "classification": classification,
                "segments": classification.get("segments", []),
                "agent_results": [],
            }

        def delegate_to_agents(state: Dict[str, Any]) -> Dict[str, Any]:
            """Nodo para delegar la consulta a los agentes especializados."""
            query = state["query"]
            classification = state["classification"]
            categories = classification.get("categories", [])
            segments = state.get("segments", [])

            agent_results = []

            # --- Lógica de ruteo real con agentes conectados ---
            # Agente de RRH
            if "RRH" in categories:
                if self.rrh_agent is not None:
                    segment_query = _extract_segment(query, segments, ["RRH"])
                    result = self.rrh_agent.answer_query(segment_query)
                    agent_results.append({"agent": "RRH", "result": result})
                else:
                    agent_results.append(
                        {"agent": "RRH", "result": "Error: Agente RRHH no configurado."}
                    )

            # Agente de Finanzas
            if "FINANCE" in categories:
                if self.finance_agent is not None:
                    segment_query = _extract_segment(query, segments, ["FINANCE"])
                    result = self.finance_agent.answer_query(segment_query)
                    agent_results.append({"agent": "FINANCE", "result": result})
                else:
                    agent_results.append(
                        {
                            "agent": "FINANCE",
                            "result": "Error: Agente Finance no configurado.",
                        }
                    )

            # Agente de IT
            if "IT" in categories or "FACILITIES" in categories:
                if self.it_agent is not None:
                    segment_query = _extract_segment(
                        query, segments, ["IT", "FACILITIES"]
                    )
                    result = self.it_agent.answer_query(segment_query)
                    agent_results.append({"agent": "IT", "result": result})
                else:
                    agent_results.append(
                        {"agent": "IT", "result": "Error: Agente IT no configurado."}
                    )

            if not categories or ("GENERAL" in categories and len(categories) == 1):
                if self.it_agent is not None:
                    result = self.it_agent.answer_query(query)
                    agent_results.append({"agent": "IT", "result": result})
                elif self.rrh_agent is not None:
                    result = self.rrh_agent.answer_query(query)
                    agent_results.append({"agent": "RRH", "result": result})

            return {"agent_results": agent_results}

        def consolidate_response(state: Dict[str, Any]) -> Dict[str, str]:
            """Nodo para reunir todas las respuestas en un formato legible."""
            all_results = state["agent_results"]

            if not all_results:
                return {
                    "final_response": "No se pudo procesar la consulta. Por favor, revisa el sistema."
                }

            formatted_responses = []
            for res in all_results:
                agent_name = res["agent"].upper()
                formatted_responses.append(
                    f"**Respuesta de {agent_name}:**\n{res['result']}"
                )

            return {"final_response": "\n\n".join(formatted_responses)}

        # --- 2. Definición de Aristas ---
        workflow.add_node("router", route_and_classify)
        workflow.add_node("delegate", delegate_to_agents)
        workflow.add_node("consolidate", consolidate_response)

        workflow.set_entry_point("router")
        workflow.add_edge("router", "delegate")
        workflow.add_edge("delegate", "consolidate")
        workflow.add_edge("consolidate", END)

        return workflow.compile()


def _extract_segment(
    query: str, segments: List[Dict], target_categories: List[str]
) -> str:
    """
    Extrae el segmento relevante de una consulta multi-tópico.
    Si no hay segmentos útiles, retorna la consulta original.
    """
    if not segments:
        return query

    for seg in segments:
        topic = seg.get("topic", "").upper()
        details = seg.get("details", "")

        for cat in target_categories:
            if cat.upper() in topic:
                # Si los detalles son genéricos, usar la query original
                if not details or "clasificada" in details.lower():
                    return query
                return details

    return query
