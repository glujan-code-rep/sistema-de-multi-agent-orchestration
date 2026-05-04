# main.py

import os
import json
from dotenv import load_dotenv

# Importar componentes del sistema
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from agents.rrh_agent import RRHAgent
from agents.it_agent import ITAgent
from agents.finance_agent import FinanceAgent
from orchestrator.graph_builder import SupportGraphBuilder
from rag_utils.retriever import VectorRetriever
from rag_utils.embedder import build_all
from utils.content_moderator import ContentModerator
from utils.security import get_security_manager
from utils.llm_providers import (
    get_llm_provider,
    check_provider_available,
    ProviderFactory,
)
from utils.tracing import get_tracer, is_tracing_enabled, ResponseEvaluator

# Cargar variables de entorno (para la clave de API)
load_dotenv()

# Asegurar que haya una API key para LM Studio (no se usa realmente)
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "sk-dummy"

LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1")


def get_available_models(lm_studio_url: str) -> list:
    """Obtiene la lista de modelos disponibles en LM Studio."""
    import urllib.request

    try:
        req = urllib.request.Request(f"{lm_studio_url}/models")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            return [model["id"] for model in data.get("data", [])]
    except Exception as e:
        print(f"Warning: No se pudo obtener modelos de LM Studio: {e}")
        return []


def get_loaded_model(lm_studio_url: str) -> str:
    """Obtiene el modelo actualmente cargado en LM Studio."""
    import urllib.request

    try:
        req = urllib.request.Request(f"{lm_studio_url}/model")
        with urllib.request.urlopen(req, timeout=3) as response:
            data = json.loads(response.read().decode())
            if data.get("data", {}).get("id"):
                return data["data"]["id"]
    except Exception:
        pass
    return None


def select_best_model(available_models: list, preferred_model: str = None) -> str:
    """Selecciona el mejor modelo disponible."""
    if not available_models:
        return "gemma-4-26b-a4b-it"

    if preferred_model and preferred_model in available_models:
        return preferred_model

    chat_models = [m for m in available_models if "embedding" not in m.lower()]
    if not chat_models:
        chat_models = available_models

    priority_order = [
        "qwen3",
        "gemma-4-26b",
        "llama-3.2-3b",
        "gemma-4-e4b",
        "glm-4",
        "llama-3.2-1b",
        "qwen2.5-0.5",
        "tinyllama",
    ]

    for keyword in priority_order:
        for model in chat_models:
            if keyword.lower() in model.lower():
                return model

    return chat_models[0]


def get_available_models(lm_studio_url: str) -> list:
    """Obtiene la lista de modelos disponibles en LM Studio."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(f"{lm_studio_url}/models")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            return [model["id"] for model in data.get("data", [])]
    except Exception as e:
        print(f"Warning: No se pudo obtener modelos de LM Studio: {e}")
        return []


def get_loaded_model(lm_studio_url: str) -> str:
    """Obtiene el modelo actualmente cargado en LM Studio."""
    import urllib.request

    try:
        # Verificar si hay un modelo cargado
        req = urllib.request.Request(f"{lm_studio_url}/model")
        with urllib.request.urlopen(req, timeout=3) as response:
            data = json.loads(response.read().decode())
            if data.get("data", {}).get("id"):
                return data["data"]["id"]
    except Exception:
        pass
    return None


def select_best_model(available_models: list, preferred_model: str = None) -> str:
    """
    Selecciona el mejor modelo disponible.
    Si se especifica un modelo preferido y está disponible, lo usa.
    Si no, selecciona el primero disponible.
    """
    if not available_models:
        return "llama-3.2-1b-instruct"

    # Si hay un modelo preferido y está disponible, usarlo
    if preferred_model and preferred_model in available_models:
        return preferred_model

    # Filtrar solo modelos de chat (excluir embeddings)
    chat_models = [m for m in available_models if "embedding" not in m.lower()]

    if not chat_models:
        chat_models = available_models

    # Seleccionar modelo preferiendo modelos más grandes y conocidos
    # Prioridad: qwen3 > gemma-4-26b > llama-3.2-3b > llama-3.2-1b > qwen2.5 > gemma > tinyllama
    priority_order = [
        "qwen3",  # Mejor calidad
        "gemma-4-26b",  # Muy grande
        "llama-3.2-3b",  # 3B params
        "gemma-4-e4b",  # 4B params
        "glm-4",  # GLM
        "llama-3.2-1b",  # 1B params (pequeño)
        "qwen2.5-0.5",  # Muy pequeño
        "tinyllama",  # Muy pequeño
    ]

    for keyword in priority_order:
        for model in chat_models:
            if keyword.lower() in model.lower():
                return model

    # Si no hay ninguno con prioridad, devolver el primero
    return chat_models[0]


def main():
    """
    Función principal que inicializa el sistema multi-agente y ejecuta un flujo de prueba.
    """
    # Determinar el proveedor a usar
    provider_name = os.getenv("LLM_PROVIDER", "lmstudio").lower()

    print("==============================================================")
    print(
        f"INICIANDO SISTEMA DE SOPORTE INTERNO AUTOMATIZADO ({provider_name.upper()})"
    )
    print("==============================================================")

    # Verificar si el proveedor está disponible
    if not check_provider_available(provider_name):
        print(f"Warning: Proveedor '{provider_name}' no está disponible.")

        # Intentar con LM Studio como fallback
        if provider_name != "lmstudio" and check_provider_available("lmstudio"):
            print("Usando LM Studio como fallback...")
            provider_name = "lmstudio"
        else:
            print("ERROR: No hay proveedores disponibles.")
            print("Configura OPENAI_API_KEY o GOOGLE_API_KEY, o inicia LM Studio.")
            return

    # --- 0. Construir bases vectoriales si no existen ---
    print("\nVerificando bases vectoriales...")
    _build_knowledge_bases_if_needed()

    # --- 1. Inicialización del LLM ---
    try:
        print(f"\nInicializando proveedor: {provider_name}")

        # Para LM Studio, usar modelo local
        if provider_name == "lmstudio":
            loaded_model = get_loaded_model(LM_STUDIO_URL)
            if loaded_model:
                llm_model = loaded_model
                print(f"Modelo LM Studio cargado: {llm_model}")
            else:
                available = get_available_models(LM_STUDIO_URL)
                if available:
                    for m in available:
                        if "gemma-4-26b" in m.lower():
                            llm_model = m
                            break
                    else:
                        for m in available:
                            if "llama" in m.lower():
                                llm_model = m
                                break
                        else:
                            llm_model = select_best_model(
                                available, os.getenv("LLM_MODEL")
                            )
                    print(f"Modelo seleccionado: {llm_model}")
                else:
                    llm_model = os.getenv("LM_STUDIO_MODEL", "gemma-4-26b-a4b-it")

            provider = get_llm_provider(
                "lmstudio", model=llm_model, base_url=LM_STUDIO_URL
            )
        else:
            # Para OpenAI o Gemini
            provider = get_llm_provider(provider_name)
            llm_model = provider.get_model_name()

        llm = provider.get_client()
        print(f"LLM inicializado: {llm_model}")

    except Exception as e:
        print(f"ERROR FATAL al inicializar el LLM: {e}")
        return

    # --- 2. Inicialización de Agentes Especialistas (RAG) ---
    try:
        print("\nInicializando Agentes Especialistas...")
        # Pasar el LLM directamente a los agentes
        rrh_agent = RRHAgent(llm=llm)
        it_agent = ITAgent(llm=llm)
        finance_agent = FinanceAgent(llm=llm)
        print("Agentes RR. HH., IT y Finance inicializados con RAG.")

    except Exception as e:
        print(f"ERROR al inicializar los agentes: {e}")
        return

    # --- 3. Inicialización del Orquestador (LangGraph) ---
    try:
        print("\nInicializando el Orquestador LangGraph...")
        graph_builder = SupportGraphBuilder(
            llm=llm, rrh_agent=rrh_agent, it_agent=it_agent, finance_agent=finance_agent
        )
        workflow = graph_builder.build_graph()
        print("Grafo de soporte construido exitosamente.")

    except Exception as e:
        print(f"ERROR al construir el grafo: {e}")
        return

    # --- 4. Moderación de contenido y seguridad ---
    print("\nVerificando seguridad del contenido...")
    moderator = ContentModerator(strict_mode=True)
    security_manager = get_security_manager()

    # test_query = "El monto de mi sueldo parece incorrecto este mes, y además mi laptop de la empresa está muy lenta."
    test_query = "quiero informacion sobre el trabajo remoto, y además No puedo acceder a mi correo electrónico corporativo."

    # Verificar contenido ofensivo
    is_safe_content, safe_response, _ = moderator.check_content(test_query)

    if not is_safe_content:
        print("\n==============================================================")
        print("CONTENIDO BLOQUEADO POR MODERACIÓN")
        print("==============================================================")
        print(f"Respuesta segura: {safe_response}")
        return

    # Verificar inyección de prompt y otros ataques
    is_safe_input, safe_input_response = security_manager.check(test_query)

    if not is_safe_input:
        print("\n==============================================================")
        print("CONTENIDO BLOQUEADO POR SEGURIDAD")
        print("==============================================================")
        print(f"Respuesta segura: {safe_input_response}")
        return

    print(f"Contenido verificado: [OK] Seguro")

    # --- 5. Inicializar tracing ---
    tracer = get_tracer()
    trace_enabled = is_tracing_enabled()

    if trace_enabled:
        print("\n[Langfuse] Tracing habilitado")
        tracer.start_trace(test_query, {"provider": provider_name})
    else:
        print("\n[Langfuse] Tracing no configurado (opcional)")
        print("   Para habilitar: export LANGFUSE_PUBLIC_KEY=pk-...")
        print("                  export LANGFUSE_SECRET_KEY=sk-...")

    # --- 5. Ejecución del Flujo de Prueba (Simulación) ---
    print("\n==============================================================")
    print(f"CONSULTA DE PRUEBA: {test_query}")
    print("==============================================================")

    # Iniciar timer para métricas
    import time

    start_time = time.time()

    final_result = workflow.invoke({"query": test_query})
    total_time = (time.time() - start_time) * 1000  # ms

    print("\n==============================================================")
    print("RESULTADO FINAL DEL SISTEMA:")
    print("==============================================================")
    print(final_result)

    # --- 6. Tracing y Evaluación ---
    if trace_enabled:
        # Log de la clasificación con input/output
        classification = final_result.get("classification", {})
        tracer.log_generation("intent_classification", test_query, classification)
        tracer.log_classification(
            classification.get("categories", []), classification.get("segments", [])
        )

        # Log de resultados de agentes
        for agent_result in final_result.get("agent_results", []):
            agent = agent_result.get("agent", "unknown")
            agent_result_text = agent_result.get("result", "")
            tracer.log_generation(f"agent_{agent}", test_query, agent_result_text)
            tracer.log_routing_decision(
                "delegate", agent, f"Categoría {agent} detectada"
            )

        # Evaluación de calidad de respuestas
        print("\n--- Evaluando calidad de respuestas ---")
        evaluator = ResponseEvaluator(llm=llm)

        agent_responses = {
            r["agent"]: r["result"] for r in final_result.get("agent_results", [])
        }

        # Aquí no tenemos acceso directo a los contextos usados
        # Pero podemos evaluar la calidad de las respuestas generadas
        quality_scores = {}

        for agent, response in agent_responses.items():
            scores = evaluator.evaluate(test_query, "", response)
            scores["overall"] = (
                scores["relevance"] + scores["completeness"] + scores["accuracy"]
            ) / 3
            quality_scores[agent] = scores

            # Log del scoring
            tracer.score_response(
                agent,
                response,
                relevance=scores.get("relevance"),
                completeness=scores.get("completeness"),
                accuracy=scores.get("accuracy"),
            )

        # Generar reporte
        report = evaluator.get_quality_report(quality_scores)
        print(report)

        # Finalizar trace
        tracer.end_trace(str(final_result.get("final_response", "")), success=True)


def _build_knowledge_bases_if_needed():
    """Construye las bases vectoriales si el directorio knowledge_bases/ está vacío."""
    import os as _os

    kb_dir = "./knowledge_bases"
    if not _os.path.exists(kb_dir):
        print("Construyendo bases vectoriales desde data/...")
        build_all()
    else:
        # Verificar si hay subdirectorios con datos
        has_data = False
        for entry in _os.listdir(kb_dir):
            full_path = _os.path.join(kb_dir, entry)
            if _os.path.isdir(full_path):
                files = [f for f in _os.listdir(full_path)]
                if files:
                    has_data = True
                    break
        if not has_data:
            print("Bases vectoriales vacías. Construyendo desde data/...")
            build_all()


if __name__ == "__main__":
    main()
