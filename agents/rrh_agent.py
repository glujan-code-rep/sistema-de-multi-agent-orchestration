# /agents/rrh_agent.py

import os
from langchain_openai import ChatOpenAI
from rag_utils.retriever import VectorRetriever  # Importamos la clase de RAG
from langchain_core.prompts import ChatPromptTemplate


class RRHAgent:
    """
    Agente Especialista para Recursos Humanos (RR. HH.).
    Utiliza RAG sobre la base vectorial de políticas internas.
    """

    def __init__(
        self,
        llm_model: str = "mistral-7b-instruct-v0.2-q4_K_M",
        base_url: str = None,
        llm=None,
    ):
        # 1. Usar LLM proporcionado o crear uno nuevo
        if llm is not None:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(
                model=llm_model,
                base_url=base_url
                or os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1"),
                temperature=0.7,
            )

        # 2. Inicializar el Retriever (Conexión a la base vectorial de RR. HH.)
        self.retriever = VectorRetriever(db_path="./knowledge_bases/rrh_vector_db")

        # 3. Definir el Prompt de Respuesta con Few-Shot Learning
        self.examples = [
            {
                "question": "¿Cuántos días de vacaciones me corresponden?",
                "context": "## 1. Vacaciones Anuales\n### 1.1 Días de vacaciones por año:\n- 1 a 2 años de servicio: 15 días hábiles\n- 3 a 5 años de servicio: 20 días hábiles\n- 6 o más años de servicio: 25 días hábiles",
                "answer": "Según las políticas de la empresa, tienes derecho a 15 días hábiles de vacaciones anuales si tienes entre 1 y 2 años de servicio. Este número aumenta a 20 días si tienes entre 3 y 5 años, y a 25 días si tienes 6 años o más.",
            },
            {
                "question": "¿Puedo trabajar desde casa?",
                "context": "### 4.2 Trabajo remoto:\n- Hasta 2 días por semana con aprobación del gerente directo.\n- Debe reportarse el horario de inicio y fin en la plataforma interna.\n- El primer día remoto debe ser acordado semanalmente antes de las 8:00 AM.",
                "answer": "Sí, puedes trabajar desde casa hasta 2 días por semana, siempre que tu gerente directo lo apruebe. Recuerda reportar tu horario en la plataforma interna y acordar el primer día remoto antes de las 8:00 AM.",
            },
            {
                "question": "¿Cuánto tiempo de licencia por paternidad tengo?",
                "context": "### 2.2 Licencia por paternidad:\n- 2 semanas consecutivas con goce de sueldo para padres biológicos y adoptivos.",
                "answer": "Tienes derecho a 2 semanas consecutivas de licencia por paternidad con goce de sueldo, tanto para padres biológicos como adoptivos.",
            },
        ]

        # Crear el template con few-shot examples
        examples_text = "\n\n".join(
            [
                f"Ejemplo {i + 1}:\nPregunta: {ex['question']}\nContexto relevante: {ex['context'][:200]}...\nRespuesta: {ex['answer']}"
                for i, ex in enumerate(self.examples)
            ]
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""Eres un Asistente experto en Recursos Humanos de la empresa.
Tu objetivo es responder preguntas sobre políticas, licencias, beneficios y compensación de empleados.

INSTRUCCIONES:
- Usa EXCLUSIVAMENTE la información del contexto proporcionado
- Si el contexto no tiene la respuesta, indica que no tienes esa información específica
- Sé conciso y directo en tus respuestas
- No inventes información
- Responde siempre en español

FORMATO DE RESPUESTA:
- Usa viñetas (-) para listar información
- Destaca términos importantes en negrita (**)
- Si es necesario, indica pasos a seguir
- Si una pregunta no es de tu área, redirige al usuario al departamento correcto (ej. RRHH, IT, etc.)""",
                ),
                (
                    "human",
                    f"""Los siguientes son ejemplos de cómo debes responder:

{examples_text}

---

Ahora responde esta nueva consulta:

Contexto disponible:\n{{context}}

Pregunta del usuario: {{question}}""",
                ),
            ]
        )

    def answer_query(self, question: str) -> str:
        """
        Función principal para procesar una consulta y generar una respuesta RAG.
        """
        try:
            # Paso 1: Recuperación (RAG)
            context = self.retriever.retrieve(question)

            # Paso 2: Generación de la Respuesta con el Contexto
            final_prompt = self.prompt.format(context=context, question=question)

            # Paso 3: Invocación del LLM
            response = self.llm.invoke(final_prompt)

            return response.content

        except Exception as e:
            print(f"Error durante el procesamiento del Agente RR. HH.: {e}")
            # En caso de error, devolvemos un mensaje seguro en lugar de fallar completamente
            return "Lo siento, hubo un error interno al procesar tu solicitud. Por favor, inténtalo de nuevo más tarde."


# ==============================================================================
# Ejemplo de cómo se usaría este agente (Para pruebas)
# ==============================================================================
if __name__ == "__main__":
    print("--- INICIANDO PRUEBA DEL AGENTE RR. HH. ---")

    # Inicializar el agente
    rrh_agent = RRHAgent()

    # Ejemplo de consulta (Esta consulta no encontrará resultados si los archivos JSON están vacíos)
    test_question = "Cuáles son las políticas de licencia para empleados nuevos?"

    print(f"\nPregunta del Usuario: {test_question}")

    # Ejecutar la respuesta RAG
    response = rrh_agent.answer_query(test_question)

    print("\nRespuesta del Agente RR. HH.:")
    print("------------------------------------------")
    print(response)
    print("------------------------------------------")
