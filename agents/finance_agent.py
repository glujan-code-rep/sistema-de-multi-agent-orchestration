# /agents/finance_agent.py

import os
from langchain_openai import ChatOpenAI
from rag_utils.retriever import VectorRetriever
from langchain_core.prompts import ChatPromptTemplate


class FinanceAgent:
    """
    Agente Especialista para Finanzas y Contabilidad.
    Utiliza RAG sobre la base vectorial de políticas financieras.
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

        # 2. Inicializar el Retriever (Conexión a la base vectorial de Finanzas)
        self.retriever = VectorRetriever(db_path="./knowledge_bases/finance_vector_db")

        # 3. Definir el Prompt de Respuesta con Few-Shot Learning
        self.examples = [
            {
                "question": "¿Cómo solicito un reembolso de gastos?",
                "context": "## Reembolso de gastos\n1. Los gastos deben быть dentro del mes\n2. Adjuntar comprobantes fiscales\n3. Llenar formato de reembolso\n4. Aprobación del gerente\n5. Tiempo de procesamiento: 5-10 días hábiles",
                "answer": "Para solicitar un reembolso de gastos:\n\n1. **Reúne los comprobantes fiscales** - Deben ser del mes en curso\n2. **Llena el formato de reembolso** - Disponible en el portal de Finanzas\n3. **Adjunta los comprobantes** - Asegúrate que sean fiscales (CFDI/CDN)\n4. **Envía a aprobación de tu gerente**\n5. **Tiempo de procesamiento**: 5-10 días hábiles\n\nNota: Los gastos de representación tienen límite de $500 por evento.",
            },
            {
                "question": "¿Cuál es la política de viáticos?",
                "context": "## Viáticos\n- Viático diario: $500 MXN dentro del país, $100 USD fuera\n- Hospedaje: Hasta $1500 MXN por noche\n- Transportación: Transporte público o taxi (no Uber/Lyft)\n- Comidas: $300 MXN por día\n- Pre aprobación requerida para viajes",
                "answer": "La política de viáticos es:\n\n- **Viático diario**: $500 MXN (dentro del país), $100 USD (fuera)\n- **Hospedaje**: Hasta $1,500 MXN por noche\n- **Transportación**: Transporte público o taxi convencional (no Uber/Lyft)\n- **Comidas**: $300 MXN por día\n\nIMPORTANTE: Se requiere pre-aprobación del gerente antes del viaje.",
            },
            {
                "question": "¿Cuándo me pagan?",
                "context": "## Fechas de pago\n- Pago quincenal: 15 y último día del mes\n- Depósito directo a cuenta registrada\n- Si es festivo, se adelantar el viernes\n- Estados de cuenta disponibles en portal",
                "answer": "Las fechas de pago son:\n\n- **Pago quincenal**: Los días 15 y el último día del mes\n- **Método**: Depósito directo a tu cuenta registrada\n- **Si es día festivo**: Se adelanta al viernes anterior\n\nPuedes consultar tus estados de cuenta en el portal de Finanzas > Mis Payslips.",
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
                    f"""Eres un Asistente de Finanzas y Contabilidad de la empresa.
Tu objetivo es responder preguntas sobre políticas financieras, reembolsos, viáticos, nóminas y procedimientos contables.

INSTRUCCIONES:
- Usa EXCLUSIVAMENTE la información del contexto proporcionado
- Si el contexto no tiene la respuesta, indica que no tienes esa información específica
- Sé conciso y directo en tus respuestas
- No inventes información
- Responde siempre en español
- Si una pregunta no es de tu área, redirige al usuario al departamento correcto (ej. RRHH, IT, etc.)

FORMATO DE RESPUESTA:
- Usa viñetas (-) para listar información
- Usa **negrita** para montos y fechas importantes
- Incluye pasos específicos cuando sea necesario
- Indica tiempos de procesamiento cuando aplique""",
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
            print(f"Error durante el procesamiento del Agente de Finanzas: {e}")
            return "Lo siento, hubo un error interno al procesar tu solicitud financiera. Por favor, contacta al departamento de Finanzas."


if __name__ == "__main__":
    print("--- INICIANDO PRUEBA DEL AGENTE DE FINANZAS ---")

    # Inicializar el agente
    finance_agent = FinanceAgent()

    # Prueba
    test_question = "¿Cómo solicito un reembolso de gastos?"
    print(f"Pregunta: {test_question}")

    response = finance_agent.answer_query(test_question)
    print(f"\nRespuesta: {response}")
