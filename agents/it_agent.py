# /agents/it_agent.py

import os
from langchain_openai import ChatOpenAI
from rag_utils.retriever import VectorRetriever  # Importamos la clase de RAG
from langchain_core.prompts import ChatPromptTemplate


class ITAgent:
    """
    Agente Especialista para Soporte Técnico (IT).
    Utiliza RAG sobre la base vectorial de FAQs y guías de troubleshooting.
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

        # 2. Inicializar el Retriever (Conexión a la base vectorial de IT)
        self.retriever = VectorRetriever(db_path="./knowledge_bases/it_vector_db")

        # 3. Definir el Prompt de Respuesta con Few-Shot Learning
        self.examples = [
            {
                "question": "Mi laptop está muy lenta",
                "context": "## Laptop lenta\n1. Reinicia el equipo (resuelve 60% de casos)\n2. Verifica espacio en disco: si <10GB, elimina archivos temporales\n3. Deshabilita programas de inicio automático: Ctrl+Shift+Esc > Inicio\n4. Ejecuta análisis con antivirus corporativo (Windows Defender/McAfee)",
                "answer": "Sigue estos pasos en orden:\n\n1. **Reinicia el equipo** - Esto resuelve el 60% de los casos.\n2. **Verifica el espacio en disco** - Si tienes menos de 10 GB libres, elimina archivos temporales o desinstala programas que no uses.\n3. **Deshabilita programas de inicio automático** - Presiona Ctrl+Shift+Esc, ve a la pestaña Inicio y deshabilita los innecesarios.\n4. **Ejecuta un análisis con el antivirus corporativo** - Usa Windows Defender o McAfee.",
            },
            {
                "question": "No puedo conectar al WiFi de la empresa",
                "context": "## Problemas de WiFi\n1. Verificar que el adaptador WiFi esté habilitado\n2. Olvidar red y reconectar\n3. Verificar que sea la red correcta (EmpresaXYZ-WiFi)\n4. Reiniciar el equipo\n5. Contactar al Helpdesk si persiste",
                "answer": "尝试 estos pasos:\n\n1. **Verifica que el adaptador WiFi esté habilitado** - Revisa en Configuración > Red e Internet\n2. **Olvida la red y reconecta** - Configuración > WiFi > EmpresaXYZ-WiFi > Olvidar\n3. **Verifica que sea la red correcta** - Debe ser 'EmpresaXYZ-WiFi'\n4. **Reinicia el equipo**\n5. **Si persiste**, contacta al Helpdesk de IT para soporte adicional.",
            },
            {
                "question": "Mi teclado no funciona algunas teclas",
                "context": "## Teclado con teclas que no responden\n1. Verificar que no esté修损坏\n2. Limpiar el teclado\n3. Probar en otro puerto USB\n4. Verificar drivers\n5. Solicitar reemplazo si es hardware",
                "answer": "Sigue estos pasos para diagnosticar:\n\n1. **Limpia el teclado** - Retira polvo y residuos entre las teclas\n2. **Prueba en otro puerto USB** - Si es teclado externo\n3. **Verifica los drivers** - Administrador de dispositivos > Teclados\n4. **Prueba con otro teclado** - Para descartar problema de hardware\n5. **Si es hardware**, reporta al Helpdesk para solicitar reemplazo.",
            },
        ]

        # Crear el template con few-shot examples
        examples_text = "\n\n".join(
            [
                f"Ejemplo {i + 1}:\nProblema: {ex['question']}\nContexto de solución: {ex['context'][:200]}...\nSolución: {ex['answer']}"
                for i, ex in enumerate(self.examples)
            ]
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""Eres un Asistente de Soporte Técnico (IT) de la empresa.
Tu objetivo es proporcionar soluciones claras, guías de troubleshooting y pasos a seguir.

INSTRUCCIONES:
- Usa EXCLUSIVAMENTE la información del contexto proporcionado
- Si el contexto no tiene la solución, indica que no tienes esa información específica
- Proporciona pasos específicos y accionables (usa numeración 1., 2., 3.)
- Sé conciso y directo
- Responde siempre en español
- Usa **negrita** para enfatizar teclas, programas o acciones importantes""",
                ),
                (
                    "human",
                    f"""Los siguientes son ejemplos de cómo debes resolver problemas:

{examples_text}

---

Ahora resuelve este nuevo problema:

Contexto disponible:\n{{context}}

Problema del usuario: {{question}}""",
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
            print(f"Error durante el procesamiento del Agente IT: {e}")
            # En caso de error, devolvemos un mensaje seguro en lugar de fallar completamente
            return "Lo siento, hubo un error interno al procesar tu solicitud técnica. Por favor, contacta al helpdesk de IT."


# ==============================================================================
# Ejemplo de cómo se usaría este agente (Para pruebas)
# ==============================================================================
if __name__ == "__main__":
    print("--- INICIANDO PRUEBA DEL AGENTE TÉCNICO (IT) ---")

    # Inicializar el agente
    it_agent = ITAgent()

    # Ejemplo de consulta
    test_question = "Mi laptop de la empresa está muy lenta, ¿qué debo hacer?"

    print(f"\nPregunta del Usuario: {test_question}")

    # Ejecutar la respuesta RAG
    response = it_agent.answer_query(test_question)

    print("\nRespuesta del Agente Técnico (IT):")
    print("------------------------------------------")
    print(response)
    print("------------------------------------------")
