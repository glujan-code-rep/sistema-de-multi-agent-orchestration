import pytest
from unittest.mock import MagicMock, patch

from agents.it_agent import ITAgent


class TestITAgentInit:
    def test_init_creates_llm(self):
        with (
            patch("agents.it_agent.ChatOpenAI") as MockLLM,
            patch("agents.it_agent.VectorRetriever"),
        ):
            agent = ITAgent()
            MockLLM.assert_called_once_with(
                model="mistral-7b-instruct-v0.2-q4_K_M",
                base_url="http://127.0.0.1:1234/v1",
                temperature=0.7,
            )

    def test_init_with_custom_model(self):
        with (
            patch("agents.it_agent.ChatOpenAI") as MockLLM,
            patch("agents.it_agent.VectorRetriever"),
        ):
            agent = ITAgent(llm_model="qwen3.6-35b-a3b")
            MockLLM.assert_called_once_with(
                model="qwen3.6-35b-a3b",
                base_url="http://127.0.0.1:1234/v1",
                temperature=0.7,
            )

    def test_init_creates_retriever(self):
        with (
            patch("agents.it_agent.ChatOpenAI"),
            patch("agents.it_agent.VectorRetriever") as MockRetriever,
        ):
            agent = ITAgent()
            MockRetriever.assert_called_once_with(
                db_path="./knowledge_bases/it_vector_db"
            )

    def test_init_has_prompt_template(self):
        with (
            patch("agents.it_agent.ChatOpenAI"),
            patch("agents.it_agent.VectorRetriever"),
        ):
            agent = ITAgent()
            assert agent.prompt is not None
            messages = agent.prompt.messages
            assert len(messages) == 2
            assert hasattr(messages[0], "prompt")
            assert hasattr(messages[1], "prompt")

    def test_system_prompt_content(self):
        with (
            patch("agents.it_agent.ChatOpenAI"),
            patch("agents.it_agent.VectorRetriever"),
        ):
            agent = ITAgent()
            system_msg = agent.prompt.messages[0].prompt.template
            assert "Soporte Técnico" in system_msg

    def test_human_prompt_template(self):
        with (
            patch("agents.it_agent.ChatOpenAI"),
            patch("agents.it_agent.VectorRetriever"),
        ):
            agent = ITAgent()
            human_msg = agent.prompt.messages[1].prompt.template
            assert "{context}" in human_msg
            assert "{question}" in human_msg
            assert "Contexto disponible" in human_msg


class TestITAgentAnswerQuery:
    def test_answer_query_returns_llm_response(self):
        with (
            patch("agents.it_agent.ChatOpenAI") as MockLLMClass,
            patch("agents.it_agent.VectorRetriever") as MockRetrieverClass,
        ):
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = "Contexto de IT sobre laptop lenta"
            MockRetrieverClass.return_value = mock_retriever

            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = (
                "Reinicia el sistema y verifica los procesos en segundo plano."
            )
            mock_llm.invoke.return_value = mock_response
            MockLLMClass.return_value = mock_llm

            agent = ITAgent()
            result = agent.answer_query("Mi laptop está muy lenta")

            assert (
                result
                == "Reinicia el sistema y verifica los procesos en segundo plano."
            )
            mock_retriever.retrieve.assert_called_once_with("Mi laptop está muy lenta")
            mock_llm.invoke.assert_called_once()

    def test_answer_query_error_returns_fallback_message(self):
        with (
            patch("agents.it_agent.ChatOpenAI") as MockLLMClass,
            patch("agents.it_agent.VectorRetriever") as MockRetrieverClass,
        ):
            mock_retriever = MagicMock()
            mock_retriever.retrieve.side_effect = Exception("DB no disponible")
            MockRetrieverClass.return_value = mock_retriever
            MockLLMClass.return_value = MagicMock()

            agent = ITAgent()
            result = agent.answer_query("Problema con red")

            assert "error interno" in result.lower()
            assert "helpdesk de IT" in result

    def test_answer_query_empty_context(self):
        with (
            patch("agents.it_agent.ChatOpenAI") as MockLLMClass,
            patch("agents.it_agent.VectorRetriever") as MockRetrieverClass,
        ):
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = (
                "No se encontraron documentos relevantes para esta consulta."
            )
            MockRetrieverClass.return_value = mock_retriever

            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = (
                "No hay información disponible en nuestra base de conocimiento."
            )
            mock_llm.invoke.return_value = mock_response
            MockLLMClass.return_value = mock_llm

            agent = ITAgent()
            result = agent.answer_query("Configurar impresora")

            assert (
                result
                == "No hay información disponible en nuestra base de conocimiento."
            )

    def test_answer_query_calls_prompt_format(self):
        with (
            patch("agents.it_agent.ChatOpenAI") as MockLLMClass,
            patch("agents.it_agent.VectorRetriever") as MockRetrieverClass,
        ):
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = "Contexto relevante"
            MockRetrieverClass.return_value = mock_retriever

            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Respuesta generada"
            mock_llm.invoke.return_value = mock_response
            MockLLMClass.return_value = mock_llm

            agent = ITAgent()
            agent.answer_query("Pregunta de prueba")

            called_args = mock_llm.invoke.call_args[0][0]
            assert "Contexto disponible:" in called_args
            assert "Problema del usuario: Pregunta de prueba" in called_args


class TestITAgentIntegration:
    def test_agent_attributes_exist(self):
        with (
            patch("agents.it_agent.ChatOpenAI"),
            patch("agents.it_agent.VectorRetriever"),
        ):
            agent = ITAgent()
            assert hasattr(agent, "llm")
            assert hasattr(agent, "retriever")
            assert hasattr(agent, "prompt")
