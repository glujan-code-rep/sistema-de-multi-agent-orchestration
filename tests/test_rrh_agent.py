import pytest
from unittest.mock import MagicMock, patch

from agents.rrh_agent import RRHAgent


class TestRRHAgentInit:
    def test_init_creates_llm(self):
        with (
            patch("agents.rrh_agent.ChatOpenAI") as MockLLM,
            patch("agents.rrh_agent.VectorRetriever"),
        ):
            agent = RRHAgent()
            MockLLM.assert_called_once_with(
                model="mistral-7b-instruct-v0.2-q4_K_M",
                base_url="http://127.0.0.1:1234/v1",
                temperature=0.7,
            )

    def test_init_with_custom_model(self):
        with (
            patch("agents.rrh_agent.ChatOpenAI") as MockLLM,
            patch("agents.rrh_agent.VectorRetriever"),
        ):
            agent = RRHAgent(llm_model="qwen3.6-35b-a3b")
            MockLLM.assert_called_once_with(
                model="qwen3.6-35b-a3b",
                base_url="http://127.0.0.1:1234/v1",
                temperature=0.7,
            )

    def test_init_creates_retriever(self):
        with (
            patch("agents.rrh_agent.ChatOpenAI"),
            patch("agents.rrh_agent.VectorRetriever") as MockRetriever,
        ):
            agent = RRHAgent()
            MockRetriever.assert_called_once_with(
                db_path="./knowledge_bases/rrh_vector_db"
            )

    def test_init_has_prompt_template(self):
        with (
            patch("agents.rrh_agent.ChatOpenAI"),
            patch("agents.rrh_agent.VectorRetriever"),
        ):
            agent = RRHAgent()
            assert agent.prompt is not None
            messages = agent.prompt.messages
            assert len(messages) == 2
            assert hasattr(messages[0], "prompt")
            assert hasattr(messages[1], "prompt")

    def test_system_prompt_content(self):
        with (
            patch("agents.rrh_agent.ChatOpenAI"),
            patch("agents.rrh_agent.VectorRetriever"),
        ):
            agent = RRHAgent()
            system_msg = agent.prompt.messages[0].prompt.template
            assert "Recursos Humanos" in system_msg

    def test_human_prompt_template(self):
        with (
            patch("agents.rrh_agent.ChatOpenAI"),
            patch("agents.rrh_agent.VectorRetriever"),
        ):
            agent = RRHAgent()
            human_msg = agent.prompt.messages[1].prompt.template
            assert "{context}" in human_msg
            assert "{question}" in human_msg
            assert "Contexto disponible" in human_msg


class TestRRHAgentAnswerQuery:
    def test_answer_query_returns_llm_response(self):
        with (
            patch("agents.rrh_agent.ChatOpenAI") as MockLLMClass,
            patch("agents.rrh_agent.VectorRetriever") as MockRetrieverClass,
        ):
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = "Contexto RRH sobre licencias"
            MockRetrieverClass.return_value = mock_retriever

            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = (
                "Los empleados nuevos tienen 15 días de licencia anual."
            )
            mock_llm.invoke.return_value = mock_response
            MockLLMClass.return_value = mock_llm

            agent = RRHAgent()
            result = agent.answer_query("Políticas de licencia para empleados nuevos")

            assert result == "Los empleados nuevos tienen 15 días de licencia anual."
            mock_retriever.retrieve.assert_called_once_with(
                "Políticas de licencia para empleados nuevos"
            )
            mock_llm.invoke.assert_called_once()

    def test_answer_query_error_returns_fallback_message(self):
        with (
            patch("agents.rrh_agent.ChatOpenAI") as MockLLMClass,
            patch("agents.rrh_agent.VectorRetriever") as MockRetrieverClass,
        ):
            mock_retriever = MagicMock()
            mock_retriever.retrieve.side_effect = Exception("DB no disponible")
            MockRetrieverClass.return_value = mock_retriever
            MockLLMClass.return_value = MagicMock()

            agent = RRHAgent()
            result = agent.answer_query("Consulta sobre beneficios")

            assert "error interno" in result.lower()
            assert "inténtalo de nuevo más tarde" in result

    def test_answer_query_empty_context(self):
        with (
            patch("agents.rrh_agent.ChatOpenAI") as MockLLMClass,
            patch("agents.rrh_agent.VectorRetriever") as MockRetrieverClass,
        ):
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = (
                "No se encontraron documentos relevantes para esta consulta."
            )
            MockRetrieverClass.return_value = mock_retriever

            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "No hay información disponible sobre este tema."
            mock_llm.invoke.return_value = mock_response
            MockLLMClass.return_value = mock_llm

            agent = RRHAgent()
            result = agent.answer_query("Políticas de viaje")

            assert result == "No hay información disponible sobre este tema."

    def test_answer_query_calls_prompt_format(self):
        with (
            patch("agents.rrh_agent.ChatOpenAI") as MockLLMClass,
            patch("agents.rrh_agent.VectorRetriever") as MockRetrieverClass,
        ):
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = "Contexto relevante"
            MockRetrieverClass.return_value = mock_retriever

            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Respuesta generada"
            mock_llm.invoke.return_value = mock_response
            MockLLMClass.return_value = mock_llm

            agent = RRHAgent()
            agent.answer_query("Pregunta de prueba")

            called_args = mock_llm.invoke.call_args[0][0]
            assert "Contexto disponible:" in called_args
            assert "Pregunta del usuario: Pregunta de prueba" in called_args


class TestRRHAgentIntegration:
    def test_agent_attributes_exist(self):
        with (
            patch("agents.rrh_agent.ChatOpenAI"),
            patch("agents.rrh_agent.VectorRetriever"),
        ):
            agent = RRHAgent()
            assert hasattr(agent, "llm")
            assert hasattr(agent, "retriever")
            assert hasattr(agent, "prompt")
