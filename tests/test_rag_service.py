"""
Tests unitarios para RAGService.

Estrategia:
  - El vector store se pasa como mock al constructor (no se necesita ChromaDB real).
  - ChatGoogleGenerativeAI se parchea para evitar llamadas a la API.
  - Se verifica la lógica: recuperación de contexto, construcción de fuentes,
    deduplicación, truncado de previews.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.config import Settings
from app.models import ChatResponse
from app.services.rag_service import RAGService

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_settings():
    return Settings(
        google_api_key="fake-test-key",
        gemini_model="gemini-2.0-flash",
        max_tokens=512,
        retriever_k=3,
    )


@pytest.fixture
def mock_vector_store():
    """ChromaDB mockeado que devuelve un retriever también mockeado."""
    vs = MagicMock()
    vs.as_retriever.return_value = MagicMock()
    return vs


@pytest.fixture
def rag_service(mock_vector_store, mock_settings):
    """RAGService con LLM mockeado."""
    with patch("app.services.rag_service.ChatGoogleGenerativeAI") as mock_llm_class:
        mock_llm_class.return_value = MagicMock()
        svc = RAGService(vector_store=mock_vector_store, settings=mock_settings)
        yield svc


# ─── Tests de RAGService.__init__ ────────────────────────────────────────────

class TestRAGServiceInit:
    def test_llm_is_initialized_with_correct_model(self, mock_vector_store, mock_settings):
        with patch("app.services.rag_service.ChatGoogleGenerativeAI") as mock_llm:
            mock_llm.return_value = MagicMock()
            RAGService(vector_store=mock_vector_store, settings=mock_settings)

            call_kwargs = mock_llm.call_args.kwargs
            assert call_kwargs["model"] == "gemini-2.0-flash"
            assert call_kwargs["google_api_key"] == "fake-test-key"

    def test_retriever_is_created_from_vector_store(self, mock_vector_store, mock_settings):
        with patch("app.services.rag_service.ChatGoogleGenerativeAI"):
            RAGService(vector_store=mock_vector_store, settings=mock_settings)
            mock_vector_store.as_retriever.assert_called_once()

    def test_settings_are_stored(self, rag_service, mock_settings):
        assert rag_service.settings.gemini_model == "gemini-2.0-flash"
        assert rag_service.settings.max_tokens == 512


# ─── Tests de RAGService.answer ──────────────────────────────────────────────

class TestRAGServiceAnswer:
    """Verifica el pipeline RAG completo: recuperación + generación + respuesta."""

    def test_answer_returns_chat_response_type(self, rag_service):
        rag_service.vector_store.similarity_search_with_score = MagicMock(return_value=[])
        rag_service.llm.invoke = MagicMock(
            return_value=AIMessage(content="Respuesta de prueba.")
        )

        result = rag_service.answer("¿Cuál es la tasa de interés?")

        assert isinstance(result, ChatResponse)

    def test_answer_preserves_question(self, rag_service):
        rag_service.vector_store.similarity_search_with_score = MagicMock(return_value=[])
        rag_service.llm.invoke = MagicMock(return_value=AIMessage(content="Respuesta."))

        result = rag_service.answer("¿Cuánto es el plazo fijo?")

        assert result.question == "¿Cuánto es el plazo fijo?"

    def test_answer_text_comes_from_llm(self, rag_service):
        rag_service.vector_store.similarity_search_with_score = MagicMock(return_value=[])
        rag_service.llm.invoke = MagicMock(
            return_value=AIMessage(content="La TNA es 45% mensual.")
        )

        result = rag_service.answer("¿Qué tasa tiene el plazo fijo?")

        assert result.answer == "La TNA es 45% mensual."

    def test_answer_model_field_matches_settings(self, rag_service):
        rag_service.vector_store.similarity_search_with_score = MagicMock(return_value=[])
        rag_service.llm.invoke = MagicMock(return_value=AIMessage(content="Ok."))

        result = rag_service.answer("pregunta")

        assert result.model == "gemini-2.0-flash"

    def test_answer_with_one_source_document(self, rag_service):
        """Los documentos recuperados deben aparecer en sources con su similarity_score."""
        doc = Document(
            page_content="Información sobre préstamos hipotecarios.",
            metadata={"source": "hipotecas.pdf", "page": 3},
        )
        rag_service.vector_store.similarity_search_with_score = MagicMock(return_value=[(doc, 0.876)])
        rag_service.llm.invoke = MagicMock(return_value=AIMessage(content="Respuesta."))

        result = rag_service.answer("¿Qué es una hipoteca?")

        assert len(result.sources) == 1
        assert result.sources[0].source == "hipotecas.pdf"
        assert result.sources[0].page == 3
        assert result.sources[0].similarity_score == 0.876

    def test_answer_with_no_docs_has_empty_sources(self, rag_service):
        """Si no hay documentos relevantes, sources debe ser lista vacía."""
        rag_service.vector_store.similarity_search_with_score = MagicMock(return_value=[])
        rag_service.llm.invoke = MagicMock(
            return_value=AIMessage(content="No tengo esa información.")
        )

        result = rag_service.answer("¿Algo que no existe?")

        assert result.sources == []

    def test_answer_deduplicates_same_source_and_page(self, rag_service):
        """Dos chunks del mismo doc/página NO deben generar dos entradas en sources."""
        docs_with_scores = [
            (Document(page_content="Chunk A", metadata={"source": "guia.pdf", "page": 1}), 0.9),
            (Document(page_content="Chunk B", metadata={"source": "guia.pdf", "page": 1}), 0.85),
            (Document(page_content="Chunk C", metadata={"source": "guia.pdf", "page": 2}), 0.7),
        ]
        rag_service.vector_store.similarity_search_with_score = MagicMock(return_value=docs_with_scores)
        rag_service.llm.invoke = MagicMock(return_value=AIMessage(content="Ok."))

        result = rag_service.answer("pregunta")

        # p.1 y p.2 → 2 fuentes únicas
        assert len(result.sources) == 2
        pages = {s.page for s in result.sources}
        assert pages == {1, 2}

    def test_answer_truncates_long_content_preview(self, rag_service):
        """Contenido de más de 300 caracteres debe truncarse con '...'"""
        long_text = "B" * 500
        docs_with_scores = [(Document(page_content=long_text, metadata={"source": "largo.txt"}), 0.75)]
        rag_service.vector_store.similarity_search_with_score = MagicMock(return_value=docs_with_scores)
        rag_service.llm.invoke = MagicMock(return_value=AIMessage(content="Ok."))

        result = rag_service.answer("pregunta larga")

        assert result.sources[0].content.endswith("...")
        assert len(result.sources[0].content) == 303  # 300 + "..."

    def test_answer_short_content_is_not_truncated(self, rag_service):
        """Contenido corto NO debe truncarse."""
        short_text = "Texto breve."
        docs_with_scores = [(Document(page_content=short_text, metadata={"source": "corto.txt"}), 0.6)]
        rag_service.vector_store.similarity_search_with_score = MagicMock(return_value=docs_with_scores)
        rag_service.llm.invoke = MagicMock(return_value=AIMessage(content="Ok."))

        result = rag_service.answer("pregunta")

        assert result.sources[0].content == short_text
        assert not result.sources[0].content.endswith("...")

    def test_answer_doc_without_source_uses_unknown(self, rag_service):
        """Documentos sin metadato 'source' deben mostrar 'Desconocido'."""
        docs_with_scores = [(Document(page_content="Texto sin fuente.", metadata={}), 0.55)]
        rag_service.vector_store.similarity_search_with_score = MagicMock(return_value=docs_with_scores)
        rag_service.llm.invoke = MagicMock(return_value=AIMessage(content="Ok."))

        result = rag_service.answer("pregunta")

        assert result.sources[0].source == "Desconocido"

    def test_answer_calls_retriever_with_question(self, rag_service):
        """similarity_search_with_score debe invocarse con la pregunta original y k correcto."""
        rag_service.vector_store.similarity_search_with_score = MagicMock(return_value=[])
        rag_service.llm.invoke = MagicMock(return_value=AIMessage(content="Ok."))

        rag_service.answer("¿Qué es el BCRA?")

        rag_service.vector_store.similarity_search_with_score.assert_called_once_with(
            "¿Qué es el BCRA?", k=3
        )

    def test_answer_calls_llm_with_system_and_human_messages(self, rag_service):
        """El LLM debe recibir una lista con SystemMessage y HumanMessage."""
        from langchain_core.messages import HumanMessage, SystemMessage

        rag_service.vector_store.similarity_search_with_score = MagicMock(return_value=[])
        rag_service.llm.invoke = MagicMock(return_value=AIMessage(content="Ok."))

        rag_service.answer("¿Qué es un bono?")

        call_args = rag_service.llm.invoke.call_args[0][0]
        assert len(call_args) == 2
        assert isinstance(call_args[0], SystemMessage)
        assert isinstance(call_args[1], HumanMessage)
