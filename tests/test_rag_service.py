"""
Tests unitarios para RAGService.

Estrategia:
  - El vector store se pasa como mock al constructor.
  - ChatGoogleGenerativeAI y CrossEncoder se parchean (sin API ni descarga de modelos).
  - Se verifica la lógica: reranking, guardas deterministas (fallback, urgencia),
    validación de citas, fuentes, deduplicación, truncado de previews.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.config import Settings
from app.models import ChatResponse
from app.services.rag_service import (
    NO_INFO_ANSWER,
    URGENCY_NOTICE,
    LLMAnswer,
    RAGService,
    is_urgent,
)

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_settings():
    return Settings(
        google_api_key="fake-test-key",
        gemini_model="gemini-2.0-flash",
        max_tokens=512,
        retriever_k=3,
    )


@pytest.fixture(autouse=True)
def mock_cross_encoder():
    """CrossEncoder mockeado: evita descargar bge-reranker (~2 GB).
    predict devuelve scores decrecientes → conserva el orden original de Pinecone."""
    with patch("app.services.rag_service.CrossEncoder") as mock_ce_class:
        mock_ce_class.return_value.predict.side_effect = lambda pairs: [
            float(len(pairs) - i) for i in range(len(pairs))
        ]
        yield mock_ce_class


@pytest.fixture
def mock_vector_store():
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


def llm_says(rag_service, answer="Ok.", cited=None, found=True):
    """Mockea la salida estructurada del LLM."""
    rag_service.structured_llm.invoke = MagicMock(
        return_value=LLMAnswer(answer=answer, cited_fragments=cited or [], found_in_context=found)
    )


def docs(*specs):
    """specs: (contenido, metadata, score) → lista (Document, score)."""
    return [(Document(page_content=c, metadata=m), sc) for c, m, sc in specs]


# ─── Tests de RAGService.__init__ ────────────────────────────────────────────

class TestRAGServiceInit:
    def test_llm_is_initialized_with_correct_model(self, mock_vector_store, mock_settings):
        with patch("app.services.rag_service.ChatGoogleGenerativeAI") as mock_llm:
            mock_llm.return_value = MagicMock()
            RAGService(vector_store=mock_vector_store, settings=mock_settings)

            call_kwargs = mock_llm.call_args.kwargs
            assert call_kwargs["model"] == "gemini-2.0-flash"
            assert call_kwargs["google_api_key"] == "fake-test-key"

    def test_structured_output_uses_schema(self, rag_service):
        rag_service.llm.with_structured_output.assert_called_once_with(LLMAnswer)

    def test_retriever_is_created_from_vector_store(self, mock_vector_store, mock_settings):
        with patch("app.services.rag_service.ChatGoogleGenerativeAI"):
            RAGService(vector_store=mock_vector_store, settings=mock_settings)
            mock_vector_store.as_retriever.assert_called_once()

    def test_settings_are_stored(self, rag_service):
        assert rag_service.settings.gemini_model == "gemini-2.0-flash"
        assert rag_service.settings.max_tokens == 512


# ─── Tests de RAGService.answer ──────────────────────────────────────────────

class TestRAGServiceAnswer:
    """Verifica el pipeline RAG completo: recuperación + generación + respuesta."""

    def test_answer_returns_chat_response_type(self, rag_service):
        rag_service.vector_store.similarity_search_with_score = MagicMock(return_value=[])
        assert isinstance(rag_service.answer("¿Cuál es la tasa?"), ChatResponse)

    def test_answer_preserves_question(self, rag_service):
        rag_service.vector_store.similarity_search_with_score = MagicMock(return_value=[])
        assert rag_service.answer("¿Cuánto es el plazo fijo?").question == "¿Cuánto es el plazo fijo?"

    def test_answer_text_comes_from_llm(self, rag_service):
        rag_service.vector_store.similarity_search_with_score = MagicMock(
            return_value=docs(("TNA 45%", {"source": "a.txt"}, 0.9))
        )
        llm_says(rag_service, "La TNA es 45%.", cited=[1])
        result = rag_service.answer("¿Qué tasa tiene el plazo fijo?")
        assert result.answer == "La TNA es 45%."
        assert result.status == "answered"

    def test_answer_model_field_matches_settings(self, rag_service):
        rag_service.vector_store.similarity_search_with_score = MagicMock(return_value=[])
        assert rag_service.answer("pregunta").model == "gemini-2.0-flash"

    def test_answer_with_one_source_document(self, rag_service):
        rag_service.vector_store.similarity_search_with_score = MagicMock(
            return_value=docs(("Préstamos hipotecarios.", {"source": "hipotecas.pdf", "page": 3}, 0.876))
        )
        llm_says(rag_service, cited=[1])
        result = rag_service.answer("¿Qué es una hipoteca?")
        assert len(result.sources) == 1
        assert result.sources[0].source == "hipotecas.pdf"
        assert result.sources[0].page == 3
        assert result.sources[0].similarity_score == 0.876

    def test_answer_deduplicates_same_source_and_page(self, rag_service):
        rag_service.vector_store.similarity_search_with_score = MagicMock(return_value=docs(
            ("Chunk A", {"source": "guia.pdf", "page": 1}, 0.9),
            ("Chunk B", {"source": "guia.pdf", "page": 1}, 0.85),
            ("Chunk C", {"source": "guia.pdf", "page": 2}, 0.7),
        ))
        llm_says(rag_service, cited=[1, 2, 3])
        result = rag_service.answer("pregunta")
        assert {s.page for s in result.sources} == {1, 2}

    def test_answer_truncates_long_content_preview(self, rag_service):
        rag_service.vector_store.similarity_search_with_score = MagicMock(
            return_value=docs(("B" * 500, {"source": "largo.txt"}, 0.75))
        )
        llm_says(rag_service, cited=[1])
        result = rag_service.answer("pregunta larga")
        assert len(result.sources[0].content) == 303  # 300 + "..."
        # La evaluación recibe el chunk completo, no el preview
        assert result.contexts == ["B" * 500]

    def test_answer_short_content_is_not_truncated(self, rag_service):
        rag_service.vector_store.similarity_search_with_score = MagicMock(
            return_value=docs(("Texto breve.", {"source": "corto.txt"}, 0.6))
        )
        llm_says(rag_service, cited=[1])
        assert rag_service.answer("pregunta").sources[0].content == "Texto breve."

    def test_answer_doc_without_source_uses_unknown(self, rag_service):
        rag_service.vector_store.similarity_search_with_score = MagicMock(
            return_value=docs(("Texto sin fuente.", {}, 0.55))
        )
        llm_says(rag_service, cited=[1])
        assert rag_service.answer("pregunta").sources[0].source == "Desconocido"

    def test_answer_calls_retriever_with_question(self, rag_service):
        """Debe pedir k * multiplier candidatos a Pinecone para dar margen al reranker."""
        rag_service.vector_store.similarity_search_with_score = MagicMock(return_value=[])
        rag_service.answer("¿Qué es el BCRA?")
        expected_k = 3 * rag_service.settings.reranker_candidates_multiplier
        rag_service.vector_store.similarity_search_with_score.assert_called_once_with(
            "¿Qué es el BCRA?", k=expected_k
        )

    def test_reranker_reorders_and_keeps_top_k(self, rag_service):
        """El orden final lo decide el CrossEncoder y se conservan solo k chunks."""
        rag_service.vector_store.similarity_search_with_score = MagicMock(
            return_value=docs(*[(f"chunk {i}", {"source": f"doc{i}.txt"}, 0.9) for i in range(6)])
        )
        rag_service.reranker.predict = MagicMock(side_effect=lambda pairs: list(range(len(pairs))))
        llm_says(rag_service, cited=[1, 2, 3])
        result = rag_service.answer("¿Pregunta?")
        assert [s.source for s in result.sources] == ["doc5.txt", "doc4.txt", "doc3.txt"]

    def test_answer_calls_llm_with_system_and_human_messages(self, rag_service):
        from langchain_core.messages import HumanMessage, SystemMessage

        rag_service.vector_store.similarity_search_with_score = MagicMock(
            return_value=docs(("x", {"source": "a.txt"}, 0.9))
        )
        llm_says(rag_service, cited=[1])
        rag_service.answer("¿Qué es un bono?")
        call_args = rag_service.structured_llm.invoke.call_args[0][0]
        assert isinstance(call_args[0], SystemMessage)
        assert isinstance(call_args[1], HumanMessage)


# ─── Guardas deterministas (CCA anti-patrón #1) ──────────────────────────────

class TestDeterministicGuards:
    """Garantías en código: no dependen de que el LLM cumpla el prompt."""

    def test_no_candidates_returns_fallback_without_llm(self, rag_service):
        rag_service.vector_store.similarity_search_with_score = MagicMock(return_value=[])
        rag_service.structured_llm.invoke = MagicMock()
        result = rag_service.answer("¿Algo que no existe?")
        assert result.answer == NO_INFO_ANSWER
        assert result.status == "no_context"
        assert result.sources == []
        rag_service.structured_llm.invoke.assert_not_called()

    def test_low_rerank_score_returns_fallback_without_llm(self, rag_service):
        rag_service.settings.min_rerank_score = 0.5
        rag_service.vector_store.similarity_search_with_score = MagicMock(
            return_value=docs(("irrelevante", {"source": "a.txt"}, 0.9))
        )
        rag_service.reranker.predict = MagicMock(return_value=[0.1])
        rag_service.structured_llm.invoke = MagicMock()
        result = rag_service.answer("¿Capital de Francia?")
        assert result.status == "no_context"
        rag_service.structured_llm.invoke.assert_not_called()

    def test_llm_not_found_returns_fixed_fallback_text(self, rag_service):
        rag_service.vector_store.similarity_search_with_score = MagicMock(
            return_value=docs(("x", {"source": "a.txt"}, 0.9))
        )
        llm_says(rag_service, answer="Inventado", found=False)
        result = rag_service.answer("¿Tasa del préstamo?")
        assert result.answer == NO_INFO_ANSWER
        assert result.status == "no_context"

    def test_urgent_question_always_gets_notice(self, rag_service):
        rag_service.vector_store.similarity_search_with_score = MagicMock(
            return_value=docs(("Bloqueo desde la app.", {"source": "a.txt"}, 0.9))
        )
        llm_says(rag_service, "Podés bloquearla desde la app.", cited=[1])
        result = rag_service.answer("Me robaron la tarjeta, ¿qué hago?")
        assert result.urgent is True
        assert URGENCY_NOTICE in result.answer

    def test_urgent_notice_also_on_fallback(self, rag_service):
        rag_service.vector_store.similarity_search_with_score = MagicMock(return_value=[])
        assert URGENCY_NOTICE in rag_service.answer("Creo que es un fraude").answer

    @pytest.mark.parametrize("q,expected", [
        ("Me robaron el celular", True),
        ("Perdí mi tarjeta", True),
        ("No reconozco un consumo", True),
        ("¿Cuál es la tasa del plazo fijo?", False),
        ("¿Cómo pido un préstamo?", False),
    ])
    def test_is_urgent(self, q, expected):
        assert is_urgent(q) is expected


# ─── Procedencia de fuentes (CCA anti-patrones #5 y #6) ──────────────────────

class TestCitations:
    """Las fuentes son las citadas por el LLM; las citas se validan en código."""

    def _three_docs(self, rag_service):
        rag_service.vector_store.similarity_search_with_score = MagicMock(
            return_value=docs(*[(f"c{i}", {"source": f"d{i}.txt"}, 0.9) for i in range(3)])
        )

    def test_only_cited_fragments_become_sources(self, rag_service):
        self._three_docs(rag_service)
        llm_says(rag_service, cited=[2])
        assert [s.source for s in rag_service.answer("p").sources] == ["d1.txt"]

    def test_invalid_citations_are_dropped(self, rag_service):
        self._three_docs(rag_service)
        llm_says(rag_service, cited=[0, 3, 99])
        assert [s.source for s in rag_service.answer("p").sources] == ["d2.txt"]

    def test_no_valid_citations_degrades_to_all_sources(self, rag_service):
        self._three_docs(rag_service)
        llm_says(rag_service, cited=[])
        assert len(rag_service.answer("p").sources) == 3

    def test_contexts_include_all_chunks_seen_by_llm(self, rag_service):
        self._three_docs(rag_service)
        llm_says(rag_service, cited=[1])
        assert rag_service.answer("p").contexts == ["c0", "c1", "c2"]

    def test_contexts_not_exposed_in_api_json(self, rag_service):
        self._three_docs(rag_service)
        llm_says(rag_service, cited=[1])
        assert "contexts" not in rag_service.answer("p").model_dump()
