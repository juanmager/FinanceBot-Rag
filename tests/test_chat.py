"""
Tests del endpoint POST /chat/ask.
Verifica el comportamiento del chatbot RAG sin consumir tokens reales.
"""

import os
from datetime import datetime, timezone

os.environ.setdefault("GOOGLE_API_KEY", "AIzaSy-test-key")


class TestChatAskEndpoint:
    """Tests del endpoint POST /chat/ask."""

    def test_ask_returns_200(self, client):
        """Una pregunta válida debe retornar HTTP 200."""
        response = client.post("/chat/ask", json={"question": "¿Cuál es la tasa del plazo fijo?"})
        assert response.status_code == 200

    def test_ask_response_has_answer(self, client):
        """La respuesta debe incluir el campo 'answer' con texto."""
        data = client.post("/chat/ask", json={"question": "¿Cómo abro una cuenta?"}).json()
        assert "answer" in data
        assert len(data["answer"]) > 0

    def test_ask_response_full_structure(self, client):
        """La respuesta debe tener todos los campos del modelo ChatResponse."""
        data = client.post("/chat/ask", json={"question": "¿Tienen tarjeta de crédito?"}).json()

        required_fields = ["question", "answer", "sources", "model", "timestamp"]
        for field in required_fields:
            assert field in data, f"Campo '{field}' ausente en la respuesta"

    def test_ask_echoes_original_question(self, client, mock_rag):
        """La respuesta debe reflejar la pregunta enviada."""
        from app.models import ChatResponse

        question = "¿Cuánto es el límite de transferencia diario?"
        mock_rag.answer.return_value = ChatResponse(
            question=question,
            answer="El límite diario estándar es $500.000.",
            sources=[],
            model="claude-3-5-sonnet-20241022",
            timestamp=datetime.now(timezone.utc),
        )

        data = client.post("/chat/ask", json={"question": question}).json()
        assert data["question"] == question

    def test_ask_sources_is_list(self, client):
        """El campo 'sources' debe ser una lista."""
        data = client.post("/chat/ask", json={"question": "¿Tienen préstamos?"}).json()
        assert isinstance(data["sources"], list)

    def test_ask_source_structure(self, client, mock_rag):
        """Cada fuente en 'sources' debe tener content, source y page."""
        from app.models import ChatResponse, SourceDocument

        mock_rag.answer.return_value = ChatResponse(
            question="test",
            answer="Respuesta de prueba",
            sources=[
                SourceDocument(
                    content="Fragmento del documento financiero.",
                    source="productos_financieros.txt",
                    page=None,
                )
            ],
            model="claude-3-5-sonnet-20241022",
            timestamp=datetime.now(timezone.utc),
        )

        sources = client.post("/chat/ask", json={"question": "test"}).json()["sources"]
        assert len(sources) == 1
        source = sources[0]
        assert "content" in source
        assert "source" in source
        assert "page" in source

    def test_ask_empty_question_returns_422(self, client):
        """Una pregunta vacía debe retornar HTTP 422 (validación Pydantic)."""
        response = client.post("/chat/ask", json={"question": ""})
        assert response.status_code == 422

    def test_ask_missing_question_returns_422(self, client):
        """Omitir el campo 'question' debe retornar HTTP 422."""
        response = client.post("/chat/ask", json={})
        assert response.status_code == 422

    def test_ask_calls_rag_service_with_question(self, client, mock_rag):
        """El endpoint debe llamar a rag_service.answer() con la pregunta correcta."""
        question = "¿Cuáles son los requisitos del préstamo hipotecario?"
        client.post("/chat/ask", json={"question": question})
        mock_rag.answer.assert_called_once_with(question)

    def test_ask_model_field_present(self, client):
        """La respuesta debe incluir el nombre del modelo utilizado."""
        data = client.post("/chat/ask", json={"question": "¿Tienen caja de ahorros?"}).json()
        assert "model" in data
        assert len(data["model"]) > 0
