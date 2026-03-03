"""
Fixtures compartidas para todos los tests del Proyecto 2.

Estrategia de testing:
  - Los servicios reales (IngestService, RAGService) son pesados: cargan modelos de ML
    y requieren conexiones a Gemini y Pinecone.
  - En tests, inyectamos mocks de esos servicios usando el patrón create_app(lifespan_fn=).
  - Los tests solo verifican comportamiento HTTP (status codes, estructuras JSON).
  - No se consumen tokens ni se cargan modelos de embeddings.
"""

import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# Asegurar variable de entorno antes de importar cualquier módulo de la app
os.environ.setdefault("GOOGLE_API_KEY", "AIzaSy-test-key")
os.environ.setdefault("GEMINI_MODEL", "gemini-2.0-flash")
os.environ.setdefault("PINECONE_API_KEY", "fake-pinecone-key")
os.environ.setdefault("PINECONE_INDEX_NAME", "test-index")


# ── Helpers para construir respuestas mock ─────────────────────────────────────

def make_mock_ingest_service(documents=None):
    """Crea un IngestService mockeado con comportamiento configurable."""
    from app.models import IngestResponse

    service = MagicMock()
    service.get_documents.return_value = documents or []
    service.ingest_file.return_value = IngestResponse(
        filename="test.txt",
        chunks_created=5,
        message="'test.txt' procesado exitosamente con 5 fragmentos indexados.",
    )
    return service


def make_mock_rag_service():
    """Crea un RAGService mockeado con una respuesta financiera de ejemplo."""
    from app.models import ChatResponse, SourceDocument

    service = MagicMock()
    service.answer.return_value = ChatResponse(
        question="¿Cuál es la tasa del plazo fijo?",
        answer="La tasa nominal anual del Plazo Fijo FinanceBot es del 45%, revisable mensualmente.",
        sources=[
            SourceDocument(
                content="TNA vigente: 45% (se actualiza mensualmente según política del BCRA)",
                source="productos_financieros.txt",
                page=None,
            )
        ],
        model="gemini-2.0-flash",
        timestamp=datetime.now(UTC),
    )
    return service


# ── Fixture principal ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_ingest():
    """IngestService mockeado sin documentos."""
    return make_mock_ingest_service()


@pytest.fixture
def mock_rag():
    """RAGService mockeado con respuesta de ejemplo."""
    return make_mock_rag_service()


@pytest.fixture
def client(mock_ingest, mock_rag):
    """
    Cliente de test con servicios mockeados inyectados vía lifespan personalizado.
    Cada test recibe un cliente con estado limpio.
    """
    @asynccontextmanager
    async def test_lifespan(app):
        # Inyectar servicios mock en el estado de la app
        app.state.ingest_service = mock_ingest
        app.state.rag_service = mock_rag
        yield

    from app.config import get_settings
    get_settings.cache_clear()

    from app.main import create_app
    test_app = create_app(lifespan_fn=test_lifespan)

    with TestClient(test_app) as c:
        yield c


