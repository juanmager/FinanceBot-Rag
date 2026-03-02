"""
Tests unitarios para IngestService y la función format_docs.

Estrategia:
  - Se mockean HuggingFaceEmbeddings, Pinecone y PineconeVectorStore para evitar cargar el modelo (~90 MB)
    y conectarse a Pinecone durante los tests.
  - Los loaders de LangChain (PyPDFLoader, TextLoader) se mockean para evitar I/O real.
  - Se testea la lógica de negocio: chunking, registro de metadatos, manejo de errores.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.config import Settings
from app.models import DocumentInfo, IngestResponse
from app.services.ingest_service import IngestService, format_docs

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_settings():
    """Settings de prueba sin necesidad de .env."""
    return Settings(
        google_api_key="fake-test-key",
        pinecone_api_key="fake-pinecone-key",
        pinecone_index_name="test-index",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=500,
        chunk_overlap=50,
        retriever_k=3,
    )


@pytest.fixture
def ingest_service(mock_settings):
    """
    IngestService con dependencias externas mockeadas.
    HuggingFaceEmbeddings, Pinecone y PineconeVectorStore no se instancian realmente.
    """
    with (
        patch("app.services.ingest_service.HuggingFaceEmbeddings") as mock_emb,
        patch("app.services.ingest_service.Pinecone") as mock_pc,
        patch("app.services.ingest_service.PineconeVectorStore") as mock_pvs,
    ):
        mock_emb.return_value = MagicMock()
        mock_pc.return_value = MagicMock()
        mock_pvs.return_value = MagicMock()
        svc = IngestService(settings=mock_settings)
        yield svc


# ─── Tests de format_docs ─────────────────────────────────────────────────────

class TestFormatDocs:
    """Verifica el formateador de fragmentos recuperados."""

    def test_empty_list_returns_no_info_message(self):
        result = format_docs([])
        assert result == "No hay información disponible en la base de conocimiento."

    def test_single_doc_without_page(self):
        doc = Document(page_content="Texto de prueba", metadata={"source": "guia.txt"})
        result = format_docs([doc])

        assert "[Fragmento 1 — guia.txt]" in result
        assert "Texto de prueba" in result

    def test_single_doc_with_page_number(self):
        doc = Document(
            page_content="Información financiera",
            metadata={"source": "informe.pdf", "page": 5},
        )
        result = format_docs([doc])
        assert "p.5" in result
        assert "informe.pdf" in result

    def test_multiple_docs_have_separator(self):
        docs = [
            Document(page_content="Primero", metadata={"source": "a.txt"}),
            Document(page_content="Segundo", metadata={"source": "b.txt"}),
        ]
        result = format_docs(docs)

        assert "[Fragmento 1 — a.txt]" in result
        assert "[Fragmento 2 — b.txt]" in result
        assert "---" in result  # separador entre fragmentos

    def test_doc_without_source_uses_unknown(self):
        doc = Document(page_content="Sin fuente", metadata={})
        result = format_docs([doc])
        assert "Desconocido" in result

    def test_page_zero_is_included(self):
        """page=0 es un valor válido (primera página en algunos loaders)."""
        doc = Document(page_content="Portada", metadata={"source": "doc.pdf", "page": 0})
        result = format_docs([doc])
        assert "p.0" in result


# ─── Tests de IngestService.__init__ ─────────────────────────────────────────

class TestIngestServiceInit:
    """Verifica que el servicio se inicialice correctamente."""

    def test_embeddings_and_pinecone_are_created(self, mock_settings):
        with (
            patch("app.services.ingest_service.HuggingFaceEmbeddings") as mock_emb,
            patch("app.services.ingest_service.Pinecone") as mock_pc,
            patch("app.services.ingest_service.PineconeVectorStore") as mock_pvs,
        ):
            mock_emb.return_value = MagicMock()
            mock_pc.return_value = MagicMock()
            mock_pvs.return_value = MagicMock()

            svc = IngestService(settings=mock_settings)

            mock_emb.assert_called_once()
            mock_pc.assert_called_once()
            mock_pvs.assert_called_once()
            assert svc._docs_metadata == {}

    def test_pinecone_initialized_with_correct_api_key(self, mock_settings):
        with (
            patch("app.services.ingest_service.HuggingFaceEmbeddings"),
            patch("app.services.ingest_service.Pinecone") as mock_pc,
            patch("app.services.ingest_service.PineconeVectorStore"),
        ):
            mock_pc.return_value = MagicMock()
            IngestService(settings=mock_settings)
            mock_pc.assert_called_once_with(api_key=mock_settings.pinecone_api_key)

    def test_text_splitter_configured_with_settings(self, ingest_service, mock_settings):
        assert ingest_service.text_splitter._chunk_size == mock_settings.chunk_size
        assert ingest_service.text_splitter._chunk_overlap == mock_settings.chunk_overlap


# ─── Tests de IngestService.get_documents / get_vector_store ─────────────────

class TestIngestServiceAccessors:
    def test_get_documents_empty_initially(self, ingest_service):
        assert ingest_service.get_documents() == []

    def test_get_vector_store_returns_pinecone_instance(self, ingest_service):
        vs = ingest_service.get_vector_store()
        assert vs is ingest_service.vector_store


# ─── Tests de IngestService.ingest_file ──────────────────────────────────────

class TestIngestServiceIngestFile:
    """Verifica el pipeline completo de ingestión de documentos."""

    def test_ingest_txt_returns_ingest_response(self, ingest_service):
        """Un archivo TXT válido debe retornar IngestResponse con datos correctos."""
        fake_docs = [Document(page_content="Contenido de texto", metadata={})]

        with patch("app.services.ingest_service.TextLoader") as mock_loader:
            mock_loader.return_value.load.return_value = fake_docs
            ingest_service.vector_store.add_documents = MagicMock()

            result = ingest_service.ingest_file(b"Contenido de texto", "reglamento.txt")

        assert isinstance(result, IngestResponse)
        assert result.filename == "reglamento.txt"
        assert result.chunks_created >= 1
        assert "exitosamente" in result.message

    def test_ingest_pdf_uses_pdf_loader(self, ingest_service):
        """Los archivos .pdf deben usar PyPDFLoader."""
        fake_docs = [Document(page_content="Contenido PDF", metadata={})]

        with patch("app.services.ingest_service.PyPDFLoader") as mock_loader:
            mock_loader.return_value.load.return_value = fake_docs
            ingest_service.vector_store.add_documents = MagicMock()

            result = ingest_service.ingest_file(b"%PDF-1.4 contenido", "manual.pdf")

        assert result.filename == "manual.pdf"
        mock_loader.assert_called_once()

    def test_ingest_unsupported_extension_raises_value_error(self, ingest_service):
        """Extensiones no soportadas deben lanzar ValueError con mensaje claro."""
        with pytest.raises(ValueError, match="no soportado"):
            ingest_service.ingest_file(b"contenido", "archivo.docx")

    def test_ingest_adds_source_metadata_to_chunks(self, ingest_service):
        """Cada fragmento debe llevar el metadato 'source' con el nombre del archivo."""
        fake_docs = [Document(page_content="Texto", metadata={})]
        captured_chunks = []

        def capture_docs(chunks):
            captured_chunks.extend(chunks)

        with patch("app.services.ingest_service.TextLoader") as mock_loader:
            mock_loader.return_value.load.return_value = fake_docs
            ingest_service.vector_store.add_documents = MagicMock(side_effect=capture_docs)

            ingest_service.ingest_file(b"Texto", "origen.txt")

        assert all(c.metadata.get("source") == "origen.txt" for c in captured_chunks)

    def test_ingest_registers_document_metadata(self, ingest_service):
        """Después de ingestar, el documento debe aparecer en get_documents()."""
        fake_docs = [Document(page_content="Texto", metadata={})]

        with patch("app.services.ingest_service.TextLoader") as mock_loader:
            mock_loader.return_value.load.return_value = fake_docs
            ingest_service.vector_store.add_documents = MagicMock()

            ingest_service.ingest_file(b"texto", "nuevo.txt")

        docs = ingest_service.get_documents()
        assert len(docs) == 1
        assert docs[0].filename == "nuevo.txt"
        assert isinstance(docs[0], DocumentInfo)

    def test_ingest_multiple_files_accumulates_metadata(self, ingest_service):
        """Ingestar dos archivos distintos debe registrar ambos."""
        fake_docs = [Document(page_content="X", metadata={})]

        with patch("app.services.ingest_service.TextLoader") as mock_loader:
            mock_loader.return_value.load.return_value = fake_docs
            ingest_service.vector_store.add_documents = MagicMock()

            ingest_service.ingest_file(b"a", "archivo1.txt")
            ingest_service.ingest_file(b"b", "archivo2.txt")

        docs = ingest_service.get_documents()
        filenames = {d.filename for d in docs}
        assert filenames == {"archivo1.txt", "archivo2.txt"}

    def test_ingest_loader_error_raises_runtime_error(self, ingest_service):
        """Si el loader lanza excepción, debe propagarse como RuntimeError."""
        with patch("app.services.ingest_service.TextLoader") as mock_loader:
            mock_loader.return_value.load.side_effect = Exception("disco lleno")

            with pytest.raises(RuntimeError, match="Error al procesar"):
                ingest_service.ingest_file(b"texto", "fallido.txt")

    def test_ingest_error_does_not_register_metadata(self, ingest_service):
        """Si hay error, el documento NO debe quedar registrado."""
        with patch("app.services.ingest_service.TextLoader") as mock_loader:
            mock_loader.return_value.load.side_effect = Exception("error")

            with pytest.raises(RuntimeError):
                ingest_service.ingest_file(b"texto", "fallido.txt")

        assert ingest_service.get_documents() == []
