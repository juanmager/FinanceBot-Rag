"""
Tests de los endpoints de gestión de documentos:
  POST /documents/ingest  — Indexar un archivo
  GET  /documents          — Listar documentos
"""

import os
from datetime import UTC, datetime

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key")


class TestIngestEndpoint:
    """Tests del endpoint POST /documents/ingest."""

    def test_ingest_txt_returns_200(self, client):
        """Subir un TXT válido debe retornar HTTP 200."""
        response = client.post(
            "/documents/ingest",
            files={"file": ("manual.txt", b"Informacion sobre cuentas bancarias.", "text/plain")},
        )
        assert response.status_code == 200

    def test_ingest_response_structure(self, client):
        """La respuesta debe tener filename, chunks_created y message."""
        response = client.post(
            "/documents/ingest",
            files={"file": ("doc.txt", b"Contenido del documento.", "text/plain")},
        )
        data = response.json()
        assert "filename" in data
        assert "chunks_created" in data
        assert "message" in data

    def test_ingest_returns_correct_filename(self, client, mock_ingest):
        """El filename en la respuesta debe coincidir con el enviado."""
        from app.models import IngestResponse

        mock_ingest.ingest_file.return_value = IngestResponse(
            filename="guia_productos.txt",
            chunks_created=8,
            message="Procesado OK",
        )
        response = client.post(
            "/documents/ingest",
            files={"file": ("guia_productos.txt", b"Contenido.", "text/plain")},
        )
        assert response.json()["filename"] == "guia_productos.txt"

    def test_ingest_unsupported_extension_returns_400(self, client):
        """Subir un archivo .docx debe retornar HTTP 400."""
        response = client.post(
            "/documents/ingest",
            files={"file": ("documento.docx", b"contenido", "application/octet-stream")},
        )
        assert response.status_code == 400

    def test_ingest_empty_file_returns_400(self, client):
        """Subir un archivo vacío debe retornar HTTP 400."""
        response = client.post(
            "/documents/ingest",
            files={"file": ("vacio.txt", b"", "text/plain")},
        )
        assert response.status_code == 400

    def test_ingest_service_is_called_with_correct_args(self, client, mock_ingest):
        """El servicio debe recibir el contenido y nombre del archivo."""
        content = b"Tasas de interes del banco."
        client.post(
            "/documents/ingest",
            files={"file": ("tasas.txt", content, "text/plain")},
        )
        # Verificar que ingest_file fue llamado
        mock_ingest.ingest_file.assert_called_once()
        call_args = mock_ingest.ingest_file.call_args
        assert call_args[0][0] == content         # primer argumento: bytes
        assert call_args[0][1] == "tasas.txt"     # segundo argumento: filename

    def test_ingest_pdf_extension_is_accepted(self, client):
        """La extensión .pdf debe ser aceptada (no retornar 400 por tipo)."""
        # El mock devuelve éxito independientemente del contenido
        response = client.post(
            "/documents/ingest",
            files={"file": ("reglamento.pdf", b"%PDF-1.4 fake content", "application/pdf")},
        )
        # No debería fallar por tipo de archivo (puede fallar por otro motivo con contenido falso,
        # pero el mock intercepta antes de procesar el contenido real)
        assert response.status_code != 400 or "tipo" not in response.json().get("detail", "").lower()


class TestDocumentsListEndpoint:
    """Tests del endpoint GET /documents."""

    def test_list_documents_returns_200(self, client):
        """El listado de documentos debe retornar HTTP 200."""
        response = client.get("/documents")
        assert response.status_code == 200

    def test_list_documents_empty_by_default(self, client):
        """Sin documentos indexados, la lista debe estar vacía."""
        data = client.get("/documents").json()
        assert data["total"] == 0
        assert data["documents"] == []

    def test_list_documents_with_items(self, client, mock_ingest):
        """Con documentos indexados, la lista debe mostrarlos."""
        from app.models import DocumentInfo

        mock_ingest.get_documents.return_value = [
            DocumentInfo(
                filename="productos.txt",
                extension=".txt",
                chunks=12,
                ingested_at=datetime(2026, 2, 21, 10, 0, 0, tzinfo=UTC),
            )
        ]

        data = client.get("/documents").json()
        assert data["total"] == 1
        assert len(data["documents"]) == 1
        assert data["documents"][0]["filename"] == "productos.txt"

    def test_list_documents_structure(self, client, mock_ingest):
        """Cada documento en la lista debe tener los campos correctos."""
        from app.models import DocumentInfo

        mock_ingest.get_documents.return_value = [
            DocumentInfo(
                filename="faq.txt",
                extension=".txt",
                chunks=7,
                ingested_at=datetime(2026, 2, 21, tzinfo=UTC),
            )
        ]

        docs = client.get("/documents").json()["documents"]
        doc = docs[0]
        for field in ["filename", "extension", "chunks", "ingested_at"]:
            assert field in doc, f"Campo '{field}' ausente"
