"""
Endpoints de gestión de documentos.
  POST /documents/ingest  — Subir e indexar un archivo PDF o TXT
  GET  /documents          — Listar documentos en la base de conocimiento
"""

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.models import DocumentsResponse, ErrorResponse, IngestResponse

router = APIRouter(prefix="/documents", tags=["Documents"])

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def _get_ingest_service(request: Request):
    """Obtiene el IngestService desde el estado de la app (inyectado en lifespan)."""
    return request.app.state.ingest_service


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingestar un documento",
    description=(
        "Sube un archivo PDF o TXT, lo divide en fragmentos, genera embeddings "
        "y los almacena en ChromaDB para ser usados como contexto en las respuestas del chatbot."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Tipo de archivo no soportado o archivo vacío"},
        413: {"model": ErrorResponse, "description": "Archivo demasiado grande (máx. 20MB)"},
        500: {"model": ErrorResponse, "description": "Error al procesar el documento"},
    },
)
async def ingest_document(
    request: Request,
    file: UploadFile = File(..., description="Archivo PDF o TXT a indexar"),
) -> IngestResponse:
    """
    Procesa e indexa el archivo subido en el vector store.
    Los documentos indexados se usarán automáticamente en las respuestas del chatbot.
    """
    # Validar extensión
    filename = file.filename or "archivo_sin_nombre.txt"
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tipo de archivo '{ext}' no soportado. Formatos válidos: PDF, TXT.",
        )

    # Leer contenido y validar tamaño
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo está vacío.",
        )
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"El archivo supera el límite de {MAX_FILE_SIZE_MB}MB.",
        )

    service = _get_ingest_service(request)

    try:
        return service.ingest_file(content, filename)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al procesar el documento: {str(e)}",
        )


@router.get(
    "",
    response_model=DocumentsResponse,
    status_code=status.HTTP_200_OK,
    summary="Listar documentos indexados",
    description=(
        "Retorna la lista de documentos que fueron procesados e indexados "
        "en el vector store durante la sesión actual."
    ),
)
def list_documents(request: Request) -> DocumentsResponse:
    """Lista todos los documentos en la base de conocimiento."""
    service = _get_ingest_service(request)
    docs = service.get_documents()
    return DocumentsResponse(total=len(docs), documents=docs)
