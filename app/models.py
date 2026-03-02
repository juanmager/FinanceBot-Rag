"""
Esquemas Pydantic centralizados para requests y responses de la API.
"""

from datetime import datetime

from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# HEALTH
# ──────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    service: str
    version: str
    timestamp: datetime


# ──────────────────────────────────────────────
# INGEST (carga de documentos)
# ──────────────────────────────────────────────

class IngestResponse(BaseModel):
    """Respuesta al ingestar un documento."""

    filename: str = Field(..., description="Nombre del archivo procesado")
    chunks_created: int = Field(..., description="Cantidad de fragmentos generados")
    message: str = Field(..., description="Mensaje de resultado")


class DocumentInfo(BaseModel):
    """Metadatos de un documento ya ingestado."""

    filename: str
    extension: str = Field(..., description="Extensión del archivo (.pdf o .txt)")
    chunks: int = Field(..., description="Fragmentos almacenados en el vector store")
    ingested_at: datetime


class DocumentsResponse(BaseModel):
    """Listado de documentos en la base de conocimiento."""

    total: int
    documents: list[DocumentInfo]


# ──────────────────────────────────────────────
# CHAT / RAG
# ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Payload para el endpoint de chat."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Pregunta del cliente al asistente financiero",
        examples=["¿Cuáles son los requisitos para abrir una cuenta?"],
    )


class SourceDocument(BaseModel):
    """Fragmento de documento utilizado como contexto para la respuesta."""

    content: str = Field(..., description="Extracto del fragmento (máx. 300 caracteres)")
    source: str = Field(..., description="Nombre del archivo de origen")
    page: int | None = Field(default=None, description="Número de página (solo para PDFs)")
    similarity_score: float | None = Field(
        default=None,
        description="Score de similitud coseno devuelto por Pinecone (0 a 1). Mayor valor = más relevante.",
    )


class ChatResponse(BaseModel):
    """Respuesta del asistente financiero RAG."""

    question: str = Field(..., description="Pregunta original")
    answer: str = Field(..., description="Respuesta generada por FinanceBot")
    sources: list[SourceDocument] = Field(..., description="Documentos usados como contexto")
    model: str = Field(..., description="Modelo de Gemini utilizado")
    timestamp: datetime = Field(..., description="Fecha y hora de la respuesta")


# ──────────────────────────────────────────────
# ERROR
# ──────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
