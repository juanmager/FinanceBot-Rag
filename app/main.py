"""
Punto de entrada de la aplicación FastAPI.

Patrón de diseño clave: create_app(lifespan_fn=None)
  - En producción: usa default_lifespan que crea los servicios reales (IngestService + RAGService)
  - En tests: recibe un lifespan mockeado que inyecta servicios falsos sin tocar Gemini ni Pinecone
"""

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.routes import chat, documents, health

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan de producción ─────────────────────────────────────────────────────
@asynccontextmanager
async def default_lifespan(app: FastAPI):
    """
    Inicializa los servicios al arrancar la app y los limpia al cerrar.
    Este proceso puede tardar 30-60 segundos la primera vez (descarga del modelo de embeddings).
    """
    settings = get_settings()
    logger.info(f"🚀 Iniciando {settings.app_name} v{settings.app_version}")

    # Importación diferida para evitar carga innecesaria en tests
    from app.services.ingest_service import IngestService
    from app.services.rag_service import RAGService

    logger.info("⏳ Cargando modelo de embeddings y vector store (puede tardar en el primer inicio)...")
    ingest_svc = IngestService(settings)
    rag_svc = RAGService(ingest_svc.get_vector_store(), settings)

    # Inyectar servicios en el estado de la app (accesible desde los endpoints via request.app.state)
    app.state.ingest_service = ingest_svc
    app.state.rag_service = rag_svc

    logger.info("✅ Servicios listos. API disponible.")
    yield

    logger.info("🛑 Cerrando FinanceBot RAG API...")


# ── Factory ────────────────────────────────────────────────────────────────────
def create_app(lifespan_fn: Callable | None = None) -> FastAPI:
    """
    Factory de la aplicación FastAPI.

    Args:
        lifespan_fn: Función de lifespan opcional. Si es None, usa default_lifespan.
                     Pasá un lifespan mockeado en tests para evitar cargar modelos reales.
    """
    settings = get_settings()
    _lifespan = lifespan_fn if lifespan_fn is not None else default_lifespan

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "## 🏦 FinanceBot — Asistente Financiero con RAG\n\n"
            "Sistema de **Retrieval-Augmented Generation** construido con LangChain, "
            "Pinecone y el modelo **Gemini** de Google.\n\n"
            "### Flujo de uso\n"
            "1. **Indexar documentos** → `POST /documents/ingest` (PDF o TXT)\n"
            "2. **Consultar al asistente** → `POST /chat/ask`\n"
            "3. **Ver documentos indexados** → `GET /documents`\n\n"
            "### Tecnologías\n"
            "- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local, sin API key extra)\n"
            "- **Vector Store**: Pinecone (serverless, persistido en la nube)\n"
            "- **LLM**: Gemini 2.5 Flash (Google)\n"
            "- **Chunking**: RecursiveCharacterTextSplitter (LangChain)\n"
        ),
        contact={"name": "AI Engineer Portfolio"},
        license_info={"name": "MIT"},
        lifespan=_lifespan,
    )

    # ── CORS ──
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ──
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(chat.router)

    # ── Handler global de excepciones no capturadas ──
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Error no manejado en {request.url}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "InternalServerError", "detail": "Error interno del servidor."},
        )

    return app


# Instancia principal (usada por uvicorn)
app = create_app()
