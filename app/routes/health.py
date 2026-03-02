"""
Health check endpoint.
No depende de la API de Anthropic ni del vector store.
"""

from datetime import UTC, datetime

from fastapi import APIRouter

from app.config import get_settings
from app.models import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    response_model=HealthResponse,
    summary="Health Check",
    description="Verifica que el servicio está activo. No consume tokens ni accede al vector store.",
)
def health_check() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        timestamp=datetime.now(UTC),
    )
