"""
Endpoint de chat RAG.
  POST /chat/ask — Hacer una pregunta al asistente financiero FinanceBot
"""

import google.api_core.exceptions as google_exceptions
from fastapi import APIRouter, HTTPException, Request, status

from app.models import ChatRequest, ChatResponse, ErrorResponse

router = APIRouter(prefix="/chat", tags=["Chat"])


def _get_rag_service(request: Request):
    """Obtiene el RAGService desde el estado de la app."""
    return request.app.state.rag_service


@router.post(
    "/ask",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Hacer una pregunta al asistente financiero",
    description=(
        "Envía una pregunta al asistente **FinanceBot**. El sistema recupera los fragmentos "
        "más relevantes de los documentos indexados y genera una respuesta contextualizada "
        "usando el modelo Gemini de Google.\n\n"
        "**Importante:** primero debés indexar documentos usando `POST /documents/ingest`."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "API key de Gemini inválida"},
        429: {"model": ErrorResponse, "description": "Rate limit de Gemini excedido"},
        503: {"model": ErrorResponse, "description": "API de Gemini no disponible"},
        500: {"model": ErrorResponse, "description": "Error interno"},
    },
)
def ask_question(payload: ChatRequest, request: Request) -> ChatResponse:
    """
    Genera una respuesta basada en los documentos financieros indexados.
    Retorna la respuesta junto con los fragmentos de documentos usados como contexto.
    """
    service = _get_rag_service(request)

    try:
        return service.answer(payload.question)

    except google_exceptions.Unauthenticated as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"API key de Gemini inválida: {str(e)}",
        )
    except google_exceptions.ResourceExhausted as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit excedido. Intentá en unos segundos: {str(e)}",
        )
    except google_exceptions.ServiceUnavailable as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No se pudo conectar a la API de Gemini: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno: {str(e)}",
        )
