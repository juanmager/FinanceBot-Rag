"""
Configuración centralizada de la aplicación usando pydantic-settings.
Las variables se leen desde el archivo .env en la raíz del proyecto.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Ajustes de la aplicación. Todos los valores pueden sobreescribirse con variables de entorno."""

    # ── Google Gemini ────────────────────────────
    google_api_key: str
    gemini_model: str = "gemini-2.0-flash"
    max_tokens: int = 2048

    # ── App ────────────────────────────────────────────────────
    app_name: str = "FinanceBot — Asistente Financiero RAG"
    app_version: str = "1.0.0"
    debug: bool = False

    # ── RAG / Chunking ─────────────────────────────────────────
    chunk_size: int = 1000          # Tamaño de cada fragmento en caracteres
    chunk_overlap: int = 200        # Superposición entre fragmentos para mantener contexto
    retriever_k: int = 4            # Número de fragmentos a recuperar por consulta

    # ── Pinecone ────────────────────────────────────────────────
    pinecone_api_key: str
    pinecone_index_name: str = "financebot-rag"
    pinecone_namespace: str = "financial_docs"

    # ── Embeddings ─────────────────────────────────────────────
    # Modelo liviano (~90MB) que corre localmente sin API key adicional
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Retorna instancia cacheada de Settings (se carga una sola vez)."""
    return Settings()
