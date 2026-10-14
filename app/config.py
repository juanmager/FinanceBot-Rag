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
    gemini_model: str = "gemini-3.8-flash"
    max_tokens: int = 2048

    # ── App ────────────────────────────────────────────────────
    app_name: str = "FinanceBot — Asistente Financiero RAG"
    app_version: str = "1.0.0"
    debug: bool = False

    # ── RAG / Chunking ─────────────────────────────────────────
    # RUN 18 (OFAT): Revert chunk_size. Testing System Prompt only.
    chunk_size: int = 400           # REVERTED to baseline (Run 17 failed)
    chunk_overlap: int = 80         # ~20% overlap for continuity
    retriever_k: int = 10           # Baseline: retrieve 10 candidates

    # ── Pinecone ────────────────────────────────────────────────
    pinecone_api_key: str
    pinecone_index_name: str = "financebot-rag"
    pinecone_namespace: str = "financial_docs"

    # ── Embeddings ─────────────────────────────────────────────
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_dimension: int = 1024

    # ── Reranking ──────────────────────────────────────────────
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_candidates_multiplier: int = 2  # Baseline multiplier
    # Si el mejor score del reranker queda por debajo -> fallback sin llamar al LLM.
    # Calibrado con evaluation/calibrate_threshold.py (14/10/2026): fuera de tema <= 0.001
    # (salvo "plazo fijo Banco Nación" 0.395), en dominio min 0.245.
    min_rerank_score: float = 0.1

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Retorna instancia cacheada de Settings (se carga una sola vez)."""
    return Settings()
