# ── Imagen base ────────────────────────────────────────────────────────────────
FROM python:3.12-slim

LABEL maintainer="AI Engineer Portfolio"
LABEL description="FinanceBot — RAG Chatbot con LangChain + Pinecone + Gemini"

# ── Variables del sistema ──────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Directorio de caché de Hugging Face (dentro del contenedor)
ENV HF_HOME=/app/.cache/huggingface

# ── Directorio de trabajo ──────────────────────────────────────────────────────
WORKDIR /app

# ── Dependencias del sistema (necesarias para sentence-transformers) ───────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencias Python ────────────────────────────────────────────────────────
# Se copian primero para aprovechar el cache de capas de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Pre-descargar el modelo de embeddings (evita la demora en el primer request) ─
# Esta capa se cachea en el build: si requirements.txt no cambia, no se vuelve a descargar
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
print('Descargando modelo de embeddings...'); \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
print('Modelo descargado OK.')"

# ── Copiar código fuente y datos ───────────────────────────────────────────────
COPY app/ ./app/
COPY data/ ./data/

# ── Usuario sin privilegios (seguridad) ───────────────────────────────────────
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Puerto ────────────────────────────────────────────────────────────────────
# Cloud Run inyecta la variable de entorno PORT en tiempo de ejecución.
# El valor por defecto 8000 aplica para desarrollo local.
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]

# ── Comando de inicio ─────────────────────────────────────────────────────────
# Cloud Run sobreescribe $PORT dinámicamente; localmente usa 8000 por defecto.
#CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
