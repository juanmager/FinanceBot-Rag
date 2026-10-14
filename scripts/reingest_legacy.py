"""
Script de re-ingestión: sube todos los documentos de /data a Pinecone.
Ejecutar una sola vez después de migrar de ChromaDB a Pinecone.

Uso:
    python reingest.py
    (con el servidor DETENIDO — accede a Pinecone directamente sin FastAPI)
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Directorios de documentos ──────────────────────────────────────────────────
DATA_DIRS = [
    Path("data/tyc_ualá"),
    Path("data/sample_docs"),
]

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def main() -> None:
    from app.services.ingest_service import IngestService

    logger.info("Iniciando IngestService (conectando a Pinecone)...")
    svc = IngestService()

    archivos = []
    for d in DATA_DIRS:
        if d.exists():
            archivos.extend([f for f in d.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS])

    if not archivos:
        logger.warning("No se encontraron archivos para indexar.")
        sys.exit(0)

    logger.info(f"Archivos a indexar: {len(archivos)}")
    total_chunks = 0

    for archivo in archivos:
        try:
            content = archivo.read_bytes()
            result = svc.ingest_file(content, archivo.name)
            total_chunks += result.chunks_created
            logger.info(f"  ✅ {archivo.name}: {result.chunks_created} chunks")
        except Exception as e:
            logger.error(f"  ❌ {archivo.name}: {e}")

    logger.info(f"\n🎉 Re-ingestión completa: {total_chunks} chunks indexados en Pinecone.")


if __name__ == "__main__":
    main()
