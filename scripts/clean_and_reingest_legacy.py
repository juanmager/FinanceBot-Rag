"""
Limpia el índice Pinecone y reindexa todos los documentos desde cero.
Ejecutar cuando haya duplicados o para un reset completo.

Uso (con el servidor DETENIDO):
    python clean_and_reingest.py
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_DIRS = [
    Path("data/tyc_ualá"),
    Path("data/sample_docs"),
]
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def main() -> None:
    from app.config import get_settings
    from app.services.ingest_service import IngestService
    from pinecone import Pinecone

    settings = get_settings()

    # ── Paso 1: Limpiar el namespace en Pinecone ───────────────────────────────
    logger.info(f"Conectando a Pinecone (índice: {settings.pinecone_index_name})...")
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    stats = index.describe_index_stats()
    namespace = settings.pinecone_namespace
    ns_stats = stats.namespaces.get(namespace)
    vector_count = ns_stats.vector_count if ns_stats else 0

    logger.info(f"Vectores actuales en namespace '{namespace}': {vector_count}")

    if vector_count > 0:
        logger.info("Eliminando todos los vectores del namespace...")
        index.delete(delete_all=True, namespace=namespace)
        logger.info("✅ Namespace limpio.")
    else:
        logger.info("El namespace ya está vacío.")

    # ── Paso 2: Reindexar desde cero ──────────────────────────────────────────
    logger.info("Inicializando IngestService...")
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

    logger.info(f"\n🎉 Listo: {total_chunks} chunks indexados en Pinecone (sin duplicados).")


if __name__ == "__main__":
    main()
