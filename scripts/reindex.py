"""
Script de re-indexación completa de Pinecone.
Pasos:
  1. Elimina todos los vectores del índice actual
  2. Re-indexa todos los documentos fuente con los parámetros de chunking actuales
     (chunk_size=800, chunk_overlap=150)

Documentos incluidos:
  - data/sample_docs/      : 2 docs base (preguntas y productos)
  - data/tyc_uala_scraped/ : 14 PDFs oficiales scraped de Ualá (TyC, préstamos,
                             refinanciación, cuotificaciones, caja de ahorros, etc.)
"""

import sys
import time
from pathlib import Path

# Agregar el root del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.services.ingest_service import IngestService

# ── Documentos a re-indexar ────────────────────────────────────────────────────
DOCS_TO_REINDEX = [
    # Documentos base
    Path("data/sample_docs/preguntas_frecuentes.txt"),
    Path("data/sample_docs/productos_financieros.txt"),
    # Documentos scraped de Ualá (13.6 MB, versiones oficiales)
    Path("data/tyc_uala_scraped/tyc_uala_app_prepaga_bis_diciembre_2025.pdf"),
    Path("data/tyc_uala_scraped/terminos_condiciones_uala_modificaciones_octubre_2025.pdf"),
    Path("data/tyc_uala_scraped/terminos_condiciones_banco_caja_ahorros.pdf"),
    Path("data/tyc_uala_scraped/tyc_uala_bank_prestamo_personal_diciembre_2025.pdf"),
    Path("data/tyc_uala_scraped/ar_contratos_tyc_alau_prestamos.pdf"),
    Path("data/tyc_uala_scraped/ar_contratos_resumen_tyc_prestamos.pdf"),
    Path("data/tyc_uala_scraped/ar_tyc_refinanciacion.pdf"),
    Path("data/tyc_uala_scraped/ar_tyc_refinanciacion_resumen.pdf"),
    Path("data/tyc_uala_scraped/alau_tyc_cuotificaciones_360_diciembre_2024.pdf"),
    Path("data/tyc_uala_scraped/resumen_oferta_cuotificaciones_360_diciembre_2024.pdf"),
    Path("data/tyc_uala_scraped/contrato_tc_web_enero_2026.pdf"),
    Path("data/tyc_uala_scraped/uala_pagos_tarjeta_celular.pdf"),
    Path("data/tyc_uala_scraped/codigo_practicas_bancarias.pdf"),
    Path("data/tyc_uala_scraped/locales_adheridos_promo35_uala_bis.pdf"),
]


def main() -> None:
    settings = get_settings()

    print("\n" + "=" * 60)
    print("  RE-INDEXACIÓN COMPLETA DE PINECONE")
    print("=" * 60)
    print(f"  chunk_size    : {settings.chunk_size}")
    print(f"  chunk_overlap : {settings.chunk_overlap}")
    print(f"  retriever_k   : {settings.retriever_k}")
    print(f"  índice        : {settings.pinecone_index_name}")
    print(f"  namespace     : {settings.pinecone_namespace}")
    print("=" * 60)

    # ── Paso 1: Limpiar índice existente ──────────────────────────────────────
    print("\n🗑️  Limpiando índice Pinecone existente...")
    from pinecone import Pinecone
    from pinecone.exceptions import NotFoundException
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    # Borrar todos los vectores del namespace (puede no existir en índice nuevo)
    try:
        index.delete(delete_all=True, namespace=settings.pinecone_namespace)
        print("  ✅ Índice limpiado correctamente.")
    except NotFoundException:
        print("  ℹ️  Namespace vacío (índice nuevo) — nada que limpiar.")

    # Esperar a que Pinecone propague la eliminación
    print("  ⏳ Esperando 5s para que Pinecone propague la eliminación...")
    time.sleep(5)

    # ── Paso 2: Re-indexar documentos ─────────────────────────────────────────
    print(f"\n📄 Re-indexando {len(DOCS_TO_REINDEX)} documentos...\n")
    ingest_service = IngestService(settings=settings)

    total_chunks = 0
    failed = []

    for i, doc_path in enumerate(DOCS_TO_REINDEX, 1):
        if not doc_path.exists():
            print(f"  [{i}/{len(DOCS_TO_REINDEX)}] ⚠️  No encontrado: {doc_path}")
            failed.append(str(doc_path))
            continue

        print(f"  [{i}/{len(DOCS_TO_REINDEX)}] 📄 Procesando: {doc_path.name}")
        try:
            file_content = doc_path.read_bytes()
            result = ingest_service.ingest_file(file_content, doc_path.name)
            total_chunks += result.chunks_created
            print(f"         ✅ {result.chunks_created} chunks indexados")
        except Exception as e:
            print(f"         ❌ Error: {e}")
            failed.append(str(doc_path))

    # ── Reporte final ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  REPORTE FINAL")
    print("=" * 60)
    print(f"  ✅ Documentos procesados : {len(DOCS_TO_REINDEX) - len(failed)}/{len(DOCS_TO_REINDEX)}")
    print(f"  📦 Total chunks creados  : {total_chunks}")
    if failed:
        print(f"  ❌ Fallidos              : {len(failed)}")
        for f in failed:
            print(f"     - {f}")
    print("=" * 60)
    print("\n🎯 Re-indexación completa. Podés correr la evaluación ahora:")
    print("   .venv\\Scripts\\python.exe evaluation\\run_evaluation.py\n")


if __name__ == "__main__":
    main()