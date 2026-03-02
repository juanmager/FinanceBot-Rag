"""
Evaluación de calidad del RAG con RAGAS.

¿Qué mide cada métrica?
────────────────────────────────────────────────────────────────────
faithfulness        → ¿La respuesta está basada en los documentos recuperados
                      o el LLM inventó información? (0 = inventó todo, 1 = 100% fiel)

answer_relevancy    → ¿La respuesta responde la pregunta del usuario?
                      (0 = irrelevante, 1 = completamente relevante)

context_precision   → De los chunks recuperados, ¿cuántos eran realmente necesarios?
                      (evita ruido en el contexto)

context_recall      → ¿Se recuperaron todos los chunks necesarios para responder?
                      (detecta información perdida en retrieval)
────────────────────────────────────────────────────────────────────

Uso:
    python evaluation/run_evaluation.py

Requisitos:
    pip install ragas datasets
    El servidor NO necesita estar corriendo. Se llama al RAGService directamente.

Salida:
    evaluation/results/ragas_report_YYYY-MM-DD.json
    evaluation/results/ragas_report_YYYY-MM-DD.csv
"""

import json
import os
import sys
from datetime import date
from pathlib import Path

# Asegura que los imports de app/ funcionen desde cualquier directorio
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)  # .env se carga relativo al cwd

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from app.config import get_settings
from app.services.ingest_service import IngestService
from app.services.rag_service import RAGService

# ── Configuración ─────────────────────────────────────────────────────────────
DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def build_ragas_dataset(rag_service: RAGService) -> Dataset:
    """
    Ejecuta cada pregunta del dataset contra el RAGService y construye
    el formato que RAGAS espera: question / answer / contexts / ground_truth.
    """
    with open(DATASET_PATH, encoding="utf-8") as f:
        eval_data = json.load(f)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print(f"\n📋 Evaluando {len(eval_data)} preguntas...\n")

    for i, item in enumerate(eval_data, 1):
        question = item["question"]
        print(f"  [{i}/{len(eval_data)}] {question[:60]}...")

        # Llamar al RAGService directamente (sin HTTP)
        chat_response = rag_service.answer(question)

        # RAGAS espera contexts como lista de strings (los chunks recuperados)
        retrieved_contexts = [src.content for src in chat_response.sources]

        questions.append(question)
        answers.append(chat_response.answer)
        contexts.append(retrieved_contexts)
        ground_truths.append(item["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


def print_report(results: dict) -> None:
    """Imprime el reporte de métricas en consola."""
    print("\n" + "=" * 60)
    print("  RAGAS — Reporte de Calidad RAG")
    print("=" * 60)

    thresholds = {
        "faithfulness": 0.80,
        "answer_relevancy": 0.80,
        "context_precision": 0.70,
        "context_recall": 0.70,
    }

    for metric, threshold in thresholds.items():
        value = results.get(metric)
        if value is None:
            continue
        status = "✅" if value >= threshold else "⚠️ "
        bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        print(f"  {status} {metric:<22} {value:.3f}  [{bar}]  (umbral: {threshold})")

    print("=" * 60 + "\n")


def main() -> None:
    settings = get_settings()

    # Inicializar servicios (sin levantar FastAPI)
    print("🔌 Conectando a Pinecone...")
    ingest_service = IngestService(settings=settings)
    vector_store = ingest_service.get_vector_store()
    rag_service = RAGService(vector_store=vector_store, settings=settings)

    # Construir dataset ejecutando todas las preguntas
    dataset = build_ragas_dataset(rag_service)

    # Ejecutar evaluación RAGAS
    print("\n🧠 Ejecutando evaluación RAGAS (requiere llamadas a Gemini)...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    # Mostrar reporte
    results_dict = result.to_pandas().mean(numeric_only=True).to_dict()
    print_report(results_dict)

    # Guardar resultados
    today = date.today().isoformat()
    json_path = RESULTS_DIR / f"ragas_report_{today}.json"
    csv_path = RESULTS_DIR / f"ragas_report_{today}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    result.to_pandas().to_csv(csv_path, index=False)

    print(f"📄 Resultados guardados en:")
    print(f"   {json_path}")
    print(f"   {csv_path}\n")


if __name__ == "__main__":
    main()
