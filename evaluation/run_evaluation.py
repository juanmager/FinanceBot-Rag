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
    python evaluation/run_evaluation.py [--use-cache]   (--use-cache: reusa respuestas ya recolectadas)

Requisitos:
    pip install ragas datasets
    El servidor NO necesita estar corriendo. Se llama al RAGService directamente.

Salida:
    evaluation/results/ragas_report_YYYY-MM-DD.json
    evaluation/results/ragas_report_YYYY-MM-DD.csv
"""

import json
import math
import os
import sys
import time
from datetime import date
from pathlib import Path

# Asegura que los imports de app/ funcionen desde cualquier directorio
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)  # .env se carga relativo al cwd

from google import genai as google_genai
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.embeddings.base import LangchainEmbeddingsWrapper  # necesario: answer_relevancy llama embed_query (BaseRagasEmbeddings)
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._faithfulness import faithfulness

from app.config import get_settings
from app.services.ingest_service import IngestService
from app.services.rag_service import RAGService

# Delay entre preguntas para respetar el rate limit de Gemini free tier (5 RPM)
REQUEST_DELAY_SECONDS = 13

# ── Configuración ─────────────────────────────────────────────────────────────
DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DATASET_CACHE_PATH = RESULTS_DIR / "eval_dataset_cache.json"


def build_ragas_dataset(rag_service: RAGService) -> Dataset:
    """
    Ejecuta cada pregunta del dataset contra el RAGService y construye
    el formato que RAGAS 0.4.x espera:
      user_input / response / retrieved_contexts / reference
    """
    with open(DATASET_PATH, encoding="utf-8") as f:
        eval_data = json.load(f)

    user_inputs = []
    responses = []
    retrieved_contexts = []
    references = []

    print(f"\n📋 Evaluando {len(eval_data)} preguntas...\n")

    for i, item in enumerate(eval_data, 1):
        question = item["question"]
        print(f"  [{i}/{len(eval_data)}] {question[:60]}...")

        # Llamar al RAGService directamente (sin HTTP)
        chat_response = rag_service.answer(question)

        if i < len(eval_data):
            print(f"  ⏳ Esperando {REQUEST_DELAY_SECONDS}s...")
            time.sleep(REQUEST_DELAY_SECONDS)

        # RAGAS 0.4.x: retrieved_contexts es lista de strings.
        # Se usan los chunks COMPLETOS que vio el LLM (antes: previews de 300 chars
        # deduplicados por pagina, lo que subestimaba faithfulness y recall).
        contexts = chat_response.contexts

        user_inputs.append(question)
        responses.append(chat_response.answer)
        retrieved_contexts.append(contexts)
        references.append(item["ground_truth"])

    # RAGAS 0.4.x usa user_input / response / retrieved_contexts / reference
    dataset = Dataset.from_dict({
        "user_input": user_inputs,
        "response": responses,
        "retrieved_contexts": retrieved_contexts,
        "reference": references,
    })

    # Guardar cache con los nombres correctos de RAGAS 0.4.x
    cache_data = {
        "user_input": user_inputs,
        "response": responses,
        "retrieved_contexts": retrieved_contexts,
        "reference": references,
    }
    with open(DATASET_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Dataset cacheado en: {DATASET_CACHE_PATH}")

    return dataset


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
        if value is None or (isinstance(value, float) and math.isnan(value)):
            print(f"  ❓ {metric:<22} N/A   (evaluación fallida)")
            continue
        status = "✅" if value >= threshold else "⚠️ "
        bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        print(f"  {status} {metric:<22} {value:.3f}  [{bar}]  (umbral: {threshold})")

    print("=" * 60 + "\n")


def main() -> None:
    settings = get_settings()

    # Cache solo con --use-cache: por defecto se recolecta de nuevo, para que un cambio
    # de código/config no se evalúe en silencio con respuestas viejas.
    if "--use-cache" in sys.argv and DATASET_CACHE_PATH.exists():
        print(f"\n📦 Cargando dataset desde cache: {DATASET_CACHE_PATH}")
        with open(DATASET_CACHE_PATH, encoding="utf-8") as f:
            cached = json.load(f)
        # Migrar cache viejo (columnas RAGAS < 0.2) al formato RAGAS 0.4.x
        if "question" in cached:
            cached = {
                "user_input": cached["question"],
                "response": cached["answer"],
                "retrieved_contexts": cached["contexts"],
                "reference": cached["ground_truth"],
            }
        dataset = Dataset.from_dict(cached)
    else:
        # Inicializar servicios (sin levantar FastAPI)
        print("🔌 Conectando a Pinecone...")
        ingest_service = IngestService(settings=settings)
        vector_store = ingest_service.get_vector_store()
        rag_service = RAGService(vector_store=vector_store, settings=settings)
        dataset = build_ragas_dataset(rag_service)

    # Configurar LLM y embeddings para RAGAS
    EVALUATOR_MODEL = settings.gemini_model
    print(f"\n🧠 Ejecutando evaluación RAGAS con {EVALUATOR_MODEL} como evaluador...")

    # Problema 1 — LLM: LangchainLLMWrapper sigue siendo necesario.
    # llm_factory(provider="google") usa el adapter instructor, que tiene un bug conocido
    # con Gemini (envía HARM_CATEGORY_JAILBREAK inválido → CancelledError en todos los jobs).
    # Ref: https://github.com/567-labs/instructor/issues/1658
    # La alternativa litellm requeriría instalar el paquete (no disponible en este env).
    # Se suprime el DeprecationWarning hasta que el bug upstream esté resuelto.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        evaluator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
            model=EVALUATOR_MODEL,
            google_api_key=settings.google_api_key,
            temperature=0,
        ))

    # Problema 1 — Embeddings: LangchainEmbeddingsWrapper es necesario a pesar de estar deprecado.
    # GoogleEmbeddings (nueva API) implementa BaseRagasEmbedding con embed_text/aembed_text,
    # pero answer_relevancy internamente llama embed_query (método de BaseRagasEmbeddings legacy).
    # Hasta que RAGAS normalice la interfaz, esta es la única ruta funcional para 4 métricas.
    # Se suprime el warning igual que el LLM.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        evaluator_embeddings = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.google_api_key,
        ))

    # Problema 2 — Concurrencia: max_workers=3 reduce saturación con Gemini.
    # timeout=360: 6 min por evaluación (Gemini 2.5 Flash ~50-120s con reasoning).
    # max_wait=120: backoff exponencial máximo entre reintentos (existe en RAGAS 0.4.3).
    from ragas.run_config import RunConfig
    run_config = RunConfig(max_workers=3, timeout=360, max_retries=3, max_wait=120)

    # Problema 3 — answer_relevancy activado DESPUÉS del fix de concurrencia (P2).
    # Requiere embeddings=evaluator_embeddings para calcular similitud semántica entre
    # la pregunta original y las preguntas regeneradas desde la respuesta.
    # batch_size=10: procesa el dataset en lotes de 10 para mayor estabilidad.
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_precision, context_recall, answer_relevancy],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
        raise_exceptions=False,
        batch_size=10,
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
