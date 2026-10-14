"""
Calibra `min_rerank_score`: compara el mejor score del reranker en preguntas
del dominio (eval_dataset.json) vs preguntas fuera de tema.
Solo usa Pinecone + CrossEncoder (no llama a Gemini → costo cero).

Uso:
    python evaluation/calibrate_threshold.py
"""

import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from sentence_transformers import CrossEncoder  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.services.ingest_service import IngestService  # noqa: E402

OFF_TOPIC = [
    "¿Cuál es la capital de Francia?",
    "¿Cómo se hace una torta de chocolate?",
    "¿Quién ganó el mundial 2022?",
    "¿Qué tasa da un plazo fijo en Banco Nación?",
    "¿Cómo configuro el wifi de mi casa?",
    "¿Cuánto cuesta un pasaje a Córdoba?",
]


def top_score(vs, reranker, k: int, question: str) -> float:
    cands = vs.similarity_search_with_score(question, k=k)
    if not cands:
        return 0.0
    return float(max(reranker.predict([(question, d.page_content) for d, _ in cands])))


def main() -> None:
    settings = get_settings()
    vs = IngestService(settings=settings).vector_store
    reranker = CrossEncoder(settings.reranker_model)
    k = settings.retriever_k * settings.reranker_candidates_multiplier

    data = json.loads((Path(__file__).parent / "eval_dataset.json").read_text(encoding="utf-8"))
    in_scores = sorted(top_score(vs, reranker, k, d["question"]) for d in data)
    off_scores = sorted(top_score(vs, reranker, k, q) for q in OFF_TOPIC)

    print(f"En dominio  (n={len(in_scores)}): min={in_scores[0]:.3f}  p10={in_scores[len(in_scores)//10]:.3f}")
    print(f"Fuera tema  (n={len(off_scores)}): max={off_scores[-1]:.3f}  todos={[round(s, 3) for s in off_scores]}")
    print(f"Más bajos en dominio: {[round(s, 3) for s in in_scores[:5]]}")


if __name__ == "__main__":
    main()
