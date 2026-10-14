"""
Análisis por pregunta de un reporte RAGAS (CCA D5: los promedios esconden fallos).
Muestra las preguntas con peor recall y peor faithfulness, y clasifica la causa:
  - recall bajo      → problema de RETRIEVAL (no se trajo el chunk correcto)
  - faithfulness baja → problema de GENERACIÓN (el LLM agregó datos)

Uso:
    python evaluation/analyze_results.py [ruta_csv]   (default: el CSV más reciente)
"""

import csv
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
METRICS = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for m in METRICS:
            try:
                r[m] = float(r[m])
            except (KeyError, ValueError):
                r[m] = None
    return rows


def worst(rows: list[dict], metric: str, n: int = 8) -> list[dict]:
    return sorted((r for r in rows if r[metric] is not None), key=lambda r: r[metric])[:n]


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else max(RESULTS_DIR.glob("ragas_report_*.csv"))
    rows = load(path)
    print(f"Reporte: {path.name} ({len(rows)} preguntas)\n")

    for metric, cause in [("context_recall", "RETRIEVAL"), ("faithfulness", "GENERACIÓN")]:
        print(f"── Peor {metric} (causa probable: {cause}) ──")
        for r in worst(rows, metric):
            print(f"  {r[metric]:.2f}  {r['user_input'][:90]}")
        print()

    retrieval = sum(1 for r in rows if (r["context_recall"] or 0) < 0.5)
    generation = sum(1 for r in rows if (r["context_recall"] or 0) >= 0.5 and (r["faithfulness"] or 0) < 0.7)
    print(f"Fallos de retrieval (recall < 0.5): {retrieval}")
    print(f"Fallos de generación (recall ok, faithfulness < 0.7): {generation}")


if __name__ == "__main__":
    main()
