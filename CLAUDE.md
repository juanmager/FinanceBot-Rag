# FinanceBot RAG

RAG (FastAPI + Gemini + Pinecone + e5 + reranker bge) sobre TyC de Ualá. Proyecto personal de portfolio.

## Reglas
- Todo cambio (código, infra, GCP) se registra en `CHANGELOG.md`: una línea por cambio, qué y por qué.
- Costo mínimo: preferir free tiers; nada que suba la factura de GCP sin avisar.
- Push a `main` dispara deploy a Cloud Run (`.github/workflows/ci.yml`). Trabajar en ramas.
- Garantías en código, criterio en el prompt (lineamiento CCA). Tests: `pytest tests -q` con todo mockeado.
- Evaluación: `evaluation/run_evaluation.py` (~30 min, Gemini free tier). Historial de runs en `MASTER_LOG_2026-03-26.md`.
