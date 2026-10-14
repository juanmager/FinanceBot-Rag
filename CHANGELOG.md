# Changelog

Registro de cambios (código, infra y GCP). Formato: fecha · qué · por qué.

## 2026-10-14
- **Tests**: CrossEncoder mockeado en `tests/test_rag_service.py` (antes descargaba ~2 GB; 110 s → 26 s). Test de k actualizado a `k * multiplier`; test nuevo de reordenamiento por reranker.
- **rag_service**: no llama al reranker si Pinecone no devuelve candidatos.
- **Git**: cambios de Runs 17–20 commiteados en rama `feature/reranker-e5-ragas` (no en `main`: push a `main` = deploy a Cloud Run, y los modelos nuevos no entran en 1 GiB).
- **GCP / Artifact Registry**: borradas 3 imágenes viejas (20,5 GB → 5,4 GB; ~US$2/mes → ~US$0,50). Queda solo `latest` (`9c0a900`, la que sirve Cloud Run).
- **GCP / Artifact Registry**: cleanup policy `keep-last-2` en `financebot-repo` (evita que se acumulen imágenes).
