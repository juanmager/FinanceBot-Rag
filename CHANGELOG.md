# Changelog

Registro de cambios (código, infra y GCP). Formato: fecha · qué · por qué.

## 2026-10-14
- **Tests**: CrossEncoder mockeado en `tests/test_rag_service.py` (antes descargaba ~2 GB; 110 s → 26 s). Test de k actualizado a `k * multiplier`; test nuevo de reordenamiento por reranker.
- **rag_service**: no llama al reranker si Pinecone no devuelve candidatos.
- **Git**: cambios de Runs 17–20 commiteados en rama `feature/reranker-e5-ragas` (no en `main`: push a `main` = deploy a Cloud Run, y los modelos nuevos no entran en 1 GiB).
- **GCP / Artifact Registry**: borradas 3 imágenes viejas (20,5 GB → 5,4 GB; ~US$2/mes → ~US$0,50). Queda solo `latest` (`9c0a900`, la que sirve Cloud Run).
- **GCP / Artifact Registry**: cleanup policy `keep-last-2` en `financebot-repo` (evita que se acumulen imágenes).
- **Eval (bug de medición)**: RAGAS recibía previews de 300 chars deduplicados por página en vez de los chunks completos que vio el LLM → subestimaba faithfulness/recall en todas las runs. Ahora usa `ChatResponse.contexts` (excluido del JSON de la API).
- **CCA AP#1 (garantía en código)**: aviso de urgencia 24/7 por regex (`is_urgent`) y respuesta "sin información" con texto fijo; si no hay contexto no se llama al LLM. Umbral `min_rerank_score` (default 0.0 = desactivado, pendiente de calibrar).
- **CCA AP#5/#6 (salida estructurada + procedencia)**: Gemini responde con schema `LLMAnswer` (answer, cited_fragments, found_in_context); las fuentes son solo las citadas; citas inválidas se descartan en código; sin citas válidas → todas las fuentes + warning (degradación elegante).
- **CCA D4 (prompt)**: se quitó el énfasis en mayúsculas; criterios concretos + 2 few-shot sintéticos (no salen del dataset de eval, para no contaminar métricas).
- **ChatResponse**: campos nuevos `status` (answered/no_context) y `urgent`.
- **Eval**: `evaluation/analyze_results.py` (análisis por pregunta, CCA D5). Run 20: 11/30 fallos de retrieval vs 4 de generación → el cuello de botella es el retrieval (medir de nuevo con el bug corregido antes de decidir búsqueda híbrida).
- **Eval**: `evaluation/calibrate_threshold.py` para calibrar el umbral (solo Pinecone + reranker, sin Gemini).
- **Claude Code**: `.claude/rules/tests.md` (convenciones de tests con carga condicional por ruta, CCA D3).
- **Tests**: 61 → 76 (guardas deterministas, citas, contexts).
- **Pendiente**: en este equipo cargar `multilingual-e5-large` crashea PyTorch (access violation, torch 2.10 / transformers 5.2 / sentence-transformers 5.2.3); bloquea calibración y evaluación local. README desactualizado (menciona Claude/ChromaDB).
