---
paths: ["tests/**/*.py"]
---
# Convenciones de tests
- Todo mockeado: nunca llamar a Gemini, Pinecone ni descargar modelos (CrossEncoder/embeddings se parchean).
- Las garantías deterministas (fallback, urgencia, validación de citas) tienen que tener un test que verifique que NO se llama al LLM o que el texto fijo aparece.
- Correr: `pytest tests -q` y `ruff check app/ tests/`.
