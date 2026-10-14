# 🏦 FinanceBot — Asistente Financiero RAG

> **Proyecto 2 — AI Engineer Portfolio**  
> Sistema de **Retrieval-Augmented Generation (RAG)** construido con LangChain, Pinecone y Gemini, sobre los términos y condiciones de Ualá.  
> Temado como asistente de atención al cliente de un banco digital argentino.

---

## 📋 Tabla de contenidos

- [Descripción y arquitectura RAG](#descripción-y-arquitectura-rag)
- [Endpoints](#endpoints)
- [Correr localmente](#correr-localmente)
- [Correr con Docker](#correr-con-docker)
- [Demo rápido](#demo-rápido)
- [Tests](#tests)
- [Variables de entorno](#variables-de-entorno)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Decisiones técnicas](#decisiones-técnicas)

---

## Descripción y arquitectura RAG

**RAG (Retrieval-Augmented Generation)** es una técnica que combina búsqueda semántica con generación de texto:

```
 Documento (PDF/TXT)
        │
        ▼
  [Chunking]  ──→  Fragmentos de ~400 chars (overlap 80)
        │
        ▼
  [Embeddings]  ──→  multilingual-e5-large (1024 dim, local)
        │
        ▼
  [Pinecone]  ──→  Vector store serverless (free tier)
        │
        │   (al recibir una pregunta)
        ▼
  [Retriever]  ──→  20 candidatos por similitud coseno
        │
        ▼
  [Reranker]  ──→  bge-reranker-v2-m3 elige los 10 mejores
        │          (score < MIN_RERANK_SCORE → respuesta fija, sin LLM)
        ▼
  [Gemini 2.5 Flash]  ──→  Salida estructurada: answer + cited_fragments + found_in_context
        │
        ▼
  Guardas en código: citas validadas, aviso de urgencia 24/7, fallback "sin información"
        │
        ▼
  Respuesta + solo las fuentes citadas
```

### Stack tecnológico

| Componente | Tecnología |
|---|---|
| Framework API | FastAPI |
| Orquestación LLM | LangChain 0.3 |
| LLM | Gemini 2.5 Flash (free tier) |
| Embeddings | `intfloat/multilingual-e5-large` (local) |
| Reranker | `BAAI/bge-reranker-v2-m3` (CrossEncoder) |
| Vector Store | Pinecone serverless |
| Evaluación | RAGAS (faithfulness, precision, recall, relevancy) |
| Deploy | Cloud Run (push a `main` → CI/CD) |
| Chunking | `RecursiveCharacterTextSplitter` |
| Loaders | `PyPDFLoader` (PDF) + `TextLoader` (TXT) |
| Containerización | Docker + Docker Compose |

---

## Endpoints

| Endpoint | Método | Descripción |
|---|---|---|
| `/health` | GET | Health check del servicio |
| `/documents/ingest` | POST | Subir e indexar un PDF o TXT |
| `/documents` | GET | Listar documentos en la base de conocimiento |
| `/chat/ask` | POST | Hacer una pregunta al asistente financiero |

---

## Correr localmente

### 1. Clonar y crear entorno virtual

```bash
git clone <tu-repo>
cd project2-rag-chatbot

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

> ⚠️ La primera instalación tarda varios minutos por `sentence-transformers` y `torch`.

### 3. Configurar variables de entorno

```bash
cp .env.example .env
# Editá .env y agregá GOOGLE_API_KEY y PINECONE_API_KEY
```

### 4. Iniciar el servidor

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

> ⚠️ El **primer inicio** puede tardar 30-60 segundos mientras descarga los modelos de embeddings y reranker (~3 GB).

### 5. Documentación interactiva

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

---

## Correr con Docker

### Docker Compose (recomendado)

```bash
# Asegurate de tener el .env configurado
docker compose up -d

# Ver logs (el primer build tarda ~5 min por la descarga del modelo)
docker compose logs -f
```

La API queda disponible en http://localhost:8001

Para detener:
```bash
docker compose down
```

> **Nota:** Los vectores viven en Pinecone, así que sobreviven a reinicios y redeploys del contenedor.

---

## Demo rápido

### 1. Verificar que el servicio está activo

```bash
curl http://localhost:8001/health
```

### 2. Indexar los documentos de muestra

```bash
# Indexar productos financieros
curl -X POST http://localhost:8001/documents/ingest \
  -F "file=@data/sample_docs/productos_financieros.txt"

# Indexar FAQ
curl -X POST http://localhost:8001/documents/ingest \
  -F "file=@data/sample_docs/preguntas_frecuentes.txt"
```

### 3. Hacer preguntas al asistente

```bash
curl -X POST http://localhost:8001/chat/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cuáles son los requisitos para abrir una cuenta?"}'
```

```bash
curl -X POST http://localhost:8001/chat/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cuánto es la tasa del plazo fijo?"}'
```

```bash
curl -X POST http://localhost:8001/chat/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Qué hago si me roban la tarjeta?"}'
```

### Respuesta de ejemplo

```json
{
  "question": "¿Qué hago si me roban la tarjeta?",
  "answer": "Podés bloquear la tarjeta desde la app...

⚠️ Si es una urgencia (tarjeta robada o fraude), llamá ya al 0800-FINANCE, disponible 24/7.",
  "sources": [
    {"content": "...", "source": "tyc_uala.pdf", "page": 12, "similarity_score": 0.84}
  ],
  "status": "answered",
  "urgent": true,
  "model": "gemini-2.5-flash",
  "timestamp": "2026-10-14T15:00:00Z"
}
```

`status` es `no_context` cuando el contexto no tiene la información (respuesta fija). `urgent` lo detecta el código, no el LLM.

### 4. Ver documentos indexados

```bash
curl http://localhost:8001/documents
```

---

## Tests

```bash
# Instalar dependencias de desarrollo
pip install -r requirements-dev.txt

# Correr todos los tests
pytest -v

# Con cobertura
pytest --cov=app --cov-report=html -v
```

Los tests usan **mocks** para los servicios de ML — no descargan modelos ni consumen tokens.

## Evaluación (RAGAS)

```bash
python evaluation/run_evaluation.py        # 30 preguntas, ~30 min (--use-cache reusa respuestas)
python evaluation/analyze_results.py       # peores preguntas: fallos de retrieval vs generación
python evaluation/calibrate_threshold.py   # calibra MIN_RERANK_SCORE (sin Gemini)
```

Historial de corridas en `MASTER_LOG_2026-03-26.md`; cambios en `CHANGELOG.md`.

---

## Variables de entorno

| Variable | Requerida | Default | Descripción |
|---|---|---|---|
| `GOOGLE_API_KEY` | ✅ Sí | — | API key de Gemini (AI Studio) |
| `PINECONE_API_KEY` | ✅ Sí | — | API key de Pinecone |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Modelo a usar |
| `MAX_TOKENS` | No | `2048` | Máximo de tokens en respuesta |
| `CHUNK_SIZE` | No | `400` | Tamaño de fragmentos (chars) |
| `CHUNK_OVERLAP` | No | `80` | Superposición entre fragmentos |
| `RETRIEVER_K` | No | `10` | Fragmentos que llegan al LLM |
| `PINECONE_INDEX_NAME` | No | `financebot-rag` | Índice de Pinecone |
| `PINECONE_NAMESPACE` | No | `financial_docs` | Namespace |
| `EMBEDDING_MODEL` | No | `intfloat/multilingual-e5-large` | Modelo de embeddings (dim 1024) |
| `RERANKER_MODEL` | No | `BAAI/bge-reranker-v2-m3` | CrossEncoder |
| `RERANKER_CANDIDATES_MULTIPLIER` | No | `2` | Candidatos = K × multiplier |
| `MIN_RERANK_SCORE` | No | `0.1` | Debajo → respuesta fija sin LLM |

---

## Estructura del proyecto

```
project2-rag-chatbot/
├── app/
│   ├── main.py               # Entry point + factory create_app()
│   ├── config.py             # Configuración con pydantic-settings
│   ├── models.py             # Schemas Pydantic
│   ├── routes/
│   │   ├── health.py         # GET /health
│   │   ├── documents.py      # POST /documents/ingest, GET /documents
│   │   └── chat.py           # POST /chat/ask
│   └── services/
│       ├── ingest_service.py # Chunking + embeddings e5 + Pinecone
│       └── rag_service.py    # Retrieval + rerank + Gemini estructurado + guardas
├── data/
│   └── sample_docs/
│       ├── productos_financieros.txt   # Documentos de demo
│       └── preguntas_frecuentes.txt
├── tests/
│   ├── conftest.py           # Fixtures con mock lifespan
│   ├── test_health.py
│   ├── test_documents.py
│   ├── test_chat.py
│   ├── test_ingest_service.py
│   └── test_rag_service.py
├── evaluation/               # RAGAS: dataset, scripts y resultados
├── scripts/                  # Reindexado de Pinecone
├── CHANGELOG.md
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── README.md
```

---

## Decisiones técnicas

### ¿Por qué Gemini + Pinecone + modelos locales?
Costo mínimo: Gemini 2.5 Flash y Pinecone tienen free tier, y embeddings/reranker corren localmente (sin costo por request).

### ¿Por qué reranker?
La similitud coseno trae candidatos razonables pero mal ordenados; el CrossEncoder lee pregunta y fragmento juntos y mejora la precisión del contexto (Runs 17–20 en `MASTER_LOG`).

### Garantías en código, criterio en el prompt
Lo que debe cumplirse siempre (aviso de urgencia, respuesta "sin información", validación de citas) está en código, no confiado al prompt. El LLM devuelve salida estructurada (`LLMAnswer`) y solo se muestran las fuentes que citó.

### Patrón `create_app(lifespan_fn=)`
Permite inyectar un lifespan mockeado en los tests, evitando cargar modelos de ML (~30s) en cada test. Los tests corren en <1 segundo.

### Separación IngestService / RAGService
Ambos comparten la misma instancia de `PineconeVectorStore`. Cuando `IngestService` agrega documentos, el retriever de `RAGService` los encuentra automáticamente en la siguiente consulta.
