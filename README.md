# 🏦 FinanceBot — Asistente Financiero RAG

> **Proyecto 2 — AI Engineer Portfolio**  
> Sistema de **Retrieval-Augmented Generation (RAG)** construido con LangChain, ChromaDB y Claude (Anthropic).  
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
  [Chunking]  ──→  Fragmentos de texto (~1000 chars)
        │
        ▼
  [Embeddings]  ──→  Vectores numéricos (all-MiniLM-L6-v2, local)
        │
        ▼
  [ChromaDB]  ──→  Vector store persistido en disco
        │
        │   (al recibir una pregunta)
        ▼
  [Retriever]  ──→  Top-K fragmentos más similares semánticamente
        │
        ▼
  [Claude LLM]  ──→  Genera respuesta basada SOLO en el contexto recuperado
        │
        ▼
  Respuesta + Fuentes citadas
```

### Stack tecnológico

| Componente | Tecnología |
|---|---|
| Framework API | FastAPI |
| Orquestación LLM | LangChain 0.3 |
| LLM | Claude 3.5 Sonnet (Anthropic) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vector Store | ChromaDB (persistido en disco) |
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

> ⚠️ La primera instalación tarda varios minutos por `sentence-transformers` y `chromadb`.

### 3. Configurar variables de entorno

```bash
cp .env.example .env
# Editá .env y agregá tu ANTHROPIC_API_KEY
```

### 4. Iniciar el servidor

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

> ⚠️ El **primer inicio** puede tardar 30-60 segundos mientras descarga el modelo de embeddings.

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

> **Nota:** La base de vectores ChromaDB se persiste en el volumen `chroma_data`. Los documentos que indexás sobreviven reinicios del contenedor.

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
  "question": "¿Cuánto es la tasa del plazo fijo?",
  "answer": "La Tasa Nominal Anual (TNA) del Plazo Fijo FinanceBot es del **45%**, revisable mensualmente según la política del BCRA. El monto mínimo es de $1.000 y podés elegir plazos de 30, 60, 90, 180 o 365 días.",
  "sources": [
    {
      "content": "TNA vigente: 45% (se actualiza mensualmente según política del BCRA)\nPlazos disponibles: 30, 60, 90, 180 y 365 días...",
      "source": "productos_financieros.txt",
      "page": null
    }
  ],
  "model": "claude-3-5-sonnet-20241022",
  "timestamp": "2026-02-21T15:00:00Z"
}
```

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

---

## Variables de entorno

| Variable | Requerida | Default | Descripción |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ Sí | — | API key de Anthropic |
| `ANTHROPIC_MODEL` | No | `claude-3-5-sonnet-20241022` | Modelo a usar |
| `MAX_TOKENS` | No | `2048` | Máximo de tokens en respuesta |
| `CHUNK_SIZE` | No | `1000` | Tamaño de fragmentos (chars) |
| `CHUNK_OVERLAP` | No | `200` | Superposición entre fragmentos |
| `RETRIEVER_K` | No | `4` | Fragmentos a recuperar por query |
| `CHROMA_PERSIST_DIR` | No | `./chroma_db` | Directorio de persistencia |
| `CHROMA_COLLECTION_NAME` | No | `financial_docs` | Nombre de la colección |
| `EMBEDDING_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | Modelo de embeddings |

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
│       ├── ingest_service.py # Chunking + embeddings + ChromaDB
│       └── rag_service.py    # Retrieval + generación con Claude
├── data/
│   └── sample_docs/
│       ├── productos_financieros.txt   # Documentos de demo
│       └── preguntas_frecuentes.txt
├── tests/
│   ├── conftest.py           # Fixtures con mock lifespan
│   ├── test_health.py
│   ├── test_documents.py
│   └── test_chat.py
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

### ¿Por qué sentence-transformers y no embeddings de Anthropic?
Anthropic no ofrece una API de embeddings. Se eligió `all-MiniLM-L6-v2` porque:
- Es gratuito y corre localmente (sin costo adicional de API)
- 384 dimensiones — buen balance entre rendimiento y velocidad
- ~90MB — liviano para un contenedor Docker

### ¿Por qué ChromaDB y no FAISS?
ChromaDB ofrece **persistencia automática en disco** sin configuración extra. FAISS requiere serialización manual. Para un proyecto con API REST donde los documentos se deben preservar entre reinicios, ChromaDB es más conveniente.

### Patrón `create_app(lifespan_fn=)`
Permite inyectar un lifespan mockeado en los tests, evitando cargar modelos de ML (~30s) en cada test. Los tests corren en <1 segundo.

### Separación IngestService / RAGService
Ambos comparten la misma instancia de `Chroma`. Cuando `IngestService` agrega documentos, el retriever de `RAGService` los encuentra automáticamente en la siguiente consulta.
