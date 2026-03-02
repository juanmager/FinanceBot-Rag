"""
Servicio RAG (Retrieval-Augmented Generation).
Responsabilidades:
  1. Recuperar los fragmentos más relevantes del vector store para una pregunta dada
  2. Construir el prompt con el contexto recuperado
  3. Generar la respuesta usando Gemini (Google)
  4. Retornar respuesta + documentos fuente para transparencia
"""

import logging
from datetime import UTC, datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

from app.config import Settings, get_settings
from app.models import ChatResponse, SourceDocument
from app.services.ingest_service import format_docs

logger = logging.getLogger(__name__)

# ── Prompt del sistema ─────────────────────────────────────────────────────────
# Define la personalidad y restricciones del asistente financiero
SYSTEM_CONTENT = """Eres **FinanceBot**, el asistente virtual de **Ualá** (también conocida como Alau), un banco digital argentino.

**Tu función:**
Responder preguntas de clientes sobre productos y servicios financieros, basándote ÚNICAMENTE en la información del contexto proporcionado.

**Reglas obligatorias:**
1. Respondé siempre en español, de manera clara, amable y profesional.
2. Usá SOLO la información del contexto. Nunca inventes datos, tasas, plazos ni condiciones.
3. "FinanceBot", "Ualá" y "Alau" hacen referencia al mismo banco digital — tratá la información de cualquiera de los tres nombres como válida para responder.
4. Si la respuesta no está en el contexto, respondé exactamente: "Lo siento, no tengo información disponible sobre ese tema en mi base de conocimiento. Para asistencia personalizada, comunicate con nuestro equipo al **0800-FINANCE** (lunes a viernes, 8 a 20 hs) o por WhatsApp."
5. Para urgencias (tarjeta robada, transacciones sospechosas, fraude), siempre indicá que llamen inmediatamente al **0800-FINANCE** disponible 24/7.
6. Sé conciso pero completo. Evitá respuestas de más de 300 palabras a menos que sea estrictamente necesario.

**Contexto disponible:**
{context}"""


class RAGService:
    """
    Orquesta el flujo RAG: recuperación de contexto + generación de respuesta con Gemini.
    Comparte la instancia de Pinecone con IngestService para ver documentos actualizados.
    """

    def __init__(self, vector_store: PineconeVectorStore, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.vector_store = vector_store

        logger.info(f"Inicializando RAGService con modelo: {self.settings.gemini_model}")
        self.llm = ChatGoogleGenerativeAI(
            model=self.settings.gemini_model,
            google_api_key=self.settings.google_api_key,
            max_output_tokens=self.settings.max_tokens,
        )

        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.settings.retriever_k},
        )

    def answer(self, question: str) -> ChatResponse:
        """
        Responde una pregunta usando RAG:
          1. Recupera fragmentos relevantes del vector store
          2. Construye el prompt con el contexto
          3. Genera la respuesta con Gemini
          4. Retorna respuesta + fuentes

        Args:
            question: Pregunta del cliente.

        Returns:
            ChatResponse con la respuesta generada y los documentos fuente.

        Raises:
            Exception: Si la API de Gemini falla o el vector store no responde.
        """
        logger.info(f"Procesando pregunta: '{question[:80]}...'")

        # ── Paso 1: Recuperar fragmentos relevantes CON scores ───────
        # similarity_search_with_score es nativo de Pinecone/LangChain;
        # retorna lista de tuplas (Document, float) donde float es el
        # score de similitud coseno (0–1, mayor = más relevante).
        docs_with_scores = self.vector_store.similarity_search_with_score(
            question, k=self.settings.retriever_k
        )
        logger.info(f"Fragmentos recuperados: {len(docs_with_scores)}")
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            src = doc.metadata.get("source", "?")
            logger.info(f"  Chunk {i}: score={score:.3f} | {src}")

        # ── Paso 2: Formatear contexto ───────────────────────────────
        relevant_docs = [doc for doc, _ in docs_with_scores]
        context = format_docs(relevant_docs)

        # ── Paso 3: Construir mensajes para Gemini ───────────────────
        system_msg = SystemMessage(content=SYSTEM_CONTENT.format(context=context))
        human_msg = HumanMessage(content=f"Pregunta del cliente: {question}")

        # ── Paso 4: Generar respuesta ────────────────────────────────
        response = self.llm.invoke([system_msg, human_msg])
        answer_text = response.content

        logger.info("Respuesta generada exitosamente.")

        # ── Paso 5: Construir lista de fuentes con score de similitud ─
        sources = []
        seen_sources: set[str] = set()
        for doc, score in docs_with_scores:
            src_key = f"{doc.metadata.get('source')}_{doc.metadata.get('page', '')}"
            if src_key in seen_sources:
                continue
            seen_sources.add(src_key)

            content_preview = doc.page_content
            if len(content_preview) > 300:
                content_preview = content_preview[:300] + "..."

            sources.append(
                SourceDocument(
                    content=content_preview,
                    source=doc.metadata.get("source", "Desconocido"),
                    page=doc.metadata.get("page"),
                    similarity_score=round(float(score), 3),
                )
            )

        return ChatResponse(
            question=question,
            answer=answer_text,
            sources=sources,
            model=self.settings.gemini_model,
            timestamp=datetime.now(UTC),
        )
