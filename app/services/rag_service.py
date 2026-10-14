"""
Servicio RAG (Retrieval-Augmented Generation).
Responsabilidades:
  1. Recuperar candidatos del vector store (Pinecone, similitud coseno)
  2. Aplicar reranking con CrossEncoder para mejorar precisión del contexto
  3. Construir el prompt con los mejores chunks
  4. Generar la respuesta usando Gemini (Google)
  5. Retornar respuesta + documentos fuente para transparencia
"""

import logging
from datetime import UTC, datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import CrossEncoder

from app.config import Settings, get_settings
from app.models import ChatResponse, SourceDocument
from app.services.ingest_service import format_docs

logger = logging.getLogger(__name__)

# ── Prompt del sistema ─────────────────────────────────────────────────────────
# BASELINE: Original System Prompt (reverted after Run 18 failure)
SYSTEM_CONTENT = """Eres **FinanceBot**, el asistente virtual de **Ualá** (también conocida como Alau), un banco digital argentino.

**Tu función:**
Responder preguntas de clientes sobre productos y servicios financieros, basándote ÚNICAMENTE en el texto del contexto proporcionado más abajo.

**Reglas — leé cada una antes de responder:**
1. Respondé siempre en español, de manera clara, amable y profesional.
2. Usá EXCLUSIVAMENTE la información que aparece textualmente en el contexto. Está prohibido usar conocimiento propio, general o externo.
3. Si un dato específico (número, porcentaje, fecha, plazo, condición, requisito) no figura con exactitud en el contexto, NO lo completes ni lo inferás.
4. "FinanceBot", "Ualá" y "Alau" hacen referencia al mismo banco digital — tratá la información de cualquiera de los tres nombres como válida.
5. Si la respuesta no está en el contexto, respondé: "Lo siento, no tengo información disponible sobre ese tema. Para asistencia, llamá al 0800-FINANCE (L-V 8-20) o WhatsApp."
6. Para urgencias (tarjeta robada, fraude), siempre indicá: "Llamá al 0800-FINANCE 24/7".
7. Sé conciso pero completo. Máx 300 palabras.

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
            temperature=0.0,  # B1: sin creatividad → máxima fidelidad al contexto
        )

        # CrossEncoder para reranking post-retrieval
        logger.info(f"Cargando CrossEncoder: {self.settings.reranker_model}")
        self.reranker = CrossEncoder(self.settings.reranker_model)

        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.settings.retriever_k},
        )

    def answer(self, question: str) -> ChatResponse:
        """
        Responde una pregunta usando RAG con reranking:
          1. Recupera k * multiplier candidatos de Pinecone (similitud coseno)
          2. Reordena con CrossEncoder (más preciso que coseno)
          3. Toma los k mejores chunks
          4. Construye el prompt con el contexto
          5. Genera la respuesta con Gemini
          6. Retorna respuesta + fuentes

        Args:
            question: Pregunta del cliente.

        Returns:
            ChatResponse con la respuesta generada y los documentos fuente.

        Raises:
            Exception: Si la API de Gemini falla o el vector store no responde.
        """
        logger.info(f"Procesando pregunta: '{question[:80]}...'")

        # ── Paso 1: Recuperar candidatos ampliados de Pinecone ──────
        # Se recuperan k * multiplier candidatos para dar margen al reranker.
        n_candidates = self.settings.retriever_k * self.settings.reranker_candidates_multiplier
        candidates = self.vector_store.similarity_search_with_score(
            question, k=n_candidates
        )
        logger.info(f"Candidatos recuperados de Pinecone: {len(candidates)}")

        # ── Paso 2: Reranking con CrossEncoder ───────────────────────
        # CrossEncoder puntúa el par (pregunta, chunk) con mayor precisión
        # que la similitud coseno usada en el retrieval inicial.
        pairs = [(question, doc.page_content) for doc, _ in candidates]
        cross_scores = self.reranker.predict(pairs) if pairs else []
        reranked = sorted(
            zip(candidates, cross_scores),
            key=lambda x: x[1],
            reverse=True,
        )
        docs_with_scores = [
            (doc, float(embed_score))
            for (doc, embed_score), _ in reranked[: self.settings.retriever_k]
        ]
        logger.info(f"Fragmentos tras reranking (top {self.settings.retriever_k}):")
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            src = doc.metadata.get("source", "?")
            logger.info(f"  Chunk {i}: embed_score={score:.3f} | {src}")

        # ── Paso 3: Formatear contexto ───────────────────────────────
        relevant_docs = [doc for doc, _ in docs_with_scores]
        context = format_docs(relevant_docs)

        # ── Paso 4: Construir mensajes para Gemini ─────────────────────
        system_msg = SystemMessage(content=SYSTEM_CONTENT.format(context=context))
        human_msg = HumanMessage(content=f"Pregunta del cliente: {question}")

        # ── Paso 5: Generar respuesta ────────────────────────────────
        response = self.llm.invoke([system_msg, human_msg])
        answer_text = response.content

        logger.info("Respuesta generada exitosamente.")

        # ── Paso 6: Construir lista de fuentes con score de similitud ─
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
