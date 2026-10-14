"""
Servicio RAG (Retrieval-Augmented Generation).
Responsabilidades:
  1. Recuperar candidatos del vector store (Pinecone, similitud coseno)
  2. Aplicar reranking con CrossEncoder para mejorar precisión del contexto
  3. Guardas deterministas en código (sin contexto → fallback; urgencia → aviso 24/7)
  4. Generar la respuesta con Gemini usando salida estructurada (respuesta + citas)
  5. Retornar respuesta + solo las fuentes citadas (procedencia)

Principio (CCA, anti-patrón #1): lo que debe estar GARANTIZADO va en código,
el prompt solo aporta criterio.
"""

import logging
import re
from datetime import UTC, datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from app.config import Settings, get_settings
from app.models import ChatResponse, SourceDocument
from app.services.ingest_service import format_docs

logger = logging.getLogger(__name__)

# ── Respuestas fijas (garantizadas por código, no por el prompt) ──────────────
NO_INFO_ANSWER = (
    "Lo siento, no tengo información disponible sobre ese tema. "
    "Para asistencia, llamá al 0800-FINANCE (L-V 8-20) o WhatsApp."
)
URGENCY_NOTICE = "⚠️ Si es una urgencia (tarjeta robada o fraude), llamá ya al 0800-FINANCE, disponible 24/7."
URGENCY_PATTERN = re.compile(
    r"\b(rob(o|ada|ado|aron)|fraude|estafa|hacke|phishing|suplantaci|extravi|"
    r"perd[ií]\s+(mi|la)\s+tarjeta|no\s+reconozco)",
    re.IGNORECASE,
)

# ── Prompt del sistema: criterios + few-shot (sin énfasis en mayúsculas) ──────
# Los ejemplos son sintéticos a propósito: NO salen del dataset de evaluación,
# para no contaminar las métricas de RAGAS.
SYSTEM_CONTENT = """Sos FinanceBot, el asistente virtual de Ualá (también llamada Alau), un banco digital argentino.
"FinanceBot", "Ualá" y "Alau" son el mismo banco.

Criterios para responder:
- Respondé en español, claro y amable, en menos de 300 palabras.
- Cada dato concreto (monto, porcentaje, plazo, requisito, condición) tiene que figurar en algún fragmento del contexto. Si un dato no figura, no lo incluyas.
- En cited_fragments poné los números de los [Fragmento N] de donde sacaste la información.
- Si el contexto no responde la pregunta, poné found_in_context=false.

Ejemplo 1 (el contexto responde):
Contexto: [Fragmento 2 — tyc.pdf] La reposición de la tarjeta tiene un costo que se descuenta del saldo de la cuenta.
Pregunta: ¿Me cobran por una tarjeta nueva?
→ answer: "Sí. La reposición de la tarjeta tiene un costo, que se descuenta del saldo de tu cuenta." · cited_fragments: [2] · found_in_context: true

Ejemplo 2 (el contexto habla del tema pero no tiene el dato pedido):
Contexto: [Fragmento 1 — prestamos.pdf] Los préstamos se acreditan en la cuenta del usuario.
Pregunta: ¿Cuál es la tasa del préstamo personal?
→ answer: "" · cited_fragments: [] · found_in_context: false

Contexto disponible:
{context}"""


class LLMAnswer(BaseModel):
    """Schema de salida estructurada que debe devolver el LLM."""

    answer: str = Field(description="Respuesta al cliente, basada solo en el contexto")
    cited_fragments: list[int] = Field(
        default_factory=list,
        description="Números N de los [Fragmento N] usados para responder",
    )
    found_in_context: bool = Field(
        description="false si el contexto no contiene la información pedida"
    )


def is_urgent(question: str) -> bool:
    """Detección determinista de urgencias (robo, fraude, extravío)."""
    return bool(URGENCY_PATTERN.search(question))


class RAGService:
    """
    Orquesta el flujo RAG: recuperación + reranking + generación estructurada con Gemini.
    Es un workflow (pasos fijos en código), no un agente.
    """

    def __init__(self, vector_store: PineconeVectorStore, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.vector_store = vector_store

        logger.info(f"Inicializando RAGService con modelo: {self.settings.gemini_model}")
        self.llm = ChatGoogleGenerativeAI(
            model=self.settings.gemini_model,
            google_api_key=self.settings.google_api_key,
            max_output_tokens=self.settings.max_tokens,
            temperature=0.0,  # sin creatividad → máxima fidelidad al contexto
        )
        # Salida estructurada: la forma la garantiza el schema, no el prompt (CCA AP#5)
        self.structured_llm = self.llm.with_structured_output(LLMAnswer)

        logger.info(f"Cargando CrossEncoder: {self.settings.reranker_model}")
        self.reranker = CrossEncoder(self.settings.reranker_model)

        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.settings.retriever_k},
        )

    def answer(self, question: str) -> ChatResponse:
        """
        Responde una pregunta usando RAG con reranking:
          1. Recupera k * multiplier candidatos de Pinecone
          2. Reordena con CrossEncoder y toma los k mejores
          3. Si el mejor score no supera el umbral → fallback fijo, sin llamar al LLM
          4. Genera respuesta estructurada (answer + cited_fragments + found_in_context)
          5. Valida citas en código y arma las fuentes solo con lo citado
          6. Si la pregunta es urgente, agrega el aviso 24/7 (determinista)
        """
        logger.info(f"Procesando pregunta: '{question[:80]}...'")
        urgent = is_urgent(question)

        # ── Paso 1: Recuperar candidatos ampliados de Pinecone ──────
        n_candidates = self.settings.retriever_k * self.settings.reranker_candidates_multiplier
        candidates = self.vector_store.similarity_search_with_score(question, k=n_candidates)
        logger.info(f"Candidatos recuperados de Pinecone: {len(candidates)}")

        # ── Paso 2: Reranking con CrossEncoder ───────────────────────
        pairs = [(question, doc.page_content) for doc, _ in candidates]
        cross_scores = self.reranker.predict(pairs) if pairs else []
        reranked = sorted(zip(candidates, cross_scores), key=lambda x: x[1], reverse=True)
        top = reranked[: self.settings.retriever_k]
        docs_with_scores = [(doc, float(embed_score)) for (doc, embed_score), _ in top]
        top_score = float(top[0][1]) if top else None
        logger.info(f"Top rerank score: {top_score}")

        relevant_docs = [doc for doc, _ in docs_with_scores]
        contexts = [doc.page_content for doc in relevant_docs]

        # ── Paso 3: Guarda determinista — sin contexto relevante ─────
        if top_score is None or top_score < self.settings.min_rerank_score:
            logger.info("Sin contexto relevante: fallback sin llamar al LLM.")
            return self._build_response(question, NO_INFO_ANSWER, [], contexts, "no_context", urgent)

        # ── Paso 4: Generación estructurada ──────────────────────────
        system_msg = SystemMessage(content=SYSTEM_CONTENT.format(context=format_docs(relevant_docs)))
        human_msg = HumanMessage(content=f"Pregunta del cliente: {question}")
        result: LLMAnswer = self.structured_llm.invoke([system_msg, human_msg])

        if not result.found_in_context or not result.answer.strip():
            return self._build_response(question, NO_INFO_ANSWER, [], contexts, "no_context", urgent)

        # ── Paso 5: Validación semántica de citas (el schema no la garantiza) ─
        valid = [n for n in dict.fromkeys(result.cited_fragments) if 1 <= n <= len(docs_with_scores)]
        if valid:
            cited = [docs_with_scores[n - 1] for n in valid]
        else:
            # Degradación elegante (AP#3): sin citas válidas, se muestran todas y se registra
            logger.warning(f"Respuesta sin citas válidas ({result.cited_fragments}); se listan todas las fuentes.")
            cited = docs_with_scores

        return self._build_response(question, result.answer, cited, contexts, "answered", urgent)

    def _build_response(
        self,
        question: str,
        answer_text: str,
        docs_with_scores: list,
        contexts: list[str],
        status: str,
        urgent: bool,
    ) -> ChatResponse:
        # ── Paso 6: Aviso de urgencia garantizado por código ─────────
        if urgent and "24/7" not in answer_text:
            answer_text = f"{answer_text}\n\n{URGENCY_NOTICE}"

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
            status=status,
            urgent=urgent,
            contexts=contexts,
            model=self.settings.gemini_model,
            timestamp=datetime.now(UTC),
        )
