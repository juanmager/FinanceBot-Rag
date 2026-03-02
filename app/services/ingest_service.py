"""
Servicio de ingestión de documentos.
Responsabilidades:
  1. Recibir archivos PDF o TXT (como bytes)
  2. Dividirlos en fragmentos (chunking)
  3. Generar embeddings con sentence-transformers (local, sin API key extra)
  4. Almacenar fragmentos + embeddings en Pinecone
  5. Llevar un registro en memoria de los documentos procesados
"""

import logging
import os
import re
import tempfile
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from app.config import Settings, get_settings
from app.models import DocumentInfo, IngestResponse

logger = logging.getLogger(__name__)

# Tipos de archivo soportados
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

# Mapa de ligaduras tipográficas comunes en PDFs a sus equivalentes ASCII
_LIGATURES = str.maketrans({
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st",
    "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2026": "...",
})


def clean_text(text: str) -> str:
    """
    Limpia el texto extraído de PDFs:
    - Reemplaza ligaduras tipográficas (ﬁ → fi, ﬂ → fl, etc.)
    - Normaliza caracteres Unicode a su forma NFC
    - Colapsa el patrón 'palabra\\n \\npalabra' (layout de PDF con espaciado raro)
    - Elimina espacios múltiples y líneas en blanco excesivas
    """
    # 1. Reemplazar ligaduras conocidas
    text = text.translate(_LIGATURES)
    # 2. Normalizar Unicode NFC
    text = unicodedata.normalize("NFC", text)
    # 3. Colapsar el patrón "\n \n" entre palabras (artefacto de PDFs con layout raro)
    text = re.sub(r"(\S)\n \n(\S)", r"\1 \2", text)
    # 4. Colapsar espacios múltiples
    text = re.sub(r" {2,}", " ", text)
    # 5. Máximo 2 líneas en blanco consecutivas
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_docs(docs: list) -> str:
    """
    Formatea una lista de documentos recuperados como texto estructurado
    para incluirlos en el prompt del LLM.
    """
    if not docs:
        return "No hay información disponible en la base de conocimiento."

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Desconocido")
        page = doc.metadata.get("page")
        header = f"[Fragmento {i} — {source}" + (f", p.{page}" if page is not None else "") + "]"
        parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


class IngestService:
    """
    Gestiona la carga de documentos en el vector store Pinecone.
    Esta clase también es la fuente de verdad del vector store compartido
    con el RAGService.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

        logger.info(f"Cargando modelo de embeddings: {self.settings.embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info(f"Inicializando Pinecone (índice: {self.settings.pinecone_index_name})")
        self.pc = Pinecone(api_key=self.settings.pinecone_api_key)

        # Crear el índice si no existe (384 dims para all-MiniLM-L6-v2, métrica coseno)
        index_name = self.settings.pinecone_index_name
        if not self.pc.has_index(index_name):
            logger.info(f"Creando índice Pinecone '{index_name}'...")
            self.pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self.vector_store = PineconeVectorStore(
            index=self.pc.Index(index_name),
            embedding=self.embeddings,
            namespace=self.settings.pinecone_namespace,
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", "===", "---", ". ", " ", ""],
        )

        # Registro en memoria de documentos procesados (se pierde al reiniciar)
        self._docs_metadata: dict[str, DocumentInfo] = {}

    def ingest_file(self, file_content: bytes, filename: str) -> IngestResponse:
        """
        Procesa e indexa un archivo en el vector store.

        Args:
            file_content: Contenido del archivo como bytes.
            filename: Nombre original del archivo (usado como metadato).

        Returns:
            IngestResponse con el resultado del procesamiento.

        Raises:
            ValueError: Si el tipo de archivo no está soportado.
            RuntimeError: Si falla el procesamiento interno.
        """
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Tipo de archivo '{ext}' no soportado. "
                f"Formatos válidos: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        # Guardar en archivo temporal para que los loaders de LangChain puedan leerlo
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        try:
            # Cargar documento según su tipo
            if ext == ".pdf":
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, encoding="utf-8")

            docs = loader.load()

            # Limpiar texto: normalizar ligaduras y artefactos de PDF
            for doc in docs:
                doc.page_content = clean_text(doc.page_content)

            # Añadir metadato de origen a cada fragmento
            for doc in docs:
                doc.metadata["source"] = filename

            # Dividir en fragmentos
            chunks = self.text_splitter.split_documents(docs)
            logger.info(f"'{filename}': {len(docs)} página(s) → {len(chunks)} fragmentos")

            # Indexar en Pinecone
            self.vector_store.add_documents(chunks)

            # Registrar metadatos
            doc_info = DocumentInfo(
                filename=filename,
                extension=ext,
                chunks=len(chunks),
                ingested_at=datetime.now(UTC),
            )
            self._docs_metadata[filename] = doc_info

            return IngestResponse(
                filename=filename,
                chunks_created=len(chunks),
                message=f"'{filename}' procesado exitosamente con {len(chunks)} fragmentos indexados.",
            )

        except Exception as e:
            logger.error(f"Error procesando '{filename}': {e}")
            raise RuntimeError(f"Error al procesar el documento: {str(e)}") from e

        finally:
            # Siempre eliminar el archivo temporal
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def get_documents(self) -> list[DocumentInfo]:
        """Retorna la lista de documentos procesados en esta sesión."""
        return list(self._docs_metadata.values())

    def get_vector_store(self) -> PineconeVectorStore:
        """Expone el vector store para que lo use el RAGService."""
        return self.vector_store
