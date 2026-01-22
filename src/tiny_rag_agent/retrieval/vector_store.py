"""Vector store wrapper for ChromaDB (local mode)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, List

from tiny_rag_agent.ingestion.schema import DocumentChunk


@dataclass(frozen=True)
class _EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimensions: int = 384


class _SimpleEmbeddingFunction:
    """Lightweight, deterministic embedding for local fallback."""

    def __init__(self, dimensions: int) -> None:
        self._dimensions = dimensions

    def __call__(self, texts: Iterable[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            vector = [0.0] * self._dimensions
            for token in text.lower().split():
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                index = int(digest, 16) % self._dimensions
                vector[index] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            embeddings.append([value / norm for value in vector])
        return embeddings


class VectorStore:
    """Wrapper for ChromaDB vector storage."""

    def __init__(
        self,
        collection_name: str = "tiny_rag_chunks",
        persist_directory: str = "chroma_db",
    ) -> None:
        self._embedding_config = _EmbeddingConfig()
        self._client = self._init_client(persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._build_embedding_function(),
        )

    def _init_client(self, persist_directory: str):
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "chromadb is required for VectorStore. Install with `pdm add chromadb`."
            ) from exc

        return chromadb.PersistentClient(path=persist_directory)

    def _build_embedding_function(self):
        try:
            from chromadb.utils import embedding_functions

            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self._embedding_config.model_name,
            )
        except Exception:
            return _SimpleEmbeddingFunction(self._embedding_config.dimensions)

    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to the vector store."""
        if not chunks:
            return

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[dict[str, object]] = []
        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadata: dict[str, object] = {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
            }
            if chunk.page is not None:
                metadata["page"] = chunk.page
            if chunk.metadata is not None:
                metadata["metadata_json"] = json.dumps(chunk.metadata)
            metadatas.append(metadata)

        self._collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, query_text: str, top_k: int) -> List[DocumentChunk]:
        """Query the vector store and return top matching chunks."""
        if top_k <= 0:
            return []

        results = self._collection.query(
            query_texts=[query_text],
            n_results=top_k,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]

        chunks: List[DocumentChunk] = []
        for chunk_id, text, metadata in zip(ids, documents, metadatas):
            metadata = metadata or {}
            metadata_json = metadata.get("metadata_json")
            parsed_metadata = (
                json.loads(metadata_json) if isinstance(metadata_json, str) else None
            )
            chunks.append(
                DocumentChunk(
                    doc_id=str(metadata.get("doc_id", "")),
                    chunk_id=str(metadata.get("chunk_id", chunk_id)),
                    source=str(metadata.get("source", "")),
                    page=metadata.get("page"),
                    text=text,
                    metadata=parsed_metadata,
                )
            )
        return chunks
