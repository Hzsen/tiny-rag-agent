"""Hybrid retriever combining vector and keyword search."""

from __future__ import annotations

from typing import Dict, List

from tiny_rag_agent.ingestion.schema import DocumentChunk
from tiny_rag_agent.retrieval.keyword_store import KeywordStore
from tiny_rag_agent.retrieval.vector_store import VectorStore


class HybridRetriever:
    """Combine vector and keyword retrieval with RRF."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        keyword_store: KeywordStore | None = None,
    ) -> None:
        self._vector_store = vector_store or VectorStore()
        self._keyword_store = keyword_store or KeywordStore()

    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add documents to both retrieval backends."""
        self._vector_store.add_documents(chunks)
        self._keyword_store.add_documents(chunks)

    def search(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """Search both stores and merge results using RRF."""
        if top_k <= 0:
            return []

        vector_results = self._vector_store.query(query, top_k=top_k)
        keyword_results = self._keyword_store.query(query, top_k=top_k)

        scores: Dict[str, float] = {}
        chunks: Dict[str, DocumentChunk] = {}

        def _apply_rrf(results: List[DocumentChunk], k: int = 60) -> None:
            for rank, chunk in enumerate(results, start=1):
                chunk_key = chunk.chunk_id
                scores[chunk_key] = scores.get(chunk_key, 0.0) + 1.0 / (k + rank)
                chunks[chunk_key] = chunk

        _apply_rrf(vector_results)
        _apply_rrf(keyword_results)

        ranked_chunks = sorted(
            chunks.values(),
            key=lambda chunk: scores.get(chunk.chunk_id, 0.0),
            reverse=True,
        )

        return ranked_chunks[:top_k]
