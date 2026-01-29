"""Keyword-based retrieval using BM25."""

from __future__ import annotations

from typing import List

from rank_bm25 import BM25Okapi

from tiny_rag_agent.ingestion.schema import DocumentChunk


class KeywordStore:
    """Wrapper for BM25 keyword search."""

    def __init__(self) -> None:
        # Keep the original chunks so we can return them by index.
        self._chunks: List[DocumentChunk] = []
        # BM25 index is built from the tokenized corpus.
        self._bm25: BM25Okapi | None = None

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase + whitespace split."""
        return text.lower().split()

    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks and rebuild the BM25 index."""
        if not chunks:
            return

        # Store chunks and build a BM25 index over their tokens.
        self._chunks = chunks
        tokenized_corpus = [self._tokenize(chunk.text) for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def query(self, query_text: str, top_k: int) -> List[DocumentChunk]:
        """Query the BM25 index and return top matching chunks."""
        if self._bm25 is None or top_k <= 0:
            return []

        tokenized_query = self._tokenize(query_text)
        # BM25 returns one score per document in the corpus.
        scores = self._bm25.get_scores(tokenized_query)
        # Rank documents by score (highest first).
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        # Return the top-k document chunks in ranked order.
        return [self._chunks[index] for index in ranked_indices[:top_k]]
