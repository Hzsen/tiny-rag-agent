"""Chunking logic for ingestion pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator, List

from tiny_rag_agent.ingestion.loader import BaseLoader, load_file
from tiny_rag_agent.ingestion.schema import DocumentChunk, IngestionConfig


class Chunker:
    """Split documents into chunks using recursive character splitting."""

    def __init__(self, config: IngestionConfig) -> None:
        self._chunk_size = config.chunk_size
        self._chunk_overlap = config.chunk_overlap

    def chunk(self, text: str) -> List[str]:
        """Split text into chunks using recursive character splitting."""

        if not text:
            return []

        splits = self._recursive_split(text, separators=["\n\n", "\n", " "])
        return self._merge_with_overlap(splits)

    def process_document(self, file_path: str) -> List[DocumentChunk]:
        """Load a file, chunk its content, and return DocumentChunk entries."""

        return self.process_file(file_path, loader=None)

    def process_file(
        self, file_path: str, loader: BaseLoader | None
    ) -> List[DocumentChunk]:
        """Load a file via loader, chunk its content, and return entries."""

        doc_id = _stable_doc_id(file_path)
        chunks: List[DocumentChunk] = []
        chunk_index = 0

        if loader is None:
            items = load_file(file_path)
        else:
            items = loader.load(file_path)

        for item in items:
            text = item.get("text", "")
            metadata = dict(item.get("metadata", {}) or {})
            page = metadata.pop("page", None)

            for chunk_text in self.chunk(text):
                chunks.append(
                    DocumentChunk(
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}-{chunk_index}",
                        source=str(Path(file_path)),
                        page=page if isinstance(page, int) else None,
                        text=chunk_text,
                        metadata=metadata or None,
                    )
                )
                chunk_index += 1

        return chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text by provided separators."""

        if len(text) <= self._chunk_size:
            return [text.strip()]
        if not separators:
            return self._split_by_size(text)

        separator = separators[0]
        if separator in text:
            parts = [part.strip() for part in text.split(separator) if part.strip()]
            splits: List[str] = []
            for part in parts:
                splits.extend(self._recursive_split(part, separators[1:]))
            return splits

        return self._recursive_split(text, separators[1:])

    def _split_by_size(self, text: str) -> List[str]:
        """Split text by fixed size with overlap."""

        if self._chunk_size <= 0:
            return [text.strip()]

        chunks: List[str] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + self._chunk_size, length)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= length:
                break
            if self._chunk_overlap > 0:
                start = max(0, end - self._chunk_overlap)
            else:
                start = end
        return chunks

    def _merge_with_overlap(self, splits: List[str]) -> List[str]:
        """Merge splits into chunks with the configured overlap."""

        if self._chunk_size <= 0:
            return [split.strip() for split in splits if split.strip()]

        chunks: List[str] = []
        current = ""

        for split in splits:
            split = split.strip()
            if not split:
                continue

            if not current:
                if len(split) <= self._chunk_size:
                    current = split
                    continue
                chunks.extend(self._split_by_size(split))
                current = ""
                continue

            candidate = f"{current} {split}"
            if len(candidate) <= self._chunk_size:
                current = candidate
                continue

            chunks.append(current)
            overlap_text = (
                current[-self._chunk_overlap :] if self._chunk_overlap > 0 else ""
            )
            if overlap_text:
                max_overlap = max(0, self._chunk_size - len(split) - 1)
                overlap_text = overlap_text[-max_overlap:] if max_overlap else ""

            if overlap_text:
                new_current = f"{overlap_text} {split}"
            else:
                new_current = split

            if len(new_current) <= self._chunk_size:
                current = new_current
                continue

            sized = self._split_by_size(split)
            if overlap_text and sized:
                max_overlap = max(0, self._chunk_size - len(sized[0]) - 1)
                if max_overlap > 0:
                    prefix = overlap_text[-max_overlap:]
                    sized[0] = f"{prefix} {sized[0]}".strip()
            if sized:
                chunks.extend(sized[:-1])
                current = sized[-1]
            else:
                current = ""

        if current:
            chunks.append(current)

        return chunks


def _stable_doc_id(file_path: str) -> str:
    """Create a stable identifier for a document path."""

    return hashlib.sha256(file_path.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    sample_path = "data/sample.pdf"
    chunker = Chunker(IngestionConfig())
    try:
        sample_chunks = chunker.process_document(sample_path)
        for chunk in sample_chunks[:3]:
            print(f"{chunk.chunk_id} | page={chunk.page}\n{chunk.text}\n")
    except FileNotFoundError as exc:
        print(f"Sample file missing: {exc}")
