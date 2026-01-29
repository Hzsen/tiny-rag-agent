"""Test the ingestion pipeline on a sample PDF."""

from __future__ import annotations

from pathlib import Path

from tiny_rag_agent.ingestion.chunker import Chunker
from tiny_rag_agent.ingestion.loader import PDFLoader
from tiny_rag_agent.ingestion.schema import IngestionConfig


def _has_number_dense_text(text: str, threshold: float = 0.35) -> bool:
    """Return True when digits are a high proportion of the text."""

    if not text:
        return False
    digits = sum(1 for char in text if char.isdigit())
    ratio = digits / max(len(text), 1)
    return ratio >= threshold


def main() -> None:
    """Run a basic ingestion pass on a sample PDF."""

    config = IngestionConfig()
    loader = PDFLoader()
    chunker = Chunker(config)

    file_path = "src/tiny_rag_agent/data/avgo_report.pdf"
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return

    chunks = chunker.process_file(file_path, loader)

    print(f"Total chunks: {len(chunks)}")
    if not chunks:
        print("No chunks generated.")
        return

    first_chunk = chunks[0]
    print("\nFirst chunk:\n")
    print(first_chunk.text)

    middle_index = 5 if len(chunks) > 5 else len(chunks) // 2
    middle_chunk = chunks[middle_index]
    print(f"\nMiddle chunk (index {middle_index}):\n")
    print(middle_chunk.text)

    print("\nFirst chunk metadata:")
    print(
        {
            "doc_id": first_chunk.doc_id,
            "source": first_chunk.source,
            "page": first_chunk.page,
        }
    )

    for chunk in chunks:
        if _has_number_dense_text(chunk.text):
            print(
                "Warning: chunk may contain a table-heavy section "
                f"(chunk_id={chunk.chunk_id}, page={chunk.page})."
            )
            break


if __name__ == "__main__":
    main()
