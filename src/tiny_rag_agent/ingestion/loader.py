"""Document loaders for ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import Any, Iterator

from pypdf import PdfReader


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, file_path: str) -> Iterator[dict[str, Any]]:
        """Yield text and metadata dictionaries for a document."""


class PDFLoader(BaseLoader):
    """Loader for PDF documents."""

    def load(self, file_path: str) -> Iterator[dict[str, Any]]:
        """Yield per-page text and metadata from a PDF."""

        reader = PdfReader(file_path)
        for page_index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            cleaned = _clean_pdf_text(text)
            if not cleaned:
                continue
            yield {
                "text": cleaned,
                "metadata": {"page": page_index},
            }


class MarkdownLoader(BaseLoader):
    """Loader for Markdown files."""

    def load(self, file_path: str) -> Iterator[dict[str, Any]]:
        """Yield the full Markdown document as a single block."""

        content = Path(file_path).read_text(encoding="utf-8")
        cleaned = _normalize_whitespace(content)
        if cleaned:
            yield {
                "text": cleaned,
                "metadata": {},
            }


def load_file(file_path: str) -> Iterator[dict[str, Any]]:
    """Load a file with the correct loader based on extension."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()
    if extension == ".pdf":
        loader: BaseLoader = PDFLoader()
    elif extension in {".md", ".markdown"}:
        loader = MarkdownLoader()
    else:
        raise ValueError(f"Unsupported file type: {extension or '<none>'}")

    return loader.load(file_path)


def _clean_pdf_text(text: str) -> str:
    """Clean PDF text by removing common artifacts."""

    normalized = _normalize_whitespace(text)
    lines = []
    for line in normalized.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.isdigit():
            continue
        if re.fullmatch(r"page\s+\d+", stripped, flags=re.IGNORECASE):
            continue
        lines.append(stripped)
    return "\n".join(lines).strip()


def _normalize_whitespace(text: str) -> str:
    """Collapse excessive whitespace while preserving newlines."""

    text = text.replace("\f", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
