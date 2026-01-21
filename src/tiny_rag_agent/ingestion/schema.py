"""Pydantic schemas for ingestion configuration and chunks."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestionConfig(BaseModel):
    """Configuration schema for ingestion settings."""

    chunk_size: int = Field(
        default=500,
        description="Size of each chunk in characters.",
    )
    chunk_overlap: int = Field(
        default=50,
        description="Number of overlapping characters between chunks.",
    )


class DocumentChunk(BaseModel):
    """Schema for a chunk produced during ingestion."""

    doc_id: str = Field(
        ...,
        description="Unique identifier for the source document.",
    )
    chunk_id: str = Field(
        ...,
        description="Unique hash or index for this specific chunk.",
    )
    source: str = Field(
        ...,
        description="Source file path or URL.",
    )
    page: int | None = Field(
        default=None,
        description="Optional page number for paginated sources (e.g., PDFs).",
    )
    text: str = Field(
        ...,
        description="Chunk content as plain text.",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata for this chunk.",
    )
