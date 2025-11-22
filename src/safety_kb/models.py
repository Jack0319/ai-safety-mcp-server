"""Data models for knowledge base documents."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    title: str
    text: str
    url: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    doc_id: str
    title: str
    snippet: str
    score: float
    url: Optional[str] = None
    source: Optional[str] = None
    topic: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
