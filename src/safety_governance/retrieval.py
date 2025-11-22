"""Governance and policy retrieval helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from safety_kb.models import SearchResult
from safety_kb.retrieval import KnowledgeBaseClient, KnowledgeBaseFilters


@dataclass
class GovernanceRetriever:
    kb: KnowledgeBaseClient
    doc_path: Path | None = None

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        filters = KnowledgeBaseFilters(topic="governance")
        results = self.kb.search(query or "governance", k=k, filters=filters)
        if results or not self.doc_path:
            return results
        # optional fallback: load docs from disk if KB lacks governance corpus
        return []
