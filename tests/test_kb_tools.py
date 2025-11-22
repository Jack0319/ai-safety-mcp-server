from __future__ import annotations

from pathlib import Path

import orjson

from safety_kb.models import Document
from safety_kb.retrieval import KnowledgeBaseClient

FIXTURE = Path(__file__).parent / "fixtures" / "sample_docs.jsonl"


def make_stub_client() -> KnowledgeBaseClient:
    client = KnowledgeBaseClient(vectorstore_url=None, collection="test", docstore_url=None, use_stub=True)
    docs = []
    for line in FIXTURE.read_text().splitlines():
        docs.append(Document(**orjson.loads(line)))
    client.load_stub_documents(docs)
    return client


def test_search_returns_results():
    client = make_stub_client()
    results = client.search("deceptive")
    assert results
    assert results[0].doc_id == "doc-1"


def test_get_document_fetches_full_text():
    client = make_stub_client()
    doc = client.get_document("doc-2")
    assert "dual-use" in doc.text or "dual" in doc.text.lower()
