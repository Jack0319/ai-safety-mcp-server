"""Knowledge base retrieval utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

from chromadb.api import ClientAPI  # type: ignore
from chromadb.api.models import Collection
from sqlmodel import Field, Session, SQLModel, create_engine

from core.errors import KnowledgeBaseNotFound, KnowledgeBaseUnavailable
from core.logging import get_logger
from safety_kb.models import Document, SearchResult

logger = get_logger(__name__)


class DocumentRecord(SQLModel, table=True):
    doc_id: str = Field(primary_key=True)
    title: str
    text: str
    url: str | None = None
    metadata_json: str | None = None  # JSON string


class SQLDocStore:
    def __init__(self, url: str):
        self.engine = create_engine(url, echo=False)
        SQLModel.metadata.create_all(self.engine)

    def get_document(self, doc_id: str) -> Document:
        with Session(self.engine) as session:
            record = session.get(DocumentRecord, doc_id)
            if not record:
                raise KnowledgeBaseNotFound(doc_id)
            metadata = {}
            if record.metadata_json:
                try:
                    import orjson

                    metadata = orjson.loads(record.metadata_json)
                except Exception:  # pragma: no cover - fallback
                    metadata = {}
            return Document(
                doc_id=record.doc_id,
                title=record.title,
                text=record.text,
                url=record.url,
                metadata=metadata,
            )


class InMemoryStub:
    """Simple stub backend for CI/test usage."""

    def __init__(self):
        self.documents: dict[str, Document] = {}

    def add(self, doc: Document) -> None:
        self.documents[doc.doc_id] = doc

    def search(self, query: str, k: int, filters: Optional[dict[str, Any]] = None) -> list[SearchResult]:
        results: list[SearchResult] = []
        for doc in self.documents.values():
            if filters and filters.get("topic") and doc.metadata.get("topic") != filters.get("topic"):
                continue
            if query.lower() in doc.text.lower() or query.lower() in doc.title.lower():
                snippet = doc.text[:200]
                results.append(
                    SearchResult(
                        doc_id=doc.doc_id,
                        title=doc.title,
                        snippet=snippet,
                        score=1.0,
                        url=doc.url,
                        source=doc.metadata.get("source"),
                        topic=doc.metadata.get("topic"),
                        metadata=doc.metadata,
                    )
                )
        return results[:k]

    def get_document(self, doc_id: str) -> Document:
        if doc_id not in self.documents:
            raise KnowledgeBaseNotFound(doc_id)
        return self.documents[doc_id]


@dataclass
class KnowledgeBaseFilters:
    topic: Optional[str] = None
    org: Optional[str] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None

    def to_chroma_where(self) -> dict[str, Any]:
        where: dict[str, Any] = {}
        if self.topic:
            where["topic"] = self.topic
        if self.org:
            where["org"] = self.org
        if self.year_min is not None or self.year_max is not None:
            where["year"] = {}
            if self.year_min is not None:
                where["year"]["$gte"] = self.year_min
            if self.year_max is not None:
                where["year"]["$lte"] = self.year_max
        return where


class ChromaBackend:
    def __init__(self, url: str, collection_name: str):
        import chromadb

        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"}:
            host = parsed.hostname or "localhost"
            port = parsed.port or 8000
            client: ClientAPI = chromadb.HttpClient(host=host, port=port, ssl=parsed.scheme == "https")
        else:
            client = chromadb.PersistentClient(path=url)
        self.client = client
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def search(self, query: str, k: int, filters: KnowledgeBaseFilters | None = None) -> list[SearchResult]:
        where = filters.to_chroma_where() if filters else None
        try:
            result = self.collection.query(
                query_texts=[query],
                where=where,
                n_results=k,
                include=["metadatas", "documents", "distances"],
            )
        except Exception as exc:  # pragma: no cover - external dependency
            raise KnowledgeBaseUnavailable(str(exc)) from exc

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        search_results: list[SearchResult] = []
        for doc_id, doc_text, metadata, distance in zip(ids, documents, metadatas, distances):
            snippet = (doc_text or "")[:400]
            score = similarity_from_distance(distance)
            meta = metadata or {}
            search_results.append(
                SearchResult(
                    doc_id=doc_id,
                    title=meta.get("title", doc_id),
                    snippet=snippet,
                    score=score,
                    url=meta.get("url"),
                    source=meta.get("source"),
                    topic=meta.get("topic"),
                    metadata=meta,
                )
            )
        return search_results

    def get_document(self, doc_id: str) -> Document:
        result = self.collection.get(ids=[doc_id])
        if not result.get("ids"):
            raise KnowledgeBaseNotFound(doc_id)
        text = (result.get("documents") or [[""]])[0][0]
        metadata = (result.get("metadatas") or [[{}]])[0][0]
        return Document(
            doc_id=doc_id,
            title=metadata.get("title", doc_id),
            text=text,
            url=metadata.get("url"),
            metadata=metadata or {},
        )


def similarity_from_distance(distance: Optional[float]) -> float:
    if distance is None:
        return 0.0
    # Convert L2 distance to similarity-like score in [0, 1]
    return 1.0 / (1.0 + math.exp(distance))


class KnowledgeBaseClient:
    def __init__(
        self,
        *,
        vectorstore_url: Optional[str],
        collection: str,
        docstore_url: Optional[str],
        use_stub: bool = False,
    ) -> None:
        self.stub = InMemoryStub() if use_stub or not vectorstore_url else None
        self.backend = None
        if vectorstore_url and not use_stub:
            self.backend = ChromaBackend(vectorstore_url, collection)
        self.docstore = SQLDocStore(docstore_url) if docstore_url and not use_stub else None

    def _ensure_backend(self) -> ChromaBackend:
        if self.backend is None:
            raise KnowledgeBaseUnavailable("Vector store backend not configured")
        return self.backend

    def search(self, query: str, k: int = 5, filters: Optional[KnowledgeBaseFilters] = None) -> list[SearchResult]:
        if self.stub:
            return self.stub.search(query, k, filters=filters.__dict__ if filters else None)
        backend = self._ensure_backend()
        return backend.search(query, k, filters)

    def get_document(self, doc_id: str) -> Document:
        if self.stub:
            return self.stub.get_document(doc_id)
        if self.docstore:
            return self.docstore.get_document(doc_id)
        backend = self._ensure_backend()
        return backend.get_document(doc_id)

    def load_stub_documents(self, docs: Iterable[Document]) -> None:
        if not self.stub:
            raise KnowledgeBaseUnavailable("Stub backend not enabled")
        for doc in docs:
            self.stub.add(doc)

    def search_by_topic(self, topic: str, k: int = 5) -> list[SearchResult]:
        filters = KnowledgeBaseFilters(topic=topic)
        return self.search("", k=k, filters=filters)
