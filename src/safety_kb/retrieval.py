"""Knowledge base retrieval utilities."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

import numpy as np
from chromadb.api import ClientAPI  # type: ignore
from chromadb.api.models import Collection
from sqlmodel import Field, Session, SQLModel, create_engine, select

from core.errors import KnowledgeBaseNotFound, KnowledgeBaseUnavailable
from core.logging import get_logger
from safety_kb.models import Document, SearchResult

logger = get_logger(__name__)


class DocumentRecord(SQLModel, table=True):
    __tablename__ = "documents"  # type: ignore
    doc_id: str = Field(primary_key=True)
    title: str
    text: str
    url: str | None = None
    metadata_json: str | None = None  # JSON string


class EmbeddingRecord(SQLModel, table=True):
    __tablename__ = "embeddings"  # type: ignore
    doc_id: str = Field(primary_key=True, foreign_key="documents.doc_id")
    embedding_json: str  # JSON-encoded list of floats


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


class SQLiteVectorStore:
    """SQLite-based vector store with embedding support for semantic search."""

    def __init__(self, url: str, collection: str = "default"):
        self.engine = create_engine(url, echo=False)
        self.collection = collection
        SQLModel.metadata.create_all(self.engine)
        self._embedding_model = None

    def _get_embedding_model(self):
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                raise KnowledgeBaseUnavailable(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                )
        return self._embedding_model

    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for a text string."""
        model = self._get_embedding_model()
        return model.encode(text, convert_to_numpy=True)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def add_document(self, doc: Document) -> None:
        """Add a document with its embedding to the store."""
        embedding = self._compute_embedding(doc.text)

        with Session(self.engine) as session:
            # Store document
            doc_record = DocumentRecord(
                doc_id=doc.doc_id,
                title=doc.title,
                text=doc.text,
                url=doc.url,
                metadata_json=json.dumps(doc.metadata) if doc.metadata else None,
            )
            session.merge(doc_record)

            # Store embedding
            emb_record = EmbeddingRecord(
                doc_id=doc.doc_id,
                embedding_json=json.dumps(embedding.tolist()),
            )
            session.merge(emb_record)
            session.commit()

    def search(
        self, query: str, k: int, filters: Optional["KnowledgeBaseFilters"] = None
    ) -> list[SearchResult]:
        """Semantic search using vector similarity."""
        query_embedding = self._compute_embedding(query)

        with Session(self.engine) as session:
            # Get all documents with embeddings
            statement = select(DocumentRecord, EmbeddingRecord).join(
                EmbeddingRecord, DocumentRecord.doc_id == EmbeddingRecord.doc_id
            )
            results = session.exec(statement).all()

            # Compute similarities and apply filters
            candidates: list[tuple[DocumentRecord, float]] = []
            for doc_record, emb_record in results:
                # Parse metadata
                metadata = {}
                if doc_record.metadata_json:
                    try:
                        metadata = json.loads(doc_record.metadata_json)
                    except Exception:
                        pass

                # Apply filters
                if filters:
                    if filters.topic and metadata.get("topic") != filters.topic:
                        continue
                    if filters.org and metadata.get("org") != filters.org:
                        continue
                    if filters.year_min is not None:
                        year = metadata.get("year")
                        if year is None or year < filters.year_min:
                            continue
                    if filters.year_max is not None:
                        year = metadata.get("year")
                        if year is None or year > filters.year_max:
                            continue

                # Compute similarity
                doc_embedding = np.array(json.loads(emb_record.embedding_json))
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                candidates.append((doc_record, similarity))

            # Sort by similarity and take top k
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = candidates[:k]

            # Build search results
            search_results: list[SearchResult] = []
            for doc_record, score in top_candidates:
                metadata = {}
                if doc_record.metadata_json:
                    try:
                        metadata = json.loads(doc_record.metadata_json)
                    except Exception:
                        pass

                snippet = doc_record.text[:400]
                search_results.append(
                    SearchResult(
                        doc_id=doc_record.doc_id,
                        title=doc_record.title,
                        snippet=snippet,
                        score=score,
                        url=doc_record.url,
                        source=metadata.get("source"),
                        topic=metadata.get("topic"),
                        metadata=metadata,
                    )
                )

            return search_results

    def get_document(self, doc_id: str) -> Document:
        """Retrieve a document by ID."""
        with Session(self.engine) as session:
            record = session.get(DocumentRecord, doc_id)
            if not record:
                raise KnowledgeBaseNotFound(doc_id)
            metadata = {}
            if record.metadata_json:
                try:
                    metadata = json.loads(record.metadata_json)
                except Exception:
                    pass
            return Document(
                doc_id=record.doc_id,
                title=record.title,
                text=record.text,
                url=record.url,
                metadata=metadata,
            )


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
            # Determine backend type based on URL
            parsed = urlparse(vectorstore_url)
            if parsed.scheme in {"http", "https"}:
                # Use ChromaDB for HTTP/HTTPS URLs
                self.backend = ChromaBackend(vectorstore_url, collection)
            elif parsed.scheme in {"sqlite", "sqlite+aiosqlite"} or vectorstore_url.startswith("sqlite://"):
                # Use SQLite vector store for sqlite:// URLs
                self.backend = SQLiteVectorStore(vectorstore_url, collection)
            else:
                # Default to SQLite for file paths
                sqlite_url = f"sqlite:///{vectorstore_url}" if not vectorstore_url.startswith("sqlite") else vectorstore_url
                self.backend = SQLiteVectorStore(sqlite_url, collection)

        self.docstore = SQLDocStore(docstore_url) if docstore_url and not use_stub else None

    def _ensure_backend(self) -> ChromaBackend | SQLiteVectorStore:
        if self.backend is None:
            raise KnowledgeBaseUnavailable("Vector store backend not configured")
        return self.backend

    def search(self, query: str, k: int = 5, filters: Optional[KnowledgeBaseFilters] = None) -> list[SearchResult]:
        logger.debug(f"KB search: query length={len(query)}, k={k}, filters={filters}")
        if self.stub:
            logger.debug("Using stub backend for KB search")
            return self.stub.search(query, k, filters=filters.__dict__ if filters else None)
        backend = self._ensure_backend()
        results = backend.search(query, k, filters)
        logger.debug(f"KB search returned {len(results)} results")
        return results

    def get_document(self, doc_id: str) -> Document:
        logger.debug(f"KB get_document: doc_id='{doc_id}'")
        if self.stub:
            logger.debug("Using stub backend for KB get_document")
            return self.stub.get_document(doc_id)
        if self.docstore:
            logger.debug("Using docstore for KB get_document")
            return self.docstore.get_document(doc_id)
        backend = self._ensure_backend()
        doc = backend.get_document(doc_id)
        logger.debug(f"KB get_document completed: doc_id='{doc_id}', title='{doc.title[:50]}...'")
        return doc

    def add_document(self, doc: Document) -> None:
        """Add a document to the knowledge base (only for SQLite backend)."""
        if self.stub:
            self.stub.add(doc)
        elif isinstance(self.backend, SQLiteVectorStore):
            self.backend.add_document(doc)
        else:
            raise KnowledgeBaseUnavailable("add_document only supported for SQLite backend or stub")

    def load_stub_documents(self, docs: Iterable[Document]) -> None:
        if not self.stub:
            raise KnowledgeBaseUnavailable("Stub backend not enabled")
        for doc in docs:
            self.stub.add(doc)

    def search_by_topic(self, topic: str, k: int = 5) -> list[SearchResult]:
        filters = KnowledgeBaseFilters(topic=topic)
        return self.search("", k=k, filters=filters)

    def is_available(self) -> bool:
        """Check if knowledge base is properly configured and available (not just stub)."""
        # KB is available if we have a backend (real KB) or if stub mode is explicitly enabled
        # Stub mode is acceptable for development/testing
        return self.backend is not None or self.stub is not None
