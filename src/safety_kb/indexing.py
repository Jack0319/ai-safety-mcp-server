"""Helpers for ingesting AI safety documents into the vector and doc stores."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from chromadb.utils import embedding_functions  # type: ignore

from safety_kb.models import Document
from safety_kb.retrieval import ChromaBackend, KnowledgeBaseClient


def ingest_documents(
    docs: Sequence[Document],
    *,
    vectorstore_url: str,
    collection: str,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> None:
    """Embed and upsert a batch of documents."""

    backend = ChromaBackend(vectorstore_url, collection)
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)

    ids = [doc.doc_id for doc in docs]
    metadatas = [doc.metadata | {"title": doc.title, "url": doc.url} for doc in docs]
    texts = [doc.text for doc in docs]
    embeddings = embedder(texts)
    backend.collection.upsert(ids=ids, metadatas=metadatas, documents=texts, embeddings=embeddings)


def ingest_directory(path: Path, kb: KnowledgeBaseClient) -> None:
    """Ingest `.jsonl` documents from disk into the stub backend (dev helper)."""

    docs: list[Document] = []
    for jsonl_file in path.glob("*.jsonl"):
        with jsonl_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                import orjson

                payload = orjson.loads(line)
                docs.append(Document(**payload))
    if docs:
        kb.load_stub_documents(docs)
