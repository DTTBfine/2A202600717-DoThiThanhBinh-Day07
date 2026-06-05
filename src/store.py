from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            # TODO: initialize chromadb client + collection
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        record_id = f"{doc.id}-{self._next_index}"
        self._next_index += 1
        return {
            "id": record_id,
            "doc_id": doc.id,
            "content": doc.content,
            "metadata": {**doc.metadata, "doc_id": doc.id},
            "embedding": self._embedding_fn(doc.content),
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0 or not query or not records:
            return []

        query_embedding = self._embedding_fn(query)
        scored: list[dict[str, Any]] = []

        for record in records:
            score = _dot(query_embedding, record["embedding"])
            scored.append(
                {
                    "id": record["id"],
                    "doc_id": record["doc_id"],
                    "content": record["content"],
                    "metadata": record["metadata"],
                    "score": score,
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        records = [self._make_record(doc) for doc in docs]

        if self._use_chroma and self._collection is not None:
            try:
                self._collection.add(
                    ids=[record["id"] for record in records],
                    documents=[record["content"] for record in records],
                    embeddings=[record["embedding"] for record in records],
                    metadatas=[record["metadata"] for record in records],
                )
            except Exception:
                pass

        self._store.extend(records)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self._use_chroma and self._collection is not None:
            try:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    include=['metadatas', 'documents', 'distances'],
                )
                hits = []
                for idx, score in enumerate(results['distances'][0]):
                    hits.append(
                        {
                            'id': results['ids'][0][idx],
                            'doc_id': results['metadatas'][0][idx].get('doc_id'),
                            'content': results['documents'][0][idx],
                            'metadata': results['metadatas'][0][idx],
                            'score': 1.0 - score if score is not None else 0.0,
                        }
                    )
                return hits
            except Exception:
                pass

        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        filtered_records = self._store
        if metadata_filter:
            filtered_records = [
                record
                for record in self._store
                if all(record["metadata"].get(key) == value for key, value in metadata_filter.items())
            ]
        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        original_length = len(self._store)
        self._store = [record for record in self._store if record["doc_id"] != doc_id]
        return len(self._store) < original_length
