"""Dense semantic retrieval module.

Provides a retriever interface over the FAISS dense index using
bi-encoder embeddings for semantic similarity search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.indexing.dense_index import DenseIndex


@dataclass
class DenseResult:
    """A single dense retrieval result.

    Attributes:
        doc_id: Document identifier.
        score: Cosine similarity score.
        method: Retrieval method identifier.
    """

    doc_id: int
    score: float
    method: str = "dense"


class DenseRetriever:
    """Dense semantic retrieval using bi-encoder embeddings.

    Encodes queries with a sentence transformer and retrieves nearest
    neighbors from the FAISS index by cosine similarity.

    Args:
        index: Pre-built dense vector index.
        model_name: Sentence transformer model for query encoding.
        default_top_k: Default number of results to return.
    """

    def __init__(
        self,
        index: Optional[DenseIndex] = None,
        model_name: str = "all-MiniLM-L6-v2",
        default_top_k: int = 100,
    ) -> None:
        self.index = index or DenseIndex()
        self.model_name = model_name
        self.default_top_k = default_top_k
        self._encoder: Optional[object] = None

    def _get_encoder(self) -> object:
        """Lazy-load the sentence transformer encoder.

        Returns:
            SentenceTransformer model, or None if unavailable.
        """
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(self.model_name)
            except (ImportError, Exception):
                self._encoder = None
        return self._encoder

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query string into a dense embedding.

        Falls back to a deterministic pseudo-random embedding if the
        sentence transformer is unavailable.

        Args:
            query: Raw query string.

        Returns:
            Query embedding vector of shape (dimension,).
        """
        encoder = self._get_encoder()
        if encoder is not None:
            return encoder.encode(query, convert_to_numpy=True)
        rng = np.random.default_rng(hash(query) % (2**31))
        return rng.standard_normal(self.index.dimension).astype(np.float32)

    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> list[DenseResult]:
        """Retrieve documents by semantic similarity to the query.

        Args:
            query: Raw query string.
            top_k: Number of results to return.

        Returns:
            List of DenseResult objects sorted by descending similarity.
        """
        k = top_k or self.default_top_k
        query_embedding = self.encode_query(query)
        raw_results = self.index.search(query_embedding, top_k=k)
        return [
            DenseResult(doc_id=doc_id, score=score, method="dense")
            for doc_id, score in raw_results
        ]

    def retrieve_by_embedding(
        self, embedding: np.ndarray, top_k: Optional[int] = None
    ) -> list[DenseResult]:
        """Retrieve documents using a pre-computed query embedding.

        Args:
            embedding: Query vector of shape (dimension,).
            top_k: Number of results to return.

        Returns:
            List of DenseResult objects.
        """
        k = top_k or self.default_top_k
        raw_results = self.index.search(embedding, top_k=k)
        return [
            DenseResult(doc_id=doc_id, score=score, method="dense")
            for doc_id, score in raw_results
        ]

    @property
    def doc_count(self) -> int:
        """Return the number of indexed documents."""
        return self.index.size


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    dim = 128
    n_docs = 50

    index = DenseIndex(dimension=dim)
    doc_ids = list(range(n_docs))
    embeddings = rng.standard_normal((n_docs, dim)).astype(np.float32)
    index.add(doc_ids, embeddings)

    retriever = DenseRetriever(index=index, default_top_k=5)
    print(f"Dense index: {retriever.doc_count} documents, dim={dim}\n")

    query_vec = rng.standard_normal(dim).astype(np.float32)
    results = retriever.retrieve_by_embedding(query_vec, top_k=5)
    print("Top-5 by embedding similarity:")
    for r in results:
        print(f"  Doc {r.doc_id}: {r.score:.4f} [{r.method}]")
