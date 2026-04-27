"""FAISS-based dense vector index for semantic search.

Wraps FAISS IndexFlatIP (inner product) for nearest-neighbor retrieval
over document embeddings produced by a bi-encoder model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class DenseIndex:
    """Dense vector index using FAISS for approximate nearest-neighbor search.

    Stores document embeddings and supports cosine-similarity retrieval
    via inner product on L2-normalized vectors.

    Attributes:
        dimension: Embedding dimensionality.
        use_gpu: Whether to use GPU FAISS (if available).
    """

    dimension: int = 384
    use_gpu: bool = False
    _index: object = field(default=None, repr=False)
    _doc_ids: list[int] = field(default_factory=list)
    _embeddings: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize the FAISS index."""
        self._build_index()

    def _build_index(self) -> None:
        """Build or rebuild the FAISS inner-product index."""
        try:
            import faiss

            self._index = faiss.IndexFlatIP(self.dimension)
            if self.use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
                except Exception:
                    pass  # Fall back to CPU
        except ImportError:
            self._index = None  # Fallback mode without FAISS

    def add(self, doc_ids: list[int], embeddings: np.ndarray) -> None:
        """Add document embeddings to the index.

        Embeddings are L2-normalized before insertion so inner product
        equals cosine similarity.

        Args:
            doc_ids: List of document identifiers.
            embeddings: Array of shape (n, dimension) with document vectors.
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = (embeddings / norms).astype(np.float32)

        self._doc_ids.extend(doc_ids)

        if self._index is not None:
            self._index.add(normalized)

        if self._embeddings is None:
            self._embeddings = normalized
        else:
            self._embeddings = np.vstack([self._embeddings, normalized])

    def search(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Search for nearest neighbors to a query embedding.

        Args:
            query_embedding: Query vector of shape (dimension,) or (1, dimension).
            top_k: Number of results to return.

        Returns:
            List of (doc_id, similarity_score) tuples, descending by score.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-10)
        query_normalized = (query_embedding / norm).astype(np.float32)

        k = min(top_k, len(self._doc_ids))
        if k == 0:
            return []

        if self._index is not None:
            try:
                import faiss  # noqa: F811

                scores, indices = self._index.search(query_normalized, k)
                results = []
                for i in range(k):
                    idx = int(indices[0][i])
                    if 0 <= idx < len(self._doc_ids):
                        results.append((self._doc_ids[idx], float(scores[0][i])))
                return results
            except Exception:
                pass

        # Fallback: brute-force numpy search
        return self._numpy_search(query_normalized, k)

    def _numpy_search(
        self, query_normalized: np.ndarray, k: int
    ) -> list[tuple[int, float]]:
        """Brute-force cosine similarity search using numpy.

        Args:
            query_normalized: L2-normalized query vector of shape (1, dim).
            k: Number of results.

        Returns:
            List of (doc_id, score) tuples.
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            return []
        similarities = (self._embeddings @ query_normalized.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:k]
        return [
            (self._doc_ids[i], float(similarities[i])) for i in top_indices
        ]

    @property
    def size(self) -> int:
        """Return the number of indexed documents."""
        return len(self._doc_ids)

    def save(self, directory: str) -> None:
        """Save the index to disk.

        Args:
            directory: Output directory path.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        if self._embeddings is not None:
            np.save(str(path / "embeddings.npy"), self._embeddings)

        metadata = {"dimension": self.dimension, "doc_ids": self._doc_ids}
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, directory: str) -> DenseIndex:
        """Load an index from disk.

        Args:
            directory: Directory containing saved index files.

        Returns:
            Reconstructed DenseIndex instance.
        """
        path = Path(directory)
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        idx = cls(dimension=metadata["dimension"])
        embeddings_path = path / "embeddings.npy"
        if embeddings_path.exists():
            embeddings = np.load(str(embeddings_path))
            idx.add(metadata["doc_ids"], embeddings)
        return idx


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    dim = 128
    n_docs = 100

    index = DenseIndex(dimension=dim)
    doc_ids = list(range(n_docs))
    embeddings = rng.standard_normal((n_docs, dim)).astype(np.float32)
    index.add(doc_ids, embeddings)

    print(f"Indexed {index.size} documents with dimension {dim}")

    query = rng.standard_normal(dim).astype(np.float32)
    results = index.search(query, top_k=5)
    print("\nTop-5 nearest neighbors:")
    for doc_id, score in results:
        print(f"  Doc {doc_id}: similarity = {score:.4f}")
