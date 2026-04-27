"""Hybrid search with Reciprocal Rank Fusion (RRF).

Combines BM25 sparse retrieval and dense semantic retrieval using RRF
to produce a single ranked list that benefits from both signal types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.retrieval.bm25 import BM25Retriever, SearchResult
from src.retrieval.dense import DenseResult, DenseRetriever


@dataclass
class HybridResult:
    """A fused search result combining sparse and dense signals.

    Attributes:
        doc_id: Document identifier.
        rrf_score: Reciprocal rank fusion score.
        bm25_rank: Rank from BM25 retrieval (None if not retrieved).
        dense_rank: Rank from dense retrieval (None if not retrieved).
        bm25_score: Raw BM25 score.
        dense_score: Raw dense similarity score.
    """

    doc_id: int
    rrf_score: float
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    bm25_score: float = 0.0
    dense_score: float = 0.0


class HybridRetriever:
    """Hybrid retriever combining BM25 and dense search via RRF.

    Reciprocal Rank Fusion merges two ranked lists by summing reciprocal
    ranks: RRF(d) = sum(1 / (k + rank_i(d))) across all systems i.

    The constant k (default 60) controls how much weight is given to
    documents ranked lower in individual lists.

    Args:
        bm25_retriever: BM25 sparse retriever.
        dense_retriever: Dense semantic retriever.
        rrf_k: RRF smoothing constant (default 60).
        bm25_weight: Weight for BM25 RRF contribution.
        dense_weight: Weight for dense RRF contribution.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> None:
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        fetch_k: int = 100,
    ) -> list[HybridResult]:
        """Retrieve documents using hybrid RRF fusion.

        Fetches candidates from both BM25 and dense retrievers, then
        fuses their rankings using weighted reciprocal rank fusion.

        Args:
            query: Raw query string.
            top_k: Number of final results to return.
            fetch_k: Number of candidates to fetch from each retriever.

        Returns:
            List of HybridResult objects sorted by descending RRF score.
        """
        bm25_results = self.bm25_retriever.retrieve(query, top_k=fetch_k)
        dense_results = self.dense_retriever.retrieve(query, top_k=fetch_k)
        return self.fuse(bm25_results, dense_results, top_k)

    def fuse(
        self,
        bm25_results: list[SearchResult],
        dense_results: list[DenseResult],
        top_k: int = 10,
    ) -> list[HybridResult]:
        """Fuse two ranked lists using Reciprocal Rank Fusion.

        RRF score for document d:
            RRF(d) = w_bm25 / (k + rank_bm25(d)) + w_dense / (k + rank_dense(d))

        Args:
            bm25_results: Ranked results from BM25 retrieval.
            dense_results: Ranked results from dense retrieval.
            top_k: Number of results to return.

        Returns:
            Fused results sorted by RRF score.
        """
        bm25_ranks: dict[int, int] = {}
        bm25_scores: dict[int, float] = {}
        for rank, result in enumerate(bm25_results, start=1):
            bm25_ranks[result.doc_id] = rank
            bm25_scores[result.doc_id] = result.score

        dense_ranks: dict[int, int] = {}
        dense_scores: dict[int, float] = {}
        for rank, result in enumerate(dense_results, start=1):
            dense_ranks[result.doc_id] = rank
            dense_scores[result.doc_id] = result.score

        all_doc_ids = set(bm25_ranks.keys()) | set(dense_ranks.keys())
        fused: list[HybridResult] = []

        for doc_id in all_doc_ids:
            rrf_score = 0.0
            bm25_rank = bm25_ranks.get(doc_id)
            dense_rank = dense_ranks.get(doc_id)

            if bm25_rank is not None:
                rrf_score += self.bm25_weight / (self.rrf_k + bm25_rank)
            if dense_rank is not None:
                rrf_score += self.dense_weight / (self.rrf_k + dense_rank)

            fused.append(
                HybridResult(
                    doc_id=doc_id,
                    rrf_score=rrf_score,
                    bm25_rank=bm25_rank,
                    dense_rank=dense_rank,
                    bm25_score=bm25_scores.get(doc_id, 0.0),
                    dense_score=dense_scores.get(doc_id, 0.0),
                )
            )

        fused.sort(key=lambda x: x.rrf_score, reverse=True)
        return fused[:top_k]


if __name__ == "__main__":
    import numpy as np

    from src.indexing.dense_index import DenseIndex
    from src.indexing.inverted_index import InvertedIndex

    corpus = [
        (0, "Information retrieval systems use inverted indexes"),
        (1, "BM25 is a probabilistic retrieval model based on term frequency"),
        (2, "Dense retrieval uses neural embeddings for semantic matching"),
        (3, "Hybrid search combines sparse and dense signals for better recall"),
        (4, "Reciprocal rank fusion merges multiple ranked lists effectively"),
    ]

    inv_index = InvertedIndex()
    for doc_id, text in corpus:
        inv_index.add_document(doc_id, text)

    rng = np.random.default_rng(42)
    dense_idx = DenseIndex(dimension=64)
    doc_ids = [d[0] for d in corpus]
    embeddings = rng.standard_normal((len(corpus), 64)).astype(np.float32)
    dense_idx.add(doc_ids, embeddings)

    bm25_ret = BM25Retriever(index=inv_index)
    dense_ret = DenseRetriever(index=dense_idx)
    hybrid = HybridRetriever(bm25_ret, dense_ret, rrf_k=60)

    query = "hybrid search retrieval"
    results = hybrid.retrieve(query, top_k=5)
    print(f"Query: '{query}'\n")
    print(f"{'Doc':>4} {'RRF':>8} {'BM25 Rank':>10} {'Dense Rank':>11}")
    print("-" * 40)
    for r in results:
        bm25_r = str(r.bm25_rank) if r.bm25_rank else "-"
        dense_r = str(r.dense_rank) if r.dense_rank else "-"
        print(f"{r.doc_id:>4} {r.rrf_score:>8.5f} {bm25_r:>10} {dense_r:>11}")
