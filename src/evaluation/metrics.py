"""Search evaluation metrics: NDCG, MAP, MRR.

Implements standard information retrieval evaluation metrics for
measuring search quality against ground-truth relevance judgments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class EvalResult:
    """Evaluation result for a single query.

    Attributes:
        query_id: Query identifier.
        ndcg: Normalized Discounted Cumulative Gain.
        average_precision: Average Precision.
        reciprocal_rank: Reciprocal Rank (1/rank of first relevant doc).
        precision_at_k: Precision at the cutoff k.
        recall_at_k: Recall at the cutoff k.
    """

    query_id: str
    ndcg: float = 0.0
    average_precision: float = 0.0
    reciprocal_rank: float = 0.0
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0


def dcg(relevances: list[float], k: Optional[int] = None) -> float:
    """Compute Discounted Cumulative Gain.

    DCG@k = sum_{i=1}^{k} (2^{rel_i} - 1) / log2(i + 1)

    Args:
        relevances: List of relevance scores in ranked order.
        k: Cutoff position (None = use all).

    Returns:
        DCG score.
    """
    if k is not None:
        relevances = relevances[:k]
    return sum(
        (2.0 ** rel - 1.0) / math.log2(i + 2)
        for i, rel in enumerate(relevances)
    )


def ndcg(relevances: list[float], k: Optional[int] = None) -> float:
    """Compute Normalized Discounted Cumulative Gain.

    NDCG@k = DCG@k / IDCG@k where IDCG is the ideal DCG
    (relevances sorted in descending order).

    Args:
        relevances: List of relevance scores in ranked order.
        k: Cutoff position (None = use all).

    Returns:
        NDCG score in [0, 1].
    """
    actual_dcg = dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal_relevances, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def average_precision(
    retrieved_ids: list[int],
    relevant_ids: set[int],
    k: Optional[int] = None,
) -> float:
    """Compute Average Precision.

    AP = (1/|relevant|) * sum_{k: doc_k is relevant} Precision@k

    Args:
        retrieved_ids: List of retrieved document IDs in ranked order.
        relevant_ids: Set of relevant document IDs.
        k: Cutoff position (None = use all).

    Returns:
        Average Precision score in [0, 1].
    """
    if not relevant_ids:
        return 0.0

    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    hits = 0
    sum_precision = 0.0
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            hits += 1
            sum_precision += hits / i

    return sum_precision / len(relevant_ids)


def mean_average_precision(
    queries: list[tuple[list[int], set[int]]],
    k: Optional[int] = None,
) -> float:
    """Compute Mean Average Precision across multiple queries.

    Args:
        queries: List of (retrieved_ids, relevant_ids) tuples.
        k: Cutoff position.

    Returns:
        MAP score in [0, 1].
    """
    if not queries:
        return 0.0
    ap_scores = [average_precision(ret, rel, k) for ret, rel in queries]
    return sum(ap_scores) / len(ap_scores)


def reciprocal_rank(
    retrieved_ids: list[int], relevant_ids: set[int]
) -> float:
    """Compute Reciprocal Rank.

    RR = 1 / rank of the first relevant document.

    Args:
        retrieved_ids: List of retrieved document IDs in ranked order.
        relevant_ids: Set of relevant document IDs.

    Returns:
        Reciprocal rank score in [0, 1].
    """
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(
    queries: list[tuple[list[int], set[int]]],
) -> float:
    """Compute Mean Reciprocal Rank across multiple queries.

    Args:
        queries: List of (retrieved_ids, relevant_ids) tuples.

    Returns:
        MRR score in [0, 1].
    """
    if not queries:
        return 0.0
    rr_scores = [reciprocal_rank(ret, rel) for ret, rel in queries]
    return sum(rr_scores) / len(rr_scores)


def precision_at_k(
    retrieved_ids: list[int], relevant_ids: set[int], k: int
) -> float:
    """Compute Precision at rank k.

    Args:
        retrieved_ids: List of retrieved document IDs in ranked order.
        relevant_ids: Set of relevant document IDs.
        k: Cutoff position.

    Returns:
        Precision@k score in [0, 1].
    """
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def recall_at_k(
    retrieved_ids: list[int], relevant_ids: set[int], k: int
) -> float:
    """Compute Recall at rank k.

    Args:
        retrieved_ids: List of retrieved document IDs in ranked order.
        relevant_ids: Set of relevant document IDs.
        k: Cutoff position.

    Returns:
        Recall@k score in [0, 1].
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def evaluate_query(
    query_id: str,
    retrieved_ids: list[int],
    relevance_labels: dict[int, float],
    k: int = 10,
) -> EvalResult:
    """Evaluate a single query across all metrics.

    Args:
        query_id: Query identifier.
        retrieved_ids: Retrieved document IDs in ranked order.
        relevance_labels: Mapping of doc_id → relevance grade.
        k: Cutoff position for metrics.

    Returns:
        EvalResult with all computed metrics.
    """
    relevances = [relevance_labels.get(doc_id, 0.0) for doc_id in retrieved_ids]
    relevant_ids = {doc_id for doc_id, rel in relevance_labels.items() if rel > 0}

    return EvalResult(
        query_id=query_id,
        ndcg=ndcg(relevances, k),
        average_precision=average_precision(retrieved_ids, relevant_ids, k),
        reciprocal_rank=reciprocal_rank(retrieved_ids, relevant_ids),
        precision_at_k=precision_at_k(retrieved_ids, relevant_ids, k),
        recall_at_k=recall_at_k(retrieved_ids, relevant_ids, k),
    )


def evaluate_run(
    run: dict[str, list[int]],
    qrels: dict[str, dict[int, float]],
    k: int = 10,
) -> dict[str, float]:
    """Evaluate a full retrieval run across all queries.

    Args:
        run: Mapping of query_id → list of retrieved doc_ids.
        qrels: Mapping of query_id → {doc_id: relevance_grade}.
        k: Cutoff position.

    Returns:
        Dictionary of aggregated metric scores.
    """
    results = []
    for query_id, retrieved in run.items():
        labels = qrels.get(query_id, {})
        results.append(evaluate_query(query_id, retrieved, labels, k))

    n = len(results) if results else 1
    return {
        f"NDCG@{k}": sum(r.ndcg for r in results) / n,
        f"MAP@{k}": sum(r.average_precision for r in results) / n,
        "MRR": sum(r.reciprocal_rank for r in results) / n,
        f"P@{k}": sum(r.precision_at_k for r in results) / n,
        f"R@{k}": sum(r.recall_at_k for r in results) / n,
    }


if __name__ == "__main__":
    print("=== NDCG Examples ===")
    perfect = [3.0, 2.0, 1.0, 0.0]
    reversed_order = [0.0, 1.0, 2.0, 3.0]
    print(f"Perfect ranking NDCG@4:  {ndcg(perfect, 4):.4f}")
    print(f"Reversed ranking NDCG@4: {ndcg(reversed_order, 4):.4f}")

    print("\n=== MAP / MRR Examples ===")
    retrieved = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
    relevant = {1, 2, 5}
    print(f"AP@10:  {average_precision(retrieved, relevant, 10):.4f}")
    print(f"RR:     {reciprocal_rank(retrieved, relevant):.4f}")
    print(f"P@5:    {precision_at_k(retrieved, relevant, 5):.4f}")
    print(f"R@5:    {recall_at_k(retrieved, relevant, 5):.4f}")

    print("\n=== Full Run Evaluation ===")
    run = {
        "q1": [1, 3, 5, 7, 9],
        "q2": [2, 4, 6, 8, 10],
        "q3": [1, 2, 3, 4, 5],
    }
    qrels = {
        "q1": {1: 3.0, 5: 2.0, 7: 1.0},
        "q2": {2: 2.0, 6: 1.0, 10: 3.0},
        "q3": {1: 1.0, 3: 2.0, 5: 3.0},
    }
    metrics = evaluate_run(run, qrels, k=5)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
