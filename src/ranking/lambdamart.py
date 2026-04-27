"""LambdaMART learning-to-rank implementation.

Implements a gradient-boosted decision tree ranker using the LambdaMART
objective, which optimizes NDCG directly through lambda gradients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RankingFeatures:
    """Feature vector for a query-document pair."""

    doc_id: int
    bm25_score: float = 0.0
    dense_score: float = 0.0
    query_term_overlap: float = 0.0
    doc_length: int = 0
    title_match: float = 0.0
    idf_sum: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert features to a numpy array."""
        return np.array([
            self.bm25_score, self.dense_score, self.query_term_overlap,
            float(self.doc_length), self.title_match, self.idf_sum,
        ], dtype=np.float32)


@dataclass
class DecisionStump:
    """A single decision stump (depth-1 tree) for gradient boosting."""

    feature_idx: int = 0
    threshold: float = 0.0
    left_value: float = 0.0
    right_value: float = 0.0

    def predict(self, x: np.ndarray) -> float:
        """Predict a score for a single feature vector."""
        return self.left_value if x[self.feature_idx] <= self.threshold else self.right_value


class LambdaMART:
    """LambdaMART learning-to-rank model.

    Uses gradient-boosted decision stumps with lambda gradients that
    directly optimize NDCG. Each boosting round fits a stump to the
    lambda gradients computed from pairwise document comparisons.

    Args:
        n_estimators: Number of boosting rounds.
        learning_rate: Shrinkage factor for each tree.
        sigma: Sigmoid steepness for pairwise probabilities.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        sigma: float = 1.0,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.stumps: list[DecisionStump] = []

    def _compute_lambdas(self, scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute lambda gradients for a single query group."""
        n = len(scores)
        lambdas = np.zeros(n, dtype=np.float64)

        sorted_idx = np.argsort(-scores)
        dcg_discounts = 1.0 / np.log2(np.arange(2, n + 2, dtype=np.float64))
        ideal_sorted = np.argsort(-labels)
        ideal_dcg = sum(
            (2.0 ** labels[ideal_sorted[i]] - 1.0) * dcg_discounts[i]
            for i in range(n)
        )
        if ideal_dcg == 0:
            return lambdas

        rank_map = np.zeros(n, dtype=int)
        for rank, idx in enumerate(sorted_idx):
            rank_map[idx] = rank

        for i in range(n):
            for j in range(n):
                if labels[i] <= labels[j]:
                    continue
                diff = scores[i] - scores[j]
                prob = 1.0 / (1.0 + math.exp(self.sigma * diff))

                rank_i = rank_map[i]
                rank_j = rank_map[j]
                gain_i = 2.0 ** labels[i] - 1.0
                gain_j = 2.0 ** labels[j] - 1.0

                delta_ndcg = abs(
                    (gain_i - gain_j)
                    * (dcg_discounts[rank_i] - dcg_discounts[rank_j])
                ) / ideal_dcg

                lam = prob * delta_ndcg
                lambdas[i] += lam
                lambdas[j] -= lam

        return lambdas

    def _fit_stump(self, features: np.ndarray, gradients: np.ndarray) -> DecisionStump:
        """Fit a decision stump to the lambda gradients."""
        n_samples, n_features = features.shape
        best_loss = float("inf")
        best_stump = DecisionStump()

        for feat_idx in range(n_features):
            values = features[:, feat_idx]
            thresholds = np.unique(values)
            if len(thresholds) > 20:
                thresholds = np.percentile(values, np.linspace(0, 100, 20))

            for thresh in thresholds:
                left_mask = values <= thresh
                right_mask = ~left_mask

                if not left_mask.any() or not right_mask.any():
                    continue

                left_val = gradients[left_mask].mean()
                right_val = gradients[right_mask].mean()

                preds = np.where(left_mask, left_val, right_val)
                loss = np.sum((gradients - preds) ** 2)

                if loss < best_loss:
                    best_loss = loss
                    best_stump = DecisionStump(
                        feature_idx=feat_idx,
                        threshold=float(thresh),
                        left_value=float(left_val),
                        right_value=float(right_val),
                    )

        return best_stump

    def fit(self, query_groups: list[tuple[np.ndarray, np.ndarray]]) -> list[float]:
        """Train the LambdaMART model.

        Args:
            query_groups: List of (features, labels) tuples, one per query.

        Returns:
            Training NDCG scores per boosting round.
        """
        all_features = np.vstack([f for f, _ in query_groups])
        all_scores = np.zeros(len(all_features), dtype=np.float64)
        ndcg_history: list[float] = []

        group_offsets: list[tuple[int, int]] = []
        offset = 0
        for features, _ in query_groups:
            group_offsets.append((offset, offset + len(features)))
            offset += len(features)

        for round_idx in range(self.n_estimators):
            all_lambdas = np.zeros(len(all_features), dtype=np.float64)

            for (start, end), (_, labels) in zip(group_offsets, query_groups):
                group_scores = all_scores[start:end]
                lambdas = self._compute_lambdas(group_scores, labels)
                all_lambdas[start:end] = lambdas

            stump = self._fit_stump(all_features, all_lambdas)
            self.stumps.append(stump)

            for i in range(len(all_features)):
                all_scores[i] += self.learning_rate * stump.predict(all_features[i])

            ndcg_history.append(self._avg_ndcg(all_scores, query_groups, group_offsets))

        return ndcg_history

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict ranking scores for feature vectors."""
        scores = np.zeros(len(features), dtype=np.float64)
        for stump in self.stumps:
            for i in range(len(features)):
                scores[i] += self.learning_rate * stump.predict(features[i])
        return scores

    def rank(self, candidates: list[RankingFeatures]) -> list[RankingFeatures]:
        """Rank candidate documents by predicted relevance."""
        if not candidates:
            return []
        features = np.array([c.to_array() for c in candidates])
        scores = self.predict(features)
        ranked_indices = np.argsort(-scores)
        return [candidates[i] for i in ranked_indices]

    @staticmethod
    def _avg_ndcg(
        scores: np.ndarray,
        query_groups: list[tuple[np.ndarray, np.ndarray]],
        offsets: list[tuple[int, int]],
    ) -> float:
        """Compute average NDCG across query groups."""
        ndcgs = []
        for (start, end), (_, labels) in zip(offsets, query_groups):
            group_scores = scores[start:end]
            sorted_idx = np.argsort(-group_scores)
            dcg = sum(
                (2.0 ** labels[sorted_idx[i]] - 1.0) / math.log2(i + 2)
                for i in range(len(labels))
            )
            ideal_idx = np.argsort(-labels)
            idcg = sum(
                (2.0 ** labels[ideal_idx[i]] - 1.0) / math.log2(i + 2)
                for i in range(len(labels))
            )
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        return float(np.mean(ndcgs)) if ndcgs else 0.0


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n_queries, n_docs, n_features = 20, 10, 6

    query_groups = []
    for _ in range(n_queries):
        features = rng.standard_normal((n_docs, n_features)).astype(np.float32)
        labels = rng.integers(0, 4, size=n_docs).astype(np.float64)
        query_groups.append((features, labels))

    model = LambdaMART(n_estimators=50, learning_rate=0.1)
    history = model.fit(query_groups)

    print(f"Trained LambdaMART with {len(model.stumps)} stumps")
    print(f"NDCG@start: {history[0]:.4f}")
    print(f"NDCG@end:   {history[-1]:.4f}")

    test_features = rng.standard_normal((5, n_features)).astype(np.float32)
    scores = model.predict(test_features)
    print(f"\nPredicted scores: {scores.round(4)}")
