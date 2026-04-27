"""Neural cross-encoder reranker.

Implements a cross-encoder that jointly encodes query-document pairs
for fine-grained relevance scoring. Falls back to a lightweight
feature-based scorer when transformers are unavailable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RerankedResult:
    """A reranked search result.

    Attributes:
        doc_id: Document identifier.
        score: Cross-encoder relevance score.
        original_rank: Rank before reranking.
    """

    doc_id: int
    score: float
    original_rank: int


class CrossEncoderReranker:
    """Neural cross-encoder for document reranking.

    Uses a transformer cross-encoder model to score query-document pairs
    jointly. When the model is unavailable, falls back to a feature-based
    scorer using term overlap and length normalization.

    Args:
        model_name: HuggingFace cross-encoder model name.
        max_length: Maximum token length for the cross-encoder.
        batch_size: Batch size for inference.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._model: Optional[object] = None
        self._tokenizer: Optional[object] = None

    def _load_model(self) -> bool:
        """Attempt to load the cross-encoder model.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if self._model is not None:
            return True
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self._model.eval()
            return True
        except (ImportError, Exception):
            return False

    def _neural_score(
        self, query: str, documents: list[str]
    ) -> list[float]:
        """Score query-document pairs using the neural cross-encoder.

        Args:
            query: Query string.
            documents: List of document texts.

        Returns:
            List of relevance scores.
        """
        import torch

        scores: list[float] = []
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i : i + self.batch_size]
            pairs = [[query, doc] for doc in batch_docs]
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits.squeeze(-1)
                scores.extend(logits.tolist())
        return scores

    @staticmethod
    def _fallback_score(query: str, document: str) -> float:
        """Compute a lightweight relevance score without neural models.

        Uses term overlap, position weighting, and length normalization
        as a proxy for cross-encoder scoring.

        Args:
            query: Query string.
            document: Document text.

        Returns:
            Heuristic relevance score in [0, 1].
        """
        query_terms = set(query.lower().split())
        doc_terms = document.lower().split()
        doc_term_set = set(doc_terms)

        if not query_terms or not doc_terms:
            return 0.0

        overlap = len(query_terms & doc_term_set)
        overlap_ratio = overlap / len(query_terms)

        position_score = 0.0
        for qt in query_terms:
            for pos, dt in enumerate(doc_terms[:50]):
                if qt == dt:
                    position_score += 1.0 / (1.0 + pos)
                    break

        length_penalty = 1.0 / (1.0 + math.log1p(max(0, len(doc_terms) - 100)))

        score = 0.5 * overlap_ratio + 0.3 * min(position_score, 1.0) + 0.2 * length_penalty
        return min(score, 1.0)

    def rerank(
        self,
        query: str,
        doc_ids: list[int],
        doc_texts: list[str],
        top_k: Optional[int] = None,
    ) -> list[RerankedResult]:
        """Rerank documents for a given query.

        Attempts neural cross-encoder scoring first, falls back to
        heuristic scoring if the model is unavailable.

        Args:
            query: Query string.
            doc_ids: List of document identifiers.
            doc_texts: List of document texts (parallel to doc_ids).
            top_k: Number of top results to return (None = all).

        Returns:
            List of RerankedResult sorted by descending relevance score.
        """
        if len(doc_ids) != len(doc_texts):
            raise ValueError("doc_ids and doc_texts must have the same length")

        if self._load_model():
            scores = self._neural_score(query, doc_texts)
        else:
            scores = [self._fallback_score(query, doc) for doc in doc_texts]

        results = [
            RerankedResult(doc_id=doc_id, score=score, original_rank=rank)
            for rank, (doc_id, score) in enumerate(zip(doc_ids, scores), start=1)
        ]
        results.sort(key=lambda x: x.score, reverse=True)

        if top_k is not None:
            results = results[:top_k]
        return results

    def score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair.

        Args:
            query: Query string.
            document: Document text.

        Returns:
            Relevance score.
        """
        results = self.rerank(query, [0], [document])
        return results[0].score if results else 0.0


if __name__ == "__main__":
    reranker = CrossEncoderReranker()

    query = "What is information retrieval?"
    documents = [
        "Information retrieval is the science of searching for information in documents.",
        "Machine learning models can classify images into categories.",
        "Search engines use inverted indexes for fast document retrieval.",
        "The weather today is sunny with a high of 75 degrees.",
        "BM25 and TF-IDF are classic information retrieval scoring functions.",
    ]
    doc_ids = list(range(len(documents)))

    results = reranker.rerank(query, doc_ids, documents, top_k=3)
    print(f"Query: '{query}'\n")
    print(f"{'Rank':>4} {'Doc':>4} {'Score':>8} {'Original':>9}")
    print("-" * 30)
    for rank, r in enumerate(results, 1):
        print(f"{rank:>4} {r.doc_id:>4} {r.score:>8.4f} {r.original_rank:>9}")
        print(f"     {documents[r.doc_id][:70]}")
