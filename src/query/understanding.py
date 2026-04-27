"""Query intent classification module.

Classifies search queries into intent categories (navigational, informational,
transactional) and extracts structured signals to guide retrieval strategy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class QueryIntent(Enum):
    """Search query intent categories."""

    NAVIGATIONAL = "navigational"
    INFORMATIONAL = "informational"
    TRANSACTIONAL = "transactional"
    AMBIGUOUS = "ambiguous"


@dataclass
class QueryAnalysis:
    """Structured analysis of a search query.

    Attributes:
        original_query: The raw query string.
        intent: Classified intent category.
        confidence: Confidence score for the classification.
        entities: Extracted named entities or key phrases.
        is_question: Whether the query is phrased as a question.
        has_negation: Whether the query contains negation.
        query_length: Number of tokens in the query.
        suggested_weights: Suggested BM25/dense weights based on intent.
    """

    original_query: str
    intent: QueryIntent = QueryIntent.AMBIGUOUS
    confidence: float = 0.0
    entities: list[str] = field(default_factory=list)
    is_question: bool = False
    has_negation: bool = False
    query_length: int = 0
    suggested_weights: dict[str, float] = field(default_factory=dict)


# Pattern-based intent signals
_NAVIGATIONAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(go to|navigate|open|visit|homepage|official)\b", re.IGNORECASE),
    re.compile(r"\.(com|org|net|io|gov|edu)\b", re.IGNORECASE),
    re.compile(r"\b(login|sign in|sign up|dashboard)\b", re.IGNORECASE),
]

_INFORMATIONAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(what|how|why|when|where|who|which|explain|define)\b", re.IGNORECASE),
    re.compile(r"\b(tutorial|guide|example|documentation|overview)\b", re.IGNORECASE),
    re.compile(r"\b(difference between|compare|vs\.?|versus)\b", re.IGNORECASE),
]

_TRANSACTIONAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(buy|purchase|order|download|install|subscribe)\b", re.IGNORECASE),
    re.compile(r"\b(price|cost|deal|discount|free|trial)\b", re.IGNORECASE),
    re.compile(r"\b(best|top|review|rating|recommend)\b", re.IGNORECASE),
]

_QUESTION_PATTERN = re.compile(
    r"^(what|how|why|when|where|who|which|is|are|can|do|does|did|will|should)\b",
    re.IGNORECASE,
)

_NEGATION_PATTERN = re.compile(
    r"\b(not|no|without|never|don't|doesn't|isn't|aren't|won't|can't|exclude)\b",
    re.IGNORECASE,
)


class QueryUnderstanding:
    """Query understanding engine for intent classification.

    Uses pattern matching and heuristic scoring to classify queries
    and extract structured signals for retrieval optimization.
    """

    def __init__(self) -> None:
        self._intent_patterns: dict[QueryIntent, list[re.Pattern[str]]] = {
            QueryIntent.NAVIGATIONAL: _NAVIGATIONAL_PATTERNS,
            QueryIntent.INFORMATIONAL: _INFORMATIONAL_PATTERNS,
            QueryIntent.TRANSACTIONAL: _TRANSACTIONAL_PATTERNS,
        }

    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze a query for intent, entities, and retrieval signals.

        Args:
            query: Raw query string.

        Returns:
            QueryAnalysis with classified intent and extracted signals.
        """
        tokens = query.strip().split()
        analysis = QueryAnalysis(
            original_query=query,
            query_length=len(tokens),
            is_question=bool(_QUESTION_PATTERN.match(query)),
            has_negation=bool(_NEGATION_PATTERN.search(query)),
        )

        intent_scores = self._score_intents(query)
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            best_score = intent_scores[best_intent]
            total = sum(intent_scores.values())
            analysis.intent = best_intent
            analysis.confidence = best_score / total if total > 0 else 0.0
        else:
            analysis.intent = QueryIntent.AMBIGUOUS
            analysis.confidence = 0.5

        analysis.entities = self._extract_entities(query)
        analysis.suggested_weights = self._suggest_weights(analysis)

        return analysis

    def _score_intents(self, query: str) -> dict[QueryIntent, float]:
        """Score each intent category by pattern matches.

        Args:
            query: Raw query string.

        Returns:
            Dictionary of intent → match score.
        """
        scores: dict[QueryIntent, float] = {}
        for intent, patterns in self._intent_patterns.items():
            score = sum(1.0 for p in patterns if p.search(query))
            if score > 0:
                scores[intent] = score
        return scores

    def _extract_entities(self, query: str) -> list[str]:
        """Extract potential named entities from the query.

        Uses capitalization and multi-word phrase heuristics.

        Args:
            query: Raw query string.

        Returns:
            List of extracted entity strings.
        """
        entities: list[str] = []
        capitalized = re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", query)
        for entity in capitalized:
            if entity.lower() not in {"the", "a", "an", "is", "are", "what", "how"}:
                entities.append(entity)

        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)

        return list(set(entities))

    def _suggest_weights(self, analysis: QueryAnalysis) -> dict[str, float]:
        """Suggest BM25/dense retrieval weights based on query analysis.

        Navigational queries favor exact BM25 matching.
        Informational queries favor dense semantic matching.
        Transactional queries use balanced weights.

        Args:
            analysis: Query analysis results.

        Returns:
            Dictionary with 'bm25_weight' and 'dense_weight' keys.
        """
        weight_map = {
            QueryIntent.NAVIGATIONAL: {"bm25_weight": 0.8, "dense_weight": 0.2},
            QueryIntent.INFORMATIONAL: {"bm25_weight": 0.3, "dense_weight": 0.7},
            QueryIntent.TRANSACTIONAL: {"bm25_weight": 0.5, "dense_weight": 0.5},
            QueryIntent.AMBIGUOUS: {"bm25_weight": 0.5, "dense_weight": 0.5},
        }
        weights = weight_map[analysis.intent].copy()

        if analysis.query_length <= 2:
            weights["bm25_weight"] = min(weights["bm25_weight"] + 0.1, 1.0)
            weights["dense_weight"] = max(weights["dense_weight"] - 0.1, 0.0)

        if analysis.is_question:
            weights["dense_weight"] = min(weights["dense_weight"] + 0.1, 1.0)
            weights["bm25_weight"] = max(weights["bm25_weight"] - 0.1, 0.0)

        return weights


if __name__ == "__main__":
    engine = QueryUnderstanding()

    queries = [
        "github.com login",
        "how does BM25 scoring work",
        "buy noise cancelling headphones",
        "python tutorial for beginners",
        "best laptop 2024 review",
        "what is reciprocal rank fusion",
        "Amazon Web Services dashboard",
        "neural network",
    ]

    for q in queries:
        result = engine.analyze(q)
        print(f"Query: '{q}'")
        print(f"  Intent: {result.intent.value} (confidence: {result.confidence:.2f})")
        print(f"  Question: {result.is_question}, Length: {result.query_length}")
        if result.entities:
            print(f"  Entities: {result.entities}")
        print(f"  Weights: BM25={result.suggested_weights.get('bm25_weight', 0):.1f}, "
              f"Dense={result.suggested_weights.get('dense_weight', 0):.1f}")
        print()
