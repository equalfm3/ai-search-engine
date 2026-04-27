"""Query expansion module.

Expands user queries with synonyms, related terms, and pseudo-relevance
feedback to improve recall without sacrificing precision.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExpandedQuery:
    """Result of query expansion.

    Attributes:
        original: The original query string.
        expanded: The expanded query string.
        added_terms: Terms added during expansion.
        method: Expansion method used.
        boost_map: Term → boost weight for expanded terms.
    """

    original: str
    expanded: str
    added_terms: list[str] = field(default_factory=list)
    method: str = "none"
    boost_map: dict[str, float] = field(default_factory=dict)


# Curated synonym groups for common search domains
_SYNONYM_GROUPS: list[set[str]] = [
    {"search", "retrieval", "lookup", "find", "query"},
    {"document", "doc", "page", "article", "text"},
    {"score", "rank", "rating", "relevance"},
    {"fast", "quick", "rapid", "efficient", "speedy"},
    {"machine learning", "ml", "deep learning", "neural"},
    {"information", "data", "knowledge", "content"},
    {"index", "catalog", "database", "store"},
    {"algorithm", "method", "technique", "approach"},
    {"embedding", "vector", "representation", "encoding"},
    {"semantic", "meaning", "conceptual", "contextual"},
]

# Build a lookup from term → synonym set
_SYNONYM_MAP: dict[str, set[str]] = {}
for group in _SYNONYM_GROUPS:
    for term in group:
        _SYNONYM_MAP[term] = group


class QueryExpander:
    """Query expansion engine supporting multiple expansion strategies.

    Supports synonym expansion, pseudo-relevance feedback (PRF),
    and combined expansion with configurable weights.

    Args:
        max_expansion_terms: Maximum number of terms to add.
        synonym_boost: Boost weight for synonym-expanded terms.
        prf_boost: Boost weight for PRF-expanded terms.
    """

    def __init__(
        self,
        max_expansion_terms: int = 5,
        synonym_boost: float = 0.5,
        prf_boost: float = 0.3,
    ) -> None:
        self.max_expansion_terms = max_expansion_terms
        self.synonym_boost = synonym_boost
        self.prf_boost = prf_boost

    def expand_synonyms(self, query: str) -> ExpandedQuery:
        """Expand query with synonyms from curated synonym groups.

        Args:
            query: Original query string.

        Returns:
            ExpandedQuery with synonym-expanded terms.
        """
        tokens = query.lower().split()
        added: list[str] = []
        boost_map: dict[str, float] = {}

        for token in tokens:
            boost_map[token] = 1.0
            if token in _SYNONYM_MAP:
                synonyms = _SYNONYM_MAP[token] - {token}
                for syn in sorted(synonyms):
                    if len(added) >= self.max_expansion_terms:
                        break
                    if syn not in tokens and syn not in added:
                        added.append(syn)
                        boost_map[syn] = self.synonym_boost

        expanded_text = query
        if added:
            expanded_text = f"{query} {' '.join(added)}"

        return ExpandedQuery(
            original=query,
            expanded=expanded_text,
            added_terms=added,
            method="synonym",
            boost_map=boost_map,
        )

    def expand_prf(
        self,
        query: str,
        feedback_docs: list[str],
        top_n_terms: Optional[int] = None,
    ) -> ExpandedQuery:
        """Expand query using pseudo-relevance feedback (PRF).

        Extracts frequent terms from top-ranked documents and adds
        the most discriminative ones to the query.

        Args:
            query: Original query string.
            feedback_docs: Text of top-k retrieved documents.
            top_n_terms: Number of feedback terms to add.

        Returns:
            ExpandedQuery with PRF-expanded terms.
        """
        n_terms = top_n_terms or self.max_expansion_terms
        query_tokens = set(query.lower().split())

        term_counts: Counter[str] = Counter()
        for doc in feedback_docs:
            doc_tokens = re.findall(r"[a-z0-9]+", doc.lower())
            for token in doc_tokens:
                if token not in query_tokens and len(token) > 2:
                    term_counts[token] += 1

        doc_freq: Counter[str] = Counter()
        for doc in feedback_docs:
            doc_tokens = set(re.findall(r"[a-z0-9]+", doc.lower()))
            for token in doc_tokens:
                doc_freq[token] += 1

        scored_terms: list[tuple[str, float]] = []
        for term, count in term_counts.items():
            df = doc_freq.get(term, 0)
            if df < len(feedback_docs) * 0.8:
                score = count * (1.0 / (1.0 + df))
                scored_terms.append((term, score))

        scored_terms.sort(key=lambda x: x[1], reverse=True)
        added = [t for t, _ in scored_terms[:n_terms]]

        boost_map: dict[str, float] = {t: 1.0 for t in query_tokens}
        for term in added:
            boost_map[term] = self.prf_boost

        expanded_text = query
        if added:
            expanded_text = f"{query} {' '.join(added)}"

        return ExpandedQuery(
            original=query,
            expanded=expanded_text,
            added_terms=added,
            method="prf",
            boost_map=boost_map,
        )

    def expand(
        self,
        query: str,
        feedback_docs: Optional[list[str]] = None,
    ) -> ExpandedQuery:
        """Expand query using the best available strategy.

        Uses PRF if feedback documents are provided, otherwise falls
        back to synonym expansion.

        Args:
            query: Original query string.
            feedback_docs: Optional top-k document texts for PRF.

        Returns:
            ExpandedQuery with the best expansion applied.
        """
        if feedback_docs:
            return self.expand_prf(query, feedback_docs)
        return self.expand_synonyms(query)


if __name__ == "__main__":
    expander = QueryExpander(max_expansion_terms=3)

    print("=== Synonym Expansion ===")
    queries = [
        "fast document retrieval",
        "semantic search algorithm",
        "machine learning index",
    ]
    for q in queries:
        result = expander.expand_synonyms(q)
        print(f"Original:  '{result.original}'")
        print(f"Expanded:  '{result.expanded}'")
        print(f"Added:     {result.added_terms}")
        print(f"Boosts:    {result.boost_map}")
        print()

    print("=== Pseudo-Relevance Feedback ===")
    query = "information retrieval"
    feedback = [
        "Information retrieval systems use inverted indexes for efficient document search",
        "BM25 is a probabilistic retrieval model based on term frequency and document length",
        "Modern search engines combine sparse and dense retrieval for better results",
    ]
    result = expander.expand_prf(query, feedback, top_n_terms=4)
    print(f"Original:  '{result.original}'")
    print(f"Expanded:  '{result.expanded}'")
    print(f"Added:     {result.added_terms}")
    print()
