"""BM25 sparse retrieval module.

Provides a retriever interface over the inverted index for BM25-scored
document retrieval with optional query-time boosting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.indexing.inverted_index import InvertedIndex


@dataclass
class SearchResult:
    """A single search result with score and metadata.

    Attributes:
        doc_id: Document identifier.
        score: Relevance score.
        method: Retrieval method that produced this result.
    """

    doc_id: int
    score: float
    method: str = "bm25"


class BM25Retriever:
    """BM25 sparse retrieval over an inverted index.

    Wraps the InvertedIndex to provide a clean retriever interface
    with configurable parameters and query-time boosting.

    Args:
        index: Pre-built inverted index.
        default_top_k: Default number of results to return.
    """

    def __init__(
        self, index: Optional[InvertedIndex] = None, default_top_k: int = 100
    ) -> None:
        self.index = index or InvertedIndex()
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        boost_terms: Optional[dict[str, float]] = None,
    ) -> list[SearchResult]:
        """Retrieve documents matching a query using BM25 scoring.

        Args:
            query: Raw query string.
            top_k: Number of results to return (overrides default).
            boost_terms: Optional term → boost_factor mapping for query-time
                boosting. Terms in the query matching a key get their BM25
                contribution multiplied by the boost factor.

        Returns:
            List of SearchResult objects sorted by descending score.
        """
        k = top_k or self.default_top_k
        query_tokens = self.index.tokenize(query)

        if not query_tokens:
            return []

        if boost_terms is None:
            raw_results = self.index.search(query, k)
            return [
                SearchResult(doc_id=doc_id, score=score, method="bm25")
                for doc_id, score in raw_results
            ]

        return self._boosted_search(query_tokens, boost_terms, k)

    def _boosted_search(
        self,
        query_tokens: list[str],
        boost_terms: dict[str, float],
        top_k: int,
    ) -> list[SearchResult]:
        """Perform BM25 search with per-term boost factors.

        Args:
            query_tokens: Tokenized query terms.
            boost_terms: Term → boost factor mapping.
            top_k: Number of results.

        Returns:
            Boosted search results.
        """
        scores: dict[int, float] = {}

        for term in query_tokens:
            boost = boost_terms.get(term, 1.0)
            if term not in self.index.postings:
                continue
            for entry in self.index.postings[term]:
                base_score = self.index.bm25_score(
                    entry.doc_id, term, entry.term_freq
                )
                scores[entry.doc_id] = scores.get(entry.doc_id, 0.0) + base_score * boost

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            SearchResult(doc_id=doc_id, score=score, method="bm25")
            for doc_id, score in ranked
        ]

    def add_document(self, doc_id: int, text: str) -> None:
        """Add a document to the underlying index.

        Args:
            doc_id: Document identifier.
            text: Raw document text.
        """
        self.index.add_document(doc_id, text)

    @property
    def doc_count(self) -> int:
        """Return the number of indexed documents."""
        return self.index.doc_count


if __name__ == "__main__":
    retriever = BM25Retriever()

    corpus = [
        (0, "Introduction to information retrieval and search engines"),
        (1, "BM25 is a bag-of-words retrieval function for ranking documents"),
        (2, "Neural networks for natural language processing tasks"),
        (3, "Inverted indexes are the backbone of modern search systems"),
        (4, "TF-IDF and BM25 are classic sparse retrieval methods"),
    ]
    for doc_id, text in corpus:
        retriever.add_document(doc_id, text)

    print(f"Indexed {retriever.doc_count} documents\n")

    query = "BM25 retrieval ranking"
    results = retriever.retrieve(query, top_k=3)
    print(f"Query: '{query}'")
    for r in results:
        print(f"  Doc {r.doc_id}: {r.score:.4f} [{r.method}]")

    print(f"\nBoosted query (boost 'bm25' by 2x):")
    boosted = retriever.retrieve(query, top_k=3, boost_terms={"bm25": 2.0})
    for r in boosted:
        print(f"  Doc {r.doc_id}: {r.score:.4f} [{r.method}]")
