"""BM25 inverted index built from scratch.

Implements a term-level inverted index with BM25 scoring for sparse retrieval.
Supports incremental document addition, serialization, and configurable parameters.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PostingEntry:
    """Single entry in a posting list: document ID and term frequency."""

    doc_id: int
    term_freq: int


@dataclass
class InvertedIndex:
    """BM25-scored inverted index for sparse document retrieval.

    Builds a term → posting list mapping with precomputed IDF values.
    Supports BM25 scoring with configurable k1 and b parameters.

    Attributes:
        k1: Term frequency saturation parameter (default 1.5).
        b: Length normalization parameter (default 0.75).
    """

    k1: float = 1.5
    b: float = 0.75
    postings: dict[str, list[PostingEntry]] = field(default_factory=dict)
    doc_lengths: dict[int, int] = field(default_factory=dict)
    doc_count: int = 0
    avg_doc_length: float = 0.0
    _idf_cache: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Tokenize text into lowercase alphanumeric tokens.

        Args:
            text: Raw text string.

        Returns:
            List of lowercase tokens.
        """
        return re.findall(r"[a-z0-9]+", text.lower())

    def add_document(self, doc_id: int, text: str) -> None:
        """Add a document to the index.

        Args:
            doc_id: Unique document identifier.
            text: Raw document text.
        """
        tokens = self.tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        self.doc_count += 1
        total_length = sum(self.doc_lengths.values())
        self.avg_doc_length = total_length / self.doc_count if self.doc_count else 0.0

        term_freqs: dict[str, int] = defaultdict(int)
        for token in tokens:
            term_freqs[token] += 1

        for term, freq in term_freqs.items():
            if term not in self.postings:
                self.postings[term] = []
            self.postings[term].append(PostingEntry(doc_id=doc_id, term_freq=freq))

        self._idf_cache.clear()

    def idf(self, term: str) -> float:
        """Compute inverse document frequency for a term.

        Uses the standard BM25 IDF formula:
            IDF(t) = ln((N - df(t) + 0.5) / (df(t) + 0.5) + 1)

        Args:
            term: The query term.

        Returns:
            IDF score for the term.
        """
        if term in self._idf_cache:
            return self._idf_cache[term]
        df = len(self.postings.get(term, []))
        score = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)
        self._idf_cache[term] = score
        return score

    def bm25_score(self, doc_id: int, term: str, term_freq: int) -> float:
        """Compute BM25 score for a single term-document pair.

        BM25(d, t) = IDF(t) * (f(t,d) * (k1 + 1)) / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))

        Args:
            doc_id: Document identifier.
            term: Query term.
            term_freq: Frequency of term in document.

        Returns:
            BM25 score contribution for this term-document pair.
        """
        dl = self.doc_lengths.get(doc_id, 0)
        idf_val = self.idf(term)
        numerator = term_freq * (self.k1 + 1)
        denominator = term_freq + self.k1 * (1 - self.b + self.b * dl / self.avg_doc_length)
        return idf_val * numerator / denominator

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Search the index with BM25 scoring.

        Args:
            query: Raw query string.
            top_k: Number of top results to return.

        Returns:
            List of (doc_id, score) tuples sorted by descending score.
        """
        query_terms = self.tokenize(query)
        scores: dict[int, float] = defaultdict(float)

        for term in query_terms:
            if term not in self.postings:
                continue
            for entry in self.postings[term]:
                scores[entry.doc_id] += self.bm25_score(
                    entry.doc_id, term, entry.term_freq
                )

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def save(self, path: str) -> None:
        """Serialize the index to a JSON file.

        Args:
            path: Output file path.
        """
        data = {
            "k1": self.k1,
            "b": self.b,
            "doc_count": self.doc_count,
            "avg_doc_length": self.avg_doc_length,
            "doc_lengths": {str(k): v for k, v in self.doc_lengths.items()},
            "postings": {
                term: [{"doc_id": e.doc_id, "term_freq": e.term_freq} for e in entries]
                for term, entries in self.postings.items()
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> InvertedIndex:
        """Load an index from a JSON file.

        Args:
            path: Input file path.

        Returns:
            Reconstructed InvertedIndex instance.
        """
        with open(path) as f:
            data = json.load(f)
        idx = cls(k1=data["k1"], b=data["b"])
        idx.doc_count = data["doc_count"]
        idx.avg_doc_length = data["avg_doc_length"]
        idx.doc_lengths = {int(k): v for k, v in data["doc_lengths"].items()}
        idx.postings = {
            term: [PostingEntry(**e) for e in entries]
            for term, entries in data["postings"].items()
        }
        return idx

    def vocabulary_size(self) -> int:
        """Return the number of unique terms in the index."""
        return len(self.postings)


if __name__ == "__main__":
    docs = [
        (0, "The quick brown fox jumps over the lazy dog"),
        (1, "A fast brown fox leaps over a sleepy hound"),
        (2, "Machine learning models for natural language processing"),
        (3, "Deep learning and neural networks for text classification"),
        (4, "Information retrieval with inverted indexes and BM25 scoring"),
    ]
    index = InvertedIndex()
    for doc_id, text in docs:
        index.add_document(doc_id, text)

    print(f"Indexed {index.doc_count} documents, {index.vocabulary_size()} terms")
    print(f"Average document length: {index.avg_doc_length:.1f}")

    queries = ["brown fox", "machine learning", "information retrieval BM25"]
    for q in queries:
        results = index.search(q, top_k=3)
        print(f"\nQuery: '{q}'")
        for doc_id, score in results:
            print(f"  Doc {doc_id}: {score:.4f} — {docs[doc_id][1][:60]}")
