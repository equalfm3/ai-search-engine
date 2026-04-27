"""Snippet generation and query term highlighting.

Extracts the most relevant text passages from documents and highlights
matching query terms for display in search results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Snippet:
    """A highlighted text snippet from a document.

    Attributes:
        text: The snippet text with highlight markers.
        score: Relevance score of this snippet window.
        start_char: Starting character offset in the original document.
        end_char: Ending character offset in the original document.
    """

    text: str
    score: float
    start_char: int
    end_char: int


class SnippetGenerator:
    """Generate and highlight search result snippets.

    Extracts the most relevant passage from a document by sliding a
    window over sentences and scoring by query term density. Highlights
    matching terms with configurable markers.

    Args:
        max_snippet_length: Maximum snippet length in characters.
        highlight_pre: Opening highlight marker (e.g., '<b>' or '**').
        highlight_post: Closing highlight marker (e.g., '</b>' or '**').
        context_sentences: Number of sentences to include in snippet.
    """

    def __init__(
        self,
        max_snippet_length: int = 200,
        highlight_pre: str = "<b>",
        highlight_post: str = "</b>",
        context_sentences: int = 2,
    ) -> None:
        self.max_snippet_length = max_snippet_length
        self.highlight_pre = highlight_pre
        self.highlight_post = highlight_post
        self.context_sentences = context_sentences

    @staticmethod
    def _split_sentences(text: str) -> list[tuple[str, int]]:
        """Split text into sentences with their character offsets.

        Args:
            text: Document text.

        Returns:
            List of (sentence, start_offset) tuples.
        """
        sentences: list[tuple[str, int]] = []
        for match in re.finditer(r"[^.!?]+[.!?]*", text):
            sent = match.group().strip()
            if sent:
                sentences.append((sent, match.start()))
        if not sentences and text.strip():
            sentences.append((text.strip(), 0))
        return sentences

    def _score_window(
        self, sentences: list[str], query_terms: set[str]
    ) -> float:
        """Score a window of sentences by query term density.

        Args:
            sentences: List of sentence strings in the window.
            query_terms: Set of lowercase query terms.

        Returns:
            Density score (matches / total words).
        """
        window_text = " ".join(sentences).lower()
        words = re.findall(r"[a-z0-9]+", window_text)
        if not words:
            return 0.0
        matches = sum(1 for w in words if w in query_terms)
        return matches / len(words)

    def generate(
        self,
        query: str,
        document: str,
        max_snippets: int = 1,
    ) -> list[Snippet]:
        """Generate highlighted snippets from a document for a query.

        Args:
            query: Search query string.
            document: Full document text.
            max_snippets: Maximum number of snippets to return.

        Returns:
            List of Snippet objects with highlighted text.
        """
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        sentences = self._split_sentences(document)

        if not sentences or not query_terms:
            truncated = document[: self.max_snippet_length]
            return [Snippet(text=truncated, score=0.0, start_char=0, end_char=len(truncated))]

        window_size = self.context_sentences
        best_windows: list[tuple[float, int]] = []

        for i in range(len(sentences)):
            end = min(i + window_size, len(sentences))
            window_sents = [s for s, _ in sentences[i:end]]
            score = self._score_window(window_sents, query_terms)
            best_windows.append((score, i))

        best_windows.sort(key=lambda x: x[0], reverse=True)

        snippets: list[Snippet] = []
        used_indices: set[int] = set()

        for score, start_idx in best_windows:
            if len(snippets) >= max_snippets:
                break
            if start_idx in used_indices:
                continue

            end_idx = min(start_idx + window_size, len(sentences))
            window_sents = [s for s, _ in sentences[start_idx:end_idx]]
            snippet_text = " ".join(window_sents)

            if len(snippet_text) > self.max_snippet_length:
                snippet_text = snippet_text[: self.max_snippet_length] + "..."

            highlighted = self._highlight(snippet_text, query_terms)
            start_char = sentences[start_idx][1]
            end_char = sentences[end_idx - 1][1] + len(sentences[end_idx - 1][0])

            snippets.append(Snippet(
                text=highlighted,
                score=score,
                start_char=start_char,
                end_char=end_char,
            ))

            for j in range(start_idx, end_idx):
                used_indices.add(j)

        return snippets

    def _highlight(self, text: str, query_terms: set[str]) -> str:
        """Highlight query terms in text.

        Args:
            text: Text to highlight.
            query_terms: Set of lowercase query terms.

        Returns:
            Text with highlight markers around matching terms.
        """
        def replacer(match: re.Match[str]) -> str:
            word = match.group()
            if word.lower() in query_terms:
                return f"{self.highlight_pre}{word}{self.highlight_post}"
            return word

        return re.sub(r"\b[a-zA-Z0-9]+\b", replacer, text)

    def generate_plain(self, query: str, document: str) -> str:
        """Generate a single plain-text snippet without highlighting.

        Args:
            query: Search query.
            document: Document text.

        Returns:
            Best matching text passage.
        """
        snippets = self.generate(query, document, max_snippets=1)
        if snippets:
            text = snippets[0].text
            text = text.replace(self.highlight_pre, "").replace(self.highlight_post, "")
            return text
        return document[: self.max_snippet_length]


if __name__ == "__main__":
    generator = SnippetGenerator(max_snippet_length=200, highlight_pre="**", highlight_post="**")

    document = (
        "Information retrieval is the activity of obtaining information system resources "
        "that are relevant to an information need. Searches can be based on full-text or "
        "other content-based indexing. BM25 is a bag-of-words retrieval function that "
        "ranks documents based on query terms appearing in each document. It is one of "
        "the most effective and widely used ranking functions in information retrieval."
    )

    queries = ["BM25 ranking function", "information retrieval", "content indexing"]
    for q in queries:
        snippets = generator.generate(q, document, max_snippets=1)
        print(f"Query: '{q}'")
        for s in snippets:
            print(f"  Score: {s.score:.3f}")
            print(f"  Snippet: {s.text}")
        print()
