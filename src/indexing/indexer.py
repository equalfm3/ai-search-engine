"""Document indexing pipeline.

Reads documents from a corpus directory, builds both inverted (BM25) and
dense (FAISS) indexes, and persists them to disk for serving.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.indexing.dense_index import DenseIndex
from src.indexing.inverted_index import InvertedIndex


@dataclass
class Document:
    """A document in the corpus.

    Attributes:
        doc_id: Unique integer identifier.
        title: Document title.
        body: Document body text.
        url: Optional source URL.
        metadata: Optional key-value metadata.
    """

    doc_id: int
    title: str
    body: str
    url: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Concatenate title and body for indexing."""
        return f"{self.title} {self.body}"


class Indexer:
    """Document indexing pipeline that builds sparse and dense indexes.

    Reads documents from JSON files, tokenizes them for BM25, generates
    embeddings for dense retrieval, and saves both indexes.

    Args:
        output_dir: Directory to save index files.
        dense_dim: Embedding dimension for the dense index.
    """

    def __init__(self, output_dir: str = "index_output", dense_dim: int = 384) -> None:
        self.output_dir = Path(output_dir)
        self.inverted_index = InvertedIndex()
        self.dense_index = DenseIndex(dimension=dense_dim)
        self.documents: dict[int, Document] = {}
        self._encoder: Optional[object] = None
        self.dense_dim = dense_dim

    def _get_encoder(self) -> object:
        """Lazy-load the sentence transformer encoder.

        Returns:
            SentenceTransformer model instance, or None if unavailable.
        """
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            except (ImportError, Exception):
                self._encoder = None
        return self._encoder

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode texts into dense embeddings.

        Falls back to random embeddings if sentence-transformers is unavailable.

        Args:
            texts: List of text strings to encode.

        Returns:
            Array of shape (len(texts), dense_dim).
        """
        encoder = self._get_encoder()
        if encoder is not None:
            return encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        rng = np.random.default_rng(hash(texts[0]) % (2**31))
        return rng.standard_normal((len(texts), self.dense_dim)).astype(np.float32)

    def load_corpus(self, corpus_path: str) -> list[Document]:
        """Load documents from a corpus directory or JSON file.

        Supports two formats:
        - Directory of .json files, each containing a single document
        - Single .jsonl file with one document per line

        Args:
            corpus_path: Path to corpus directory or JSONL file.

        Returns:
            List of loaded Document objects.
        """
        path = Path(corpus_path)
        docs: list[Document] = []

        if path.is_file() and path.suffix in (".json", ".jsonl"):
            docs = self._load_jsonl(path)
        elif path.is_dir():
            for json_file in sorted(path.glob("*.json")):
                docs.extend(self._load_json_file(json_file))
            for jsonl_file in sorted(path.glob("*.jsonl")):
                docs.extend(self._load_jsonl(jsonl_file))
        else:
            raise FileNotFoundError(f"Corpus path not found: {corpus_path}")

        return docs

    def _load_json_file(self, path: Path) -> list[Document]:
        """Load documents from a single JSON file."""
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return [self._parse_doc(d) for d in data]
        return [self._parse_doc(data)]

    def _load_jsonl(self, path: Path) -> list[Document]:
        """Load documents from a JSONL file (one JSON object per line)."""
        docs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(self._parse_doc(json.loads(line)))
        return docs

    def _parse_doc(self, data: dict) -> Document:
        """Parse a dictionary into a Document object."""
        doc_id = data.get("doc_id", len(self.documents))
        return Document(
            doc_id=doc_id,
            title=data.get("title", ""),
            body=data.get("body", data.get("text", "")),
            url=data.get("url", ""),
            metadata=data.get("metadata", {}),
        )

    def index_documents(self, docs: list[Document], batch_size: int = 32) -> None:
        """Index a list of documents into both sparse and dense indexes.

        Args:
            docs: Documents to index.
            batch_size: Batch size for dense encoding.
        """
        print(f"Indexing {len(docs)} documents...")
        start = time.time()

        for doc in docs:
            self.documents[doc.doc_id] = doc
            self.inverted_index.add_document(doc.doc_id, doc.full_text)

        texts = [doc.full_text for doc in docs]
        doc_ids = [doc.doc_id for doc in docs]

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_ids = doc_ids[i : i + batch_size]
            embeddings = self._encode_texts(batch_texts)
            self.dense_index.add(batch_ids, embeddings)

        elapsed = time.time() - start
        print(f"Indexed {len(docs)} docs in {elapsed:.2f}s")
        print(f"  Vocabulary: {self.inverted_index.vocabulary_size()} terms")
        print(f"  Dense index: {self.dense_index.size} vectors")

    def save(self) -> None:
        """Save all indexes and document metadata to disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.inverted_index.save(str(self.output_dir / "inverted_index.json"))
        self.dense_index.save(str(self.output_dir / "dense_index"))

        doc_data = {
            str(doc_id): {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "body": doc.body,
                "url": doc.url,
                "metadata": doc.metadata,
            }
            for doc_id, doc in self.documents.items()
        }
        with open(self.output_dir / "documents.json", "w") as f:
            json.dump(doc_data, f, indent=2)

        print(f"Index saved to {self.output_dir}/")

    def get_document(self, doc_id: int) -> Optional[Document]:
        """Retrieve a document by ID.

        Args:
            doc_id: Document identifier.

        Returns:
            Document if found, None otherwise.
        """
        return self.documents.get(doc_id)


def main() -> None:
    """CLI entrypoint for the indexing pipeline."""
    parser = argparse.ArgumentParser(description="Index a document corpus")
    parser.add_argument("--input", required=True, help="Path to corpus directory or JSONL file")
    parser.add_argument("--output", default="index_output", help="Output directory for indexes")
    parser.add_argument("--dim", type=int, default=384, help="Dense embedding dimension")
    args = parser.parse_args()

    indexer = Indexer(output_dir=args.output, dense_dim=args.dim)
    docs = indexer.load_corpus(args.input)
    indexer.index_documents(docs)
    indexer.save()


if __name__ == "__main__":
    main()
