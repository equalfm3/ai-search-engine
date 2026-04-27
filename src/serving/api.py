"""FastAPI search endpoint.

Provides a REST API for searching the indexed document collection
with hybrid retrieval, reranking, and snippet generation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.indexing.dense_index import DenseIndex
from src.indexing.indexer import Document, Indexer
from src.indexing.inverted_index import InvertedIndex
from src.query.expansion import QueryExpander
from src.query.understanding import QueryUnderstanding
from src.ranking.cross_encoder import CrossEncoderReranker
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.serving.snippets import SnippetGenerator


@dataclass
class SearchResultItem:
    """A single search result for API response.

    Attributes:
        doc_id: Document identifier.
        title: Document title.
        snippet: Highlighted text snippet.
        score: Final relevance score.
        url: Document URL if available.
    """

    doc_id: int
    title: str
    snippet: str
    score: float
    url: str = ""


class SearchEngine:
    """Full search engine combining retrieval, ranking, and serving.

    Orchestrates the complete search pipeline: query understanding,
    expansion, hybrid retrieval, reranking, and snippet generation.
    """

    def __init__(self) -> None:
        self.indexer = Indexer()
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.dense_retriever: Optional[DenseRetriever] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.reranker = CrossEncoderReranker()
        self.query_understanding = QueryUnderstanding()
        self.query_expander = QueryExpander()
        self.snippet_generator = SnippetGenerator(
            highlight_pre="<b>", highlight_post="</b>"
        )

    def load_index(self, index_dir: str) -> None:
        """Load pre-built indexes from disk.

        Args:
            index_dir: Directory containing saved index files.
        """
        path = Path(index_dir)
        inv_index = InvertedIndex.load(str(path / "inverted_index.json"))
        dense_index = DenseIndex.load(str(path / "dense_index"))

        doc_path = path / "documents.json"
        if doc_path.exists():
            with open(doc_path) as f:
                doc_data = json.load(f)
            for doc_id_str, d in doc_data.items():
                self.indexer.documents[int(doc_id_str)] = Document(**d)

        self.bm25_retriever = BM25Retriever(index=inv_index)
        self.dense_retriever = DenseRetriever(index=dense_index)
        self.hybrid_retriever = HybridRetriever(
            self.bm25_retriever, self.dense_retriever
        )

    def index_corpus(self, corpus_path: str) -> None:
        """Index a document corpus from scratch.

        Args:
            corpus_path: Path to corpus directory or JSONL file.
        """
        docs = self.indexer.load_corpus(corpus_path)
        self.indexer.index_documents(docs)

        self.bm25_retriever = BM25Retriever(index=self.indexer.inverted_index)
        self.dense_retriever = DenseRetriever(index=self.indexer.dense_index)
        self.hybrid_retriever = HybridRetriever(
            self.bm25_retriever, self.dense_retriever
        )

    def index_documents(self, docs: list[Document]) -> None:
        """Index a list of Document objects directly.

        Args:
            docs: Documents to index.
        """
        self.indexer.index_documents(docs)
        self.bm25_retriever = BM25Retriever(index=self.indexer.inverted_index)
        self.dense_retriever = DenseRetriever(index=self.indexer.dense_index)
        self.hybrid_retriever = HybridRetriever(
            self.bm25_retriever, self.dense_retriever
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_reranking: bool = True,
        use_expansion: bool = True,
    ) -> list[SearchResultItem]:
        """Execute a full search pipeline.

        Args:
            query: Raw search query.
            top_k: Number of results to return.
            use_reranking: Whether to apply cross-encoder reranking.
            use_expansion: Whether to apply query expansion.

        Returns:
            List of SearchResultItem objects.
        """
        if self.hybrid_retriever is None:
            return []

        analysis = self.query_understanding.analyze(query)
        search_query = query
        if use_expansion:
            expanded = self.query_expander.expand_synonyms(query)
            search_query = expanded.expanded

        fetch_k = min(top_k * 5, 100)
        hybrid_results = self.hybrid_retriever.retrieve(
            search_query, top_k=fetch_k
        )

        if not hybrid_results:
            return []

        doc_ids = [r.doc_id for r in hybrid_results]
        doc_texts = []
        for doc_id in doc_ids:
            doc = self.indexer.get_document(doc_id)
            doc_texts.append(doc.full_text if doc else "")

        if use_reranking and len(doc_ids) > 1:
            reranked = self.reranker.rerank(query, doc_ids, doc_texts, top_k=top_k)
            final_ids = [r.doc_id for r in reranked]
            final_scores = [r.score for r in reranked]
        else:
            final_ids = doc_ids[:top_k]
            final_scores = [r.rrf_score for r in hybrid_results[:top_k]]

        results: list[SearchResultItem] = []
        for doc_id, score in zip(final_ids, final_scores):
            doc = self.indexer.get_document(doc_id)
            if doc is None:
                continue
            snippets = self.snippet_generator.generate(query, doc.body or doc.full_text)
            snippet_text = snippets[0].text if snippets else doc.title
            results.append(SearchResultItem(
                doc_id=doc_id,
                title=doc.title,
                snippet=snippet_text,
                score=score,
                url=doc.url,
            ))

        return results


def create_app() -> object:
    """Create the FastAPI application.

    Returns:
        FastAPI app instance.
    """
    from fastapi import FastAPI, Query
    from fastapi.responses import JSONResponse

    app = FastAPI(title="AI Search Engine", version="1.0.0")
    engine = SearchEngine()

    sample_docs = [
        Document(0, "Information Retrieval", "Information retrieval is the science of searching for information in documents and databases."),
        Document(1, "BM25 Scoring", "BM25 is a bag-of-words retrieval function that ranks documents based on query term frequency."),
        Document(2, "Dense Retrieval", "Dense retrieval uses neural network embeddings to capture semantic similarity between queries and documents."),
        Document(3, "Hybrid Search", "Hybrid search combines sparse BM25 and dense semantic signals using reciprocal rank fusion."),
        Document(4, "Learning to Rank", "LambdaMART and neural rerankers learn to optimize ranking metrics like NDCG directly."),
    ]
    engine.index_documents(sample_docs)

    @app.get("/search")
    async def search(
        q: str = Query(..., description="Search query"),
        top_k: int = Query(10, ge=1, le=100, description="Number of results"),
        rerank: bool = Query(True, description="Apply reranking"),
    ) -> JSONResponse:
        """Search endpoint."""
        results = engine.search(q, top_k=top_k, use_reranking=rerank)
        return JSONResponse(content={
            "query": q,
            "results": [
                {
                    "doc_id": r.doc_id,
                    "title": r.title,
                    "snippet": r.snippet,
                    "score": round(r.score, 4),
                    "url": r.url,
                }
                for r in results
            ],
            "total": len(results),
        })

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the search API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    app = create_app()
    print(f"Starting search API on {args.host}:{args.port}")
    print("Endpoints: GET /search?q=<query>, GET /health")

    try:
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)
    except ImportError:
        print("uvicorn not installed. Install with: pip install uvicorn")
        print("API app created successfully (run with uvicorn manually)")
