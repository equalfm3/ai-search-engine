"""Gradio search UI for the AI search engine.

Provides an interactive web interface for searching the indexed
document collection with real-time results and highlighting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indexing.indexer import Document
from src.serving.api import SearchEngine


def build_sample_engine() -> SearchEngine:
    """Build a search engine with sample documents for demo purposes.

    Returns:
        SearchEngine loaded with sample documents.
    """
    engine = SearchEngine()

    sample_docs = [
        Document(0, "BM25 Scoring Algorithm",
                 "BM25 is a bag-of-words retrieval function that ranks documents based on "
                 "query terms appearing in each document. It considers term frequency, "
                 "document length, and inverse document frequency to compute relevance scores."),
        Document(1, "Dense Retrieval with Neural Embeddings",
                 "Dense retrieval encodes queries and documents into dense vector representations "
                 "using neural networks. Similarity is computed via cosine distance in the "
                 "embedding space, capturing semantic meaning beyond exact term matching."),
        Document(2, "Reciprocal Rank Fusion",
                 "Reciprocal rank fusion combines multiple ranked lists by summing reciprocal "
                 "ranks. It is a simple yet effective method for hybrid search that merges "
                 "sparse BM25 and dense semantic retrieval results."),
        Document(3, "LambdaMART Learning to Rank",
                 "LambdaMART is a gradient boosted decision tree model that directly optimizes "
                 "NDCG through lambda gradients. It uses pairwise document comparisons weighted "
                 "by the change in NDCG to learn an effective ranking function."),
        Document(4, "Cross-Encoder Reranking",
                 "Cross-encoder models jointly encode query-document pairs through a transformer "
                 "to produce fine-grained relevance scores. They are more accurate than "
                 "bi-encoders but slower, making them ideal for reranking a small candidate set."),
        Document(5, "Inverted Index Data Structure",
                 "An inverted index maps terms to the documents containing them, along with "
                 "term frequencies and positions. It is the fundamental data structure behind "
                 "all modern search engines, enabling sub-linear time lookups."),
        Document(6, "Query Understanding and Intent Classification",
                 "Query understanding classifies search queries into intent categories such as "
                 "navigational, informational, and transactional. This helps the search engine "
                 "adapt its retrieval strategy to match user expectations."),
        Document(7, "NDCG Evaluation Metric",
                 "Normalized Discounted Cumulative Gain measures ranking quality by comparing "
                 "the actual ranking to the ideal ranking. It accounts for graded relevance "
                 "and position bias, making it the standard metric for search evaluation."),
        Document(8, "Semantic Search Architecture",
                 "Modern semantic search systems combine sparse keyword matching with dense "
                 "neural retrieval. The hybrid approach captures both exact term matches and "
                 "conceptual similarity for comprehensive document retrieval."),
        Document(9, "Query Expansion Techniques",
                 "Query expansion adds related terms to the original query to improve recall. "
                 "Techniques include synonym expansion, pseudo-relevance feedback from top "
                 "documents, and neural query reformulation."),
    ]

    engine.index_documents(sample_docs)
    return engine


def create_gradio_app(engine: SearchEngine) -> object:
    """Create the Gradio search interface.

    Args:
        engine: Initialized SearchEngine instance.

    Returns:
        Gradio Blocks app.
    """
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Install with: pip install gradio")
        print("Falling back to CLI search mode.")
        return None

    def search_fn(query: str, top_k: int, use_reranking: bool) -> str:
        """Execute search and format results as HTML."""
        if not query.strip():
            return "<p>Enter a search query above.</p>"

        results = engine.search(
            query, top_k=int(top_k), use_reranking=use_reranking
        )

        if not results:
            return "<p>No results found.</p>"

        html_parts = [f"<p><b>{len(results)} results for:</b> {query}</p><hr>"]
        for i, r in enumerate(results, 1):
            html_parts.append(
                f"<div style='margin-bottom:12px;'>"
                f"<b>{i}. {r.title}</b> "
                f"<span style='color:#666;'>(score: {r.score:.4f})</span><br>"
                f"<span style='font-size:0.9em;'>{r.snippet}</span>"
                f"</div>"
            )
        return "\n".join(html_parts)

    with gr.Blocks(title="AI Search Engine") as app:
        gr.Markdown("# AI Search Engine Demo")
        gr.Markdown("Search with BM25 + dense retrieval + hybrid fusion + reranking")

        with gr.Row():
            query_input = gr.Textbox(
                label="Search Query",
                placeholder="Enter your search query...",
                scale=4,
            )
            search_btn = gr.Button("Search", variant="primary", scale=1)

        with gr.Row():
            top_k_slider = gr.Slider(1, 20, value=5, step=1, label="Results")
            rerank_toggle = gr.Checkbox(value=True, label="Reranking")

        results_html = gr.HTML(label="Results")

        search_btn.click(
            fn=search_fn,
            inputs=[query_input, top_k_slider, rerank_toggle],
            outputs=results_html,
        )
        query_input.submit(
            fn=search_fn,
            inputs=[query_input, top_k_slider, rerank_toggle],
            outputs=results_html,
        )

    return app


def cli_search(engine: SearchEngine) -> None:
    """Run an interactive CLI search loop.

    Args:
        engine: Initialized SearchEngine instance.
    """
    print("AI Search Engine — CLI Mode")
    print("Type a query and press Enter. Type 'quit' to exit.\n")

    while True:
        query = input("Query> ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        results = engine.search(query, top_k=5)
        if not results:
            print("  No results found.\n")
            continue

        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r.score:.4f}] {r.title}")
            snippet = r.snippet.replace("<b>", "").replace("</b>", "")
            print(f"     {snippet[:100]}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Search Engine Demo")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--cli", action="store_true", help="Use CLI mode instead of Gradio")
    args = parser.parse_args()

    engine = build_sample_engine()

    if args.cli:
        cli_search(engine)
    else:
        app = create_gradio_app(engine)
        if app is not None:
            app.launch(server_port=args.port)
        else:
            cli_search(engine)
