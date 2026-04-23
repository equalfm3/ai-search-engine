# AI-Powered Search Engine

Full search engine with BM25 sparse retrieval, dense semantic search, hybrid fusion, learned ranking (LambdaMART + neural), query understanding, and a search UI. Indexes and searches document collections.

## What This Covers

- Inverted index with BM25 scoring from scratch
- Dense retrieval with bi-encoder embeddings
- Hybrid search: reciprocal rank fusion (RRF)
- Learned ranking: LambdaMART and neural cross-encoder reranker
- Query understanding: intent classification, query expansion
- Snippet generation and highlighting
- FastAPI search API + Gradio UI

## Structure

```
├── src/
│   ├── indexing/
│   │   ├── inverted_index.py  # BM25 inverted index
│   │   ├── dense_index.py     # FAISS dense index
│   │   └── indexer.py         # Document indexing pipeline
│   ├── retrieval/
│   │   ├── bm25.py            # BM25 retrieval
│   │   ├── dense.py           # Dense retrieval
│   │   └── hybrid.py          # RRF hybrid fusion
│   ├── ranking/
│   │   ├── lambdamart.py      # LambdaMART ranker
│   │   └── cross_encoder.py   # Neural reranker
│   ├── query/
│   │   ├── understanding.py   # Intent classification
│   │   └── expansion.py       # Query expansion
│   ├── serving/
│   │   ├── api.py             # FastAPI search endpoint
│   │   └── snippets.py        # Snippet generation
│   └── evaluation/
│       └── metrics.py         # NDCG, MAP, MRR
├── demo/
│   └── app.py                 # Gradio search UI
├── data/
│   └── sample_corpus/         # Sample documents
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
python -m src.indexing.indexer --input data/sample_corpus/
python -m src.serving.api  # starts search API on :8000
```
