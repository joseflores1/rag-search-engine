# RAG Search Engine

A Retrieval-Augmented Generation (RAG) search engine built in Python. It
currently implements the retrieval stage: keyword search over a corpus of
documents (each with a title and description).

## Features
- Inverted-index keyword retrieval over document titles + descriptions
- Text preprocessing pipeline: lowercase, punctuation removal, stopword
  removal, and Porter stemming (via NLTK)
- `search` and `build` CLI commands

## Project structure
- `cli/keyword_search_cli.py` — command-line interface
- `cli/lib/keyword_search.py` — inverted index + tokenization
- `cli/lib/search_utils.py` — shared constants and helpers
- `data/` — corpus dataset + stopwords (gitignored)
- `cache/` — generated index files (gitignored)

## Requirements
- Python >= 3.14
- [uv](https://docs.astral.sh/uv/) (package manager)

## Setup
    uv sync

## Usage
Run the CLI from the `cli/` directory:

    cd cli

Build the inverted index:
    uv run python -m keyword_search_cli build

Search documents (keyword matching):
    uv run python -m keyword_search_cli search "<query>"

## Roadmap
- BM25 scoring (params defined in `search_utils.py`)
- Reciprocal rank fusion (RRF) for result merging
- Semantic / vector search with embeddings
- Document chunking and chunk-level retrieval