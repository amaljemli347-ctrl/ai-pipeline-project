# AI Pipeline Project (Week 1 & Week 2)

This repository contains the codebase for the AI pipeline project.
It covers Data Ingestion, Text Preprocessing, Chunking, Embedding, and Vector Database Population.

## Project Structure

```
├── .gitignore
├── requirements.txt
├── README.md
├── scripts/
│   ├── verify_env.py          # Verifies environment and dependencies
│   ├── ingest_data.py         # Downloads raw data from HuggingFace
│   ├── process_data.py        # Cleans and processes the raw data
│   ├── build_vectordb.py      # [Week 2] Chunks, embeds, and populates ChromaDB
│   └── benchmark_retrieval.py # [Week 2] Tests semantic search and logs latency
├── src/
│   ├── preprocessing.py       # Modular text cleaning functions
│   └── chunking.py            # [Week 2] Recursive character chunking strategy
└── tests/
    ├── test_preprocessing.py  # Unit tests for preprocessing
    └── test_chunking.py       # [Week 2] Unit tests for chunking logic
```

## Setup Instructions

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .\.venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

## Week 1: Data Ingestion & Preprocessing

- **Ingestion**: `scripts/ingest_data.py` downloads documents from Wikipedia.
- **Preprocessing**: `scripts/process_data.py` strips HTML, normalizes unicode, cleans whitespace, filters for English, and saves to `data/processed/clean_corpus.parquet`.

## Week 2: Chunking, Embeddings & Vector DB

### Chunking Strategy
The chunking strategy is implemented in `src/chunking.py`. It uses a **Recursive Character Splitter** with a target chunk size of 500 characters and an overlap of 50 characters. 
- It attempts to split on `\n\n` (paragraphs), then `\n`, then `. ` (sentences), and finally ` ` (words).
- This ensures that chunks are created at the most natural text boundaries, preserving context and meaning.
- Unit tests are located in `tests/test_chunking.py`.

### Generating Embeddings & ChromaDB Edge Cases
The vector database is built using `scripts/build_vectordb.py`.
- **Embeddings**: We use `all-MiniLM-L6-v2` via `sentence-transformers`. It's very fast for CPU encoding. The script logs the total encoding time and chunks per second.
- **ChromaDB limits**: ChromaDB has a maximum batch payload limit (around 41,666 for local sqlite constraints or 5461 elements max depending on SQL compilation). To handle this edge case safely, the script batches ingestions strictly into sizes of 5,000 chunks.

Run the build script:
```bash
python scripts/build_vectordb.py
```
This populates the `data/chroma_db/` persistent directory.

### Retrieval Benchmarking
To test semantic search performance, run:
```bash
python scripts/benchmark_retrieval.py
```
This script queries the vector DB with 5 benchmark queries and logs the latency (in ms) along with the top retrieved snippet.

## Running Tests
Run the entire test suite using `pytest`:
```bash
pytest
```
