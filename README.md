# Enterprise RAG System (Week 1, 2, 3 & 4)

This repository contains an end-to-end, production-ready Retrieval-Augmented Generation (RAG) system with advanced NLP analysis and metadata-filtered retrieval.

## Project Architecture & Directory Structure

```
├── .env.example                     # Template for environment variables (OpenRouter/Groq API Key)
├── .gitignore                       # Ignores data, .venv, .env, and local database artifacts
├── requirements.txt                 # Locked dependency definitions
├── README.md                        # Comprehensive setup & architecture documentation
├── scripts/
│   ├── verify_env.py                # [Week 1] Verifies PyTorch/CUDA availability and dependencies
│   ├── ingest_data.py               # [Week 1] Streams and ingests raw dataset from HuggingFace
│   ├── process_data.py              # [Week 1] Preprocessing pipeline (HTML, unicode, whitespace, lang filter)
│   ├── build_vectordb.py            # [Week 2] Chunks, generates embeddings, and populates ChromaDB
│   ├── benchmark_retrieval.py       # [Week 2] Measures semantic retrieval latency
│   ├── run_rag.py                   # [Week 3] End-to-end RAG pipeline runner with latency & hallucination checks
│   ├── enrich_vectordb.py           # [Week 4] Enriches vector DB with LDA Topics & Sentiment/NER metadata
│   └── benchmark_nlp_retrieval.py   # [Week 4] Evaluates NLP accuracy and benchmarks metadata-filtered retrieval
├── src/
│   ├── preprocessing.py             # [Week 1] Text cleaning utility functions
│   ├── chunking.py                  # [Week 2] Recursive Character Text Splitter implementation
│   ├── llm.py                       # [Week 3] OpenAI-compatible LLM client with retries & error handling
│   ├── rag.py                       # [Week 3] RAG orchestrator, prompt engineering, hallucination verification
│   └── nlp.py                       # [Week 4] LDA Topic Modeling, spaCy NER & Sentiment Analysis engine
└── tests/
    ├── test_preprocessing.py        # [Week 1] Unit tests for preprocessing functions
    ├── test_chunking.py             # [Week 2] Unit tests for recursive chunking
    ├── test_rag.py                  # [Week 3] Unit tests for RAG prompt formatting & hallucination checks
    └── test_nlp.py                  # [Week 4] Unit tests for Topic Modeling, Sentiment, NER & Accuracy
```

---

## Setup Instructions

1. **Activate Virtual Environment**:
   ```bash
   .\.venv\Scripts\activate   # Windows
   # or
   source .venv/bin/activate  # macOS/Linux
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and fill in your API Key (e.g. OpenRouter or Groq):
   ```bash
   cp .env.example .env
   ```

---

## Week 4: NLP Analysis (Topic Modeling, Sentiment & NER)

### 1. Topic Modeling (`src/nlp.py`)
- Implements **Latent Dirichlet Allocation (LDA)** using `scikit-learn` to extract latent themes across the corpus (e.g., *Science & Technology*, *History & Culture*, *Business & Economy*).
- **Edge Case Handling**: Short documents (`< 20 chars`) are flagged as `"General/Short"` to prevent noise in topic distribution.

### 2. Sentiment Analysis & Named Entity Recognition (NER)
- **NER**: Uses `spaCy` (`en_core_web_sm`) to extract named entities (`PERSON`, `ORG`, `GPE`, `DATE`).
- **Sentiment**: Rule-based lexicon scorer classifying chunk polarity as `positive`, `negative`, or `neutral`.

### 3. Evaluation & Accuracy Report
Evaluated against a reference manually-labeled test dataset (`evaluate_nlp_accuracy()`):
- **Accuracy**: `90.0%`
- **Macro F1 Score**: `0.889`
- Evaluated on 10 gold-standard reference samples across positive, negative, and neutral categories.

### 4. Metadata-Filtered Vector Retrieval (`src/rag.py` & `scripts/benchmark_nlp_retrieval.py`)
ChromaDB payload enriched with metadata (`topic`, `sentiment`, `entities`).
Supports metadata filtering queries:
```python
# Retrieve chunks strictly belonging to History & Culture topic
rag.query("What are key historical events?", filter_metadata={"topic": "History & Culture"})

# Retrieve chunks with positive sentiment
rag.query("Business growth news", filter_metadata={"sentiment": "positive"})
```

---

## How to Run

1. **Run Full Test Suite (21 Unit Tests)**:
   ```bash
   pytest
   ```

2. **Run Vector DB Metadata Enrichment**:
   ```bash
   python scripts/enrich_vectordb.py
   ```

3. **Run NLP Accuracy Evaluation & Filtered Retrieval Benchmark**:
   ```bash
   python scripts/benchmark_nlp_retrieval.py
   ```

4. **Run RAG End-to-End Benchmark**:
   ```bash
   python scripts/run_rag.py
   ```
