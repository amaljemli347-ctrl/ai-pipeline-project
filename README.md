# Enterprise RAG System (Week 1, 2 & 3)

This repository contains an end-to-end, production-ready Retrieval-Augmented Generation (RAG) system built for domain-specific AI applications.

## Project Architecture & Directory Structure

```
├── .env.example               # Template for environment variables (OpenRouter/Groq API Key)
├── .gitignore                 # Ignores data, .venv, .env, and local database artifacts
├── requirements.txt           # Locked dependency definitions
├── README.md                  # Comprehensive setup & architecture documentation
├── scripts/
│   ├── verify_env.py          # [Week 1] Verifies PyTorch/CUDA availability and dependencies
│   ├── ingest_data.py         # [Week 1] Streams and ingests raw dataset from HuggingFace
│   ├── process_data.py        # [Week 1] Preprocessing pipeline (HTML, unicode, whitespace, lang filter)
│   ├── build_vectordb.py      # [Week 2] Chunks, generates embeddings, and populates ChromaDB
│   ├── benchmark_retrieval.py # [Week 2] Measures semantic retrieval latency
│   └── run_rag.py             # [Week 3] End-to-end RAG pipeline runner with latency & hallucination checks
├── src/
│   ├── preprocessing.py       # [Week 1] Text cleaning utility functions
│   ├── chunking.py            # [Week 2] Recursive Character Text Splitter implementation
│   ├── llm.py                 # [Week 3] OpenAI-compatible LLM client with retries & error handling
│   └── rag.py                 # [Week 3] RAG orchestrator, prompt engineering, hallucination verification
└── tests/
    ├── test_preprocessing.py  # [Week 1] Unit tests for preprocessing functions
    ├── test_chunking.py       # [Week 2] Unit tests for recursive chunking
    └── test_rag.py            # [Week 3] Unit tests for RAG prompt formatting & hallucination checks
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
   Set `OPENROUTER_API_KEY=your_key_here` in `.env`.

---

## Week 3: LLM Integration & Prompt Engineering

### 1. LLM Integration & API Error Handling (`src/llm.py`)
- Standardized `LLMClient` supporting OpenAI-compatible APIs (OpenRouter, Groq, OpenAI).
- **Error Handling & Retries**: Implements automatic exponential backoff retries for:
  - `RateLimitError` (HTTP 429)
  - `APITimeoutError`
  - Model 404 / Invalid Model ID detection
- **Fallback Mock Mode**: If no API key is supplied in `.env`, the client operates in an informative mock mode allowing offline evaluation and seamless CI/CD without crashing.

### 2. Prompt Engineering & Context Injection (`src/rag.py`)
- **System Prompt**: Enforces strict grounding rules. The LLM is instructed to answer *strictly* using the retrieved context.
- **Out-of-Domain & Hallucination Prevention**: If the context does not contain sufficient facts to answer the question, the system prompt instructs the LLM to explicitly return:
  > *"The provided context does not contain enough information to answer your query."*

### 3. Hallucination Detection & Verification
The `RAGPipeline` contains a `verify_groundedness()` layer that computes lexical and semantic overlap between the LLM output and source context chunks to flag ungrounded claims or confirm valid out-of-domain rejections.

### 4. End-to-End Latency Logging
Running `python scripts/run_rag.py` measures and outputs:
- **Retrieval Latency** (ChromaDB vector search time)
- **Generation Latency** (LLM completion response time)
- **Total End-to-End Latency**

---

## How to Run

1. **Run Full Test Suite**:
   ```bash
   pytest
   ```

2. **Run End-to-End RAG Benchmark**:
   ```bash
   python scripts/run_rag.py
   ```
