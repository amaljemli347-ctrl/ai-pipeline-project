# Week 1: Environment, Data Ingestion & Preprocessing

This repository contains the codebase for the Week 1 deliverable. 
It focuses on initializing the project, setting up the virtual environment, downloading a real-world text dataset, and implementing a text preprocessing pipeline.

## Project Structure

```
├── .gitignore
├── requirements.txt
├── README.md
├── scripts/
│   ├── verify_env.py      # Script to verify environment and dependencies
│   ├── ingest_data.py     # Script to download raw data
│   └── process_data.py    # Script to clean and process the data
├── src/
│   └── preprocessing.py   # Modular text cleaning functions
└── tests/
    └── test_preprocessing.py # Unit tests for the preprocessing pipeline
```

## Setup Instructions

1. **Clone the repository and set it as active workspace**.
2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .\.venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

## Verifying the Environment

Run the verification script to test basic imports and check if CUDA/GPU is available on your machine.
```bash
python scripts/verify_env.py
```

## Data Ingestion

The dataset chosen is a subset of the Wikipedia dataset (approx 5,000 documents). 
We use the Hugging Face `datasets` library in streaming mode to efficiently fetch the records.

Run the ingestion script:
```bash
python scripts/ingest_data.py
```
This will save the dataset as a Parquet file at `data/raw/raw_wikipedia.parquet`.

## Text Preprocessing

The preprocessing pipeline includes:
- **HTML Stripping**: Removing any HTML tags using `BeautifulSoup`.
- **Unicode Normalization**: Converting characters to NFKC form.
- **Whitespace Cleanup**: Removing extra spaces, tabs, and newlines.
- **Language Filtering**: Ensuring the text is written in English (using `langdetect`).

To process the raw data and generate the cleaned corpus, run:
```bash
python scripts/process_data.py
```
This script reads the raw parquet file, applies the preprocessing steps, filters out non-English documents, and saves the cleaned dataset to `data/processed/clean_corpus.parquet`.

## Running Tests

Unit tests are written using `pytest`. To run the tests, execute:
```bash
pytest tests/test_preprocessing.py
```
This will test each modular function in `src/preprocessing.py` to ensure it correctly handles edge cases and standard input.
