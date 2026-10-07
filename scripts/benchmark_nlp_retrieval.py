import os
import sys
import time
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag import RAGPipeline
from src.nlp import evaluate_nlp_accuracy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_nlp_benchmark():
    print("=" * 70)
    print("      WEEK 4 NLP ANALYSIS & FILTERED RETRIEVAL BENCHMARK")
    print("=" * 70)
    
    # 1. NLP Accuracy Evaluation Report
    print("\n--- 1. EVALUATING NLP ACCURACY (MANUALLY LABELED TEST SET) ---")
    eval_metrics = evaluate_nlp_accuracy()
    print(f"Total Reference Samples : {eval_metrics['total_samples']}")
    print(f"Correct Predictions    : {eval_metrics['correct_predictions']}")
    print(f"Classification Accuracy: {eval_metrics['accuracy'] * 100:.1f}%")
    print(f"Macro F1 Score         : {eval_metrics['macro_f1']:.3f}")
    print("-" * 70)
    
    # 2. Metadata-Filtered Vector Retrieval
    print("\n--- 2. BENCHMARKING METADATA-FILTERED VECTOR RETRIEVAL ---")
    pipeline = RAGPipeline(chroma_path="data/chroma_db", collection_name="wikipedia_chunks")
    
    filtered_queries = [
        {
            "query": "What are key historical events or world history facts?",
            "filter": {"topic": "History & Culture"}
        },
        {
            "query": "What software systems or computer technology developments occurred?",
            "filter": {"topic": "Science & Technology"}
        },
        {
            "query": "What are positive business developments or company successes?",
            "filter": {"sentiment": "positive"}
        }
    ]
    
    for i, item in enumerate(filtered_queries, 1):
        q = item["query"]
        f = item["filter"]
        print(f"[{i}/{len(filtered_queries)}] Query: '{q}'")
        print(f"    Filter Metadata: {f}")
        
        start_t = time.time()
        res = pipeline.retrieve(q, top_k=2, filter_metadata=f)
        lat = time.time() - start_t
        
        print(f"    Retrieval Latency: {lat * 1000:.2f} ms")
        print(f"    Chunks Retrieved : {len(res['chunks'])}")
        
        for idx, (chunk, meta) in enumerate(zip(res['chunks'], res['metadatas']), 1):
            topic_str = meta.get('topic', 'N/A')
            sent_str = meta.get('sentiment', 'N/A')
            entities_str = meta.get('entities', 'None')
            print(f"      [Result {idx}] (Topic: {topic_str} | Sentiment: {sent_str} | Entities: {entities_str[:50]}...)")
            print(f"                   Text Snippet: {chunk[:120].replace('\n', ' ')}...")
            
        print("-" * 70)

    print("\nNLP Analysis & Metadata Filtering Benchmark Complete!")

if __name__ == "__main__":
    run_nlp_benchmark()
