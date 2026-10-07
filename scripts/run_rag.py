import os
import sys
import time
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag import RAGPipeline
from src.llm import LLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RAG_Runner")

def run_rag_benchmark():
    print("=" * 70)
    print("      ENTERPRISE RAG SYSTEM - WEEK 3 BENCHMARK RUNNER")
    print("=" * 70)
    
    # Initialize pipeline
    llm = LLMClient()
    pipeline = RAGPipeline(
        chroma_path="data/chroma_db",
        collection_name="wikipedia_chunks",
        llm_client=llm
    )
    
    test_queries = [
        # In-domain / In-corpus queries (Wikipedia context expected)
        "What is the history or definition of Wikipedia?",
        "Who founded Microsoft or Apple?",
        "What are the main events of World War II?",
        
        # Out-of-domain / Unrelated / Off-topic queries (Hallucination check)
        "What is the private password for the enterprise server?",
        "Who won the 2030 World Cup in Tokyo?",
        "What is the secret recipe for Coca-Cola according to internal documents?"
    ]
    
    benchmark_results = []
    
    print(f"\nExecuting {len(test_queries)} test queries through RAG pipeline...\n")
    
    for i, q in enumerate(test_queries, 1):
        print(f"[{i}/{len(test_queries)}] Query: '{q}'")
        res = pipeline.query(q, top_k=3)
        
        print(f"    |- LLM Status:  {res['llm_status']} ({res['model_used']})")
        print(f"    |- Latency:     Retrieval: {res['latency']['retrieval_s']}s | Generation: {res['latency']['generation_s']}s | Total: {res['latency']['total_s']}s")
        print(f"    |- Grounded:    {res['groundedness_check']['grounded']} (Out-of-Domain: {res['groundedness_check']['out_of_domain']})")
        print(f"    |- Reason:      {res['groundedness_check']['reason']}")
        print(f"    +- Answer:      {res['answer'][:200]}...")
        print("-" * 70)
        
        benchmark_results.append(res)
        time.sleep(0.2)

    # Average latencies
    avg_retrieval = sum(r['latency']['retrieval_s'] for r in benchmark_results) / len(benchmark_results)
    avg_generation = sum(r['latency']['generation_s'] for r in benchmark_results) / len(benchmark_results)
    avg_total = sum(r['latency']['total_s'] for r in benchmark_results) / len(benchmark_results)
    
    print("\n" + "=" * 70)
    print("                    BENCHMARK LATENCY SUMMARY")
    print("=" * 70)
    print(f"Average Retrieval Latency  : {avg_retrieval:.4f} s ({avg_retrieval*1000:.1f} ms)")
    print(f"Average Generation Latency : {avg_generation:.4f} s ({avg_generation*1000:.1f} ms)")
    print(f"Average End-to-End Latency : {avg_total:.4f} s ({avg_total*1000:.1f} ms)")
    print("=" * 70)

if __name__ == "__main__":
    run_rag_benchmark()
