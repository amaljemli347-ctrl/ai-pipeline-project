import chromadb
import time
from sentence_transformers import SentenceTransformer

def benchmark_retrieval():
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Connecting to ChromaDB...")
    try:
        client = chromadb.PersistentClient(path="data/chroma_db")
        collection = client.get_collection(name="wikipedia_chunks")
    except Exception as e:
        print(f"Error connecting to ChromaDB collection: {e}")
        print("Make sure you run build_vectordb.py first!")
        return
    
    queries = [
        "What is the history of the Internet?",
        "Who discovered penicillin?",
        "Explain the theory of relativity.",
        "How do black holes form?",
        "World War 2 major events"
    ]
    
    print(f"\nRunning {len(queries)} benchmark queries...")
    print("-" * 50)
    
    total_time = 0
    
    for query in queries:
        print(f"Query: '{query}'")
        
        start_time = time.time()
        # Embed the query
        query_embedding = model.encode([query])[0].tolist()
        
        # Search ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        end_time = time.time()
        
        latency = end_time - start_time
        total_time += latency
        print(f"Latency: {latency * 1000:.2f} ms")
        
        # Print top result
        if results['documents'] and results['documents'][0]:
            # Replace newlines for clean printing
            snippet = results['documents'][0][0].replace('\n', ' ')
            print(f"Top result: {snippet[:150]}...")
        else:
            print("No results found.")
            
        print("-" * 50)
        
    print(f"Average latency per query: {(total_time / len(queries)) * 1000:.2f} ms")

if __name__ == "__main__":
    benchmark_retrieval()
