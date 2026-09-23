import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import time
import os
import sys

# Add src to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.chunking import recursive_character_split

def build_vectordb():
    input_path = "data/processed/clean_corpus.parquet"
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
        
    print("Loading cleaned dataset...")
    df = pd.read_parquet(input_path)
    
    # To keep embedding time reasonable for a weekly assignment without GPU,
    # we'll use a subset if the dataset is too large (e.g. 500 docs instead of 5500)
    # The prompt says "populated vector database". 5000 docs could yield 20,000 chunks
    # which might take a long time on CPU. Let's process the first 1000 docs.
    # 1000 documents should be enough for demonstrating semantic search.
    max_docs = 1000
    if len(df) > max_docs:
        print(f"Using a subset of {max_docs} documents (from {len(df)}) for faster embedding on CPU.")
        df = df.head(max_docs)
    
    print("Chunking documents...")
    all_chunks = []
    all_ids = []
    
    for i, row in df.iterrows():
        text = row.get("text", "")
        doc_id = str(row.get("id", i))
        
        chunks = recursive_character_split(text, chunk_size=500, chunk_overlap=50)
        for j, chunk in enumerate(chunks):
            if chunk.strip():
                all_chunks.append(chunk)
                all_ids.append(f"{doc_id}_{j}")
    
    print(f"Total chunks created: {len(all_chunks)}")
    
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Generating embeddings...")
    start_time = time.time()
    # Batch size controls memory usage
    embeddings = model.encode(all_chunks, batch_size=32, show_progress_bar=True)
    end_time = time.time()
    
    embedding_time = end_time - start_time
    print(f"Embedding generation took {embedding_time:.2f} seconds ({len(all_chunks) / embedding_time:.2f} chunks/sec).")
    
    print("Initializing ChromaDB...")
    os.makedirs("data/chroma_db", exist_ok=True)
    client = chromadb.PersistentClient(path="data/chroma_db")
    
    # Delete existing collection to avoid duplicates if re-run
    try:
        client.delete_collection("wikipedia_chunks")
    except:
        pass
        
    collection = client.create_collection(name="wikipedia_chunks")
    
    print("Ingesting into ChromaDB (handling batch edge cases)...")
    # ChromaDB has max batch size limit. We use 5000 to be safe.
    batch_size = 5000
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i + batch_size]
        batch_ids = all_ids[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size].tolist()
        
        collection.add(
            ids=batch_ids,
            documents=batch_chunks,
            embeddings=batch_embeddings
        )
        print(f"Ingested batch {i // batch_size + 1}/{(len(all_chunks) // batch_size) + 1}")
        
    print("Vector database populated successfully!")

if __name__ == "__main__":
    build_vectordb()
