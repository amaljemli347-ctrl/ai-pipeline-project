import os
import sys
import time
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.chunking import recursive_character_split
from src.nlp import TopicModeler, SentimentAndNERAnalyzer

def enrich_and_populate_vectordb():
    input_path = "data/processed/clean_corpus.parquet"
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please run Week 1 preprocessing first.")
        return

    print("Loading cleaned dataset for NLP Enrichment...")
    df = pd.read_parquet(input_path)
    
    max_docs = 1000
    if len(df) > max_docs:
        print(f"Using top {max_docs} documents for enrichment & embedding.")
        df = df.head(max_docs)

    raw_texts = df["text"].tolist()
    
    print("Fitting Topic Modeling (LDA) on corpus...")
    topic_modeler = TopicModeler(n_topics=5)
    topic_modeler.fit(raw_texts)
    
    print("Initializing Sentiment & NER Analyzer...")
    nlp_analyzer = SentimentAndNERAnalyzer()

    print("Chunking documents and extracting NLP metadata...")
    all_chunks = []
    all_ids = []
    all_metadatas = []
    
    for i, row in df.iterrows():
        text = row.get("text", "")
        doc_id = str(row.get("id", i))
        
        chunks = recursive_character_split(text, chunk_size=500, chunk_overlap=50)
        
        for j, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
                
            # NLP Metadata Extraction
            topic_info = topic_modeler.predict_topic(chunk)
            nlp_info = nlp_analyzer.analyze(chunk)
            
            metadata = {
                "doc_id": doc_id,
                "topic": topic_info["topic_name"],
                "topic_confidence": float(topic_info["confidence"]),
                "sentiment": nlp_info["sentiment"],
                "sentiment_score": float(nlp_info["sentiment_score"]),
                "entities": str(nlp_info["entities"])[:200]
            }
            
            all_chunks.append(chunk)
            all_ids.append(f"{doc_id}_{j}")
            all_metadatas.append(metadata)

    print(f"Total metadata-enriched chunks created: {len(all_chunks)}")
    
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Generating embeddings...")
    start_time = time.time()
    embeddings = model.encode(all_chunks, batch_size=32, show_progress_bar=True)
    end_time = time.time()
    print(f"Embedding generation took {end_time - start_time:.2f} seconds.")
    
    print("Connecting to ChromaDB and populating collection with NLP metadata...")
    os.makedirs("data/chroma_db", exist_ok=True)
    client = chromadb.PersistentClient(path="data/chroma_db")
    
    try:
        client.delete_collection("wikipedia_chunks")
    except Exception:
        pass
        
    collection = client.create_collection(name="wikipedia_chunks")
    
    batch_size = 5000
    for b in range(0, len(all_chunks), batch_size):
        b_chunks = all_chunks[b:b + batch_size]
        b_ids = all_ids[b:b + batch_size]
        b_embeds = embeddings[b:b + batch_size].tolist()
        b_metas = all_metadatas[b:b + batch_size]
        
        collection.add(
            ids=b_ids,
            documents=b_chunks,
            embeddings=b_embeds,
            metadatas=b_metas
        )
        print(f"Ingested enriched batch {b // batch_size + 1}/{(len(all_chunks) // batch_size) + 1}")
        
    print("Vector DB enriched with NLP metadata successfully!")

if __name__ == "__main__":
    enrich_and_populate_vectordb()
