import os
import pandas as pd
import sys

# Add src to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import process_document

def process_corpus(input_path="data/raw/raw_data.parquet", output_dir="data/processed", filename="clean_corpus.parquet"):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
        
    print(f"Loading raw data from {input_path}...")
    df = pd.read_parquet(input_path)
    
    print(f"Processing {len(df)} documents...")
    
    cleaned_docs = []
    
    for i, row in df.iterrows():
        text = row.get("text", "")
        
        result = process_document(text)
        
        if result["is_valid"]:
            cleaned_row = row.copy()
            cleaned_row["text"] = result["cleaned_text"]
            cleaned_docs.append(cleaned_row)
            
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} documents...")
            
    cleaned_df = pd.DataFrame(cleaned_docs)
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    cleaned_df.to_parquet(out_path, index=False)
    print(f"Saved cleaned corpus to {out_path} with {len(cleaned_df)} records (filtered from {len(df)}).")

if __name__ == "__main__":
    process_corpus()
