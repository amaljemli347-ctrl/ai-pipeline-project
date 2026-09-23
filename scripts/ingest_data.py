import os
import pandas as pd
from datasets import load_dataset

def download_dataset_subset(num_docs=5500, output_dir="data/raw", filename="raw_data.parquet"):
    print(f"Downloading {num_docs} documents from wikimedia/wikipedia (20231101.en)...")
    
    try:
        # load_dataset with explicit namespace
        dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        docs = []
        for i, row in enumerate(dataset):
            if i >= num_docs:
                break
            docs.append(row)
            if (i + 1) % 1000 == 0:
                print(f"Fetched {i + 1} documents...")
    except Exception as e:
        print(f"Error fetching from huggingface: {e}")
        return

    df = pd.DataFrame(docs)
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    df.to_parquet(out_path, index=False)
    print(f"Saved raw data to {out_path} with {len(df)} records.")

if __name__ == "__main__":
    download_dataset_subset()
