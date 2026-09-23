import sys

def verify_environment():
    print("Verifying Environment...")
    print(f"Python Version: {sys.version}")

    # Check dependencies
    try:
        import pandas as pd
        print(f"[OK] pandas installed (version: {pd.__version__})")
    except ImportError:
        print("[FAIL] pandas not installed")

    try:
        import sentence_transformers
        print(f"[OK] sentence-transformers installed (version: {sentence_transformers.__version__})")
    except ImportError:
        print("[FAIL] sentence-transformers not installed")

    try:
        import chromadb
        print(f"[OK] chromadb installed (version: {chromadb.__version__})")
    except ImportError:
        print("[FAIL] chromadb not installed")

    try:
        import spacy
        print(f"[OK] spacy installed (version: {spacy.__version__})")
    except ImportError:
        print("[FAIL] spacy not installed")

    # Check CUDA/GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] CUDA is available! Device: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARN] CUDA is not available. PyTorch will use CPU.")
    except ImportError:
        print("[FAIL] PyTorch (torch) not installed")

if __name__ == "__main__":
    verify_environment()
