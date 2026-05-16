from sentence_transformers import SentenceTransformer
import os
import sys
import json
import numpy as np

# Handle imports for both script and module contexts
try:
    from .loader import load_documents
    from .chunker import chunk_documents
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from loader import load_documents # pyright: ignore[reportMissingImports]
    from chunker import chunk_documents # pyright: ignore[reportMissingImports]

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def embed_chunks(chunks):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    return embeddings


if __name__ == "__main__":
    # Resolve data path relative to project root.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    resolved_data_path = os.path.join(project_root, "data", "raw")

    documents = load_documents(resolved_data_path)
    chunks = chunk_documents(documents)
    embeddings = embed_chunks(chunks)

    print(f"Loaded {len(documents)} document(s)")
    print(f"Created {len(chunks)} chunk(s)")
    print(f"Generated {len(embeddings)} embedding(s)")
    print(f"Embedding dimension: {embeddings[0].shape[0]}")
    print(f"\nSample embedding (first 10 values):")
    print(embeddings[0][:10])