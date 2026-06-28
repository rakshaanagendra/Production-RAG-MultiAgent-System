#!/bin/bash

echo "Downloading index data from Hugging Face Hub..."

python - <<EOF
from huggingface_hub import snapshot_download
import os

snapshot_download(
    repo_id="rakshanagendra/rag-index-data",
    repo_type="dataset",
    local_dir="data/processed",
    token=os.environ.get("HF_TOKEN")
)

print("Data downloaded successfully.")
EOF

echo "Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port 8000