# Base image — slim Python 3.11
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed by some Python packages
# libgomp1 is required by faiss-cpu for OpenMP (parallel computation)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — Docker caches this layer
# If requirements.txt hasn't changed, pip install is skipped on rebuild
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project into container
COPY . .

# Download FAISS index data at build time
ARG HF_TOKEN
RUN python - <<EOF
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id="rakshanagendra/rag-index-data",
    repo_type="dataset",
    local_dir="data/processed",
    token=os.environ.get("HF_TOKEN", "$HF_TOKEN")
)
print("Data downloaded successfully.")
EOF

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]