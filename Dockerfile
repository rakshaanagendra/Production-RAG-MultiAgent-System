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

# Tell Docker this container listens on port 8000
EXPOSE 8000

# Start FastAPI with uvicorn when container runs
# host 0.0.0.0 means accept connections from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]