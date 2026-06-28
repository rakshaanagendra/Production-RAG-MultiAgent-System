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

# Make startup script executable
RUN chmod +x startup.sh

EXPOSE 8000

CMD ["./startup.sh"]