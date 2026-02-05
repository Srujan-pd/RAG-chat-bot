# Use slim Python image (not full)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install ONLY essential system dependencies
# - build-essential: C compiler for some Python packages
# - curl: For health checks
# - Clean up apt cache to reduce image size
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first (Docker layer caching - if requirements don't change, this layer is reused)
COPY requirements.txt .

# Install Python packages
# --no-cache-dir: Don't store pip cache (saves ~500MB)
# --no-deps for torch: Prevents installing unnecessary CUDA dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model during build (avoids runtime download)
# This adds ~100MB but saves time and prevents rate limit errors
RUN mkdir -p /app/model_cache && \
    python3 -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/app/model_cache')"

# Set environment variable so the model is found at runtime
ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache

# Copy application code (done last so code changes don't invalidate previous layers)
COPY . .

# Create runtime directory for vector store downloads
RUN mkdir -p /tmp/vectorstore

# Environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Start application
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
