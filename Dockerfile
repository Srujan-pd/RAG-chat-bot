# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python packages with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model (adds ~100MB but saves runtime)
RUN mkdir -p /app/model_cache && \
    python3 -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/app/model_cache')"

# Set environment for model cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache

# Copy only necessary application files
COPY main.py .
COPY rag_engine.py .
COPY database.py .
COPY models.py .
COPY supabase_manager.py .

# Create runtime directories
RUN mkdir -p /tmp/vectorstore

# Environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start application
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1
