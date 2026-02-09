# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set build arguments
ARG DATABASE_URL
ARG GEMINI_API_KEY
ARG SUPABASE_URL
ARG SUPABASE_KEY

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/models \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    # Optimize for Cloud Run
    TOKENIZERS_PARALLELISM=false \
    # Pass build args to runtime environment
    DATABASE_URL=${DATABASE_URL} \
    GEMINI_API_KEY=${GEMINI_API_KEY} \
    SUPABASE_URL=${SUPABASE_URL} \
    SUPABASE_KEY=${SUPABASE_KEY} \
    SUPABASE_BUCKET_NAME=vectorstore-bucket

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model during build (CRITICAL for performance)
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; \
    print('Downloading embeddings model...'); \
    model = HuggingFaceEmbeddings(\
        model_name='sentence-transformers/all-MiniLM-L6-v2',\
        model_kwargs={'device': 'cpu'},\
        encode_kwargs={'normalize_embeddings': True}\
    ); \
    print('✅ Embeddings model downloaded and cached')"

# Copy application code
COPY . .

# Create cache directory with proper permissions
RUN mkdir -p /app/.cache/huggingface && \
    mkdir -p /tmp/vectorstore && \
    chmod -R 777 /app/.cache && \
    chmod -R 777 /tmp/vectorstore

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /tmp/vectorstore

USER appuser

# Expose port (Cloud Run expects 8080)
EXPOSE 8080

# Health check - give more time for startup
HEALTHCHECK --interval=60s --timeout=10s --start-period=180s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8080}/alive || exit 1

# Run with uvicorn
# Using --timeout-keep-alive to keep connections alive longer
# Using --workers 1 to avoid memory issues on Cloud Run
CMD exec uvicorn main:app \
    --host 0.0.0.0 \
    --port ${PORT:-8080} \
    --workers 1 \
    --timeout-keep-alive 75 \
    --log-level info

