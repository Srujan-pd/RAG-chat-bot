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
    PORT=8080 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/models \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
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

# Pre-download the embedding model during build
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; \
    print('Downloading embeddings model...'); \
    model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'); \
    print('✅ Embeddings model downloaded')"

# Copy application code
COPY . .

# Create cache directory with proper permissions
RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8080

# Health check with longer timeout
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/ready', timeout=10)"

# Run the application
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1 --log-level info
