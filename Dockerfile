# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDWRITEBYTECODE=1 \
    PORT=8080 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/models \
    HF_HUB_ENABLE_HF_TRANSFER=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model during build
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; \
    model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'); \
    print('Model downloaded successfully')"

# Create cache directory with proper permissions
RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8080

# Health check with longer timeout
HEALTHCHECK --interval=30s --timeout=30s --start-period=90s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health', timeout=10)"

# Run the application
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1
