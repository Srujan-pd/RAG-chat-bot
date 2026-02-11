FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    MODEL_CACHE=/app/model_cache \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies including audio for TTS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create cache directory for HuggingFace models
RUN mkdir -p /app/model_cache /tmp/vectorstore /data && \
    chmod 777 /app/model_cache /tmp/vectorstore /data

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make startup check executable
RUN chmod +x startup_check.py

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application with proper startup
CMD python startup_check.py && \
    exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 75
