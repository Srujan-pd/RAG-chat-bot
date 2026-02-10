# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py chat.py rag_engine.py database.py models.py supabase_manager.py ./

# Create necessary directories
RUN mkdir -p /tmp/vectorstore /tmp/huggingface

# Expose port (Cloud Run will inject PORT env var)
ENV PORT=8080
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1
