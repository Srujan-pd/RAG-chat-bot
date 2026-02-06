# Use a smaller base image
FROM python:3.11-slim

WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY main.py .
COPY database.py .
COPY models.py .
COPY rag_engine.py .
COPY chat.py .
COPY voice_chat.py .
COPY supabase_manager.py .

# Create directories
RUN mkdir -p /tmp/vectorstore

# Create a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Command to run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
