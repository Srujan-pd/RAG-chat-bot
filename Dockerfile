# Use slim Python image with minimal dependencies
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Install Python packages without cache
RUN pip install --no-cache-dir -r requirements.txt

# PRE-DOWNLOAD THE EMBEDDING MODEL DURING BUILD (CRITICAL!)
# This avoids HuggingFace rate limits at runtime
RUN python3 -c "from sentence_transformers import SentenceTransformer; \
    print('📥 Downloading embedding model...'); \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/tmp'); \
    print('✅ Model downloaded successfully')"

# Set environment variables to use local cache
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV SENTENCE_TRANSFORMERS_HOME=/tmp
ENV TRANSFORMERS_CACHE=/tmp
ENV HF_HOME=/tmp
ENV TORCH_HOME=/tmp

# Copy application code
COPY . .

# Create runtime directories for Cloud Run
RUN mkdir -p /tmp/vectorstore

# Use PORT environment variable
ENV PORT=8080
EXPOSE 8080

# Start application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
