FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages (use latest faiss-cpu)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn[standard]==0.29.0 \
    python-dotenv==1.2.1 \
    sqlalchemy==2.0.29 \
    psycopg2-binary==2.9.11 \
    google-generativeai==0.8.6 \
    supabase==2.27.3 \
    faiss-cpu==1.13.2 \
    sentence-transformers==2.2.2 \
    pydantic==2.12.5 \
    numpy==1.24.3 \
    requests==2.32.5

# Copy app
COPY . .

# Set environment
ENV PORT=8080

# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
