# APEX SIGNAL™ — Production Dockerfile
FROM python:3.11-slim

LABEL maintainer="APEX SIGNAL™ Team"
LABEL description="Production-grade quantitative trading signal platform"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for ML models and data
RUN mkdir -p /app/ml/models /app/data/cache /app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/healthz'); assert r.status_code == 200"

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "apex_signal.api.app:app", "--host", "0.0.0.0", "--port", "8000"]