FROM python:3.11-slim

# Install system dependencies for libspatialindex (R-tree)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libspatialindex-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e "."

# Create output directories
RUN mkdir -p experiments/results docs/figures

# Expose API port
EXPOSE 8000

# Default command: start the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
