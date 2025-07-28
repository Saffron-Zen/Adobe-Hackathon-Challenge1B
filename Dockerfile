# Use Python 3.12 slim base image for better performance and smaller size
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for PDF processing and other requirements
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt-dev \
    zlib1g-dev \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories with proper permissions
RUN mkdir -p /app/documents /app/output /app/chroma_db /app/analysis_results && \
    chown -R appuser:appuser /app

# Copy application files
COPY --chown=appuser:appuser *.py ./
COPY --chown=appuser:appuser *.json ./
COPY --chown=appuser:appuser setup.sh ./

# Make setup script executable
RUN chmod +x setup.sh

# Copy documents directory if it exists
COPY --chown=appuser:appuser documents/ ./documents/

# Switch to non-root user
USER appuser

# Create volume mount points for persistent data
VOLUME ["/app/documents", "/app/output", "/app/chroma_db", "/app/analysis_results"]

# Expose port for FastAPI (if used)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command - run the document analyzer
CMD ["python", "document_analyzer.py", "input.json", "output.json"]

# Alternative commands you can use:
# For FastAPI server: CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# For interactive mode: CMD ["python", "-i"]
# For bash shell: CMD ["/bin/bash"]
