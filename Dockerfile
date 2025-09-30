# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080

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

# Copy application code
COPY . .
COPY chave.json /app/chave.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/chave.json

# Create temp directories
RUN mkdir -p temp_graficos/arquivados

# Remove arquivos desnecessários do container
RUN rm -rf /app/__pycache__ /app/temp_graficos /app/Programa.rar /app/README.md /app/LICENSE

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/_stcore/health || exit 1

# Run the application with proper upload limits for interceptation
CMD ["sh", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --browser.serverAddress=0.0.0.0 --browser.serverPort=$PORT --server.maxUploadSize=200 --server.maxMessageSize=200 --server.enableCORS=false"]