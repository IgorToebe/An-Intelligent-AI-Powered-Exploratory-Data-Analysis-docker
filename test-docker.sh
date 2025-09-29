#!/bin/bash
# Teste local da aplicação Docker

echo "🐳 Testando aplicação Docker localmente..."

# Build da imagem
echo "📦 Building Docker image..."
docker build -t i2a2-eda-platform-local .

# Parar container anterior se existir
echo "🛑 Stopping previous container..."
docker stop i2a2-eda-platform-local 2>/dev/null || true
docker rm i2a2-eda-platform-local 2>/dev/null || true

# Executar container
echo "🚀 Starting container..."
docker run -d \
    --name i2a2-eda-platform-local \
    -p 8080:8080 \
    -e GOOGLE_API_KEY="${GOOGLE_API_KEY}" \
    -v "$(pwd)/temp_graficos:/app/temp_graficos" \
    i2a2-eda-platform-local

echo "✅ Container started!"
echo "🌐 Application available at: http://localhost:8080"
echo ""
echo "📋 Useful commands:"
echo "  docker logs i2a2-eda-platform-local     # View logs"
echo "  docker stop i2a2-eda-platform-local     # Stop container"
echo "  docker exec -it i2a2-eda-platform-local bash  # Access container shell"