#!/bin/bash

# Start Monitoring Stack for Movie Recommendation API
# This script starts Prometheus and Grafana for monitoring

set -e

echo "Starting Movie Recommendation Monitoring Stack..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if model exists
MODEL_DIR="./models/content_based"
if [ ! -d "$MODEL_DIR" ]; then
    echo "WARNING: Model directory not found at $MODEL_DIR"
    echo "The API service may not start properly without a trained model."
fi

# Set environment variables
export COMPOSE_PROJECT_NAME=movie-monitoring

echo "Starting monitoring services..."

# Start the monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

# Check if services are running
echo "Checking service status..."

# Check Prometheus
if curl -s http://localhost:9090/api/v1/query?query=up > /dev/null; then
    echo "✅ Prometheus is running at http://localhost:9090"
else
    echo "❌ Prometheus is not responding"
fi

# Check Grafana
if curl -s http://localhost:3000/api/health > /dev/null; then
    echo "✅ Grafana is running at http://localhost:3000"
    echo "   Default credentials: admin/admin123"
else
    echo "❌ Grafana is not responding"
fi

# Check API (if running)
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Movie API is running at http://localhost:8000"
    echo "   Metrics available at http://localhost:8000/metrics"
else
    echo "ℹ️  Movie API is not running. Start it separately if needed."
fi

echo ""
echo "Monitoring stack is ready!"
echo ""
echo "Access points:"
echo "- Prometheus: http://localhost:9090"
echo "- Grafana: http://localhost:3000 (admin/admin123)"
echo "- API Health: http://localhost:8000/health"
echo "- API Metrics: http://localhost:8000/metrics"
echo "- API Docs: http://localhost:8000/docs"
echo ""
echo "To stop the monitoring stack:"
echo "  docker-compose -f docker-compose.monitoring.yml down"
