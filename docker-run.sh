#!/bin/bash

# MLOps Movie Recommendation - Docker Launcher
# This script builds and starts the entire MLOps pipeline using Docker Compose

set -e  # Exit on any error

echo "🎬 MLOps Movie Recommendation - Docker Deployment"
echo "=================================================="

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     - Start all services (default)"
    echo "  stop      - Stop all services"
    echo "  restart   - Restart all services"
    echo "  build     - Build/rebuild all Docker images"
    echo "  logs      - Show logs from all services"
    echo "  status    - Show status of all services"
    echo "  clean     - Stop and remove all containers, networks, and volumes"
    echo "  api-only  - Start only API service"
    echo "  help      - Show this help message"
}

# Function to check if .env file exists and has required variables
check_env() {
    if [[ ! -f .env ]]; then
        echo "❌ .env file not found. Creating a sample .env file..."
        cp .env.production .env
        echo "📝 Please edit .env file and add your TMDB_API_KEY"
        exit 1
    fi
    
    if ! grep -q "TMDB_API_KEY" .env || grep -q "your_tmdb_api_key_here" .env; then
        echo "⚠️  Warning: TMDB_API_KEY not set in .env file"
        echo "📝 Please add your TMDB API key to the .env file"
    fi
}

# Function to create necessary directories
create_directories() {
    echo "📁 Creating necessary directories..."
    mkdir -p logs monitoring airflow/logs airflow/plugins
    
    # Set permissions for Airflow
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "🔐 Setting permissions for Airflow (Linux)..."
        echo -e "AIRFLOW_UID=$(id -u)" >> .env
    fi
}

# Function to start services
start_services() {
    echo "🚀 Starting MLOps Movie Recommendation services..."
    
    check_env
    create_directories
    
    # Build and start services
    docker-compose up -d
    
    echo ""
    echo "✅ Services are starting up. This may take a few minutes..."
    echo ""
    echo "🔗 Service URLs:"
    echo "   📊 Dashboard:     http://localhost:8501"
    echo "   🔌 API:          http://localhost:8000"
    echo "   📚 API Docs:     http://localhost:8000/docs"
    echo "   ✈️  Airflow:      http://localhost:8080 (admin/admin)"
    echo "   📈 Grafana:      http://localhost:3000 (admin/admin)"
    echo ""
    echo "⏳ Waiting for services to be ready..."
    
    # Wait for API to be ready
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "✅ API service is ready!"
            break
        elif [[ $i -eq 30 ]]; then
            echo "⚠️  API service is taking longer than expected to start"
            echo "   You can check logs with: $0 logs"
        else
            echo "   Waiting for API... ($i/30)"
            sleep 10
        fi
    done
    
    echo ""
    echo "🎉 MLOps Movie Recommendation is ready!"
    echo "📖 Check the logs with: $0 logs"
}

# Function to stop services
stop_services() {
    echo "🛑 Stopping all services..."
    docker-compose down
    echo "✅ All services stopped"
}

# Function to restart services
restart_services() {
    echo "🔄 Restarting all services..."
    stop_services
    start_services
}

# Function to build images
build_images() {
    echo "🔨 Building Docker images..."
    docker-compose build --no-cache
    echo "✅ All images built successfully"
}

# Function to show logs
show_logs() {
    echo "📋 Showing logs from all services..."
    docker-compose logs -f
}

# Function to show status
show_status() {
    echo "📊 Service Status:"
    docker-compose ps
    
    echo ""
    echo "💾 Volume Usage:"
    docker system df
    
    echo ""
    echo "🌐 Network Status:"
    docker network ls | grep mlops
}

# Function to clean up everything
clean_up() {
    echo "🧹 Cleaning up all Docker resources..."
    read -p "This will remove all containers, networks, and volumes. Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose down -v --rmi all --remove-orphans
        docker system prune -f
        echo "✅ Cleanup completed"
    else
        echo "❌ Cleanup cancelled"
    fi
}

# Function to start only API service
start_api_only() {
    echo "🚀 Starting API service only..."
    check_env
    create_directories
    
    docker-compose up -d movie-api postgres
    
    echo ""
    echo "✅ API service is starting..."
    echo "🔗 API URL: http://localhost:8000"
    echo "📚 API Docs: http://localhost:8000/docs"
}

# Main script logic
case "${1:-start}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    build)
        build_images
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    clean)
        clean_up
        ;;
    api-only)
        start_api_only
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac
