#!/bin/bash

# Health Check Script for MLOps Movie Recommendation Services

echo "🔍 MLOps Movie Recommendation - Health Check"
echo "============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check service health
check_service() {
    local service_name=$1
    local url=$2
    local expected_status=${3:-200}

    echo -n "Checking $service_name... "

    response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null)

    if [[ $response -eq $expected_status ]]; then
        echo -e "${GREEN}✅ Healthy${NC}"
        return 0
    else
        echo -e "${RED}❌ Unhealthy (HTTP $response)${NC}"
        return 1
    fi
}

# Function to check database connection
check_database() {
    echo -n "Checking PostgreSQL... "

    if docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Connected${NC}"
        return 0
    else
        echo -e "${RED}❌ Connection failed${NC}"
        return 1
    fi
}

# Function to check Airflow database
check_airflow_db() {
    echo -n "Checking Airflow PostgreSQL... "

    if docker-compose exec -T postgres-airflow pg_isready -U airflow > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Connected${NC}"
        return 0
    else
        echo -e "${RED}❌ Connection failed${NC}"
        return 1
    fi
}

# Function to check if containers are running
check_containers() {
    echo "📦 Container Status:"

    services=("movie-api" "dashboard" "postgres" "postgres-airflow" "airflow-webserver" "airflow-scheduler")

    for service in "${services[@]}"; do
        echo -n "  $service... "

        if docker-compose ps "$service" 2>/dev/null | grep -q "Up"; then
            echo -e "${GREEN}✅ Running${NC}"
        else
            echo -e "${RED}❌ Not running${NC}"
        fi
    done
    echo
}

# Function to check API endpoints
check_api_endpoints() {
    echo "🔌 API Endpoints:"

    # Check health endpoint
    check_service "Health" "http://localhost:8000/health"

    # Check model stats endpoint
    check_service "Model Stats" "http://localhost:8000/model/stats"

    # Check docs endpoint
    check_service "API Docs" "http://localhost:8000/docs"

    echo
}

# Function to check web services
check_web_services() {
    echo "🌐 Web Services:"

    # Check Dashboard
    check_service "Streamlit Dashboard" "http://localhost:8501/healthz"

    # Check Airflow
    check_service "Airflow Webserver" "http://localhost:8080/health"

    # Check Grafana
    check_service "Grafana" "http://localhost:3000/api/health"

    echo
}

# Function to test recommendation functionality
test_recommendations() {
    echo "🎬 Testing Recommendation API:"

    echo -n "  Movie recommendations... "

    response=$(curl -s -X POST "http://localhost:8000/recommendations" \
        -H "Content-Type: application/json" \
        -d '{"movie_id": 278, "top_k": 3}' 2>/dev/null)

    if echo "$response" | grep -q "recommendations"; then
        echo -e "${GREEN}✅ Working${NC}"
    else
        echo -e "${RED}❌ Failed${NC}"
    fi

    echo -n "  Movie search... "

    response=$(curl -s -X POST "http://localhost:8000/search" \
        -H "Content-Type: application/json" \
        -d '{"query": "action", "top_k": 3}' 2>/dev/null)

    if echo "$response" | grep -q "results"; then
        echo -e "${GREEN}✅ Working${NC}"
    else
        echo -e "${RED}❌ Failed${NC}"
    fi

    echo
}

# Function to show resource usage
show_resource_usage() {
    echo "💻 Resource Usage:"

    echo "  Docker containers:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -10

    echo
    echo "  Disk usage:"
    docker system df

    echo
}

# Function to show recent logs
show_recent_logs() {
    echo "📋 Recent Logs (last 10 lines per service):"

    services=("movie-api" "dashboard" "airflow-webserver" "airflow-scheduler")

    for service in "${services[@]}"; do
        echo "--- $service ---"
        docker-compose logs --tail=5 "$service" 2>/dev/null || echo "No logs available"
        echo
    done
}

# Main health check execution
main() {
    echo "$(date): Starting health check..."
    echo

    # Check if docker-compose is available
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        echo -e "${RED}❌ Docker Compose not found${NC}"
        exit 1
    fi

    # Run all checks
    check_containers
    check_database
    check_airflow_db
    check_api_endpoints
    check_web_services
    test_recommendations

    # Optional detailed checks
    if [[ "$1" == "--detailed" || "$1" == "-d" ]]; then
        show_resource_usage
        show_recent_logs
    fi

    echo "🏁 Health check completed at $(date)"
    echo
    echo "💡 Tips:"
    echo "  - Use '$0 --detailed' for more information"
    echo "  - Check logs with: docker-compose logs [service-name]"
    echo "  - Restart services with: ./docker-run.sh restart"
}

# Show usage if help requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --detailed, -d    Show detailed resource usage and logs"
    echo "  --help, -h        Show this help message"
    echo
    echo "This script checks the health of all MLOps Movie Recommendation services."
    exit 0
fi

# Run main function
main "$@"
