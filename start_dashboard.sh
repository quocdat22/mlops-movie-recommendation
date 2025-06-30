#!/bin/bash

# MLOps Movie Recommendation Dashboard Launcher
# This script starts both the API service and the Streamlit dashboard

echo "🎬 Starting MLOps Movie Recommendation Dashboard..."

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: No virtual environment detected. Consider activating your venv."
fi

# Install requirements if needed
echo "📦 Checking dependencies..."
pip install -r requirements.txt --quiet

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -i:$port >/dev/null 2>&1; then
        echo "Port $port is already in use"
        return 1
    else
        return 0
    fi
}

# Start API service in background
echo "🚀 Starting API service..."
if check_port 8000; then
    python -m src.api.api_service &
    API_PID=$!
    echo "API service started with PID: $API_PID"
    sleep 5  # Give API time to start
else
    echo "⚠️  API service may already be running on port 8000"
fi

# Start Streamlit dashboard
echo "📊 Starting Streamlit dashboard..."
if check_port 8501; then
    # Set environment variables to avoid email prompt
    export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
    streamlit run src/dashboard/mlops_dashboard.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &
    DASHBOARD_PID=$!
    echo "Dashboard started with PID: $DASHBOARD_PID"
else
    echo "⚠️  Dashboard may already be running on port 8501"
fi

# Print access information
echo ""
echo "✅ Services started successfully!"
echo "🔗 Access the dashboard at: http://localhost:8501"
echo "🔗 API endpoints at: http://localhost:8000"
echo "📚 API documentation at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup processes on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    if [[ ! -z "$API_PID" ]]; then
        kill $API_PID 2>/dev/null
        echo "API service stopped"
    fi
    if [[ ! -z "$DASHBOARD_PID" ]]; then
        kill $DASHBOARD_PID 2>/dev/null
        echo "Dashboard stopped"
    fi
    echo "✅ All services stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Keep script running
while true; do
    sleep 1
done
