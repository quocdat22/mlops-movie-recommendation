#!/bin/bash

# Start Movie Recommendation API Server
# This script starts the FastAPI server with proper configuration

set -e

echo "Starting Movie Recommendation API Server..."

# Set default environment variables
export API_HOST=${API_HOST:-"0.0.0.0"}
export API_PORT=${API_PORT:-"8000"}
export API_WORKERS=${API_WORKERS:-"1"}
export LOG_LEVEL=${LOG_LEVEL:-"INFO"}

# Check if model exists
MODEL_DIR="./models/content_based"
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found at $MODEL_DIR"
    echo "Please train the model first using:"
    echo "  python src/models/train_model.py"
    exit 1
fi

# Check if model files exist
if [ ! -f "$MODEL_DIR/tfidf_vectorizer.pkl" ] || [ ! -f "$MODEL_DIR/similarity_matrix.npy" ]; then
    echo "ERROR: Required model files not found in $MODEL_DIR"
    echo "Please ensure the following files exist:"
    echo "  - tfidf_vectorizer.pkl"
    echo "  - similarity_matrix.npy"
    echo "  - movie_metadata.pkl"
    exit 1
fi

echo "Model files found. Starting server..."

# Start the server
python -m uvicorn src.api.api_service:app \
    --host $API_HOST \
    --port $API_PORT \
    --workers $API_WORKERS \
    --log-level $(echo $LOG_LEVEL | tr '[:upper:]' '[:lower:]') \
    --reload
