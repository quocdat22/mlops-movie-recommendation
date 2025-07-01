#!/bin/bash

# Movie Recommendation API with GCS Model Loading
# This script builds and runs the API with model downloading from Google Cloud Storage

set -e

echo "🎬 Movie Recommendation API with GCS"
echo "======================================"

# Check if credentials file exists
if [ ! -f "gcs-credentials.json" ]; then
    echo "❌ ERROR: gcs-credentials.json not found!"
    echo "Please place your Google Cloud Service Account credentials file as 'gcs-credentials.json'"
    echo "in the current directory."
    exit 1
fi

# Set default environment variables
export GCS_BUCKET_NAME="${GCS_BUCKET_NAME:-staging.doanandroid-e3111.appspot.com}"
export GCS_MODEL_PATH="${GCS_MODEL_PATH:-content_based}"

echo "📋 Configuration:"
echo "   GCS Bucket: $GCS_BUCKET_NAME"
echo "   Model Path: $GCS_MODEL_PATH"
echo "   Port: 8000"
echo ""

# Create logs directory
mkdir -p logs

echo "🔨 Building Docker image..."
docker-compose -f docker-compose.gcs.yml build

echo "🚀 Starting API service..."
echo "This will download the model from GCS on first run..."
echo ""

# Run with proper environment variables
docker-compose -f docker-compose.gcs.yml up

echo "✅ API service stopped"