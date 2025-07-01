#!/bin/bash
set -e

echo "Starting Movie Recommendation API..."

# Check if GCS credentials are available
if [ ! -f "/gcs-credentials.json" ]; then
    echo "ERROR: GCS credentials file not found at /gcs-credentials.json"
    echo "Please mount the credentials file as a volume"
    exit 1
fi

# Set up Google Cloud credentials
export GOOGLE_APPLICATION_CREDENTIALS="/gcs-credentials.json"

# Check required environment variables
if [ -z "$GCS_BUCKET_NAME" ]; then
    echo "ERROR: GCS_BUCKET_NAME environment variable is required"
    exit 1
fi

if [ -z "$GCS_MODEL_PATH" ]; then
    echo "WARNING: GCS_MODEL_PATH not set, using default 'content_based'"
    export GCS_MODEL_PATH="content_based"
fi

echo "Configuration:"
echo "  GCS Bucket: $GCS_BUCKET_NAME"
echo "  GCS Model Path: $GCS_MODEL_PATH"
echo "  Model Directory: ${MODEL_DIR:-./models/content_based}"

# Create models directory
mkdir -p /app/models/content_based

# Try to download model if not exists
echo "Checking if model needs to be downloaded..."

python3 -c "
import os
import sys
sys.path.append('/app/src')

try:
    from models.model_inference import ContentBasedRecommender
    
    model_dir = os.getenv('MODEL_DIR', '/app/models/content_based')
    print(f'Using model directory: {model_dir}')
    
    recommender = ContentBasedRecommender(model_dir)
    print('Model setup completed successfully')
except Exception as e:
    print(f'Error setting up model: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to download/setup model"
    exit 1
fi

echo "Model setup completed. Starting API server..."

# Start the API service
exec python -m src.api.api_service