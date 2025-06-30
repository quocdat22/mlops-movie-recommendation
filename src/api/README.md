# Movie Recommendation API Documentation

## Overview

The Movie Recommendation API provides content-based movie recommendations using TF-IDF vectorization and cosine similarity. The API is built with FastAPI and provides comprehensive endpoints for movie recommendations, search, and information retrieval.

## Quick Start

### 1. Prerequisites
- Python 3.11+
- Trained model artifacts in `models/content_based/`
- Required dependencies (see `requirements.txt`)

### 2. Start the API Server

```bash
# Using the start script
./start_api.sh

# Or manually
python -m uvicorn src.api.api_service:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access API Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Health & Information

#### GET `/health`
Check API health status
```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime_seconds": 3600.5,
  "total_requests": 150
}
```

#### GET `/model/stats`
Get model statistics
```json
{
  "total_movies": 45000,
  "vocabulary_size": 15000,
  "matrix_shape": [45000, 15000],
  "model_loaded": true,
  "cache_size": 500
}
```

### Movie Information

#### GET `/movies/{movie_id}`
Get information about a specific movie
```json
{
  "movie_id": 550,
  "title": "Fight Club",
  "release_date": "1999-10-15",
  "vote_average": 8.4,
  "genres": "Drama"
}
```

#### GET `/movies/popular?top_k=20`
Get popular movies

#### GET `/movies/random?count=10`
Get random movies for exploration

### Recommendations

#### POST `/recommendations`
Get movie recommendations
```json
{
  "movie_id": 550,
  "top_k": 10,
  "min_similarity": 0.1,
  "exclude_same_year": false
}
```

Response:
```json
{
  "movie_id": 550,
  "recommendations": [
    {
      "movie_id": 807,
      "title": "Se7en",
      "release_date": "1995-09-22",
      "vote_average": 8.4,
      "genres": "Crime, Drama, Mystery",
      "similarity": 0.85
    }
  ],
  "request_time": "2023-12-01T10:00:00",
  "processing_time_ms": 45.2
}
```

#### GET `/recommendations/{movie_id}`
Get recommendations via GET method

#### POST `/recommendations/batch`
Get recommendations for multiple movies
```json
{
  "movie_ids": [550, 807, 155],
  "top_k": 5,
  "min_similarity": 0.1
}
```

### Search

#### POST `/search`
Search for movies using text query
```json
{
  "query": "action adventure superhero",
  "top_k": 10
}
```

#### GET `/search?query=action&top_k=10`
Search via GET method

## Authentication

Optional API key authentication can be enabled by setting the `API_KEYS` environment variable:

```bash
export API_KEYS=your-secret-key-1,your-secret-key-2
```

Include the API key in requests:
```bash
curl -H "X-API-Key: your-secret-key-1" http://localhost:8000/recommendations/550
```

## Rate Limiting

The API includes rate limiting (60 requests per minute per IP by default). Rate-limited requests will receive a 429 status code.

## Error Handling

The API returns standard HTTP status codes:
- 200: Success
- 400: Bad Request
- 401: Unauthorized (invalid API key)
- 404: Not Found
- 429: Too Many Requests
- 500: Internal Server Error
- 503: Service Unavailable (model not loaded)

Error responses include detailed messages:
```json
{
  "detail": "Movie 999999 not found"
}
```

## Docker Deployment

### Build and run with Docker
```bash
# Build the image
docker build -t movie-api -f src/api/Dockerfile .

# Run the container
docker run -p 8000:8000 -v $(pwd)/models:/app/models movie-api
```

### Use Docker Compose
```bash
docker-compose -f docker-compose.api.yml up
```

## Performance

- **Cold start**: ~2-3 seconds (model loading)
- **Recommendation**: ~10-50ms per request
- **Search**: ~50-100ms per request
- **Batch recommendations**: ~100-300ms for 10 movies

## Monitoring

The API includes built-in logging and performance tracking:
- Request/response logging
- Processing time tracking
- Error logging
- Health check endpoint

## Example Usage

### Python Client
```python
import requests

# Get recommendations
response = requests.post("http://localhost:8000/recommendations", json={
    "movie_id": 550,
    "top_k": 5
})
recommendations = response.json()
```

### JavaScript/Frontend
```javascript
// Get recommendations
const response = await fetch('http://localhost:8000/recommendations', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        movie_id: 550,
        top_k: 5
    })
});
const recommendations = await response.json();
```

### cURL
```bash
# Get recommendations
curl -X POST "http://localhost:8000/recommendations" \
     -H "Content-Type: application/json" \
     -d '{"movie_id": 550, "top_k": 5}'

# Search movies
curl "http://localhost:8000/search?query=action%20adventure&top_k=5"
```

## Troubleshooting

### Common Issues

1. **Model not loaded**
   - Ensure model files exist in `models/content_based/`
   - Check file permissions
   - Verify model training completed successfully

2. **Connection refused**
   - Check if the server is running
   - Verify port 8000 is not in use
   - Check firewall settings

3. **Slow responses**
   - Model artifacts might be large
   - Consider model optimization
   - Check available memory

### Logs

Check application logs for detailed error information:
```bash
# If running directly
python -m uvicorn src.api.api_service:app --log-level debug

# If using Docker
docker logs <container_name>
```

For more information, visit the interactive API documentation at `/docs` when the server is running.
