#!/usr/bin/env python3
"""
Movie Recommendation API Service

FastAPI-based REST API for serving content-based movie recommendations.
Provides endpoints for movie recommendations, search, and model information.

Features:
- RESTful API endpoints
- Real-time recommendations
- Text-based movie search
- Model health checks
- Performance monitoring
- Error handling and validation
- API documentation with Swagger

Author: MLOps Movie Recommendation System
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import json
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, HTTPException, Query, Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator

# Add src to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))

# Import configuration and middleware
from .config import settings
from .middleware import APIKeyMiddleware, LoggingMiddleware, RateLimitMiddleware
from .metrics import metrics

# Import our recommendation model
from models.model_inference import ContentBasedRecommender

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Pydantic models for API
class RecommendationRequest(BaseModel):
    movie_id: int = Field(..., description="ID of the movie to get recommendations for")
    top_k: int = Field(10, ge=1, le=settings.MAX_RECOMMENDATIONS, description="Number of recommendations to return")
    min_similarity: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity threshold")
    exclude_same_year: bool = Field(False, description="Exclude movies from the same year")

class BatchRecommendationRequest(BaseModel):
    movie_ids: List[int] = Field(..., description="List of movie IDs")
    top_k: int = Field(10, ge=1, le=50, description="Number of recommendations per movie")
    min_similarity: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity threshold")

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Text query describing movie features")
    top_k: int = Field(10, ge=1, le=50, description="Number of similar movies to return")

class MovieInfo(BaseModel):
    movie_id: int
    title: str
    release_date: str
    vote_average: float
    genres: str
    similarity: Optional[float] = None

class RecommendationResponse(BaseModel):
    movie_id: int
    recommendations: List[MovieInfo]
    request_time: str
    processing_time_ms: float

class BatchRecommendationResponse(BaseModel):
    results: Dict[int, List[MovieInfo]]
    request_time: str
    processing_time_ms: float

class SearchResponse(BaseModel):
    query: str
    results: List[MovieInfo]
    request_time: str
    processing_time_ms: float

class ModelStats(BaseModel):
    total_movies: int
    vocabulary_size: int
    matrix_shape: List[int]
    model_loaded: bool
    cache_size: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
    total_requests: int

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    global recommender
    
    # Startup
    logger.info("Starting Movie Recommendation API...")
    
    # Find model directory
    model_dir = settings.get_model_path()
    
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        logger.error("Please train the model first using train_model.py")
    else:
        # Initialize and load model
        recommender = ContentBasedRecommender(str(model_dir))
        success = recommender.load_model()
        
        if success:
            stats = recommender.get_model_stats()
            logger.info(f"Model loaded successfully with {stats['total_movies']:,} movies")
            # Update model metrics
            metrics.update_model_status(
                is_loaded=True,
                total_movies=stats['total_movies'],
                vocab_size=stats['vocabulary_size']
            )
        else:
            logger.error("Failed to load recommendation model")
            metrics.update_model_status(is_loaded=False)
    
    yield  # This is where the app runs
    
    # Shutdown
    logger.info("Shutting down Movie Recommendation API...")

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
app.add_middleware(APIKeyMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument app with Prometheus
Instrumentator().instrument(app).expose(app)

# Global variables
recommender = None
app_start_time = datetime.now()
request_count = 0

# Utility functions
def track_request():
    """Track API requests"""
    global request_count
    request_count += 1

def get_processing_time(start_time: datetime) -> float:
    """Calculate processing time in milliseconds"""
    return (datetime.now() - start_time).total_seconds() * 1000

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    track_request()
    return {
        "message": "Movie Recommendation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    track_request()
    
    uptime = (datetime.now() - app_start_time).total_seconds()
    
    return HealthResponse(
        status="healthy" if recommender and recommender.is_loaded else "unhealthy",
        model_loaded=recommender.is_loaded if recommender else False,
        uptime_seconds=uptime,
        total_requests=request_count
    )

@app.get("/model/stats", response_model=ModelStats)
async def get_model_stats():
    """Get model statistics"""
    track_request()
    
    if not recommender or not recommender.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    stats = recommender.get_model_stats()
    
    # Update cache metrics
    metrics.update_cache_size(stats.get('cache_size', 0))
    
    return ModelStats(
        total_movies=stats['total_movies'],
        vocabulary_size=stats['vocabulary_size'],
        matrix_shape=list(stats['matrix_shape']),
        model_loaded=stats['is_loaded'],
        cache_size=stats['cache_size']
    )

@app.get("/monitoring/summary")
async def get_monitoring_summary():
    """Get monitoring metrics summary"""
    track_request()
    
    try:
        # Get current metric values safely
        def get_metric_value(metric, default=0):
            try:
                if hasattr(metric, '_value'):
                    if hasattr(metric._value, '_value'):
                        return metric._value._value
                    return metric._value
                elif hasattr(metric, 'get'):
                    return metric.get()
                else:
                    # For Counter objects, try to get sample value
                    samples = list(metric.collect()[0].samples)
                    return samples[0].value if samples else default
            except Exception:
                return default
        
        return {
            "model_status": {
                "loaded": bool(get_metric_value(metrics.model_loaded, 0)),
                "total_movies": int(get_metric_value(metrics.model_total_movies, 0)),
                "vocabulary_size": int(get_metric_value(metrics.model_vocabulary_size, 0))
            },
            "cache_metrics": {
                "hits": int(get_metric_value(metrics.cache_hits_total, 0)),
                "misses": int(get_metric_value(metrics.cache_misses_total, 0)),
                "current_size": int(get_metric_value(metrics.cache_size, 0))
            },
            "request_metrics": {
                "recommendations_generated": int(get_metric_value(metrics.recommendations_generated_total, 0)),
                "search_queries": int(get_metric_value(metrics.search_queries_total, 0)),
                "model_errors": int(get_metric_value(metrics.model_errors_total, 0))
            }
        }
    except Exception as e:
        logger.error(f"Error getting monitoring summary: {e}")
        # Return basic info even if metrics fail
        return {
            "model_status": {
                "loaded": bool(recommender and recommender.is_loaded),
                "total_movies": 0,
                "vocabulary_size": 0
            },
            "cache_metrics": {
                "hits": 0,
                "misses": 0,
                "current_size": 0
            },
            "request_metrics": {
                "recommendations_generated": 0,
                "search_queries": 0,
                "model_errors": 0
            }
        }

@app.get("/movies/popular", response_model=List[MovieInfo])
async def get_popular_movies(
    top_k: int = Query(20, ge=1, le=100, description="Number of popular movies")
):
    """Get most popular movies based on vote count and average"""
    track_request()
    start_time = datetime.now()
    
    if not recommender or not recommender.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Recommender returns a list of dicts, not a DataFrame
        popular_movies_list = recommender.get_popular_movies(top_k=top_k)
        
        # Convert list of dicts to list of MovieInfo objects
        result = [MovieInfo(**movie) for movie in popular_movies_list]
        
        processing_time = get_processing_time(start_time)
        metrics.record_popular_movies_request(processing_time)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting popular movies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/movies/random", response_model=List[MovieInfo])
async def get_random_movies(
    count: int = Query(10, ge=1, le=50, description="Number of random movies")
):
    """Get a random selection of movies"""
    track_request()
    
    if not recommender or not recommender.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        # Recommender returns a list of dicts
        random_movies_list = recommender.get_random_movies(count=count)
        
        # Convert list of dicts to list of MovieInfo objects
        result = [MovieInfo(**movie) for movie in random_movies_list]
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting random movies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/movies/{movie_id}", response_model=MovieInfo)
async def get_movie_info(movie_id: int = PathParam(..., description="Movie ID")):
    """Get information about a specific movie"""
    track_request()
    
    if not recommender or not recommender.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    movie_info = recommender.get_movie_info(movie_id)
    if not movie_info:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")
    
    return MovieInfo(**movie_info)

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get movie recommendations"""
    start_time = datetime.now()
    track_request()
    
    if not recommender or not recommender.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Record request size
        metrics.recommendation_request_size.labels(endpoint="recommendations").observe(request.top_k)
        
        # Use metrics decorator for inference timing
        @metrics.record_inference_time("recommendations", 1)
        def get_recs():
            return recommender.get_recommendations(
                movie_id=request.movie_id,
                top_k=request.top_k,
                min_similarity=request.min_similarity,
                exclude_same_year=request.exclude_same_year
            )
        
        recommendations = get_recs()
        
        if not recommendations:
            metrics.record_recommendations_generated("recommendations", 0, "not_found")
            raise HTTPException(
                status_code=404, 
                detail=f"No recommendations found for movie {request.movie_id}"
            )
        
        # Record successful recommendations
        metrics.record_recommendations_generated("recommendations", len(recommendations), "success")
        
        # Convert to Pydantic models
        rec_models = [MovieInfo(**rec) for rec in recommendations]
        
        return RecommendationResponse(
            movie_id=request.movie_id,
            recommendations=rec_models,
            request_time=start_time.isoformat(),
            processing_time_ms=get_processing_time(start_time)
        )
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        metrics.record_recommendations_generated("recommendations", 0, "error")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommendations/{movie_id}", response_model=RecommendationResponse)
async def get_recommendations_get(
    movie_id: int = PathParam(..., description="Movie ID"),
    top_k: int = Query(10, ge=1, le=50, description="Number of recommendations"),
    min_similarity: float = Query(0.0, ge=0.0, le=1.0, description="Minimum similarity"),
    exclude_same_year: bool = Query(False, description="Exclude same year movies")
):
    """Get movie recommendations via GET request"""
    request_obj = RecommendationRequest(
        movie_id=movie_id,
        top_k=top_k,
        min_similarity=min_similarity,
        exclude_same_year=exclude_same_year
    )
    return await get_recommendations(request_obj)

@app.post("/recommendations/batch", response_model=BatchRecommendationResponse)
async def get_batch_recommendations(request: BatchRecommendationRequest):
    """Get recommendations for multiple movies"""
    start_time = datetime.now()
    track_request()
    
    if not recommender or not recommender.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.movie_ids) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 movies per batch request")
    
    try:
        batch_results = recommender.get_batch_recommendations(
            movie_ids=request.movie_ids,
            top_k=request.top_k,
            min_similarity=request.min_similarity
        )
        
        # Convert to Pydantic models
        formatted_results = {}
        for movie_id, recommendations in batch_results.items():
            formatted_results[movie_id] = [MovieInfo(**rec) for rec in recommendations]
        
        return BatchRecommendationResponse(
            results=formatted_results,
            request_time=start_time.isoformat(),
            processing_time_ms=get_processing_time(start_time)
        )
        
    except Exception as e:
        logger.error(f"Error getting batch recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse)
async def search_movies(request: SearchRequest):
    """Search for movies similar to text query"""
    start_time = datetime.now()
    track_request()
    
    if not recommender or not recommender.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Use metrics decorator for search timing
        @metrics.record_search_time()
        def search():
            return recommender.search_movies(
                query=request.query,
                top_k=request.top_k,
                search_type='hybrid'
            )
        
        results = search()
        
        # Record successful search
        metrics.record_search_query("success")
        
        # Convert to Pydantic models
        result_models = [MovieInfo(**result) for result in results]
        
        return SearchResponse(
            query=request.query,
            results=result_models,
            request_time=start_time.isoformat(),
            processing_time_ms=get_processing_time(start_time)
        )
        
    except Exception as e:
        logger.error(f"Error searching movies: {e}")
        metrics.record_search_query("error")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search", response_model=SearchResponse)
async def search_movies_get(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(10, ge=1, le=50, description="Number of results to return")
):
    """Search for movies using GET method (for convenience)"""
    # Convert to POST request format
    request = SearchRequest(query=query, top_k=top_k)
    return await search_movies(request)

@app.delete("/cache")
async def clear_cache():
    """Clear recommendation cache"""
    track_request()
    
    if not recommender or not recommender.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    recommender.clear_cache()
    return {"message": "Cache cleared successfully"}

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Run the API
def main():
    """Run the API server"""
    uvicorn.run(
        "src.api.api_service:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
        workers=settings.WORKERS
    )

if __name__ == "__main__":
    main()
