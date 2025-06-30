#!/usr/bin/env python3
"""
Custom Prometheus Metrics for Movie Recommendation API

This module defines custom metrics specific to the movie recommendation system
that will be collected by Prometheus for monitoring model performance.
"""

from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps
from typing import Callable, Any

# Custom metrics for movie recommendation system
class RecommendationMetrics:
    """Custom Prometheus metrics for recommendation system"""
    
    def __init__(self):
        # Model inference metrics
        self.model_inference_duration = Histogram(
            'recommendation_model_inference_duration_seconds',
            'Time spent on model inference',
            ['endpoint', 'movie_count'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        # Recommendation metrics
        self.recommendations_generated_total = Counter(
            'recommendations_generated_total',
            'Total number of movie recommendations generated',
            ['endpoint', 'status']
        )
        
        self.recommendation_request_size = Histogram(
            'recommendation_request_size',
            'Number of recommendations requested',
            ['endpoint'],
            buckets=[1, 5, 10, 20, 50, 100]
        )
        
        # Cache metrics
        self.cache_hits_total = Counter(
            'recommendation_cache_hits_total',
            'Total number of cache hits'
        )
        
        self.cache_misses_total = Counter(
            'recommendation_cache_misses_total',
            'Total number of cache misses'
        )
        
        self.cache_size = Gauge(
            'recommendation_cache_size',
            'Current number of items in cache'
        )
        
        # Model health metrics
        self.model_loaded = Gauge(
            'recommendation_model_loaded',
            'Whether the recommendation model is loaded (1) or not (0)'
        )
        
        self.model_total_movies = Gauge(
            'recommendation_model_total_movies',
            'Total number of movies in the model'
        )
        
        self.model_vocabulary_size = Gauge(
            'recommendation_model_vocabulary_size',
            'Size of the TF-IDF vocabulary'
        )
        
        # Search metrics
        self.search_queries_total = Counter(
            'movie_search_queries_total',
            'Total number of movie search queries',
            ['status']
        )
        
        self.search_duration = Histogram(
            'movie_search_duration_seconds',
            'Time spent on movie search',
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )
        
        # Popular movies metrics
        self.popular_movies_request_duration = Histogram(
            'popular_movies_request_duration_seconds',
            'Time spent fetching popular movies',
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )
        
        # Error metrics
        self.model_errors_total = Counter(
            'recommendation_model_errors_total',
            'Total number of model errors',
            ['error_type', 'endpoint']
        )
    
    def record_inference_time(self, endpoint: str, movie_count: int = 1):
        """Decorator to record model inference time"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    # Record successful inference
                    duration = time.time() - start_time
                    self.model_inference_duration.labels(
                        endpoint=endpoint,
                        movie_count=str(movie_count)
                    ).observe(duration)
                    return result
                except Exception as e:
                    # Record error
                    self.model_errors_total.labels(
                        error_type=type(e).__name__,
                        endpoint=endpoint
                    ).inc()
                    raise
            return wrapper
        return decorator
    
    def record_recommendations_generated(self, endpoint: str, count: int, status: str = "success"):
        """Record number of recommendations generated"""
        self.recommendations_generated_total.labels(
            endpoint=endpoint,
            status=status
        ).inc(count)
    
    def record_cache_hit(self):
        """Record a cache hit"""
        self.cache_hits_total.inc()
    
    def record_cache_miss(self):
        """Record a cache miss"""
        self.cache_misses_total.inc()
    
    def update_cache_size(self, size: int):
        """Update current cache size"""
        self.cache_size.set(size)
    
    def update_model_status(self, is_loaded: bool, total_movies: int = 0, vocab_size: int = 0):
        """Update model status metrics"""
        self.model_loaded.set(1 if is_loaded else 0)
        if is_loaded:
            self.model_total_movies.set(total_movies)
            self.model_vocabulary_size.set(vocab_size)
    
    def record_search_query(self, status: str = "success"):
        """Record a search query"""
        self.search_queries_total.labels(status=status).inc()
    
    def record_search_time(self):
        """Decorator to record search time"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                self.search_duration.observe(duration)
                return result
            return wrapper
        return decorator
    
    def record_popular_movies_request(self, duration_ms: float):
        """Record duration of popular movies request in milliseconds"""
        # Convert milliseconds to seconds for Prometheus Histogram
        self.popular_movies_request_duration.observe(duration_ms / 1000.0)
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        try:
            # Get values safely from counters
            hits_samples = list(self.cache_hits_total.collect()[0].samples)
            misses_samples = list(self.cache_misses_total.collect()[0].samples)
            
            hits = hits_samples[0].value if hits_samples else 0
            misses = misses_samples[0].value if misses_samples else 0
            
            total = hits + misses
            return hits / total if total > 0 else 0.0
        except Exception:
            return 0.0

# Global metrics instance
metrics = RecommendationMetrics()
