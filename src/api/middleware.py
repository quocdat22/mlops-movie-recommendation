"""
Middleware for Movie Recommendation API
"""

import logging
import time
from datetime import datetime

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .config import settings

logger = logging.getLogger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication (optional)"""

    def __init__(self, app, exclude_paths: list = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]

    async def dispatch(self, request: Request, call_next):
        # Skip API key check for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Skip if no API keys configured
        if not settings.ALLOWED_API_KEYS or not settings.ALLOWED_API_KEYS[0]:
            return await call_next(request)

        # Check API key
        api_key = request.headers.get(settings.API_KEY_HEADER)
        if not api_key or api_key not in settings.ALLOWED_API_KEYS:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing API key"},
            )

        return await call_next(request)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")

        # Process request
        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            # Log response
            logger.info(
                f"Response: {response.status_code} - "
                f"Time: {process_time:.3f}s - "
                f"Path: {request.url.path}"
            )

            # Add timing header
            response.headers["X-Process-Time"] = str(process_time)

            return response

        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"Error: {str(e)} - Time: {process_time:.3f}s")
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware"""

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.client_requests = {}

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = datetime.now()

        # Clean old entries (older than 1 minute)
        cutoff_time = current_time.timestamp() - 60
        if client_ip in self.client_requests:
            self.client_requests[client_ip] = [
                req_time
                for req_time in self.client_requests[client_ip]
                if req_time > cutoff_time
            ]

        # Check rate limit
        if client_ip not in self.client_requests:
            self.client_requests[client_ip] = []

        if len(self.client_requests[client_ip]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded"},
            )

        # Add current request
        self.client_requests[client_ip].append(current_time.timestamp())

        return await call_next(request)
