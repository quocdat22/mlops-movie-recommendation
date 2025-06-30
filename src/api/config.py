"""
Configuration settings for Movie Recommendation API
"""
import os
from pathlib import Path
from typing import List

class APISettings:
    """API Configuration Settings"""
    
    # API Settings
    API_TITLE: str = "Movie Recommendation API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Content-based movie recommendation system using TF-IDF and cosine similarity"
    
    # Server Settings
    HOST: str = os.getenv("API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("API_PORT", "8000"))
    WORKERS: int = int(os.getenv("API_WORKERS", "1"))
    RELOAD: bool = os.getenv("API_RELOAD", "true").lower() == "true"
    
    # CORS Settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://localhost:8080",  # Vue dev server
        "*"  # Allow all origins in development
    ]
    
    # Model Settings
    MODEL_DIR: str = os.getenv("MODEL_DIR", "./models/content_based")
    MODEL_CACHE_SIZE: int = int(os.getenv("MODEL_CACHE_SIZE", "1000"))
    
    # Request Limits
    MAX_RECOMMENDATIONS: int = 50
    MAX_BATCH_SIZE: int = 20
    MAX_SEARCH_RESULTS: int = 100
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Database (if needed for logging/analytics)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    
    # Security
    API_KEY_HEADER: str = "X-API-Key"
    ALLOWED_API_KEYS: List[str] = os.getenv("API_KEYS", "").split(",") if os.getenv("API_KEYS") else []
    
    # Health Check
    HEALTH_CHECK_TIMEOUT: int = 30
    
    @classmethod
    def get_model_path(cls) -> Path:
        """Get absolute path to model directory"""
        model_dir = Path(cls.MODEL_DIR)
        if model_dir.is_absolute():
            return model_dir
        else:
            # If relative path, make it relative to project root
            current_dir = Path(__file__).parent.parent.parent  # Go to project root
            return (current_dir / cls.MODEL_DIR).resolve()

# Create settings instance
settings = APISettings()
