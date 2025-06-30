#!/usr/bin/env python3
"""
Test script to verify database connection and basic operations.
"""

import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

from src.data.database_manager import DatabaseManager
import logging

def test_database_connection():
    """Test database connection and basic operations."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create database manager instance
        db_manager = DatabaseManager()
        
        # Test connection
        logger.info("Testing database connection...")
        db_manager.connect()
        logger.info("✓ Database connection successful!")
        
        # Test table creation
        logger.info("Testing table creation...")
        db_manager.create_movies_table()
        logger.info("✓ Movies table created or already exists!")
        
        # Test getting movies count
        logger.info("Testing movies count query...")
        db_manager.disconnect()  # Disconnect before using get_movies_count (it handles its own connection)
        count = db_manager.get_movies_count()
        logger.info(f"✓ Current movies count: {count}")
        
        # Test getting sample movies
        logger.info("Testing sample movies query...")
        movies = db_manager.get_movies(limit=5)
        logger.info(f"✓ Retrieved {len(movies)} sample movies")
        
        if movies:
            logger.info("Sample movie data:")
            for movie in movies[:2]:  # Show first 2 movies
                logger.info(f"  - {movie.get('title', 'Unknown')} (ID: {movie.get('id', 'Unknown')})")
        
        logger.info("✓ All database tests passed successfully!")
        # Use assertion so pytest receives None (pass) and avoids return-value warning
        assert True
        
    except Exception as e:
        logger.error(f"✗ Database test failed: {e}")
        # Fail the test by re-raising
        raise

if __name__ == "__main__":
    test_database_connection()
    sys.exit(0)
