#!/usr/bin/env python3
"""
Script to load movie data from JSON file to Supabase database.
"""

import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.database_manager import DatabaseManager
import logging

def main():
    """Main function to execute data loading."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Path to the JSON file
    json_file_path = "data/raw/top_rated_movies.json"
    
    # Check if file exists
    if not os.path.exists(json_file_path):
        logger.error(f"JSON file not found: {json_file_path}")
        return 1
    
    try:
        # Create database manager instance
        db_manager = DatabaseManager()
        
        # Load movies from JSON file
        logger.info("Starting to load movies data to Supabase...")
        db_manager.load_movies_from_json(json_file_path)
        
        # Get count of movies in database
        count = db_manager.get_movies_count()
        logger.info(f"Successfully completed! Total movies in database: {count}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
