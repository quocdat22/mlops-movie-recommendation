import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        """Initialize database connection using environment variables."""
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL not found in environment variables")
        
        self.connection = None
        self.cursor = None
    
    def connect(self):
        """Establish connection to the database."""
        try:
            self.connection = psycopg2.connect(self.database_url)
            self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            logger.info("Successfully connected to the database")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("Database connection closed")
    
    def create_movies_table(self):
        """Create movies table if it doesn't exist."""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS movies (
            id INTEGER PRIMARY KEY,
            adult BOOLEAN,
            backdrop_path TEXT,
            genre_ids INTEGER[],
            original_language VARCHAR(10),
            original_title TEXT,
            overview TEXT,
            popularity DECIMAL,
            poster_path TEXT,
            release_date DATE,
            title TEXT NOT NULL,
            video BOOLEAN,
            vote_average DECIMAL,
            vote_count INTEGER,
            movie_cast TEXT[],
            keywords TEXT[],
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        try:
            self.cursor.execute(create_table_query)
            self.connection.commit()
            logger.info("Movies table created or already exists")
        except Exception as e:
            logger.error(f"Error creating movies table: {e}")
            self.connection.rollback()
            raise
    
    def check_movie_exists(self, movie_id: int) -> bool:
        """Check if a movie with given ID already exists."""
        try:
            self.cursor.execute("SELECT 1 FROM movies WHERE id = %s", (movie_id,))
            return self.cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking if movie exists: {e}")
            return False
    
    def insert_movie(self, movie_data: Dict[str, Any]) -> bool:
        """Insert a single movie into the database."""
        insert_query = """
        INSERT INTO movies (
            id, adult, backdrop_path, genre_ids, original_language, 
            original_title, overview, popularity, poster_path, 
            release_date, title, video, vote_average, vote_count, 
            movie_cast, keywords
        ) VALUES (
            %(id)s, %(adult)s, %(backdrop_path)s, %(genre_ids)s, 
            %(original_language)s, %(original_title)s, %(overview)s, 
            %(popularity)s, %(poster_path)s, %(release_date)s, 
            %(title)s, %(video)s, %(vote_average)s, %(vote_count)s, 
            %(movie_cast)s, %(keywords)s
        )
        ON CONFLICT (id) DO UPDATE SET
            adult = EXCLUDED.adult,
            backdrop_path = EXCLUDED.backdrop_path,
            genre_ids = EXCLUDED.genre_ids,
            original_language = EXCLUDED.original_language,
            original_title = EXCLUDED.original_title,
            overview = EXCLUDED.overview,
            popularity = EXCLUDED.popularity,
            poster_path = EXCLUDED.poster_path,
            release_date = EXCLUDED.release_date,
            title = EXCLUDED.title,
            video = EXCLUDED.video,
            vote_average = EXCLUDED.vote_average,
            vote_count = EXCLUDED.vote_count,
            movie_cast = EXCLUDED.movie_cast,
            keywords = EXCLUDED.keywords,
            updated_at = CURRENT_TIMESTAMP
        """
        
        try:
            # Handle None values for release_date
            if movie_data.get('release_date') == '':
                movie_data['release_date'] = None
                
            self.cursor.execute(insert_query, movie_data)
            return True
        except Exception as e:
            logger.error(f"Error inserting movie {movie_data.get('id', 'unknown')}: {e}")
            return False
    
    def batch_insert_movies(self, movies_data: List[Dict[str, Any]], batch_size: int = 100):
        """Insert movies in batches for better performance."""
        total_movies = len(movies_data)
        inserted_count = 0
        skipped_count = 0
        error_count = 0
        
        logger.info(f"Starting to insert {total_movies} movies in batches of {batch_size}")
        
        for i in range(0, total_movies, batch_size):
            batch = movies_data[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_movies + batch_size - 1)//batch_size}")
            
            try:
                for movie in batch:
                    if self.insert_movie(movie):
                        inserted_count += 1
                    else:
                        error_count += 1
                
                self.connection.commit()
                logger.info(f"Committed batch {i//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                self.connection.rollback()
                error_count += len(batch)
        
        logger.info(f"Insertion completed. Inserted: {inserted_count}, Errors: {error_count}")
        return inserted_count, error_count
    
    def load_movies_from_json(self, json_file_path: str):
        """Load movies from JSON file and insert into database."""
        try:
            # Check if file exists
            if not os.path.exists(json_file_path):
                raise FileNotFoundError(f"JSON file not found: {json_file_path}")
            
            # Load JSON data
            with open(json_file_path, 'r', encoding='utf-8') as file:
                movies_data = json.load(file)
            
            logger.info(f"Loaded {len(movies_data)} movies from {json_file_path}")
            
            # Connect to database
            self.connect()
            
            # Create table if not exists
            self.create_movies_table()
            
            # Insert movies
            inserted, errors = self.batch_insert_movies(movies_data)
            
            logger.info(f"Successfully inserted {inserted} movies with {errors} errors")
            
        except Exception as e:
            logger.error(f"Error loading movies from JSON: {e}")
            raise
        finally:
            self.disconnect()
    
    def get_movies_count(self) -> int:
        """Get total number of movies in the database."""
        try:
            self.connect()
            self.cursor.execute("SELECT COUNT(*) FROM movies")
            count = self.cursor.fetchone()['count']
            return count
        except Exception as e:
            logger.error(f"Error getting movies count: {e}")
            return 0
        finally:
            self.disconnect()
    
    def get_movies(self, limit: int = 10, offset: int = 0) -> List[Dict]:
        """Get movies from database with pagination."""
        try:
            self.connect()
            query = "SELECT * FROM movies ORDER BY vote_average DESC LIMIT %s OFFSET %s"
            self.cursor.execute(query, (limit, offset))
            movies = self.cursor.fetchall()
            return [dict(movie) for movie in movies]
        except Exception as e:
            logger.error(f"Error getting movies: {e}")
            return []
        finally:
            self.disconnect()


def main():
    """Main function to load movies data."""
    # Path to the JSON file
    json_file_path = "/home/hoang/Projects/web/movie/mlops-movie-recommendation/data/raw/top_rated_movies.json"
    
    # Create database manager instance
    db_manager = DatabaseManager()
    
    try:
        # Load movies from JSON file
        db_manager.load_movies_from_json(json_file_path)
        
        # Get count of movies in database
        count = db_manager.get_movies_count()
        logger.info(f"Total movies in database: {count}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()