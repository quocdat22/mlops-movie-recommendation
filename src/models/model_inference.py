#!/usr/bin/env python3
"""
Content-Based Movie Recommendation Model Inference

This module provides inference capabilities for the trained content-based
recommendation model. It loads model artifacts and provides efficient
recommendation generation for API serving.

Features:
- Fast model artifact loading
- Efficient similarity-based recommendations
- Batch recommendation support
- Comprehensive error handling
- Performance optimization for production

Author: MLOps Movie Recommendation System
"""

import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from google.cloud import storage

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    """Content-based movie recommendation inference engine"""

    def __init__(self, model_dir: str):
        """
        Initialize recommender with model artifacts

        Args:
            model_dir: Directory containing trained model artifacts
        """
        self.model_dir = Path(model_dir)

        # Model components
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.movie_to_idx = None
        self.idx_to_movie = None
        self.movie_metadata = None
        self.raw_movies_data = None  # Store raw movie data for complete info

        # Performance tracking
        self._recommendation_cache = {}
        self.cache_size_limit = 1000  # Maximum cache size
        self.is_loaded = False

        self.gcs_bucket_name = os.getenv("GCS_BUCKET_NAME")
        self.gcs_model_path = os.getenv("GCS_MODEL_PATH", "content_based")

        # Load model on initialization
        self.load_model()

    def download_from_gcs(self):
        """Download model artifacts from Google Cloud Storage"""
        if not self.gcs_bucket_name or not self.gcs_model_path:
            logger.warning("GCS bucket name or model path not configured, skipping GCS download")
            return

        try:
            # Initialize GCS client
            client = storage.Client()
            bucket = client.bucket(self.gcs_bucket_name)
            
            # Create model directory if it doesn't exist
            self.model_dir.mkdir(parents=True, exist_ok=True)
            
            # Define the model files we need to download
            model_files = [
                'tfidf_vectorizer.pkl',
                'tfidf_matrix.pkl', 
                'similarity_matrix.npy',
                'movie_mappings.pkl',
                'movie_metadata.pkl'
            ]
            
            logger.info(f"Downloading model files from GCS bucket: {self.gcs_bucket_name}")
            
            for file_name in model_files:
                # GCS path
                gcs_path = f"{self.gcs_model_path}/{file_name}"
                # Local path
                local_path = self.model_dir / file_name
                
                # Skip if file already exists locally
                if local_path.exists():
                    logger.info(f"File {file_name} already exists locally, skipping download")
                    continue
                
                try:
                    blob = bucket.blob(gcs_path)
                    if blob.exists():
                        logger.info(f"Downloading {gcs_path} to {local_path}")
                        blob.download_to_filename(str(local_path))
                        logger.info(f"Successfully downloaded {file_name}")
                    else:
                        logger.warning(f"File not found in GCS: {gcs_path}")
                except Exception as e:
                    logger.error(f"Failed to download {file_name}: {e}")
                    raise e
                    
            logger.info("Model download from GCS completed successfully")
            
        except Exception as e:
            logger.error(f"Error downloading model from GCS: {e}")
            raise e

    def load_model(self) -> bool:
        """Load all model artifacts"""
        try:
            self.download_from_gcs()
        except Exception as e:
            logger.error(
                f"Cannot proceed with model loading due to GCS download failure: {e}"
            )
            self.is_loaded = False
            return False

        logger.info(f"Loading model from {self.model_dir}")

        try:
            # Load TF-IDF vectorizer
            vectorizer_path = self.model_dir / "tfidf_vectorizer.pkl"
            with open(vectorizer_path, "rb") as f:
                self.tfidf_vectorizer = pickle.load(f)

            # Load TF-IDF matrix
            tfidf_matrix_path = self.model_dir / "tfidf_matrix.pkl"
            with open(tfidf_matrix_path, "rb") as f:
                self.tfidf_matrix = pickle.load(f)

            # Load similarity matrix
            similarity_path = self.model_dir / "similarity_matrix.npy"
            self.similarity_matrix = np.load(similarity_path)

            # Load movie mappings
            mappings_path = self.model_dir / "movie_mappings.pkl"
            with open(mappings_path, "rb") as f:
                mappings = pickle.load(f)
                self.movie_to_idx = mappings["movie_to_idx"]
                self.idx_to_movie = mappings["idx_to_movie"]

            # Load movie metadata
            metadata_path = self.model_dir / "movie_metadata.pkl"
            self.movie_metadata = pd.read_pickle(metadata_path)

            # Load raw movies data for complete information
            self.raw_movies_data = {}
            raw_data_path = self.model_dir / "raw_movies_data.json"

            if not raw_data_path.exists():
                # Try to find raw data in the project data directory
                project_root = self.model_dir.parent.parent
                raw_data_files = [
                    project_root / "data" / "raw" / "top_rated_movies.json",
                    project_root / "data" / "raw" / "recent_movies.json",
                    project_root / "airflow" / "data" / "raw" / "recent_movies.json",
                ]

                for raw_file in raw_data_files:
                    if raw_file.exists():
                        try:
                            with open(raw_file, encoding="utf-8") as f:
                                raw_data = json.load(f)
                                # Index by movie id for fast lookup
                                for movie in raw_data:
                                    if "id" in movie:
                                        self.raw_movies_data[movie["id"]] = movie
                            logger.info(f"Loaded raw data from {raw_file}")
                        except Exception as e:
                            logger.warning(
                                f"Error loading raw data from {raw_file}: {e}"
                            )
            else:
                try:
                    with open(raw_data_path, encoding="utf-8") as f:
                        self.raw_movies_data = json.load(f)
                except Exception as e:
                    logger.warning(f"Error loading raw data: {e}")
                    self.raw_movies_data = {}

            # Enrich metadata with vote average/count from raw data
            if self.raw_movies_data and isinstance(self.raw_movies_data, dict):
                vote_data = [
                    {
                        "movie_id": int(mid),
                        "vote_average": data.get("vote_average", 0),
                        "vote_count": data.get("vote_count", 0),
                    }
                    for mid, data in self.raw_movies_data.items()
                ]
                if vote_data:
                    vote_df = pd.DataFrame(vote_data)
                    self.movie_metadata = pd.merge(
                        self.movie_metadata, vote_df, on="movie_id", how="left"
                    )
                    # Fill missing values for movies not in raw data
                    self.movie_metadata["vote_average"].fillna(0, inplace=True)
                    self.movie_metadata["vote_count"].fillna(0, inplace=True)

            self.is_loaded = True

            logger.info("Model loaded successfully:")
            logger.info(f"- Movies: {len(self.movie_to_idx):,}")
            logger.info(f"- Vocabulary: {len(self.tfidf_vectorizer.vocabulary_):,}")
            logger.info(f"- Matrix shape: {self.similarity_matrix.shape}")

            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_loaded = False
            return False

    def _check_model_loaded(self):
        """Check if model is loaded"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

    def get_movie_info(self, movie_id: int) -> Optional[Dict]:
        """Get movie information by ID"""
        self._check_model_loaded()

        # Start with metadata from model
        movie_info = self.movie_metadata[self.movie_metadata["movie_id"] == movie_id]
        if movie_info.empty:
            return None

        movie = movie_info.iloc[0]
        result = {
            "movie_id": int(movie["movie_id"]),
            "title": movie["title"],
            "release_date": "",
            "vote_average": 0.0,
            "genres": "",
        }

        # Enhance with raw data if available
        if self.raw_movies_data and movie_id in self.raw_movies_data:
            raw_movie = self.raw_movies_data[movie_id]
            result.update(
                {
                    "release_date": raw_movie.get("release_date", ""),
                    "vote_average": float(raw_movie.get("vote_average", 0)),
                    "genres": (
                        ", ".join([str(g) for g in raw_movie.get("genre_ids", [])])
                        if raw_movie.get("genre_ids")
                        else ""
                    ),
                    "overview": raw_movie.get("overview", ""),
                    "popularity": float(raw_movie.get("popularity", 0)),
                }
            )

        return result

    def get_recommendations(
        self,
        movie_id: int,
        top_k: int = 10,
        min_similarity: float = 0.0,
        exclude_same_year: bool = False,
        include_metadata: bool = True,
    ) -> List[Dict]:
        """
        Get recommendations for a movie

        Args:
            movie_id: ID of the movie to get recommendations for
            top_k: Number of recommendations to return
            min_similarity: Minimum similarity threshold
            exclude_same_year: Whether to exclude movies from the same year
            include_metadata: Whether to include full movie metadata

        Returns:
            List of recommended movies with similarity scores
        """
        self._check_model_loaded()

        # Check cache first
        cache_key = f"{movie_id}_{top_k}_{min_similarity}_{exclude_same_year}"
        if cache_key in self._recommendation_cache:
            return self._recommendation_cache[cache_key]

        # Check if movie exists
        if movie_id not in self.movie_to_idx:
            logger.warning(f"Movie ID {movie_id} not found in trained model")
            return []

        movie_idx = self.movie_to_idx[movie_id]

        # Get source movie info for filtering
        source_movie = None
        if exclude_same_year:
            source_movie = self.get_movie_info(movie_id)

        # Get similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[movie_idx]))

        # Filter and sort
        filtered_scores = []
        for idx, score in sim_scores:
            if idx == movie_idx:  # Skip self
                continue
            if score < min_similarity:  # Skip low similarity
                continue

            rec_movie_id = self.idx_to_movie[idx]

            # Apply year filtering if requested
            if exclude_same_year and source_movie:
                rec_movie = self.get_movie_info(rec_movie_id)
                if (
                    rec_movie
                    and source_movie.get("release_date")
                    and rec_movie.get("release_date")
                ):
                    source_year = source_movie["release_date"][:4]
                    rec_year = rec_movie["release_date"][:4]
                    if source_year == rec_year:
                        continue

            filtered_scores.append((idx, score, rec_movie_id))

        # Sort by similarity
        filtered_scores.sort(key=lambda x: x[1], reverse=True)

        # Build recommendations
        recommendations = []
        for idx, score, rec_movie_id in filtered_scores[:top_k]:
            if include_metadata:
                movie_info = self.get_movie_info(rec_movie_id)
                if movie_info:
                    movie_info["similarity"] = float(score)
                    recommendations.append(movie_info)
            else:
                recommendations.append(
                    {"movie_id": rec_movie_id, "similarity": float(score)}
                )

        # Cache result
        if len(self._recommendation_cache) < self.cache_size_limit:
            self._recommendation_cache[cache_key] = recommendations

        return recommendations

    def get_batch_recommendations(
        self, movie_ids: List[int], top_k: int = 10, **kwargs
    ) -> Dict[int, List[Dict]]:
        """Get recommendations for multiple movies"""
        self._check_model_loaded()

        results = {}
        for movie_id in movie_ids:
            try:
                recommendations = self.get_recommendations(movie_id, top_k, **kwargs)
                results[movie_id] = recommendations
            except Exception as e:
                logger.error(f"Error getting recommendations for movie {movie_id}: {e}")
                results[movie_id] = []

        return results

    def search_similar_movies(self, query_features: str, top_k: int = 10) -> List[Dict]:
        """
        Find movies similar to given text features

        Args:
            query_features: Text describing movie features
            top_k: Number of similar movies to return

        Returns:
            List of similar movies
        """
        self._check_model_loaded()

        try:
            # Transform query using fitted vectorizer
            query_vector = self.tfidf_vectorizer.transform([query_features])

            # Compute similarities with all movies
            similarities = np.array(
                query_vector.dot(self.tfidf_matrix.T).toarray()
            ).flatten()

            # Get top similar movies
            top_indices = similarities.argsort()[-top_k:][::-1]

            recommendations = []
            for idx in top_indices:
                if (
                    similarities[idx] > 0
                ):  # Only include movies with positive similarity
                    movie_id = self.idx_to_movie[idx]
                    movie_info = self.get_movie_info(movie_id)
                    if movie_info:
                        movie_info["similarity"] = float(similarities[idx])
                        recommendations.append(movie_info)

            return recommendations

        except Exception as e:
            logger.error(f"Error in text-based search: {e}")
            return []

    def search_by_title(self, title_query: str, top_k: int = 10) -> List[Dict]:
        """
        Search movies by title (fuzzy matching)

        Args:
            title_query: Movie title to search for
            top_k: Maximum number of results to return

        Returns:
            List of matching movies
        """
        self._check_model_loaded()

        try:
            title_query = title_query.lower().strip()
            matches = []

            # Search through all movies in metadata
            for movie_id in self.movie_to_idx.keys():
                movie_info = self.get_movie_info(movie_id)
                if movie_info and "title" in movie_info:
                    movie_title = movie_info["title"].lower()

                    # Calculate simple similarity score
                    if title_query in movie_title:
                        # Exact substring match gets higher score
                        if title_query == movie_title:
                            score = 1.0  # Exact match
                        elif movie_title.startswith(title_query):
                            score = 0.9  # Starts with query
                        else:
                            score = 0.7  # Contains query

                        movie_info["similarity"] = score
                        matches.append(movie_info)

            # Sort by similarity score and title length (shorter titles first for same score)
            matches.sort(key=lambda x: (-x["similarity"], len(x["title"])))

            return matches[:top_k]

        except Exception as e:
            logger.error(f"Error in title search: {e}")
            return []

    def search_movies(
        self, query: str, top_k: int = 10, search_type: str = "hybrid"
    ) -> List[Dict]:
        """
        Combined search function that tries both title and content search

        Args:
            query: Search query
            top_k: Maximum number of results
            search_type: 'title', 'content', or 'hybrid'

        Returns:
            List of matching movies
        """
        self._check_model_loaded()

        try:
            if search_type == "title":
                return self.search_by_title(query, top_k)
            elif search_type == "content":
                return self.search_similar_movies(query, top_k)
            else:  # hybrid
                # Try title search first
                title_results = self.search_by_title(query, top_k // 2)

                # If we have some title matches, also get content matches
                if title_results:
                    content_results = self.search_similar_movies(query, top_k // 2)

                    # Combine and deduplicate
                    all_results = title_results.copy()
                    seen_ids = {movie["movie_id"] for movie in title_results}

                    for movie in content_results:
                        if movie["movie_id"] not in seen_ids:
                            all_results.append(movie)

                    return all_results[:top_k]
                else:
                    # No title matches, try content search
                    return self.search_similar_movies(query, top_k)

        except Exception as e:
            logger.error(f"Error in combined search: {e}")
            return []

    def get_model_stats(self) -> Dict:
        """Get model statistics"""
        self._check_model_loaded()

        return {
            "total_movies": len(self.movie_to_idx),
            "vocabulary_size": len(self.tfidf_vectorizer.vocabulary_),
            "matrix_shape": self.similarity_matrix.shape,
            "model_dir": str(self.model_dir),
            "cache_size": len(self._recommendation_cache),
            "is_loaded": self.is_loaded,
        }

    def clear_cache(self):
        """Clear recommendation cache"""
        self._recommendation_cache.clear()
        logger.info("Recommendation cache cleared")

    def get_popular_movies(self, top_k: int = 20) -> List[Dict]:
        """Get popular movies based on vote average"""
        self._check_model_loaded()

        # Sort by vote average
        popular = self.movie_metadata.nlargest(top_k, "vote_average")

        results = []
        for _, movie in popular.iterrows():
            results.append(
                {
                    "movie_id": int(movie["movie_id"]),
                    "title": movie["title"],
                    "release_date": movie.get("release_date", ""),
                    "vote_average": float(movie.get("vote_average", 0)),
                    "genres": movie.get("genres", ""),
                }
            )

        return results

    def get_random_movies(self, count: int = 10) -> List[Dict]:
        """Get random movies for exploration"""
        self._check_model_loaded()

        sample = self.movie_metadata.sample(n=min(count, len(self.movie_metadata)))

        results = []
        for _, movie in sample.iterrows():
            results.append(
                {
                    "movie_id": int(movie["movie_id"]),
                    "title": movie["title"],
                    "release_date": movie.get("release_date", ""),
                    "vote_average": float(movie.get("vote_average", 0)),
                    "genres": movie.get("genres", ""),
                }
            )

        return results


def main():
    """Demo the recommendation system"""

    # Paths
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    model_dir = project_root / "models" / "content_based"

    # Check if model exists
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        print("Please run train_model.py first to train the model.")
        return

    # Initialize recommender
    recommender = ContentBasedRecommender(str(model_dir))

    # Load model
    if not recommender.load_model():
        print("Failed to load model")
        return

    # Get model stats
    stats = recommender.get_model_stats()
    print("\n" + "=" * 50)
    print("CONTENT-BASED MOVIE RECOMMENDER")
    print("=" * 50)
    print(f"Total Movies: {stats['total_movies']:,}")
    print(f"Vocabulary Size: {stats['vocabulary_size']:,}")
    print(f"Matrix Shape: {stats['matrix_shape']}")
    print("=" * 50)

    # Demo with popular movies
    print("\nPopular Movies:")
    popular = recommender.get_popular_movies(5)
    for i, movie in enumerate(popular, 1):
        print(
            f"{i}. {movie['title']} ({movie['release_date'][:4] if movie['release_date'] else 'N/A'}) - {movie['vote_average']:.1f}"
        )

    # Demo recommendations
    if popular:
        demo_movie = popular[0]
        print(f"\nRecommendations for '{demo_movie['title']}':")

        recommendations = recommender.get_recommendations(
            demo_movie["movie_id"], top_k=5
        )
        for i, rec in enumerate(recommendations, 1):
            print(
                f"{i}. {rec['title']} ({rec['release_date'][:4] if rec['release_date'] else 'N/A'}) - "
                f"Similarity: {rec['similarity']:.3f}"
            )

    # Demo text-based search
    print("\nSimilar movies to 'action superhero adventure':")
    text_recommendations = recommender.search_similar_movies(
        "action superhero adventure", top_k=5
    )
    for i, rec in enumerate(text_recommendations, 1):
        print(
            f"{i}. {rec['title']} ({rec['release_date'][:4] if rec['release_date'] else 'N/A'}) - "
            f"Similarity: {rec['similarity']:.3f}"
        )

    # Demo title search
    print("\nSearch results for title 'batman':")
    title_recommendations = recommender.search_by_title("batman", top_k=5)
    for i, rec in enumerate(title_recommendations, 1):
        print(
            f"{i}. {rec['title']} ({rec['release_date'][:4] if rec['release_date'] else 'N/A'}) - "
            f"Similarity: {rec['similarity']:.3f}"
        )

    # Demo combined search
    print("\nCombined search results for 'superhero':")
    combined_recommendations = recommender.search_movies(
        "superhero", top_k=5, search_type="hybrid"
    )
    for i, rec in enumerate(combined_recommendations, 1):
        print(
            f"{i}. {rec['title']} ({rec['release_date'][:4] if rec['release_date'] else 'N/A'}) - "
            f"Similarity: {rec['similarity']:.3f}"
        )

    print("\nRecommender ready for API integration!")


if __name__ == "__main__":
    main()
