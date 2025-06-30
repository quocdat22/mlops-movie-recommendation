#!/usr/bin/env python3
"""
API Client Example for Movie Recommendation API

This script demonstrates how to interact with the Movie Recommendation API.
It includes examples for all available endpoints.
"""

import requests
import json
from typing import Dict, List


class MovieRecommendationAPIClient:
    """Client for Movie Recommendation API"""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

        if api_key:
            self.session.headers.update({"X-API-Key": api_key})

    def health_check(self) -> Dict:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def get_model_stats(self) -> Dict:
        """Get model statistics"""
        response = self.session.get(f"{self.base_url}/model/stats")
        response.raise_for_status()
        return response.json()

    def get_movie_info(self, movie_id: int) -> Dict:
        """Get information about a specific movie"""
        response = self.session.get(f"{self.base_url}/movies/{movie_id}")
        response.raise_for_status()
        return response.json()

    def get_recommendations(
        self,
        movie_id: int,
        top_k: int = 10,
        min_similarity: float = 0.0,
        exclude_same_year: bool = False,
    ) -> Dict:
        """Get movie recommendations"""
        payload = {
            "movie_id": movie_id,
            "top_k": top_k,
            "min_similarity": min_similarity,
            "exclude_same_year": exclude_same_year,
        }
        response = self.session.post(f"{self.base_url}/recommendations", json=payload)
        response.raise_for_status()
        return response.json()

    def get_batch_recommendations(
        self, movie_ids: List[int], top_k: int = 10, min_similarity: float = 0.0
    ) -> Dict:
        """Get recommendations for multiple movies"""
        payload = {
            "movie_ids": movie_ids,
            "top_k": top_k,
            "min_similarity": min_similarity,
        }
        response = self.session.post(
            f"{self.base_url}/recommendations/batch", json=payload
        )
        response.raise_for_status()
        return response.json()

    def search_movies(self, query: str, top_k: int = 10) -> Dict:
        """Search for movies similar to text query"""
        payload = {"query": query, "top_k": top_k}
        response = self.session.post(f"{self.base_url}/search", json=payload)
        response.raise_for_status()
        return response.json()

    def get_popular_movies(self, top_k: int = 20) -> List[Dict]:
        """Get popular movies"""
        response = self.session.get(f"{self.base_url}/movies/popular?top_k={top_k}")
        response.raise_for_status()
        return response.json()

    def get_random_movies(self, count: int = 10) -> List[Dict]:
        """Get random movies"""
        response = self.session.get(f"{self.base_url}/movies/random?count={count}")
        response.raise_for_status()
        return response.json()


def main():
    """Example usage of the API client"""
    # Initialize client
    client = MovieRecommendationAPIClient()

    try:
        # Health check
        print("=== Health Check ===")
        health = client.health_check()
        print(json.dumps(health, indent=2))

        if not health.get("model_loaded"):
            print(
                "ERROR: Model not loaded. Please start the API server with a trained model."
            )
            return

        # Model stats
        print("\n=== Model Statistics ===")
        stats = client.get_model_stats()
        print(json.dumps(stats, indent=2))

        # Get popular movies to find a movie ID for testing
        print("\n=== Popular Movies ===")
        popular = client.get_popular_movies(top_k=5)
        print(f"Found {len(popular)} popular movies")

        if popular:
            test_movie = popular[0]
            movie_id = test_movie["movie_id"]
            print(f"Using movie: {test_movie['title']} (ID: {movie_id})")

            # Get movie info
            print(f"\n=== Movie Info for {movie_id} ===")
            movie_info = client.get_movie_info(movie_id)
            print(json.dumps(movie_info, indent=2))

            # Get recommendations
            print(f"\n=== Recommendations for {movie_id} ===")
            recommendations = client.get_recommendations(movie_id, top_k=5)
            print(f"Processing time: {recommendations['processing_time_ms']:.2f}ms")
            for i, rec in enumerate(recommendations["recommendations"][:3], 1):
                print(
                    f"{i}. {rec['title']} (Similarity: {rec.get('similarity', 'N/A')})"
                )

        # Search movies
        print("\n=== Movie Search ===")
        search_results = client.search_movies("action adventure superhero", top_k=3)
        print(f"Search processing time: {search_results['processing_time_ms']:.2f}ms")
        for i, result in enumerate(search_results["results"], 1):
            print(f"{i}. {result['title']} - {result['genres']}")

        # Random movies
        print("\n=== Random Movies ===")
        random_movies = client.get_random_movies(count=3)
        for i, movie in enumerate(random_movies, 1):
            print(f"{i}. {movie['title']}")

    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API server. Is it running?")
        print("Start the server with: ./start_api.sh")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
