import sys
from pathlib import Path

# Ensure project root is in Python path so `import src` works when tests are
# executed from any directory.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import pytest
from fastapi.testclient import TestClient

from src.api import api_service


class DummyRecommender:
    """Lightweight dummy recommender used for fast unit-tests."""

    def __init__(self):
        self.is_loaded = True

    # ---------------------- Model meta --------------------- #
    def get_model_stats(self):
        return {
            "total_movies": 10,
            "vocabulary_size": 100,
            "matrix_shape": (10, 10),
            "is_loaded": True,
            "cache_size": 0,
        }

    # ---------------------- Movies ------------------------- #
    def get_popular_movies(self, top_k: int = 20):
        return [
            {
                "movie_id": i,
                "title": f"Movie {i}",
                "release_date": "2020-01-01",
                "vote_average": 8.0,
                "genres": "Drama",
                "similarity": None,
            }
            for i in range(top_k)
        ]

    def get_random_movies(self, count: int = 10):
        return [
            {
                "movie_id": i + 1000,
                "title": f"Random {i}",
                "release_date": "2021-01-01",
                "vote_average": 7.0,
                "genres": "Comedy",
                "similarity": None,
            }
            for i in range(count)
        ]

    def get_movie_info(self, movie_id: int):
        return {
            "movie_id": movie_id,
            "title": "Movie",
            "release_date": "2020-01-01",
            "vote_average": 8.0,
            "genres": "Drama",
            "similarity": None,
        }

    # ---------------------- Recommendations --------------- #
    def get_recommendations(
        self,
        movie_id: int,
        top_k: int = 10,
        min_similarity: float = 0.0,
        exclude_same_year: bool = False,
        **_,
    ):
        return [
            {
                "movie_id": movie_id + i + 1,
                "title": f"Rec {i}",
                "release_date": "2020-01-01",
                "vote_average": 7.5,
                "genres": "Action",
                "similarity": 0.1 * (i + 1),
            }
            for i in range(top_k)
        ]

    # ---------------------- Search ------------------------ #
    def search_movies(self, query: str, top_k: int = 10, **_):
        return [
            {
                "movie_id": i + 2000,
                "title": f"{query} {i}",
                "release_date": "2022-01-01",
                "vote_average": 6.5,
                "genres": "Action",
                "similarity": None,
            }
            for i in range(top_k)
        ]

    # ---------------------- Cache ------------------------- #
    def clear_cache(self):
        pass


@pytest.fixture(scope="session")
def test_client():
    """Provide FastAPI TestClient with dummy recommender injected."""

    # Inject dummy recommender before tests start
    api_service.recommender = DummyRecommender()
    client = TestClient(api_service.app)
    return client
