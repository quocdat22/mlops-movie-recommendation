import pytest


@pytest.mark.parametrize(
    "endpoint",
    [
        "/health",
        "/model/stats",
    ],
)
def test_get_endpoints_success(test_client, endpoint):
    response = test_client.get(endpoint)
    assert response.status_code == 200


def test_popular_movies(test_client):
    resp = test_client.get("/movies/popular?top_k=5")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 5
    assert data[0]["title"].startswith("Movie")


def test_random_movies(test_client):
    resp = test_client.get("/movies/random?count=3")
    assert resp.status_code == 200
    assert len(resp.json()) == 3


def test_recommendations_post(test_client):
    payload = {"movie_id": 1, "top_k": 3}
    resp = test_client.post("/recommendations", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["movie_id"] == 1
    assert len(data["recommendations"]) == 3


def test_search_movies_get(test_client):
    resp = test_client.get("/search?query=hero&top_k=4")
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 4
