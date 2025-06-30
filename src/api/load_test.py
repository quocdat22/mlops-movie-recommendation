#!/usr/bin/env python3
"""
Load Testing Script for Movie Recommendation API

This script generates load on the API to test monitoring capabilities.
It makes various types of requests to different endpoints.
"""

import requests
import time
import random
import threading
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict


class APILoadTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.results = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
        }

    def health_check(self) -> bool:
        """Check if API is available"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def get_popular_movies(self) -> List[Dict]:
        """Get list of popular movies for testing"""
        try:
            response = self.session.get(f"{self.base_url}/movies/popular?top_k=50")
            if response.status_code == 200:
                return response.json()
            return []
        except Exception:
            return []

    def make_recommendation_request(self, movie_id: int) -> Dict:
        """Make a recommendation request"""
        start_time = time.time()
        try:
            payload = {
                "movie_id": movie_id,
                "top_k": random.randint(5, 20),
                "min_similarity": round(random.uniform(0.0, 0.3), 2),
            }

            response = self.session.post(
                f"{self.base_url}/recommendations", json=payload, timeout=10
            )

            response_time = time.time() - start_time

            result = {
                "endpoint": "recommendations",
                "status_code": response.status_code,
                "response_time": response_time,
                "success": response.status_code == 200,
            }

            if response.status_code == 200:
                data = response.json()
                result["recommendations_count"] = len(data.get("recommendations", []))

            return result

        except Exception as e:
            return {
                "endpoint": "recommendations",
                "status_code": 0,
                "response_time": time.time() - start_time,
                "success": False,
                "error": str(e),
            }

    def make_search_request(self) -> Dict:
        """Make a search request"""
        search_queries = [
            "action adventure",
            "comedy romance",
            "sci-fi thriller",
            "drama mystery",
            "horror supernatural",
            "fantasy epic",
            "animated family",
            "documentary crime",
        ]

        start_time = time.time()
        try:
            payload = {
                "query": random.choice(search_queries),
                "top_k": random.randint(5, 15),
            }

            response = self.session.post(
                f"{self.base_url}/search", json=payload, timeout=10
            )

            response_time = time.time() - start_time

            result = {
                "endpoint": "search",
                "status_code": response.status_code,
                "response_time": response_time,
                "success": response.status_code == 200,
            }

            if response.status_code == 200:
                data = response.json()
                result["results_count"] = len(data.get("results", []))

            return result

        except Exception as e:
            return {
                "endpoint": "search",
                "status_code": 0,
                "response_time": time.time() - start_time,
                "success": False,
                "error": str(e),
            }

    def make_health_request(self) -> Dict:
        """Make a health check request"""
        start_time = time.time()
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response_time = time.time() - start_time

            return {
                "endpoint": "health",
                "status_code": response.status_code,
                "response_time": response_time,
                "success": response.status_code == 200,
            }

        except Exception as e:
            return {
                "endpoint": "health",
                "status_code": 0,
                "response_time": time.time() - start_time,
                "success": False,
                "error": str(e),
            }

    def worker_thread(self, movie_ids: List[int], duration: int) -> List[Dict]:
        """Worker thread that makes requests for a specified duration"""
        results = []
        start_time = time.time()

        while time.time() - start_time < duration:
            # Choose random endpoint
            endpoint_choice = random.choices(
                ["recommendation", "search", "health"],
                weights=[0.6, 0.3, 0.1],  # 60% recommendations, 30% search, 10% health
                k=1,
            )[0]

            if endpoint_choice == "recommendation" and movie_ids:
                movie_id = random.choice(movie_ids)
                result = self.make_recommendation_request(movie_id)
            elif endpoint_choice == "search":
                result = self.make_search_request()
            else:
                result = self.make_health_request()

            results.append(result)

            # Random delay between requests
            time.sleep(random.uniform(0.1, 1.0))

        return results

    def run_load_test(self, duration: int = 60, concurrent_users: int = 5):
        """Run load test for specified duration with concurrent users"""
        print(
            f"Starting load test for {duration} seconds with {concurrent_users} concurrent users..."
        )

        # Check API availability
        if not self.health_check():
            print("ERROR: API is not available. Please start the API first.")
            return

        # Get popular movies for testing
        print("Getting popular movies for testing...")
        popular_movies = self.get_popular_movies()
        if not popular_movies:
            print("WARNING: Could not get popular movies. Using random IDs.")
            movie_ids = list(range(1, 1000))
        else:
            movie_ids = [movie["movie_id"] for movie in popular_movies]
            print(f"Using {len(movie_ids)} popular movies for testing")

        # Start load test
        start_time = time.time()
        all_results = []

        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            # Submit worker threads
            futures = [
                executor.submit(self.worker_thread, movie_ids, duration)
                for _ in range(concurrent_users)
            ]

            # Collect results
            for future in as_completed(futures):
                try:
                    thread_results = future.result()
                    all_results.extend(thread_results)
                except Exception as e:
                    print(f"Thread error: {e}")

        # Calculate statistics
        total_time = time.time() - start_time
        self.print_results(all_results, total_time)

    def print_results(self, results: List[Dict], total_time: float):
        """Print test results"""
        if not results:
            print("No results to display")
            return

        # Basic stats
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r["success"])
        failed_requests = total_requests - successful_requests

        # Response time stats
        response_times = [r["response_time"] for r in results]
        avg_response_time = sum(response_times) / len(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)

        # Throughput
        throughput = total_requests / total_time

        # Error rate by status code
        status_codes = {}
        for result in results:
            code = result["status_code"]
            status_codes[code] = status_codes.get(code, 0) + 1

        # Endpoint breakdown
        endpoint_stats = {}
        for result in results:
            endpoint = result["endpoint"]
            if endpoint not in endpoint_stats:
                endpoint_stats[endpoint] = {"total": 0, "success": 0, "avg_time": 0}

            endpoint_stats[endpoint]["total"] += 1
            if result["success"]:
                endpoint_stats[endpoint]["success"] += 1
            endpoint_stats[endpoint]["avg_time"] += result["response_time"]

        # Calculate averages
        for endpoint in endpoint_stats:
            total = endpoint_stats[endpoint]["total"]
            endpoint_stats[endpoint]["avg_time"] /= total
            endpoint_stats[endpoint]["success_rate"] = (
                endpoint_stats[endpoint]["success"] / total
            )

        # Print results
        print("\n" + "=" * 50)
        print("LOAD TEST RESULTS")
        print("=" * 50)
        print(f"Total Duration: {total_time:.2f} seconds")
        print(f"Total Requests: {total_requests}")
        print(f"Successful Requests: {successful_requests}")
        print(f"Failed Requests: {failed_requests}")
        print(f"Success Rate: {(successful_requests/total_requests)*100:.2f}%")
        print(f"Throughput: {throughput:.2f} requests/second")
        print()

        print("Response Time Statistics:")
        print(f"  Average: {avg_response_time:.3f} seconds")
        print(f"  Minimum: {min_response_time:.3f} seconds")
        print(f"  Maximum: {max_response_time:.3f} seconds")
        print()

        print("Status Code Distribution:")
        for code, count in sorted(status_codes.items()):
            print(f"  {code}: {count} ({(count/total_requests)*100:.1f}%)")
        print()

        print("Endpoint Performance:")
        for endpoint, stats in endpoint_stats.items():
            print(f"  {endpoint}:")
            print(f"    Requests: {stats['total']}")
            print(f"    Success Rate: {stats['success_rate']*100:.1f}%")
            print(f"    Avg Response Time: {stats['avg_time']:.3f}s")
        print()


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Load test Movie Recommendation API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument(
        "--duration", type=int, default=60, help="Test duration in seconds"
    )
    parser.add_argument(
        "--users", type=int, default=5, help="Number of concurrent users"
    )

    args = parser.parse_args()

    tester = APILoadTester(args.url)
    tester.run_load_test(args.duration, args.users)


if __name__ == "__main__":
    main()
