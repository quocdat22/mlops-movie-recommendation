import json
import logging
import os
import time
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def fetch_movie_details(api_key, movie_id, detail_type):
    """
    Lấy thông tin chi tiết (credits hoặc keywords) cho một bộ phim.
    """
    base_url = f"https://api.themoviedb.org/3/movie/{movie_id}/{detail_type}"
    params = {"api_key": api_key}
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Lỗi khi lấy {detail_type} cho movie_id {movie_id}: {e}")
        return None


def fetch_recent_movies(api_key, days_back=7):
    """
    Lấy danh sách các phim mới phát hành trong vài ngày qua.
    Sử dụng endpoint 'now_playing' hoặc 'discover' với filter ngày.

    Args:
        api_key (str): API key cho TMDB.
        days_back (int): Số ngày quay lại để tìm phim mới.

    Returns:
        list: Danh sách các phim mới hoặc None nếu có lỗi.
    """
    all_movies = []
    # base_url = "https://api.themoviedb.org/3/discover/movie"
    base_url = "https://api.themoviedb.org/3/movie/now_playing"
    # Tính toán ngày bắt đầu (days_back ngày trước)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    logging.info(f"Đang tìm phim mới từ {start_date} đến {end_date}...")

    for page in range(1, 6):  # Giới hạn 5 trang để tránh quá tải
        params = {
            "api_key": api_key,
            "language": "en-US",
            "page": page,
            "primary_release_date.gte": start_date,
            "primary_release_date.lte": end_date,
            "sort_by": "popularity.desc",
        }
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            movies = data.get("results", [])
            if not movies:
                logging.warning(f"Không tìm thấy phim mới nào ở trang {page}.")
                break

            # Lấy thêm credits và keywords cho mỗi phim
            for movie in movies:
                movie_id = movie["id"]
                time.sleep(0.1)  # Rate limiting

                credits = fetch_movie_details(api_key, movie_id, "credits")
                if credits and "cast" in credits:
                    movie["cast"] = [actor["name"] for actor in credits["cast"]][:3]
                else:
                    movie["cast"] = []

                keywords = fetch_movie_details(api_key, movie_id, "keywords")
                if keywords and "keywords" in keywords:
                    movie["keywords"] = [kw["name"] for kw in keywords["keywords"]]
                else:
                    movie["keywords"] = []

                logging.info(f"Đã xử lý chi tiết cho phim mới: {movie.get('title')}")

            all_movies.extend(movies)
            logging.info(f"Đã lấy thành công {len(movies)} phim từ trang {page}.")

        except requests.exceptions.RequestException as e:
            logging.error(f"Lỗi khi gọi API ở trang {page}: {e}")
            return None

    logging.info(f"Tổng cộng đã lấy được {len(all_movies)} phim mới.")
    return all_movies


def save_recent_movies_data(data, output_path):
    """
    Lưu dữ liệu phim mới vào file JSON.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"Đã lưu {len(data)} phim mới vào: {output_path}")
        return True
    except OSError as e:
        logging.error(f"Lỗi khi ghi file: {e}")
        return False


def main():
    """
    Hàm chính để Airflow gọi.
    """
    load_dotenv()

    tmdb_api_key = os.getenv("TMDB_API_KEY")
    if not tmdb_api_key:
        logging.error("TMDB_API_KEY không được thiết lập.")
        raise ValueError("TMDB_API_KEY is not set")

    # Lấy phim mới trong 1 ngày qua
    recent_movies = fetch_recent_movies(api_key=tmdb_api_key, days_back=1)

    if recent_movies:
        output_path = "data/raw/recent_movies.json"
        if save_recent_movies_data(recent_movies, output_path):
            logging.info(
                f"Task 1 hoàn thành: Đã lấy và lưu {len(recent_movies)} phim mới."
            )
        else:
            raise Exception("Không thể lưu dữ liệu phim mới.")
    else:
        logging.warning("Không có phim mới nào được tìm thấy.")


if __name__ == "__main__":
    main()
