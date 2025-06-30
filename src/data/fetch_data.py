import json
import logging
import os
import time

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


def fetch_top_rated_movies(api_key, output_path, pages=500):
    """
    Lấy danh sách các phim được đánh giá cao nhất từ TMDB API, bổ sung chi tiết và lưu sau mỗi trang.
    """
    all_movies = []
    base_url = "https://api.themoviedb.org/3/movie/top_rated"

    logging.info(f"Bắt đầu lấy dữ liệu từ TMDB cho {pages} trang...")

    for page in range(1, pages + 1):
        params = {"api_key": api_key, "language": "en-US", "page": page}
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            movies = data.get("results", [])
            if not movies:
                logging.warning(f"Không tìm thấy phim nào ở trang {page}.")
                break

            # Lấy thêm credits và keywords cho mỗi phim
            for movie in movies:
                movie_id = movie["id"]
                # Thêm một khoảng dừng nhỏ để tránh bị giới hạn API rate
                time.sleep(0.0001)  # 1ms

                credits = fetch_movie_details(api_key, movie_id, "credits")
                if credits and "cast" in credits:
                    movie["cast"] = [actor["name"] for actor in credits["cast"]][
                        :3
                    ]  # Lấy 3 diễn viên đầu
                else:
                    movie["cast"] = []

                keywords = fetch_movie_details(api_key, movie_id, "keywords")
                if keywords and "keywords" in keywords:
                    movie["keywords"] = [kw["name"] for kw in keywords["keywords"]]
                else:
                    movie["keywords"] = []

                logging.info(f"Đã xử lý chi tiết cho phim: {movie.get('title')}")

            all_movies.extend(movies)
            # Lưu dữ liệu vào file sau mỗi trang
            save_data_to_raw_file(all_movies, output_path)
            logging.info(
                f"Đã lấy, xử lý và lưu thành công dữ liệu từ trang {page}/{pages}."
            )

        except requests.exceptions.RequestException as e:
            logging.error(f"Lỗi khi gọi API ở trang {page}: {e}")
            return None

    logging.info(f"Tổng cộng đã lấy và xử lý được {len(all_movies)} phim.")
    return all_movies


def save_data_to_raw_file(data, output_path):
    """
    Lưu dữ liệu vào file JSON.

    Args:
        data (list or dict): Dữ liệu cần lưu.
        output_path (str): Đường dẫn đến file output.
    """
    try:
        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"Đã lưu dữ liệu thành công vào: {output_path}")
    except OSError as e:
        logging.error(f"Lỗi khi ghi file: {e}")


if __name__ == "__main__":
    # Tải các biến môi trường từ file .env
    load_dotenv()

    # Lấy API key từ biến môi trường
    tmdb_api_key = os.getenv("TMDB_API_KEY")

    if not tmdb_api_key:
        logging.error(
            "Biến môi trường TMDB_API_KEY không được thiết lập. Vui lòng thêm vào file .env."
        )
    else:
        # Định nghĩa đường dẫn file output
        raw_data_path = "data/raw/top_rated_movies.json"

        # Lấy và lưu dữ liệu phim
        fetch_top_rated_movies(
            api_key=tmdb_api_key, output_path=raw_data_path, pages=499
        )
