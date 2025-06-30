import os
import pandas as pd
from sqlalchemy import create_engine, inspect, text, types
from dotenv import load_dotenv
import logging
import json

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_db_engine():
    """Tạo và trả về một SQLAlchemy engine kết nối tới database."""
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logging.error("DATABASE_URL không được thiết lập.")
        raise ValueError("DATABASE_URL is not set")

    try:
        engine = create_engine(database_url)
        logging.info("Kết nối database thành công.")
        return engine
    except Exception as e:
        logging.error(f"Lỗi khi kết nối database: {e}")
        raise


def ensure_movies_table_exists(engine):
    """Đảm bảo bảng 'movies' tồn tại với cấu trúc đầy đủ."""
    inspector = inspect(engine)
    if not inspector.has_table("movies"):
        logging.info("Bảng 'movies' không tồn tại. Đang tạo bảng mới...")
        # Sử dụng cấu trúc bảng giống như trong database_manager.py
        create_table_query = text(
            """
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
        )
        try:
            with engine.connect() as connection:
                with connection.begin():
                    connection.execute(create_table_query)
            logging.info("Đã tạo bảng 'movies' thành công.")
        except Exception as e:
            logging.error(f"Lỗi khi tạo bảng: {e}")
            raise
    else:
        logging.info("Bảng 'movies' đã tồn tại.")


def process_and_insert_recent_movies(engine, json_file_path):
    """
    Đọc file JSON chứa phim mới và chèn vào database.
    Chỉ chèn những phim chưa có trong database.
    """
    try:
        # Kiểm tra file tồn tại
        if not os.path.exists(json_file_path):
            logging.warning(f"File không tồn tại: {json_file_path}")
            return 0

        df = pd.read_json(json_file_path)
        logging.info(f"Đã đọc thành công {len(df)} records từ {json_file_path}")

        if df.empty:
            logging.info("Không có dữ liệu phim mới để xử lý.")
            return 0

        # Giữ lại tất cả các cột cần thiết như trong database_manager.py
        columns_to_keep = [
            "id",
            "adult",
            "backdrop_path",
            "genre_ids",
            "original_language",
            "original_title",
            "overview",
            "popularity",
            "poster_path",
            "release_date",
            "title",
            "video",
            "vote_average",
            "vote_count",
            "movie_cast",
            "keywords",
        ]

        # Kiểm tra các cột có tồn tại không
        available_columns = [col for col in columns_to_keep if col in df.columns]
        df = df[available_columns]

        # Xử lý dữ liệu để phù hợp với PostgreSQL
        # Chuyển đổi list thành mảng PostgreSQL TEXT[] cho movie_cast và keywords
        if "movie_cast" in df.columns:
            df["movie_cast"] = df["movie_cast"].apply(
                lambda x: x if isinstance(x, list) else []
            )
        if "keywords" in df.columns:
            df["keywords"] = df["keywords"].apply(
                lambda x: x if isinstance(x, list) else []
            )

        # Xử lý release_date - đặt None cho chuỗi rỗng
        if "release_date" in df.columns:
            df["release_date"] = df["release_date"].replace("", None)

        # Lấy danh sách các ID phim đã có trong database
        with engine.connect() as connection:
            try:
                existing_ids_df = pd.read_sql("SELECT id FROM movies", connection)
                existing_ids = (
                    existing_ids_df["id"].tolist() if not existing_ids_df.empty else []
                )
            except Exception:
                # Bảng có thể trống hoặc chưa có dữ liệu
                existing_ids = []

        # Lọc ra những phim chưa có trong database
        new_movies_df = df[~df["id"].isin(existing_ids)]

        if not new_movies_df.empty:
            logging.info(
                f"Tìm thấy {len(new_movies_df)} phim mới cần chèn vào database."
            )

            # Chèn dữ liệu mới vào bảng - pandas sẽ tự động xử lý PostgreSQL arrays
            new_movies_df.to_sql("movies", engine, if_exists="append", index=False)
            logging.info(
                f"Đã chèn thành công {len(new_movies_df)} phim mới vào database."
            )
            return len(new_movies_df)
        else:
            logging.info(
                "Không có phim mới nào để chèn (tất cả đã tồn tại trong database)."
            )
            return 0

    except Exception as e:
        logging.error(f"Lỗi trong quá trình xử lý và chèn dữ liệu: {e}")
        raise


def cleanup_recent_movies_file(json_file_path):
    """
    Xóa file dữ liệu tạm sau khi xử lý xong để tránh tích tụ file không cần thiết.
    """
    try:
        if os.path.exists(json_file_path):
            os.remove(json_file_path)
            logging.info(f"Đã xóa file tạm: {json_file_path}")
    except Exception as e:
        logging.warning(f"Không thể xóa file tạm {json_file_path}: {e}")


def get_total_movies_count(engine):
    """Lấy tổng số phim trong database."""
    try:
        with engine.connect() as connection:
            result = pd.read_sql("SELECT COUNT(*) as total FROM movies", connection)
            return result.iloc[0]["total"]
    except Exception as e:
        logging.error(f"Lỗi khi đếm phim: {e}")
        return 0


def main():
    """
    Hàm chính để Airflow gọi.
    """
    load_dotenv()

    recent_movies_file = "data/raw/recent_movies.json"

    try:
        # Kết nối database
        db_engine = get_db_engine()

        # Đảm bảo bảng tồn tại
        ensure_movies_table_exists(db_engine)

        # Xử lý và chèn phim mới
        new_movies_added = process_and_insert_recent_movies(
            db_engine, recent_movies_file
        )

        # Lấy tổng số phim trong database
        total_movies = get_total_movies_count(db_engine)

        # Dọn dẹp file tạm
        cleanup_recent_movies_file(recent_movies_file)

        logging.info(
            f"Task 2 hoàn thành: Đã thêm {new_movies_added} phim mới. Tổng số phim trong database: {total_movies}"
        )

    except Exception as e:
        logging.error(f"Task 2 thất bại: {e}")
        raise


if __name__ == "__main__":
    main()
