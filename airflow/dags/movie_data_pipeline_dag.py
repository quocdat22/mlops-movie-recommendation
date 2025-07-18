import os
import sys
from datetime import datetime, timedelta

from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from airflow import DAG

# Thêm đường dẫn src vào sys.path để có thể import các module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_path = os.path.join(project_root, "src")
sys.path.append(src_path)

# Import các hàm từ script đã tạo
from data.airflow_fetch_movies import main as fetch_movies_main
from data.airflow_process_movies import main as process_movies_main


# Định nghĩa hàm wrapper cho model training
def run_preprocessing_and_training():
    """Run data preprocessing and model training"""
    import os
    import subprocess
    import sys

    # Get project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    # Run preprocessing
    preprocess_script = os.path.join(project_root, "src", "data", "preprocess_data.py")
    preprocess_cmd = [sys.executable, preprocess_script]
    result = subprocess.run(
        preprocess_cmd, capture_output=True, text=True, cwd=project_root
    )

    if result.returncode != 0:
        raise Exception(f"Preprocessing failed: {result.stderr}")

    # Run model training
    train_script = os.path.join(project_root, "src", "models", "train_model.py")
    train_cmd = [sys.executable, train_script]
    result = subprocess.run(train_cmd, capture_output=True, text=True, cwd=project_root)

    if result.returncode != 0:
        raise Exception(f"Model training failed: {result.stderr}")

    print("Preprocessing and model training completed successfully!")


# Định nghĩa các tham số mặc định cho DAG
default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "start_date": datetime(2025, 6, 29),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "catchup": False,  # Không chạy lại các lần bị miss
}

# Khởi tạo DAG
dag = DAG(
    "movie_data_pipeline",
    default_args=default_args,
    description="Complete MLOps pipeline: data collection, processing, model training",
    schedule_interval="0 2 * * 1",  # Chạy hàng tuần vào thứ 2 lúc 2:00 AM
    max_active_runs=1,  # Chỉ cho phép 1 instance chạy cùng lúc
    tags=["movies", "data-pipeline", "mlops", "model-training"],
)

# Task 1: Lấy dữ liệu phim mới từ TMDB API
fetch_movies_task = PythonOperator(
    task_id="fetch_new_movies",
    python_callable=fetch_movies_main,
    dag=dag,
    doc_md="""
    ### Fetch New Movies Task

    Task này thực hiện việc:
    - Kết nối đến TMDB API
    - Lấy danh sách phim mới phát hành trong 7 ngày qua
    - Bổ sung thông tin cast (3 diễn viên đầu) và keywords
    - Lưu dữ liệu vào file `data/raw/recent_movies.json`

    **Input**: TMDB API Key từ biến môi trường
    **Output**: File JSON chứa dữ liệu phim mới
    """,
)

# Task 2: Xử lý và lưu dữ liệu vào database
process_movies_task = PythonOperator(
    task_id="process_and_store_movies",
    python_callable=process_movies_main,
    dag=dag,
    doc_md="""
    ### Process and Store Movies Task

    Task này thực hiện việc:
    - Đọc file JSON từ Task 1
    - Kiểm tra và lọc ra những phim chưa có trong database
    - Chèn phim mới vào PostgreSQL database
    - Dọn dẹp file tạm sau khi hoàn thành

    **Input**: File `data/raw/recent_movies.json`
    **Output**: Dữ liệu được cập nhật trong PostgreSQL
    """,
)

# Task 3: Preprocessing và Training Model (chạy hàng tuần)
train_model_task = PythonOperator(
    task_id="preprocess_and_train_model",
    python_callable=run_preprocessing_and_training,
    dag=dag,
    execution_timeout=timedelta(hours=2),  # Set timeout for long-running task
    doc_md="""
    ### Data Preprocessing and Model Training Task

    Task này thực hiện việc:
    - Chạy script preprocessing để chuẩn bị dữ liệu cho model
    - Thực hiện feature engineering với tags được tối ưu
    - Huấn luyện mô hình content-based recommendation
    - Tạo similarity matrix và lưu model artifacts
    - Sinh báo cáo training và validation metrics

    **Input**: Dữ liệu từ PostgreSQL database
    **Output**:
    - File processed_movies.parquet
    - Model artifacts (TF-IDF vectorizer, similarity matrix)
    - Training report và metrics
    """,
)

# Task 4: Kiểm tra trạng thái pipeline
check_pipeline_status = BashOperator(
    task_id="check_pipeline_status",
    bash_command="""
    echo "=== MLOps Movie Pipeline Status ==="
    echo "Pipeline completed at: $(date)"
    echo "Current working directory: $(pwd)"
    echo "Checking data directory..."
    ls -la data/ 2>/dev/null || echo "Data directory not found"
    echo "Checking model directory..."
    ls -la models/ 2>/dev/null || echo "Model directory not found"
    echo "Pipeline execution finished successfully!"
    """,
    dag=dag,
    doc_md="""
    ### Pipeline Status Check

    Task cuối cùng để:
    - Ghi log trạng thái hoàn thành pipeline
    - Kiểm tra sự tồn tại của thư mục dữ liệu và model
    - Xác nhận pipeline chạy thành công
    - Tạo log summary cho monitoring
    """,
)

# Thiết lập dependencies (thứ tự thực thi)
fetch_movies_task >> process_movies_task >> train_model_task >> check_pipeline_status

# Cấu hình thông tin DAG để hiển thị trong Airflow UI
dag.doc_md = """
# MLOps Movie Recommendation Pipeline

## Mô tả
Pipeline hoàn chỉnh cho hệ thống gợi ý phim sử dụng MLOps best practices, bao gồm:
- Thu thập dữ liệu tự động từ TMDB API
- Xử lý và lưu trữ dữ liệu vào PostgreSQL
- Feature engineering và data preprocessing
- Huấn luyện mô hình content-based recommendation
- Tạo model artifacts sẵn sàng cho deployment

## Luồng xử lý
1. **Fetch Movies**: Lấy dữ liệu phim mới từ TMDB API với thông tin chi tiết
2. **Process & Store**: Xử lý và lưu vào database, tránh duplicate
3. **Train Model**: Preprocessing dữ liệu và huấn luyện mô hình recommendation
4. **Status Check**: Kiểm tra và báo cáo trạng thái pipeline

## Lịch chạy
- **Data Collection**: Hàng ngày lúc 2:00 AM (các task 1-2)
- **Model Training**: Hàng tuần vào thứ 2 lúc 2:00 AM (full pipeline)

## Output
- Dữ liệu phim được cập nhật trong PostgreSQL
- Model artifacts: TF-IDF vectorizer, similarity matrix
- Training reports và validation metrics
- Ready-to-deploy recommendation system

## Monitoring
- Task logs trong Airflow UI
- Training metrics và reports
- Database connection status
- Model performance validation

## Environment Variables Required
- `TMDB_API_KEY`: API key cho The Movie Database
- `DATABASE_URL`: Connection string cho PostgreSQL database

## Usage
Để chạy pipeline:
1. Đảm bảo Airflow đã được cấu hình đúng
2. Copy DAG file vào thư mục `/opt/airflow/dags/`
3. Restart Airflow scheduler và webserver
4. Enable DAG trong Airflow UI
"""
