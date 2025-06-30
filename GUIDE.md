Tôi sẽ hướng dẫn bạn xây dựng hệ thống MLOps cho gợi ý phim từ cơ bản đến nâng cao, ưu tiên sử dụng các công nghệ miễn phí.

## 1. Kiến trúc tổng quan MLOps cho hệ thống gợi ý phim

### Các thành phần chính:
- **Data Pipeline**: Thu thập và xử lý dữ liệu phim mới
- **Model Training**: Huấn luyện và cập nhật mô hình
- **Model Serving**: Triển khai mô hình phục vụ API
- **Monitoring**: Giám sát hiệu suất và drift
- **Orchestration**: Điều phối các workflow tự động

## 2. Tech Stack miễn phí đề xuất

### Core Technologies:
- **Python**: Ngôn ngữ chính
- **Git + GitHub**: Version control (miễn phí cho public repos)
- **DVC (Data Version Control)**: Phiên bản hóa dữ liệu và mô hình
- **Docker**: Containerization
- **GitHub Actions**: CI/CD pipeline
- **MLflow**: Experiment tracking và model registry
- **Apache Airflow**: Workflow orchestration
- **FastAPI**: Model serving API
- **PostgreSQL**: Database
- **Grafana + Prometheus**: Monitoring
- **Configuration Management**: Tách biệt cấu hình (ví dụ: `.env`, `config.yaml`)

### Cloud Platform (Free Tier):
- **Render/Railway**: Deploy miễn phí với giới hạn

## 3. Roadmap từng bước

### Phase 1: Foundation Setup (Tuần 1-2)

```python
# 1. Setup project structure
mlops-movie-recommendation/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── api/
├── tests/
├── docker/
├── airflow/
├── monitoring/
└── deployment/
```

### Phase 2: Data Pipeline (Tuần 2-3)

```python
# Ví dụ data pipeline với Apache Airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

def extract_new_movies():
    # Crawl TMDB API hoặc IMDb
    # Lưu vào database
    pass

def preprocess_data():
    # Clean và feature engineering
    pass

def update_user_interactions():
    # Cập nhật dữ liệu tương tác người dùng
    pass

dag = DAG(
    'movie_data_pipeline',
    default_args={
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    schedule_interval='@daily'
)

extract_task = PythonOperator(
    task_id='extract_movies',
    python_callable=extract_new_movies,
    dag=dag
)
```

### Phase 3: Model Development (Tuần 3-4)

```python
# Model training với MLflow tracking
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error

def train_recommendation_model():
    with mlflow.start_run():
        # Load data
        # Train model (Collaborative Filtering, Matrix Factorization, etc.)
        # Log parameters, metrics, model

        mlflow.log_param("algorithm", "matrix_factorization")
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, "recommendation_model")

# Model cho phim mới (Cold Start Problem)
def train_content_based_model():
    # Sử dụng metadata: genre, director, actors, etc.
    pass
```

### Phase 4: Model Serving (Tuần 4-5)

```python
# FastAPI serving
from fastapi import FastAPI
import mlflow.pyfunc

app = FastAPI()

# Load model
model = mlflow.pyfunc.load_model("models:/recommendation_model/production")

@app.post("/recommend")
async def get_recommendations(user_id: int, num_recommendations: int = 10):
    # Get user preferences
    # Generate recommendations
    # Handle new movies
    recommendations = model.predict(user_features)
    return {"recommendations": recommendations}

@app.post("/recommend_new_movies")
async def recommend_new_movies(user_id: int):
    # Sử dụng content-based filtering cho phim mới
    pass
```

### Phase 5: Monitoring & Retraining (Tuần 5-6)

```python
# Model monitoring
import prometheus_client
from prometheus_client import Counter, Histogram

# Metrics
prediction_counter = Counter('model_predictions_total', 'Total predictions')
prediction_latency = Histogram('model_prediction_duration_seconds', 'Prediction latency')

def monitor_model_performance():
    # Track accuracy, coverage, diversity
    # Detect concept drift
    # Alert when retraining needed
    pass

# Auto-retraining trigger
def check_retraining_criteria():
    # Kiểm tra:
    # - Số lượng phim mới
    # - Model performance degradation
    # - User feedback patterns
    pass
```

## 4. Testing Strategy

Để đảm bảo chất lượng và độ tin cậy của hệ thống, cần áp dụng nhiều loại testing khác nhau:

- **Unit Tests**: Kiểm tra từng thành phần nhỏ, độc lập của code (ví dụ: hàm xử lý dữ liệu, hàm tính toán feature).
- **Data Validation Tests**: Sử dụng các công cụ như [Great Expectations](https://greatexpectations.io/) để tự động xác thực schema, ràng buộc và chất lượng của dữ liệu đầu vào.
- **Model Evaluation Tests**: Thiết lập một pipeline để đánh giá model mới so với model đang chạy trên production. Model mới chỉ được deploy nếu hiệu suất (ví dụ: RMSE, precision@k) vượt qua một ngưỡng nhất định.
- **Integration Tests**: Kiểm tra sự tương tác giữa các thành phần, ví dụ như kiểm tra toàn bộ luồng từ API request đến khi nhận được kết quả gợi ý.

## 5. Giải pháp cho phim mới (Cold Start)

### Hybrid Approach:
```python
class HybridRecommender:
    def __init__(self):
        self.collaborative_model = load_collaborative_model()
        self.content_based_model = load_content_based_model()

    def recommend(self, user_id, include_new_movies=True):
        # Existing movies: Collaborative Filtering
        existing_recs = self.collaborative_model.recommend(user_id)

        if include_new_movies:
            # New movies: Content-based
            new_movie_recs = self.content_based_model.recommend_new(user_id)
            # Blend recommendations
            return self.blend_recommendations(existing_recs, new_movie_recs)

        return existing_recs
```

## 6. Deployment Strategy và Bảo mật

### Docker Setup:
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### GitHub Actions CI/CD:
```yaml
# .github/workflows/deploy.yml
name: MLOps Pipeline
on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: pytest tests/

  train:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Train model
        run: python src/models/train.py

  deploy:
    needs: train
    runs-on: ubuntu-latest
    environment: production # Sử dụng environment để truy cập secrets
    steps:
      - name: Deploy to production
        env:
          RENDER_API_KEY: ${{ secrets.RENDER_API_KEY }}
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: |
          docker build -t movie-recommender .
          # Deploy to cloud (sử dụng các biến env ở trên)
```

### Quản lý Bảo mật:
- **Không hardcode credentials**: Luôn sử dụng các biến môi trường để quản lý API keys, mật khẩu database, và các thông tin nhạy cảm khác.
- **GitHub Secrets**: Tận dụng GitHub Secrets để lưu trữ an toàn các biến môi trường cho CI/CD pipeline.
- **Tách biệt cấu hình**: Sử dụng file `.env` (được thêm vào `.gitignore`) cho môi trường local và các secrets cho môi trường production.

## 7. Timeline và mục tiêu

**Tuần 1-2**: Setup cơ bản, data pipeline đơn giản
**Tuần 3-4**: Model development và experiment tracking
**Tuần 5-6**: Model serving và monitoring cơ bản
**Tuần 7-8**: Auto-retraining và optimization
**Tuần 9-10**: Testing, debugging và documentation

## 8. Bước tiếp theo

1. **Bắt đầu với MVP**: Tập trung vào pipeline đơn giản trước
2. **Data Collection**: Thu thập dữ liệu phim từ TMDB API
3. **Simple Model**: Bắt đầu với collaborative filtering cơ bản
4. **Iterative Development**: Từng bước thêm tính năng
