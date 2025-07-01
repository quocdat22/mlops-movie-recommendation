# Movie Recommendation API - GCS Model Loading Fix

## Vấn đề
API không thể tìm thấy model vì model được lưu trữ trên Google Cloud Storage (GCS) thông qua DVC, nhưng container Docker không được cấu hình để tải model từ GCS.

## Giải pháp đã thực hiện

### 1. Thêm function download_from_gcs()
- Đã thêm function `download_from_gcs()` vào `src/models/model_inference.py`
- Function này tự động tải các file model từ GCS bucket khi container khởi động

### 2. Cập nhật Dockerfile
- Thêm entrypoint script để tải model trước khi khởi động API
- Cài đặt `curl` cho health check
- Tạo thư mục `/app/models` trong container

### 3. Tạo entrypoint script
- `src/api/entrypoint.sh` kiểm tra credentials và tải model
- Xử lý lỗi và thông báo chi tiết

### 4. Docker Compose mới
- `docker-compose.gcs.yml` với cấu hình GCS
- Mount credentials file
- Environment variables cho GCS

## Cách sử dụng

### Phương pháp 1: Sử dụng script tự động
```bash
# Đảm bảo file credentials đã có
cp oceanic-hash-461113-m7-5016a3d0b291.json gcs-credentials.json

# Chạy script
./run_api_with_gcs.sh
```

### Phương pháp 2: Docker Compose trực tiếp
```bash
# Set environment variables
export GCS_BUCKET_NAME="staging.doanandroid-e3111.appspot.com"
export GCS_MODEL_PATH="content_based"

# Build và run
docker-compose -f docker-compose.gcs.yml up --build
```

### Phương pháp 3: Docker run (như command ban đầu)
```bash
# Build image
docker build -t movie-api-gcs -f src/api/Dockerfile .

# Run với GCS config
docker run --rm -p 8000:8000 \
  --network movie-net \
  --name movie-api \
  -e GOOGLE_APPLICATION_CREDENTIALS="/gcs-credentials.json" \
  -e GCS_BUCKET_NAME="staging.doanandroid-e3111.appspot.com" \
  -e GCS_MODEL_PATH="content_based" \
  -e MODEL_DIR="/app/models/content_based" \
  -v "$(pwd)/gcs-credentials.json:/gcs-credentials.json:ro" \
  movie-api-gcs
```

## Kiểm tra hoạt động

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Model Stats  
```bash
curl http://localhost:8000/model/stats
```

### 3. Test Recommendation
```bash
curl -X POST "http://localhost:8000/recommendations" \
  -H "Content-Type: application/json" \
  -d '{"movie_id": 1, "top_k": 5}'
```

## Logs để debug

### Container logs
```bash
docker-compose -f docker-compose.gcs.yml logs -f movie-api
```

### Log files
- Container logs: `./logs/`
- Model download progress được log trong container startup

## Files đã thay đổi

1. `src/models/model_inference.py` - Thêm GCS download
2. `src/api/Dockerfile` - Cập nhật build process  
3. `src/api/entrypoint.sh` - Script khởi động
4. `src/api/config.py` - Cập nhật model path
5. `docker-compose.gcs.yml` - Cấu hình GCS
6. `run_api_with_gcs.sh` - Script tiện ích

## Troubleshooting

### Lỗi credentials
```
ERROR: GCS credentials file not found
```
**Giải pháp:** Đảm bảo file `gcs-credentials.json` tồn tại và được mount đúng.

### Lỗi permissions
```
Error downloading model from GCS: 403
```  
**Giải pháp:** Kiểm tra service account có quyền đọc GCS bucket.

### Lỗi model không tồn tại
```
File not found in GCS: content_based/...
```
**Giải pháp:** Kiểm tra model đã được upload lên GCS chưa bằng DVC.

### Upload model lên GCS (nếu cần)
```bash
# Nếu model chưa có trên GCS
dvc push models/content_based.dvc
```