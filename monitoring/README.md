# Movie Recommendation API Monitoring

Hệ thống monitoring cho Movie Recommendation API sử dụng Prometheus và Grafana để theo dõi hiệu suất và trạng thái của API và model.

## 🚀 Quick Start

### 1. Khởi động Monitoring Stack

```bash
# Khởi động tất cả monitoring services
./start_monitoring.sh
```

### 2. Khởi động API Service (nếu chưa chạy)

```bash
# Trong terminal khác
./start_api.sh
```

### 3. Truy cập Monitoring Dashboards

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin123)
- **API Health**: http://localhost:8000/health
- **API Metrics**: http://localhost:8000/metrics

## 📊 Metrics được theo dõi

### System Metrics (Prometheus + FastAPI Instrumentator)
- **Request Rate**: Số lượng requests per second
- **Response Latency**: Thời gian phản hồi (P50, P95, P99)
- **HTTP Status Codes**: Phân bố status codes
- **Active Connections**: Số kết nối đang hoạt động

### Custom Model Metrics
- **Model Status**: Trạng thái model (loaded/not loaded)
- **Model Inference Time**: Thời gian xử lý recommendations
- **Recommendations Generated**: Số lượng recommendations được tạo
- **Cache Performance**: Hit rate và cache size
- **Search Performance**: Thời gian và tần suất search
- **Error Tracking**: Theo dõi lỗi model theo loại

### Business Metrics
- **API Usage**: Endpoint usage patterns
- **User Behavior**: Request patterns và preferences
- **Performance Trends**: Long-term performance trends

## 🔧 Cấu hình

### Prometheus Configuration
File: `monitoring/prometheus.yml`
- Scrape interval: 10 seconds
- Targets: API service trên port 8000
- Metrics endpoint: `/metrics`

### Grafana Configuration
- **Datasource**: Prometheus tự động cấu hình
- **Dashboard**: Movie Recommendation API Dashboard
- **Refresh**: 5 seconds auto-refresh

## 📈 Dashboard Panels

### 1. API Health Status
- Model loading status
- Request rate over time
- Service uptime

### 2. Request Latency
- Response time percentiles (P50, P95, P99)
- Latency trends
- Performance degradation alerts

### 3. Model Performance
- Model inference time
- Recommendations generation rate
- Model error tracking

### 4. Cache Performance
- Cache hit/miss rates
- Cache size over time
- Cache efficiency metrics

## 🧪 Load Testing

### Chạy Load Test
```bash
# Basic load test (60 seconds, 5 users)
python src/api/load_test.py

# Custom configuration
python src/api/load_test.py --duration 120 --users 10 --url http://localhost:8000
```

### Load Test Features
- **Multiple Endpoints**: Tests recommendations, search, health
- **Realistic Patterns**: Weighted request distribution
- **Concurrent Users**: Simulates multiple users
- **Detailed Metrics**: Response times, success rates, throughput

## 📊 Monitoring Endpoints

### API Endpoints for Monitoring
- `GET /health` - Service health check
- `GET /metrics` - Prometheus metrics
- `GET /monitoring/summary` - Human-readable metrics summary
- `GET /model/stats` - Model statistics

### Example Monitoring Summary
```bash
curl http://localhost:8000/monitoring/summary
```

Response:
```json
{
  "model_status": {
    "loaded": true,
    "total_movies": 45466,
    "vocabulary_size": 20000
  },
  "cache_metrics": {
    "hits": 150,
    "misses": 25,
    "hit_rate": 0.8571,
    "current_size": 100
  },
  "request_metrics": {
    "recommendations_generated": 1250,
    "search_queries": 300,
    "model_errors": 2
  }
}
```

## 🚨 Alerts và Monitoring

### Key Metrics to Watch
1. **Model Health**: `recommendation_model_loaded = 0`
2. **High Latency**: `http_request_duration_seconds > 2.0`
3. **Error Rate**: `http_requests_total{status_code=~"5.."} > 0.05`
4. **Low Cache Hit Rate**: `cache_hit_rate < 0.7`

### Future Enhancements
- **Alerting Rules**: Prometheus AlertManager configuration
- **Data Drift Detection**: Integration với Evidently AI
- **Custom Dashboards**: Thêm business-specific metrics
- **Log Aggregation**: ELK stack integration

## 🛠️ Troubleshooting

### Common Issues

1. **Metrics not showing in Grafana**
   ```bash
   # Check if Prometheus can scrape API
   curl http://localhost:9090/api/v1/targets
   
   # Check API metrics endpoint
   curl http://localhost:8000/metrics
   ```

2. **API not accessible from Prometheus**
   ```bash
   # Check if API is running
   curl http://localhost:8000/health
   
   # Check Docker network (if using Docker)
   docker network ls
   ```

3. **Grafana dashboard not loading**
   - Verify Prometheus datasource connection
   - Check dashboard JSON configuration
   - Ensure correct metric names in queries

### Stop Monitoring Stack
```bash
docker-compose -f docker-compose.monitoring.yml down
```

## 📚 Next Steps

1. **Data Drift Monitoring**: Implement Evidently AI for data drift detection
2. **Alerting**: Set up AlertManager for proactive monitoring
3. **Business Metrics**: Add user engagement and recommendation quality metrics
4. **Log Analysis**: Integrate structured logging and log analysis
5. **A/B Testing**: Add metrics for model version comparison
