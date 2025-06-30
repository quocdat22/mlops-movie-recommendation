
Hoàn thành! Tôi đã tạo DAG file `airflow/dags/movie_data_pipeline_dag.py` với các tính năng sau:

## 🔧 **Cấu trúc DAG**:

### **3 Tasks được định nghĩa**:
1. **`fetch_new_movies`**: Gọi Task 1 (lấy dữ liệu từ TMDB)
2. **`process_and_store_movies`**: Gọi Task 2 (xử lý và lưu vào database)  
3. **`check_pipeline_status`**: Task kiểm tra cuối (ghi log trạng thái)

### **Dependencies (Thứ tự thực thi)**:
```
fetch_new_movies → process_and_store_movies → check_pipeline_status
```

## ⚙️ **Cấu hình quan trọng**:

- **Schedule**: Chạy hàng ngày lúc 2:00 AM (`'0 2 * * *'`)
- **Retry**: 2 lần nếu có lỗi, delay 5 phút giữa các lần
- **Max Active Runs**: 1 (tránh chạy đồng thời nhiều instance)
- **Catchup**: False (không chạy lại các lần bị miss)

## 📋 **Để sử dụng DAG này**:

1. **Restart Airflow** để nó nhận diện DAG mới:
   ```bash
   cd airflow
   docker-compose restart
   ```

2. **Kiểm tra trong Airflow UI**:
   - Mở `http://localhost:8080`
   - Bạn sẽ thấy DAG mới tên `movie_data_pipeline`
   - Bật (enable) DAG bằng cách click vào toggle switch

3. **Test thủ công** (nếu muốn):
   - Click vào DAG name → "Trigger DAG" để chạy ngay
   - Hoặc đợi đến 2:00 AM để nó tự động chạy

4. **Monitor execution**:
   - Click vào DAG để xem Graph View
   - Click vào từng task để xem logs chi tiết

## 📊 **Lợi ích của DAG này**:

- **Tự động hóa hoàn toàn**: Không cần can thiệp thủ công
- **Fault tolerance**: Retry khi có lỗi
- **Monitoring**: Log chi tiết cho từng bước
- **Documentation**: Mô tả đầy đủ trong Airflow UI
- **Scalable**: Dễ dàng thêm tasks mới khi cần

Bây giờ hệ thống **Phase 2: Data Pipeline** của bạn đã hoàn thiện với việc tự động cập nhật dữ liệu phim mới hàng ngày! 🎉