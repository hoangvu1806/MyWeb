# Sử dụng image python với phiên bản 3.11
FROM python:3.11-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Sao chép tất cả các file vào thư mục làm việc
COPY . /app

# Cài đặt FastAPI với tất cả các phụ thuộc
RUN pip install --no-cache-dir fastapi[all]

# Mở cổng 8000 để truy cập vào FastAPI
EXPOSE 8000

# Chạy FastAPI với Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
