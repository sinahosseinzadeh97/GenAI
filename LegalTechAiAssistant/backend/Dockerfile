FROM python:3.11-slim

# پایتون و pip بهینه
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# وابستگی‌های سیستمی کم‌حجم
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
  && rm -rf /var/lib/apt/lists/*

# ابزارهای پایه پایتون
RUN python -m pip install --upgrade pip setuptools wheel

# مهم: نصب Torch فقط-CPU (بدون CUDA)
# اگر از sentence-transformers استفاده می‌کنی این خط رو نگه دار؛ در غیر این صورت می‌تونی حذفش کنی.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1

# نصب وابستگی‌ها (httpx پین شده تا با openai==1.44.0 سازگار باشد)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# کپی کد برنامه
COPY app ./app

# ساخت دایرکتوری‌های ذخیره‌سازی
RUN mkdir -p /app/app/storage/files /app/app/storage/vectorstore

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
