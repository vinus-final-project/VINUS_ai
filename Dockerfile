# Python 3.10 slim 기반
FROM python:3.10-slim

# 1. AI 및 C++ 컴파일에 필요한 기본 시스템 패키지 일괄 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. pip, setuptools, wheel 최신 버전으로 업그레이드
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# 3. 소스 코드 복사
COPY . /app

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]