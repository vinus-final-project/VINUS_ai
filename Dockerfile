# 1. Python 3.10 기반 이미지 사용 (PyTorch 및 AI 라이브러리 호환성 우수)
FROM python:3.10-slim

# 2. 필수 시스템 패키지 및 C++ 빌드 도구 설치 (ChromaDB, C extension 라이브러리용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 의존성 파일 복사 및 pip 패키지 설치
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 전체 복사
COPY . /app

# 6. AI 서버 실행 포트 노출 (Task Definition 설정과 동일한 8001 포트)
EXPOSE 8001

# 7. 서버 실행 명령 (FastAPI/Uvicorn 기준, 메인 파일명이 main.py 일 경우)
# 만약 실행 파일이 app/main.py 라면 "app.main:app" 으로 수정
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]