FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 설치 (chromadb, sentence-transformers 빌드에 필요)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 의존성 먼저 복사 (레이어 캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY src/ ./src/
COPY main.py .

# docs 폴더 생성 (볼륨 마운트 전 기본 디렉토리)
RUN mkdir -p docs

CMD ["python", "main.py"]
