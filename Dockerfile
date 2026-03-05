FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 (chromadb 빌드, docling 빌드에 필요)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 의존성 먼저 복사 (레이어 캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# HuggingFace 모델 캐시 디렉토리 (볼륨 마운트 대상)
ENV HF_HOME=/app/.cache/huggingface

# 소스 코드 복사
COPY src/ ./src/
COPY main.py .

# docs, chroma_db 디렉토리 생성 (볼륨 마운트 전 기본 디렉토리)
RUN mkdir -p docs chroma_db

CMD ["python", "main.py"]
