import os
from dotenv import load_dotenv

load_dotenv()

# API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM 프로바이더
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "claude").lower()

# 모델
MODEL_NAME = "claude-sonnet-4-6"
GEMINI_MODEL_NAME = "gemini-2.0-flash"

# 임베딩 모델
# BAAI/bge-m3 선택 이유:
#   - 다국어(한국어) 특화로 한국어 PDF 임베딩 품질 우수
#   - 완전 무료, 로컬 실행으로 API 비용 없음
#   - MTEB 벤치마크 지속 상위권
EMBEDDING_MODEL = "BAAI/bge-m3"

# 경로
CHROMA_PERSIST_DIR = "./chroma_db"
DOCS_DIR = "./docs"

# 청킹 설정
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 검색 설정
RETRIEVER_K = 5
MAX_RETRIES = 2
