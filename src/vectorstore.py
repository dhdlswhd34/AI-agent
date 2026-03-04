import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from src.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL


class BGEEmbeddings(Embeddings):
    """
    transformers를 직접 사용하는 커스텀 임베딩 클래스.

    선택 이유:
    - sentence_transformers 3.x에서 BAAI/bge-m3 tokenizer 호환 문제 우회
    - tokenizer에 명시적으로 str 입력을 보장
    - CLS 토큰 풀링 + L2 정규화로 BAAI/bge-m3 공식 권장 방식 구현
    """

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def _encode(self, texts: list) -> list:
        texts = [str(t) for t in texts]
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = self.model(**encoded)
        embeddings = output.last_hidden_state[:, 0, :]  # CLS 토큰
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.numpy().tolist()

    def embed_documents(self, texts: list) -> list:
        return self._encode(texts)

    def embed_query(self, text: str) -> list:
        return self._encode([text])[0]


def get_embeddings() -> BGEEmbeddings:
    """
    임베딩 모델을 초기화합니다.

    BAAI/bge-m3 선택 이유:
    - 다국어(한국어) 특화 임베딩으로 한국어 PDF 처리에 최적
    - 무료, 로컬 실행 가능 (API 비용 없음)
    - normalize_embeddings=True로 코사인 유사도 검색 최적화
    """
    return BGEEmbeddings(model_name=EMBEDDING_MODEL)


def create_vectorstore(chunks: list) -> Chroma:
    """청크로부터 ChromaDB 벡터스토어를 생성하고 저장합니다."""
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return vectorstore


def load_vectorstore() -> Chroma:
    """저장된 ChromaDB 벡터스토어를 로드합니다."""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )


def get_or_create_vectorstore(chunks: list = None) -> Chroma:
    """
    ChromaDB 벡터스토어를 불러오거나 새로 생성합니다.

    ChromaDB 선택 이유:
    - 로컬 설치, 별도 서버 불필요 → 개발 환경 구성 간편
    - 영구 저장(persist) 지원 → PDF 재처리 없이 재사용 가능
    - LangChain 통합이 가장 성숙하고 안정적
    - Pinecone(클라우드 의존), FAISS(영구저장 불편) 대비 균형 최적
    """
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        print("   기존 벡터스토어 로드 중... (--rebuild 옵션으로 재생성 가능)")
        return load_vectorstore()

    if chunks is None:
        raise ValueError("최초 실행 시 문서 청크가 필요합니다.")

    print("   새 벡터스토어 생성 중...")
    return create_vectorstore(chunks)
