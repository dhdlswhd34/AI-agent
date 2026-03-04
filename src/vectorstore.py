import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    임베딩 모델을 초기화합니다.

    BAAI/bge-m3 선택 이유:
    - 다국어(한국어) 특화 임베딩으로 한국어 PDF 처리에 최적
    - 무료, 로컬 실행 가능 (API 비용 없음)
    - normalize_embeddings=True로 코사인 유사도 검색 최적화
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


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
