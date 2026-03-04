from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from src.config import RETRIEVER_K


def create_retriever(vectorstore: Chroma, documents: list) -> EnsembleRetriever:
    """
    Ensemble Retriever (BM25 + Vector MMR)를 생성합니다.

    선택 이유:
    1. BM25 (키워드 기반 검색):
       - PDF 문서의 전문 용어·고유명사 정확 매칭에 강함
       - 희소(sparse) 표현으로 exact match 보장

    2. Vector MMR (의미 기반 검색):
       - 문맥·유사 표현의 의미 기반 검색에 강함
       - MMR(Maximal Marginal Relevance)로 중복 청크 제거 및 다양한 관점 확보
       - fetch_k > k 설정으로 후보군에서 다양성 극대화

    3. Ensemble (가중치 BM25:Vector = 4:6):
       - 두 방식을 결합하여 Recall 극대화
       - 벡터 검색에 더 높은 가중치로 의미 이해 중심 검색
       - 순수 벡터 검색 대비 PDF에서 관련 정보 검색률 향상
    """

    # BM25 Retriever (키워드 기반)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = RETRIEVER_K

    # Vector Retriever with MMR (의미 기반 + 다양성)
    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVER_K,
            "fetch_k": RETRIEVER_K * 3,
            "lambda_mult": 0.7,  # 1.0=관련성 중심, 0.0=다양성 중심
        },
    )

    # Ensemble Retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6],
    )

    return ensemble_retriever
