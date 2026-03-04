import os
import glob
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_documents(docs_dir: str = DOCS_DIR) -> list:
    """docs/ 폴더의 모든 PDF 파일을 로드합니다."""
    pdf_files = glob.glob(os.path.join(docs_dir, "**/*.pdf"), recursive=True)

    if not pdf_files:
        return []

    documents = []
    for pdf_path in pdf_files:
        print(f"   로드 중: {os.path.basename(pdf_path)}")
        loader = PDFPlumberLoader(pdf_path)
        docs = loader.load()
        documents.extend(docs)

    return documents


def split_documents(documents: list) -> list:
    """
    문서를 청크로 분할합니다.

    RecursiveCharacterTextSplitter 선택 이유:
    - 단락 → 문장 → 단어 순으로 재귀적 분할하여 문맥 보존
    - PDF 특성상 줄바꿈 기반 분할이 효과적
    - chunk_overlap으로 청크 경계의 정보 손실 방지
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks
