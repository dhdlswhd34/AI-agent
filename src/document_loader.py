import os
import glob
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_documents(docs_dir: str = DOCS_DIR, loader: str = "pdfplumber") -> list:
    """docs/ 폴더의 모든 PDF 파일을 로드합니다."""
    pdf_files = glob.glob(os.path.join(docs_dir, "**/*.pdf"), recursive=True)
    if not pdf_files:
        return []

    if loader == "docling":
        return _load_with_docling(pdf_files)
    return _load_with_pdfplumber(pdf_files)


def _load_with_pdfplumber(pdf_files: list) -> list:
    documents = []
    for pdf_path in pdf_files:
        print(f"   로드 중 (pdfplumber): {os.path.basename(pdf_path)}")
        loader = PDFPlumberLoader(pdf_path)
        docs = loader.load()
        docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]
        documents.extend(docs)
    return documents


def _load_with_docling(pdf_files: list) -> list:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        raise ImportError("docling이 설치되지 않았습니다. pip install docling 실행 후 다시 시도하세요.")

    converter = DocumentConverter()
    documents = []
    for pdf_path in pdf_files:
        print(f"   로드 중 (docling): {os.path.basename(pdf_path)}")
        result = converter.convert(pdf_path)
        md_text = result.document.export_to_markdown()
        if md_text and md_text.strip():
            documents.append(Document(
                page_content=md_text,
                metadata={"source": pdf_path},
            ))
    return documents


def split_documents(documents: list, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list:
    """
    문서를 청크로 분할합니다.

    RecursiveCharacterTextSplitter 선택 이유:
    - 단락 → 문장 → 단어 순으로 재귀적 분할하여 문맥 보존
    - PDF 특성상 줄바꿈 기반 분할이 효과적
    - chunk_overlap으로 청크 경계의 정보 손실 방지
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    return splitter.split_documents(documents)
