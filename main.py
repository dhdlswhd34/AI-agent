import os
import argparse
import shutil
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

from src.document_loader import load_documents, split_documents
from src.vectorstore import get_or_create_vectorstore
from src.retriever import create_retriever
from src.graph import create_agent_graph
from src.config import LLM_PROVIDER


def parse_args():
    parser = argparse.ArgumentParser(description="PDF 문서 참조 AI 에이전트")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="벡터스토어를 강제로 재생성합니다 (새 PDF 추가 시 사용)",
    )
    return parser.parse_args()


def check_env():
    """환경 변수 및 문서 폴더 확인"""
    if LLM_PROVIDER == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            print("[오류] GOOGLE_API_KEY가 설정되지 않았습니다.")
            print("       .env 파일에 GOOGLE_API_KEY를 설정해주세요.")
            return False
    else:
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("[오류] ANTHROPIC_API_KEY가 설정되지 않았습니다.")
            print("       .env 파일에 ANTHROPIC_API_KEY를 설정해주세요.")
            return False

    docs_dir = "./docs"
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"[안내] {docs_dir}/ 폴더를 생성했습니다. PDF 파일을 넣어주세요.")
        return False

    return True


def print_banner():
    model_label = "Gemini 2.0 Flash" if LLM_PROVIDER == "gemini" else "Claude claude-sonnet-4-6"
    print("=" * 60)
    print("       PDF 문서 참조 AI 에이전트")
    print(f"       LangChain + LangGraph + {model_label}")
    print("=" * 60)


def run_chatbot(agent, chat_history: list):
    """CLI 챗봇 루프"""
    print("\n준비 완료! 질문을 입력하세요.")
    print("명령어: 'quit'/'exit' 종료 | 'clear' 대화 초기화 | 'history' 대화 내역")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n질문: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n종료합니다.")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "종료"]:
            print("종료합니다.")
            break

        if user_input.lower() in ["clear", "초기화"]:
            chat_history.clear()
            print("[대화 내역이 초기화되었습니다]")
            continue

        if user_input.lower() in ["history", "내역"]:
            if not chat_history:
                print("[대화 내역이 없습니다]")
            else:
                print("\n--- 대화 내역 ---")
                for msg in chat_history:
                    role = "질문" if isinstance(msg, HumanMessage) else "답변"
                    print(f"[{role}] {msg.content[:100]}...")
            continue

        print()

        # 에이전트 실행
        result = agent.invoke(
            {
                "question": user_input,
                "documents": [],
                "generation": "",
                "chat_history": chat_history,
                "retries": 0,
            }
        )

        answer = result["generation"]
        print(f"\n답변:\n{answer}")
        print("-" * 60)

        # 대화 히스토리 업데이트 (최근 10턴 유지)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=answer))
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]


def main():
    args = parse_args()
    print_banner()

    if not check_env():
        return

    # 벡터스토어 재생성 옵션
    if args.rebuild:
        chroma_dir = "./chroma_db"
        if os.path.exists(chroma_dir):
            for item in os.listdir(chroma_dir):
                item_path = os.path.join(chroma_dir, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            print("[재생성] 기존 벡터스토어를 삭제했습니다.")

    # 1. 문서 로드
    print("\n[1/3] 문서 로드 중...")
    documents = load_documents()
    if not documents:
        print("[오류] docs/ 폴더에 PDF 파일이 없습니다.")
        print("       docs/ 폴더에 참조할 PDF 파일을 넣고 다시 실행하세요.")
        return

    chunks = split_documents(documents)
    print(f"   완료: {len(documents)}개 문서, {len(chunks)}개 청크")

    # 2. 벡터스토어 초기화
    print("\n[2/3] 벡터스토어 초기화 중...")
    vectorstore = get_or_create_vectorstore(chunks)
    print("   완료")

    # 3. 에이전트 초기화
    print("\n[3/3] 에이전트 초기화 중...")
    retriever = create_retriever(vectorstore, chunks)
    agent = create_agent_graph(retriever)
    print("   완료")

    # 챗봇 실행
    chat_history = []
    run_chatbot(agent, chat_history)


if __name__ == "__main__":
    main()
