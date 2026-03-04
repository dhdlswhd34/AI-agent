from typing import TypedDict, List
from langchain_core.messages import BaseMessage
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from src.prompts import RAG_PROMPT, GRADE_PROMPT, REWRITE_PROMPT
from src.config import MODEL_NAME, MAX_RETRIES, LLM_PROVIDER, GOOGLE_API_KEY, GEMINI_MODEL_NAME


class GraphState(TypedDict):
    """LangGraph 상태 정의"""
    question: str           # 현재 질문 (재작성 시 갱신)
    documents: List         # 검색된 문서 청크
    generation: str         # 최종 생성 답변
    chat_history: List[BaseMessage]  # 대화 히스토리
    retries: int            # 쿼리 재작성 횟수


def create_agent_graph(retriever):
    """
    Adaptive RAG LangGraph 에이전트를 생성합니다.

    패턴 선택 이유:
    - 단순 체인 대비 조건부 흐름으로 품질 향상
    - 관련 문서 없을 시 자동 쿼리 재작성 후 재검색
    - 상태 기반 멀티턴 대화 컨텍스트 유지
    - 재시도 제한(MAX_RETRIES)으로 무한 루프 방지

    그래프 흐름:
    retrieve → grade_documents → [generate | rewrite_query]
                                              ↓
                                          retrieve (반복)
    """

    if LLM_PROVIDER == "gemini":
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, temperature=0, google_api_key=GOOGLE_API_KEY)
    else:
        llm = ChatAnthropic(model=MODEL_NAME, temperature=0)

    # -------------------------------------------------------------------------
    # 노드 정의
    # -------------------------------------------------------------------------

    def retrieve_node(state: GraphState) -> dict:
        """Ensemble Retriever로 관련 문서를 검색합니다."""
        print("  [검색] 관련 문서 검색 중...")
        question = state["question"]
        documents = retriever.invoke(question)
        print(f"  [검색] {len(documents)}개 청크 검색 완료")
        return {"documents": documents}

    def grade_documents_node(state: GraphState) -> dict:
        """검색된 문서의 관련성을 평가하여 필터링합니다."""
        print("  [평가] 문서 관련성 평가 중...")
        question = state["question"]
        documents = state["documents"]

        grade_chain = GRADE_PROMPT | llm
        relevant_docs = []

        for doc in documents:
            result = grade_chain.invoke(
                {"question": question, "document": doc.page_content}
            )
            if "yes" in result.content.strip().lower():
                relevant_docs.append(doc)

        print(f"  [평가] 관련 문서: {len(relevant_docs)}/{len(documents)}개")
        return {"documents": relevant_docs}

    def rewrite_query_node(state: GraphState) -> dict:
        """관련 문서가 없을 때 검색 최적화를 위해 쿼리를 재작성합니다."""
        print("  [재작성] 쿼리 최적화 중...")
        question = state["question"]

        rewrite_chain = REWRITE_PROMPT | llm
        result = rewrite_chain.invoke({"question": question})
        new_question = result.content.strip()

        print(f"  [재작성] 원본: {question}")
        print(f"  [재작성] 변환: {new_question}")

        return {
            "question": new_question,
            "retries": state.get("retries", 0) + 1,
        }

    def generate_node(state: GraphState) -> dict:
        """검색된 문서를 기반으로 최종 답변을 생성합니다."""
        print("  [생성] 답변 생성 중...")
        question = state["question"]
        documents = state["documents"]
        chat_history = state.get("chat_history", [])

        if not documents:
            return {
                "generation": "제공된 문서에서 해당 질문에 관련된 정보를 찾을 수 없습니다. 다른 방식으로 질문해보시거나 관련 문서가 있는지 확인해주세요."
            }

        # 문서 컨텍스트 구성 (문서명 + 페이지 포함)
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "문서")
            page = doc.metadata.get("page", "?")
            source_name = source.split("/")[-1].split("\\")[-1]
            context_parts.append(
                f"[{i}] 출처: {source_name}, p.{page}\n{doc.page_content}"
            )
        context = "\n\n---\n\n".join(context_parts)

        rag_chain = RAG_PROMPT | llm
        result = rag_chain.invoke(
            {
                "context": context,
                "question": question,
                "chat_history": chat_history,
            }
        )

        return {"generation": result.content}

    # -------------------------------------------------------------------------
    # 조건부 엣지 함수
    # -------------------------------------------------------------------------

    def decide_to_generate(state: GraphState) -> str:
        """관련 문서 유무와 재시도 횟수에 따라 다음 노드를 결정합니다."""
        documents = state["documents"]
        retries = state.get("retries", 0)

        if not documents and retries < MAX_RETRIES:
            print(f"  [판단] 관련 문서 없음 → 쿼리 재작성 ({retries + 1}/{MAX_RETRIES})")
            return "rewrite"

        return "generate"

    # -------------------------------------------------------------------------
    # 그래프 구성
    # -------------------------------------------------------------------------

    graph = StateGraph(GraphState)

    # 노드 등록
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("generate", generate_node)

    # 엣지 연결
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "rewrite": "rewrite_query",
            "generate": "generate",
        },
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()
