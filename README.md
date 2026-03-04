# PDF 문서 참조 AI 에이전트

LangChain + LangGraph를 활용한 Adaptive RAG 기반 PDF 문서 참조 챗봇

---

## 시스템 아키텍처

```
사용자 질문
     │
     ▼
[Retrieve] Ensemble 검색 (BM25 + Vector MMR)
     │
     ▼
[Grade] 문서 관련성 평가
     │
     ├─ 관련 있음 ──────────────────────────┐
     │                                       │
     └─ 관련 없음 + 재시도 가능              │
          │                                  │
          ▼                                  │
     [Rewrite] 쿼리 최적화 → 재검색          │
                                             ▼
                                       [Generate]
                                   LLM으로 답변 생성
                                   (출처 + 근거 포함)
                                             │
                                             ▼
                                        최종 답변
```

---

## 기술 선정 이유

### 개발 언어: Python

| 항목 | 내용 |
|------|------|
| **생태계** | LangChain, LangGraph의 공식 지원 언어로 가장 풍부한 통합 제공 |
| **AI/ML 표준** | Hugging Face, PyTorch 등 AI 생태계의 사실상 표준 언어 |
| **문서 처리** | pdfplumber 등 PDF 처리 라이브러리가 가장 성숙 |
| **개발 속도** | 빠른 프로토타이핑과 간결한 코드로 RAG 파이프라인 구현에 최적 |

---

### LLM: Claude Sonnet 4.6 (Anthropic) / Gemini 2.5 Flash (Google)

기본값은 Claude Sonnet 4.6이며, `.env`의 `LLM_PROVIDER` 설정으로 Gemini 2.5 Flash로 전환 가능합니다.

#### Claude Sonnet 4.6

| 항목 | 내용 |
|------|------|
| **컨텍스트 윈도우** | **200,000 토큰** — 긴 PDF 문서의 전체 내용 처리 가능 |
| **문서 이해력** | 복잡한 PDF 구조와 전문 용어 해석 능력이 타 모델 대비 우수 |
| **Hallucination 억제** | 문서 기반 답변 시 근거 없는 정보 생성 억제 능력이 뛰어남 |
| **한국어 지원** | 한국어 문서 처리 및 답변 생성의 품질이 높음 |

#### Gemini 2.5 Flash

| 항목 | 내용 |
|------|------|
| **속도** | 응답 속도가 빠르고 비용 효율적 |
| **멀티모달** | 이미지·표 포함 PDF 처리에 유리 |
| **무료 티어** | Google AI Studio 무료 할당량 제공 |

> Claude는 문서 이해·한국어 처리 품질, Gemini는 속도·비용 효율 면에서 각각 강점

---

### VectorDB: ChromaDB

| 항목 | 내용 |
|------|------|
| **로컬 실행** | 별도 서버·클라우드 불필요 → 개발 환경 구성 간편 |
| **영구 저장** | `persist_directory` 지원 → PDF 재처리 없이 재사용 가능 |
| **LangChain 통합** | LangChain과의 통합이 가장 성숙하고 안정적 |
| **중소규모 최적** | 수백~수천 페이지 규모 문서에 최적화된 성능 |

> Pinecone(클라우드 의존, 비용 발생), FAISS(영구 저장 불편) 대비 개발~운영 균형 최적

---

### 임베딩 모델: BAAI/bge-m3

| 항목 | 내용 |
|------|------|
| **다국어 지원** | 한국어를 포함한 다국어 특화 임베딩 → 한국어 PDF 품질 우수 |
| **무료·로컬** | 로컬 실행으로 API 비용 없음 |
| **벤치마크** | MTEB(Massive Text Embedding Benchmark) 지속적 상위권 |
| **최적화** | `normalize_embeddings=True`로 코사인 유사도 검색 최적화 |

> OpenAI Embedding(유료), ko-sroberta(한국어 단일 언어) 대비 다국어 무료 모델 중 최상위 성능

---

### PDF 로더: PDFPlumberLoader

| 항목 | 내용 |
|------|------|
| **표(Table) 추출** | 셀 구조를 유지하며 정확하게 추출 |
| **레이아웃 보존** | 다단 구성, 헤더/푸터 처리가 PyPDFLoader 대비 우수 |
| **특수문자** | 수식·특수 폰트 깨짐 최소화 |

> PyPDFLoader는 표·복잡한 레이아웃에서 텍스트가 뒤섞이는 문제가 있어 PDFPlumberLoader로 교체

---

### Retriever: Ensemble (BM25 + Vector) + MMR

**왜 단일 검색 방식이 아닌 Ensemble인가?**

```
BM25 (키워드 기반)       Vector MMR (의미 기반)
       +                       +
  전문 용어·고유명사       문맥·유사 표현 검색
  정확한 매칭에 강함       의미 이해에 강함
       │                       │
       └──────── Ensemble ──────┘
                    │
              Recall 극대화
```

| 구성 요소 | 가중치 | 역할 |
|----------|--------|------|
| BM25 | 0.4 | 키워드 정확 매칭 (전문 용어, 고유명사) |
| Vector MMR | 0.6 | 의미 기반 검색 + 중복 청크 제거 |

- **MMR (Maximal Marginal Relevance)**: 유사한 청크 중복 제거, 다양한 관점의 컨텍스트 확보
- `lambda_mult=0.7`: 관련성(0.7)과 다양성(0.3)의 균형
- 순수 벡터 검색 대비 PDF 전문 문서에서 관련 정보 검색률 향상

---

### Prompt Engineering 전략

#### 1. RAG 답변 생성 프롬프트
```
역할 정의     → "제공된 문서만을 기반으로 답변하는 전문가"
환각 방지     → "문서에 없는 내용은 '찾을 수 없습니다' 명시" 규칙 강제
출처 인용     → "문서명, 페이지 번호 반드시 포함" → 신뢰성 확보
답변 형식     → 핵심 먼저, 상세 설명, 출처 순으로 구조화
대화 히스토리 → MessagesPlaceholder로 멀티턴 컨텍스트 유지
```

#### 2. 문서 관련성 평가 프롬프트
```
이진 분류 (yes/no) → 명확한 판단, 파싱 용이
추가 설명 금지     → 일관된 출력 형식 강제
```

#### 3. 쿼리 재작성 프롬프트
```
검색 최적화 목적   → 핵심 키워드 추출, 모호한 표현 구체화
의도 보존 원칙     → 원래 질문의 의미 유지하되 검색에 유리한 형태로 변환
```

---

### LangGraph 패턴: Adaptive RAG

| 항목 | 내용 |
|------|------|
| **조건부 흐름** | 단순 체인 대비 문서 관련성에 따른 분기 처리 가능 |
| **자동 재검색** | 관련 문서 없을 시 쿼리 재작성 → 재검색 자동 수행 |
| **내결함성** | MAX_RETRIES로 무한 루프 방지 |
| **상태 관리** | TypedDict 기반 상태로 대화 컨텍스트 유지 |

---

## 프로젝트 구조

```
├── README.md              # 선정 이유 및 실행 방법
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example           # 환경 변수 템플릿
├── .gitignore
├── requirements.txt       # 의존성 목록
├── main.py                # 진입점 (CLI 챗봇)
├── src/
│   ├── __init__.py
│   ├── config.py          # 설정값 관리
│   ├── document_loader.py # PDF 로드 + 청킹
│   ├── vectorstore.py     # ChromaDB 초기화/로드
│   ├── retriever.py       # Ensemble Retriever 생성
│   ├── prompts.py         # 프롬프트 템플릿 정의
│   └── graph.py           # LangGraph 에이전트 구성
└── docs/                  # PDF 파일 보관 폴더
```

---

## 설치 및 실행

### 사전 요구사항

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) 설치

### 1. API 키 설정

```bash
cp .env.example .env
```

`.env` 파일을 열어 API 키와 LLM 프로바이더를 입력합니다:

```
# Claude 사용 시
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Gemini 사용 시
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_google_api_key_here
```

> Anthropic API 키: https://console.anthropic.com
> Google API 키: https://aistudio.google.com

### 2. PDF 문서 준비

```bash
# docs/ 폴더에 참조할 PDF 파일 복사
cp your_document.pdf docs/
```

### 3. 이미지 빌드 및 실행

```bash
docker-compose run --rm rag-agent
```

> 최초 실행 시 이미지 빌드 + BAAI/bge-m3 모델 다운로드(약 1.5GB)로 시간이 걸립니다.
> 이후 실행부터는 캐시를 사용하므로 빠르게 시작됩니다.

### 4. 새 PDF 추가 시 (벡터스토어 재생성)

```bash
docker-compose run --rm rag-agent python main.py --rebuild
```

---

## 사용 방법

```
========================================================
       PDF 문서 참조 AI 에이전트
       LangChain + LangGraph + Claude Sonnet 4.6
========================================================

[1/3] 문서 로드 중...
[2/3] 벡터스토어 초기화 중...
[3/3] 에이전트 초기화 중...

준비 완료! 질문을 입력하세요.
명령어: 'quit'/'exit' 종료 | 'clear' 대화 초기화 | 'history' 대화 내역
------------------------------------------------------------

질문: 문서에서 설명하는 주요 개념은 무엇인가요?

  [검색] 관련 문서 검색 중...
  [평가] 문서 관련성 평가 중...
  [생성] 답변 생성 중...

답변:
(문서 기반 답변 + 출처 표시)
------------------------------------------------------------
```

### 명령어

| 명령어 | 동작 |
|--------|------|
| `quit` / `exit` | 프로그램 종료 |
| `clear` / `초기화` | 대화 내역 초기화 |
| `history` / `내역` | 대화 내역 확인 |

---

## 기술 스택 요약

| 구분 | 선택 | 이유 요약 |
|------|------|-----------|
| 언어 | Python 3.11 | LangChain/LangGraph 공식 지원, AI 생태계 표준 |
| LLM | Claude Sonnet 4.6 / Gemini 2.5 Flash | 문서 이해력·한국어 품질 / 속도·비용 효율 |
| VectorDB | ChromaDB | 로컬, 영구 저장, LangChain 통합 안정적 |
| 임베딩 | BAAI/bge-m3 | 다국어(한국어) 특화, 무료, 고성능 |
| PDF 로더 | PDFPlumberLoader | 표·레이아웃 정확 추출 |
| Retriever | Ensemble (BM25+Vector) + MMR | 키워드+의미 결합으로 Recall 극대화 |
| 프레임워크 | LangChain + LangGraph | RAG 파이프라인 + 조건부 에이전트 흐름 |
