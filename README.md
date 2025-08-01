# KGRAG - Knowledge Graph-based Retrieval Augmented Generation

지식 그래프 기반 RAG (Retrieval-Augmented Generation) 시스템입니다.

## 🚀 빠른 시작

### 1. 설치 및 환경 설정

```bash
# 저장소 클론
git clone https://github.com/Tiabet/KGRAG.git
cd KGRAG

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate.bat  # Windows

# 패키지 설치
pip install -r requirements.txt

# OpenAI API 키 설정
export OPENAI_API_KEY="your-api-key-here"  # Linux/Mac
# 또는
set OPENAI_API_KEY=your-api-key-here  # Windows
```

### 2. 통합 실행 인터페이스

**Windows:**
```cmd
run_all.bat
```

**Linux/Mac:**
```bash
chmod +x run_all.sh
./run_all.sh
```

## 📁 프로젝트 구조

```
KGRAG/
├── 📁 index/              # 그래프 구축 관련
│   ├── build_graph.py
│   ├── graph_construction.py
│   ├── json_to_gexf.py
│   ├── edge_embedding.py
│   ├── build_index.sh     # Linux/Mac용
│   └── build_index.bat    # Windows용
│
├── 📁 generate/           # 답변 생성 관련
│   ├── graph_based_rag_short.py
│   ├── graph_based_rag_long.py
│   ├── answer_generation_short.py
│   ├── answer_generation_long.py
│   ├── generate_answers.sh   # Linux/Mac용
│   └── generate_answers.bat  # Windows용
│
├── 📁 evaluate/           # 답변 평가 관련
│   ├── judge_F1.py
│   ├── judge_Ultradomain.py
│   ├── evaluate_answers.sh   # Linux/Mac용
│   └── evaluate_answers.bat  # Windows용
│
├── 📁 prompt/             # 프롬프트 템플릿
├── 📁 hotpotQA/           # 데이터셋 예시
├── 📁 UltraDomain/        # 데이터셋 예시
├── 📁 MultihopRAG/        # 데이터셋 예시
│
├── Retriever.py           # 공통 리트리버
├── subtopic_choice.py     # 서브토픽 선택
├── topic_choice.py        # 토픽 선택
├── requirements.txt       # 패키지 목록
├── SETUP_GUIDE.md         # 상세 가이드
├── run_all.sh             # 통합 실행 (Linux/Mac)
└── run_all.bat            # 통합 실행 (Windows)
```

## 🔧 사용법

### 1. 환경 설정 ⚙️

```bash
# .env 파일 생성 (설정 템플릿 복사)
cp .env.example .env

# 또는 테스트 스크립트로 샘플 생성
python test_config.py --create-env

# .env 파일에서 API 키 설정
# OPENAI_API_KEY=your_actual_api_key_here
```

**주요 설정 항목:**
- `OPENAI_API_KEY`: OpenAI API 키 (필수)
- `DEFAULT_MODEL`: 기본 사용 모델 (기본값: gpt-4o-mini)
- `TOP_K1`, `TOP_K2`: RAG 검색 파라미터 (기본값: 50, 10)
- `TOPIC_CHOICE_MIN/MAX`: 토픽 선택 개수 범위 (기본값: 5-10)
- `SUBTOPIC_CHOICE_MIN/MAX`: 서브토픽 선택 개수 범위 (기본값: 10-25)
- `MAX_TOKENS`, `OVERLAP`: 텍스트 청킹 설정 (기본값: 3000, 300)
- `TEMPERATURE`: 모델 생성 온도 (기본값: 0.5)

```bash
# 설정 확인
python test_config.py
```

### 2. 그래프 인덱스 구축 🏗️

먼저 텍스트 데이터에서 지식 그래프를 구축하고 FAISS 인덱스를 생성합니다.

**필요한 입력:** `[데이터셋]/contexts.txt`

**생성 파일:**
- `graph_v1.json` - 추출된 트리플
- `graph_v1.gexf` - 그래프 파일
- `edge_index_v1.faiss` - FAISS 벡터 인덱스
- `edge_payloads_v1.npy` - 메타데이터

### 3. 답변 생성 🤖

구축된 그래프를 사용하여 질문에 대한 답변을 생성합니다.

**필요한 입력:** 
- 인덱싱된 데이터셋
- `[데이터셋]/qa.json` - 질문 파일

**출력:** `Result/Generated/` 폴더에 답변 결과

**모드:**
- 짧은 답변 (빠른 처리)
- 긴 답변 (상세한 처리)
- 대화형 모드 (실시간 질문)

### 3. 답변 평가 📊

생성된 답변을 골드 스탠다드와 비교하여 성능을 평가합니다.

**평가 지표:**
- **F1 스코어** - 정확도, 재현율, F1
- **UltraDomain 평가** - LLM 기반 품질 평가

**출력:** `Result/Evaluation/` 폴더에 평가 결과

## 💡 주요 특징

- **모듈화된 구조**: 각 기능별로 폴더 분리
- **크로스 플랫폼**: Windows, Linux, Mac 지원
- **대화형 인터페이스**: 사용하기 쉬운 메뉴 시스템
- **병렬 처리**: 멀티스레딩으로 빠른 처리
- **유연한 설정**: 다양한 옵션과 건너뛰기 기능
- **상세한 로깅**: 진행 상황과 오류 추적

## 📊 성능

- **처리 속도**: 멀티스레딩으로 빠른 그래프 구축
- **확장성**: 대용량 데이터셋 지원
- **정확도**: 고품질 트리플 추출 및 검색

## 🛠️ 개발자 가이드

### 개별 스크립트 실행

```bash
# 그래프 구축
python index/build_graph.py --dataset hotpotQA

# 답변 생성
python generate/answer_generation_short.py

# 평가
python evaluate/judge_F1.py
```

### 새로운 데이터셋 추가

1. `[데이터셋명]/contexts.txt` 생성
2. `[데이터셋명]/qa.json` 생성 (선택사항)
3. 인덱스 구축 스크립트 실행

## 📝 라이센스

Apache License 2.0

## 🤝 기여

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다!

---

더 자세한 가이드는 [SETUP_GUIDE.md](SETUP_GUIDE.md)를 참조하세요.
