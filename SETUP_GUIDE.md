# KGRAG 그래프 구축 가이드

이 프로젝트는 텍스트에서 지식 그래프를 구축하고 RAG(Retrieval-Augmented Generation)를 수행하는 시스템입니다.

## 🚀 빠른 시작

### 1. 환경 설정

```bash
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

### 2. 데이터 준비

각 데이터셋 폴더에 `contexts.txt` 파일을 준비하세요:

```
hotpotQA/
  └── contexts.txt
UltraDomain/
  ├── CS/
  │   └── contexts.txt
  ├── Agriculture/
  │   └── contexts.txt
  └── ...
```

### 3. 그래프 구축

#### 방법 1: 통합 스크립트 사용 (권장)

**Windows:**
```cmd
run_all.bat
```

**Linux/Mac:**
```bash
chmod +x run_all.sh
./run_all.sh
```

#### 방법 2: Python 스크립트 직접 사용

```bash
# 단일 데이터셋 처리
python build_graph.py --dataset hotpotQA

# 사용자 정의 입력 파일
python build_graph.py --dataset mydata --input /path/to/contexts.txt

# 특정 단계만 실행
python build_graph.py --dataset hotpotQA --skip-extraction  # 트리플 추출 건너뛰기
python build_graph.py --dataset hotpotQA --skip-gexf       # GEXF 변환 건너뛰기
python build_graph.py --dataset hotpotQA --skip-index      # 인덱스 생성 건너뛰기
```

#### 방법 3: 개별 스크립트 사용

```bash
# 1. 트리플 추출
python graph_construction.py

# 2. GEXF 변환
python json_to_gexf.py input.json output.gexf

# 3. 인덱스 생성
python edge_embedding.py
```

## 📁 출력 파일 구조

각 데이터셋 폴더에 다음 파일들이 생성됩니다:

```
dataset_name/
├── contexts.txt              # 입력 텍스트
├── graph_v1.json            # 추출된 트리플 (JSON)
├── graph_v1.gexf            # 그래프 파일 (GEXF)
├── graph_v1_processed.gexf  # 처리된 그래프 파일
├── edge_index_v1.faiss      # FAISS 벡터 인덱스
├── edge_payloads_v1.npy     # 메타데이터
└── kv_store_text_chunks.json # 텍스트 청크 저장소
```

## 🔧 RAG 시스템 사용

그래프가 구축되면 RAG 시스템을 사용할 수 있습니다:

```bash
# 짧은 답변용
python graph_based_rag_short.py

# 긴 답변용  
python graph_based_rag_long.py

# 답변 생성 (배치 처리)
python answer_generation_short.py
python answer_generation_long.py
```

## 📊 평가

```bash
# F1 스코어 평가
python judge_F1.py

# UltraDomain 평가
python judge_Ultradomain.py
```

## ⚠️ 주의사항

1. **OpenAI API 키**: 반드시 환경변수 또는 `.env` 파일에 설정
2. **메모리**: 대용량 데이터셋의 경우 충분한 RAM 필요
3. **인터넷 연결**: OpenAI API 호출을 위해 안정적인 인터넷 연결 필요
4. **처리 시간**: 데이터셋 크기에 따라 수분~수시간 소요 가능

## 🛠️ 문제 해결

### 일반적인 오류들:

1. **API 키 오류**
   ```
   ValueError: OPENAI_API_KEY env var required.
   ```
   → 환경변수 `OPENAI_API_KEY` 설정 확인

2. **패키지 누락**
   ```
   ModuleNotFoundError: No module named 'xxx'
   ```
   → `pip install -r requirements.txt` 실행

3. **메모리 부족**
   ```
   OutOfMemoryError
   ```
   → 더 작은 청크 크기 사용 또는 더 많은 RAM 확보

4. **파일 경로 오류**
   ```
   FileNotFoundError: contexts.txt
   ```
   → 입력 파일 경로 확인

## 📈 성능 최적화

- **병렬 처리**: `MAX_WORKERS` 값 조정
- **청크 크기**: `MAX_TOKENS` 값 조정  
- **배치 크기**: API 호출 배치 크기 조정
- **캐싱**: 중간 결과 캐싱 활용

## 🤝 기여

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다!

## 📄 라이센스

Apache License 2.0
