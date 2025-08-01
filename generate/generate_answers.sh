#!/bin/bash

# =============================================================================
# KGRAG 답변 생성 스크립트
# =============================================================================
# 구축된 지식 그래프를 사용하여 질문에 대한 답변을 생성합니다.
# =============================================================================

set -e  # 에러 발생시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🤖 KGRAG 답변 생성 시작${NC}"
echo "========================================"

# 현재 스크립트 디렉터리로 이동
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 환경 변수 체크
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}❌ 오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.${NC}"
    echo "다음 명령어로 설정하세요:"
    echo "export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

# Python 가상환경 활성화 (있는 경우)
if [ -d "venv" ]; then
    echo -e "${YELLOW}📦 Python 가상환경 활성화 중...${NC}"
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo -e "${YELLOW}📦 Python 가상환경 활성화 중...${NC}"
    source .venv/bin/activate
fi

# 필요한 패키지 설치 확인
echo -e "${YELLOW}📋 필요한 패키지 설치 확인 중...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt > /dev/null 2>&1
else
    echo -e "${RED}❌ requirements.txt 파일을 찾을 수 없습니다.${NC}"
    exit 1
fi

# 사용 가능한 데이터셋 검색
echo -e "${BLUE}📂 인덱싱된 데이터셋 검색 중...${NC}"
AVAILABLE_DATASETS=()
for dir in */; do
    if [ -f "${dir}edge_index_v1.faiss" ] && [ -f "${dir}edge_payloads_v1.npy" ]; then
        AVAILABLE_DATASETS+=("${dir%/}")
    fi
done

if [ ${#AVAILABLE_DATASETS[@]} -eq 0 ]; then
    echo -e "${RED}❌ 인덱싱된 데이터셋을 찾을 수 없습니다.${NC}"
    echo "먼저 index/ 폴더의 스크립트를 사용하여 그래프를 구축하세요."
    exit 1
fi

# 데이터셋 선택
echo -e "${BLUE}📊 사용 가능한 데이터셋:${NC}"
for i in "${!AVAILABLE_DATASETS[@]}"; do
    dataset="${AVAILABLE_DATASETS[$i]}"
    echo "  $((i+1)). $dataset"
    
    # 통계 정보 표시
    if [ -f "$dataset/qa.json" ]; then
        qa_count=$(python -c "import json; print(len(json.load(open('$dataset/qa.json'))))" 2>/dev/null || echo "N/A")
        echo -e "      질문 수: ${YELLOW}$qa_count${NC}"
    fi
done

read -p "답변을 생성할 데이터셋을 선택하세요 (번호): " choice

if [[ "$choice" =~ ^[1-9][0-9]*$ ]] && [ "$choice" -le "${#AVAILABLE_DATASETS[@]}" ]; then
    SELECTED_DATASET="${AVAILABLE_DATASETS[$((choice-1))]}"
else
    echo -e "${RED}❌ 잘못된 선택입니다.${NC}"
    exit 1
fi

# 답변 유형 선택
echo ""
echo -e "${BLUE}📝 답변 생성 유형:${NC}"
echo "  1. 짧은 답변 (graph_based_rag_short.py)"
echo "  2. 긴 답변 (graph_based_rag_long.py)"
echo "  3. 대화형 모드 (단일 질문)"

read -p "답변 유형을 선택하세요 (1-3): " answer_type

# 출력 디렉터리 생성
mkdir -p "Result/Generated"
mkdir -p "Result/Generated/Chunks"

case $answer_type in
    1)
        echo -e "${GREEN}📝 짧은 답변 생성 시작${NC}"
        
        # answer_generation_short.py 경로 수정하여 실행
        python -c "
import sys
sys.path.append('generate')
exec(open('generate/answer_generation_short.py').read().replace(
    'input_path = \"hotpotQA/qa.json\"', 
    'input_path = \"$SELECTED_DATASET/qa.json\"'
).replace(
    'output_path = \"Result/Ours/hotpot_30_5.json\"',
    'output_path = \"Result/Generated/${SELECTED_DATASET//\//_}_short.json\"'
).replace(
    'chunk_log_path = \"Result/Ours/Chunks/used_chunks_1000_multihop.jsonl\"',
    'chunk_log_path = \"Result/Generated/Chunks/${SELECTED_DATASET//\//_}_short_chunks.jsonl\"'
))
"
        ;;
    2)
        echo -e "${GREEN}📝 긴 답변 생성 시작${NC}"
        
        # answer_generation_long.py 경로 수정하여 실행
        python -c "
import sys
sys.path.append('generate')
exec(open('generate/answer_generation_long.py').read().replace(
    'input_path = \"UltraDomain/Mix/qa.json\"', 
    'input_path = \"$SELECTED_DATASET/qa.json\"'
).replace(
    'output_path = \"Result/Ours/mix_result.json\"',
    'output_path = \"Result/Generated/${SELECTED_DATASET//\//_}_long.json\"'
).replace(
    'chunk_log_path = \"Result/Ours/Chunks/used_chunks_mix.jsonl\"',
    'chunk_log_path = \"Result/Generated/Chunks/${SELECTED_DATASET//\//_}_long_chunks.jsonl\"'
))
"
        ;;
    3)
        echo -e "${GREEN}💬 대화형 모드 시작${NC}"
        echo "질문을 입력하세요 (종료하려면 'quit' 입력):"
        
        while true; do
            read -p "질문: " question
            if [ "$question" = "quit" ] || [ "$question" = "exit" ]; then
                break
            fi
            
            if [ -n "$question" ]; then
                echo -e "${YELLOW}🤔 답변 생성 중...${NC}"
                
                # 단일 질문에 대한 답변 생성
                python -c "
import sys
sys.path.append('generate')
from graph_based_rag_short import GraphRAG

# 데이터셋 경로 설정
import os
os.chdir('$PROJECT_ROOT')

# GraphRAG 인스턴스 생성 (경로 동적 설정)
exec(open('generate/graph_based_rag_short.py').read().replace(
    'GEXF_PATH        = \"hotpotQA/graph_v1.gexf\"',
    'GEXF_PATH        = \"$SELECTED_DATASET/graph_v1.gexf\"'
).replace(
    'JSON_PATH        = \"hotpotQA/graph_v1.json\"',
    'JSON_PATH        = \"$SELECTED_DATASET/graph_v1.json\"'
).replace(
    'KV_JSON_PATH     = \"hotpotQA/kv_store_text_chunks.json\"',
    'KV_JSON_PATH     = \"$SELECTED_DATASET/kv_store_text_chunks.json\"'
).replace(
    'INDEX_PATH       = \"hotpotQA/edge_index_v1.faiss\"',
    'INDEX_PATH       = \"$SELECTED_DATASET/edge_index_v1.faiss\"'
).replace(
    'PAYLOAD_PATH     = \"hotpotQA/edge_payloads_v1.npy\"',
    'PAYLOAD_PATH     = \"$SELECTED_DATASET/edge_payloads_v1.npy\"'
))

# 질문 답변
rag = GraphRAG()
answer = rag.answer('$question', top_k1=30, top_k2=5)
print(f'\\n답변: {answer}\\n')
"
            fi
        done
        
        echo -e "${GREEN}👋 대화형 모드를 종료합니다.${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}❌ 잘못된 선택입니다.${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✅ 답변 생성 완료!${NC}"
echo ""

# 결과 파일 확인
echo -e "${BLUE}📊 생성된 결과 파일:${NC}"
for file in Result/Generated/*.json; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        count=$(python -c "import json; print(len(json.load(open('$file'))))" 2>/dev/null || echo "N/A")
        echo -e "  ✅ $file (${YELLOW}$size${NC}, ${YELLOW}$count${NC} 답변)"
    fi
done

echo ""
echo -e "${GREEN}✨ 답변 생성이 성공적으로 완료되었습니다!${NC}"
echo -e "결과는 ${BLUE}Result/Generated/${NC} 폴더에서 확인할 수 있습니다."
