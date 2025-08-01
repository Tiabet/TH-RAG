#!/bin/bash

# =============================================================================
# KGRAG 그래프 인덱스 구축 스크립트
# =============================================================================
# 텍스트에서 지식 그래프를 구축하고 FAISS 인덱스를 생성합니다.
# =============================================================================

set -e  # 에러 발생시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🏗️  KGRAG 그래프 인덱스 구축 시작${NC}"
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

# 데이터셋 배열 정의
DATASETS=(
    "hotpotQA"
    "UltraDomain/Agriculture"
    "UltraDomain/CS"
    "UltraDomain/Mix"
    "UltraDomain/Legal"
    "MultihopRAG"
)

# 사용자 입력으로 데이터셋 선택
echo -e "${BLUE}📂 사용 가능한 데이터셋:${NC}"
for i in "${!DATASETS[@]}"; do
    echo "  $((i+1)). ${DATASETS[$i]}"
done
echo "  a. 모든 데이터셋"
echo "  c. 사용자 정의 경로"

read -p "인덱싱할 데이터셋을 선택하세요 (번호, 'a', 또는 'c'): " choice

if [ "$choice" = "a" ] || [ "$choice" = "A" ]; then
    SELECTED_DATASETS=("${DATASETS[@]}")
elif [ "$choice" = "c" ] || [ "$choice" = "C" ]; then
    read -p "데이터셋 경로를 입력하세요 (예: mydata): " custom_path
    if [ -z "$custom_path" ]; then
        echo -e "${RED}❌ 잘못된 경로입니다.${NC}"
        exit 1
    fi
    SELECTED_DATASETS=("$custom_path")
elif [[ "$choice" =~ ^[1-9][0-9]*$ ]] && [ "$choice" -le "${#DATASETS[@]}" ]; then
    SELECTED_DATASETS=("${DATASETS[$((choice-1))]}")
else
    echo -e "${RED}❌ 잘못된 선택입니다.${NC}"
    exit 1
fi

# 인덱싱 옵션 선택
echo ""
echo -e "${BLUE}🔧 인덱싱 옵션:${NC}"
echo "  1. 전체 파이프라인 (트리플 추출 → GEXF 변환 → 인덱스 생성)"
echo "  2. GEXF 변환부터 (기존 JSON 사용)"
echo "  3. 인덱스 생성만 (기존 GEXF 사용)"

read -p "옵션을 선택하세요 (1-3): " pipeline_choice

case $pipeline_choice in
    1) SKIP_EXTRACTION=false; SKIP_GEXF=false; SKIP_INDEX=false ;;
    2) SKIP_EXTRACTION=true; SKIP_GEXF=false; SKIP_INDEX=false ;;
    3) SKIP_EXTRACTION=true; SKIP_GEXF=true; SKIP_INDEX=false ;;
    *) echo -e "${RED}❌ 잘못된 선택입니다.${NC}"; exit 1 ;;
esac

# 각 데이터셋에 대해 인덱싱 실행
total_datasets=${#SELECTED_DATASETS[@]}
current=0

for dataset in "${SELECTED_DATASETS[@]}"; do
    current=$((current + 1))
    echo ""
    echo -e "${GREEN}🔄 [$current/$total_datasets] [$dataset] 인덱싱 시작${NC}"
    echo "=================================="
    
    # build_graph.py 실행
    python_args="index/build_graph.py --dataset $dataset"
    
    if [ "$SKIP_EXTRACTION" = true ]; then
        python_args="$python_args --skip-extraction"
    fi
    if [ "$SKIP_GEXF" = true ]; then
        python_args="$python_args --skip-gexf"
    fi
    if [ "$SKIP_INDEX" = true ]; then
        python_args="$python_args --skip-index"
    fi
    
    if python $python_args; then        
        echo -e "${GREEN}✅ [$dataset] 인덱싱 완료${NC}"
        
        # 결과 파일 크기 표시
        if [ -f "$dataset/edge_index_v1.faiss" ]; then
            index_size=$(du -h "$dataset/edge_index_v1.faiss" | cut -f1)
            echo -e "  📊 인덱스 크기: ${YELLOW}$index_size${NC}"
        fi
        if [ -f "$dataset/edge_payloads_v1.npy" ]; then
            payload_size=$(du -h "$dataset/edge_payloads_v1.npy" | cut -f1)
            echo -e "  📦 페이로드 크기: ${YELLOW}$payload_size${NC}"
        fi
    else
        echo -e "${RED}❌ [$dataset] 인덱싱 실패${NC}"
    fi
done

echo ""
echo -e "${GREEN}🎊 모든 데이터셋 인덱싱 완료!${NC}"
echo ""

# 통계 요약
echo -e "${BLUE}📊 인덱싱 결과 요약:${NC}"
total_size=0
for dataset in "${SELECTED_DATASETS[@]}"; do
    if [ -d "$dataset" ]; then
        echo "[$dataset]:"
        
        # 트리플 수 계산
        if [ -f "$dataset/graph_v1.json" ]; then
            triples_count=$(python -c "
import json, sys
try:
    with open('$dataset/graph_v1.json') as f:
        data = json.load(f)
    if isinstance(data, list):
        total = sum(len(json.loads(item.get('result', '[]'))) for item in data if isinstance(item, dict) and 'result' in item)
    else:
        total = len(data.get('triples', []))
    print(total)
except:
    print('N/A')
" 2>/dev/null)
            echo -e "  🔢 트리플 수: ${YELLOW}$triples_count${NC}"
        fi
        
        # 파일 크기
        if [ -f "$dataset/edge_index_v1.faiss" ]; then
            index_size=$(du -h "$dataset/edge_index_v1.faiss" | cut -f1)
            echo -e "  📊 인덱스: ${YELLOW}$index_size${NC}"
        fi
        if [ -f "$dataset/edge_payloads_v1.npy" ]; then
            payload_size=$(du -h "$dataset/edge_payloads_v1.npy" | cut -f1)
            echo -e "  📦 페이로드: ${YELLOW}$payload_size${NC}"
        fi
        echo ""
    fi
done

echo -e "${GREEN}✨ 그래프 인덱싱이 성공적으로 완료되었습니다!${NC}"
echo -e "이제 ${BLUE}generate/${NC} 폴더의 스크립트를 사용하여 답변을 생성할 수 있습니다."
