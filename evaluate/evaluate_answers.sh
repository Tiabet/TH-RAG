#!/bin/bash

# =============================================================================
# KGRAG 답변 평가 스크립트
# =============================================================================
# 생성된 답변을 골드 스탠다드와 비교하여 성능을 평가합니다.
# =============================================================================

set -e  # 에러 발생시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}📊 KGRAG 답변 평가 시작${NC}"
echo "========================================"

# 현재 스크립트 디렉터리로 이동
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 환경 변수 체크 (UltraDomain 평가용)
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  OPENAI_API_KEY가 설정되지 않았습니다.${NC}"
    echo "F1 평가만 수행하고, UltraDomain 평가는 건너뜁니다."
    SKIP_ULTRADOMAIN=true
else
    SKIP_ULTRADOMAIN=false
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

# 생성된 결과 파일 검색
echo -e "${BLUE}🔍 생성된 결과 파일 검색 중...${NC}"
RESULT_FILES=()

# Result/Generated 폴더에서 검색
if [ -d "Result/Generated" ]; then
    for file in Result/Generated/*.json; do
        if [ -f "$file" ]; then
            RESULT_FILES+=("$file")
        fi
    done
fi

# Result/Ours 폴더에서도 검색 (기존 결과)
if [ -d "Result/Ours" ]; then
    for file in Result/Ours/*.json; do
        if [ -f "$file" ]; then
            RESULT_FILES+=("$file")
        fi
    done
fi

if [ ${#RESULT_FILES[@]} -eq 0 ]; then
    echo -e "${RED}❌ 평가할 결과 파일을 찾을 수 없습니다.${NC}"
    echo "먼저 generate/ 폴더의 스크립트를 사용하여 답변을 생성하세요."
    exit 1
fi

# 결과 파일 선택
echo -e "${BLUE}📁 사용 가능한 결과 파일:${NC}"
for i in "${!RESULT_FILES[@]}"; do
    file="${RESULT_FILES[$i]}"
    filename=$(basename "$file")
    size=$(du -h "$file" | cut -f1)
    count=$(python -c "import json; print(len(json.load(open('$file'))))" 2>/dev/null || echo "N/A")
    echo "  $((i+1)). $filename (${YELLOW}$size${NC}, ${YELLOW}$count${NC} 답변)"
done
echo "  a. 모든 파일"

read -p "평가할 결과 파일을 선택하세요 (번호 또는 'a'): " choice

if [ "$choice" = "a" ] || [ "$choice" = "A" ]; then
    SELECTED_FILES=("${RESULT_FILES[@]}")
elif [[ "$choice" =~ ^[1-9][0-9]*$ ]] && [ "$choice" -le "${#RESULT_FILES[@]}" ]; then
    SELECTED_FILES=("${RESULT_FILES[$((choice-1))]}")
else
    echo -e "${RED}❌ 잘못된 선택입니다.${NC}"
    exit 1
fi

# 평가 유형 선택
echo ""
echo -e "${BLUE}📏 평가 유형:${NC}"
echo "  1. F1 스코어 평가 (자동)"
echo "  2. UltraDomain 평가 (LLM 기반)"
echo "  3. 둘 다"

read -p "평가 유형을 선택하세요 (1-3): " eval_type

# 평가 결과 디렉터리 생성
mkdir -p "Result/Evaluation"
timestamp=$(date +"%Y%m%d_%H%M%S")

# 각 파일에 대해 평가 수행
for result_file in "${SELECTED_FILES[@]}"; do
    filename=$(basename "$result_file" .json)
    echo ""
    echo -e "${GREEN}🔄 [$filename] 평가 시작${NC}"
    echo "=================================="
    
    # 해당하는 골드 스탠다드 파일 찾기
    gold_file=""
    if [[ "$filename" == *"hotpot"* ]]; then
        gold_file="hotpotQA/qa.json"
    elif [[ "$filename" == *"UltraDomain"* ]] || [[ "$filename" == *"Mix"* ]] || [[ "$filename" == *"mix"* ]]; then
        if [ -f "UltraDomain/Mix/qa.json" ]; then
            gold_file="UltraDomain/Mix/qa.json"
        elif [ -f "MultihopRAG/qa.json" ]; then
            gold_file="MultihopRAG/qa.json"
        fi
    elif [[ "$filename" == *"MultihopRAG"* ]]; then
        gold_file="MultihopRAG/qa.json"
    fi
    
    if [ -z "$gold_file" ] || [ ! -f "$gold_file" ]; then
        echo -e "${YELLOW}⚠️  골드 스탠다드 파일을 찾을 수 없습니다: $gold_file${NC}"
        continue
    fi
    
    # F1 평가
    if [ "$eval_type" = "1" ] || [ "$eval_type" = "3" ]; then
        echo -e "${BLUE}📊 F1 스코어 평가 중...${NC}"
        
        # judge_F1.py 실행 (경로 수정)
        python -c "
import sys
sys.path.append('evaluate')

# judge_F1.py 내용 읽기 및 경로 수정
with open('evaluate/judge_F1.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace('PRED_PATH = Path(\"Result/Ours/30_5.json\")', 'PRED_PATH = Path(\"$result_file\")')
code = code.replace('GOLD_PATH = Path(\"MultihopRAG/qa.json\")', 'GOLD_PATH = Path(\"$gold_file\")')

print(f'\\n=== F1 평가 결과: $filename ===')
exec(code)
" > "Result/Evaluation/${filename}_f1_${timestamp}.txt"
        
        echo -e "${GREEN}✅ F1 평가 완료${NC}"
        cat "Result/Evaluation/${filename}_f1_${timestamp}.txt"
    fi
    
    # UltraDomain 평가
    if [ "$eval_type" = "2" ] || [ "$eval_type" = "3" ]; then
        if [ "$SKIP_ULTRADOMAIN" = false ]; then
            echo -e "${BLUE}🧠 UltraDomain LLM 평가 중...${NC}"
            echo "이 평가는 시간이 오래 걸릴 수 있습니다..."
            
            # judge_Ultradomain.py 실행 (경로 수정)
            python -c "
import sys
sys.path.append('evaluate')

# judge_Ultradomain.py 내용 읽기 및 경로 수정
with open('evaluate/judge_Ultradomain.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace('my_rag_path    = \"Result/Ours/mix_result.json\"', 'my_rag_path    = \"$result_file\"')
code = code.replace('other_rag_path = \"Result/PathRAG/mix_result.json\"', 'other_rag_path = \"$gold_file\"')

print(f'\\n=== UltraDomain 평가 결과: $filename ===')
exec(code)
" > "Result/Evaluation/${filename}_ultradomain_${timestamp}.txt" 2>&1
            
            echo -e "${GREEN}✅ UltraDomain 평가 완료${NC}"
            tail -20 "Result/Evaluation/${filename}_ultradomain_${timestamp}.txt"
        else
            echo -e "${YELLOW}⚠️  UltraDomain 평가 건너뜀 (API 키 없음)${NC}"
        fi
    fi
done

echo ""
echo -e "${GREEN}🎊 모든 평가 완료!${NC}"
echo ""

# 평가 결과 요약
echo -e "${BLUE}📊 평가 결과 요약:${NC}"
echo "평가 결과 파일들이 다음 위치에 저장되었습니다:"
for file in Result/Evaluation/*_${timestamp}*.txt; do
    if [ -f "$file" ]; then
        echo -e "  ✅ ${YELLOW}$file${NC}"
    fi
done

echo ""
echo -e "${GREEN}✨ 답변 평가가 성공적으로 완료되었습니다!${NC}"
echo -e "자세한 결과는 ${BLUE}Result/Evaluation/${NC} 폴더에서 확인할 수 있습니다."
