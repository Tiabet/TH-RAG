#!/bin/bash

# =============================================================================
# KGRAG 통합 실행 스크립트
# =============================================================================
# 이 스크립트는 KGRAG 시스템의 모든 기능에 대한 통합 인터페이스를 제공합니다.
# =============================================================================

set -e  # 에러 발생시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 KGRAG 통합 실행 스크립트${NC}"
echo "========================================"
echo ""
echo -e "${YELLOW}KGRAG (Knowledge Graph-based Retrieval Augmented Generation)${NC}"
echo "지식 그래프 기반 RAG 시스템"
echo ""

# 현재 디렉터리 확인
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ 프로젝트 루트 디렉터리에서 실행해주세요.${NC}"
    exit 1
fi

# 메뉴 표시
while true; do
    echo -e "${BLUE}� 사용 가능한 기능:${NC}"
    echo "  1. 🏗️  그래프 인덱스 구축 (Index Building)"
    echo "  2. 🤖 답변 생성 (Answer Generation)"
    echo "  3. 📊 답변 평가 (Answer Evaluation)"
    echo "  4. 📖 도움말 (Help)"
    echo "  5. 🚪 종료 (Exit)"
    echo ""
    
    read -p "원하는 기능을 선택하세요 (1-5): " choice
    
    case $choice in
        1)
            echo -e "${GREEN}🏗️  그래프 인덱스 구축을 시작합니다...${NC}"
            echo ""
            if [ -f "index/build_index.sh" ]; then
                chmod +x index/build_index.sh
                ./index/build_index.sh
            else
                echo -e "${RED}❌ index/build_index.sh 파일을 찾을 수 없습니다.${NC}"
            fi
            ;;
        2)
            echo -e "${GREEN}🤖 답변 생성을 시작합니다...${NC}"
            echo ""
            if [ -f "generate/generate_answers.sh" ]; then
                chmod +x generate/generate_answers.sh
                ./generate/generate_answers.sh
            else
                echo -e "${RED}❌ generate/generate_answers.sh 파일을 찾을 수 없습니다.${NC}"
            fi
            ;;
        3)
            echo -e "${GREEN}� 답변 평가를 시작합니다...${NC}"
            echo ""
            if [ -f "evaluate/evaluate_answers.sh" ]; then
                chmod +x evaluate/evaluate_answers.sh
                ./evaluate/evaluate_answers.sh
            else
                echo -e "${RED}❌ evaluate/evaluate_answers.sh 파일을 찾을 수 없습니다.${NC}"
            fi
            ;;
        4)
            echo -e "${BLUE}📖 KGRAG 도움말${NC}"
            echo "=================================="
            echo ""
            echo -e "${YELLOW}🏗️  그래프 인덱스 구축:${NC}"
            echo "  - 텍스트에서 지식 그래프를 구축하고 FAISS 인덱스를 생성합니다."
            echo "  - 필요한 입력: [데이터셋]/contexts.txt 파일"
            echo "  - 출력: JSON, GEXF, FAISS 인덱스 파일들"
            echo ""
            echo -e "${YELLOW}🤖 답변 생성:${NC}"
            echo "  - 구축된 그래프를 사용하여 질문에 대한 답변을 생성합니다."
            echo "  - 필요한 입력: 인덱싱된 데이터셋, [데이터셋]/qa.json 파일"
            echo "  - 출력: 답변 결과 JSON 파일"
            echo ""
            echo -e "${YELLOW}📊 답변 평가:${NC}"
            echo "  - 생성된 답변을 골드 스탠다드와 비교하여 성능을 평가합니다."
            echo "  - 평가 지표: F1 스코어, UltraDomain LLM 평가"
            echo ""
            echo -e "${YELLOW}📁 프로젝트 구조:${NC}"
            echo "  index/     - 그래프 구축 관련 스크립트"
            echo "  generate/  - 답변 생성 관련 스크립트"
            echo "  evaluate/  - 답변 평가 관련 스크립트"
            echo "  prompt/    - 프롬프트 템플릿"
            echo ""
            echo -e "${YELLOW}⚙️  환경 설정:${NC}"
            echo "  - OpenAI API 키: export OPENAI_API_KEY='your-key'"
            echo "  - Python 패키지: pip install -r requirements.txt"
            echo ""
            read -p "계속하려면 Enter를 누르세요..."
            ;;
        5)
            echo -e "${GREEN}👋 KGRAG 시스템을 종료합니다.${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ 잘못된 선택입니다. 1-5 사이의 숫자를 입력하세요.${NC}"
            ;;
    esac
    
    echo ""
    echo "=================================="
    echo ""
done