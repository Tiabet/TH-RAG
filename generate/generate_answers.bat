@echo off
REM =============================================================================
REM KGRAG 답변 생성 스크립트 (Windows)    python -c "
import sys, os
from pathlib import Path

# 프로젝트 루트로 이동
project_root = Path(__file__).parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

# generate 폴더를 경로에 추가
sys.path.insert(0, str(project_root / 'generate'))

# 파일 읽기 및 경로 수정
with open('generate/answer_generation_long.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace('input_path = \"hotpotQA/qa.json\"', 'input_path = \"%selected_dataset%/qa.json\"')
code = code.replace('output_path = \"Result/Ours/hotpot_30_5.json\"', 'output_path = \"Result/Generated/%selected_dataset%_long.json\"'.replace('/', '_'))
code = code.replace('chunk_log_path = \"Result/Ours/Chunks/used_chunks_1000_multihop.jsonl\"', 'chunk_log_path = \"Result/Generated/Chunks/%selected_dataset%_long_chunks.jsonl\"'.replace('/', '_'))

exec(code)
"====================================================================

echo 🤖 KGRAG 답변 생성 시작
echo ========================================

REM 프로젝트 루트로 이동
cd /d "%~dp0\.."

REM 환경 변수 체크
if "%OPENAI_API_KEY%"=="" (
    echo ❌ 오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.
    echo 다음 명령어로 설정하세요:
    echo set OPENAI_API_KEY=your-api-key-here
    pause
    exit /b 1
)

REM Python 가상환경 활성화
if exist "venv\Scripts\activate.bat" (
    echo 📦 Python 가상환경 활성화 중...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo 📦 Python 가상환경 활성화 중...
    call .venv\Scripts\activate.bat
)

REM 패키지 설치
echo 📋 필요한 패키지 설치 확인 중...
if exist "requirements.txt" (
    pip install -r requirements.txt >nul 2>&1
) else (
    echo ❌ requirements.txt 파일을 찾을 수 없습니다.
    pause
    exit /b 1
)

REM 사용 가능한 데이터셋 검색
echo 📂 인덱싱된 데이터셋 검색 중...
set "dataset_list="
set "dataset_count=0"

for /d %%d in (*) do (
    if exist "%%d\edge_index_v1.faiss" (
        if exist "%%d\edge_payloads_v1.npy" (
            set /a dataset_count+=1
            set "dataset_list=!dataset_list! %%d"
            echo   !dataset_count!. %%d
        )
    )
)

if %dataset_count%==0 (
    echo ❌ 인덱싱된 데이터셋을 찾을 수 없습니다.
    echo 먼저 index\ 폴더의 스크립트를 사용하여 그래프를 구축하세요.
    pause
    exit /b 1
)

REM 데이터셋 선택
set /p choice="답변을 생성할 데이터셋을 선택하세요 (번호): "

REM 선택된 데이터셋 찾기
set "current=0"
set "selected_dataset="
for %%d in (%dataset_list%) do (
    set /a current+=1
    if !current!==%choice% (
        set "selected_dataset=%%d"
    )
)

if "%selected_dataset%"=="" (
    echo ❌ 잘못된 선택입니다.
    pause
    exit /b 1
)

REM 답변 유형 선택
echo.
echo 📝 답변 생성 유형:
echo   1. 짧은 답변 (graph_based_rag_short.py)
echo   2. 긴 답변 (graph_based_rag_long.py)
echo   3. 대화형 모드 (단일 질문)

set /p answer_type="답변 유형을 선택하세요 (1-3): "

REM 출력 디렉터리 생성
if not exist "Result\Generated" mkdir "Result\Generated"
if not exist "Result\Generated\Chunks" mkdir "Result\Generated\Chunks"

if "%answer_type%"=="1" (
    echo 📝 짧은 답변 생성 시작
    
    REM answer_generation_short.py 실행
    python -c "
import sys, os
from pathlib import Path

# 프로젝트 루트로 이동
project_root = Path(__file__).parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

# generate 폴더를 경로에 추가
sys.path.insert(0, str(project_root / 'generate'))

# 파일 읽기 및 경로 수정
with open('generate/answer_generation_short.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace('input_path = \"hotpotQA/qa.json\"', 'input_path = \"%selected_dataset%/qa.json\"')
code = code.replace('output_path = \"Result/Ours/hotpot_30_5.json\"', 'output_path = \"Result/Generated/%selected_dataset%_short.json\"'.replace('/', '_'))
code = code.replace('chunk_log_path = \"Result/Ours/Chunks/used_chunks_1000_multihop.jsonl\"', 'chunk_log_path = \"Result/Generated/Chunks/%selected_dataset%_short_chunks.jsonl\"'.replace('/', '_'))

exec(code)
"
    
) else if "%answer_type%"=="2" (
    echo 📝 긴 답변 생성 시작
    
    REM answer_generation_long.py 실행
    python -c "
import sys, os
sys.path.append('generate')
os.chdir(r'%CD%')

# 파일 읽기 및 경로 수정
with open('generate/answer_generation_long.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace('input_path = \"UltraDomain/Mix/qa.json\"', 'input_path = \"%selected_dataset%/qa.json\"')
code = code.replace('output_path = \"Result/Ours/mix_result.json\"', 'output_path = \"Result/Generated/%selected_dataset%_long.json\"'.replace('/', '_'))
code = code.replace('chunk_log_path = \"Result/Ours/Chunks/used_chunks_mix.jsonl\"', 'chunk_log_path = \"Result/Generated/Chunks/%selected_dataset%_long_chunks.jsonl\"'.replace('/', '_'))

exec(code)
"

) else if "%answer_type%"=="3" (
    echo 💬 대화형 모드 시작
    echo 질문을 입력하세요 (종료하려면 'quit' 입력):
    
    :question_loop
    set /p question="질문: "
    if /i "%question%"=="quit" goto :end_interactive
    if /i "%question%"=="exit" goto :end_interactive
    
    if not "%question%"=="" (
        echo 🤔 답변 생성 중...
        
        python -c "
import sys, os
sys.path.append('generate')
os.chdir(r'%CD%')

# GraphRAG 클래스만 로드
with open('generate/graph_based_rag_short.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 경로 수정
code = code.replace('GEXF_PATH        = \"hotpotQA/graph_v1.gexf\"', 'GEXF_PATH        = \"%selected_dataset%/graph_v1.gexf\"')
code = code.replace('JSON_PATH        = \"hotpotQA/graph_v1.json\"', 'JSON_PATH        = \"%selected_dataset%/graph_v1.json\"')
code = code.replace('KV_JSON_PATH     = \"hotpotQA/kv_store_text_chunks.json\"', 'KV_JSON_PATH     = \"%selected_dataset%/kv_store_text_chunks.json\"')
code = code.replace('INDEX_PATH       = \"hotpotQA/edge_index_v1.faiss\"', 'INDEX_PATH       = \"%selected_dataset%/edge_index_v1.faiss\"')
code = code.replace('PAYLOAD_PATH     = \"hotpotQA/edge_payloads_v1.npy\"', 'PAYLOAD_PATH     = \"%selected_dataset%/edge_payloads_v1.npy\"')

# 실행 부분 제거
lines = code.split('\n')
filtered_lines = []
skip_example = False
for line in lines:
    if 'if __name__ == \"__main__\":' in line:
        skip_example = True
    if not skip_example:
        filtered_lines.append(line)

exec('\n'.join(filtered_lines))

# 질문 답변
rag = GraphRAG()
answer = rag.answer('%question%', top_k1=30, top_k2=5)
print(f'\n답변: {answer}\n')
"
    )
    
    goto :question_loop
    
    :end_interactive
    echo 👋 대화형 모드를 종료합니다.
    goto :end

) else (
    echo ❌ 잘못된 선택입니다.
    pause
    exit /b 1
)

echo.
echo ✅ 답변 생성 완료!
echo.

REM 결과 파일 확인
echo 📊 생성된 결과 파일:
for %%f in ("Result\Generated\*.json") do (
    if exist "%%f" (
        for %%s in ("%%f") do echo   ✅ %%f (%%~zs bytes)
    )
)

:end
echo.
echo ✨ 답변 생성이 성공적으로 완료되었습니다!
echo 결과는 Result\Generated\ 폴더에서 확인할 수 있습니다.

pause
