@echo off
REM =============================================================================
REM KGRAG 그래프 인덱스 구축 스크립트 (Windows)
REM =============================================================================

echo 🏗️  KGRAG 그래프 인덱스 구축 시작
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

REM 데이터셋 선택
echo 📂 사용 가능한 데이터셋:
echo   1. hotpotQA
echo   2. UltraDomain/Agriculture
echo   3. UltraDomain/CS
echo   4. UltraDomain/Mix
echo   5. UltraDomain/Legal
echo   6. MultihopRAG
echo   a. 모든 데이터셋
echo   c. 사용자 정의 경로

set /p choice="인덱싱할 데이터셋을 선택하세요 (번호, 'a', 또는 'c'): "

if /i "%choice%"=="a" (
    set "datasets=hotpotQA UltraDomain/Agriculture UltraDomain/CS UltraDomain/Mix UltraDomain/Legal MultihopRAG"
    set "dataset_count=6"
) else if /i "%choice%"=="c" (
    set /p custom_path="데이터셋 경로를 입력하세요 (예: mydata): "
    if "!custom_path!"=="" (
        echo ❌ 잘못된 경로입니다.
        pause
        exit /b 1
    )
    set "datasets=!custom_path!"
    set "dataset_count=1"
) else if "%choice%"=="1" (
    set "datasets=hotpotQA"
    set "dataset_count=1"
) else if "%choice%"=="2" (
    set "datasets=UltraDomain/Agriculture"
    set "dataset_count=1"
) else if "%choice%"=="3" (
    set "datasets=UltraDomain/CS"
    set "dataset_count=1"
) else if "%choice%"=="4" (
    set "datasets=UltraDomain/Mix"
    set "dataset_count=1"
) else if "%choice%"=="5" (
    set "datasets=UltraDomain/Legal"
    set "dataset_count=1"
) else if "%choice%"=="6" (
    set "datasets=MultihopRAG"
    set "dataset_count=1"
) else (
    echo ❌ 잘못된 선택입니다.
    pause
    exit /b 1
)

REM 인덱싱 옵션 선택
echo.
echo 🔧 인덱싱 옵션:
echo   1. 전체 파이프라인 (트리플 추출 → GEXF 변환 → 인덱스 생성)
echo   2. GEXF 변환부터 (기존 JSON 사용)
echo   3. 인덱스 생성만 (기존 GEXF 사용)

set /p pipeline_choice="옵션을 선택하세요 (1-3): "

set "skip_args="
if "%pipeline_choice%"=="2" (
    set "skip_args=--skip-extraction"
) else if "%pipeline_choice%"=="3" (
    set "skip_args=--skip-extraction --skip-gexf"
) else if not "%pipeline_choice%"=="1" (
    echo ❌ 잘못된 선택입니다.
    pause
    exit /b 1
)

REM 각 데이터셋 처리
set "current=0"
for %%d in (%datasets%) do (
    set /a current+=1
    echo.
    echo 🔄 [!current!/%dataset_count%] [%%d] 인덱싱 시작
    echo ==================================
    
    python index\build_graph.py --dataset "%%d" %skip_args%
    
    if errorlevel 1 (
        echo ❌ [%%d] 인덱싱 실패
    ) else (
        echo ✅ [%%d] 인덱싱 완료
        
        REM 결과 파일 크기 표시
        if exist "%%d\edge_index_v1.faiss" (
            for %%s in ("%%d\edge_index_v1.faiss") do echo   📊 인덱스 크기: %%~zs bytes
        )
        if exist "%%d\edge_payloads_v1.npy" (
            for %%s in ("%%d\edge_payloads_v1.npy") do echo   📦 페이로드 크기: %%~zs bytes
        )
    )
)

echo.
echo 🎊 모든 데이터셋 인덱싱 완료!
echo.
echo ✨ 그래프 인덱싱이 성공적으로 완료되었습니다!
echo 이제 generate\ 폴더의 스크립트를 사용하여 답변을 생성할 수 있습니다.

pause
