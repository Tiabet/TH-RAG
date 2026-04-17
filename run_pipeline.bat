@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🚀 KGRAG 파이프라인 실행 도구
echo ===============================
echo.

REM 프로젝트 루트로 이동
cd /d "%~dp0"

REM Python 가상환경 확인 및 활성화
if exist "venv\Scripts\activate.bat" (
    echo 📦 가상환경 활성화 중...
    call venv\Scripts\activate.bat
    echo.
) else if exist ".venv\Scripts\activate.bat" (
    echo 📦 가상환경 활성화 중...
    call .venv\Scripts\activate.bat
    echo.
)

REM 설정 파일 확인
if not exist ".env" (
    echo ⚠️  .env 파일이 없습니다. 설정 파일을 생성하시겠습니까?
    set /p create_env=Y/N: 
    if /i "!create_env!"=="Y" (
        echo 📝 .env 파일 생성 중...
        python test_config.py --create-env
        echo.
        echo ✅ .env 파일이 생성되었습니다!
        echo ⚠️  .env 파일에서 OPENAI_API_KEY를 설정해주세요.
        echo.
        pause
        exit /b 0
    )
)

:menu
cls
echo 🚀 KGRAG 파이프라인 실행 도구
echo ===============================
echo.
echo 📋 작업을 선택하세요:
echo.
echo [1] 📂 사용 가능한 데이터셋 목록 보기
echo [2] 🔄 전체 파이프라인 실행 (그래프 구축 → 답변 생성 → 평가)
echo [3] 🏗️  그래프 구축만 실행
echo [4] 💬 답변 생성만 실행  
echo [5] 📊 평가만 실행
echo [6] 🔧 설정 확인
echo [7] ❓ 도움말
echo [0] ❌ 종료
echo.
set /p choice=선택하세요 (0-7): 

if "%choice%"=="0" goto :end
if "%choice%"=="1" goto :list_datasets
if "%choice%"=="2" goto :full_pipeline
if "%choice%"=="3" goto :build_graph
if "%choice%"=="4" goto :generate_answers
if "%choice%"=="5" goto :evaluate
if "%choice%"=="6" goto :check_config
if "%choice%"=="7" goto :help

echo ❌ 잘못된 선택입니다.
timeout /t 2 >nul
goto :menu

:list_datasets
echo.
echo 📂 사용 가능한 데이터셋:
python pipeline.py --list-datasets
echo.
pause
goto :menu

:full_pipeline
echo.
echo 🔄 전체 파이프라인 실행
echo =====================
echo.
python pipeline.py --list-datasets
echo.
set /p dataset=실행할 데이터셋 이름을 입력하세요: 
if "!dataset!"=="" (
    echo ❌ 데이터셋 이름을 입력해주세요.
    pause
    goto :menu
)

echo.
echo 🚀 전체 파이프라인 시작: !dataset!
echo.
python pipeline.py --dataset "!dataset!"
echo.
if %ERRORLEVEL% EQU 0 (
    echo ✅ 파이프라인이 성공적으로 완료되었습니다!
) else (
    echo ❌ 파이프라인 실행 중 오류가 발생했습니다.
)
pause
goto :menu

:build_graph
echo.
echo 🏗️  그래프 구축
echo ==============
echo.
python pipeline.py --list-datasets
echo.
set /p dataset=그래프를 구축할 데이터셋 이름: 
if "!dataset!"=="" (
    echo ❌ 데이터셋 이름을 입력해주세요.
    pause
    goto :menu
)

echo.
echo 🏗️  그래프 구축 시작: !dataset!
python pipeline.py --dataset "!dataset!" --steps graph_construction,json_to_gexf,edge_embedding
echo.
if %ERRORLEVEL% EQU 0 (
    echo ✅ 그래프 구축이 완료되었습니다!
) else (
    echo ❌ 그래프 구축 중 오류가 발생했습니다.
)
pause
goto :menu

:generate_answers
echo.
echo 💬 답변 생성
echo ============
echo.
python pipeline.py --list-datasets
echo.
set /p dataset=답변을 생성할 데이터셋 이름: 
if "!dataset!"=="" (
    echo ❌ 데이터셋 이름을 입력해주세요.
    pause
    goto :menu
)

echo.
echo 💬 답변 생성 시작: !dataset!
python pipeline.py --dataset "!dataset!" --steps answer_generation
echo.
if %ERRORLEVEL% EQU 0 (
    echo ✅ 답변 생성이 완료되었습니다!
) else (
    echo ❌ 답변 생성 중 오류가 발생했습니다.
)
pause
goto :menu

:evaluate
echo.
echo 📊 평가
echo =======
echo.
python pipeline.py --list-datasets
echo.
set /p dataset=평가할 데이터셋 이름: 
if "!dataset!"=="" (
    echo ❌ 데이터셋 이름을 입력해주세요.
    pause
    goto :menu
)

echo.
echo 📊 평가 시작: !dataset!
python pipeline.py --dataset "!dataset!" --steps evaluation
echo.
if %ERRORLEVEL% EQU 0 (
    echo ✅ 평가가 완료되었습니다!
) else (
    echo ❌ 평가 중 오류가 발생했습니다.
)
pause
goto :menu

:check_config
echo.
echo 🔧 설정 확인
echo ============
python test_config.py
echo.
pause
goto :menu

:help
echo.
echo ❓ KGRAG 파이프라인 도움말
echo =========================
echo.
echo 📖 KGRAG는 지식 그래프 기반 RAG 시스템입니다.
echo.
echo 🔧 주요 기능:
echo   • 텍스트에서 자동으로 지식 그래프 구축
echo   • FAISS 기반 고속 벡터 검색
echo   • GPT 모델을 활용한 답변 생성
echo   • 자동 성능 평가
echo.
echo 📁 데이터셋 준비:
echo   1. data/your_dataset/ 폴더 생성
echo   2. contexts.txt 파일에 텍스트 데이터 저장
echo   3. questions.txt 파일에 질문 목록 저장 (선택사항)
echo.
echo 🛠️  설정:
echo   • .env 파일에서 API 키 및 하이퍼파라미터 설정
echo   • test_config.py로 설정 확인 가능
echo.
echo 💡 명령줄 사용법:
echo   python pipeline.py --dataset your_dataset
echo   python pipeline.py --list-datasets
echo   python pipeline.py --dataset your_dataset --steps graph_construction
echo.
pause
goto :menu

:end
echo.
echo 👋 KGRAG 파이프라인 도구를 종료합니다.
echo.
