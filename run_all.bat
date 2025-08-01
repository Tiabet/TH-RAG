@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🚀 KGRAG 통합 실행 도구
echo ========================
echo.

REM 프로젝트 루트로 이동
cd /d "%~dp0"

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

:menu
echo.
echo 📋 실행할 작업을 선택하세요:
echo [1] 📊 그래프 인덱스 구축 (Index Building)
echo [2] 💬 답변 생성 (Answer Generation)
echo [3] 📊 답변 평가 (Answer Evaluation) 
echo [4] 🔄 전체 파이프라인 실행 (Full Pipeline)
echo [5] 🛠️  시스템 설정 (System Setup)
echo [0] ❌ 종료 (Exit)
echo.
set /p choice=선택하세요 (0-5): 

if "%choice%"=="1" (
    echo.
    echo 🔄 그래프 인덱스 구축 시작...
    call index\build_index.bat
    if errorlevel 1 (
        echo ❌ 인덱스 구축 실패
        pause
        goto menu
    )
    echo ✅ 인덱스 구축 완료
    pause
    goto menu
) else if "%choice%"=="2" (
    echo.
    echo 🔄 답변 생성 시작...
    call generate\generate_answers.bat
    if errorlevel 1 (
        echo ❌ 답변 생성 실패
        pause
        goto menu
    )
    echo ✅ 답변 생성 완료
    pause
    goto menu
) else if "%choice%"=="3" (
    echo.
    echo 🔄 답변 평가 시작...
    call evaluate\evaluate_answers.bat
    if errorlevel 1 (
        echo ❌ 답변 평가 실패
        pause
        goto menu
    )
    echo ✅ 답변 평가 완료
    pause
    goto menu
) else if "%choice%"=="4" (
    echo.
    echo 🔄 전체 파이프라인 실행 시작...
    echo.
    echo 1단계: 그래프 인덱스 구축
    call index\build_index.bat
    if errorlevel 1 (
        echo ❌ 인덱스 구축 실패
        pause
        goto menu
    )
    echo.
    echo 2단계: 답변 생성
    call generate\generate_answers.bat
    if errorlevel 1 (
        echo ❌ 답변 생성 실패
        pause
        goto menu
    )
    echo.
    echo 3단계: 답변 평가
    call evaluate\evaluate_answers.bat
    if errorlevel 1 (
        echo ❌ 답변 평가 실패
        pause
        goto menu
    )
    echo ✅ 전체 파이프라인 완료
    pause
    goto menu
) else if "%choice%"=="5" (
    echo.
    echo 🛠️ 시스템 설정
    echo ================
    echo.
    
    REM 디렉터리 구조 확인
    echo 📁 프로젝트 디렉터리 구조:
    if exist "index\" (
        echo   ✅ index\ 폴더 존재
    ) else (
        echo   ❌ index\ 폴더 없음
        mkdir index 2>nul
        echo   📁 index\ 폴더 생성
    )
    
    if exist "generate\" (
        echo   ✅ generate\ 폴더 존재
    ) else (
        echo   ❌ generate\ 폴더 없음
        mkdir generate 2>nul
        echo   📁 generate\ 폴더 생성
    )
    
    if exist "evaluate\" (
        echo   ✅ evaluate\ 폴더 존재
    ) else (
        echo   ❌ evaluate\ 폴더 없음
        mkdir evaluate 2>nul
        echo   📁 evaluate\ 폴더 생성
    )
    
    if exist "Result\" (
        echo   ✅ Result\ 폴더 존재
    ) else (
        echo   ❌ Result\ 폴더 없음
        mkdir Result\Generated\Chunks 2>nul
        echo   📁 Result\ 폴더 생성
    )
    
    REM Python 환경 확인
    echo.
    echo 🐍 Python 환경:
    python --version 2>nul
    if errorlevel 1 (
        echo   ❌ Python이 설치되어 있지 않습니다
    ) else (
        echo   ✅ Python 설치됨
    )
    
    REM 필수 패키지 확인
    echo.
    echo 📦 패키지 재설치...
    pip install -r requirements.txt
    
    echo.
    echo ✅ 시스템 설정 완료
    pause
    goto menu
) else if "%choice%"=="0" (
    echo.
    echo 👋 KGRAG 도구를 종료합니다.
    exit /b 0
) else (
    echo ❌ 잘못된 선택입니다. 다시 선택해주세요.
    pause
    goto menu
)
