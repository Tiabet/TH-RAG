@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🚀 KGRAG 통합 파이프라인 도구
echo ================================
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
echo 📋 필요한 패키지 확인 중...
if exist "requirements.txt" (
    pip install -q -r requirements.txt
) else (
    echo ❌ requirements.txt 파일을 찾을 수 없습니다.
    pause
    exit /b 1
)

:menu
echo.
echo 📋 작업을 선택하세요:
echo [1] 📂 사용 가능한 데이터셋 목록 보기
echo [2] 🔄 전체 파이프라인 실행 (모든 단계)
echo [3] 📊 그래프 구축만 실행
echo [4] 💬 답변 생성만 실행
echo [5] 📊 평가만 실행
echo [6] 🛠️  개별 단계 선택 실행
echo [7] 🔧 고급 옵션
echo [0] ❌ 종료
echo.
set /p choice=선택하세요 (0-7): 

if "%choice%"=="1" (
    echo.
    echo 📂 사용 가능한 데이터셋 목록:
    python pipeline.py --list-datasets
    echo.
    pause
    goto menu
) else if "%choice%"=="2" (
    echo.
    echo 📂 처리할 데이터셋을 입력하세요:
    set /p dataset_name=데이터셋 이름: 
    if "!dataset_name!"=="" (
        echo ❌ 데이터셋 이름을 입력해야 합니다.
        pause
        goto menu
    )
    
    echo.
    echo 🔄 전체 파이프라인 실행 중...
    python pipeline.py --dataset "!dataset_name!"
    
    if errorlevel 1 (
        echo ❌ 파이프라인 실행 실패
    ) else (
        echo ✅ 파이프라인 실행 완료
    )
    echo.
    pause
    goto menu
) else if "%choice%"=="3" (
    echo.
    echo 📂 처리할 데이터셋을 입력하세요:
    set /p dataset_name=데이터셋 이름: 
    if "!dataset_name!"=="" (
        echo ❌ 데이터셋 이름을 입력해야 합니다.
        pause
        goto menu
    )
    
    echo.
    echo 📊 그래프 구축 실행 중...
    python pipeline.py --dataset "!dataset_name!" --steps graph_construction json_to_gexf edge_embedding
    
    if errorlevel 1 (
        echo ❌ 그래프 구축 실패
    ) else (
        echo ✅ 그래프 구축 완료
    )
    echo.
    pause
    goto menu
) else if "%choice%"=="4" (
    echo.
    echo 📂 처리할 데이터셋을 입력하세요:  
    set /p dataset_name=데이터셋 이름: 
    if "!dataset_name!"=="" (
        echo ❌ 데이터셋 이름을 입력해야 합니다.
        pause
        goto menu
    )
    
    echo.
    echo 💬 답변 생성 타입을 선택하세요:
    echo [1] Short answers only
    echo [2] Long answers only  
    echo [3] Both short and long
    set /p answer_type=선택 (1-3): 
    
    if "!answer_type!"=="1" (
        python pipeline.py --dataset "!dataset_name!" --steps answer_generation_short
    ) else if "!answer_type!"=="2" (
        python pipeline.py --dataset "!dataset_name!" --steps answer_generation_long
    ) else if "!answer_type!"=="3" (
        python pipeline.py --dataset "!dataset_name!" --steps answer_generation_short answer_generation_long
    ) else (
        echo ❌ 잘못된 선택입니다.
        pause
        goto menu
    )
    
    if errorlevel 1 (
        echo ❌ 답변 생성 실패
    ) else (
        echo ✅ 답변 생성 완료
    )
    echo.
    pause
    goto menu
) else if "%choice%"=="5" (
    echo.
    echo 📂 평가할 데이터셋을 입력하세요:
    set /p dataset_name=데이터셋 이름: 
    if "!dataset_name!"=="" (
        echo ❌ 데이터셋 이름을 입력해야 합니다.
        pause
        goto menu
    )
    
    echo.
    echo 📊 평가 실행 중...
    python pipeline.py --dataset "!dataset_name!" --steps evaluation_f1
    
    if errorlevel 1 (
        echo ❌ 평가 실패
    ) else (
        echo ✅ 평가 완료
    )
    echo.
    pause
    goto menu
) else if "%choice%"=="6" (
    echo.
    echo 📂 처리할 데이터셋을 입력하세요:
    set /p dataset_name=데이터셋 이름: 
    if "!dataset_name!"=="" (
        echo ❌ 데이터셋 이름을 입력해야 합니다.
        pause
        goto menu
    )
    
    echo.
    echo 🛠️ 실행할 단계를 선택하세요 (쉼표로 구분):
    echo 사용 가능한 단계:
    echo   - graph_construction: QA 데이터 생성
    echo   - json_to_gexf: 그래프 형식 변환
    echo   - edge_embedding: 엣지 임베딩 생성
    echo   - answer_generation_short: 짧은 답변 생성
    echo   - answer_generation_long: 긴 답변 생성  
    echo   - evaluation_f1: F1 스코어 평가
    echo.
    set /p selected_steps=단계들 (예: graph_construction,edge_embedding): 
    
    if "!selected_steps!"=="" (
        echo ❌ 단계를 입력해야 합니다.
        pause
        goto menu
    )
    
    REM 쉼표를 공백으로 변환
    set "selected_steps=!selected_steps:,= !"
    
    echo.
    echo 🔄 선택된 단계들 실행 중...
    python pipeline.py --dataset "!dataset_name!" --steps !selected_steps!
    
    if errorlevel 1 (
        echo ❌ 선택된 단계 실행 실패
    ) else (
        echo ✅ 선택된 단계 실행 완료
    )
    echo.
    pause
    goto menu
) else if "%choice%"=="7" (
    echo.
    echo 🔧 고급 옵션:
    echo [1] 강제 재실행 (기존 결과 무시)
    echo [2] 파이프라인 상태 확인
    echo [3] 결과 파일 정리
    echo [0] 메인 메뉴로 돌아가기
    echo.
    set /p adv_choice=선택하세요 (0-3): 
    
    if "!adv_choice!"=="1" (
        echo.
        echo 📂 강제 재실행할 데이터셋을 입력하세요:
        set /p dataset_name=데이터셋 이름: 
        if "!dataset_name!"=="" (
            echo ❌ 데이터셋 이름을 입력해야 합니다.
            pause
            goto menu
        )
        
        echo.
        echo 🔄 강제 재실행 중 (기존 결과 무시)...
        python pipeline.py --dataset "!dataset_name!" --force
        
        if errorlevel 1 (
            echo ❌ 강제 재실행 실패
        ) else (
            echo ✅ 강제 재실행 완료
        )
        echo.
        pause
        goto menu
    ) else if "!adv_choice!"=="2" (
        echo.
        echo 📊 파이프라인 상태 확인:
        if exist "temp\pipeline_state.json" (
            echo 현재 파이프라인 상태:
            type "temp\pipeline_state.json"
        ) else (
            echo 📂 아직 실행된 파이프라인이 없습니다.
        )
        echo.
        pause
        goto menu
    ) else if "!adv_choice!"=="3" (
        echo.
        echo 🧹 결과 파일 정리:
        echo 이 작업은 모든 생성된 결과를 삭제합니다.
        set /p confirm=정말 삭제하시겠습니까? (y/N): 
        if "!confirm!"=="y" (
            if exist "results\" rmdir /s /q "results\"
            if exist "temp\" rmdir /s /q "temp\"
            echo ✅ 결과 파일 정리 완료
        ) else (
            echo ❌ 취소되었습니다.
        )
        echo.
        pause
        goto menu
    ) else (
        goto menu
    )
) else if "%choice%"=="0" (
    echo.
    echo 👋 KGRAG 파이프라인 도구를 종료합니다.
    exit /b 0
) else (
    echo ❌ 잘못된 선택입니다. 다시 선택해주세요.
    pause
    goto menu
)
