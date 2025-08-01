@echo off
REM =============================================================================
REM KGRAG 답변 평가 스크립트 (Windows)
REM =============================================================================

echo 📊 KGRAG 답변 평가 시작
echo ========================================

REM 프로젝트 루트로 이동
cd /d "%~dp0\.."

REM 환경 변수 체크 (UltraDomain 평가용)
set "SKIP_ULTRADOMAIN=false"
if "%OPENAI_API_KEY%"=="" (
    echo ⚠️  OPENAI_API_KEY가 설정되지 않았습니다.
    echo F1 평가만 수행하고, UltraDomain 평가는 건너뜁니다.
    set "SKIP_ULTRADOMAIN=true"
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

REM 생성된 결과 파일 검색
echo 🔍 생성된 결과 파일 검색 중...
set "result_list="
set "result_count=0"

REM Result/Generated 폴더에서 검색
if exist "Result\Generated" (
    for %%f in ("Result\Generated\*.json") do (
        if exist "%%f" (
            set /a result_count+=1
            set "result_list=!result_list! %%f"
        )
    )
)

REM Result/Ours 폴더에서도 검색
if exist "Result\Ours" (
    for %%f in ("Result\Ours\*.json") do (
        if exist "%%f" (
            set /a result_count+=1
            set "result_list=!result_list! %%f"
        )
    )
)

if %result_count%==0 (
    echo ❌ 평가할 결과 파일을 찾을 수 없습니다.
    echo 먼저 generate\ 폴더의 스크립트를 사용하여 답변을 생성하세요.
    pause
    exit /b 1
)

REM 결과 파일 표시
echo 📁 사용 가능한 결과 파일:
set "current=0"
for %%f in (%result_list%) do (
    set /a current+=1
    for %%s in ("%%f") do echo   !current!. %%~nxf (%%~zs bytes)
)
echo   a. 모든 파일

set /p choice="평가할 결과 파일을 선택하세요 (번호 또는 'a'): "

REM 선택된 파일 목록 생성
if /i "%choice%"=="a" (
    set "selected_files=%result_list%"
) else (
    set "current=0"
    set "selected_files="
    for %%f in (%result_list%) do (
        set /a current+=1
        if !current!==%choice% (
            set "selected_files=%%f"
        )
    )
    
    if "!selected_files!"=="" (
        echo ❌ 잘못된 선택입니다.
        pause
        exit /b 1
    )
)

REM 평가 유형 선택
echo.
echo 📏 평가 유형:
echo   1. F1 스코어 평가 (자동)
echo   2. UltraDomain 평가 (LLM 기반)
echo   3. 둘 다

set /p eval_type="평가 유형을 선택하세요 (1-3): "

REM 평가 결과 디렉터리 생성
if not exist "Result\Evaluation" mkdir "Result\Evaluation"

REM 타임스탬프 생성
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "timestamp=%dt:~0,8%_%dt:~8,6%"

REM 각 파일에 대해 평가 수행
for %%f in (%selected_files%) do (
    echo.
    echo 🔄 [%%~nf] 평가 시작
    echo ==================================
    
    REM 골드 스탠다드 파일 찾기
    set "gold_file="
    echo %%f | findstr /i "hotpot" >nul && set "gold_file=hotpotQA\qa.json"
    echo %%f | findstr /i /c:"UltraDomain" >nul && set "gold_file=UltraDomain\Mix\qa.json"
    echo %%f | findstr /i /c:"Mix" >nul && set "gold_file=UltraDomain\Mix\qa.json"
    echo %%f | findstr /i /c:"mix" >nul && set "gold_file=UltraDomain\Mix\qa.json"
    echo %%f | findstr /i /c:"MultihopRAG" >nul && set "gold_file=MultihopRAG\qa.json"
    
    if "!gold_file!"=="" (
        echo ⚠️  골드 스탠다드 파일을 찾을 수 없습니다.
        goto :next_file
    )
    
    if not exist "!gold_file!" (
        echo ⚠️  골드 스탠다드 파일이 존재하지 않습니다: !gold_file!
        goto :next_file
    )
    
    REM F1 평가
    if "%eval_type%"=="1" (
        goto :f1_eval
    ) else if "%eval_type%"=="3" (
        goto :f1_eval
    ) else (
        goto :ultra_eval
    )
    
    :f1_eval
    echo 📊 F1 스코어 평가 중...
    
    python -c "
import sys
sys.path.append('evaluate')

# judge_F1.py 내용 읽기 및 경로 수정
with open('evaluate/judge_F1.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace('PRED_PATH = Path(\"Result/Ours/30_5.json\")', 'PRED_PATH = Path(r\"%%f\")')
code = code.replace('GOLD_PATH = Path(\"MultihopRAG/qa.json\")', 'GOLD_PATH = Path(r\"!gold_file!\")')

print('\\n=== F1 평가 결과: %%~nf ===')
exec(code)
" > "Result\Evaluation\%%~nf_f1_%timestamp%.txt"
    
    echo ✅ F1 평가 완료
    type "Result\Evaluation\%%~nf_f1_%timestamp%.txt"
    
    if "%eval_type%"=="1" goto :next_file
    
    :ultra_eval
    if "%SKIP_ULTRADOMAIN%"=="false" (
        echo 🧠 UltraDomain LLM 평가 중...
        echo 이 평가는 시간이 오래 걸릴 수 있습니다...
        
        python -c "
import sys
sys.path.append('evaluate')

# judge_Ultradomain.py 내용 읽기 및 경로 수정
with open('evaluate/judge_Ultradomain.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace('my_rag_path    = \"Result/Ours/mix_result.json\"', 'my_rag_path    = r\"%%f\"')
code = code.replace('other_rag_path = \"Result/PathRAG/mix_result.json\"', 'other_rag_path = r\"!gold_file!\"')

print('\\n=== UltraDomain 평가 결과: %%~nf ===')
exec(code)
" > "Result\Evaluation\%%~nf_ultradomain_%timestamp%.txt" 2>&1
        
        echo ✅ UltraDomain 평가 완료
        REM 마지막 20줄만 표시
        powershell -command "Get-Content 'Result\Evaluation\%%~nf_ultradomain_%timestamp%.txt' | Select-Object -Last 20"
    ) else (
        echo ⚠️  UltraDomain 평가 건너뜀 (API 키 없음)
    )
    
    :next_file
)

echo.
echo 🎊 모든 평가 완료!
echo.

REM 평가 결과 요약
echo 📊 평가 결과 요약:
echo 평가 결과 파일들이 다음 위치에 저장되었습니다:
for %%f in ("Result\Evaluation\*_%timestamp%*.txt") do (
    if exist "%%f" echo   ✅ %%f
)

echo.
echo ✨ 답변 평가가 성공적으로 완료되었습니다!
echo 자세한 결과는 Result\Evaluation\ 폴더에서 확인할 수 있습니다.

pause
