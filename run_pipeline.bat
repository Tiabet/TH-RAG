@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

if not exist ".env" (
    echo Missing .env. Create one from .env.example first.
    echo Run: python test_config.py --create-env
    pause
    exit /b 1
)

:menu
cls
echo TH-RAG Pipeline Helper
echo ======================
echo.
echo [1] List datasets
echo [2] Run full pipeline
echo [3] Build graph artifacts
echo [4] Generate short answers
echo [5] Generate long answers
echo [6] Run F1 evaluation
echo [7] Validate configuration
echo [0] Exit
echo.
set /p choice=Select an option (0-7): 

if "%choice%"=="0" goto :end
if "%choice%"=="1" goto :list_datasets
if "%choice%"=="2" goto :full_pipeline
if "%choice%"=="3" goto :graph_build
if "%choice%"=="4" goto :short_answers
if "%choice%"=="5" goto :long_answers
if "%choice%"=="6" goto :evaluation
if "%choice%"=="7" goto :config_check

echo Invalid selection.
pause
goto :menu

:list_datasets
python pipeline.py --list-datasets
pause
goto :menu

:full_pipeline
set /p dataset=Dataset name: 
if "!dataset!"=="" goto :menu
python pipeline.py --dataset "!dataset!"
pause
goto :menu

:graph_build
set /p dataset=Dataset name: 
if "!dataset!"=="" goto :menu
python pipeline.py --dataset "!dataset!" --steps graph_build
pause
goto :menu

:short_answers
set /p dataset=Dataset name: 
if "!dataset!"=="" goto :menu
python pipeline.py --dataset "!dataset!" --steps answer_generation_short
pause
goto :menu

:long_answers
set /p dataset=Dataset name: 
if "!dataset!"=="" goto :menu
python pipeline.py --dataset "!dataset!" --steps answer_generation_long
pause
goto :menu

:evaluation
set /p dataset=Dataset name: 
if "!dataset!"=="" goto :menu
python pipeline.py --dataset "!dataset!" --steps evaluation
pause
goto :menu

:config_check
python test_config.py
pause
goto :menu

:end
endlocal
