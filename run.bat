@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=D:\My-Neuro\lianlian-v0.1\venv_app\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

rem 按 models.json 同步 llama-server 的模型路径
if exist "%~dp0models.json" "%PY%" "%~dp0tools\sync_models.py" >nul 2>&1

rem 大模型后端没起就先起（llama.cpp 走 8080）
if "%LLM_BACKEND%"=="" call "%~dp0start_llama.bat"

rem 用 GPT-SoVITS 合成时，先把它的服务也拉起来（走 9880）
rem TTS 用不用 GPT-SoVITS 由 models.json/.env 决定
"%PY%" -c "from config import CONFIG; import sys; sys.exit(0 if CONFIG.tts_backend=='gsv' else 1)" >nul 2>&1
if %errorlevel%==0 call "%~dp0start_gsv.bat"

"%PY%" -u "%~dp0core\voice_chat.py" %*
