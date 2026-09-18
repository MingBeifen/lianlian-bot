@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=D:\My-Neuro\lianlian-v0.1\venv_app\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" "%~dp0live2d\native_pet_setup.py"
pause
