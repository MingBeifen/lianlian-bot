@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=python"
where py >nul 2>&1 && set "PY=py -3.12"
%PY% "%~dp0tools\setup_new_machine.py" %*
pause
