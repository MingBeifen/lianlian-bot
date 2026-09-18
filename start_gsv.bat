@echo off
setlocal
chcp 65001 >nul

set "GSV=%~dp0..\gsv-repo\GPT-SoVITS-main"
set "PY=%~dp0..\gsv-venv\Scripts\python.exe"
set "M=%~dp0..\gsv-models"

rem api_v2.py 不会自己设这两个环境变量，必须在这里设（TTS_infer_pack 这条路径要）
set "bert_path=%M%\chinese-roberta-wwm-ext-large"
set "cnhubert_base_path=%M%\chinese-hubert-base"
rem 避开不可写的默认路径
set "MPLCONFIGDIR=%~dp0_mpl"
set "HF_HOME=%~dp0_hfcache"
set "PYTHONIOENCODING=utf-8"

rem 已经在跑就跳过
powershell -NoProfile -Command "try{$c=New-Object Net.Sockets.TcpClient;$c.Connect('127.0.0.1',9880);$c.Dispose();exit 0}catch{exit 1}" >nul 2>&1
if %errorlevel%==0 (
  echo [GPT-SoVITS] 9880 端口已在运行，跳过启动
  exit /b 0
)

if not exist "%PY%" ( echo [GPT-SoVITS] 找不到 %PY% & exit /b 1 )
if not exist "%GSV%\api_v2.py" ( echo [GPT-SoVITS] 找不到 %GSV%\api_v2.py & exit /b 1 )

echo [GPT-SoVITS] 启动中（加载模型约 10-30 秒）...
cd /d "%GSV%"
start "gpt-sovits-api" /min "%PY%" api_v2.py -a 127.0.0.1 -p 9880 -c "%~dp0config\gsv_infer.yaml"

powershell -NoProfile -Command "for($i=0;$i -lt 180;$i++){try{$c=New-Object Net.Sockets.TcpClient;$c.Connect('127.0.0.1',9880);$c.Dispose();exit 0}catch{Start-Sleep -Milliseconds 1000}};exit 1"
if %errorlevel%==0 (
  echo [GPT-SoVITS] 就绪
) else (
  echo [GPT-SoVITS] 启动超时 —— 多半是显存不够（先关掉 llama-server 试试）
  exit /b 1
)
