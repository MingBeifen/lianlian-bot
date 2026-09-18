@echo off
setlocal
chcp 65001 >nul

rem ===== 后端可执行文件 =====
rem CUDA 12.4 版（默认）：JIT 缓存预热后启动 6 秒、首字 0.05s
set "LLAMA=%~dp0..\llamacpp-cu124"
rem Vulkan 版（备用，不挑 CUDA 版本，启动 14 秒）：
rem set "LLAMA=%~dp0..\llamacpp-vk"

rem ===== 模型（run.bat 会按 models.json 自动写入）=====
set "MODEL="
set "LORA="
set "LORA_ARG="

rem CUDA 12.4 的 nvcc 不认识 sm_120，显卡靠驱动 JIT 编译 PTX。
rem 把 JIT 缓存固定到这里，避免被系统清理（清了就要重新编译约 30 秒）。
set "CUDA_CACHE_PATH=%~dp0jit-cache"
set "CUDA_CACHE_MAXSIZE=4294967296"

rem 已经在跑就跳过
powershell -NoProfile -Command "try{$c=New-Object Net.Sockets.TcpClient;$c.Connect('127.0.0.1',8080);$c.Dispose();exit 0}catch{exit 1}" >nul 2>&1
if %errorlevel%==0 (
  echo [llama.cpp] 8080 端口已在运行，跳过启动
  exit /b 0
)

if not exist "%LLAMA%\llama-server.exe" (
  echo [llama.cpp] 找不到 %LLAMA%\llama-server.exe
  exit /b 1
)

if "%MODEL%"=="" (
  echo [llama.cpp] models.json 里还没配置 process.gguf
  echo             编辑 voicebot\models.json 后运行 run.bat（会自动同步路径）
  exit /b 1
)
if not exist "%MODEL%" (
  echo [llama.cpp] 找不到模型文件：%MODEL%
  exit /b 1
)
if not "%LORA%"=="" set "LORA_ARG=--lora "%LORA%""

if not exist "%CUDA_CACHE_PATH%" (
  mkdir "%CUDA_CACHE_PATH%" >nul 2>&1
  echo [llama.cpp] 首次运行：要为 RTX 5060 现编译 CUDA 内核，会多等约 30 秒
  echo [llama.cpp] （只此一次，之后每次启动 6 秒）
)

echo [llama.cpp] 启动中 ...
start "llama-server" /min "%LLAMA%\llama-server.exe" -m "%MODEL%" %LORA_ARG% -ngl 99 -c 8192 --host 127.0.0.1 --port 8080 --jinja -t 6 --repeat-penalty 1.15 --repeat-last-n 256

powershell -NoProfile -Command "for($i=0;$i -lt 180;$i++){try{$c=New-Object Net.Sockets.TcpClient;$c.Connect('127.0.0.1',8080);$c.Dispose();exit 0}catch{Start-Sleep -Milliseconds 1000}};exit 1"
if %errorlevel%==0 (
  echo [llama.cpp] 就绪
) else (
  echo [llama.cpp] 启动超时 —— 检查显存是否被占满（Ollama 里的模型先 ollama stop 掉）
  exit /b 1
)
