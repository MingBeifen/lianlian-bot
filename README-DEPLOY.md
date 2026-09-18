# 把 voicebot 部署到另一台电脑

这份文档对应 `voicebot/` 项目。已经在代码和脚本里做了路径相对化：
- 配置默认值不再写死 `D:\`；
- `models.json` 统一决定所有模型路径；
- `run.bat` 会自动找 `.venv` / `python`；
- `start_llama.bat` / `start_gsv.bat` 优先用项目旁边的目录。

## 0. 先决条件

| 项目 | 要求 |
|---|---|
| 系统 | Windows 10 / 11 x64 |
| Python | **3.12**（live2d-py 只有 cp312 的轮子；3.10/3.11 也能跑其余功能）|
| 显卡 | 建议 NVIDIA 8GB+（ASR/TTS/LLM 用 CUDA）；没有也能跑，见下文降级 |
| 磁盘 | 代码+依赖约 2~3GB；模型另算：LLM 约 5GB、GSV 约 3GB、bge 0.4GB、Kokoro 0.3GB |
| 内存 | 建议 16GB |

## 方式 A：源码包（推荐）

1. **打包**（在你现在的机器上）：
   ```bat
   cd voicebot
   python tools\\package_release.py
   ```
   产物在 `D:\DeepSeek-Harness\dist\voicebot-src-日期.zip`（约 11MB，含源码/脚本/前端/_vendor）。

2. 把 zip 拷到对方电脑，解压到任意目录（例如 `D:\voicebot\`）。

3. 安装 Python 3.12（官网版，安装时勾选 **Add python.exe to PATH**）。

4. 双击 `voicebot\setup_new_machine.bat`：
   - 自动建 `.venv`
   - `pip install -r requirements.txt`
   - 有 NVIDIA 显卡则装 CUDA（cu128）版 torch
   - 首次会把 `models.example.json` 复制成 `models.json`
   第一次安装依赖大约 10~30 分钟（视网速）。

5. **编辑 `models.json`**（关键一步）：
   - `process.llama_exe`：llama.cpp 目录（没有就改用 Ollama，见下）
   - `process.gguf` / `process.lora`：你的 LLM 模型文件
   - `asr.device` / `tts.device`：`cuda:0` 或 `cpu`
   - `tts`：用哪个后端（kokoro 最省事）
   - `memory.embed_model`：留空会自动从 HF 下载
   - `live2d.native_python`：默认 `..\live2d-venv\Scripts\python.exe`
   - `live2d.native_model`：你自己的 Live2D 模型路径

6. **准备模型**（见下一节），然后：
   ```bat
   python tools\\check_env.py
   ```
   直到没有 `[X]`，最后双击 `run.bat`，浏览器打开 http://127.0.0.1:8765 。

## 方式 B：整盘拷贝（省事但受限）

把整个 `D:\DeepSeek-Harness` 拷到对方电脑，**保持同样的盘符和路径**（很多默认值和 venv 的 `pyvenv.cfg` 认绝对路径）。可行的前提：

- 对方不要把目录放到别的盘/别的路径；
- Python / CUDA 驱动版本接近（torch cu128 需要较新的驱动，RTX 50 系必须）；
- 接受整包 ~8GB+ 的体积，并且注意不要把个人数据发给别人（见下）。

更稳的做法：只拷**模型和运行时目录**到同样路径，然后对方自己按方式 A 建 venv。

## 模型从哪来

| 模型 | 获取方式 |
|---|---|
| LLM（聊天）| **推荐 Ollama**：对方装 Ollama 拉一个模型（如 `qwen2.5:7b`），`models.json` 改 `llm.base_url=http://127.0.0.1:11434/v1`、`llm.model=...`，`process` 段不用管；<br>或者自己下 gguf（Qwen2.5-7B q4_k_m 等），把 `llamacpp-cu124/` 放到项目根目录，填 `process.gguf` |
| ASR（识别）| 不用手动准备：首次运行自动从 ModelScope 下载 SenseVoiceSmall（要联网）|
| 记忆嵌入 | 不填 `memory.embed_model` 就自动从 HF 下载 `bge-base-zh`（约 400MB）|
| TTS | `kokoro`：首次自动下载；`edge`：要联网；`gsv`：需要自己部署 GPT-SoVITS（`gsv-repo` + `gsv-venv` + `gsv-models`），参考 `start_gsv.bat` |
| Live2D | 模型有各自的授权，**不能随包分发**：让对方使用自己合法获得的 `.model3.json`（安装了 VTube Studio 的机器可在其 `Live2DModels` 目录找到自带的 hiyori）|

## 没有 NVIDIA 显卡怎么办

`models.json` 里改：

```json
{
  "asr":    { "device": "cpu" },
  "tts":    { "device": "cpu" },
  "llm":    { "base_url": "http://127.0.0.1:11434/v1", "model": "qwen2.5:3b" },
  "live2d": { "backend": "native" }
}
```

LLM 用 Ollama（CPU 也能跑小模型），Live2D 走 OpenGL 不挑显卡；延迟会从不到 1 秒变成几秒。

## 隐私与版权（发给别人前必看）

**不要打包/发送这些：**

- `voicebot/.env`（可能含 API Key）
- `voicebot/lianlian_memory.db`（你的聊天记忆）
- `voicebot/vtube_token.json`（VTS 授权 token）
- `voicebot/out/`、`jit-cache/`、日志
- `persona.txt` / LoRA（如果包含个人设定或隐私语料）
- **Live2D 模型、声音克隆参考音频、任何未经授权的语音数据**

`package_release.py` 已经自动排除前几项；模型和音频要你自己确认。

**依赖的许可证**（以各自仓库为准）：llama.cpp = MIT；Kokoro = Apache-2.0；bge = MIT；Qwen2.5 = Apache-2.0（小模型有额外条款）；GPT-SoVITS = MIT；Live2D Cubism SDK / 模型另需遵守 Live2D 的许可。

## 常见问题

- **`check_env.py` 报 [X]**：按提示修路径，改完再跑一次。
- **端口被占用**：改 `models.json` 的 `process.port` / `webui.port`。
- **首次运行很慢**：在下载 SenseVoice / Kokoro / bge，属于正常；之后再启动 20~40 秒。
- **Live2D 窗口不显示**：确认装了显卡驱动；或者改 `live2d.backend=vtube` 用 VTube Studio 渲染。
- **改了 `models.json` 不生效**：要重启 `run.bat`；llama-server 的路径由 `sync_models.py` 在启动时同步。
- **想要更小/更快**：可以删掉 `gsv` 相关目录，改 `tts.backend=kokoro` 或 `edge`。