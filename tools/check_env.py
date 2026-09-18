"""voicebot 环境体检：Python 依赖 / CUDA / 音频设备 / models.json 里的模型和端口。

    python check_env.py

只检查，不改东西；有 [X] 就按提示修，修完再跑一次。
"""
from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # voicebot/
PROJECT = ROOT.parent                           # 工作区（模型/venv 所在）
MODELS = ROOT / "models.json"
OK, WARN, BAD = "[OK]", "[!]", "[X]"
results: list = []


def add(level: str, title: str, detail: str = "") -> None:
    results.append(level)
    print(f"{level} {title}" + (f" — {detail}" if detail else ""))


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def check_file(label: str, path: str, level: str = BAD, optional: bool = False) -> None:
    if not path:
        add(WARN if optional else BAD, label, "未配置")
        return
    p = Path(path)
    if p.exists():
        kind = "目录" if p.is_dir() else f"{p.stat().st_size / 1024 / 1024:.1f}MB"
        add(OK, label, f"{path}（{kind}）")
    else:
        add(WARN if optional else BAD, label, f"不存在：{path}")


def main() -> int:
    print("=" * 60)
    print("voicebot 环境检查")
    print("=" * 60)
    add(OK if sys.version_info >= (3, 10) else BAD,
        f"Python {sys.version.split()[0]}", "推荐 3.12" if sys.version_info[:2] != (3, 12) else "")

    required = ["numpy", "torch", "funasr", "modelscope", "transformers",
                "soundfile", "openai", "aiohttp", "websockets", "kokoro", "jieba"]
    missing = [p for p in required if importlib.util.find_spec(p) is None]
    add(OK if not missing else BAD, "Python 依赖",
        "缺少: " + ", ".join(missing) if missing else f"{len(required)} 个核心包已装")

    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
            add(OK, "CUDA", f"{name} / {mem:.1f}GB / torch {torch.__version__}")
        else:
            add(WARN, "CUDA 不可用", "ASR/TTS 会走 CPU；把 models.json 的 device 改成 cpu")
    except Exception as exc:
        add(BAD, "torch 导入失败", f"{type(exc).__name__}: {exc}")

    try:
        import sounddevice as sd
        devs = sd.query_devices()
        ins = [d for d in devs if d.get("max_input_channels", 0) > 0]
        outs = [d for d in devs if d.get("max_output_channels", 0) > 0]
        add(OK if ins and outs else WARN, "音频设备", f"输入 {len(ins)} 个 / 输出 {len(outs)} 个")
    except Exception:
        add(WARN, "sounddevice", "未安装（项目 _vendor 里自带，或 pip install sounddevice）")

    if not MODELS.exists():
        add(BAD, "models.json", "不存在：先运行 python tools\\gen_models.py，或从 models.example.json 复制")
        return finish()

    try:
        data = json.loads(MODELS.read_text(encoding="utf-8"))
        add(OK, "models.json", str(MODELS))
    except Exception as exc:
        add(BAD, "models.json 解析", f"{type(exc).__name__}: {exc}")
        return finish()

    proc = data.get("process") or {}
    check_file("LLM 基础模型", proc.get("gguf"))
    if proc.get("lora"):
        check_file("LLM LoRA", proc.get("lora"), optional=True)
    exe = Path(str(proc.get("llama_exe") or "")) / "llama-server.exe"
    check_file("llama-server.exe", str(exe), optional=True)
    port = int(proc.get("port") or 8080)
    add(OK if port_open(port) else WARN, f"llama-server 端口 {port}",
        "已就绪" if port_open(port) else "未启动（run.bat 会拉起）")

    asr = data.get("asr") or {}
    model = str(asr.get("model") or "")
    p = Path(model)
    local = bool(model) and p.exists()
    remote = "/" in model and not model.startswith(("D:", "C:"))
    add(OK if local or remote else WARN, "ASR 模型",
        f"{model}（本地）" if local else f"{model}（首次运行会自动下载）" if remote else "未配置")

    tts = data.get("tts") or {}
    backend = str(tts.get("backend") or "")
    if backend == "gsv":
        gsv = tts.get("gsv") or {}
        check_file("GSV 参考音频", gsv.get("ref_audio"))
        url = str(gsv.get("api_url") or "http://127.0.0.1:9880")
        hostport = url.split("//", 1)[-1].split("/", 1)[0]
        host, _, port_s = hostport.partition(":")
        alive = port_open(int(port_s or 80), host or "127.0.0.1")
        add(OK if alive else WARN, "GPT-SoVITS 服务", f"{url} " + ("在线" if alive else "未启动（run.bat 会拉起）"))
    elif backend == "kokoro":
        repo = str(tts.get("repo") or "")
        pp = Path(repo)
        add(OK if pp.exists() else WARN, "Kokoro 模型",
            f"{repo}（本地）" if pp.exists() else f"{repo}（首次自动下载）")
    else:
        add(OK, "TTS 后端", backend or "未配置")

    emb = str((data.get("memory") or {}).get("embed_model") or "")
    if emb:
        add(OK if Path(emb).exists() else WARN, "记忆嵌入模型",
            f"{emb}（本地）" if Path(emb).exists() else f"{emb}（首次自动下载）")
    else:
        add(WARN, "记忆嵌入模型", "未配置（默认走 HF 下载）")

    l2d = data.get("live2d") or {}
    if l2d.get("backend") == "native":
        check_file("Live2D 模型", l2d.get("native_model"))
        py = Path(str(l2d.get("native_python") or ""))
        if py.exists():
            add(OK, "渲染器 python", str(py))
            try:
                r = subprocess.run([str(py), "-c", "import live2d, glfw, OpenGL, PIL; print('ok')"],
                                   capture_output=True, text=True, timeout=40)
                add(OK if r.returncode == 0 else BAD, "live2d 依赖",
                    "ok" if r.returncode == 0 else (r.stderr or "")[-120:])
            except Exception as exc:
                add(WARN, "live2d 依赖检查", f"{type(exc).__name__}: {exc}")
        else:
            add(WARN, "渲染器 python", f"不存在：{py}（运行 setup_native_pet.bat）")
    else:
        add(OK, "Live2D 后端", str(l2d.get("backend") or ""))

    wport = int((data.get("webui") or {}).get("port") or 8765)
    add(OK if not port_open(wport) else WARN, f"Web 控制台端口 {wport}",
        "空闲" if not port_open(wport) else "被占用，改 models.json 的 webui.port")
    return finish()


def finish() -> int:
    bad = results.count(BAD)
    warn = results.count(WARN)
    print("-" * 60)
    print(f"结果：{results.count(OK)} 项通过，{warn} 项提醒，{bad} 项错误")
    if bad:
        print("有必须修的项（[X]），参考上面的路径/提示。")
    elif warn:
        print("基本可用，[!] 的项按需处理（模型首次会自动下载等）。")
    else:
        print("全部就绪，直接运行 run.bat。")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())