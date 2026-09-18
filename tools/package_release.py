"""打包一份可以发给别人的源码包（不含模型、记忆、密钥、venv）。

    python tools\\package_release.py

产物：项目根目录 dist/voicebot-src-日期.zip
对方解压后运行 setup_new_machine.bat 即可。
"""
from __future__ import annotations

import datetime
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # voicebot/
DIST = ROOT.parent / "dist"                        # 工作区/dist

INCLUDE_SUFFIX = {".py", ".bat", ".md", ".txt", ".json", ".yaml", ".yml",
                  ".html", ".css", ".js", ".jit", ".dll", ".onnx", ".so", ".dylib", ".ttf"}
EXCLUDE_FILES = {".env", "models.json", "lianlian_memory.db", "vtube_token.json",
                 "native_pet.log", "_pet_backup.json", "reply.wav", "_selftest_tts.wav"}
EXCLUDE_DIRS = {"__pycache__", "out", "jit-cache", ".venv", "_mpl", "_hfcache", "data"}


def is_wanted(path: Path) -> bool:
    """path 是相对于 voicebot/ 的路径。只排除根级目录，避免误伤 _vendor/.../data。"""
    if path.name in EXCLUDE_FILES:
        return False
    if path.parts and path.parts[0] in EXCLUDE_DIRS:
        return False
    if path.suffix.lower() not in INCLUDE_SUFFIX:
        return False
    return True


def portable_start_llama(text: str) -> str:
    out = []
    for ln in text.splitlines():
        st = ln.strip()
        if st.startswith('set "LLAMA=') and "llamacpp-vk" not in st:
            out.append('set "LLAMA=%~dp0..\\llamacpp-cu124"')
            continue
        if 'set "LLAMA=D:' in ln:            # 旧机器的绝对路径回退行，删掉
            continue
        if st.startswith('set "MODEL='):
            out.append('set "MODEL="')
            continue
        if st.startswith('set "LORA='):
            out.append('set "LORA="')
            continue
        out.append(ln)
    return "\n".join(out) + "\n"


def portable_start_gsv(text: str) -> str:
    out = []
    for ln in text.splitlines():
        if 'set "GSV=D:' in ln or 'set "PY=D:' in ln or 'set "M=D:' in ln:
            continue
        out.append(ln)
    return "\n".join(out) + "\n"


def example_models() -> dict:
    data = json.loads((ROOT / "models.json").read_text(encoding="utf-8"))
    data["_说明"] = ("复制成 models.json 后按本机路径修改；"
                     "llama-server 的路径由 run.bat 调 sync_models.py 自动写进 start_llama.bat")
    data["process"]["llama_exe"] = r"D:\path\to\llamacpp-cu124"
    data["process"]["gguf"] = r"D:\models\your-llm.gguf"
    data["process"]["lora"] = ""
    if "gsv" in data.get("tts", {}):
        data["tts"]["gsv"]["ref_audio"] = r"D:\models\ref.wav"
    data["memory"]["embed_model"] = r"D:\models\bge-base-zh"
    data["live2d"]["native_python"] = r"..\live2d-venv\Scripts\python.exe"
    data["live2d"]["native_model"] = r"D:\models\Live2D\your-model\model.model3.json"
    return data


def main() -> None:
    DIST.mkdir(exist_ok=True)
    example = example_models()
    (ROOT / "models.example.json").write_text(
        json.dumps(example, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    out = DIST / f"voicebot-src-{stamp}.zip"
    files = [p for p in ROOT.rglob("*")
             if p.is_file() and is_wanted(p.relative_to(ROOT))]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted(files):
            rel = path.relative_to(ROOT)
            if rel.as_posix() == "start_llama.bat":
                z.writestr("voicebot/start_llama.bat",
                           portable_start_llama(path.read_text(encoding="utf-8", errors="replace")))
            elif rel.as_posix() == "start_gsv.bat":
                z.writestr("voicebot/start_gsv.bat",
                           portable_start_gsv(path.read_text(encoding="utf-8", errors="replace")))
            else:
                z.write(path, f"voicebot/{rel.as_posix()}")
    size = out.stat().st_size / 1024 / 1024
    print(f"已生成：{out}  ({size:.1f} MB, {len(files) + 1} 个文件)")
    print("包含：源码/脚本/前端/_vendor/models.example.json/requirements/部署文档")
    print("不包含：.env、models.json、记忆库、token、out、jit-cache、模型文件")
    print("发给别人后：解压 -> 装 Python 3.12 -> setup_new_machine.bat")


if __name__ == "__main__":
    main()