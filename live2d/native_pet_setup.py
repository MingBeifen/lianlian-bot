"""一键搭建自渲染桌宠环境（live2d-venv）。由 setup_native_pet.bat 调用。

正常情况下就是建个 venv 装 live2d-py + pygame + glfw；
下面那段 sitecustomize 是给「受限 ACL 环境」（比如本项目开发时的沙箱）用的：
os.mkdir(mode=0o700) 会创建出自己都无权限的目录，导致 pip 失败，所以替换掉 mkdtemp。
普通电脑上这段也不会造成副作用。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # 工作区（live2d-venv 放这里）
VENV = ROOT / "live2d-venv"
PY = VENV / "Scripts" / "python.exe"
SITE = VENV / "Lib" / "site-packages"

SITECUSTOMIZE = '''"""临时修复：受限 ACL 环境下 tempfile.mkdtemp 自建目录无权限。"""
import os
import random
import string
import tempfile


def _mkdtemp(suffix=None, prefix=None, dir=None):
    suffix = suffix or ""
    prefix = prefix or "tmp"
    dir = dir or tempfile.gettempdir()
    for _ in range(100):
        name = prefix + "".join(random.choices(string.ascii_lowercase + string.digits, k=8)) + suffix
        path = os.path.join(dir, name)
        try:
            os.mkdir(path)
        except FileExistsError:
            continue
        return path
    raise FileExistsError("mkdtemp: no usable dir")


tempfile.mkdtemp = _mkdtemp
'''


def run(cmd: list) -> None:
    print(">", " ".join(str(c) for c in cmd))
    subprocess.check_call([str(c) for c in cmd])


def main() -> None:
    if not PY.exists():
        run([sys.executable, "-m", "venv", "--without-pip", VENV])
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
    if not (SITE / "pip").exists():
        run([PY, "-m", "ensurepip", "--upgrade"])
    run([PY, "-m", "pip", "install", "--upgrade",
         "live2d-py==0.7.0.4", "pygame==2.6.1", "glfw", "--no-cache-dir"])
    print("\n完成！把 .env 里 PET_BACKEND 改成 native 就能用了。")


if __name__ == "__main__":
    main()