"""在另一台电脑上初始化 voicebot 环境。

    python tools\\setup_new_machine.py          # 建 .venv + 装依赖 + 体检
    python tools\\setup_new_machine.py --dry-run
    python setup_new_machine.py --skip-torch    # 跳过 CUDA 版 torch
    python setup_new_machine.py --mirror https://pypi.tuna.tsinghua.edu.cn/simple

前提：已安装 Python 3.12（live2d-py 只有 cp312 轮子）。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # voicebot/


def run(cmd: list, dry: bool) -> None:
    print(">", " ".join(str(c) for c in cmd))
    if not dry:
        subprocess.check_call([str(c) for c in cmd])


def main() -> None:
    ap = argparse.ArgumentParser(description="voicebot 新机器初始化")
    ap.add_argument("--venv", default=".venv", help="虚拟环境目录（默认 voicebot/.venv）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-venv", action="store_true")
    ap.add_argument("--skip-torch", action="store_true")
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--mirror", default="", help="pip 镜像，例如清华源")
    ap.add_argument("--torch-index", default="https://download.pytorch.org/whl/cu128",
                    help="torch 的 CUDA 轮子源（默认 cu128，适配 RTX 50 系）")
    args = ap.parse_args()

    print("=" * 60)
    print("voicebot 新机器初始化")
    print(f"Python: {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info[:2] != (3, 12):
        print("[!] 建议用 Python 3.12（live2d-py 只有 cp312 的轮子）")
    if sys.version_info < (3, 10):
        raise SystemExit("[X] Python 太旧，请装 3.12")

    venv = ROOT / args.venv
    py = venv / "Scripts" / "python.exe" if sys.platform == "win32" else venv / "bin" / "python"

    if not args.skip_venv and not py.exists():
        run([sys.executable, "-m", "venv", str(venv)], args.dry_run)
    if not args.skip_install:
        pip = [str(py), "-m", "pip"]
        mirror = ["-i", args.mirror] if args.mirror else []
        run(pip + ["install", "--upgrade", "pip"] + mirror, args.dry_run)
        run(pip + ["install", "-r", str(ROOT / "requirements.txt")] + mirror, args.dry_run)
    if (not args.skip_torch and not args.skip_install
            and shutil.which("nvidia-smi")):
        print("[i] 检测到 NVIDIA 显卡，装 CUDA 版 torch/torchaudio")
        run([str(py), "-m", "pip", "install", "--upgrade", "torch", "torchaudio",
             "--index-url", args.torch_index], args.dry_run)
    elif not args.skip_torch and not args.skip_install:
        print("[i] 没检测到 nvidia-smi，torch 用 pip 默认版（CPU，会比较慢）")

    example = ROOT / "models.example.json"
    target = ROOT / "models.json"
    if example.exists() and not target.exists():
        print(f"> 复制 {example.name} -> {target.name}（记得改成本机路径）")
        if not args.dry_run:
            shutil.copy(example, target)

    print()
    print("接下来：")
    print(f"  1. 编辑 {target}（模型路径、后端、设备）")
    print(f"  2. {py} {ROOT / 'tools' / 'check_env.py'}  直到没有 [X]")
    print("  3. 双击 run.bat 启动（首次会自动下载 ASR/嵌入模型）")
    print(f"  4. 浏览器打开 http://127.0.0.1:8765")


if __name__ == "__main__":
    main()