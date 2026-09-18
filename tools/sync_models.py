"""把 models.json 里的 llama-server 配置同步进 start_llama.bat。

run.bat 每次启动会自动调用；一般不用手动跑。
    python sync_models.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # voicebot/
MODELS = ROOT / "models.json"
BAT = ROOT / "start_llama.bat"


def set_bat_var(text: str, var: str, value: str) -> tuple:
    pattern = rf'(set\s+"{re.escape(var)}=)[^"]*(")'
    new_text, n = re.subn(pattern, lambda m: m.group(1) + value + m.group(2), text, count=1)
    return new_text, bool(n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not MODELS.exists() or not BAT.exists():
        print("[sync] 缺 models.json 或 start_llama.bat，跳过")
        return
    data = json.loads(MODELS.read_text(encoding="utf-8"))
    proc = data.get("process") or {}
    text = BAT.read_text(encoding="utf-8", errors="replace")
    changed = []
    for key, var in (("llama_exe", "LLAMA"), ("gguf", "MODEL"), ("lora", "LORA")):
        value = str(proc.get(key) or "").strip()
        if not value:
            continue
        text, ok = set_bat_var(text, var, value)
        if ok:
            changed.append(f"{var}={Path(value).name}")
    port = proc.get("port")
    if port:
        new_text, n = re.subn(r"--port\s+\d+", f"--port {int(port)}", text, count=1)
        if n:
            text = new_text
            changed.append(f"port={port}")
    print("[sync] " + (", ".join(changed) if changed else "没有变化"))
    if not args.dry_run and changed:
        BAT.write_text(text, encoding="utf-8")
        print("[sync] start_llama.bat 已更新（重启 llama-server 生效）")


if __name__ == "__main__":
    main()