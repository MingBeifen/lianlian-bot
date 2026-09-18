"""读写 voicebot/.env：保留注释和顺序，只改/加 KEY=VALUE。"""
from __future__ import annotations

from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def read_env(path: Path | None = None) -> dict:
    p = path or ENV_PATH
    data: dict = {}
    if not p.exists():
        return data
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        data[key.strip()] = val.strip()
    return data


def set_env(mapping: dict, path: Path | None = None) -> None:
    p = path or ENV_PATH
    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
    wanted = {str(k): str(v) for k, v in mapping.items()}
    out, seen = [], set()
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in wanted:
                out.append(f"{key}={wanted[key]}")
                seen.add(key)
                continue
        out.append(line)
    for key, value in wanted.items():
        if key not in seen:
            out.append(f"{key}={value}")
    p.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


def edit_bat_var(bat_path: Path, var: str, value: str) -> bool:
    """改 start_llama.bat 里 set "VAR=..." 的值（改完要重启 llama-server 才生效）。"""
    if not bat_path.exists():
        return False
    import re
    text = bat_path.read_text(encoding="utf-8", errors="replace")
    pattern = rf'(set\s+"{re.escape(var)}=)[^"]*(")'
    new_text, n = re.subn(pattern, lambda m: m.group(1) + value + m.group(2), text, count=1)
    if n:
        bat_path.write_text(new_text, encoding="utf-8")
    return bool(n)