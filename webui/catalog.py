"""模型/资源发现：把本机可用的 LLM、TTS 音色、嵌入、ASR、Live2D 模型扫出来。"""
from __future__ import annotations

import json
from pathlib import Path

from .env_store import read_env

import os

# 项目根目录（voicebot 的上一级），所有扫描都基于它，换机器不用改代码
HARNESS = Path(__file__).resolve().parent.parent.parent
VTS_MODELS = Path(os.getenv(
    "VTS_MODELS_DIR",
    r"D:\software\steam\steamapps\common\VTube Studio\VTube Studio_Data\StreamingAssets\Live2DModels"))
GGUF_DIRS = [HARNESS / "models",
             Path(os.getenv("LLAMA_GGUF_DIR", str(HARNESS / "models"))),
             HARNESS / "gguf"]
WAV_DIRS = [HARNESS, HARNESS / "models" / "refs"]


def _glob(dirs, pattern: str, limit: int = 300) -> list:
    out = set()
    for d in dirs:
        try:
            if d.exists():
                out.update(str(x) for x in d.glob(pattern))
        except Exception:
            pass
    return sorted(out)[:limit]


def llm_gguf() -> dict:
    models = [p for p in _glob(GGUF_DIRS, "*.gguf") if "lora" not in Path(p).name.lower()]
    loras = [p for p in _glob(GGUF_DIRS, "*.gguf") if "lora" in Path(p).name.lower()]
    return {"models": models, "loras": loras}


def kokoro_voices() -> list:
    voices = [p.stem for p in (HARNESS / "kokoro-zh-v11" / "voices").glob("*.pt")] \
        if (HARNESS / "kokoro-zh-v11" / "voices").exists() else []
    voices += [f"zf_{i:03d}" for i in range(1, 101)]      # v1.1-zh 的 103 个中文音色
    seen, out = set(), []
    for v in voices:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return sorted(out)


def embed_models() -> list:
    out = []
    for root in [HARNESS, HARNESS / "models"]:
        if not root.exists():
            continue
        for cfg in root.glob("*/config.json"):
            d = cfg.parent
            if (d / "pytorch_model.bin").exists() or (d / "model.safetensors").exists() \
                    or list(d.glob("*.onnx")):
                out.append(str(d))
    return sorted(set(out))


def asr_models() -> list:
    out = []
    for base in [Path.home() / ".cache" / "modelscope" / "hub",
                 HARNESS / "models"]:
        if not base.exists():
            continue
        for cfg in base.glob("**/config.yaml"):
            if "sensevoice" in str(cfg).lower() or "paraformer" in str(cfg).lower():
                out.append(str(cfg.parent))
    return sorted(set(out))[:50]


def gsv_refs() -> list:
    return _glob(WAV_DIRS, "*.wav", limit=100)


def live2d_models(extra_paths: list | None = None) -> list:
    """扫出所有可用的 model3.json。"""
    roots = [VTS_MODELS, HARNESS / "models", HARNESS / "live2d-models"]
    found: dict = {}
    for root in roots:
        if not root.exists():
            continue
        for model in root.glob("**/*.model3.json"):
            d = model.parent
            preview = None
            for name in ("icon.jpg", "icon.png", "preview.png"):
                if (d / name).exists():
                    preview = str(d / name)
                    break
            if preview is None:
                pngs = sorted(d.glob("*.png"))
                preview = str(pngs[0]) if pngs else None
            motions = len(list(d.glob("animations/*.motion3.json"))) + \
                len(list(d.glob("motions/*.motion3.json")))
            expr = len(list(d.glob("*.exp3.json")))
            found[str(model)] = {
                "name": d.name, "path": str(model), "dir": str(d),
                "preview": preview, "motions": motions, "expressions": expr,
                "group": root.name if root != VTS_MODELS else "VTube Studio",
            }
    for path in extra_paths or []:
        if not path:
            continue
        p = Path(path)
        if p.exists() and str(p) not in found:
            found[str(p)] = {"name": p.parent.name, "path": str(p),
                             "dir": str(p.parent), "preview": None,
                             "motions": len(list(p.parent.glob("animations/*.motion3.json"))),
                             "expressions": 0, "group": "当前"}
    return sorted(found.values(), key=lambda x: (x["group"], x["name"]))


def snapshot(cfg) -> dict:
    """给前端展示的当前模型配置。"""
    return {
        "llm": {"base_url": cfg.llm_base_url, "model": cfg.llm_model,
                "api_key": cfg.llm_api_key, "prompt_file": cfg.llm_system_prompt_file,
                "max_tokens": cfg.llm_max_tokens, "temperature": cfg.llm_temperature,
                "history_turns": cfg.llm_history_turns},
        "asr": {"backend": cfg.asr_backend, "model": cfg.asr_model,
                "device": cfg.asr_device, "language": cfg.asr_language},
        "tts": {"backend": cfg.tts_backend, "voice": cfg.tts_voice, "speed": cfg.tts_speed,
                "device": cfg.tts_device, "repo": cfg.tts_repo, "rate": cfg.tts_rate,
                "ref_audio": cfg.gsv_ref_audio, "prompt": cfg.gsv_prompt_text,
                "gsv_url": cfg.gsv_api_url},
        "vad": {"backend": cfg.vad_backend, "threshold": cfg.vad_threshold,
                "silence_ms": cfg.silence_ms, "min_speech_ms": cfg.min_speech_ms,
                "input_gain": cfg.input_gain, "auto_calibrate": cfg.vad_auto_calibrate},
        "memory": {"top_k": cfg.memory_top_k, "auto_write": cfg.memory_auto_write,
                   "db": cfg.memory_db,
                   "embed_model_bge": read_env().get("MEMORY_EMBED_MODEL_BGE", "")},
        "pet": {"backend": cfg.pet_backend,
                "model": cfg.native_pet_model,
                "size": cfg.pet_size, "position": cfg.pet_position,
                "mouth_gain": cfg.native_pet_mouth_gain, "idle_motion": cfg.native_pet_idle_motion,
                "react_motion": cfg.native_pet_react_motion,
                "action_motion": cfg.native_pet_action_motion,
                "click_through": cfg.pet_click_through, "topmost": cfg.pet_topmost,
                "idle_motion_gap": [cfg.native_pet_motion_gap_min, cfg.native_pet_motion_gap_max]},
    }