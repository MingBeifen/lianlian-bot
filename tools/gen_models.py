"""生成统一模型配置文件 models.json。

用法：
    python tools\\gen_models.py      # 根据当前 .env + 机器上的模型文件生成
    python tools\\gen_models.py --print

生成后所有模型都按 models.json 加载（优先级：models.json > .env > 代码默认）。
llama-server 的 gguf/LoRA 路径会在 run.bat 启动时由 sync_models.py 自动同步进 start_llama.bat。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # voicebot/tools
ROOT = HERE.parent                               # voicebot/
for _p in (str(ROOT), str(ROOT / "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import CONFIG                      # noqa: E402
from webui import catalog                      # noqa: E402

BAT = ROOT / "start_llama.bat"
OUT = ROOT / "models.json"


def bat_var(name: str) -> str:
    if not BAT.exists():
        return ""
    m = re.search(rf'set "{re.escape(name)}=([^"]*)"', BAT.read_text(encoding="utf-8", errors="replace"))
    return m.group(1).strip() if m else ""


def build() -> dict:
    gguf = bat_var("MODEL") or CONFIG.llama_gguf_path
    lora = bat_var("LORA") or CONFIG.llama_lora_path
    if not gguf:
        models = catalog.llm_gguf().get("models") or []
        gguf = models[0] if models else ""
    if not lora:
        loras = catalog.llm_gguf().get("loras") or []
        lora = loras[0] if loras else ""
    return {
        "_说明": "恋恋的统一模型配置。改完保存，重启 run.bat 生效；"
                 "llama-server 的 gguf/LoRA 由 sync_models.py 同步进 start_llama.bat。",
        "llm": {
            "base_url": CONFIG.llm_base_url,
            "api_key": CONFIG.llm_api_key,
            "model": CONFIG.llm_model,
            "system_prompt_file": CONFIG.llm_system_prompt_file,
            "max_tokens": CONFIG.llm_max_tokens,
            "temperature": CONFIG.llm_temperature,
            "top_p": CONFIG.llm_top_p,
            "history_turns": CONFIG.llm_history_turns,
            "keep_alive": CONFIG.llm_keep_alive,
            "timeout": CONFIG.llm_timeout,
        },
        "process": {
            "llama_exe": bat_var("LLAMA") or CONFIG.llama_exe_path,
            "gguf": gguf,
            "lora": lora,
            "port": CONFIG.llama_server_port,
        },
        "asr": {
            "backend": CONFIG.asr_backend,
            "model": CONFIG.asr_model,
            "device": CONFIG.asr_device,
            "language": CONFIG.asr_language,
            "api_base": CONFIG.asr_api_base,
            "api_key": CONFIG.asr_api_key,
            "api_model": CONFIG.asr_api_model,
        },
        "tts": {
            "backend": CONFIG.tts_backend,
            "lang": CONFIG.tts_lang,
            "voice": CONFIG.tts_voice,
            "speed": CONFIG.tts_speed,
            "rate": CONFIG.tts_rate,
            "en_mode": CONFIG.tts_en_mode,
            "device": CONFIG.tts_device,
            "repo": CONFIG.tts_repo,
            "sample_rate": CONFIG.tts_sample_rate,
            "trim_silence": CONFIG.tts_trim_silence,
            "strip_brackets": CONFIG.tts_strip_brackets,
            "gsv": {
                "api_url": CONFIG.gsv_api_url,
                "ref_audio": CONFIG.gsv_ref_audio,
                "prompt_text": CONFIG.gsv_prompt_text,
                "prompt_lang": CONFIG.gsv_prompt_lang,
                "text_lang": CONFIG.gsv_text_lang,
                "sample_rate": CONFIG.gsv_sample_rate,
                "timeout": CONFIG.gsv_timeout,
            },
        },
        "vad": {
            "backend": CONFIG.vad_backend,
            "threshold": CONFIG.vad_threshold,
            "energy_ratio": CONFIG.energy_ratio,
            "min_rms": CONFIG.vad_min_rms,
            "auto_calibrate": CONFIG.vad_auto_calibrate,
            "gate": CONFIG.vad_gate,
            "silence_ms": CONFIG.silence_ms,
            "min_speech_ms": CONFIG.min_speech_ms,
            "max_utterance_s": CONFIG.max_utterance_s,
            "preroll_ms": CONFIG.preroll_ms,
            "input_gain": CONFIG.input_gain,
        },
        "memory": {
            "enable": CONFIG.memory_enable,
            "db": CONFIG.memory_db,
            "top_k": CONFIG.memory_top_k,
            "auto_write": CONFIG.memory_auto_write,
            "overlap": CONFIG.memory_overlap,
            "embed_model": os.getenv("MEMORY_EMBED_MODEL_BGE", ""),
            "debug": CONFIG.debug_memory,
        },
        "live2d": {
            "enable": CONFIG.pet_enable,
            "backend": CONFIG.pet_backend,
            "native_python": CONFIG.native_pet_python,
            "native_model": CONFIG.native_pet_model,
            "port": CONFIG.native_pet_port,
            "fps": CONFIG.native_pet_fps,
            "mouth_gain": CONFIG.native_pet_mouth_gain,
            "arm": CONFIG.native_pet_arm,
            "idle_motion": CONFIG.native_pet_idle_motion,
            "motion_gap_min": CONFIG.native_pet_motion_gap_min,
            "motion_gap_max": CONFIG.native_pet_motion_gap_max,
            "react_motion": CONFIG.native_pet_react_motion,
            "action_motion": CONFIG.native_pet_action_motion,
            "action_map": CONFIG.native_pet_action_map,
            "idle_exit": CONFIG.native_pet_idle_exit,
            "size": CONFIG.pet_size,
            "position": CONFIG.pet_position,
            "margin": CONFIG.pet_margin,
            "topmost": CONFIG.pet_topmost,
            "borderless": CONFIG.pet_borderless,
            "hide_taskbar": CONFIG.pet_hide_taskbar,
            "click_through": CONFIG.pet_click_through,
            "hotkeys": CONFIG.pet_hotkeys,
        },
        "webui": {
            "enable": CONFIG.webui_enable,
            "host": CONFIG.webui_host,
            "port": CONFIG.webui_port,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 models.json")
    ap.add_argument("--print", action="store_true", help="只打印，不写文件")
    args = ap.parse_args()
    data = build()
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if args.print:
        print(text)
        return
    OUT.write_text(text + "\n", encoding="utf-8")
    print(f"已生成 {OUT}")
    print(f"  LLM   : {data['llm']['model']} @ {data['llm']['base_url']}")
    print(f"  ASR   : {data['asr']['backend']} / {data['asr']['model']}")
    print(f"  TTS   : {data['tts']['backend']} / {data['tts']['voice']}")
    print(f"  Live2D: {data['live2d']['backend']} / {Path(data['live2d']['native_model']).parent.name}")
    print(f"  模型文件: {data['process']['gguf']}")
    print("之后改 models.json 即可；重启 run.bat 生效。")


if __name__ == "__main__":
    main()