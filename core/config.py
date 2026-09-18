"""恋恋语音通话：所有配置都可以用环境变量或同目录下的 .env 覆盖。"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ.setdefault(key.strip(), val)


_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent          # voicebot/
_WORKSPACE = _ROOT.parent            # 模型/venv 所在的工作区

_load_dotenv(_ROOT / ".env")
_HERE = _ROOT                        # 项目根（persona.txt / models.json 都在这）


def _s(key: str, default: str = "") -> str:
    return (os.getenv(key) or default).strip()


def _f(key: str, default: float) -> float:
    try:
        return float(_s(key) or default)
    except ValueError:
        return default


def _i(key: str, default: int) -> int:
    try:
        return int(float(_s(key) or default))
    except ValueError:
        return default


@dataclass
class Config:
    # ---------------- 大模型（默认指向本机 Ollama 的恋恋） ----------------
    llm_base_url: str = _s("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    llm_api_key: str = _s("LLM_API_KEY", "ollama")
    llm_model: str = _s("LLM_MODEL", "lianlian_v3:latest")
    # 用 Ollama 时留空 = 用 Modelfile 里的人设；
    # 用 llama.cpp 时没有 Modelfile，必须靠这里把人设发过去（推荐指向 persona.txt）
    llm_system_prompt_file: str = _s("LLM_SYSTEM_PROMPT_FILE", "")
    llm_system_prompt: str = _s("LLM_SYSTEM_PROMPT", "")
    llm_temperature: float = _f("LLM_TEMPERATURE", -1)   # <0 = 用 Modelfile 的参数
    llm_top_p: float = _f("LLM_TOP_P", -1)
    llm_max_tokens: int = _i("LLM_MAX_TOKENS", 220)      # 说话别太长
    llm_keep_alive: str = _s("LLM_KEEP_ALIVE", "30m")    # 模型常驻显存，省加载时间
    llm_history_turns: int = _i("LLM_HISTORY_TURNS", 10)
    llm_action: bool = _s("LLM_ACTION", "1") not in ("0", "false", "False")  # 从回复判断伴随动作
    llm_timeout: float = _f("LLM_TIMEOUT", 120)
    # llama.cpp 进程（由 models.json 统一管理，start_llama.bat 由 sync_models.py 同步）
    llama_exe_path: str = _s("LLAMA_EXE", str(_HERE.parent / "llamacpp-cu124"))
    llama_gguf_path: str = _s("LLAMA_GGUF", "")
    llama_lora_path: str = _s("LLAMA_LORA", "")
    llama_server_port: int = _i("LLAMA_PORT", 8080)

    # ---------------- 情绪 ----------------
    emotion_enable: bool = _s("EMOTION", "1") not in ("0", "false", "False")
    emotion_half_life: float = _f("EMOTION_HALF_LIFE_SEC", 180)   # 心情衰减一半的秒数
    emotion_affects_speed: bool = _s("EMOTION_SPEED", "1") not in ("0", "false", "False")
    debug_emotion: bool = _s("DEBUG_EMOTION", "1") not in ("0", "false", "False")
    # 每种情绪可以用不同的参考音频（有就换，没有就用默认的）
    gsv_ref_happy: str = _s("GSV_REF_HAPPY", "")
    gsv_ref_angry: str = _s("GSV_REF_ANGRY", "")
    gsv_ref_sad: str = _s("GSV_REF_SAD", "")
    gsv_ref_shy: str = _s("GSV_REF_SHY", "")
    gsv_prompt_happy: str = _s("GSV_PROMPT_HAPPY", "")
    gsv_prompt_angry: str = _s("GSV_PROMPT_ANGRY", "")
    gsv_prompt_sad: str = _s("GSV_PROMPT_SAD", "")
    # ---------------- Live2D 形象（VTube Studio 插件） ----------------
    # 项目只做驱动，渲染交给 VTube Studio；没开或没连上都不影响聊天。
    live2d_enable: bool = _s("LIVE2D", "0") not in ("0", "false", "False")
    live2d_url: str = _s("LIVE2D_URL", "ws://127.0.0.1:8001")
    live2d_plugin_name: str = _s("LIVE2D_PLUGIN_NAME", "恋恋语音伴侣")
    live2d_plugin_dev: str = _s("LIVE2D_PLUGIN_DEV", "lianlian")
    live2d_token_file: str = _s("LIVE2D_TOKEN_FILE", "")        # 留空 = voicebot/data/vtube_token.json
    live2d_fps: float = _f("LIVE2D_FPS", 24)                    # 口型帧率；被 VTS 限流会自动降
    live2d_mouth_param: str = _s("LIVE2D_MOUTH_PARAM", "MouthOpen")
    live2d_mouth_gain: float = _f("LIVE2D_MOUTH_GAIN", 6.0)     # 嘴张太大就调小
    live2d_mouth_floor: float = _f("LIVE2D_MOUTH_FLOOR", 0.003)  # 低于这个电平当静音
    # 情绪 -> 表情文件（VTS 模型目录里的 .exp3.json 文件名），例：
    # LIVE2D_EXPR=happy=exp_01.exp3.json,shy=exp_02.exp3.json
    live2d_expr: str = _s("LIVE2D_EXPR", "")
    # 情绪 -> VTS 热键 ID（没做表情文件时用这个），例：LIVE2D_HOTKEY=angry=ht_angry
    live2d_hotkey: str = _s("LIVE2D_HOTKEY", "")
    live2d_reset_neutral: bool = _s("LIVE2D_RESET_NEUTRAL", "1") not in ("0", "false", "False")
    live2d_debug: bool = _s("LIVE2D_DEBUG", "0") not in ("0", "false", "False")

    # ---------------- 桌宠模式（把 VTube Studio 变成桌面挂件） ----------------
    pet_enable: bool = _s("PET", "0") not in ("0", "false", "False")
    pet_title: str = _s("PET_TITLE", "VTube Studio")             # VTS 主窗口标题
    pet_topmost: bool = _s("PET_TOPMOST", "1") not in ("0", "false", "False")
    pet_borderless: bool = _s("PET_BORDERLESS", "1") not in ("0", "false", "False")
    pet_hide_taskbar: bool = _s("PET_HIDE_TASKBAR", "1") not in ("0", "false", "False")
    pet_click_through: bool = _s("PET_CLICK_THROUGH", "0") not in ("0", "false", "False")  # 1=启动就穿透
    pet_hotkeys: bool = _s("PET_HOTKEYS", "1") not in ("0", "false", "False")
    pet_restore_on_exit: bool = _s("PET_RESTORE_ON_EXIT", "1") not in ("0", "false", "False")
    pet_position: str = _s("PET_POSITION", "br").lower()         # tl/tr/bl/br/center/keep
    pet_margin: int = _i("PET_MARGIN", 30)                       # 离屏幕边缘像素
    pet_size: str = _s("PET_SIZE", "")                           # 例 520x900，留空=保持

    # ---- 自渲染桌宠（PET_BACKEND=native）：不依赖 VTube Studio ----
    pet_backend: str = _s("PET_BACKEND", "vtube").lower()        # vtube | native
    native_pet_python: str = _s(
        "NATIVE_PET_PYTHON", str(_HERE.parent / "live2d-venv" / "Scripts" / "python.exe"))
    native_pet_model: str = _s("NATIVE_PET_MODEL", "")   # 由 models.json / .env 指定
    native_pet_port: int = _i("NATIVE_PET_PORT", 9890)
    native_pet_fps: int = _i("NATIVE_PET_FPS", 60)
    native_pet_mouth_gain: float = _f("NATIVE_PET_MOUTH_GAIN", 6.0)  # 真实语音电平只有 0.1 左右
    native_pet_arm: str = _s("NATIVE_PET_ARM", "auto").lower()   # auto/a/b/both 手臂姿态
    native_pet_idle_motion: bool = _s("NATIVE_PET_IDLE_MOTION", "1") not in ("0", "false", "False")
    native_pet_motion_gap_min: float = _f("NATIVE_PET_MOTION_GAP_MIN", 10)   # 待机动作间隔
    native_pet_motion_gap_max: float = _f("NATIVE_PET_MOTION_GAP_MAX", 25)
    native_pet_debug: bool = _s("NATIVE_PET_DEBUG", "0") not in ("0", "false", "False")
    native_pet_react_motion: bool = _s("NATIVE_PET_REACT_MOTION", "1") not in ("0", "false", "False")
    native_pet_action_motion: bool = _s("NATIVE_PET_ACTION_MOTION", "1") not in ("0", "false", "False")
    native_pet_action_map: str = _s("NATIVE_PET_ACTION_MAP", "")   # 例 nod=Idle_2,wave=Idle_5
    native_pet_idle_exit: int = _i("NATIVE_PET_IDLE_EXIT", 15)   # 主程序断联多久后渲染器退出

    # ---------------- 主动性（她自己找时机开口）----------------
    proactive_enable: bool = _s("PROACTIVE", "1") not in ("0", "false", "False")
    proactive_idle_sec: float = _f("PROACTIVE_IDLE_SEC", 600)        # 安静多久她可以开口
    proactive_min_gap: float = _f("PROACTIVE_MIN_GAP_SEC", 480)      # 两次主动的最小间隔
    proactive_backoff_base: float = _f("PROACTIVE_BACKOFF_BASE", 2.0)  # 没被理一次，间隔乘几倍
    proactive_max_per_hour: int = _i("PROACTIVE_MAX_PER_HOUR", 2)
    proactive_max_per_day: int = _i("PROACTIVE_MAX_PER_DAY", 12)
    proactive_unanswered_limit: int = _i("PROACTIVE_UNANSWERED_LIMIT", 3)  # 连N次没理就当天闭嘴
    proactive_quiet_hours: str = _s("PROACTIVE_QUIET_HOURS", "0-7")  # 这个时段不主动
    proactive_require_sound: bool = _s("PROACTIVE_REQUIRE_SOUND", "1") not in ("0", "false", "False")
    proactive_noise_floor: float = _f("PROACTIVE_NOISE_FLOOR", 0.0015)  # 低于这个音量认为你不在
    proactive_temperature: float = _f("PROACTIVE_TEMPERATURE", 0.9)
    proactive_poll_sec: float = _f("PROACTIVE_POLL_SEC", 20)          # 多久检查一次

    # ---------------- 长期记忆 ----------------
    memory_enable: bool = _s("MEMORY_ENABLE", "1") not in ("0", "false", "False")
    memory_db: str = _s("MEMORY_DB", "")             # 留空用 data/lianlian_memory.db
    memory_top_k: int = _i("MEMORY_TOP_K", 8)        # 每轮注入几条
    memory_auto_write: bool = _s("MEMORY_AUTO_WRITE", "1") not in ("0", "false", "False")
    memory_overlap: float = _f("MEMORY_OVERLAP", 0.45)   # 抽出的内容至少要有多少字来自原话
    debug_memory: int = _i("DEBUG_MEMORY", 1)          # 0=不打印 1=一行摘要 2=打印全部

    # ---------------- 语音识别 ----------------
    asr_backend: str = _s("ASR_BACKEND", "sensevoice").lower()   # sensevoice | api
    asr_model: str = _s("ASR_MODEL", "iic/SenseVoiceSmall")
    asr_device: str = _s("ASR_DEVICE", "cuda:0")                 # cuda:0 | cpu
    asr_language: str = _s("ASR_LANGUAGE", "zh")
    asr_api_base: str = _s("ASR_API_BASE", "https://api.openai.com/v1")
    asr_api_key: str = _s("ASR_API_KEY", "")
    asr_api_model: str = _s("ASR_API_MODEL", "whisper-1")

    # ---------------- 语音合成 ----------------
    tts_backend: str = _s("TTS_BACKEND", "kokoro").lower()       # kokoro | api
    tts_lang: str = _s("TTS_LANG", "z")                          # z=中文
    tts_voice: str = _s("TTS_VOICE", "zf_xiaoxiao")
    tts_speed: float = _f("TTS_SPEED", 1.05)                    # Kokoro 用
    tts_rate: str = _s("TTS_RATE", "+8%")                       # Edge 用，形如 +8% / -5%
    tts_en_mode: str = _s("TTS_EN_MODE", "ipa").lower()         # 英文怎么念：ascii | ipa | zh

    # ---- GPT-SoVITS（TTS_BACKEND=gsv）：它在独立 venv 里，通过 HTTP 服务调用 ----
    gsv_api_url: str = _s("GSV_API_URL", "http://127.0.0.1:9880")
    gsv_ref_audio: str = _s("GSV_REF_AUDIO", str(_HERE.parent / "gsv-ref.wav"))
    gsv_prompt_text: str = _s("GSV_PROMPT_TEXT", "行吧。只是因为你感冒了，怕传染死我而已。")
    gsv_prompt_lang: str = _s("GSV_PROMPT_LANG", "zh")
    gsv_text_lang: str = _s("GSV_TEXT_LANG", "zh")
    gsv_sample_rate: int = _i("GSV_SAMPLE_RATE", 32000)
    gsv_timeout: float = _f("GSV_TIMEOUT", 120)
    tts_device: str = _s("TTS_DEVICE", "cuda:0")                  # 实测 GPU 快 3~4 倍；显存不够就改 cpu
    tts_repo: str = _s("TTS_REPO", "hexgrad/Kokoro-82M")        # 换成 ...-v1.1-zh 中文更好
    tts_strip_brackets: bool = _s("TTS_STRIP_BRACKETS", "1") not in ("0", "false", "False")  # 去掉（动作）旁白不朗读
    tts_trim_silence: bool = _s("TTS_TRIM_SILENCE", "1") not in ("0", "false", "False")
    tts_api_base: str = _s("TTS_API_BASE", "https://api.openai.com/v1")
    tts_api_key: str = _s("TTS_API_KEY", "")
    tts_api_model: str = _s("TTS_API_MODEL", "tts-1")
    tts_api_voice: str = _s("TTS_API_VOICE", "nova")

    # ---------------- 麦克风 / 扬声器 ----------------
    sample_rate: int = _i("SAMPLE_RATE", 16000)      # 识别要求 16k
    frame_ms: int = _i("FRAME_MS", 32)               # 32ms = 512 采样点
    input_device: str = _s("INPUT_DEVICE", "")       # 设备号或名字片段，留空用系统默认
    output_device: str = _s("OUTPUT_DEVICE", "")
    prefer_wasapi: bool = _s("PREFER_WASAPI", "0") in ("1", "true", "True")
    tts_sample_rate: int = _i("TTS_SAMPLE_RATE", 24000)   # Kokoro 固定 24k

    # ---------------- 断句（VAD） ----------------
    vad_backend: str = _s("VAD_BACKEND", "auto")     # auto | silero | webrtc | energy
    vad_threshold: float = _f("VAD_THRESHOLD", 0.5)  # silero 用的概率阈值
    energy_ratio: float = _f("ENERGY_RATIO", 2.2)    # 比环境噪声大多少倍算说话
    vad_min_rms: float = _f("VAD_MIN_RMS", 0.006)    # 绝对音量下限（自动校准会按实测底噪覆盖它）
    vad_auto_calibrate: bool = _s("VAD_AUTO_CALIBRATE", "1") not in ("0", "false", "False")
    vad_gate: bool = _s("VAD_GATE", "0") not in ("0", "false", "False")   # 给神经网络 VAD 加能量闸门（噪声极大时试）
    input_gain: float = _f("INPUT_GAIN", 1.0)        # 麦克风太轻就放大，比如 5、10
    silence_ms: int = _i("SILENCE_MS", 650)          # 静音多久算说完
    min_speech_ms: int = _i("MIN_SPEECH_MS", 300)    # 短于这个当咳嗽/杂音丢掉
    max_utterance_s: float = _f("MAX_UTTERANCE_S", 12.0)   # 兜底：再久也强制断句
    preroll_ms: int = _i("PREROLL_MS", 300)          # 保留语音起点前的一小段，避免吃掉第一个字

    # ---------------- 交互 ----------------
    barge_in: bool = _s("BARGE_IN", "0") not in ("0", "false", "False")   # 1=允许说话打断（要戴耳机，外放会自问自答）
    barge_in_confirm_ms: int = _i("BARGE_IN_CONFIRM_MS", 160)
    greeting: str = _s("GREETING", "")               # 启动时先说一句，留空则不说
    debug_latency: bool = _s("DEBUG_LATENCY", "1") not in ("0", "false", "False")
    debug_mic: bool = _s("DEBUG_MIC", "1") not in ("0", "false", "False")   # 打印"检测到说话/一句说完"
    allow_typing: bool = _s("ALLOW_TYPING", "1") not in ("0", "false", "False")  # 说话的同时也能打字
    # ---------------- Web 控制台 ----------------
    webui_enable: bool = _s("WEBUI", "1") not in ("0", "false", "False")
    webui_host: str = _s("WEBUI_HOST", "127.0.0.1")
    webui_port: int = _i("WEBUI_PORT", 8765)
    text_mode: bool = False
    wav_input: str = ""
    wav_output: str = ""

    @property
    def frame_len(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)

    @property
    def preroll_frames(self) -> int:
        return max(1, int(self.preroll_ms / self.frame_ms))

    def describe(self) -> str:
        return (
            f"  大模型 : {self.llm_model}  @ {self.llm_base_url}\n"
            f"  识别   : {self.asr_backend} / {self.asr_model} on {self.asr_device}\n"
            f"  合成   : {self.tts_backend} / {self.tts_voice} on {self.tts_device} (lang={self.tts_lang})\n"
            f"  音频   : 输入[{self.input_device or '默认'}] 输出[{self.output_device or '默认'}] "
            f"{self.sample_rate}Hz→{self.tts_sample_rate}Hz\n"
            f"  打断   : {'开' if self.barge_in else '关（半双工，外放时用这个）'}\n"
            f"  Live2D : {'开 ' + self.live2d_url if self.live2d_enable else '关'}\n"
            f"  桌宠   : {'开 ' + self.pet_backend + ' ' + self.pet_position if self.pet_enable else '关'}\n"
            f"  模型配置 : {'models.json' if _MODELS_FILE.exists() else '.env（未找到 models.json）'}"
        )


CONFIG = Config()

# ---------------- models.json：所有模型的统一配置 ----------------
# 优先级：models.json > .env > 代码默认值；只覆盖 JSON 里出现的字段。
_MODELS_FILE = _ROOT / "models.json"
_models_loaded = False


def _bool_val(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() not in ("0", "false", "no", "off", "")


_MODEL_FIELDS = [
    (("llm", "base_url"), "llm_base_url", str),
    (("llm", "api_key"), "llm_api_key", str),
    (("llm", "model"), "llm_model", str),
    (("llm", "system_prompt_file"), "llm_system_prompt_file", str),
    (("llm", "max_tokens"), "llm_max_tokens", int),
    (("llm", "temperature"), "llm_temperature", float),
    (("llm", "top_p"), "llm_top_p", float),
    (("llm", "history_turns"), "llm_history_turns", int),
    (("llm", "keep_alive"), "llm_keep_alive", str),
    (("llm", "timeout"), "llm_timeout", float),
    (("process", "llama_exe"), "llama_exe_path", str),
    (("process", "gguf"), "llama_gguf_path", str),
    (("process", "lora"), "llama_lora_path", str),
    (("process", "port"), "llama_server_port", int),
    (("asr", "backend"), "asr_backend", str),
    (("asr", "model"), "asr_model", str),
    (("asr", "device"), "asr_device", str),
    (("asr", "language"), "asr_language", str),
    (("asr", "api_base"), "asr_api_base", str),
    (("asr", "api_key"), "asr_api_key", str),
    (("asr", "api_model"), "asr_api_model", str),
    (("tts", "backend"), "tts_backend", str),
    (("tts", "lang"), "tts_lang", str),
    (("tts", "voice"), "tts_voice", str),
    (("tts", "speed"), "tts_speed", float),
    (("tts", "rate"), "tts_rate", str),
    (("tts", "en_mode"), "tts_en_mode", str),
    (("tts", "device"), "tts_device", str),
    (("tts", "repo"), "tts_repo", str),
    (("tts", "sample_rate"), "tts_sample_rate", int),
    (("tts", "trim_silence"), "tts_trim_silence", _bool_val),
    (("tts", "strip_brackets"), "tts_strip_brackets", _bool_val),
    (("tts", "gsv", "api_url"), "gsv_api_url", str),
    (("tts", "gsv", "ref_audio"), "gsv_ref_audio", str),
    (("tts", "gsv", "prompt_text"), "gsv_prompt_text", str),
    (("tts", "gsv", "prompt_lang"), "gsv_prompt_lang", str),
    (("tts", "gsv", "text_lang"), "gsv_text_lang", str),
    (("tts", "gsv", "sample_rate"), "gsv_sample_rate", int),
    (("tts", "gsv", "timeout"), "gsv_timeout", float),
    (("vad", "backend"), "vad_backend", str),
    (("vad", "threshold"), "vad_threshold", float),
    (("vad", "energy_ratio"), "energy_ratio", float),
    (("vad", "min_rms"), "vad_min_rms", float),
    (("vad", "auto_calibrate"), "vad_auto_calibrate", _bool_val),
    (("vad", "gate"), "vad_gate", _bool_val),
    (("vad", "silence_ms"), "silence_ms", int),
    (("vad", "min_speech_ms"), "min_speech_ms", int),
    (("vad", "max_utterance_s"), "max_utterance_s", float),
    (("vad", "preroll_ms"), "preroll_ms", int),
    (("vad", "input_gain"), "input_gain", float),
    (("memory", "enable"), "memory_enable", _bool_val),
    (("memory", "db"), "memory_db", str),
    (("memory", "top_k"), "memory_top_k", int),
    (("memory", "auto_write"), "memory_auto_write", _bool_val),
    (("memory", "overlap"), "memory_overlap", float),
    (("memory", "debug"), "debug_memory", int),
    (("live2d", "enable"), "pet_enable", _bool_val),
    (("live2d", "backend"), "pet_backend", str),
    (("live2d", "native_python"), "native_pet_python", str),
    (("live2d", "native_model"), "native_pet_model", str),
    (("live2d", "port"), "native_pet_port", int),
    (("live2d", "fps"), "native_pet_fps", int),
    (("live2d", "mouth_gain"), "native_pet_mouth_gain", float),
    (("live2d", "arm"), "native_pet_arm", str),
    (("live2d", "idle_motion"), "native_pet_idle_motion", _bool_val),
    (("live2d", "motion_gap_min"), "native_pet_motion_gap_min", float),
    (("live2d", "motion_gap_max"), "native_pet_motion_gap_max", float),
    (("live2d", "react_motion"), "native_pet_react_motion", _bool_val),
    (("live2d", "action_motion"), "native_pet_action_motion", _bool_val),
    (("live2d", "action_map"), "native_pet_action_map", str),
    (("live2d", "idle_exit"), "native_pet_idle_exit", int),
    (("live2d", "size"), "pet_size", str),
    (("live2d", "position"), "pet_position", str),
    (("live2d", "margin"), "pet_margin", int),
    (("live2d", "topmost"), "pet_topmost", _bool_val),
    (("live2d", "borderless"), "pet_borderless", _bool_val),
    (("live2d", "hide_taskbar"), "pet_hide_taskbar", _bool_val),
    (("live2d", "click_through"), "pet_click_through", _bool_val),
    (("live2d", "hotkeys"), "pet_hotkeys", _bool_val),
    (("webui", "enable"), "webui_enable", _bool_val),
    (("webui", "host"), "webui_host", str),
    (("webui", "port"), "webui_port", int),
]


def _dig(data: dict, path: tuple):
    node = data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _apply_models_json(cfg) -> bool:
    """用 models.json 覆盖模型相关配置（只动 JSON 里写了的字段）。"""
    if not _MODELS_FILE.exists():
        print("[config] 未找到 models.json（可选）：运行 python tools\\gen_models.py 生成统一配置")
        return False
    try:
        data = json.loads(_MODELS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[config] models.json 解析失败（{type(exc).__name__}: {exc}），改用 .env")
        return False
    applied = []
    for path, attr, cast in _MODEL_FIELDS:
        value = _dig(data, path)
        if value is None:
            continue
        try:
            setattr(cfg, attr, cast(value))
            applied.append(attr)
        except Exception as exc:
            print(f"[config] models.json 字段 {'.'.join(path)} 无效：{exc}")
    embed = _dig(data, ("memory", "embed_model"))
    if embed:
        os.environ["MEMORY_EMBED_MODEL_BGE"] = str(embed)
        applied.append("memory_embed_model")
    if applied:
        print(f"[config] 已按 models.json 加载 {len(applied)} 项模型配置")
    return True


_apply_models_json(CONFIG)

# 人设文件优先：llama.cpp 没有 Modelfile，人设得由程序发过去
if CONFIG.llm_system_prompt_file:
    _pf = Path(CONFIG.llm_system_prompt_file)
    if not _pf.is_absolute():
        _pf = _HERE / _pf
    if not _pf.exists():                     # 兜底：自动去 config/ 里找同名文件
        _alt = _HERE / "config" / _pf.name
        if _alt.exists():
            _pf = _alt
    if _pf.exists():
        CONFIG.llm_system_prompt = _pf.read_text(encoding="utf-8").strip()
    else:
        print(f"[config] 警告：找不到人设文件 {_pf}")
