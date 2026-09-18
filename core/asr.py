"""语音识别：SenseVoice（本地，你的 5060 上很快）或 OpenAI 兼容的 whisper 接口。"""
from __future__ import annotations

import asyncio
import io
import os
import re
import tempfile
import wave
from pathlib import Path

import numpy as np

from emotion import parse_sensevoice

_TAG = re.compile(r"<\|[^|]*\|>")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200B]")
_EVENT = re.compile(
    r"[\[\(（【]\s*(?:laughter|applause|music|noise|speech|sigh|笑|笑声|掌声|音乐|噪音|咳嗽|叹气|哭声|背景音)\s*[\]\)）】]"
)
_JUNK = ("谢谢观看", "请不吝点赞", "订阅", "转发", "字幕", "thanks for watching",
         "please subscribe", "subtitle", "amara.org")


def clean_asr_text(text: str) -> str:
    """去掉 SenseVoice 的 <|zh|><|NEUTRAL|> 之类标签和表情符号。"""
    if not text:
        return ""
    text = _TAG.sub("", text)
    text = _EVENT.sub("", text)
    text = _EMOJI.sub("", text)
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def is_noise(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if len(t) == 1 and not t.isalnum():
        return True
    low = t.lower()
    return any(k in low for k in _JUNK)


def wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm)
    return buf.getvalue()



def local_model_dir(model_id: str) -> str:
    """把 iic/SenseVoiceSmall 这种 id 换成 modelscope 本地缓存目录。

    funasr 默认每次启动都要连 hub 校验文件，网一慢就卡在启动界面；
    直接用本地快照就完全离线，秒开。
    """
    if not model_id or "/" not in model_id:
        return ""
    org, name = model_id.split("/", 1)
    root = Path(os.environ.get("MODELSCOPE_CACHE", Path.home() / ".cache" / "modelscope"))
    base = root / "models" / f"{org}--{name}" / "snapshots"
    if not base.is_dir():
        return ""
    for rev in ("master", "main"):
        d = base / rev
        if d.is_dir() and any(d.iterdir()):
            return str(d)
    subs = sorted([d for d in base.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
    return str(subs[0]) if subs else ""


class SenseVoiceASR:
    def __init__(self, cfg):
        from funasr import AutoModel

        self.cfg = cfg
        self.last_emotion = "neutral"
        self.last_event = ""
        local = local_model_dir(cfg.asr_model)
        if local:
            print(f"[asr] 加载 {cfg.asr_model}（{cfg.asr_device}，本地缓存）...")
        else:
            print(f"[asr] 加载 {cfg.asr_model}（{cfg.asr_device}）...（第一次要从网上拉模型）")
        self.model = AutoModel(
            model=local or cfg.asr_model,
            device=cfg.asr_device,
            disable_update=True,
            disable_pbar=True,
        )
        print("[asr] 就绪")

        self.last_emotion = "neutral"     # 从语音里听出来的哥哥的情绪
        self.last_event = ""              # 笑声/叹气之类

    async def transcribe(self, audio: np.ndarray) -> str:
        return await asyncio.to_thread(self._sync, audio)

    def _sync(self, audio: np.ndarray) -> str:
        try:
            res = self.model.generate(
                input=audio, language=self.cfg.asr_language, use_itn=True, batch_size_s=60
            )
        except Exception as exc:
            print(f"[asr] 传内存数组失败（{type(exc).__name__}: {exc}），改用临时 wav")
            res = self._via_file(audio)
        raw = res[0].get("text", "") if res else ""
        # SenseVoice 的 <|HAPPY|> <|Laughter|> 标签以前是直接丢掉的，
        # 其实那是免费的真实情感信号
        text, self.last_emotion, self.last_event = parse_sensevoice(raw)
        return clean_asr_text(text)

    def _via_file(self, audio: np.ndarray):
        import soundfile as sf

        fd, path = tempfile.mkstemp(suffix=".wav", prefix="lianlian_")
        os.close(fd)
        try:
            sf.write(path, audio, self.cfg.sample_rate, subtype="PCM_16")
            return self.model.generate(input=path, language=self.cfg.asr_language, use_itn=True)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


class ApiASR:
    """任何 OpenAI 兼容的 /audio/transcriptions（OpenAI、Groq、硅基流动…）。"""

    last_emotion = "neutral"
    last_event = ""

    def __init__(self, cfg):
        from openai import AsyncOpenAI

        self.cfg = cfg
        self.client = AsyncOpenAI(base_url=cfg.asr_api_base, api_key=cfg.asr_api_key or "none")

    async def transcribe(self, audio: np.ndarray) -> str:
        kwargs = {
            "model": self.cfg.asr_api_model,
            "file": ("speech.wav", wav_bytes(audio, self.cfg.sample_rate), "audio/wav"),
        }
        if self.cfg.asr_language:
            kwargs["language"] = self.cfg.asr_language
        resp = await self.client.audio.transcriptions.create(**kwargs)
        return clean_asr_text(getattr(resp, "text", "") or "")


def make_asr(cfg):
    if cfg.asr_backend == "api":
        return ApiASR(cfg)
    return SenseVoiceASR(cfg)