"""自检：不开麦克风，验证 合成 → 识别 → 大模型 三段链路。

    python selftest.py            # 跑完整自检
    python selftest.py --play     # 顺便放一遍合成出来的声音
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent        # voicebot/
_VENV_PY = _ROOT / ".venv" / "Scripts" / "python.exe"
if not _VENV_PY.exists():                              # 兼容旧机器
    _legacy = Path(r"D:\My-Neuro\lianlian-v0.1\venv_app\Scripts\python.exe")
    _VENV_PY = _legacy if _legacy.exists() else Path(sys.executable)


def _ensure_venv() -> None:
    """没跑在项目 venv 里就自动用它重跑一遍。

    系统 python 没有 torchaudio/soundfile，silero VAD 和录音都会缺件，
    与其让你踩坑，不如自己切过去。
    """
    if os.environ.get("LIANLIAN_NO_REEXEC") == "1" or not _VENV_PY.exists():
        return
    try:
        if Path(sys.executable).resolve() == _VENV_PY.resolve():
            return
    except OSError:
        return
    os.environ["LIANLIAN_NO_REEXEC"] = "1"
    import subprocess

    raise SystemExit(subprocess.call([str(_VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]]))


_ensure_venv()

for _p in (str(_ROOT), str(_ROOT / "core"), str(_ROOT / "live2d")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


import argparse
import asyncio
import time
from pathlib import Path

import numpy as np

from config import CONFIG
from asr import is_noise, make_asr
from audio_io import Player, list_devices
from llm import ChatLLM
from tts import make_tts

HERE = Path(__file__).parent
SENTENCE = "早上好呀，今天也要一起加油哦。"
WAV = HERE / "_selftest_tts.wav"


def to_16k(audio: np.ndarray, sr: int, target: int = 16000) -> np.ndarray:
    if sr == target:
        return audio.astype(np.float32)
    n = int(round(len(audio) * target / sr))
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


async def main(play: bool) -> int:
    cfg = CONFIG
    ok = True

    print("\n=== 0. 音频设备 ===")
    list_devices()

    print(f"\n=== 1. 语音合成（{cfg.tts_backend} / {cfg.tts_voice}）===")
    t0 = time.perf_counter()
    tts = make_tts(cfg)
    t1 = time.perf_counter()
    pcm = await tts.synth(SENTENCE)
    t2 = time.perf_counter()
    dur = len(pcm) / tts.sample_rate
    print(f"文本 : {SENTENCE}")
    print(f"耗时 : 加载 {t1-t0:.1f}s | 合成 {t2-t1:.2f}s，音频 {dur:.2f}s（{tts.sample_rate}Hz）")
    if pcm.size == 0:
        print("[FAIL] 合成结果为空")
        return 1
    import soundfile as sf
    sf.write(str(WAV), pcm, tts.sample_rate)
    print(f"已保存 {WAV}")

    print("\n=== 2. 语音识别（把上面那段读回来）===")
    asr = make_asr(cfg)
    t3 = time.perf_counter()
    text = await asr.transcribe(to_16k(pcm, tts.sample_rate, cfg.sample_rate))
    t4 = time.perf_counter()
    print(f"识别 : {text!r}   耗时 {t4-t3:.2f}s")
    hit = sum(1 for ch in "早上好今天也要一起加油" if ch in text)
    if not text or is_noise(text):
        print("[FAIL] 没识别出内容")
        ok = False
    else:
        print(f"命中关键字 {hit}/12 —— {'看起来正常' if hit >= 8 else '识别质量偏低，检查一下麦克风/模型'}")

    print("\n=== 3. 大模型（真实一轮对话）===")
    llm = ChatLLM(cfg)
    t5 = time.perf_counter()
    first = None
    chunks = []
    async for piece in llm.stream_reply("哥哥：我今天有点累，陪我说说话好不好？"):
        if first is None:
            first = time.perf_counter()
        chunks.append(piece)
        print(piece, end="", flush=True)
    print()
    if not chunks:
        print("[FAIL] 大模型没有回复，确认 ollama serve 在跑、模型名对不对")
        ok = False
    else:
        print(f"首字 {first - t5:.2f}s | 全部 {time.perf_counter() - t5:.2f}s")
    print(f"（历史里现在有 {len(llm.history)} 条消息，正式跑的时候会一直带着上下文）")

    if play:
        print("\n=== 4. 播放刚才合成的语音 ===")
        player = Player(cfg, tts.sample_rate)
        player.start()
        player.play(pcm)
        await player.wait_drained()
        player.close()

    print("\n" + ("全 部 通 过 ✔" if ok else "有项目失败 ✘，看上面 [FAIL]") + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--play", action="store_true", help="播放合成出来的语音")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main(args.play)))
