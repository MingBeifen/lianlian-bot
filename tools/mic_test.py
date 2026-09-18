"""麦克风体检：录几秒，实时画音量条，告诉你声音到底有没有进来。

    python mic_test.py             # 跟着提示做（1.5 秒安静 + 4 秒说话）
    python mic_test.py --seconds 8 # 说话时间加长

结束会给出：底噪、说话电平、建议的 VAD_MIN_RMS / INPUT_GAIN，
并且把录音存成 _mic_test.wav —— 用播放器放一下就知道录进去的是不是你的声音。
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
import sys
import time
from pathlib import Path

import numpy as np

from config import CONFIG
from audio_io import MIC_HINT, resolve_device

import sounddevice as sd  # audio_io 已经把 _vendor 加进 sys.path

HERE = Path(__file__).parent


def save_wav(path, audio: np.ndarray, sr: int) -> None:
    """用标准库写 wav —— 这样系统 python（没有 soundfile）也能跑。"""
    import wave

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm)


def bar(rms: float, width: int = 44) -> str:
    """0 ~ 0.05 的音量映射成一条可见的条（开方让小声也看得见）。"""
    level = min(1.0, max(0.0, (rms / 0.02) ** 0.5))
    n = int(level * width)
    return "[" + "#" * n + "." * (width - n) + "]"


def record(stream, seconds: float, frame_len: int, gain: float, label: str) -> np.ndarray:
    total = int(seconds * stream.samplerate / frame_len)
    out = []
    peak = 0.0
    for i in range(total):
        data, _ = stream.read(frame_len)
        frame = np.array(data[:, 0], dtype=np.float32) * gain
        np.clip(frame, -1.0, 1.0, out=frame)
        out.append(frame)
        rms = float(np.sqrt(np.mean(frame ** 2)))
        peak = max(peak, rms)
        sys.stdout.write(f"\r  {label} {bar(rms)} {rms:.5f}  峰值窗口 {peak:.5f}")
        sys.stdout.flush()
    sys.stdout.write("\n")
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=4.0, help="说话这段录多久")
    ap.add_argument("--gain", type=float, default=None, help="先按这个增益试（默认用 .env 里的 INPUT_GAIN）")
    args = ap.parse_args()

    cfg = CONFIG
    gain = args.gain if args.gain is not None else cfg.input_gain
    dev = resolve_device(cfg.input_device, output=False, prefer_wasapi=cfg.prefer_wasapi)
    idx = dev if dev is not None else sd.default.device[0]
    try:
        name = sd.query_devices(idx)["name"]
    except Exception:
        name = "?"
    print(f"\n输入设备：{name}（{idx}）  采样率 {cfg.sample_rate}  增益 x{gain:g}\n")

    try:
        stream = sd.InputStream(device=dev, samplerate=cfg.sample_rate, channels=1,
                                dtype="float32", blocksize=cfg.frame_len)
    except Exception as exc:
        print(f"[X] 打不开麦克风：{exc}\n{MIC_HINT}")
        return 1

    with stream:
        print("第 1 步：请【保持安静】1.5 秒，测一下环境底噪 ...")
        time.sleep(0.4)
        noise_audio = record(stream, 1.5, cfg.frame_len, gain, "底噪 ")
        print(f"\n第 2 步：现在【正常说一句话】，比如「今天天气不错」，录 {args.seconds:g} 秒 ...")
        time.sleep(0.3)
        speech_audio = record(stream, args.seconds, cfg.frame_len, gain, "说话 ")

    noise = float(np.sqrt(np.mean(noise_audio ** 2))) if noise_audio.size else 0.0
    speech_rms = float(np.sqrt(np.mean(speech_audio ** 2))) if speech_audio.size else 0.0
    # 说话那段里最响的 0.2 秒，比整体平均更能代表你的音量
    win = max(1, int(0.2 * cfg.sample_rate))
    if speech_audio.size >= win:
        seg = np.lib.stride_tricks.sliding_window_view(speech_audio, win)[:: win // 2]
        speech_peak = float(np.max(np.sqrt(np.mean(seg ** 2, axis=1))))
        raw_peak = float(np.max(np.abs(speech_audio)))
    else:
        speech_peak = raw_peak = 0.0

    print("\n" + "=" * 64)
    print(f"  环境底噪 RMS : {noise:.6f}   ({20 * np.log10(max(noise, 1e-9)):.1f} dBFS)")
    print(f"  说话 RMS     : {speech_rms:.6f}   (最响 0.2 秒 {speech_peak:.6f}，峰值 {raw_peak:.6f})")
    print("=" * 64)

    wav = HERE / "_mic_test.wav"
    try:
        save_wav(wav, np.concatenate([noise_audio, speech_audio]), cfg.sample_rate)
        print(f"\n录音已存到 {wav}（前 1.5 秒是底噪，之后是说话）")
        print("用播放器放一下 —— 能听见你自己说话，就说明声音确实进系统了。")
    except Exception as exc:
        print(f"（存 wav 失败：{exc}）")

    print("\n【结论】")
    if raw_peak < 1e-4:
        print("  ✘ 说话时几乎没有信号 —— 这是系统/硬件层面的问题，调阈值没用。按顺序查：")
        print("     1) 设置 → 系统 → 声音 → 输入 → 选中这个麦克风 → 把「音量」拉到 100")
        print("     2) 老版面板更全：运行 mmsys.cpl → 录制 → 麦克风 → 属性 → 级别 →")
        print("        音量 100，并把「麦克风加强」开到 +20dB 或 +30dB")
        print("     3) 笔记本可能有麦克风静音快捷键（Fn + F4/F8 之类），看键盘上的麦克风图标灯")
        print("     4) 确认不是选中了「未插孔」的设备：--devices 里认准 ACTIVE 的那个")
        print("   （另一个可能：你的笔记本用的是阵列麦克风，换个设备号试试：")
        print("     在 .env 里写 INPUT_DEVICE=0 或别的号，通常 MME 和 WASAPI 各有一条）")
    elif noise > 0.02:
        db = 20 * np.log10(max(noise, 1e-9))
        print(f"  △ 麦克风信号很好（峰值 {raw_peak:.3f}），但**环境底噪太高**（{db:.0f}dBFS）。")
        print("    能量 VAD 的门槛是「底噪 × 2.2」，底噪一高就盖过你的说话音量 → 程序里看着像没反应。")
        print("    项目 _vendor 里已经放好了 silero 神经网络 VAD，程序会自动优先用它，直接 run.bat 就行。")
        print("    想确认有没有生效：启动日志里应该出现 [vad] 使用 silero 神经网络 VAD。")
        print("    另外值得一试：把系统「麦克风加强」调低一档再测 —— 如果底噪降得比人声多，说明增益过头了。")
    elif speech_peak < 0.006:
        suggest = round(max(0.0008, speech_peak * 0.35), 5)
        need_gain = round(min(50.0, 0.03 / max(speech_peak, 1e-6)), 1)
        print("  △ 有信号，但很轻。两种改法（二选一或都做）：")
        print(f"     · 在 .env 里设 VAD_MIN_RMS={suggest}   （现在的门槛是 {cfg.vad_min_rms}，太高了）")
        print(f"     · 或者在 .env 里设 INPUT_GAIN={need_gain}  让程序先把声音放大再识别")
        print("     另外建议顺手把系统麦克风音量拉满、加强开 +20dB，效果最好。")
    else:
        print("  ✔ 麦克风电平正常。如果程序里还是没反应，多半是 VAD_MIN_RMS 设太高或者")
        print(f"     选错设备了（当前门槛 {cfg.vad_min_rms}，设备 {idx}）。")
    print("\n【程序现在会用的 VAD】")
    try:
        from vad import make_vad
        vad = make_vad(cfg)
        print(f"  → {type(vad).__name__}")
    except Exception as exc:
        print(f"  探测失败：{exc}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
