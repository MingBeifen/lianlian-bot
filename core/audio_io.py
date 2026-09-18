"""麦克风采集（自动断句）和扬声器播放（可随时打断）。"""
from __future__ import annotations

import asyncio
import collections
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# 项目自带的 _vendor 里有 sounddevice，这样不用往你的 venv 里装东西
_ROOT = Path(__file__).resolve().parent.parent
_VENDOR = _ROOT / "_vendor"
if _VENDOR.is_dir() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

import numpy as np
import sounddevice as sd

from vad import BaseVAD


MIC_HINT = (
    "打不开麦克风。常见原因按顺序排查：\n"
    "  1) Windows 麦克风权限被关（本机现在就是这个）：\n"
    "     设置 → 隐私和安全性 → 麦克风 → 打开「麦克风访问」+「允许桌面应用访问你的麦克风」\n"
    "     快捷命令： start ms-settings:privacy-microphone\n"
    "  2) .env 里 INPUT_DEVICE 选错设备号（用 --devices 看列表）\n"
    "  3) 麦克风被别的程序独占（会议软件、OBS 等）"
)


def mic_selftest(cfg) -> bool:
    """录一小段看看麦克风到底能不能用，失败就直说怎么修。"""
    dev = resolve_device(cfg.input_device, output=False, prefer_wasapi=cfg.prefer_wasapi)
    idx = dev if dev is not None else sd.default.device[0]
    try:
        name = sd.query_devices(idx)["name"]
    except Exception:
        name = "?"
    print(f"[麦克风] 测试 {name} ...")
    try:
        frames = []
        with sd.InputStream(samplerate=cfg.sample_rate, channels=1, dtype="float32",
                            blocksize=cfg.frame_len, device=dev) as stream:
            for _ in range(max(4, int(1.2 * cfg.sample_rate / cfg.frame_len))):
                data, _ = stream.read(cfg.frame_len)
                frames.append(np.array(data[:, 0], dtype=np.float32))
    except Exception as exc:
        print(f"[麦克风] 打开失败：{exc}")
        print(MIC_HINT)
        return False
    audio = np.concatenate(frames)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    print(f"[麦克风] 打开正常，1.2 秒采样：RMS {rms:.5f} / 峰值 {peak:.4f}")
    if peak < 1e-3:
        print("[麦克风] 但音量几乎是 0 —— 对着麦克风说句话再测，还是这样就是设备没选对或被静音了")
    return True


def list_devices() -> None:
    print(sd.query_devices())
    print("默认输入/输出:", list(sd.default.device))


def resolve_device(spec: str, output: bool, prefer_wasapi: bool = False):
    """把 空 / 设备号 / 名字片段 解析成 sounddevice 的设备号。"""
    spec = (spec or "").strip()
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    need = "max_output_channels" if output else "max_input_channels"

    def usable(idx: int) -> bool:
        return devices[idx][need] > 0

    def wasapi_of(idxs: list) -> list:
        return [i for i in idxs if "wasapi" in hostapis[devices[i]["hostapi"]]["name"].lower()]

    if spec.isdigit():
        return int(spec)
    if spec:
        hits = [i for i, d in enumerate(devices) if spec.lower() in d["name"].lower() and usable(i)]
        if not hits:
            kind = "输出" if output else "输入"
            raise SystemExit(f"找不到匹配 '{spec}' 的{kind}设备，运行 --devices 看设备号")
        return (wasapi_of(hits) or hits)[0]

    if not prefer_wasapi:
        return None
    default_idx = sd.default.device[1 if output else 0]
    if default_idx is None or default_idx < 0:
        return None
    name = devices[default_idx]["name"]
    hits = [i for i, d in enumerate(devices) if d["name"] == name and usable(i)]
    return (wasapi_of(hits) or [default_idx])[0]


class MicListener:
    """后台线程持续读麦克风，靠 VAD 切出「一整句话」，塞进 asyncio 队列。"""

    def __init__(self, cfg, vad: BaseVAD, loop: asyncio.AbstractEventLoop,
                 out_queue: Optional[asyncio.Queue] = None,
                 speech_event: Optional[asyncio.Event] = None):
        self.cfg = cfg
        self.vad = vad
        self.loop = loop
        self.frame_len = cfg.frame_len
        # 队列和事件可以从外面传进来，这样键盘输入能和语音共用同一条流水线
        self.queue: asyncio.Queue = out_queue if out_queue is not None else asyncio.Queue()
        self.speech_started = speech_event if speech_event is not None else asyncio.Event()
        self.error: Optional[str] = None
        self.device = resolve_device(cfg.input_device, output=False, prefer_wasapi=cfg.prefer_wasapi)
        self._muted = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._confirm = max(1, int(cfg.barge_in_confirm_ms / cfg.frame_ms))
        self.stats = {"frames": 0, "utterances": 0}
        self._level = 0.0          # 最近的平滑音量（主动性用它判断"房间里有没有人"）

    # ---------------- 生命周期 ----------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="mic", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

    def set_muted(self, muted: bool) -> None:
        """半双工：外放时 AI 说话期间把麦克风闭掉，免得它听见自己。"""
        self._muted.set() if muted else self._muted.clear()

    def _signal(self, fn, *args) -> None:
        try:
            self.loop.call_soon_threadsafe(fn, *args)
        except RuntimeError:
            pass

    def _fail(self, msg: str) -> None:
        self.error = msg
        self._signal(self.queue.put_nowait, None)

    # ---------------- 采集线程 ----------------
    def _open_stream(self):
        kwargs = dict(samplerate=self.cfg.sample_rate, channels=1, dtype="float32",
                      blocksize=self.frame_len, device=self.device)
        try:
            return sd.InputStream(**kwargs, latency="low")
        except Exception:
            return sd.InputStream(**kwargs)

    def _calibrate(self, stream) -> None:
        """开机先安静听 1 秒，按这台机器的实际电平把 VAD 门槛调好。

        不同麦克风电平能差 100 倍，写死一个 VAD_MIN_RMS 必然有一半人用不了。
        """
        if not self.cfg.vad_auto_calibrate:
            return
        gain = self.cfg.input_gain
        frames = []
        try:
            for _ in range(max(4, int(1.0 * self.cfg.sample_rate / self.frame_len))):
                data, _ = stream.read(self.frame_len)
                f = np.ascontiguousarray(data[:, 0], dtype=np.float32)
                if gain != 1.0:
                    f = np.clip(f * gain, -1.0, 1.0)
                frames.append(f)
        except Exception as exc:
            print(f"[mic] 校准失败（{type(exc).__name__}），沿用配置里的阈值")
            return
        if not frames:
            return
        noise = float(np.sqrt(np.mean(np.concatenate(frames) ** 2)))
        if not np.isfinite(noise) or noise <= 0:
            noise = 1e-6
        db = 20 * np.log10(max(noise, 1e-9))
        if hasattr(self.vad, "min_rms"):      # 只有能量法需要手调门槛
            ratio = getattr(self.vad, "ratio", 2.2)
            self.vad.noise = max(noise, 1e-7)
            self.vad.min_rms = max(noise * ratio * 0.6, 3e-5)
            print(f"[mic] 底噪 RMS {noise:.6f}（{db:.0f}dBFS）→ 能量 VAD 门槛 {self.vad.min_rms:.6f}"
                  + (f"（增益 x{gain:g}）" if gain != 1.0 else ""))
        else:
            print(f"[mic] 底噪 RMS {noise:.6f}（{db:.0f}dBFS）")
        if noise < 2e-5:
            print("[mic] 警告：底噪几乎为 0（低于 -95dB），麦克风基本没送声音进来。")
            print("      先跑一次 `python mic_test.py`，它会告诉你该调系统音量还是加强。")
        elif noise > 0.02:
            print(f"[mic] 注意：环境底噪偏高（{db:.0f}dBFS）。能量 VAD 的门槛会被抬到说话音量之上，")
            print("      如果识别没反应，用 silero 神经网络 VAD（项目 _vendor 里已备好，会自动启用）。")

    def recent_rms(self) -> float:
        """最近的环境音量（指数平滑）。主动性用它判断你在不在。"""
        return self._level

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _log(self, msg: str) -> None:
        if self.cfg.debug_mic:
            print(f"[mic] {msg}", flush=True)

    def _run(self) -> None:
        """线程入口：任何意外都要报给主线程，否则就是「语音突然没反应」还不留痕迹。"""
        try:
            self._run_inner()
        except Exception:
            import traceback

            self._fail("麦克风采集线程异常退出：\n" + traceback.format_exc())

    def _run_inner(self) -> None:
        cfg = self.cfg
        gain = cfg.input_gain
        preroll: collections.deque = collections.deque(maxlen=cfg.preroll_frames)
        buf: list = []
        speaking = False
        hangover = 0
        run = 0
        try:
            stream = self._open_stream()
        except Exception as exc:
            self._fail(f"打不开麦克风（设备号 {self.device}）：{exc}\n{MIC_HINT}")
            return

        with stream:
            self._calibrate(stream)
            self._log(f"开始监听（静音 {cfg.silence_ms}ms 判定说完，最短 {cfg.min_speech_ms}ms）")

            while not self._stop.is_set():
                try:
                    data, _overflow = stream.read(self.frame_len)
                except Exception as exc:
                    self._fail(f"读麦克风失败（设备被拔掉/被别的程序独占？）：{exc}")
                    return

                frame = np.ascontiguousarray(data[:, 0], dtype=np.float32)
                if gain != 1.0:
                    frame = np.clip(frame * gain, -1.0, 1.0)
                # 音量要在这里算（静音时也要算），供主动性判断房间有没有人
                self._level = 0.9 * self._level + 0.1 * float(
                    np.sqrt(np.mean(np.square(frame, dtype=np.float32))) + 1e-12)

                if self._muted.is_set():
                    if speaking or preroll:
                        speaking, hangover, run = False, 0, 0
                        buf.clear()
                        preroll.clear()
                    continue

                speech = self.vad.is_speech(frame)
                level = float(np.sqrt(np.mean(np.square(frame, dtype=np.float32))) + 1e-12)
                self.stats["frames"] += 1

                if not speaking:
                    preroll.append(frame)
                    if speech:
                        speaking, hangover, run = True, 0, 1
                        buf = list(preroll)
                        self._log(f"检测到你说话（电平 {level:.4f}）")
                    continue

                buf.append(frame)
                if speech:
                    hangover = 0
                    run += 1
                    if run == self._confirm:
                        self._signal(self.speech_started.set)
                else:
                    hangover += 1

                held = len(buf) * cfg.frame_ms / 1000.0
                if not hasattr(self, "_warned") or not self._warned:
                    if speaking and held > 6.0 and hangover == 0:
                        self._warned = True
                        print(f"[mic] 警告：已经连续 {held:.0f} 秒判定为「在说话」，很可能环境噪声太大、")
                        print("      或者麦克风加强开太高，VAD 分不出你和噪声。建议把「麦克风加强」调低一档，")
                        print("      或调大 .env 的 ENERGY_RATIO（现在 %s）。" % cfg.energy_ratio)
                too_long = held >= cfg.max_utterance_s
                if hangover * cfg.frame_ms >= cfg.silence_ms or too_long:
                    audio = np.concatenate(buf) if buf else np.zeros(0, dtype=np.float32)
                    speaking, hangover, run = False, 0, 0
                    buf.clear()
                    preroll.clear()
                    seconds = len(audio) / float(cfg.sample_rate)
                    if seconds * 1000.0 >= cfg.min_speech_ms:
                        self.stats["utterances"] += 1
                        self._log(f"一句说完 {seconds:.1f}s（峰值 {float(np.max(np.abs(audio))):.3f}）→ 送识别")
                        self._signal(self.queue.put_nowait, audio)
                    else:
                        self._log(f"太短（{seconds:.2f}s）当杂音丢掉")

    async def get_utterance(self) -> np.ndarray:
        audio = await self.queue.get()
        if audio is None:
            raise RuntimeError(self.error or "麦克风异常退出")
        return audio


class Player:
    """把 PCM 片段排进播放队列；stop() 用于打断（立刻静音 + 丢掉没播的）。"""

    def __init__(self, cfg, sample_rate: int):
        self.sample_rate = sample_rate
        self.cfg = cfg
        self.device = resolve_device(cfg.output_device, output=True, prefer_wasapi=cfg.prefer_wasapi)
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._lock = threading.Lock()
        self._current: Optional[np.ndarray] = None
        self._pos = 0
        self._stream: Optional[sd.OutputStream] = None
        self._last_active = 0.0
        self._level = 0.0

    def start(self) -> None:
        kwargs = dict(samplerate=self.sample_rate, channels=1, dtype="float32",
                      blocksize=0, device=self.device, callback=self._callback)
        try:
            self._stream = sd.OutputStream(**kwargs, latency="low")
        except Exception:
            try:
                self._stream = sd.OutputStream(**kwargs)
            except Exception as exc:
                raise SystemExit(
                    f"打不开扬声器（设备号 {self.device}）：{exc}\n"
                    f"  用 --devices 看设备号，然后在 .env 里写 OUTPUT_DEVICE=5（比如内置扬声器）"
                ) from exc
        self._stream.start()

    def close(self) -> None:
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        self._stream = None

    def _callback(self, outdata, frames, time_info, status) -> None:
        filled = 0
        with self._lock:
            while filled < frames:
                if self._current is None:
                    try:
                        self._current = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    self._pos = 0
                take = min(frames - filled, len(self._current) - self._pos)
                outdata[filled : filled + take, 0] = self._current[self._pos : self._pos + take]
                self._last_active = time.monotonic()
                self._pos += take
                filled += take
                if self._pos >= len(self._current):
                    self._current = None
        # 播放电平（RMS）：Live2D 口型的唯一数据源。
        # 起音立即跟上、收音慢一点，嘴形不会抖。
        if filled > 0:
            block = outdata[:filled, 0]
            rms = float(np.sqrt(np.dot(block, block) / filled))
            self._level = max(rms, self._level * 0.55)
        else:
            self._level *= 0.5
        if filled < frames:
            outdata[filled:, 0] = 0.0

    def play(self, pcm: np.ndarray) -> None:
        if pcm is None or len(pcm) == 0:
            return
        self._queue.put(np.ascontiguousarray(pcm, dtype=np.float32))

    def stop(self) -> None:
        with self._lock:
            self._current = None
            self._pos = 0
            self._level = 0.0
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

    def active_recently(self, seconds: float = 2.0) -> bool:
        """刚才（或现在）是不是在放她的声音 —— 用来识别"麦克风串回自己的声音"。"""
        return (time.monotonic() - self._last_active) < seconds

    @property
    def level(self) -> float:
        """当前播放电平 RMS（0~1）—— Live2D 口型的数据源。"""
        return self._level

    @property
    def idle(self) -> bool:
        with self._lock:
            return self._current is None and self._queue.empty()

    async def wait_drained(self, tail: float = 0.15) -> None:
        while not self.idle:
            await asyncio.sleep(0.02)
        await asyncio.sleep(tail)