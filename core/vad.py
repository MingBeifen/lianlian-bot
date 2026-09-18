"""说话检测（VAD）

优先级：silero（神经网络，嘈杂环境最稳）→ webrtcvad → 能量法（零依赖保底）

能量法在噪声大的房间里天生吃力：门槛按底噪倍数算，底噪一高门槛就盖过说话音量。
所以只要装了 silero-vad / webrtcvad，程序会自动优先用它们。
"""
from __future__ import annotations

import numpy as np


class BaseVAD:
    def is_speech(self, frame: np.ndarray) -> bool:
        raise NotImplementedError

    def rms(self, audio: np.ndarray) -> float:
        """给校准用的、和内部判定同一套处理后的电平。"""
        return float(np.sqrt(np.mean(np.square(audio, dtype=np.float32))) + 1e-12)


class EnergyVAD(BaseVAD):
    """零依赖保底方案：自适应噪声地板 + 一阶高通（砍掉风扇/空调低频轰鸣）。"""

    def __init__(self, ratio: float = 2.2, min_rms: float = 0.0005,
                 sample_rate: int = 16000, hp_hz: float = 150.0):
        self.ratio = max(1.2, float(ratio))
        self.min_rms = max(1e-6, float(min_rms))
        self.noise = max(1e-4, self.min_rms)
        self._a = float(np.exp(-2.0 * np.pi * hp_hz / sample_rate))
        self._px = 0.0
        self._py = 0.0

    def _highpass(self, x: np.ndarray) -> np.ndarray:
        a, px, py = self._a, self._px, self._py
        y = np.empty_like(x)
        for i in range(x.size):
            v = float(x[i])
            py = a * (py + v - px)
            px = v
            y[i] = py
        self._px, self._py = px, py
        return y

    def rms(self, audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(self._highpass(audio), dtype=np.float32))) + 1e-12)

    def is_speech(self, frame: np.ndarray) -> bool:
        level = self.rms(frame)
        speech = level > max(self.noise * self.ratio, self.min_rms)
        # 说话时几乎不更新噪声地板，安静时快速跟上
        alpha = 0.998 if speech else 0.97
        self.noise = min(max(alpha * self.noise + (1 - alpha) * level, 1e-7), 0.3)
        return speech


class WebrtcVAD(BaseVAD):
    """webrtcvad 要 10/20/30ms 的 16bit PCM，这里按 20ms 切块投票。"""

    def __init__(self, aggressiveness: int = 2, sample_rate: int = 16000):
        import webrtcvad

        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.step = int(sample_rate * 0.02) * 2

    def is_speech(self, frame: np.ndarray) -> bool:
        pcm = (np.clip(frame, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        votes = [
            self.vad.is_speech(pcm[i : i + self.step], self.sample_rate)
            for i in range(0, len(pcm) - self.step + 1, self.step)
        ]
        return bool(votes) and sum(votes) * 2 >= len(votes)


class SileroVAD(BaseVAD):
    """silero-vad：小神经网络 VAD，噪声环境下比能量法强很多。

    安装：venv_app\\Scripts\\pip install silero-vad   （torch 你已经有了）
    """

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000):
        import torch
        from silero_vad import load_silero_vad

        if sample_rate != 16000:
            raise ValueError("silero 需要 16k 采样率")
        self.torch = torch
        self.model = load_silero_vad()
        self.threshold = float(threshold)
        self.chunk = 512          # 16k 下 silero 固定吃 512 点
        self._buf = np.zeros(0, dtype=np.float32)

    def is_speech(self, frame: np.ndarray) -> bool:
        self._buf = np.concatenate([self._buf, frame.astype(np.float32, copy=False)])
        hit = False
        while self._buf.size >= self.chunk:
            block = np.ascontiguousarray(self._buf[: self.chunk])
            self._buf = self._buf[self.chunk :]
            with self.torch.no_grad():
                tensor = self.torch.from_numpy(block)
                try:
                    prob = float(self.model(tensor, 16000).item())
                except TypeError:
                    prob = float(self.model(tensor).item())
            hit = hit or prob >= self.threshold
        return hit


class GatedVAD(BaseVAD):
    """在神经网络 VAD 外面再套一层能量闸门（silero AND 够响）。

    噪声大的环境里，silero 会把稳态噪声也判成"说话"（实测：底噪 0.018，
    它照样输出 >0.5），结果就是一整句话永远结束不了 —— 表现就是"说话没反应"。
    这里要求同时满足：神经网络说你在说话 **且** 音量显著高于自适应底噪。
    """

    def __init__(self, inner: BaseVAD, ratio: float = 1.6, min_rms: float = 3e-5):
        self.inner = inner
        self.ratio = max(1.05, float(ratio))
        self.min_rms = max(1e-7, float(min_rms))
        self.noise = 1e-3

    def rms(self, audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(audio, dtype=np.float32))) + 1e-12)

    def is_speech(self, frame: np.ndarray) -> bool:
        level = self.rms(frame)
        loud = level > max(self.noise * self.ratio, self.min_rms)
        if not loud:
            # 只在不"响"的时候更新底噪，免得把说话声当成环境噪声
            self.noise = min(max(0.97 * self.noise + 0.03 * level, 1e-7), 0.3)
        neural = self.inner.is_speech(frame)
        return bool(neural and loud)


def make_vad(cfg) -> BaseVAD:
    backend = (cfg.vad_backend or "auto").lower()
    order = ["silero", "webrtc", "energy"] if backend == "auto" else [backend]
    for name in order:
        try:
            if name == "silero":
                vad = SileroVAD(cfg.vad_threshold, cfg.sample_rate)
                if cfg.vad_gate:
                    vad = GatedVAD(vad, cfg.energy_ratio * 0.7)
                print(f"[vad] 使用 silero 神经网络 VAD（阈值 {cfg.vad_threshold}"
                      + ("，带能量闸门）" if cfg.vad_gate else "）"))
                return vad
            if name == "webrtc":
                vad = WebrtcVAD(2, cfg.sample_rate)
                if cfg.vad_gate:
                    vad = GatedVAD(vad, cfg.energy_ratio * 0.7)
                print("[vad] 使用 webrtcvad" + ("（带能量闸门）" if cfg.vad_gate else ""))
                return vad
            if name == "energy":
                print(f"[vad] 使用能量 VAD（阈值 x{cfg.energy_ratio}，最低音量 {cfg.vad_min_rms}）")
                return EnergyVAD(cfg.energy_ratio, cfg.vad_min_rms, cfg.sample_rate)
        except Exception as exc:
            if backend != "auto":
                raise
            print(f"[vad] {name} 不可用（{type(exc).__name__}: {exc}），换下一个")
    return EnergyVAD(cfg.energy_ratio, cfg.vad_min_rms, cfg.sample_rate)