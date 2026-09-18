r"""恋恋的情绪系统。

两个方向，别混：

  【感知】哥哥的情绪 —— 来自 SenseVoice 的情感标签（语音里自带的，免费且真实）
                        → 影响她怎么回应（共情 / 打趣 / 嘴硬心软）

  【表达】她自己的情绪 —— 来自大模型输出的标签
                        → 影响 TTS（语速，可选换参考音频）+ 以后的 Live2D 表情

再加一个**心情值**：情绪会随时间平复，不会被某一句话永久带偏。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

# 她能表达的情绪（也是给大模型选的范围）
EMOTIONS = ("neutral", "happy", "shy", "angry", "sad", "surprised")
ZH = {"neutral": "平静", "happy": "开心", "shy": "害羞", "angry": "生气",
      "sad": "低落", "surprised": "惊讶"}

# 语速倍率：激动说得快，低落说得慢
SPEED = {"neutral": 1.0, "happy": 1.08, "shy": 0.95,
         "angry": 1.12, "sad": 0.90, "surprised": 1.15}

# SenseVoice 的情感标签 -> 我们的说法
SENSEVOICE_EMO = {"HAPPY": "happy", "SAD": "sad", "ANGRY": "angry",
                  "NEUTRAL": "neutral", "FEARFUL": "sad", "DISGUSTED": "angry",
                  "SURPRISED": "surprised"}
# SenseVoice 的事件标签（笑声之类）也有用
SENSEVOICE_EVENT = {"Laughter": "在笑", "Applause": "在鼓掌", "Cry": "在哭",
                    "Sneeze": "打喷嚏", "Cough": "咳嗽", "Sigh": "叹气", "BGM": "有背景音乐"}

_TAG = re.compile(r"<\|([^|]+)\|>")


@dataclass
class Mood:
    """她当前的心情。"""
    name: str = "neutral"
    intensity: float = 0.0
    since: float = 0.0

    def get(self) -> str:
        return self.name if self.intensity >= 0.15 else "neutral"

    def zh(self) -> str:
        return ZH.get(self.get(), "平静")


class EmotionState:
    """心情状态机：会被影响、也会自己平复。

    用法：
        st = EmotionState()
        st.update("happy", 0.8)        # 她被逗笑了
        st.decay()                     # 每次用之前先衰减
        st.get()                       # -> "happy" / "neutral"
        st.speed_factor()              # 给 TTS 的语速倍率
    """

    def __init__(self, half_life: float = 180.0, floor: float = 0.15):
        self.half_life = float(half_life)   # 情绪强度衰减一半需要的秒数
        self.floor = float(floor)
        self.mood = Mood()
        self.log: list[tuple[float, str, float]] = []    # (时间, 情绪, 强度)

    # ---------------------------------------------------------------- 更新
    def update(self, emotion: str, intensity: float = 0.7) -> None:
        """被某句话影响。强度取「新情绪」和「残留情绪」里更强的那个。"""
        name = (emotion or "neutral").strip().lower()
        if name not in EMOTIONS:
            name = "neutral"
        self.decay()
        intensity = max(0.0, min(1.0, float(intensity)))
        if name == "neutral":
            self.mood = Mood("neutral", 0.0, time.time())
            return
        # 同一种情绪会叠加，换了一种情绪则看谁更强
        if name == self.mood.name:
            intensity = min(1.0, self.mood.intensity + intensity * 0.5)
        elif intensity < self.mood.intensity:
            return                      # 原来的情绪更强烈，不被覆盖
        self.mood = Mood(name, intensity, time.time())
        self.log.append((time.time(), name, intensity))
        if len(self.log) > 50:
            self.log = self.log[-50:]

    def decay(self, now: float | None = None) -> None:
        """按半衰期平复。"""
        now = now or time.time()
        if self.mood.intensity <= 0:
            return
        el = now - self.mood.since
        if el > 0:
            self.mood.intensity *= 0.5 ** (el / self.half_life)
            self.mood.since = now
        if self.mood.intensity < self.floor:
            self.mood = Mood("neutral", 0.0, now)

    # ---------------------------------------------------------------- 查询
    def get(self) -> str:
        self.decay()
        return self.mood.get()

    def zh(self) -> str:
        self.decay()
        return self.mood.zh()

    def speed_factor(self) -> float:
        return SPEED.get(self.get(), 1.0)

    def hint(self) -> str:
        """给大模型的一段提示，让她知道自己现在什么心情。"""
        self.decay()
        m = self.mood
        if m.intensity < self.floor:
            return ""
        return (f"【你现在的心情：{m.zh()}（强度 {m.intensity:.1f}/1）。"
                f"让它自然体现在语气里，但别直接说出来。】\n")


def parse_sensevoice(text: str) -> tuple[str, str, str]:
    """从 SenseVoice 的输出里剥出 (干净文本, 情绪, 事件)。

    SenseVoice 的原始输出形如：<|zh|><|HAPPY|><|Laughter|><|withitn|>真实文本
    以前这些标签是被直接丢掉的 —— 其实**情感和笑声都是免费的真实信号**。
    """
    emo, event = "neutral", ""
    for raw in _TAG.findall(text or ""):
        tag = raw.strip()
        if tag.upper() in SENSEVOICE_EMO:
            emo = SENSEVOICE_EMO[tag.upper()]
        elif tag in SENSEVOICE_EVENT:
            event = SENSEVOICE_EVENT[tag]
    clean = _TAG.sub("", text or "").strip()
    return clean, emo, event


def user_hint(emo: str, event: str = "") -> str:
    """把哥哥的情绪变成给大模型的一句话。"""
    if emo == "neutral" and not event:
        return ""
    parts = []
    if emo == "happy":
        parts.append("听起来心情不错")
    elif emo == "sad":
        parts.append("听起来有点低落")
    elif emo == "angry":
        parts.append("听起来有点火气")
    elif emo == "surprised":
        parts.append("听起来有点意外")
    if event:
        parts.append(event)
    return ("【哥哥现在" + "、".join(parts) + "。自然回应就好，别直接点破。】\n") if parts else ""