"""恋恋 · 语音 + 键盘 混合对话

    麦克风 → VAD 断句 → SenseVoice 识别 ┐
                                        ├→ 恋恋(Ollama) → Kokoro 合成 → 扬声器
    键盘打字（随时可插） ───────────────┘

语音和打字抢同一个输入队列，谁先来算谁。她说话的时候，你说话或者打字都能打断她。

    run.bat                         混合模式（默认）
    run.bat --text                  只用键盘，不开麦克风
    run.bat --wav a.wav --out reply.wav
    run.bat --devices               看设备 + 测麦克风
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]          # voicebot/
for _p in (str(_ROOT), str(_ROOT / "core"), str(_ROOT / "live2d")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import asyncio
import difflib
import collections
import sys
import threading
import time

import numpy as np

from config import CONFIG
from asr import is_noise, make_asr
from audio_io import MicListener, Player, list_devices, mic_selftest
from avatar import make_avatar
from llm import ChatLLM
from memory import MemoryExtractor, MemoryStore
from proactive import BotContext, ProactiveBrain
from tts import make_tts
from vad import make_vad

HARD = "。！？!?…\n；;"
SOFT = "，,、：:～~"
MAX_CHUNK = 40      # 一句话最多攒这么多字，超了硬切
SOFT_MIN = 16       # 逗号处至少攒这么多字才切，免得碎成一片

EOF = object()      # 键盘输入结束的哨兵


def _break_at(buf: str) -> int:
    for i, ch in enumerate(buf):
        if ch in HARD and i + 1 >= 2:
            return i + 1
        if ch in SOFT and i + 1 >= SOFT_MIN:
            return i + 1
    if len(buf) >= MAX_CHUNK:
        return len(buf)
    return -1


async def sentences(tokens):
    """把流式 token 攒成一句句吐出去，让 TTS 不用等整段生成完。"""
    buf = ""
    async for tok in tokens:
        buf += tok
        while True:
            i = _break_at(buf)
            if i < 0:
                break
            piece, buf = buf[:i], buf[i:]
            if piece.strip():
                yield piece.strip()
    if buf.strip():
        yield buf.strip()


class EchoGuard:
    """防止她把扬声器里自己的声音当成用户输入（自问自答死循环）。

    实测：外放时她的回复被麦克风收进去 → 当成用户说话 → 她再回 → 无限循环，
    而且每轮都会触发一次"打断"。这里拿识别结果和她最近说过的话比一比，像就丢掉。
    """

    def __init__(self, history: int = 8, threshold: float = 0.7):
        self.recent: collections.deque = collections.deque(maxlen=history)
        self.threshold = threshold
        self.dropped = 0

    def remember(self, text: str) -> None:
        t = (text or "").strip()
        if t:
            self.recent.append(t)

    def is_echo(self, text: str, playing_recently: bool = False) -> bool:
        t = (text or "").strip()
        if len(t) < 2:
            return False
        # 她自己的声音串回来时，先被"打断"截断、再被 VAD 切碎，
        # 识别出来往往是"真的""好的"这种几字碎片，所以短 + 刚在放音 = 判为回声
        if playing_recently and len(t) <= 6:
            return True
        for old in self.recent:
            if len(t) >= 3 and t in old:          # 整句都是她刚说过的一部分
                return True
            if difflib.SequenceMatcher(None, t, old).ratio() >= self.threshold:
                return True
        return False


class TypedInput:
    """后台线程读键盘，回车即算一句用户输入，塞进跟麦克风共用的队列。

    用线程而不是 asyncio 的 stdin 管道：Windows 上 asyncio 加不了控制台输入，
    而线程里的 readline() 会老实阻塞等人敲回车。
    """

    def __init__(self, loop, queue: asyncio.Queue, signal: asyncio.Event,
                 eof_stops: bool, busy=None):
        self.loop = loop
        self.queue = queue
        self.signal = signal
        self.eof_stops = eof_stops
        self.busy = busy or (lambda: False)
        self._thread = threading.Thread(target=self._run, name="keyboard", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while True:
            try:
                line = sys.stdin.readline()
            except Exception:
                line = ""
            if not line:                       # EOF：管道读完 / Ctrl+Z
                if self.eof_stops:
                    self._push(EOF)
                else:
                    print("[键盘] 输入流已关闭，继续语音模式")
                return
            line = line.strip()
            if line:
                self._push(line)

    def _push(self, item) -> None:
        try:
            self.loop.call_soon_threadsafe(self._deliver, item)
        except RuntimeError:
            pass

    def _deliver(self, item) -> None:
        # 关键：先看她是不是正在出声。没在出声时打字不能触发"打断"，
        # 否则刚收到这句就把它自己那一轮给取消了（同一瞬间的竞态）。
        try:
            busy = bool(self.busy())
        except Exception:
            busy = False
        self.queue.put_nowait(item)
        if busy:
            self.signal.set()   # 她正在说的话，立刻闭嘴


class VoiceBot:
    def __init__(self, cfg):
        self.cfg = cfg
        self.inputs: asyncio.Queue = asyncio.Queue()
        self.interrupt = asyncio.Event()   # 说话 or 打字 → 打断她
        self.mic = None
        self.typed = None
        self.player = None
        self.avatar = None
        self.pet = None
        self.native_pet = None
        self.asr = None
        self.tts = None
        self.llm = None
        self.turn = None
        self._barged = False
        self.collected: list = []
        self.echo = EchoGuard()
        self.memory = None
        self.extractor = None
        self.brain = None
        self._last_turn_end = time.time()
        self._mem_task = None
        self.webui = None
        self.event_sinks: list = []       # Web 控制台事件订阅
        self.mic_user_muted = False       # 网页上手动静音的标记

    # ---------------- 初始化 ----------------
    async def setup(self, with_mic: bool = True, with_typing: bool = False) -> None:
        cfg = self.cfg
        loop = asyncio.get_running_loop()
        print("[1/5] 语音识别 ...")
        self.asr = make_asr(cfg)
        print("[2/5] 语音合成 ...")
        self.tts = make_tts(cfg)
        print("[3/5] 大模型 ...")
        self.llm = ChatLLM(cfg)
        await self.llm.warmup()
        # 音频设备放到最后开：加载模型要几十秒，万一麦克风有问题也不挡在前面
        if cfg.memory_enable:
            print("[4/5] 长期记忆 ...")
            try:
                self.memory = MemoryStore(cfg.memory_db or None)
                await asyncio.to_thread(self.memory.warmup)
                self.llm.memory = self.memory
                self.extractor = MemoryExtractor(cfg, self.memory)
                print(f"[memory] 就绪，库里现有 {len(self.memory)} 条记忆"
                      + ("（自动写入已开）" if cfg.memory_auto_write else "（自动写入已关）"))
            except Exception as exc:
                # 嵌入模型没下下来 / 没网 也不该让整个程序起不来
                self.memory = None
                print(f"[memory] 不可用（{type(exc).__name__}: {str(exc)[:80]}）")
                print("[memory] 这轮先不带记忆，其余功能正常。想启用就联网跑一次 python memory.py --demo")
        print("[5/5] 打开音频设备 ...")
        # 播放采样率必须跟着 TTS 后端走！Kokoro/Edge 是 24k，GPT-SoVITS 是 32k，
        # 用错会让音调偏移（实测 32k 按 24k 播 → 237Hz 的女声变成 178Hz 的男声）
        rate = int(getattr(self.tts, "sample_rate", cfg.tts_sample_rate))
        if rate != cfg.tts_sample_rate:
            print(f"[audio] 播放采样率跟随 TTS 后端: {rate} Hz（配置里写的是 {cfg.tts_sample_rate}）")
        self.player = Player(cfg, rate)
        self.player.start()
        backend = str(getattr(cfg, "pet_backend", "vtube")).lower()
        if backend == "native" and cfg.pet_enable:
            # 自渲染桌宠：不依赖 VTube Studio，渲染器和主程序走 TCP
            try:
                from native_pet_link import NativePetLink
                self.native_pet = NativePetLink(cfg, lambda: self.player.level)
                self.native_pet.start()
                print("[pet] 自渲染桌宠启动中（live2d-py + GLFW，日志 native_pet.log）")
            except Exception as exc:
                self.native_pet = None
                print(f"[pet] 自渲染桌宠没起来（{type(exc).__name__}: {exc}）；其余功能正常")
        elif cfg.live2d_enable:
            self.avatar = make_avatar(cfg, self.player)
            if self.avatar is not None:
                self.avatar.start()
                print(f"[avatar] Live2D 已启用：{cfg.live2d_url}"
                      "（首次连接要在 VTube Studio 里点 Allow；没开也不影响聊天）")
        if cfg.pet_enable and backend != "native":
            try:
                from desktop_pet import DesktopPet
                self.pet = DesktopPet(cfg)
                self.pet.start()
            except Exception as exc:
                self.pet = None
                print(f"[pet] 桌宠模式没起来（{type(exc).__name__}: {exc}）；其余功能正常")
        if with_mic:
            self.mic = MicListener(cfg, make_vad(cfg), loop,
                                   out_queue=self.inputs, speech_event=self.interrupt)
            self.mic.start()
        if with_typing:
            # 只有她"正在出声"时，打字才算打断；她还在思考（还没出声）时打字只是排队，
            # 否则连发两条消息会把第一条那轮直接取消掉（实测踩过）
            self.typed = TypedInput(loop, self.inputs, self.interrupt, eof_stops=not with_mic,
                                    busy=lambda: self.player is not None
                                                and self.player.active_recently(1.2))
            self.typed.start()
        if cfg.proactive_enable:
            self.brain = ProactiveBrain(cfg, client=self.llm.client, model=cfg.llm_model,
                                        memory=self.memory, history=lambda: self.llm.history)
            print("[proactive] 主动性已开启（她自己会找时机开口）")
        if cfg.webui_enable:
            try:
                from webui.server import WebUI
                self.webui = WebUI(self)
                await self.webui.start()
            except Exception as exc:
                self.webui = None
                print(f"[webui] 启动失败：{type(exc).__name__}: {exc}")

    async def close(self) -> None:
        if self.webui is not None:
            await self.webui.stop()
            self.webui = None
        # 最后那句的记忆可能还在抽，等它写完再退（最多 15 秒）
        if self._mem_task is not None and not self._mem_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._mem_task), timeout=15)
                print("[memory] 最后一条记忆已写入")
            except Exception:
                pass
        if self.native_pet is not None:
            await self.native_pet.stop()
            self.native_pet = None
        if self.pet is not None:
            self.pet.stop()
            self.pet = None
        if self.avatar is not None:
            print(f"[avatar] {self.avatar.describe()}")
            await self.avatar.close()
            self.avatar = None
        if self.mic:
            self.mic.stop()
        if self.player:
            self.player.stop()
            self.player.close()

    # ---------------- Web 控制台接口 ----------------
    @property
    def busy(self) -> bool:
        return self.turn is not None and not self.turn.done()

    async def emit(self, event: dict) -> None:
        for sink in list(self.event_sinks):
            try:
                await sink(event)
            except Exception:
                pass

    def submit_web_text(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self.inputs.put_nowait(text)
        if self.player is not None and self.player.active_recently(1.2):
            self.interrupt.set()

    def submit_web_audio(self, audio) -> None:
        if audio is None or len(audio) == 0:
            return
        self.inputs.put_nowait(np.asarray(audio, dtype=np.float32))

    def set_local_mic_muted(self, muted: bool) -> None:
        self.mic_user_muted = bool(muted)
        if self.mic is not None:
            self.mic.set_muted(self.mic_user_muted)
        print(f"[webui] 本机麦克风：{'静音' if self.mic_user_muted else '开麦'}")

    async def apply_llm(self, values: dict) -> dict:
        for key, val in values.items():
            setattr(self.cfg, key, val)
        self.llm = ChatLLM(self.cfg)
        self.llm.memory = self.memory
        await self.llm.warmup()
        if self.brain is not None:
            self.brain.client = self.llm.client
            self.brain.model = self.cfg.llm_model
        return {"model": self.cfg.llm_model, "base_url": self.cfg.llm_base_url}

    async def apply_asr(self, values: dict) -> dict:
        for key, val in values.items():
            setattr(self.cfg, key, val)
        self.asr = await asyncio.to_thread(make_asr, self.cfg)
        return {"backend": self.cfg.asr_backend, "model": self.cfg.asr_model}

    async def apply_tts(self, values: dict) -> dict:
        for key, val in values.items():
            setattr(self.cfg, key, val)
        tts = await asyncio.to_thread(make_tts, self.cfg)
        self.tts = tts
        rate = int(getattr(tts, "sample_rate", self.cfg.tts_sample_rate))
        if self.player is not None and rate != self.player.sample_rate:
            self.player.stop()
            self.player.close()
            self.player = Player(self.cfg, rate)
            self.player.start()
        return {"backend": self.cfg.tts_backend, "voice": self.cfg.tts_voice,
                "sample_rate": rate}

    async def apply_vad(self, values: dict) -> dict:
        for key, val in values.items():
            setattr(self.cfg, key, val)
        if self.mic is not None:
            flag = getattr(self.mic, "_muted", None)
            was_muted = bool(flag.is_set()) if hasattr(flag, "is_set") else False
            self.mic.stop()
            self.mic = MicListener(self.cfg, make_vad(self.cfg), asyncio.get_running_loop(),
                                   out_queue=self.inputs, speech_event=self.interrupt)
            self.mic.set_muted(was_muted)
            self.mic.start()
        return {"backend": self.cfg.vad_backend, "threshold": self.cfg.vad_threshold}

    async def apply_memory(self, values: dict) -> dict:
        import os
        old_store = self.memory
        for key, val in values.items():
            if key == "memory_embed_model_bge":
                os.environ["MEMORY_EMBED_MODEL_BGE"] = str(val)
                continue
            setattr(self.cfg, key, val)
        if not self.cfg.memory_enable:
            return {"enabled": False}
        store = MemoryStore(self.cfg.memory_db or None)
        await asyncio.to_thread(store.warmup)
        self.memory = store
        if self.llm is not None:
            self.llm.memory = store
        self.extractor = MemoryExtractor(self.cfg, store)
        if old_store is not None:
            try:
                old_store.close()
            except Exception:
                pass
        return {"count": len(store), "top_k": self.cfg.memory_top_k}

    async def apply_pet(self, values: dict) -> dict:
        for key, val in values.items():
            setattr(self.cfg, key, val)
        if self.native_pet is not None:
            await self.native_pet.restart_renderer()
        return {"backend": self.cfg.pet_backend}

    # ---------------- Live2D ----------------
    async def _avatar_emotion(self, text: str = "") -> None:
        """把这句话的情绪 + 伴随动作同步给 Live2D；失败也不影响聊天。"""
        if self.avatar is None and self.native_pet is None:
            return
        text = (text or "").strip()
        if not text:
            return
        want_emo = bool(self.cfg.emotion_enable)
        want_act = bool(getattr(self.cfg, "llm_action", True)
                        and self.native_pet is not None)

        async def _emo():
            try:
                return await self.llm.classify_emotion(text)
            except Exception:
                return "neutral"

        async def _act():
            try:
                return await self.llm.classify_action(text)
            except Exception:
                return "none"

        emo, act = await asyncio.gather(
            _emo() if want_emo else asyncio.sleep(0, result="neutral"),
            _act() if want_act else asyncio.sleep(0, result="none"))
        if want_emo:
            self.llm.last_emotion = emo
            if self.avatar is not None:
                self.avatar.set_emotion(emo)
            if self.native_pet is not None:
                self.native_pet.set_emotion(emo)
            if self.cfg.debug_emotion:
                print(f"[emotion] 表情 -> {emo}")
        self.llm.last_action = act
        if want_act and act != "none":
            self.native_pet.set_action(act)
            if self.cfg.debug_emotion:
                print(f"[action] 动作 -> {act}")

    def _last_reply_text(self) -> str:
        for msg in reversed(self.llm.history):
            if msg.get("role") == "assistant":
                return str(msg.get("content") or "")
        return ""

    # ---------------- 说 -----------------
    async def say(self, text: str) -> None:
        """直接让她出声（不过大模型），给问候语用。"""
        print(f"恋恋 ：{text}")
        pcm = await self.tts.synth(text)
        self.player.play(pcm)
        await self._avatar_emotion(text)
        await self.player.wait_drained()
        await self.emit({"type": "assistant_done", "text": text})

    async def _stream_out(self, user_text: str) -> None:
        """大模型流式回复：一边出字一边合成一边播。"""
        cfg = self.cfg
        await self.emit({"type": "state", "state": "thinking"})
        print("恋恋 ：", end="", flush=True)
        first_at = None
        async for sentence in sentences(self.llm.stream_reply(user_text)):
            print(sentence, end="", flush=True)
            try:
                pcm = await self.tts.synth(sentence)
            except Exception as exc:
                # 单句合成失败（网络抖动 / 文本被服务端拒绝）不该让整轮崩掉
                print(f"\n[tts] 这句合成失败，跳过：{type(exc).__name__}: {str(exc)[:80]}")
                continue
            if first_at is None:
                first_at = time.perf_counter()
            self.echo.remember(sentence)
            self.collected.append(pcm)
            await self.emit({"type": "assistant_sentence", "text": sentence})
            self.player.play(pcm)
        print()
        # Live2D：趁语音还在播，判断这句话的情绪并切表情（不额外占用等待时间）
        if self.avatar is not None:
            await self._avatar_emotion(self._last_reply_text())
        # 她正在出声，这几秒 GPU 是空的 —— 正好拿来抽记忆，等于零延迟成本
        if self.extractor is not None:
            self._mem_task = asyncio.create_task(self.extractor.write(user_text))
        if cfg.debug_latency and first_at:
            print(f"       [首句出声 {first_at - self._t_in:.2f}s | 整轮 {time.perf_counter() - self._t_in:.2f}s]")
        await self.player.wait_drained()
        await self.emit({"type": "assistant_done", "text": self._last_reply_text()})
        await self.emit({"type": "state", "state": "idle"})

    # ---------------- 两种输入各走一遍 ----------------
    async def handle_audio(self, audio: np.ndarray) -> None:
        self._t_in = time.perf_counter()
        text = await self.asr.transcribe(audio)
        if is_noise(text):
            print(f"[忽略] 没听清（{text!r}）")
            return
        if self.echo.is_echo(text, self.player.active_recently(2.0)):
            self.echo.dropped += 1
            print(f"[跳过] 像是扬声器串回来的我自己的声音（连续 {self.echo.dropped} 次）：{text!r}")
            if self.echo.dropped >= 3 and self.cfg.barge_in:
                # 自愈：外放时她自己听见自己 → 干脆切成半双工，她说话时不听
                self.cfg.barge_in = False
                print("[自动] 检测到「她听见自己」的死循环，已切换成半双工模式：")
                print("       她说话期间麦克风自动闭上。想恢复说话打断就戴耳机，并设 BARGE_IN=1。")
            return
        self.echo.dropped = 0
        print(f"你   ：{text}    \t[识别 {time.perf_counter() - self._t_in:.2f}s]")
        await self.emit({"type": "user_text", "text": text, "source": "voice"})
        await self._stream_out(text)

    async def handle_text(self, text: str) -> None:
        self._t_in = time.perf_counter()
        self.echo.dropped = 0
        print(f"你   ：{text}    \t[键盘]")
        await self.emit({"type": "user_text", "text": text, "source": "keyboard"})
        await self._stream_out(text)

    # ---------------- 打断 ----------------
    async def _speak_proactive(self, text: str) -> None:
        """把她主动想说的话说出来（不再走一遍大模型，直接合成）。"""
        self.echo.remember(text)
        self.llm.history.append({"role": "assistant", "content": text})
        try:
            pcm = await self.tts.synth(text)
        except Exception as exc:
            print(f"[主动] 合成失败：{type(exc).__name__}: {str(exc)[:70]}")
            return
        half_duplex = not self.cfg.barge_in and self.mic is not None
        if half_duplex:
            self.mic.set_muted(True)
        try:
            self.player.play(pcm)
            await self._avatar_emotion(text)
            await self.player.wait_drained()
        finally:
            if half_duplex:
                self.player.stop()
                self.mic.set_muted(False)

    async def _proactive_loop(self) -> None:
        """每隔一会儿问一次大脑：现在要不要自己开口。"""
        cfg = self.cfg
        while True:
            await asyncio.sleep(max(5.0, cfg.proactive_poll_sec))
            if self.turn is not None and not self.turn.done():
                continue
            ctx = BotContext(
                idle_sec=time.time() - self._last_turn_end,
                busy=False,
                noise_rms=self.mic.recent_rms() if self.mic else 0.0,
            )
            try:
                text = await self.brain.maybe_speak(ctx)
            except Exception as exc:
                print(f"[主动] 出错：{type(exc).__name__}: {str(exc)[:70]}")
                continue
            if not text:
                continue
            # 生成期间哥哥可能说话了，那就让给他
            if self.turn is not None and not self.turn.done():
                continue
            print(f"\n恋恋（主动）：{text}")
            await self.emit({"type": "proactive", "text": text})
            self.brain.note_proactive_spoke(text)
            self.turn = asyncio.create_task(self._speak_proactive(text))
            try:
                await self.turn
            except asyncio.CancelledError:
                pass
            self._last_turn_end = time.time()

    async def _interrupt_watcher(self) -> None:
        """你一说话（或打字），她立刻闭嘴。"""
        while True:
            await self.interrupt.wait()
            self.interrupt.clear()
            if self.turn and not self.turn.done():
                self._barged = True
                self.player.stop()
                if self.avatar is not None:
                    self.avatar.set_emotion("neutral")   # 被打断，脸先收回来
                if self.native_pet is not None:
                    self.native_pet.set_emotion("neutral")
                self.turn.cancel()
                print("\n[打断] 你说吧～")

    async def _mic_watchdog(self) -> None:
        """万一采集线程悄悄死了，至少让用户看见原因。"""
        while True:
            await asyncio.sleep(2)
            if self.mic and not self.mic.is_alive():
                print(f"\n[麦克风] 采集线程已停止（已处理 {self.mic.stats['frames']} 帧、"
                      f"识别 {self.mic.stats['utterances']} 句）。要恢复请重启程序。")
                return

    # ---------------- 主循环 ----------------
    async def run(self) -> None:
        cfg = self.cfg
        text_only = cfg.text_mode
        await self.setup(with_mic=not text_only, with_typing=text_only or cfg.allow_typing)
        print("\n" + "=" * 62)
        if text_only:
            print("  键盘模式（没开麦克风）—— 打字回车发送，Ctrl+C 退出")
        else:
            print("  恋恋已上线 —— 直接说话，或者打字回车发消息")
            print("  打断已开启：她说着话你也能插嘴" + ("（戴耳机效果最好）" if cfg.barge_in else "（半双工模式）"))
        if self.typed:
            print("  打字提示：输入完按回车，她会停下来先听你说")
        print("=" * 62 + "\n")

        watcher = asyncio.create_task(self._interrupt_watcher()) if (self.mic or self.typed) else None
        dog = asyncio.create_task(self._mic_watchdog()) if self.mic else None
        pro = asyncio.create_task(self._proactive_loop()) if self.brain else None
        try:
            if cfg.greeting:
                await self.say(cfg.greeting)

            while True:
                if cfg.text_mode:
                    print("你（打字）：", end="", flush=True)
                item = await self.inputs.get()

                if item is EOF:
                    print("\n[键盘] 输入结束，退出。")
                    break
                if item is None:                       # 麦克风线程挂了的哨兵
                    if self.typed:                     # 还能打字就降级，别整个退出
                        print("\n" + "!" * 62)
                        print(f"[麦克风] 已停止工作：{self.mic.error}")
                        print("[麦克风] 已自动切换成键盘模式，继续打字没问题；要恢复语音请重启程序。")
                        print("!" * 62)
                        self.mic = None
                        cfg.text_mode = True
                        continue
                    raise RuntimeError(self.mic.error or "麦克风异常退出")

                self.interrupt.clear()
                if self.brain:
                    self.brain.note_user_spoke()       # 哥哥开口了，退避清零
                if isinstance(item, str):              # 打字
                    self.turn = asyncio.create_task(self.handle_text(item))
                else:                                  # 语音（本机麦克风或网页上传）
                    if not cfg.barge_in and self.mic:
                        self.mic.set_muted(True)       # 外放：她说话时先别听
                    self.turn = asyncio.create_task(self.handle_audio(item))

                try:
                    await self.turn
                except asyncio.CancelledError:
                    if not self._barged:
                        raise                          # 真被 Ctrl+C 了，往外抛
                    self._barged = False
                except Exception as exc:
                    print(f"\n[错误] {type(exc).__name__}: {exc}")
                    await self.emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                finally:
                    self._last_turn_end = time.time()
                    if not cfg.barge_in and self.mic:
                        self.player.stop()
                        self.mic.set_muted(self.mic_user_muted)
        finally:
            if watcher:
                watcher.cancel()
            if dog:
                dog.cancel()
            if pro:
                pro.cancel()
                if self.brain:
                    print(f"[proactive] {self.brain.describe()}")
            await self.close()

    # ---------------- 离线跑一个 wav ----------------
    async def run_wav(self, path: str, out: str = "") -> None:
        import soundfile as sf

        await self.setup(with_mic=False, with_typing=False)
        try:
            audio, sr = sf.read(path, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != self.cfg.sample_rate:
                n = int(round(len(audio) * self.cfg.sample_rate / sr))
                audio = np.interp(np.linspace(0, len(audio) - 1, n),
                                  np.arange(len(audio)), audio).astype(np.float32)
            await self.handle_audio(audio)
            if out and self.collected:
                sf.write(out, np.concatenate(self.collected), self.tts.sample_rate)
                print(f"[保存] 回复语音 -> {out}")
        finally:
            await self.close()      # 别把桌宠样式留在 VTS 上


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="恋恋 · 语音 + 键盘 混合对话")
    p.add_argument("--devices", action="store_true", help="列出音频设备并测试麦克风后退出")
    p.add_argument("--text", action="store_true", help="只用键盘，不开麦克风")
    p.add_argument("--no-typing", action="store_true", help="屏蔽键盘，只用语音")
    p.add_argument("--wav", metavar="FILE", help="跑一个 wav 文件，不用麦克风")
    p.add_argument("--out", metavar="FILE", help="配合 --wav：把回复语音存成 wav")
    return p.parse_args()


async def _amain(cfg) -> None:
    print(cfg.describe())
    bot = VoiceBot(cfg)
    if cfg.wav_input:
        await bot.run_wav(cfg.wav_input, cfg.wav_output)
    else:
        await bot.run()


def main() -> None:
    args = parse_args()
    if args.devices:
        list_devices()
        mic_selftest(CONFIG)
        return
    cfg = CONFIG
    cfg.text_mode = args.text
    cfg.allow_typing = cfg.allow_typing and not args.no_typing
    cfg.wav_input = args.wav or ""
    cfg.wav_output = args.out or ""
    try:
        asyncio.run(_amain(cfg))
    except KeyboardInterrupt:
        print("\n再见啦～")


if __name__ == "__main__":
    main()