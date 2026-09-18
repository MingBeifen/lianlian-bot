"""主程序 <-> 自渲染桌宠（native_pet.py）的桥。

主程序在 127.0.0.1:<端口> 起一个 TCP 服务，按 30fps 推状态：

    {"mouth": 0.42, "emotion": "happy"}

同时负责用独立 venv 的 python 启动渲染器进程，退出时一起收掉。
渲染器连不上主程序时会在 NATIVE_PET_IDLE_EXIT 秒后自己退出，不会留孤儿进程。
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import subprocess
import time
from pathlib import Path

EMOTIONS = ("neutral", "happy", "shy", "angry", "sad", "surprised")
ACTIONS = ("none", "nod", "shake", "wave", "think", "shy_hide", "laugh")


class NativePetLink:
    def __init__(self, cfg, level_provider, log=print):
        self.cfg = cfg
        self.level = level_provider or (lambda: 0.0)
        self.log = log
        self.port = int(cfg.native_pet_port)
        self.connected = False
        self._emotion = "neutral"
        self._action = "none"
        self._action_seq = 0
        self._clients: list = []
        self._server = None
        self._task = None
        self._proc: subprocess.Popen | None = None
        self._logfile = None
        self._last_seen = 0.0

    # ---------------- 对外 ----------------
    def set_emotion(self, emotion: str) -> None:
        emo = (emotion or "neutral").strip().lower()
        self._emotion = emo if emo in EMOTIONS else "neutral"

    def set_action(self, action: str) -> None:
        """LLM 判断出的伴随动作（wave/nod/...）；每次调用都算新的一次触发。"""
        act = (action or "none").strip().lower()
        if act not in ACTIONS:
            act = "none"
        self._action = act
        self._action_seq += 1

    def status_line(self) -> str:
        alive = "运行中" if (self._proc is not None and self._proc.poll() is None) else "未启动"
        return f"渲染器 {alive} | 主程序连接 {'是' if self.connected else '否'} | TCP {self.port}"

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="native-pet")

    async def set_model(self, model_path: str) -> None:
        """网页切换 Live2D 模型：只重启渲染器进程，TCP 服务不动。"""
        self.cfg.native_pet_model = str(model_path)
        await self.restart_renderer()

    async def restart_renderer(self) -> None:
        for writer in list(self._clients):
            with contextlib.suppress(Exception):
                writer.close()
        self._clients.clear()
        self.connected = False
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self._proc.wait, 3)
        self._proc = None
        if self._logfile is not None:
            with contextlib.suppress(Exception):
                self._logfile.close()
            self._logfile = None
        self._launch()

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        for writer in list(self._clients):
            with contextlib.suppress(Exception):
                writer.close()
        self._clients.clear()
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                await asyncio.to_thread(self._proc.wait, 3)
            except Exception:
                with contextlib.suppress(Exception):
                    self._proc.kill()
            self.log("[pet] 自渲染桌宠已关闭")
        self._proc = None
        if self._logfile is not None:
            with contextlib.suppress(Exception):
                self._logfile.close()
            self._logfile = None

    # ---------------- 内部 ----------------
    async def _run(self) -> None:
        try:
            self._server = await asyncio.start_server(self._on_client, "127.0.0.1", self.port)
        except OSError as exc:
            self.log(f"[pet] TCP 端口 {self.port} 用不了（{exc}），自渲染桌宠没起来")
            return
        self._launch()
        warned = False
        while True:
            if self._clients:
                data = (json.dumps({"mouth": float(self.level()),
                                    "emotion": self._emotion,
                                    "action": self._action,
                                    "action_seq": self._action_seq})
                        + "\n").encode("utf-8")
                for writer in list(self._clients):
                    try:
                        writer.write(data)
                    except Exception:
                        with contextlib.suppress(ValueError):
                            self._clients.remove(writer)
                if self._clients:
                    self.connected = True
                    self._last_seen = time.monotonic()
            elif self.connected and time.monotonic() - self._last_seen > 2:
                self.connected = False
            if (not warned and self._clients == [] and self._proc is not None
                    and self._proc.poll() is not None):
                warned = True
                self.log("[pet] 自渲染桌宠进程退出了，看 voicebot/native_pet.log 里的原因")
            await asyncio.sleep(1 / 30)

    async def _on_client(self, reader, writer) -> None:
        self.log(f"[pet] 自渲染桌宠已连接（{writer.get_extra_info('peername')}）")
        self._clients.append(writer)
        self.connected = True
        self._last_seen = time.monotonic()
        try:
            while True:
                if not await reader.read(1024):     # 渲染器不发数据，读到 EOF 说明它退了
                    break
        except Exception:
            pass
        finally:
            if writer in self._clients:
                self._clients.remove(writer)
            self.connected = False
            with contextlib.suppress(Exception):
                writer.close()

    def _launch(self) -> None:
        cfg = self.cfg
        root = Path(__file__).resolve().parents[1]        # voicebot/（相对路径按它解析）
        py = Path(cfg.native_pet_python or "")
        if not py.is_absolute():
            py = root / py
        model = Path(cfg.native_pet_model or "")
        if not model.is_absolute():
            model = root / model
        script = Path(__file__).with_name("native_pet.py")
        if not py.exists():
            self.log(f"[pet] 找不到渲染器 python：{py}")
            self.log("[pet] 先运行一次 setup_native_pet.bat 建好 live2d-venv")
            return
        if not cfg.native_pet_model or not model.exists():
            self.log(f"[pet] 找不到 Live2D 模型：{cfg.native_pet_model or '（未配置）'}")
            self.log("[pet] 在 models.json 的 live2d.native_model 里填 xxx.model3.json 路径")
            return
        args = [str(py), "-u", str(script),
                "--model", str(model),
                "--port", str(self.port),
                "--size", str(getattr(cfg, "pet_size", "") or "520x760"),
                "--position", str(getattr(cfg, "pet_position", "") or "br"),
                "--margin", str(int(getattr(cfg, "pet_margin", 20))),
                "--fps", str(int(getattr(cfg, "native_pet_fps", 60))),
                "--mouth-gain", str(float(getattr(cfg, "native_pet_mouth_gain", 6.0))),
                "--arm-pose", str(getattr(cfg, "native_pet_arm", "auto")),
                "--motion-gap-min", str(float(getattr(cfg, "native_pet_motion_gap_min", 10.0))),
                "--motion-gap-max", str(float(getattr(cfg, "native_pet_motion_gap_max", 25.0))),
                "--idle-exit", str(int(getattr(cfg, "native_pet_idle_exit", 15)))]
        if bool(getattr(cfg, "pet_click_through", False)):
            args.append("--click-through")
        if not bool(getattr(cfg, "pet_hotkeys", True)):
            args.append("--no-hotkeys")
        if not bool(getattr(cfg, "native_pet_idle_motion", True)):
            args.append("--no-idle-motion")
        if bool(getattr(cfg, "native_pet_debug", False)):
            args.append("--debug")
        if not bool(getattr(cfg, "native_pet_react_motion", True)):
            args.append("--no-react-motion")
        if not bool(getattr(cfg, "native_pet_action_motion", True)):
            args.append("--no-action-motion")
        if str(getattr(cfg, "native_pet_action_map", "") or "").strip():
            args += ["--action-map", str(cfg.native_pet_action_map).strip()]
        log_path = Path(__file__).resolve().parents[1] / "out" / "native_pet.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._logfile = open(log_path, "a", encoding="utf-8")
        self._logfile.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 启动 =====\n")
        self._logfile.flush()
        flags = 0x08000000 if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        self._proc = subprocess.Popen(
            args, stdout=self._logfile, stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parent),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.log(f"[pet] 已启动自渲染桌宠（pid {self._proc.pid}），日志 native_pet.log")