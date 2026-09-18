"""恋恋自渲染桌宠 —— 不依赖 VTube Studio 的 Live2D 实现。

用 live2d-py（Cubism Native Core）+ GLFW 在透明无边框窗口里直接渲染模型：
  · GLFW 透明帧缓冲：真正的逐像素透明，没有色键毛边；
  · 主程序通过 TCP 推口型（Player 电平）和情绪，本脚本只负责画；
  · 自动眨眼 / 呼吸由模型框架负责，另外加了视线跟随鼠标、待机摆动；
  · 按住左键可以拖动她；点一下（没拖动）= 惊讶小反应；
  · Ctrl+Alt+P 鼠标穿透、T 置顶、H 显示/隐藏、R 复位、Q 退出，
    Ctrl+Alt+Shift+方向键 微移、+/- 缩放。

这个脚本要跑在独立的 live2d-venv 里（见 setup_native_pet.bat），
主程序 voice_chat.py 会自动启动它，不需要手动开。
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import queue
import random
import socket
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

import glfw

if sys.platform != "win32":
    raise SystemExit("自渲染桌宠暂时只支持 Windows")

try:
    import live2d.v3 as live2d
except ImportError as exc:
    raise SystemExit("没找到 live2d-py，请先运行 setup_native_pet.bat") from exc

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# ---------------- Win32 ----------------
GWL_EXSTYLE = -20
WS_EX_TOPMOST = 0x00000008
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_APPWINDOW = 0x00040000
SWP_NOSIZE, SWP_NOMOVE, SWP_NOZORDER = 0x0001, 0x0002, 0x0004
SWP_NOACTIVATE, SWP_FRAMECHANGED = 0x0010, 0x0020
SPI_GETWORKAREA = 0x0030
MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x4000
WM_HOTKEY, WM_QUIT = 0x0312, 0x0012
GWL_WNDPROC = -4
WM_NCHITTEST = 0x0084
HTTRANSPARENT = -1

user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetWindowLongW.restype = ctypes.c_long
user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.SystemParametersInfoW.argtypes = [wintypes.UINT, wintypes.UINT,
                                         ctypes.POINTER(wintypes.RECT), wintypes.UINT]
user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_uint, ctypes.c_uint]
user32.RegisterHotKey.restype = wintypes.BOOL
kernel32.GetCurrentThreadId.restype = wintypes.DWORD

# 情绪 -> 标准参数偏移（模型没有的参数会在应用时自动跳过）
EMO_PARAMS = {
    "neutral": {},
    "happy": {"ParamMouthForm": 0.7, "ParamEyeLSmile": 1.0, "ParamEyeRSmile": 1.0,
              "ParamBrowLY": 0.3, "ParamBrowRY": 0.3, "ParamCheek": 0.6},
    "shy": {"ParamMouthForm": 0.5, "ParamCheek": 0.9,
            "ParamEyeLSmile": 0.4, "ParamEyeRSmile": 0.4},
    "angry": {"ParamMouthForm": -0.6, "ParamBrowLY": -0.5, "ParamBrowRY": -0.5,
              "ParamCheek": 0.3},
    "sad": {"ParamMouthForm": -0.5, "ParamBrowLY": -0.4, "ParamBrowRY": -0.4,
            "ParamEyeLOpen": 0.7, "ParamEyeROpen": 0.7},
    "surprised": {"ParamEyeLOpen": 1.3, "ParamEyeROpen": 1.3,
                  "ParamBrowLY": 0.5, "ParamBrowRY": 0.5},
}
ALL_EMO_IDS = sorted({pid for v in EMO_PARAMS.values() for pid in v})

ACTION_TAGS = ("nod", "shake", "wave", "think", "shy_hide", "laugh")


WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_longlong, wintypes.HWND, ctypes.c_uint,
                          ctypes.c_ulonglong, ctypes.c_longlong)


class ClickThroughFilter:
    """子类化窗口过程：穿透开启时让 WM_NCHITTEST 返回 HTTRANSPARENT。

    比 WS_EX_LAYERED|WS_EX_TRANSPARENT 可靠，而且不碰 DWM 的逐像素透明。
    """

    def __init__(self, hwnd: int):
        self.hwnd = hwnd
        self.enabled = False
        self._set = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
        self._get = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
        self._set.restype = ctypes.c_void_p
        self._set.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
        self._get.restype = ctypes.c_void_p
        self._get.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.CallWindowProcW.restype = ctypes.c_longlong
        user32.CallWindowProcW.argtypes = [ctypes.c_void_p, wintypes.HWND,
                                           ctypes.c_uint, ctypes.c_ulonglong,
                                           ctypes.c_longlong]
        self._orig = self._get(hwnd, GWL_WNDPROC)
        self._proc = WNDPROC(self._wndproc)
        self._set(hwnd, GWL_WNDPROC, ctypes.cast(self._proc, ctypes.c_void_p))

    def _wndproc(self, hwnd, msg, wparam, lparam):
        if msg == WM_NCHITTEST and self.enabled:
            return HTTRANSPARENT
        return user32.CallWindowProcW(self._orig, hwnd, msg, wparam, lparam)

    def set_enabled(self, on: bool) -> None:
        self.enabled = bool(on)

    def restore(self) -> None:
        try:
            self._set(self.hwnd, GWL_WNDPROC, self._orig)
        except Exception:
            pass


class PetHotkeys(threading.Thread):
    """全局热键；GLFW 不是线程安全的，所以只把动作塞进队列，主线程执行。"""

    _HK = MOD_CONTROL | MOD_ALT | MOD_NOREPEAT
    _HKS = MOD_CONTROL | MOD_ALT | MOD_SHIFT | MOD_NOREPEAT

    def __init__(self, actions: queue.Queue, log=print):
        super().__init__(name="pet-hotkeys", daemon=True)
        self.actions = actions
        self.log = log
        self.registered = 0
        self._tid = 0
        self._stop_flag = threading.Event()

    def run(self) -> None:
        self._tid = kernel32.GetCurrentThreadId()
        combos = [
            ("click", self._HK, 0x50), ("top", self._HK, 0x54),
            ("hide", self._HK, 0x48), ("reset", self._HK, 0x52),
            ("quit", self._HK, 0x51),
            ("left", self._HKS, 0x25), ("right", self._HKS, 0x27),
            ("up", self._HKS, 0x26), ("down", self._HKS, 0x28),
            ("grow", self._HKS, 0xBB), ("shrink", self._HKS, 0xBD),
        ]
        ids = {}
        for i, (name, mods, vk) in enumerate(combos, 1):
            if user32.RegisterHotKey(None, i, mods, vk):
                ids[i] = name
            else:
                self.log(f"[pet] 热键被占用：{name}")
        self.registered = len(ids)
        msg = wintypes.MSG()
        while not self._stop_flag.is_set():
            if user32.GetMessageW(ctypes.byref(msg), None, 0, 0) <= 0:
                break
            if msg.message == WM_HOTKEY:
                name = ids.get(int(msg.wParam))
                if name:
                    self.actions.put(name)
        for i in list(ids):
            user32.UnregisterHotKey(None, i)

    def stop(self) -> None:
        self._stop_flag.set()
        if self._tid:
            user32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
class StateClient(threading.Thread):
    """从主程序收 JSON 行：{"mouth": 0..1, "emotion": "happy"}。"""

    def __init__(self, port: int, idle_exit: float = 0.0, log=print):
        super().__init__(name="pet-state", daemon=True)
        self.port = port
        self.idle_exit = idle_exit
        self.log = log
        self.mouth = 0.0
        self.emotion = "neutral"
        self.action = "none"
        self.action_seq = 0
        self.connected = False
        self.expired = False
        self.last_rx = time.monotonic()
        self._got_first = False
        self._stop_flag = threading.Event()

    def run(self) -> None:
        while not self._stop_flag.is_set():
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=3) as sock:
                    sock.settimeout(1.0)
                    self.connected = True
                    self.last_rx = time.monotonic()
                    self.log("[pet] 已连上主程序")
                    buf = b""
                    while not self._stop_flag.is_set():
                        try:
                            data = sock.recv(8192)
                        except socket.timeout:
                            continue
                        if not data:
                            break
                        buf += data
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            if not line.strip():
                                continue
                            try:
                                msg = json.loads(line)
                            except Exception:
                                continue
                            self.mouth = max(0.0, min(1.0, float(msg.get("mouth") or 0.0)))
                            emo = str(msg.get("emotion") or "neutral").lower()
                            if emo in EMO_PARAMS:
                                self.emotion = emo
                            act = str(msg.get("action") or "none").lower()
                            if act in ACTION_TAGS or act == "none":
                                self.action = act
                            self.action_seq = int(msg.get("action_seq") or 0)
                            self.last_rx = time.monotonic()
                            if not self._got_first:
                                self._got_first = True
                                self.log("[pet] 收到主程序状态，开始跟口型")
            except OSError:
                pass
            self.connected = False
            if self.idle_exit > 0 and time.monotonic() - self.last_rx > self.idle_exit:
                self.log(f"[pet] 主程序 {self.idle_exit:.0f}s 没消息，桌宠退出")
                self.expired = True
                return
            time.sleep(1.0)

    def stop(self) -> None:
        self._stop_flag.set()


class NativePet:
    def __init__(self, args):
        self.args = args
        self.actions: queue.Queue = queue.Queue()
        self.win = None
        self.model = None
        self.ids: set = set()
        self.hwnd = 0
        self.click_filter = None
        self._frames = 0
        self._arm_opacity: dict = {}
        self.motion_groups: list = []
        self.motion_stats: dict = {}
        self._emo_motion: dict = {}
        self._pending_reaction = ""
        self._pending_action = ""
        self._action_seq = 0
        self._action_motion: dict = {}
        self._last_emo = "neutral"
        self._speech_pose: dict = {}
        self._speech_energy = 0.0
        self._motion_active = False
        self._next_motion_at = 0.0
        self._arm_static = False
        self.client = StateClient(args.port, args.idle_exit) if args.port else None
        self.hotkeys = None
        self.click_through = bool(args.click_through)
        self.topmost = True
        self.mouth_cur = 0.0
        self.face: dict = {}
        self.dragging = False
        self.drag_moved = False
        self.drag_start = (0, 0)
        self.drag_win = (0, 0)
        self.tap_until = 0.0
        self.t0 = time.monotonic()

    # ---------------- 窗口 / 模型 ----------------
    def setup(self) -> None:
        if not glfw.init():
            raise SystemExit("glfw 初始化失败")
        glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, glfw.TRUE)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        glfw.window_hint(glfw.DECORATED, glfw.FALSE)
        glfw.window_hint(glfw.FLOATING, glfw.TRUE)
        w, h = self.args.size
        self.win = glfw.create_window(w, h, self.args.title, None, None)
        if not self.win:
            raise SystemExit("创建透明窗口失败")
        self._place(self.args.position, self.args.margin)
        glfw.make_context_current(self.win)
        glfw.swap_interval(0)
        glfw.show_window(self.win)

        live2d.init()
        live2d.glInit()
        self.model = live2d.LAppModel()
        self.model.LoadModelJson(self.args.model)
        self.model.Resize(w, h)
        self.model.SetAutoBlinkEnable(True)
        self.model.SetAutoBreathEnable(True)
        self.ids = set(self.model.GetParamIds())
        self._setup_arm_pose()
        self._load_motions()
        missing = [p for p in ALL_EMO_IDS if p not in self.ids]
        if missing:
            print(f"[pet] 模型缺这些表情参数，已跳过：{missing}")
        self.hwnd = glfw.get_win32_window(self.win)
        if not getattr(self.args, "no_filter", False):
            self.click_filter = ClickThroughFilter(self.hwnd)
        elif self.click_through:      # 诊断用：退回老式样式穿透
            ex = user32.GetWindowLongW(self.hwnd, GWL_EXSTYLE) | WS_EX_TRANSPARENT
            user32.SetWindowLongW(self.hwnd, GWL_EXSTYLE, ex)
        self._set_toolwindow(bool(self.args.hide_taskbar))
        self._apply_click_through()
        if self.args.hotkeys:
            self.hotkeys = PetHotkeys(self.actions)
            self.hotkeys.start()
        if self.client is not None:
            self.client.start()
        print(f"[pet] 自渲染桌宠就绪：{w}x{h} | 透明={not self.args.no_transparent} "
              f"| 全局热键 {getattr(self.hotkeys, 'registered', 0)} 个"
              + (" | 演示模式" if self.args.demo else ""))

    def _setup_arm_pose(self) -> None:
        """没有 pose3.json 的模型会同时显示 PartArmA/PartArmB（四只手）。

        官方 Hiyori 的默认动作驱动 A 组手臂，所以这里锁定 A、隐藏 B；
        想要另一组把 NATIVE_PET_ARM 改成 B，想自己用 pose 文件就写 both。
        """
        parts = self.model.GetPartIds()
        arm_parts = {name: parts.index(name) for name in ("PartArmA", "PartArmB")
                     if name in parts}
        if len(arm_parts) < 2:
            return
        has_pose = False
        try:
            cfg = json.loads(Path(self.args.model).read_text(encoding="utf-8"))
            has_pose = bool((cfg.get("FileReferences") or {}).get("Pose"))
        except Exception:
            pass
        mode = (self.args.arm_pose or "auto").lower()
        if has_pose and mode == "auto":
            print("[pet] 模型带 pose3.json，手臂姿态交给 Cubism")
            return
        self._arm_static = mode in ("a", "b", "both")
        show = {"a": "PartArmA", "b": "PartArmB", "both": ""}.get(mode, "PartArmA")
        self._arm_opacity = {idx: (1.0 if name == show else 0.0)
                             for name, idx in arm_parts.items()}
        print(f"[pet] 模型无 pose3.json，锁定手臂：{show or '原始(两组都显示)'}")

    def _load_motions(self) -> None:
        """把模型目录里的动作文件都装进来，让她没在说话时自己播。"""
        if not getattr(self.args, "idle_motion", True) or self.args.demo > 0:
            return
        folder = Path(self.args.model).parent
        files = sorted(folder.glob("animations/*.motion3.json")) or \
            sorted(folder.glob("motions/*.motion3.json"))
        for i, f in enumerate(files):
            group = f"Idle_{i}"
            try:
                self.model.LoadExtraMotion(group, str(f))
                self.motion_groups.append(group)
                self.motion_stats[group] = self._read_motion_stats(f)
            except Exception as exc:
                print(f"[pet] 动作加载失败 {f.name}: {exc}")
        if self.motion_groups:
            print(f"[pet] 已加载 {len(self.motion_groups)} 个待机动作")
        self._build_reaction_map()
        self._build_action_map()
        self._next_motion_at = time.monotonic() + 3.0

    @staticmethod
    def _segments_range(segments: list) -> float:
        """从 motion3 的 Segments 里取参数波动范围（近似）。"""
        if not segments:
            return 0.0
        vals = [segments[1]]
        i = 2
        while i < len(segments):
            typ = segments[i]
            i += 1
            if typ == 1:            # bezier
                if i + 5 >= len(segments):
                    break
                vals.extend([segments[i + 1], segments[i + 3], segments[i + 5]])
                i += 6
            elif typ in (0, 2, 3):  # linear / stepped / inverse
                if i + 1 >= len(segments):
                    break
                vals.append(segments[i + 1])
                i += 2
            else:
                break
        return max(vals) - min(vals)

    def _read_motion_stats(self, path: Path) -> dict:
        """分析动作文件：时长 + 各参数波动，用来给情绪挑动作。"""
        st = {"duration": 0.0, "angle_x": 0.0, "angle_y": 0.0, "angle_z": 0.0,
              "body": 0.0, "arm_a": 0.0, "arm_b": 0.0, "mouth": 0.0}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return st
        st["duration"] = float((data.get("Meta") or {}).get("Duration") or 0.0)
        for curve in data.get("Curves") or []:
            pid = str(curve.get("Id") or "")
            rng = self._segments_range(curve.get("Segments") or [])
            if pid == "ParamAngleX":
                st["angle_x"] = rng
            elif pid == "ParamAngleY":
                st["angle_y"] = rng
            elif pid == "ParamAngleZ":
                st["angle_z"] = rng
            elif pid in ("ParamBodyAngleX", "ParamBodyAngleZ"):
                st["body"] = max(st["body"], rng)
            elif pid in ("ParamArmLA", "ParamArmRA"):
                st["arm_a"] = max(st["arm_a"], rng)
            elif pid in ("ParamArmLB", "ParamArmRB"):
                st["arm_b"] = max(st["arm_b"], rng)
            elif pid == "ParamMouthOpenY":
                st["mouth"] = rng
        st["lively"] = (st["angle_x"] + st["angle_y"] + st["angle_z"]
                        + st["body"] + st["arm_a"] + st["arm_b"])
        return st

    def _build_action_map(self) -> None:
        """动作标签 -> 动作文件：按特征自动挑，可用 NATIVE_PET_ACTION_MAP 覆盖。"""
        if not self.motion_stats or not getattr(self.args, "action_motion", True):
            return
        st = self.motion_stats
        auto = {
            "nod": max(st, key=lambda g: st[g]["angle_y"] * 2 - st[g]["duration"] * 0.2),
            "shake": max(st, key=lambda g: st[g]["angle_x"] + st[g]["angle_z"] * 1.5),
            "wave": max(st, key=lambda g: st[g]["arm_b"] * 2 + st[g]["arm_a"]),
            "think": max(st, key=lambda g: st[g]["duration"] + st[g]["angle_x"] * 0.5
                         - st[g]["lively"] * 0.3),
            "shy_hide": max(st, key=lambda g: st[g]["arm_a"] * 1.5 + st[g]["mouth"]),
            "laugh": max(st, key=lambda g: st[g]["lively"] + st[g]["mouth"]),
        }
        text = str(getattr(self.args, "action_map", "") or "")
        for item in text.replace("；", ",").replace(";", ",").split(","):
            item = item.strip()
            if "=" not in item:
                continue
            key, val = item.split("=", 1)
            key, val = key.strip().lower(), val.strip()
            if key in auto and val in self.motion_stats:
                auto[key] = val
            elif key in auto:
                print(f"[pet] 动作映射 {key}={val} 无效（没有这个动作组）")
        self._action_motion = auto
        print("[pet] 动作标签: " + ", ".join(f"{k}->{v}" for k, v in auto.items()))

    def _build_reaction_map(self) -> None:
        """情绪 -> 动作：用动作文件的参数特征猜哪一条适合（可关）。"""
        if not self.motion_stats or not getattr(self.args, "react_motion", True):
            return
        st = self.motion_stats
        pick = {
            "happy": max(st, key=lambda g: st[g]["arm_a"] + st[g]["arm_b"] + st[g]["lively"]),
            "angry": max(st, key=lambda g: st[g]["angle_z"] + st[g]["lively"] * 0.5),
            "sad": max(st, key=lambda g: st[g]["duration"] - st[g]["lively"]),
            "surprised": min(st, key=lambda g: st[g]["duration"]),
            "shy": max(st, key=lambda g: st[g]["arm_b"] + st[g]["lively"] * 0.3),
        }
        self._emo_motion = pick
        print("[pet] 情绪动作: " + ", ".join(f"{k}->{v}" for k, v in pick.items()))

    def _update_motion_state(self) -> None:
        """说话时暂停待机动作（把嘴让给口型），安静一会儿再随机播一个。"""
        now = time.monotonic()
        emo = self._active_emotion()
        if emo != self._last_emo:
            self._last_emo = emo
            group = self._emo_motion.get(emo)
            if group:
                self._pending_reaction = group
                if getattr(self.args, "debug", False):
                    print(f"[pet] 情绪 {emo} -> 反应动作 {group}")
        if self.client is not None and self.client.action_seq != self._action_seq:
            self._action_seq = self.client.action_seq
            act = self.client.action
            group = self._action_motion.get(act, "")
            self._pending_action = group if act != "none" else ""
            if getattr(self.args, "debug", False):
                print(f"[pet] 动作标签 {act} -> {group or '（不触发）'}")
        speaking = self._mouth_target() > 0.03 or self.mouth_cur > 0.06
        if speaking:
            if self._motion_active:
                self.model.StopAllMotions()
                self._motion_active = False
                if getattr(self.args, "debug", False):
                    print("[pet] 说话，暂停待机动作")
            self._next_motion_at = now + max(3.0, self.args.motion_gap_min * 0.4)
            return
        if self._motion_active and self.model.IsMotionFinished():
            self._motion_active = False
        if self._pending_action and self.motion_groups:
            group, self._pending_action = self._pending_action, ""
            self.model.StartMotion(group, 0, 2)
            self._motion_active = True
            self._next_motion_at = now + random.uniform(self.args.motion_gap_min,
                                                        self.args.motion_gap_max)
            if getattr(self.args, "debug", False):
                print(f"[pet] 播放动作 {group}")
            return
        if self._pending_reaction and self.motion_groups:
            group, self._pending_reaction = self._pending_reaction, ""
            self.model.StartMotion(group, 0, 2)          # NORMAL 优先级，盖过待机
            self._motion_active = True
            self._next_motion_at = now + random.uniform(self.args.motion_gap_min,
                                                        self.args.motion_gap_max)
            if getattr(self.args, "debug", False):
                print(f"[pet] 播放反应动作 {group}")
            return
        if self.motion_groups and not self._motion_active and now >= self._next_motion_at:
            group = random.choice(self.motion_groups)
            self.model.StartMotion(group, 0, 1)
            self._motion_active = True
            self._next_motion_at = now + random.uniform(self.args.motion_gap_min,
                                                        self.args.motion_gap_max)
            if getattr(self.args, "debug", False):
                print(f"[pet] 播放待机动作 {group}")

    def _place(self, position: str, margin: int) -> None:
        wa = wintypes.RECT()
        user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(wa), 0)
        w, h = self.args.size
        pos = (position or "br").lower()
        xs = {"tl": wa.left + margin, "tr": wa.right - w - margin,
              "bl": wa.left + margin, "br": wa.right - w - margin,
              "center": (wa.left + wa.right - w) // 2}
        ys = {"tl": wa.top + margin, "tr": wa.top + margin,
              "bl": wa.bottom - h - margin, "br": wa.bottom - h - margin,
              "center": (wa.top + wa.bottom - h) // 2}
        glfw.set_window_pos(self.win, int(xs.get(pos, wa.right - w - margin)),
                            int(ys.get(pos, wa.bottom - h - margin)))

    def _set_toolwindow(self, on: bool) -> None:
        ex = user32.GetWindowLongW(self.hwnd, GWL_EXSTYLE)
        if on:
            ex = (ex | WS_EX_TOOLWINDOW) & ~WS_EX_APPWINDOW
        else:
            ex = (ex & ~WS_EX_TOOLWINDOW) | WS_EX_APPWINDOW
        user32.SetWindowLongW(self.hwnd, GWL_EXSTYLE, ex)
        user32.SetWindowPos(self.hwnd, 0, 0, 0, 0, 0,
                            SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER
                            | SWP_NOACTIVATE | SWP_FRAMECHANGED)

    def _apply_click_through(self) -> None:
        if self.click_filter is not None:
            self.click_filter.set_enabled(self.click_through)

    # ---------------- 每帧逻辑 ----------------
    def _cursor(self) -> tuple:
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y

    def _active_emotion(self) -> str:
        if time.monotonic() < self.tap_until:
            return "surprised"
        if self.args.demo:
            return ("happy", "angry", "shy", "sad")[int(time.monotonic() - self.t0) // 2 % 4]
        return self.client.emotion if self.client else "neutral"

    def _mouth_target(self) -> float:
        if self.args.demo:
            return 0.5 + 0.45 * math.sin(2 * math.pi * (time.monotonic() - self.t0) / 1.6)
        return self.client.mouth if self.client else 0.0

    def _update_params(self, dt: float) -> None:
        # 没播动作时锁住手臂（四只手问题）；动作文件自带 PartArm 曲线，播放时交给它
        if self._arm_opacity and (self._arm_static or not self._motion_active):
            for idx, opacity in self._arm_opacity.items():
                self.model.SetPartOpacity(idx, opacity)
        t = time.monotonic() - self.t0
        # 视线和脑袋跟着鼠标（就算穿透模式也能追踪）
        cx, cy = self._cursor()
        wx, wy = glfw.get_window_pos(self.win)
        w, h = glfw.get_window_size(self.win)
        nx = max(-1.0, min(1.0, (cx - (wx + w / 2)) / max(1, w / 2)))
        ny = max(-1.0, min(1.0, (cy - (wy + h / 2)) / max(1, h / 2)))
        # 说话能量：口型电平的平滑值，用来带动身体小动作
        target_energy = max(0.0, min(1.0, self._mouth_target() * 4.0))
        self._speech_energy += (target_energy - self._speech_energy) * 0.15
        energy = self._speech_energy
        self._set("ParamEyeBallX", nx * 0.8)
        self._set("ParamEyeBallY", -ny * 0.8)
        self._set("ParamAngleX", nx * 12 + 3 * math.sin(t * 0.5))
        self._set("ParamAngleY", -ny * 8 + 2 * math.sin(t * 0.7)
                  + 2.5 * math.sin(t * 2.1) * energy)
        self._set("ParamAngleZ", 2 * math.sin(t * 0.4)
                  + 1.2 * math.sin(t * 2.7) * energy)
        self._set("ParamBodyAngleX", 1.5 * math.sin(t * 0.5))
        # 手臂/身体的说话动作（平滑衰减，说完自动归位）
        for pid, goal in {
            "ParamBodyAngleZ": 2.5 * math.sin(t * 3.2) * energy,
            "ParamArmLA": 1.5 * math.sin(t * 2.4) * energy,
            "ParamArmRA": -1.5 * math.sin(t * 2.4 + 0.4) * energy,
        }.items():
            cur = self._speech_pose.get(pid, 0.0)
            cur += (goal - cur) * 0.15
            self._speech_pose[pid] = cur
            self._set(pid, cur)

        # 口型：张得快、收得慢
        target = max(0.0, min(1.0, self._mouth_target()))
        k = 0.55 if target > self.mouth_cur else 0.18
        self.mouth_cur += (target - self.mouth_cur) * min(1.0, k * (dt * 60 if dt else 1))
        # 真实语音的 RMS 只有 0.05~0.2，必须乘增益再压曲线，嘴才张得明显
        raw = min(1.0, max(0.0, self.mouth_cur * self.args.mouth_gain))
        self._set("ParamMouthOpenY", raw ** 0.7)

        # 表情：平滑过渡到目标偏移
        goal = EMO_PARAMS.get(self._active_emotion(), {})
        for pid in ALL_EMO_IDS:
            cur = self.face.get(pid, 0.0)
            cur += (goal.get(pid, 0.0) - cur) * 0.10
            self.face[pid] = cur
            self._set(pid, cur)

    def _set(self, pid: str, value: float) -> None:
        if pid in self.ids:
            self.model.SetParameterValue(pid, float(value))

    def _handle_drag(self) -> None:
        if self.click_through:
            return
        btn = glfw.get_mouse_button(self.win, glfw.MOUSE_BUTTON_LEFT)
        cx, cy = self._cursor()
        if btn == glfw.PRESS and not self.dragging:
            self.dragging = True
            self.drag_moved = False
            self.drag_start = (cx, cy)
            self.drag_win = glfw.get_window_pos(self.win)
        elif btn == glfw.PRESS and self.dragging:
            dx, dy = cx - self.drag_start[0], cy - self.drag_start[1]
            if abs(dx) + abs(dy) > 4:
                self.drag_moved = True
            if self.drag_moved:
                glfw.set_window_pos(self.win, self.drag_win[0] + dx, self.drag_win[1] + dy)
        elif btn == glfw.RELEASE and self.dragging:
            self.dragging = False
            if not self.drag_moved:
                self.tap_until = time.monotonic() + 0.6      # 点一下：惊讶小反应
                if self.motion_groups:
                    self.model.StartMotion(random.choice(self.motion_groups), 0, 2)
                    self._motion_active = True

    def _handle_hotkeys(self) -> None:
        while True:
            try:
                name = self.actions.get_nowait()
            except queue.Empty:
                return
            if name == "click":
                self.click_through = not self.click_through
                self._apply_click_through()
                print("[pet] 鼠标穿透：" + ("开" if self.click_through else "关"))
            elif name == "top":
                self.topmost = not getattr(self, "topmost", True)
                glfw.set_window_attrib(self.win, glfw.FLOATING, self.topmost)
                print("[pet] 置顶：" + ("开" if self.topmost else "关"))
            elif name == "hide":
                if glfw.get_window_attrib(self.win, glfw.VISIBLE):
                    glfw.hide_window(self.win)
                    print("[pet] 已隐藏（Ctrl+Alt+H 叫回来）")
                else:
                    glfw.show_window(self.win)
            elif name == "reset":
                self.click_through = False
                self._apply_click_through()
                glfw.set_window_attrib(self.win, glfw.FLOATING, True)
                self._place(self.args.position, self.args.margin)
            elif name == "quit":
                glfw.set_window_should_close(self.win, True)
            elif name in ("left", "right", "up", "down"):
                dx, dy = {"left": (-20, 0), "right": (20, 0),
                          "up": (0, -20), "down": (0, 20)}[name]
                x, y = glfw.get_window_pos(self.win)
                glfw.set_window_pos(self.win, x + dx, y + dy)
            elif name in ("grow", "shrink"):
                w, h = glfw.get_window_size(self.win)
                f = 1.05 if name == "grow" else 1 / 1.05
                x, y = glfw.get_window_pos(self.win)
                glfw.set_window_size(self.win, max(120, int(w * f)), max(160, int(h * f)))
                glfw.set_window_pos(self.win, x, y)

    # ---------------- 主循环 ----------------
    def run(self) -> None:
        self.setup()
        try:
            last = time.monotonic()
            while not glfw.window_should_close(self.win):
                now = time.monotonic()
                dt = max(1e-3, min(0.1, now - last))
                last = now
                glfw.poll_events()
                self._handle_hotkeys()
                self._handle_drag()
                self._update_motion_state()
                if self.client is not None and self.client.expired:
                    break
                if self.args.demo > 0 and now - self.t0 > self.args.demo:
                    break
                live2d.clearBuffer(0.0, 0.0, 0.0, 0.0)
                self._update_params(dt)
                self.model.Update()
                self.model.Draw()
                glfw.swap_buffers(self.win)
                self._frames += 1
                if getattr(self.args, "debug", False) and self._frames % 60 == 0:
                    print(f"[pet] {self._frames} 帧 | mouth={self._mouth_target():.2f}"
                          f" | emo={self._active_emotion()} | 窗口={glfw.get_window_size(self.win)}")
                frame = time.monotonic() - now
                time.sleep(max(0.0, 1.0 / max(15, self.args.fps) - frame))
        finally:
            self.close()

    def close(self) -> None:
        if self.click_filter is not None:
            self.click_filter.restore()
            self.click_filter = None
        if self.hotkeys is not None:
            self.hotkeys.stop()
        if self.client is not None:
            self.client.stop()
        try:
            live2d.glRelease()
            live2d.dispose()
        except Exception:
            pass
        try:
            glfw.destroy_window(self.win)
            glfw.terminate()
        except Exception:
            pass
DEFAULT_MODEL = (r"D:\software\steam\steamapps\common\VTube Studio\VTube Studio_Data"
                 r"\StreamingAssets\Live2DModels\hiyori_vts\hiyori.model3.json")


def parse_size(text: str) -> tuple:
    try:
        w, h = (text or "").lower().replace("×", "x").split("x")
        return max(160, int(w)), max(200, int(h))
    except Exception:
        return 520, 760


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="恋恋自渲染桌宠（live2d-py + GLFW）")
    p.add_argument("--model", default=DEFAULT_MODEL, help="model3.json 路径")
    p.add_argument("--port", type=int, default=9890, help="接收主程序状态的 TCP 端口；0=不连")
    p.add_argument("--size", default="520x760", help="窗口大小，例 520x760")
    p.add_argument("--position", default="br", help="tl/tr/bl/br/center")
    p.add_argument("--margin", type=int, default=20, help="离屏幕边缘像素")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--mouth-gain", type=float, default=6.0)
    p.add_argument("--arm-pose", default="auto", help="auto/a/b/both：没有 pose 文件时显示哪组手臂")
    p.add_argument("--no-idle-motion", action="store_true", help="关掉待机动作")
    p.add_argument("--motion-gap-min", type=float, default=10.0, help="两个待机动作最少间隔秒数")
    p.add_argument("--motion-gap-max", type=float, default=25.0, help="两个待机动作最多间隔秒数")
    p.add_argument("--no-react-motion", action="store_true", help="关掉情绪反应动作")
    p.add_argument("--no-action-motion", action="store_true", help="关掉 LLM 动作标签")
    p.add_argument("--action-map", default="", help="覆盖动作映射，例 nod=Idle_2,wave=Idle_5")
    p.add_argument("--title", default="恋恋桌宠")
    p.add_argument("--demo", type=float, default=0.0, help="演示 N 秒后自动退出（口型+表情循环）")
    p.add_argument("--idle-exit", type=float, default=0.0,
                   help="主程序断联 N 秒后自动退出；0=一直留着")
    p.add_argument("--click-through", action="store_true", help="启动即鼠标穿透")
    p.add_argument("--show-in-taskbar", action="store_true", help="默认不占任务栏")
    p.add_argument("--no-transparent", action="store_true", help="调试：不透明背景")
    p.add_argument("--no-hotkeys", action="store_true")
    p.add_argument("--no-filter", action="store_true", help="诊断：不用窗口过程子类化")
    p.add_argument("--debug", action="store_true", help="打印帧数/口型")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.size = parse_size(args.size)
    args.hide_taskbar = not args.show_in_taskbar
    args.hotkeys = not args.no_hotkeys
    args.idle_motion = not args.no_idle_motion
    args.react_motion = not args.no_react_motion
    args.action_motion = not args.no_action_motion
    if not Path(args.model).exists():
        raise SystemExit(f"[pet] 找不到模型：{args.model}\n"
                         "       用 --model 指向 xxx.model3.json，或改 .env 的 NATIVE_PET_MODEL")
    pet = NativePet(args)
    pet.run()


if __name__ == "__main__":
    main()