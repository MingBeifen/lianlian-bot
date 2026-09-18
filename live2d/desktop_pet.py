"""恋恋桌宠模式：把 VTube Studio 窗口改造成桌面挂件（Windows 专用）。

VTS 原生支持透明背景，但不支持窗口置顶 / 鼠标穿透 / 隐藏任务栏 / 无边框，
这里用 Win32 API 补齐，所有改动都能一键恢复：

    Ctrl+Alt+P              鼠标穿透（点不到她，桌面上其它东西正常点）/ 再按恢复
    Ctrl+Alt+T              置顶开关
    Ctrl+Alt+B              无边框开关
    Ctrl+Alt+H              显示 / 隐藏
    Ctrl+Alt+R              恢复成普通窗口
    Ctrl+Alt+Shift+方向键    微调位置（每次 20 像素）
    Ctrl+Alt+Shift+加/减     缩放窗口

命令行：
    python desktop_pet.py --status     # 看当前样式
    python desktop_pet.py              # 按 .env 的 PET_* 应用（Ctrl+C 退出并恢复）
    python desktop_pet.py --keep       # 应用后保持样式（不随退出恢复）
    python desktop_pet.py --restore    # 恢复原样（上次异常退出也能救回来）

透明背景在 VTube Studio 里开：设置 -> General -> Background 附近找
Transparent Background（或去 Hotkeys 里给 Toggle Transparency 建个热键按一下）。
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]          # voicebot/
for _p in (str(_ROOT), str(_ROOT / "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if sys.platform != "win32":
    raise SystemExit("桌宠模式仅支持 Windows")

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# ---------------- Win32 常量 / 签名 ----------------
GWL_STYLE, GWL_EXSTYLE = -16, -20
WS_CAPTION = 0x00C00000
WS_THICKFRAME = 0x00040000
WS_MINIMIZEBOX = 0x00020000
WS_MAXIMIZEBOX = 0x00010000
WS_SYSMENU = 0x00080000
WS_EX_TOPMOST = 0x00000008
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_APPWINDOW = 0x00040000
WS_EX_LAYERED = 0x00080000
SWP_NOSIZE, SWP_NOMOVE, SWP_NOZORDER = 0x0001, 0x0002, 0x0004
SWP_NOACTIVATE, SWP_FRAMECHANGED = 0x0010, 0x0020
HWND_TOPMOST, HWND_NOTOPMOST = -1, -2
SW_HIDE, SW_SHOWNOACTIVATE = 0, 4
SPI_GETWORKAREA = 0x0030
LWA_ALPHA = 0x00000002
MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x4000
WM_HOTKEY, WM_QUIT = 0x0312, 0x0012

user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
user32.FindWindowW.restype = wintypes.HWND
user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetWindowLongW.restype = ctypes.c_long
user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
user32.SetWindowLongW.restype = ctypes.c_long
user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, ctypes.c_uint]
user32.SetWindowPos.restype = wintypes.BOOL
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.SetLayeredWindowAttributes.argtypes = [wintypes.HWND, wintypes.COLORREF,
                                              ctypes.c_ubyte, wintypes.DWORD]
user32.SystemParametersInfoW.argtypes = [wintypes.UINT, wintypes.UINT,
                                         ctypes.POINTER(wintypes.RECT), wintypes.UINT]
user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_uint, ctypes.c_uint]
user32.RegisterHotKey.restype = wintypes.BOOL
kernel32.GetCurrentThreadId.restype = wintypes.DWORD


class WINDOWPLACEMENT(ctypes.Structure):
    _fields_ = [("length", wintypes.UINT), ("flags", wintypes.UINT),
                ("showCmd", wintypes.UINT),
                ("ptMinPosition", wintypes.POINT), ("ptMaxPosition", wintypes.POINT),
                ("rcNormalPosition", wintypes.RECT)]


user32.GetWindowPlacement.argtypes = [wintypes.HWND, ctypes.POINTER(WINDOWPLACEMENT)]
user32.SetWindowPlacement.argtypes = [wintypes.HWND, ctypes.POINTER(WINDOWPLACEMENT)]
SW_SHOWMINIMIZED, SW_RESTORE = 2, 9

_BACKUP = Path(__file__).resolve().parents[1] / "data" / "_pet_backup.json"


def _s32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


class PetWindow:
    """VTube Studio 主窗口的样式控制器（只改窗口样式，不动 VTS 本体）。"""

    def __init__(self, title: str = "VTube Studio"):
        self.title = title
        self.hwnd = user32.FindWindowW(None, title)
        if not self.hwnd:
            raise RuntimeError(f"找不到窗口「{title}」：请先启动 VTube Studio"
                               "（或用 PET_TITLE= 指定窗口标题）")
        self.orig_style: int | None = None
        self.orig_ex: int | None = None
        self.orig_rect: tuple | None = None
        self.orig_show: int = 4          # 4 = SW_SHOWNOACTIVATE

    # ---------------- 读写样式 ----------------
    @property
    def style(self) -> int:
        return user32.GetWindowLongW(self.hwnd, GWL_STYLE) & 0xFFFFFFFF

    @property
    def ex(self) -> int:
        return user32.GetWindowLongW(self.hwnd, GWL_EXSTYLE) & 0xFFFFFFFF

    @property
    def rect(self) -> tuple:
        r = wintypes.RECT()
        user32.GetWindowRect(self.hwnd, ctypes.byref(r))
        return (r.left, r.top, r.right, r.bottom)

    def _write(self, index: int, value: int) -> None:
        user32.SetWindowLongW(self.hwnd, index, _s32(value))

    def _refresh_frame(self) -> None:
        user32.SetWindowPos(self.hwnd, 0, 0, 0, 0, 0,
                            SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER
                            | SWP_NOACTIVATE | SWP_FRAMECHANGED)

    def placement(self) -> WINDOWPLACEMENT:
        wp = WINDOWPLACEMENT()
        wp.length = ctypes.sizeof(WINDOWPLACEMENT)
        user32.GetWindowPlacement(self.hwnd, ctypes.byref(wp))
        return wp

    def capture(self) -> None:
        """记住原始样式和位置（含最小化状态），并落盘一份，异常退出也能救。"""
        if _BACKUP.exists():          # 上次异常退出残留：先按备份复原，再重新采集
            try:
                self.restore()
            except Exception:
                pass
        wp = self.placement()
        r = wp.rcNormalPosition
        self.orig_style, self.orig_ex = self.style, self.ex
        self.orig_show = int(wp.showCmd or 4)
        self.orig_rect = (r.left, r.top, r.right, r.bottom)
        try:
            _BACKUP.parent.mkdir(parents=True, exist_ok=True)
            _BACKUP.write_text(json.dumps({
                "title": self.title, "style": self.orig_style, "ex": self.orig_ex,
                "rect": list(self.orig_rect), "show_cmd": self.orig_show,
            }, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def ensure_visible(self) -> None:
        """桌宠模式启动时：如果 VTS 是最小化的，先还原出来。"""
        if user32.IsIconic(self.hwnd):
            user32.ShowWindow(self.hwnd, SW_RESTORE)

    # ---------------- 各种桌宠样式 ----------------
    def set_topmost(self, on: bool = True) -> None:
        user32.SetWindowPos(self.hwnd, HWND_TOPMOST if on else HWND_NOTOPMOST,
                            0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)
        self._refresh_frame()

    def hide_taskbar_icon(self, on: bool = True) -> None:
        if on:
            ex = (self.ex | WS_EX_TOOLWINDOW) & ~WS_EX_APPWINDOW
        else:
            ex = (self.ex & ~WS_EX_TOOLWINDOW) | WS_EX_APPWINDOW
        self._write(GWL_EXSTYLE, ex)
        # 隐藏后再显示，任务栏/Alt+Tab 才会立刻刷新（不抢焦点）
        user32.ShowWindow(self.hwnd, SW_HIDE)
        self._refresh_frame()
        user32.ShowWindow(self.hwnd, SW_SHOWNOACTIVATE)

    def set_borderless(self, on: bool = True) -> None:
        mask = WS_CAPTION | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU
        style = self.style | mask if not on else self.style & ~mask
        self._write(GWL_STYLE, style)
        self._refresh_frame()

    def set_click_through(self, on: bool = True) -> None:
        if on:
            self._write(GWL_EXSTYLE, self.ex | WS_EX_LAYERED | WS_EX_TRANSPARENT)
            user32.SetLayeredWindowAttributes(self.hwnd, 0, 255, LWA_ALPHA)
        else:
            self._write(GWL_EXSTYLE, self.ex & ~WS_EX_TRANSPARENT)
        self._refresh_frame()

    def set_visible(self, on: bool = True) -> None:
        user32.ShowWindow(self.hwnd, SW_SHOWNOACTIVATE if on else SW_HIDE)
    # ---------------- 位置 / 大小 ----------------
    def place(self, position: str = "keep", margin: int = 30, size: str = "") -> None:
        left, top, right, bottom = self.rect
        w, h = right - left, bottom - top
        if size:
            try:
                parts = size.lower().replace("×", "x").split("x")
                w, h = max(120, int(parts[0])), max(120, int(parts[1]))
            except Exception:
                pass
        pos = (position or "keep").lower()
        if pos in ("", "keep") and not size:
            return
        wa = wintypes.RECT()
        user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(wa), 0)
        xs = {"tl": wa.left + margin, "tr": wa.right - w - margin,
              "bl": wa.left + margin, "br": wa.right - w - margin,
              "center": (wa.left + wa.right - w) // 2}
        ys = {"tl": wa.top + margin, "tr": wa.top + margin,
              "bl": wa.bottom - h - margin, "br": wa.bottom - h - margin,
              "center": (wa.top + wa.bottom - h) // 2}
        x, y = (left, top) if pos == "keep" else (xs.get(pos, left), ys.get(pos, top))
        user32.SetWindowPos(self.hwnd, 0, x, y, w, h, SWP_NOZORDER | SWP_NOACTIVATE)

    def nudge(self, dx: int, dy: int) -> None:
        left, top, _, _ = self.rect
        user32.SetWindowPos(self.hwnd, 0, left + dx, top + dy, 0, 0,
                            SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE)

    def scale_by(self, factor: float) -> None:
        left, top, right, bottom = self.rect
        w = max(120, int((right - left) * factor))
        h = max(120, int((bottom - top) * factor))
        user32.SetWindowPos(self.hwnd, 0, left, top, w, h, SWP_NOZORDER | SWP_NOACTIVATE)

    # ---------------- 恢复 ----------------
    def restore(self) -> bool:
        src = None
        if self.orig_style is not None:
            src = (self.orig_style, self.orig_ex, self.orig_rect, self.orig_show)
        else:                                   # 换了个进程：从备份文件里读
            try:
                data = json.loads(_BACKUP.read_text(encoding="utf-8"))
                src = (int(data["style"]), int(data["ex"]),
                       tuple(int(v) for v in data["rect"]),
                       int(data.get("show_cmd") or 4))
            except Exception:
                pass
        if not src:
            return False
        style, ex, rect, show_cmd = src
        self._write(GWL_STYLE, style)
        self._write(GWL_EXSTYLE, ex)
        # 光写 ex style 不会退出置顶层，必须用 SetWindowPos 真正设置 z-order
        z = HWND_TOPMOST if ex & WS_EX_TOPMOST else HWND_NOTOPMOST
        user32.SetWindowPos(self.hwnd, z, 0, 0, 0, 0,
                            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED)
        wp = WINDOWPLACEMENT()
        wp.length = ctypes.sizeof(WINDOWPLACEMENT)
        # 正常大小(1) 映射成不抢焦点的 4；最小化(2)/最大化(3) 原样恢复
        wp.showCmd = show_cmd if show_cmd in (SW_SHOWMINIMIZED, 3) else SW_SHOWNOACTIVATE
        left, top, right, bottom = rect
        wp.rcNormalPosition = wintypes.RECT(left, top, right, bottom)
        user32.SetWindowPlacement(self.hwnd, ctypes.byref(wp))
        try:
            _BACKUP.unlink()
        except Exception:
            pass
        return True

    def describe(self) -> str:
        s, e = self.style, self.ex
        left, top, right, bottom = self.rect
        flags = [
            "置顶" if e & WS_EX_TOPMOST else "未置顶",
            "鼠标穿透" if e & WS_EX_TRANSPARENT else "可点击",
            "无边框" if not (s & WS_CAPTION) else "有边框",
            "任务栏隐藏" if e & WS_EX_TOOLWINDOW else "任务栏可见",
            "显示中" if user32.IsWindowVisible(self.hwnd) else "已隐藏",
        ]
        return (f"hwnd={self.hwnd} 位置=({left},{top}) "
                f"尺寸={right - left}x{bottom - top} | " + " | ".join(flags))


# ---------------- 全局热键 ----------------
_HK = MOD_CONTROL | MOD_ALT | MOD_NOREPEAT
_HKS = MOD_CONTROL | MOD_ALT | MOD_SHIFT | MOD_NOREPEAT
_VK = {"P": 0x50, "T": 0x54, "B": 0x42, "H": 0x48, "R": 0x52,
       "LEFT": 0x25, "UP": 0x26, "RIGHT": 0x27, "DOWN": 0x28,
       "PLUS": 0xBB, "MINUS": 0xBD}


class HotkeyThread(threading.Thread):
    """注册全局热键（不抢 VTS 焦点）：Ctrl+Alt+P/T/B/H/R、微移、缩放。"""

    def __init__(self, win: PetWindow, log=print):
        super().__init__(name="pet-hotkeys", daemon=True)
        self.win = win
        self.log = log
        self.registered = 0
        self._tid = 0
        self._actions: dict = {}
        self._stop_flag = threading.Event()

    def run(self) -> None:
        self._tid = kernel32.GetCurrentThreadId()
        combos = [
            ("click_through", _HK, _VK["P"]), ("topmost", _HK, _VK["T"]),
            ("borderless", _HK, _VK["B"]), ("visible", _HK, _VK["H"]),
            ("restore", _HK, _VK["R"]),
            ("move_left", _HKS, _VK["LEFT"]), ("move_right", _HKS, _VK["RIGHT"]),
            ("move_up", _HKS, _VK["UP"]), ("move_down", _HKS, _VK["DOWN"]),
            ("grow", _HKS, _VK["PLUS"]), ("shrink", _HKS, _VK["MINUS"]),
        ]
        for i, (name, mods, vk) in enumerate(combos, 1):
            if user32.RegisterHotKey(None, i, mods, vk):
                self._actions[i] = name
            else:
                self.log(f"[pet] 热键被占用，跳过：{name}")
        self.registered = len(self._actions)
        msg = wintypes.MSG()
        while not self._stop_flag.is_set():
            ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if ret <= 0:
                break
            if msg.message == WM_HOTKEY:
                name = self._actions.get(int(msg.wParam))
                if name:
                    try:
                        self._handle(name)
                    except Exception as exc:
                        self.log(f"[pet] 热键动作失败：{type(exc).__name__}: {exc}")
        for i in list(self._actions):
            user32.UnregisterHotKey(None, i)
        self._actions.clear()

    def _handle(self, name: str) -> None:
        w = self.win
        if name == "click_through":
            on = not bool(w.ex & WS_EX_TRANSPARENT)
            w.set_click_through(on)
            self.log("[pet] 鼠标穿透：" + ("开（再按 Ctrl+Alt+P 可点她）" if on else "关"))
        elif name == "topmost":
            on = not bool(w.ex & WS_EX_TOPMOST)
            w.set_topmost(on)
            self.log("[pet] 窗口置顶：" + ("开" if on else "关"))
        elif name == "borderless":
            on = bool(w.style & WS_CAPTION)
            w.set_borderless(on)
            self.log("[pet] 无边框：" + ("开" if on else "关"))
        elif name == "visible":
            on = not bool(user32.IsWindowVisible(w.hwnd))
            w.set_visible(on)
            self.log("[pet] 显示/隐藏：" + ("显示" if on else "隐藏"))
        elif name == "restore":
            w.restore()
            self.log("[pet] 已恢复成普通窗口（热键仍然可用）")
        elif name.startswith("move_"):
            dx, dy = {"move_left": (-20, 0), "move_right": (20, 0),
                      "move_up": (0, -20), "move_down": (0, 20)}[name]
            w.nudge(dx, dy)
        elif name == "grow":
            w.scale_by(1.05)
        elif name == "shrink":
            w.scale_by(1 / 1.05)

    def stop(self) -> None:
        self._stop_flag.set()
        if self._tid:
            user32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
        self.join(timeout=2)
class DesktopPet:
    """给 voice_chat / 独立运行用：应用桌宠样式，退出时按配置恢复。"""

    def __init__(self, cfg, log=print):
        self.cfg = cfg
        self.log = log
        self.win: PetWindow | None = None
        self.hotkeys: HotkeyThread | None = None

    def start(self) -> None:
        self.win = PetWindow(self.cfg.pet_title)
        self.win.capture()
        self.win.ensure_visible()      # VTS 最小化时先还原出来
        w = self.win
        if self.cfg.pet_topmost:
            w.set_topmost(True)
        if self.cfg.pet_hide_taskbar:
            w.hide_taskbar_icon(True)
        if self.cfg.pet_borderless:
            w.set_borderless(True)
        w.place(self.cfg.pet_position, self.cfg.pet_margin, self.cfg.pet_size)
        if self.cfg.pet_click_through:
            w.set_click_through(True)
        if self.cfg.pet_hotkeys:
            self.hotkeys = HotkeyThread(w, self.log)
            self.hotkeys.start()
        self.log(f"[pet] 桌宠模式已开启：{w.describe()}")
        if self.cfg.pet_hotkeys:
            self.log("[pet] 热键：Ctrl+Alt+P 穿透 | T 置顶 | B 无边框 | H 显隐 | "
                     "R 恢复 | Ctrl+Alt+Shift+方向键 微移 | +/- 缩放")

    def stop(self) -> None:
        if self.hotkeys is not None:
            self.hotkeys.stop()
            self.hotkeys = None
        if self.win is not None and self.cfg.pet_restore_on_exit:
            self.win.restore()
            self.log("[pet] 已恢复 VTube Studio 窗口样式")
        self.win = None


def main() -> None:
    from config import CONFIG

    parser = argparse.ArgumentParser(description="把 VTube Studio 变成桌面桌宠")
    parser.add_argument("--status", action="store_true", help="只显示当前窗口样式")
    parser.add_argument("--restore", action="store_true", help="恢复原始窗口样式")
    parser.add_argument("--keep", action="store_true", help="退出时保留样式（默认退出即恢复）")
    parser.add_argument("--title", default="", help="VTS 窗口标题（默认读 .env）")
    parser.add_argument("--position", default="", help="tl/tr/bl/br/center/keep")
    parser.add_argument("--size", default="", help="例 520x900（留空保持现有尺寸）")
    parser.add_argument("--margin", type=int, default=-1, help="离屏幕边缘的像素")
    parser.add_argument("--click-through", action="store_true", help="启动就鼠标穿透")
    parser.add_argument("--no-borderless", action="store_true")
    parser.add_argument("--no-topmost", action="store_true")
    parser.add_argument("--no-taskbar-hide", action="store_true")
    parser.add_argument("--no-hotkeys", action="store_true")
    parser.add_argument("--no-cfg", action="store_true", help="忽略 .env，用命令行参数")
    args = parser.parse_args()

    cfg = CONFIG
    if args.no_cfg or args.title:
        cfg.pet_title = args.title or "VTube Studio"
    if args.position:
        cfg.pet_position = args.position
    if args.size:
        cfg.pet_size = args.size
    if args.margin >= 0:
        cfg.pet_margin = args.margin
    if args.click_through:
        cfg.pet_click_through = True
    if args.no_borderless:
        cfg.pet_borderless = False
    if args.no_topmost:
        cfg.pet_topmost = False
    if args.no_taskbar_hide:
        cfg.pet_hide_taskbar = False
    if args.no_hotkeys:
        cfg.pet_hotkeys = False

    try:
        if args.status:
            print(PetWindow(cfg.pet_title).describe())
            return
        if args.restore:
            win = PetWindow(cfg.pet_title)
            print("已恢复原样" if win.restore() else "没有样式备份，窗口本来就是原样")
            print(win.describe())
            return
        pet = DesktopPet(cfg)
        pet.start()
        if args.keep:
            return
        print("[pet] 按 Ctrl+C 退出并恢复窗口")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            pet.stop()
    except RuntimeError as exc:
        raise SystemExit(f"[pet] {exc}")


if __name__ == "__main__":
    main()