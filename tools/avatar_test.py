"""Live2D / VTube Studio 连通性测试。

    python avatar_test.py                 # 连真 VTS：状态 + 4 秒口型 + 表情轮流切
    python avatar_test.py --emotion happy # 只切一个表情，看效果
    python avatar_test.py --mock          # 内置假 VTS 服务端，不装 VTube Studio 也能自测

怎么知道连没连上：看 VTube Studio 里她的嘴会不会跟着你说话动。
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent        # voicebot/
for _p in (str(_ROOT), str(_ROOT / "core"), str(_ROOT / "live2d")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import asyncio
import contextlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path

from avatar import AvatarController
from config import CONFIG

try:
    from websockets.asyncio.server import serve as ws_serve
except ImportError:
    from websockets.server import serve as ws_serve


class SpeechLevel:
    """模拟说话时的播放电平（和真实 Player.level 一个量级）。"""

    def __init__(self, on: float = 2.0, off: float = 0.5):
        self.on, self.off = on, off
        self.t0 = time.monotonic()

    def __call__(self) -> float:
        now = time.monotonic()
        if (now - self.t0) % (self.on + self.off) > self.on:
            return 0.0
        return max(0.0, 0.08 + 0.05 * math.sin(2 * math.pi * 3.3 * now))


class MockVTS:
    """假 VTube Studio：实现授权、状态、注入、表情、热键这几个消息。"""

    def __init__(self):
        self.token_requests = 0
        self.auth_requests = 0
        self.injects: list = []
        self.expressions: list = []
        self.hotkeys: list = []
        self.server = None
        self.port = 0

    async def start(self) -> int:
        self.server = await ws_serve(self._handler, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self.port

    async def stop(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

    async def _handler(self, ws) -> None:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            t = str(msg.get("messageType") or "")
            d = msg.get("data") or {}
            if t == "AuthenticationTokenRequest":
                self.token_requests += 1
                data = {"authenticationToken": "MOCK-TOKEN-123"}
            elif t == "AuthenticationRequest":
                self.auth_requests += 1
                data = {"authenticated": d.get("authenticationToken") == "MOCK-TOKEN-123"}
            elif t == "APIStateRequest":
                data = {"modelName": "MockModel", "modelLoaded": True}
            elif t == "InjectParameterDataRequest":
                self.injects.append({v.get("id"): v.get("value")
                                     for v in (d.get("parameterValues") or [])})
                data = {}
            elif t == "ExpressionActivationRequest":
                self.expressions.append((d.get("expressionFile"), bool(d.get("active"))))
                data = {"isActive": bool(d.get("active")),
                        "expressionFile": d.get("expressionFile")}
            elif t.startswith("TriggerHotkey") or t == "HotkeyTriggerRequest":
                self.hotkeys.append(d.get("hotkeyID"))
                data = {"hotkeyID": d.get("hotkeyID")}
            else:
                data = {}
            await ws.send(json.dumps({
                "apiName": "VTubeStudioPublicAPI", "apiVersion": "1.0",
                "requestID": msg.get("requestID"),
                "messageType": t.replace("Request", "Response"), "data": data}))
async def _wait_connected(ctrl: AvatarController, timeout: float) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if ctrl.connected:
            return True
        await asyncio.sleep(0.2)
    return False


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"：{detail}" if detail else ""))
    return bool(ok)


# ---------------------------------------------------------------- 真 VTS
async def run_real(args) -> int:
    cfg = CONFIG
    cfg.live2d_enable = True
    if args.url:
        cfg.live2d_url = args.url
    if args.expr:
        cfg.live2d_expr = args.expr
    if args.hotkey:
        cfg.live2d_hotkey = args.hotkey
    if args.debug:
        cfg.live2d_debug = True

    ctrl = AvatarController(cfg, SpeechLevel())
    ctrl.start()
    print(f"[test] 连接 {cfg.live2d_url} ...")
    if not await _wait_connected(ctrl, max(5.0, args.wait)):
        print("\n[test] 连不上 VTube Studio。检查顺序：")
        print("  1. VTube Studio 已经启动（Steam 免费版即可），并且载入了一个模型；")
        print("  2. VTS 设置里开启了 API / 插件功能（默认端口 8001）；")
        print("  3. 本测试会等 --wait 秒；VTS 里弹出授权窗口就点 Allow。")
        await ctrl.close()
        return 1

    print(f"[test] 已连接：{ctrl.client.model_name or '未知模型'}")
    if args.emotion:
        print(f"[test] 切表情：{args.emotion}")
        ctrl.set_emotion(args.emotion)
        await asyncio.sleep(1.2)
    else:
        if not args.no_mouth:
            print(f"[test] 口型测试 {args.seconds:.0f} 秒 —— 看她的嘴会不会动 ...")
            await asyncio.sleep(args.seconds)
            n = ctrl._injects
            ok = n > 5
            print(f"  [{'PASS' if ok else 'FAIL'}] 口型注入 {n} 帧" +
                  ("" if ok else "（一帧都没发出去，看上面 [avatar] 的提示）"))
        print("[test] 表情轮流切：happy -> angry -> neutral")
        for emo in ("happy", "angry", "neutral"):
            ctrl.set_emotion(emo)
            await asyncio.sleep(0.8)
        print("  没配 LIVE2D_EXPR / LIVE2D_HOTKEY 时看不到变化，属正常。")
    await ctrl.close()
    return 0


# ---------------------------------------------------------------- 假 VTS 自测
async def run_mock(args) -> int:
    mock = MockVTS()
    await mock.start()
    token_file = Path(tempfile.gettempdir()) / "lianlian_vts_mock_token.json"
    with contextlib.suppress(Exception):
        token_file.unlink()

    cfg = CONFIG
    cfg.live2d_enable = True
    cfg.live2d_url = f"ws://127.0.0.1:{mock.port}"
    cfg.live2d_token_file = str(token_file)
    cfg.live2d_fps = 24
    cfg.live2d_debug = True
    cfg.live2d_expr = "happy=exp_01.exp3.json"
    cfg.live2d_hotkey = "sad=ht_sad"
    print(f"[mock] 假 VTube Studio 已启动：ws://127.0.0.1:{mock.port}")

    results = []
    ctrl = AvatarController(cfg, SpeechLevel(on=1.0, off=0.25))
    ctrl.start()
    connected = await _wait_connected(ctrl, 5)
    results.append(_check("连接 + 首次授权（Token 申请）", connected))
    if not connected:
        await ctrl.close()
        await mock.stop()
        return 1

    await asyncio.sleep(1.1)                        # 让它说一「句」
    ctrl.set_emotion("happy")                       # 走表情文件
    await asyncio.sleep(0.35)
    ctrl.set_emotion("sad")                         # 走热键
    await asyncio.sleep(0.35)
    ctrl.set_emotion("neutral")                     # 无映射 -> 取消表情
    await asyncio.sleep(0.35)
    n_inject = len(mock.injects)
    mouth_max = max((v.get("MouthOpen", 0.0) for v in mock.injects), default=0.0)
    await ctrl.close()

    results.append(_check("口型注入", n_inject >= 8,
                          f"{n_inject} 帧，最大开口 {mouth_max:.2f}"))
    results.append(_check("表情激活 exp_01", ("exp_01.exp3.json", True) in mock.expressions))
    results.append(_check("表情取消（neutral）", ("exp_01.exp3.json", False) in mock.expressions))
    results.append(_check("热键触发 ht_sad", mock.hotkeys == ["ht_sad"],
                          str(mock.hotkeys)))
    results.append(_check("MouthOpen 数值范围", 0.05 < mouth_max <= 1.0))

    # 再连一次：应该复用 token，不再弹授权
    ctrl2 = AvatarController(cfg, lambda: 0.0)
    ctrl2.start()
    connected2 = await _wait_connected(ctrl2, 5)
    await ctrl2.close()
    results.append(_check("重连复用 token", connected2 and mock.token_requests == 1,
                          f"token 申请 {mock.token_requests} 次，认证 {mock.auth_requests} 次"))
    results.append(_check("模型名读取", ctrl.client.model_name == "MockModel",
                          ctrl.client.model_name))

    await mock.stop()
    with contextlib.suppress(Exception):
        token_file.unlink()
    print(f"\n[mock] {sum(results)}/{len(results)} 项通过")
    if all(results):
        print("[mock] 程序侧没问题，可以装 VTube Studio 看真效果了：python avatar_test.py")
        return 0
    print("[mock] 有失败项，把上面的输出发我。")
    return 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live2D / VTube Studio 连通性测试")
    p.add_argument("--mock", action="store_true",
                   help="内置假 VTS 服务端，不装 VTube Studio 也能自测")
    p.add_argument("--url", default="", help="VTS WebSocket 地址（默认读 .env 的 LIVE2D_URL）")
    p.add_argument("--seconds", type=float, default=4.0, help="口型测试时长（秒）")
    p.add_argument("--wait", type=float, default=12.0,
                   help="等 VTS 授权/连接的秒数（首次授权点 Allow 要留够时间）")
    p.add_argument("--emotion", default="",
                   help="只切一个表情后退出：happy/shy/angry/sad/surprised/neutral")
    p.add_argument("--expr", default="", help="临时指定情绪->表情映射，例 happy=exp_01.exp3.json")
    p.add_argument("--hotkey", default="", help="临时指定情绪->热键映射，例 angry=ht_angry")
    p.add_argument("--no-mouth", action="store_true", help="跳过口型测试")
    p.add_argument("--debug", action="store_true", help="打印每次情绪切换")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        code = asyncio.run(run_mock(args) if args.mock else run_real(args))
    except KeyboardInterrupt:
        code = 130
    sys.exit(code)


if __name__ == "__main__":
    main()