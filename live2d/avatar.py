"""恋恋的 Live2D 形象控制（VTube Studio 插件协议）。

本模块只负责「驱动」，渲染交给 VTube Studio（免费，自带示例模型）：

    Player 播放电平 ──每帧──> InjectParameterDataRequest（MouthOpen，口型）
    大模型情绪判断 ──────────> ExpressionActivationRequest / 热键（表情）

Live2D 是锦上添花，绝不能拖累聊天：
  · VTS 没装 / 没开 / 没连上 → 后台静默重连，其余功能照常；
  · 首次连接要在 VTube Studio 里点一次 Allow，token 会自动存下来；
  · 口型帧率被 VTS 限流时自动降速。

    python avatar_test.py            # 连真 VTS 测试
    python avatar_test.py --mock     # 不用 VTS，内置假服务端自测
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import math
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

try:  # websockets >= 14 的推荐入口
    from websockets.asyncio.client import connect as _ws_connect
except ImportError:  # 兼容旧版
    try:
        from websockets.client import connect as _ws_connect
    except ImportError:  # 真没装：Live2D 自动关闭，不影响聊天
        _ws_connect = None

API_NAME = "VTubeStudioPublicAPI"
API_VERSION = "1.0"

EMOTIONS = ("neutral", "happy", "shy", "angry", "sad", "surprised")
ZH = {"neutral": "平静", "happy": "开心", "shy": "害羞",
      "angry": "生气", "sad": "低落", "surprised": "惊讶"}


class APIError(RuntimeError):
    """VTube Studio 明确拒绝了请求（服务端返回 APIError）。"""

    def __init__(self, error_id: int, message: str):
        super().__init__(f"VTS 拒绝请求（errorID {error_id}）：{message}")
        self.error_id = error_id
        self.message = message


def parse_map(text: str) -> dict:
    """把 happy=exp_01.exp3.json,shy=exp_02.exp3.json 解析成 dict。"""
    out: dict = {}
    for item in (text or "").replace("；", ",").replace(";", ",").split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, val = item.split("=", 1)
        key, val = key.strip().lower(), val.strip()
        if key and val:
            out[key] = val
    return out


class VTubeStudioClient:
    """VTube Studio 公共 API 的最小客户端：连接 / 授权 / 请求响应 / 注入。"""

    def __init__(self, url: str, plugin_name: str, plugin_dev: str,
                 token_file: Path, log: Callable[[str], None] = print):
        self.url = url
        self.plugin_name = plugin_name
        self.plugin_dev = plugin_dev
        self.token_file = token_file
        self.log = log
        self._ws: Any = None
        self._pending: dict = {}
        self._rx: Optional[asyncio.Task] = None
        self.token: str = self._load_token()
        self.authenticated = False
        self.model_name = ""

    # ---------------- 连接 / 授权 ----------------
    async def connect(self, timeout: float = 8.0) -> None:
        if _ws_connect is None:
            raise RuntimeError("没装 websockets 库（pip install websockets）")
        self._ws = await _ws_connect(self.url, max_size=4 * 1024 * 1024,
                                     open_timeout=timeout, ping_interval=20)
        self._rx = asyncio.create_task(self._recv_loop(), name="vts-rx")
        await self._authenticate()

    async def close(self) -> None:
        self.authenticated = False
        if self._rx is not None:
            self._rx.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._rx
            self._rx = None
        ws, self._ws = self._ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("与 VTube Studio 的连接已断开"))
        self._pending.clear()

    async def _recv_loop(self) -> None:
        ws = self._ws
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                fut = self._pending.pop(str(msg.get("requestID") or ""), None)
                if fut is not None and not fut.done():
                    fut.set_result(msg)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        finally:
            self.authenticated = False
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("与 VTube Studio 的连接已断开"))
            self._pending.clear()

    async def _request(self, message_type: str, data: Optional[dict] = None,
                       timeout: float = 6.0) -> dict:
        if self._ws is None:
            raise ConnectionError("还没连接 VTube Studio")
        rid = uuid.uuid4().hex
        fut = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        payload = {"apiName": API_NAME, "apiVersion": API_VERSION,
                   "requestID": rid, "messageType": message_type,
                   "data": data or {}}
        try:
            await self._ws.send(json.dumps(payload, ensure_ascii=False))
        except Exception:
            self._pending.pop(rid, None)
            raise
        try:
            resp = await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise TimeoutError(f"{message_type} 超时（{timeout:.0f}s）") from None
        if resp.get("messageType") == "APIError":
            d = resp.get("data") or {}
            raise APIError(int(d.get("errorID") or 0), str(d.get("message") or ""))
        return resp

    async def _authenticate(self) -> None:
        if self.token:
            try:
                resp = await self._request("AuthenticationRequest", {
                    "pluginName": self.plugin_name,
                    "pluginDeveloper": self.plugin_dev,
                    "authenticationToken": self.token})
                if bool((resp.get("data") or {}).get("authenticated")):
                    self.authenticated = True
                    await self._fetch_state()
                    self.log(f"[avatar] 已连接 VTube Studio（token 复用，模型："
                             f"{self.model_name or '未知'}）")
                    return
                self.log("[avatar] 保存的 token 已失效，重新申请授权")
            except (APIError, TimeoutError, ConnectionError, OSError) as exc:
                self.log(f"[avatar] token 校验失败（{exc}），重新申请授权")
            self._save_token("")

        self.log("[avatar] 正在申请插件授权 —— 请在 VTube Studio 窗口里点 Allow"
                 "（若没弹窗，检查设置里是否开启了 API 插件功能）")
        try:
            resp = await self._request("AuthenticationTokenRequest", {
                "pluginName": self.plugin_name,
                "pluginDeveloper": self.plugin_dev}, timeout=120.0)
        except TimeoutError as exc:
            raise ConnectionError("等不到授权确认（VTS 里没点 Allow？）") from exc
        token = str(((resp.get("data") or {}).get("authenticationToken") or "")).strip()
        if not token:
            raise ConnectionError("VTube Studio 没有返回 authenticationToken")
        self.token = token
        self._save_token(token)
        resp = await self._request("AuthenticationRequest", {
            "pluginName": self.plugin_name,
            "pluginDeveloper": self.plugin_dev,
            "authenticationToken": token})
        if not bool((resp.get("data") or {}).get("authenticated")):
            raise ConnectionError("授权被拒绝（在 VTS 里点了 Deny？）")
        self.authenticated = True
        await self._fetch_state()
        self.log(f"[avatar] 授权成功，模型：{self.model_name or '未知'}")

    async def _fetch_state(self) -> None:
        # VTS 1.35 的 APIStateRequest 不返回模型名，CurrentModelRequest 才有
        try:
            resp = await self._request("CurrentModelRequest", {})
            d = resp.get("data") or {}
            if d.get("modelLoaded"):
                self.model_name = str(d.get("modelName") or d.get("vtsModelName") or "")
                return
        except Exception:
            pass
        try:
            resp = await self._request("APIStateRequest", {})
            self.model_name = str((resp.get("data") or {}).get("modelName") or "")
        except Exception:
            self.model_name = ""

    # ---------------- 常用 API ----------------
    async def inject(self, values: dict) -> None:
        """把一组参数值写进模型（MouthOpen 等）。"""
        await self._request("InjectParameterDataRequest", {
            "faceFound": False,
            "mode": "set",
            "parameterValues": [
                {"id": k, "value": round(float(v), 4), "weight": 1.0}
                for k, v in values.items()],
        }, timeout=4.0)

    async def activate_expression(self, expression_file: str, active: bool = True) -> None:
        await self._request("ExpressionActivationRequest", {
            "expressionFile": expression_file, "active": bool(active)})

    async def trigger_hotkey(self, hotkey_id: str) -> None:
        """优先用新版接口，老版本 VTS 退回已废弃但仍可用的 HotkeyTriggerRequest。"""
        try:
            await self._request("TriggerHotkeyByKeySequenceRequest",
                                {"hotkeyID": hotkey_id})
        except APIError:
            await self._request("HotkeyTriggerRequest", {"hotkeyID": hotkey_id})

    # ---------------- token 落盘 ----------------
    def _load_token(self) -> str:
        try:
            if self.token_file.exists():
                return str(json.loads(self.token_file.read_text(encoding="utf-8"))
                           .get("token") or "")
        except Exception:
            pass
        return ""

    def _save_token(self, token: str) -> None:
        try:
            if not token:
                with contextlib.suppress(Exception):
                    self.token_file.unlink()
                return
            self.token_file.parent.mkdir(parents=True, exist_ok=True)
            self.token_file.write_text(
                json.dumps({"token": token}, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            self.log(f"[avatar] token 存不下来（{exc}），下次启动要重新授权")
class AvatarController:
    """把播放电平和情绪翻译成 VTube Studio 请求；断了自动重连。"""

    def __init__(self, cfg, level_provider: Callable[[], float],
                 log: Callable[[str], None] = print):
        self.cfg = cfg
        self.level_provider = level_provider or (lambda: 0.0)
        self.log = log
        token_file = (Path(cfg.live2d_token_file) if cfg.live2d_token_file
                      else Path(__file__).resolve().parents[1] / "data" / "vtube_token.json")
        self.client = VTubeStudioClient(
            cfg.live2d_url, cfg.live2d_plugin_name, cfg.live2d_plugin_dev,
            token_file, log)
        self.expr_map = parse_map(cfg.live2d_expr)
        self.hotkey_map = parse_map(cfg.live2d_hotkey)
        self.fps = max(4.0, min(60.0, float(cfg.live2d_fps)))
        self._task: Optional[asyncio.Task] = None
        self._emotion: Optional[str] = None
        self._pending_emotion: Optional[str] = None
        self._active_expr = ""
        self._warned: set = set()
        self._injects = 0
        self._emotions = 0

    @property
    def connected(self) -> bool:
        return self.client.authenticated

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="avatar")

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await self.client.close()

    def set_emotion(self, emotion: str) -> None:
        """由主循环调用（同步、不阻塞）；真正切表情在后台任务里做。"""
        emo = (emotion or "neutral").strip().lower()
        if emo not in EMOTIONS:
            emo = "neutral"
        if emo != self._emotion:
            self._emotion = emo
            self._pending_emotion = emo

    def describe(self) -> str:
        state = (f"已连接（{self.client.model_name or '模型名未知'}）"
                 if self.connected else "未连接（后台重试中）")
        return (f"Live2D: {state} | 口型注入 {self._injects} 帧 | "
                f"情绪切换 {self._emotions} 次")

    # ---------------- 内部 ----------------
    async def _run(self) -> None:
        delay = 3.0
        while True:
            try:
                await self.client.connect()
                delay = 3.0
                await self._pump()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._warn_once(f"[avatar] 连不上 {self.cfg.live2d_url}："
                                f"{type(exc).__name__}: {str(exc)[:90]}；{delay:.0f}s 后重试。"
                                "VTube Studio 启动并开启插件 API 后会自动连上")
            await self.client.close()
            await asyncio.sleep(delay)
            delay = min(delay * 1.7, 60.0)

    async def _pump(self) -> None:
        dt = 1.0 / self.fps
        last_sent = 0.0
        while True:
            if self._pending_emotion is not None:
                emo, self._pending_emotion = self._pending_emotion, None
                await self._apply_emotion(emo)
            else:
                value = self._mouth_value(self.level_provider())
                if value > 0.01 or last_sent > 0.01:
                    try:
                        await self.client.inject({self.cfg.live2d_mouth_param: value})
                        self._injects += 1
                        last_sent = value
                    except APIError as exc:
                        if self._is_rate_limit(exc):
                            self.fps = max(6.0, self.fps * 0.5)
                            dt = 1.0 / self.fps
                            self._warn_once(f"[avatar] VTS 限流，口型帧率自动降到 {self.fps:.0f}fps")
                        else:
                            self._warn_once(f"[avatar] 口型参数被拒绝：{exc}。"
                                            "标准模型一般是 MouthOpen，自定义模型可能是 "
                                            "ParamMouthOpenY，用 LIVE2D_MOUTH_PARAM 指定")
                            last_sent = 0.0
            await asyncio.sleep(dt)

    async def _apply_emotion(self, emo: str) -> None:
        self._emotions += 1
        if self.cfg.live2d_debug:
            self.log(f"[avatar] 情绪 -> {emo}（{ZH.get(emo, emo)}）")
        expr = self.expr_map.get(emo, "")
        if expr:
            try:
                if self._active_expr and self._active_expr != expr:
                    await self.client.activate_expression(self._active_expr, False)
                await self.client.activate_expression(expr, True)
                self._active_expr = expr
                return
            except APIError as exc:
                self._warn_once(f"[avatar] 表情文件 {expr!r} 用不了：{exc}。"
                                "文件名要和模型目录里的 .exp3.json 一致，或改用 LIVE2D_HOTKEY")
        if (not expr and emo == "neutral" and self._active_expr
                and self.cfg.live2d_reset_neutral):
            with contextlib.suppress(APIError):
                await self.client.activate_expression(self._active_expr, False)
                self._active_expr = ""
        hotkey = self.hotkey_map.get(emo, "")
        if hotkey:
            try:
                await self.client.trigger_hotkey(hotkey)
            except APIError as exc:
                self._warn_once(f"[avatar] 热键 {hotkey!r} 触发失败：{exc}。"
                                "在 VTS 里建个热键，把它的 ID 填到 LIVE2D_HOTKEY")

    def _mouth_value(self, level: float) -> float:
        """播放电平 RMS -> 开口度 0~1（先扣噪声底，再压曲线，看起来更自然）。"""
        try:
            x = float(level)
        except Exception:
            x = 0.0
        if not math.isfinite(x) or x <= 0:
            return 0.0
        floor = max(0.0, float(self.cfg.live2d_mouth_floor))
        gain = max(0.1, float(self.cfg.live2d_mouth_gain))
        if x <= floor:
            return 0.0
        return round(min(1.0, (x - floor) * gain) ** 0.7, 4)

    @staticmethod
    def _is_rate_limit(exc: APIError) -> bool:
        text = f"{exc.error_id} {exc.message}".lower()
        return any(k in text for k in ("rate", "limit", "too many", "限流"))

    def _warn_once(self, msg: str) -> None:
        key = msg.split("；")[0][:80]   # 忽略会变的「Xs 后重试」部分，避免刷屏
        if key in self._warned:
            return
        self._warned.add(key)
        self.log(msg)


def make_avatar(cfg, player, log: Callable[[str], None] = print) -> Optional[AvatarController]:
    """按配置创建控制器；没开或没装 websockets 就返回 None（其余功能照常）。"""
    if not cfg.live2d_enable:
        return None
    if _ws_connect is None:
        log("[avatar] 想开 Live2D 但没装 websockets：venv\\Scripts\\pip install websockets")
        return None
    return AvatarController(cfg, lambda: getattr(player, "level", 0.0), log)