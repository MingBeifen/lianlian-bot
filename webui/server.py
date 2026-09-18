"""恋恋 Web 控制台：模型管理 + 记忆管理 + Live2D 切换 + 聊天（WebSocket）。

跑在 voice_chat 进程里（WEBUI=1 时自动启动），
浏览器打开 http://127.0.0.1:8765 即可。
"""
from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

from . import catalog
from .env_store import set_env

STATIC = Path(__file__).with_name("static")
HARNESS = catalog.HARNESS
ALLOWED_PREVIEW_ROOTS = [catalog.VTS_MODELS, HARNESS / "models",
                         HARNESS / "live2d-models"]

MODELS_FILE = Path(__file__).resolve().parent.parent / "models.json"


def patch_models_json(updates: dict) -> bool:
    """把改动写回 models.json（如 live2d.backend / live2d.native_model）。"""
    if not MODELS_FILE.exists():
        return False
    try:
        data = json.loads(MODELS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return False
    for dotted, value in updates.items():
        node = data
        parts = dotted.split(".")
        for key in parts[:-1]:
            node = node.setdefault(key, {})
        node[parts[-1]] = value
    MODELS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return True


class WebUI:
    def __init__(self, bot, log=print):
        self.bot = bot
        self.cfg = bot.cfg
        self.log = log
        self.host = self.cfg.webui_host
        self.port = int(self.cfg.webui_port)
        self.clients: set = set()
        self._runner = None
        self._site = None
        self._t0 = time.time()

    # ---------------- 生命周期 ----------------
    async def start(self) -> None:
        app = self._app()
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        sinks = getattr(self.bot, "event_sinks", None)
        if sinks is not None and self._on_event not in sinks:
            sinks.append(self._on_event)
        self.log(f"[webui] 控制台已启动：http://{self.host}:{self.port}")

    async def stop(self) -> None:
        sinks = getattr(self.bot, "event_sinks", None)
        if sinks and self._on_event in sinks:
            sinks.remove(self._on_event)
        for ws in list(self.clients):
            try:
                await ws.close()
            except Exception:
                pass
        self.clients.clear()
        if self._site is not None:
            await self._site.stop()
            self._site = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        self.log("[webui] 控制台已停止")

    async def _on_event(self, event: dict) -> None:
        await self.broadcast(event)

    async def broadcast(self, event: dict) -> None:
        if not self.clients:
            return
        text = json.dumps(event, ensure_ascii=False)
        for ws in list(self.clients):
            try:
                await ws.send_str(text)
            except Exception:
                self.clients.discard(ws)

    # ---------------- 状态 ----------------
    def _status(self) -> dict:
        bot, cfg = self.bot, self.cfg
        memory_count = len(bot.memory) if getattr(bot, "memory", None) else 0
        return {
            "ok": True,
            "uptime": round(time.time() - self._t0, 1),
            "busy": bool(getattr(bot, "busy", False)),
            "web_clients": len(self.clients),
            "llm": {"model": cfg.llm_model, "base_url": cfg.llm_base_url,
                    "history": len(getattr(getattr(bot, "llm", None), "history", []) or [])},
            "asr": {"backend": cfg.asr_backend, "model": cfg.asr_model,
                    "device": cfg.asr_device},
            "tts": {"backend": cfg.tts_backend, "voice": cfg.tts_voice,
                    "sample_rate": getattr(getattr(bot, "tts", None), "sample_rate", None)},
            "memory": {"count": memory_count, "enabled": cfg.memory_enable,
                       "auto_write": cfg.memory_auto_write},
            "mic": {"muted": bool(getattr(bot, "mic_user_muted", False)),
                    "has_mic": getattr(bot, "mic", None) is not None},
            "pet": {"backend": cfg.pet_backend,
                    "model": Path(cfg.native_pet_model).parent.name if cfg.native_pet_model else "",
                    "connected": bool(getattr(getattr(bot, "native_pet", None), "connected", False)),
                    "vts": bool(getattr(getattr(bot, "avatar", None), "connected", False))},
        }

    async def h_status(self, request):
        return web.json_response(self._status())

    # ---------------- 模型目录 / 切换 ----------------
    async def h_models(self, request):
        cfg = self.cfg
        cat = {
            "llm_gguf": catalog.llm_gguf(),
            "kokoro_voices": catalog.kokoro_voices(),
            "embed_models": catalog.embed_models(),
            "asr_models": catalog.asr_models(),
            "gsv_refs": catalog.gsv_refs(),
            "live2d": catalog.live2d_models([cfg.native_pet_model]),
            "vts": [],
        }
        bot = self.bot
        avatar = getattr(bot, "avatar", None)
        if avatar is not None and getattr(avatar, "connected", False):
            try:
                resp = await avatar.client._request("AvailableModelsRequest", {})
                cat["vts"] = (resp.get("data") or {}).get("availableModels") or []
            except Exception:
                pass
        return web.json_response({"current": catalog.snapshot(cfg), "catalog": cat})

    async def h_live2d_apply(self, request):
        if getattr(self.bot, "busy", False):
            return web.json_response({"ok": False, "error": "她正在说话/思考，稍后再切"}, status=409)
        body = await request.json()
        path = str(body.get("path") or "").strip()
        name = str(body.get("name") or "").strip()
        backend = str(body.get("backend") or "native").lower()
        if backend == "native":
            if not path or not Path(path).exists():
                return web.json_response({"ok": False, "error": f"模型文件不存在：{path}"}, status=400)
        elif not (name or path):
            return web.json_response({"ok": False, "error": "没有指定 VTS 模型"}, status=400)
        if path:
            self.cfg.native_pet_model = path
        if backend == "native":
            link = getattr(self.bot, "native_pet", None)
            if link is None:
                return web.json_response({"ok": False, "error": "当前不是 native 桌宠模式（PET_BACKEND=native 才支持热切换）"}, status=400)
            await link.set_model(path)
        else:
            avatar = getattr(self.bot, "avatar", None)
            if avatar is None or not getattr(avatar, "connected", False):
                return web.json_response({"ok": False, "error": "VTube Studio 未连接"}, status=400)
            target = (name or Path(path).parent.name).lower()
            model_id = ""
            try:
                resp = await avatar.client._request("AvailableModelsRequest", {})
                for m in (resp.get("data") or {}).get("availableModels") or []:
                    name = str(m.get("modelName") or "").lower()
                    if name == target or target in name:
                        model_id = str(m.get("modelID") or "")
                        break
            except Exception as exc:
                return web.json_response({"ok": False, "error": f"读取 VTS 模型列表失败：{exc}"}, status=500)
            if not model_id:
                return web.json_response({"ok": False, "error": "VTS 里找不到匹配模型（先在 VTS 里加载一次）"}, status=404)
            try:
                await avatar.client._request("ModelLoadRequest", {"modelID": model_id})
            except Exception as exc:
                return web.json_response({"ok": False, "error": f"VTS 加载失败：{exc}"}, status=500)
        env = {"PET_BACKEND": backend}
        if path:
            env["NATIVE_PET_MODEL"] = path
        set_env(env)
        updates = {"live2d.backend": backend}
        if path:
            updates["live2d.native_model"] = path
        if patch_models_json(updates):
            self.log(f"[webui] models.json 已更新：{updates}")
        label = (name or Path(path).parent.name) if backend != "native" else Path(path).parent.name
        await self.broadcast({"type": "pet", "model": label})
        return web.json_response({"ok": True, "model": label})

    async def h_live2d_preview(self, request):
        raw = request.query.get("path") or ""
        p = Path(raw)
        if not p.exists() or p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            return web.Response(status=404)
        allowed = any(str(p).lower().startswith(str(root).lower())
                      for root in ALLOWED_PREVIEW_ROOTS)
        if not allowed and "Live2DModels" not in str(p):
            return web.Response(status=403)
        ctype = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                 "png": "image/png", "webp": "image/webp"}[p.suffix.lower().lstrip(".")]
        return web.Response(body=p.read_bytes(), content_type=ctype)
    # ---------------- 记忆 ----------------
    def _store(self):
        return getattr(self.bot, "memory", None)

    async def h_memories(self, request):
        store = self._store()
        if store is None:
            return web.json_response({"ok": False, "error": "记忆功能未启用"}, status=503)
        q = (request.query.get("q") or "").strip()
        kind = (request.query.get("kind") or "").strip()
        try:
            if q:
                items = store.recall(q, k=50, kinds=[kind] if kind else None)
            else:
                items = store.all()
                if kind:
                    items = [m for m in items if m.kind == kind]
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"检索失败：{exc}"}, status=500)
        data = []
        for m in items:
            d = m.to_dict()
            d["sim"] = round(float(m.sim), 3)
            d["score"] = round(float(m.score), 3)
            data.append(d)
        return web.json_response({"ok": True, "count": len(data), "items": data})

    async def h_memory_add(self, request):
        store = self._store()
        if store is None:
            return web.json_response({"ok": False, "error": "记忆功能未启用"}, status=503)
        body = await request.json()
        content = str(body.get("content") or "").strip()
        if not content:
            return web.json_response({"ok": False, "error": "内容不能为空"}, status=400)
        kind = str(body.get("kind") or "fact").lower()
        try:
            importance = float(body.get("importance", 0.5))
        except Exception:
            importance = 0.5
        try:
            action, mid = store.add(content, kind, importance)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        item = store.get(mid)
        return web.json_response({"ok": True, "action": action, "id": mid,
                                  "item": item.to_dict() if item else None})

    async def h_memory_update(self, request):
        store = self._store()
        if store is None:
            return web.json_response({"ok": False, "error": "记忆功能未启用"}, status=503)
        try:
            mid = int(request.match_info["mid"])
        except Exception:
            return web.json_response({"ok": False, "error": "id 不对"}, status=400)
        body = await request.json()
        importance = body.get("importance")
        try:
            importance = float(importance) if importance is not None else None
        except Exception:
            return web.json_response({"ok": False, "error": "权重必须是数字"}, status=400)
        ok = store.update(mid, content=body.get("content"), kind=body.get("kind"),
                          importance=importance)
        if not ok:
            return web.json_response({"ok": False, "error": "没找到这条记忆"}, status=404)
        item = store.get(mid)
        return web.json_response({"ok": True, "item": item.to_dict() if item else None})

    async def h_memory_delete(self, request):
        store = self._store()
        if store is None:
            return web.json_response({"ok": False, "error": "记忆功能未启用"}, status=503)
        try:
            mid = int(request.match_info["mid"])
        except Exception:
            return web.json_response({"ok": False, "error": "id 不对"}, status=400)
        ok = store.forget(mid)
        return web.json_response({"ok": ok, "id": mid})

    async def h_reembed(self, request):
        store = self._store()
        if store is None:
            return web.json_response({"ok": False, "error": "记忆功能未启用"}, status=503)
        try:
            n = await asyncio.to_thread(store.reembed_all)
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"重算失败：{exc}"}, status=500)
        return web.json_response({"ok": True, "count": n})

    async def h_say(self, request):
        if getattr(self.bot, "busy", False):
            return web.json_response({"ok": False, "error": "她正在说话"}, status=409)
        body = await request.json()
        text = str(body.get("text") or "").strip()
        if not text:
            return web.json_response({"ok": False, "error": "文本为空"}, status=400)
        self.bot.turn = asyncio.create_task(self.bot.say(text))
        return web.json_response({"ok": True})

    async def h_mic(self, request):
        body = await request.json()
        bot = self.bot
        if hasattr(bot, "set_local_mic_muted"):
            bot.set_local_mic_muted(bool(body.get("muted")))
        return web.json_response({"ok": True, "muted": bool(getattr(bot, "mic_user_muted", False))})

    # ---------------- 聊天 WebSocket ----------------
    async def h_ws(self, request):
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        self.clients.add(ws)
        try:
            await ws.send_str(json.dumps({"type": "hello", "status": self._status()},
                                         ensure_ascii=False))
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except Exception:
                    continue
                t = data.get("type")
                if t == "text":
                    self.bot.submit_web_text(str(data.get("text") or ""))
                elif t == "audio":
                    raw = base64.b64decode(data.get("pcm") or "")
                    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                    sr = int(data.get("sample_rate") or 16000)
                    if sr != 16000 and len(audio) > 1:
                        n = int(len(audio) * 16000 / sr)
                        audio = np.interp(np.linspace(0, len(audio) - 1, n),
                                          np.arange(len(audio)), audio).astype(np.float32)
                    self.bot.submit_web_audio(audio)
                elif t == "interrupt":
                    if getattr(self.bot, "busy", False):
                        self.bot.interrupt.set()
                elif t == "mic":
                    if hasattr(self.bot, "set_local_mic_muted"):
                        self.bot.set_local_mic_muted(bool(data.get("muted")))
                elif t == "status":
                    await ws.send_str(json.dumps({"type": "status", "status": self._status()},
                                                 ensure_ascii=False))
        finally:
            self.clients.discard(ws)
        return ws

    async def h_index(self, request):
        index = STATIC / "index.html"
        if not index.exists():
            return web.Response(text="webui/static/index.html 不存在", status=500)
        return web.FileResponse(index, headers={"Cache-Control": "no-store"})

    # ---------------- 路由 ----------------
    def _app(self):
        app = web.Application()
        add = app.router.add_route
        add("GET", "/", self.h_index)
        add("GET", "/api/status", self.h_status)
        add("GET", "/api/models", self.h_models)
        add("POST", "/api/models/live2d", self.h_live2d_apply)
        add("GET", "/api/models/live2d/preview", self.h_live2d_preview)
        add("GET", "/api/memories", self.h_memories)
        add("POST", "/api/memories", self.h_memory_add)
        add("PATCH", "/api/memories/{mid}", self.h_memory_update)
        add("DELETE", "/api/memories/{mid}", self.h_memory_delete)
        add("POST", "/api/memory/reembed", self.h_reembed)
        add("POST", "/api/mic", self.h_mic)
        add("POST", "/api/say", self.h_say)
        add("GET", "/ws", self.h_ws)
        if STATIC.exists():
            app.router.add_static("/static/", STATIC)
        return app