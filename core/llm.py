"""对话大脑：走 OpenAI 兼容接口，默认指向本机的 llama.cpp / 恋恋。

情绪是**独立判断**出来的，不塞进主回复里：

    主回复：完全自由生成（质量优先）
    情绪  ：一次极短的调用，只输出一个词，用 GBNF 约束（不可能出错）

为什么不做成「回复里带标签」：试过 `情绪|正文` 和 `情绪\t正文` 两种分隔符，
模型都会把分隔符当标点用（`angry|. |可恶|`），或者直接输出代码片段把正文搞烂。
分开做还多花 0.15 秒，但稳定得多。
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional

EMOTIONS = ("neutral", "happy", "shy", "angry", "sad", "surprised")

# 只允许输出一个情绪词 —— 短到不可能出错
EMO_GRAMMAR = r"""
root ::= "neutral" | "happy" | "shy" | "angry" | "sad" | "surprised"
"""

EMO_PROMPT = """判断下面这句话是什么情绪。只输出一个词，不要别的。

neutral（平静/普通） happy（开心/得意） shy（害羞/嘴硬）
angry（生气/不爽） sad（低落/委屈） surprised（惊讶）

判断的是**说话人的情绪**，不是内容。拿不准就用 neutral。"""


ACTIONS = ("none", "nod", "shake", "wave", "think", "shy_hide", "laugh")

ACT_PROMPT = """判断下面这句话最自然配什么动作。只输出一个词，不要别的。

none（普通陈述、没有明显动作） nod（同意/鼓励/答应） shake（摇头/否认/拒绝）
wave（打招呼/再见/挥手致意） think（思考/犹豫/回忆） shy_hide（害羞/嘴硬/掩饰）
laugh（大笑/得意/兴奋）

规则：
- 问候或道别（你好/早上好/晚安/再见/拜拜）-> wave
- 只是陈述事实、没有明显动作 -> none
例：你好呀 -> wave；嗯嗯你说得对 -> nod；不行！ -> shake；
    让我想想 -> think；哼，才不是 -> shy_hide；哈哈哈哈 -> laugh；
    今天天气不错 -> none
拿不准就输出 none。"""

ACT_GRAMMAR = r"""
root ::= "none" | "nod" | "shake" | "wave" | "think" | "shy_hide" | "laugh"
"""


class ChatLLM:
    def __init__(self, cfg):
        from openai import AsyncOpenAI

        self.cfg = cfg
        self.client = AsyncOpenAI(
            base_url=cfg.llm_base_url,
            api_key=cfg.llm_api_key or "none",
            timeout=cfg.llm_timeout,
        )
        self.history = []
        self.memory = None            # 由 voice_chat 注入 MemoryStore
        self.last_memories = ""
        self.last_emotion = "neutral"  # 她这轮的情绪
        self.last_action = "none"      # 她这轮的伴随动作
        if cfg.llm_system_prompt:
            self.history.append({"role": "system", "content": cfg.llm_system_prompt})
        self._extra = {"keep_alive": cfg.llm_keep_alive} if cfg.llm_keep_alive else {}
        self._use_extra = bool(self._extra)
        self._grammar_ok = True

    # ---------------------------------------------------------------- 底层
    def _kwargs(self, messages: list, stream: bool, max_tokens: Optional[int] = None) -> dict:
        kw = {
            "model": self.cfg.llm_model,
            "messages": messages,
            "stream": stream,
            "max_tokens": max_tokens or self.cfg.llm_max_tokens,
        }
        if self.cfg.llm_temperature >= 0:
            kw["temperature"] = self.cfg.llm_temperature
        if self.cfg.llm_top_p >= 0:
            kw["top_p"] = self.cfg.llm_top_p
        return kw

    async def _create(self, messages: list, stream: bool, extra: Optional[dict] = None,
                      max_tokens: Optional[int] = None):
        body = dict(self._extra)
        if extra:
            body.update(extra)
        if body and self._use_extra:
            try:
                return await self.client.chat.completions.create(
                    **self._kwargs(messages, stream, max_tokens), extra_body=body
                )
            except Exception as exc:
                print(f"[llm] 服务端不接受附加参数（{type(exc).__name__}），退化成普通模式")
                self._use_extra = False
        return await self.client.chat.completions.create(
            **self._kwargs(messages, stream, max_tokens))

    async def warmup(self) -> None:
        try:
            await self._create([{"role": "user", "content": "在吗"}], stream=False, max_tokens=4)
            print(f"[llm] {self.cfg.llm_model} 已就绪")
        except Exception as exc:
            print(f"[llm] 预热失败（{type(exc).__name__}: {exc}）")

    # ---------------------------------------------------------------- 记忆
    def _memory_block(self, user_text: str) -> str:
        if self.memory is None:
            return ""
        try:
            block = self.memory.format_block(user_text, k=self.cfg.memory_top_k)
        except Exception as exc:
            print(f"[memory] 检索失败（{type(exc).__name__}: {exc}），这轮不带记忆")
            return ""
        if block and self.cfg.debug_memory:
            items = [ln[2:] for ln in block.splitlines() if ln.startswith("- ")]
            newest = items[-1] if items else ""
            print(f"[memory] 注入 {len(items)} 条（{items[0][:24] if items else ''}… 最近一条：{newest[:28]}）")
            if int(self.cfg.debug_memory) >= 2:
                for it in items:
                    print(f"         - {it}")
        return block

    # ---------------------------------------------------------------- 情绪
    async def classify_emotion(self, text: str) -> str:
        """判断她刚说的这句话是什么情绪。失败就返回 neutral，绝不影响主流程。"""
        if not self.cfg.emotion_enable or not (text or "").strip():
            return "neutral"
        extra = {"grammar": EMO_GRAMMAR, "temperature": 0.0} if self._grammar_ok else {}
        try:
            resp = await self._create(
                [{"role": "system", "content": EMO_PROMPT},
                 {"role": "user", "content": text[:200]}],
                stream=False, extra=extra, max_tokens=6)
            word = (resp.choices[0].message.content or "").strip().lower()
        except Exception:
            return "neutral"
        for e in EMOTIONS:            # 兜底：从返回里找有没有已知的情绪词
            if e in word:
                return e
        return "neutral"

    async def classify_action(self, text: str) -> str:
        """判断她这句话配什么动作；失败返回 none，绝不影响主流程。"""
        if not getattr(self.cfg, "llm_action", True) or not (text or "").strip():
            return "none"
        extra = {"grammar": ACT_GRAMMAR, "temperature": 0.0} if self._grammar_ok else {}
        try:
            resp = await self._create(
                [{"role": "system", "content": ACT_PROMPT},
                 {"role": "user", "content": text[:200]}],
                stream=False, extra=extra, max_tokens=6)
            word = (resp.choices[0].message.content or "").strip().lower()
        except Exception:
            return "none"
        for act in ACTIONS:
            if act in word:
                return act
        return "none"

    # ---------------------------------------------------------------- 主流程
    async def stream_reply(self, user_text: str, hint: str = "") -> AsyncIterator[str]:
        """流式回复。hint 是上层注入的额外提示（哥哥的情绪 / 她自己的心情）。"""
        self._trim()
        messages = list(self.history)
        block = await asyncio.to_thread(self._memory_block, user_text)
        self.last_memories = block
        if hint:
            # 放在历史之后：前缀不变，KV 缓存照样命中
            messages.append({"role": "system", "content": hint})
        if block:
            messages.append({"role": "system", "content": block})
        messages.append({"role": "user", "content": user_text})

        stream = await self._create(messages, stream=True)
        parts: list[str] = []
        async for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            piece = getattr(choices[0].delta, "content", None)
            if piece:
                parts.append(piece)
                yield piece
        reply = "".join(parts).strip()
        if reply:
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": reply})

    def _trim(self) -> None:
        limit = max(2, self.cfg.llm_history_turns * 2)
        if len(self.history) > limit + 1:
            head = self.history[:1] if self.history[0]["role"] == "system" else []
            self.history = head + self.history[-limit:]

    def reset(self) -> None:
        self.history = self.history[:1] if self.history and self.history[0]["role"] == "system" else []