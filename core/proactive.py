r"""恋恋的主动性：让她在该开口的时候自己说话。

设计原则（都是踩出来的）：
  1. **宁可不说，也不说废话** —— LLM 可以输出 <SILENT> 主动放弃
  2. **你不在时别说** —— 用麦克风音量判断房间里有没有人
  3. **说了没人理就退避** —— 否则她会对着空气自言自语停不下来
  4. **有理由才说** —— 优先级：记忆提醒 > 时间情境 > 空闲搭话

四层闸门（便宜的先查，LLM 最后）：
    硬规则 → 频率控制 → 环境判断 → LLM 决定说不说

用法：
    brain = ProactiveBrain(cfg, llm_client, memory)
    text = await brain.maybe_speak(BotContext(idle_sec=..., busy=..., noise_rms=...))
    if text: await speak(text)
"""
from __future__ import annotations

import collections
import re
import time
from dataclasses import dataclass, field
from typing import Optional

SILENT = "<SILENT>"
# 实测模型会拼错：<silnet> <silient> <silently> <sil> …… 一律当沉默处理
_SILENT_RE = re.compile(r"^\s*<\s*s?i?l", re.I)
# 她"动不了手"：这些都在声称做了物理动作（实测提示词压不住，只能代码兜底）
_FAKE_ACTION = ("我刚冲", "我刚泡", "我刚煮", "我刚做", "我刚买", "我刚拿", "我刚倒",
                "我刚收拾", "我刚准备", "我刚看", "我刚听", "我帮你", "我给你", "我替你",
                "我已经帮你", "我已经给你", "我给你准备", "我这就去", "我先去给你",
                "我买了", "我给买了", "我给你带", "我带了", "我拿来了", "我留了")

PROMPT = """（这是你脑子里的念头，**不是**哥哥说的话）

现在是 {now}，你们已经 {idle} 分钟没说话了。
{occasion}{recent}{said}{memories}
从下面挑一个角度（**每次换一个，别重复上次的角度**）：
  A. 问他在做什么、吃了没、累不累
  B. 提一件你记得的他的事（他提过的计划、他喜欢的东西）
  C. 说说你现在的想法或心情
  D. 顺着刚才的话题往下聊一句

主动跟哥哥说一句话。要求：
- 用恋恋的语气，不超过 25 个字，口语，能被直接念出来
- 像突然想起来一样自然，不要解释、不要引号、不要括号动作
- 直接输出这句话本身，不要任何前缀

**铁律：**
1. **你动不了手**。不许说你做了什么（泡茶、做饭、出去走、帮你拿东西）、答应过什么、
   有什么东西。你只能陪他说话。说漏嘴会被他发现的。
2. 不许编造关于他的事：没发生过的事、他沒说过的话。
3. 可以说你的**想法、心情、记得的事**、问他在干嘛 —— 这些不会露馅。
4. **换个角度**，别接着说同一件事（上面「上次你说」那句的话题别再提）。
"""

# 先判断"有没有想说的"（只有在没有明确时机时才用，避免没话找话）
JUDGE = """你在判断"现在有没有想主动跟哥哥说的话"。

现在是 {now}，你们已经 {idle} 分钟没说话。
{recent}{memories}
判断标准：
- YES：你确实想起了什么想跟他说的（他提过的事、你在意的事、他最近的状态）
- NO：只是"好像该说点什么"但没实际内容

只输出 YES 或 NO。"""


def _too_similar(a: str, b: str, thresh: float = 0.55) -> bool:
    """两条话太像就算重复（字符集合的 Jaccard 相似度）。"""
    if not a or not b:
        return False
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb)) >= thresh


def _daypart(now: float) -> str:
    h = time.localtime(now).tm_hour
    if 5 <= h < 10:
        return "早上"
    if 10 <= h < 12:
        return "上午"
    if 12 <= h < 14:
        return "中午"
    if 14 <= h < 18:
        return "下午"
    if 18 <= h < 23:
        return "晚上"
    return "深夜"


@dataclass
class BotContext:
    """voice_chat 每次询问时提供的现场情况。"""
    idle_sec: float = 0.0          # 距上次对话结束多久（秒）
    busy: bool = False             # 是否正在对话（有 turn 在跑）
    noise_rms: float = 0.0         # 最近一秒的环境音量（判断人在不在）
    now: float = field(default_factory=time.time)


class ProactiveBrain:
    def __init__(self, cfg, client=None, model: str = "", memory=None, history=None):
        self.cfg = cfg
        self.client = client           # openai 的 AsyncOpenAI（复用对话那个）
        self.model = model
        self.memory = memory
        self.history = history         # 可调用对象，返回最近的对话消息列表
        # ---- 防骚扰状态 ----
        self.last_proactive = 0.0
        self.unanswered = 0            # 主动说了但哥哥没理的次数
        self.hour_key, self.hour_count = -1, 0
        self.day_key, self.day_count = "", 0
        self.muted_until = 0.0         # 连续没理 → 禁言到什么时候
        self._seen: dict[str, str] = {}   # 当天只触发一次的时机
        self._said: collections.deque = collections.deque(maxlen=1)   # 最近主动说过的话
        self.stats = {"asked": 0, "spoke": 0, "silent": 0, "blocked": 0}

    # ------------------------------------------------------------------ 对外
    def note_user_spoke(self) -> None:
        """哥哥说话了 → 她在，退避清零。"""
        if self.unanswered:
            self.unanswered = 0

    def note_proactive_spoke(self, text: str) -> None:
        now = time.time()
        self.last_proactive = now
        self._said.append(text)
        self.unanswered += 1
        self._bump_counters(now)
        # 连续 N 次没被理 → 今天先闭嘴
        limit = max(1, self.cfg.proactive_unanswered_limit)
        if self.unanswered >= limit:
            self.muted_until = self._end_of_day(now)
            print(f"[主动] 连续 {self.unanswered} 次没被理，今天不再主动开口")

    async def maybe_speak(self, ctx: BotContext) -> Optional[str]:
        """返回要说的话；返回 None 表示这次不说话。"""
        ok, why = self._gates(ctx)
        if not ok:
            self.stats["blocked"] += 1
            return None
        self.stats["asked"] += 1
        occasion = self._occasion(ctx)      # 有明确时机（问候/饭点/催睡）就直接说
        if not occasion and not await self._judge(ctx):
            self.stats["silent"] += 1
            print("[主动] 她想了想，没什么要说的")
            return None
        text = await self._generate(ctx, occasion)
        if not text:
            self.stats["silent"] += 1
            return None
        self.stats["spoke"] += 1
        return text

    def describe(self) -> str:
        s = self.stats
        return (f"询问 {s['asked']} 次 / 开口 {s['spoke']} 次 / 她主动放弃 {s['silent']} 次 / "
                f"被规则拦下 {s['blocked']} 次")

    # ------------------------------------------------------------ 闸门（规则）
    def _gates(self, ctx: BotContext) -> tuple[bool, str]:
        cfg, now = self.cfg, ctx.now
        if not cfg.proactive_enable:
            return False, "总开关关着"
        if ctx.busy:
            return False, "正在对话"
        if now < self.muted_until:
            return False, "连续没理，今天静默"
        if ctx.idle_sec < cfg.proactive_idle_sec:
            return False, f"才安静 {ctx.idle_sec:.0f}s"
        # 退避：没被理的次数越多，等得越久
        gap = cfg.proactive_min_gap * (cfg.proactive_backoff_base ** self.unanswered)
        if now - self.last_proactive < gap:
            return False, f"退避中（{gap/60:.0f} 分钟内不重复）"
        if self._count_today(now) >= cfg.proactive_max_per_day:
            return False, "今天说得够多了"
        if self._count_this_hour(now) >= cfg.proactive_max_per_hour:
            return False, "这一小时说得够多了"
        if self._is_quiet_hour(now):
            return False, "静默时段"
        if cfg.proactive_require_sound and ctx.noise_rms < cfg.proactive_noise_floor:
            return False, "房间太安静，你可能不在"
        return True, ""

    def _is_quiet_hour(self, now: float) -> bool:
        """配置形如 "0-7"（0 点到 7 点不主动）"""
        spec = (self.cfg.proactive_quiet_hours or "").strip()
        if not spec:
            return False
        h = time.localtime(now).tm_hour
        for part in spec.split(","):
            part = part.strip()
            if "-" in part:
                try:
                    a, b = (int(x) for x in part.split("-", 1))
                except ValueError:
                    continue
                if a <= h < b if a <= b else (h >= a or h < b):
                    return True
            elif part.isdigit() and h == int(part):
                return True
        return False

    def _bump_counters(self, now: float) -> None:
        hk = int(now // 3600)
        if hk != self.hour_key:
            self.hour_key, self.hour_count = hk, 0
        self.hour_count += 1
        dk = time.strftime("%Y%m%d", time.localtime(now))
        if dk != self.day_key:
            self.day_key, self.day_count = dk, 0
        self.day_count += 1

    def _count_this_hour(self, now: float) -> int:
        return self.hour_count if int(now // 3600) == self.hour_key else 0

    def _count_today(self, now: float) -> int:
        return self.day_count if time.strftime("%Y%m%d", time.localtime(now)) == self.day_key else 0

    @staticmethod
    def _end_of_day(now: float) -> float:
        lt = time.localtime(now)
        return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 23, 59, 59, 0, 0, -1))

    # ------------------------------------------------------ 生成（含"可沉默"）
    def _occasion(self, ctx: BotContext) -> str:
        """当天只触发一次的时机（问候/催睡/饭点）。"""
        now = ctx.now
        h = time.localtime(now).tm_hour
        today = time.strftime("%Y%m%d", time.localtime(now))
        if 5 <= h < 10 and self._seen.get("morning") != today:
            self._seen["morning"] = today
            return "这是你们今天第一次说话，可以打个招呼。\n"
        if h >= 23 and self._seen.get("night") != today:
            self._seen["night"] = today
            return "已经很晚了，可以催他睡觉。\n"
        if 11 <= h < 13 and self._seen.get("lunch") != today:
            self._seen["lunch"] = today
            return "到饭点了，可以提醒他吃饭。\n"
        if 17 <= h < 19 and self._seen.get("dinner") != today:
            self._seen["dinner"] = today
            return "到晚饭点了，可以提醒他吃饭。\n"
        return ""

    def _recent(self, n: int = 6) -> str:
        """最近几轮对话 —— 主动开口最自然的素材来源（比凭空想好得多）。"""
        if not self.history:
            return ""
        try:
            msgs = [m for m in self.history() if m.get("role") in ("user", "assistant")][-n:]
        except Exception:
            return ""
        if not msgs:
            return ""
        lines = [f"{'哥哥' if m['role'] == 'user' else '你'}：{m['content'][:50]}" for m in msgs]
        return "你们刚才聊的：\n" + "\n".join(lines) + "\n"

    def _said_block(self) -> str:
        if not self._said:
            return ""
        return f'上次你说的是：「{self._said[-1]}」（这次别再说这个）\n'

    def _memory_block(self) -> str:
        """把最近的记忆交给她，让她自己决定有没有该提的事。"""
        if self.memory is None:
            return ""
        try:
            items = self.memory.all()[:12]
        except Exception:
            return ""
        if not items:
            return ""
        lines = [f"- [{time.strftime('%m月%d日', time.localtime(m.created_at))}] {m.content}"
                 for m in items]
        return ("你记得这些事：\n" + "\n".join(lines) +
                "\n（只有确实该提起某件时才提，别硬凑）\n")

    async def _judge(self, ctx: BotContext) -> bool:
        """没时机时先问一句"有没有想说的"，避免没话找话。"""
        if self.client is None:
            return False
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": JUDGE.format(
                    now=time.strftime("%H:%M", time.localtime(ctx.now)),
                    idle=int(ctx.idle_sec // 60),
                    recent=self._recent(),
                    memories=self._memory_block())}],
                temperature=0.3, max_tokens=6,
            )
            ans = (resp.choices[0].message.content or "").strip().upper()
            return ans.startswith("YES") or ans.startswith("Y")
        except Exception:
            return False

    async def _generate(self, ctx: BotContext, occasion: str = "") -> Optional[str]:
        if self.client is None:
            return None
        prompt = PROMPT.format(
            now=time.strftime("%Y-%m-%d %H:%M", time.localtime(ctx.now)),
            idle=int(ctx.idle_sec // 60),
            occasion=occasion,
            recent=self._recent(),
            said=self._said_block(),
            memories=self._memory_block(),
        )
        messages = []
        if self.cfg.llm_system_prompt:
            messages.append({"role": "system", "content": self.cfg.llm_system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = await self.client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=self.cfg.proactive_temperature, max_tokens=60,
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            print(f"[主动] 生成失败：{type(exc).__name__}: {str(exc)[:80]}")
            return None
        text = self._clean(text)
        bad_reason = ""
        if text and text.startswith("FAKE:"):
            bad_reason = "编造了动作"
            text = text[5:]
        elif text and self._said and _too_similar(text, self._said[-1]):
            bad_reason = "和上次太像"
        if bad_reason:
            print(f"[主动] {bad_reason}，重来一次：{text}")
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model, messages=messages + [
                        {"role": "assistant", "content": text},
                        {"role": "user", "content": "不行。" + (
                            "你动不了手，别说自己做了什么。" if bad_reason == "编造了动作"
                            else "和上次说的太像了，换个完全不同的角度。") +
                         "重新说一句。"},
                    ], temperature=self.cfg.proactive_temperature, max_tokens=60)
                text = self._clean((resp.choices[0].message.content or "").strip())
            except Exception:
                return None
        if text and text.startswith("FAKE:"):
            return None
        if text and self._said and _too_similar(text, self._said[-1]):
            print(f"[主动] 重试后还是重复，这次就不说了")
            return None
        return text

    @staticmethod
    def _clean(text: str) -> Optional[str]:
        if not text or _SILENT_RE.match(text) or "<" in text[:12]:
            return None
        text = re.sub(r'^["”「」\s]+|["”「」\s]+$', "", text)
        text = re.sub(r"^恋恋[：:]\s*", "", text)
        text = re.sub(r"^[A-Da-d][.、)）]\s*", "", text)     # 角度清单的编号，别抄进来
        text = text.split("\n")[0].strip()
        if len(text) < 2 or len(text) > 60:
            return None
        if not re.search(r"[0-9A-Za-z\u4e00-\u9fff]", text):
            return None
        for bad in _FAKE_ACTION:
            if bad in text:
                return "FAKE:" + text      # 交给上层重试
        return text