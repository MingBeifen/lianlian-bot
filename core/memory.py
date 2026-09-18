"""恋恋的长期记忆：SQLite + 嵌入向量 + numpy 暴力检索。

为什么不上向量数据库：实测 1 万条记忆 1.65ms、内存 29MB，
等超过 50 万条再考虑 Chroma/Qdrant 也不迟。

    python memory.py --demo      # 塞一批样例记忆，跑查询看召回质量
    python memory.py --add "哥哥下周三有考试"
    python memory.py --query "我考试的事"
    python memory.py --list
    python memory.py --reset
"""
from __future__ import annotations

import argparse
import asyncio
import re
import json
import math
import os
import sqlite3
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = _ROOT / "data" / "lianlian_memory.db"
EMBED_MODEL = os.getenv("MEMORY_EMBED_MODEL", "nomic-embed-text")
BASE_URL = os.getenv("MEMORY_BASE_URL", "http://127.0.0.1:11434/v1")

# 打分权重：向量相似度为主，时间和重要性做微调
W_SIM, W_RECENCY, W_IMPORTANCE = 0.75, 0.15, 0.10
HALF_LIFE_DAYS = 30.0
DEDUP_SIM = 0.90          # 高于这个相似度就不新增，改为更新旧条目
RELATED_SIM = 0.70        # 参考线：低于这个基本算"不相关"


def _post_json(url: str, payload: dict, timeout: float = 60) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _fmt_time(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def _human_delta(ts: float) -> str:
    d = (time.time() - ts) / 86400.0
    if d < 1:
        return "今天"
    if d < 2:
        return "昨天"
    if d < 30:
        return f"{int(d)}天前"
    return f"{int(d/30)}个月前"


@dataclass
class Memory:
    id: int
    content: str
    kind: str
    importance: float
    created_at: float
    last_used_at: float
    score: float = 0.0
    sim: float = 0.0

    def __str__(self) -> str:
        return (f"[{_fmt_time(self.created_at)}/{_human_delta(self.created_at)}] "
                f"({self.kind}, 重要度{self.importance:.1f}, 相似{self.sim:.3f}, 总分{self.score:.3f}) "
                f"{self.content}")

    def to_dict(self) -> dict:
        return {"id": self.id, "content": self.content, "kind": self.kind,
                "importance": round(float(self.importance), 3),
                "created_at": self.created_at, "last_used_at": self.last_used_at,
                "age": _human_delta(self.created_at), "time": _fmt_time(self.created_at)}


class Embedder:
    """嵌入后端。

    实测（同一批中文记忆 + 7 个查询，看正确答案落在第几名）：
        nomic-embed-text (Ollama, 768d)   R@1 1/5   ← 中文召回很差，别用
        bge-small-zh-v1.5 (24M, 512d)     R@1 3/5
        bge-base-zh-v1.5  (102M, 768d)    R@1 6/7  R@3 7/7   ← 默认用这个
    所以默认在 CPU 上跑 bge-base-zh：单条查询 43ms，12 条记忆编码 0.24s，够快。

    想换回 Ollama 的模型：MEMORY_EMBED_BACKEND=ollama
    """

    QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

    def __init__(self, backend: str | None = None, model_name: str | None = None,
                 ollama_model: str = EMBED_MODEL, base_url: str = BASE_URL, max_len: int = 128):
        self.backend = (backend or os.getenv("MEMORY_EMBED_BACKEND", "bge")).lower()
        self.model_name = model_name or os.getenv("MEMORY_EMBED_MODEL_BGE", "BAAI/bge-base-zh-v1.5")
        self.ollama_model = ollama_model
        self.base_url = base_url.rstrip("/")
        self.max_len = max_len
        self._tok = None
        self._model = None

    def _lazy(self):
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer

            print(f"[memory] 加载嵌入模型 {self.model_name}（首次会下载，之后走缓存）...")
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).eval()
        return self._tok, self._model

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)

    def _encode_bge(self, texts: list[str], is_query: bool) -> np.ndarray:
        import torch

        tok, model = self._lazy()
        if is_query:
            texts = [self.QUERY_PREFIX + t for t in texts]
        with torch.no_grad():
            batch = tok(texts, padding=True, truncation=True,
                        max_length=self.max_len, return_tensors="pt")
            emb = model(**batch).last_hidden_state[:, 0]        # bge 用 CLS
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.numpy().astype(np.float32)

    def _encode_ollama(self, texts: list[str]) -> np.ndarray:
        try:
            data = _post_json(f"{self.base_url}/embeddings",
                              {"model": self.ollama_model, "input": texts})
            vecs = [d["embedding"] for d in data.get("data", [])]
        except Exception:
            root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
            vecs = [(_post_json(f"{root}/api/embeddings",
                                {"model": self.ollama_model, "prompt": t})["embedding"])
                    for t in texts]
        if len(vecs) != len(texts):
            raise RuntimeError(f"嵌入返回条数不对：要 {len(texts)}，拿到 {len(vecs)}")
        return np.asarray(vecs, dtype=np.float32)

    def encode(self, texts, is_query: bool = False) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = [t for t in texts]
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        if self.backend == "ollama":
            arr = self._encode_ollama(texts)
        else:
            arr = self._encode_bge(texts, is_query)
        return self._normalize(arr)


class MemoryStore:
    def __init__(self, db_path: str | Path | None = None, embedder: Embedder | None = None):
        self.db_path = Path(db_path or os.getenv("MEMORY_DB") or DEFAULT_DB)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or Embedder()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._cache = None            # (ids, 归一化后的矩阵)
        self._embed_calls = 0
        self._embed_seconds = 0.0

    # ---------------- 基础设施 ----------------
    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id           INTEGER PRIMARY KEY,
                created_at   REAL NOT NULL,
                updated_at   REAL NOT NULL,
                last_used_at REAL NOT NULL,
                kind         TEXT NOT NULL,
                content      TEXT NOT NULL,
                importance   REAL NOT NULL DEFAULT 0.5,
                embedding    BLOB NOT NULL,
                source       TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_kind ON memories(kind);
        """)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    # ---------------- 嵌入 ----------------
    def embed(self, texts, is_query: bool = False) -> np.ndarray:
        """把文本变成归一化向量（余弦相似度 = 点积）。"""
        t0 = time.perf_counter()
        arr = self.embedder.encode(texts, is_query=is_query)
        self._embed_calls += 1
        self._embed_seconds += time.perf_counter() - t0
        return arr

    def warmup(self) -> None:
        """提前把嵌入模型加载好（首次要下载几百 MB），别让第一句话卡在那儿。"""
        self.embed(["预热"])

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text], is_query=True)[0]

    # ---------------- 读 ----------------
    def _matrix(self):
        if self._cache is None:
            rows = self._conn.execute(
                "SELECT id, embedding FROM memories ORDER BY id").fetchall()
            if not rows:
                self._cache = (np.zeros(0, dtype=np.int64),
                               np.zeros((0, 768), dtype=np.float32))
            else:
                ids = np.array([r["id"] for r in rows], dtype=np.int64)
                mat = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
                self._cache = (ids, mat)
        return self._cache

    def recall(self, query: str, k: int = 8, kinds: list[str] | None = None,
               touch: bool = True) -> list[Memory]:
        ids, mat = self._matrix()
        if ids.size == 0:
            return []
        sims = mat @ self.embed_query(query)
        rows = self._conn.execute(
            "SELECT * FROM memories" + (" WHERE kind IN (%s)" % ",".join("?" * len(kinds)) if kinds else ""),
            kinds or [],
        ).fetchall()
        by_id = {r["id"]: r for r in rows}
        now = time.time()
        scored = []
        for i, mid in enumerate(ids):
            row = by_id.get(int(mid))
            if row is None:
                continue
            recency = math.exp(-(now - row["created_at"]) / 86400.0 / HALF_LIFE_DAYS)
            score = (W_SIM * float(sims[i]) + W_RECENCY * recency
                     + W_IMPORTANCE * float(row["importance"]))
            scored.append((score, float(sims[i]), row))
        scored.sort(key=lambda x: -x[0])
        out = []
        for score, sim, row in scored[:k]:
            out.append(Memory(row["id"], row["content"], row["kind"], row["importance"],
                              row["created_at"], row["last_used_at"], score, sim))
        if touch and out:
            self._conn.executemany("UPDATE memories SET last_used_at=? WHERE id=?",
                                   [(now, m.id) for m in out])
            self._conn.commit()
        return out

    def profile(self) -> list[Memory]:
        """档案层：永远注入，不参与检索。"""
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE kind='profile' ORDER BY importance DESC").fetchall()
        return [Memory(r["id"], r["content"], r["kind"], r["importance"],
                       r["created_at"], r["last_used_at"]) for r in rows]

    # ---------------- 写 ----------------
    def remember(self, content: str, kind: str = "fact", importance: float = 0.5,
                 source: str = "", created_at: float | None = None,
                 dedup: bool = True) -> tuple[str, int]:
        content = (content or "").strip()
        if not content:
            return "empty", 0
        vec = self.embed([content])[0]
        created_at = created_at or time.time()

        if dedup and len(self) > 0:
            ids, mat = self._matrix()
            sims = mat @ vec
            j = int(np.argmax(sims))
            if float(sims[j]) >= DEDUP_SIM:
                mid = int(ids[j])
                # 相似度这么高就是同一件事：保留信息更多的那条，
                # 否则"是只橘猫"会被后来的"养了只猫"覆盖掉（实测踩过）
                row = self._conn.execute("SELECT content FROM memories WHERE id=?",
                                         (mid,)).fetchone()
                old_content = row["content"] if row else ""
                keep = content if len(content) >= len(old_content) else old_content
                if keep != content:
                    print(f"[memory] 新写的比旧的短，保留旧内容：{old_content}")
                self._conn.execute(
                    "UPDATE memories SET content=?, kind=?, importance=?, embedding=?,"
                    " updated_at=?, source=? WHERE id=?",
                    (keep, kind, max(importance, 0.0), vec.tobytes(),
                     time.time(), source, mid))
                self._conn.commit()
                self._cache = None
                return "updated", mid

        cur = self._conn.execute(
            "INSERT INTO memories (created_at, updated_at, last_used_at, kind, content,"
            " importance, embedding, source) VALUES (?,?,?,?,?,?,?,?)",
            (created_at, time.time(), 0.0, kind, content, float(importance),
             vec.tobytes(), source))
        self._conn.commit()
        self._cache = None
        return "added", int(cur.lastrowid)

    def forget(self, mid: int) -> bool:
        cur = self._conn.execute("DELETE FROM memories WHERE id=?", (int(mid),))
        self._conn.commit()
        self._cache = None
        return cur.rowcount > 0

    # 网页控制台用的别名
    delete = forget

    def all(self) -> list[Memory]:
        rows = self._conn.execute("SELECT * FROM memories ORDER BY created_at DESC").fetchall()
        return [Memory(r["id"], r["content"], r["kind"], r["importance"],
                       r["created_at"], r["last_used_at"]) for r in rows]

    def get(self, mid: int) -> Memory | None:
        row = self._conn.execute("SELECT * FROM memories WHERE id=?", (int(mid),)).fetchone()
        if row is None:
            return None
        return Memory(row["id"], row["content"], row["kind"], row["importance"],
                      row["created_at"], row["last_used_at"])

    def set_importance(self, mid: int, importance: float) -> bool:
        """修改权重（0~1，检索打分里占 10%）。"""
        value = min(1.0, max(0.0, float(importance)))
        cur = self._conn.execute(
            "UPDATE memories SET importance=?, updated_at=? WHERE id=?",
            (value, time.time(), int(mid)))
        self._conn.commit()
        return cur.rowcount > 0

    def update(self, mid: int, content: str | None = None, kind: str | None = None,
               importance: float | None = None) -> bool:
        """改内容/类型/权重；内容变了会重新算嵌入向量。"""
        row = self._conn.execute("SELECT * FROM memories WHERE id=?", (int(mid),)).fetchone()
        if row is None:
            return False
        new_content = (content if content is not None else row["content"]).strip()
        if not new_content:
            return False
        new_kind = kind if kind in ("profile", "fact", "preference", "event") else row["kind"]
        new_imp = (min(1.0, max(0.0, float(importance)))
                   if importance is not None else float(row["importance"]))
        if new_content != row["content"]:
            vec = np.asarray(self.embed([new_content])[0], dtype=np.float32)
        else:
            vec = np.frombuffer(row["embedding"], dtype=np.float32)
        self._conn.execute(
            "UPDATE memories SET content=?, kind=?, importance=?, embedding=?, updated_at=? WHERE id=?",
            (new_content, new_kind, new_imp, vec.tobytes(), time.time(), int(mid)))
        self._conn.commit()
        self._cache = None
        return True

    def add(self, content: str, kind: str = "fact", importance: float = 0.5) -> tuple:
        """给控制台用的新增（走和自动写入一样的去重逻辑）。"""
        return self.remember(content, kind, importance, source="webui")

    def reembed_all(self, progress=None) -> int:
        """换嵌入模型后重算全部向量。返回条数。"""
        rows = self._conn.execute("SELECT id, content FROM memories").fetchall()
        done = 0
        for row in rows:
            vec = np.asarray(self.embed([row["content"]])[0], dtype=np.float32)
            self._conn.execute("UPDATE memories SET embedding=?, updated_at=? WHERE id=?",
                               (vec.tobytes(), time.time(), row["id"]))
            done += 1
            if progress and done % 20 == 0:
                progress(done, len(rows))
        self._conn.commit()
        self._cache = None
        return done

    # ---------------- 给提示词用 ----------------
    def format_block(self, query: str | None = None, k: int = 8,
                     profile: bool = True) -> str:
        """拼成能直接塞进 system 消息的记忆块。"""
        lines = []
        if profile:
            for m in self.profile():
                lines.append(f"- {m.content}")
        if query:
            for m in self.recall(query, k=k, kinds=["fact", "event", "preference"]):
                lines.append(f"- [{_human_delta(m.created_at)}] {m.content}")
        if not lines:
            return ""
        return ("【你本来就记得的事】\n"
                + "\n".join(lines) +
                "\n这些是你自己经历过、一直记得的事，哥哥不问就不用刻意提。回答时：\n"
                "- 像真人一样自然想起来，只挑和当前话题有关的那条说，不要逐条复述\n"
                "- 绝对不要提「记忆」「系统」「数据」「设定」「助手」「AI」「模型」这类词，更不要说自己是助手或程序，也不要解释你是怎么知道的\n"
                "- 细节以这里写的为准：不要自己加颜色、品种、地点、时间这些没写的细节\n"
                "- 这里没写的就是你不知道，别编")


# ---------------------------------------------------------------- 自测
DEMO_MEMORIES = [
    # (内容, 类型, 重要度, 几天前)
    ("哥哥叫 Chen SJ，在一台 RTX 5060 的笔记本上折腾 AI", "profile", 1.0, 60),
    ("哥哥喜欢喝美式咖啡，不加糖", "preference", 0.6, 45),
    ("哥哥下周三有一场很重要的考试", "fact", 0.9, 1),
    ("哥哥是学深度学习的，最近在学 Qt 做界面", "fact", 0.7, 10),
    ("哥哥养了一只叫豆豆的橘猫", "profile", 0.9, 30),
    ("9月初哥哥说考试考砸了，我嘴硬但悄悄安慰了他", "event", 0.8, 15),
    ("哥哥答应周末带我去看海", "event", 0.7, 5),
    ("哥哥不喜欢吃香菜", "preference", 0.5, 40),
    ("哥哥的生日是 3 月 12 日", "profile", 0.9, 20),
    ("哥哥最近在减肥，晚上不吃主食", "fact", 0.6, 3),
    ("哥哥说他最讨厌下雨天", "preference", 0.4, 25),
    ("上个星期和哥哥吵架了，因为他半夜还在敲代码", "event", 0.7, 7),
]

DEMO_QUERIES = [
    ("我考试的事你还记得吗", ["哥哥下周三有一场很重要的考试"]),
    ("我最近在忙什么", ["哥哥是学深度学习的，最近在学 Qt 做界面"]),
    ("我喜欢喝什么", ["哥哥喜欢喝美式咖啡，不加糖"]),
    ("周末有什么安排", ["哥哥答应周末带我去看海"]),
    ("我生日是什么时候", ["哥哥的生日是 3 月 12 日"]),
    ("今天中午吃什么好", []),          # 故意问一个记忆库里没有的
]


# ---------------------------------------------------------------- 自动写入
# 用 llama.cpp 的 GBNF 语法约束，输出必定符合这个格式（不可能解析失败）。
# 故意不用 JSON：GBNF 里写 JSON 的转义又长又容易错，而且行格式 token 更少、更快。
GBNF_EXTRACT = r"""
root ::= line*
line ::= kind "|" imp "|" text "\n"
kind ::= "FACT" | "EVENT" | "PREFERENCE" | "PROFILE"
imp  ::= "0.3" | "0.5" | "0.7" | "0.9"
text ::= [^\n|]+
"""

EXTRACT_PROMPT = """你是数据抽取程序，不是聊天角色。唯一任务：从一句话里提取事实条目。

铁律（违反即失败）：
1. 只提取那句话里【明确说出来】的事实。没说的一律不写。
2. 严禁编造、严禁推测、严禁补充细节、严禁写对话、严禁写情节、严禁任何情感描写。
3. 不要扮演任何角色，不要以任何人的口吻说话。
4. 主语统一写「哥哥」，不要用「我」「你」「他」。
5. 内容要【几乎照抄原话】，只允许把「我」换成「哥哥」。不要换词、不要改写、不要解释、不要删改数字或时间。
6. 最多输出 2 条。大多数时候 0 条或 1 条就够。宁可少写，绝不编造。
7. 没有明确事实就什么都不输出。

格式（每行一条，不要别的文字）：
类型|重要度|内容
  类型：FACT / EVENT / PREFERENCE / PROFILE
  重要度：0.3 / 0.5 / 0.7 / 0.9

例1  输入：我下周三要考驾照，有点紧张
     输出：FACT|0.7|哥哥下周三要考驾照

例2  输入：今天天气不错
     输出：（空）

例3  输入：我养了只猫叫豆豆
     输出：PROFILE|0.9|哥哥养了只猫叫豆豆

例4  输入：我叫陈思杰，你以后别叫错了
     输出：PROFILE|0.9|哥哥叫陈思杰

例5  输入：我最喜欢吃我妈做的红烧肉
     输出：PREFERENCE|0.5|哥哥最喜欢吃妈妈做的红烧肉

例6  输入：今天天气不错
     输出：（空）

例7  输入：嗯嗯好的
     输出：（空）

例8  输入：哈哈，你真有意思
     输出：（空）

再强调：天气、心情、寒暄、应答，一律不记。只有【关于哥哥本人、以后还用得上】的才记。"""


_JUNK_TOKEN = re.compile(r"<[^>]{0,40}>")          # <tool_call> 这类特殊 token
_PUNCT = "，。！？、,.!?「」”\"'：:；;【】[]…～~*# "


_FILLER = ("天气", "哈哈哈", "嗯嗯", "好的", "好吧", "在吗", "吃了吗", "早上好", "晚安",
           "谢谢", "哈哈", "没事", "随便", "还行", "不知道", "再见", "拜拜")


_Q_WORDS = ("什么", "怎么", "为何", "为什么", "哪儿", "哪里", "谁", "多少", "几点", "几号",
            "是不是", "有没有", "还记得", "知道吗")


def looks_like_question(text: str) -> bool:
    """问句里没有新事实 —— 拿它去抽记忆，模型只能编。

    实测："我养了什么宠物" → 抽出"哥哥喜欢养宠物狗"（凭空捏造）
    """
    t = (text or "").strip()
    if not t:
        return True
    if t.endswith(("吗", "呢", "么", "??", "？？")):
        return True
    return any(w in t for w in _Q_WORDS)


def is_filler(content: str) -> bool:
    """寒暄、应答、天气这类不该进记忆库的短内容。"""
    t = clean_item(content).replace("哥哥", "").strip(_PUNCT).strip()
    if len(t) <= 3:
        return True
    return len(t) <= 8 and any(f in t for f in _FILLER)


def clean_item(content: str) -> str:
    """去掉模型吐出来的特殊 token 和多余标点。"""
    t = _JUNK_TOKEN.sub("", content or "")
    t = t.strip().strip(_PUNCT).strip()
    return re.sub(r"\s+", " ", t)


def _key_chars(text: str) -> list[str]:
    """归一化：去掉"哥哥"（模型必须加的）和标点，只留有信息的字。"""
    t = (text or "").replace("哥哥", "").replace("恋恋", "")
    return [c for c in t if not c.isspace() and c not in _PUNCT]


def overlaps_enough(content: str, source: str, threshold: float = 0.30) -> bool:
    """抽出来的内容要有足够多的字来自原话 —— 掐掉模型自己编的部分。

    角色扮演微调过的模型很容易"入戏"编故事，光靠提示词压不住，这是最后一道闸。
    注意要归一化：模型会把"我"改写成"哥哥"，还会换措辞，直接比字符会误杀。
    """
    chars = _key_chars(content)
    if len(chars) < 3:            # 太短的内容一律不要（噪声大，也没价值）
        return False
    src = set(_key_chars(source))
    hit = sum(1 for c in chars if c in src)
    return hit / len(chars) >= threshold


class MemoryExtractor:
    """聊完一轮后自动抽记忆。用 GBNF 保证输出格式，不会解析失败。

    时机：放在"她正在播语音"的那几秒执行 —— 那时候 GPU 是空的，等于零延迟成本。
    """

    def __init__(self, cfg, store: "MemoryStore", min_chars: int = 4):
        from openai import AsyncOpenAI

        self.cfg = cfg
        self.store = store
        self.min_chars = min_chars
        self.lock = asyncio.Lock()          # 串行化，避免和下一轮抢 —— 也避免两条抽取交错
        self._grammar_ok = True
        self.client = AsyncOpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key or "none",
                                  timeout=cfg.llm_timeout)
        self.stats = {"runs": 0, "added": 0, "updated": 0, "failed": 0}

    @classmethod
    def _parse(cls, text: str, source: str = "", overlap: float = 0.45,
               max_items: int = 2) -> list[tuple[str, float, str]]:
        out = []
        for raw in (text or "").splitlines():
            parts = raw.strip().split("|", 2)
            if len(parts) != 3:
                continue
            kind, imp, content = parts[0].strip().upper(), parts[1].strip(), parts[2].strip()
            if kind not in ("FACT", "EVENT", "PREFERENCE", "PROFILE"):
                continue
            content = clean_item(content)
            if len(content) < 5 or len(content) > 60:
                continue
            # 在写对话/编情节的特征：带引号、冒号、换行
            if any(ch in content for ch in "「」”\"：:"):
                print(f"[memory] 丢弃（像是在编故事）：{content[:40]}")
                continue
            if is_filler(content):
                print(f"[memory] 丢弃（寒暄/废话）：{content[:30]}")
                continue
            if source and not overlaps_enough(content, source, overlap):
                print(f"[memory] 丢弃（原话里没有）：{content[:40]}")
                continue
            try:
                importance = min(1.0, max(0.0, float(imp)))
            except ValueError:
                importance = 0.5
            out.append((kind.lower(), importance, content))
            if len(out) >= max_items:
                break
        return out

    async def _ask(self, user_text: str) -> str:
        kwargs = dict(model=self.cfg.llm_model, temperature=0.0, top_p=1.0, max_tokens=110,
                      messages=[{"role": "system", "content": EXTRACT_PROMPT},
                                {"role": "user", "content": user_text}])
        # grammar / repeat_penalty 都是 llama.cpp 私有字段，必须走 extra_body
        extra = {"grammar": GBNF_EXTRACT, "repeat_penalty": 1.15}
        for attempt in (0, 1):
            try:
                if attempt == 0 and self._grammar_ok:
                    resp = await self.client.chat.completions.create(**kwargs, extra_body=extra)
                else:
                    resp = await self.client.chat.completions.create(**kwargs)
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                if attempt == 0 and self._grammar_ok:
                    print(f"[memory] 后端不支持 GBNF 语法约束（{type(exc).__name__}），退回普通模式")
                    self._grammar_ok = False
                    continue
                raise
        return ""

    async def write(self, user_text: str) -> int:
        """返回新增条数。任何异常都只打印，不影响对话。"""
        if not self.cfg.memory_auto_write or len((user_text or "").strip()) < self.min_chars:
            return 0
        if looks_like_question(user_text) or looks_like_question(user_text.replace("哥哥：", "")):
            return 0          # 问句不抽（否则模型只会编）
        async with self.lock:
            self.stats["runs"] += 1
            try:
                text = await self._ask(user_text)
            except Exception as exc:
                self.stats["failed"] += 1
                print(f"[memory] 抽取失败（{type(exc).__name__}: {str(exc)[:60]}）")
                return 0
            added = 0
            items = self._parse(text, source=user_text, overlap=self.cfg.memory_overlap)
            for kind, importance, content in items:
                try:
                    action, mid = await asyncio.to_thread(
                        self.store.remember, content, kind, importance, user_text)
                except Exception as exc:
                    print(f"[memory] 写入失败（{type(exc).__name__}: {str(exc)[:60]}）")
                    continue
                self.stats["added" if action == "added" else "updated"] += 1
                icon = "＋" if action == "added" else "↻"
                print(f"[memory] {icon} #{mid} [{kind} {importance:.1f}] {content}")
                added += action == "added"
            return added


def _demo(db_path: str | None) -> int:
    store = MemoryStore(db_path)
    e = store.embedder
    print(f"数据库：{store.db_path}   现有 {len(store)} 条记录")
    print(f"嵌入后端：{e.backend}  {e.model_name if e.backend != 'ollama' else e.ollama_model}")

    print("\n=== 写入样例记忆 ===")
    for content, kind, imp, days in DEMO_MEMORIES:
        action, mid = store.remember(content, kind, imp, source="demo",
                                     created_at=time.time() - days * 86400)
        print(f"  {action:8s} #{mid:<3d} [{kind}] {content}")

    print(f"\n=== 检索（{len(DEMO_QUERIES)} 个查询，各取 Top-3）===")
    hit1 = 0
    for query, expected in DEMO_QUERIES:
        t0 = time.perf_counter()
        results = store.recall(query, k=3)
        dt = (time.perf_counter() - t0) * 1000
        print(f"\n问：{query}   ({dt:.0f}ms)")
        for r in results:
            print(f"    {r}")
        if expected:
            top1 = results[0].content if results else ""
            ok = any(e in top1 or top1 in e for e in expected)
            hit1 += ok
            print(f"    期望第一条命中：{'✔' if ok else '✘ 期望 ' + expected[0]}")
        else:
            top_sim = results[0].sim if results else 0
            print(f"    （这题记忆库里没有，最高相似度 {top_sim:.3f}，"
                  f"{'正常' if top_sim < RELATED_SIM else '偏高，可能误召回'}）")

    print(f"\n=== 结论 ===")
    print(f"  带期望答案的 {sum(1 for _, e in DEMO_QUERIES if e)} 题里，Top-1 命中 {hit1} 题")
    print(f"  嵌入调用 {store._embed_calls} 次，共 {store._embed_seconds*1000:.0f}ms")
    store.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="恋恋的长期记忆（存储 + 检索）")
    ap.add_argument("--db", default=None, help="数据库路径")
    ap.add_argument("--demo", action="store_true", help="跑样例：写入+检索质量")
    ap.add_argument("--add", metavar="内容", help="手工加一条记忆")
    ap.add_argument("--kind", default="fact")
    ap.add_argument("--importance", type=float, default=0.5)
    ap.add_argument("--query", metavar="内容", help="检索")
    ap.add_argument("--list", action="store_true", help="列出全部")
    ap.add_argument("--reset", action="store_true", help="清空")
    args = ap.parse_args()

    if args.demo:
        return _demo(args.db)

    store = MemoryStore(args.db)
    if args.reset:
        store._conn.execute("DELETE FROM memories")
        store._conn.commit()
        store._cache = None
        print(f"已清空：{store.db_path}")
    elif args.add:
        action, mid = store.remember(args.add, args.kind, args.importance)
        print(f"{action} #{mid}: {args.add}")
    elif args.query:
        for m in store.recall(args.query, k=5):
            print(" ", m)
    elif args.list:
        for m in store.all():
            print(" ", m)
    else:
        print(f"{store.db_path}：{len(store)} 条记忆（用 --demo / --add / --query / --list）")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())