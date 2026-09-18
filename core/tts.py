"""语音合成：Kokoro（本地、免费、24k）或 OpenAI 兼容 TTS。"""
from __future__ import annotations

import asyncio
import io
import json
import re

from pathlib import Path

import numpy as np

_BRACKET = re.compile(r"[（(\[【][^）)\]】]{0,10}[）)\]】]")
_MD = re.compile(r"[*_`#>\[\]{}~|]+")
# 同一个标点连着 3 个以上就压成一个（模型退化时会吐一长串 ～～～～～）
_REPEAT_PUNCT = re.compile(r"([^\w\s])\1{2,}")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]")
_WS = re.compile(r"\s+")
KOKORO_SR = 24000


_HAS_CONTENT = re.compile(r"[0-9A-Za-z\u4e00-\u9fff]")


def clean_tts_text(text: str, strip_brackets: bool = True) -> str:
    """念出来会很奇怪的东西（markdown、表情、括号旁白）先去掉。

    最后还会检查有没有"实际内容"：纯标点的文本（如 "……"、"。"，大模型偶尔会只回这个）
    直接当空处理 —— 否则 GPT-SoVITS 会返回 400「请输入有效文本」把整轮搞崩。
    """
    text = text or ""
    if strip_brackets:
        # 恋恋爱写（慢半拍）（敲门）这种旁白，朗读出来很怪
        text = _BRACKET.sub(" ", text)
    text = _MD.sub(" ", text)
    text = _REPEAT_PUNCT.sub(r"\1", text)
    text = _EMOJI.sub("", text)
    text = _WS.sub(" ", text).strip()
    return text if _HAS_CONTENT.search(text) else ""


def trim_silence(audio: np.ndarray, sr: int, thresh: float = 0.004, margin_ms: int = 20) -> np.ndarray:
    """掐掉首尾静音，出声更快。"""
    if audio.size == 0:
        return audio
    loud = np.flatnonzero(np.abs(audio) > thresh)
    if loud.size == 0:
        return audio
    m = int(sr * margin_ms / 1000)
    out = audio[max(0, loud[0] - m) : min(audio.size, loud[-1] + m + 1)]
    return out if out.size > sr // 10 else audio


class KokoroTTS:
    def __init__(self, cfg):
        from kokoro import KPipeline

        self.cfg = cfg
        self.sample_rate = KOKORO_SR
        self.voice = cfg.tts_voice
        self.speed = cfg.tts_speed
        print(f"[tts] 加载 Kokoro（{cfg.tts_device or 'auto'}）音色 {self.voice} ...")
        # ---- 1) 先确定用哪个模型：repo id 还是本地目录 ----
        kw = {"lang_code": cfg.tts_lang, "device": cfg.tts_device or None}
        repo = (cfg.tts_repo or "").strip()
        local_dir = None
        if repo and Path(repo).is_dir():
            # 支持指向本地目录：把 kokoro 里"从 hub 下"换成"从本地取"
            import kokoro.model as _km
            import kokoro.pipeline as _kp

            local_dir = Path(repo)

            def _from_local(repo_id, filename, **kw2):
                return str(local_dir / filename)

            _km.hf_hub_download = _from_local
            if hasattr(_kp, "hf_hub_download"):
                _kp.hf_hub_download = _from_local
            # kokoro 内部有张 repo_id -> 模型文件名 的硬编码表，repo_id 必须写真名
            kw["repo_id"] = ("hexgrad/Kokoro-82M-v1.1-zh"
                             if (local_dir / "kokoro-v1_1-zh.pth").exists()
                             else "hexgrad/Kokoro-82M")
            print(f"[tts] 用本地模型目录：{local_dir}")
        elif repo:
            kw["repo_id"] = repo

        # ---- 2) 中文模型：读词表 + 装英文转换器 ----
        # 两个坑：
        #   a) 不传 en_callable，中文前端会把整段英文换成 ❓ 丢掉
        #   b) 就算传了，词表里没有 '-' 和小写 'g'，tokenizer 会把这些字符静默丢弃
        if (cfg.tts_lang or "").lower() == "z":
            vocab = self._read_vocab(kw.get("repo_id"), local_dir)
            self.en_mode = (cfg.tts_en_mode or "ipa").lower()
            kw["en_callable"] = make_english_callable(vocab, self.en_mode)
            if self.en_mode == "zh":
                print("[tts] 英文按中文谐音念（诶艾/欧剋 这种）")

        self.pipeline = KPipeline(**kw)
        self._synth_sync("你好呀。")   # 预热：第一次调用要建图，特别慢
        print("[tts] 就绪")

    @staticmethod
    def _read_vocab(repo_id, local_dir) -> set:
        """读模型词表，用来过滤英文里模型不认识的字符（不读也能跑，只是个别字母会丢）。"""
        try:
            import json as _json

            if local_dir is not None:
                path = Path(local_dir) / "config.json"
            else:
                from huggingface_hub import hf_hub_download

                path = Path(hf_hub_download(repo_id=repo_id, filename="config.json"))
            vocab = set(_json.loads(path.read_text(encoding="utf-8")).get("vocab", {}).keys())
            print(f"[tts] 读到词表 {len(vocab)} 个符号")
            return vocab
        except Exception as exc:
            print(f"[tts] 读不到词表（{type(exc).__name__}: {str(exc)[:60]}），英文个别字母可能被丢")
            return set()

    def _synth_sync(self, text: str, speed: float | None = None) -> np.ndarray:
        chunks = []
        for result in self.pipeline(text, voice=self.voice, speed=speed or self.speed):
            audio = getattr(result, "audio", None)
            if audio is None and isinstance(result, (tuple, list)):
                audio = result[-1]
            if audio is None:
                continue
            chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        out = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        return trim_silence(out, self.sample_rate) if self.cfg.tts_trim_silence else out

    async def synth(self, text: str, speed: float | None = None) -> np.ndarray:
        text = clean_tts_text(text, self.cfg.tts_strip_brackets)
        if getattr(self, "en_mode", "ipa") == "zh":
            text = zhify_english(text)
        if not text:
            return np.zeros(0, dtype=np.float32)
        return await asyncio.to_thread(self._synth_sync, text, speed)


class ApiTTS:
    def __init__(self, cfg):
        from openai import AsyncOpenAI

        self.cfg = cfg
        self.sample_rate = 24000
        self.client = AsyncOpenAI(base_url=cfg.tts_api_base, api_key=cfg.tts_api_key or "none")

    async def synth(self, text: str, speed: float | None = None) -> np.ndarray:
        text = clean_tts_text(text, self.cfg.tts_strip_brackets)
        if not text:
            return np.zeros(0, dtype=np.float32)
        resp = await self.client.audio.speech.create(
            model=self.cfg.tts_api_model,
            voice=self.cfg.tts_api_voice,
            input=text,
            response_format="pcm",
        )
        data = getattr(resp, "content", None)
        if not isinstance(data, (bytes, bytearray)):
            data = await resp.aread()
        pcm = np.frombuffer(bytes(data), dtype="<i2").astype(np.float32) / 32768.0
        return trim_silence(pcm, self.sample_rate)


# 英文词怎么念。中文模型的词表里只有注音 + IPA + a-z（缺 g 和连字符），
# 所以"直接喂 ASCII 字母"会被按拼音读（a→啊），而且字母之间会黏住。
_EN_IPA = {"a":"eɪ","b":"biː","c":"siː","d":"diː","e":"iː","f":"ɛf","g":"dʒiː","h":"eɪtʃ","i":"aɪ",
           "j":"dʒeɪ","k":"keɪ","l":"ɛl","m":"ɛm","n":"ɛn","o":"oʊ","p":"piː","q":"kjuː","r":"ɑɹ",
           "s":"ɛs","t":"tiː","u":"juː","v":"viː","w":"dʌbəljuː","x":"ɛks","y":"waɪ","z":"ziː"}
_EN_ZH = {"a":"诶","b":"比","c":"西","d":"迪","e":"伊","f":"艾弗","g":"吉","h":"艾尺","i":"艾","j":"杰",
          "k":"开","l":"艾勒","m":"艾姆","n":"恩","o":"欧","p":"屁","q":"丘","r":"阿尔","s":"艾丝",
          "t":"提","u":"优","v":"威","w":"达不溜","x":"艾克斯","y":"歪","z":"贼德"}
_EN_WORDS_ZH = {"OK":"欧剋","APP":"艾普","HELLO":"哈喽","HI":"嗨","BYE":"拜","WIFI":"歪fai",
                "GITHUB":"吉特哈布","EMAIL":"伊妹儿","GOOD":"古德"}


def zhify_english(text: str) -> str:
    """把英文换成中文谐音（在文本层做，不能走 en_callable —— 词表里没有那些汉字）。

    中文模型念"诶艾"比念"AI"自然得多，这也是中文母语者的实际读法。
    """
    def repl(m):
        w = m.group(0)
        if w.upper() in _EN_WORDS_ZH:
            return _EN_WORDS_ZH[w.upper()]
        return "".join(_EN_ZH.get(c, "") for c in w.lower())
    return re.sub(r"[A-Za-z]+", repl, text)


def make_english_callable(vocab, mode: str = "ipa"):
    """造一个把英文片段转成「模型念得出来」的函数。"""
    def conv(en: str) -> str:
        s = (en or "").strip()
        if not s:
            return s
        spell = len(s) <= 3 or s.isupper()      # 缩写按字母逐个念，长词整体念
        if mode == "ipa":
            parts = [_EN_IPA.get(c, c) for c in s.lower()] if spell else [s.lower()]
            out = [p for p in parts if all(ch in vocab for ch in p)] if vocab else parts
            return " ".join(out)
        # ascii 模式：只保留词表里有的字符（缺 g 用 IPA 的 ɡ 顶替）
        out = []
        for ch in s.lower():
            if not vocab or ch in vocab:
                out.append(ch)
            elif ch == "g":
                out.append("ɡ")
        return " ".join(out) if spell else "".join(out)
    return conv


def _decode_audio_bytes(data: bytes, target_rate: int) -> np.ndarray:
    """把服务返回的音频字节（wav / mp3）解成 float32 单声道 PCM。

    soundfile 自带 libsndfile 1.2+，wav 和 mp3 都能读，不用装 ffmpeg。
    """
    import soundfile as sf

    a, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != target_rate:
        n = int(round(len(a) * target_rate / sr))
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)
    return a.astype(np.float32)


class EdgeTTS:
    """微软 Edge 在线语音：音色自然、零显存、不用下模型，但必须联网。

    中文音色：zh-CN-XiaoxiaoNeural(女) / zh-CN-XiaoyiNeural(女)
             / zh-CN-YunxiNeural(男) / zh-CN-liaoning-XiaobeiNeural(东北话)
    """

    def __init__(self, cfg):
        import edge_tts  # noqa: F401  只是检查装没装

        self.cfg = cfg
        self.voice = cfg.tts_voice
        self.rate = cfg.tts_rate
        self.sample_rate = 24000
        print(f"[tts] 使用 Edge 在线语音 音色 {self.voice} 语速 {self.rate}")

    async def synth(self, text: str, speed: float | None = None) -> np.ndarray:
        import edge_tts

        text = clean_tts_text(text, self.cfg.tts_strip_brackets)
        if not text:
            return np.zeros(0, dtype=np.float32)
        rate = f"{(speed - 1) * 100:+.0f}%" if speed else self.rate
        comm = edge_tts.Communicate(text, self.voice, rate=rate)
        buf = bytearray()
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                buf.extend(chunk["data"])
        if not buf:
            return np.zeros(0, dtype=np.float32)
        pcm = await asyncio.to_thread(_decode_audio_bytes, bytes(buf), self.sample_rate)
        return trim_silence(pcm, self.sample_rate)


class GPTSoVITSTTS:
    r"""调用本机 GPT-SoVITS 的 HTTP 服务（官方 api_v2.py）。

    为什么不直接 import：GPT-SoVITS 要求 numpy<2、transformers<5，和本项目环境冲突，
    只能跑在独立 venv（D:\DeepSeek-Harness\gsv-venv）里。官方 api_v2.py 正好提供 /tts 接口。

    启动：start_gsv.bat（run.bat 在 TTS_BACKEND=gsv 时会自动调用）
    音色：由 GSV_REF_AUDIO（几秒参考音频）+ GSV_PROMPT_TEXT（它的逐字稿）决定，
          换音色只要换这两项，不用重训。
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.api = cfg.gsv_api_url.rstrip("/")
        self.sample_rate = int(cfg.gsv_sample_rate)
        self.voice = f"{Path(cfg.gsv_ref_audio).stem}"
        print(f"[tts] GPT-SoVITS 服务 {self.api}  参考音频 {Path(cfg.gsv_ref_audio).name}")
        ok, msg = self._health()
        if ok:
            print(f"[tts] 服务在线 {msg}")
        else:
            print(f"[tts] ⚠ 连不上服务：{msg}")
            print("[tts]   先运行 start_gsv.bat 把 GPT-SoVITS 服务起起来")

    def _health(self):
        import urllib.request

        try:
            with urllib.request.urlopen(f"{self.api}/docs", timeout=5) as r:
                return True, f"HTTP {r.status}"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {str(exc)[:60]}"

    def _post(self, payload: dict) -> bytes:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            f"{self.api}/tts", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.gsv_timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            # 服务端把原因放在响应体里（比如「请输入有效文本」），别只报个 400
            detail = exc.read().decode("utf-8", "ignore")[:200]
            raise RuntimeError(f"GPT-SoVITS 返回 {exc.code}：{detail}") from None
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"连不上 GPT-SoVITS 服务（{self.api}）：{exc.reason}。先跑 start_gsv.bat") from None

    def _ref_for(self, emotion: str = "") -> tuple[str, str]:
        """每种情绪可以用不同的参考音频（配了就用，没配就用默认的）。

        注意：换参考音频必须同时给它对应的逐字稿，否则克隆会失真。
        """
        cfg = self.cfg
        table = {"happy": ("gsv_ref_happy", "gsv_prompt_happy"),
                 "angry": ("gsv_ref_angry", "gsv_prompt_angry"),
                 "sad": ("gsv_ref_sad", "gsv_prompt_sad"),
                 "shy": ("gsv_ref_shy", "gsv_prompt_shy")}
        if emotion in table:
            ref_key, text_key = table[emotion]
            ref = getattr(cfg, ref_key, "")
            if ref:
                return ref, getattr(cfg, text_key, "") or cfg.gsv_prompt_text
        return cfg.gsv_ref_audio, cfg.gsv_prompt_text

    async def synth(self, text: str, speed: float | None = None,
                    emotion: str = "") -> np.ndarray:
        text = clean_tts_text(text, self.cfg.tts_strip_brackets)
        if not text:
            return np.zeros(0, dtype=np.float32)
        ref, prompt = self._ref_for(emotion)
        payload = {
            "text": text,
            "text_lang": self.cfg.gsv_text_lang,
            "ref_audio_path": ref,
            "prompt_text": prompt,
            "prompt_lang": self.cfg.gsv_prompt_lang,
            "top_k": 15, "top_p": 1.0, "temperature": 1.0,
            "text_split_method": "cut5", "batch_size": 1,
            "speed_factor": float(speed or self.cfg.tts_speed),
            "repetition_penalty": 1.35,
            "parallel_infer": True, "split_bucket": True,
            "media_type": "wav", "streaming_mode": False,
        }
        data = await asyncio.to_thread(self._post, payload)
        if not data:
            return np.zeros(0, dtype=np.float32)
        pcm = await asyncio.to_thread(_decode_audio_bytes, data, self.sample_rate)
        return trim_silence(pcm, self.sample_rate)


def make_tts(cfg):
    backend = (cfg.tts_backend or "kokoro").lower()
    if backend == "api":
        return ApiTTS(cfg)
    if backend == "edge":
        return EdgeTTS(cfg)
    if backend in ("gsv", "gpt-sovits", "gptsovits"):
        return GPTSoVITSTTS(cfg)
    return KokoroTTS(cfg)