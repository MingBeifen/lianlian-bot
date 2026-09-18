# 本项目完全由deepseek生成，遇到任何问题请在issue中提出（会尽量修）

```
注意，要想效果好的话先使用lora微调一下模型
本项目具有一定的使用门槛（
主要因为懒，如果你想一键启动的话，那么本项目不适合你，去找找其他开源项目，他们的接口各方面做得都比我好，我只是向大肥鱼许愿才能做出来这个项目（
```

# 实时语音 + 键盘 混合对话

```
麦克风 → VAD 断句 → SenseVoice 识别 ┐
                                     ├→ 恋恋(Ollama) → Kokoro 合成 → 扬声器
键盘打字（随时可插） ─────────────────┘
```

**说话和打字可以同时用**，谁先来算谁。她说话的时候，你说话或者打字都能打断她。

---

## 一、实测速度（本机）

| 环节 | 模型 | 位置 | 实测 |
|---|---|---|---|
| 识别 | SenseVoiceSmall | RTX 5060 | 2.5s 语音 → **0.16s** |
| 大模型 | lianlian_v2 (Qwen2.5-7B q4 + LoRA) | Ollama 100% GPU | 首字 **0.06~0.3s**（热态）|
| 合成 | Kokoro-82M (`zf_xiaoxiao`) | RTX 5060 | 一句 0.25~0.31s（3.8~11.5 倍速）|
| **首句出声** | | | **0.29 ~ 0.65s** |

启动要 **24~45 秒**（一次性加载三个模型 + 预热），显存占 **6.6GB / 8GB**。
启动时先加载模型、最后才开麦克风，所以哪怕麦克风有问题也不会卡在启动界面。

真实的一轮（打字，实测日志）：

```
你   ：哥哥，我今天考试考砸了……         [键盘]
那挺不好的啊~不过不是所有人都有学霸命吧？你已经很努力了，放松一下也挺好。
       [首句出声 0.65s | 整轮 0.85s]
```

---

## 二、怎么用

> **想装到另一台电脑？** 看 [README-DEPLOY.md](README-DEPLOY.md)：
> `python tools\\package_release.py` 打源码包 -> 对方装 Python 3.12 -> 双击
> `setup_new_machine.bat` -> 改 `models.json` -> `python tools\\check_env.py` -> `run.bat`。
> 环境体检、路径相对化、个人数据剔除都已做好。


```bat
run.bat                      :: 混合模式：直接说话，也可以随时打字回车发消息
run.bat --text               :: 只用键盘（不开麦克风）
run.bat --no-typing          :: 只用语音（屏蔽键盘，防止误触）
run.bat --devices            :: 看设备列表 + 测麦克风
python tools\\mic_test.py           :: 麦克风体检：画音量条、存录音、给调参建议
run.bat --wav a.wav --out reply.wav   :: 离线跑一段录音，顺手存回复语音
python tools\\selftest.py --play    :: 三段链路自检（合成→识别→大模型）
```

**打字怎么用**：直接在窗口里敲字、回车。她正在说话时你打字，她会立刻闭上嘴先听你说
（日志里会打印 `[打断] 你说吧～`）。

程序用的是你已有的 `D:\My-Neuro\lianlian-v0.1\venv_app`（funasr / kokoro / torch / openai 都齐了）。
里面唯一缺的 `sounddevice` 已解包放进 `_vendor\`，**不需要装任何东西**。
想装进 venv 更规范：`venv_app\Scripts\pip install sounddevice`，然后删掉 `_vendor` 照样跑。

---

## 三、麦克风（重点）

权限已经解决（之前是 `HKLM\...\ConsentStore\microphone = Deny` 机器级总开关，现在是 Allow）。
但**权限开了 ≠ 有声音**：本机实测采集到的信号只有 **-96dBFS**，基本是数字静音，
所以说话时 VAD 永远触发不了 —— 这不是程序问题，是系统/硬件层面没把声音送进来。

### 用体检工具定位（推荐先做这个）

```bat
python tools\\mic_test.py
```

它会：安静 1.5 秒测底噪 → 让你说一句话 → 实时画音量条 → 给出结论和建议阈值，
并把录音存成 `_mic_test.wav`（**放一下就知道录进去的是不是你的声音**）。

如果结论是「几乎没有信号」，按顺序查：

1. **设置 → 系统 → 声音 → 输入** → 选中该麦克风 → 「音量」拉到 100
2. 老面板更全：`mmsys.cpl` → 录制 → 麦克风 → 属性 → **级别** → 音量 100 +「麦克风加强」+20dB
3. 笔记本可能有**麦克风静音快捷键**（Fn + F4/F8 之类），看键盘上的麦克风指示灯
4. 换一个麦克风试试（蓝牙耳机 / USB 麦）：蓝牙耳机配对后会多出一条输入设备
5. 确认没选到「未插孔」的设备：`run.bat --devices`，用 `INPUT_DEVICE=` 指定 ACTIVE 那条

### 噪声大导致「说话没反应」（你这台机器的实测情况）

| 项目 | 实测 | 正常房间 |
|---|---|---|
| 环境底噪 | **-24 dBFS** | -60 ~ -70 dBFS |
| 说话 RMS | 0.091 | — |
| 最响 0.2 秒 | 0.193 | — |
| 峰值 | 0.65（接近削顶）| — |

能量 VAD 的门槛 = 底噪 × 2.2 = **0.137**，**比你的说话音量还高** → 永远触发不了，
表现就是「麦克风明明能用，但程序里说什么都没反应」。

**解法：silero 神经网络 VAD。** 已经放好在 `_vendor\`（只用 torch，你已经有了），
程序 `auto` 模式下会优先选它。启动日志里应该出现：

```
[vad] 使用 silero 神经网络 VAD（阈值 0.5）
```

同一段真人语音实测：silero 判定 80% 的帧为说话、纯静音 0 误报；能量法只有 68% 且挑环境。

阈值不够就调 `VAD_THRESHOLD`（默认 0.5，环境吵可以降到 0.35~0.4）。

**另外建议**：峰值 0.65 已经接近削顶、底噪 -24dBFS，说明系统的「麦克风加强」开太高了。
把加强调低一档再跑一次 `mic_test.py`：如果底噪降得比人声多，说明增益过头，降下来识别会更准。

### 程序已经自适应的部分

- **开机自动校准**：先安静听 1 秒，按实测底噪设 VAD 门槛（`VAD_AUTO_CALIBRATE=1`），
  不同麦克风电平能差 100 倍，写死阈值必然有一半人用不了。启动时会打印
  `[mic] 底噪 RMS 0.000015 → VAD 门槛自动设为 0.000073`。
- **底噪过低会警告**：低于 -95dB 时直接提示"麦克风基本没送声音进来"。
- **`INPUT_GAIN`**：麦克风信号太轻就放大（比如 `INPUT_GAIN=5`），再送 VAD 和识别。

## 三·五、大模型后端

### llama.cpp vs Ollama

切换只改 `.env` 三行，**应用代码一行不用动**（全靠 OpenAI 兼容接口）。

| | **llama.cpp**（当前） | Ollama |
|---|---|---|
| 显存 | 5138 MiB | 5544 MiB |
| 首字延迟（热）| **0.05s** | 0.06s |
| 启动 | **4 秒**（JIT 预热后）| 10 秒 |
| 人设 | 放 `persona.txt`，改完重启程序即可 | 烧在 Modelfile，要 `ollama create` |
| 额外能力 | GBNF 语法约束、LoRA 热切换、采样精细控制 | 模型管理省心、自动起停 |

换回 Ollama：`.env` 里改 `LLM_BASE_URL=http://127.0.0.1:11434/v1`、
`LLM_MODEL=lianlian_v3:latest`，并删掉 `LLM_SYSTEM_PROMPT_FILE` 那行（人设已在 Modelfile 里）。

### CUDA 12.4 包还是 Vulkan 包？

本机驱动 573.22 最高只支持 CUDA 12.8，而 llama.cpp 官方预编译的 CUDA 包是 **13.4**（要 r580+ 驱动），
直接跑会 `ggml_cuda_init failed`。但**官方那个 CUDA 12.4 包可以用**：

CUDA 12.4 的 nvcc 根本不认识 sm_120，能跑是因为内核里带了 PTX，由驱动**现场 JIT 编译**给 5060 用。

| | Vulkan 包（`llamacpp-vk`）| **CUDA 12.4 包**（`llamacpp-cu124`，当前）|
|---|---|---|
| 第一次运行 | 14 秒就绪 | 6 秒加载 + **29 秒 JIT 编译** |
| 之后每次启动 | 14 秒 | **4 秒** |
| 冷启首轮首字 | 0.40s | **0.11s** |
| 首字（热）| 0.08s | **0.05s** |
| 长上下文 prefill | 0.22~0.28s | 0.26s |
| 显存 | 4961 MiB | 5138 MiB |

**关键点：JIT 结果会落盘缓存**（实测 20 个文件 / 68.5MB）。
默认缓存在 `%LOCALAPPDATA%\NVIDIA\ComputeCache`，被系统清理或驱动更新后要重新编译 30 秒。
`start_llama.bat` 已经把 `CUDA_CACHE_PATH` 固定到 `voicebot\jit-cache`，不会被误清。

想切回 Vulkan：编辑 `start_llama.bat`，把 `set "LLAMA=...cu124"` 换成 `...llamacpp-vk` 那行即可。

## 三·六、长期记忆

她会记住你之前说过的事。"我下周三有什么事来着？" → "考试。你这记性。"

### 三层结构

| 层 | 内容 | 用法 |
|---|---|---|
| **档案层** `kind=profile` | 名字、生日、宠物这类不变的事 | 每轮**全部注入**，不检索 |
| **事实层** `fact/preference` | "下周三有考试"、"喜欢美式" | 向量检索 |
| **事件层** `event` | "9月初说考试考砸了，我嘴硬但安慰了他" | 向量检索 + 时间衰减 |

### 检索怎么打分

```
score = 0.75 × 余弦相似度 + 0.15 × 时间衰减（半衰期 30 天）+ 0.10 × 重要度
```

取 Top-8（`MEMORY_TOP_K`）**不做阈值过滤**——中文嵌入的相似度基线很高
（实测不相关句子也能到 0.59），靠阈值卡必然误杀，所以交给大模型自己判断相关性。

### 嵌入模型必须用中文优化的

| 模型 | R@1 | 结论 |
|---|---|---|
| `nomic-embed-text`（Ollama 自带）| **1/5** | ❌ 中文召回完全不可用 |
| `bge-small-zh-v1.5`（24M）| 3/5 | 勉强 |
| **`bge-base-zh-v1.5`（102M）** | **6/7，R@3 7/7** | ✅ 默认 |

CPU 上跑，单条查询 43ms、1 万条记忆检索 1.65ms（numpy 暴力点积），
**不需要向量数据库**。首次使用自动从 HF 下载约 400MB，之后走缓存。

### 注入位置很讲究

记忆块放在**「对话历史之后、本轮用户消息之前」**，而不是塞进开头的 system prompt：

```
[system 人设] [历史对话...] [system 记忆块] [user 本轮消息]
                              ↑ 每轮都变         ↑ 每轮都变
```

这样前缀（人设 + 历史）保持不变，**KV 缓存照样命中**，只有记忆块和这一句话要重新 prefill（实测 0.26s）。
如果塞进开头，每轮都要重新 prefill 整个上下文。
另外记忆块**不进历史**，只在那一轮有效（否则会塞满上下文、旧记忆反复出现）。

### 管理记忆

```bat
python core\\memory.py --list                    :: 看现有记忆
python core\\memory.py --add "哥哥喜欢喝美式" --kind preference --importance 0.7
python core\\memory.py --query "我喝什么"         :: 测检索
python core\\memory.py --reset                   :: 清空
python core\\memory.py --demo                    :: 塞 12 条样例 + 跑召回质量自测
```

配置项：`MEMORY_ENABLE` `MEMORY_TOP_K` `MEMORY_DB` `DEBUG_MEMORY`（1=一行摘要 2=打印全部）

### 自动写入（聊完自动记）

正常聊天就行，不用手工 `--add`。实测：

```
你：我下周三要考驾照，有点紧张
    [memory] ＋ #1 [fact 0.7] 哥哥下周三要考驾照

你：今天天气不错
    （什么都没记）

你：我叫陈思杰，你以后别叫错了
    [memory] ＋ #2 [profile 0.9] 哥哥叫陈思杰

你：我养了只猫叫豆豆，是只橘猫
    [memory] ＋ #3 [profile 0.9] 哥哥养了只猫叫豆豆 是只橘猫
    [memory] ↻ #3 [fact 0.7] 哥哥养了只猫叫豆豆      ← 去重：更新而不是新增
```

**时机**：放在"她正在播语音"的那几秒 —— 那时 GPU 是空的，**等于零延迟成本**。

#### 三道防线（因为恋恋的 LoRA 会"入戏"编故事）

第一版直接用角色模型抽记忆，结果是**编故事**：

```
输入：我下周三要考驾照，有点紧张
输出：哥哥喜欢开车，但害怕考试。我得陪他。他总是说：「别担心，我肯定能行。」
      PROFILE|0.3|我陪他复习，他总说：「你陪我复习，我感觉好点。」
      ……全是编的
```

所以现在有三道闸：

| 防线 | 作用 |
|---|---|
| **1. 严格提示词 + 零温度** | 「你是数据抽取程序，不是聊天角色」+ 5 条铁律 + 正反例；温度 0 |
| **2. GBNF 语法约束** | llama.cpp 在采样层强制格式，**不可能**输出非法格式 |
| **3. 归一化重叠校验** | 抽出的内容必须有 ≥30% 的字来自原话（先去掉「哥哥」和标点再比），编造的内容直接丢 |

效果：**6 条输入 → 4 条正确提取、2 条正确忽略、0 条误报**。

#### 为什么不用 JSON 格式

GBNF 里写 JSON 要疯狂转义（`"{\"memories\": ["`），又长又容易错。改成行格式：

```
FACT|0.7|哥哥下周三要考驾照
EVENT|0.9|哥哥答应周末带我去看海
```

语法只有 5 行，解析就是 `split("|", 2)`，**结构上不可能解析失败**，而且 token 更少、生成更快。

#### 实测踩过的坑（都已修）

| 现象 | 根因 | 修法 |
|---|---|---|
| 她说「不是因为你，**是系统告诉我的**」 | 记忆块写的是"以下是你之前记住的事，**不是哥哥刚说的**"——这句话把记忆框成"外部资料" | 改成「【你本来就记得的事】」，并明说"绝对不要提记忆/系统/数据这类词" |
| 输出几百个 `～～～～～` | **从 Ollama 迁移时漏了 `repeat_penalty`**（Ollama 的 Modelfile 里有 1.1，llama.cpp 默认 1.0）| `start_llama.bat` 加 `--repeat-penalty 1.15 --repeat-last-n 256`；TTS 层再兜底把 3 个以上连续标点压成一个 |
| 问「我养了什么宠物」→ 存进「哥哥喜欢养宠物狗」 | **问句里没有新事实，模型只能编** | 加问句闸门：含"什么/吗/呢/谁/还记得"等一律不抽取 |
| 记住的是"橘猫"，她答"**白猫**" | 去重时后来更笼统的"哥哥养了只猫"**覆盖**了带颜色的那条 | 相似度 >0.90 时**保留信息更多的那条**（长的优先） |
| 回答里加戏（颜色、品种、地点）| 模型脑补 | 记忆块里强调"细节以这里写的为准，不要自己加颜色/品种/地点" |
| **她自称"语音助手"**（"我是你的语音助手～"）| 两个原因叠加：① 身份类问题本身容易触发模型的"助手默认人格"；② `temperature=0.6` 有随机性，实测同一个问题 10 次里有 1 次破功 | ① `persona.txt` 末尾加【身份铁律】：明确禁止自称助手/AI/程序/模型/虚拟角色，并规定"问你是谁就说自己是恋恋"；② 记忆块的禁令词表也补上"助手/AI/模型" |

> 改完实测：4 个身份类问题 × 10 次 = **0/40 破功**（改之前同等条件约 4/40）。
> 改 `persona.txt` **不需要重建模型**，重启程序即可（人设是运行时发过去的）。

#### 开关

```ini
MEMORY_AUTO_WRITE=1     # 0 = 只读不写（调试用）
MEMORY_OVERLAP=0.45     # 重叠阈值，调低=记得更多但可能混入编造
```

## 三·七、语音合成（TTS）

当前：**Kokoro-82M-v1.1-zh** + 音色 `zf_021`，GPU 上跑，首句 0.5s 出声。

### 为什么必须用 v1.1-zh

| | v1.0（原来的）| **v1.1-zh（现在）** |
|---|---|---|
| 中文音色数 | 4 个（实验性）| **103 个** |
| 训练 | 中文只是附带 | **专门为中文训练** |

官方在 v1.0 的模型卡里就写了"中文支持不佳，建议用 v1.1-zh"。原来的 `zf_xiaoxiao` 恰好在
那 4 个实验性音色里 —— 这就是"听着很怪"的原因。

切换只需两行：
```ini
TTS_REPO=hexgrad/Kokoro-82M-v1.1-zh     # 或指向本地目录（已下好的话）
TTS_VOICE=zf_021
```

### 踩过的坑：英文词被静默丢弃

中文前端默认 `en_callable=None`，会把**整段英文替换成 `❓` 直接丢掉**：

```
输入：我喜欢 Qt 和 AI
音素：我2ㄒㄧ3ㄏ万5 ❓ ㄏㄜ2 ❓      ← Qt、AI 都没了，听起来就是"我喜欢 和 "
```

修法分两层，**第二层更坑**：

**第一层**：给 `KPipeline` 传 `en_callable`，别让英文被换成 ❓。

**第二层**：kokoro 的 tokenizer 会把**不在模型词表里的字符静默丢掉**，而这个中文模型的词表里：

| 字符 | 在词表里吗 | 后果 |
|---|---|---|
| 连字符 `-` | **没有** | 用 `"-".join()` 拼的 `a-i` → `ai`（实测 26 个横杠全丢）|
| 小写 `g` | **没有** | 字母 g 直接消失 ← 就是"有些字母不念出来"|
| a-f、h-z、空格 | 有 | 正常 |
| 注音符号 | 38 个齐全 | 正常 |

所以最终做法是**按词表过滤**：读 `config.json` 拿到 171 个符号，转换时只输出词表里存在的字符，
缺的用形近的顶替（ASCII `g` → IPA 的 `ɡ` U+0261，长得一样且在词表里）：

```python
def make_english_callable(vocab):
    def conv(en):
        spell = len(en) <= 3 or en.isupper()   # 缩写按字母念，长词整体念
        out = []
        for ch in en.lower():
            if ch in vocab:      out.append(ch)
            elif ch == "g":      out.append("ɡ")   # 词表里没有 ASCII g
        return " ".join(out) if spell else "".join(out)
    return conv
```

实测 26 个字母 + AI/Qt/OK/GPT/GitHub/wifi/PDF **零丢失**。

### 备选：微软 Edge 在线语音

`TTS_BACKEND=edge` + `TTS_VOICE=zh-CN-XiaoxiaoNeural`。音色更自然、零显存占用，
但要联网，而且不是本地模型。想试就改这两行（`TTS_RATE=+8%` 可调语速）。

## 三·八、GPT-SoVITS 后端（TTS_BACKEND=gsv）

除了 Kokoro / Edge，还可以用 **GPT-SoVITS**（音色克隆，韵律明显更自然）。

### 为什么要走 HTTP 服务

GPT-SoVITS 要求 `numpy<2`、`transformers<5`，和本项目环境**直接冲突**（硬装会搞坏
Kokoro / SenseVoice / bge / misaki）。所以它跑在**独立 venv** 里：

```
venv_app（应用）  ──HTTP:9880──>  gsv-venv（GPT-SoVITS api_v2.py）
    numpy 2.5                            numpy 1.x
    transformers 5.17                    transformers 4.x
```

```bat
start_gsv.bat     :: 启动 GPT-SoVITS 服务（run.bat 在 TTS_BACKEND=gsv 时自动调用）
run.bat           :: 会自动拉起 llama-server + GPT-SoVITS 两个服务
```

### 换音色：不用训练

只要换两个配置项（**零样本克隆**，实测 4 秒参考音频就够）：

```ini
GSV_REF_AUDIO=D:\path\to\你的参考音频.wav   # 3~10 秒干净人声
GSV_PROMPT_TEXT=这段音频的逐字稿              # 必须和音频内容一致
```

改完重启程序即可。想要更稳/更像再考虑微调（1 分钟数据就够）。

### 实测延迟（这是主要代价）

| 句子长度 | GPT-SoVITS | Kokoro |
|---|---|---|
| 3 字 | 1.35s | ~0.3s |
| 7 字 | 1.21s | ~0.3s |
| 23 字 | 2.55s | ~0.4s |

**有个 ~1.2 秒的固定开销**（每次请求都要跑一遍 roberta-large 做文本编码），
所以短句也快不了。`streaming_mode=3` 实测反而更慢（3.49s），没用。

对比：**首句出声从 0.5s 变成 ~2.5s**。韵律的自然度提升是否值这 2 秒，得自己听。

显存：GPT-SoVITS 约 1.5GB（llama-server 5.1GB + 它 1.5GB = 6.6GB / 8.1GB ✓ 装得下）

### 集成时踩的坑（都已解决）

| 坑 | 表现 | 解决 |
|---|---|---|
| `torchaudio.load` 报 TorchCodec | `Could not load libtorchcodec_core4.dll`（torchcodec 和 torch 2.11 不匹配）| 在 gsv-venv 里放了个 `sitecustomize.py`，让 `torchaudio.load` 自动退回 soundfile |
| 服务端不认模型路径 | `HFValidationError: ... 'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large'` | `api_v2.py` **不会自己设** `bert_path` / `cnhubert_base_path`，得在 `start_gsv.bat` 里设 |
| 依赖装不上 | `jieba`/`jieba_fast`/`pyopenjtalk` 无 wheel、`distance` 等 5 个传递依赖要编译 | 手工装纯 Python 源 + 给 `jieba_fast` 写垫片转发给 `jieba`、`pyopenjtalk` 写存根（只影响日语）|
| 输出音量爆表 | 峰值 17328 | GPT-SoVITS 返回 **int16 量级**，必须 `/32768` 归一化 |
| **克隆出来变成男声** | 女声参考克隆后听着像男的 | **播放采样率不匹配**：GPT-SoVITS 输出 **32kHz**，而播放器硬编码了 24kHz，按 24k 播 32k 音频 = 音调压低 25%（实测 237Hz → 178Hz，正好落进男声区间），时长还变成 1.33 倍。修法：`Player` 用 `tts.sample_rate` 而不是配置里的固定值 |

> 这个坑很隐蔽：Kokoro 和 Edge 恰好都是 24kHz 所以一直没暴露，换 GPT-SoVITS（32kHz）才撞上。
> **教训**：音频链路上任何一环都不能硬编码采样率，必须跟着上游走。

## 三·九、主动性（她自己找时机开口）

现在她不只是"你问才答"——安静久了会自己搭话，早上会打招呼，还会想起你提过的事。

### 核心设计：规则决定"说不说"，LLM 只管"说什么"

一开始让 LLM 自己决定"要不要说"，结果是**她每次都选沉默**（模型面对"可以说也可以不说"
时永远选安全的那个），而且连 `<SILENT>` 都拼错（`<silnet>`/`<silient>`/`<silently>`）。
所以改成：

```
硬规则闸门（该不该说） → LLM（说什么） → 后置校验（合不合格）
```

### 四层闸门（便宜的先查）

| 层 | 检查 |
|---|---|
| 1. 硬规则 | 总开关、是否正在对话、安静够不够久、静默时段（默认 0-7 点） |
| 2. 频率 | 每小时 ≤2 次、每天 ≤12 次、两次间隔 ≥8 分钟 |
| 3. 退避 ⭐ | **没被理一次，下次间隔翻倍**（8→16→32→64 分钟）；连续 3 次没理 → 当天闭嘴 |
| 4. 环境 ⭐ | 房间太安静（你可能不在）就不自言自语 |

你只要回一句话，退避计数立刻清零。

### 内容生成：只在她"有理由"时才说

- **有明确时机**（早上问候 / 饭点 / 深夜催睡，每天各一次）→ 直接生成
- **没有时机** → 先跑一次快速判断（"有没有想说的"），说是才生成
  - 实测约一半的触发她会选择不说 ✓ 这正是"宁可不说也不说废话"

### 三道内容校验（提示词压不住的用代码压）

| 校验 | 为什么 |
|---|---|
| **静默变体** | 模型会拼错 `<SILENT>`，任何 `<sil...>` 都当沉默 |
| **编造动作** ⭐ | 她会说"我刚帮你泡好了咖啡"——**她动不了手**。检测到就重试一次，还犯就不说了 |
| **相似度去重** | 会翻来覆去说同一个话题（"看你写代码姿势难看"×4）。和上次的话 Jaccard 相似度 >0.55 就重试 |

### 素材来源（决定她说不说得有意思）

```
你记得的事（记忆库最近 12 条）      ← 最重要，"下周三考驾照"这种
你们刚才聊的（最近 6 轮对话）        ← 让她能顺着话题说
上次你说的是「…」（别再说这个）
当前时间 + 角度清单（A问他在干嘛/B提记得的事/C说想法/D顺着聊）
```

实测她能说出"明天要考驾照啊～那你复习一下啦～"这种**真的用了记忆**的话。

### 配置

```ini
PROACTIVE=1                    # 总开关
PROACTIVE_IDLE_SEC=600         # 安静多久可以开口
PROACTIVE_MIN_GAP_SEC=480      # 两次主动最小间隔
PROACTIVE_BACKOFF_BASE=2.0     # 没被理一次，间隔翻几倍
PROACTIVE_MAX_PER_HOUR=2
PROACTIVE_UNANSWERED_LIMIT=3   # 连续几次没理就当天闭嘴
PROACTIVE_QUIET_HOURS=0-7      # 这个时段不主动
PROACTIVE_REQUIRE_SOUND=1      # 房间太安静就不说
PROACTIVE_POLL_SEC=20          # 多久检查一次（调试时可调小）
```

**调试建议**：先把 `PROACTIVE_IDLE_SEC` 调到 60、`PROACTIVE_POLL_SEC` 调到 10，
快速感受频率合不合适，再改回去。

## 三·十、Live2D 形象（VTube Studio）

本项目只做「驱动」，渲染交给 **VTube Studio**（Steam 免费版即可，自带示例模型）：

```
Player 播放电平 ──24fps──> InjectParameterDataRequest(MouthOpen)  ← 口型自动跟声音
大模型情绪判断 ──────────> ExpressionActivationRequest / 热键     ← 表情
```

### 开启步骤

1. Steam 装 VTube Studio（免费），启动并载入一个模型（自带的 Hiyori 就行）。
2. VTS 设置里打开 **API / 插件功能**（默认端口 8001，保持默认即可）。
3. `.env` 里 `LIVE2D=1`（默认已开），重启 `run.bat`。首次连接 VTS 会弹授权窗口，
   点 **Allow**；token 自动存进 `voicebot/vtube_token.json`，以后不再弹。
4. 说话试试，她的嘴会跟着声音动。灵敏度用 `LIVE2D_MOUTH_GAIN` 调（嘴张太大就调小）。
5. 自测：`python tools\\avatar_test.py --mock`（不用装 VTS，验证程序侧）；
   装好 VTS 后 `python tools\\avatar_test.py`（测口型 + 轮流切表情）。

### 表情怎么配

在 VTS 里给模型做表情文件（`.exp3.json`）或热键，把名字填进 `.env`：

```ini
# 表情文件：happy=exp_01.exp3.json,shy=exp_02.exp3.json
LIVE2D_EXPR=happy=exp_01.exp3.json
# 或者用热键：angry=ht_angry
LIVE2D_HOTKEY=sad=ht_sad
```

情绪由 `llm.classify_emotion()` 在读完整句后判断（趁语音还在播，不占首句出声时间），
范围：`neutral / happy / shy / angry / sad / surprised`。
`neutral` 默认会取消上一个表情（`LIVE2D_RESET_NEUTRAL=0` 可关），被打断时脸立刻收回 neutral。

### 注意

- VTS 没装 / 没开 / 授权失败都不影响聊天：后台 3~60 秒静默重试，日志只提示一次。
- 标准 Cubism 模型都有 `MouthOpen`；自定义模型可能叫 `ParamMouthOpenY`，
  用 `LIVE2D_MOUTH_PARAM` 指定。
- VTS 里自带的麦克风口型会和我们抢嘴，建议关掉（设置里的 microphone lipsync）。
- 口型帧率默认 24fps，被 VTS 限流时自动降速。
- `EMOTION=0` 可整体关掉情绪判断，表情就一直是 neutral。

## 三·十一、桌宠模式（把 VTube Studio 变成桌面挂件）

驱动和渲染已经打通，剩下的是把 VTS 窗口调成桌宠：置顶、无边框、不占任务栏、
鼠标穿透，再把模型摆到屏幕角落。VTS 自身只支持透明背景，不支持这些窗口行为，
所以项目里带了 `desktop_pet.py`（纯 Win32 API、零依赖），`run.bat` 启动时自动应用。

### 第一步：VTS 开透明背景

VTube Studio -> 设置 -> General -> Background 附近找 **Transparent Background** 打开
（找不到就去 Hotkeys 里给 Toggle Transparency 绑个键按一下）。
这一步只需做一次，VTS 会记住；不开的话模型会带一块底色矩形。

### 第二步：`.env` 打开桌宠模式

```ini
PET=1
PET_POSITION=br        # tl/tr/bl/br/center/keep
PET_SIZE=              # 例 520x760，留空 = 保持 VTS 当前大小
PET_MARGIN=20
PET_BORDERLESS=1
PET_TOPMOST=1
PET_HIDE_TASKBAR=1
PET_CLICK_THROUGH=0    # 1 = 启动就鼠标穿透
```

**顺序：先启动 VTube Studio，再 `run.bat`**。退出程序时窗口自动恢复原样。

### 全局热键（任何时候可用）

| 热键 | 作用 |
|---|---|
| `Ctrl+Alt+P` | 鼠标穿透开关（穿透时点不到她，桌面其它东西正常点）|
| `Ctrl+Alt+T` | 窗口置顶开关 |
| `Ctrl+Alt+B` | 无边框开关 |
| `Ctrl+Alt+H` | 显示 / 隐藏 |
| `Ctrl+Alt+R` | 恢复成普通窗口 |
| `Ctrl+Alt+Shift+方向键` | 微调位置（20 像素）|
| `Ctrl+Alt+Shift+加号 / 减号` | 放大 / 缩小窗口 |

### 单独调 / 出问题救援

```bat
python live2d\\desktop_pet.py --status          :: 看当前窗口样式
python live2d\\desktop_pet.py                   :: 单独应用（Ctrl+C 退出并恢复）
python live2d\\desktop_pet.py --keep            :: 应用后不随退出恢复
python live2d\\desktop_pet.py --restore         :: 恢复原样（程序被强杀过也能救）
python live2d\\desktop_pet.py --click-through   :: 直接以穿透模式启动
```

注意：

- 鼠标穿透开着时点不到 VTS，按 `Ctrl+Alt+P` 切回来即可；
- 程序被任务管理器强杀后样式可能留在 VTS 上，跑一次 `--restore` 就好
  （样式备份存在 `_pet_backup.json`，正常退出会自动删）；
- VTS 没启动时桌宠模式只打印一行提示，聊天功能不受影响。

## 三·十二、自渲染桌宠（PET_BACKEND=native，不依赖 VTube Studio）

上一节是让 VTube Studio 当渲染器；这一节是项目自己渲染：live2d-py（Cubism Native
Core）+ GLFW，把模型直接画在**逐像素透明**的置顶窗口里，不需要装 VTS、不需要授权插件，
也没有色键抠图的毛边。

### 两种方式怎么选

| | `PET_BACKEND=vtube` | `PET_BACKEND=native` |
|---|---|---|
| 渲染 | VTube Studio（额外装、单独进程）| 项目自带（live2d-py + GLFW）|
| 透明 | VTS 透明背景 | GLFW 逐像素透明，边缘干净 |
| 交互 | VTS 自带鼠标互动/动作 | 拖动、点击反应、视线跟随、自动眨眼呼吸 |
| 模型 | VTS 模型库 | 任意 Cubism 3/4 的 model3.json |

### 开启步骤

1. 双击 `setup_native_pet.bat`：自动建 `D:\DeepSeek-Harness\live2d-venv` 并安装
   live2d-py / pygame / glfw（第一次约几十 MB）。
2. `.env` 改成：

   ```ini
   PET=1
   PET_BACKEND=native
   NATIVE_PET_MODEL=D:\...\hiyori_vts\hiyori.model3.json
   NATIVE_PET_PYTHON=D:\DeepSeek-Harness\live2d-venv\Scripts\python.exe
   PET_SIZE=480x700
   ```

3. 直接 `run.bat`（不用启动 VTube Studio）。主程序会自动拉起渲染器、退出时一起收掉，
   渲染器日志在 `voicebot/native_pet.log`。

### 她能做到什么

- 口型跟着 TTS 播放电平动（主程序 -> TCP -> 渲染器，30fps）；
  语音 RMS 只有 0.05~0.2，所以实际按 `(电平 x NATIVE_PET_MOUTH_GAIN) ** 0.7` 映射，
  默认增益 6；觉得嘴张太大/太小就调 `NATIVE_PET_MOUTH_GAIN`；
- 情绪切表情：开心笑眼/上扬、生气皱眉、害羞脸红、难过垂眉、惊讶瞪眼；
- 自动眨眼、呼吸，视线和脑袋跟着鼠标走；
- **待机动作**：安静时从模型 `animations/` 目录随机播一个动作（hiyori 有 10 个），
  说话时自动暂停、把嘴让给口型；间隔用 `NATIVE_PET_MOTION_GAP_MIN / MAX` 调，
  不要就把 `NATIVE_PET_IDLE_MOTION=0`；
- **情绪反应动作**：每轮回复的情绪（开心/生气/难过/害羞/惊讶）会挑一个「气质相符」的
  动作立刻播（按动作文件里各参数的波动特征自动匹配，日志会打印映射表）；不要就
  `NATIVE_PET_REACT_MOTION=0`；
- **说话身体动作**：说话时身体、脑袋、手臂会跟着语音能量轻微摆动，说完自动归位
  （待机动作暂停，所以不会被抢参数）；
- **LLM 动作标签**：每轮回复会让 LLM 额外判断一个伴随动作（`nod / shake / wave /
  think / shy_hide / laugh / none`），和情绪判断并行、不占说话延迟；说完立刻触发
  对应动作。动作映射会按动作文件特征自动生成（日志会打印），想指定就
  `NATIVE_PET_ACTION_MAP=nod=Idle_2,wave=Idle_5`；不要就 `LLM_ACTION=0`。
- 按住左键拖到屏幕任意位置；点一下（不拖动）= 惊讶小反应；
- 全局热键：`Ctrl+Alt+P` 穿透 / `T` 置顶 / `H` 显隐 / `R` 复位 /
  `Ctrl+Alt+Shift+方向键` 微移 / `+ -` 缩放 / `Ctrl+Alt+Q` 关闭。

穿透用的是窗口过程子类化（WM_NCHITTEST 返回 HTTRANSPARENT），不是
`WS_EX_TRANSPARENT` 样式，所以不会影响透明边缘。

### 动作是怎么调度的（Neuro 式分层）

像 Neuro-sama 这类 AI VTuber，动作不是模型「自己想动」，而是四层叠加：

| 层 | 内容 | 我们对应实现 |
|---|---|---|
| 1. 绑定 | 画师拆分部件、头发物理 | 模型自带的 `.moc3` + `physics3.json` |
| 2. 动作库 | 手工/动捕的 `motion3.json` | `animations/` 里的 10 个动作（可换模型扩充）|
| 3. 调度 | 状态机决定播哪个动作 | 说话 -> 口型 + 身体摆动；情绪 -> 反应动作；空闲 -> 随机待机 |
| 4. 语义 | LLM 输出里的动作标签 | 回复文本 -> GBNF 动作分类（nod/wave/...）-> 触发动作 |

调度优先级：情绪反应 / 动作标签（NORMAL）> 待机动作（IDLE）；说话时动作全部暂停
（这些动作都带嘴部曲线，播了会抢口型），没播的动作排队，等说完立刻补上。

### 手臂姿态 / 四只手

有些模型（包括 VTS 自带的 `hiyori_vts`）**没有 pose3.json**，模型里
`PartArmA` 和 `PartArmB` 两组手臂会同时显示，看起来就是四只手。
渲染器会自动处理：检测到没有 pose 文件时默认锁定显示 A 组（官方动作的默认姿态），
隐藏 B 组。想看另一组就设 `NATIVE_PET_ARM=B`，想恢复两组都显示就设 `both`；
模型自带 pose3.json 时不用管（`auto` 会交给 Cubism 处理）。

### 模型 / 许可

- `NATIVE_PET_MODEL` 指向任意 Cubism 3/4 的 `.model3.json`，同目录要有 moc3、
  纹理、physics3.json 等完整资源；
- 默认的 `hiyori_vts` 是 VTube Studio 自带的 Live2D 官方示例模型，**只在本机自己用，
  不要打包分发**；商用请遵守 Live2D Cubism SDK 许可。

### 出问题怎么查

- 看 `voicebot/native_pet.log`（模型加载、热键数量、连接状态都在里面）；
- 手动跑一次看报错：
  `D:\DeepSeek-Harness\live2d-venv\Scripts\python.exe native_pet.py --demo 10`；
- 口型不动：日志里应有「收到主程序状态，开始跟口型」；
- 完全不动：确认 `.env` 里 `NATIVE_PET_IDLE_MOTION=1`，日志里应有
  「已加载 N 个待机动作」；模型目录要有 `animations/*.motion3.json`；
- 想看动作调度：`NATIVE_PET_DEBUG=1`，日志会打印「播放待机动作 Idle_x」、
  「动作标签 wave -> Idle_7」「播放动作 Idle_7」「说话，暂停待机动作」；
- 觉得某个标签动作不合适：`NATIVE_PET_ACTION_MAP=nod=Idle_2,wave=Idle_5` 覆盖映射
  （动作组名就是日志里打印的 Idle_0~Idle_9）；
- 想切回 VTS：`.env` 里 `PET_BACKEND=vtube`。

## 三·十三、Web 控制台（网页管理一切）

`WEBUI=1`（默认开）时，程序启动后浏览器打开 **http://127.0.0.1:8765**：

| 页面 | 能做什么 |
|---|---|
| 对话 | 聊天记录流式显示；键盘输入；点麦克风用浏览器实时语音输入（16k PCM -> 服务端 VAD/识别）；打断、本机麦克风静音 |
| 记忆 | 查看全部长期记忆；每条可改内容/类型/权重、单独删除；语义搜索；新增；换嵌入模型后一键重算向量 |
| Live2D | 扫描并预览本机所有 model3.json，点击即切换（native 热重启渲染器；VTS 走插件 API 按名字加载）；桌宠大小/位置/口型增益/待机动作等选项 |

说明：

- 页面只监听 `127.0.0.1`，纯本地、无外部依赖（不需要联网、没有 CDN）；
- 模型相关配置已统一到 `models.json`（见下一节），网页不再提供模型编辑页；
- 浏览器语音输入和本机麦克风是两条路：用网页说话时建议勾选「本机麦克风静音」，
  避免同一个房间里的声音被触发两次；
- 她说话的声音默认仍从本机扬声器播放，网页负责文字显示和输入；
- Live2D 切换时会自动回写 `models.json`，下次启动仍然是新模型。

## 三·十四、模型统一配置 models.json

所有模型（LLM / ASR / TTS / VAD / 记忆嵌入 / Live2D / Web 控制台）统一由
`voicebot/models.json` 决定，程序启动时由 `config.py` 读取；

优先级：**models.json > .env > 代码默认值**（只覆盖 JSON 里写了的字段）。

### 生成 / 修改

```bat
python tools\\gen_models.py            :: 按当前 .env + 机器上的模型文件生成
python tools\\gen_models.py --print    :: 只打印不写
```

生成后直接编辑 `models.json`，重启 `run.bat` 生效。主要字段：

```json
{
  "llm":     { "base_url": "http://127.0.0.1:8080/v1", "model": "lianlian", "max_tokens": 220 },
  "process": { "llama_exe": "...", "gguf": "...", "lora": "...", "port": 8080 },
  "asr":     { "backend": "sensevoice", "model": "iic/SenseVoiceSmall", "device": "cuda:0" },
  "tts":     { "backend": "gsv", "voice": "zf_021",
               "gsv": { "ref_audio": "...", "prompt_text": "..." } },
  "vad":     { "backend": "auto", "threshold": 0.5 },
  "memory":  { "top_k": 8, "embed_model": "...\\bge-base-zh" },
  "live2d":  { "backend": "native", "native_model": "...\\hiyori.model3.json", "size": "480x700" },
  "webui":   { "enable": true, "port": 8765 }
}
```

### llama-server 的模型文件

`process.gguf` / `process.lora` / `process.llama_exe` 由 `run.bat` 启动时调用
`sync_models.py` **自动同步进 `start_llama.bat`**，不需要手改 bat；改完 models.json
重启一次 `run.bat`（llama-server 会按新文件启动）。

### 注意

- 换 Live2D 模型也可以直接在网页上点，会自动回写 models.json；
- `.env` 仍然可用（临时覆盖某台机器的路径），但 models.json 里写了的字段优先；
- 换过嵌入模型后，去网页「记忆」页点一次「重算向量」；
- 字段名和 `.env` 一一对应，少写/不写就用 `.env` 或默认值。

## 四、文件结构

```
voicebot/
├── run.bat                    启动（混合语音+键盘）
├── setup_new_machine.bat      新电脑初始化（建 .venv / 装依赖）
├── setup_native_pet.bat       建自渲染桌宠环境（live2d-venv）
├── start_llama.bat            启动 llama-server（按 models.json 同步路径）
├── start_gsv.bat              启动 GPT-SoVITS 服务
├── .env / models.json / models.example.json   配置
├── config/                    persona.txt / Modelfile.voice / gsv_infer.yaml
├── core/                      对话核心
│   ├── voice_chat.py          主循环（语音 + 键盘 + 打断）
│   ├── config.py              配置加载（models.json > .env > 默认）
│   ├── audio_io.py / vad.py   采集 / 断句 / 播放
│   ├── asr.py / tts.py / llm.py
│   ├── memory.py              长期记忆（SQLite + bge）
│   └── emotion.py / proactive.py
├── live2d/                    形象
│   ├── native_pet.py          自渲染桌宠本体（跑在 live2d-venv）
│   ├── native_pet_link.py     主程序桥（口型/情绪/动作）
│   ├── avatar.py              VTube Studio 驱动
│   ├── desktop_pet.py         VTS 窗口桌宠化
│   └── native_pet_setup.py    自渲染环境安装脚本
├── webui/                     Web 控制台
│   ├── server.py              aiohttp REST + WebSocket
│   ├── catalog.py             模型/资源扫描
│   ├── env_store.py           .env / bat 读写
│   └── static/                前端页面（index/app/style）
├── tools/                     工具（不放运行库）
│   ├── gen_models.py          生成 models.json
│   ├── sync_models.py         同步 llama 路径进 bat
│   ├── check_env.py           环境体检
│   ├── setup_new_machine.py   新电脑初始化
│   ├── package_release.py     打可分发源码包
│   ├── mic_test.py / selftest.py / avatar_test.py
├── data/                      个人数据（lianlian_memory.db / vtube_token.json）
├── _vendor/                   sounddevice + silero VAD（免安装）
├── out/                       音频输出 / native_pet.log
└── jit-cache/                 CUDA JIT 缓存（别删，省 30 秒启动）
```

---

## 五、配置速查（复制 `.env.example` 成 `.env`）

| 想干嘛 | 改哪个 |
|---|---|
| 打字也想屏蔽，只用语音 | `ALLOW_TYPING=0` 或 `--no-typing` |
| 换人设 / 改性格 | 改 `Modelfile.voice` 后 `ollama create lianlian_voice -f Modelfile.voice`，再把 `LLM_MODEL` 指过去 |
| 说话更短更快 | `LLM_MAX_TOKENS=120`、`SILENCE_MS=450`（更早判定你说完了）|
| 换音色 | `TTS_VOICE=zf_021`（v1.1-zh 有 103 个音色，`zf_001`~`zf_100`）|
| TTS 模型位置 | `TTS_REPO` 可以是 repo id（自动下载）或**本地目录** |
| 换成微软在线语音 | `TTS_BACKEND=edge` + `TTS_VOICE=zh-CN-XiaoyiNeural`（更自然，要联网）|
| 语速 | `TTS_SPEED=1.05`（>1 更快）|
| 用内置扬声器 | `OUTPUT_DEVICE=5`（数字看 `--devices`）|
| 想说话打断她 | `BARGE_IN=1` **必须戴耳机**（外放会让她听见自己 → 自问自答）|
| 外放（默认）| `BARGE_IN=0` 半双工：她说话时麦克风自动闭上 |
| 显存告急 | `TTS_DEVICE=cpu`（慢 ~0.7s，省 0.7GB）|
| 麦克风太灵敏/太钝 | `VAD_MIN_RMS` 调大挡底噪、调小更灵敏（自动校准会覆盖它）|
| VAD 选型 | `VAD_BACKEND=auto/silero/webrtc/energy`；吵的环境用 silero，阈值 `VAD_THRESHOLD` |
| 麦克风声音太轻 | `INPUT_GAIN=5`（先跑 `mic_test.py` 看建议值）|
| 旁白（动作描写）也念出来 | `TTS_STRIP_BRACKETS=0` |
| Live2D 开关 / 口型灵敏度 | `LIVE2D=1`、`LIVE2D_MOUTH_GAIN=6` |
| Live2D 表情 | `LIVE2D_EXPR=happy=exp_01.exp3.json` 或 `LIVE2D_HOTKEY=sad=ht_sad` |
| 桌宠模式 | `PET=1`、`PET_POSITION=br`、`PET_SIZE=520x760` |
| Web 控制台 | `WEBUI=1`、`WEBUI_PORT=8765`（浏览器打开 http://127.0.0.1:8765）|
| 换模型 | 改 `voicebot/models.json`（`python tools\\gen_models.py` 生成，重启 run.bat）|
| 装到新电脑 | `python tools\\package_release.py` 打包 -> `setup_new_machine.bat` -> `check_env.py` |
| 鼠标穿透 | 热键 `Ctrl+Alt+P`（或 `PET_CLICK_THROUGH=1` 默认开）|
| 桌宠实现方式 | `PET_BACKEND=vtube`（VTS）或 `native`（自渲染）|
| 自渲染模型 | `NATIVE_PET_MODEL=D:\...\xxx.model3.json` |

**`LLM_SYSTEM_PROMPT` 默认留空**，直接用 `Modelfile` 里写好的恋恋人设。填了会覆盖人设，别重复写角色设定。

---

## 六、内部是怎么串起来的

1. **两种输入，一个队列**：麦克风线程切出一句话 → `np.ndarray`；键盘线程读到一行 → `str`。
   两者都塞进同一个 `asyncio.Queue`，主循环 `await queue.get()` 谁先来算谁。
2. **打断**：`asyncio.Event`。麦克风在你说满 160ms 时 set；打字则是**先判断她是不是正在说话**，
   正在说才 set（否则刚收到这句就把自己那一轮取消了 —— 这个竞态踩过一次）。
   事件触发 → `player.stop()` 清空播放队列 + cancel 当前任务。
3. **识别**：SenseVoice 在 GPU 上跑，输出带 `<|zh|><|NEUTRAL|>` 标签，`clean_asr_text()` 剥掉。
4. **大模型**：流式 token 不是等整段，而是**按标点攒句**（`sentences()`），够长就送去合成
   —— 这是"首句出声 0.3s"的关键。
5. **合成**：Kokoro 出 24k float32，掐掉首尾静音、去掉 Markdown 和 `（慢半拍）` 这类旁白。
6. **播放**：PortAudio 回调 + 队列，`stop()` 能立刻静音并丢掉没播的部分。

---

## 七、常见问题

**她自问自答 / 说了两轮之后就不理我了**
外放时麦克风会听到扬声器 → 她把**自己的声音**当成你的输入 → 无限自我对话，
而且每轮她的声音都会触发"打断"，所以她的回复总是被截断成碎片。
**默认已经是半双工（`BARGE_IN=0`）**：她说话期间麦克风自动闭上，从根本上不可能发生。
想恢复"说话打断"，戴耳机再把 `BARGE_IN=1` 打开。
（程序还留了道保险：连续 3 次识别出她的碎片会**自动切回半双工**并提示。）

**怎么确认语音链路卡在哪一步**
新版默认打印麦克风状态，看这几行就知道：
```
[mic] 检测到你说话（电平 0.0376）          ← VAD 触发了
[mic] 一句说完 3.7s（峰值 0.120）→ 送识别   ← 成功切出一整句
你   ：早上好呀今天也要一起加油我。          ← 识别结果
```
- 只有"检测到你说话"、**没有**"一句说完" → VAD 分不出你和噪声，把 `ENERGY_RATIO` 调大或降低麦克风加强
- 出现 `[mic] 警告：已经连续 X 秒判定为「在说话」` → 同上，环境太吵或增益过高
- 出现 `[跳过] 像是扬声器串回来的我自己的声音` → 外放串音，戴耳机或保持 `BARGE_IN=0`
- 出现 `[麦克风] 采集线程已停止` → 设备被抢/拔出，日志里有具体原因

**识别老听错**
先把 `VAD_MIN_RMS` 调大（0.01~0.02）挡环境噪声，再确认 `--devices` 里输入设备选对。

**打字没反应**
混合模式下打字是后台线程读键盘，输入法候选框里的回车不算提交 —— 用英文输入法敲回车，或者用 `--text` 模式。

**改了 .env 没生效**
`.env` 只在启动时读一次，重启程序。

**退出**
`Ctrl+C`。混合模式下关掉输入流（管道结束）不会退出，会继续用语音模式跑。

---

## 八、下一步可以加什么

- ~~**Live2D / VTube Studio**~~ ✅ 已实现：见「三·十、Live2D 形象（VTube Studio）」。
- **记忆**：现在只带最近 10 轮上下文。想让她记住"上次聊过的事"，加一层向量库
  （`nomic-embed-text` 你本地已经拉了）+ 每轮抽取事实。
- **情绪同步**：SenseVoice 输出里其实带 `<|HAPPY|>` `<|ANGRY|>` 情感标签，现在被剥掉了，可以拿来驱动表情。
- **Kokoro 真流式**：目前"一句一合成"，改成按 KPipeline 的 yield 边出边播，还能再快 ~0.1s。
- **开机自启 + 全局快捷键**：`win+alt+空格` 静音/切人设之类。
