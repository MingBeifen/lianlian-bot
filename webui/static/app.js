/* 恋恋控制台 —— 纯原生 JS，无外部依赖 */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  let data = {};
  try { data = await res.json(); } catch (e) {}
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}
function toast(msg, kind = "") {
  const box = $("#toast-box");
  const el = document.createElement("div");
  el.className = "toast " + kind;
  el.textContent = msg;
  box.appendChild(el);
  setTimeout(() => { el.style.opacity = "0"; el.style.transition = ".4s"; }, 3200);
  setTimeout(() => el.remove(), 3800);
}
const esc = (s) => String(s ?? "");

/* ---------------- 标签页 ---------------- */
$$(".tab-btn").forEach((btn) => btn.addEventListener("click", () => {
  $$(".tab-btn").forEach((b) => b.classList.toggle("active", b === btn));
  $$(".tab-page").forEach((p) => p.classList.toggle("active", p.id === "tab-" + btn.dataset.tab));
  if (btn.dataset.tab === "memory") loadMemories();
  if (btn.dataset.tab === "live2d") loadLive2D();
}));

/* ---------------- 状态 ---------------- */
let lastStatus = null;
function applyStatus(st) {
  lastStatus = st;
  const el = $("#status");
  const ok = st && st.llm;
  el.className = "status " + (ok ? "ok" : "bad");
  el.innerHTML = `<i></i>${ok ? "服务运行中" : "离线"} · 记忆 ${st?.memory?.count ?? "-"} · 浏览器 ${st?.web_clients ?? 0}`;
  const micMute = $("#mic-mute");
  if (st?.mic && document.activeElement !== micMute) micMute.checked = !!st.mic.muted;
}
async function refreshStatus() {
  try { applyStatus(await api("/api/status")); }
  catch (e) { $("#status").className = "status bad"; }
}
refreshStatus();
setInterval(refreshStatus, 5000);

/* ---------------- 聊天 WebSocket ---------------- */
let ws = null, wsRetry = 0, streamEl = null, thinking = false;
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => { wsRetry = 0; };
  ws.onclose = () => {
    wsRetry++;
    setTimeout(connectWS, Math.min(8000, 600 * wsRetry));
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (ev) => {
    let d = {};
    try { d = JSON.parse(ev.data); } catch (e) { return; }
    handleEvent(d);
  };
}
function send(obj) {
  if (ws && ws.readyState === 1) { ws.send(JSON.stringify(obj)); return true; }
  toast("还没连上后端", "err");
  return false;
}
function addMsg(cls, text, who) {
  const div = document.createElement("div");
  div.className = "msg " + cls;
  if (who) {
    const w = document.createElement("span");
    w.className = "who"; w.textContent = who;
    div.appendChild(w);
  }
  const body = document.createElement("span");
  body.className = "body"; body.textContent = text;
  div.appendChild(body);
  $("#messages").appendChild(div);
  $("#messages").scrollTop = $("#messages").scrollHeight;
  return div;
}
function setStatePill(state) {
  const pill = $("#chat-state");
  const map = { thinking: "思考中…", speaking: "说话中…", idle: "空闲" };
  pill.textContent = map[state] || state;
  pill.className = "pill " + state;
}
function handleEvent(d) {
  switch (d.type) {
    case "hello":
    case "status":
      if (d.status) applyStatus(d.status);
      break;
    case "user_text":
      addMsg("user", d.text, d.source === "voice" ? "你（语音）" : "你");
      break;
    case "assistant_sentence":
      if (!streamEl) streamEl = addMsg("her", "", "恋恋");
      streamEl.querySelector(".body").textContent += d.text;
      $("#messages").scrollTop = $("#messages").scrollHeight;
      break;
    case "assistant_done":
      if (streamEl) {
        if (d.text) streamEl.querySelector(".body").textContent = d.text;
        streamEl = null;
      } else if (d.text) addMsg("her", d.text, "恋恋");
      setStatePill("idle");
      break;
    case "proactive":
      addMsg("her", d.text, "恋恋（主动）");
      break;
    case "state":
      setStatePill(d.state);
      break;
    case "error":
      addMsg("error", d.message || "出错了");
      break;
    case "pet":
      toast("已切换 Live2D 模型：" + (d.model || ""));
      break;
  }
}
connectWS();

/* ---------------- 文字发送 ---------------- */
function sendText() {
  const box = $("#chat-input");
  const text = box.value.trim();
  if (!text) return;
  if (send({ type: "text", text })) { box.value = ""; box.style.height = "auto"; }
}
$("#send-btn").addEventListener("click", sendText);
$("#chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendText(); }
});
$("#chat-input").addEventListener("input", (e) => {
  e.target.style.height = "auto";
  e.target.style.height = Math.min(130, e.target.scrollHeight) + "px";
});
$("#btn-interrupt").addEventListener("click", () => send({ type: "interrupt" }));
$("#btn-clear").addEventListener("click", () => { $("#messages").innerHTML = ""; });
$("#mic-mute").addEventListener("change", (e) => send({ type: "mic", muted: e.target.checked }));

/* ---------------- 浏览器实时语音输入 ---------------- */
let micOn = false, audioCtx = null, mediaStream = null, processor = null, sinkGain = null;
let recState = "idle", recChunks = [], speechFrames = 0, silenceFrames = 0, recStart = 0;
const MIC_RMS_ON = 0.025, MIC_RMS_OFF = 0.018, SILENCE_MS = 800, MAX_MS = 15000;

async function toggleMic() { micOn ? stopMic() : await startMic(); }
async function startMic() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const src = audioCtx.createMediaStreamSource(mediaStream);
    processor = audioCtx.createScriptProcessor(4096, 1, 1);
    sinkGain = audioCtx.createGain();
    sinkGain.gain.value = 0;
    src.connect(processor); processor.connect(sinkGain); sinkGain.connect(audioCtx.destination);
    processor.onaudioprocess = (e) => onAudio(e.inputBuffer.getChannelData(0));
    micOn = true;
    $("#mic-btn").classList.add("recording");
    $("#mic-hint").textContent = "正在听…说话即可，停顿约 0.8 秒自动发送";
  } catch (e) {
    toast("麦克风打不开：" + e.message, "err");
  }
}
function stopMic() {
  micOn = false;
  recState = "idle"; recChunks = [];
  try { processor && (processor.onaudioprocess = null); } catch (e) {}
  try { processor && processor.disconnect(); } catch (e) {}
  try { sinkGain && sinkGain.disconnect(); } catch (e) {}
  try { mediaStream && mediaStream.getTracks().forEach((t) => t.stop()); } catch (e) {}
  try { audioCtx && audioCtx.close(); } catch (e) {}
  processor = sinkGain = mediaStream = audioCtx = null;
  $("#mic-btn").classList.remove("recording");
  $("#mic-hint").textContent = "语音输入：点麦克风开始，说完停顿约 0.8 秒自动发送";
}
function onAudio(chunk) {
  if (!micOn) return;
  const rms = Math.sqrt(chunk.reduce((s, v) => s + v * v, 0) / chunk.length);
  const frameMs = (chunk.length / (audioCtx ? audioCtx.sampleRate : 48000)) * 1000;
  if (recState === "idle") {
    speechFrames = rms > MIC_RMS_ON ? speechFrames + 1 : 0;
    if (speechFrames >= 2) { recState = "rec"; recChunks = []; silenceFrames = 0; recStart = Date.now(); recChunks.push(new Float32Array(chunk)); }
    return;
  }
  recChunks.push(new Float32Array(chunk));
  silenceFrames = rms < MIC_RMS_OFF ? silenceFrames + frameMs : 0;
  if (silenceFrames >= SILENCE_MS || Date.now() - recStart >= MAX_MS) finishRec();
}
function finishRec() {
  const total = recChunks.reduce((s, c) => s + c.length, 0);
  recState = "idle";
  if (total < 1600) { recChunks = []; return; }          // 太短当噪声
  const merged = new Float32Array(total);
  let off = 0;
  for (const c of recChunks) { merged.set(c, off); off += c.length; }
  recChunks = [];
  const sr = audioCtx ? audioCtx.sampleRate : 48000;
  const n = Math.floor(merged.length * 16000 / sr);
  const out = new Int16Array(n);
  for (let i = 0; i < n; i++) {
    const pos = i * sr / 16000;
    const i0 = Math.floor(pos), i1 = Math.min(i0 + 1, merged.length - 1);
    const v = merged[i0] + (merged[i1] - merged[i0]) * (pos - i0);
    out[i] = Math.max(-32768, Math.min(32767, Math.round(v * 32767)));
  }
  const bytes = new Uint8Array(out.buffer);
  let bin = "";
  for (let i = 0; i < bytes.length; i += 8192) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 8192));
  if (send({ type: "audio", sample_rate: 16000, pcm: btoa(bin) })) {
    $("#mic-hint").textContent = "已发送，等她回答…";
    setTimeout(() => { if (micOn) $("#mic-hint").textContent = "正在听…说话即可，停顿约 0.8 秒自动发送"; }, 1200);
  }
}
$("#mic-btn").addEventListener("click", toggleMic);
/* ==================== 记忆管理 ==================== */
let memTimer = null;
async function loadMemories() {
  const q = $("#mem-search").value.trim();
  const kind = $("#mem-kind").value;
  try {
    const data = await api(`/api/memories?q=${encodeURIComponent(q)}&kind=${encodeURIComponent(kind)}`);
    renderMemories(data.items || []);
    $("#mem-count").textContent = `（${data.count} 条）`;
  } catch (e) { toast("读取记忆失败：" + e.message, "err"); }
}
function renderMemories(items) {
  const list = $("#memory-list");
  list.innerHTML = "";
  if (!items.length) {
    list.innerHTML = '<div class="empty">还没有记忆。聊聊天让她自己记，或者点「＋ 新增」。</div>';
    return;
  }
  items.forEach((it) => list.appendChild(memoryCard(it)));
}
function memoryCard(it) {
  const card = document.createElement("div");
  card.className = "mem-card";
  card.dataset.id = it.id;
  const created = new Date((it.created_at || 0) * 1000).toLocaleString();
  card.innerHTML = `
    <div class="mem-head">
      <span class="mem-id">#${it.id}</span>
      <span class="mem-kind ${it.kind}">${it.kind}</span>
      <span class="mem-id">${created} · ${it.age || ""}${it.score ? ` · 相关度 ${it.score}` : ""}</span>
    </div>
    <textarea class="mem-body" rows="2"></textarea>
    <div class="mem-foot">
      <div class="weight">权重
        <input type="range" min="0" max="1" step="0.05" value="${it.importance}">
        <b>${Number(it.importance).toFixed(2)}</b>
      </div>
      <div class="mem-actions">
        <button class="btn sm ghost del">删除</button>
        <button class="btn sm primary save">保存</button>
      </div>
    </div>`;
  const body = card.querySelector(".mem-body");
  body.value = it.content;
  const range = card.querySelector("input[type=range]");
  const val = card.querySelector(".weight b");
  const mark = () => card.classList.add("dirty");
  body.addEventListener("input", mark);
  range.addEventListener("input", () => { val.textContent = Number(range.value).toFixed(2); mark(); });
  card.querySelector(".save").addEventListener("click", async () => {
    try {
      await api(`/api/memories/${it.id}`, {
        method: "PATCH",
        body: { content: body.value, importance: parseFloat(range.value) },
      });
      toast(`已保存 #${it.id}`, "ok");
      card.classList.remove("dirty");
      loadMemories();
    } catch (e) { toast("保存失败：" + e.message, "err"); }
  });
  card.querySelector(".del").addEventListener("click", async () => {
    if (!confirm(`删除记忆 #${it.id}？\n\n${body.value.slice(0, 80)}`)) return;
    try {
      await api(`/api/memories/${it.id}`, { method: "DELETE" });
      toast("已删除", "ok");
      loadMemories();
    } catch (e) { toast("删除失败：" + e.message, "err"); }
  });
  return card;
}
$("#mem-add").addEventListener("click", () => {
  if ($("#mem-new")) { $("#mem-new").remove(); return; }
  const card = document.createElement("div");
  card.className = "mem-card dirty";
  card.id = "mem-new";
  card.innerHTML = `
    <div class="mem-head">
      <span class="mem-id">新记忆</span>
      <select class="mem-kind-sel">
        <option value="fact">fact 事实</option>
        <option value="preference">preference 偏好</option>
        <option value="profile">profile 档案</option>
        <option value="event">event 事件</option>
      </select>
    </div>
    <textarea class="mem-body" rows="2" placeholder="要记住的内容，例如：哥哥下周三要考驾照"></textarea>
    <div class="mem-foot">
      <div class="weight">权重
        <input type="range" min="0" max="1" step="0.05" value="0.7">
        <b>0.70</b>
      </div>
      <div class="mem-actions">
        <button class="btn sm ghost cancel">取消</button>
        <button class="btn sm primary save">保存</button>
      </div>
    </div>`;
  const range = card.querySelector("input[type=range]");
  const val = card.querySelector(".weight b");
  range.addEventListener("input", () => (val.textContent = Number(range.value).toFixed(2)));
  card.querySelector(".cancel").addEventListener("click", () => card.remove());
  card.querySelector(".save").addEventListener("click", async () => {
    const content = card.querySelector(".mem-body").value.trim();
    if (!content) return toast("内容不能为空", "err");
    try {
      const data = await api("/api/memories", {
        method: "POST",
        body: { content, kind: card.querySelector(".mem-kind-sel").value,
                importance: parseFloat(range.value) },
      });
      toast(`已${data.action === "updated" ? "更新（合并到已有记忆）" : "新增"} #${data.id}`, "ok");
      loadMemories();
    } catch (e) { toast("保存失败：" + e.message, "err"); }
  });
  $("#memory-list").prepend(card);
});
let memSearchTimer = null;
$("#mem-search").addEventListener("input", () => {
  clearTimeout(memSearchTimer);
  memSearchTimer = setTimeout(loadMemories, 350);
});
$("#mem-kind").addEventListener("change", loadMemories);
$("#mem-refresh").addEventListener("click", loadMemories);
$("#mem-reembed").addEventListener("click", async () => {
  if (!confirm("重新计算所有记忆的向量？\n换过嵌入模型后才需要，条数多会花点时间。")) return;
  try {
    const data = await api("/api/memory/reembed", { method: "POST" });
    toast(`已重算 ${data.count} 条向量`, "ok");
    loadMemories();
  } catch (e) { toast("重算失败：" + e.message, "err"); }
});

/* ==================== Live2D ==================== */
let l2dData = null;
async function loadLive2D() {
  try {
    l2dData = await api("/api/models");
    renderLive2D();
  } catch (e) { toast("读取 Live2D 列表失败：" + e.message, "err"); }
}
function renderLive2D() {
  const cur = l2dData.current?.pet || {};
  $("#l2d-backend").value = cur.backend || "native";
  const grid = $("#l2d-grid");
  grid.innerHTML = "";
  const items = [...(l2dData.catalog.live2d || []),
                 ...(l2dData.catalog.vts || []).map((m) => ({
                   name: m.modelName || m.vtsModelName || "VTS 模型",
                   path: "", vts: true, preview: null,
                   moves: "-", group: "VTube Studio",
                 }))];
  if (!items.length) grid.innerHTML = '<div class="empty">没扫到模型，检查模型目录。</div>';
  items.forEach((m) => {
    const card = document.createElement("div");
    card.className = "l2d-card";
    const isCur = m.path && cur.model && m.path.toLowerCase() === String(cur.model).toLowerCase();
    if (isCur) card.classList.add("selected");
    const img = m.preview ? `<img src="/api/models/live2d/preview?path=${encodeURIComponent(m.preview)}" loading="lazy">`
                          : `<div class="noimg">◕‿◕</div>`;
    card.innerHTML = `${img}
      <div class="badge">${esc(m.group || "")}</div>
      <div class="meta"><b title="${esc(m.name)}">${esc(m.name)}</b>
      <span>${m.vts ? "VTS 模型" : `动作 ${m.motions ?? 0} · 表情 ${m.expressions ?? 0}`}</span></div>`;
    card.addEventListener("click", async () => {
      if (!confirm(`切换到「${m.name}」？\n说话时会短暂重启桌宠渲染器。`)) return;
      try {
        const body = m.vts
          ? { name: m.name, backend: "vtube" }
          : { path: m.path, backend: $("#l2d-backend").value };
        const d = await api("/api/models/live2d", { method: "POST", body });
        toast("已切换：" + (d.model || m.name), "ok");
        loadLive2D();
        refreshStatus();
      } catch (e) { toast("切换失败：" + e.message, "err"); }
    });
    grid.appendChild(card);
  });
  // 桌宠选项
  const wrap = $("#pet-options");
  wrap.innerHTML = "";
  const card = document.createElement("div");
  card.className = "card";
  card.innerHTML = `<h3>桌宠选项<span class="tag">热重启生效</span></h3>
    <div class="desc">这些参数会重启 native 渲染器；VTS 模式只影响窗口设置</div>
    <div class="form-grid">
      <div class="field"><label>窗口大小（例 480x700）</label><input data-k="pet_size" value="${esc(cur.size || "")}"></div>
      <div class="field"><label>位置</label><select data-k="pet_position">
        ${["br", "bl", "tr", "tl", "center", "keep"].map((p) => `<option value="${p}"${p === cur.position ? " selected" : ""}>${p}</option>`).join("")}
      </select></div>
      <div class="field"><label>口型增益</label><input type="number" step="0.5" data-k="native_pet_mouth_gain" value="${esc(cur.mouth_gain ?? 6)}"></div>
      <div class="field"><label>待机动作间隔（秒）</label>
        <input type="number" step="1" data-k="native_pet_motion_gap_min" value="${esc(cur.idle_motion_gap?.[0] ?? 10)}"></div>
      <div class="field"><label class="bool"><input type="checkbox" data-k="native_pet_idle_motion"${cur.idle_motion ? " checked" : ""}> 待机动作</label></div>
      <div class="field"><label class="bool"><input type="checkbox" data-k="native_pet_react_motion"${cur.react_motion ? " checked" : ""}> 情绪反应动作</label></div>
      <div class="field"><label class="bool"><input type="checkbox" data-k="native_pet_action_motion"${cur.action_motion ? " checked" : ""}> LLM 动作标签</label></div>
      <div class="field"><label class="bool"><input type="checkbox" data-k="pet_click_through"${cur.click_through ? " checked" : ""}> 鼠标穿透</label></div>
      <div class="field"><label class="bool"><input type="checkbox" data-k="pet_topmost"${cur.topmost ? " checked" : ""}> 窗口置顶</label></div>
    </div>
    <div class="card-actions"><button class="btn primary apply">应用（重启渲染器）</button><span class="result"></span></div>`;
  card.querySelector(".apply").addEventListener("click", async () => {
    const vals = {};
    card.querySelectorAll("[data-k]").forEach((el) => {
      vals[el.dataset.k] = el.type === "checkbox" ? el.checked
        : el.type === "number" ? parseFloat(el.value) : el.value;
    });
    try {
      await api("/api/models/apply", { method: "POST", body: { section: "pet", values: vals } });
      toast("桌宠选项已应用", "ok");
    } catch (e) { toast("应用失败：" + e.message, "err"); }
  });
  wrap.appendChild(card);
}
$("#l2d-refresh").addEventListener("click", loadLive2D);
$("#l2d-backend").addEventListener("change", (e) => {
  toast(e.target.value === "native" ? "native 模式可网页热切换模型" : "VTS 模式请先在 VTS 里开启 API，重启 run.bat 生效", "");
});

/* ---------------- 初始化 ---------------- */
refreshStatus();