/* Hachimi Sonata · 哈基米音律工坊 · 前端逻辑（无构建步骤） */
"use strict";

const state = {
  config: null,
  session: null,
  analysis: null,
  tracks: [],
  sectionPlans: [],
  autoSectionTemplates: [],
  sectionBaseMarkov: 0.7,
  jobId: null,
  es: null,
  esCloseTimer: null,
  subtitles: [],
  result: null,
  audioUrl: null,
  activeSub: -1,
  ffmpegWarned: false,
  lastAutoDownload: null,
  resultSig: null,
  starting: false,
  paramsList: [],
  pendingParams: null,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const DEFAULT_MARKOV = 0.7;
const AUTO_MARKOV_MIN = 0.38;
const AUTO_MARKOV_MAX = 0.94;
const DEFAULT_SOURCE_WEIGHTS = {
  "哈基米": 0.76,
  "叮咚鸡": 0.16,
  "曼巴": 0.05,
  "曼波only": 0.03,
};
const SOURCE_KEYS = ["哈基米", "叮咚鸡", "曼巴", "曼波only"];
const SOURCE_LABELS = { "哈基米": "哈基米", "叮咚鸡": "叮咚鸡", "曼巴": "曼巴", "曼波only": "曼波" };

/* ------------------------------------------------------------------ */
/* 工具                                                                */
/* ------------------------------------------------------------------ */

function toast(message, kind = "") {
  const wrap = $("#toastWrap");
  const el = document.createElement("div");
  el.className = "toast " + kind;
  el.textContent = message;
  wrap.appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    el.style.transition = "opacity .3s";
    setTimeout(() => el.remove(), 320);
  }, 3200);
}

function fmtTime(t) {
  t = Math.max(0, Number(t) || 0);
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${m}:${s.toFixed(2).padStart(5, "0")}`;
}

function fmtDuration(t) {
  t = Math.max(0, Number(t) || 0);
  const m = Math.floor(t / 60);
  const s = Math.round(t % 60);
  return m > 0 ? `${m}分${s}秒` : `${s}秒`;
}

function fmtTempo(analysis) {
  const changes = (analysis && analysis.tempo_changes) || [];
  const total = Number(
    (analysis && (analysis.tempo_changes_total || (analysis.stats || {}).n_tempo_changes)) || changes.length || 0
  );
  if (total > 1) {
    let lo = Number(analysis.tempo_bpm_min) || 0, hi = Number(analysis.tempo_bpm_max) || 0;
    if (!(lo > 0) || !(hi > 0)) {
      lo = Infinity; hi = 0;
      changes.forEach((c) => {
        const b = Number(c.bpm) || 0;
        if (b > 0) { lo = Math.min(lo, b); hi = Math.max(hi, b); }
      });
    }
    if (!Number.isFinite(lo) || lo <= 0) return `${total} 次变速`;
    const range = Math.abs(hi - lo) < 0.5 ? `${Math.round(lo)}` : `${Math.round(lo)}–${Math.round(hi)}`;
    const first = Number(analysis.tempo) || Number(changes[0] && changes[0].bpm) || lo;
    return `${Math.round(first)} 起 · ${range} BPM · ${total} 次变速`;
  }
  if (analysis && analysis.tempo) return `${Math.round(analysis.tempo)} BPM`;
  return "速度未知";
}

function fmtSize(bytes) {
  if (!bytes) return "";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0, v = bytes;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
}

async function fetchJSON(url, options) {
  const resp = await fetch(url, options);
  let data = null;
  try { data = await resp.json(); } catch (e) { /* ignore */ }
  if (!resp.ok) {
    const detail = data && (data.detail || data.error) ? (data.detail || data.error) : `HTTP ${resp.status}`;
    throw new Error(detail);
  }
  return data;
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

/* ------------------------------------------------------------------ */
/* 初始化                                                              */
/* ------------------------------------------------------------------ */

async function init() {
  await loadConfig();
  bindEvents();
  drawEmptyCanvas();
  loadLibrary();
  loadParamsList();
  const savedMode =
    localStorage.getItem("hachimi-sonata.downloadMode") ||
    localStorage.getItem("hachimi.downloadMode");
  if (savedMode && $("#downloadMode")) $("#downloadMode").value = savedMode;
}

async function loadConfig() {
  try {
    const cfg = await fetchJSON("/api/config");
    state.config = cfg;
    const modelSel = $("#modelSelect");
    const voiceSel = $("#globalVoice");
    modelSel.innerHTML = "";
    voiceSel.innerHTML = "";
    cfg.models.forEach((m) => {
      const o = el("option", null, m.label);
      o.value = m.value;
      modelSel.appendChild(o);
    });
    cfg.voices.forEach((v) => {
      const o = el("option", null, v.label);
      o.value = v.value;
      voiceSel.appendChild(o);
    });
    modelSel.value = cfg.defaults.model || "markov";
    voiceSel.value = cfg.defaults.voice || "auto";
    $("#gammaInput").value = cfg.defaults.velocity_gamma || 1.6;
    $("#gammaVal").textContent = Number($("#gammaInput").value).toFixed(1);
    $("#workersInput").value = cfg.defaults.workers ?? 0;
    if (cfg.sample_sets && $("#sampleSet")) {
      const sampleSel = $("#sampleSet");
      sampleSel.innerHTML = "";
      cfg.sample_sets.forEach((s) => {
        const o = el("option", null, s.label);
        o.value = String(s.value);
        sampleSel.appendChild(o);
      });
      sampleSel.value = String(cfg.defaults.sample_set || 1);
    }
    if (cfg.source_dir && $("#sourceDir")) $("#sourceDir").textContent = cfg.source_dir;
    if ($("#syntaxInput")) {
      $("#syntaxInput").value = cfg.defaults.syntax_weight ?? 0.3;
      updateSyntaxLabels();
    }
    if ($("#gpuMode")) $("#gpuMode").value = cfg.defaults.gpu || "auto";
    if ($("#gpuMemory")) $("#gpuMemory").value = String(cfg.defaults.gpu_memory_mb || 2048);
    if ($("#autoMixInput")) $("#autoMixInput").checked = cfg.defaults.auto_mix !== false;
    if ($("#premixInput")) $("#premixInput").checked = cfg.defaults.write_premix !== false;
    if ($("#badNotesInput")) $("#badNotesInput").checked = cfg.defaults.write_bad_note_midis !== false;
    if ($("#videosInput")) {
      $("#videosInput").checked = cfg.defaults.write_subtitle_videos !== false;
    }
    if ($("#subtitleFontSize")) {
      $("#subtitleFontSize").value = String(cfg.defaults.subtitle_video_font_size || 8);
      $("#subtitleFontFamily").value = cfg.defaults.subtitle_video_font_family || "SimSun";
      $("#subtitleTextColor").value = cfg.defaults.subtitle_video_text_color || "#FFFFFF";
      updateSubtitleStyle();
    }
    const gpuHint = $("#gpuHint");
    if (gpuHint) {
      if (cfg.gpu && cfg.gpu.available) {
        gpuHint.textContent = `GPU：${cfg.gpu.name}（${cfg.gpu.total_mb} MB，当前可用 ${cfg.gpu.free_mb} MB）· 仅用于混音落位`;
      } else {
        gpuHint.textContent = "未检测到可用 CUDA GPU，将使用 CPU（功能不受影响）";
      }
    }
  } catch (e) {
    toast("无法读取服务配置：" + e.message, "error");
  }
}

function bindEvents() {
  const input = $("#fileInput");
  $("#browseBtn").addEventListener("click", () => input.click());
  $("#changeFileBtn").addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files && input.files[0]) uploadFile(input.files[0]);
    input.value = "";
  });

  const drop = $("#dropCard");
  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("dragover"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("dragover"); })
  );
  drop.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) uploadFile(file);
  });

  $("#localBtn").addEventListener("click", openLocalModal);
  $("#closeLocal").addEventListener("click", () => { $("#localModal").hidden = true; });
  $("#localModal").addEventListener("click", (e) => {
    if (e.target.id === "localModal") $("#localModal").hidden = true;
  });
  $("#localSearch").addEventListener("input", filterLocalList);
  $("#refreshLibrary").addEventListener("click", loadLibrary);
  const libraryCard = $("#libraryCard");
  if (libraryCard) {
    libraryCard.open = localStorage.getItem("hachimi-sonata.libraryOpen") === "1";
    libraryCard.addEventListener("toggle", () => {
      localStorage.setItem("hachimi-sonata.libraryOpen", libraryCard.open ? "1" : "0");
    });
  }
  const paramsCard = $("#paramsCard");
  if (paramsCard) {
    paramsCard.open = localStorage.getItem("hachimi-sonata.paramsOpen") === "1";
    paramsCard.addEventListener("toggle", () => {
      localStorage.setItem("hachimi-sonata.paramsOpen", paramsCard.open ? "1" : "0");
    });
  }
  $("#paramsApply").addEventListener("click", applySelectedParams);
  $("#paramsRefresh").addEventListener("click", loadParamsList);
  $("#paramsImportBtn").addEventListener("click", () => $("#paramsFile").click());
  $("#paramsFile").addEventListener("change", () => {
    const file = $("#paramsFile").files && $("#paramsFile").files[0];
    if (file) importParamsFile(file);
    $("#paramsFile").value = "";
  });

  $("#helpBtn").addEventListener("click", () => { $("#helpModal").hidden = false; });
  $("#closeHelp").addEventListener("click", () => { $("#helpModal").hidden = true; });
  $("#helpModal").addEventListener("click", (e) => {
    if (e.target.id === "helpModal") $("#helpModal").hidden = true;
  });

  $("#gammaInput").addEventListener("input", () => {
    $("#gammaVal").textContent = Number($("#gammaInput").value).toFixed(1);
  });

  $("#syntaxInput").addEventListener("input", updateSyntaxLabels);
  $("#syntaxInput").addEventListener("change", rebuildAutoSectionPlans);
  if ($("#subtitleFontSize")) {
    $("#subtitleFontSize").addEventListener("input", updateSubtitleStyle);
  }
  $("#seedInput").addEventListener("change", () => {
    const base = Number($("#seedInput").value) || 1337;
    state.sectionPlans.forEach((plan) => {
      if (plan.auto) plan.seed = base + plan.section_index * 7919;
    });
    renderSectionPlans();
  });
  $("#refreshSamples").addEventListener("click", refreshSamples);
  $("#resetSectionPlans").addEventListener("click", resetAllSectionPlans);

  $("#downloadMode").addEventListener("change", () => {
    localStorage.setItem("hachimi-sonata.downloadMode", $("#downloadMode").value);
  });

  $("#selectAll").addEventListener("click", () => setAllSelected(true, "all"));
  $("#selectNone").addEventListener("click", () => setAllSelected(false, "all"));
  $("#selectMelody").addEventListener("click", () => setAllSelected(true, "melody"));

  $("#globalVoice").addEventListener("change", () => {
    const mode = $("#globalVoice").value;
    state.tracks.forEach((t) => { t.voice = mode; });
    renderTracks();
  });

  $("#renderBtn").addEventListener("click", startRender);
  $("#cancelBtn").addEventListener("click", cancelRender);

  $("#analysisTabs").addEventListener("click", (e) => {
    const tab = e.target.closest(".tab");
    if (!tab) return;
    $$("#analysisTabs .tab").forEach((t) => t.classList.toggle("active", t === tab));
    ["chords", "sections", "motifs"].forEach((name) => {
      $("#tab-" + name).hidden = name !== tab.dataset.tab;
    });
  });

  window.addEventListener("resize", debounce(() => {
    if (state.analysis) drawOverview(state.analysis);
  }, 180));

  $("#audioPlayer").addEventListener("timeupdate", () => {
    syncSubtitleToAudio($("#audioPlayer").currentTime);
  });
}

function debounce(fn, delay) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

function updateSyntaxLabels() {
  const value = Number($("#syntaxInput").value) || 0;
  $("#syntaxVal").textContent = Math.round(value * 100) + "%";
  $("#markovVal").textContent = Math.round((1 - value) * 100) + "%";
}

function updateSubtitleStyle() {
  if (!$("#subtitleFontSize")) return;
  const units = Math.max(1, Number($("#subtitleFontSize").value) || 8);
  const out = $("#subtitleFontSizeOut");
  if (out) out.textContent = `${units} 号 · ${Math.round(units * 10)} px · 轻微黑边`;
}

async function refreshSamples() {
  const btn = $("#refreshSamples");
  if (btn.disabled) return;
  btn.disabled = true;
  btn.textContent = "刷新中…";
  try {
    const data = await fetchJSON("/api/samples/refresh", { method: "POST" });
    toast(`音源已刷新：${data.files} 个采样`, "ok");
    appendMiniLog(`音源已刷新：${data.source_dir}`);
  } catch (e) {
    toast("刷新音源失败：" + e.message, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "刷新";
  }
}

/* ------------------------------------------------------------------ */
/* 段落编曲（每段独立）                                                */
/* ------------------------------------------------------------------ */

function currentBaseMarkov() {
  const syntax = Number($("#syntaxInput").value);
  return Math.max(0, Math.min(1, 1 - (Number.isFinite(syntax) ? syntax : 0.3)));
}

function clampMarkov(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.max(0, Math.min(1, n)) : (fallback ?? DEFAULT_MARKOV);
}

function makeAutoPlan(section, index, markov) {
  return {
    section_index: index,
    label: section.label || String.fromCharCode(65 + (index % 26)),
    start: section.start || 0,
    end: section.end || 0,
    key: section.key || "",
    energy: section.energy || 0,
    progression: section.progression || [],
    phrase_count: section.phrase_count || 0,
    character: section.character || "",
    markov_ratio: clampMarkov(markov == null ? currentBaseMarkov() : markov),
    seed: 1337 + index * 7919,
    source_weights: { ...DEFAULT_SOURCE_WEIGHTS },
    auto: true,
    source: "auto",
  };
}

function normalizeSectionPlans(rawPlans, sections) {
  const source = Array.isArray(rawPlans) ? rawPlans : [];
  const byIndex = new Map();
  source.forEach((plan, i) => {
    const idx = Number(plan.section_index ?? plan.index ?? i);
    byIndex.set(Number.isFinite(idx) ? idx : i, plan);
  });
  const result = [];
  (sections || []).forEach((section, i) => {
    const plan = byIndex.get(i) || makeAutoPlan(section, i, currentBaseMarkov());
    const weights = { ...DEFAULT_SOURCE_WEIGHTS, ...(plan.source_weights || plan.weights || {}) };
    const rawMarkov = plan.markov_ratio ?? (plan.syntax_weight != null ? 1 - Number(plan.syntax_weight) : currentBaseMarkov());
    result.push({
      ...makeAutoPlan(section, i, rawMarkov),
      ...plan,
      section_index: i,
      label: section.label || plan.label || String.fromCharCode(65 + (i % 26)),
      start: section.start ?? plan.start ?? 0,
      end: section.end ?? plan.end ?? 0,
      key: section.key || plan.key || "",
      progression: section.progression || plan.progression || [],
      phrase_count: section.phrase_count || plan.phrase_count || 0,
      markov_ratio: clampMarkov(rawMarkov, currentBaseMarkov()),
      seed: Number(plan.seed ?? (1337 + i * 7919)),
      source_weights: weightMap(weights),
      auto: plan.auto !== false,
    });
  });
  if (!result.length) {
    result.push(makeAutoPlan({ label: "A", start: 0, end: state.analysis?.duration || 0 }, 0));
  }
  return result;
}

function weightMap(weights) {
  const out = {};
  SOURCE_KEYS.forEach((key) => {
    const value = Number(weights[key] ?? 0);
    out[key] = Number.isFinite(value) ? Math.max(0, value) : 0;
  });
  const total = SOURCE_KEYS.reduce((sum, key) => sum + out[key], 0);
  if (total <= 1e-9) return { ...DEFAULT_SOURCE_WEIGHTS };
  SOURCE_KEYS.forEach((key) => { out[key] = out[key] / total; });
  return out;
}

function renderSectionPlans() {
  const box = $("#sectionPlanList");
  const card = $("#arrangementCard");
  if (!box || !card) return;
  box.innerHTML = "";
  if (!state.sectionPlans.length) {
    card.hidden = true;
    return;
  }
  state.sectionPlans.forEach((plan, index) => {
    const row = el("div", "section-plan-row");
    row.dataset.index = String(index);

    const head = el("div", "section-plan-head");
    const left = el("div", "section-plan-title");
    left.appendChild(el("span", "tag", plan.label || "A"));
    left.appendChild(document.createTextNode(`  ${fmtTime(plan.start)} - ${fmtTime(plan.end)}`));
    if (plan.key) left.appendChild(el("span", "plan-key", plan.key));
    if (plan.progression && plan.progression.length) {
      left.appendChild(el("span", "plan-progression", plan.progression.slice(0, 8).join(" · ")));
    }
    if (plan.phrase_count) left.appendChild(el("span", "plan-phrase", `${plan.phrase_count} 句`));
    head.appendChild(left);

    const right = el("div", "section-plan-actions");
    const badge = el("span", "plan-badge " + (plan.auto ? "auto" : "manual"), plan.auto ? "自动" : "手动");
    right.appendChild(badge);
    const reset = el("button", "ghost-btn tiny", "恢复自动");
    reset.type = "button";
    reset.addEventListener("click", () => {
      state.sectionPlans[index] = makeAutoPlan(sectionOfPlan(plan), index, currentBaseMarkov());
      renderSectionPlans();
    });
    right.appendChild(reset);
    head.appendChild(right);
    row.appendChild(head);

    const body = el("div", "section-plan-body");

    // 马尔科夫比例
    const markovField = el("label", "plan-field");
    markovField.appendChild(document.createTextNode("马尔科夫"));
    const markovOut = el("output", "plan-out", `${Math.round(plan.markov_ratio * 100)}%`);
    markovField.appendChild(markovOut);
    const markov = document.createElement("input");
    markov.type = "range"; markov.min = "0"; markov.max = "100"; markov.step = "1";
    markov.value = String(Math.round(clampMarkov(plan.markov_ratio) * 100));
    markov.addEventListener("input", () => {
      plan.markov_ratio = Number(markov.value) / 100;
      markovOut.textContent = markov.value + "%";
      plan.auto = false;
      plan.source = "manual";
      badge.className = "plan-badge manual";
      badge.textContent = "手动";
    });
    markovField.appendChild(markov);
    body.appendChild(markovField);

    // 随机种子
    const seedField = el("label", "plan-field");
    seedField.appendChild(document.createTextNode("随机种子"));
    const seed = document.createElement("input");
    seed.type = "number"; seed.min = "0"; seed.max = "2147483647"; seed.step = "1";
    seed.value = String(plan.seed);
    seed.addEventListener("change", () => {
      plan.seed = Math.max(0, Math.min(2147483647, Number(seed.value) || 0));
      seed.value = String(plan.seed);
      plan.auto = false;
      plan.source = "manual";
      badge.className = "plan-badge manual";
      badge.textContent = "手动";
    });
    seedField.appendChild(seed);
    body.appendChild(seedField);

    // 四个音源配比
    SOURCE_KEYS.forEach((key) => {
      const field = el("label", "plan-field weight-field");
      field.appendChild(document.createTextNode(SOURCE_LABELS[key]));
      const input = document.createElement("input");
      input.type = "number"; input.min = "0"; input.max = "100"; input.step = "1";
      input.value = String(Math.round((plan.source_weights[key] || 0) * 100));
      input.addEventListener("change", () => {
        const raw = { ...plan.source_weights };
        raw[key] = Math.max(0, Number(input.value) || 0) / 100;
        plan.source_weights = weightMap(raw);
        plan.auto = false;
        plan.source = "manual";
        badge.className = "plan-badge manual";
        badge.textContent = "手动";
        refreshWeightInputs(row, plan);
      });
      field.appendChild(input);
      field.dataset.sourceKey = key;
      body.appendChild(field);
    });
    row.appendChild(body);

    const contrast = Number(plan.auto_contrast);
    const meta = el(
      "div",
      "plan-meta",
      [
        plan.character,
        plan.energy ? `能量 ${Math.round(plan.energy * 100)}%` : "",
        Number.isFinite(contrast) && contrast > 0 ? `段间对比 ${Math.round(contrast * 100)}` : "",
      ]
        .filter(Boolean).join(" · ") || "按分析结果自动匹配"
    );
    row.appendChild(meta);
    box.appendChild(row);
  });
  card.hidden = false;
  $("#sectionPlanHint").textContent =
    `共 ${state.sectionPlans.length} 段 · 全局基线 ${Math.round(currentBaseMarkov() * 100)}% 马尔科夫 · ` +
    `自动段已拉开段间差异（${Math.round(AUTO_MARKOV_MIN * 100)}–${Math.round(AUTO_MARKOV_MAX * 100)}%）；` +
    `哈基米默认主导，叮咚鸡辅助；修改后立即生效`;
}

function sectionOfPlan(plan) {
  return {
    label: plan.label,
    start: plan.start,
    end: plan.end,
    key: plan.key,
    energy: plan.energy,
    progression: plan.progression,
    phrase_count: plan.phrase_count,
    character: plan.character,
  };
}

function refreshWeightInputs(row, plan) {
  row.querySelectorAll(".weight-field").forEach((field) => {
    const key = field.dataset.sourceKey;
    const input = field.querySelector("input");
    if (input) input.value = String(Math.round((plan.source_weights[key] || 0) * 100));
  });
}

function templateForSection(index) {
  return (state.autoSectionTemplates || []).find(
    (item) => Number(item.section_index ?? item.index) === Number(index)
  );
}

function rebuildAutoSectionPlans() {
  if (!state.sectionPlans.length) { updateSyntaxLabels(); return; }
  const newBase = currentBaseMarkov();
  const oldBase = Number.isFinite(Number(state.sectionBaseMarkov))
    ? Number(state.sectionBaseMarkov)
    : newBase;
  const delta = newBase - oldBase;
  state.sectionPlans.forEach((plan, index) => {
    if (!plan.auto) return;
    const template = templateForSection(index);
    if (template) {
      // 保留后端算出的段间差异，只把整套计划随全局基线平移。
      plan.markov_ratio = clampMarkov(Number(template.markov_ratio ?? plan.markov_ratio) + delta);
      plan.seed = Number(template.seed ?? plan.seed);
      plan.source_weights = weightMap(template.source_weights || plan.source_weights);
      plan.auto_contrast = template.auto_contrast;
    } else {
      plan.markov_ratio = clampMarkov(plan.markov_ratio + delta);
    }
  });
  state.sectionBaseMarkov = newBase;
  updateSyntaxLabels();
  renderSectionPlans();
}

function resetAllSectionPlans() {
  if (!state.sectionPlans.length) return;
  const newBase = currentBaseMarkov();
  const oldBase = Number.isFinite(Number(state.sectionBaseMarkov))
    ? Number(state.sectionBaseMarkov)
    : newBase;
  const delta = newBase - oldBase;
  state.sectionPlans = state.sectionPlans.map((plan, index) => {
    const template = templateForSection(index);
    if (!template) return makeAutoPlan(sectionOfPlan(plan), index, newBase);
    return {
      ...template,
      section_index: index,
      label: plan.label || template.label,
      start: plan.start ?? template.start,
      end: plan.end ?? template.end,
      key: plan.key || template.key,
      energy: plan.energy ?? template.energy,
      progression: plan.progression || template.progression || [],
      phrase_count: plan.phrase_count || template.phrase_count || 0,
      character: plan.character || template.character || "",
      markov_ratio: clampMarkov(Number(template.markov_ratio ?? newBase) + delta),
      source_weights: weightMap(template.source_weights || DEFAULT_SOURCE_WEIGHTS),
      auto: true,
      source: "auto",
    };
  });
  state.sectionBaseMarkov = newBase;
  renderSectionPlans();
  toast("已恢复自动段落计划；每段的随机比例差异保留", "ok");
}

/* ------------------------------------------------------------------ */
/* 工程参数复用                                                        */
/* ------------------------------------------------------------------ */

async function loadParamsList() {
  const sel = $("#paramsSelect");
  if (!sel) return;
  try {
    const data = await fetchJSON("/api/params/list");
    state.paramsList = data.items || [];
    const count = $("#paramsCount");
    if (count) count.textContent = String(state.paramsList.length);
    sel.innerHTML = "";
    if (!state.paramsList.length) {
      const o = el("option", null, "还没有可复用的参数包");
      o.value = "";
      sel.appendChild(o);
      $("#paramsHint").textContent = "生成一次作品后，会自动出现参数包";
      return;
    }
    state.paramsList.forEach((item) => {
      const o = el(
        "option",
        null,
        `${item.project || "未命名"}${item.legacy_only ? "（旧作品 manifest）" : ""}`
      );
      o.value = item.read_url || "";
      o.title = [item.created_at, item.file_name, item.voice, `${item.tracks || 0} 轨`]
        .filter(Boolean).join(" · ");
      sel.appendChild(o);
    });
    $("#paramsHint").textContent = `已发现 ${state.paramsList.length} 个参数包；选择后点「读取并套用」`;
  } catch (e) {
    sel.innerHTML = "<option value=''>读取失败</option>";
    $("#paramsHint").textContent = "参数包列表读取失败：" + e.message;
  }
}

async function applySelectedParams() {
  const sel = $("#paramsSelect");
  const url = sel && sel.value;
  if (!url) { toast("请选择一个工程参数包", "error"); return; }
  const btn = $("#paramsApply");
  btn.disabled = true;
  btn.textContent = "读取中…";
  try {
    const data = await fetchJSON(url);
    applyParamsPayload(data.params || data, data.source || "");
  } catch (e) {
    toast("读取参数包失败：" + e.message, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "读取并套用";
  }
}

async function importParamsFile(file) {
  if (!/\.(zip|json)$/i.test(file.name || "")) {
    toast("请选择 .zip 或 .json 参数包", "error");
    return;
  }
  if (file.size > 32 * 1024 * 1024) {
    toast("参数包超过 32 MB", "error");
    return;
  }
  const btn = $("#paramsImportBtn");
  btn.disabled = true;
  btn.textContent = "导入中…";
  try {
    const resp = await fetch(`/api/params/import?name=${encodeURIComponent(file.name)}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file,
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || data.error || `HTTP ${resp.status}`);
    applyParamsPayload(data.params || data, file.name);
  } catch (e) {
    toast("导入参数包失败：" + e.message, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "导入参数包 ZIP / JSON";
  }
}

function applyParamsPayload(payload, sourceLabel) {
  const params = (payload && payload.params) || payload || {};
  const opt = params.options || {};
  const tracks = params.tracks || opt.tracks || [];
  const has = (key) => Object.prototype.hasOwnProperty.call(opt, key);
  const setVal = (selector, value) => {
    const node = $(selector);
    if (node && value !== undefined && value !== null) node.value = String(value);
  };
  const setCheck = (selector, value) => {
    const node = $(selector);
    if (node && value !== undefined) node.checked = Boolean(value);
  };

  if (has("model_kind")) setVal("#modelSelect", opt.model_kind);
  if (has("voice")) setVal("#globalVoice", opt.voice);
  if (has("sample_set")) setVal("#sampleSet", opt.sample_set);
  if (has("seed")) setVal("#seedInput", opt.seed);
  if (has("syntax_weight")) setVal("#syntaxInput", opt.syntax_weight);
  if (has("velocity_gamma")) setVal("#gammaInput", opt.velocity_gamma);
  if (has("workers")) setVal("#workersInput", opt.workers);
  if (has("gpu")) setVal("#gpuMode", opt.gpu);
  if (has("gpu_memory_mb")) setVal("#gpuMemory", opt.gpu_memory_mb);
  if (has("max_seconds")) setVal("#maxSeconds", opt.max_seconds == null ? "" : opt.max_seconds);
  setCheck("#autoMixInput", opt.auto_mix);
  setCheck("#stemsInput", opt.write_stems);
  setCheck("#premixInput", opt.write_premix);
  setCheck("#badNotesInput", opt.write_bad_note_midis);
  setCheck("#zipInput", opt.write_zip);
  setCheck("#videosInput", opt.write_subtitle_videos);
  setCheck("#decomposedInput", opt.decomposed_merge);
  if (has("subtitle_video_font_size")) setVal("#subtitleFontSize", opt.subtitle_video_font_size);
  if (has("subtitle_video_font_family")) setVal("#subtitleFontFamily", opt.subtitle_video_font_family);
  if (has("subtitle_video_text_color")) setVal("#subtitleTextColor", opt.subtitle_video_text_color);
  updateSyntaxLabels();
  updateSubtitleStyle();
  state.sectionBaseMarkov = currentBaseMarkov();

  if (!state.session || !state.analysis) {
    state.pendingParams = params;
    $("#paramsHint").textContent =
      `已暂存「${params.project || sourceLabel || "参数包"}」；导入 MIDI / 乐谱后自动套用到轨道与段落`;
    toast("参数已暂存；导入文件后会自动套用轨道 / 段落设置", "ok");
    return;
  }

  const byIndex = new Map((Array.isArray(tracks) ? tracks : []).map((t) => [Number(t.index), t]));
  const selectedSet = new Set((params.selected || []).map((n) => Number(n)));
  let appliedTracks = 0;
  state.tracks.forEach((track) => {
    const item = byIndex.get(Number(track.index));
    if (!item) return;
    if (Array.isArray(params.selected) && params.selected.length) {
      track.selected = selectedSet.has(Number(track.index));
    } else if (item.selected !== undefined) {
      track.selected = Boolean(item.selected);
    }
    if (item.voice) track.voice = item.voice;
    if (item.octave !== undefined) track.octave = Math.max(-3, Math.min(3, Number(item.octave) || 0));
    if (item.gain !== undefined) track.gain = Math.max(0, Math.min(2, Number(item.gain)));
    appliedTracks += 1;
  });
  renderTracks();
  updateTrackSummary();

  const plans = opt.section_plans || params.section_plans || [];
  if (plans.length && state.sectionPlans.length && state.analysis) {
    state.sectionPlans = normalizeSectionPlans(plans, state.analysis.sections || []);
    renderSectionPlans();
  }

  const label = params.project || sourceLabel || "参数包";
  $("#paramsHint").textContent =
    `已套用「${label}」：${appliedTracks} 条匹配轨道，` +
    `${plans.length || 0} 段编曲，字幕 ${opt.subtitle_video_font_size || 8} 号`;
  toast(
    `参数已套用：${appliedTracks} 条轨道${plans.length ? ` / ${plans.length} 段编曲` : ""}` +
    (params.legacy ? "（旧作品，部分参数缺失）" : ""),
    "ok"
  );
}

/* ------------------------------------------------------------------ */
/* 上传与分析                                                          */
/* ------------------------------------------------------------------ */

function uploadFile(file) {
  const name = file.name || "upload.mid";
  if (!/\.(mid|midi|kar|rmi|xml|musicxml|mxl)$/i.test(name)) {
    toast("请选择 .mid / .midi / .kar / .rmi / .xml / .musicxml / .mxl 文件", "error");
    return;
  }
  const maxMb = (state.config && state.config.max_upload_mb) || 64;
  if (file.size > maxMb * 1024 * 1024) {
    toast(`文件超过 ${maxMb} MB`, "error");
    return;
  }

  const progress = $("#dropProgress");
  const bar = progress.querySelector("div");
  progress.hidden = false;
  bar.style.width = "4%";
  $("#fileName").textContent = name;
  $("#fileMeta").textContent = `正在上传 / 分析… ${fmtSize(file.size)}`;

  const xhr = new XMLHttpRequest();
  xhr.open("POST", `/api/analyze?name=${encodeURIComponent(name)}`);
  xhr.responseType = "json";
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) bar.style.width = `${Math.min(90, (e.loaded / e.total) * 90)}%`;
  };
  xhr.onload = () => {
    bar.style.width = "100%";
    setTimeout(() => { progress.hidden = true; bar.style.width = "0"; }, 350);
    if (xhr.status >= 200 && xhr.status < 300) {
      handleAnalyze(xhr.response);
    } else {
      const msg = (xhr.response && (xhr.response.detail || xhr.response.error)) || `HTTP ${xhr.status}`;
      toast("分析失败：" + msg, "error");
      $("#fileMeta").textContent = "分析失败，请重试";
    }
  };
  xhr.onerror = () => {
    progress.hidden = true;
    toast("上传失败，请检查服务是否运行", "error");
  };
  xhr.send(file);
}

function handleAnalyze(data) {
  // 重新导入时停掉旧任务的事件流，避免旧结果覆盖新分析
  closeES();
  if (state.jobId) {
    fetchJSON(`/api/jobs/${state.jobId}/cancel`, { method: "POST" }).catch(() => {});
    state.jobId = null;
  }
  state.starting = false;
  state.resultSig = null;
  state.lastAutoDownload = null;
  const oldAudio = $("#audioPlayer");
  oldAudio.pause();
  oldAudio.removeAttribute("src");

  state.session = { id: data.session_id, filename: data.filename, path: data.path };
  state.analysis = data.analysis;
  state.analysis.overview = data.overview || { pitch: [], density: [] };
  state.result = null;
  state.subtitles = [];
  state.activeSub = -1;

  const suggested = new Set(
    data.analysis.tracks.filter((t) => t.suggested_selected && !t.is_drum).map((t) => t.index)
  );
  const chosen = new Set((data.default_selected && data.default_selected.length) ? data.default_selected : Array.from(suggested));

  state.autoSectionTemplates = JSON.parse(JSON.stringify(data.section_plans || []));
  state.sectionBaseMarkov = currentBaseMarkov();
  state.tracks = data.analysis.tracks.map((t) => ({
    index: t.index,
    name: t.name,
    instrument: t.instrument,
    role: t.role,
    role_label: t.role_label,
    n_notes: t.n_notes,
    avg_velocity: t.avg_velocity,
    pitch_min: t.pitch_min,
    pitch_max: t.pitch_max,
    is_drum: t.is_drum,
    reason: t.reason,
    selected: chosen.has(t.index),
    voice: t.suggested_voice || "auto",
    octave: t.suggested_octave || 0,
    gain: t.suggested_gain != null ? t.suggested_gain : 1,
  }));
  state.sectionPlans = normalizeSectionPlans(data.section_plans, data.analysis.sections || []);

  renderFileCard();
  renderTracks();
  renderAnalysis();
  renderSectionPlans();
  $("#fileCard").hidden = false;
  $("#trackCard").hidden = false;
  $("#analysisCard").hidden = false;
  $("#resultCard").hidden = true;
  $("#subtitleCard").hidden = true;
  $("#renderBtn").disabled = false;
  $("#progressBox").hidden = true;
  $("#cancelBtn").hidden = true;
  $("#renderBtn").textContent = "开始生成";
  drawOverview(state.analysis);
  // 布局稳定后再画一次，确保概览曲线不因首帧宽度/数据未就绪而空白
  setTimeout(() => {
    if (state.analysis === data.analysis) drawOverview(state.analysis);
  }, 250);
  updateTrackSummary();
  if (state.pendingParams) {
    const pending = state.pendingParams;
    state.pendingParams = null;
    applyParamsPayload(pending, "暂存参数");
  }
  toast(`分析完成：${data.analysis.stats.n_tracks} 轨 / ${data.analysis.stats.n_chords} 个和弦窗`, "ok");
  $("#fileCard").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderFileCard() {
  const a = state.analysis;
  const stats = a.stats || {};
  $("#fileName").textContent = state.session.filename;
  $("#fileMeta").textContent = `${fmtDuration(a.duration)} · ${stats.n_notes || 0} 音符 · ${fmtTempo(a)}`;
  const grid = $("#statGrid");
  grid.innerHTML = "";
  const isScore = a.source_format === "musicxml" || a.source_format === "mxl";
  const changes = a.tempo_changes || [];
  const tempoTotal = Number(a.tempo_changes_total || stats.n_tempo_changes || changes.length || 0);
  const tempoTitleParts = changes.slice(0, 24).map(
    (c) => `${fmtTime(c.time)}  ${c.approx ? "≈" : ""}${Number(c.bpm).toFixed(1)} BPM${c.label ? "  " + c.label : ""}`
  );
  if (tempoTotal > Math.min(changes.length, 24)) tempoTitleParts.push(`…共 ${tempoTotal} 处`);
  const tempoTitle = tempoTitleParts.join("\n");
  const items = [
    ["时长", fmtDuration(a.duration)],
    ["调性", a.key?.label_cn || a.key?.label || stats.key || "—"],
    ["轨道", `${stats.n_tracks || 0}`],
    ["音符", `${stats.n_notes || 0}`],
    ["速度变化", `${tempoTotal || 1} 次`, tempoTitle],
    ["和弦窗", `${stats.n_chords || 0}`],
    ["分解和弦", `${stats.decomposed_chords || 0}`],
    ["段落", `${stats.n_sections || 0}`],
    ["乐句", `${stats.n_phrases || 0}`],
    ["动机", `${stats.n_motifs || 0}`],
  ];
  if (isScore) {
    items.push(["力度标记", `${stats.n_dynamics || 0}`, "MusicXML 力度 / 发夹数量"]);
    items.push(["奏法标记", `${stats.n_articulations || 0}`, "连音线、断奏、重音、tremolo 等"]);
  }
  items.forEach(([k, v, title]) => {
    const box = el("div", "stat");
    if (title) box.title = title;
    box.appendChild(el("div", "k", k));
    box.appendChild(el("div", "v", v));
    grid.appendChild(box);
  });
  renderTempoMap(a);
}

function renderTempoMap(a) {
  const box = $("#tempoMap");
  const list = $("#tempoList");
  if (!box || !list) return;
  const changes = a.tempo_changes || [];
  const total = Number(a.tempo_changes_total || (a.stats || {}).n_tempo_changes || changes.length || 0);
  const compressed = Boolean(a.tempo_changes_compressed);
  list.innerHTML = "";
  if (!changes.length) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  $("#tempoHint").textContent = compressed
    ? `${total} 处 · 密集自动化，显示 ${changes.length} 个代表点`
    : `${total || changes.length} 处 · 按谱面位置`;

  const maxChips = 28;
  changes.slice(0, maxChips).forEach((c) => {
    const chip = el("span", "tempo-chip");
    const bpm = Number(c.bpm) || 0;
    chip.appendChild(el("b", null, `${c.approx ? "≈" : ""}${bpm.toFixed(1).replace(/\.0$/, "")}`));
    chip.appendChild(document.createTextNode(" BPM"));
    chip.appendChild(el("span", null, fmtTime(c.time).slice(0, -3)));
    if (c.label) chip.title = c.label;
    list.appendChild(chip);
  });
  if (changes.length > maxChips) {
    const more = el("span", "tempo-chip more", `… 另 ${changes.length - maxChips} 个`);
    more.title = "完整速度明细见下表";
    list.appendChild(more);
  }

  const body = $("#tempoDetailBody");
  const detail = $("#tempoDetails");
  if (!body || !detail) return;
  body.innerHTML = "";
  changes.forEach((c, i) => {
    const tr = document.createElement("tr");
    const idx = el("td", null, `${i + 1}`);
    const time = el("td", null, fmtTime(c.time));
    const bpmCell = el("td", c.approx ? "approx" : null, `${c.approx ? "≈" : ""}${Number(c.bpm).toFixed(1)}`);
    const label = el("td", "tempo-label", c.label || (c.approx ? `窗口 ${Number(c.samples) || 0} 个事件` : "—"));
    tr.append(idx, time, bpmCell, label);
    body.appendChild(tr);
  });
  $("#tempoDetailHint").textContent = compressed
    ? `共 ${total} 处，按约 ${Number(a.tempo_changes_window) || 0} 秒窗口取中位数；播放仍按全部事件`
    : `共 ${total || changes.length} 处，全部列出`;
}

function renderTracks() {
  const body = $("#trackBody");
  body.innerHTML = "";
  const voiceOptions = (state.config && state.config.voices) || [
    { value: "auto", label: "自动混合" },
    { value: "哈基米", label: "哈基米" },
    { value: "哈基米only", label: "哈基米 only" },
    { value: "曼波", label: "曼波" },
    { value: "曼波only", label: "曼波 only" },
    { value: "叮咚鸡", label: "叮咚鸡" },
    { value: "曼巴", label: "曼巴" },
  ];

  state.tracks.forEach((t) => {
    const tr = el("tr");
    if (!t.selected) tr.classList.add("unselected");

    // 参与
    const tdCheck = el("td", "c-check");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = t.selected;
    cb.addEventListener("change", () => {
      t.selected = cb.checked;
      tr.classList.toggle("unselected", !t.selected);
      updateTrackSummary();
    });
    tdCheck.appendChild(cb);
    tr.appendChild(tdCheck);

    tr.appendChild(el("td", "c-idx", t.index));

    const tdName = el("td");
    tdName.appendChild(el("div", "track-name", t.name));
    tdName.appendChild(el("div", "track-sub", `${t.instrument} · ${t.n_notes} 音符${t.is_drum ? " · 打击乐" : ""}`));
    tdName.title = t.reason || "";
    tr.appendChild(tdName);

    const tdRole = el("td");
    tdRole.appendChild(el("span", "role-badge " + (t.role || ""), t.role_label || t.role));
    tr.appendChild(tdRole);

    tr.appendChild(el("td", "c-num", `${t.n_notes}`));

    const tdVel = el("td", "c-vel");
    const velBar = el("div", "vel-bar");
    const velFill = el("i");
    velFill.style.width = `${Math.min(100, ((t.avg_velocity || 0) / 127) * 100)}%`;
    velBar.appendChild(velFill);
    tdVel.appendChild(velBar);
    tdVel.title = `平均力度 ${t.avg_velocity || 0}`;
    tr.appendChild(tdVel);

    const tdVoice = el("td");
    const sel = document.createElement("select");
    voiceOptions.forEach((v) => {
      const o = el("option", null, v.label);
      o.value = v.value;
      if (v.value === t.voice) o.selected = true;
      sel.appendChild(o);
    });
    sel.addEventListener("change", () => { t.voice = sel.value; });
    tdVoice.appendChild(sel);
    tr.appendChild(tdVoice);

    const tdOct = el("td", "c-oct");
    const oct = document.createElement("input");
    oct.type = "number"; oct.min = -3; oct.max = 3; oct.step = 1; oct.value = t.octave;
    oct.addEventListener("change", () => { t.octave = Math.max(-3, Math.min(3, Number(oct.value) || 0)); });
    tdOct.appendChild(oct);
    tr.appendChild(tdOct);

    const tdGain = el("td", "c-gain");
    const wrap = el("div");
    wrap.style.display = "flex"; wrap.style.alignItems = "center"; wrap.style.gap = "6px";
    const gain = document.createElement("input");
    gain.type = "range"; gain.min = 0; gain.max = 2; gain.step = 0.05; gain.value = t.gain;
    const val = el("span", "gain-val", Number(t.gain).toFixed(2));
    gain.addEventListener("input", () => {
      t.gain = Number(gain.value);
      val.textContent = t.gain.toFixed(2);
    });
    wrap.appendChild(gain); wrap.appendChild(val);
    tdGain.appendChild(wrap);
    tr.appendChild(tdGain);

    body.appendChild(tr);
  });
}

function setAllSelected(value, mode) {
  state.tracks.forEach((t) => {
    if (!value) { t.selected = false; return; }
    if (mode === "melody") t.selected = t.role === "melody";
    else t.selected = !t.is_drum && t.n_notes > 0;
  });
  renderTracks();
  updateTrackSummary();
}

function updateTrackSummary() {
  const chosen = state.tracks.filter((t) => t.selected);
  const notes = chosen.reduce((s, t) => s + t.n_notes, 0);
  const voices = new Set(chosen.map((t) => t.voice));
  $("#trackSummary").textContent =
    `已选 ${chosen.length} / ${state.tracks.length} 轨 · 约 ${notes} 个音符 · 语音：${Array.from(voices).join("、") || "无"}`;
}

/* ------------------------------------------------------------------ */
/* 分析可视化                                                          */
/* ------------------------------------------------------------------ */

function drawEmptyCanvas() {
  const canvas = $("#overviewCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600, h = 150;
  canvas.width = w * dpr; canvas.height = h * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "rgba(102, 112, 133, .72)";
  ctx.font = "12px 'Microsoft YaHei UI'";
  ctx.fillText("导入 MIDI 或乐谱后显示音高轮廓与段落", 14, h / 2);
}

function drawOverview(analysis) {
  const canvas = $("#overviewCanvas");
  if (!canvas || !analysis.overview) return;
  const data = analysis.overview;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 0, h = 150;
  // 首次分析时 fileCard 刚从 hidden 切换为显示，同一次任务里布局宽度可能还是 0。
  // 用 setTimeout 等布局完成后再画（比 requestAnimationFrame 在无头浏览器里更可靠）。
  if (!w) {
    setTimeout(() => {
      if (state.analysis === analysis) drawOverview(analysis);
    }, 60);
    return;
  }
  canvas.width = w * dpr; canvas.height = h * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const duration = Math.max(0.001, analysis.duration);
  const pad = 8;
  const innerW = w - pad * 2, innerH = h - pad * 2;

  // 段落背景
  (analysis.sections || []).forEach((s, i) => {
    const x = pad + (s.start / duration) * innerW;
    const sw = Math.max(1, ((s.end - s.start) / duration) * innerW);
    ctx.fillStyle = i % 2 === 0 ? "rgba(245, 158, 11, .10)" : "rgba(108, 92, 231, .09)";
    ctx.fillRect(x, pad, sw, innerH);
    if (sw > 16) {
      ctx.fillStyle = "rgba(154, 103, 0, .82)";
      ctx.font = "9px Consolas";
      ctx.fillText(s.label, x + 3, pad + 10);
    }
  });

  // 密度柱
  const dens = data.density || [];
  const maxD = Math.max(1, ...dens);
  if (dens.length) {
    const bw = innerW / dens.length;
    dens.forEach((d, i) => {
      if (!d) return;
      const bh = (d / maxD) * innerH * 0.55;
      ctx.fillStyle = "rgba(79, 140, 255, .20)";
      ctx.fillRect(pad + i * bw, pad + innerH - bh, Math.max(1, bw - 0.5), bh);
    });
  }

  // 音高轮廓
  const pitch = data.pitch || [];
  if (pitch.length) {
    const valid = pitch.filter((p) => p != null);
    if (!valid.length) return;
    const minP = Math.min(...valid) - 2, maxP = Math.max(...valid) + 2;
    ctx.strokeStyle = "rgba(108, 92, 231, .92)";
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    let started = false;
    pitch.forEach((p, i) => {
      if (p == null) { started = false; return; }
      const x = pad + (i / pitch.length) * innerW;
      const y = pad + innerH - ((p - minP) / Math.max(1, maxP - minP)) * innerH;
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // 时间轴
  ctx.fillStyle = "rgba(102,112,133,.72)";
  ctx.font = "9px Consolas";
  for (let i = 0; i <= 4; i++) {
    const t = (duration * i) / 4;
    const x = pad + (i / 4) * innerW;
    ctx.fillText(fmtTime(t).slice(0, -3), Math.min(x, w - 26), h - 2);
  }
}

function renderAnalysis() {
  const a = state.analysis;
  renderChords(a.chords || []);
  renderSections(a.sections || [], a.duration);
  renderPhrases(a.phrases || []);
  renderMotifs(a.motifs || []);
}

function renderChords(chords) {
  const strip = $("#chordStrip");
  strip.innerHTML = "";
  const show = chords.slice(0, 220);
  show.forEach((c) => {
    const b = el("span", "chord-block", c.label);
    if (c.decomposed) b.classList.add("decomposed");
    if (c.quality === "m" || c.quality === "m7" || c.quality === "dim") b.classList.add("minor");
    b.title = `${c.label}  ${fmtTime(c.start)} - ${fmtTime(c.end)}  声部 ${c.tracks.length}`;
    strip.appendChild(b);
  });
  const decomposed = chords.filter((c) => c.decomposed).length;
  $("#chordInfo").textContent =
    `共 ${chords.length} 个和弦窗口（展示前 ${show.length} 个）· 其中 ${decomposed} 个为分解和弦（跨声部共用歌词）`;
}

function renderSections(sections, duration) {
  const strip = $("#sectionStrip");
  const list = $("#sectionList");
  strip.innerHTML = "";
  list.innerHTML = "";
  const total = Math.max(0.001, duration);
  const palette = ["#ffd166", "#9b8cff", "#6ee7a8", "#ff9ecb", "#7fb4ff", "#ffb37a"];
  sections.forEach((s, i) => {
    const seg = el("div", "section-seg", s.label);
    seg.style.flex = String(Math.max(0.02, (s.end - s.start) / total));
    seg.style.background = palette[i % palette.length];
    const keyText = s.key ? ` · ${s.key}` : "";
    const prog = (s.progression || []).slice(0, 8).join("–");
    seg.title = `${s.label}  ${fmtTime(s.start)} - ${fmtTime(s.end)}${keyText}${prog ? " · " + prog : ""}`;
    strip.appendChild(seg);

    const row = el("div", "section-row");
    const left = el("div");
    left.appendChild(el("span", "tag", s.label + " 段"));
    left.appendChild(document.createTextNode(`  ${fmtTime(s.start)} - ${fmtTime(s.end)}`));
    if (s.key) left.appendChild(el("span", "section-key", s.key));
    if (s.character) left.appendChild(el("span", "section-char", s.character));
    row.appendChild(left);
    const right = el("div", "section-right");
    if (prog) right.appendChild(el("span", "section-prog", prog));
    right.appendChild(
      el("span", "muted", `${s.n_notes} 音符 · ${s.phrase_count || 0} 句 · 能量 ${(s.energy * 100).toFixed(0)}%`)
    );
    row.appendChild(right);
    list.appendChild(row);
  });
  if (!sections.length) list.appendChild(el("p", "muted small", "没有检测到明显的段落重复。"));
}

function renderPhrases(phrases) {
  const box = $("#phraseList");
  if (!box) return;
  box.innerHTML = "";
  if (!phrases.length) {
    box.appendChild(el("p", "muted small", "没有检测到独立乐句（可能是持续长音或过短片段）。"));
    return;
  }
  phrases.forEach((p) => {
    const chip = el("div", "phrase-chip");
    chip.appendChild(el("span", "phrase-id", p.label || "A1"));
    chip.appendChild(el("span", "phrase-time", `${fmtTime(p.start)} - ${fmtTime(p.end)}`));
    const meta = [p.key, (p.progression || []).slice(0, 4).join("–"), `${p.n_notes || 0} 音`]
      .filter(Boolean).join(" · ");
    chip.appendChild(el("span", "phrase-meta", meta));
    chip.title = `${p.section_label || ""} 段 · ${p.bars || 0} 小节`;
    box.appendChild(chip);
  });
}

function renderMotifs(motifs) {
  const box = $("#motifList");
  box.innerHTML = "";
  if (!motifs.length) {
    box.appendChild(el("p", "muted small", "没有检测到重复动机（单声部或变化较多的曲子常见）。"));
    return;
  }
  motifs.forEach((m) => {
    const row = el("div", "motif-row");
    row.appendChild(el("span", "motif-id", m.id));
    const mid = el("div");
    mid.appendChild(el("div", null, `${m.length} 个音 · 音程 ${m.intervals.join(" ")}`));
    mid.appendChild(el("div", "motif-intervals", `首次出现 @ ${fmtTime(m.occurrences[0].start)}`));
    row.appendChild(mid);
    row.appendChild(el("span", "motif-count", `×${m.count} 次`));
    box.appendChild(row);
  });
}

/* ------------------------------------------------------------------ */
/* 生成                                                                */
/* ------------------------------------------------------------------ */

function collectOptions() {
  const maxSecondsRaw = $("#maxSeconds").value.trim();
  const maxSeconds = maxSecondsRaw === "" ? null : Number(maxSecondsRaw);
  return {
    model_kind: $("#modelSelect").value || "markov",
    voice: $("#globalVoice").value || "auto",
    sample_set: Number($("#sampleSet")?.value) || 1,
    seed: Number($("#seedInput").value) || 1337,
    syntax_weight: Number($("#syntaxInput").value),
    section_plans: state.sectionPlans.map((plan) => ({
      section_index: plan.section_index,
      label: plan.label,
      start: plan.start,
      end: plan.end,
      key: plan.key,
      markov_ratio: Number(plan.markov_ratio),
      seed: Number(plan.seed),
      source_weights: { ...plan.source_weights },
      auto: Boolean(plan.auto),
      source: plan.source || (plan.auto ? "auto" : "manual"),
    })),
    velocity_gamma: Number($("#gammaInput").value),
    workers: Number($("#workersInput").value) || 0,
    gpu: $("#gpuMode").value,
    gpu_memory_mb: Number($("#gpuMemory").value) || 2048,
    max_seconds: maxSeconds && maxSeconds > 0 ? maxSeconds : null,
    decomposed_merge: $("#decomposedInput").checked,
    auto_mix: $("#autoMixInput") ? $("#autoMixInput").checked : true,
    write_stems: $("#stemsInput").checked,
    write_premix: $("#premixInput") ? $("#premixInput").checked : true,
    write_bad_note_midis: $("#badNotesInput") ? $("#badNotesInput").checked : true,
    write_zip: $("#zipInput").checked,
    write_subtitle_videos: $("#videosInput") ? $("#videosInput").checked : true,
    subtitle_video_width: 1920,
    subtitle_video_fps: 30,
    subtitle_video_font_family: $("#subtitleFontFamily") ? $("#subtitleFontFamily").value : "SimSun",
    subtitle_video_font_size: $("#subtitleFontSize") ? Number($("#subtitleFontSize").value) || 8 : 8,
    subtitle_video_text_color: $("#subtitleTextColor") ? $("#subtitleTextColor").value : "#FFFFFF",
    normalize: "peak",
  };
}

async function startRender() {
  if (!state.session) { toast("请先导入 MIDI 或乐谱", "error"); return; }
  if (state.starting) return;
  const tracks = state.tracks.map((t) => ({
    index: t.index,
    selected: Boolean(t.selected),
    voice: t.voice,
    octave: Number(t.octave) || 0,
    gain: Number(t.gain),
  }));
  if (!tracks.some((t) => t.selected)) { toast("请至少勾选一个轨道", "error"); return; }

  const options = collectOptions();
  state.starting = true;
  state.result = null;
  state.resultSig = null;
  state.lastAutoDownload = null;
  state.audioUrl = null;
  state.subtitles = [];
  state.activeSub = -1;
  closeES();
  state.jobId = null;
  const audio = $("#audioPlayer");
  audio.pause();
  audio.removeAttribute("src");
  $("#renderBtn").disabled = true;
  $("#cancelBtn").hidden = false;
  $("#progressBox").hidden = false;
  $("#resultCard").hidden = true;
  $("#subtitleCard").hidden = true;
  $("#progressBar").style.width = "0%";
  $("#pctText").textContent = "0%";
  $("#stageText").textContent = "提交任务…";
  $("#miniLog").textContent = "";
  setProgress(0);

  try {
    const resp = await fetchJSON("/api/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.session.id, options, tracks }),
    });
    state.jobId = resp.job_id;
    watchJob(resp.job_id);
  } catch (e) {
    toast("无法启动生成：" + e.message, "error");
    finishRender();
  }
}

function watchJob(jobId) {
  if (state.es) { state.es.close(); state.es = null; }
  const es = new EventSource(`/api/jobs/${jobId}/events`);
  state.es = es;
  es.onmessage = (msg) => {
    let event;
    try { event = JSON.parse(msg.data); } catch (e) { return; }
    handleJobEvent(event);
  };
  es.onerror = () => {
    // EventSource 会自动重连；仅在任务已结束或明确失败时提示
    fetchJSON(`/api/jobs/${jobId}`).then((job) => {
      if (job.status === "done" && job.result) {
        showResult(job.result);
        es.close(); state.es = null;
      } else if (job.status === "error") {
        toast("生成失败：" + job.error, "error");
        es.close(); state.es = null; finishRender();
      }
    }).catch(() => {});
  };
}

function handleJobEvent(event) {
  switch (event.type) {
    case "snapshot":
      if (event.subtitles && event.subtitles.length) setSubtitles(event.subtitles);
      if (event.pct) setProgress(event.pct, event.message || event.stage);
      if (Array.isArray(event.log)) $("#miniLog").textContent = event.log.join("\n");
      if (event.status === "done" && event.result) { showResult(event.result); scheduleESClose(); }
      if (event.status === "error") { toast("生成失败：" + event.error, "error"); scheduleESClose(); finishRender(); }
      if (event.status === "cancelled") { toast("任务已取消"); scheduleESClose(); finishRender(); }
      break;
    case "stage":
      setProgress(event.pct, event.message, event.stage);
      break;
    case "log":
      appendMiniLog(event.message);
      break;
    case "subtitle":
      setSubtitles(event.lines || []);
      $("#subtitleMeta").textContent = `${(event.lines || []).length} 行 · 已实时推送`;
      break;
    case "done":
      showResult(event.result);
      // 等后端发完 end 再关闭，避免浏览器先 RST 造成 Windows Proactor 10054。
      scheduleESClose();
      break;
    case "error":
      toast("生成失败：" + (event.message || "未知错误"), "error");
      appendMiniLog("ERROR " + (event.message || ""));
      scheduleESClose();
      finishRender();
      break;
    case "cancelled":
      toast("任务已取消");
      scheduleESClose();
      finishRender();
      break;
    case "end":
      closeES();
      break;
    default:
      break;
  }
}

function scheduleESClose() {
  if (!state.es) return;
  if (state.esCloseTimer) clearTimeout(state.esCloseTimer);
  // 正常情况下紧接着会收到 end；预留 1.5s 兜底，避免服务端异常时连接泄漏。
  state.esCloseTimer = setTimeout(() => {
    state.esCloseTimer = null;
    closeES();
  }, 1500);
}

function closeES() {
  if (state.esCloseTimer) {
    clearTimeout(state.esCloseTimer);
    state.esCloseTimer = null;
  }
  if (state.es) { state.es.close(); state.es = null; }
}

function setProgress(pct, message, stage) {
  const value = Math.max(0, Math.min(100, Number(pct) || 0));
  $("#progressBar").style.width = value + "%";
  $("#pctText").textContent = Math.round(value) + "%";
  if (message) $("#stageText").textContent = message;
  if (stage === "render" || stage === "post") $("#cancelBtn").hidden = false;
}

function appendMiniLog(text) {
  const box = $("#miniLog");
  if (!text) return;
  box.textContent += (box.textContent ? "\n" : "") + text;
  box.scrollTop = box.scrollHeight;
}

function finishRender() {
  state.starting = false;
  $("#renderBtn").disabled = !state.session;
  $("#cancelBtn").hidden = true;
}

async function cancelRender() {
  if (!state.jobId) return;
  try {
    await fetchJSON(`/api/jobs/${state.jobId}/cancel`, { method: "POST" });
    toast("已请求取消…");
  } catch (e) {
    toast("取消失败：" + e.message, "error");
  }
}

/* ------------------------------------------------------------------ */
/* 字幕                                                                */
/* ------------------------------------------------------------------ */

function setSubtitles(lines) {
  state.subtitles = Array.isArray(lines) ? lines : [];
  state.activeSub = -1;
  $("#subtitleCard").hidden = false;
  const list = $("#subtitleList");
  list.innerHTML = "";
  state.subtitles.forEach((line, i) => {
    const row = el("div", "sub-line");
    row.dataset.i = String(i);
    row.appendChild(el("span", "t", fmtTime(line.start)));
    row.appendChild(el("span", "txt", line.text));
    row.addEventListener("click", () => {
      const audio = $("#audioPlayer");
      if (state.audioUrl) { audio.currentTime = line.start + 0.001; audio.play().catch(() => {}); }
    });
    list.appendChild(row);
  });
  if (state.subtitles.length) {
    showSubtitleLine(0, 0);
    if (!state.audioUrl) $("#nowLine").textContent = state.subtitles[0].text;
  }
}

function showSubtitleLine(index, time) {
  if (index < 0 || index >= state.subtitles.length) return;
  if (state.activeSub === index) {
    if (time != null) highlightChars(index, time);
    return;
  }
  const rows = $$("#subtitleList .sub-line");
  rows.forEach((r) => r.classList.remove("active"));
  const row = rows[index];
  if (row) {
    row.classList.add("active");
    row.scrollIntoView({ block: "center", behavior: "smooth" });
  }
  const line = state.subtitles[index];
  state.activeSub = index;
  highlightChars(index, time == null ? line.start : time);
}

function highlightChars(index, time) {
  const line = state.subtitles[index];
  if (!line) return;
  const now = $("#nowLine");
  now.textContent = "";
  let current = null;
  (line.chars || [{ w: line.text, s: line.start }]).forEach((c) => {
    if (time >= c.s) current = c;
  });
  (line.chars || [{ w: line.text, s: line.start }]).forEach((c) => {
    const span = el("span", c === current ? "on" : "", c.w);
    now.appendChild(span);
  });
}

function syncSubtitleToAudio(t) {
  const lines = state.subtitles;
  if (!lines.length) return;
  let lo = 0, hi = lines.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (lines[mid].start <= t) { ans = mid; lo = mid + 1; } else hi = mid - 1;
  }
  if (ans >= 0) showSubtitleLine(ans, t);
}

/* ------------------------------------------------------------------ */
/* 结果                                                                */
/* ------------------------------------------------------------------ */

function showResult(result) {
  const sig = `${result.project || ""}|${result.audio_path || ""}|${result.duration || 0}`;
  if (state.resultSig === sig) return;
  state.resultSig = sig;
  state.result = result;
  $("#resultCard").hidden = false;
  $("#progressBar").style.width = "100%";
  $("#pctText").textContent = "100%";
  $("#stageText").textContent = "生成完成";
  finishRender();

  const audio = $("#audioPlayer");
  state.audioUrl = result.audio_url + "?v=" + Date.now();
  audio.src = state.audioUrl;
  audio.load();

  const project = result.project || "hachimi";
  $("#resultMeta").textContent = `${project} · ${fmtDuration(result.duration)} · ${result.output_dir}`;

  const grid = $("#downloadGrid");
  grid.innerHTML = "";
  const files = result.files || {};
  const order = [
    ["zip", "打包下载 ZIP", "MP3 + 分轨 + 字幕"],
    ["audio", "混音 MP3", "audio"],
    ["srt", "字幕 SRT", "subtitle"],
    ["lrc", "歌词 LRC", "subtitle"],
    ["html", "实时字幕 HTML", "karaoke"],
    ["lyrics", "歌词 JSON", "events"],
    ["subtitle_json", "字幕 JSON", "timeline"],
    ["markdown", "歌词表格 MD", "document"],
    ["analysis", "分析 JSON", "chords"],
    ["instrument_report", "乐器清单 MD", "输出 / 未输出"],
    ["instrument_report_csv", "乐器清单 CSV", "表格"],
    ["instrument_report_json", "乐器清单 JSON", "machine"],
    ["verification", "发声校验 JSON", "polyphony"],
    ["bad_notes", "不良音符报告 JSON", "过高 / 过低 / 音量低", "bad"],
    ["supplement_midi", "补充 MIDI", "不良音 _NG → 空白轨 → 全谱", "bad"],
    ["percussion_midi", "打击乐 MIDI", "单独导出 · 轨名 _NOAUDIO", "bad"],
    ["params", "工程参数包 JSON", "可复制全部设置", "params"],
    ["params_zip", "工程参数包 ZIP", "发给别人复用", "params"],
    ["premix_zip", "混音前干声 ZIP", "全部轨道 FLAC", "premix"],
  ];
  const addLink = (info, label, sub, cls) => {
    if (!info || !info.url) return;
    const a = document.createElement("a");
    a.className = "dl-btn" + (cls ? " " + cls : "");
    a.href = info.url;
    a.download = info.name || "";
    a.appendChild(el("span", null, label));
    a.appendChild(el("small", null, sub || fmtSize(info.size)));
    grid.appendChild(a);
  };
  order.forEach(([key, label, sub, cls]) =>
    addLink(files[key], label, sub, cls || (key === "zip" ? "zip" : ""))
  );
  const videos = files.subtitle_videos || {};
  Object.entries(videos).forEach(([key, info]) => {
    const label = key === "all"
      ? "全部字幕视频（每次 1 行）"
      : key === "melody"
        ? "主要字幕视频（旧版）"
        : "辅助字幕视频（旧版）";
    addLink(info, label, "绿幕 1920 宽 · 全部轨道合并 · 单条", "video");
  });
  const stems = files.stems || {};
  Object.entries(stems).forEach(([idx, info]) => {
    const track = (result.tracks || []).find((t) => String(t.index) === String(idx)) || {};
    const voice = track.voice === "auto" ? "自动混合" : (track.voice || "");
    const octave = Number(track.octave || 0);
    const desc = [track.instrument || track.name || "", voice, `${octave >= 0 ? "+" : ""}${octave}`]
      .filter(Boolean).join(" · ");
    addLink(info, `轨道 ${idx} MP3`, desc || "stem", "stem");
  });
  const trackSrt = files.track_srt || {};
  Object.entries(trackSrt).forEach(([idx, info]) => addLink(info, `轨道 ${idx} SRT`, "per-track"));
  const stemsManifest = files.stems_manifest;
  if (stemsManifest) addLink(stemsManifest, "分轨索引 JSON", "stems.json");
  const premix = files.premix || {};
  Object.entries(premix).forEach(([idx, info]) => {
    const track = (result.tracks || []).find((t) => String(t.index) === String(idx)) || {};
    addLink(info, `轨道 ${idx} 混音前 FLAC`, track.instrument || "premix", "premix");
  });
  const badMidis = files.bad_note_midis || {};
  Object.entries(badMidis).forEach(([idx, info]) => {
    const label = idx === "all" ? "全部不良音符 MIDI" : `轨道 ${idx} 不良音符 MIDI`;
    addLink(info, label, idx === "all" ? "合并检查" : "仅含不良音", "bad");
  });

  // ---- 和弦发声校验 ----
  const verify = result.verification;
  const verifyBox = $("#verifyBox");
  if (verify) {
    verifyBox.hidden = false;
    verifyBox.className = "verify-box " + (verify.ok ? "ok" : "bad");
    verifyBox.innerHTML = "";
    verifyBox.appendChild(
      el("div", "verify-title", verify.ok ? "✓ 和弦发声检查通过" : "⚠ 和弦发声检查发现问题")
    );
    verifyBox.appendChild(el("div", "verify-msg", verify.message || ""));
    verifyBox.appendChild(
      el(
        "div",
        "verify-detail",
        `检查轨道 ${verify.checked_tracks || 0} 条 · 和弦组 ${verify.total_chords || 0} 个 · ` +
        `和弦音 ${verify.verified_chord_notes || 0}/${verify.total_chord_notes || 0} 已发声`
      )
    );
  } else {
    verifyBox.hidden = true;
  }

  // ---- 不良音符质量检查 ----
  const quality = result.bad_notes || (result.stats || {}).bad_notes;
  const supplement = result.supplement_midi || quality?.supplement_midi || {};
  const qualityBox = $("#qualityBox");
  if (quality) {
    const ok = Number(quality.bad_notes || 0) === 0;
    qualityBox.hidden = false;
    qualityBox.className = "verify-box " + (ok ? "ok" : "bad");
    qualityBox.innerHTML = "";
    qualityBox.appendChild(
      el("div", "verify-title", ok ? "✓ 音符质量检查通过" : "⚠ 发现不良音符，已导出专用 MIDI")
    );
    qualityBox.appendChild(el("div", "verify-msg", quality.message || ""));
    qualityBox.appendChild(
      el(
        "div",
        "verify-detail",
        `过高 ${quality.too_high || 0} · 过低 ${quality.too_low || 0} · ` +
        `音量过低 ${quality.too_quiet || 0} · 共 ${quality.bad_notes || 0}/${quality.total_notes || 0} 个`
      )
    );
    if (supplement.path || supplement.n_tracks) {
      qualityBox.appendChild(
        el(
          "div",
          "verify-detail",
          `补充 MIDI：${supplement.n_bad_tracks || 0} 条不良音轨 → 空白轨 → ` +
          `${supplement.n_score_tracks || 0} 条全谱轨；共 ${supplement.n_tracks || 0} 轨 / ` +
          `${supplement.n_notes || 0} 音（不良 ${supplement.bad_notes || 0} + 全谱 ${supplement.score_notes || 0}）`
        )
      );
    }
    const percussion = result.percussion_midi || {};
    if (percussion.path) {
      qualityBox.appendChild(
        el(
          "div",
          "verify-detail",
          `打击乐 MIDI：${percussion.n_tracks || 0} 轨 / ${percussion.n_notes || 0} 音；` +
          "不参与音频合成，轨名带 _NOAUDIO"
        )
      );
    }
  } else {
    qualityBox.hidden = true;
  }

  const stats = result.stats || {};
  const render = stats.render || {};
  const mix = stats.mix || {};
  const lyricsStats = stats.lyrics || {};
  const resultStats = $("#resultStats");
  resultStats.innerHTML = "";
  const items = [
    ["时长", fmtDuration(result.duration)],
    ["调性", result.analysis?.key?.label_cn || result.analysis?.key?.label || "—"],
    ["段落 / 乐句", `${result.analysis?.stats?.n_sections || 0} / ${result.analysis?.stats?.n_phrases || 0}`],
    ["全局马尔科夫", `${Math.round((lyricsStats.markov_ratio ?? 0.7) * 100)}%`],
    ["渲染耗时", `${(result.render_seconds || 0).toFixed(1)} 秒`],
    ["实时倍率", `${(render.realtime_factor || 0).toFixed(1)}×`],
    ["合成单位", `${render.n_unique_units || 0}`],
    ["单位内存", `${Math.round((render.unit_bytes || 0) / 1048576)} MB`],
    ["时长量化", `${Math.round((render.duration_quantum || 0.02) * 1000)} ms`],
    ["设备", render.device === "cuda" ? `GPU ${render.gpu_memory_mb || ""}MB` : "CPU"],
    ["单位复用", `${Math.round((render.unit_reuse_rate || 0) * 100)}%`],
    ["自动混音", mix.enabled === false ? "关闭" : `开启 · ${mix.loudness_db ?? "—"} dB`],
    ["母带动态", mix.enabled === false ? "传统后处理" : `${mix.compression_ratio ?? "—"}:1 · 补偿 +${mix.makeup_db ?? 0} dB · 峰值因数 ${mix.crest_db ?? "—"} dB`],
    ["表情音符", `${render.n_dynamic_notes || 0} 渐强/渐弱`],
    ["高音补偿", `${render.n_audibility_notes || 0} 个`],
    ["分轨", `${Object.keys(stems).length} 条`],
    ["字幕视频", (() => {
      const video = stats.video || {};
      if (!video.available) return video.error || "未生成";
      const cover = video.coverage || {};
      const covered = cover.covered_slots != null ? ` · 覆盖${cover.covered_slots}/${cover.input_slots}` : "";
      return `${video.width}宽×${video.height}高 · ${video.font_size ?? 8}号 · 每次${video.max_lines || 1}行${covered} · ${video.fps}fps · ${video.elapsed_seconds ?? "—"}s`;
    })()],
    ["弱化音头 / 连奏", `${render.n_attack_softened || 0} / ${render.n_legato_links || 0}`],
    ["辅音 / 爆破音", `${render.n_consonant_notes || 0} / ${render.n_plosive_notes || 0}`],
    ["混音前干声", `${stats.premix?.n_tracks || 0} 条 FLAC${stats.premix?.zip ? " · ZIP" : ""}`],
    ["不良音符", `${quality?.bad_notes || 0} / ${quality?.total_notes || 0}`],
    ["补充 MIDI", supplement.n_tracks ? `${supplement.n_tracks} 轨 / ${supplement.n_notes} 音` : "—"],
    ["打击乐 MIDI", result.percussion_midi?.path ? `${result.percussion_midi.n_tracks} 轨 / ${result.percussion_midi.n_notes} 音` : "—"],
    ["发声检查", verify ? (verify.ok ? "通过" : "异常") : "—"],
    ["句法 / 马尔科夫", `${lyricsStats.syntax_phrases || 0} / ${lyricsStats.markov_phrases || 0}`],
    ["分解和弦槽", `${lyricsStats.n_decomposed_slots || 0}`],
    ["动机复用", `${lyricsStats.n_motifs || 0} 个`],
    ["语音分布", Object.entries(lyricsStats.voices || {}).map(([k, v]) => `${k}×${v}`).join(" ") || "—"],
    ["乐器输出", `${stats.instruments?.n_outputs || 0} 独立 / ${stats.instruments?.n_mix_only || 0} 混音 / ${stats.instruments?.n_missing || 0} 未输出`],
  ];
  items.forEach(([k, v]) => {
    const box = el("div", "stat");
    box.appendChild(el("div", "k", k));
    box.appendChild(el("div", "v", String(v)));
    resultStats.appendChild(box);
  });

  if (result.subtitles && result.subtitles.length) setSubtitles(result.subtitles);
  autoDownload(result);
  toast("生成完成 🎉", "ok");
  loadLibrary();
  loadParamsList();
  $("#resultCard").scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ------------------------------------------------------------------ */
/* 自动下载                                                            */
/* ------------------------------------------------------------------ */

function triggerDownload(info) {
  if (!info || !info.url) return false;
  const a = document.createElement("a");
  a.href = info.url;
  a.download = info.name || "";
  a.style.display = "none";
  document.body.appendChild(a);
  a.click();
  setTimeout(() => a.remove(), 1000);
  return true;
}

function autoDownload(result) {
  const mode = ($("#downloadMode") && $("#downloadMode").value) || "all";
  if (mode === "off") return;
  const key = `${result.project}|${result.duration}|${result.audio_path || ""}`;
  if (state.lastAutoDownload === key) return;
  state.lastAutoDownload = key;

  const files = result.files || {};
  const queue = [];
  if (files.zip) queue.push(files.zip);
  if (mode === "all") {
    if (files.audio) queue.push(files.audio);
    if (files.srt) queue.push(files.srt);
    if (files.lrc) queue.push(files.lrc);
    // 每条分轨也单独下载，保证“所有音频”都有独立 MP3
    Object.values(files.stems || {}).forEach((info) => queue.push(info));
  }
  if (!queue.length) return;
  appendMiniLog(`自动下载 ${queue.length} 个文件（MP3 / 字幕）`);
  toast("已开始自动下载；如浏览器询问多文件下载，请选择允许", "ok");
  queue.forEach((info, index) => {
    setTimeout(() => triggerDownload(info), index * 900 + 300);
  });
}

/* ------------------------------------------------------------------ */
/* 项目曲库（MIDI / MusicXML）                                          */
/* ------------------------------------------------------------------ */

let localFiles = [];

async function openLocalModal() {
  $("#localModal").hidden = false;
  $("#localList").innerHTML = "<p class='muted small'>加载中…</p>";
  try {
    const data = await fetchJSON("/api/midi/list");
    localFiles = data.files || [];
    renderLocalList(localFiles);
  } catch (e) {
    $("#localList").innerHTML = "";
    $("#localList").appendChild(el("p", "muted small", "无法读取项目曲库：" + e.message));
  }
}

function filterLocalList() {
  const q = $("#localSearch").value.trim().toLowerCase();
  renderLocalList(localFiles.filter((f) => !q || f.name.toLowerCase().includes(q) || f.rel.toLowerCase().includes(q)));
}

function renderLocalList(files) {
  const box = $("#localList");
  box.innerHTML = "";
  if (!files.length) {
    box.appendChild(el("p", "muted small", "没有匹配的 MIDI / 乐谱文件。"));
    return;
  }
  files.forEach((f) => {
    const row = el("div", "local-row");
    row.appendChild(el("span", "nm", f.name));
    row.appendChild(el("span", "path", f.rel));
    row.addEventListener("click", async () => {
      $("#localModal").hidden = true;
      $("#fileName").textContent = f.name;
      $("#fileMeta").textContent = "正在分析…";
      try {
        const data = await fetchJSON("/api/analyze-path", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path: f.path }),
        });
        handleAnalyze(data);
      } catch (e) {
        toast("分析失败：" + e.message, "error");
      }
    });
    box.appendChild(row);
  });
}

/* ------------------------------------------------------------------ */
/* 最近作品                                                            */
/* ------------------------------------------------------------------ */

async function loadLibrary() {
  try {
    const data = await fetchJSON("/api/library");
    const box = $("#libraryList");
    const items = data.items || [];
    const count = $("#libraryCount");
    if (count) count.textContent = String(items.length);
    box.innerHTML = "";
    if (!items.length) {
      box.appendChild(el("p", "muted small", "还没有生成过作品。"));
      return;
    }
    items.slice(0, 12).forEach((item) => {
      const row = el("div", "library-row");
      row.appendChild(el("span", "name", item.project));
      row.appendChild(el("span", "time", item.duration ? fmtDuration(item.duration) : ""));
      const a = document.createElement("a");
      a.href = item.audio_url;
      a.textContent = "试听";
      a.target = "_blank";
      row.appendChild(a);
      const dl = document.createElement("a");
      dl.href = item.audio_url;
      dl.textContent = "下载";
      dl.download = `${item.project || "hachimi"}.mp3`;
      row.appendChild(dl);
      box.appendChild(row);
    });
  } catch (e) {
    $("#libraryList").textContent = "读取失败：" + e.message;
    const count = $("#libraryCount");
    if (count) count.textContent = "—";
  }
}

/* ------------------------------------------------------------------ */

init();
