"""字幕生成：SRT / LRC / JSON / Markdown / 独立 HTML 实时字幕播放器。

生成音频时会同步产出这些文件，因此「生成的同时」即可获得可直接使用的字幕文档；
网页端还会通过 SSE 把字幕行实时推送到浏览器，实现渲染过程中的实时字幕预览。
"""

from __future__ import annotations

import html
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from hachimi.config import APP_NAME, APP_NAME_CN

logger = logging.getLogger(__name__)


@dataclass
class SubtitleLine:
    index: int
    start: float
    end: float
    text: str
    chars: list[dict[str, Any]] = field(default_factory=list)
    section: str = ""
    voice: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "chars": self.chars,
            "section": self.section,
            "voice": self.voice,
        }


# ---------------------------------------------------------------------------
# 时间格式
# ---------------------------------------------------------------------------


def sec_to_srt(t: float) -> str:
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms >= 1000:
        ms = 999
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def sec_to_lrc(t: float) -> str:
    t = max(0.0, float(t))
    m = int(t // 60)
    s = t - m * 60
    return f"[{m:02d}:{s:05.2f}]"


# ---------------------------------------------------------------------------
# 从歌词槽构建字幕行
# ---------------------------------------------------------------------------


def _make_lines(
    items: Sequence[dict[str, Any]],
    *,
    reset_gap: float = 1.0,
    max_len: int = 12,
    max_line_seconds: float = 6.0,
    min_seconds: float = 0.6,
    hold_gap: float = 0.0,
) -> list[SubtitleLine]:
    """把 (word,start,end) 序列合并成字幕行。

    ``hold_gap`` 用于把短于该值的句间空隙并入当前字幕（最多延长到下一句前
    0.01 秒），避免频繁闪绿；真正的间奏仍保持空白。
    """
    items = sorted(items, key=lambda x: (x["start"], x["end"]))
    if not items:
        return []

    chunks: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    for it in items:
        if cur:
            gap = it["start"] - cur[-1]["start"]
            span = it["end"] - cur[0]["start"]
            if gap > reset_gap or len(cur) >= max_len or span > max_line_seconds:
                chunks.append(cur)
                cur = []
        cur.append(it)
    if cur:
        chunks.append(cur)

    lines: list[SubtitleLine] = []
    for idx, chunk in enumerate(chunks):
        start = float(chunk[0]["start"])
        last_end = float(max(c["end"] for c in chunk))
        end = max(last_end + 0.25, start + min_seconds)
        if idx + 1 < len(chunks):
            next_start = float(chunks[idx + 1][0]["start"])
            if hold_gap > 0 and next_start - end <= hold_gap:
                end = max(end, next_start - 0.01)
            end = min(end, next_start - 0.01)
        end = max(end, start + 0.05)
        text = "".join(str(c["word"]) for c in chunk)
        chars = [
            {
                "w": str(c["word"]),
                "s": round(float(c["start"]), 3),
                "e": round(float(c["end"]), 3),
                "slot": c.get("slot"),
            }
            for c in chunk
        ]
        lines.append(
            SubtitleLine(
                index=idx + 1,
                start=start,
                end=end,
                text=text,
                chars=chars,
                section=str(chunk[0].get("section", "")),
                voice=str(chunk[0].get("voice", "")),
            )
        )
    return lines


def build_subtitles(
    slots: Sequence[Any],
    *,
    selected_tracks: Iterable[int] | None = None,
    preferred_tracks: Iterable[int] | None = None,
    primary_tracks: Iterable[int] | None = None,
    voices: dict[int, str] | None = None,
    preferred_voice: str | None = None,
    reset_gap: float = 1.0,
    max_len: int = 12,
) -> list[SubtitleLine]:
    """从歌词槽构建全局字幕行。

    ``preferred_voice`` / ``preferred_tracks`` 让全局字幕跟随主旋律，
    避免把伴奏 / 对位声部的字与旋律交错在一起（各轨字幕仍会单独输出）。
    """
    selected = set(selected_tracks) if selected_tracks is not None else None
    preferred = set(preferred_tracks) if preferred_tracks else set()
    primary = set(primary_tracks) if primary_tracks else set()
    usable = list(slots)
    if primary:
        filtered = [s for s in usable if getattr(s, "primary_track", -1) in primary]
        if filtered:
            usable = filtered
    elif preferred_voice and voices:
        filtered = [s for s in usable if voices.get(s.index) == preferred_voice]
        if filtered:
            usable = filtered
    elif preferred:
        filtered = [s for s in usable if set(s.track_ids) & preferred]
        if filtered:
            usable = filtered
    items: list[dict[str, Any]] = []
    for slot in usable:
        if selected is not None and not (set(slot.track_ids) & selected):
            continue
        if not slot.word:
            continue
        items.append(
            {
                "word": slot.word,
                "start": slot.start,
                "end": slot.end,
                "slot": slot.index,
                "section": slot.section,
                "voice": "",
            }
        )
    return _make_lines(items, reset_gap=reset_gap, max_len=max_len)


def build_stream_subtitles(
    slots: Sequence[Any],
    track: int,
    *,
    reset_gap: float = 1.0,
    max_len: int = 16,
    max_line_seconds: float = 8.0,
    min_seconds: float = 0.35,
) -> list[SubtitleLine]:
    """只取某一条主声部的字幕行（旧版主 / 辅视频使用）。

    与全局字幕不同，这里按 ``slot.primary_track`` 精确过滤：一个和弦槽即使
    同时包含旋律与伴奏，也只会归入其主声部。
    """
    items: list[dict[str, Any]] = []
    seen: set[int] = set()
    for slot in slots:
        if int(getattr(slot, "primary_track", -1)) != int(track):
            continue
        if not getattr(slot, "word", ""):
            continue
        key = int(getattr(slot, "index", -1))
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "word": slot.word,
                "start": float(slot.start),
                "end": float(slot.end),
                "slot": key,
                "section": getattr(slot, "section", ""),
                "voice": "",
            }
        )
    return _make_lines(
        items,
        reset_gap=reset_gap,
        max_len=max(4, min(24, int(max_len))),
        max_line_seconds=max(1.0, float(max_line_seconds)),
        min_seconds=max(0.05, float(min_seconds)),
    )


def _serialize_lines(lines: list[SubtitleLine], gap: float = 0.001) -> list[SubtitleLine]:
    """把字幕行压成严格首尾相接的时间线，避免同一时刻出现两行。

    合并全部轨道时，多个流可能在同一时刻各自结束 / 开始；``_make_lines``
    的“至少 0.05 秒”规则会让相邻行重叠，视频时间轴会直接丢掉后一行。
    这里把后一行顺延到前一行结束后，并保证每行仍有最短显示时间。
    """
    cursor = -1e9
    for line in lines:
        if line.start < cursor + gap:
            line.start = cursor + gap
        if line.end < line.start + 0.05:
            line.end = line.start + 0.05
        cursor = line.end
    return lines


def build_group_subtitles(
    slots: Sequence[Any],
    tracks: Iterable[int],
    *,
    reset_gap: float = 1.1,
    max_len: int = 16,
    max_line_seconds: float = 8.0,
    min_seconds: float = 0.35,
    hold_gap: float = 1.0,
) -> list[SubtitleLine]:
    """把一组轨道的主声部槽合成连续字幕。

    每个歌词槽只取一次（槽内多个轨道共用同一个字），既覆盖该组的全部文字，
    又不会因为和弦 / 齐奏重复。多个轨道同时唱不同字时会按时间合并到同一行；
    传入全部轨道时，即可得到单条视频使用的完整字幕时间线。
    """
    track_set = {int(t) for t in tracks}
    if not track_set:
        return []
    items: list[dict[str, Any]] = []
    seen: set[int] = set()
    for slot in slots:
        primary = int(getattr(slot, "primary_track", -1))
        if primary not in track_set:
            continue
        word = str(getattr(slot, "word", "") or "")
        if not word:
            continue
        key = int(getattr(slot, "index", -1))
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "word": word,
                "start": float(getattr(slot, "start", 0.0)),
                "end": float(getattr(slot, "end", 0.0)),
                "slot": key,
                "section": str(getattr(slot, "section", "") or ""),
                "voice": "",
                "track": primary,
            }
        )
    lines = _make_lines(
        items,
        reset_gap=max(0.1, float(reset_gap)),
        max_len=max(4, min(32, int(max_len))),
        max_line_seconds=max(1.0, float(max_line_seconds)),
        min_seconds=max(0.05, float(min_seconds)),
        hold_gap=max(0.0, float(hold_gap)),
    )
    return _serialize_lines(lines)


def build_track_subtitles(
    events: Sequence[dict[str, Any]],
    track: int,
    *,
    reset_gap: float = 1.0,
    max_len: int = 12,
) -> list[SubtitleLine]:
    """单轨字幕：同一槽只显示一次，避免和弦内多个音重复同一个字。"""
    seen: set[int] = set()
    items: list[dict[str, Any]] = []
    for ev in events:
        if int(ev.get("track", -1)) != track:
            continue
        slot = ev.get("slot")
        key = int(slot) if slot is not None else id(ev)
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "word": ev.get("state") or ev.get("word"),
                "start": float(ev["start"]),
                "end": float(ev["end"]),
                "slot": slot,
                "section": ev.get("section", ""),
                "voice": ev.get("voice", ""),
            }
        )
    return _make_lines(items, reset_gap=reset_gap, max_len=max_len)


# ---------------------------------------------------------------------------
# 写文件
# ---------------------------------------------------------------------------


def write_srt(lines: Sequence[SubtitleLine], path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig") as f:
        for line in lines:
            f.write(f"{line.index}\n")
            f.write(f"{sec_to_srt(line.start)} --> {sec_to_srt(line.end)}\n")
            f.write(f"{line.text}\n\n")
    return path


def write_lrc(lines: Sequence[SubtitleLine], path: str, title: str = "", artist: str = "哈基米") -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"[ti:{title}]\n[ar:{artist}]\n[by:{APP_NAME}]\n")
        for line in lines:
            f.write(f"{sec_to_lrc(line.start)}{line.text}\n")
    return path


def write_subtitle_json(
    lines: Sequence[SubtitleLine],
    path: str,
    *,
    meta: dict[str, Any] | None = None,
) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta or {}, "lines": [line.to_dict() for line in lines]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def write_markdown(
    lines: Sequence[SubtitleLine],
    path: str,
    *,
    title: str = "",
    meta: dict[str, Any] | None = None,
) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    meta = meta or {}
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title or '哈基米歌词'}\n\n")
        if meta:
            f.write("| 项目 | 值 |\n| --- | --- |\n")
            for k, v in meta.items():
                f.write(f"| {k} | {v} |\n")
            f.write("\n")
        f.write("| 时间 | 段落 | 歌词 |\n| --- | --- | --- |\n")
        for line in lines:
            f.write(f"| {sec_to_srt(line.start)} | {line.section or '-'} | {line.text} |\n")
    return path


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ · 实时字幕</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; min-height: 100vh; font-family: "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
         background: radial-gradient(1200px 600px at 20% -10%, #3b1d5a 0%, #130f24 45%, #0a0a12 100%);
         color: #f2ecff; display: flex; flex-direction: column; align-items: center; padding: 28px 16px 60px; }
  .wrap { width: min(1080px, 100%); }
  h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: 1px; }
  .sub { color: #a99cc9; font-size: 13px; margin-bottom: 18px; }
  .player { background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.12);
            border-radius: 20px; padding: 22px 24px; backdrop-filter: blur(8px);
            box-shadow: 0 24px 60px rgba(0,0,0,.45); }
  #now { font-size: clamp(30px, 5vw, 54px); font-weight: 800; text-align: center; min-height: 1.4em;
         letter-spacing: .12em; text-shadow: 0 0 30px rgba(180,120,255,.45); transition: opacity .15s; }
  #now .on { color: #ffd166; }
  .meta { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; margin: 14px 0 4px; font-size: 12px; color: #b6a9d6; }
  .meta span { background: rgba(255,255,255,.06); padding: 3px 10px; border-radius: 999px; }
  audio { width: 100%; margin-top: 16px; filter: drop-shadow(0 8px 20px rgba(0,0,0,.4)); }
  .timeline { position: relative; height: 8px; background: rgba(255,255,255,.08); border-radius: 99px; margin-top: 18px; overflow: hidden; }
  #bar { position: absolute; inset: 0 auto 0 0; width: 0; background: linear-gradient(90deg, #9b5cff, #ff7ad9); border-radius: 99px; }
  .controls { display: flex; gap: 10px; margin-top: 12px; justify-content: center; }
  button { background: rgba(255,255,255,.08); color: #f2ecff; border: 1px solid rgba(255,255,255,.16);
           border-radius: 10px; padding: 8px 16px; cursor: pointer; font-size: 13px; }
  button:hover { background: rgba(255,255,255,.16); }
  .lyrics { margin-top: 22px; max-height: 46vh; overflow-y: auto; padding-right: 6px; scroll-behavior: smooth; }
  .line { display: flex; gap: 14px; padding: 9px 12px; border-radius: 12px; color: #9d90bd; font-size: 16px;
          transition: background .2s, color .2s, transform .2s; }
  .line .t { font-variant-numeric: tabular-nums; font-size: 12px; min-width: 64px; opacity: .7; padding-top: 3px; }
  .line .txt { letter-spacing: .08em; }
  .line .sec { margin-left: auto; font-size: 11px; opacity: .55; padding-top: 4px; }
  .line.active { background: linear-gradient(90deg, rgba(155,92,255,.28), rgba(255,122,217,.12)); color: #fff;
                 transform: translateX(4px); box-shadow: inset 3px 0 0 #ffd166; }
  .line.active .txt { font-weight: 700; }
  .footer { text-align: center; color: #6f6688; font-size: 12px; margin-top: 24px; }
  ::-webkit-scrollbar { width: 8px; } ::-webkit-scrollbar-thumb { background: rgba(255,255,255,.16); border-radius: 8px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <div class="sub">__BRAND__ · __BRAND_CN__ · 实时字幕文档（卡拉 OK 模式，可直接双击打开）</div>
  <div class="player">
    <div id="now">准备播放…</div>
    <div class="meta" id="meta"></div>
    <audio id="audio" controls preload="metadata" src="__AUDIO__"></audio>
    <div class="timeline"><div id="bar"></div></div>
    <div class="controls">
      <button id="prev">上一句</button>
      <button id="toggle">播放 / 暂停</button>
      <button id="next">下一句</button>
      <button id="copy">复制当前歌词</button>
    </div>
    <div class="lyrics" id="lyrics"></div>
  </div>
  <div class="footer">由 __BRAND__ · __BRAND_CN__ 生成 · 字幕与音频同目录，移动时请一起移动</div>
</div>
<script>
const DATA = __DATA__;
const audio = document.getElementById('audio');
const nowEl = document.getElementById('now');
const lyricsEl = document.getElementById('lyrics');
const bar = document.getElementById('bar');
const metaEl = document.getElementById('meta');

function fmt(t){ t=Math.max(0,t); const m=Math.floor(t/60), s=t-m*60; return m+':'+s.toFixed(2).padStart(5,'0'); }

// 元信息
const meta = DATA.meta || {};
Object.entries(meta).forEach(([k,v]) => {
  if (v === null || v === undefined || v === '') return;
  const el = document.createElement('span');
  el.textContent = k + '：' + v;
  metaEl.appendChild(el);
});

// 歌词列表
DATA.lines.forEach((line, i) => {
  const row = document.createElement('div');
  row.className = 'line';
  row.dataset.i = i;
  row.innerHTML = '<span class="t"></span><span class="txt"></span><span class="sec"></span>';
  row.querySelector('.t').textContent = fmt(line.start);
  row.querySelector('.txt').textContent = line.text;
  row.querySelector('.sec').textContent = line.section || '';
  row.addEventListener('click', () => { audio.currentTime = line.start + 0.001; audio.play(); });
  lyricsEl.appendChild(row);
});
const rows = [...lyricsEl.children];

let active = -1;
function findLine(t){
  let lo=0, hi=DATA.lines.length-1, ans=-1;
  while(lo<=hi){ const mid=(lo+hi)>>1; if(DATA.lines[mid].start <= t){ ans=mid; lo=mid+1; } else hi=mid-1; }
  return ans;
}
function render(t){
  const i = findLine(t);
  if (DATA.lines[i] && t > DATA.lines[i].end + 0.8) { /* 间奏保持上一句 */ }
  if (i !== active) {
    if (active >= 0) rows[active]?.classList.remove('active');
    active = i;
    if (i >= 0) {
      rows[i]?.classList.add('active');
      rows[i]?.scrollIntoView({block:'center', behavior:'smooth'});
      nowEl.textContent = DATA.lines[i].text;
    }
  }
  if (i >= 0) {
    const line = DATA.lines[i];
    // 字符级高亮
    let html = '';
    for (const c of line.chars) {
      const on = t >= c.s - 0.01;
      html += on ? '<span class="on">' + c.w + '</span>' : c.w;
    }
    if (nowEl.dataset.line !== String(i)) { nowEl.innerHTML = html; nowEl.dataset.line = String(i); }
    else { nowEl.innerHTML = html; }
  }
  if (audio.duration) bar.style.width = Math.min(100, t/audio.duration*100) + '%';
}
function tick(){ render(audio.currentTime || 0); requestAnimationFrame(tick); }
requestAnimationFrame(tick);

document.getElementById('toggle').onclick = () => audio.paused ? audio.play() : audio.pause();
document.getElementById('prev').onclick = () => { const i=Math.max(0,findLine(audio.currentTime)-1); audio.currentTime=DATA.lines[i]?.start||0; };
document.getElementById('next').onclick = () => { const i=Math.min(DATA.lines.length-1,findLine(audio.currentTime)+1); audio.currentTime=DATA.lines[i]?.start||0; };
document.getElementById('copy').onclick = async () => {
  const i=findLine(audio.currentTime);
  try { await navigator.clipboard.writeText(i>=0?DATA.lines[i].text:''); document.getElementById('copy').textContent='已复制'; setTimeout(()=>document.getElementById('copy').textContent='复制当前歌词',1200);} catch(e){}
};
</script>
</body>
</html>
"""


def write_html(
    lines: Sequence[SubtitleLine],
    path: str,
    *,
    title: str,
    audio_name: str,
    meta: dict[str, Any] | None = None,
) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta or {},
        "lines": [line.to_dict() for line in lines],
    }
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html_text = (
        _HTML_TEMPLATE.replace("__TITLE__", html.escape(title or "哈基米歌词"))
        .replace("__AUDIO__", html.escape(audio_name))
        .replace("__BRAND__", html.escape(APP_NAME))
        .replace("__BRAND_CN__", html.escape(APP_NAME_CN))
        .replace("__DATA__", data_json)
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_text)
    return path


# ---------------------------------------------------------------------------
# 兼容接口
# ---------------------------------------------------------------------------


def make_track_srt(
    note2word,
    track_idx: int,
    output_path: str,
    reset_gap: float = 1.0,
    max_duration: float = 2.5,
    max_len: int = 12,
) -> None:
    """上游兼容：``note2word`` 为 ``[{pretty_midi.Note: word}]``。"""
    if track_idx >= len(note2word) or not note2word[track_idx]:
        logger.warning("SRT: 轨道 %d 不存在或为空，跳过", track_idx)
        return
    items = [
        {"word": w, "start": float(n.start), "end": float(min(n.end, n.start + max_duration))}
        for n, w in note2word[track_idx].items()
    ]
    lines = _make_lines(items, reset_gap=reset_gap, max_len=max_len)
    write_srt(lines, output_path)
    logger.info("SRT written: %s (%d subtitles)", output_path, len(lines))


__all__ = [
    "SubtitleLine",
    "build_group_subtitles",
    "build_stream_subtitles",
    "build_subtitles",
    "build_track_subtitles",
    "make_track_srt",
    "sec_to_lrc",
    "sec_to_srt",
    "write_html",
    "write_lrc",
    "write_markdown",
    "write_srt",
    "write_subtitle_json",
]
