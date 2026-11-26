"""端到端渲染流程：分析 → 歌词 → 合成 → 混音 → 字幕 / 文件输出。

:func:`render_midi` 是 CLI、网页后端与启动器共用的唯一入口。
通过 ``progress_cb`` 可以把「阶段 / 百分比 / 日志 / 字幕行」实时推送给前端。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hachimi import models as model_factory
from hachimi.analysis import AnalysisResult, analyze
from hachimi.audio import apply_post_chain
from hachimi.audio_io import write_audio
from hachimi.config import (
    BAD_NOTES_DIRNAME,
    DEFAULT_VOICE,
    MODEL_DIR,
    OUTPUT_DIR,
    PREMIX_DIRNAME,
    RENDER_TAIL_SECONDS,
    SR,
    STEMS_DIRNAME,
    SUBTITLE_VIDEO_MAX_CHARS,
    TRACK_LOW_CUT_FREQ,
    VOICE_MODES,
    ZIP_SUFFIX,
    RenderOptions,
    ensure_dirs,
    ensure_writable_dir,
)
from hachimi.lyrics import compose_lyrics
from hachimi.loudness import limit_peaks
from hachimi.params import build_project_params, write_project_params
from hachimi.qc import (
    assess_bad_notes,
    write_bad_and_all_midi,
    write_bad_note_midis,
    write_percussion_midi,
)
from hachimi.mixing import (
    TrackPlacement,
    auto_mix_plan,
    instrument_auto_gain,
    master_bus,
    place_track,
    room_reverb,
)
from hachimi.render import RenderStats, build_render_notes, render_notes
from hachimi.report import build_instrument_report, write_instrument_report
from hachimi.verify import build_chord_groups, check_stem_energy, verify_polyphony
from hachimi.subtitles import (
    build_group_subtitles,
    build_subtitles,
    build_track_subtitles,
    write_html,
    write_lrc,
    write_markdown,
    write_srt,
    write_subtitle_json,
)
from hachimi.video import render_subtitle_video

logger = logging.getLogger(__name__)


class RenderCancelled(RuntimeError):
    """用户取消渲染。"""


ProgressCb = Callable[[dict[str, Any]], None]


# ---------------------------------------------------------------------------
# 结果
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    project: str
    midi_path: str
    output_dir: str
    audio_path: str
    duration: float
    render_seconds: float
    files: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    lyrics_stats: dict[str, Any] = field(default_factory=dict)
    tracks: list[dict[str, Any]] = field(default_factory=list)
    subtitles: list[dict[str, Any]] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    stems: dict[str, str] = field(default_factory=dict)
    instrument_report: dict[str, Any] = field(default_factory=dict)
    mix_report: dict[str, Any] = field(default_factory=dict)
    subtitle_videos: dict[str, Any] = field(default_factory=dict)
    premix: dict[str, str] = field(default_factory=dict)
    bad_notes: dict[str, Any] = field(default_factory=dict)
    supplement_midi: dict[str, Any] = field(default_factory=dict)
    percussion_midi: dict[str, Any] = field(default_factory=dict)
    project_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "midi_path": self.midi_path,
            "output_dir": self.output_dir,
            "audio_path": self.audio_path,
            "duration": round(self.duration, 3),
            "render_seconds": round(self.render_seconds, 2),
            "files": self.files,
            "stats": self.stats,
            "analysis": self.analysis,
            "lyrics_stats": self.lyrics_stats,
            "tracks": self.tracks,
            "subtitles": self.subtitles,
            "verification": self.verification,
            "stems": self.stems,
            "instrument_report": self.instrument_report,
            "mix_report": self.mix_report,
            "subtitle_videos": self.subtitle_videos,
            "premix": self.premix,
            "bad_notes": self.bad_notes,
            "supplement_midi": self.supplement_midi,
            "percussion_midi": self.percussion_midi,
            "project_params": self.project_params,
        }


# ---------------------------------------------------------------------------
# 进度
# ---------------------------------------------------------------------------


class StageProgress:
    """把各阶段进度映射到整体 0~100%。"""

    def __init__(self, cb: ProgressCb | None) -> None:
        self.cb = cb

    def emit(self, stage: str, pct: float, message: str = "", **extra: Any) -> None:
        if not self.cb:
            return
        payload = {
            "type": "stage",
            "stage": stage,
            "pct": round(max(0.0, min(100.0, pct)), 2),
            "message": message,
            "ts": time.time(),
        }
        payload.update(extra)
        try:
            self.cb(payload)
        except Exception:  # pragma: no cover
            logger.exception("progress callback failed")

    def log(self, message: str, level: str = "info") -> None:
        if not self.cb:
            return
        try:
            self.cb({"type": "log", "level": level, "message": message, "ts": time.time()})
        except Exception:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# 模型缓存
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def load_lyrics_model(kind: str, cls: str, progress: StageProgress | None = None):
    """加载或训练一个歌词模型（带进程内 + 磁盘缓存）。"""
    key = (kind, cls)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    ensure_dirs()
    os.makedirs(MODEL_DIR, exist_ok=True)
    if kind == "transformer":
        path = os.path.join(MODEL_DIR, f"{cls}.npz")
    else:
        path = os.path.join(MODEL_DIR, f"{cls}.json")

    if os.path.exists(path):
        try:
            model = model_factory.load_model(kind, path)
            logger.info("加载已有歌词模型 [%s] %s", kind, cls)
            if progress:
                progress.log(f"加载已有模型 [{kind}] {cls}")
            _MODEL_CACHE[key] = model
            return model
        except Exception as exc:  # pragma: no cover
            logger.warning("模型加载失败（%s），将重新训练：%s", path, exc)

    if progress:
        progress.log(f"训练歌词模型 [{kind}] {cls} …")
    if kind == "transformer":
        from hachimi.config import TRANSFORMER

        model = model_factory.create_model("transformer", [cls], **TRANSFORMER)
    else:
        model = model_factory.create_model("markov", [cls])

    try:
        model.save(path)
        if kind == "markov" and progress:
            try:
                from hachimi.animate import draw_markov_graph

                draw_markov_graph(
                    model.to_dict(),
                    save_path=os.path.join(MODEL_DIR, f"{cls}.png"),
                )
            except Exception as exc:  # pragma: no cover - 绘图失败不影响主流程
                logger.debug("马尔科夫图绘制失败：%s", exc)
    except Exception as exc:  # pragma: no cover
        logger.warning("模型保存失败：%s", exc)

    _MODEL_CACHE[key] = model
    return model


# ---------------------------------------------------------------------------
# 轨道参数
# ---------------------------------------------------------------------------


def resolve_track_options(
    analysis: AnalysisResult,
    options: RenderOptions,
) -> list[dict[str, Any]]:
    """合并「分析建议」与「用户选择」，得到最终每轨参数。"""
    user = options.track_options()
    has_explicit_tracks = bool(options.tracks)
    tracks: list[dict[str, Any]] = []
    for t in analysis.tracks:
        u = user.get(t.index, {})
        # 前端显式提交轨道列表时，未提交的轨道视为「不参与」，
        # 避免用户取消勾选后又被分析建议重新打开。
        selected = u.get("selected", False if has_explicit_tracks else t.suggested_selected)
        if t.n_notes == 0 or t.is_drum:
            selected = bool(u.get("selected", False))
        if "voice" in u and u["voice"]:
            voice = u["voice"]
        elif options.tracks:
            voice = options.voice
        else:
            voice = t.suggested_voice or options.voice
        if options.voice != DEFAULT_VOICE and not options.tracks:
            voice = options.voice
        octave = int(u.get("octave", t.suggested_octave))
        gain = float(u.get("gain", t.suggested_gain if t.suggested_gain else 1.0))
        tracks.append(
            {
                "index": t.index,
                "name": t.name,
                "instrument": t.instrument,
                "role": t.role,
                "role_label": t.role_label,
                "n_notes": t.n_notes,
                "selected": bool(selected),
                "voice": voice,
                "octave": max(-3, min(3, octave)),
                "gain": max(0.0, min(2.0, gain)),
                "suggested": t.to_dict(),
            }
        )
    return tracks


def _safe_name(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return name[:80] or "hachimi"


def _stem_filename(track: dict[str, Any], fmt: str = "mp3") -> str:
    """分轨文件名：乐器 + 轨道号 + 语音类型 + 八度修改值。

    例如 ``Cello_02_自动混合_+0.mp3``、
    ``Violin_00_哈基米_+1.mp3``。
    """
    instrument = str(track.get("instrument") or track.get("name") or "Track")
    instrument = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", instrument)
    instrument = re.sub(r"\s+", "_", instrument).strip("._") or "Track"
    instrument = instrument[:40]

    voice = str(track.get("voice") or "auto")
    if voice == "auto":
        voice = "自动混合"
    voice = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", voice)[:20] or "自动混合"

    index = int(track.get("index", 0))
    octave = int(track.get("octave", 0))
    ext = "wav" if str(fmt).lower() == "wav" else "mp3"
    return f"{instrument}_{index:02d}_{voice}_{octave:+d}.{ext}"


def _video_slot_counts(slots: list[Any]) -> Counter[int]:
    """每个主声部轨道的歌词槽数；只统计真正有字的槽。"""
    counts: Counter[int] = Counter()
    for slot in slots:
        track = int(getattr(slot, "primary_track", -1))
        if track >= 0 and str(getattr(slot, "word", "") or ""):
            counts[track] += 1
    return counts


def _video_group_info(
    track_options: dict[int, dict[str, Any]],
    tracks: list[int],
    slots: list[Any],
    label: str,
) -> dict[str, Any]:
    """单条字幕视频的轨道组信息；保留兼容字段 index / name / instrument。"""
    if not tracks:
        return {}
    counts = _video_slot_counts(slots)
    primary = max(tracks, key=lambda t: (counts[t], -t))
    options = [track_options.get(t, {}) for t in tracks]
    names = [
        str(o.get("name", f"Track {t}"))
        for t, o in zip(tracks, options)
    ]
    instruments: list[str] = []
    for option in options:
        name = str(option.get("instrument", "") or "")
        if name and name not in instruments:
            instruments.append(name)
    return {
        "index": int(primary),
        "name": names[0] if len(names) == 1 else f"{names[0]} 等 {len(names)} 轨",
        "instrument": " / ".join(instruments),
        "role": label,
        "tracks": tracks,
        "track_names": names,
        "track_count": len(tracks),
        "track_label": "、".join(str(t) for t in tracks),
        "slots": sum(counts[t] for t in tracks),
    }


def _event_in_range(ev: dict[str, Any], start: float, limit: float | None) -> bool:
    if float(ev["end"]) <= start:
        return False
    if limit is not None and float(ev["start"]) >= limit:
        return False
    return True


_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
_STORED_EXTS = _AUDIO_EXTS | {".mp4", ".mov", ".mkv", ".webm"}


def _flatten_files(files: dict[str, Any]) -> list[str]:
    """把 ``files`` 中的单路径 / 嵌套路径展平。"""
    out: list[str] = []
    for value in files.values():
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                if isinstance(item, str):
                    out.append(item)
    return out


def _write_zip(paths: list[str], zip_path: str, out_dir: str) -> str:
    """把指定文件打包为 ZIP（音频存储、文本压缩），保留相对目录结构。"""
    base = Path(out_dir).resolve()
    target = Path(zip_path).resolve()
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zf:
        for raw in sorted(set(paths)):
            path = Path(raw)
            if not path.is_file() or path.resolve() == target:
                continue
            try:
                arcname = path.resolve().relative_to(base).as_posix()
            except ValueError:
                arcname = path.name
            compress = (
                zipfile.ZIP_STORED
                if path.suffix.lower() in _STORED_EXTS
                else zipfile.ZIP_DEFLATED
            )
            zf.write(path, arcname, compress_type=compress)
    return zip_path


def _merge_render_stats(items: list[RenderStats], audio_seconds: float, elapsed: float) -> RenderStats:
    """把逐轨渲染统计合并成整首统计。"""
    merged = RenderStats()
    merged.audio_seconds = audio_seconds
    merged.elapsed = elapsed
    merged.realtime_factor = audio_seconds / elapsed if elapsed > 0 else 0.0
    for item in items:
        merged.n_notes += item.n_notes
        merged.n_units += item.n_units
        merged.n_unique_units += item.n_unique_units
        merged.cache_hits += item.cache_hits
        merged.unit_bytes += item.unit_bytes
        merged.n_skipped += item.n_skipped
        merged.skipped_notes.extend(item.skipped_notes)
        merged.n_silent_units += item.n_silent_units
        merged.silent_units.extend(item.silent_units)
        merged.n_dynamic_notes += item.n_dynamic_notes
        merged.n_audibility_notes += item.n_audibility_notes
        merged.n_tremolo_notes += item.n_tremolo_notes
        merged.n_attack_softened += item.n_attack_softened
        merged.n_legato_links += item.n_legato_links
        merged.n_consonant_notes += item.n_consonant_notes
        merged.n_plosive_notes += item.n_plosive_notes
        merged.duration_quantum = item.duration_quantum
        if item.device == "cuda":
            merged.device = "cuda"
            merged.gpu_memory_mb = max(merged.gpu_memory_mb, item.gpu_memory_mb)
        merged.workers = max(merged.workers, item.workers)
        merged.notes_per_second = merged.n_notes / elapsed if elapsed > 0 else 0.0
    return merged


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def render_midi(
    midi_path: str,
    options: RenderOptions | None = None,
    *,
    output_dir: str | None = None,
    project_name: str | None = None,
    progress_cb: ProgressCb | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> PipelineResult:
    """把 MIDI / MusicXML 渲染成哈基米音乐，并输出音频 + 歌词 + 字幕。

    Args:
        midi_path: MIDI 或 MusicXML / MXL 文件路径。
        options: 渲染参数；``None`` 使用默认（全部语音模式、自动选择轨道）。
        output_dir: 自定义输出目录，默认 ``output/<project>``。
        project_name: 项目名，默认 MIDI 文件名。
        progress_cb: 进度回调（收到 stage / log / subtitle / done 事件）。
        cancel_cb: 返回 True 时尽快停止。
    """
    started = time.perf_counter()
    options = options or RenderOptions()
    ensure_dirs()

    if not os.path.exists(midi_path):
        raise FileNotFoundError(f"MIDI / 乐谱文件不存在：{midi_path}")

    stem = _safe_name(project_name or os.path.splitext(os.path.basename(midi_path))[0])
    requested_dir = output_dir or os.path.join(OUTPUT_DIR, stem)
    # 只读安装目录 / Low integrity 环境下，输出目录可能不可写；
    # 回退到 _runtime 或用户可写目录，保证渲染不中断。
    out_dir = ensure_writable_dir(requested_dir, stem)
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"无法创建输出目录：{out_dir}（{exc}）。"
            "请把项目放到可写目录，或检查杀毒软件的“受控文件夹访问”。"
        ) from exc

    progress = StageProgress(progress_cb)

    def check_cancel() -> None:
        if cancel_cb and cancel_cb():
            raise RenderCancelled("已取消")

    # ================= 1. 分析 =================
    progress.emit("analyze", 2, "读取 MIDI / 乐谱与轨道信息")
    analysis = analyze(midi_path)
    check_cancel()

    # 首次解析得到的建议选择 → 与用户参数合并
    initial_tracks = resolve_track_options(analysis, options)
    selected = [t["index"] for t in initial_tracks if t["selected"]]
    if not selected:
        selected = [t.index for t in analysis.tracks if t.suggested_selected]
    default_selected = [t.index for t in analysis.tracks if t.suggested_selected]
    if set(selected) != set(default_selected):
        progress.emit("analyze", 5, "按所选轨道重新分析和弦与结构")
        analysis = analyze(midi_path, selected_tracks=selected)
        check_cancel()

    track_options = {t["index"]: t for t in resolve_track_options(analysis, options)}
    selected = [i for i, t in track_options.items() if t["selected"]]
    audible = [
        i for i in selected
        if float(track_options[i].get("gain", 1.0)) > 0
    ]
    if not selected:
        raise ValueError("没有勾选任何轨道，无法生成。")
    progress.emit(
        "analyze",
        8,
        f"分析完成：{len(analysis.tracks)} 轨 / {len(analysis.chords)} 个和弦 / "
        f"{len(analysis.sections)} 段 / {len(analysis.motifs)} 个动机",
    )

    # ================= 2. 歌词模型 =================
    needed_classes: set[str] = set()
    for idx in selected:
        mode = track_options[idx]["voice"]
        needed_classes.update(VOICE_MODES.get(mode, ["哈基米"]))
    if not needed_classes:
        needed_classes = {"哈基米"}

    progress.emit("model", 10, f"准备歌词模型：{'、'.join(sorted(needed_classes))}")
    models = {}
    for cls in sorted(needed_classes):
        models[cls] = load_lyrics_model(options.model_kind, cls, progress)
        check_cancel()
    progress.emit("model", 16, "歌词模型就绪")

    # ================= 3. 歌词 =================
    progress.emit("lyrics", 18, "和弦 / 动机感知的歌词生成")
    lyrics = compose_lyrics(
        analysis,
        models,
        track_options,
        seed=options.seed,
        default_voice=options.voice,
        syntax_weight=options.syntax_weight,
        section_plans=options.section_plans,
        decomposed_merge=options.decomposed_merge,
        onset_tol=options.chord_onset_tol,
        merge_gap=options.chord_merge_gap,
        progress_cb=lambda i, n: progress.emit("lyrics", 18 + 10 * i / max(1, n), "歌词生成"),
    )
    check_cancel()

    # ================= 4. 字幕（先生成，实时推送） =================
    progress.emit("subtitles", 33, "生成实时字幕")
    time_limit = options.end_time
    if options.max_seconds is not None:
        cap = options.start_time + options.max_seconds
        time_limit = cap if time_limit is None else min(time_limit, cap)

    def slot_in_range(slot) -> bool:
        if slot.end <= options.start_time:
            return False
        if time_limit is not None and slot.start >= time_limit:
            return False
        return True

    sub_slots = [
        s for s in lyrics.slots
        if slot_in_range(s) and (not selected or set(s.track_ids) & set(selected))
    ]
    melody_tracks = [
        idx for idx in selected if track_options[idx].get("role") == "melody"
    ]
    primary_voice = None
    if melody_tracks:
        melody_set = set(melody_tracks)
        for slot in sub_slots:
            if set(slot.track_ids) & melody_set:
                primary_voice = lyrics.voices.get(slot.index)
                if primary_voice:
                    break
    subtitle_lines = build_subtitles(
        sub_slots,
        selected_tracks=selected,
        preferred_tracks=melody_tracks,
        primary_tracks=melody_tracks,
        voices=lyrics.voices,
        preferred_voice=primary_voice,
    )
    if time_limit is not None:
        for line in subtitle_lines:
            line.end = min(line.end, max(line.start + 0.2, time_limit))
            for ch in line.chars:
                ch["e"] = min(float(ch["e"]), line.end)
    sub_events = [e for e in lyrics.events if _event_in_range(e, options.start_time, time_limit)]
    track_subs = {
        idx: build_track_subtitles(sub_events, idx) for idx in selected
    }
    if time_limit is not None:
        for lines in track_subs.values():
            for line in lines:
                line.end = min(line.end, max(line.start + 0.2, time_limit))
    subtitle_payload = [line.to_dict() for line in subtitle_lines]

    # ---- 绿幕字幕视频：全部轨道合并成一条，每次最多一行 ----
    # 每个歌词槽只出现一次（按 primary_track 归属），既覆盖全部可听声部，
    # 又不会因为和弦 / 齐奏重复。所有字共用同一条时间线，视频里任何时刻
    # 都只有一行字幕；相邻槽按时间顺序合并，短句间尽量保持连续。
    audible_set = set(audible)
    video_slots = [
        s for s in sub_slots
        if set(s.track_ids) & audible_set and str(getattr(s, "word", "") or "")
    ]
    video_tracks = sorted(
        {
            int(getattr(s, "primary_track", -1))
            for s in video_slots
            if int(getattr(s, "primary_track", -1)) >= 0
        }
    )
    video_lines = build_group_subtitles(
        video_slots,
        video_tracks,
        reset_gap=1.1,
        max_len=SUBTITLE_VIDEO_MAX_CHARS,
        max_line_seconds=8.0,
        min_seconds=0.35,
        hold_gap=1.0,
    )
    video_track_info = {
        "all": _video_group_info(
            track_options, video_tracks, video_slots, "全部字幕"
        )
    }

    def _line_slot_ids(lines: list[Any]) -> set[int]:
        ids: set[int] = set()
        for line in lines:
            for ch in getattr(line, "chars", []) or []:
                slot = ch.get("slot")
                if slot is not None:
                    ids.add(int(slot))
        return ids

    video_covered_slots = _line_slot_ids(video_lines)
    video_input_slots = {
        int(s.index) for s in video_slots
    }
    video_coverage = {
        "input_slots": len(video_input_slots),
        "covered_slots": len(video_covered_slots),
        "missing_slots": len(video_input_slots - video_covered_slots),
        "tracks": video_tracks,
        "lines": len(video_lines),
        "one_line": True,
    }
    if video_coverage["missing_slots"]:
        progress.log(
            "字幕视频仍有 %d 个歌词槽未覆盖" % video_coverage["missing_slots"],
            level="warning",
        )

    if progress_cb:
        progress_cb(
            {
                "type": "subtitle",
                "lines": subtitle_payload,
                "sections": [s.to_dict() for s in analysis.sections],
                "voices": {str(k): v for k, v in lyrics.voices.items()},
                "stats": lyrics.stats,
                "ts": time.time(),
            }
        )

    # ================= 5. 逐轨渲染 + 自动混音 =================
    progress.emit("render", 35, "按乐器逐轨合成（力度 / 表情 / 可听度）")
    render_notes_list = build_render_notes(
        lyrics.events,
        track_options,
        start_time=options.start_time,
        end_time=options.end_time,
        max_seconds=options.max_seconds,
        sample_set=options.sample_set,
    )
    if not render_notes_list:
        raise ValueError("所选时间段内没有可渲染的音符。")

    render_started = time.perf_counter()
    global_t0 = max(float(options.start_time), min(n.start for n in render_notes_list))
    global_t1 = max(n.end for n in render_notes_list) + max(0.0, float(options.tail_seconds))
    if options.max_seconds is not None:
        # 试听/测试时不要把「5 秒素材」自动扩成 8 秒文件；尾巴也受上限约束
        global_t1 = min(
            global_t1,
            options.start_time + float(options.max_seconds) + max(0.0, float(options.tail_seconds)),
        )
    mix_samples = max(1, int(np.ceil((global_t1 - global_t0) * SR)) + 1)
    mix = np.zeros((mix_samples, 2), dtype=np.float32)
    reverb_bus: np.ndarray | None = (
        np.zeros((mix_samples, 2), dtype=np.float32) if options.auto_mix else None
    )

    mix_tracks = [track_options[idx] for idx in audible if track_options.get(idx)]
    mix_plan = auto_mix_plan(mix_tracks, enabled=bool(options.auto_mix))
    placements: list[dict[str, Any]] = []

    stems: dict[str, str] = {}
    stem_energy: dict[int, list[dict[str, Any]]] = {}
    stem_entries: list[dict[str, Any]] = []
    premixs: dict[str, str] = {}
    premix_entries: list[dict[str, Any]] = []
    track_stats: list[RenderStats] = []
    audio_format = options.audio_format if options.audio_format in ("mp3", "wav") else "mp3"

    stem_dir = os.path.join(out_dir, STEMS_DIRNAME)
    if options.write_stems:
        os.makedirs(stem_dir, exist_ok=True)
        # 清掉上一次生成遗留的分轨，避免旧轨道混淆
        for old in os.listdir(stem_dir):
            if old == "stems.json" or old.endswith((".wav", ".mp3", ".flac", ".ogg")):
                try:
                    os.remove(os.path.join(stem_dir, old))
                except OSError:
                    pass

    premix_dir = os.path.join(out_dir, PREMIX_DIRNAME)
    if options.write_premix:
        os.makedirs(premix_dir, exist_ok=True)
        for old in os.listdir(premix_dir):
            if old.endswith((".wav", ".mp3", ".flac", ".ogg", ".json")):
                try:
                    os.remove(os.path.join(premix_dir, old))
                except OSError:
                    pass

    for pos, idx in enumerate(audible):
        check_cancel()
        track_notes = [n for n in render_notes_list if n.track == idx]
        if not track_notes:
            progress.log(f"轨道 {idx} 没有可渲染音符，跳过")
            continue

        last_emit = [0.0]

        def track_progress(done: int, total: int, message: str, *, _idx=idx, _pos=pos) -> None:
            now = time.time()
            if now - last_emit[0] < 0.2 and done < total:
                return
            last_emit[0] = now
            overall = 35.0 + 46.0 * (_pos + done / max(1, total)) / max(1, len(audible))
            progress.emit(
                "render",
                overall,
                f"轨道 {_idx}：{message}（{done}/{total}）",
            )

        track_audio, stem_stats = render_notes(
            track_notes,
            sr=SR,
            velocity_gamma=options.velocity_gamma,
            workers=options.workers,
            duration_quantum=options.duration_quantum,
            max_units=options.max_units,
            gpu=options.gpu,
            gpu_memory_mb=options.gpu_memory_mb,
            align_start=global_t0,
            total_samples=mix_samples,
            progress_cb=track_progress,
            cancel_cb=cancel_cb,
        )
        track_stats.append(stem_stats)

        if options.auto_mix:
            # 每轨干声：低切 + 防削波；不做全局 spread / delay，避免破坏相位
            track_audio = apply_post_chain(
                track_audio,
                sr=SR,
                low_cut=TRACK_LOW_CUT_FREQ,
                spread=0.0,
                taps=0,
                damping=18500,
                normalize="limit",
                inplace=True,
            )
        else:
            track_audio = apply_post_chain(track_audio, sr=SR, normalize="limit", inplace=True)

        track = track_options[idx]
        mixed_name = _stem_filename(track, audio_format)
        # 混音前干声：此时尚未做声像 / 深度 / 自动电平 / 混响，保留系统低切与
        # 防削波处理。统一用 FLAC 无损压缩，随后再打成一个 ZIP。
        if options.write_premix:
            premix_name = os.path.splitext(mixed_name)[0] + ".premix.flac"
            premix_base = os.path.splitext(premix_name)[0]
            premix_path = write_audio(
                os.path.join(premix_dir, premix_base), track_audio, SR, "flac"
            )
            premixs[str(idx)] = premix_path
            premix_entries.append(
                {
                    "index": idx,
                    "name": track.get("name", f"Track {idx}"),
                    "instrument": track.get("instrument", ""),
                    "voice": track.get("voice", ""),
                    "octave": track.get("octave", 0),
                    "gain": track.get("gain", 1.0),
                    "file": os.path.basename(premix_path),
                    "n_notes": len(track_notes),
                    "duration": round(len(track_audio) / SR, 3),
                    "loudness_db": round(
                        float(
                            20.0
                            * np.log10(
                                np.sqrt(np.mean(np.square(track_audio.astype(np.float64)))) + 1e-12
                            )
                        ),
                        2,
                    ),
                    "peak": round(float(np.max(np.abs(track_audio))) if track_audio.size else 0.0, 4),
                    "stage": "premix",
                }
            )

        placement = mix_plan.get(idx)
        if placement is None:
            placement = TrackPlacement(index=idx, instrument=str(track_options[idx].get("instrument", "")))

        if options.auto_mix:
            auto_gain = instrument_auto_gain(track_audio, placement.target_rms_db)
            placement.auto_gain_db = auto_gain
            placed, phase_info = place_track(track_audio, placement, SR, auto_gain_db=auto_gain)
        else:
            auto_gain = 0.0
            placed, phase_info = track_audio, {}

        mix += placed
        if reverb_bus is not None:
            reverb_bus += placed * placement.reverb_send
        placement_info = placement.to_dict()
        placement_info.update(phase_info)
        placement_info["file"] = ""
        placements.append(placement_info)

        # 分轨与主混音共用同一套摆位 / 自动电平，保证回放关系一致
        if options.write_stems:
            stem_audio = limit_peaks(placed, SR, ceiling_db=-1.0)
            track = track_options[idx]
            filename = _stem_filename(track, audio_format)
            filename_base = os.path.splitext(filename)[0]
            path = write_audio(os.path.join(stem_dir, filename_base), stem_audio, SR, audio_format)
            filename = os.path.basename(path)
            stems[str(idx)] = path
            placement_info["file"] = f"stems/{filename}"

            groups = build_chord_groups([n for n in analysis.notes if n.track == idx])
            stem_energy[idx] = check_stem_energy(
                stem_audio,
                SR,
                global_t0,
                groups.get(idx, []),
                start_time=options.start_time,
                end_time=time_limit,
            )
            stem_entries.append(
                {
                    "index": idx,
                    "name": track.get("name", f"Track {idx}"),
                    "instrument": track.get("instrument", ""),
                    "voice": track.get("voice", ""),
                    "octave": track.get("octave", 0),
                    "gain": track.get("gain", 1.0),
                    "auto_gain_db": round(auto_gain, 2),
                    "pan": round(placement.pan, 3),
                    "depth": round(placement.depth, 3),
                    "file": filename,
                    "n_notes": len(track_notes),
                    "duration": round(len(stem_audio) / SR, 3),
                    "loudness_db": round(
                        float(
                            20.0
                            * np.log10(
                                np.sqrt(np.mean(np.square(stem_audio.astype(np.float64)))) + 1e-12
                            )
                        ),
                        2,
                    ),
                    "peak": round(float(np.max(np.abs(stem_audio))) if stem_audio.size else 0.0, 4),
                    "render_stats": stem_stats.to_dict(),
                }
            )
            del stem_audio

        progress.emit(
            "stems",
            81.0 + 4.0 * (pos + 1) / max(1, len(audible)),
            f"轨道 {idx} 已摆位 / 混音（{len(track_notes)} 音符）",
        )
        del placed, track_audio

    if not track_stats:
        raise ValueError("所选轨道内没有可渲染的音符。")

    # ---- 母带 / 总线 ----
    progress.emit("post", 85, "自动混音母带：响度 / 限幅 / 空间")
    if options.auto_mix:
        if reverb_bus is not None and np.any(reverb_bus):
            wet = room_reverb(reverb_bus, SR, mix=1.0)
            wet *= 0.85
            mix += wet
            del wet
        mix, mix_info = master_bus(mix, SR, room_mix=0.0)
        mix_info["send_reverb"] = True
        del reverb_bus
    else:
        mix = apply_post_chain(mix, sr=SR, normalize=options.normalize, inplace=True)
        mix_info = {
            "enabled": False,
            "loudness_db": round(
                float(20.0 * np.log10(np.sqrt(np.mean(np.square(mix.astype(np.float64)))) + 1e-12)),
                2,
            ),
            "peak": round(float(np.max(np.abs(mix))) if mix.size else 0.0, 4),
        }
    check_cancel()
    mix_duration = mix_samples / SR
    render_elapsed = max(1e-6, time.perf_counter() - render_started)
    rstats = _merge_render_stats(track_stats, mix_duration, render_elapsed)
    rstats.peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    rstats.audio_seconds = mix_duration
    mix_info["tracks"] = placements
    mix_info["enabled"] = bool(options.auto_mix)
    mix_info["n_dynamic_notes"] = rstats.n_dynamic_notes
    mix_info["n_audibility_notes"] = rstats.n_audibility_notes
    mix_info["n_tremolo_notes"] = rstats.n_tremolo_notes
    mix_info["n_consonant_notes"] = rstats.n_consonant_notes
    mix_info["n_plosive_notes"] = rstats.n_plosive_notes
    progress.emit("render", 86, "逐轨合成与自动混音完成", render=rstats.to_dict(), mix=mix_info)

    # ================= 6. 写混音 + 分轨清单 =================
    progress.emit("write", 87, "写入主混音音频")
    audio_path = write_audio(os.path.join(out_dir, stem), mix, SR, audio_format)
    files: dict[str, Any] = {"audio": audio_path}
    del mix

    if stems:
        files["stems"] = stems
        index_path = os.path.join(stem_dir, "stems.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(
                {"tracks": stem_entries, "mix": mix_info},
                f,
                ensure_ascii=False,
                indent=2,
            )
        files["stems_manifest"] = index_path

    premix_zip = ""
    if premixs:
        files["premix"] = premixs
        premix_index = os.path.join(premix_dir, "stems_premix.json")
        with open(premix_index, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "format": "flac",
                    "stage": "混音前（已做系统低切 / 防削波，未做声像、深度、自动电平与混响）",
                    "tracks": premix_entries,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        files["premix_manifest"] = premix_index
        progress.emit("write", 92, "打包混音前干声（FLAC + ZIP）")
        premix_zip = os.path.join(out_dir, f"{stem}_stems_premix.zip")
        _write_zip(
            list(premixs.values()) + [premix_index, premix_zip],
            premix_zip,
            out_dir,
        )
        files["premix_zip"] = premix_zip

    # ---- 和弦音发声校验 ----
    progress.emit("verify", 93, "校验和弦音是否全部发声")
    track_names = {idx: track_options[idx].get("name", f"Track {idx}") for idx in audible}
    verification = verify_polyphony(
        [n for n in analysis.notes if n.track in set(audible)],
        lyrics.events,
        render_notes_list,
        stats=rstats,
        track_names=track_names,
        start_time=options.start_time,
        end_time=time_limit,
        stem_energy=stem_energy,
    )
    verification["stems"] = {k: os.path.basename(v) for k, v in stems.items()}
    verification_path = os.path.join(out_dir, f"{stem}_verification.json")
    with open(verification_path, "w", encoding="utf-8") as f:
        json.dump(verification, f, ensure_ascii=False, indent=2)
    files["verification"] = verification_path
    if verification["ok"]:
        logger.info(verification["message"])
    else:
        logger.warning(verification["message"])
        progress.log(verification["message"], level="warning")
    progress.emit("verify", 94, verification["message"], verification=verification)

    # ---- 不良音符检查：过高 / 过低 / 实际音量过低 ----
    progress.emit("quality", 94.05, "检查过高 / 过低 / 音量过低的不良音符")
    bad_dir = os.path.join(out_dir, BAD_NOTES_DIRNAME)
    if os.path.isdir(bad_dir):
        for old in os.listdir(bad_dir):
            if old.lower().endswith(".mid"):
                try:
                    os.remove(os.path.join(bad_dir, old))
                except OSError:
                    pass
    bad_assessment = assess_bad_notes(analysis.notes, track_options, analysis.tracks)
    bad_files: dict[str, str] = {}
    supplement_info: dict[str, Any] = {}
    percussion_info: dict[str, Any] = {}
    if options.write_bad_note_midis and bad_assessment.get("bad_notes"):
        try:
            bad_files = write_bad_note_midis(
                analysis.notes,
                bad_assessment,
                analysis.tracks,
                out_dir,
                stem,
                tempo=float(getattr(analysis, "tempo", 120.0) or 120.0),
                tempo_changes=getattr(analysis, "tempo_changes", None),
                time_signature=str(getattr(analysis, "time_signature", "") or ""),
            )
        except Exception as exc:  # pragma: no cover - MIDI 写出失败不影响音频
            logger.warning("不良音符 MIDI 写出失败：%s", exc)
            progress.log(f"不良音符 MIDI 写出失败：{exc}", level="warning")

    # ---- 补充 MIDI：不良音轨 → 空白轨 → 全谱 ----
    # 未输出 = 没有进入主混音（未勾选 / 静音 / 时间段内无音符），或本次要求
    # 独立分轨却没有生成音频的轨道。这里只用于在补充 MIDI 里标注原因；
    # 全谱段本身始终包含所有有声轨道，``mix_only`` 也会完整保留。
    audible_set = set(audible)
    rendered_audio_tracks = {int(n.track) for n in render_notes_list}
    stem_set = {int(k) for k in stems}
    missing_for_midi: list[int] = []
    missing_reasons: dict[int, str] = {}
    for track in analysis.tracks:
        idx = int(track.index)
        if int(track.n_notes) <= 0:
            # 空轨道无法写音符，但保留在补充清单里说明为什么没有它。
            if idx not in selected or idx not in audible_set:
                missing_for_midi.append(idx)
                missing_reasons[idx] = "空轨道（没有音符）"
            continue
        reason = ""
        if idx not in selected:
            reason = "未勾选参与生成"
        elif idx not in audible_set:
            reason = "音量为 0（静音）"
        elif idx not in rendered_audio_tracks:
            reason = "所选时间段内没有可渲染音符"
        elif options.write_stems and idx not in stem_set:
            reason = "未生成独立分轨"
        if reason:
            missing_for_midi.append(idx)
            missing_reasons[idx] = reason

    if options.write_bad_note_midis and (bad_files or missing_for_midi):
        try:
            supplement_info = write_bad_and_all_midi(
                getattr(analysis, "notes_all", None) or analysis.notes,
                bad_assessment,
                analysis.tracks,
                out_dir,
                stem,
                missing_indices=missing_for_midi,
                missing_reasons=missing_reasons,
                tempo=float(getattr(analysis, "tempo", 120.0) or 120.0),
                tempo_changes=getattr(analysis, "tempo_changes", None),
                time_signature=str(getattr(analysis, "time_signature", "") or ""),
            )
        except Exception as exc:  # pragma: no cover - MIDI 写出失败不影响音频
            logger.warning("补充 MIDI 写出失败：%s", exc)
            progress.log(f"补充 MIDI 写出失败：{exc}", level="warning")
            supplement_info = {"path": "", "message": f"写出失败：{exc}"}

    # ---- 单独打击乐 MIDI ----
    # 打击乐不进入哈基米音频合成，但必须能被 DAW / 鼓采样器单独使用。
    if any(bool(getattr(t, "is_drum", False)) for t in analysis.tracks):
        try:
            percussion_info = write_percussion_midi(
                getattr(analysis, "notes_all", None) or analysis.notes,
                analysis.tracks,
                out_dir,
                stem,
                tempo=float(getattr(analysis, "tempo", 120.0) or 120.0),
                tempo_changes=getattr(analysis, "tempo_changes", None),
                time_signature=str(getattr(analysis, "time_signature", "") or ""),
            )
        except Exception as exc:  # pragma: no cover - MIDI 写出失败不影响音频
            logger.warning("打击乐 MIDI 写出失败：%s", exc)
            progress.log(f"打击乐 MIDI 写出失败：{exc}", level="warning")
            percussion_info = {"path": "", "message": f"写出失败：{exc}"}

    bad_assessment["files"] = bad_files
    bad_assessment["midi_enabled"] = bool(options.write_bad_note_midis)
    bad_assessment["supplement_midi"] = supplement_info
    bad_assessment["percussion_midi"] = percussion_info
    bad_json_path = os.path.join(out_dir, f"{stem}_bad_notes.json")
    with open(bad_json_path, "w", encoding="utf-8") as f:
        json.dump(bad_assessment, f, ensure_ascii=False, indent=2)
    files["bad_notes"] = bad_json_path
    if bad_files:
        files["bad_note_midis"] = bad_files
    if supplement_info.get("path"):
        files["supplement_midi"] = supplement_info["path"]
    if percussion_info.get("path"):
        files["percussion_midi"] = percussion_info["path"]
    if bad_assessment.get("bad_notes"):
        progress.log(
            bad_assessment["message"]
            + ("；已为每条问题轨道导出仅含不良音符的 MIDI" if bad_files else ""),
            level="warning",
        )
    else:
        progress.log(bad_assessment["message"])
    if supplement_info.get("path"):
        progress.log(
            supplement_info.get("message", "补充 MIDI 已写出"),
            level="info",
        )
    if percussion_info.get("path"):
        progress.log(
            percussion_info.get("message", "打击乐 MIDI 已写出"),
            level="info",
        )

    # ================= 7. 绿幕字幕视频（音频完成后串行生成） =================
    subtitle_video_info: dict[str, Any] = {
        "available": False,
        "files": {},
        "srt": {},
        "tracks": {},
        "error": "未开启",
    }
    if options.write_subtitle_videos:
        if video_lines:
            progress.emit(
                "video",
                94.1,
                "生成绿幕字幕视频：全部轨道合并为一条（每次 1 行）",
            )

            def video_progress(pos: int, total: int, message: str) -> None:
                pct = 94.1 + 0.8 * (pos + 0.5) / max(1, total)
                progress.emit("video", pct, message)

            try:
                subtitle_video_info = render_subtitle_video(
                    video_lines,
                    out_dir=out_dir,
                    project=stem,
                    duration=mix_duration,
                    time_offset=global_t0,
                    width=max(640, min(3840, int(options.subtitle_video_width or 1920))),
                    fps=max(12, min(60, int(options.subtitle_video_fps or 30))),
                    font_family=str(options.subtitle_video_font_family or "SimSun"),
                    font_size=float(options.subtitle_video_font_size or 8.0),
                    text_color=str(options.subtitle_video_text_color or "#FFFFFF"),
                    track_info=video_track_info,
                    coverage=video_coverage,
                    progress_cb=video_progress,
                )
            except Exception as exc:  # pragma: no cover - 视频失败不能影响音频
                logger.warning("绿幕字幕视频生成失败：%s", exc)
                subtitle_video_info = {
                    "available": False,
                    "files": {},
                    "srt": {},
                    "tracks": {},
                    "error": str(exc),
                }
            subtitle_video_info.setdefault("coverage", video_coverage)
            if subtitle_video_info.get("available"):
                files["subtitle_videos"] = subtitle_video_info.get("files") or {}
                if subtitle_video_info.get("meta_path"):
                    files["subtitle_videos_meta"] = subtitle_video_info["meta_path"]
                for key, path in (subtitle_video_info.get("files") or {}).items():
                    track_item = (subtitle_video_info.get("tracks") or {}).get(key)
                    if track_item is not None:
                        track_item["file_relative"] = os.path.relpath(path, out_dir).replace("\\", "/")
                coverage_info = subtitle_video_info.get("coverage") or {}
                progress.emit(
                    "video",
                    94.9,
                    "字幕视频完成：单条 · "
                    f"{subtitle_video_info.get('width')} 宽 × "
                    f"{subtitle_video_info.get('height')} 高 · "
                    f"每次 {subtitle_video_info.get('max_lines', 1)} 行 · "
                    f"覆盖 {coverage_info.get('covered_slots', 0)}/"
                    f"{coverage_info.get('input_slots', 0)} 槽 · "
                    f"{subtitle_video_info.get('fps')} fps",
                )
            else:
                progress.log(
                    "字幕视频跳过：" + str(subtitle_video_info.get("error") or "未知原因"),
                    level="warning",
                )
        else:
            progress.log("没有可用于字幕视频的歌词槽，跳过视频", level="warning")

    # ================= 8. 字幕 / 分析 / 打包 =================
    progress.emit("write", 95, "写入字幕与分析文件")
    lyrics_path = os.path.join(out_dir, f"{stem}_lyrics.json")
    with open(lyrics_path, "w", encoding="utf-8") as f:
        json.dump(lyrics.events, f, ensure_ascii=False, indent=2)
    files["lyrics"] = lyrics_path

    srt_path = os.path.join(out_dir, f"{stem}_subtitles.srt")
    write_srt(subtitle_lines, srt_path)
    files["srt"] = srt_path

    lrc_path = os.path.join(out_dir, f"{stem}_subtitles.lrc")
    write_lrc(subtitle_lines, lrc_path, title=stem)
    files["lrc"] = lrc_path

    meta = {
        "曲目": stem,
        "时长": f"{mix_duration:.1f} 秒",
        "调性": analysis.key.get("label_cn") or analysis.key.get("label") or "—",
        "段落 / 乐句": f"{len(analysis.sections)} 段 / {len(analysis.phrases)} 句",
        "轨道": len(selected),
        "歌词槽": lyrics.stats.get("n_slots", 0),
        "分解和弦槽": lyrics.stats.get("n_decomposed_slots", 0),
        "动机": lyrics.stats.get("n_motifs", 0),
        "语音": "、".join(f"{k}×{v}" for k, v in lyrics.stats.get("voices", {}).items()),
        "混音响度": f"{mix_info.get('loudness_db', '—')} dB RMS",
        "母带动态": (
            f"{mix_info.get('compression_ratio', '—')}:1 / "
            f"自动补偿 +{mix_info.get('makeup_db', 0)} dB / "
            f"弱段 +{mix_info.get('upward_gain_db', 0)} dB / "
            f"峰值因数 {mix_info.get('crest_db', '—')} dB"
        ),
        "自动混音": "开启" if options.auto_mix else "关闭",
        "渐强 / 渐弱": f"{rstats.n_dynamic_notes} 个音符",
        "Tremolo 奏法": f"{rstats.n_tremolo_notes} 个音符",
        "高音补偿": f"{rstats.n_audibility_notes} 个音符",
        "弱化音头 / 连奏": f"{rstats.n_attack_softened} / {rstats.n_legato_links} 个音符",
        "辅音 / 爆破音衰减": f"{rstats.n_consonant_notes} / {rstats.n_plosive_notes} 个音符",
        "不良音符": f"{bad_assessment.get('bad_notes', 0)} / {bad_assessment.get('total_notes', 0)} 个",
        "补充 MIDI": (
            f"{supplement_info.get('n_tracks', 0)} 轨 / "
            f"{supplement_info.get('n_notes', 0)} 个音符"
            if supplement_info.get("path")
            else "未生成"
        ),
        "打击乐 MIDI": (
            f"{percussion_info.get('n_tracks', 0)} 轨 / "
            f"{percussion_info.get('n_notes', 0)} 个音符"
            if percussion_info.get("path")
            else "未生成"
        ),
        "绿幕字幕视频": (
            f"单条 {subtitle_video_info.get('width')}宽×{subtitle_video_info.get('height')}高 · "
            f"{subtitle_video_info.get('font_size')}号 · "
            f"每次{max(1, int(subtitle_video_info.get('max_lines') or 1))}行 · "
            f"{subtitle_video_info.get('fps')}fps"
            if subtitle_video_info.get("available")
            else "未生成"
        ),
        "渲染耗时": f"{rstats.elapsed:.1f} 秒",
        "实时倍率": f"{rstats.realtime_factor:.1f}×",
    }

    json_path = os.path.join(out_dir, f"{stem}_subtitles.json")
    write_subtitle_json(
        subtitle_lines,
        json_path,
        meta={
            **meta,
            "sections": [s.to_dict() for s in analysis.sections],
            "tracks": [track_options[i] for i in selected],
        },
    )
    files["subtitle_json"] = json_path

    md_path = os.path.join(out_dir, f"{stem}_subtitles.md")
    write_markdown(subtitle_lines, md_path, title=stem, meta=meta)
    files["markdown"] = md_path

    html_path = os.path.join(out_dir, f"{stem}_realtime.html")
    write_html(
        subtitle_lines,
        html_path,
        title=stem,
        audio_name=os.path.basename(audio_path),
        meta={k: v for k, v in meta.items() if k not in ("渲染耗时", "实时倍率")},
    )
    files["html"] = html_path

    analysis_path = os.path.join(out_dir, f"{stem}_analysis.json")
    analysis_dict = analysis.to_dict()
    analysis_dict["render_tracks"] = [track_options[i] for i in selected]
    analysis_dict["section_plans"] = lyrics.stats.get("section_plans", [])
    analysis_dict["section_stats"] = lyrics.stats.get("section_stats", [])
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis_dict, f, ensure_ascii=False, indent=2)
    files["analysis"] = analysis_path

    track_srt_files: dict[str, str] = {}
    for idx, lines in track_subs.items():
        if not lines:
            continue
        p = os.path.join(out_dir, f"track{idx}.srt")
        write_srt(lines, p)
        track_srt_files[str(idx)] = p
    if track_srt_files:
        files["track_srt"] = track_srt_files

    # ---- 工程参数包：完整复制一次工程的全部设置 ----
    progress.emit("params", 96.2, "保存工程参数包（可复制到其他工程）")
    project_params = build_project_params(
        project=stem,
        midi_path=midi_path,
        options=options,
        track_options=track_options,
        selected=selected,
        audible=audible,
        analysis=analysis,
        outputs={
            "audio": os.path.basename(audio_path),
            "duration": round(mix_duration, 3),
            "stems": len(stems),
            "premix_zip": os.path.basename(premix_zip) if premix_zip else "",
            "bad_notes": int(bad_assessment.get("bad_notes", 0)),
            "bad_note_midis": sum(1 for k in bad_files if k != "all"),
            "supplement_midi": bool(supplement_info.get("path")),
            "percussion_midi": bool(percussion_info.get("path")),
            "subtitle_videos": bool(subtitle_video_info.get("available")),
        },
    )
    params_paths = write_project_params(project_params, out_dir, stem)
    files["params"] = params_paths["json"]
    files["params_zip"] = params_paths["zip"]

    # ---- 乐器输出清单（输出 / 未输出 / 音频文件名对应） ----
    progress.emit("report", 96, "生成乐器输出清单")
    instrument_report = build_instrument_report(
        project=stem,
        mix_filename=os.path.basename(audio_path),
        analysis=analysis,
        track_options=track_options,
        selected=selected,
        audible=audible,
        stems=stems,
        stem_entries=stem_entries,
        premix=premixs,
        premix_entries=premix_entries,
        premix_zip=premix_zip,
        bad_notes=bad_assessment,
        bad_note_files=bad_files,
        supplement_midi=supplement_info,
        percussion_midi=percussion_info,
        params_paths=params_paths,
        write_stems=bool(options.write_stems),
        output_dir=out_dir,
        mix_report=mix_info,
        video_report=subtitle_video_info,
    )
    report_paths = write_instrument_report(instrument_report, out_dir, stem)
    files["instrument_report"] = report_paths["markdown"]
    files["instrument_report_csv"] = report_paths["csv"]
    files["instrument_report_json"] = report_paths["json"]

    duration = mix_duration
    output_files = {
        int(key): os.path.basename(path) for key, path in stems.items()
    }
    premix_output = {
        int(key): os.path.basename(path) for key, path in premixs.items()
    }
    bad_by_track = {
        int(item.get("index", -1)): item
        for item in (bad_assessment.get("tracks") or [])
        if item.get("index") is not None
    }
    result_tracks = []
    for idx in selected:
        item = dict(track_options[idx])
        item["output_file"] = output_files.get(int(idx), "")
        item["premix_file"] = premix_output.get(int(idx), "")
        item["bad_notes"] = int(bad_by_track.get(int(idx), {}).get("bad_notes", 0))
        item["bad_note_midi"] = os.path.basename(bad_files.get(str(int(idx)), "")) if bad_files else ""
        item["output_status"] = (
            "stem" if int(idx) in output_files else ("mix" if int(idx) in set(audible) else "muted")
        )
        result_tracks.append(item)

    payload = {
        "project": stem,
        "midi_path": midi_path,
        "output_dir": out_dir,
        "audio_path": audio_path,
        "duration": duration,
        "render_seconds": rstats.elapsed,
        "files": files,
        "stats": {
            "render": rstats.to_dict(),
            "mix": mix_info,
            "lyrics": lyrics.stats,
            "analysis": analysis.stats(),
            "verification": {
                "ok": verification["ok"],
                "total_chords": verification["total_chords"],
                "total_chord_notes": verification["total_chord_notes"],
                "verified_chord_notes": verification["verified_chord_notes"],
                "problems": verification["problems"],
            },
            "instruments": instrument_report["summary"],
            "premix": {
                "n_tracks": len(premixs),
                "format": "flac",
                "zip": os.path.basename(premix_zip) if premix_zip else "",
            },
            "bad_notes": {
                key: bad_assessment.get(key)
                for key in (
                    "ok", "total_notes", "bad_notes", "bad_ratio", "too_high",
                    "too_low", "too_quiet", "message",
                )
            },
            "supplement_midi": {
                key: supplement_info.get(key)
                for key in (
                    "n_tracks", "n_entries", "n_bad_tracks", "n_score_tracks",
                    "n_spacer_tracks", "n_missing_tracks", "n_notes", "bad_notes",
                    "score_notes", "missing_notes", "message",
                )
            },
            "percussion_midi": {
                key: percussion_info.get(key)
                for key in ("n_tracks", "n_notes", "message")
            },
            "video": {
                "available": bool(subtitle_video_info.get("available")),
                "width": subtitle_video_info.get("width"),
                "height": subtitle_video_info.get("height"),
                "max_lines": subtitle_video_info.get("max_lines", 1),
                "single_video": True,
                "line_mode": "one_line",
                "font": subtitle_video_info.get("font"),
                "font_size": subtitle_video_info.get("font_size"),
                "font_px": subtitle_video_info.get("font_px"),
                "text_color": subtitle_video_info.get("text_color"),
                "stroke_width": subtitle_video_info.get("stroke_width"),
                "stroke_color": subtitle_video_info.get("stroke_color"),
                "coverage": subtitle_video_info.get("coverage") or video_coverage,
                "layouts": subtitle_video_info.get("layouts") or {},
                "fps": subtitle_video_info.get("fps"),
                "files": {
                    k: os.path.basename(v)
                    for k, v in (subtitle_video_info.get("files") or {}).items()
                },
                "elapsed_seconds": subtitle_video_info.get("elapsed_seconds"),
                "error": subtitle_video_info.get("error"),
            },
        },
        "analysis": analysis_dict,
        "lyrics_stats": lyrics.stats,
        "tracks": result_tracks,
        "subtitles": subtitle_payload,
        "verification": verification,
        "stems": {k: os.path.basename(v) for k, v in stems.items()},
        "instrument_report": instrument_report,
        "mix_report": mix_info,
        "subtitle_videos": subtitle_video_info,
        "premix": {k: os.path.basename(v) for k, v in premixs.items()},
        "premix_entries": premix_entries,
        "premix_zip": premix_zip,
        "bad_notes": bad_assessment,
        "supplement_midi": supplement_info,
        "percussion_midi": percussion_info,
        "project_params": project_params,
    }

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if options.write_zip:
        progress.emit("zip", 99, "打包全部输出文件")
        zip_path = os.path.join(out_dir, f"{stem}{ZIP_SUFFIX}")
        zip_paths = _flatten_files(files) + [manifest_path]
        if premix_zip:
            # 主 ZIP 里只放混音前干声的压缩包，避免同一批 FLAC 被存两份。
            premix_root = os.path.abspath(premix_dir) + os.sep
            zip_paths = [
                p for p in zip_paths
                if not os.path.abspath(p).startswith(premix_root)
            ]
        _write_zip(zip_paths, zip_path, out_dir)
        files["zip"] = zip_path
        payload["files"] = files
        # 重新写一次 manifest，让其中的文件清单包含 ZIP 本身
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    progress.emit(
        "done",
        100,
        f"完成：{duration:.1f} 秒音频，渲染 {rstats.elapsed:.1f} 秒（{rstats.realtime_factor:.1f}× 实时）",
        result=payload,
    )
    logger.info("输出目录: %s", out_dir)

    return PipelineResult(
        project=stem,
        midi_path=midi_path,
        output_dir=out_dir,
        audio_path=audio_path,
        duration=duration,
        render_seconds=rstats.elapsed,
        files=files,
        stats=payload["stats"],
        analysis=analysis_dict,
        lyrics_stats=lyrics.stats,
        tracks=payload["tracks"],
        subtitles=subtitle_payload,
        verification=verification,
        stems=stems,
        instrument_report=instrument_report,
        mix_report=mix_info,
        subtitle_videos=subtitle_video_info,
        premix={k: os.path.basename(v) for k, v in premixs.items()},
        bad_notes=bad_assessment,
        supplement_midi=supplement_info,
        percussion_midi=percussion_info,
        project_params=project_params,
    )


__all__ = [
    "PipelineResult",
    "RenderCancelled",
    "StageProgress",
    "load_lyrics_model",
    "render_midi",
    "resolve_track_options",
]
