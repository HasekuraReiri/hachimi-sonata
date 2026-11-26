"""MIDI / MusicXML 音乐分析：轨道/乐器识别、调性、和弦、结构、乐句与动机检测。

设计要点
--------
1. 使用自有的轻量 :class:`NoteEvent`（而不是 ``pretty_midi.Note``）作为内部表示，
   便于序列化、去重和按 id 引用；MusicXML 的力度 / 奏法也会带到 ``NoteEvent``。
2. **和弦分析**：先按起音聚簇（默认 60ms 容差），再在时间轴上合并成「和弦窗口」；
   合并条件是同根音/同和弦，或后一簇的音级集合是当前和弦音集的子集（用于识别
   分解和弦与琶音）。
3. **歌词槽（lyric slot）**：把每个和弦窗口内的音符按起音分组；若整个窗口符合
   「分解和弦」特征（全部为和弦音、相邻音间隔足够小），则把窗口内所有声部的音符
   合并为**同一个槽**——这保证跨声部同一个分解和弦得到相同歌词。
4. **动机检测**：在每条轨道上提取旋律序列，用「音程序列 + 节奏比例」的移调不变
   签名做哈希，找出重复出现的动机，供歌词阶段复用同一句歌词。
5. **结构分析**：按拍窗口提取 chroma + 节奏/力度特征，做自相似聚类，得到 A/B/A'
   式的段落划分，段落编号也用于自动语音模式（哈基米 / 曼波）切换。
"""

from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Sequence

import numpy as np
import pretty_midi

from hachimi.config import (
    ANALYSIS_MAX_NOTES,
    BASS_GAIN,
    BASS_MAX_PITCH,
    CHORD_MAX_WINDOW,
    CHORD_MERGE_GAP,
    CHORD_MIN_NOTES,
    CHORD_ONSET_TOL,
    DECOMPOSED_GAP,
    DEFAULT_VIOLIN_GAIN,
    MOTIF_MAX_NOTES,
    MOTIF_MAX_RESULTS,
    MOTIF_MIN_NOTES,
    MOTIF_MIN_OCCURRENCES,
    SECTION_BEATS,
    SECTION_SIMILARITY,
    SYNTAX_PHRASE_GAP,
    instrument_name,
)

logger = logging.getLogger(__name__)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

#: Krumhansl–Schmuckler 调性轮廓。音乐心理学中最常用的调性判断模板，
#: 比单纯统计音级频率更能区分「大调 / 小调」与近关系调。
KEY_PROFILES: dict[str, list[float]] = {
    "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
}

#: 自然大调 / 自然小调的级数音程
SCALE_INTERVALS: dict[str, list[int]] = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
}

#: (后缀, 音程集合)
CHORD_QUALITIES: list[tuple[str, tuple[int, ...]]] = [
    ("", (0, 4, 7)),
    ("m", (0, 3, 7)),
    ("dim", (0, 3, 6)),
    ("aug", (0, 4, 8)),
    ("sus2", (0, 2, 7)),
    ("sus4", (0, 5, 7)),
    ("5", (0, 7)),
    ("maj7", (0, 4, 7, 11)),
    ("m7", (0, 3, 7, 10)),
    ("7", (0, 4, 7, 10)),
    ("m7b5", (0, 3, 6, 10)),
    ("dim7", (0, 3, 6, 9)),
    ("mMaj7", (0, 3, 7, 11)),
    ("add9", (0, 2, 4, 7)),
    ("6", (0, 4, 7, 9)),
    ("m6", (0, 3, 7, 9)),
]


def pc_name(pc: int) -> str:
    return NOTE_NAMES[int(pc) % 12]


def note_name(pitch: int) -> str:
    return f"{pc_name(pitch)}{pitch // 12 - 1}"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class NoteEvent:
    """一个音符（与 pretty_midi 解耦）。

    MusicXML 来源的音符会额外带上乐谱标记：``velocity_end``（发夹 /
    力度变化终点）、``articulation``（奏法）、``gate``（实际发音比例）
    与 ``tremolo``。MIDI 来源保持默认值，行为与旧版完全一致。
    """

    id: int
    track: int
    index: int
    pitch: int
    start: float
    end: float
    velocity: int
    velocity_end: int | None = None
    articulation: str = ""
    gate: float = 1.0
    dynamic: str = ""
    tremolo: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def pitch_class(self) -> int:
        return self.pitch % 12

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "track": self.track,
            "pitch": self.pitch,
            "name": note_name(self.pitch),
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "velocity": self.velocity,
            "velocity_end": self.velocity_end,
            "articulation": self.articulation,
            "gate": round(self.gate, 3),
            "dynamic": self.dynamic,
            "tremolo": self.tremolo,
        }


@dataclass
class TrackAnalysis:
    index: int
    name: str
    program: int
    instrument: str
    is_drum: bool
    n_notes: int
    pitch_min: int | None
    pitch_max: int | None
    median_pitch: float | None
    avg_velocity: float
    mean_duration: float
    density: float
    polyphony: float
    active_start: float
    active_end: float
    role: str
    role_label: str
    suggested_selected: bool
    suggested_voice: str
    suggested_octave: int
    suggested_gain: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChordWindow:
    index: int
    start: float
    end: float
    root: int
    quality: str
    label: str
    tones: list[int]
    note_ids: list[int]
    track_ids: list[int]
    confidence: float
    is_decomposed: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self, compact: bool = True) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "label": self.label,
            "root": pc_name(self.root),
            "quality": self.quality,
            "confidence": round(self.confidence, 3),
            "n_notes": len(self.note_ids),
            "tracks": sorted(set(self.track_ids)),
            "decomposed": self.is_decomposed,
        }


@dataclass
class LyricSlot:
    """一个歌词槽：槽内所有音符共用同一个字。"""

    index: int
    start: float
    end: float
    note_ids: list[int]
    chord_label: str = ""
    chord_index: int = -1
    track_ids: list[int] = field(default_factory=list)
    #: 槽的“主声部”：歌词流按主声部组织，保证每条声部自己的句子连续
    primary_track: int = -1
    is_decomposed: bool = False
    section: str = ""
    section_index: int = -1
    motif_id: str = ""
    word: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Section:
    index: int
    label: str
    start: float
    end: float
    n_notes: int
    energy: float
    repeat_of: str | None = None
    # ---- 调性与和声（更专业的段落摘要） ----
    key: str = ""
    key_tonic: int = -1
    key_mode: str = ""
    key_confidence: float = 0.0
    n_chords: int = 0
    progression: list[str] = field(default_factory=list)
    chord_labels: list[str] = field(default_factory=list)
    bars: int = 0
    phrase_count: int = 0
    density: float = 0.0
    register: float = 0.0
    character: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "n_notes": self.n_notes,
            "energy": round(self.energy, 3),
            "repeat_of": self.repeat_of,
            "key": self.key,
            "key_tonic": self.key_tonic,
            "key_mode": self.key_mode,
            "key_confidence": round(self.key_confidence, 3),
            "n_chords": self.n_chords,
            "progression": self.progression[:16],
            "chord_labels": self.chord_labels[:16],
            "bars": self.bars,
            "phrase_count": self.phrase_count,
            "density": round(self.density, 2),
            "register": round(self.register, 1),
            "character": self.character,
        }


@dataclass
class Motif:
    id: str
    length: int
    intervals: list[int]
    occurrences: list[dict[str, Any]]
    score: float = 0.0
    sample_pitches: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "length": self.length,
            "intervals": self.intervals,
            "count": len(self.occurrences),
            "score": round(self.score, 2),
            "occurrences": self.occurrences[:8],
            "pitches": self.sample_pitches,
        }


#: 网页最多展示多少个速度点；密集速度自动化（每 64 分音符一个标记）
#: 会有数万个事件，必须按时间窗口聚合，否则浏览器会生成数万个 DOM 节点。
TEMPO_DISPLAY_POINTS = 60


def compress_tempo_changes(
    changes: Sequence[dict[str, Any]], max_points: int = TEMPO_DISPLAY_POINTS
) -> tuple[list[dict[str, Any]], bool, float]:
    """把速度地图压缩为最多 ``max_points`` 个显示点。

    返回 ``(rows, compressed, window_seconds)``。真实播放仍使用
    :class:`hachimi.score.TempoMap` 的完整事件，不在这里做任何近似；
    这里只影响网页 / 报告的展示密度。压缩时按等宽时间窗口取 BPM 中位数，
    避免 60/61 这类量化抖动把界面刷爆。
    """
    rows: list[dict[str, Any]] = []
    for item in changes:
        try:
            t = float(item.get("time", 0.0))
            bpm = float(item.get("bpm", 0.0))
            pos = float(item.get("position", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(t) or not math.isfinite(bpm) or bpm <= 0:
            continue
        rows.append(
            {
                "time": round(t, 4),
                "position": round(pos, 4),
                "bpm": round(bpm, 4),
                "label": str(item.get("label", "") or ""),
            }
        )
    rows.sort(key=lambda r: (r["time"], r["position"]))
    if len(rows) <= max_points or max_points <= 0:
        return rows, False, 0.0

    t0, t1 = rows[0]["time"], rows[-1]["time"]
    if t1 - t0 < 1e-9:
        return rows[:max_points], True, 0.0

    width = (t1 - t0) / max_points
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(max_points)]
    for row in rows:
        idx = int((row["time"] - t0) / width)
        idx = max(0, min(max_points - 1, idx))
        buckets[idx].append(row)

    out: list[dict[str, Any]] = []
    for bucket in buckets:
        if not bucket:
            continue
        bpms = sorted(float(r["bpm"]) for r in bucket)
        mid = len(bpms) // 2
        if len(bpms) % 2:
            median = bpms[mid]
        else:
            median = (bpms[mid - 1] + bpms[mid]) / 2.0
        item = dict(bucket[0])
        item["bpm"] = round(median, 3)
        item["label"] = ""
        item["approx"] = True
        item["samples"] = len(bucket)
        out.append(item)
    if out:
        out[0]["time"] = round(t0, 4)
        out[-1]["time"] = round(t1, 4)
    return out, True, width


@dataclass
class AnalysisResult:
    midi_path: str
    title: str
    duration: float
    tempo: float | None
    tracks: list[TrackAnalysis]
    chords: list[ChordWindow]
    sections: list[Section]
    motifs: list[Motif]
    notes: list[NoteEvent]
    warnings: list[str] = field(default_factory=list)
    #: 全曲调性（Krumhansl 轮廓）与段落内乐句
    key: dict[str, Any] = field(default_factory=dict)
    phrases: list[dict[str, Any]] = field(default_factory=list)
    time_signature: str = "4/4"
    #: midi / musicxml / mxl
    source_format: str = "midi"
    #: 完整速度地图（秒 / 位置 / BPM / 文字标记）
    tempo_changes: list[dict[str, Any]] = field(default_factory=list)
    #: 乐谱来源的额外统计（力度、奏法、连音线、拍号变化等）
    score_meta: dict[str, Any] = field(default_factory=dict)
    #: 全部轨道的音符（``notes`` 可能只含用户勾选的轨道；用于补充 MIDI）
    notes_all: list[NoteEvent] = field(default_factory=list)

    # ---- 摘要 ----
    def selected_track_indices(self) -> list[int]:
        return [t.index for t in self.tracks if t.suggested_selected]

    def stats(self) -> dict[str, Any]:
        return {
            "duration": round(self.duration, 2),
            "tempo": round(self.tempo, 1) if self.tempo else None,
            "n_tracks": len(self.tracks),
            "n_notes": len(self.notes),
            "n_chords": len(self.chords),
            "n_sections": len(self.sections),
            "n_motifs": len(self.motifs),
            "n_phrases": len(self.phrases),
            "decomposed_chords": sum(1 for c in self.chords if c.is_decomposed),
            "key": self.key.get("label", ""),
            "key_confidence": self.key.get("confidence", 0.0),
            "time_signature": self.time_signature,
            "source_format": self.source_format,
            "n_tempo_changes": len(self.tempo_changes),
            "n_dynamics": int(self.score_meta.get("n_dynamics", 0) or 0),
            "n_articulations": int(self.score_meta.get("n_articulations", 0) or 0),
            "n_ties": int(self.score_meta.get("n_ties", 0) or 0),
        }

    def tempo_display(
        self, max_points: int = TEMPO_DISPLAY_POINTS
    ) -> tuple[list[dict[str, Any]], bool, float]:
        return compress_tempo_changes(self.tempo_changes, max_points)

    def to_dict(
        self, max_chords: int = 600, max_tempo_changes: int = TEMPO_DISPLAY_POINTS
    ) -> dict[str, Any]:
        """转为网页 / JSON 友好的字典。

        密集速度自动化只在展示层聚合；完整事件仍保存在
        ``self.tempo_changes`` 与 ``TempoMap`` 中用于精确计时。
        """
        tempo_rows, compressed, window = self.tempo_display(max_tempo_changes)
        bpms = [float(c.get("bpm", 0) or 0) for c in self.tempo_changes if float(c.get("bpm", 0) or 0) > 0]
        score_meta = dict(self.score_meta)
        # 这两个字段是完整的数万条原始数据，网页 / JSON 不需要重复携带。
        score_meta.pop("tempo_map", None)
        score_meta.pop("tempo_changes", None)
        score_meta["n_tempo_changes"] = len(self.tempo_changes)
        score_meta["tempo_changes_displayed"] = len(tempo_rows)
        score_meta["tempo_changes_compressed"] = compressed
        return {
            "midi_path": self.midi_path,
            "title": self.title,
            "duration": round(self.duration, 3),
            "tempo": round(self.tempo, 2) if self.tempo else None,
            "time_signature": self.time_signature,
            "source_format": self.source_format,
            "tempo_changes": tempo_rows,
            "tempo_changes_total": len(self.tempo_changes),
            "tempo_changes_compressed": compressed,
            "tempo_changes_window": round(window, 2),
            "tempo_bpm_min": round(min(bpms), 2) if bpms else None,
            "tempo_bpm_max": round(max(bpms), 2) if bpms else None,
            "score_meta": score_meta,
            "key": self.key,
            "stats": self.stats(),
            "tracks": [t.to_dict() for t in self.tracks],
            "chords": [c.to_dict() for c in self.chords[:max_chords]],
            "sections": [s.to_dict() for s in self.sections],
            "phrases": self.phrases,
            "motifs": [m.to_dict() for m in self.motifs],
            "warnings": self.warnings,
            "default_selected": self.selected_track_indices(),
        }

    def note_map(self) -> dict[int, NoteEvent]:
        return {n.id: n for n in self.notes}


# ---------------------------------------------------------------------------
# MIDI 读取
# ---------------------------------------------------------------------------

ROLE_LABELS = {
    "melody": "旋律",
    "harmony": "和声",
    "bass": "低音声部",
    "pad": "铺底",
    "percussion": "打击乐",
    "mixed": "复合",
}


def load_midi(path: str) -> tuple[pretty_midi.PrettyMIDI, list[str]]:
    """读取 MIDI / MusicXML，返回 (统一对象, 警告列表)。

    ``.xml`` / ``.musicxml`` / ``.mxl`` 会先由 :mod:`hachimi.score` 转成
    ``PrettyMIDI``；音符仍然是秒级时间轴，因此后续分析、歌词与渲染完全
    复用 MIDI 流程。
    """
    import warnings as _warnings

    warnings: list[str] = []
    ext = os.path.splitext(path)[1].lower()
    is_score = ext in (".xml", ".musicxml", ".mxl")
    if not is_score:
        # 有些 .mxl 会被改名为 .xml 或没有扩展名，用文件头兜底
        try:
            with open(path, "rb") as f:
                is_score = f.read(2) == b"PK"
        except OSError:
            pass

    if is_score:
        from hachimi.score import read_score

        try:
            pm = read_score(path)
        except Exception as exc:
            raise ValueError(f"无法解析 MusicXML 乐谱: {exc}") from exc
        warnings.extend(pm.score_meta.get("warnings") or [])
        if not pm.instruments:
            warnings.append("该乐谱没有可用乐器声部。")
        return pm, warnings

    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            pm = pretty_midi.PrettyMIDI(path)
    except Exception as exc:  # pragma: no cover - 依赖第三方解析
        raise ValueError(f"无法解析 MIDI 文件: {exc}") from exc

    if not pm.instruments:
        warnings.append("该 MIDI 没有可用乐器轨道（可能是格式 2 或纯控制文件）。")
    return pm, warnings


def collect_notes(
    pm: pretty_midi.PrettyMIDI,
    selected_tracks: Iterable[int] | None = None,
    max_notes: int = ANALYSIS_MAX_NOTES,
) -> list[NoteEvent]:
    """把 pretty_midi 音符转换为 :class:`NoteEvent` 列表。"""
    selected = set(selected_tracks) if selected_tracks is not None else None
    out: list[NoteEvent] = []
    nid = 0
    for t_idx, inst in enumerate(pm.instruments):
        if selected is not None and t_idx not in selected:
            continue
        for n_idx, note in enumerate(inst.notes):
            start = max(0.0, float(note.start))
            end = float(note.end)
            if end - start < 1e-3:
                end = start + 0.05
            velocity_end = getattr(note, "velocity_end", None)
            out.append(
                NoteEvent(
                    id=nid,
                    track=t_idx,
                    index=n_idx,
                    pitch=int(note.pitch),
                    start=start,
                    end=end,
                    velocity=max(1, min(127, int(note.velocity or 64))),
                    velocity_end=(
                        max(1, min(127, int(velocity_end)))
                        if velocity_end is not None
                        else None
                    ),
                    articulation=str(getattr(note, "articulation", "") or ""),
                    gate=float(getattr(note, "gate", 1.0) or 1.0),
                    dynamic=str(getattr(note, "dynamic", "") or ""),
                    tremolo=bool(getattr(note, "tremolo", False)),
                )
            )
            nid += 1
    out.sort(key=lambda n: (n.start, n.pitch, n.track))
    if len(out) > max_notes:
        logger.warning("音符过多（%d），分析阶段截断到 %d 个", len(out), max_notes)
        out = out[:max_notes]
    return out


# ---------------------------------------------------------------------------
# 轨道 / 乐器分析
# ---------------------------------------------------------------------------


def analyze_tracks(pm: pretty_midi.PrettyMIDI, notes: list[NoteEvent]) -> list[TrackAnalysis]:
    by_track: dict[int, list[NoteEvent]] = defaultdict(list)
    for n in notes:
        by_track[n.track].append(n)

    all_notes = notes
    global_median = float(np.median([n.pitch for n in all_notes])) if all_notes else 60.0

    tracks: list[TrackAnalysis] = []
    for idx, inst in enumerate(pm.instruments):
        tnotes = by_track.get(idx, [])
        name = inst.name.strip() if inst.name else ""
        instr = instrument_name(inst.program, inst.is_drum)
        if not name:
            name = f"{instr} #{idx}"

        if not tnotes:
            tracks.append(
                TrackAnalysis(
                    index=idx,
                    name=name,
                    program=int(inst.program),
                    instrument=instr,
                    is_drum=bool(inst.is_drum),
                    n_notes=0,
                    pitch_min=None,
                    pitch_max=None,
                    median_pitch=None,
                    avg_velocity=0.0,
                    mean_duration=0.0,
                    density=0.0,
                    polyphony=0.0,
                    active_start=0.0,
                    active_end=0.0,
                    role="percussion" if inst.is_drum else "mixed",
                    role_label=ROLE_LABELS["percussion" if inst.is_drum else "mixed"],
                    suggested_selected=False,
                    suggested_voice="auto",
                    suggested_octave=0,
                    suggested_gain=0.0,
                    reason="空轨道",
                )
            )
            continue

        pitches = np.array([n.pitch for n in tnotes], dtype=np.float32)
        vels = np.array([n.velocity for n in tnotes], dtype=np.float32)
        durs = np.array([n.duration for n in tnotes], dtype=np.float32)
        start = float(min(n.start for n in tnotes))
        end = float(max(n.end for n in tnotes))
        span = max(0.25, end - start)

        # 同时发声数（复调度）
        events = sorted([(n.start, 1) for n in tnotes] + [(n.end, -1) for n in tnotes])
        active = 0
        poly_sum = 0.0
        poly_time = 0.0
        prev_t = events[0][0]
        for t, delta in events:
            dt = t - prev_t
            if dt > 0:
                poly_sum += active * dt
                poly_time += dt
            active += delta
            prev_t = t
        polyphony = poly_sum / poly_time if poly_time > 0 else 0.0
        density = len(tnotes) / span

        median_pitch = float(np.median(pitches))
        highest_pitch = int(pitches.max())
        # 最高音过低 → 低音声部：默认不参与，音量也降低
        low_bass = (not inst.is_drum) and highest_pitch < BASS_MAX_PITCH
        if inst.is_drum:
            role = "percussion"
        elif low_bass:
            role = "bass"
        elif median_pitch < 48 and polyphony < 1.6:
            role = "bass"
        elif polyphony >= 1.6 and density > 2.0:
            role = "harmony"
        elif polyphony < 1.3 and (pitches.max() - pitches.min()) > 7 and density < 4.0 and median_pitch >= 60:
            role = "melody"
        elif density > 6.0:
            role = "harmony"
        else:
            role = "pad" if polyphony >= 1.4 else "mixed"

        # 推荐选择 / 语音 / 八度 / 音量
        selected = not inst.is_drum
        if role == "percussion":
            gain = 0.0
            voice = "auto"
            octave = 0
            reason = "打击乐没有音高，默认静音（避免噪声）"
        else:
            if role == "melody":
                voice, gain, reason = "哈基米", 1.0, "单声部高音区，作为主旋律"
            elif role == "bass":
                voice, gain = "auto", BASS_GAIN
                octave = 1
                if low_bass:
                    reason = (
                        f"最高音 {highest_pitch} < {BASS_MAX_PITCH}，判定为低音声部；"
                        f"默认不勾选、音量降为 {BASS_GAIN:.2f}，可手动勾选并调整"
                    )
                    selected = False
                else:
                    reason = (
                        f"中位音高偏低，判定为低音声部；音量降为 {BASS_GAIN:.2f}"
                    )
            elif role == "harmony":
                voice, gain, reason = "auto", 0.55, "密集和声，降低音量避免糊成一团"
            else:
                voice, gain, reason = "auto", 0.7, "中音区铺底/对位"
            if role != "bass":
                if median_pitch < 50:
                    octave = 1
                    reason += "；整体偏低，升八度更接近人声"
                elif median_pitch > 76:
                    octave = -1
                    reason += "；整体偏高，降八度更自然"
                else:
                    octave = 0
            if len(tnotes) < 4:
                selected = False
                reason += "；仅 %d 个音符，默认不参与" % len(tnotes)

        # 小提琴（GM #41 / program 40）默认音量 1.5：弦乐持续音在哈基米
        # 采样混音中容易被淹没，因此给一个更高的默认增益；用户仍可手动改。
        if (
            not inst.is_drum
            and (int(inst.program) == 40 or "violin" in instr.lower() or "fiddle" in instr.lower())
        ):
            gain = DEFAULT_VIOLIN_GAIN
            reason += f"；小提琴默认音量 {DEFAULT_VIOLIN_GAIN:.2f}"

        tracks.append(
            TrackAnalysis(
                index=idx,
                name=name,
                program=int(inst.program),
                instrument=instr,
                is_drum=bool(inst.is_drum),
                n_notes=len(tnotes),
                pitch_min=int(pitches.min()),
                pitch_max=int(pitches.max()),
                median_pitch=round(median_pitch, 1),
                avg_velocity=round(float(vels.mean()), 1),
                mean_duration=round(float(durs.mean()), 3),
                density=round(density, 2),
                polyphony=round(polyphony, 2),
                active_start=round(start, 3),
                active_end=round(end, 3),
                role=role,
                role_label=ROLE_LABELS.get(role, role),
                suggested_selected=selected,
                suggested_voice=voice,
                suggested_octave=octave,
                suggested_gain=gain,
                reason=reason,
            )
        )
    # 兜底：如果没有任何轨道被判为旋律，把中位音高最高的一条作为主旋律
    # （但不把已判定为“低音声部”的轨道强行重新勾选）
    playable = [t for t in tracks if not t.is_drum and t.n_notes >= 4]
    if playable and not any(t.role == "melody" for t in playable):
        high_enough = [
            t for t in playable
            if (t.median_pitch or 0) >= BASS_MAX_PITCH
            or (t.pitch_max or 0) >= BASS_MAX_PITCH
        ]
        if high_enough:
            lead = max(high_enough, key=lambda t: (t.median_pitch or 0, t.n_notes))
            lead.role = "melody"
            lead.role_label = ROLE_LABELS["melody"]
            lead.suggested_voice = "哈基米"
            lead.suggested_selected = True
            lead.reason = "主旋律（兜底选择：中位音高最高）"
    return tracks


# ---------------------------------------------------------------------------
# 和弦分析
# ---------------------------------------------------------------------------


def _template_mask(intervals: Sequence[int], root: int) -> int:
    mask = 0
    for iv in intervals:
        mask |= 1 << ((root + iv) % 12)
    return mask


def _popcount(x: int) -> int:
    return bin(x).count("1")


def detect_chord(
    pitches: Sequence[int],
    weights: Sequence[float] | None = None,
) -> tuple[int, str, str, list[int], float]:
    """从一组音高检测和弦。

    Returns:
        (root_pc, quality, label, chord_pcs, confidence)
    """
    if not pitches:
        return 0, "none", "—", [], 0.0

    pcs = sorted({int(p) % 12 for p in pitches})
    bass = min(pitches) % 12

    if len(pcs) == 1:
        return pcs[0], "single", pc_name(pcs[0]), pcs, 1.0
    if len(pcs) == 2:
        interval = (pcs[1] - pcs[0]) % 12
        return pcs[0], "interval", f"{pc_name(pcs[0])}+{pc_name(pcs[1])}", pcs, 0.8

    pc_mask = 0
    for pc in pcs:
        pc_mask |= 1 << pc

    best = (0, "", "", pcs, -1e9)
    for root in range(12):
        for suffix, intervals in CHORD_QUALITIES:
            tmask = _template_mask(intervals, root)
            matched = _popcount(tmask & pc_mask)
            extra = _popcount(pc_mask & ~tmask)
            missing = _popcount(tmask & ~pc_mask)
            score = matched * 1.0 - extra * 0.85 - missing * 1.15
            if root == bass:
                score += 0.45
            if matched == len(intervals) and extra == 0:
                score += 0.5
            # 偏好更少音的和弦，避免把三和弦解释成七和弦
            score -= 0.02 * len(intervals)
            if score > best[4]:
                label = f"{pc_name(root)}{suffix}"
                best = (root, suffix, label, sorted((root + iv) % 12 for iv in intervals), score)

    root, quality, label, tones, score = best
    # 归一化置信度到 0~1
    max_possible = 1.6 + 0.45 + 0.5
    confidence = max(0.0, min(1.0, (score + 1.2) / max_possible))
    return root, quality, label, tones, confidence


def _weighted_chord_pitches(active: Sequence[NoteEvent]) -> list[int]:
    """按「时长 × 力度」加权挑选和弦音，抑制快速经过音对和弦判断的干扰。"""
    weights: dict[int, float] = defaultdict(float)
    for n in active:
        weights[n.pitch_class] += min(n.duration, 2.0) * (0.5 + n.velocity / 127.0)
    if not weights:
        return []
    max_w = max(weights.values())
    selected = {pc for pc, w in weights.items() if w >= 0.35 * max_w}
    reps: list[int] = []
    for pc in sorted(selected):
        reps.append(min(n.pitch for n in active if n.pitch_class == pc))
    return reps


def detect_chords(
    notes: list[NoteEvent],
    onset_tol: float = CHORD_ONSET_TOL,
    merge_gap: float = CHORD_MERGE_GAP,
    max_window: float = CHORD_MAX_WINDOW,
    min_notes: int = CHORD_MIN_NOTES,
) -> list[ChordWindow]:
    """把音符流切分为「和弦窗口」序列。"""
    if not notes:
        return []

    ordered = sorted(notes, key=lambda n: (n.start, n.pitch))
    by_id = {n.id: n for n in ordered}

    # 1) 按起音聚簇
    clusters: list[list[NoteEvent]] = []
    current: list[NoteEvent] = [ordered[0]]
    cluster_start = ordered[0].start
    for note in ordered[1:]:
        if note.start - cluster_start <= onset_tol:
            current.append(note)
        else:
            clusters.append(current)
            current = [note]
            cluster_start = note.start
    clusters.append(current)

    # 2) 每个簇计算「当前发声音级集合」（含持续音）
    #    用滑动指针维护活动音符，避免 O(簇数 × 音符数) 的朴素扫描。
    cluster_info: list[dict[str, Any]] = []
    active: list[NoteEvent] = []
    ptr = 0
    for cl in clusters:
        t = cl[0].start
        while ptr < len(ordered) and ordered[ptr].start <= t + 1e-6:
            active.append(ordered[ptr])
            ptr += 1
        active = [n for n in active if n.end > t + 1e-6]
        if not active:
            active = list(cl)
        pitches = _weighted_chord_pitches(active) or [n.pitch for n in active]
        root, quality, label, tones, conf = detect_chord(pitches)
        cluster_info.append(
            {
                "start": min(n.start for n in cl),
                "end": max(n.end for n in cl),
                "notes": cl,
                "root": root,
                "quality": quality,
                "label": label,
                "tones": tones,
                "confidence": conf,
            }
        )

    # 3) 合并为和弦窗口
    windows: list[dict[str, Any]] = []
    for info in cluster_info:
        merged = False
        if windows:
            w = windows[-1]
            ws, we = w["start"], w["end"]
            cs = info["start"]
            gap = cs - we
            tone_set = set(w["tones"]) or set()
            new_set = set(info["tones"]) or set()
            same_chord = (w["root"], w["quality"]) == (info["root"], info["quality"])
            subset = new_set and tone_set and (new_set <= tone_set or tone_set <= new_set)
            if (
                gap <= merge_gap
                and cs - ws <= max_window
                and (same_chord or subset)
                and info["confidence"] > 0.12
                and not (len(new_set | tone_set) > 6)
            ):
                w["notes"].extend(info["notes"])
                w["end"] = max(we, info["end"])
                # 用更大的和弦（更多和弦音）作为窗口标签
                if subset and len(new_set) > len(tone_set):
                    w.update(root=info["root"], quality=info["quality"], label=info["label"], tones=info["tones"])
                    w["confidence"] = info["confidence"]
                elif same_chord:
                    w["confidence"] = (w["confidence"] + info["confidence"]) / 2
                merged = True
        if not merged:
            windows.append(dict(info))

    # 4) 生成 ChordWindow
    out: list[ChordWindow] = []
    for idx, w in enumerate(windows):
        wnotes = w["notes"]
        # 去重
        seen: set[int] = set()
        ids: list[int] = []
        for n in wnotes:
            if n.id not in seen:
                seen.add(n.id)
                ids.append(n.id)
        tracks = sorted({by_id[i].track for i in ids})
        out.append(
            ChordWindow(
                index=idx,
                start=float(w["start"]),
                end=float(w["end"]),
                root=int(w["root"]),
                quality=str(w["quality"]),
                label=str(w["label"]),
                tones=[int(t) for t in w["tones"]],
                note_ids=ids,
                track_ids=tracks,
                confidence=float(min(1.0, max(0.0, w["confidence"]))),
            )
        )
    return out


# ---------------------------------------------------------------------------
# 歌词槽（跨声部一致）
# ---------------------------------------------------------------------------


def build_lyric_slots(
    notes: list[NoteEvent],
    chords: list[ChordWindow],
    onset_tol: float = CHORD_ONSET_TOL,
    decomposed_merge: bool = True,
    decomposed_gap: float = DECOMPOSED_GAP,
) -> list[LyricSlot]:
    """把和弦窗口转化为歌词槽。

    规则：

    - 窗口内按**轨道**分别做起音聚簇（同轨同时发声的音属于同一组）；
    - 把各声部的「第 i 个和弦音组」对齐：若不同声部在同一时间位置奏出和弦音
      （间隔 ≤ ``decomposed_gap``），则合并为**同一个歌词槽**——跨声部的
      分解和弦 / 柱式和弦因此共用同一个字；
    - 同一声部内部的连续音（旋律、单声部琶音）各自独立成槽，
      不会因为共享和弦而把整条旋律压成一个字；
    - 含非和弦音的组（旋律经过音等）独立成槽。
    """
    by_id = {n.id: n for n in notes}
    slots: list[LyricSlot] = []

    def add_slot(group: list[NoteEvent], chord: ChordWindow, decomposed: bool = False) -> None:
        slots.append(
            LyricSlot(
                index=len(slots),
                start=min(n.start for n in group),
                end=max(n.end for n in group),
                note_ids=[n.id for n in group],
                chord_label=chord.label,
                chord_index=chord.index,
                track_ids=sorted({n.track for n in group}),
                is_decomposed=decomposed,
            )
        )

    for chord in chords:
        wnotes = sorted(
            (by_id[i] for i in chord.note_ids if i in by_id),
            key=lambda n: (n.start, n.pitch),
        )
        if not wnotes:
            continue

        # 按轨道分簇
        per_track: dict[int, list[NoteEvent]] = defaultdict(list)
        for note in wnotes:
            per_track[note.track].append(note)

        groups: list[list[NoteEvent]] = []
        for track in sorted(per_track):
            tnotes = sorted(per_track[track], key=lambda n: (n.start, n.pitch))
            cur: list[NoteEvent] = [tnotes[0]]
            gstart = tnotes[0].start
            for note in tnotes[1:]:
                if note.start - gstart <= onset_tol:
                    cur.append(note)
                else:
                    groups.append(cur)
                    cur = [note]
                    gstart = note.start
            groups.append(cur)

        tone_set = (
            set(chord.tones)
            if chord.quality not in ("single", "interval", "none")
            else set()
        )

        def is_chord_group(group: list[NoteEvent]) -> bool:
            if not tone_set:
                return True
            return all(n.pitch_class in tone_set for n in group)

        chord_groups = [g for g in groups if is_chord_group(g)]
        other_groups = [g for g in groups if not is_chord_group(g)]

        # 按「每个声部的第 i 个和弦音组」对齐：
        # 同一时间位置、不同声部的和弦音共用同一个歌词槽；
        # 同一声部内部的连续音（旋律 / 琶音）各自独立，避免旋律被压成一个字。
        by_track: dict[int, list[list[NoteEvent]]] = defaultdict(list)
        for group in chord_groups:
            by_track[group[0].track].append(group)
        for lst in by_track.values():
            lst.sort(key=lambda g: (g[0].start, g[0].pitch))
        n_positions = max((len(lst) for lst in by_track.values()), default=0)

        for i in range(n_positions):
            candidates = [lst[i] for lst in by_track.values() if i < len(lst)]
            candidates.sort(key=lambda g: (g[0].start, g[0].pitch))
            if (
                decomposed_merge
                and len(candidates) >= 2
                and candidates[-1][0].start - candidates[0][0].start <= decomposed_gap
            ):
                add_slot([n for g in candidates for n in g], chord, decomposed=True)
                chord.is_decomposed = True
            else:
                for group in candidates:
                    add_slot(group, chord)

        for group in other_groups:
            add_slot(group, chord)

    slots.sort(key=lambda s: (s.start, s.end))
    for i, s in enumerate(slots):
        s.index = i
    return slots


# ---------------------------------------------------------------------------
# 调性 / 级数 / 乐句
# ---------------------------------------------------------------------------


def detect_time_signature(pm: pretty_midi.PrettyMIDI) -> str:
    """读取第一个拍号；缺省按 4/4。"""
    meta = getattr(pm, "score_meta", {}) or {}
    if meta.get("time_signature"):
        return str(meta["time_signature"])
    try:
        changes = pm.time_signature_changes
        if changes:
            first = changes[0]
            num, den = int(first.numerator), int(first.denominator)
            if 1 <= num <= 16 and den in (1, 2, 4, 8, 16):
                return f"{num}/{den}"
    except Exception:  # pragma: no cover - 第三方解析差异
        pass
    return "4/4"


def _key_correlations(notes: Sequence[NoteEvent]) -> tuple[np.ndarray, np.ndarray]:
    """返回音级权重向量与每个调（12 大调 + 12 小调）的相关系数。"""
    chroma = np.zeros(12, dtype=np.float64)
    for n in notes:
        weight = min(max(n.duration, 0.03), 4.0) * (0.35 + n.velocity / 127.0)
        if n.pitch < 55:
            weight *= 1.12
        chroma[n.pitch % 12] += weight
    if chroma.sum() <= 1e-9:
        return chroma, np.zeros(24, dtype=np.float64)
    chroma = chroma / chroma.sum()

    scores = np.zeros(24, dtype=np.float64)
    for mode_index, mode in enumerate(("major", "minor")):
        profile = np.asarray(KEY_PROFILES[mode], dtype=np.float64)
        profile = (profile - profile.mean()) / (profile.std() + 1e-9)
        for tonic in range(12):
            rotated = np.roll(profile, tonic)
            x = chroma - chroma.mean()
            denom = float(np.linalg.norm(x) * np.linalg.norm(rotated))
            scores[mode_index * 12 + tonic] = (
                float(np.dot(x, rotated) / denom) if denom > 1e-9 else 0.0
            )
    return chroma, scores


def detect_key(
    notes: Sequence[NoteEvent],
    *,
    prior: dict[str, Any] | None = None,
    start: float | None = None,
    end: float | None = None,
) -> dict[str, Any]:
    """用 Krumhansl–Schmuckler 调性轮廓判断大调 / 小调。

    Args:
        notes: 参与分析的音符。
        prior: 全曲调性；用于给段落调性加一个很轻的先验，避免短段落乱跳。
        start/end: 只统计该时间段内的音符。

    Returns:
        ``{"tonic": 0..11, "mode": "major"/"minor", "label": "C major",
        "label_cn": "C 大调", "confidence": 0..1, "alternatives": [...]}``
    """
    subset = [
        n
        for n in notes
        if (start is None or n.start >= start - 1e-6)
        and (end is None or n.start < end - 1e-6)
    ]
    chroma, scores = _key_correlations(subset)
    if scores.size == 0 or not np.any(scores):
        fallback = dict(prior or {})
        fallback.setdefault("tonic", 0)
        fallback.setdefault("mode", "major")
        fallback.setdefault("label", pc_name(fallback["tonic"]) + " " + fallback["mode"])
        fallback.setdefault("label_cn", pc_name(fallback["tonic"]) + (" 大调" if fallback["mode"] == "major" else " 小调"))
        fallback.setdefault("confidence", 0.0)
        fallback.setdefault("alternatives", [])
        return fallback

    if prior:
        try:
            prior_index = int(prior.get("tonic", 0)) % 12
            prior_mode = str(prior.get("mode", "major"))
            idx = (0 if prior_mode == "major" else 12) + prior_index
            scores[idx] += 0.035
            # 关系大小调也加一点先验
            relative = (prior_index + (9 if prior_mode == "major" else 3)) % 12
            rel_idx = (12 if prior_mode == "major" else 0) + relative
            scores[rel_idx] += 0.012
        except Exception:
            pass

    order = np.argsort(scores)[::-1]
    best = int(order[0])
    second = int(order[1]) if len(order) > 1 else best
    mode = "major" if best < 12 else "minor"
    tonic = best % 12
    confidence = max(0.0, min(1.0, (float(scores[best]) - float(scores[second])) / 0.18))
    label = f"{pc_name(tonic)} {mode}"
    major_mode = mode == "major"

    alternatives = []
    for idx in order[1:4]:
        idx = int(idx)
        alt_mode = "major" if idx < 12 else "minor"
        alternatives.append(
            {
                "label": f"{pc_name(idx % 12)} {alt_mode}",
                "score": round(float(scores[idx]), 4),
            }
        )

    return {
        "tonic": tonic,
        "mode": mode,
        "label": label,
        "label_cn": f"{pc_name(tonic)} {'大调' if major_mode else '小调'}",
        "confidence": round(confidence, 3),
        "alternatives": alternatives,
        "chroma": [round(float(v), 4) for v in chroma],
        "profile": "Krumhansl-Schmuckler",
    }


_ROMAN_UPPER = ["I", "II", "III", "IV", "V", "VI", "VII"]


def roman_numeral(chord: ChordWindow | Mapping[str, Any], tonic: int, mode: str = "major") -> str:
    """把和弦转成相对调性的罗马数字级数（用于段落和声摘要）。"""
    if isinstance(chord, ChordWindow):
        root, quality = chord.root, chord.quality
    else:
        root = int(chord.get("root", 0))
        quality = str(chord.get("quality", ""))
    if quality in ("none", "single", "interval", ""):
        if quality:
            return ""
    rel = (int(root) - int(tonic)) % 12
    scale = SCALE_INTERVALS.get(mode if mode in SCALE_INTERVALS else "major", SCALE_INTERVALS["major"])
    degree = -1
    accidental = ""
    # 先找精确音级，避免把 F（IV）误判成 #III；找不到再接受半音邻级。
    for idx, interval in enumerate(scale):
        if rel == interval:
            degree = idx
            accidental = ""
            break
    if degree < 0:
        for idx, interval in enumerate(scale):
            diff = ((rel - interval + 6) % 12) - 6
            if abs(diff) <= 1:
                degree = idx
                accidental = "b" if diff < 0 else ("#" if diff > 0 else "")
                break
    if degree < 0:
        return ""
    numeral = _ROMAN_UPPER[degree]
    if quality in ("m", "m6", "m7", "mMaj7"):
        numeral = numeral.lower()
    if quality == "dim":
        numeral = numeral.lower() + "°"
    elif quality == "dim7":
        numeral = numeral.lower() + "°7"
    elif quality == "m7b5":
        numeral = numeral.lower() + "ø7"
    elif quality == "aug":
        numeral += "+"
    elif quality == "maj7":
        numeral += "maj7"
    elif quality in ("7", "m7"):
        numeral += "7"
    elif quality in ("sus2", "sus4"):
        numeral += quality
    elif quality in ("add9", "6"):
        numeral += quality
    return f"{accidental}{numeral}"


def _section_chords(
    chords: Sequence[ChordWindow],
    start: float,
    end: float,
) -> list[ChordWindow]:
    return [c for c in chords if start - 1e-6 <= c.start < end - 1e-6]


def _annotate_sections(
    sections: list[Section],
    notes: Sequence[NoteEvent],
    chords: Sequence[ChordWindow],
    key_info: dict[str, Any],
    bpm: float,
    time_signature: str,
) -> None:
    """把调性、和弦级数、小节数、织体特征写入每个段落。"""
    try:
        numerator = int(str(time_signature).split("/", 1)[0])
    except Exception:
        numerator = 4
    numerator = max(1, numerator)
    seconds_per_bar = (60.0 / bpm) * numerator if bpm > 0 else 2.0

    for sec in sections:
        sec_notes = [n for n in notes if sec.start - 1e-6 <= n.start < sec.end - 1e-6]
        info = detect_key(sec_notes, prior=key_info, start=sec.start, end=sec.end)
        sec.key = str(info.get("label", ""))
        sec.key_tonic = int(info.get("tonic", -1))
        sec.key_mode = str(info.get("mode", ""))
        sec.key_confidence = float(info.get("confidence", 0.0))

        sec_chords = _section_chords(chords, sec.start, sec.end)
        sec.n_chords = len(sec_chords)
        sec.chord_labels = [c.label for c in sec_chords[:16]]
        progression: list[str] = []
        for c in sec_chords:
            numeral = roman_numeral(c, sec.key_tonic if sec.key_tonic >= 0 else key_info.get("tonic", 0), sec.key_mode or key_info.get("mode", "major"))
            if numeral and (not progression or progression[-1] != numeral):
                progression.append(numeral)
        sec.progression = progression[:16]

        duration = max(1e-6, sec.end - sec.start)
        sec.bars = max(1, int(round(duration / max(0.2, seconds_per_bar))))
        sec.density = round(len(sec_notes) / duration, 2)
        sec.register = round(float(np.median([n.pitch for n in sec_notes])), 1) if sec_notes else 0.0
        energy_word = "强奏" if sec.energy >= 0.62 else ("中强" if sec.energy >= 0.34 else "轻奏")
        density_word = "密集" if sec.density >= 4.0 else ("流动" if sec.density >= 1.5 else "留白")
        mode_word = "大调" if (sec.key_mode or key_info.get("mode")) == "major" else "小调"
        sec.character = f"{mode_word} · {energy_word} · {density_word}"


def analyze_phrases(
    notes: Sequence[NoteEvent],
    sections: Sequence[Section],
    chords: Sequence[ChordWindow],
    *,
    bpm: float,
    time_signature: str = "4/4",
    max_phrases: int = 160,
) -> list[dict[str, Any]]:
    """按休止、小节超周期把每个段落切成「乐句」。

    规则：
    * 音符之间的静音超过 ``SYNTAX_PHRASE_GAP``（至少 1.5 拍）→ 新乐句；
    * 一个乐句最长 4 小节（避免长段没有呼吸）；
    * 同一段落内编号 A1 / A2 …，网页和段落编曲都能看到。
    """
    if not notes or not sections:
        return []
    try:
        numerator = int(str(time_signature).split("/", 1)[0])
    except Exception:
        numerator = 4
    beat_seconds = 60.0 / bpm if bpm > 0 else 0.5
    gap_limit = max(float(SYNTAX_PHRASE_GAP), beat_seconds * 1.5)
    phrase_limit = max(1.5, beat_seconds * numerator * 4.0)
    ordered = sorted(notes, key=lambda n: (n.start, n.pitch))
    phrases: list[dict[str, Any]] = []

    for sec in sections:
        sec_notes = [n for n in ordered if sec.start - 1e-6 <= n.start < sec.end - 1e-6]
        if not sec_notes:
            sec.phrase_count = 0
            continue
        groups: list[list[NoteEvent]] = []
        current: list[NoteEvent] = [sec_notes[0]]
        current_start = sec_notes[0].start
        current_end = sec_notes[0].end
        for note in sec_notes[1:]:
            gap = note.start - current_end
            if gap > gap_limit or note.start - current_start >= phrase_limit:
                groups.append(current)
                current = [note]
                current_start = note.start
                current_end = note.end
            else:
                current.append(note)
                current_end = max(current_end, note.end)
        groups.append(current)

        # 合并过短的尾句，避免出现只有一个装饰音的“乐句”
        merged_groups: list[list[NoteEvent]] = []
        for group in groups:
            dur = max(n.end for n in group) - min(n.start for n in group)
            if merged_groups and dur < 0.28:
                merged_groups[-1].extend(group)
            else:
                merged_groups.append(group)
        groups = merged_groups
        sec.phrase_count = len(groups)

        for pidx, group in enumerate(groups):
            p_start = min(n.start for n in group)
            p_end = max(n.end for n in group)
            if p_end - p_start < 0.05:
                continue
            p_chords = _section_chords(chords, p_start, p_end)
            progression: list[str] = []
            for c in p_chords:
                numeral = roman_numeral(c, sec.key_tonic if sec.key_tonic >= 0 else 0, sec.key_mode or "major")
                if numeral and (not progression or progression[-1] != numeral):
                    progression.append(numeral)
            phrases.append(
                {
                    "index": len(phrases),
                    "label": f"{sec.label}{pidx + 1}",
                    "section": sec.index,
                    "section_label": sec.label,
                    "start": round(p_start, 3),
                    "end": round(p_end, 3),
                    "n_notes": len(group),
                    "key": sec.key,
                    "progression": progression[:8],
                    "bars": max(1, int(round((p_end - p_start) / max(0.2, beat_seconds * numerator)))),
                }
            )
            if len(phrases) >= max_phrases:
                return phrases
    return phrases


# ---------------------------------------------------------------------------
# 结构分析
# ---------------------------------------------------------------------------


def _beat_grid(pm: pretty_midi.PrettyMIDI, notes: list[NoteEvent]) -> tuple[np.ndarray, float]:
    """返回拍点数组与估计 BPM。

    MusicXML 来源会直接使用按完整速度地图生成的拍点，而不是从音频/MIDI
    事件反推，因此速度变化处的小节边界仍然准确。
    """
    score_beats = getattr(pm, "_score_beats", None)
    if score_beats is not None and len(score_beats) >= 2:
        return np.asarray(score_beats, dtype=np.float64), float(
            getattr(pm, "_score_bpm", 120.0)
        )
    try:
        beats = pm.get_beats()
    except Exception:  # pragma: no cover
        beats = np.array([])
    if beats is None or len(beats) < 4:
        # 用音符起点间隔中位数估计（假定时值以八分音符为主）
        starts = np.array([n.start for n in notes], dtype=np.float64)
        if len(starts) < 4:
            return np.array([0.0, 0.5]), 120.0
        iois = np.diff(np.unique(np.round(starts, 3)))
        iois = iois[(iois > 0.05) & (iois < 2.0)]
        beat = float(np.median(iois)) * 2 if len(iois) else 0.5
        beat = max(0.2, min(1.5, beat))
        end = float(starts.max() + 1.0)
        beats = np.arange(0.0, end, beat)
        return beats, 60.0 / beat
    beats = np.asarray(beats, dtype=np.float64)
    if len(beats) >= 2:
        bpm = 60.0 / float(np.median(np.diff(beats)))
    else:
        bpm = 120.0
    return beats, bpm


def analyze_structure(
    pm: pretty_midi.PrettyMIDI,
    notes: list[NoteEvent],
    beats_per_section: int = SECTION_BEATS,
    similarity: float = SECTION_SIMILARITY,
) -> tuple[list[Section], float]:
    """自相似段落分析。"""
    if not notes:
        return [], 0.0

    beats, bpm = _beat_grid(pm, notes)
    if len(beats) < 2:
        beats = np.array([0.0, max(n.end for n in notes)])
    if len(beats) < beats_per_section:
        # 曲子太短：整体作为一个段落
        end = float(max(n.end for n in notes))
        return (
            [Section(0, "A", 0.0, end, len(notes), float(np.mean([n.velocity for n in notes])) / 127.0)],
            bpm,
        )

    # 段落边界
    boundaries = list(beats[::beats_per_section])
    end_time = float(max(n.end for n in notes))
    if boundaries[-1] < end_time:
        boundaries.append(end_time)
    else:
        boundaries[-1] = max(boundaries[-1], end_time)

    note_starts = np.array([n.start for n in notes])
    note_pitch = np.array([n.pitch for n in notes])
    note_vel = np.array([n.velocity for n in notes])
    note_dur = np.array([n.duration for n in notes])

    features: list[np.ndarray] = []
    energies: list[float] = []
    counts: list[int] = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        mask = (note_starts >= s) & (note_starts < e)
        idx = np.where(mask)[0]
        counts.append(len(idx))
        if len(idx) == 0:
            features.append(np.zeros(28))
            energies.append(0.0)
            continue
        pcs = note_pitch[idx] % 12
        chroma = np.zeros(12)
        for pc, d in zip(pcs, note_dur[idx]):
            chroma[int(pc)] += max(0.05, float(d))
        chroma = chroma / (chroma.sum() + 1e-9)
        # 低音 chroma：和声进行的根音/转位对段落辨识非常重要，
        # 因此把低音区的音级单独作为一组特征，而不是只统计全频段 chroma。
        low_mask = note_pitch[idx] < 55
        bass_chroma = np.zeros(12)
        if np.any(low_mask):
            for pc, d in zip(pcs[low_mask], note_dur[idx][low_mask]):
                bass_chroma[int(pc)] += max(0.05, float(d))
            bass_chroma = bass_chroma / (bass_chroma.sum() + 1e-9)
        rhythm = np.array(
            [
                min(1.0, len(idx) / max(1.0, (e - s)) / 8.0),
                float(note_vel[idx].mean()) / 127.0,
                (float(note_pitch[idx].mean()) - 48) / 48.0,
                min(1.0, float(np.mean(note_dur[idx])) / 1.5),
            ]
        )
        feat = np.concatenate([chroma * 1.2, bass_chroma * 0.8, rhythm * 0.8])
        features.append(feat)
        energies.append(float(note_vel[idx].mean()) / 127.0 * min(1.0, len(idx) / max(1.0, (e - s)) / 4.0))

    labels: list[str] = []
    repeat_of: list[str | None] = []
    for i, feat in enumerate(features):
        best_j = -1
        best_sim = -1.0
        if np.linalg.norm(feat) > 1e-6:
            # 跳过相邻窗口：相邻段落本来就相似，只寻找「隔开一段」的重复
            for j in range(i - 1):
                if np.linalg.norm(features[j]) < 1e-6:
                    continue
                sim = float(
                    np.dot(feat, features[j])
                    / (np.linalg.norm(feat) * np.linalg.norm(features[j]) + 1e-9)
                )
                if sim > best_sim:
                    best_sim, best_j = sim, j
        if best_j >= 0 and best_sim >= similarity:
            labels.append(labels[best_j])
            repeat_of.append(labels[best_j])
        else:
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            new_label = alphabet[len(set(labels)) % len(alphabet)]
            labels.append(new_label)
            repeat_of.append(None)

    # 合并相邻同标签
    sections: list[Section] = []
    cur_label = labels[0]
    cur_start = float(boundaries[0])
    cur_count = counts[0]
    cur_energy = [energies[0]]
    cur_repeat = repeat_of[0]
    for i in range(1, len(labels)):
        if labels[i] == cur_label:
            cur_count += counts[i]
            cur_energy.append(energies[i])
        else:
            sections.append(
                Section(
                    index=len(sections),
                    label=cur_label,
                    start=round(cur_start, 3),
                    end=round(float(boundaries[i]), 3),
                    n_notes=cur_count,
                    energy=float(np.mean(cur_energy)) if cur_energy else 0.0,
                    repeat_of=cur_repeat,
                )
            )
            cur_label = labels[i]
            cur_start = float(boundaries[i])
            cur_count = counts[i]
            cur_energy = [energies[i]]
            cur_repeat = repeat_of[i]
    sections.append(
        Section(
            index=len(sections),
            label=cur_label,
            start=round(cur_start, 3),
            end=round(float(boundaries[-1]), 3),
            n_notes=cur_count,
            energy=float(np.mean(cur_energy)) if cur_energy else 0.0,
            repeat_of=cur_repeat,
        )
    )

    # 合并后重新计算 repeat_of（同标签的第二次及以后）
    seen_labels: set[str] = set()
    for sec in sections:
        if sec.label in seen_labels:
            sec.repeat_of = sec.label
        else:
            seen_labels.add(sec.label)
            sec.repeat_of = None

    return sections, bpm


# ---------------------------------------------------------------------------
# 动机检测
# ---------------------------------------------------------------------------


def _melodic_sequence(notes: list[NoteEvent], onset_tol: float = 0.06) -> list[NoteEvent]:
    """把复调音符压缩成单声部旋律序列（每个起音簇取最高音，过短装饰音保留）。"""
    if not notes:
        return []
    ordered = sorted(notes, key=lambda n: (n.start, -n.pitch))
    seq: list[NoteEvent] = []
    cur = ordered[0]
    for note in ordered[1:]:
        if abs(note.start - cur.start) <= onset_tol:
            if note.pitch > cur.pitch:
                cur = note
        else:
            seq.append(cur)
            cur = note
    seq.append(cur)
    return seq


def detect_motifs(
    notes: list[NoteEvent],
    min_notes: int = MOTIF_MIN_NOTES,
    max_notes: int = MOTIF_MAX_NOTES,
    min_occurrences: int = MOTIF_MIN_OCCURRENCES,
    max_results: int = MOTIF_MAX_RESULTS,
) -> list[Motif]:
    """检测重复出现的旋律动机（移调不变）。"""
    if not notes:
        return []

    by_track: dict[int, list[NoteEvent]] = defaultdict(list)
    for n in notes:
        by_track[n.track].append(n)

    # signature -> 出现列表 [(track, [note_ids], start, end, pitches, iois)]
    occurrences: dict[tuple, list[dict[str, Any]]] = defaultdict(list)

    for track, tnotes in by_track.items():
        seq = _melodic_sequence(tnotes)
        if len(seq) < min_notes:
            continue
        # 限制单轨计算量
        if len(seq) > 4000:
            seq = seq[:4000]
        pitches = np.array([n.pitch for n in seq], dtype=np.int32)
        starts = np.array([n.start for n in seq], dtype=np.float64)
        iois = np.diff(starts)
        base = float(np.median(iois[iois > 0])) if np.any(iois > 0) else 0.25
        base = max(base, 1e-2)

        for length in range(max_notes, min_notes - 1, -1):
            for i in range(0, len(seq) - length + 1):
                intervals = tuple(int(x) for x in np.diff(pitches[i:i + length]))
                span = float(starts[i + length - 1] - starts[i])
                if span <= 0 or span > 12.0:
                    continue
                rhythm = tuple(
                    round(math.log2(max(iois[i + k], 1e-3) / base), 1)
                    for k in range(length - 1)
                )
                # 全部同音 / 无节奏差异的重复不视为动机
                if not any(intervals) and len(set(rhythm)) <= 1:
                    continue
                sig = (intervals, rhythm)
                occurrences[sig].append(
                    {
                        "track": track,
                        "ids": [seq[i + k].id for k in range(length)],
                        "start": round(float(starts[i]), 4),
                        "end": round(float(max(seq[i + k].end for k in range(length))), 4),
                        "pitches": [int(p) for p in pitches[i:i + length]],
                        "ioi": [round(float(x), 4) for x in iois[i:i + length - 1]],
                    }
                )

    # 过滤 + 去重（同一段音乐的多个长度只保留最长）
    candidates: list[Motif] = []
    for sig, occ in occurrences.items():
        if len(occ) < min_occurrences:
            continue
        intervals, rhythm = sig
        # 不同起点的出现才算重复
        unique_starts = {(o["track"], o["start"]) for o in occ}
        if len(unique_starts) < min_occurrences:
            continue
        # 去掉几乎完全重叠的出现（同一次演奏被多个窗口捕获）
        occ_sorted = sorted(occ, key=lambda o: (o["start"], -len(o["ids"])))
        filtered: list[dict[str, Any]] = []
        for o in occ_sorted:
            overlap = False
            for p in filtered:
                if p["track"] == o["track"] and not (
                    o["end"] <= p["start"] + 1e-3 or o["start"] >= p["end"] - 1e-3
                ):
                    overlap = True
                    break
            if not overlap:
                filtered.append(o)
        if len(filtered) < min_occurrences:
            continue
        score = (len(intervals) + 1) * len(filtered)
        candidates.append(
            Motif(
                id="",
                length=len(intervals) + 1,
                intervals=list(intervals),
                occurrences=filtered[:16],
                score=float(score),
                sample_pitches=filtered[0]["pitches"],
            )
        )

    candidates.sort(key=lambda m: (m.score, m.length), reverse=True)

    # 抑制与已选动机重叠严重的候选
    chosen: list[Motif] = []
    used: set[int] = set()
    for cand in candidates:
        ids = {i for o in cand.occurrences for i in o["ids"]}
        if ids and len(ids & used) / len(ids) > 0.6:
            continue
        cand.id = f"M{len(chosen) + 1}"
        chosen.append(cand)
        used |= ids
        if len(chosen) >= max_results:
            break
    return chosen


# ---------------------------------------------------------------------------
# 组合分析
# ---------------------------------------------------------------------------


def analyze(
    midi_path: str,
    selected_tracks: Iterable[int] | None = None,
    onset_tol: float = CHORD_ONSET_TOL,
    merge_gap: float = CHORD_MERGE_GAP,
    max_chords: int | None = None,
) -> AnalysisResult:
    """完整分析一个 MIDI 文件。"""
    pm, warnings = load_midi(midi_path)
    notes_all = collect_notes(pm)
    tracks = analyze_tracks(pm, notes_all)

    if selected_tracks is None:
        selected = [t.index for t in tracks if t.suggested_selected]
    else:
        selected = [int(i) for i in selected_tracks]

    notes = [n for n in notes_all if n.track in set(selected)] or notes_all

    chords = detect_chords(notes, onset_tol=onset_tol, merge_gap=merge_gap)
    if max_chords is not None:
        chords = chords[:max_chords]
    # 预先标记「分解和弦」窗口，便于网页展示与统计
    try:
        build_lyric_slots(notes, chords, onset_tol=onset_tol)
    except Exception:  # pragma: no cover
        logger.debug("分解和弦标记失败", exc_info=True)
    sections, bpm = analyze_structure(pm, notes)
    time_signature = detect_time_signature(pm)
    key_info = detect_key(notes)
    _annotate_sections(sections, notes, chords, key_info, bpm, time_signature)
    phrases = analyze_phrases(
        notes,
        sections,
        chords,
        bpm=bpm,
        time_signature=time_signature,
    )
    motifs = detect_motifs(notes)

    score_meta = dict(getattr(pm, "score_meta", {}) or {})
    source_format = str(score_meta.get("format", "midi") or "midi")
    tempo_changes: list[dict[str, Any]] = []
    if score_meta.get("tempo_changes"):
        tempo_changes = list(score_meta["tempo_changes"])
    else:
        try:
            times, bpms = pm.get_tempo_changes()
            tempo_changes = [
                {
                    "time": round(float(t), 4),
                    "bpm": round(float(b), 4),
                    "label": "",
                }
                for t, b in zip(times, bpms)
            ]
        except Exception:  # pragma: no cover - 第三方解析差异
            tempo_changes = []

    title = os.path.splitext(os.path.basename(midi_path))[0]
    return AnalysisResult(
        midi_path=midi_path,
        title=title or "未命名",
        duration=float(pm.get_end_time()),
        tempo=bpm if bpm else None,
        tracks=tracks,
        chords=chords,
        sections=sections,
        motifs=motifs,
        notes=notes,
        warnings=warnings,
        key=key_info,
        phrases=phrases,
        time_signature=time_signature,
        source_format=source_format,
        tempo_changes=tempo_changes,
        score_meta=score_meta,
        notes_all=notes_all,
    )


__all__ = [
    "AnalysisResult",
    "ChordWindow",
    "LyricSlot",
    "Motif",
    "NoteEvent",
    "Section",
    "TrackAnalysis",
    "analyze",
    "analyze_phrases",
    "analyze_structure",
    "analyze_tracks",
    "build_lyric_slots",
    "collect_notes",
    "detect_chord",
    "detect_chords",
    "detect_key",
    "detect_motifs",
    "detect_time_signature",
    "load_midi",
    "note_name",
    "pc_name",
    "roman_numeral",
]


if __name__ == "__main__":  # pragma: no cover - 手动调试
    import json
    import sys

    logging.basicConfig(level=logging.INFO)
    path = sys.argv[1] if len(sys.argv) > 1 else str(
        __import__("pathlib").Path(__file__).resolve().parents[1] / "midi"
    )
    result = analyze(path)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2)[:4000])
