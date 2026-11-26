"""音符质量检查与“不良音符”MIDI 导出。

判定范围
--------
* **过高 / 过低**：按轨道八度移调后的实际发声音高判断。超出 C2~C6
  （36~84）或偏离本轨音高中位数两个八度，视为发声困难。
* **音量过低**：用实际发声音高、MIDI 力度（含渐弱终点）和轨道音量估算
  该音的相对电平；明显低于本轨中位电平才算不良。
* **与力度记号区分**：MusicXML 的 p / pp / mp 或渐弱属于作曲意图，默认
  不判为不良；只有低到几乎听不见（绝对电平过低或力度极小）才记录。
* **只导出不良音**：每轨生成一个只含不良音符的新 MIDI，合格音符全部剔除；
  另生成一个合并 MIDI，方便快速检查问题音。
* **补充 MIDI**：按「不良音轨（`*_NG`）→ 空白分隔轨 → 谱面全部轨道」排列，
  所有轨名使用英文基名（如 `violin1_NG` / `violin1`），未参与音频合成的
  轨道追加 `_NOAUDIO`（如 `triangle1_NOAUDIO`），便于一次导入后对照问题音、
  完整配器与音频缺口，又不会和音频文件名混淆。

该模块只做检查与 MIDI 写出，不修改原始 MIDI / MusicXML。
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from hachimi.analysis import NoteEvent
from hachimi.config import (
    BAD_NOTE_EXPRESSIVE_REL_DB,
    BAD_NOTE_QUIET_ABS_DB,
    BAD_NOTE_QUIET_REL_DB,
    BAD_NOTE_QUIET_VELOCITY,
    BAD_NOTE_RELATIVE_PITCH_SPAN,
    SINGABLE_MAX_PITCH,
    SINGABLE_MIN_PITCH,
    SR,
)
from hachimi.loudness import audibility_gain

_QUIET_DYNAMICS = {"ppp", "pp", "p", "mp"}
_REASON_LABELS = {
    "too_high": "过高",
    "too_low": "过低",
    "too_quiet": "音量过低",
}


@dataclass
class BadNote:
    id: int
    track: int
    pitch: int
    start: float
    end: float
    velocity: int
    velocity_end: int | None
    dynamic: str
    effective_pitch: int
    level_db: float
    relative_db: float
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "track": self.track,
            "pitch": self.pitch,
            "note": _pitch_name(self.pitch),
            "effective_pitch": self.effective_pitch,
            "effective_note": _pitch_name(self.effective_pitch),
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "velocity": self.velocity,
            "velocity_end": self.velocity_end,
            "dynamic": self.dynamic,
            "level_db": round(self.level_db, 2),
            "relative_db": round(self.relative_db, 2),
            "reasons": list(self.reasons),
            "reason_labels": [_REASON_LABELS[r] for r in self.reasons],
        }


def _pitch_name(pitch: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    p = int(pitch)
    return f"{names[p % 12]}{p // 12 - 1}"


def _velocity_gain(velocity: int, gamma: float = 1.6) -> float:
    v = max(1, min(127, int(velocity))) / 127.0
    return max(0.02, float(v) ** max(0.1, gamma))


def _note_field(note: Any, key: str, default: Any = None) -> Any:
    if isinstance(note, Mapping):
        return note.get(key, default)
    return getattr(note, key, default)


def _is_expressive_soft(note: Any) -> bool:
    """p / pp / mp 或明显渐弱：属于谱面力度意图。"""
    dynamic = str(_note_field(note, "dynamic", "") or "").strip().lower()
    if dynamic in _QUIET_DYNAMICS:
        return True
    vel = int(_note_field(note, "velocity", 64) or 64)
    end = _note_field(note, "velocity_end", None)
    return end is not None and int(end) <= vel - 12


def assess_bad_notes(
    notes: Sequence[NoteEvent],
    track_options: Mapping[int, Mapping[str, Any]],
    tracks: Sequence[Any],
    *,
    min_pitch: int = SINGABLE_MIN_PITCH,
    max_pitch: int = SINGABLE_MAX_PITCH,
    relative_span: int = BAD_NOTE_RELATIVE_PITCH_SPAN,
    quiet_rel_db: float = BAD_NOTE_QUIET_REL_DB,
    expressive_rel_db: float = BAD_NOTE_EXPRESSIVE_REL_DB,
    quiet_abs_db: float = BAD_NOTE_QUIET_ABS_DB,
    quiet_velocity: int = BAD_NOTE_QUIET_VELOCITY,
    sample_rate: int = SR,
) -> dict[str, Any]:
    """检查每条轨道的音符，返回可 JSON 序列化的质量问题报告。"""
    del sample_rate  # 目前只做电平估算，保留参数便于未来做真实音频采样
    meta = {int(_note_field(t, "index", -1)): t for t in tracks}
    selected = {
        int(i)
        for i, opt in track_options.items()
        if bool(opt.get("selected", True)) and float(opt.get("gain", 1.0) or 0.0) > 0
    }
    usable = [n for n in notes if int(_note_field(n, "track", -1)) in selected]

    by_track: dict[int, list[Any]] = defaultdict(list)
    for note in usable:
        by_track[int(_note_field(note, "track", -1))].append(note)

    tracks_out: list[dict[str, Any]] = []
    bad_notes: list[BadNote] = []
    for idx in sorted(by_track):
        items = sorted(by_track[idx], key=lambda n: (float(_note_field(n, "start", 0.0)), int(_note_field(n, "pitch", 0))))
        opt = track_options.get(idx, {})
        octave = int(opt.get("octave", 0) or 0)
        gain = max(0.001, float(opt.get("gain", 1.0) or 0.0))
        info = meta.get(idx)
        name = str(_note_field(info, "name", opt.get("name", f"Track {idx}")) or f"Track {idx}")
        instrument = str(_note_field(info, "instrument", opt.get("instrument", "")) or "")

        rows: list[dict[str, Any]] = []
        eff_pitches: list[int] = []
        levels: list[float] = []
        for note in items:
            pitch = int(_note_field(note, "pitch", 60))
            eff = int(np.clip(pitch + 12 * octave, 21, 108))
            vel = int(_note_field(note, "velocity", 64) or 64)
            vel_end = _note_field(note, "velocity_end", None)
            avg_vel = (vel + int(vel_end)) / 2.0 if vel_end is not None else float(vel)
            level = 20.0 * math.log10(
                max(1e-9, _velocity_gain(avg_vel) * audibility_gain(eff) * gain)
            )
            eff_pitches.append(eff)
            levels.append(level)
            rows.append(
                {
                    "note": note,
                    "pitch": pitch,
                    "eff": eff,
                    "vel": vel,
                    "vel_end": int(vel_end) if vel_end is not None else None,
                    "dynamic": str(_note_field(note, "dynamic", "") or ""),
                    "level": level,
                }
            )

        median_pitch = float(np.median(eff_pitches)) if eff_pitches else 0.0
        median_level = float(np.median(levels)) if levels else -60.0
        reason_counts: Counter[str] = Counter()
        track_bad: list[BadNote] = []
        for row in rows:
            note = row["note"]
            reasons: list[str] = []
            if row["eff"] < min_pitch or (
                row["eff"] < 60 and median_pitch - row["eff"] >= relative_span
            ):
                reasons.append("too_low")
            if row["eff"] > max_pitch or (
                row["eff"] > 60 and row["eff"] - median_pitch >= relative_span
            ):
                reasons.append("too_high")

            expressive = _is_expressive_soft(note)
            rel_db = float(row["level"] - median_level)
            duration = float(_note_field(note, "end", 0.0)) - float(_note_field(note, "start", 0.0))
            limit = expressive_rel_db if expressive else quiet_rel_db
            too_quiet = False
            if duration >= 0.08:
                if row["level"] < quiet_abs_db:
                    too_quiet = True
                elif rel_db < limit:
                    too_quiet = True
                elif not expressive and row["vel"] <= quiet_velocity:
                    too_quiet = True
                elif expressive and row["vel"] <= 8:
                    # 谱面 p 可以保留，但力度低到 1~8 时实际已几乎听不见。
                    too_quiet = True
            # 力度记号导致的弱音单独记录；除非已经低到绝对听不见，否则不算不良。
            if too_quiet:
                reasons.append("too_quiet")

            if not reasons:
                continue
            item = BadNote(
                id=int(_note_field(note, "id", -1)),
                track=idx,
                pitch=row["pitch"],
                start=float(_note_field(note, "start", 0.0)),
                end=float(_note_field(note, "end", 0.0)),
                velocity=row["vel"],
                velocity_end=row["vel_end"],
                dynamic=row["dynamic"],
                effective_pitch=row["eff"],
                level_db=row["level"],
                relative_db=rel_db,
                reasons=reasons,
            )
            bad_notes.append(item)
            track_bad.append(item)
            reason_counts.update(reasons)

        tracks_out.append(
            {
                "index": idx,
                "name": name,
                "instrument": instrument,
                "octave": octave,
                "gain": round(gain, 3),
                "total_notes": len(items),
                "bad_notes": len(track_bad),
                "too_high": int(reason_counts.get("too_high", 0)),
                "too_low": int(reason_counts.get("too_low", 0)),
                "too_quiet": int(reason_counts.get("too_quiet", 0)),
                "expressive_soft": sum(1 for n in items if _is_expressive_soft(n)),
                "median_pitch": round(median_pitch, 2),
                "median_level_db": round(median_level, 2),
                "bad_ids": [n.id for n in track_bad],
                "examples": [n.to_dict() for n in track_bad[:8]],
            }
        )

    reason_total: Counter[str] = Counter()
    for note in bad_notes:
        reason_total.update(note.reasons)
    total_notes = len(usable)
    bad_total = len(bad_notes)
    return {
        "ok": bad_total == 0,
        "total_notes": total_notes,
        "bad_notes": bad_total,
        "bad_ratio": round(bad_total / total_notes, 4) if total_notes else 0.0,
        "too_high": int(reason_total.get("too_high", 0)),
        "too_low": int(reason_total.get("too_low", 0)),
        "too_quiet": int(reason_total.get("too_quiet", 0)),
        "tracks": tracks_out,
        "notes": [n.to_dict() for n in bad_notes[:2000]],
        "notes_truncated": max(0, bad_total - 2000),
        "thresholds": {
            "min_pitch": int(min_pitch),
            "max_pitch": int(max_pitch),
            "relative_span": int(relative_span),
            "quiet_rel_db": float(quiet_rel_db),
            "expressive_rel_db": float(expressive_rel_db),
            "quiet_abs_db": float(quiet_abs_db),
            "quiet_velocity": int(quiet_velocity),
        },
        "message": (
            "音符质量检查通过：没有发现过高、过低或音量过低的不良音符"
            if bad_total == 0
            else f"发现 {bad_total} 个不良音符（过高 {reason_total.get('too_high', 0)} / "
            f"过低 {reason_total.get('too_low', 0)} / 音量过低 {reason_total.get('too_quiet', 0)}）"
        ),
    }


def _program_for(track_info: Any, fallback: int = 0) -> int:
    value = _note_field(track_info, "program", fallback)
    try:
        return max(0, min(127, int(value)))
    except (TypeError, ValueError):
        return fallback


def _safe_midi_name(value: Any, fallback: str = "") -> str:
    """MIDI 轨道名只保留 ASCII，避免部分音序器把中文元事件读成乱码。"""
    text = str(value or fallback)
    return text.encode("ascii", "ignore").decode("ascii")[:48] or fallback


_TRACK_BASE_ALIASES = {
    # 统一成简短英文名，避免与音频文件名（含中文语音 / 八度）混淆。
    "stringensemble": "strings",
    "drumkit": "drums",
    "grtr": "bassdrum",
    "beck": "cymbals",
    "orchestralharp": "harp",
    "tubularbells": "bells",
    "acousticgrandpiano": "piano",
    "brightacousticpiano": "piano",
    "electricgrandpiano": "piano",
    "honkytonkpiano": "piano",
    "electricpiano1": "piano",
    "electricpiano2": "piano",
}


def _ascii_slug(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    # 去掉重音 / 变音符号后再剔除标点，确保得到纯英文轨道名。
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9]+", "", text).lower()


def _track_base_slug(track_info: Any) -> str:
    # 打击乐的 GM 音色都是 Drum Kit，只有 part-name（Triangle / Gr. Tr. /
    # Tamtam / Beck.）能区分具体乐器；优先用名称而不是 Drum Kit。
    if bool(_note_field(track_info, "is_drum", False)):
        source = str(_note_field(track_info, "name", "") or "")
    else:
        source = str(_note_field(track_info, "instrument", "") or "")
    if not _ascii_slug(source):
        source = str(_note_field(track_info, "name", "") or "")
    slug = _ascii_slug(source) or "track"
    # "String Ensemble 1" / "Synth Bass 2" 之类的编号交给后面的实例计数器，
    # 避免出现 stringensemble11 这种重复编号。
    stem = re.sub(r"\d+$", "", slug) or slug
    return _TRACK_BASE_ALIASES.get(stem, stem)


def _build_track_names(tracks: Sequence[Any]) -> dict[int, str]:
    """为每条有声轨道分配英文名：violin1、violin2、flute1 …

    只统计真正会写入 MIDI 的轨道（``n_notes > 0``），空轨道不占用序号。
    """
    names: dict[int, str] = {}
    counts: Counter[str] = Counter()
    ordered = sorted(tracks, key=lambda t: int(_note_field(t, "index", 0) or 0))
    for info in ordered:
        if int(_note_field(info, "n_notes", 0) or 0) <= 0:
            continue
        idx = int(_note_field(info, "index", -1))
        base = _track_base_slug(info)
        counts[base] += 1
        names[idx] = _safe_midi_name(f"{base}{counts[base]}", f"track{idx + 1}")
    return names


def _instrument_for(
    track_info: Any,
    name: str,
    notes: Sequence[Any],
) -> Any:
    """按源轨道音色创建 Instrument，并写入给定音符。"""
    import pretty_midi

    is_drum = bool(_note_field(track_info, "is_drum", False))
    program = 0 if is_drum else _program_for(track_info)
    instrument = pretty_midi.Instrument(
        program=program,
        is_drum=is_drum,
        name=_safe_midi_name(name, "track"),
    )
    for note in sorted(
        notes,
        key=lambda n: (
            float(_note_field(n, "start", 0.0)),
            int(_note_field(n, "pitch", 60)),
        ),
    ):
        start = max(0.0, float(_note_field(note, "start", 0.0)))
        end = max(start + 0.01, float(_note_field(note, "end", start + 0.01)))
        instrument.notes.append(
            pretty_midi.Note(
                velocity=max(1, min(127, int(_note_field(note, "velocity", 64) or 64))),
                pitch=max(0, min(127, int(_note_field(note, "pitch", 60)))),
                start=start,
                end=end,
            )
        )
    return instrument


def _write_with_blank_track(pm: Any, path: str, after_tracks: int) -> None:
    """写出 MIDI，并在指定数量的乐器轨后插入一条真正的空白轨。

    ``pretty_midi`` 读取时会丢弃没有 note 事件的轨，所以空白分隔轨必须在
    ``PrettyMIDI.write`` 之后用 ``mido`` 直接插入；这样文件本身仍保留该轨，
    DAW 打开时能看到一条空轨，而不是被解析器悄悄吞掉。
    """
    import mido

    pm.write(path)
    mid = mido.MidiFile(path)
    blank = mido.MidiTrack()
    blank.append(mido.MetaMessage("end_of_track", time=0))
    # 第 0 轨是速度 / 拍号轨，之后才是乐器轨。
    insert_at = max(1, min(len(mid.tracks), 1 + max(0, int(after_tracks))))
    mid.tracks.insert(insert_at, blank)
    mid.save(path)


def _apply_tempo_map(
    pm: Any,
    tempo: float,
    tempo_changes: Sequence[Mapping[str, Any]] | None,
    time_signature: str = "",
) -> None:
    """把乐谱的完整速度地图写入新建的 ``PrettyMIDI``。

    MusicXML 转 MIDI 时，速度变化密集到每个十六分音符都可能有自动化
    （例如 sync (8) 有 29056 处）。只写初始 BPM 会让诊断 MIDI 的小节线
    与原始乐谱错位；这里按谱面 ``position``（四分音符数）恢复速度地图。
    音符本身仍使用秒级时间轴，因此绝对播放位置不变。
    """
    import pretty_midi

    changes: dict[int, float] = {}
    for item in tempo_changes or []:
        try:
            bpm = float(item.get("bpm", 0) or 0)
        except (TypeError, ValueError):
            continue
        if bpm <= 0:
            continue
        position = item.get("position")
        if position is None:
            continue
        try:
            tick = int(round(float(position) * pm.resolution))
        except (TypeError, ValueError):
            continue
        if tick >= 0:
            changes[tick] = bpm

    initial = max(1.0, min(1000.0, float(tempo or 120.0)))
    if 0 not in changes:
        changes[0] = initial
    scales = [
        (tick, 60.0 / (bpm * pm.resolution))
        for tick, bpm in sorted(changes.items())
    ]
    if not scales:
        return
    pm._tick_scales = scales

    if time_signature:
        try:
            num, den = str(time_signature).split("/", 1)
            numerator = max(1, min(32, int(num)))
            denominator = max(1, min(32, int(den)))
            pm.time_signature_changes.append(
                pretty_midi.TimeSignature(numerator, denominator, 0.0)
            )
        except (TypeError, ValueError):
            pass

    # PrettyMIDI.write 会通过 time_to_tick 取音符位置；先按最晚结束时间
    # 建好 tick→秒表。这里故意取一个略大的上界，避免最后几个音被截断。
    end_time = max(
        (float(getattr(n, "end", 0.0) or 0.0) for inst in pm.instruments for n in inst.notes),
        default=1.0,
    )
    max_bpm = max([initial, *[bpm for bpm in changes.values()]])
    max_tick = max(
        scales[-1][0] + 1,
        int(max(1.0, end_time) * pm.resolution * max_bpm / 60.0) + pm.resolution * 4,
    )
    pm._update_tick_to_time(max_tick)


def write_bad_note_midis(
    notes: Sequence[NoteEvent],
    assessment: Mapping[str, Any],
    tracks: Sequence[Any],
    out_dir: str,
    project: str,
    *,
    tempo: float = 120.0,
    tempo_changes: Sequence[Mapping[str, Any]] | None = None,
    time_signature: str = "",
) -> dict[str, str]:
    """把每轨不良音符写成独立 MIDI，并额外写一个合并 MIDI。"""
    import pretty_midi

    folder = os.path.join(out_dir, "bad_notes")
    os.makedirs(folder, exist_ok=True)
    meta = {int(_note_field(t, "index", -1)): t for t in tracks}
    by_track: dict[int, list[NoteEvent]] = defaultdict(list)
    for note in notes:
        by_track[int(_note_field(note, "track", -1))].append(note)

    bad_ids: dict[int, set[int]] = defaultdict(set)
    for item in assessment.get("tracks", []) or []:
        idx = int(item.get("index", -1))
        bad_ids[idx].update(int(v) for v in item.get("bad_ids", []) or [])

    files: dict[str, str] = {}
    track_names = _build_track_names(tracks)
    combined = pretty_midi.PrettyMIDI(initial_tempo=max(20.0, min(400.0, float(tempo or 120.0))))
    for idx in sorted(bad_ids):
        ids = bad_ids[idx]
        if not ids:
            continue
        info = meta.get(idx)
        chosen = [
            note
            for note in by_track.get(idx, [])
            if int(_note_field(note, "id", -1)) in ids
        ]
        if not chosen:
            continue
        base_name = track_names.get(idx, f"track{idx + 1}")
        instrument = _instrument_for(info, f"{base_name}_NG", chosen)
        pm = pretty_midi.PrettyMIDI(
            initial_tempo=max(20.0, min(400.0, float(tempo or 120.0)))
        )
        pm.instruments.append(instrument)
        # 单轨文件保持轻量，只写初始 BPM；完整速度地图放在合并 / 补充 MIDI。
        path = os.path.join(folder, f"track{idx}_bad.mid")
        pm.write(path)
        files[str(idx)] = path
        combined.instruments.append(instrument)

    combined_path = ""
    if combined.instruments and any(inst.notes for inst in combined.instruments):
        combined_path = os.path.join(folder, f"{project}_bad_notes.mid")
        _apply_tempo_map(combined, float(tempo or 120.0), tempo_changes, time_signature)
        combined.write(combined_path)
        files["all"] = combined_path
    return files


def write_bad_and_all_midi(
    notes: Sequence[NoteEvent],
    assessment: Mapping[str, Any],
    tracks: Sequence[Any],
    out_dir: str,
    project: str,
    *,
    missing_indices: Sequence[int] = (),
    missing_reasons: Mapping[int, str] | None = None,
    tempo: float = 120.0,
    tempo_changes: Sequence[Mapping[str, Any]] | None = None,
    time_signature: str = "",
) -> dict[str, Any]:
    """写一条“全谱检查 MIDI”。

    轨道顺序固定为三段：

    1. **不良音轨**：每条有问题轨道一条，名称 ``<英文名>_NG``，只含不良音；
    2. **空白轨**：一条没有任何音符的空轨，作为视觉分隔；
    3. **全谱轨**：谱面所有有声轨道各一条，名称 ``<英文名>``，含完整音符。

    同一轨道的 ``_NG`` 与全谱版本使用同一个英文基名（如 ``violin1_NG`` /
    ``violin1``），但不会与音频文件名（含语音 / 八度 / 中文）重名。
    空轨道无法写入音符，单独记录在 ``skipped`` 中。
    """
    import pretty_midi

    reason_by_index = {int(k): str(v) for k, v in (missing_reasons or {}).items()}
    missing_set = {int(i) for i in missing_indices}
    meta = {int(_note_field(t, "index", -1)): t for t in tracks}
    by_track: dict[int, list[NoteEvent]] = defaultdict(list)
    for note in notes:
        by_track[int(_note_field(note, "track", -1))].append(note)

    bad_ids: dict[int, set[int]] = defaultdict(set)
    for item in assessment.get("tracks", []) or []:
        idx = int(item.get("index", -1))
        bad_ids[idx].update(int(v) for v in item.get("bad_ids", []) or [])

    track_names = _build_track_names(tracks)
    ordered: list[int] = [int(_note_field(t, "index", -1)) for t in tracks]
    for idx in sorted(by_track):
        if idx not in ordered:
            ordered.append(idx)
    ordered = [idx for idx in ordered if idx >= 0]

    entries: list[dict[str, Any]] = []
    instruments: list[Any] = []
    skipped: list[dict[str, Any]] = []
    bad_tracks = 0
    score_tracks = 0
    total_bad = 0
    total_score = 0
    total_missing = 0

    # ---- 1. 不良音轨（_NG） ----
    for idx in ordered:
        track_notes = by_track.get(idx, [])
        ids = bad_ids.get(idx, set())
        if not track_notes or not ids:
            continue
        info = meta.get(idx)
        chosen = [
            note
            for note in track_notes
            if int(_note_field(note, "id", -1)) in ids
        ]
        if not chosen:
            continue
        base_name = track_names.get(idx, f"track{idx + 1}")
        no_audio = idx in missing_set
        label = f"{base_name}_NG" + ("_NOAUDIO" if no_audio else "")
        instrument = _instrument_for(info, label, chosen)
        instruments.append(instrument)
        bad_tracks += 1
        total_bad += len(chosen)
        entries.append(
            {
                "index": idx,
                "section": "bad",
                "kind": "bad",
                "name": str(_note_field(info, "name", f"Track {idx}") or ""),
                "instrument": str(_note_field(info, "instrument", "") or ""),
                "midi_track_name": label,
                "n_notes": len(track_notes),
                "included_notes": len(instrument.notes),
                "bad_notes": len(instrument.notes),
                "reason": "只含不良音符" + ("（未参与音频合成）" if no_audio else ""),
                "is_unoutput": no_audio,
            }
        )

    # ---- 2. 空白分隔轨 ----
    # 只记一条元数据；真正的空轨在写文件后用 mido 插入（PrettyMIDI 会丢弃
    # 没有 note 的 Instrument）。
    entries.append(
        {
            "index": None,
            "section": "spacer",
            "kind": "spacer",
            "name": "",
            "instrument": "",
            "midi_track_name": "",
            "n_notes": 0,
            "included_notes": 0,
            "bad_notes": 0,
            "reason": "空白分隔轨",
        }
    )

    # ---- 3. 谱面全部有声轨道 ----
    for idx in ordered:
        track_notes = by_track.get(idx, [])
        if not track_notes:
            info = meta.get(idx)
            if info is not None:
                skipped.append(
                    {
                        "index": idx,
                        "name": str(_note_field(info, "name", f"Track {idx}") or ""),
                        "instrument": str(_note_field(info, "instrument", "") or ""),
                        "reason": reason_by_index.get(idx, "空轨道（没有音符）"),
                    }
                )
            continue
        info = meta.get(idx)
        base_name = track_names.get(idx, f"track{idx + 1}")
        reason = reason_by_index.get(idx, "未输出") if idx in missing_set else ""
        full_name = base_name + ("_NOAUDIO" if reason else "")
        instrument = _instrument_for(info, full_name, track_notes)
        instruments.append(instrument)
        score_tracks += 1
        total_score += len(track_notes)
        if reason:
            total_missing += len(track_notes)
        entries.append(
            {
                "index": idx,
                "section": "full",
                "kind": "full",
                "name": str(_note_field(info, "name", f"Track {idx}") or ""),
                "instrument": str(_note_field(info, "instrument", "") or ""),
                "midi_track_name": full_name,
                "n_notes": len(track_notes),
                "included_notes": len(track_notes),
                "bad_notes": sum(
                    1 for n in track_notes
                    if int(_note_field(n, "id", -1)) in bad_ids.get(idx, set())
                ),
                "reason": reason or "全谱完整轨道",
                "is_unoutput": bool(reason),
                "unoutput_reason": reason,
            }
        )

    if not bad_tracks and not score_tracks:
        return {
            "path": "",
            "file": "",
            "n_tracks": 0,
            "n_entries": 0,
            "n_bad_tracks": 0,
            "n_score_tracks": 0,
            "n_spacer_tracks": 0,
            "n_missing_tracks": 0,
            "n_notes": 0,
            "bad_notes": 0,
            "score_notes": 0,
            "missing_notes": 0,
            "tracks": [],
            "skipped": skipped,
            "message": "没有可写入补充 MIDI 的不良音或谱面轨道",
        }

    folder = os.path.join(out_dir, "bad_notes")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{project}_bad_and_all.mid")
    pm = pretty_midi.PrettyMIDI(
        initial_tempo=max(20.0, min(1000.0, float(tempo or 120.0)))
    )
    pm.instruments.extend(instruments)
    _apply_tempo_map(pm, float(tempo or 120.0), tempo_changes, time_signature)
    _write_with_blank_track(pm, path, bad_tracks)

    n_tracks = bad_tracks + 1 + score_tracks
    return {
        "path": path,
        "file": path,
        "n_tracks": n_tracks,
        "n_entries": len(entries),
        "n_bad_tracks": bad_tracks,
        "n_score_tracks": score_tracks,
        "n_spacer_tracks": 1,
        "n_missing_tracks": sum(
            1 for item in entries
            if item.get("section") == "full" and item.get("is_unoutput")
        ),
        "n_notes": total_bad + total_score,
        "bad_notes": total_bad,
        "score_notes": total_score,
        "missing_notes": total_missing,
        "tracks": entries,
        "skipped": skipped,
        "message": (
            f"补充 MIDI：{bad_tracks} 条不良音轨 → 1 条空白轨 → "
            f"{score_tracks} 条全谱轨，共 {n_tracks} 轨 / "
            f"{total_bad + total_score} 个音符"
            f"（不良 {total_bad} + 全谱 {total_score}）"
        ),
    }


def write_percussion_midi(
    notes: Sequence[NoteEvent],
    tracks: Sequence[Any],
    out_dir: str,
    project: str,
    *,
    tempo: float = 120.0,
    tempo_changes: Sequence[Mapping[str, Any]] | None = None,
    time_signature: str = "",
) -> dict[str, Any]:
    """单独导出打击乐 MIDI。

    打击乐没有可唱的元音，默认不进入哈基米音频合成；这里按 GM 鼓音高
    （由 MusicXML ``midi-unpitched`` 或 MIDI 原音符还原）单独写一条文件，
    便于在 DAW / 鼓采样器里直接使用。轨名统一带 ``_NOAUDIO`` 标记。
    """
    import pretty_midi

    by_track: dict[int, list[NoteEvent]] = defaultdict(list)
    for note in notes:
        by_track[int(_note_field(note, "track", -1))].append(note)
    names = _build_track_names(tracks)

    instruments: list[Any] = []
    entries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    total = 0
    for info in sorted(tracks, key=lambda t: int(_note_field(t, "index", 0) or 0)):
        idx = int(_note_field(info, "index", -1))
        if not bool(_note_field(info, "is_drum", False)):
            continue
        track_notes = by_track.get(idx, [])
        if not track_notes:
            skipped.append(
                {
                    "index": idx,
                    "name": str(_note_field(info, "name", f"Track {idx}") or ""),
                    "instrument": str(_note_field(info, "instrument", "") or ""),
                    "reason": "空轨道（没有打击乐音符）",
                }
            )
            continue
        base_name = names.get(idx, f"drums{idx + 1}")
        label = f"{base_name}_NOAUDIO"
        instrument = _instrument_for(info, label, track_notes)
        instruments.append(instrument)
        total += len(track_notes)
        entries.append(
            {
                "index": idx,
                "name": str(_note_field(info, "name", f"Track {idx}") or ""),
                "instrument": str(_note_field(info, "instrument", "") or ""),
                "midi_track_name": label,
                "n_notes": len(track_notes),
            }
        )

    if not instruments:
        return {
            "path": "",
            "file": "",
            "n_tracks": 0,
            "n_notes": 0,
            "tracks": [],
            "skipped": skipped,
            "message": "没有可导出的打击乐音符",
        }

    path = os.path.join(out_dir, f"{project}_percussion.mid")
    pm = pretty_midi.PrettyMIDI(
        initial_tempo=max(20.0, min(1000.0, float(tempo or 120.0)))
    )
    pm.instruments.extend(instruments)
    _apply_tempo_map(pm, float(tempo or 120.0), tempo_changes, time_signature)
    pm.write(path)
    return {
        "path": path,
        "file": path,
        "n_tracks": len(instruments),
        "n_notes": total,
        "tracks": entries,
        "skipped": skipped,
        "message": f"打击乐 MIDI：{len(instruments)} 轨 / {total} 个音符",
    }


__all__ = [
    "BadNote",
    "assess_bad_notes",
    "write_bad_note_midis",
    "write_bad_and_all_midi",
    "write_percussion_midi",
]
