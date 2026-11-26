"""复调发声校验：确认「同一时刻多个音」的每个音都真的被渲染/发声。

背景
----
上游旧版每个和弦只把歌词分配给最高音，其余音符不会进入渲染；
新版虽然会把整个和弦槽的所有音都送入渲染器，但仍需要一个**显式校验**，
避免未来改动（例如量化、时间范围裁剪、单位缺失）导致某个和弦音被漏掉。

校验分三层：

1. **歌词事件**：和弦组内每个音都必须有歌词事件（``state`` 非空）；
2. **渲染落位**：每个音都必须进入 ``RenderNote`` 列表，且没有被跳过，
   合成单位不是静音；
3. **分轨音频**：在每轨单独导出的 WAV 中，和弦起始窗口必须有实际能量。

任一层发现问题都会写入 ``verification.json`` 并在网页端显示警告。
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from hachimi.analysis import NoteEvent
from hachimi.render import RenderNote, RenderStats, note_unit_key

#: 判定「同时发声」的起音容差
DEFAULT_ONSET_TOL = 0.06
#: 分轨和弦窗口的能量阈值（RMS）
DEFAULT_ENERGY_THRESHOLD = 1e-5


def _key(track: int, pitch: int, start: float, end: float) -> tuple:
    return (int(track), int(pitch), round(float(start), 4), round(float(end), 4))


def build_chord_groups(
    notes: Sequence[NoteEvent],
    onset_tol: float = DEFAULT_ONSET_TOL,
) -> dict[int, list[list[NoteEvent]]]:
    """按轨道把音符分成起音簇，返回每轨的全部簇（含单音簇）。"""
    by_track: dict[int, list[NoteEvent]] = defaultdict(list)
    for n in notes:
        by_track[n.track].append(n)

    result: dict[int, list[list[NoteEvent]]] = {}
    for track, items in by_track.items():
        items = sorted(items, key=lambda n: (n.start, n.pitch))
        groups: list[list[NoteEvent]] = []
        cur = [items[0]]
        start = items[0].start
        for note in items[1:]:
            if note.start - start <= onset_tol:
                cur.append(note)
            else:
                groups.append(cur)
                cur = [note]
                start = note.start
        groups.append(cur)
        result[track] = groups
    return result


@dataclass
class TrackCheck:
    index: int
    name: str
    total_notes: int = 0
    verified_notes: int = 0
    chord_groups: int = 0
    chord_notes: int = 0
    verified_chord_notes: int = 0
    missing_events: list[dict[str, Any]] = field(default_factory=list)
    missing_render: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    silent_units: list[dict[str, Any]] = field(default_factory=list)
    silent_groups: list[dict[str, Any]] = field(default_factory=list)
    out_of_range: int = 0

    @property
    def ok(self) -> bool:
        return not (
            self.missing_events
            or self.missing_render
            or self.skipped
            or self.silent_units
            or self.silent_groups
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "total_notes": self.total_notes,
            "verified_notes": self.verified_notes,
            "chord_groups": self.chord_groups,
            "chord_notes": self.chord_notes,
            "verified_chord_notes": self.verified_chord_notes,
            "out_of_range": self.out_of_range,
            "ok": self.ok,
            "missing_events": self.missing_events[:20],
            "missing_render": self.missing_render[:20],
            "skipped": self.skipped[:20],
            "silent_units": self.silent_units[:20],
            "silent_groups": self.silent_groups[:20],
        }


def _note_brief(n: NoteEvent) -> dict[str, Any]:
    return {
        "track": n.track,
        "pitch": n.pitch,
        "start": round(n.start, 4),
        "end": round(n.end, 4),
    }


def check_stem_energy(
    audio: np.ndarray,
    sr: int,
    t0: float,
    groups: Sequence[Sequence[NoteEvent]],
    *,
    start_time: float = 0.0,
    end_time: float | None = None,
    threshold: float = DEFAULT_ENERGY_THRESHOLD,
) -> list[dict[str, Any]]:
    """检查单轨音频中每个和弦起音窗口是否有实际能量。"""
    problems: list[dict[str, Any]] = []
    for group in groups:
        if len(group) < 2:
            continue
        starts = [
            n.start
            for n in group
            if n.start >= start_time - 1e-6 and (end_time is None or n.start < end_time)
        ]
        if not starts:
            continue
        start = max(min(starts), t0)
        b = max(0, int(round((start - t0) * sr)))
        e = min(len(audio), b + max(1, int(round(0.12 * sr))))
        if e <= b:
            continue
        window = audio[b:e].astype(np.float64)
        rms = float(math.sqrt(float(np.mean(window * window)))) if window.size else 0.0
        if rms < threshold:
            problems.append(
                {
                    "track": group[0].track,
                    "start": round(start, 4),
                    "rms": round(rms, 9),
                    "pitches": [n.pitch for n in group],
                }
            )
    return problems


def verify_polyphony(
    notes: Sequence[NoteEvent],
    events: Sequence[dict[str, Any]],
    render_notes: Sequence[RenderNote],
    stats: RenderStats | None = None,
    track_names: dict[int, str] | None = None,
    *,
    onset_tol: float = DEFAULT_ONSET_TOL,
    start_time: float = 0.0,
    end_time: float | None = None,
    stem_energy: dict[int, list[dict[str, Any]]] | None = None,
    energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
) -> dict[str, Any]:
    """校验所有音符（尤其是和弦音）是否都被安排发声。

    Args:
        notes: 参与生成的音符（``analysis.notes``）。
        events: 歌词事件列表。
        render_notes: 送入渲染器的音符列表。
        stats: 渲染统计（用于跳过/静音单位信息）。
        track_names: 轨道名映射。
        start_time/end_time: 渲染时间范围（范围外音符不计为缺失）。
        stem_energy: ``{track: check_stem_energy() 的问题列表}``。
        energy_threshold: 分轨能量阈值。
    """
    track_names = track_names or {}
    event_keys = {
        _key(e["track"], e["pitch"], e["start"], e["end"])
        for e in events
        if e.get("state") or e.get("word")
    }

    def _render_end(note: RenderNote) -> float:
        # 连奏会把前音尾部延伸进后音，校验仍按原始时值匹配歌词事件。
        return note.source_end if note.source_end is not None else note.end

    render_keys = {
        _key(n.track, n.pitch, n.start, _render_end(n)) for n in render_notes
    }
    render_map = {
        _key(n.track, n.pitch, n.start, _render_end(n)): n for n in render_notes
    }
    skipped_keys = {
        _key(x.get("track", -1), x.get("pitch", -1), x.get("start", 0), x.get("end", 0))
        for x in (stats.skipped_notes if stats else [])
    }
    silent_unit_keys = {
        (x.get("sample") or x.get("word"), x.get("pitch"), x.get("duration_samples"))
        for x in (stats.silent_units if stats else [])
    }
    quantum = stats.duration_quantum if stats else 0.02

    groups_by_track = build_chord_groups(notes, onset_tol=onset_tol)
    checks: list[TrackCheck] = []

    for track in sorted(groups_by_track):
        groups = groups_by_track[track]
        check = TrackCheck(index=track, name=track_names.get(track, f"Track {track}"))
        for group in groups:
            in_range = [
                n
                for n in group
                if n.start >= start_time - 1e-6
                and (end_time is None or n.start < end_time)
            ]
            check.out_of_range += len(group) - len(in_range)
            if len(group) > 1 and in_range:
                check.chord_groups += 1
            for note in in_range:
                check.total_notes += 1
                key = _key(note.track, note.pitch, note.start, note.end)
                missing = False
                if key not in event_keys:
                    check.missing_events.append(_note_brief(note))
                    missing = True
                if key not in render_keys:
                    check.missing_render.append(_note_brief(note))
                    missing = True
                if key in skipped_keys:
                    check.skipped.append(_note_brief(note))
                    missing = True
                rn = render_map.get(key)
                if rn is not None and note_unit_key(rn, quantum) in silent_unit_keys:
                    check.silent_units.append(_note_brief(note))
                    missing = True
                if not missing:
                    check.verified_notes += 1

                if len(group) > 1:
                    check.chord_notes += 1
                    if not missing:
                        check.verified_chord_notes += 1

        # 分轨音频能量检查（由 pipeline 在导出分轨时计算后传入）
        if stem_energy and track in stem_energy:
            check.silent_groups.extend(stem_energy[track])

        checks.append(check)

    total_chords = sum(c.chord_groups for c in checks)
    total_chord_notes = sum(c.chord_notes for c in checks)
    verified_chord_notes = sum(c.verified_chord_notes for c in checks)
    total_notes = sum(c.total_notes for c in checks)
    verified_notes = sum(c.verified_notes for c in checks)
    ok = (
        all(c.ok for c in checks)
        and total_chord_notes == verified_chord_notes
        and total_notes == verified_notes
    )
    problems = sum(
        len(c.missing_events) + len(c.missing_render) + len(c.skipped)
        + len(c.silent_units) + len(c.silent_groups)
        for c in checks
    )

    if ok:
        message = f"发声检查通过：{total_chord_notes} 个和弦音全部发声，共 {total_chords} 个和弦组"
    else:
        message = f"发声检查发现 {problems} 个问题（和弦音 {verified_chord_notes}/{total_chord_notes}）"

    return {
        "ok": ok,
        "message": message,
        "checked_tracks": len(checks),
        "total_chords": total_chords,
        "total_chord_notes": total_chord_notes,
        "verified_chord_notes": verified_chord_notes,
        "total_notes": total_notes,
        "verified_notes": verified_notes,
        "problems": problems,
        "tracks": [c.to_dict() for c in checks],
    }


__all__ = [
    "TrackCheck",
    "build_chord_groups",
    "check_stem_energy",
    "verify_polyphony",
]
