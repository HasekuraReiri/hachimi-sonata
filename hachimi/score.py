"""MusicXML / MXL 乐谱读取。

本模块把 ``.xml`` / ``.musicxml`` / ``.mxl``（压缩 MusicXML）转成
:class:`pretty_midi.PrettyMIDI`，因此后续的轨道分析、歌词、渲染、字幕
全部复用 MIDI 流程。

与普通 MIDI 转换相比，这里额外保留了乐谱里真正影响演奏的标记：

* **完整速度地图**：读取每个 ``<sound tempo>`` / ``<metronome>``，按谱面
  位置（而不是小节起点）换算成秒；同一小节日内的速度变化也能精确落位。
* **力度**：读取 ``<sound dynamics>`` 与 ``<dynamics>``（ppp…ffff、fp、sfz…），
  并把 ``<wedge type="crescendo/diminuendo">`` 以及 cresc./dim. 文字
  展开成逐音力度曲线。
* **奏法**：staccato / staccatissimo / tenuto / accent / strong-accent /
  marcato、pizz.、tremolo、fermata、grace note、slur 等。
* **连音线**：把跨小节的 tie / tied 合并成一个持续音，避免同一个长音
  被重复切成多个字。
* **移调乐器**：读取 ``<transpose>`` 的 chromatic / octave-change，
  输出实际发声音高（例如低音提琴低八度）。

实现只依赖标准库 ``xml.etree.ElementTree`` 与 ``zipfile``，不引入
music21 之类的大依赖。
"""

from __future__ import annotations

import io
import logging
import math
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
from xml.etree import ElementTree as ET

import numpy as np
import pretty_midi

logger = logging.getLogger(__name__)

SCORE_EXTS = (".xml", ".musicxml", ".mxl")

#: MuseScore / MusicXML 的 <sound dynamics> 以「100 ≈ MIDI 90」为标尺，
#: 因此乘 0.9 后与 MuseScore 导出的 MIDI velocity 基本一致。
_DYNAMIC_SCALE = 0.9

#: 没有 <sound dynamics> 时的符号力度（已换算到 0~127 的 velocity）。
DYNAMIC_VELOCITIES: dict[str, int] = {
    "pppppp": 4,
    "ppppp": 8,
    "pppp": 12,
    "ppp": 16,
    "pp": 33,
    "p": 49,
    "mp": 64,
    "mf": 80,
    "f": 96,
    "ff": 112,
    "fff": 126,
    "ffff": 127,
    "fffff": 127,
    "ffffff": 127,
    "sf": 104,
    "sfz": 108,
    "sffz": 112,
    "fz": 104,
    "fp": 96,
    "pf": 64,
}

#: 带重音性质的力度记号。
_ACCENT_DYNAMICS = {"sf", "sfz", "sffz", "fz"}

_ARTICULATION_ALIASES = {
    "strong-accent": "marcato",
    "spiccato": "staccato",
}

_STEP_TO_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

#: 一小节没有音符时，用拍号推算小节长度（以四分音符为单位）。
def _nominal_measure_quarters(beats: float, beat_type: float) -> float:
    if beats <= 0 or beat_type <= 0:
        return 4.0
    return max(0.25, beats * 4.0 / beat_type)


# ---------------------------------------------------------------------------
# 轻量数据结构
# ---------------------------------------------------------------------------


@dataclass
class TempoEvent:
    """谱面上的一个速度标记（position 以四分音符计）。"""

    position: float
    bpm: float
    label: str = ""
    part: int = 0
    order: int = 0

    def to_dict(self, time: float) -> dict[str, Any]:
        return {
            "time": round(float(time), 4),
            "position": round(float(self.position), 4),
            "bpm": round(float(self.bpm), 4),
            "label": self.label,
        }


@dataclass
class DynamicMark:
    position: float
    velocity: float
    label: str = ""
    part: int = 0
    order: int = 0


@dataclass
class Wedge:
    start: float
    end: float
    kind: str = "crescendo"
    number: str = "1"
    part: int = 0
    order: int = 0
    start_value: float = 80.0
    end_value: float = 80.0
    label: str = ""


@dataclass
class TextEvent:
    position: float
    text: str
    part: int = 0
    order: int = 0


@dataclass
class RawNote:
    part: int
    order: int
    pitch: int
    start_q: float
    duration_q: float
    voice: str = "1"
    staff: int = 1
    tie_start: bool = False
    tie_stop: bool = False
    articulations: set[str] = field(default_factory=set)
    fermata: bool = False
    grace: bool = False
    tremolo: bool = False
    ornament: bool = False
    technique: str = ""
    dynamic: str = ""
    velocity: int = 80
    velocity_end: int = 80
    gate: float = 1.0
    #: 由 tie 合并链中的第一颗音保留；后续音只延长时值
    merged_ties: int = 0

    @property
    def end_q(self) -> float:
        return self.start_q + max(0.0, self.duration_q)


@dataclass
class ScorePart:
    index: int
    part_id: str
    name: str
    abbreviation: str
    program: int
    is_drum: bool
    notes: list[RawNote] = field(default_factory=list)
    dynamics: list[DynamicMark] = field(default_factory=list)
    wedges: list[Wedge] = field(default_factory=list)
    texts: list[TextEvent] = field(default_factory=list)
    time_signatures: list[dict[str, Any]] = field(default_factory=list)
    tempo_events: list[TempoEvent] = field(default_factory=list)
    measures: int = 0
    end_q: float = 0.0
    open_wedges: dict[str, tuple[float, str, int]] = field(default_factory=dict)
    #: 打击乐谱的 ``midi-instrument id -> MIDI 音高``（已转成 0~127）
    drum_map: dict[str, int] = field(default_factory=dict)
    #: 该声部实际读到的 <unpitched> 音符数
    unpitched_count: int = 0


@dataclass
class ScoreData:
    path: str
    fmt: str
    title: str = ""
    composer: str = ""
    parts: list[ScorePart] = field(default_factory=list)
    tempo_events: list[TempoEvent] = field(default_factory=list)
    time_signatures: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_note_count: int = 0
    tie_count: int = 0
    grace_count: int = 0
    ornament_count: int = 0
    articulation_count: int = 0
    dynamics_count: int = 0
    wedge_count: int = 0


# ---------------------------------------------------------------------------
# XML 工具
# ---------------------------------------------------------------------------


def _local(tag: Any) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _strip_namespaces(root: ET.Element) -> None:
    for elem in root.iter():
        elem.tag = _local(elem.tag)
        for key in list(elem.attrib):
            if key.startswith("{"):
                elem.attrib[_local(key)] = elem.attrib.pop(key)


def _child(elem: ET.Element | None, name: str) -> ET.Element | None:
    if elem is None:
        return None
    for item in list(elem):
        if _local(item.tag) == name:
            return item
    return None


def _children(elem: ET.Element | None, name: str) -> list[ET.Element]:
    if elem is None:
        return []
    return [item for item in list(elem) if _local(item.tag) == name]


def _descendants(elem: ET.Element | None, name: str) -> list[ET.Element]:
    if elem is None:
        return []
    return [item for item in elem.iter() if _local(item.tag) == name]


def _text(elem: ET.Element | None, name: str, default: str = "") -> str:
    child = _child(elem, name)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def _float_text(elem: ET.Element | None, name: str, default: float = 0.0) -> float:
    raw = _text(elem, name, "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _int_text(elem: ET.Element | None, name: str, default: int = 0) -> int:
    raw = _text(elem, name, "")
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def _as_number(text: str, default: float = 0.0) -> float:
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# 文件读取（.xml / .musicxml / .mxl）
# ---------------------------------------------------------------------------


def _read_mxl_bytes(raw: bytes, path: str) -> bytes:
    """从 MXL（ZIP）中按 container.xml 找到根乐谱。"""
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = zf.namelist()
            root_name = ""
            if "META-INF/container.xml" in names:
                try:
                    container = ET.fromstring(zf.read("META-INF/container.xml"))
                    for elem in container.iter():
                        if _local(elem.tag) == "rootfile":
                            root_name = elem.attrib.get("full-path", "")
                            if root_name:
                                break
                except ET.ParseError:
                    root_name = ""
            if root_name and root_name in names:
                return zf.read(root_name)
            candidates = [
                name
                for name in names
                if not name.startswith("META-INF/")
                and name.lower().endswith((".xml", ".musicxml"))
            ]
            if not candidates:
                raise ValueError("MXL 压缩包内没有 MusicXML 文件")
            # 通常根文件在顶层；优先选择最浅的路径
            candidates.sort(key=lambda n: (n.count("/"), len(n)))
            return zf.read(candidates[0])
    except zipfile.BadZipFile as exc:
        raise ValueError(f"不是有效的 MXL（ZIP）文件：{path}") from exc


def _load_xml_root(path: str) -> tuple[ET.Element, str]:
    ext = os.path.splitext(path)[1].lower()
    raw = Path(path).read_bytes()
    fmt = "musicxml"
    if ext == ".mxl" or raw[:2] == b"PK":
        raw = _read_mxl_bytes(raw, path)
        fmt = "mxl"
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError(f"MusicXML 解析失败：{exc}") from exc
    _strip_namespaces(root)
    if _local(root.tag) not in ("score-partwise", "score-timewise"):
        raise ValueError("不是 MusicXML 乐谱（根元素应为 score-partwise）")
    if _local(root.tag) == "score-timewise":
        raise ValueError("暂不支持 score-timewise 乐谱，请用 MuseScore 另存为 score-partwise")
    return root, fmt


# ---------------------------------------------------------------------------
# 速度地图
# ---------------------------------------------------------------------------


class TempoMap:
    """按四分音符位置精确换算秒数的速度地图。"""

    def __init__(self, events: Sequence[TempoEvent], default_bpm: float = 120.0) -> None:
        cleaned: list[TempoEvent] = []
        for ev in events:
            if not math.isfinite(ev.bpm) or ev.bpm <= 0:
                continue
            if not math.isfinite(ev.position) or ev.position < 0:
                continue
            cleaned.append(ev)
        cleaned.sort(key=lambda e: (e.position, e.part, e.order))

        # 同一位置只保留一个：优先第一声部（通常是总谱速度轨），
        # 同声部后出现的标记覆盖先出现的。
        by_pos: dict[float, TempoEvent] = {}
        for ev in cleaned:
            key = round(ev.position, 6)
            old = by_pos.get(key)
            if old is None:
                by_pos[key] = ev
            elif ev.part < old.part:
                by_pos[key] = ev
            elif ev.part == old.part and ev.order >= old.order:
                by_pos[key] = ev
        deduped = sorted(by_pos.values(), key=lambda e: (e.position, e.part, e.order))
        if not deduped or deduped[0].position > 1e-9:
            deduped.insert(0, TempoEvent(0.0, default_bpm, "默认速度"))

        self.events: list[TempoEvent] = deduped
        self._positions = np.array([e.position for e in deduped], dtype=np.float64)
        self._bpms = np.array([e.bpm for e in deduped], dtype=np.float64)
        self._times = np.zeros_like(self._positions)
        for i in range(1, len(deduped)):
            prev = deduped[i - 1]
            cur = deduped[i]
            self._times[i] = self._times[i - 1] + (cur.position - prev.position) * 60.0 / prev.bpm

    @property
    def initial_bpm(self) -> float:
        return float(self._bpms[0])

    def bpm_at(self, position: float) -> float:
        idx = int(np.searchsorted(self._positions, position, side="right") - 1)
        idx = max(0, min(len(self._bpms) - 1, idx))
        return float(self._bpms[idx])

    def seconds_at(self, position: float) -> float:
        if not math.isfinite(position):
            return 0.0
        if position <= self._positions[0]:
            return float(position * 60.0 / self._bpms[0])
        idx = int(np.searchsorted(self._positions, position, side="right") - 1)
        idx = max(0, min(len(self._positions) - 1, idx))
        return float(
            self._times[idx] + (position - self._positions[idx]) * 60.0 / self._bpms[idx]
        )

    def changes(self) -> list[dict[str, Any]]:
        return [ev.to_dict(self.seconds_at(ev.position)) for ev in self.events]

    def beat_times(self, end_q: float, beat_q: float = 1.0) -> np.ndarray:
        beat_q = max(0.125, float(beat_q))
        n = max(2, int(math.ceil(max(0.0, end_q) / beat_q)) + 1)
        positions = np.arange(n, dtype=np.float64) * beat_q
        return np.array([self.seconds_at(float(p)) for p in positions], dtype=np.float64)


# ---------------------------------------------------------------------------
# 解析单个部分
# ---------------------------------------------------------------------------

def _parse_time_signature(elem: ET.Element | None) -> tuple[int, int] | None:
    if elem is None:
        return None
    beats_raw = _text(elem, "beats", "4")
    # 复合拍号可能写成 3+2；取总和
    beats = 0
    for part in beats_raw.replace(" ", "").split("+"):
        try:
            beats += int(float(part))
        except (TypeError, ValueError):
            pass
    beat_type = _int_text(elem, "beat-type", 4)
    if beats <= 0 or beat_type <= 0:
        return None
    return beats, beat_type


def _parse_pitch(note: ET.Element, transpose: int) -> int | None:
    pitch_elem = _child(note, "pitch")
    if pitch_elem is None:
        return None
    step = _text(pitch_elem, "step", "C").upper()
    if step not in _STEP_TO_SEMITONE:
        return None
    alter = int(round(_float_text(pitch_elem, "alter", 0.0)))
    octave = _int_text(pitch_elem, "octave", 4)
    midi = (octave + 1) * 12 + _STEP_TO_SEMITONE[step] + alter + transpose
    return max(0, min(127, int(midi)))


def _parse_unpitched(note: ET.Element, part: "ScorePart") -> int | None:
    """读取打击乐 <unpitched>，用 <midi-unpitched> 还原 GM 鼓音高。

    MusicXML 的 ``midi-unpitched`` 是 1-based：36 表示 MIDI note 35
    （GM Acoustic Bass Drum），因此这里减 1。部分导出器只写
    ``display-step/octave`` 或省略 instrument 引用，此时回退到该声部
    唯一的鼓映射，最后兜底为 38（Acoustic Snare）。
    """
    if _child(note, "unpitched") is None:
        return None
    instrument = _child(note, "instrument")
    inst_id = str(instrument.attrib.get("id", "")) if instrument is not None else ""
    pitch = part.drum_map.get(inst_id)
    if pitch is None and part.drum_map:
        unique = sorted(set(part.drum_map.values()))
        pitch = unique[0] if len(unique) == 1 else part.drum_map.get(inst_id, 38)
    return max(0, min(127, int(pitch if pitch is not None else 38)))


def _has_type(elem: ET.Element | None, name: str, wanted: str) -> bool:
    for item in _descendants(elem, name):
        if (item.attrib.get("type") or "").lower() == wanted:
            return True
    return False


def _parse_articulations(note: ET.Element) -> tuple[set[str], bool, bool, bool]:
    arts: set[str] = set()
    notations = _child(note, "notations")
    articulation_elem = _child(notations, "articulations")
    if articulation_elem is not None:
        for item in list(articulation_elem):
            tag = _local(item.tag)
            if tag:
                arts.add(_ARTICULATION_ALIASES.get(tag, tag))
    fermata = _child(notations, "fermata") is not None
    ornaments = _descendants(notations, "ornaments")
    ornament_tags = [
        _local(item.tag)
        for group in ornaments
        for item in group.iter()
    ]
    tremolo = any(tag == "tremolo" for tag in ornament_tags)
    ornament = any(
        tag in ("trill-mark", "mordent", "inverted-mordent", "turn", "shake")
        for tag in ornament_tags
    )
    return arts, fermata, tremolo, ornament


def _direction_label(direction: ET.Element) -> str:
    for elem in direction.iter():
        tag = _local(elem.tag)
        if tag in ("words", "rehearsal"):
            text = "".join(elem.itertext()).strip()
            if text:
                return text
    for elem in _descendants(direction, "metronome"):
        beat_unit = _text(elem, "beat-unit", "quarter")
        per_minute = _text(elem, "per-minute", "")
        if per_minute:
            return f"{beat_unit} = {per_minute}"
    return ""


def _metronome_bpm(elem: ET.Element) -> float | None:
    per_minute = _float_text(elem, "per-minute", 0.0)
    if per_minute <= 0:
        return None
    unit = _text(elem, "beat-unit", "quarter").lower()
    unit_q = {
        "maxima": 32.0,
        "long": 16.0,
        "breve": 8.0,
        "whole": 4.0,
        "half": 2.0,
        "quarter": 1.0,
        "eighth": 0.5,
        "16th": 0.25,
        "32nd": 0.125,
        "64th": 0.0625,
        "128th": 0.03125,
        "256th": 0.015625,
    }.get(unit, 1.0)
    dots = 0
    for item in list(elem):
        if _local(item.tag) == "beat-unit-dot":
            dots += 1
    unit_q *= 1.5 if dots == 1 else (1.75 if dots == 2 else 1.0)
    return float(per_minute) * unit_q


def _dynamic_velocity_from_sound(raw: str) -> float | None:
    value = _as_number(raw, -1.0)
    if value <= 0:
        return None
    return max(1.0, min(127.0, value * _DYNAMIC_SCALE))


def _parse_dynamics(direction: ET.Element) -> tuple[float | None, str]:
    label = ""
    dynamics_elem = None
    for elem in _descendants(direction, "dynamics"):
        dynamics_elem = elem
        for child in list(elem):
            tag = _local(child.tag).lower()
            if tag:
                label = tag
                break
        if label:
            break
    sound_value = None
    for sound in _descendants(direction, "sound"):
        raw = sound.attrib.get("dynamics")
        if raw:
            sound_value = _dynamic_velocity_from_sound(raw)
            if sound_value is not None:
                break
    if sound_value is None and label:
        sound_value = float(DYNAMIC_VELOCITIES.get(label, 80))
    return sound_value, label


def _parse_direction(
    direction: ET.Element,
    *,
    part: ScorePart,
    position: float,
    divisions: float,
    order: int,
) -> None:
    # 速度：<sound tempo> 优先，其次 <metronome>
    tempo_value: float | None = None
    for sound in _descendants(direction, "sound"):
        raw = sound.attrib.get("tempo")
        if raw:
            value = _as_number(raw, 0.0)
            if value > 0:
                tempo_value = value
                break
    if tempo_value is None:
        for metronome in _descendants(direction, "metronome"):
            tempo_value = _metronome_bpm(metronome)
            if tempo_value:
                break
    if tempo_value and tempo_value > 0:
        part.tempo_events.append(
            TempoEvent(position, float(tempo_value), _direction_label(direction), part.index, order)
        )

    dyn_value, dyn_label = _parse_dynamics(direction)
    if dyn_value is not None:
        part.dynamics.append(
            DynamicMark(position, float(dyn_value), dyn_label, part.index, order)
        )
        # fp 是“强后立刻弱”，用一个很短的内部 wedge 表示，避免整小节一直强。
        if dyn_label in ("fp", "pf"):
            target = 49.0 if dyn_label == "fp" else 96.0
            part.wedges.append(
                Wedge(
                    position,
                    position + 0.75,
                    "fp" if dyn_label == "fp" else "pf",
                    f"fp{order}",
                    part.index,
                    order,
                    start_value=float(dyn_value),
                    end_value=target,
                    label=dyn_label,
                )
            )

    # 渐强 / 渐弱发夹
    for wedge in _descendants(direction, "wedge"):
        kind = (wedge.attrib.get("type") or "").lower()
        number = str(wedge.attrib.get("number", "1") or "1")
        if kind in ("crescendo", "diminuendo"):
            part.open_wedges[number] = (position, kind, order)
        elif kind == "stop":
            opened = part.open_wedges.pop(number, None)
            if opened is not None:
                start, start_kind, start_order = opened
                part.wedges.append(
                    Wedge(
                        start,
                        position,
                        start_kind,
                        number,
                        part.index,
                        start_order,
                        label="cresc." if start_kind == "crescendo" else "dim.",
                    )
                )

    # 文字记号：cresc. / dim. 也作为速度/力度曲线；pizz. / arco 作为奏法
    for elem in direction.iter():
        tag = _local(elem.tag)
        if tag not in ("words", "rehearsal"):
            continue
        text = "".join(elem.itertext()).strip()
        if not text:
            continue
        part.texts.append(TextEvent(position, text, part.index, order))
        low = text.lower().replace(" ", "")
        if low in ("cresc.", "cresc", "crescendo", "crescendo."):
            # 没有发夹时，补一条到下一个力度标记 / 8 拍后的渐强
            if not any(w.start <= position + 0.5 and w.start >= position - 0.5 for w in part.wedges):
                part.wedges.append(
                    Wedge(position, position + 8.0, "crescendo", f"text{order}", part.index, order, label=text)
                )
        elif low in ("dim.", "dim", "diminuendo", "diminuendo.", "decresc.", "decrescendo"):
            if not any(w.start <= position + 0.5 and w.start >= position - 0.5 for w in part.wedges):
                part.wedges.append(
                    Wedge(position, position + 8.0, "diminuendo", f"text{order}", part.index, order, label=text)
                )
        elif low in ("pizz.", "pizzicato"):
            part.open_wedges["__technique__"] = (position, "pizzicato", order)
        elif low == "arco":
            part.open_wedges["__technique__"] = (position, "arco", order)


def _parse_note(
    note: ET.Element,
    *,
    part: ScorePart,
    cursor_q: float,
    last_note_start_q: float,
    divisions: float,
    transpose: int,
    order: int,
) -> tuple[RawNote | None, float, float, str]:
    """返回 (音符, 新 cursor, 新 last_note_start, 当前奏法)。"""
    technique_info = part.open_wedges.get("__technique__")
    technique = ""
    if technique_info and technique_info[1] in ("pizzicato", "arco"):
        technique = technique_info[1]

    is_grace = _child(note, "grace") is not None
    is_chord = _child(note, "chord") is not None
    is_cue = _child(note, "cue") is not None
    if is_cue:
        return None, cursor_q, last_note_start_q, technique

    duration_div = _float_text(note, "duration", 0.0)
    duration_q = max(0.0, duration_div / max(1.0, divisions))
    if is_grace:
        start_q = cursor_q
        duration_q = 0.0
    elif is_chord:
        start_q = last_note_start_q
    else:
        start_q = cursor_q

    pitch = _parse_pitch(note, transpose)
    if pitch is None:
        pitch = _parse_unpitched(note, part)
        if pitch is not None:
            part.unpitched_count += 1
    if pitch is None:
        # 休止符 / 非音高音符：推进时间但不生成音符
        new_cursor = cursor_q if (is_chord or is_grace) else cursor_q + duration_div / max(1.0, divisions)
        new_last = last_note_start_q if (is_chord or is_grace) else start_q
        return None, new_cursor, new_last, technique

    arts, fermata, tremolo, ornament = _parse_articulations(note)
    tie_start = _has_type(note, "tie", "start") or _has_type(note, "tied", "start")
    tie_stop = _has_type(note, "tie", "stop") or _has_type(note, "tied", "stop")

    voice = _text(note, "voice", "1") or "1"
    staff = _int_text(note, "staff", 1)

    raw = RawNote(
        part=part.index,
        order=order,
        pitch=pitch,
        start_q=start_q,
        duration_q=duration_q,
        voice=voice,
        staff=staff,
        tie_start=tie_start,
        tie_stop=tie_stop,
        articulations=arts,
        fermata=fermata,
        grace=is_grace,
        tremolo=tremolo,
        ornament=ornament,
        technique=technique,
    )
    new_cursor = cursor_q if (is_chord or is_grace) else cursor_q + duration_q
    new_last = last_note_start_q if (is_chord or is_grace) else start_q
    return raw, new_cursor, new_last, technique


def _parse_part(
    part_elem: ET.Element,
    meta: dict[str, Any],
    part_index: int,
    data: ScoreData,
) -> ScorePart:
    part = ScorePart(
        index=part_index,
        part_id=str(part_elem.attrib.get("id", f"P{part_index + 1}")),
        name=str(meta.get("name") or f"Part {part_index + 1}"),
        abbreviation=str(meta.get("abbreviation") or ""),
        program=int(meta.get("program", 0)),
        is_drum=bool(meta.get("is_drum", False)),
        drum_map={
            str(key): int(value)
            for key, value in (meta.get("drum_map") or {}).items()
        },
    )
    divisions = float(meta.get("divisions", 1) or 1)
    transpose = int(meta.get("transpose", 0))
    time_sig = (4, 4)
    measure_start_q = 0.0
    order = 0
    total_measures = 0

    for measure in _children(part_elem, "measure"):
        total_measures += 1
        cursor_q = 0.0
        max_cursor_q = 0.0
        last_note_start_q = 0.0
        for elem in list(measure):
            tag = _local(elem.tag)
            if tag == "attributes":
                raw_div = _text(elem, "divisions", "")
                if raw_div:
                    try:
                        divisions = max(1.0, float(raw_div))
                    except ValueError:
                        pass
                time_elem = _child(elem, "time")
                parsed_time = _parse_time_signature(time_elem)
                if parsed_time:
                    time_sig = parsed_time
                    if not part.time_signatures or part.time_signatures[-1]["signature"] != f"{parsed_time[0]}/{parsed_time[1]}":
                        part.time_signatures.append(
                            {
                                "position": round(measure_start_q, 4),
                                "time": round(measure_start_q, 4),
                                "signature": f"{parsed_time[0]}/{parsed_time[1]}",
                            }
                        )
                tr = _child(elem, "transpose")
                if tr is not None:
                    chromatic = _int_text(tr, "chromatic", 0)
                    octave_change = _int_text(tr, "octave-change", 0)
                    transpose = chromatic + 12 * octave_change
            elif tag == "direction":
                offset_div = _float_text(elem, "offset", 0.0)
                position = max(0.0, measure_start_q + cursor_q + offset_div / max(1.0, divisions))
                order += 1
                _parse_direction(
                    elem,
                    part=part,
                    position=position,
                    divisions=divisions,
                    order=order,
                )
            elif tag == "note":
                order += 1
                raw, cursor_q, last_note_start_q, _ = _parse_note(
                    elem,
                    part=part,
                    cursor_q=cursor_q,
                    last_note_start_q=last_note_start_q,
                    divisions=divisions,
                    transpose=transpose,
                    order=order,
                )
                if raw is not None:
                    # _parse_note 使用小节内坐标；这里统一加上小节起点
                    raw.start_q += measure_start_q
                    part.notes.append(raw)
                    data.raw_note_count += 1
                    if raw.tie_start or raw.tie_stop:
                        data.tie_count += 1
                    if raw.grace:
                        data.grace_count += 1
                    if raw.ornament:
                        data.ornament_count += 1
                    if raw.articulations:
                        data.articulation_count += 1
                max_cursor_q = max(max_cursor_q, cursor_q)
            elif tag == "backup":
                cursor_q -= _float_text(elem, "duration", 0.0) / max(1.0, divisions)
            elif tag == "forward":
                cursor_q += _float_text(elem, "duration", 0.0) / max(1.0, divisions)
            elif tag == "sound":
                # 少数导出器把 <sound> 直接放在 measure 下
                raw_tempo = elem.attrib.get("tempo")
                if raw_tempo:
                    value = _as_number(raw_tempo, 0.0)
                    if value > 0:
                        order += 1
                        part.tempo_events.append(
                            TempoEvent(
                                max(0.0, measure_start_q + cursor_q),
                                value,
                                "速度",
                                part.index,
                                order,
                            )
                        )
                raw_dyn = elem.attrib.get("dynamics")
                if raw_dyn:
                    value = _dynamic_velocity_from_sound(raw_dyn)
                    if value is not None:
                        order += 1
                        part.dynamics.append(
                            DynamicMark(
                                max(0.0, measure_start_q + cursor_q),
                                value,
                                "",
                                part.index,
                                order,
                            )
                        )
        nominal = _nominal_measure_quarters(float(time_sig[0]), float(time_sig[1]))
        measure_len = max(max_cursor_q, cursor_q)
        if measure_len <= 1e-9:
            measure_len = nominal
        measure_start_q += max(0.25, measure_len)

    part.measures = total_measures
    part.end_q = measure_start_q
    if part.unpitched_count:
        part.is_drum = True
    # 未闭合的发夹：延伸到最后一个小节末
    for number, (start, kind, order0) in list(part.open_wedges.items()):
        if number == "__technique__":
            continue
        part.wedges.append(
            Wedge(
                start,
                measure_start_q,
                kind,
                number,
                part.index,
                order0,
                label="cresc." if kind == "crescendo" else "dim.",
            )
        )
    part.open_wedges.clear()
    return part


# ---------------------------------------------------------------------------
# 连音线 / 装饰音 / 力度
# ---------------------------------------------------------------------------


def _merge_ties(notes: list[RawNote]) -> list[RawNote]:
    """合并 tie / tied 链；跨小节的同音持续音只保留一颗。"""
    notes = sorted(notes, key=lambda n: (n.start_q, n.order, n.pitch))
    open_map: dict[tuple[int, str, int, int], RawNote] = {}
    out: list[RawNote] = []
    for note in notes:
        key = (note.part, note.voice, note.staff, note.pitch)
        prev = open_map.get(key)
        if note.tie_stop and prev is not None:
            prev_end = note.start_q + note.duration_q
            prev.duration_q = max(prev.duration_q, prev_end - prev.start_q)
            prev.merged_ties += 1
            # 如果连音链还有下一段，继续挂在同一颗音上
            if note.tie_start:
                open_map[key] = prev
            else:
                open_map.pop(key, None)
            continue

        out.append(note)
        if note.tie_start:
            open_map[key] = note
        else:
            open_map.pop(key, None)

    out.sort(key=lambda n: (n.start_q, n.order, n.pitch))
    return out


def _schedule_graces(notes: list[RawNote], tempo_map: TempoMap) -> None:
    """把装饰音放到主音前约 80ms，避免它与主音完全重叠。"""
    notes.sort(key=lambda n: (n.start_q, n.order))
    graces = [n for n in notes if n.grace]
    if not graces:
        return
    principals = [n for n in notes if not n.grace]
    for grace in graces:
        nxt = next(
            (
                n
                for n in principals
                if n.voice == grace.voice
                and n.start_q >= grace.start_q - 1e-6
            ),
            None,
        )
        if nxt is None:
            continue
        bpm = tempo_map.bpm_at(nxt.start_q)
        dur_q = max(0.02, 0.08 * bpm / 60.0)
        grace.start_q = max(0.0, nxt.start_q - dur_q)
        grace.duration_q = dur_q


def _mark_value_at(
    pos: float,
    marks: Sequence[DynamicMark],
    wedges: Sequence[Wedge],
    default: float = 80.0,
) -> float:
    value = default
    best = -1.0
    for mark in marks:
        if mark.position <= pos + 1e-6 and mark.position >= best:
            best = mark.position
            value = mark.velocity
    for wedge in wedges:
        if wedge.end <= pos + 1e-6 and wedge.end >= best and wedge.kind not in ("fp", "pf"):
            best = wedge.end
            value = wedge.end_value
    return value


def _resolve_wedges(
    part: ScorePart,
    default: float = 80.0,
) -> None:
    """给每根发夹确定起止力度；没有结束力度时按音乐常识外推。"""
    marks = sorted(part.dynamics, key=lambda m: (m.position, m.order))
    for wedge in part.wedges:
        if wedge.kind in ("fp", "pf"):
            continue
        wedge.start_value = _mark_value_at(wedge.start, marks, [], default)
        end_mark = next((m for m in marks if m.position >= wedge.end - 1e-6), None)
        if end_mark is not None and end_mark.position <= wedge.end + 2.0:
            wedge.end_value = end_mark.velocity
        elif wedge.kind == "crescendo":
            wedge.end_value = min(127.0, wedge.start_value + max(8.0, (127.0 - wedge.start_value) * 0.35))
        else:
            wedge.end_value = max(8.0, wedge.start_value - max(8.0, (wedge.start_value - 8.0) * 0.35))


def _velocity_at(
    pos: float,
    marks: Sequence[DynamicMark],
    wedges: Sequence[Wedge],
    default: float = 80.0,
) -> float:
    active = [
        w
        for w in wedges
        if w.start - 1e-6 <= pos <= w.end + 1e-6 and w.end > w.start + 1e-9
    ]
    if active:
        wedge = max(active, key=lambda w: (w.start, w.order))
        t = (pos - wedge.start) / max(1e-9, wedge.end - wedge.start)
        t = max(0.0, min(1.0, t))
        # smoothstep 比线性更接近真实渐强 / 渐弱
        shaped = t * t * (3.0 - 2.0 * t)
        return float(wedge.start_value + (wedge.end_value - wedge.start_value) * shaped)

    value = default
    best = -1.0
    for mark in marks:
        if mark.position <= pos + 1e-6 and mark.position >= best:
            best = mark.position
            value = mark.velocity
    for wedge in wedges:
        if wedge.end <= pos + 1e-6 and wedge.end >= best:
            best = wedge.end
            value = wedge.end_value
    return float(value)


def _dynamic_label_at(pos: float, marks: Sequence[DynamicMark]) -> str:
    label = "mf"
    for mark in sorted(marks, key=lambda m: (m.position, m.order)):
        if mark.position <= pos + 1e-6 and mark.label:
            label = mark.label
    return label


def _apply_dynamics(part: ScorePart, tempo_map: TempoMap) -> None:
    _resolve_wedges(part, 80.0)
    marks = sorted(part.dynamics, key=lambda m: (m.position, m.order))
    wedges = sorted(part.wedges, key=lambda w: (w.start, w.order))
    for note in part.notes:
        start_value = _velocity_at(note.start_q, marks, wedges, 80.0)
        end_value = _velocity_at(max(note.start_q, note.end_q - 0.01), marks, wedges, 80.0)

        boost = 0.0
        if "accent" in note.articulations:
            boost += 10.0
        if "marcato" in note.articulations:
            boost += 16.0
        if "tenuto" in note.articulations:
            boost += 2.0
        if note.technique == "pizzicato":
            boost += 2.0
        # 装饰音不抢主音
        if note.grace:
            boost -= 8.0

        note.velocity = int(np.clip(round(start_value + boost), 1, 127))
        # 重音只在音头；长音仍回到谱面力度，形成自然衰减
        note.velocity_end = int(np.clip(round(end_value), 1, 127))
        if note.fermata:
            note.gate = max(note.gate, 1.15)

        gate = 1.0
        arts = note.articulations
        if "staccatissimo" in arts:
            gate = 0.28
        elif "staccato" in arts:
            gate = 0.48
        elif "marcato" in arts:
            gate = 0.72
        elif "tenuto" in arts:
            gate = 1.0
        if note.technique == "pizzicato":
            gate = min(gate, 0.42)
        if note.fermata:
            gate = max(gate, 1.15)
        note.gate = max(0.12, min(1.6, gate))
        note.dynamic = _dynamic_label_at(note.start_q, marks)


# ---------------------------------------------------------------------------
# 构建 PrettyMIDI
# ---------------------------------------------------------------------------


class ScoreMIDI(pretty_midi.PrettyMIDI):
    """保留乐谱速度地图 / 拍点 / 元数据的 PrettyMIDI 子类。"""

    def __init__(
        self,
        *,
        tempo_map: TempoMap,
        end_time: float,
        time_signatures: list[dict[str, Any]],
        meta: dict[str, Any],
    ) -> None:
        super().__init__()
        self.score_meta = meta
        self._score_tempo_map = tempo_map
        self._score_end_time = max(0.0, float(end_time))
        self._score_bpm = tempo_map.initial_bpm
        self._score_beats = tempo_map.beat_times(max(0.0, float(meta.get("_end_q", 0.0))))
        self.score_meta["tempo_changes"] = tempo_map.changes()
        self.time_signature_changes = []
        for item in time_signatures:
            try:
                self.time_signature_changes.append(
                    pretty_midi.TimeSignature(
                        int(item["beats"]),
                        int(item["beat_type"]),
                        float(item.get("time", 0.0)),
                    )
                )
            except Exception:
                pass

    def get_end_time(self) -> float:
        return self._score_end_time

    def get_beats(self) -> np.ndarray:
        return np.asarray(self._score_beats, dtype=np.float64)

    def get_tempo_changes(self) -> tuple[np.ndarray, np.ndarray]:
        events = self._score_tempo_map.events
        times = np.array([self._score_tempo_map.seconds_at(e.position) for e in events], dtype=np.float64)
        bpms = np.array([e.bpm for e in events], dtype=np.float64)
        return times, bpms


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def _part_meta(root: ET.Element) -> tuple[list[dict[str, Any]], str, str]:
    part_list = _child(root, "part-list")
    score_parts = _children(part_list, "score-part")
    metas: list[dict[str, Any]] = []
    for idx, sp in enumerate(score_parts):
        name = _text(sp, "part-name", f"Part {idx + 1}")
        abbreviation = _text(sp, "part-abbreviation", "")
        score_inst = _child(sp, "score-instrument")
        inst_sound = _text(score_inst, "instrument-sound", "").lower()

        midi_inst = _child(sp, "midi-instrument")
        program = 0
        channel = 0
        if midi_inst is not None:
            program = _int_text(midi_inst, "midi-program", 0)
            channel = _int_text(midi_inst, "midi-channel", 0)
            if program > 0:
                program -= 1  # MusicXML 1~128 → pretty_midi 0~127
            else:
                program = 0
        # 打击乐声部可含多个 <score-instrument>，每个 <midi-instrument>
        # 用 midi-unpitched 指定 GM 鼓音高（1-based）。
        drum_map: dict[str, int] = {}
        for item in _children(sp, "midi-instrument"):
            raw_unpitched = _int_text(item, "midi-unpitched", 0)
            if raw_unpitched > 0:
                iid = str(item.attrib.get("id", "") or "")
                drum_map[iid] = max(0, min(127, raw_unpitched - 1))
        is_drum = (
            channel == 10
            or "percussion" in inst_sound
            or "drum" in name.lower()
            or bool(drum_map)
        )
        metas.append(
            {
                "id": str(sp.attrib.get("id", "")),
                "name": name or abbreviation or f"Part {idx + 1}",
                "abbreviation": abbreviation,
                "program": max(0, min(127, program)),
                "channel": channel,
                "is_drum": bool(is_drum),
                "instrument_sound": inst_sound,
                "drum_map": drum_map,
                "divisions": 1.0,
                "transpose": 0,
            }
        )
    work = _child(root, "work")
    title = _text(work, "work-title", "")
    identification = _child(root, "identification")
    composer = ""
    for creator in _children(identification, "creator"):
        if (creator.attrib.get("type") or "").lower() == "composer":
            composer = "".join(creator.itertext()).strip()
            break
    if not composer:
        composer = _text(identification, "creator", "")
    return metas, title, composer


def read_score(path: str) -> ScoreMIDI:
    """读取 MusicXML / MXL，返回带乐谱演奏信息的 ``PrettyMIDI``。"""
    root, fmt = _load_xml_root(path)
    metas, title, composer = _part_meta(root)
    data = ScoreData(path=path, fmt=fmt, title=title, composer=composer)
    if not metas:
        raise ValueError("乐谱没有 part-list，无法确定乐器轨道")

    # 按 part-list 顺序解析；没有 part-list 的 part 也兜底加入
    part_elems = _children(root, "part")
    meta_by_id = {str(m.get("id")): m for m in metas}
    for idx, elem in enumerate(part_elems):
        pid = str(elem.attrib.get("id", ""))
        meta = meta_by_id.get(pid)
        if meta is None:
            meta = {
                "id": pid,
                "name": f"Part {idx + 1}",
                "program": 0,
                "is_drum": False,
                "divisions": 1.0,
                "transpose": 0,
            }
        part = _parse_part(elem, meta, idx, data)
        data.parts.append(part)

    if not data.parts:
        raise ValueError("乐谱没有 part 元素")

    # 全局速度标记：所有声部合并，同一位置优先第一声部
    for part in data.parts:
        data.tempo_events.extend(part.tempo_events)
    data.tempo_events.sort(key=lambda e: (e.position, e.part, e.order))
    tempo_map = TempoMap(data.tempo_events)

    # 拍号：取第一声部；保留拍号变化
    if data.parts:
        data.time_signatures = list(data.parts[0].time_signatures)
    if not data.time_signatures:
        data.time_signatures = [{"position": 0.0, "time": 0.0, "signature": "4/4"}]
    first_sig = data.time_signatures[0]["signature"]
    try:
        num, den = [int(x) for x in first_sig.split("/", 1)]
    except Exception:
        num, den = 4, 4
    # 给每个 time signature 补上 beats / beat_type，方便 pretty_midi
    for item in data.time_signatures:
        try:
            n, d = [int(x) for x in str(item["signature"]).split("/", 1)]
        except Exception:
            n, d = 4, 4
        item["beats"] = n
        item["beat_type"] = d
        item["time"] = tempo_map.seconds_at(float(item.get("position", 0.0)))

    # 每声部：合并 tie、放置装饰音、计算力度、转秒
    total_end_q = 0.0
    for part in data.parts:
        part.notes = _merge_ties(part.notes)
        _schedule_graces(part.notes, tempo_map)
        _apply_dynamics(part, tempo_map)
        for note in part.notes:
            note.velocity = max(1, min(127, int(note.velocity)))
            note.velocity_end = max(1, min(127, int(note.velocity_end)))
        total_end_q = max(total_end_q, part.end_q, max((n.end_q for n in part.notes), default=0.0))
        data.dynamics_count += len(part.dynamics)
        data.wedge_count += len(part.wedges)

    end_time = tempo_map.seconds_at(total_end_q)
    if end_time <= 0:
        end_time = 1.0

    # 记录速度变化（用于分析 JSON / 网页）
    tempo_changes = tempo_map.changes()
    warnings = list(data.warnings)
    jump_count = sum(
        1
        for elem in root.iter()
        if _local(elem.tag) in ("repeat", "segno", "coda", "dacapo", "fine", "ending")
    )
    if jump_count:
        warnings.append(f"乐谱含 {jump_count} 个反复 / 跳转记号，当前按单遍播放（未展开反复）")
    if data.grace_count:
        warnings.append(f"已读取 {data.grace_count} 个装饰音（按主音前 80ms 播放）")
    if data.ornament_count:
        warnings.append(f"已标记 {data.ornament_count} 个装饰音/颤音记号（按长音奏法近似）")
    if data.wedge_count:
        warnings.append(f"已展开 {data.wedge_count} 条力度渐变（发夹 / cresc. / fp）")
    if data.tie_count:
        warnings.append(f"已合并 {data.tie_count} 个连音线标记")

    meta = {
        "format": fmt,
        "format_label": "MXL（压缩 MusicXML）" if fmt == "mxl" else "MusicXML",
        "title": title,
        "composer": composer,
        "n_parts": len(data.parts),
        "n_measures": max((p.measures for p in data.parts), default=0),
        "n_notes_raw": data.raw_note_count,
        "n_notes": sum(len(p.notes) for p in data.parts),
        "n_ties": data.tie_count,
        "n_grace": data.grace_count,
        "n_ornaments": data.ornament_count,
        "n_articulations": data.articulation_count,
        "n_dynamics": data.dynamics_count,
        "n_wedges": data.wedge_count,
        "n_tempo_changes": len(tempo_changes),
        "tempo_changes": tempo_changes,
        "time_signatures": [
            {"time": round(float(t["time"]), 4), "signature": t["signature"]}
            for t in data.time_signatures
        ],
        "warnings": warnings,
        "_end_q": total_end_q,
    }

    midi = ScoreMIDI(
        tempo_map=tempo_map,
        end_time=end_time,
        time_signatures=data.time_signatures,
        meta=meta,
    )

    for part in data.parts:
        instrument = pretty_midi.Instrument(
            program=int(part.program),
            is_drum=bool(part.is_drum),
            name=part.name,
        )
        for note in part.notes:
            start = tempo_map.seconds_at(note.start_q)
            end = tempo_map.seconds_at(note.end_q)
            if end - start < 0.03:
                end = start + 0.03
            pm_note = pretty_midi.Note(
                velocity=int(note.velocity),
                pitch=int(note.pitch),
                start=float(start),
                end=float(end),
            )
            # pretty_midi.Note 是普通类，可以安全地挂附加字段
            pm_note.velocity_end = int(note.velocity_end)
            pm_note.articulation = ",".join(sorted(note.articulations))
            pm_note.gate = float(note.gate)
            pm_note.dynamic = str(note.dynamic)
            pm_note.tremolo = bool(note.tremolo)
            pm_note.tie_merged = int(note.merged_ties)
            pm_note.technique = str(note.technique)
            instrument.notes.append(pm_note)
        midi.instruments.append(instrument)

    midi.score_meta.update(meta)
    # 让分析模块可以直接读取这些字段
    midi.score_meta["warnings"] = warnings
    midi.score_meta["duration"] = round(end_time, 4)
    midi.score_meta["initial_tempo"] = round(tempo_map.initial_bpm, 4)
    midi.score_meta["time_signature"] = first_sig
    midi.score_meta["tempo_map"] = [
        {"position": round(e.position, 4), "bpm": round(e.bpm, 4), "label": e.label}
        for e in tempo_map.events
    ]
    midi.score_meta["_end_q"] = total_end_q
    # 覆盖速度为拍点数组（四分音符拍）
    midi._score_beats = tempo_map.beat_times(total_end_q)
    return midi


__all__ = [
    "DYNAMIC_VELOCITIES",
    "SCORE_EXTS",
    "ScoreData",
    "ScoreMIDI",
    "TempoMap",
    "read_score",
]
