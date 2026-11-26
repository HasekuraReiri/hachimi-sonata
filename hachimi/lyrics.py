"""歌词生成：和弦一致、动机复用、标准句法 + 马尔科夫混合。

与上游版本的差异
----------------
* 以上游按轨道独立马尔科夫为基础，增加：
  - 同一个分解和弦（含跨声部琶音）内的所有音符 → 同一个字；
  - 同一个动机（移调不变的音高/节奏签名）的多次出现 → 完全相同的歌词；
  - 主题动机、长音强制使用「哈基米 / 叮咚鸡 / 曼巴」标准句法；
  - 普通旋律按每段的 ``markov_ratio`` 在标准句法与马尔科夫链之间混合；
  - 每段有独立的随机种子与音源配比（哈基米主导，叮咚鸡辅助）；
  - 语音模式 ``auto`` 以哈基米为主旋律，叮咚鸡 / 曼巴作为对位与伴奏辅助。
"""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import pretty_midi

from hachimi.analysis import (
    AnalysisResult,
    LyricSlot,
    build_lyric_slots,
    detect_chords,
)
from hachimi.arrangement import (
    DEFAULT_MARKOV_RATIO,
    DEFAULT_SOURCE_WEIGHTS,
    SOURCE_KEYS,
    build_section_plans,
    normalize_source_weights,
    plan_by_section,
    section_index_at,
)
from hachimi.base import LyricsModel
from hachimi.config import (
    SIMULTANEOUS_THRESHOLD,
    START_TIME,
    SYNTAX_LONG_NOTE_SECONDS,
    SYNTAX_MAX_PHRASE_SLOTS,
    SYNTAX_PHRASE_GAP,
    SYNTAX_WEIGHT,
    VOICE_MODES,
)
from hachimi.syntax import generate_phrase

logger = logging.getLogger(__name__)


@dataclass
class LyricsResult:
    slots: list[LyricSlot]
    events: list[dict[str, Any]]
    note2word: dict[int, str]
    voices: dict[int, str]
    voice_counts: dict[str, int]
    motif_reuse: dict[str, int]
    stats: dict[str, Any] = field(default_factory=dict)

    def track_events(self, track: int) -> list[dict[str, Any]]:
        return [e for e in self.events if e["track"] == track]


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _voice_from_mode(mode: str | None) -> str:
    """把模式名解析成实际采样类别。"""
    if mode in ("auto", "", None):
        return "auto"
    if mode in VOICE_MODES:
        return mode
    return "auto"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def _segment_phrases(slots: list[LyricSlot]) -> list[list[LyricSlot]]:
    """按段落 / 停顿 / 最大长度把槽位切成“乐句”。"""
    phrases: list[list[LyricSlot]] = []
    current: list[LyricSlot] = []
    for slot in slots:
        if current:
            gap = slot.start - current[-1].end
            if (
                slot.section_index != current[-1].section_index
                or gap > SYNTAX_PHRASE_GAP
                or len(current) >= SYNTAX_MAX_PHRASE_SLOTS
            ):
                phrases.append(current)
                current = []
        current.append(slot)
    if current:
        phrases.append(current)
    return phrases


def compose_lyrics(
    analysis: AnalysisResult,
    models: dict[str, LyricsModel],
    track_options: dict[int, dict[str, Any]],
    *,
    seed: int = 1337,
    default_voice: str = "auto",
    syntax_weight: float = SYNTAX_WEIGHT,
    section_plans: list[dict[str, Any]] | None = None,
    decomposed_merge: bool = True,
    onset_tol: float = 0.06,
    merge_gap: float = 0.4,
    progress_cb=None,
) -> LyricsResult:
    """为核心分析结果生成歌词。

    Args:
        analysis: :func:`hachimi.analysis.analyze` 的结果。
        models: ``{语音类别: LyricsModel}``，只需包含实际会用到的类别。
        track_options: ``{track_index: {selected, voice, ...}}``。
        seed: 全局随机种子（保证同一输入可复现）。
        default_voice: 轨道未指定时的语音模式。
        syntax_weight: 全局句法权重，作为没有手动段落计划时的基线。
        section_plans: 每段编曲计划。每项可包含
            ``section_index / markov_ratio / seed / source_weights``；
            ``auto=False`` 的手动值优先。为 ``None`` 时按分析结果自动生成。
        decomposed_merge: 是否把分解和弦合并为同一个歌词槽。

    Returns:
        :class:`LyricsResult`
    """
    random.seed(seed)
    syntax_weight = max(0.0, min(1.0, float(syntax_weight)))

    # 段落计划必须在分配语音 / 生成乐句之前准备好：它同时控制每段的
    # 马尔科夫比例、随机种子和音源配比。
    plans = build_section_plans(
        analysis.sections,
        base_markov=1.0 - syntax_weight,
        base_seed=seed,
        duration=analysis.duration,
        overrides=section_plans,
    )
    plans_map = plan_by_section(plans)

    # ---- 1. 过滤参与轨道 ----
    selected = [idx for idx, opt in track_options.items() if opt.get("selected", True)]
    selected_set = set(selected) if selected else {
        t.index for t in analysis.tracks if t.suggested_selected
    }
    notes = [n for n in analysis.notes if n.track in selected_set]
    if not notes:
        notes = analysis.notes
    if not notes:
        raise ValueError("没有可参与生成的音符，请至少勾选一个轨道。")

    # ---- 2. 和弦窗口 + 歌词槽 ----
    chords = detect_chords(notes, onset_tol=onset_tol, merge_gap=merge_gap)
    slots = build_lyric_slots(
        notes,
        chords,
        onset_tol=onset_tol,
        decomposed_merge=decomposed_merge,
    )
    if not slots:
        raise ValueError("未能构建歌词槽。")

    note_by_id = {n.id: n for n in notes}
    note_track = {n.id: n.track for n in notes}

    for slot in slots:
        sec_idx = section_index_at(analysis.sections, slot.start) if analysis.sections else 0
        slot.section_index = sec_idx
        slot.section = analysis.sections[sec_idx].label if analysis.sections else "A"

    # ---- 3. 每个槽的语音 ----
    track_voice_mode = {
        idx: _voice_from_mode(track_options.get(idx, {}).get("voice", default_voice))
        for idx in selected_set
    }
    roles = {t.index: t.role for t in analysis.tracks}
    track_pitch = {t.index: (t.median_pitch or 0.0) for t in analysis.tracks}

    # 每个槽指定一个主声部：旋律优先，其次音区较高者，最后轨道号
    for slot in slots:
        tids = [t for t in slot.track_ids if roles.get(t) != "percussion"]
        if not tids:
            tids = list(slot.track_ids)
        slot.primary_track = min(
            tids,
            key=lambda t: (roles.get(t) != "melody", -track_pitch.get(t, 0.0), t),
        ) if tids else -1

    voices: dict[int, str] = {}
    available_sources = [k for k in SOURCE_KEYS if k in models]
    if "哈基米" not in available_sources:
        available_sources = ["哈基米"] + available_sources
    section_rng: dict[int, random.Random] = {}
    section_weights: dict[int, dict[str, float]] = {}
    for sec_idx, plan in plans_map.items():
        raw_weights = normalize_source_weights(
            plan.get("source_weights"), fallback=DEFAULT_SOURCE_WEIGHTS
        )
        filtered = {k: (raw_weights.get(k, 0.0) if k in available_sources else 0.0) for k in SOURCE_KEYS}
        section_weights[sec_idx] = normalize_source_weights(
            filtered, fallback=DEFAULT_SOURCE_WEIGHTS
        )
        section_rng[sec_idx] = random.Random(int(plan.get("seed", seed)) ^ 0xA11CE)

    def _auto_voice_for_slot(slot: LyricSlot) -> str:
        tracks = {note_track[i] for i in slot.note_ids if i in note_track}
        if any(roles.get(t) == "melody" for t in tracks):
            # 主旋律永远哈基米：保证「哈基米绝对主导」。
            return "哈基米"
        sec_idx = slot.section_index if slot.section_index >= 0 else 0
        weights = section_weights.get(sec_idx) or normalize_source_weights(
            DEFAULT_SOURCE_WEIGHTS
        )
        rng = section_rng.get(sec_idx) or random.Random(seed ^ 0xA11CE)
        choices = [k for k in SOURCE_KEYS if weights.get(k, 0.0) > 0 and k in models]
        if not choices:
            return "哈基米"
        # 按段落配比逐槽加权抽取：哈基米多数、叮咚鸡为主辅助，
        # 曼巴/曼波少量出现；手动配比也按同一方式生效。
        return rng.choices(choices, weights=[weights[k] for k in choices], k=1)[0]

    for slot in slots:
        tracks = sorted({note_track[i] for i in slot.note_ids if i in note_track})
        mode = None
        for tr in sorted(tracks, key=lambda x: (roles.get(x) != "melody", x)):
            m = track_voice_mode.get(tr, default_voice)
            if m != "auto":
                mode = m
                break
        if mode is None:
            mode = default_voice
        voices[slot.index] = (
            _auto_voice_for_slot(slot) if mode == "auto" else mode
        )

    # ---- 4. 按「语音流」+「乐句」生成歌词（标准句法 / 马尔科夫混合） ----
    rng = random.Random(seed ^ 0x5EED)
    contexts: dict[str, str] = defaultdict(str)
    words: list[str] = [""] * len(slots)
    processed = 0
    syntax_phrases = 0
    markov_phrases = 0
    section_phrase_seq: dict[int, int] = defaultdict(int)
    section_acc: dict[int, dict[str, Any]] = {}
    for sec_idx, plan in plans_map.items():
        section_acc[sec_idx] = {
            "section_index": sec_idx,
            "label": plan.get("label", ""),
            "key": plan.get("key", ""),
            "start": plan.get("start", 0.0),
            "end": plan.get("end", 0.0),
            "markov_ratio": float(plan.get("markov_ratio", DEFAULT_MARKOV_RATIO)),
            "seed": int(plan.get("seed", seed)),
            "source_weights": dict(plan.get("source_weights", {})),
            "phrases": 0,
            "syntax_phrases": 0,
            "markov_phrases": 0,
            "slots": 0,
            "voices": Counter(),
        }

    # 每个「主声部 + 语音」单独成流：主旋律的歌词不会和伴奏的字交错
    streams: dict[tuple[int, str], list[LyricSlot]] = defaultdict(list)
    for slot in slots:
        streams[(slot.primary_track, voices[slot.index])].append(slot)
    stream_positions: dict[tuple[int, str], dict[int, int]] = {
        key: {slot.index: i for i, slot in enumerate(items)}
        for key, items in streams.items()
    }

    for key in sorted(streams):
        track, voice = key
        for phrase in _segment_phrases(streams[key]):
            sec_idx = phrase[0].section_index if phrase[0].section_index >= 0 else 0
            plan = plans_map.get(sec_idx) or {
                "markov_ratio": DEFAULT_MARKOV_RATIO,
                "seed": seed,
            }
            phrase_number = section_phrase_seq[sec_idx]
            section_phrase_seq[sec_idx] += 1
            phrase_seed = int(plan.get("seed", seed)) + phrase_number * 104729
            phrase_rng = random.Random(phrase_seed ^ 0x5EED)
            markov_ratio = max(0.0, min(1.0, float(plan.get("markov_ratio", DEFAULT_MARKOV_RATIO))))
            syntax_ratio = 1.0 - markov_ratio
            acc = section_acc.setdefault(
                sec_idx,
                {
                    "section_index": sec_idx,
                    "label": plan.get("label", ""),
                    "markov_ratio": markov_ratio,
                    "seed": int(plan.get("seed", seed)),
                    "phrases": 0,
                    "syntax_phrases": 0,
                    "markov_phrases": 0,
                    "slots": 0,
                    "voices": Counter(),
                },
            )
            acc["phrases"] += 1
            acc["slots"] += len(phrase)
            for slot in phrase:
                acc["voices"][voice] += 1

            # 主题长旋律（哈基米主旋律中的长音）无论权重如何都使用标准句法
            forced = voice == "哈基米" and any(
                s.duration >= SYNTAX_LONG_NOTE_SECONDS for s in phrase
            )
            use_syntax = forced or phrase_rng.random() < syntax_ratio
            if use_syntax:
                syntax_phrases += 1
                acc["syntax_phrases"] += 1
                text = generate_phrase(voice, len(phrase), phrase_rng)
                for slot, ch in zip(phrase, text):
                    slot.word = ch
                    words[slot.index] = ch
                contexts[voice] = (contexts[voice] + text)[-64:]
            else:
                markov_phrases += 1
                acc["markov_phrases"] += 1
                model = models.get(voice) or next(iter(models.values()))
                # 每段/每句使用自己的种子，让「每段随机数」真正可控。
                random.seed(phrase_seed ^ 0x9E3779B9)
                for slot in phrase:
                    ctx = contexts[voice][-16:]
                    try:
                        word = model.sample(ctx) or "哈"
                    except Exception as exc:  # pragma: no cover
                        logger.warning("歌词采样失败（%s）：%s", voice, exc)
                        word = "哈"
                    slot.word = word
                    words[slot.index] = word
                    contexts[voice] = (contexts[voice] + word)[-64:]
            processed += len(phrase)
            if progress_cb:
                progress_cb(processed, len(slots))

    # ---- 5. 动机复用：同一动机 → 同一串歌词 ----
    # 以「音符 → 槽」映射为基础，直接按音符序列比较与改写。
    # 这样即使某个动机音与伴奏共用一个和弦槽，也能保证各次出现的
    # 歌词逐音一致；只有产生冲突（同一槽被要求不同字）时才跳过该动机。
    note_slot: dict[int, int] = {}
    for slot in slots:
        for nid in slot.note_ids:
            note_slot[nid] = slot.index

    note2word: dict[int, str] = {}
    for slot in slots:
        for nid in slot.note_ids:
            if nid in note_by_id:
                note2word[nid] = slot.word

    def set_slot_word(si: int, w: str) -> None:
        slots[si].word = w
        words[si] = w
        for nid in slots[si].note_ids:
            if nid in note_by_id:
                note2word[nid] = w

    reuse_count: dict[str, int] = defaultdict(int)
    locked: set[int] = set()
    candidates = []
    for motif in analysis.motifs:
        occs = [o for o in motif.occurrences if o.get("ids")]
        if len(occs) < 2:
            continue
        if len({len(o["ids"]) for o in occs}) != 1:
            continue
        candidates.append((motif, occs))
    # 优先保证「旋律声部 + 长且出现次数多」的动机
    def _motif_rank(item) -> tuple:
        motif, occs = item
        is_melody = any(roles.get(o.get("track", -1)) == "melody" for o in occs)
        return (0 if is_melody else 1, -motif.score, -len(occs[0]["ids"]), motif.id)

    candidates.sort(key=_motif_rank)

    for motif, occs in candidates:
        base_slots = [note_slot.get(int(nid)) for nid in occs[0]["ids"]]
        if any(si is None for si in base_slots):
            continue
        if len(set(base_slots)) != len(base_slots):
            continue
        voice = voices[base_slots[0]]  # type: ignore[index]
        track = slots[base_slots[0]].primary_track  # type: ignore[index]

        # 动机必须在同一主声部流里占据连续槽位，否则插入句法歌词会把
        # 主旋律的其他字挤乱；这种重叠动机直接跳过。
        voice_pos = stream_positions.get((track, voice), {})
        occurrence_slots: list[list[int]] = []
        contiguous = True
        for occ in occs:
            slots_occ = [note_slot.get(int(nid)) for nid in occ["ids"]]
            if any(si is None for si in slots_occ) or len(set(slots_occ)) != len(slots_occ):
                contiguous = False
                break
            if any(
                slots[int(si)].primary_track != track or voices[int(si)] != voice
                for si in slots_occ
            ):
                contiguous = False
                break
            positions = [voice_pos.get(si, -1) for si in slots_occ]  # type: ignore[arg-type]
            if any(p < 0 for p in positions) or positions != list(range(positions[0], positions[0] + len(positions))):
                contiguous = False
                break
            occurrence_slots.append([int(si) for si in slots_occ])  # type: ignore[arg-type]
        if not contiguous:
            continue

        # 主题动机强制使用标准句法，并复制到所有出现位置
        phrase_text = generate_phrase(voice, len(base_slots), rng)

        desired: dict[int, str] = {}
        conflict = False
        for slots_occ in occurrence_slots:
            for si, ch in zip(slots_occ, phrase_text):
                old = desired.get(si)
                if old is not None and old != ch:
                    conflict = True
                    break
                desired[si] = ch
            if conflict:
                break
        if conflict or not desired:
            continue
        if set(desired) & locked:
            continue
        for si, ch in desired.items():
            set_slot_word(si, ch)
        locked |= set(desired)
        reuse_count[motif.id] = len(occs) - 1

    # ---- 6. 输出映射 ----

    events: list[dict[str, Any]] = []
    for slot in slots:
        for nid in slot.note_ids:
            n = note_by_id.get(nid)
            if n is None:
                continue
            events.append(
                {
                    "start": round(n.start, 4),
                    "end": round(n.end, 4),
                    "state": slot.word,
                    "word": slot.word,
                    "track": n.track,
                    "pitch": n.pitch,
                    "velocity": n.velocity,
                    "velocity_end": n.velocity_end,
                    "articulation": n.articulation,
                    "gate": n.gate,
                    "tremolo": n.tremolo,
                    "slot": slot.index,
                    "chord": slot.chord_label,
                    "section": slot.section,
                    "voice": voices[slot.index],
                    "decomposed": slot.is_decomposed,
                }
            )
    events.sort(key=lambda e: (e["start"], e["track"], e["pitch"]))

    voice_counts = Counter(voices[i] for i in range(len(slots)))
    section_stats = []
    for sec_idx in sorted(section_acc):
        acc = section_acc[sec_idx]
        acc["voices"] = {k: int(v) for k, v in acc["voices"].items()}
        section_stats.append(acc)
    effective_markov = (
        sum(float(p.get("markov_ratio", DEFAULT_MARKOV_RATIO)) for p in plans_map.values()) / max(1, len(plans_map))
        if plans_map
        else DEFAULT_MARKOV_RATIO
    )
    stats = {
        "n_slots": len(slots),
        "n_events": len(events),
        "n_decomposed_slots": sum(1 for s in slots if s.is_decomposed),
        "n_motifs": len(reuse_count),
        "motif_reused_slots": int(sum(reuse_count.values())),
        "motif_reuse": {k: int(v) for k, v in reuse_count.items()},
        "syntax_weight": round(1.0 - effective_markov, 3),
        "markov_ratio": round(effective_markov, 3),
        "syntax_phrases": syntax_phrases,
        "markov_phrases": markov_phrases,
        "voices": {k: int(v) for k, v in voice_counts.items()},
        "unique_words": len(set(words)),
        "section_plans": plans,
        "section_stats": section_stats,
    }

    return LyricsResult(
        slots=slots,
        events=events,
        note2word=note2word,
        voices=voices,
        voice_counts=dict(voice_counts),
        motif_reuse=dict(reuse_count),
        stats=stats,
    )


# ---------------------------------------------------------------------------
# 事件 / 兼容接口
# ---------------------------------------------------------------------------


def build_events(note2word) -> list[dict[str, Any]]:
    """兼容旧接口：把 {note → word} 映射转为事件列表。"""
    events = []
    for track_idx, track_map in enumerate(note2word):
        for note, word in track_map.items():
            events.append(
                {
                    "start": float(note.start),
                    "end": float(note.end),
                    "state": word,
                    "word": word,
                    "track": track_idx,
                    "pitch": int(note.pitch),
                    "velocity": int(getattr(note, "velocity", 90)),
                    "velocity_end": getattr(note, "velocity_end", None),
                    "articulation": str(getattr(note, "articulation", "") or ""),
                    "gate": float(getattr(note, "gate", 1.0) or 1.0),
                    "tremolo": bool(getattr(note, "tremolo", False)),
                }
            )
    events.sort(key=lambda e: (e["start"], e["end"]))
    return events


def generate_lyrics(
    midi: pretty_midi.PrettyMIDI,
    models: list[LyricsModel | None],
    start_time: float = START_TIME,
    end_time: float | None = None,
) -> tuple[list[list[str]], list[dict]]:
    """上游兼容接口：按轨道独立生成歌词（不含和弦/动机分析）。

    新代码请使用 :func:`compose_lyrics`。
    """
    from tqdm import tqdm

    note2word: list[dict] = []
    track_words_all: list[list[str]] = []
    total = sum(len(i.notes) for i in midi.instruments)
    pbar = tqdm(total=total, desc="编写歌词中", disable=None)

    for idx, inst in enumerate(midi.instruments):
        notes = sorted(inst.notes, key=lambda n: (n.start, n.end))
        if not notes:
            note2word.append({})
            track_words_all.append([])
            continue
        groups: list[list] = []
        cur = [notes[0]]
        cur_start = notes[0].start
        for note in notes[1:]:
            if note.start - cur_start <= SIMULTANEOUS_THRESHOLD:
                cur.append(note)
            else:
                groups.append(cur)
                cur = [note]
                cur_start = note.start
        groups.append(cur)

        model = models[idx] if idx < len(models) else models[0]
        if not model:
            note2word.append({})
            track_words_all.append([])
            continue

        context = ""
        mapping = {}
        words = []
        for group in groups:
            word = model.sample(context)
            words.append(word)
            mapping[max(group, key=lambda n: n.pitch)] = word
            context = (context + word)[-256:]
            pbar.update(len(group))
        note2word.append(mapping)
        track_words_all.append(words)

    pbar.close()
    return track_words_all, note2word


def make_track_srt(*args, **kwargs):
    """兼容旧接口，实际实现位于 :mod:`hachimi.subtitles`。"""
    from hachimi.subtitles import make_track_srt as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "LyricsResult",
    "build_events",
    "compose_lyrics",
    "generate_lyrics",
    "make_track_srt",
]
