"""高性能 MIDI → 人声渲染器。

核心优化
--------
1. **合成单位缓存**：把「同一个字 + 同一个音高 + 接近的时长」视为同一个合成单位。
   一首曲子里大量音符会命中同一单位（尤其是分解和弦、伴奏音型、重复动机），
   只需合成一次即可反复叠加。
2. **多进程并行**：不同单位在进程池中并行合成（Windows 下使用 spawn，
   worker 从磁盘缓存读取 WORLD 参数，避免传输大数组）。
3. **时长量化自适应**：若单位数量过多，自动放宽时长量化步长，
   保证单次渲染的合成次数有上界（默认 1400）。
4. **float32 + 精确时间轴**：整段缓冲区 float32，按采样点精确落位；
   **MIDI 力度** 通过 ``gain = (velocity/127) ** gamma`` 映射到振幅，
   每个音符独立增益，轨道增益再叠乘。
5. **原地后处理**：见 :mod:`hachimi.audio`。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from hachimi.config import (
    AP_SCALE,
    ATTACK_SOFTEN_CURVE,
    ATTACK_SOFTEN_MS,
    EXPRESSION_MAX_GAP,
    EXPRESSION_MIN_DELTA,
    FRAME_PERIOD,
    GPU_MIN_NOTES,
    LEGATO_ATTACK_MS,
    LEGATO_MAX_EXTEND,
    LEGATO_MAX_GAP,
    LEGATO_OVERLAP_SECONDS,
    PLOSIVE_ATTACK_SOFTEN_MS,
    RENDER_CACHE_BYTES,
    RENDER_DEFAULT_WORKERS,
    RENDER_MAX_UNITS,
    RENDER_MAX_WORKERS,
    RENDER_MIN_PARALLEL_UNITS,
    RENDER_NOTE_FADE_OUT,
    RENDER_NOTE_MAX_SECONDS,
    RENDER_TAIL_SECONDS,
    RENDER_UNIT_BYTES,
    SP_SCALE,
    SR,
    VELOCITY_GAMMA,
    VELOCITY_MIN,
)
from hachimi.gpu import gpu_budget_bytes, gpu_enabled, gpu_info
from hachimi.loudness import audibility_gain, expression_curve
from hachimi.synthesis import (
    choose_sample,
    cons_frames_for,
    get_preprocessed,
    is_plosive_sample,
    refresh_sample_bank,
    synthesize_note,
)

logger = logging.getLogger(__name__)

ProgressCb = Callable[[int, int, str], None]


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class RenderNote:
    pitch: int
    start: float
    end: float
    velocity: int
    track: int
    word: str
    gain: float = 1.0
    shift: int = 0
    #: 表情终点力度：同一乐句下一音的力度；用于渐强 / 渐弱
    velocity_end: int | None = None
    #: 实际使用的采样 id（变体采样，如 鸡2 / 曼1）；空 = 歌词字本身
    sample: str = ""
    #: 乐谱奏法（staccato / tenuto / accent / pizzicato ...）
    articulation: str = ""
    #: 实际发音比例：staccato 缩短、fermata 延长
    gate: float = 1.0
    #: 弦乐 tremolo：用轻微振幅调制近似
    tremolo: bool = False
    #: 连奏：本音从前一个音的尾部接过来；前一个音 / 本音的起音会更柔。
    legato_from_prev: bool = False
    #: 本音尾部延伸进下一个音的起音，形成换弓 / 换气过渡。
    legato_to_next: bool = False
    #: 与下一个音实际重叠的秒数（仅统计用）
    legato_overlap: float = 0.0
    #: 连奏延伸前的原始结束时间；校验 / 字幕 / 事件匹配仍用原始时值
    source_end: float | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def sample_id(self) -> str:
        return self.sample or self.word

    @property
    def render_pitch(self) -> int:
        return int(np.clip(self.pitch + 12 * self.shift, 21, 108))


@dataclass
class RenderStats:
    n_notes: int = 0
    n_units: int = 0
    n_unique_units: int = 0
    cache_hits: int = 0
    workers: int = 1
    elapsed: float = 0.0
    audio_seconds: float = 0.0
    realtime_factor: float = 0.0
    peak: float = 0.0
    duration_quantum: float = 0.02
    unit_bytes: int = 0
    notes_per_second: float = 0.0
    device: str = "cpu"
    gpu_memory_mb: int = 0
    #: 落位失败的音符（正常情况下应为 0）
    n_skipped: int = 0
    skipped_notes: list[dict[str, Any]] = field(default_factory=list)
    #: 合成结果近乎静音的单位（正常情况下应为 0）
    n_silent_units: int = 0
    silent_units: list[dict[str, Any]] = field(default_factory=list)
    #: 带渐强 / 渐弱表情的音符数量
    n_dynamic_notes: int = 0
    #: 触发音高可听度补偿的音符数量
    n_audibility_notes: int = 0
    #: 带 tremolo 奏法的音符数量
    n_tremolo_notes: int = 0
    #: 软起音（弱化音头）的音符数量
    n_attack_softened: int = 0
    #: 连奏连接数量（前音延伸 / 后音柔起音）
    n_legato_links: int = 0
    #: 应用了“所有辅音降低”的音符数
    n_consonant_notes: int = 0
    #: 其中属于爆破音字、起始瞬态被额外大幅削减的音符数
    n_plosive_notes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_notes": self.n_notes,
            "n_units": self.n_units,
            "n_unique_units": self.n_unique_units,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(self.cache_hits / self.n_notes, 3) if self.n_notes else 0.0,
            "unit_reuse": max(0, self.n_notes - self.n_unique_units),
            "unit_reuse_rate": round(1 - self.n_unique_units / self.n_notes, 3) if self.n_notes else 0.0,
            "workers": self.workers,
            "elapsed": round(self.elapsed, 2),
            "audio_seconds": round(self.audio_seconds, 2),
            "realtime_factor": round(self.realtime_factor, 2),
            "peak": round(self.peak, 4),
            "duration_quantum": self.duration_quantum,
            "unit_bytes": self.unit_bytes,
            "notes_per_second": round(self.notes_per_second, 1),
            "device": self.device,
            "gpu_memory_mb": self.gpu_memory_mb,
            "n_skipped": self.n_skipped,
            "skipped_notes": self.skipped_notes[:50],
            "n_silent_units": self.n_silent_units,
            "silent_units": self.silent_units[:50],
            "n_dynamic_notes": self.n_dynamic_notes,
            "n_audibility_notes": self.n_audibility_notes,
            "n_tremolo_notes": self.n_tremolo_notes,
            "n_attack_softened": self.n_attack_softened,
            "n_legato_links": self.n_legato_links,
            "n_consonant_notes": self.n_consonant_notes,
            "n_plosive_notes": self.n_plosive_notes,
        }


# ---------------------------------------------------------------------------
# 单位缓存（进程内 LRU，按字节计）
# ---------------------------------------------------------------------------


class UnitCache:
    def __init__(self, max_bytes: int = RENDER_CACHE_BYTES) -> None:
        self.max_bytes = max_bytes
        self._data: OrderedDict[tuple, tuple[np.ndarray, int]] = OrderedDict()
        self.bytes = 0
        self._lock = threading.Lock()

    def get(self, key: tuple) -> np.ndarray | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            self._data.move_to_end(key)
            return item[0]

    def put(self, key: tuple, array: np.ndarray) -> np.ndarray:
        nbytes = int(array.nbytes)
        if nbytes > self.max_bytes:
            return array
        with self._lock:
            if key in self._data:
                old = self._data.pop(key)
                self.bytes -= old[1]
            self._data[key] = (array, nbytes)
            self.bytes += nbytes
            while self.bytes > self.max_bytes and self._data:
                _, (_, size) = self._data.popitem(last=False)
                self.bytes -= size
        return array

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.bytes = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


_UNIT_CACHE = UnitCache()
#: 多任务同时渲染时串行使用 GPU，避免多个任务叠加占用显存
_GPU_LOCK = threading.Lock()


def clear_unit_cache() -> None:
    _UNIT_CACHE.clear()


# ---------------------------------------------------------------------------
# 并行 worker
# ---------------------------------------------------------------------------

_WORKER_PREPROC: dict[str, list[dict]] = {}


def _worker_preproc(word: str) -> list[dict]:
    item = _WORKER_PREPROC.get(word)
    if item is None:
        item = get_preprocessed(word)
        _WORKER_PREPROC[word] = item
    return item


def _synth_task(spec: tuple[str, int, int, float, float, int, float]) -> tuple[tuple, np.ndarray]:
    """进程池任务：合成一个单位。"""
    word, pitch, dur_samples, ap_scale, sp_scale, sr, frame_period = spec
    preproc = _worker_preproc(word)
    y = synthesize_note(
        preproc,
        pitch,
        dur_samples,
        sr=sr,
        frame_period=frame_period,
        ap_scale=ap_scale,
        sp_scale=sp_scale,
        sample_id=word,
    )
    key = (word, pitch, dur_samples)
    return key, y


# ---------------------------------------------------------------------------
# 单位规划
# ---------------------------------------------------------------------------


def _quantize_duration(dur: float, quantum: float) -> float:
    q = max(0.005, quantum)
    return max(q, round(dur / q) * q)


def note_unit_key(note: RenderNote, quantum: float = 0.02, sr: int = SR) -> tuple:
    """返回音符对应的合成单位键（与 :func:`plan_units` / 落位逻辑一致）。"""
    dur = min(max(note.duration, 0.04), RENDER_NOTE_MAX_SECONDS)
    return (note.sample_id, note.render_pitch, int(round(_quantize_duration(dur, quantum) * sr)))


def plan_units(
    notes: Sequence[RenderNote],
    quantum: float = 0.02,
    max_units: int = RENDER_MAX_UNITS,
    max_bytes: int = RENDER_UNIT_BYTES,
    sr: int = SR,
) -> tuple[list[tuple[str, int, int]], float]:
    """规划需要合成哪些单位，返回 ([(word, pitch, dur_samples)], 实际量化步长)。

    同时限制「单位数量」和「单位音频估算字节数」；长曲会自动加粗时长量化，
    避免一次把所有单位都堆在内存里。
    """
    q = max(0.005, quantum)
    while True:
        seen: dict[tuple[str, int, int], None] = {}
        estimated = 0
        for n in notes:
            dur = min(max(n.duration, 0.04), RENDER_NOTE_MAX_SECONDS)
            dq = int(round(_quantize_duration(dur, q) * sr))
            key = (n.sample_id, n.render_pitch, dq)
            if key not in seen:
                seen[key] = None
                estimated += dq * 2 * 4  # float32 立体声
        if (len(seen) <= max_units and estimated <= max_bytes) or q >= 0.5:
            return list(seen.keys()), q
        q *= 1.35


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------


def _velocity_gain(velocity: int, gamma: float) -> float:
    v = max(1, min(127, int(velocity))) / 127.0
    return max(VELOCITY_MIN, float(v) ** max(0.1, gamma))


def soft_attack_envelope(n: int, curve: float = ATTACK_SOFTEN_CURVE) -> np.ndarray:
    """软起音包络：从 0 平滑升到 1，用于弱化音头瞬态。

    采用幂曲线而不是线性淡入：前几毫秒上升较慢，能削掉“啪”的起音；
    普通音 14 ms 到达满音量，爆破音 22 ms，连奏后的下一个音 30 ms。
    所有声库共用同一包络；声库只决定采样，不改变这里的调音算法。
    """
    n = max(0, int(n))
    if n <= 1:
        return np.ones(max(0, n), dtype=np.float32)
    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)
    return np.power(t, max(0.2, float(curve))).astype(np.float32)


def _synthesize_units(
    specs: list[tuple[str, int, int]],
    *,
    workers: int,
    ap_scale: float,
    sp_scale: float,
    sr: int,
    frame_period: float,
    progress_cb: ProgressCb | None,
    cancel_cb: Callable[[], bool] | None,
    on_unit: Callable[[tuple, np.ndarray], None] | None = None,
) -> tuple[dict[tuple, np.ndarray], int, int]:
    """合成所有单位。

    ``on_unit`` 传入时，每合成/命中一个单位就立即交给调用方落位，随后丢弃
    大数组，不再把全部单位堆在内存里；返回值中的 ``units`` 为空。
    """
    units: dict[tuple, np.ndarray] = {}
    handled: set[tuple] = set()
    hits = 0

    def emit(key: tuple, y: np.ndarray) -> None:
        handled.add(key)
        if on_unit is not None:
            on_unit(key, y)
        else:
            units[key] = y

    # 先查进程内缓存
    todo: list[tuple[str, int, int]] = []
    for key in specs:
        cached = _UNIT_CACHE.get(key)
        if cached is not None:
            emit(key, cached)
            hits += 1
        else:
            todo.append(key)

    total = len(specs)
    done = len(specs) - len(todo)
    if progress_cb and done:
        progress_cb(done, total, "命中合成缓存")

    if not todo:
        return units, 1, hits

    # 串行（少量单位 / 单进程）
    if workers <= 1 or len(todo) < RENDER_MIN_PARALLEL_UNITS:
        for key in todo:
            if cancel_cb and cancel_cb():
                break
            word, pitch, dur_samples = key
            preproc = get_preprocessed(word)
            y = synthesize_note(
                preproc, pitch, dur_samples,
                sr=sr,
                frame_period=frame_period,
                ap_scale=ap_scale,
                sp_scale=sp_scale,
                # 串行与多进程必须走完全相同的辅音 / 爆破音处理；
                # 漏传 sample_id 会让小规模渲染（含大量非哈基米声库）失去音头衰减。
                sample_id=word,
            )
            emit(key, _UNIT_CACHE.put(key, y))
            done += 1
            if progress_cb:
                progress_cb(done, total, f"合成 {word}@{pitch}")
        return units, 1, hits

    # 多进程
    payload = [
        (word, pitch, dur_samples, ap_scale, sp_scale, sr, frame_period)
        for (word, pitch, dur_samples) in todo
    ]
    actual_workers = max(1, min(workers, RENDER_MAX_WORKERS, len(payload)))
    chunksize = max(1, len(payload) // (actual_workers * 4))
    logger.info("并行合成 %d 个单位（%d 个进程）", len(payload), actual_workers)

    try:
        with ProcessPoolExecutor(max_workers=actual_workers) as pool:
            pending = {pool.submit(_synth_task, item): None for item in payload}
            for fut in as_completed(pending):
                if cancel_cb and cancel_cb():
                    for f in pending:
                        f.cancel()
                    break
                # 及时移除 Future，避免它继续持有已落位单位的数组
                pending.pop(fut, None)
                key, y = fut.result()
                emit(key, _UNIT_CACHE.put(key, y))
                del y
                done += 1
                if progress_cb:
                    progress_cb(done, total, f"并行合成 {key[0]}@{key[1]}")
    except Exception as exc:  # pragma: no cover - 进程池不可用时降级
        logger.warning("多进程合成失败（%s），降级为串行", exc)
        for key in todo:
            if key in handled:
                continue
            if cancel_cb and cancel_cb():
                break
            word, pitch, dur_samples = key
            y = synthesize_note(
                get_preprocessed(word), pitch, dur_samples,
                sr=sr,
                frame_period=frame_period,
                ap_scale=ap_scale,
                sp_scale=sp_scale,
                sample_id=word,
            )
            emit(key, _UNIT_CACHE.put(key, y))
            done += 1
            if progress_cb:
                progress_cb(done, total, f"合成 {word}@{pitch}")
        actual_workers = 1

    return units, actual_workers, hits


def render_notes(
    notes: Sequence[RenderNote],
    *,
    sr: int = SR,
    frame_period: float = FRAME_PERIOD,
    velocity_gamma: float = VELOCITY_GAMMA,
    workers: int = RENDER_DEFAULT_WORKERS,
    duration_quantum: float = 0.02,
    max_units: int = RENDER_MAX_UNITS,
    ap_scale: float = AP_SCALE,
    sp_scale: float = SP_SCALE,
    tail_seconds: float = RENDER_TAIL_SECONDS,
    gpu: str = "auto",
    gpu_memory_mb: int = 2048,
    align_start: float | None = None,
    total_samples: int | None = None,
    progress_cb: ProgressCb | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, RenderStats]:
    """把音符序列渲染成立体声混音。

    Returns:
        (audio ``(N, 2) float32``, :class:`RenderStats`)
    """
    t_start = time.perf_counter()
    notes = [n for n in notes if n.duration > 0 and n.sample_id]
    if not notes:
        return np.zeros((1, 2), dtype=np.float32), RenderStats()

    # 本地音源被修改后：清除单位缓存，预处理缓存由文件签名自动失效
    if refresh_sample_bank():
        clear_unit_cache()

    notes.sort(key=lambda n: (n.start, n.end))
    workers = int(workers or 0)
    if workers <= 0:
        workers = min(RENDER_MAX_WORKERS, max(1, (os.cpu_count() or 4) - 1))

    # ---- 1. 规划 ----
    specs, quantum = plan_units(notes, quantum=duration_quantum, max_units=max_units, sr=sr)
    if progress_cb:
        progress_cb(0, len(specs), "分析合成单位")

    # 先在父进程预热 WORLD 采样缓存：既避免多进程同时写缓存冲突，
    # 也让 worker 直接读缓存（首次渲染更快）。
    words_needed = sorted({w for w, _, _ in specs})
    t_pre = time.perf_counter()
    for word in words_needed:
        get_preprocessed(word)
    if words_needed:
        logger.info("准备 %d 个采样（%.2fs）", len(words_needed), time.perf_counter() - t_pre)

    # ---- 2. 分配缓冲区 ----
    # align_start / total_samples 让分轨直接写入与混音等长的缓冲区，
    # 避免之后再 np.concatenate / np.pad 复制整段音频（长曲可省数百 MB）。
    t0 = max(0.0, align_start if align_start is not None else min(n.start for n in notes))
    t1 = max(n.end for n in notes) + max(0.0, tail_seconds)
    n_samples = (
        max(1, int(total_samples))
        if total_samples is not None
        else int(np.ceil((t1 - t0) * sr)) + 1
    )
    y = np.zeros((n_samples, 2), dtype=np.float32)

    total = len(notes)
    done = 0
    used_units = 0
    skipped: list[dict[str, Any]] = []
    silent: list[dict[str, Any]] = []
    silent_seen: set[tuple] = set()
    rms_cache: dict[tuple, float] = {}
    attack_n = max(1, int(ATTACK_SOFTEN_MS * sr / 1000.0))
    legato_attack_n = max(attack_n, int(LEGATO_ATTACK_MS * sr / 1000.0))
    plosive_attack_n = max(attack_n, int(PLOSIVE_ATTACK_SOFTEN_MS * sr / 1000.0))
    fade_out_n = max(1, int(RENDER_NOTE_FADE_OUT * sr))

    # ---- 3. 预先规划每个音符的落位参数，不持有单位音频 ----
    spec_set = set(specs)
    by_pitch: dict[tuple[str, int], list[tuple]] = defaultdict(list)
    for key in specs:
        by_pitch[(key[0], key[1])].append(key)

    prepared: list[tuple[RenderNote, tuple, int, int, float, float, float]] = []
    groups: dict[tuple, list[tuple]] = defaultdict(list)
    n_dynamic_notes = 0
    n_audibility_notes = 0
    n_tremolo_notes = 0
    n_attack_softened = 0
    n_legato_links = 0
    n_consonant_notes = 0
    n_plosive_notes = 0
    for n in notes:
        if cancel_cb and cancel_cb():
            break
        dur = min(max(n.duration, 0.04), RENDER_NOTE_MAX_SECONDS)
        dq = int(round(_quantize_duration(dur, quantum) * sr))
        key = (n.sample_id, n.render_pitch, dq)
        if key not in spec_set:
            # 量化边界兜底：在同字/同音高的单位中选最接近的时长
            candidates = by_pitch.get((n.sample_id, n.render_pitch), [])
            if not candidates:
                skipped.append(
                    {
                        "track": n.track,
                        "pitch": n.pitch,
                        "start": round(n.start, 4),
                        "end": round(n.end, 4),
                        "word": n.word,
                        "sample": n.sample_id,
                        "reason": "缺少合成单位",
                    }
                )
                done += 1
                continue
            key = min(candidates, key=lambda k: abs(k[2] - dq))

        b = max(0, int(round((n.start - t0) * sr)))
        e = min(b + int(round(n.duration * sr)), n_samples)
        d = e - b
        if d <= 0 or b >= n_samples:
            skipped.append(
                {
                    "track": n.track,
                    "pitch": n.pitch,
                    "start": round(n.start, 4),
                    "end": round(n.end, 4),
                    "word": n.word,
                    "sample": n.sample_id,
                    "reason": "时长为 0 或超出音频缓冲区",
                }
            )
            done += 1
            continue

        # 音高可听度：高音补 presence / air，低音补厚度（单位合成时也做频谱塑形）
        aud = audibility_gain(n.render_pitch)
        if abs(aud - 1.0) > 0.02:
            n_audibility_notes += 1

        v0 = int(np.clip(n.velocity, 1, 127))
        v1 = int(np.clip(n.velocity_end if n.velocity_end is not None else v0, 1, 127))
        if abs(v1 - v0) < EXPRESSION_MIN_DELTA:
            v1 = v0
        else:
            n_dynamic_notes += 1

        if n.tremolo:
            n_tremolo_notes += 1
        if d >= 4:
            n_attack_softened += 1
        if n.legato_from_prev:
            n_legato_links += 1
        if cons_frames_for(n.sample_id) > 0:
            n_consonant_notes += 1
        if is_plosive_sample(n.sample_id):
            n_plosive_notes += 1
        gain = float(n.gain) * aud
        item = (n, key, b, d, gain, v0, v1)
        prepared.append(item)
        groups[key].append(item)

    unit_estimate = sum(int(k[2]) * 2 * 4 for k in specs)
    done = total - len(prepared)

    def unit_rms(k: tuple, u: np.ndarray) -> float:
        value = rms_cache.get(k)
        if value is None:
            value = float(np.sqrt(np.mean(np.square(u.astype(np.float64))))) if u.size else 0.0
            rms_cache[k] = value
        return value

    def mark_silent(key: tuple, unit: np.ndarray) -> None:
        if unit_rms(key, unit) < 1e-6 and key not in silent_seen:
            silent_seen.add(key)
            silent.append(
                {
                    "sample": key[0],
                    "word": key[0],
                    "pitch": key[1],
                    "duration_samples": key[2],
                    "rms": round(unit_rms(key, unit), 8),
                }
            )

    def make_segment(item: tuple, unit: np.ndarray) -> np.ndarray:
        n, key, b, d, gain, v0, v1 = item
        seg = unit[:d] if len(unit) >= d else np.pad(unit, ((0, d - len(unit)), (0, 0)))
        seg = np.array(seg, dtype=np.float32, copy=True)
        # 弱化音头：普通音 14 ms；基 / 库 / 叮 / 咚 / 巴 / 波等爆破音
        # 22 ms；连奏进入音 30 ms。只作用于起音段，不改变辅音长度，
        # 也不影响后面元音的时值拉伸。
        plosive = is_plosive_sample(n.sample_id)
        if n.legato_from_prev:
            soft_n = max(legato_attack_n, plosive_attack_n if plosive else attack_n)
        elif plosive:
            soft_n = plosive_attack_n
        else:
            soft_n = attack_n
        f = min(soft_n, d // 3)
        if f > 1:
            seg[:f] *= soft_attack_envelope(f)[:, None]
        f = min(fade_out_n, d // 4)
        if f > 1:
            seg[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)[:, None]
        g0 = _velocity_gain(v0, velocity_gamma)
        if v1 != v0:
            # 按乐谱相邻力度做渐强 / 渐弱，而不是整首一个固定增益
            curve = expression_curve(d, g0, _velocity_gain(v1, velocity_gamma))
            seg *= curve[:, None]
        else:
            seg *= g0
        if n.tremolo:
            # 弦乐 tremolo：约 11Hz 的振幅调制；保留音头，不做成机械颤音
            local_t = np.arange(d, dtype=np.float32) / float(sr)
            mod = 0.64 + 0.36 * (0.5 + 0.5 * np.sin(2.0 * np.pi * 11.0 * local_t))
            seg *= mod[:, None]
        return seg * gain

    # ---- 4. 合成 / 落位 ----
    budget = gpu_budget_bytes(gpu_memory_mb)
    use_gpu = (
        len(prepared) >= GPU_MIN_NOTES
        and unit_estimate <= RENDER_UNIT_BYTES
        and n_samples * 2 * 4 <= budget * 0.6
        and gpu_enabled(gpu, gpu_memory_mb)
    )
    device_name = "cpu"
    cache_hits = 0
    actual_workers = 1
    unit_bytes_total = 0
    units: dict[tuple, np.ndarray] = {}

    if use_gpu:
        try:
            # GPU 需要一次性拿到单位；仅在单位估算较小（<= 256MB）时走此路径
            units, actual_workers, cache_hits = _synthesize_units(
                specs,
                workers=workers,
                ap_scale=ap_scale,
                sp_scale=sp_scale,
                sr=sr,
                frame_period=frame_period,
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
            )
            unit_bytes_total = sum(int(u.nbytes) for u in units.values())
            for key, unit in units.items():
                mark_silent(key, unit)
            import torch

            with _GPU_LOCK:
                dev = torch.device("cuda")
                y_gpu = torch.zeros((n_samples, 2), dtype=torch.float32, device=dev)
                chunk_notes = max(8, int((budget // 4) / (RENDER_NOTE_MAX_SECONDS * sr * 2 * 4)))
                for key, items in groups.items():
                    unit = units.get(key)
                    if unit is None:
                        continue
                    for start in range(0, len(items), chunk_notes):
                        chunk = items[start:start + chunk_notes]
                        idxs = []
                        segs = []
                        for item in chunk:
                            n, key2, b, d, gain, v0, v1 = item
                            idxs.append(np.arange(b, b + d, dtype=np.int64))
                            segs.append(make_segment(item, unit))
                        if not idxs:
                            continue
                        index = torch.from_numpy(np.concatenate(idxs)).to(dev)
                        source = torch.from_numpy(np.concatenate(segs)).to(dev)
                        y_gpu[:, 0].index_add_(0, index, source[:, 0])
                        y_gpu[:, 1].index_add_(0, index, source[:, 1])
                        done += len(chunk)
                        if progress_cb:
                            progress_cb(done, total, "GPU 混音落位")
                torch.cuda.synchronize()
                y = y_gpu.detach().cpu().numpy()
                used_units = len(prepared)
                device_name = "cuda"
                del y_gpu, index, source
                torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover - GPU 不可用时回退
            logger.warning("GPU 混音失败，回退 CPU：%s", exc)
            use_gpu = False
            device_name = "cpu"
            done = total - len(prepared)

    if not use_gpu:
        if units:
            # GPU 失败但单位已合成：直接 CPU 落位
            unit_bytes_total = 0
            for key, unit in units.items():
                unit_bytes_total += int(unit.nbytes)
                mark_silent(key, unit)
                for item in groups.get(key, []):
                    n, key2, b, d, gain, v0, v1 = item
                    y[b:b + d] += make_segment(item, unit)
                    used_units += 1
                    done += 1
            device_name = "cpu"
        else:
            def place_unit(key: tuple, unit: np.ndarray) -> None:
                nonlocal done, used_units, unit_bytes_total
                unit_bytes_total += int(unit.nbytes)
                mark_silent(key, unit)
                for item in groups.get(key, []):
                    n, key2, b, d, gain, v0, v1 = item
                    y[b:b + d] += make_segment(item, unit)
                    used_units += 1
                    done += 1

            # 关键：合成一个单位就立即落位并释放，单位总数再多也不会全部驻留内存
            units, actual_workers, cache_hits = _synthesize_units(
                specs,
                workers=workers,
                ap_scale=ap_scale,
                sp_scale=sp_scale,
                sr=sr,
                frame_period=frame_period,
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
                on_unit=place_unit,
            )
            device_name = "cpu"

    # ---- 4. 统计 ----
    elapsed = time.perf_counter() - t_start
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    stats = RenderStats(
        n_notes=used_units,
        n_units=total,
        n_unique_units=len(specs),
        cache_hits=cache_hits,
        workers=actual_workers,
        elapsed=elapsed,
        audio_seconds=len(y) / sr,
        realtime_factor=(len(y) / sr) / elapsed if elapsed > 0 else 0.0,
        peak=peak,
        duration_quantum=quantum,
        unit_bytes=unit_bytes_total,
        notes_per_second=used_units / elapsed if elapsed > 0 else 0.0,
        n_skipped=len(skipped),
        skipped_notes=skipped,
        n_silent_units=len(silent),
        silent_units=silent,
        n_dynamic_notes=n_dynamic_notes,
        n_audibility_notes=n_audibility_notes,
        n_tremolo_notes=n_tremolo_notes,
        n_attack_softened=n_attack_softened,
        n_legato_links=n_legato_links,
        n_consonant_notes=n_consonant_notes,
        n_plosive_notes=n_plosive_notes,
        device=device_name,
        gpu_memory_mb=gpu_memory_mb if device_name == "cuda" else 0,
    )
    logger.info(
        "渲染完成：%d 音符 / %d 单位（缓存命中 %d，%d 进程）耗时 %.2fs（%.1fx 实时）",
        stats.n_notes, stats.n_unique_units, stats.cache_hits, stats.workers,
        stats.elapsed, stats.realtime_factor,
    )
    return y, stats


# ---------------------------------------------------------------------------
# 从歌词结果构造渲染音符
# ---------------------------------------------------------------------------


def _transition_has_other_voice(items: Sequence[RenderNote], i: int, j: int) -> bool:
    """连奏只处理单声部相邻音；和弦 / 重叠声部保持原样。"""
    a, b = items[i], items[j]
    for k, other in enumerate(items):
        if k == i or k == j:
            continue
        # 有第三个音横跨这次换音，说明不是单声部连续音
        if other.start < b.start - 0.02 and other.end > a.end - 0.02:
            return True
    return False


def apply_legato(
    notes: Sequence[RenderNote],
    *,
    max_gap: float = LEGATO_MAX_GAP,
    overlap: float = LEGATO_OVERLAP_SECONDS,
    max_extend: float = LEGATO_MAX_EXTEND,
) -> list[RenderNote]:
    """把同一轨的连续单声部音连接起来，模拟真实乐器的连奏过渡。

    做法不是简单把两个音叠在一起，而是：
    * 前一个音的尾巴最多向后再延伸 ``max_extend`` 秒，盖住到下一个音起音
      之间的空隙；延伸量受下一个音长度限制，避免吃掉下一句；
    * 下一个音标记 ``legato_from_prev``，渲染时使用 30 ms 软起音；
    * 前一个音标记 ``legato_to_next``，其自然释放与下一音起音形成交叉过渡。

    只有间隔 <= ``max_gap`` 且中间没有第三个持续音（和弦 / 复调）时才连接，
    因此不会把和弦各声部误连成滑音。
    """
    by_track: dict[int, list[RenderNote]] = defaultdict(list)
    for note in notes:
        by_track[note.track].append(note)

    for items in by_track.values():
        items.sort(key=lambda n: (n.start, n.pitch))
        for i in range(len(items) - 1):
            a, b = items[i], items[i + 1]
            gap = b.start - a.end
            if gap < -0.03 or gap > max_gap:
                continue
            if b.start <= a.start + 0.03:
                continue
            if a.duration < 0.06 or b.duration < 0.06:
                continue
            if _transition_has_other_voice(items, i, i + 1):
                continue
            # 前音尾延伸到后音起音之后一点；不能超过后音本身，避免吞字。
            target = b.start + max(0.008, float(overlap))
            new_end = min(target, a.end + max(0.0, float(max_extend)), b.end - 0.02)
            # 必须真正盖到后音起音（至少留 5ms 重叠），否则不算连奏。
            if new_end <= a.end + 0.001 or new_end < b.start - 0.005:
                continue
            if a.source_end is None:
                a.source_end = a.end
            a.end = new_end
            a.legato_to_next = True
            a.legato_overlap = round(max(0.0, new_end - b.start), 4)
            b.legato_from_prev = True
    return list(notes)


def build_render_notes(
    events: Sequence[dict[str, Any]],
    track_options: dict[int, dict[str, Any]],
    *,
    start_time: float = 0.0,
    end_time: float | None = None,
    max_seconds: float | None = None,
    sample_set: int = 1,
) -> list[RenderNote]:
    """把歌词事件转成 :class:`RenderNote`，并套用轨道增益/八度。"""
    limit = end_time
    if max_seconds is not None:
        limit = start_time + max_seconds if limit is None else min(limit, start_time + max_seconds)

    out: list[RenderNote] = []
    for ev in events:
        start = max(float(ev["start"]), float(start_time))
        written_end = float(ev["end"])
        if written_end <= start_time:
            continue
        if limit is not None and start >= limit:
            continue
        track = int(ev["track"])
        opt = track_options.get(track, {})
        if not opt.get("selected", True):
            continue
        gain = float(opt.get("gain", 1.0))
        if gain <= 0:
            continue
        word = str(ev.get("state") or ev.get("word") or "哈")
        voice = str(ev.get("voice") or opt.get("voice") or "")
        sample = choose_sample(
            word, voice, track, int(ev["pitch"]), start, sample_set=sample_set
        )
        gate = float(ev.get("gate", 1.0) or 1.0)
        gate = max(0.12, min(1.6, gate))
        end = start + max(0.04, (written_end - start) * gate)
        out.append(
            RenderNote(
                pitch=int(ev["pitch"]),
                start=start,
                end=end,
                velocity=int(ev.get("velocity", 90)),
                track=track,
                word=word,
                gain=gain,
                shift=int(opt.get("octave", 0)),
                velocity_end=(
                    int(ev["velocity_end"]) if ev.get("velocity_end") is not None else None
                ),
                sample=sample,
                articulation=str(ev.get("articulation", "") or ""),
                gate=gate,
                tremolo=bool(ev.get("tremolo", False)),
                # 校验 / 字幕 / 事件匹配使用谱面原始结束时间；gate（断奏、
                # 装饰音等）只改变实际发声长度，不能把音符误判为“未渲染”。
                source_end=written_end,
            )
        )
    notes = apply_velocity_trajectory(out)
    return apply_legato(notes)


def apply_velocity_trajectory(notes: Sequence[RenderNote]) -> list[RenderNote]:
    """用同一轨相邻音的力度补出「渐强 / 渐弱」终点。

    只有相邻音落在同一乐句（间隔 <= ``EXPRESSION_MAX_GAP``）时才连线；
    这样长音会根据下一句的力度自然渐强或渐弱，而不是每个音一个死音量。
    """
    by_track: dict[int, list[RenderNote]] = defaultdict(list)
    for note in notes:
        by_track[note.track].append(note)

    for items in by_track.values():
        items.sort(key=lambda n: (n.start, n.pitch))
        j = 0
        for i, note in enumerate(items):
            if note.velocity_end is not None:
                continue
            # start 有序，用单调指针找「之后第一个不同起音」，避免 O(n²)
            if j <= i:
                j = i + 1
            while j < len(items) and items[j].start <= note.start + 0.02:
                j += 1
            if j >= len(items):
                continue
            nxt = items[j]
            gap = nxt.start - note.end
            if -0.05 <= gap <= EXPRESSION_MAX_GAP:
                note.velocity_end = int(nxt.velocity)
    return list(notes)


__all__ = [
    "RenderNote",
    "RenderStats",
    "UnitCache",
    "apply_legato",
    "apply_velocity_trajectory",
    "build_render_notes",
    "clear_unit_cache",
    "note_unit_key",
    "plan_units",
    "render_notes",
    "soft_attack_envelope",
]
