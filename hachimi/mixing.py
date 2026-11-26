"""自动混音：真实乐器摆位、相位对齐、自动电平与母带响度。

设计原则
--------
* **先分轨、后摆位**：每件乐器先渲染成独立的干声，再按乐器家族放到声场里，
  而不是把所有音符直接糊成一条总线；
* **相位优先**：直达声保持左右相关（不为了“宽”而故意延时），
  先检测极性，再做等功率摆位；只有早期反射才按深度延时；
* **电平服从角色**：旋律、和声、低音、铺底、打击乐各有目标响度，
  但在用户手动音量基础上最多只修正 ±6 dB；
* **母带做动态整理**：3.4:1 总线压缩 + 自动增益补偿 + 弱段向上压缩，
  收住强音、抬起弱音，再用目标响度与前瞻限幅保证流媒体安全；
  仍保留 MIDI 力度与渐强渐弱，不做流行音乐式压扁。

摆位坐标以听众视角：左负右正，0 为中央。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from hachimi.config import (
    AUTO_MIX_MAX_ADJUST_DB,
    MASTER_CEILING_DB,
    MASTER_TARGET_DB,
    MIX_COMPRESSOR_ATTACK_MS,
    MIX_COMPRESSOR_KNEE_DB,
    MIX_COMPRESSOR_MAX_MAKEUP_DB,
    MIX_COMPRESSOR_RATIO,
    MIX_COMPRESSOR_RELEASE_MS,
    MIX_COMPRESSOR_THRESHOLD_DB,
    MIX_UPWARD_ATTACK_MS,
    MIX_UPWARD_MAX_DB,
    MIX_UPWARD_RATIO,
    MIX_UPWARD_RELEASE_MS,
    MIX_UPWARD_THRESHOLD_DB,
    SR,
    STEM_TARGET_DB,
    TRACK_LOW_CUT_FREQ,
)
from hachimi.loudness import (
    bus_compress,
    crest_factor_db,
    db_to_gain,
    gain_to_db,
    limit_peaks,
    loudness_limit,
    normalize_loudness,
    rms_db,
)

_EPS = 1e-12


# ---------------------------------------------------------------------------
# 乐器摆位
# ---------------------------------------------------------------------------


@dataclass
class TrackPlacement:
    """一件乐器的混音摆位。"""

    index: int
    name: str = ""
    instrument: str = ""
    role: str = ""
    family: str = "other"
    pan: float = 0.0                 # -1 左 / 0 中 / +1 右
    depth: float = 0.18              # 0 近 / 1 远
    width: float = 1.0               # 立体声宽度
    target_rms_db: float = -20.0     # 该角色的目标响度
    auto_gain_db: float = 0.0        # 渲染后由实际能量计算
    phase_ms: float = 8.0            # 早期反射延时（直达声保持对齐）
    reverb_send: float = 0.10
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "instrument": self.instrument,
            "role": self.role,
            "family": self.family,
            "pan": round(self.pan, 3),
            "depth": round(self.depth, 3),
            "width": round(self.width, 3),
            "target_rms_db": round(self.target_rms_db, 2),
            "auto_gain_db": round(self.auto_gain_db, 2),
            "phase_ms": round(self.phase_ms, 2),
            "reverb_send": round(self.reverb_send, 3),
            "reason": self.reason,
        }


#: (关键字, 声像, 深度, 乐器家族)。按顺序匹配，先具体后笼统。
_LAYOUT: list[tuple[str, float, float, str]] = [
    ("violin", -0.58, 0.16, "strings"),
    ("fiddle", -0.58, 0.16, "strings"),
    ("viola", -0.34, 0.18, "strings"),
    ("cello", 0.30, 0.22, "strings"),
    ("contrabass", 0.16, 0.20, "strings"),
    ("double bass", 0.16, 0.20, "strings"),
    ("string ensemble", -0.10, 0.30, "strings"),
    ("tremolo strings", -0.12, 0.32, "strings"),
    ("pizzicato", -0.20, 0.24, "strings"),
    ("harp", 0.24, 0.26, "strings"),
    ("piano", -0.14, 0.18, "keys"),
    ("harpsichord", -0.10, 0.18, "keys"),
    ("organ", 0.00, 0.28, "keys"),
    ("accordion", -0.04, 0.22, "keys"),
    ("guitar", -0.26, 0.16, "plucked"),
    ("banjo", -0.30, 0.14, "plucked"),
    ("sitar", -0.28, 0.18, "plucked"),
    ("koto", 0.20, 0.20, "plucked"),
    ("flute", 0.10, 0.18, "woodwind"),
    ("piccolo", 0.20, 0.16, "woodwind"),
    ("recorder", 0.08, 0.18, "woodwind"),
    ("oboe", 0.00, 0.16, "woodwind"),
    ("clarinet", -0.08, 0.16, "woodwind"),
    ("bassoon", 0.14, 0.20, "woodwind"),
    ("sax", 0.22, 0.16, "woodwind"),
    ("trumpet", 0.42, 0.14, "brass"),
    ("trombone", 0.34, 0.16, "brass"),
    ("french horn", 0.22, 0.22, "brass"),
    ("tuba", 0.12, 0.22, "brass"),
    ("brass", 0.30, 0.18, "brass"),
    ("choir", 0.00, 0.36, "vocal"),
    ("voice", 0.00, 0.30, "vocal"),
    ("pad", 0.00, 0.38, "synth"),
    ("synth", 0.00, 0.30, "synth"),
    ("lead", 0.00, 0.12, "synth"),
    ("bass", 0.00, 0.08, "bass"),
    ("drum", 0.00, 0.26, "drums"),
    ("percussion", 0.18, 0.30, "drums"),
]

_ROLE_DEPTH = {
    "melody": 0.12,
    "harmony": 0.26,
    "bass": 0.08,
    "pad": 0.38,
    "percussion": 0.28,
    "mixed": 0.20,
}

_ROLE_TARGET_DB = {
    "melody": -25.5,
    "harmony": -29.5,
    "bass": -27.0,
    "pad": -31.0,
    "percussion": -29.5,
    "mixed": -28.0,
}


def _layout_for(instrument: str, role: str, index: int) -> tuple[float, float, str, str]:
    text = f"{instrument} {role}".lower()
    for key, pan, depth, family in _LAYOUT:
        if key in text:
            # 低音声部无论乐器名都靠近中央：低频没有方向感，偏太多会一边倒
            if role == "bass":
                pan, depth = 0.04, min(depth, 0.10)
            # 同家族多轨时轻微错开，避免完全重叠
            offset = ((index % 3) - 1) * 0.035
            return max(-0.92, min(0.92, pan + offset)), depth, family, key
    role_depth = _ROLE_DEPTH.get(role, 0.20)
    if role == "bass":
        return 0.04, 0.08, "bass", "bass"
    return 0.0, role_depth, "other", "role"


def auto_mix_plan(
    tracks: Sequence[Mapping[str, Any]],
    *,
    enabled: bool = True,
) -> dict[int, TrackPlacement]:
    """根据乐器名 / 角色生成每轨摆位。``tracks`` 为已解析的轨道参数。"""
    plan: dict[int, TrackPlacement] = {}
    for track in tracks:
        idx = int(track.get("index", 0))
        instrument = str(track.get("instrument") or track.get("name") or "")
        role = str(track.get("role") or "mixed")
        pan, depth, family, matched = _layout_for(instrument, role, idx)
        if not enabled:
            pan, depth, family, matched = 0.0, 0.0, "off", "关闭"
        else:
            depth = max(depth, _ROLE_DEPTH.get(role, depth))
        target = _ROLE_TARGET_DB.get(role, -28.0)
        if family == "bass":
            target = -27.0
        elif role == "melody":
            target = -25.5
        reason = (
            f"{matched} 摆位：{'左' if pan < -0.05 else '右' if pan > 0.05 else '中'}"
            f" / 深度 {depth:.2f}"
            if enabled
            else "自动混音已关闭"
        )
        plan[idx] = TrackPlacement(
            index=idx,
            name=str(track.get("name") or ""),
            instrument=instrument,
            role=role,
            family=family,
            pan=pan,
            depth=depth,
            width=1.0,
            target_rms_db=target,
            phase_ms=8.0 + 22.0 * depth,
            reverb_send=0.06 + 0.26 * depth,
            reason=reason,
        )
    return plan


def instrument_auto_gain(audio: np.ndarray, target_rms_db: float) -> float:
    """按角色目标响度计算自动电平修正（最多 ±6 dB）。"""
    current = rms_db(audio)
    if current < -70.0:
        return 0.0
    return float(
        np.clip(
            (target_rms_db - current) * 0.75,
            -AUTO_MIX_MAX_ADJUST_DB,
            AUTO_MIX_MAX_ADJUST_DB,
        )
    )


# ---------------------------------------------------------------------------
# 相位 / 摆位 / 空间
# ---------------------------------------------------------------------------


def _stereo(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[1] == 1:
        arr = np.repeat(arr, 2, axis=1)
    return np.ascontiguousarray(arr[:, :2])


def channel_correlation(audio: np.ndarray) -> float:
    arr = _stereo(audio)
    if len(arr) == 0:
        return 0.0
    left = arr[:, 0].astype(np.float64)
    right = arr[:, 1].astype(np.float64)
    denom = float(np.sqrt(np.sum(left * left) * np.sum(right * right)))
    if denom < _EPS:
        return 0.0
    return float(np.sum(left * right) / denom)


def phase_align(audio: np.ndarray) -> tuple[np.ndarray, bool, float]:
    """相位对齐：检测左右相关性，反相时翻转右声道。

    Returns:
        (对齐后的音频, 是否翻转, 原始相关系数)
    """
    arr = _stereo(audio).copy()
    corr = channel_correlation(arr)
    flipped = False
    if corr < -0.12:
        arr[:, 1] *= -1.0
        flipped = True
        corr = -corr
    return arr, flipped, corr


def _lowpass(audio: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    cutoff = float(np.clip(cutoff, 300.0, sr * 0.45))
    sos = butter(2, cutoff / (sr * 0.5), btype="low", output="sos")
    return sosfilt(sos, audio, axis=0).astype(np.float32, copy=False)


def _pan_gains(pan: float) -> tuple[float, float]:
    """等功率声像；中央保持 0 dB，极端声像单边约 +3 dB。"""
    pan = float(np.clip(pan, -1.0, 1.0))
    theta = (pan + 1.0) * np.pi * 0.25
    return float(np.cos(theta) * np.sqrt(2.0)), float(np.sin(theta) * np.sqrt(2.0))


def place_track(
    audio: np.ndarray,
    placement: TrackPlacement,
    sr: int = SR,
    *,
    auto_gain_db: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """把一轨干声按摆位放到立体声声场。

    直达声：极性检查 → M/S 控宽 → 等功率声像；
    空间感：按深度做高频吸收 + 对侧早期反射。
    """
    arr, flipped, corr = phase_align(audio)
    n = len(arr)
    if n == 0:
        return arr, {"flipped": flipped, "correlation": corr}

    left = arr[:, 0]
    right = arr[:, 1]
    mid = (left + right) * 0.5
    side = (left - right) * 0.5 * float(placement.width)

    gl, gr = _pan_gains(placement.pan)
    out = np.empty((n, 2), dtype=np.float32)
    out[:, 0] = mid * gl + side
    out[:, 1] = mid * gr - side

    # 距离感：越远高频越暗
    depth = float(np.clip(placement.depth, 0.0, 1.0))
    if depth > 0.01:
        out = _lowpass(out, sr, 18500.0 - 7500.0 * depth)
        direct_gain = db_to_gain(-1.2 * depth)
        out *= direct_gain

        # 早期反射：只延迟反射声，直达声保持相位对齐
        delay = max(1, int(sr * placement.phase_ms / 1000.0))
        if delay < n:
            reflect = np.zeros_like(out)
            mono = mid[: n - delay] * (0.14 + 0.10 * depth)
            reflect[delay:, 0] = mono * gr
            reflect[delay:, 1] = mono * gl
            reflect = _lowpass(reflect, sr, 5200.0 - 1800.0 * depth)
            out += reflect
            del reflect

    gain_db = float(placement.auto_gain_db if auto_gain_db is None else auto_gain_db)
    if abs(gain_db) > 1e-4:
        out *= db_to_gain(gain_db)

    info = {
        "flipped": flipped,
        "correlation": round(corr, 4),
        "pan": round(placement.pan, 3),
        "depth": round(depth, 3),
        "auto_gain_db": round(gain_db, 2),
        "early_reflection_ms": round(placement.phase_ms, 2) if depth > 0.01 else 0.0,
    }
    return out.astype(np.float32, copy=False), info


# ---------------------------------------------------------------------------
# 母带 / 总线
# ---------------------------------------------------------------------------


def room_reverb(audio: np.ndarray, sr: int = SR, mix: float = 0.055) -> np.ndarray:
    """短房间混响（仅作为总线胶合，不掩盖乐器摆位）。"""
    arr = _stereo(audio)
    if mix <= 0 or len(arr) == 0:
        return arr
    length = int(sr * 1.15)
    t = np.arange(length, dtype=np.float64) / sr
    rng = np.random.default_rng(20240607)
    wet = np.empty_like(arr)
    for ch in range(2):
        ir = np.exp(-t * 6.5) * rng.standard_normal(length)
        ir[: int(sr * 0.006)] *= np.linspace(0.0, 1.0, int(sr * 0.006))
        ir /= np.sqrt(np.sum(ir * ir) + _EPS)
        wet[:, ch] = fftconvolve(arr[:, ch], ir.astype(np.float32), mode="full")[: len(arr)]
    out = (1.0 - mix) * arr + mix * wet
    return out.astype(np.float32, copy=False)


def master_bus(
    audio: np.ndarray,
    sr: int = SR,
    *,
    target_db: float = MASTER_TARGET_DB,
    ceiling_db: float = MASTER_CEILING_DB,
    room_mix: float = 0.055,
) -> tuple[np.ndarray, dict[str, Any]]:
    """母带：动态压缩（自动补偿 + 弱段提升）→ 目标响度 → 峰值保护。

    动态处理分两层：

    * 向下 3.4:1 的总线压缩收住强音，RMS 包络 + 14ms 启动保留字头；
    * 压缩后自动计算 makeup，把整体 RMS 补回压缩前；
    * 对 -36dB 以下的弱段做最多 +4.5dB 的向上压缩，让弱音不被埋掉。

    这样比旧版 1.8:1 的“轻胶合”动态更稳，适合流媒体回放；但仍保留
    约 9~15dB 的峰值因数，不会像流行音乐母带那样压成一条墙。
    """
    arr = _stereo(audio).copy()
    if len(arr) == 0:
        return arr, {"loudness_db": -120.0, "peak_db": -120.0}

    # 去除次低频，保留低音基频
    sos = butter(2, TRACK_LOW_CUT_FREQ / (sr * 0.5), btype="high", output="sos")
    arr = sosfilt(sos, arr, axis=0).astype(np.float32, copy=False)

    input_loudness = rms_db(arr)
    arr, comp_info = bus_compress(
        arr,
        sr,
        threshold_db=MIX_COMPRESSOR_THRESHOLD_DB,
        ratio=MIX_COMPRESSOR_RATIO,
        attack_ms=MIX_COMPRESSOR_ATTACK_MS,
        release_ms=MIX_COMPRESSOR_RELEASE_MS,
        knee_db=MIX_COMPRESSOR_KNEE_DB,
        detector="rms",
        makeup="auto",
        max_makeup_db=MIX_COMPRESSOR_MAX_MAKEUP_DB,
        upward_threshold_db=MIX_UPWARD_THRESHOLD_DB,
        upward_ratio=MIX_UPWARD_RATIO,
        upward_max_db=MIX_UPWARD_MAX_DB,
        upward_attack_ms=MIX_UPWARD_ATTACK_MS,
        upward_release_ms=MIX_UPWARD_RELEASE_MS,
    )
    after_compression = rms_db(arr)
    # 推到目标响度，再用前瞻限幅器压峰值；不会因一个瞬态把整首拉小
    arr = loudness_limit(arr, target_db=target_db, ceiling_db=ceiling_db)
    if room_mix > 0:
        arr = room_reverb(arr, sr, mix=room_mix)
    arr = limit_peaks(arr, sr, ceiling_db=ceiling_db)

    # 首尾 10ms 防爆音
    k = min(int(sr * 0.010), len(arr) // 2)
    if k > 1:
        ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)[:, None]
        arr[:k] *= ramp
        arr[-k:] *= ramp[::-1]

    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    info = {
        "loudness_db": round(rms_db(arr), 2),
        "before_loudness_db": round(input_loudness, 2),
        "after_compression_db": round(after_compression, 2),
        "peak_db": round(gain_to_db(peak), 2),
        "target_db": round(target_db, 2),
        "correlation": round(channel_correlation(arr), 4),
        "reverb_mix": round(room_mix, 3),
        "profile": "streaming-dynamic",
        "compression_ratio": comp_info["ratio"],
        "compressor_threshold_db": comp_info["threshold_db"],
        "makeup_db": comp_info["makeup_db"],
        "gain_reduction_db": comp_info["gain_reduction_db"],
        "max_gain_reduction_db": comp_info["max_gain_reduction_db"],
        "upward_gain_db": comp_info["upward_gain_db"],
        "max_upward_gain_db": comp_info["max_upward_gain_db"],
        "crest_db": round(crest_factor_db(arr), 2),
    }
    return np.clip(arr, -1.0, 1.0).astype(np.float32, copy=False), info


def stem_bus(audio: np.ndarray, sr: int = SR) -> tuple[np.ndarray, dict[str, Any]]:
    """分轨总线：只做目标响度与峰值保护，不加强势空间效果。"""
    arr = _stereo(audio).copy()
    if len(arr) == 0:
        return arr, {"loudness_db": -120.0}
    arr = normalize_loudness(arr, target_db=STEM_TARGET_DB, ceiling_db=MASTER_CEILING_DB)
    return arr, {
        "loudness_db": round(rms_db(arr), 2),
        "target_db": STEM_TARGET_DB,
        "peak_db": round(gain_to_db(float(np.max(np.abs(arr)))), 2),
        "crest_db": round(crest_factor_db(arr), 2),
    }


__all__ = [
    "TrackPlacement",
    "auto_mix_plan",
    "channel_correlation",
    "instrument_auto_gain",
    "master_bus",
    "phase_align",
    "place_track",
    "room_reverb",
    "stem_bus",
]
