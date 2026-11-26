"""音频后处理：延迟、混响、滤波、声场展开、限幅。

与上游相比：

* 全部使用 ``float32``，内存占用减半；
* :func:`multitap_delay` 改为原地叠加，不再为每一级回声分配整段 ``zeros_like``
  （长曲子上这是明显的 CPU / 内存浪费）；
* 新增 :func:`apply_post_chain` 统一处理链与 :func:`limit_peak` 限幅。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve, butter, lfilter

from hachimi.config import (
    HF_DAMPING_CUTOFF,
    LOW_CUT_FREQ,
    MULTITAP_DECAY,
    MULTITAP_DELAY_MS,
    MULTITAP_TAPS,
    NORMALIZE_PEAK,
    SR,
    STEREO_SPREAD,
)


def _as_2d(y: np.ndarray) -> tuple[np.ndarray, bool]:
    """保证音频为 (N, C) float32，并返回是否原本是单声道。"""
    arr = np.asarray(y)
    mono = arr.ndim == 1
    if mono:
        arr = arr[:, None]
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return np.ascontiguousarray(arr), mono


def multitap_delay(
    y: np.ndarray,
    sr: int = SR,
    decay: float = MULTITAP_DECAY,
    delay_ms: float = MULTITAP_DELAY_MS,
    num_taps: int = MULTITAP_TAPS,
) -> np.ndarray:
    """多级延迟（原地叠加实现，避免大量临时数组）。"""
    out = np.array(y, dtype=np.float32, copy=True)
    delay_samples = int(sr * delay_ms / 1000.0)
    if delay_samples <= 0:
        return out

    n = len(out)
    for i in range(1, num_taps + 1):
        shift = delay_samples * i
        if shift >= n:
            break
        atten = decay ** i
        out[shift:] += y[:-shift] * atten
    return out


def vocal_low_cut(
    audio: np.ndarray,
    sr: int = SR,
    cutoff: float = LOW_CUT_FREQ,
    order: int = 4,
) -> np.ndarray:
    """高通滤波，切除人声低频隆隆声。"""
    nyq = sr * 0.5
    b, a = butter(order, cutoff / nyq, btype="high", analog=False)
    return lfilter(b, a, audio, axis=0).astype(np.float32, copy=False)


def stereo_spread(
    stereo_audio: np.ndarray,
    spread_amount: float = STEREO_SPREAD,
    *,
    inplace: bool = False,
) -> np.ndarray:
    """Mid/Side 声场展开。

    Args:
        stereo_audio: ``(2, N)`` 立体声（保持上游接口）。
        inplace: 在输入缓冲上直接改写，省掉一整段输出数组。
    """
    arr = np.asarray(stereo_audio)
    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValueError("stereo_spread 需要形状为 (2, N) 的立体声数组")
    L, R = arr[0], arr[1]
    mid = (L + R) * 0.5
    side = (L - R) * 0.5 * (1.0 + spread_amount)
    out = arr if inplace else np.empty_like(arr, dtype=np.float32)
    out[0] = mid + side
    out[1] = mid - side
    return out.astype(np.float32, copy=False)


def hf_damping(audio: np.ndarray, sr: int = SR, cutoff: float = HF_DAMPING_CUTOFF) -> np.ndarray:
    """高频衰减（低通滤波）。"""
    nyq = sr * 0.5
    b, a = butter(2, min(0.999, cutoff / nyq), btype="low")
    return lfilter(b, a, audio, axis=0).astype(np.float32, copy=False)


def cathedral_reverb(
    audio: np.ndarray,
    sr: int = SR,
    mix: float = 0.25,
    decay: float = 3.5,
) -> np.ndarray:
    """大教堂风格长混响（FFT 卷积）。"""
    arr, mono = _as_2d(audio)

    ir_len = int(sr * decay)
    if ir_len <= 1:
        return arr[:, 0] if mono else arr
    t = np.linspace(0, decay, ir_len)
    ir = np.exp(-t * 5)
    rng = np.random.default_rng(1234)
    ir += 0.3 * rng.standard_normal(ir_len)

    nyq = sr * 0.5
    b, a = butter(1, 5000 / nyq, btype="low")
    ir = lfilter(b, a, ir)
    ir /= np.max(np.abs(ir) + 1e-9)
    ir = ir.astype(np.float32)

    wet = np.zeros_like(arr)
    for ch in range(arr.shape[1]):
        wet[:, ch] = fftconvolve(arr[:, ch], ir, mode="full")[: len(arr)]
    out = (1.0 - mix) * arr + mix * wet
    return out[:, 0] if mono else out.astype(np.float32, copy=False)


def soft_normalize(
    audio: np.ndarray,
    target_peak: float = NORMALIZE_PEAK,
    clip: bool = True,
) -> np.ndarray:
    """温和归一化：只做衰减，不改变相对动态。"""
    peak = float(np.max(np.abs(audio))) if np.size(audio) else 0.0
    if peak < 1e-9:
        return audio
    if peak > target_peak:
        audio = audio * (target_peak / peak)
    if clip:
        audio = np.clip(audio, -1.0, 1.0)
    return audio.astype(np.float32, copy=False)


def limit_peak(
    audio: np.ndarray,
    target_peak: float = NORMALIZE_PEAK,
    *,
    inplace: bool = False,
) -> np.ndarray:
    """峰值限幅（仅衰减，保留力度动态）。"""
    peak = float(np.max(np.abs(audio))) if np.size(audio) else 0.0
    if peak > target_peak and peak > 0:
        gain = target_peak / peak
        if inplace:
            audio *= gain
            return audio
        audio = audio * gain
    return audio.astype(np.float32, copy=False)


def peak_normalize(
    audio: np.ndarray,
    target_peak: float = NORMALIZE_PEAK,
    max_boost: float = 4.0,
    *,
    inplace: bool = False,
) -> np.ndarray:
    """整体增益归一化（单一增益，不改变音符间的力度比例）。

    与 ``limit_peak`` 的区别：峰值偏低时也会提升（最多 ``max_boost`` 倍），
    避免整体音量过小；所有音符使用同一增益，因此 MIDI 力度差异被完整保留。
    """
    peak = float(np.max(np.abs(audio))) if np.size(audio) else 0.0
    if peak < 1e-9:
        return audio
    gain = target_peak / peak
    gain = min(max(gain, 1.0 / max_boost), max_boost)
    if inplace:
        audio *= gain
        return audio
    return (audio * gain).astype(np.float32, copy=False)


def normalize_rms(audio: np.ndarray, target_rms: float = 0.12) -> np.ndarray:
    """RMS 归一化（可选，会压缩力度差异，默认不启用）。"""
    rms = float(np.sqrt(np.mean(np.square(audio)))) if np.size(audio) else 0.0
    if rms > 1e-6:
        audio = audio * (target_rms / rms)
    return limit_peak(audio)


def fade_edges(audio: np.ndarray, sr: int = SR, fade_ms: float = 5.0) -> np.ndarray:
    """给整段音频加首尾淡入淡出，避免爆音。"""
    n = len(audio)
    k = min(int(sr * fade_ms / 1000.0), n // 2)
    if k <= 1:
        return audio
    audio[:k] *= np.linspace(0.0, 1.0, k, dtype=np.float32)[:, None]
    audio[-k:] *= np.linspace(1.0, 0.0, k, dtype=np.float32)[:, None]
    return audio


def apply_post_chain(
    y: np.ndarray,
    sr: int = SR,
    low_cut: float = LOW_CUT_FREQ,
    spread: float = STEREO_SPREAD,
    delay_decay: float = MULTITAP_DECAY,
    delay_ms: float = MULTITAP_DELAY_MS,
    taps: int = MULTITAP_TAPS,
    damping: float = HF_DAMPING_CUTOFF,
    normalize: str = "peak",
    target_peak: float = NORMALIZE_PEAK,
    *,
    inplace: bool = False,
) -> np.ndarray:
    """统一后处理链：高通 → 声场 → 延迟 → 高频衰减 → 限幅。

    ``inplace=True`` 时尽量复用输入缓冲（调用方拥有该数组时使用），
    可显著降低长音频的峰值内存。
    """
    y = _as_2d(y)[0]
    if low_cut:
        y = vocal_low_cut(y, sr, cutoff=low_cut)
    if spread:
        y = stereo_spread(y.T, spread_amount=spread, inplace=inplace).T
    if taps > 0:
        y = multitap_delay(y, sr, decay=delay_decay, delay_ms=delay_ms, num_taps=taps)
    if damping:
        y = hf_damping(y, sr, cutoff=damping)
    if normalize == "peak":
        y = peak_normalize(y, target_peak, inplace=inplace)
    elif normalize == "limit":
        # 仅防止削波，不做提升：适合分轨（保持各轨相对音量）
        y = limit_peak(y, target_peak, inplace=inplace)
    elif normalize == "rms":
        y = normalize_rms(y)
    else:
        y = np.clip(y, -1.0, 1.0)
    return fade_edges(y, sr)


__all__ = [
    "apply_post_chain",
    "cathedral_reverb",
    "fade_edges",
    "hf_damping",
    "limit_peak",
    "multitap_delay",
    "normalize_rms",
    "peak_normalize",
    "soft_normalize",
    "stereo_spread",
    "vocal_low_cut",
]
