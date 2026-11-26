"""响度、压缩、齿音与音高可听度工具。

本模块解决四个问题：

1. **音源先稳定**：WORLD 分析前，把每个采样做「压缩 + 补偿 + 目标响度」，
   让不同字、不同音源的底子响度一致，模拟真实乐器稳定发声；
2. **齿音 / 清辅音控制**：逐帧分析 4.5~10kHz 能量占比与自相关周期度，
   压低噪声性齿音与无周期清辅音，同时不误伤元音；
3. **乐谱力度再塑形**：合成单位只负责音色，实际音量由 MIDI velocity
   与相邻音的力度走向（渐强 / 渐弱）在落位时决定；
4. **极端音高可听**：根据音高与频段能量，做动态高频/低频补偿，
   避免很高的音只剩气声、很低的音被低频掩蔽。

所有函数都只依赖 numpy / scipy，不引入额外依赖。
"""

from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import minimum_filter1d
from scipy.signal import butter, lfilter, sosfilt, sosfiltfilt

from hachimi.config import (
    CONSONANT_ATTENUATION_DB,
    PLOSIVE_ATTENUATION_DB,
    PLOSIVE_BURST_MS,
    PLOSIVE_RAMP_MS,
    SIBILANCE_MAX_REDUCTION_DB,
    SIBILANCE_PERIODICITY_THRESHOLD,
    SIBILANCE_RATIO_THRESHOLD,
    SIBILANCE_UNIT_REDUCTION_DB,
    SR,
    VOICELESS_MAX_REDUCTION_DB,
)

_EPS = 1e-12


# ---------------------------------------------------------------------------
# 基础换算
# ---------------------------------------------------------------------------


def db_to_gain(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def gain_to_db(gain: float) -> float:
    return 20.0 * math.log10(max(_EPS, float(gain)))


def midi_to_hz(pitch: float) -> float:
    return 440.0 * 2.0 ** ((float(pitch) - 69.0) / 12.0)


def rms(audio: np.ndarray) -> float:
    arr = np.asarray(audio, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr))))


def rms_db(audio: np.ndarray) -> float:
    return gain_to_db(rms(audio) + _EPS)


def peak_db(audio: np.ndarray) -> float:
    arr = np.asarray(audio)
    if arr.size == 0:
        return -120.0
    return gain_to_db(float(np.max(np.abs(arr))))


def crest_factor_db(audio: np.ndarray) -> float:
    """峰值因数（峰值 dB - RMS dB），用来判断混音是否被压得过扁。"""
    peak = peak_db(audio)
    level = rms_db(audio)
    if peak < -100.0 or level < -100.0:
        return 0.0
    return float(peak - level)


def _as_stereo(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# 压缩 / 采样稳定化
# ---------------------------------------------------------------------------


def _block_envelope(audio: np.ndarray, frame: int) -> np.ndarray:
    """按块取峰值包络（多声道联动，避免声像被压缩拉偏）。"""
    arr = _as_stereo(audio)
    n = len(arr)
    if n == 0:
        return np.zeros(1, dtype=np.float64)
    blocks = int(math.ceil(n / frame))
    padded = np.pad(arr, ((0, blocks * frame - n), (0, 0)))
    view = padded.reshape(blocks, frame, arr.shape[1])
    return np.max(np.abs(view), axis=(1, 2)).astype(np.float64)


def _block_rms(audio: np.ndarray, frame: int) -> np.ndarray:
    """按块取 RMS 包络；比峰值更接近人耳对“响度”的感受。"""
    arr = _as_stereo(audio).astype(np.float64)
    n = len(arr)
    if n == 0:
        return np.zeros(1, dtype=np.float64)
    blocks = int(math.ceil(n / frame))
    padded = np.pad(arr, ((0, blocks * frame - n), (0, 0)))
    view = padded.reshape(blocks, frame, arr.shape[1])
    return np.sqrt(np.mean(view * view, axis=(1, 2))).astype(np.float64)


def _soft_knee_reduction_db(
    level_db: np.ndarray,
    threshold_db: float,
    ratio: float,
    knee_db: float,
) -> np.ndarray:
    """软拐点压缩的增益衰减量（正数，dB）。"""
    over = np.asarray(level_db, dtype=np.float64) - float(threshold_db)
    ratio = max(1.0, float(ratio))
    knee = max(0.0, float(knee_db))
    if knee > 0:
        soft = np.clip(over + knee * 0.5, 0.0, knee)
        hard = np.maximum(over - knee * 0.5, 0.0)
        return (soft * soft) / (2.0 * knee) * (1.0 - 1.0 / ratio) + hard * (
            1.0 - 1.0 / ratio
        )
    return np.maximum(over, 0.0) * (1.0 - 1.0 / ratio)


def _smooth_amount(
    target: np.ndarray,
    block_ms: float,
    attack_ms: float,
    release_ms: float,
) -> np.ndarray:
    """对“衰减量 / 提升量”做单极平滑。

    ``target`` 为正数：变大时按 attack 追赶，变小时按 release 回落。
    这样压缩启动快、恢复慢，不会跟着每个音符抽气。
    """
    target = np.asarray(target, dtype=np.float64)
    attack = math.exp(-float(block_ms) / max(0.2, float(attack_ms)))
    release = math.exp(-float(block_ms) / max(1.0, float(release_ms)))
    out = np.empty_like(target)
    current = 0.0
    for i, value in enumerate(target):
        coeff = attack if value > current else release
        current = coeff * current + (1.0 - coeff) * float(value)
        out[i] = current
    return out


def compress(
    audio: np.ndarray,
    sr: int = SR,
    *,
    threshold_db: float = -26.0,
    ratio: float = 6.0,
    attack_ms: float = 6.0,
    release_ms: float = 120.0,
    knee_db: float = 6.0,
    makeup_db: float | None = None,
    frame: int = 256,
) -> np.ndarray:
    """前馈峰值压缩（块包络 + 攻击/释放平滑）。

    这是「音源稳定化」用的强压缩：阈值较低、比例较高，让每个采样的
    平均能量接近，同时保留字头瞬态。合成后再由 MIDI 力度决定实际响度。
    """
    arr = _as_stereo(audio)
    n = len(arr)
    if n == 0:
        return arr.astype(np.float32, copy=False)

    env = _block_envelope(arr, frame)
    env_db = 20.0 * np.log10(env + _EPS)
    reduction = _soft_knee_reduction_db(env_db, threshold_db, ratio, knee_db)
    block_ms = frame / max(1, sr) * 1000.0
    smoothed = _smooth_amount(reduction, block_ms, attack_ms, release_ms)

    gain_db = -np.repeat(smoothed, frame)[:n]
    out = arr.astype(np.float32, copy=True)
    out *= np.power(10.0, gain_db / 20.0).astype(np.float32)[:, None]

    if makeup_db is None:
        makeup_db = 0.0
    if makeup_db:
        out *= db_to_gain(makeup_db)
    return out.astype(np.float32, copy=False)


def bus_compress(
    audio: np.ndarray,
    sr: int = SR,
    *,
    threshold_db: float = -24.0,
    ratio: float = 3.4,
    attack_ms: float = 14.0,
    release_ms: float = 220.0,
    knee_db: float = 8.0,
    detector: str = "rms",
    frame: int = 512,
    makeup: str | float = "auto",
    max_makeup_db: float = 10.0,
    upward_threshold_db: float = -36.0,
    upward_ratio: float = 1.7,
    upward_max_db: float = 4.5,
    upward_attack_ms: float = 40.0,
    upward_release_ms: float = 320.0,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """混音总线压缩：向下压缩 + 自动补偿 + 弱段向上压缩。

    与 :func:`compress` 的“音源稳定化”不同，这里处理的是整首混音：

    * **向下压缩**用 RMS 包络、3.4:1 比例，把过强的段落和峰值收住；
    * **自动补偿**按压缩前后的整体 RMS 计算 makeup，不会因为压缩把整首变小；
    * **向上压缩**只抬低于 ``upward_threshold_db`` 的弱段，最多 ``upward_max_db``，
      慢启动 / 慢释放，避免弱音被埋掉，也不产生抽气感；
    * 最终仍由 :func:`loudness_limit` 推到目标响度，所以这里只负责动态比例。

    Returns:
        ``(处理后的音频, 压缩统计)``。统计用于母带报告与自检。
    """
    arr = _as_stereo(audio).astype(np.float32, copy=True)
    n = len(arr)
    info: dict[str, float | str] = {
        "detector": detector,
        "ratio": round(max(1.0, float(ratio)), 2),
        "threshold_db": round(float(threshold_db), 2),
        "upward_ratio": round(max(1.0, float(upward_ratio)), 2),
        "upward_threshold_db": round(float(upward_threshold_db), 2),
        "makeup_db": 0.0,
        "gain_reduction_db": 0.0,
        "max_gain_reduction_db": 0.0,
        "upward_gain_db": 0.0,
        "max_upward_gain_db": 0.0,
        "crest_db": round(crest_factor_db(arr), 2),
        "input_rms_db": round(rms_db(arr), 2),
    }
    if n == 0:
        info["output_rms_db"] = -120.0
        return arr, info

    if detector == "peak":
        env = _block_envelope(arr, frame)
    else:
        env = _block_rms(arr, frame)
    env_db = 20.0 * np.log10(env + _EPS)

    block_ms = frame / max(1, sr) * 1000.0
    reduction = _soft_knee_reduction_db(env_db, threshold_db, ratio, knee_db)
    reduction = _smooth_amount(reduction, block_ms, attack_ms, release_ms)

    # 弱段向上压缩：只在“有信号但很轻”的块上抬，静音段不抬噪声。
    active = env_db > -58.0
    boost = np.zeros_like(env_db)
    quiet = active & (env_db < float(upward_threshold_db))
    if np.any(quiet):
        boost[quiet] = (
            float(upward_threshold_db) - env_db[quiet]
        ) * (1.0 - 1.0 / max(1.0, float(upward_ratio)))
    boost = np.clip(boost, 0.0, max(0.0, float(upward_max_db)))
    boost = _smooth_amount(boost, block_ms, upward_attack_ms, upward_release_ms)

    gain_db = -reduction
    makeup_db = 0.0
    if makeup == "auto":
        # 先只看向下压缩，量出 RMS 损失；这部分用单一增益补回。
        down_gain = np.power(
            10.0,
            np.repeat(gain_db, frame)[:n].astype(np.float32) / 20.0,
            dtype=np.float32,
        )
        down_rms = rms_db(arr * down_gain[:, None])
        makeup_db = float(
            np.clip(
                rms_db(arr) - down_rms,
                0.0,
                max(0.0, float(max_makeup_db)),
            )
        )
    elif makeup not in ("none", "", None):
        makeup_db = float(np.clip(float(makeup), -max_makeup_db, max_makeup_db))

    total_db = gain_db + makeup_db + boost
    total_gain = np.power(
        10.0,
        np.repeat(total_db, frame)[:n].astype(np.float32) / 20.0,
        dtype=np.float32,
    )
    out = arr * total_gain[:, None]

    info.update(
        {
            "makeup_db": round(makeup_db, 2),
            "gain_reduction_db": round(float(np.mean(reduction[active])), 2)
            if np.any(active)
            else 0.0,
            "max_gain_reduction_db": round(float(np.max(reduction)), 2),
            "upward_gain_db": round(float(np.mean(boost[boost > 0.05])), 2)
            if np.any(boost > 0.05)
            else 0.0,
            "max_upward_gain_db": round(float(np.max(boost)), 2),
            "output_rms_db": round(rms_db(out), 2),
            "crest_db": round(crest_factor_db(out), 2),
        }
    )
    return out.astype(np.float32, copy=False), info


def normalize_loudness(
    audio: np.ndarray,
    target_db: float = -18.0,
    *,
    ceiling_db: float = -1.0,
    max_gain_db: float = 18.0,
) -> np.ndarray:
    """用「单一增益」把 RMS 响度推到目标，再用峰值上限保护。

    单一增益不会压缩音符之间的力度差；峰值保护只在必要时衰减。
    """
    arr = np.asarray(audio, dtype=np.float32)
    current = rms_db(arr)
    if current < -90.0:
        return arr
    gain_db = float(np.clip(target_db - current, -max_gain_db, max_gain_db))
    out = arr * db_to_gain(gain_db)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    ceiling = db_to_gain(ceiling_db)
    if peak > ceiling > 0:
        out *= ceiling / peak
    return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)


def limit_peaks(
    audio: np.ndarray,
    sr: int = SR,
    *,
    ceiling_db: float = -1.0,
    lookahead_ms: float = 6.0,
    release_ms: float = 120.0,
    frame: int = 64,
) -> np.ndarray:
    """前瞻峰值限制器：只压短时峰值，不整体回落。

    与「峰值归一化」不同：这里先把整体推到目标响度，再把超过 ceiling 的
    瞬态用前瞻包络压下来，因此不会因为一个短促峰值就把整首歌拉小。
    """
    arr = _as_stereo(audio).astype(np.float32, copy=True)
    n = len(arr)
    if n == 0:
        return arr
    ceiling = db_to_gain(ceiling_db)
    env = _block_envelope(arr, frame)
    desired = np.minimum(1.0, ceiling / (env + _EPS))

    look = max(1, int(round(sr * lookahead_ms / 1000.0 / frame)))
    if look > 1:
        desired = minimum_filter1d(desired, size=look * 2 + 1, mode="nearest")

    block_ms = frame / max(1, sr) * 1000.0
    release = math.exp(-block_ms / max(1.0, release_ms))
    smooth = np.empty_like(desired)
    current = 1.0
    for i, target in enumerate(desired):
        if target < current:
            current = float(target)  # 攻击：立即跟随（已有前瞻）
        else:
            current = release * current + (1.0 - release) * float(target)
        smooth[i] = current

    gain = np.repeat(smooth, frame)[:n].astype(np.float32)
    out = arr * gain[:, None]
    return np.clip(out, -ceiling, ceiling).astype(np.float32, copy=False)


def loudness_limit(
    audio: np.ndarray,
    target_db: float = -14.0,
    *,
    ceiling_db: float = -1.0,
    max_gain_db: float = 24.0,
) -> np.ndarray:
    """推到目标 RMS 响度，再用前瞻限幅器控制峰值。

    这是母带使用的响度标准；单音 / 采样仍使用 :func:`normalize_loudness`。
    """
    arr = np.asarray(audio, dtype=np.float32)
    current = rms_db(arr)
    if current < -90.0:
        return arr
    gain_db = float(np.clip(target_db - current, -max_gain_db, max_gain_db))
    out = (arr * db_to_gain(gain_db)).astype(np.float32, copy=False)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > db_to_gain(ceiling_db):
        out = limit_peaks(out, ceiling_db=ceiling_db)
    return out


def _periodicity(frame: np.ndarray, sr: int) -> float:
    """帧内 80~1200Hz 的自相关周期度（0=噪声，1=周期波）。"""
    x = np.asarray(frame, dtype=np.float64)
    x = x - float(np.mean(x))
    energy = float(np.dot(x, x))
    if energy < 1e-10:
        return 0.0
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac /= ac[0] + _EPS
    lo = max(1, int(sr / 1200.0))
    hi = min(len(ac) - 1, int(sr / 80.0))
    if hi <= lo:
        return 0.0
    return float(np.clip(np.max(ac[lo:hi]), 0.0, 1.0))


def sibilance_analysis(
    audio: np.ndarray,
    sr: int = SR,
    *,
    frame_length: int = 2048,
    hop: int = 512,
) -> dict[str, np.ndarray]:
    """逐帧分析齿音 / 清辅音特征。

    返回：
    * ``high_ratio``：4.5~10kHz 能量占比；
    * ``periodicity``：自相关周期度；
    * ``zcr``：过零率；
    * ``rms_db``：帧响度；
    * ``times``：帧中心时间。
    """
    arr = _as_stereo(audio)
    mono = np.mean(arr.astype(np.float64), axis=1)
    if mono.size == 0:
        empty = np.zeros(0, dtype=np.float64)
        return {
            "times": empty,
            "high_ratio": empty,
            "periodicity": empty,
            "zcr": empty,
            "rms_db": empty,
        }
    if len(mono) < frame_length:
        mono = np.pad(mono, (0, frame_length - len(mono)))
    frames = sliding_window_view(mono, frame_length)[::hop]
    window = np.hanning(frame_length)
    windowed = frames * window[None, :]

    spec = np.abs(np.fft.rfft(windowed, axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame_length, 1.0 / sr)
    total = np.sum(spec, axis=1) + _EPS
    high_mask = (freqs >= 4500.0) & (freqs <= 10000.0)
    high_ratio = np.sum(spec[:, high_mask], axis=1) / total

    signs = np.signbit(windowed)
    zcr = np.mean(np.abs(np.diff(signs.astype(np.int8), axis=1)), axis=1)
    rms = np.sqrt(np.mean(windowed * windowed, axis=1))
    rms_db = 20.0 * np.log10(rms + _EPS)
    periodicity = np.array([_periodicity(f, sr) for f in windowed], dtype=np.float64)
    times = (np.arange(len(frames)) * hop + frame_length * 0.5) / sr
    return {
        "times": times,
        "high_ratio": high_ratio,
        "periodicity": periodicity,
        "zcr": zcr,
        "rms_db": rms_db,
    }


def _gain_curve(
    gain_db: np.ndarray,
    times: np.ndarray,
    n: int,
    sr: int,
) -> np.ndarray:
    """把帧级 dB 增益插值到采样点。"""
    if len(times) == 0 or n <= 0:
        return np.ones(max(1, n), dtype=np.float32)
    x = np.arange(n, dtype=np.float64) / sr
    values = np.interp(x, times, gain_db, left=gain_db[0], right=gain_db[-1])
    return np.power(10.0, values / 20.0).astype(np.float32)


def reduce_sibilance(
    audio: np.ndarray,
    sr: int = SR,
    *,
    max_reduction_db: float = SIBILANCE_MAX_REDUCTION_DB,
    ratio_threshold: float = SIBILANCE_RATIO_THRESHOLD,
    periodicity_threshold: float = SIBILANCE_PERIODICITY_THRESHOLD,
) -> np.ndarray:
    """动态齿音衰减：只压 4.5~10kHz 的噪声性高频，不动元音低频。"""
    arr = _as_stereo(audio).astype(np.float32, copy=False)
    if arr.size == 0 or max_reduction_db <= 0:
        return arr
    info = sibilance_analysis(arr, sr)
    if len(info["times"]) == 0:
        return arr

    ratio = info["high_ratio"]
    periodic = info["periodicity"]
    strength = np.clip(
        (ratio - ratio_threshold) / max(0.05, 1.0 - ratio_threshold), 0.0, 1.0
    ) * np.clip(
        (periodicity_threshold - periodic) / 0.35, 0.0, 1.0
    )
    strength[info["rms_db"] < -55.0] = 0.0
    if float(np.max(strength)) < 0.02:
        return arr

    gain_db = -float(max_reduction_db) * strength
    curve = _gain_curve(gain_db, info["times"], len(arr), sr)[:, None]

    nyq = sr * 0.5
    upper = min(10000.0, nyq * 0.92)
    if upper <= 4600.0:
        return arr
    sos = butter(3, [4500.0 / nyq, upper / nyq], btype="bandpass", output="sos")
    try:
        band = sosfiltfilt(sos, arr, axis=0).astype(np.float32, copy=False)
    except ValueError:  # pragma: no cover - 极短采样
        band = sosfilt(sos, arr, axis=0).astype(np.float32, copy=False)
    return (arr - band + band * curve).astype(np.float32, copy=False)


def reduce_voiceless_consonants(
    audio: np.ndarray,
    sr: int = SR,
    *,
    cons_frame: int = 0,
    frame_period: float = 5.0,
    max_reduction_db: float = VOICELESS_MAX_REDUCTION_DB,
) -> np.ndarray:
    """降低清辅音（unvoiced consonant）整体响度。

    只在采样开头的辅音区内、且自相关周期度低时衰减，元音和浊辅音不受影响。
    """
    arr = _as_stereo(audio).astype(np.float32, copy=False)
    if arr.size == 0 or cons_frame <= 0 or max_reduction_db <= 0:
        return arr
    info = sibilance_analysis(arr, sr)
    if len(info["times"]) == 0:
        return arr

    # 辅音区（WORLD 5ms 帧）+ 25ms 过渡，覆盖分析帧中心
    limit = float(cons_frame) * float(frame_period) / 1000.0 + 0.025
    mask = info["times"] <= limit
    periodic = info["periodicity"]
    strength = np.clip((0.58 - periodic) / 0.40, 0.0, 1.0)
    strength[~mask] = 0.0
    strength[info["rms_db"] < -55.0] = 0.0
    if float(np.max(strength)) < 0.02:
        return arr

    gain_db = -float(max_reduction_db) * strength
    curve = _gain_curve(gain_db, info["times"], len(arr), sr)[:, None]
    return (arr * curve).astype(np.float32, copy=False)


def consonant_gain_curve(
    n: int,
    sr: int = SR,
    cons_frame: int = 8,
    frame_period: float = 5.0,
    *,
    plosive: bool = False,
    consonant_db: float = CONSONANT_ATTENUATION_DB,
    plosive_db: float = PLOSIVE_ATTENUATION_DB,
    burst_ms: float = PLOSIVE_BURST_MS,
    ramp_ms: float = PLOSIVE_RAMP_MS,
) -> np.ndarray:
    """生成单个音节的辅音衰减包络（线性幅度）。

    所有音节的开头辅音区先整体降低 ``consonant_db``；对 ``基 / 库 / 叮 / 咚``
    等爆破音字，再在最初 ``burst_ms`` 内叠加 ``plosive_db``。两段都使用平滑
    斜坡，避免在采样中间产生新的“咔哒”声。
    """
    n = max(0, int(n))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    cons_sec = max(0.0, float(cons_frame) * float(frame_period) / 1000.0)
    if cons_sec <= 0 and not plosive:
        return np.ones(n, dtype=np.float32)

    t = np.arange(n, dtype=np.float64) / max(1, int(sr))
    gain_db = np.zeros(n, dtype=np.float64)

    # 所有辅音：辅音区内衰减，随后在 35ms 内平滑回到 0dB。
    if cons_sec > 0 and consonant_db > 0:
        fade = max(0.012, min(0.045, cons_sec * 0.8))
        gain_db -= float(consonant_db) * np.clip(
            (cons_sec + fade - t) / max(1e-6, fade), 0.0, 1.0
        )

    # 爆破音：起音处额外大幅衰减，再平滑过渡回普通辅音电平。
    if plosive and plosive_db > 0:
        burst = max(0.004, float(burst_ms) / 1000.0)
        ramp = max(0.008, float(ramp_ms) / 1000.0)
        extra = np.zeros(n, dtype=np.float64)
        extra[t <= burst] = 1.0
        mask = (t > burst) & (t < burst + ramp)
        extra[mask] = 1.0 - (t[mask] - burst) / max(1e-6, ramp)
        gain_db -= float(plosive_db) * extra

    return np.power(10.0, gain_db / 20.0).astype(np.float32)


def attenuate_consonants(
    audio: np.ndarray,
    sr: int = SR,
    cons_frame: int = 8,
    frame_period: float = 5.0,
    *,
    plosive: bool = False,
) -> np.ndarray:
    """在合成单位输出端压低辅音；爆破音字的起始瞬态额外大幅衰减。

    这一步放在最终响度归一化之后，因此不会被 ``loudness_limit`` 重新推高；
    元音主体保持原样，字头仍然可辨。
    """
    arr = _as_stereo(audio).astype(np.float32, copy=False)
    curve = consonant_gain_curve(
        len(arr), sr, cons_frame, frame_period, plosive=plosive
    )
    if curve.size == 0 or float(np.min(curve)) >= 0.999:
        return arr
    return (arr * curve[:, None]).astype(np.float32, copy=False)


def condition_sample(
    audio: np.ndarray,
    sr: int = SR,
    *,
    target_db: float = -18.0,
    low_cut: float = 24.0,
    threshold_db: float = -28.0,
    ratio: float = 8.0,
    attack_ms: float = 4.0,
    release_ms: float = 90.0,
    ceiling_db: float = -1.0,
    cons_frame: int = 0,
    frame_period: float = 5.0,
    deess: bool = True,
    deess_reduction_db: float = SIBILANCE_MAX_REDUCTION_DB,
    voiceless_reduction_db: float = VOICELESS_MAX_REDUCTION_DB,
) -> np.ndarray:
    """音源预处理：去直流 → 低切 → 强压缩 → 去齿音 → 清辅音衰减 → 目标响度。

    目的：让任意字、任意音源套件在进入 WORLD 前具有相同的稳定响度底子，
    同时压低齿音和清辅音，避免高音 presence 提升后出现刺耳 / 过响的辅音。
    """
    arr = _as_stereo(audio).astype(np.float32, copy=True)
    if arr.size == 0:
        return arr

    # 去直流，避免 WORLD 分析出错误的低频能量
    arr -= np.mean(arr, axis=0, keepdims=True)

    if low_cut and low_cut > 0:
        nyq = sr * 0.5
        b, a = butter(2, max(10.0, low_cut) / nyq, btype="high")
        arr = lfilter(b, a, arr, axis=0).astype(np.float32, copy=False)

    arr = compress(
        arr,
        sr,
        threshold_db=threshold_db,
        ratio=ratio,
        attack_ms=attack_ms,
        release_ms=release_ms,
        knee_db=6.0,
    )
    if deess:
        arr = reduce_sibilance(arr, sr, max_reduction_db=deess_reduction_db)
    if voiceless_reduction_db > 0:
        arr = reduce_voiceless_consonants(
            arr,
            sr,
            cons_frame=cons_frame,
            frame_period=frame_period,
            max_reduction_db=voiceless_reduction_db,
        )
    arr = normalize_loudness(arr, target_db=target_db, ceiling_db=ceiling_db)

    # 采样首尾 1.5ms 防爆音（不影响字头音色）
    k = min(int(sr * 0.0015), len(arr) // 2)
    if k > 1:
        ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)[:, None]
        arr[:k] *= ramp
        arr[-k:] *= ramp[::-1]
    return arr


# ---------------------------------------------------------------------------
# 音高 / 频段可听度
# ---------------------------------------------------------------------------


def pitch_audibility_db(pitch: float) -> float:
    """按音高给一点等响度补偿（dB）。

    真实听感在 2~5 kHz 最灵敏；很高的音（> C7）基频已接近听觉上限，
    需要额外抬升才不会「只剩气声」。低频则只做轻微补偿，避免轰头。
    """
    p = float(pitch)
    if p <= 24:
        return 5.0
    if p <= 36:
        return 5.0 - (p - 24.0) / 12.0 * 3.0  # +5 → +2
    if p <= 48:
        return 2.0 - (p - 36.0) / 12.0 * 2.0  # +2 → 0
    if p <= 84:
        return 0.0
    if p <= 96:
        return (p - 84.0) / 12.0 * 2.0  # 0 → +2
    if p <= 108:
        return 2.0 + (p - 96.0) / 12.0 * 5.0  # +2 → +7
    return 7.0 + min(5.0, (p - 108.0) / 6.0 * 3.0)


def audibility_gain(pitch: float) -> float:
    return db_to_gain(pitch_audibility_db(pitch))


def spectrum_weight(pitch: float, freqs: np.ndarray) -> np.ndarray:
    """公开接口：返回 WORLD 频谱使用的音高相关线性增益曲线。"""
    return _weight_curve(freqs, pitch)


def _weight_curve(freqs: np.ndarray, pitch: float) -> np.ndarray:
    """给 WORLD 频谱用的音高相关增益曲线（线性幅度）。"""
    f = np.asarray(freqs, dtype=np.float64)
    db = np.zeros_like(f)

    # 低频：保留基频，切掉基频以下的次低频隆隆声
    if pitch < 48:
        f0 = midi_to_hz(pitch)
        sub_cut = float(np.clip(f0 * 0.68, 16.0, 42.0))
        db += np.where(f < sub_cut, -10.0, 0.0)
        db += np.where((f >= sub_cut) & (f < max(90.0, f0 * 1.2)), -2.0, 0.0)

    # 高音：补偿 presence / air，避免基频升高后高频泛音不足
    high = float(np.clip((float(pitch) - 72.0) / 24.0, 0.0, 1.0))
    if high > 0:
        # 2~7 kHz 的临场感，最多 +5.5 dB
        presence = np.exp(-0.5 * ((np.log2(np.maximum(f, 1.0) / 3600.0)) / 1.05) ** 2)
        db += high * 5.5 * presence
        # 8 kHz 以上空气感，最多 +4 dB
        air = np.clip((f - 7000.0) / 9000.0, 0.0, 1.0)
        db += high * 4.0 * air

    # 低音：给基频一点厚度，但不要盖住中频
    if pitch < 48:
        low = float(np.clip((48.0 - float(pitch)) / 18.0, 0.0, 1.0))
        body = np.exp(-0.5 * ((np.log2(np.maximum(f, 1.0) / 140.0)) / 0.8) ** 2)
        db += low * 2.5 * body

    return np.power(10.0, db / 20.0)


def band_energies(audio: np.ndarray, sr: int = SR) -> dict[str, float]:
    """粗略四段能量（dB），用于动态响度分析。"""
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=1)
    if arr.size < 8:
        return {"sub": -120.0, "low": -120.0, "mid": -120.0, "high": -120.0}

    spec = np.abs(np.fft.rfft(arr * np.hanning(len(arr)))) ** 2
    freqs = np.fft.rfftfreq(len(arr), 1.0 / sr)
    total = float(np.sum(spec) + _EPS)

    bands = {
        "sub": (20.0, 90.0),
        "low": (90.0, 500.0),
        "mid": (500.0, 2000.0),
        "high": (2000.0, min(sr * 0.5, 16000.0)),
    }
    out: dict[str, float] = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        value = float(np.sum(spec[mask]) / total)
        out[name] = round(10.0 * math.log10(max(_EPS, value)), 2)
    return out


def dynamic_high_gain(audio: np.ndarray, pitch: float, sr: int = SR) -> float:
    """分析高频占比，决定额外的高频补偿（dB）。

    高音若 2 kHz 以上能量占比过低，会听不见；这里给最多 +6 dB 的动态补偿。
    """
    if pitch < 72:
        return 0.0
    bands = band_energies(audio, sr)
    ratio_db = bands.get("high", -120.0)
    # 高频占比低于 -12 dB 时开始补偿，越低补得越多
    deficit = max(0.0, -12.0 - ratio_db)
    return float(min(6.0, deficit * 0.6))


def air_shelf(
    audio: np.ndarray,
    sr: int = SR,
    gain_db: float = 0.0,
    cutoff: float = 6000.0,
) -> np.ndarray:
    """一阶高频搁架（用于动态高频补偿）。"""
    if abs(gain_db) < 0.05:
        return np.asarray(audio)
    arr = _as_stereo(audio).astype(np.float32, copy=False)
    nyq = sr * 0.5
    wc = min(0.98, max(0.02, cutoff / nyq))
    # 简单一阶差分高频 + 原声混合，避免复杂双线性设计
    b, a = butter(1, wc, btype="high")
    high = lfilter(b, a, arr, axis=0).astype(np.float32, copy=False)
    mix = db_to_gain(gain_db) - 1.0
    return (arr + mix * high).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# 动态包络
# ---------------------------------------------------------------------------


def expression_curve(
    n: int,
    start_gain: float,
    end_gain: float,
    *,
    shape: str = "smooth",
) -> np.ndarray:
    """生成一个音符内部的渐强 / 渐弱曲线。"""
    n = max(0, int(n))
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if abs(end_gain - start_gain) < 1e-6:
        return np.full(n, start_gain, dtype=np.float32)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    if shape == "smooth":
        t = t * t * (3.0 - 2.0 * t)
    return (start_gain + (end_gain - start_gain) * t).astype(np.float32)


__all__ = [
    "air_shelf",
    "attenuate_consonants",
    "audibility_gain",
    "band_energies",
    "bus_compress",
    "compress",
    "condition_sample",
    "consonant_gain_curve",
    "crest_factor_db",
    "db_to_gain",
    "dynamic_high_gain",
    "expression_curve",
    "gain_to_db",
    "limit_peaks",
    "loudness_limit",
    "midi_to_hz",
    "normalize_loudness",
    "peak_db",
    "pitch_audibility_db",
    "reduce_sibilance",
    "reduce_voiceless_consonants",
    "rms",
    "rms_db",
    "sibilance_analysis",
    "spectrum_weight",
]
