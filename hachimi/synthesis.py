"""WORLD vocoder 语音合成：采样分析缓存、元音/辅音分离、音高与时值变换。

性能说明
--------
上游实现每个音符都要做一次「逐频点 ``np.interp`` 循环 + ``pw.synthesize``」，
一首曲子有上千个音符时 CPU 会长时间满载。这里做了三件事：

1. **分析结果磁盘缓存**：``harvest / cheaptrick / d4c`` 只在首次使用时计算，
   之后直接从 ``cache/preproc`` 读取（多进程渲染时各 worker 也能共享）；
2. **向量化频谱插值**：用整数索引 + 线性权重一次性插值整张 (帧 × 频点) 矩阵，
   替代 513 次 Python 循环；
3. 输出 ``float32``，并由 :mod:`hachimi.render` 按 (实际采样 id, 音高, 时长) 复用合成结果。
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pyworld as pw
import soundfile as sf
from scipy.signal import resample_poly

from hachimi.config import (
    CACHE_DIR,
    CONS_FRAMES,
    EXTRA_SAMPLE_DIRS,
    FRAME_PERIOD,
    HACHIMI_SET2_CHARS,
    HACHIMI_SET2_DIRS,
    SAMPLE_ALIASES,
    SAMPLE_COMPRESSOR_ATTACK_MS,
    SAMPLE_COMPRESSOR_RATIO,
    SAMPLE_COMPRESSOR_RELEASE_MS,
    SAMPLE_COMPRESSOR_THRESHOLD_DB,
    SAMPLE_FALLBACK,
    SAMPLE_PATH,
    SAMPLE_SOURCE_DIRS,
    SAMPLE_TARGET_DB,
    SIBILANCE_UNIT_REDUCTION_DB,
    SOURCE_DD_DIR,
    SOURCE_DIR,
    SOURCE_HACHIMI_DIR,
    SOURCE_MANBO_DIR,
    SOURCE_MB_DIR,
    SR,
    UNIT_TARGET_DB,
    VOICE_SAMPLE_VARIANTS,
)
from hachimi.loudness import (
    air_shelf,
    attenuate_consonants,
    condition_sample,
    dynamic_high_gain,
    loudness_limit,
    reduce_sibilance,
    spectrum_weight,
)

logger = logging.getLogger(__name__)

_CACHE_VERSION = "v10"

#: 采样搜索目录：本地“音源”优先，其次 assets/samples，最后旧“补充”目录
SAMPLE_DIRS = [
    SOURCE_DIR,
    SOURCE_HACHIMI_DIR,
    SOURCE_DD_DIR,
    SOURCE_MB_DIR,
    SOURCE_MANBO_DIR,
    SAMPLE_PATH,
    *EXTRA_SAMPLE_DIRS,
]


# ---------------------------------------------------------------------------
# 采样解析
# ---------------------------------------------------------------------------


def sample_exists(name: str) -> bool:
    if name.startswith("h2_"):
        base = name[3:]
        return any(os.path.exists(os.path.join(folder, f"{base}.wav")) for folder in HACHIMI_SET2_DIRS)
    return any(os.path.exists(os.path.join(folder, f"{name}.wav")) for folder in SAMPLE_DIRS)


def sample_path_for(char: str) -> str:
    """返回字符 / 采样 id 对应的 wav 路径；缺失时使用别名或兜底采样。"""
    if char.startswith("h2_"):
        base = char[3:]
        for folder in HACHIMI_SET2_DIRS:
            path = os.path.join(folder, f"{base}.wav")
            if os.path.exists(path):
                return path
        char = base  # 第二套缺该字时回退第一套

    for folder in SAMPLE_DIRS:
        path = os.path.join(folder, f"{char}.wav")
        if os.path.exists(path):
            return path

    alias = SAMPLE_ALIASES.get(char)
    if alias:
        for folder in SAMPLE_DIRS:
            path = os.path.join(folder, f"{alias}.wav")
            if os.path.exists(path):
                return path

    for folder in SAMPLE_DIRS:
        path = os.path.join(folder, f"{SAMPLE_FALLBACK}.wav")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"找不到字符「{char}」的采样，且兜底采样不存在。"
    )


def available_sample_chars() -> set[str]:
    """当前可用的采样字符 / 采样 id（不含别名）。"""
    names: set[str] = set()
    for folder in SAMPLE_DIRS:
        if not os.path.isdir(folder):
            continue
        names.update(
            Path(p).stem
            for p in os.listdir(folder)
            if p.lower().endswith((".wav", ".flac", ".ogg"))
        )
    for folder in HACHIMI_SET2_DIRS:
        if not os.path.isdir(folder):
            continue
        names.update(
            f"h2_{Path(p).stem}"
            for p in os.listdir(folder)
            if p.lower().endswith((".wav", ".flac", ".ogg"))
        )
    return names


def bank_signature() -> str:
    """音源目录签名（文件路径 + 修改时间 + 大小），用于热更新检测。"""
    parts: list[str] = []
    folders = list(dict.fromkeys(SAMPLE_DIRS + HACHIMI_SET2_DIRS + SAMPLE_SOURCE_DIRS))
    for folder in folders:
        if not folder or not os.path.isdir(folder):
            continue
        try:
            for name in sorted(os.listdir(folder)):
                if not name.lower().endswith((".wav", ".flac", ".ogg")):
                    continue
                path = os.path.join(folder, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                parts.append(f"{path}|{st.st_mtime_ns}|{st.st_size}")
        except OSError:
            continue
    return hashlib.md5("\n".join(parts).encode("utf-8", "replace")).hexdigest()


_BANK_SIGNATURE: str | None = None


def refresh_sample_bank(force: bool = False) -> bool:
    """检测本地音源是否被修改；返回是否需要清除合成单位缓存。"""
    global _BANK_SIGNATURE
    sig = bank_signature()
    changed = force or (_BANK_SIGNATURE is not None and sig != _BANK_SIGNATURE)
    _BANK_SIGNATURE = sig
    if changed:
        logger.info("检测到音源变化，已刷新采样缓存")
    return changed


def sample_signature(char: str) -> str:
    """单个采样文件签名，参与预处理缓存键；修改音源后自动失效。"""
    try:
        path = sample_path_for(char)
        st = os.stat(path)
        return f"{st.st_mtime_ns:x}_{st.st_size:x}"
    except Exception:
        return "0"


def resolve_sample_id(char: str) -> str:
    """把歌词字解析成**实际采样文件**的 id（处理别名，不处理音高/变体）。

    曼波等声库只提供了部分字（曼 / 波），其余字通过 ``SAMPLE_ALIASES``
    借用哈基米采样。如果直接把歌词字当作 sample id，会出现两个问题：

    * 同一个采样文件在不同声库下生成不同缓存单位，浪费合成；
    * 更严重的是辅音帧数 / 爆破音判定按**歌词字**而不是实际采样字执行，
      例如曼波的「达」借用「那」，却按「达」的爆破音参数再压一次音头。

    因此这里统一返回实际采样字，让所有声库共用同一套辅音 / 音头处理。
    """
    current = str(char or "")
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        if current.startswith("h2_"):
            if sample_exists(current):
                return current
            current = current[3:]
            continue
        if any(os.path.exists(os.path.join(folder, f"{current}.wav")) for folder in SAMPLE_DIRS):
            return current
        alias = SAMPLE_ALIASES.get(current)
        if not alias:
            break
        current = str(alias)
    return str(char or "")


def choose_sample(
    word: str,
    voice: str = "",
    track: int = 0,
    pitch: int = 60,
    start: float = 0.0,
    sample_set: int = 1,
) -> str:
    """根据语音类别 / 音源套件选择采样。

    - 先按语音类别选择变体采样（鸡/鸡2、曼1/2/3、巴1/2/3 …）；
    - 再解析别名，返回实际存在的采样 id（如曼波「夏」→「哈」）；
    - ``sample_set=2`` 且**实际采样字**属于哈基米 15 字时，使用
      ``音源/哈基米第二版``（兼容旧 ``补充/哈基米音源``）的第二套采样
      （内部 id 加 ``h2_`` 前缀）。
    歌词显示的仍然是 ``word``，只有合成用的采样 id 改变。
    """
    if word.startswith("h2_"):
        return word if sample_exists(word) else resolve_sample_id(word[3:])

    variants = VOICE_SAMPLE_VARIANTS.get(voice, {}).get(word)
    if variants:
        offset = int(round(start * 1000)) + int(pitch) * 7 + int(track)
        ordered = variants[offset % len(variants):] + variants[: offset % len(variants)]
        for candidate in ordered:
            if sample_exists(candidate):
                return candidate

    resolved = resolve_sample_id(word)
    if (
        int(sample_set) == 2
        and resolved in HACHIMI_SET2_CHARS
        and sample_exists(f"h2_{resolved}")
    ):
        return f"h2_{resolved}"
    return resolved


def _resolved_base(char: str) -> str:
    """采样 id → 实际采样字（去掉 h2_，并解析别名）。"""
    resolved = resolve_sample_id(char)
    return resolved[3:] if resolved.startswith("h2_") else resolved


def cons_frames_for(char: str) -> int:
    """采样 id（如 鸡2 / 曼1 / h2_哈）→ 辅音帧数。

    先解析别名，保证直接调用 ``synthesize_note(sample_id="夏")`` 时，
    也按实际采样「哈」的 3 帧辅音处理，而不是按歌词字的 8 帧。
    """
    base = _resolved_base(char)
    return CONS_FRAMES.get(base, CONS_FRAMES.get(base[:1], 8))


#: 爆破 / 塞擦起始的字：基、库、叮、咚 等。它们的最初 20ms 额外衰减。
PLOSIVE_CHARS = set("基鸡吉机叫健库康叮咚大带达豆的波巴北不出搞狗噶")


def is_plosive_sample(char: str) -> bool:
    """采样 id 是否属于需要重压起始爆破音的字。

    别名先解析到实际采样：曼波「达」借用「那」，不应按「达」的爆破音
    参数重复压低音头；「不」借用「波」则仍按爆破音处理。
    """
    if not char:
        return False
    base = _resolved_base(char)
    return bool(base) and (base[0] in PLOSIVE_CHARS or base in PLOSIVE_CHARS)


# ---------------------------------------------------------------------------
# WORLD 分析
# ---------------------------------------------------------------------------


def preprocess_sample(
    sample_path: str,
    cons_frame: int = 8,
    sr: int = SR,
    frame_period: float = FRAME_PERIOD,
) -> list[dict]:
    """分析一个采样，返回两个声道的 WORLD 参数与元音/辅音掩码。"""
    x, orig_sr = sf.read(sample_path)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    elif x.shape[1] == 1:
        x = np.concatenate([x, x], axis=1)
    else:
        x = x[:, :2]

    if orig_sr != sr:
        # 用 scipy 重采样，避免 worker 进程加载 librosa/numba（每个 worker 可省数百 MB）
        x = resample_poly(x, sr, orig_sr, axis=0).astype(np.float64, copy=False)

    # 音源稳定化：强压缩 + 补偿 + 统一目标响度。
    # 这样不同字 / 不同套音源的底子一致，后续只由乐谱力度决定强弱。
    x = condition_sample(
        x,
        sr,
        target_db=SAMPLE_TARGET_DB,
        threshold_db=SAMPLE_COMPRESSOR_THRESHOLD_DB,
        ratio=SAMPLE_COMPRESSOR_RATIO,
        attack_ms=SAMPLE_COMPRESSOR_ATTACK_MS,
        release_ms=SAMPLE_COMPRESSOR_RELEASE_MS,
        cons_frame=cons_frame,
        frame_period=frame_period,
    )

    preproc: list[dict] = []
    for ch in range(2):
        channel = np.ascontiguousarray(x[:, ch], dtype=np.float64)
        f0, t = pw.harvest(channel, sr, frame_period=frame_period)
        sp = pw.cheaptrick(channel, f0, t, sr)
        ap = pw.d4c(channel, f0, t, sr)

        is_vowel = np.ones_like(f0, dtype=bool)
        # 至少保留 1 帧元音，避免极短采样被整段当作辅音 / 整段被拉伸
        n_cons = min(int(cons_frame), max(0, len(f0) - 1))
        if n_cons > 0:
            is_vowel[:n_cons] = False

        preproc.append(
            {
                "f0": f0,
                "sp": sp.astype(np.float32),
                "ap": ap.astype(np.float32),
                "t": t,
                "is_vowel": is_vowel,
            }
        )
    return preproc


# ---------------------------------------------------------------------------
# 磁盘缓存
# ---------------------------------------------------------------------------


def preproc_cache_path(char: str, sr: int = SR, frame_period: float = FRAME_PERIOD) -> str:
    """缓存文件路径（ASCII 安全，供子进程读取）。"""
    cons = cons_frames_for(char)
    code = "_".join(f"{ord(c):x}" for c in char) or "0"
    # 文件修改时间/大小参与 key：本地音源改完后自动重新预处理，不会读到旧缓存
    sig = sample_signature(char)
    key = f"{code}_{sr}_{frame_period:g}_{cons}_{_CACHE_VERSION}_{sig}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:10]
    folder = os.path.join(CACHE_DIR, "preproc")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{code}_{digest}.npz")


def get_preprocessed(
    char: str,
    sr: int = SR,
    frame_period: float = FRAME_PERIOD,
    use_cache: bool = True,
) -> list[dict]:
    """获取字符的 WORLD 分析结果（带磁盘缓存）。"""
    path = preproc_cache_path(char, sr, frame_period)
    if use_cache and os.path.exists(path):
        try:
            with np.load(path, allow_pickle=False) as data:
                out = []
                for ch in range(2):
                    out.append(
                        {
                            "f0": data[f"f0_{ch}"],
                            "sp": data[f"sp_{ch}"],
                            "ap": data[f"ap_{ch}"],
                            "t": data[f"t_{ch}"],
                            "is_vowel": data[f"v_{ch}"].astype(bool),
                        }
                    )
                return out
        except Exception as exc:  # pragma: no cover - 缓存损坏时重新计算
            logger.warning("读取缓存失败（%s）：%s，重新分析", path, exc)

    sample_path = sample_path_for(char)
    cons = cons_frames_for(char)
    preproc = preprocess_sample(sample_path, cons_frame=cons, sr=sr, frame_period=frame_period)
    if use_cache:
        try:
            folder = os.path.dirname(path)
            fd, tmp = tempfile.mkstemp(dir=folder, suffix=".npz.tmp")
            os.close(fd)
            arrays = {}
            for ch, item in enumerate(preproc):
                arrays[f"f0_{ch}"] = item["f0"]
                arrays[f"sp_{ch}"] = item["sp"]
                arrays[f"ap_{ch}"] = item["ap"]
                arrays[f"t_{ch}"] = item["t"]
                arrays[f"v_{ch}"] = item["is_vowel"]
            # 注意：不能用 np.savez(tmp_path)，它会在文件名后再追加 .npz。
            with open(tmp, "wb") as fh:
                np.savez(fh, **arrays)
            try:
                os.replace(tmp, path)
            except OSError as exc:
                # 多个进程同时写同一缓存时可能失败；忽略即可，worker 内存里已有结果
                logger.debug("缓存替换失败（忽略）：%s", exc)
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        except Exception as exc:  # pragma: no cover
            logger.warning("写入缓存失败：%s", exc)
    return preproc


def preprocess_chars(
    chars: Iterable[str],
    progress_cb=None,
    sr: int = SR,
    frame_period: float = FRAME_PERIOD,
) -> dict[str, list[dict]]:
    """批量预处理（去重）并返回 ``{char: preproc}``。"""
    unique = list(dict.fromkeys(chars))
    out: dict[str, list[dict]] = {}
    for i, char in enumerate(unique):
        out[char] = get_preprocessed(char, sr=sr, frame_period=frame_period)
        if progress_cb:
            progress_cb(i + 1, len(unique), char)
    return out


# ---------------------------------------------------------------------------
# 快速合成
# ---------------------------------------------------------------------------


def _interp_rows(mat: np.ndarray, n: int) -> np.ndarray:
    """把 (T, D) 矩阵沿时间轴线性插值到 n 行（向量化）。"""
    if n <= 0:
        return np.zeros((0, mat.shape[1]), dtype=mat.dtype)
    m = mat.shape[0]
    if m == 0:
        return np.zeros((n, mat.shape[1]), dtype=mat.dtype)
    if m == 1:
        return np.repeat(mat, n, axis=0)
    idx = np.linspace(0.0, m - 1, n, dtype=np.float64)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, m - 1)
    frac = (idx - i0).astype(np.float32)[:, None]
    return (mat[i0] * (1.0 - frac) + mat[i1] * frac).astype(mat.dtype, copy=False)


def _shape_spectrum(sp: np.ndarray, pitch: int, sr: int) -> np.ndarray:
    """按目标音高调整 WORLD 频谱：高音补 presence/air，低音补厚度。"""
    if sp.size == 0:
        return sp
    freqs = np.linspace(0.0, sr * 0.5, sp.shape[1], dtype=np.float64)
    # sp 是功率谱，因此把幅度权重平方后应用，等效于 dB 增益
    weight = np.square(spectrum_weight(float(pitch), freqs))
    return (sp * weight[None, :]).astype(sp.dtype, copy=False)


def _audibility_post(y: np.ndarray, pitch: int, sr: int) -> np.ndarray:
    """对合成单位做动态频段分析 + 目标响度归一化。

    - 先把单位 RMS 归一到统一标准，保证音源本身平稳；
    - 高音若高频占比不足，再做一次动态 air shelf，确保听得见。
    """
    if y.size == 0:
        return y
    # 用前瞻限幅把低音的峰值压下来，使其 RMS 也能到达统一目标；
    # 只做一次固定增益会把低音压得很轻，导致低音“听不见”。
    out = loudness_limit(y, target_db=UNIT_TARGET_DB, ceiling_db=-1.0, max_gain_db=30.0)
    # 合成单位再补一次轻量去齿音：防止高音 presence / air 补偿把残留齿音推亮
    out = reduce_sibilance(out, sr, max_reduction_db=SIBILANCE_UNIT_REDUCTION_DB)
    out = loudness_limit(out, target_db=UNIT_TARGET_DB, ceiling_db=-1.0, max_gain_db=30.0)
    extra_db = dynamic_high_gain(out, pitch, sr)
    if extra_db > 0.35:
        out = air_shelf(out, sr, gain_db=min(4.0, extra_db), cutoff=5500.0)
        out = loudness_limit(out, target_db=UNIT_TARGET_DB, ceiling_db=-1.0, max_gain_db=30.0)
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# 时值拉伸：只拉元音中段，辅音 / 音头 / 尾音保持原型
# ---------------------------------------------------------------------------


def _split_source_vowel(n: int) -> tuple[int, int, int]:
    """把源元音帧分为（音头, 中段, 尾音）。"""
    n = max(1, int(n))
    if n <= 2:
        return 0, n, 0
    attack = min(4, max(1, int(round(n * 0.12))))
    release = min(6, max(1, int(round(n * 0.18))))
    if attack + release >= n:
        attack = max(1, n // 4)
        release = max(1, n // 4)
    middle = max(1, n - attack - release)
    # 极端短元音：保证三者相加准确
    if attack + middle + release != n:
        attack = min(attack, max(0, n - 2))
        release = min(release, max(0, n - attack - 1))
        middle = n - attack - release
    return attack, middle, release


def _split_target_vowel(total: int, attack: int, release: int) -> tuple[int, int, int]:
    """目标元音帧分配：音头 / 尾音保持固定，只让中段伸缩。"""
    total = max(1, int(total))
    if total < attack + release + 1:
        # 目标比“音头+尾音”还短：只能按比例压缩，优先保留中段
        scale = total / max(1, attack + release)
        a = max(0, int(round(attack * scale)))
        r = max(0, int(round(release * scale)))
        if a + r >= total:
            a, r = 0, 0
        middle = total - a - r
    else:
        a, r = attack, release
        middle = total - a - r
    if middle < 1:
        a = max(0, total // 3)
        r = max(0, total // 3)
        middle = total - a - r
    return a, middle, r


def vowel_stretch_plan(n_vowel: int, n_target: int) -> dict[str, int | float]:
    """返回元音时值分配，便于测试和调试。

    正常长音：``target_attack`` / ``target_release`` 与源相同，
    只有 ``target_middle`` 变长；缩短时优先压缩中段。
    """
    a, m, r = _split_source_vowel(n_vowel)
    ta, tm, tr = _split_target_vowel(n_target, a, r)
    return {
        "source_attack": a,
        "source_middle": m,
        "source_release": r,
        "target_attack": ta,
        "target_middle": tm,
        "target_release": tr,
        "stretch_ratio": (tm / m) if m else 0.0,
    }


def _interp_series(values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if values.size == 0:
        return np.zeros(n, dtype=np.float64)
    if values.size == 1:
        return np.full(n, values[0], dtype=np.float64)
    return np.interp(np.linspace(0, values.size - 1, n), np.arange(values.size), values)


def _loop_series(values: np.ndarray, n: int, crossfade: int = 3) -> np.ndarray:
    """把中段循环铺满，循环点做交叉淡化，保证长音没有断点。"""
    values = np.asarray(values, dtype=np.float64)
    m = values.size
    if n <= 0 or m == 0:
        return np.zeros(max(0, n), dtype=np.float64)
    if m == 1:
        return np.full(n, values[0], dtype=np.float64)
    xf = max(1, min(crossfade, m // 3))
    hop = max(1, m - xf)
    out = np.zeros(n, dtype=np.float64)
    weight = np.zeros(n, dtype=np.float64)
    for start in range(0, n, hop):
        length = min(m, n - start)
        if length <= 0:
            break
        seg = values[:length]
        w = np.ones(length, dtype=np.float64)
        if start > 0:
            k = min(xf, length)
            w[:k] = np.linspace(0.0, 1.0, k, endpoint=False)
        out[start:start + length] += seg * w
        weight[start:start + length] += w
        if start + length >= n:
            break
    return out / np.maximum(weight, 1e-9)


def _loop_rows(mat: np.ndarray, n: int, crossfade: int = 3) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float64)
    m = arr.shape[0]
    if n <= 0 or m == 0:
        return np.zeros((max(0, n), arr.shape[1]), dtype=arr.dtype)
    if m == 1:
        return np.repeat(arr, n, axis=0)
    xf = max(1, min(crossfade, m // 3))
    hop = max(1, m - xf)
    out = np.zeros((n, arr.shape[1]), dtype=np.float64)
    weight = np.zeros(n, dtype=np.float64)
    for start in range(0, n, hop):
        length = min(m, n - start)
        if length <= 0:
            break
        seg = arr[:length]
        w = np.ones(length, dtype=np.float64)
        if start > 0:
            k = min(xf, length)
            w[:k] = np.linspace(0.0, 1.0, k, endpoint=False)
        out[start:start + length] += seg * w[:, None]
        weight[start:start + length] += w
        if start + length >= n:
            break
    return out / np.maximum(weight, 1e-9)[:, None]


def _stretch_series(values: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if values.size <= 1:
        return _interp_series(values, n)
    # 中度拉伸用插值，保留中段内部自然过渡；极限长音用带交叉淡化的循环
    if n <= values.size * 2.5:
        return _interp_series(values, n)
    return _loop_series(values, n)


def _stretch_rows(mat: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, mat.shape[1]), dtype=mat.dtype)
    if mat.shape[0] <= 1:
        return _interp_rows(mat, n)
    if n <= mat.shape[0] * 2.5:
        return _interp_rows(mat, n)
    return _loop_rows(mat, n)


def _stretch_vowel_params(
    f0_v: np.ndarray,
    sp_v: np.ndarray,
    ap_v: np.ndarray,
    n_target: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """只拉伸元音中段；音头、尾音与辅音保持原始时值。"""
    n_v = int(len(f0_v))
    n_target = max(1, int(n_target))
    if n_v <= 0:
        return (
            np.full(n_target, 0.0, dtype=np.float64),
            np.zeros((n_target, sp_v.shape[1]), dtype=sp_v.dtype),
            np.zeros((n_target, ap_v.shape[1]), dtype=ap_v.dtype),
        )
    if n_v < 2 or n_target < 4:
        return (
            _interp_series(f0_v, n_target),
            _interp_rows(sp_v, n_target),
            _interp_rows(ap_v, n_target),
        )

    a, m, r = _split_source_vowel(n_v)
    ta, tm, tr = _split_target_vowel(n_target, a, r)
    if m <= 0:
        return (
            _interp_series(f0_v, n_target),
            _interp_rows(sp_v, n_target),
            _interp_rows(ap_v, n_target),
        )

    # 音头 / 尾音：目标更长时原样保留，目标更短时才压缩
    def resize_1d(seg: np.ndarray, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros(0, dtype=np.float64)
        if n == len(seg):
            return np.asarray(seg, dtype=np.float64)
        return _interp_series(seg, n)

    def resize_rows(seg: np.ndarray, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, seg.shape[1]), dtype=seg.dtype)
        if n == len(seg):
            return seg
        return _interp_rows(seg, n)

    f0_parts = [
        resize_1d(f0_v[:a], ta),
        _stretch_series(f0_v[a:a + m], tm),
        resize_1d(f0_v[n_v - r:], tr),
    ]
    sp_parts = [
        resize_rows(sp_v[:a], ta),
        _stretch_rows(sp_v[a:a + m], tm),
        resize_rows(sp_v[n_v - r:], tr),
    ]
    ap_parts = [
        resize_rows(ap_v[:a], ta),
        _stretch_rows(ap_v[a:a + m], tm),
        resize_rows(ap_v[n_v - r:], tr),
    ]
    f0_out = np.concatenate([p for p in f0_parts if len(p)])
    sp_out = np.concatenate([p for p in sp_parts if len(p)], axis=0)
    ap_out = np.concatenate([p for p in ap_parts if len(p)], axis=0)

    # 数值兜底：保证帧数严格等于目标
    if len(f0_out) < n_target:
        pad = n_target - len(f0_out)
        f0_out = np.pad(f0_out, (0, pad), mode="edge")
        sp_out = np.pad(sp_out, ((0, pad), (0, 0)), mode="edge")
        ap_out = np.pad(ap_out, ((0, pad), (0, 0)), mode="edge")
    elif len(f0_out) > n_target:
        f0_out = f0_out[:n_target]
        sp_out = sp_out[:n_target]
        ap_out = ap_out[:n_target]
    return f0_out, sp_out, ap_out


def _apply_note_release(y: np.ndarray, sr: int) -> np.ndarray:
    """给单个音符加自然的收尾，避免长音末尾被硬切。"""
    n = len(y)
    if n <= 4:
        return y
    seconds = n / max(1, sr)
    release_sec = 0.020 + min(0.025, seconds * 0.008)
    release_n = min(int(sr * release_sec), n // 4)
    if release_n <= 1:
        return y
    env = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, release_n)))
    y[-release_n:] *= env[:, None].astype(np.float32)
    return y


def synthesize_note(
    preproc: list[dict],
    target_pitch: int,
    dur_samples: int,
    sr: int = SR,
    frame_period: float = FRAME_PERIOD,
    ap_scale: float = 0.05,
    sp_scale: float = 1.0,
    sample_id: str = "",
) -> np.ndarray:
    """用 WORLD 合成单个音符，返回 ``(dur_samples, 2) float32``。

    ``sample_id`` 用于判断辅音区与爆破音字；不传时保持旧行为（仅合成），
    方便单元测试直接验证 WORLD 结果。
    """
    dur_samples = int(dur_samples)
    if dur_samples <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    target_f0 = 440.0 * 2.0 ** ((int(target_pitch) - 69) / 12.0)
    total_frames_target = max(1, int(round(dur_samples / sr * 1000.0 / frame_period)))

    channels: list[np.ndarray] = []
    for ch in range(2):
        orig = preproc[ch]
        f0_orig = orig["f0"]
        is_vowel = orig["is_vowel"]

        voiced = f0_orig[f0_orig > 0]
        if len(voiced) > 0:
            factor = target_f0 / float(np.mean(voiced))
        else:
            factor = 1.0
        f0_new = f0_orig.copy()
        f0_new[f0_orig > 0] *= factor

        vowel_idx = np.where(is_vowel)[0]
        cons_idx = np.where(~is_vowel)[0]

        f0_v = f0_new[vowel_idx]
        f0_u = f0_new[cons_idx]
        n_u = len(f0_u)
        n_v_target = max(1, total_frames_target - n_u)

        sp = orig["sp"]
        ap = orig["ap"]
        sp_u = sp[cons_idx] * sp_scale
        ap_u = ap[cons_idx] * ap_scale
        sp_v = sp[vowel_idx]
        ap_v = ap[vowel_idx]

        # 关键：只拉伸元音中段，辅音 / 音头 / 尾音时值不变
        f0_v_new, sp_v_new, ap_v_new = _stretch_vowel_params(
            f0_v, sp_v, ap_v, n_v_target
        )
        sp_v_new = sp_v_new * sp_scale
        ap_v_new = ap_v_new * ap_scale

        f0_final = np.concatenate([f0_u, f0_v_new]).astype(np.float64)
        sp_final = np.concatenate([sp_u, sp_v_new]).astype(np.float64)
        ap_final = np.concatenate([ap_u, ap_v_new]).astype(np.float64)
        sp_final = _shape_spectrum(sp_final, target_pitch, sr)

        if len(f0_final) < 2:
            channels.append(np.zeros(dur_samples, dtype=np.float32))
            continue

        y = pw.synthesize(f0_final, sp_final, ap_final, sr, frame_period)
        if len(y) > dur_samples:
            y = y[:dur_samples]
        elif len(y) < dur_samples:
            y = np.pad(y, (0, dur_samples - len(y)))
        channels.append(y.astype(np.float32, copy=False))

    out = np.stack(channels, axis=1)
    out = _audibility_post(out, target_pitch, sr)
    if sample_id:
        # 所有辅音统一降低；基 / 库 / 叮 / 咚 等爆破音开头再重压。
        out = attenuate_consonants(
            out,
            sr,
            cons_frames_for(sample_id),
            frame_period,
            plosive=is_plosive_sample(sample_id),
        )
    out = _apply_note_release(out, sr)
    return out.astype(np.float32, copy=False)


def synthesize_note_from_char(
    char: str,
    target_pitch: int,
    dur_samples: int,
    **kwargs,
) -> np.ndarray:
    """便捷接口：按字符取预处理缓存后合成。

    别名（如曼波「夏」→「哈」）也先解析成实际采样，保证与主渲染链路
    使用相同的辅音帧数 / 爆破音判定。
    """
    sample_id = resolve_sample_id(char)
    preproc = get_preprocessed(sample_id)
    kwargs.setdefault("sample_id", sample_id)
    return synthesize_note(preproc, target_pitch, dur_samples, **kwargs)


__all__ = [
    "available_sample_chars",
    "PLOSIVE_CHARS",
    "choose_sample",
    "cons_frames_for",
    "get_preprocessed",
    "is_plosive_sample",
    "preprocess_chars",
    "preprocess_sample",
    "preproc_cache_path",
    "resolve_sample_id",
    "sample_exists",
    "sample_path_for",
    "synthesize_note",
    "synthesize_note_from_char",
    "vowel_stretch_plan",
]
