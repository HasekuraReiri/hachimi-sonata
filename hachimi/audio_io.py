"""音频输出：默认编码为 MP3（失败时回退 WAV）。

``soundfile`` 的 libsndfile 在绝大多数平台已内置 MP3 编码；若当前运行环境
不支持，会再尝试 ``ffmpeg``，最后才回退为 WAV，保证生成流程不中断。
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from hachimi.config import MP3_SUBTYPE, SR, TEMP_DIR

logger = logging.getLogger(__name__)


def _ffmpeg_mp3(wav_path: str, mp3_path: str) -> bool:
    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    try:
        subprocess.run(
            [exe, "-y", "-loglevel", "error", "-i", wav_path, "-b:a", "320k", mp3_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0
    except Exception as exc:  # pragma: no cover
        logger.warning("ffmpeg 转 MP3 失败：%s", exc)
        return False


def write_audio(path_without_ext: str, audio: np.ndarray, sr: int = SR, fmt: str = "mp3") -> str:
    """写出音频，返回实际文件路径（自动附加扩展名）。

    Args:
        path_without_ext: 不含扩展名的目标路径。
        audio: ``(N,)`` 或 ``(N, 2)`` 音频。
        sr: 采样率。
        fmt: ``"mp3"``（默认）、``"wav"`` 或 ``"flac"``。
    """
    fmt = (fmt or "mp3").lower()
    if fmt not in ("mp3", "wav", "flac"):
        fmt = "mp3"
    os.makedirs(os.path.dirname(os.path.abspath(path_without_ext)), exist_ok=True)

    if fmt == "flac":
        path = path_without_ext + ".flac"
        sf.write(path, audio, sr, format="FLAC", subtype="PCM_16")
        return path

    if fmt == "wav":
        path = path_without_ext + ".wav"
        sf.write(path, audio, sr, subtype="PCM_16")
        return path

    mp3_path = path_without_ext + ".mp3"
    try:
        sf.write(mp3_path, audio, sr, format="MP3", subtype=MP3_SUBTYPE)
        if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
            return mp3_path
    except Exception as exc:
        logger.warning("libsndfile MP3 编码失败：%s", exc)

    # 回退 1：临时 WAV → ffmpeg MP3（临时文件放项目盘，避免占 C 盘）
    os.makedirs(TEMP_DIR, exist_ok=True)
    fd, tmp_wav = tempfile.mkstemp(prefix="hachimi_mp3_", suffix=".wav", dir=TEMP_DIR)
    os.close(fd)
    try:
        sf.write(tmp_wav, audio, sr, subtype="PCM_16")
        if _ffmpeg_mp3(tmp_wav, mp3_path):
            return mp3_path
    finally:
        try:
            os.remove(tmp_wav)
        except OSError:
            pass

    # 回退 2：WAV
    logger.warning("无法编码 MP3，回退为 WAV")
    wav_path = path_without_ext + ".wav"
    sf.write(wav_path, audio, sr, subtype="PCM_16")
    return wav_path


__all__ = ["write_audio"]
