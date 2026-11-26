"""MIDI 合成兼容层。

上游的 ``synthesize_midi`` 直接逐音符调用 WORLD，速度较慢；新实现位于
:mod:`hachimi.render`。本模块保留旧函数签名，内部转调新渲染器，
方便旧脚本 / 外部代码继续工作。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pretty_midi

from hachimi.config import FRAME_PERIOD, SR, VELOCITY_GAMMA
from hachimi.render import RenderNote, render_notes

logger = logging.getLogger(__name__)


def midi_info(midi: pretty_midi.PrettyMIDI) -> list[dict[str, Any]]:
    """返回轨道信息摘要。"""
    from hachimi.config import instrument_name

    return [
        {
            "index": i,
            "name": inst.name or f"Track {i}",
            "program": inst.program,
            "instrument": instrument_name(inst.program, inst.is_drum),
            "is_drum": inst.is_drum,
            "n_notes": len(inst.notes),
        }
        for i, inst in enumerate(midi.instruments)
    ]


def synthesize_midi(
    preproc: dict[str, Any] | None,
    note2word: list[dict],
    midi: pretty_midi.PrettyMIDI,
    sr: int = SR,
    frame_period: float = FRAME_PERIOD,
    start_time: float = 0.0,
    end_time: float | None = None,
    octave_shift: list[int] | None = None,
    volume_factor: list[float] | None = None,
    ap_scale: float = 0.05,
    sp_scale: float = 1.0,
) -> np.ndarray:
    """兼容旧接口：把 MIDI 渲染为立体声 numpy 数组。

    ``preproc`` / ``ap_scale`` / ``sp_scale`` 参数为兼容保留，内部由
    :mod:`hachimi.synthesis` 的磁盘缓存统一管理。
    """
    n_tracks = len(midi.instruments)
    octave_shift = octave_shift or [0] * n_tracks
    volume_factor = volume_factor if volume_factor is not None else [1.0] * n_tracks

    notes: list[RenderNote] = []
    for idx, inst in enumerate(midi.instruments):
        if idx >= len(volume_factor) or volume_factor[idx] == 0:
            continue
        shift = octave_shift[idx] if idx < len(octave_shift) else 0
        mapping = note2word[idx] if idx < len(note2word) else {}
        for note in inst.notes:
            word = mapping.get(note)
            if not word:
                continue
            notes.append(
                RenderNote(
                    pitch=int(note.pitch),
                    start=float(note.start),
                    end=float(note.end),
                    velocity=int(note.velocity),
                    track=idx,
                    word=str(word),
                    gain=float(volume_factor[idx]),
                    shift=int(shift),
                )
            )

    audio, _ = render_notes(
        notes,
        sr=sr,
        frame_period=frame_period,
        velocity_gamma=VELOCITY_GAMMA,
        duration_quantum=0.02,
    )
    total_time = midi.get_end_time() if end_time is None else end_time
    n_samples = int(total_time * sr)
    if len(audio) < n_samples:
        audio = np.pad(audio, ((0, n_samples - len(audio)), (0, 0)))
    elif len(audio) > n_samples:
        audio = audio[:n_samples]
    return audio


__all__ = ["midi_info", "synthesize_midi"]
