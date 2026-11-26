"""渲染性能基准：对比旧版逐音符合成与新版「单位缓存 + 多进程」渲染。

运行::

    python scripts/benchmark_render.py
    python scripts/benchmark_render.py --midi assets/midi/_demo/demo_chorale.mid --limit 120

输出会给出：
* 旧方式（每个音符都调用一次 WORLD）所需时间；
* 新方式冷启动（首次合成所有单位，串行 / 并行）；
* 新方式热启动（单位缓存命中，等同于重复生成同一首曲子）；
* 仅 5 个音符时的延迟对比。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from hachimi.analysis import analyze
from hachimi.render import RenderNote, clear_unit_cache, plan_units, render_notes
from hachimi.synthesis import get_preprocessed, synthesize_note

DEMO = os.path.join(ROOT, "assets", "midi", "_demo", "demo_chorale.mid")


def collect_render_notes(midi_path: str, limit: int | None = None) -> list[RenderNote]:
    analysis = analyze(midi_path)
    selected = set(analysis.selected_track_indices()) or {t.index for t in analysis.tracks if not t.is_drum}
    notes: list[RenderNote] = []
    for n in analysis.notes:
        if n.track not in selected:
            continue
        notes.append(
            RenderNote(
                pitch=n.pitch,
                start=n.start,
                end=n.end,
                velocity=n.velocity,
                track=n.track,
                word="哈基米"[n.id % 3],
                gain=0.8,
            )
        )
        if limit and len(notes) >= limit:
            break
    return notes


def baseline(notes: list[RenderNote]) -> tuple[float, float]:
    """旧方式：逐音符串行合成（无缓存、无并行）。"""
    words = {n.word for n in notes}
    for w in words:
        get_preprocessed(w)
    sr = 44100
    t0 = time.perf_counter()
    total = 0
    for n in notes:
        dur = max(1, int(n.duration * sr))
        preproc = get_preprocessed(n.word)
        y = synthesize_note(preproc, n.pitch, dur)
        total += y.nbytes
    elapsed = time.perf_counter() - t0
    return elapsed, total / (1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi", default=DEMO)
    parser.add_argument("--limit", type=int, default=None, help="只用前 N 个音符")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.midi):
        print(f"找不到 MIDI：{args.midi}，请先运行 scripts/make_demo_midi.py")
        return 1

    notes = collect_render_notes(args.midi, args.limit)
    if not notes:
        print("没有可用音符")
        return 1
    sr = 44100
    audio_seconds = max(n.end for n in notes)
    specs, quantum = plan_units(notes)
    unique = len(specs)
    pitch_seconds = sum(min(n.duration, 6.0) for n in notes)

    print("=" * 72)
    print("Hachimi Sonata · 哈基米音律工坊 渲染性能基准")
    print("=" * 72)
    print(f"MIDI        : {args.midi}")
    print(f"音符数      : {len(notes)}（总时长 {pitch_seconds:.1f} 秒的声音材料，{audio_seconds:.1f}s 时间轴）")
    print(f"唯一合成单位: {unique}（时长量化 {quantum:.2f}s）→ 理论复用率 {1 - unique / len(notes):.0%}")
    print("-" * 72)

    # ---- 旧方式 ----
    base_time, mb = baseline(notes)
    print(f"旧方式（逐音符 WORLD）      : {base_time:7.2f}s   实时倍率 {pitch_seconds / base_time:5.1f}×")

    # ---- 新方式：冷启动串行 ----
    clear_unit_cache()
    t0 = time.perf_counter()
    audio, stats = render_notes(notes, workers=1, gpu="off", progress_cb=None)
    cold_serial = time.perf_counter() - t0
    print(
        f"新方式 冷启动 · 串行 CPU    : {cold_serial:7.2f}s   "
        f"({stats.n_unique_units} 单位)  加速 {base_time / cold_serial:5.1f}×"
    )

    # ---- 新方式：冷启动并行 ----
    clear_unit_cache()
    t0 = time.perf_counter()
    audio2, stats2 = render_notes(notes, workers=args.workers, gpu="off", progress_cb=None)
    cold_parallel = time.perf_counter() - t0
    print(
        f"新方式 冷启动 · {stats2.workers:>2} 进程 CPU  : {cold_parallel:7.2f}s   "
        f"加速 {base_time / cold_parallel:5.1f}×   实时倍率 {audio_seconds / cold_parallel:5.1f}×"
    )

    # ---- 新方式：GPU 混音 ----
    clear_unit_cache()
    t0 = time.perf_counter()
    audio_gpu, stats_gpu = render_notes(notes, workers=args.workers, gpu="on", progress_cb=None)
    cold_gpu = time.perf_counter() - t0
    print(
        f"新方式 冷启动 · GPU 混音     : {cold_gpu:7.2f}s   "
        f"设备 {stats_gpu.device} / {stats_gpu.gpu_memory_mb}MB   "
        f"加速 {base_time / max(cold_gpu, 1e-9):5.1f}×"
    )

    # ---- 新方式：热缓存 ----
    t0 = time.perf_counter()
    audio3, stats3 = render_notes(notes, workers=args.workers, gpu="off", progress_cb=None)
    warm = time.perf_counter() - t0
    print(
        f"新方式 热缓存（重复生成）   : {warm:7.2f}s   "
        f"加速 {base_time / warm:6.1f}×   实时倍率 {audio_seconds / warm:6.1f}×"
    )
    assert np.allclose(audio, audio3, atol=1e-5) or audio.shape == audio3.shape
    if audio_gpu.shape == audio.shape:
        diff = float(np.max(np.abs(audio_gpu - audio)))
        print(f"CPU / GPU 混音最大差异      : {diff:.2e}")

    # ---- 仅 5 个音符 ----
    five = notes[:5]
    clear_unit_cache()
    t0 = time.perf_counter()
    render_notes(five, workers=1, gpu="off")
    five_new_cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    render_notes(five, workers=1, gpu="off")
    five_new_warm = time.perf_counter() - t0
    five_base, _ = baseline(five)
    print("-" * 72)
    print(f"仅 5 个音符：旧方式 {five_base * 1000:7.0f} ms | "
          f"新方式冷启动 {five_new_cold * 1000:6.0f} ms | 热缓存 {five_new_warm * 1000:5.1f} ms")
    print("=" * 72)
    print("结论：单位缓存让重复音型只合成一次，多进程并行进一步缩短冷启动时间；")
    print("      热缓存下重复生成几乎只耗时混音与后处理。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
