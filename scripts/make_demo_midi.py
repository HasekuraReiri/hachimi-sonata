"""生成一个用于测试 / 演示的多轨道 MIDI。

包含：
* 旋律轨：重复出现的动机（移调出现），不同力度；
* 和声轨：柱式和弦；
* 低音轨：分解和弦（跨声部分解和弦测试）；
* 鼓轨：默认应被静音。

运行::

    python scripts/make_demo_midi.py
    python scripts/make_demo_midi.py --out assets/midi/demo.mid
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pretty_midi

# 4 小节和弦进行：C - Am - F - G
CHORDS = [
    (48, [60, 64, 67]),   # C
    (45, [57, 60, 64]),   # Am
    (41, [53, 57, 60]),   # F
    (43, [55, 59, 62]),   # G
]

# 动机：音高序列 + 时值（以拍为单位）
MOTIF = [(0, 0.5), (2, 0.5), (4, 0.5), (7, 0.5), (4, 0.5), (2, 0.5)]


def build_demo(path: str, bars: int = 8, bpm: float = 100.0) -> str:
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    beat = 60.0 / bpm
    bar = beat * 4

    melody = pretty_midi.Instrument(program=0, name="Melody")
    harmony = pretty_midi.Instrument(program=48, name="Strings")
    bass = pretty_midi.Instrument(program=42, name="Cello")
    drums = pretty_midi.Instrument(program=0, name="Drums", is_drum=True)

    transpositions = [0, 5, 7, 5, 0, 7, 5, 0]
    velocities = [92, 108, 120, 76, 100, 116, 68, 88]
    for b in range(bars):
        t0 = b * bar
        chord_idx = b % len(CHORDS)
        root, pcs = CHORDS[chord_idx]
        tr = transpositions[b % len(transpositions)]

        # ---- 旋律：动机 ----
        t = t0
        for i, (interval, dur_beats) in enumerate(MOTIF):
            pitch = 64 + tr + interval
            melody.notes.append(
                pretty_midi.Note(
                    velocity=velocities[(b + i) % len(velocities)],
                    pitch=pitch,
                    start=t,
                    end=t + dur_beats * beat * 0.92,
                )
            )
            t += dur_beats * beat
        # 每小节后半：长音
        melody.notes.append(
            pretty_midi.Note(
                velocity=velocities[b % len(velocities)],
                pitch=64 + tr,
                start=t0 + 2.0 * beat,
                end=t0 + 3.8 * beat,
            )
        )

        # ---- 和声：柱式和弦（每两拍换一次）----
        for half in range(2):
            start = t0 + half * 2 * beat
            for pc in pcs:
                harmony.notes.append(
                    pretty_midi.Note(
                        velocity=64,
                        pitch=pc - 12,
                        start=start,
                        end=start + 1.9 * beat,
                    )
                )

        # ---- 低音：分解和弦（同一个和弦的琶音，跨小节/声部）----
        arp = [root, root + 7, root + 12, root + 16, root + 12, root + 7]
        for i, pitch in enumerate(arp):
            start = t0 + i * 0.5 * beat
            bass.notes.append(
                pretty_midi.Note(
                    velocity=80 if i % 2 == 0 else 62,
                    pitch=pitch,
                    start=start,
                    end=start + 0.45 * beat,
                )
            )

        # ---- 鼓 ----
        for beat_i in range(4):
            start = t0 + beat_i * beat
            drums.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=start, end=start + 0.1))
            drums.notes.append(
                pretty_midi.Note(velocity=70, pitch=42, start=start + 0.5 * beat, end=start + 0.55 * beat)
            )

    for inst in (melody, harmony, bass, drums):
        inst.notes.sort(key=lambda n: (n.start, n.pitch))
        pm.instruments.append(inst)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pm.write(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="生成演示用多轨 MIDI")
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "assets", "midi", "_demo", "demo_chorale.mid",
        ),
    )
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--bpm", type=float, default=100.0)
    args = parser.parse_args()
    path = build_demo(args.out, bars=args.bars, bpm=args.bpm)
    print(f"已生成：{path}")
    pm = pretty_midi.PrettyMIDI(path)
    print(f"时长 {pm.get_end_time():.2f}s，{len(pm.instruments)} 轨")
    for i, inst in enumerate(pm.instruments):
        print(f"  [{i}] {inst.name}: {len(inst.notes)} 音符, drum={inst.is_drum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
