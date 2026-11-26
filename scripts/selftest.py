"""项目自检脚本：验证核心功能的关键行为。

运行::

    python scripts/selftest.py

覆盖：
1. 和弦识别（C / Am / G7）；
2. 调性判断、每段编曲计划（默认 70% 马尔科夫 / 哈基米主导 / 每段独立种子 /
   段间明显对比）与手动覆盖；
3. GM 小提琴默认音量 1.50；
4. 跨声部「分解和弦」共用同一个歌词字；
5. 同一动机重复出现时复用同一串歌词；
6. MIDI 力度精确映射到音量（velocity^gamma），并按相邻力度渐强 / 渐弱；
7. 音源强压缩 / 补偿后的稳定响度，以及极端高音的可听度补偿；
8. 齿音 / 清辅音分析衰减（噪声性高频与无周期辅音）；
9. 时值拉伸只拉元音中段，长音连贯、音头清晰、结尾自然；
10. 弱化音头 / 连奏（前音延伸、后音柔起音、和弦不误连）；
11. 绿幕字幕视频（宽 1920 / 单条全部字幕 / 每次 1 行 / 纯绿 / 白字 / 30fps）；
12. 自动混音的真实乐器摆位、相位反相修正与母带响度；
13. 总线压缩：3.4:1 向下压缩 + 自动补偿 + 弱段向上压缩，强弱差收窄且峰值安全；
14. 实时字幕（行不重叠、SRT/LRC/HTML 正常输出）；
15. 端到端渲染（音频 / 歌词 / 字幕 / 分析 / 混音前干声 / 参数包 / 视频齐全）；
16. 默认语音模式同时包含哈基米与辅助音色；
17. 断奏 gate 不会被和弦发声校验误判；
18. 辅音 / 爆破音衰减（基 / 库 起音约 -16dB，串行 / 并行一致，元音不变）；
19. 不良音符判定与只含不良音的 MIDI；
20. 工程参数包往返与旧 manifest 复用；
21. 开源版无 LLM 代码 / 入口 / 界面，源码扫描无 API key。
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import soundfile as sf

from hachimi.analysis import (
    NoteEvent,
    Section,
    analyze,
    build_lyric_slots,
    detect_chord,
    detect_chords,
    detect_key,
)
from hachimi.arrangement import (
    AUTO_MARKOV_RANGE,
    build_section_plans,
    normalize_source_weights,
)
from hachimi.config import RenderOptions
from hachimi.lyrics import compose_lyrics
from hachimi.models import create_model
from hachimi.pipeline import render_midi, resolve_track_options
from hachimi.render import RenderNote, apply_legato, render_notes, soft_attack_envelope
from hachimi.subtitles import build_subtitles, sec_to_srt, write_html, write_lrc, write_srt
from hachimi.video import find_ffmpeg, measure_layout, probe_video, render_subtitle_video

DEMO = os.path.join(ROOT, "assets", "midi", "_demo", "demo_chorale.mid")
SCORE = os.path.join(ROOT, "测试", "爱之死测试.xml")
OUT = os.path.join(ROOT, "output", "_selftest")

PASS = 0
FAIL = 0


def check(name: str, fn) -> None:
    global PASS, FAIL
    t0 = time.time()
    try:
        detail = fn()
        PASS += 1
        print(f"  [PASS] {name}  ({time.time() - t0:.2f}s) {detail or ''}")
    except Exception as exc:
        FAIL += 1
        print(f"  [FAIL] {name}  ({time.time() - t0:.2f}s) -> {exc}")


# ---------------------------------------------------------------------------


def test_chords() -> str:
    root, quality, label, tones, conf = detect_chord([60, 64, 67])
    assert (root, quality) == (0, ""), (root, quality, label)
    assert label == "C", label
    root2, q2, label2, _, _ = detect_chord([57, 60, 64])
    assert (root2, q2) == (9, "m"), (root2, q2, label2)
    root3, q3, label3, _, _ = detect_chord([55, 59, 62, 65])
    assert q3 == "7", (label3, q3)
    return "C / Am / G7 识别正确"


def test_decomposed_slots() -> str:
    # 三个声部依次奏出 C-E-G（同一个和弦的分解），外加低音 C
    notes = [
        NoteEvent(id=0, track=0, index=0, pitch=60, start=0.00, end=0.4, velocity=80),
        NoteEvent(id=1, track=1, index=0, pitch=64, start=0.12, end=0.4, velocity=80),
        NoteEvent(id=2, track=2, index=0, pitch=67, start=0.24, end=0.4, velocity=80),
        NoteEvent(id=3, track=3, index=0, pitch=48, start=0.00, end=0.4, velocity=70),
    ]
    chords = detect_chords(notes)
    slots = build_lyric_slots(notes, chords, decomposed_merge=True)
    merged = [s for s in slots if s.is_decomposed]
    assert merged, "没有检测到分解和弦槽"
    assert len(merged[0].track_ids) >= 2, merged[0].track_ids
    assert set(merged[0].note_ids) == {0, 1, 2, 3}, merged[0].note_ids

    slots_no = build_lyric_slots(notes, chords, decomposed_merge=False)
    assert not any(s.is_decomposed for s in slots_no)
    return f"分解和弦合并为 1 个槽（{len(merged[0].track_ids)} 个声部）"


def test_analysis_demo():
    result = analyze(DEMO)
    assert len(result.tracks) == 4, len(result.tracks)
    assert not result.tracks[3].suggested_selected, "鼓轨应默认静音"
    assert result.tracks[0].suggested_selected
    assert len(result.chords) >= 8, len(result.chords)
    assert sum(1 for c in result.chords if c.is_decomposed) > 0, "未标记分解和弦"
    assert len(result.motifs) >= 1, "未检测到动机"
    assert len(result.sections) >= 1
    return (
        f"{len(result.tracks)} 轨 / {len(result.chords)} 和弦 / "
        f"{len(result.sections)} 段 / {len(result.motifs)} 动机"
    )


def test_key_and_section_plans() -> str:
    """调性判断 + 每段默认 70% 马尔科夫 + 哈基米主导 + 手动覆盖。"""
    c_major = [
        NoteEvent(id=i, track=0, index=i, pitch=p, start=i * 0.5, end=i * 0.5 + 0.5, velocity=80)
        for i, p in enumerate([60, 62, 64, 65, 67, 69, 71, 72])
    ]
    a_minor = [
        NoteEvent(id=i, track=0, index=i, pitch=p, start=i * 0.5, end=i * 0.5 + 0.5, velocity=80)
        for i, p in enumerate([57, 59, 60, 62, 64, 65, 67, 69])
    ]
    assert detect_key(c_major)["label"] == "C major", detect_key(c_major)
    assert detect_key(a_minor)["label"] == "A minor", detect_key(a_minor)
    from hachimi.analysis import roman_numeral

    assert roman_numeral({"root": 5, "quality": ""}, 0) == "IV"
    assert roman_numeral({"root": 9, "quality": "m"}, 0) == "vi"
    assert roman_numeral({"root": 7, "quality": "7"}, 0) == "V7"

    analysis = analyze(DEMO)
    sections = [
        Section(0, "A", 0.0, analysis.duration / 2, analysis.stats()["n_notes"], 0.5),
        Section(1, "B", analysis.duration / 2, analysis.duration, analysis.stats()["n_notes"], 0.7),
    ]
    plans = build_section_plans(sections, base_markov=0.7, base_seed=42, duration=analysis.duration)
    assert len(plans) == 2
    assert all(AUTO_MARKOV_RANGE[0] <= p["markov_ratio"] <= AUTO_MARKOV_RANGE[1] for p in plans)
    assert max(p["markov_ratio"] for p in plans) - min(p["markov_ratio"] for p in plans) >= 0.06
    assert 0.55 <= sum(p["markov_ratio"] for p in plans) / 2 <= 0.85
    assert len({p["seed"] for p in plans}) == 2
    for plan in plans:
        weights = normalize_source_weights(plan["source_weights"])
        assert weights["哈基米"] == max(weights.values()), weights
        assert weights["叮咚鸡"] >= weights["曼巴"], weights

    # 多段时差异必须明显大于旧版 ±5% 的轻微抖动。
    contrast_sections = [
        Section(i, chr(65 + i), i * 10.0, (i + 1) * 10.0, 100, energy)
        for i, energy in enumerate([0.2, 0.8, 0.3, 0.7, 0.4, 0.9])
    ]
    contrast_plans = build_section_plans(
        contrast_sections, base_markov=0.7, base_seed=42, duration=60.0
    )
    contrast = max(p["markov_ratio"] for p in contrast_plans) - min(
        p["markov_ratio"] for p in contrast_plans
    )
    assert contrast >= 0.15, contrast
    hachimi_weights = [normalize_source_weights(p["source_weights"])["哈基米"] for p in contrast_plans]
    assert max(hachimi_weights) - min(hachimi_weights) >= 0.10, hachimi_weights
    manual = build_section_plans(
        sections,
        base_markov=0.7,
        base_seed=42,
        duration=analysis.duration,
        overrides=[
            {
                "section_index": 0,
                "markov_ratio": 0.88,
                "seed": 12345,
                "source_weights": {"哈基米": 0.7, "叮咚鸡": 0.2, "曼巴": 0.07, "曼波only": 0.03},
                "auto": False,
            },
            {"section_index": 1, "markov_ratio": 0.55, "seed": 54321, "auto": False},
        ],
    )
    assert manual[0]["markov_ratio"] == 0.88 and manual[0]["seed"] == 12345
    assert manual[1]["markov_ratio"] == 0.55 and manual[1]["seed"] == 54321

    # 让歌词生成真正按 A/B 两段运行，检查每段计划被应用。
    analysis.sections = sections
    options = RenderOptions()
    tracks = resolve_track_options(analysis, options)
    for track in tracks:
        track["selected"] = track["index"] in (0, 1, 2)
        track["voice"] = "auto"
    models = {
        name: create_model("markov", [name])
        for name in ("哈基米", "叮咚鸡", "曼巴", "曼波only")
    }
    lyrics = compose_lyrics(
        analysis,
        models,
        {t["index"]: t for t in tracks},
        seed=42,
        default_voice="auto",
        section_plans=manual,
    )
    stats = lyrics.stats
    assert len(stats["section_stats"]) == 2, stats["section_stats"]
    assert stats["section_stats"][0]["markov_ratio"] == 0.88
    assert stats["section_stats"][1]["markov_ratio"] == 0.55
    assert stats["voices"].get("哈基米", 0) > max(
        [v for k, v in stats["voices"].items() if k != "哈基米"] or [0]
    )
    return "C 大调 / A 小调识别正确；两段计划独立应用，哈基米主导"


def test_violin_default_gain() -> str:
    """GM 小提琴（program 40）默认音量必须是 1.5。"""
    import pretty_midi

    path = os.path.join(OUT, "_violin.mid")
    os.makedirs(OUT, exist_ok=True)
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=40, name="Violin")
    for i, pitch in enumerate([72, 74, 76, 77, 79, 81, 79, 76]):
        inst.notes.append(
            pretty_midi.Note(velocity=90, pitch=pitch, start=i * 0.5, end=i * 0.5 + 0.45)
        )
    pm.instruments.append(inst)
    pm.write(path)
    analysis = analyze(path)
    track = analysis.tracks[0]
    assert track.instrument == "Violin", track.instrument
    assert abs(track.suggested_gain - 1.5) < 1e-9, track.suggested_gain
    assert "1.50" in track.reason, track.reason
    resolved = resolve_track_options(analysis, RenderOptions())[0]
    assert abs(float(resolved["gain"]) - 1.5) < 1e-9, resolved["gain"]
    return f"Violin（轨道 {track.index}）默认音量 {track.suggested_gain:.2f}"


def _build_lyrics():
    result = analyze(DEMO)
    options = RenderOptions()
    tracks = resolve_track_options(result, options)
    for t in tracks:
        t["selected"] = t["index"] != 3
        t["voice"] = "哈基米"
    track_options = {t["index"]: t for t in tracks}
    model = create_model("markov", ["哈基米"])
    lyrics = compose_lyrics(
        result,
        {"哈基米": model},
        track_options,
        default_voice="哈基米",
        decomposed_merge=True,
    )
    return result, lyrics
    result = analyze(DEMO)
    options = RenderOptions()
    tracks = resolve_track_options(result, options)
    for t in tracks:
        t["selected"] = t["index"] != 3
        t["voice"] = "哈基米"
    track_options = {t["index"]: t for t in tracks}
    model = create_model("markov", ["哈基米"])
    lyrics = compose_lyrics(
        result,
        {"哈基米": model},
        track_options,
        default_voice="哈基米",
        decomposed_merge=True,
    )
    return result, lyrics


def test_cross_voice_same_word() -> str:
    _, lyrics = _build_lyrics()
    note2word = lyrics.note2word
    decomposed = [s for s in lyrics.slots if s.is_decomposed and len(set(s.track_ids)) >= 2]
    assert decomposed, "没有跨声部分解和弦槽"
    for slot in decomposed:
        words = {note2word[nid] for nid in slot.note_ids if nid in note2word}
        assert len(words) == 1, f"槽 {slot.index} 出现多个字：{words}"
    return f"{len(decomposed)} 个跨声部分解和弦槽全部同字"


def test_motif_same_lyrics() -> str:
    analysis, lyrics = _build_lyrics()
    applied = set(lyrics.motif_reuse)
    assert applied, "没有任何动机被应用"
    checked = 0
    for motif in analysis.motifs:
        if motif.id not in applied:
            continue
        seqs = []
        for occ in motif.occurrences:
            seq = [lyrics.note2word.get(int(nid)) for nid in occ.get("ids", [])]
            if any(w is None for w in seq):
                seq = None
            seqs.append(seq)
        seqs = [s for s in seqs if s]
        if len(seqs) < 2:
            continue
        assert all(s == seqs[0] for s in seqs), f"动机 {motif.id} 歌词不一致：{seqs}"
        checked += 1
    assert checked > 0, "没有可校验的动机复用"
    return f"{checked} 个动机的多次出现逐音歌词完全一致"


def test_velocity_gain() -> str:
    gamma = 1.6
    notes = [
        RenderNote(pitch=64, start=0.0, end=1.0, velocity=32, track=0, word="哈", gain=1.0),
        RenderNote(pitch=64, start=2.0, end=3.0, velocity=127, track=0, word="哈", gain=1.0),
    ]
    audio, stats = render_notes(notes, workers=1, velocity_gamma=gamma)
    sr = 44100
    a = audio[int(0.1 * sr):int(0.9 * sr), :]
    b = audio[int(2.1 * sr):int(2.9 * sr), :]
    rms_a = float(np.sqrt(np.mean(a ** 2)))
    rms_b = float(np.sqrt(np.mean(b ** 2)))
    assert rms_a > 0 and rms_b > 0
    ratio = rms_b / rms_a
    expected = (127 / 32) ** gamma
    err = abs(ratio / expected - 1)
    assert err < 0.05, f"力度响度比 {ratio:.2f} 与期望 {expected:.2f} 偏差 {err:.1%}"
    return f"velocity 32→127 响度比 {ratio:.2f}（期望 {expected:.2f}）"


def test_audio_loudness_and_extreme_pitch() -> str:
    """音源强压缩后响度稳定；高音有 presence / 可听度补偿。"""
    from hachimi.config import SAMPLE_TARGET_DB
    from hachimi.loudness import (
        band_energies,
        condition_sample,
        peak_db,
        pitch_audibility_db,
        rms_db,
    )
    from hachimi.synthesis import get_preprocessed, sample_path_for, synthesize_note

    raw, _sr = sf.read(sample_path_for("哈"))
    conditioned = condition_sample(raw)
    loud = rms_db(conditioned)
    assert abs(loud - SAMPLE_TARGET_DB) < 1.5, (loud, SAMPLE_TARGET_DB)
    assert peak_db(conditioned) <= -0.5, peak_db(conditioned)

    pre = get_preprocessed("哈")
    high = synthesize_note(pre, 96, 44100)
    mid = synthesize_note(pre, 60, 44100)
    assert rms_db(high) > -23.0 and rms_db(mid) > -23.0, (rms_db(high), rms_db(mid))
    bands = band_energies(high)
    assert bands["high"] > bands["mid"], bands
    assert pitch_audibility_db(108) > pitch_audibility_db(60) + 5
    return f"采样响度 {loud:.1f}dB；C7 高频占比 {bands['high']:.1f}dB"


def test_expression_and_auto_mix() -> str:
    """力度渐强 / 渐弱 + 乐器摆位 + 反相检测。"""
    from hachimi.mixing import auto_mix_plan, channel_correlation, phase_align, place_track

    notes = [
        RenderNote(
            pitch=64, start=0.0, end=0.9, velocity=30, track=0, word="哈",
            velocity_end=60,
        ),
        RenderNote(
            pitch=64, start=1.1, end=2.0, velocity=60, track=0, word="哈",
            velocity_end=100,
        ),
        RenderNote(pitch=64, start=2.2, end=3.0, velocity=100, track=0, word="哈"),
    ]
    audio, stats = render_notes(notes, workers=1)
    assert stats.n_dynamic_notes == 2, stats.n_dynamic_notes
    sr = 44100
    windows = []
    for start in (0.1, 1.2, 2.3):
        seg = audio[int(start * sr):int((start + 0.5) * sr)]
        windows.append(float(np.sqrt(np.mean(seg ** 2))))
    assert windows[0] < windows[1] < windows[2], windows

    plan = auto_mix_plan(
        [
            {"index": 0, "instrument": "Violin", "role": "melody"},
            {"index": 1, "instrument": "Cello", "role": "harmony"},
            {"index": 2, "instrument": "Acoustic Bass", "role": "bass"},
        ]
    )
    assert plan[0].pan < -0.2 and plan[1].pan > 0.15, (plan[0].pan, plan[1].pan)
    assert abs(plan[2].pan) < 0.1, plan[2].pan

    t = np.arange(sr, dtype=np.float32) / sr
    mono = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    dry = np.stack([mono, mono], axis=1)
    placed, info = place_track(dry, plan[0], sr, auto_gain_db=0)
    assert channel_correlation(placed) > 0.2, info
    inverted = np.stack([mono, -mono], axis=1)
    fixed, flipped, corr = phase_align(inverted)
    assert flipped and corr > 0.9, (flipped, corr)
    return f"渐强 3 级 {['%.3f' % v for v in windows]}；小提琴左 / 大提琴右；反相已修正"


def test_mix_bus_compression() -> str:
    """总线压缩：更大压缩比 + 自动补偿 + 弱段提升，强音不过冲。"""
    from hachimi.loudness import bus_compress, crest_factor_db, peak_db, rms_db
    from hachimi.mixing import master_bus

    sr = 44100
    seg = int(sr * 1.6)
    t = np.arange(seg) / sr
    loud = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    quiet = 0.016 * np.sin(2 * np.pi * 330.0 * t)
    mono = np.concatenate([loud, quiet, loud, quiet]).astype(np.float32)
    signal = np.stack([mono, mono], axis=1)

    def steady(audio: np.ndarray, start: float) -> float:
        part = audio[int((start + 0.55) * sr): int((start + 1.55) * sr)]
        return rms_db(part)

    input_loud = steady(signal, 0.0)
    input_quiet = steady(signal, seg / sr)
    assert input_loud - input_quiet > 25.0, (input_loud, input_quiet)

    out, info = bus_compress(signal, sr)
    assert float(info["ratio"]) >= 3.0, info
    assert float(info["gain_reduction_db"]) > 3.0, info
    assert float(info["max_gain_reduction_db"]) > 6.0, info
    assert float(info["makeup_db"]) > 4.0, info
    assert float(info["upward_gain_db"]) > 0.4, info
    assert np.all(np.isfinite(out)), "压缩输出出现 NaN / Inf"

    out_loud = steady(out, 0.0)
    out_quiet = steady(out, seg / sr)
    assert out_loud - out_quiet < (input_loud - input_quiet) - 8.0, (
        input_loud,
        input_quiet,
        out_loud,
        out_quiet,
    )
    assert out_quiet - input_quiet > 6.0, (input_quiet, out_quiet)
    assert -3.0 < out_loud - input_loud < 1.5, (input_loud, out_loud)
    assert crest_factor_db(out) > 7.0, crest_factor_db(out)

    master, mix_info = master_bus(signal, sr, room_mix=0.0)
    assert abs(float(mix_info["loudness_db"]) - float(mix_info["target_db"])) < 1.5, mix_info
    assert float(mix_info["peak_db"]) <= -0.9, mix_info
    assert 7.0 < float(mix_info["crest_db"]) < 18.0, mix_info
    assert float(mix_info["compression_ratio"]) >= 3.0, mix_info
    assert peak_db(master) <= -0.9, peak_db(master)
    return (
        f"强弱差 {input_loud - input_quiet:.1f} → {out_loud - out_quiet:.1f} dB；"
        f"补偿 +{info['makeup_db']} dB / 弱段 +{info['upward_gain_db']} dB；"
        f"母带 {mix_info['loudness_db']} dB RMS / 峰值因数 {mix_info['crest_db']} dB"
    )


def test_sibilance_and_voiceless() -> str:
    """齿音 / 清辅音分析：高频噪声段应被降低，元音段基本不动。"""
    from hachimi.loudness import (
        reduce_sibilance,
        reduce_voiceless_consonants,
        sibilance_analysis,
    )
    from hachimi.synthesis import sample_path_for

    sr = 44100
    t = np.arange(int(sr * 0.8), dtype=np.float64) / sr
    vowel = 0.10 * np.sin(2 * np.pi * 220 * t) + 0.04 * np.sin(2 * np.pi * 440 * t)
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(len(t))
    # 5~9kHz 噪声作为齿音
    from scipy.signal import butter, sosfiltfilt

    sos = butter(3, [5000 / (sr / 2), 9000 / (sr / 2)], btype="bandpass", output="sos")
    sibilant = sosfiltfilt(sos, noise) * 0.8
    sibilant[: int(0.30 * sr)] = 0
    sibilant[int(0.55 * sr):] = 0
    mixed = (vowel + sibilant).astype(np.float32)

    info = sibilance_analysis(mixed, sr)
    assert float(np.max(info["high_ratio"])) > 0.75, info["high_ratio"]
    assert float(np.min(info["periodicity"])) < 0.35, info["periodicity"]
    reduced = reduce_sibilance(mixed, sr, max_reduction_db=6.0).mean(axis=1)

    def band_rms(sig: np.ndarray, lo: float, hi: float) -> float:
        n = int(0.20 * sr)
        spec = np.abs(np.fft.rfft(sig[int(0.32 * sr):int(0.32 * sr) + n])) ** 2
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.sqrt(np.sum(spec[mask])))

    before = band_rms(mixed, 4500, 10000)
    after = band_rms(reduced, 4500, 10000)
    assert after < before * 0.75, (before, after)

    # 实际音源：西 /x/ 的齿音段
    real, _ = sf.read(sample_path_for("西"))
    real = real.mean(axis=1) if real.ndim > 1 else real
    real_after = reduce_sibilance(real, sr, max_reduction_db=6.0).mean(axis=1)
    n = int(0.12 * sr)
    def real_high(z: np.ndarray) -> float:
        spec = np.abs(np.fft.rfft(z[:n] * np.hanning(n))) ** 2
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        return float(np.sqrt(np.sum(spec[(freqs >= 4500) & (freqs < 10000)])))
    real_before = real_high(real)
    real_after = real_high(real_after)
    assert real_after < real_before * 0.8, (real_before, real_after)

    # 清辅音：开头 80ms 噪声整体降低，后面元音不变
    voiced = (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    consonant = (0.2 * rng.standard_normal(int(0.08 * sr))).astype(np.float32)
    sample = np.concatenate([consonant, voiced]).astype(np.float32)
    fixed = reduce_voiceless_consonants(sample, sr, cons_frame=16, max_reduction_db=2.5)
    early_before = float(np.sqrt(np.mean(sample[: int(0.06 * sr)] ** 2)))
    early_after = float(np.sqrt(np.mean(fixed[: int(0.06 * sr)] ** 2)))
    late_before = float(np.sqrt(np.mean(sample[int(0.20 * sr):int(0.35 * sr)] ** 2)))
    late_after = float(np.sqrt(np.mean(fixed[int(0.20 * sr):int(0.35 * sr)] ** 2)))
    assert early_after < early_before * 0.85, (early_before, early_after)
    assert abs(late_after / late_before - 1.0) < 0.05, (late_before, late_after)
    return (
        f"齿音高频降低 {(1 - after / before) * 100:.0f}%（西 {1 - real_after / real_before:.0%}）；"
        f"清辅音降低 {(1 - early_after / early_before) * 100:.0f}%"
    )


def test_note_stretch_middle_only() -> str:
    """时值拉伸只拉元音中段：辅音/音头/尾音时值不变，长音连贯。"""
    from hachimi.synthesis import get_preprocessed, synthesize_note, vowel_stretch_plan

    base = vowel_stretch_plan(20, 20)
    long_plan = vowel_stretch_plan(20, 80)
    assert long_plan["target_attack"] == base["target_attack"], (base, long_plan)
    assert long_plan["target_release"] == base["target_release"], (base, long_plan)
    assert long_plan["target_middle"] > base["target_middle"], (base, long_plan)
    assert base["target_attack"] + base["target_middle"] + base["target_release"] == 20
    assert long_plan["target_attack"] + long_plan["target_middle"] + long_plan["target_release"] == 80

    sr = 44100
    pre = get_preprocessed("哈")
    short = synthesize_note(pre, 60, int(0.40 * sr))
    long = synthesize_note(pre, 60, int(3.00 * sr))
    assert len(short) == int(0.40 * sr) and len(long) == int(3.00 * sr)

    # 音头（辅音 + 元音起始）必须保持同一波形形状
    n = int(0.012 * sr)
    a = short[:n].mean(axis=1)
    b = long[:n].mean(axis=1)
    a = a / (np.sqrt(np.mean(a * a)) + 1e-12)
    b = b / (np.sqrt(np.mean(b * b)) + 1e-12)
    corr = float(np.corrcoef(a, b)[0, 1])
    assert corr > 0.98, corr

    def rms_at(x: np.ndarray, start: float, length: float = 0.10) -> float:
        seg = x[int(start * sr):int((start + length) * sr)]
        return float(np.sqrt(np.mean(seg ** 2)))

    middle = [rms_at(long, t) for t in (0.2, 0.6, 1.2, 1.8, 2.4)]
    assert min(middle) > max(middle) * 0.55, middle
    tail = rms_at(long, 2.985, 0.012)
    assert tail < np.mean(middle) * 0.5, (tail, middle)
    return (
        f"20→80 帧仅中段 {base['target_middle']}→{long_plan['target_middle']}；"
        f"音头相关 {corr:.3f}；长音 RMS {['%.3f' % v for v in middle]}；尾音已收"
    )


def test_legato_and_soft_attack() -> str:
    """连续音延伸 + 弱化音头；和弦 / 复调不能被误连。"""
    notes = [
        RenderNote(pitch=60, start=0.0, end=0.50, velocity=90, track=0, word="哈"),
        RenderNote(pitch=62, start=0.55, end=1.00, velocity=90, track=0, word="基"),
    ]
    linked = apply_legato(notes)
    assert linked[0].legato_to_next, linked[0]
    assert linked[1].legato_from_prev, linked[1]
    assert linked[0].end > 0.50, linked[0].end
    assert linked[0].legato_overlap > 0, linked[0].legato_overlap
    assert abs(linked[1].start - 0.55) < 1e-9

    chord = [
        RenderNote(pitch=60, start=0.0, end=0.5, velocity=90, track=0, word="哈"),
        RenderNote(pitch=64, start=0.0, end=0.5, velocity=90, track=0, word="基"),
        RenderNote(pitch=67, start=0.55, end=1.0, velocity=90, track=0, word="米"),
    ]
    chord_out = apply_legato(chord)
    assert not any(n.legato_to_next or n.legato_from_prev for n in chord_out)

    env = soft_attack_envelope(64)
    assert env[0] == 0.0 and env[-1] > 0.9, (env[0], env[-1])
    return f"前音延伸 {linked[0].legato_overlap:.3f}s；起音 14ms 软包络；和弦不误连"


def test_subtitle_video() -> str:
    """绿幕视频：宽 1920、单条视频、每次 1 行、8 号白字黑边。"""
    from PIL import Image
    import subprocess

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("未找到 ffmpeg，无法验证字幕视频")
    from hachimi.subtitles import SubtitleLine

    out = os.path.join(OUT, "video")
    # 两句顺序字幕，全部进入同一条视频；每句都保持单行。
    lines = [
        SubtitleLine(1, 0.10, 0.60, "哈基米" * 5, section="A"),
        SubtitleLine(2, 0.65, 1.15, "伴奏字幕", section="A"),
    ]
    result = render_subtitle_video(
        lines,
        out_dir=out,
        project="_selftest_video",
        duration=1.2,
        fps=30,
    )
    assert result["available"], result
    assert result["width"] == 1920, result
    assert 150 <= result["height"] <= 200, result
    assert result["max_lines"] == 1, result
    assert result["single_video"] is True, result
    assert result["font"] == "SimSun", result
    assert result["font_size"] == 8, result
    assert result["font_px"] == 80, result
    assert result["text_color"] == "#FFFFFF", result
    assert result["stroke_width"] == 2, result
    assert result["stroke_color"] == "#000000", result
    assert set(result["files"]) == {"all"}, result["files"]

    # 单行模式：24 个全角字绝不拆行，只缩小字号。
    one_line = measure_layout(
        [[SubtitleLine(1, 0.0, 1.0, "哈" * 24, section="A")]],
        width=1920,
        max_lines=1,
    )
    assert one_line.width == 1920, one_line
    assert one_line.max_lines == 1, one_line
    assert one_line.font_size < 80, one_line
    assert one_line.height < 200, one_line

    # 字体 / 字号 / 颜色可调；黑边仍保留。
    custom = measure_layout(
        [[SubtitleLine(1, 0.0, 1.0, "自定义样式", section="A")]],
        width=1920,
        font_family="Microsoft YaHei",
        font_size=14,
        text_color="#FF0000",
        max_lines=1,
    )
    assert custom.font_family == "Microsoft YaHei", custom
    assert custom.font_size == 140, custom
    assert custom.text_rgb == (255, 0, 0), custom
    assert custom.stroke_width == 2, custom
    assert custom.stroke_rgb == (0, 0, 0), custom

    video_info = probe_video(result["files"]["all"])
    assert video_info.get("width") == 1920, video_info
    assert abs(int(video_info.get("height") or 0) - result["height"]) <= 2, video_info
    assert video_info.get("fps") == "30/1", video_info
    assert 1.0 <= float(video_info.get("duration") or 0) <= 1.4, video_info

    frame_path = os.path.join(out, "frame.png")
    proc = subprocess.run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", "0.2", "-i", result["files"]["all"],
            "-frames:v", "1", frame_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    arr = np.asarray(Image.open(frame_path).convert("RGB"))
    for y, x in ((0, 0), (arr.shape[0] - 1, arr.shape[1] - 1), (arr.shape[0] // 2, 0)):
        r, g, b = (int(v) for v in arr[y, x])
        assert r < 16 and b < 16 and g > 230, (y, x, r, g, b)
    white_pixels = int(np.all(arr >= 245, axis=2).sum())
    black_pixels = int(np.all(arr <= 20, axis=2).sum())
    assert white_pixels > 500, white_pixels
    assert black_pixels > 100, black_pixels
    return (
        f"{result['width']}×{result['height']} · 单条 · 每次 1 行 · 30fps · 纯绿底 · "
        f"{white_pixels} 白字 / {black_pixels} 黑边像素 / 一张 MP4"
    )


def test_subtitles() -> str:
    _, lyrics = _build_lyrics()
    lines = build_subtitles(lyrics.slots)
    assert lines, "没有字幕行"
    for i in range(len(lines) - 1):
        assert lines[i].end <= lines[i + 1].start + 1e-6, f"字幕重叠：{i}"
        assert lines[i].text, f"空字幕：{i}"
    for line in lines:
        assert line.end > line.start
    os.makedirs(OUT, exist_ok=True)
    srt = write_srt(lines, os.path.join(OUT, "test.srt"))
    lrc = write_lrc(lines, os.path.join(OUT, "test.lrc"), title="selftest")
    html_path = write_html(
        lines, os.path.join(OUT, "test.html"), title="selftest", audio_name="test.wav"
    )
    text = open(srt, encoding="utf-8-sig").read()
    assert "-->" in text and sec_to_srt(lines[0].start) in text
    assert "[ti:selftest]" in open(lrc, encoding="utf-8").read()
    html_text = open(html_path, encoding="utf-8").read()
    assert "test.wav" in html_text and "const DATA" in html_text
    # HTML 内嵌 JSON 必须可解析
    start = html_text.index("const DATA = ") + len("const DATA = ")
    end = html_text.index(";\n", start)
    payload = json.loads(html_text[start:end].replace("<\\/", "</"))
    assert len(payload["lines"]) == len(lines)
    return f"{len(lines)} 行字幕 / SRT+LRC+HTML 正常"


def test_verification_negative() -> str:
    """人为漏掉一个和弦音，校验必须能发现。"""
    from hachimi.analysis import NoteEvent
    from hachimi.render import RenderNote
    from hachimi.verify import check_stem_energy, verify_polyphony

    notes = [
        NoteEvent(id=0, track=0, index=0, pitch=60, start=0.0, end=1.0, velocity=90),
        NoteEvent(id=1, track=1, index=0, pitch=64, start=0.0, end=1.0, velocity=90),
        NoteEvent(id=2, track=2, index=0, pitch=67, start=0.0, end=1.0, velocity=90),
    ]
    render_list = [
        RenderNote(pitch=n.pitch, start=n.start, end=n.end, velocity=n.velocity, track=n.track, word="哈")
        for n in notes
    ]
    events = [
        {"track": n.track, "pitch": n.pitch, "start": n.start, "end": n.end, "state": "哈"}
        for n in notes
    ]
    full = verify_polyphony(notes, events, render_list)
    assert full["ok"], full
    broken = verify_polyphony(notes, events[:2], render_list)
    assert not broken["ok"], "漏掉和弦音时校验未报警"
    assert broken["tracks"][0]["missing_events"] or any(t["missing_events"] for t in broken["tracks"])

    # 分轨静音也必须被发现
    import numpy as np

    groups = [[notes[0], notes[1], notes[2]]]
    silent = check_stem_energy(np.zeros((44100, 2), dtype=np.float32), 44100, 0.0, groups)
    assert silent, "静音分轨未被发现"
    return "漏音 / 静音分轨均可被检出"


def test_gate_verification() -> str:
    """断奏 / 装饰音的 gate 缩短发声，但不能被误判为漏渲染。"""
    from hachimi.analysis import NoteEvent
    from hachimi.render import build_render_notes
    from hachimi.verify import verify_polyphony

    note = NoteEvent(
        id=0, track=0, index=0, pitch=60, start=0.0, end=1.0,
        velocity=90, gate=0.48, articulation="staccato",
    )
    event = {
        "track": 0, "pitch": 60, "start": 0.0, "end": 1.0,
        "state": "哈", "gate": 0.48, "articulation": "staccato",
    }
    options = {0: {"index": 0, "selected": True, "gain": 1.0, "octave": 0, "voice": "auto"}}
    render_notes_list = build_render_notes([event], options)
    assert abs(render_notes_list[0].duration - 0.48) < 1e-6, render_notes_list[0]
    assert abs(render_notes_list[0].source_end - 1.0) < 1e-6, render_notes_list[0]
    result = verify_polyphony([note], [event], render_notes_list, track_names={0: "T0"})
    assert result["ok"], result
    assert result["verified_notes"] == 1, result
    return "staccato gate 0.48 实际发声 0.48s，校验按谱面 1.0s 匹配：通过"


def test_consonant_plosive() -> str:
    """辅音统一压低；基 / 库 起始更柔，且串行 / 并行路径一致。"""
    from hachimi.config import AP_SCALE, FRAME_PERIOD, SP_SCALE, SR
    from hachimi.loudness import consonant_gain_curve
    from hachimi.render import RenderNote, _synth_task, clear_unit_cache, render_notes
    from hachimi.synthesis import get_preprocessed, is_plosive_sample, synthesize_note

    curve = consonant_gain_curve(44100, 44100, 8, plosive=True)
    assert float(curve[0]) < 0.18, curve[0]  # 起音约 -16 dB
    assert float(curve[-1]) > 0.999, curve[-1]
    assert is_plosive_sample("基") and is_plosive_sample("库")
    assert not is_plosive_sample("哈")

    def rms(x: np.ndarray, a: float, b: float) -> float:
        return float(np.sqrt(np.mean(x[int(a * 44100):int(b * 44100)] ** 2)))

    sr = 44100
    for char in ("基", "库"):
        pre = get_preprocessed(char)
        dry = synthesize_note(pre, 60, sr)
        wet = synthesize_note(pre, 60, sr, sample_id=char)
        early_ratio = rms(wet, 0.0, 0.02) / max(1e-12, rms(dry, 0.0, 0.02))
        vowel_ratio = rms(wet, 0.20, 0.40) / max(1e-12, rms(dry, 0.20, 0.40))
        assert early_ratio < 0.22, (char, early_ratio)
        assert abs(vowel_ratio - 1.0) < 0.02, (char, vowel_ratio)

        # 串行渲染分支也必须套用 sample_id 的爆破音处理；
        # 旧实现漏传 sample_id，会让小规模 / 非哈基米声库渲染丢失音头衰减。
        clear_unit_cache()
        note = RenderNote(
            pitch=60,
            start=0.0,
            end=0.5,
            velocity=90,
            track=0,
            word=char,
            sample=char,
        )
        serial, _ = render_notes(
            [note],
            workers=1,
            gpu="off",
            align_start=0.0,
            total_samples=int(0.6 * sr),
        )
        serial_ratio = rms(serial, 0.0, 0.02) / max(1e-12, rms(serial, 0.20, 0.40))
        assert serial_ratio < 0.15, (char, serial_ratio)

        # 多进程 worker 与串行必须合成出完全相同的单位（都带 sample_id）。
        spec = (char, 60, int(0.5 * SR), AP_SCALE, SP_SCALE, SR, FRAME_PERIOD)
        _, worker_audio = _synth_task(spec)
        direct_audio = synthesize_note(
            get_preprocessed(char),
            60,
            int(0.5 * SR),
            sr=SR,
            frame_period=FRAME_PERIOD,
            ap_scale=AP_SCALE,
            sp_scale=SP_SCALE,
            sample_id=char,
        )
        assert np.allclose(worker_audio, direct_audio), char
    return "基 / 库 起音约 -16dB（相对元音 -24~-27dB）；串行 / 并行一致；元音不变"


def test_bad_notes_and_midi() -> str:
    """过高 / 过低 / 音量过低判为不良；p 力度不误判；MIDI 只含不良音。"""
    import pretty_midi

    from hachimi.analysis import analyze as analyze_file
    from hachimi.qc import assess_bad_notes, write_bad_note_midis

    analysis = analyze_file(DEMO)
    options = RenderOptions()
    tracks = resolve_track_options(analysis, options)
    for track in tracks:
        track["selected"] = track["index"] == 0
        track["gain"] = 1.0
        track["octave"] = 0
    notes = [
        NoteEvent(id=0, track=0, index=0, pitch=24, start=0.0, end=0.5, velocity=90),
        NoteEvent(id=1, track=0, index=1, pitch=90, start=0.6, end=1.1, velocity=90),
        NoteEvent(id=2, track=0, index=2, pitch=60, start=1.2, end=1.7, velocity=8),
        NoteEvent(id=3, track=0, index=3, pitch=60, start=1.8, end=2.3, velocity=20, dynamic="p"),
        NoteEvent(id=4, track=0, index=4, pitch=60, start=2.4, end=2.9, velocity=90),
    ]
    quality = assess_bad_notes(notes, {t["index"]: t for t in tracks}, analysis.tracks)
    assert quality["too_low"] == 1 and quality["too_high"] == 1, quality
    assert quality["too_quiet"] == 1, quality  # 仅 velocity=8；p 力度不误判
    files = write_bad_note_midis(
        notes, quality, analysis.tracks, os.path.join(OUT, "qc"), "_selftest_qc"
    )
    assert "0" in files and "all" in files, files
    for path in files.values():
        pm = pretty_midi.PrettyMIDI(path)
        pitches = [n.pitch for inst in pm.instruments for n in inst.notes]
        assert sorted(pitches) == [24, 60, 90], pitches
    return "过高 1 / 过低 1 / 音量低 1；p 力度保留；MIDI 只含 3 个不良音"


def test_supplement_midi() -> str:
    """补充 MIDI：不良音轨 _NG → 空白轨 → 全谱轨，轨道名全英文。"""
    import pretty_midi

    from hachimi.analysis import analyze as analyze_file
    from hachimi.qc import assess_bad_notes, write_bad_and_all_midi

    analysis = analyze_file(DEMO)
    options = RenderOptions()
    tracks = resolve_track_options(analysis, options)
    for track in tracks:
        track["selected"] = track["index"] == 0
        track["gain"] = 1.0
        track["octave"] = 0
    notes = [
        NoteEvent(id=0, track=0, index=0, pitch=24, start=0.0, end=0.5, velocity=90),
        NoteEvent(id=1, track=0, index=1, pitch=90, start=0.6, end=1.1, velocity=90),
        NoteEvent(id=2, track=0, index=2, pitch=60, start=1.2, end=1.7, velocity=8),
        NoteEvent(id=3, track=0, index=3, pitch=60, start=1.8, end=2.3, velocity=20, dynamic="p"),
        NoteEvent(id=4, track=0, index=4, pitch=60, start=2.4, end=2.9, velocity=90),
    ]
    missing_notes = [n for n in analysis.notes_all if n.track == 1]
    assert missing_notes, "演示 MIDI 第 2 轨应有音符"
    quality = assess_bad_notes(
        notes + missing_notes,
        {t["index"]: t for t in tracks},
        analysis.tracks,
    )
    assert quality["bad_notes"] == 3, quality
    folder = os.path.join(OUT, "qc_supplement")
    info = write_bad_and_all_midi(
        notes + missing_notes,
        quality,
        analysis.tracks,
        folder,
        "_selftest_supplement",
        missing_indices=[1, 2],
        missing_reasons={1: "未勾选", 2: "空轨道"},
        tempo=analysis.tempo or 120.0,
        tempo_changes=analysis.tempo_changes,
        time_signature=analysis.time_signature,
    )
    assert info["path"] and os.path.exists(info["path"]), info
    assert info["n_tracks"] == 4, info
    assert info["n_bad_tracks"] == 1 and info["n_score_tracks"] == 2, info
    assert info["n_entries"] == 4 and info["n_spacer_tracks"] == 1, info
    assert info["n_notes"] == 3 + 5 + len(missing_notes), info
    assert info["bad_notes"] == 3 and info["score_notes"] == 5 + len(missing_notes), info
    assert info["missing_notes"] == len(missing_notes), info
    assert info["n_missing_tracks"] == 1, info
    assert [item["index"] for item in info["skipped"]] == [2, 3], info["skipped"]
    pm = pretty_midi.PrettyMIDI(info["path"])
    names = [inst.name for inst in pm.instruments]
    assert names == ["piano1_NG", "piano1", "strings1_NOAUDIO"], names
    assert sorted(n.pitch for n in pm.instruments[0].notes) == [24, 60, 90]
    assert len(pm.instruments[1].notes) == 5, "全谱轨应保留所有音符"
    assert len(pm.instruments[2].notes) == len(missing_notes)
    for name in names:
        assert name.isascii() and name, name
        assert "_" not in name or name.endswith("_NG") or name.endswith("_NOAUDIO"), name

    import mido

    mid = mido.MidiFile(info["path"])
    midi_names = []
    for track in mid.tracks:
        midi_names.append(next((m.name for m in track if m.type == "track_name"), ""))
    assert midi_names == ["", "piano1_NG", "", "piano1", "strings1_NOAUDIO"], midi_names
    assert len(mid.tracks[2]) == 1 and mid.tracks[2][0].type == "end_of_track", mid.tracks[2]
    return (
        f"{info['n_tracks']} 轨（{info['n_bad_tracks']} NG → 空白 → "
        f"{info['n_score_tracks']} 全谱）/ {info['n_notes']} 音；"
        f"名称 {names[0]} / {names[1]} / {names[2]}"
    )


def test_percussion_unpitched() -> str:
    """MusicXML 打击乐：<unpitched> 按 midi-unpitched 还原，并单独导出。"""
    import pretty_midi

    from hachimi.analysis import analyze as analyze_file
    from hachimi.qc import write_percussion_midi
    from hachimi.score import read_score

    xml = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1">
      <part-name>Violin</part-name>
      <score-instrument id="P1-I1"><instrument-name>Violin</instrument-name></score-instrument>
      <midi-instrument id="P1-I1"><midi-channel>1</midi-channel><midi-program>41</midi-program></midi-instrument>
    </score-part>
    <score-part id="P2">
      <part-name>Triangle</part-name>
      <score-instrument id="P2-I82"><instrument-name>Open Triangle</instrument-name></score-instrument>
      <midi-instrument id="P2-I82"><midi-channel>10</midi-channel><midi-program>49</midi-program><midi-unpitched>82</midi-unpitched></midi-instrument>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><type>whole</type></note>
    </measure>
  </part>
  <part id="P2">
    <measure number="1">
      <attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
      <note><unpitched><display-step>E</display-step><display-octave>5</display-octave></unpitched><duration>4</duration><instrument id="P2-I82"/><voice>1</voice><type>whole</type></note>
    </measure>
  </part>
</score-partwise>
"""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "_percussion_test.musicxml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    score = read_score(path)
    assert len(score.instruments) == 2, len(score.instruments)
    assert score.instruments[1].is_drum is True
    assert len(score.instruments[1].notes) == 1, score.instruments[1].notes
    assert score.instruments[1].notes[0].pitch == 81, score.instruments[1].notes[0].pitch
    analysis = analyze_file(path)
    drums = [t for t in analysis.tracks if t.is_drum]
    assert len(drums) == 1 and drums[0].n_notes == 1, drums
    folder = os.path.join(OUT, "percussion")
    os.makedirs(folder, exist_ok=True)
    info = write_percussion_midi(
        analysis.notes_all,
        analysis.tracks,
        folder,
        "_selftest_percussion",
        tempo=analysis.tempo or 120.0,
        tempo_changes=analysis.tempo_changes,
        time_signature=analysis.time_signature,
    )
    assert info["path"] and os.path.exists(info["path"]), info
    assert info["n_tracks"] == 1 and info["n_notes"] == 1, info
    assert info["tracks"][0]["midi_track_name"] == "triangle1_NOAUDIO", info["tracks"]
    pm = pretty_midi.PrettyMIDI(info["path"])
    assert len(pm.instruments) == 1 and pm.instruments[0].is_drum, pm.instruments
    assert pm.instruments[0].name == "triangle1_NOAUDIO", pm.instruments[0].name
    assert [n.pitch for n in pm.instruments[0].notes] == [81], pm.instruments[0].notes
    return "unpitched → GM 81；打击乐 1 轨 / 1 音，轨名 triangle1_NOAUDIO"


def test_params_roundtrip() -> str:
    """参数 ZIP 可读回；旧 manifest 可还原部分设置。"""
    from hachimi.params import (
        build_project_params,
        load_params_bytes,
        load_params_file,
        params_from_manifest,
        write_project_params,
    )

    analysis = analyze(DEMO)
    options = RenderOptions(
        voice="auto",
        sample_set=2,
        seed=2468,
        syntax_weight=0.22,
        write_premix=True,
        write_bad_note_midis=True,
    )
    tracks = resolve_track_options(analysis, options)
    for track in tracks:
        track["selected"] = track["index"] in (0, 2)
        track["gain"] = 0.73
        track["octave"] = 1
    payload = build_project_params(
        project="_selftest_params",
        midi_path=DEMO,
        options=options,
        track_options={t["index"]: t for t in tracks},
        selected=[0, 2],
        audible=[0, 2],
        analysis=analysis,
        outputs={"duration": analysis.duration},
    )
    os.makedirs(os.path.join(OUT, "params"), exist_ok=True)
    paths = write_project_params(payload, os.path.join(OUT, "params"), "_selftest_params")
    loaded = load_params_file(paths["zip"])
    assert loaded["project"] == "_selftest_params", loaded
    assert loaded["options"]["seed"] == 2468, loaded["options"]
    assert len(loaded["tracks"]) == len(tracks), loaded["tracks"]
    assert load_params_bytes(open(paths["json"], "rb").read(), "p.json")["options"]["sample_set"] == 2

    manifest = {
        "project": "old",
        "midi_path": "old.mid",
        "duration": 12.0,
        "tracks": [{"index": 0, "selected": True, "voice": "哈基米", "octave": -1, "gain": 0.8}],
        "analysis": {"source_format": "musicxml", "render_tracks": []},
        "stats": {"mix": {"enabled": True}},
    }
    legacy = params_from_manifest(manifest)
    assert legacy["legacy"] and legacy["tracks"][0]["octave"] == -1, legacy
    return "参数 JSON / ZIP 往返一致；旧 manifest 可恢复轨道设置"


def test_sample_set_and_bass() -> str:
    """第二套哈基米音源 + 本地音源 + 低音声部自动不勾选。"""
    from hachimi.config import (
        BASS_GAIN,
        BASS_MAX_PITCH,
        SOURCE_DD_DIR,
        SOURCE_HACHIMI_DIR,
        SOURCE_MANBO_DIR,
        SOURCE_MB_DIR,
    )
    from hachimi.synthesis import (
        choose_sample,
        preproc_cache_path,
        sample_exists,
        sample_path_for,
    )

    assert sample_path_for("哈").startswith(SOURCE_HACHIMI_DIR), sample_path_for("哈")
    assert sample_path_for("鸡").startswith(SOURCE_DD_DIR), sample_path_for("鸡")
    assert sample_path_for("曼1").startswith(SOURCE_MB_DIR), sample_path_for("曼1")
    assert sample_path_for("波").startswith(SOURCE_MANBO_DIR), sample_path_for("波")
    assert sample_path_for("h2_哈").startswith(SOURCE_HACHIMI_DIR.replace("哈基米", "哈基米第二版"))

    # 修改音源 mtime 后预处理缓存 key 必须变化，然后恢复原 mtime
    sample = sample_path_for("哈")
    st = os.stat(sample)
    before = preproc_cache_path("哈")
    try:
        os.utime(sample, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        after = preproc_cache_path("哈")
        assert before != after, "修改音源后预处理缓存未失效"
    finally:
        os.utime(sample, ns=(st.st_atime_ns, st.st_mtime_ns))

    assert sample_exists("h2_哈"), "缺少第二套哈基米采样"
    assert choose_sample("哈", "哈基米", 0, 60, 0.0, sample_set=2) == "h2_哈"
    assert choose_sample("哈", "哈基米", 0, 60, 0.0, sample_set=1) == "哈"

    # 别名必须先解析成实际采样，辅音帧数 / 爆破音判定才和原声库一致。
    from hachimi.synthesis import is_plosive_sample, resolve_sample_id

    aliases = {
        "夏": "哈",
        "达": "那",
        "马": "那",
        "哇": "呀",
        "不": "波",
        "耶": "呀",
        "巴": "巴1",
    }
    for word, expected in aliases.items():
        actual = choose_sample(word, "曼波", 0, 60, 0.0)
        assert actual == expected, (word, actual, expected)
    assert choose_sample("夏", "曼波", 0, 60, 0.0, sample_set=2) == "h2_哈"
    assert resolve_sample_id("夏") == "哈"
    assert not is_plosive_sample(choose_sample("达", "曼波", 0, 60, 0.0))
    assert is_plosive_sample(choose_sample("不", "曼波", 0, 60, 0.0))

    analysis = analyze(DEMO)
    low = [
        t for t in analysis.tracks
        if not t.is_drum and t.pitch_max is not None and t.pitch_max < BASS_MAX_PITCH
    ]
    assert low, "演示 MIDI 应存在低音声部"
    for t in low:
        assert t.role == "bass", (t.index, t.role)
        assert t.role_label == "低音声部"
        assert not t.suggested_selected, (t.index, t.suggested_selected)
        assert t.suggested_gain <= BASS_GAIN + 1e-6, t.suggested_gain
    defaults = analysis.selected_track_indices()
    assert not (set(defaults) & {t.index for t in low}), defaults
    return f"本地音源可用；低音声部 {[t.index for t in low]} 默认不勾选、音量 {BASS_GAIN}"


def test_voice_tuning_consistency() -> str:
    """非哈基米声库与哈基米共用同一套调音 / 音头处理。

    逐字检查所有声库的字母表都能解析到实际存在的采样（别名不能变成兜底「哈」），
    再对每个声库的爆破音代表字跑一次串行渲染，确认起音都被同样削弱。
    """
    from hachimi.config import ALPHABET
    from hachimi.render import RenderNote, clear_unit_cache, render_notes
    from hachimi.synthesis import (
        choose_sample,
        sample_exists,
        sample_path_for,
    )

    for voice, chars in ALPHABET.items():
        for char in chars:
            sid = choose_sample(char, voice, 0, 60, 0.0)
            assert sample_exists(sid), (voice, char, sid)
            path = sample_path_for(sid)
            assert os.path.basename(path) == f"{sid}.wav", (voice, char, sid, path)

    # 代表性爆破音：每次都用对应声库实际会选的采样。
    representatives = (
        ("哈基米", "基"),
        ("叮咚鸡", "叮"),
        ("曼巴", "巴"),
        ("曼波", "波"),
        ("哈基米only", "基"),
        ("曼波only", "波"),
    )
    sr = 44100

    def rms(x: np.ndarray, a: float, b: float) -> float:
        return float(np.sqrt(np.mean(x[int(a * sr):int(b * sr)] ** 2)))

    ratios: list[tuple[str, str, float]] = []
    for voice, char in representatives:
        sid = choose_sample(char, voice, 0, 60, 0.0)
        clear_unit_cache()
        note = RenderNote(
            pitch=60,
            start=0.0,
            end=0.5,
            velocity=90,
            track=0,
            word=char,
            sample=sid,
        )
        audio, _ = render_notes(
            [note],
            workers=1,
            gpu="off",
            align_start=0.0,
            total_samples=int(0.6 * sr),
        )
        ratio = rms(audio, 0.0, 0.02) / max(1e-12, rms(audio, 0.20, 0.40))
        assert ratio < 0.20, (voice, char, sid, ratio)
        ratios.append((voice, char, ratio))

    worst = max(ratios, key=lambda item: item[2])
    return (
        f"{len(ALPHABET)} 个声库字母表全部指向实际采样；"
        f"{len(ratios)} 个爆破音代表字起音均 < -14dB（最弱 {worst[0]}「{worst[1]}」）"
    )


def test_stems_zip_and_polyphony() -> str:
    if not os.path.exists(DEMO):
        raise FileNotFoundError(DEMO)
    options = RenderOptions(
        model_kind="markov",
        voice="auto",
        workers=1,
        max_seconds=5.0,
        write_stems=True,
        write_zip=True,
        write_subtitle_videos=False,
        tracks=[
            {"index": 0, "selected": True, "voice": "auto", "octave": 0, "gain": 0.6},
            {"index": 1, "selected": True, "voice": "auto", "octave": 0, "gain": 0.6},
            {"index": 2, "selected": True, "voice": "auto", "octave": 0, "gain": 0.6},
        ],
    )
    result = render_midi(
        DEMO, options, output_dir=os.path.join(OUT, "stems"), project_name="_selftest_stems"
    )
    stems = result.stems
    assert len(stems) == 3, stems
    for idx, path in stems.items():
        data, sr = sf.read(path)
        assert data.ndim == 2 and data.shape[1] == 2, (idx, data.shape)
        assert len(data) > 44100 * 4, (idx, len(data))
        assert float(np.sqrt(np.mean(data ** 2))) > 0, f"分轨 {idx} 静音"
        # 命名：乐器 + 轨道号 + 语音类型 + 八度修改值
        name = os.path.basename(path)
        assert name.endswith(".mp3"), name
        assert f"_{int(idx):02d}_" in name, name
        assert ("+0" in name or "-1" in name or "+1" in name), name

    zip_path = result.files.get("zip")
    assert zip_path and os.path.exists(zip_path), "缺少 ZIP"
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    stem_zip = [n for n in names if n.startswith("stems/") and n.endswith(".mp3")]
    assert len(stem_zip) == 3, stem_zip
    for expected in (
        "_selftest_stems.mp3",
        "_selftest_stems_subtitles.srt",
        "_selftest_stems_verification.json",
        "_selftest_stems_instrument_report.md",
        "_selftest_stems_instrument_report.csv",
        "_selftest_stems_instrument_report.json",
        "_selftest_stems_params.json",
        "_selftest_stems_params.zip",
        "_selftest_stems_stems_premix.zip",
        "_selftest_stems_bad_notes.json",
        "manifest.json",
    ):
        assert expected in names, f"ZIP 缺少 {expected}: {sorted(names)}"
    assert len(result.premix) == 3, result.premix
    assert not any(n.startswith("stems_premix/") for n in names), "主 ZIP 不应重复存单条 FLAC"
    with zipfile.ZipFile(result.files["premix_zip"]) as zf:
        premix_names = set(zf.namelist())
    assert any(n.startswith("stems_premix/") and n.endswith(".flac") for n in premix_names), premix_names
    assert result.project_params.get("options", {}).get("write_premix") is True

    # 乐器清单：输出乐器 + 对应音频 + 未输出名单
    for key in ("instrument_report", "instrument_report_csv", "instrument_report_json"):
        assert key in result.files and os.path.exists(result.files[key]), result.files
    report_text = open(result.files["instrument_report"], encoding="utf-8").read()
    assert "输出乐器名单" in report_text and "对应音频文件" in report_text
    assert "未输出名单" in report_text and "Drums" in report_text
    assert "自动混音摆位" in report_text and "母带响度" in report_text
    assert "Acoustic_Grand_Piano_00_" in report_text or "_00_" in report_text
    report_summary = result.instrument_report["summary"]
    assert report_summary["n_outputs"] == 3, report_summary
    assert report_summary["n_missing"] >= 1, report_summary
    assert all(t.get("output_file") for t in result.tracks), result.tracks

    verification = result.verification
    assert verification and verification["ok"], verification
    assert verification["total_chord_notes"] > 0
    assert verification["verified_chord_notes"] == verification["total_chord_notes"]
    assert all(t["ok"] for t in verification["tracks"]), verification["tracks"]
    return (
        f"{len(stems)} 条分轨 + ZIP({len(names)} 文件)；"
        f"{verification['verified_chord_notes']}/{verification['total_chord_notes']} 和弦音已发声"
    )


def test_syntax_and_supplement() -> str:
    """标准句法 + 补充采样（叮咚鸡 / 曼巴）。"""
    from hachimi.config import ALPHABET
    from hachimi.syntax import SYNTAX_BANKS, generate_phrase, split_units

    for voice in ("哈基米", "叮咚鸡", "曼巴"):
        for n in (3, 4, 6, 8, 12):
            text = generate_phrase(voice, n, random.Random(7))
            assert len(text) == n, (voice, n, text)
            assert set(text) <= set(ALPHABET[voice]), (voice, text)
            units = split_units(voice, text)
            assert "".join(units) == text
    # 补充采样是否可被解析
    from hachimi.synthesis import sample_exists, choose_sample

    for sample in ("叮", "咚", "鸡2", "叫2", "曼1", "巴3", "欧", "嘛"):
        assert sample_exists(sample), f"缺少补充采样 {sample}"
    assert choose_sample("鸡", "叮咚鸡", 0, 60, 0.0) in ("鸡", "鸡2")
    assert choose_sample("曼", "曼巴", 0, 60, 0.0).startswith("曼")
    return f"{len(SYNTAX_BANKS)} 种句法；叮咚鸡/曼巴采样可用"


def test_gpu_mixing() -> str:
    """GPU 混音与 CPU 结果一致，并遵守显存预算。"""
    from hachimi.render import RenderNote, clear_unit_cache, render_notes

    notes = [
        RenderNote(
            pitch=60 + (i % 12),
            start=i * 0.08,
            end=i * 0.08 + 0.3,
            velocity=80,
            track=i % 3,
            word="哈",
            sample="哈",
            gain=0.5,
        )
        for i in range(160)
    ]
    clear_unit_cache()
    cpu, _ = render_notes(notes, workers=2, gpu="off")
    clear_unit_cache()
    gpu, stats = render_notes(notes, workers=2, gpu="on")
    if stats.device != "cuda":
        return "无 CUDA，已自动回退 CPU"
    diff = float(np.max(np.abs(cpu - gpu)))
    assert diff < 1e-5, f"CPU/GPU 混音差异过大: {diff}"
    assert stats.gpu_memory_mb <= 3072
    return f"GPU 混音一致（diff={diff:.1e}，显存预算 {stats.gpu_memory_mb}MB）"


def test_musicxml_analysis() -> str:
    """MusicXML：速度地图、力度、奏法、连音线与秒级时间轴。"""
    if not os.path.exists(SCORE):
        raise FileNotFoundError(SCORE)
    result = analyze(SCORE, selected_tracks=list(range(10)))
    assert result.source_format == "musicxml", result.source_format
    assert len(result.tracks) == 10, len(result.tracks)
    assert abs(result.duration - 393.2857) < 0.05, result.duration
    assert result.time_signature == "4/4", result.time_signature

    expected_times = [0.0, 55.0, 153.2857, 160.1429, 323.3429, 330.2, 364.4857, 369.2857]
    expected_bpms = [48.0, 52.5, 35.0, 50.0, 35.0, 35.0, 50.0, 40.0]
    assert len(result.tempo_changes) == len(expected_times), result.tempo_changes
    for item, t, bpm in zip(result.tempo_changes, expected_times, expected_bpms):
        assert abs(float(item["time"]) - t) < 0.01, (item, t)
        assert abs(float(item["bpm"]) - bpm) < 0.01, (item, bpm)

    p1 = [n for n in result.notes if n.track == 0]
    assert len(p1) == 262, len(p1)
    assert p1[0].start == 0.0 and abs(p1[1].start - 1.25) < 1e-6
    assert abs(p1[2].start - 2.5) < 1e-6
    # 小节内变速：m33 的 Grave 在 q130，而不是小节起点 q128
    assert any(abs(n.start - 153.2857) < 0.002 for n in p1), "m33 变速落位不正确"
    # m12 的 Lento 在 q44，对应 55.0s
    assert any(abs(n.start - 55.0) < 0.7 for n in p1), "m12 变速落位不正确"

    # p / mp / f 等力度与 cresc. / dim. 发夹
    assert result.score_meta["n_dynamics"] >= 270, result.score_meta
    assert result.score_meta["n_wedges"] >= 250, result.score_meta
    assert any(n.velocity <= 55 for n in p1), "未读取到弱力度"
    assert any(n.velocity >= 90 for n in p1), "未读取到强力度"
    assert any(
        n.velocity != n.velocity_end for n in p1
    ), "未生成渐强 / 渐弱力度终点"

    # 奏法：staccato / tenuto / accent / marcato / tremolo
    assert result.score_meta["n_articulations"] >= 30, result.score_meta
    assert any(n.articulation for n in result.notes)
    assert any(n.gate < 0.6 for n in result.notes), "未应用断奏时值"
    assert any(n.tremolo for n in result.notes), "未标记 tremolo"

    # tie 合并：原始 4113 个音符 -> 3961 个持续音
    assert result.score_meta["n_notes_raw"] > result.score_meta["n_notes"]
    assert result.score_meta["n_ties"] >= 250, result.score_meta
    # 低音提琴移调：Contrabass 第一音实际发声音高 39
    bass = [n for n in result.notes if n.track == 9]
    assert bass and bass[0].pitch == 39, bass[:1]

    return (
        f"{len(result.tracks)} 声部 / {result.duration:.2f}s / "
        f"{len(result.tempo_changes)} 处变速 {expected_bpms[0]:g}→{expected_bpms[-1]:g} BPM / "
        f"{result.score_meta['n_dynamics']} 处力度 / {result.score_meta['n_ties']} 连音线"
    )


def test_musicxml_mxl() -> str:
    """.mxl（ZIP + container.xml）与 .xml 结果一致。"""
    import zipfile

    if not os.path.exists(SCORE):
        raise FileNotFoundError(SCORE)
    os.makedirs(OUT, exist_ok=True)
    mxl_path = os.path.join(OUT, "_musicxml_test.mxl")
    xml_bytes = open(SCORE, "rb").read()
    container = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<container version="1.0" '
        'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        "<rootfiles><rootfile full-path=\"score.xml\" "
        'media-type="application/vnd.recordare.musicxml+xml"/></rootfiles></container>'
    ).encode("utf-8")
    with zipfile.ZipFile(mxl_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("META-INF/container.xml", container)
        zf.writestr("score.xml", xml_bytes)

    result = analyze(mxl_path)
    assert result.source_format == "mxl", result.source_format
    assert len(result.tracks) == 10, len(result.tracks)
    assert abs(result.duration - 393.2857) < 0.05, result.duration
    assert len(result.tempo_changes) == 8, result.tempo_changes
    return f"MXL 解析正常：{result.duration:.2f}s / {len(result.tempo_changes)} 处变速"


def test_musicxml_render_5s() -> str:
    """MusicXML 只渲染 5 秒试听，验证不会输出整首 393 秒音频。"""
    import soundfile as sf

    if not os.path.exists(SCORE):
        raise FileNotFoundError(SCORE)
    options = RenderOptions(
        model_kind="markov",
        voice="哈基米",
        workers=1,
        start_time=150.0,
        max_seconds=5.0,
        tail_seconds=0.0,
        write_stems=False,
        write_zip=False,
        write_subtitle_videos=False,
        auto_mix=False,
        tracks=[
            {"index": 0, "selected": True, "voice": "哈基米", "octave": 0, "gain": 0.6},
            {"index": 2, "selected": True, "voice": "哈基米", "octave": 0, "gain": 0.6},
        ],
    )
    result = render_midi(
        SCORE,
        options,
        output_dir=os.path.join(OUT, "score_5s"),
        project_name="_selftest_score_5s",
    )
    data, sr = sf.read(result.audio_path)
    assert sr == 44100, sr
    assert 4.7 <= len(data) / sr <= 5.2, len(data) / sr
    assert result.duration <= 5.2, result.duration
    assert result.stats["analysis"]["source_format"] == "musicxml"
    assert result.stats["analysis"]["n_tempo_changes"] == 8
    # 150~155s 跨 153.2857 的 52.5→35 BPM 变速，力度也应来自谱面
    assert result.stats["render"]["n_dynamic_notes"] > 0, result.stats["render"]
    return (
        f"{len(data) / sr:.2f}s 试听（整首 {result.stats['analysis']['duration']:.0f}s）/ "
        f"{result.stats['render']['n_notes']} 音符"
    )


def test_pipeline() -> str:
    if not os.path.exists(DEMO):
        raise FileNotFoundError(f"缺少演示 MIDI：{DEMO}（先运行 scripts/make_demo_midi.py）")
    options = RenderOptions(
        model_kind="markov",
        voice="auto",
        workers=1,
        max_seconds=5.0,
        tracks=[
            {"index": 0, "selected": True, "voice": "auto", "octave": 0, "gain": 0.6},
            {"index": 1, "selected": True, "voice": "auto", "octave": 0, "gain": 0.35},
            {"index": 2, "selected": True, "voice": "auto", "octave": 0, "gain": 0.6},
        ],
    )
    result = render_midi(DEMO, options, output_dir=OUT, project_name="_selftest")
    for key in ("audio", "lyrics", "srt", "lrc", "html", "analysis", "subtitle_json", "markdown"):
        assert key in result.files, f"缺少输出：{key}"
        assert os.path.exists(result.files[key]), result.files[key]

    data, sr = sf.read(result.audio_path)
    assert sr == 44100, sr
    assert data.ndim == 2 and data.shape[1] == 2, data.shape
    assert len(data) > 44100, len(data)

    # 默认 auto 模式应包含哈基米主旋律 + 至少一种辅助音色
    voices = result.stats["lyrics"].get("voices", {})
    assert "哈基米" in voices, voices
    assert any(v in voices for v in ("叮咚鸡", "曼巴", "曼波only")), voices

    # 自动混音 / 母带响度
    mix = result.stats.get("mix") or {}
    assert mix.get("enabled") is True, mix
    assert mix.get("loudness_db", -99) > -20.0, mix
    assert len(mix.get("tracks") or []) == 3, mix
    assert -1 <= mix.get("correlation", 0) <= 1, mix
    assert any(t.get("pan", 0) < 0 for t in mix["tracks"]), mix["tracks"]
    assert any(t.get("pan", 0) > 0 for t in mix["tracks"]), mix["tracks"]

    # 渲染统计
    render_stats = result.stats["render"]
    assert render_stats["n_units"] > 0
    assert render_stats["n_dynamic_notes"] >= 1, render_stats
    assert render_stats["n_attack_softened"] > 0, render_stats
    assert render_stats["n_consonant_notes"] > 0, render_stats
    assert result.premix, "缺少混音前干声"
    assert result.files.get("params_zip") and os.path.exists(result.files["params_zip"])
    assert result.files.get("premix_zip") and os.path.exists(result.files["premix_zip"])
    assert result.bad_notes.get("total_notes", 0) > 0, result.bad_notes
    supplement = result.files.get("supplement_midi")
    assert supplement and os.path.exists(supplement), supplement
    import pretty_midi

    supplement_pm = pretty_midi.PrettyMIDI(supplement)
    assert supplement_pm.instruments, "补充 MIDI 应包含未输出轨道"
    assert result.supplement_midi.get("n_missing_tracks", 0) >= 1, result.supplement_midi
    percussion = result.files.get("percussion_midi")
    assert percussion and os.path.exists(percussion), percussion
    percussion_pm = pretty_midi.PrettyMIDI(percussion)
    assert percussion_pm.instruments and percussion_pm.instruments[0].is_drum, percussion_pm.instruments
    assert percussion_pm.instruments[0].name.endswith("_NOAUDIO"), percussion_pm.instruments[0].name
    assert result.duration > 0
    assert result.files["html"].endswith(".html")

    # 绿幕字幕视频必须随音频一起输出，且写进 ZIP
    video = result.subtitle_videos or {}
    assert video.get("available"), video
    assert video.get("width") == 1920, video
    assert 150 <= int(video.get("height") or 0) <= 700, video
    coverage = video.get("coverage") or {}
    assert coverage.get("missing_slots") == 0, coverage
    assert video.get("max_lines") == 1, video
    assert video.get("single_video") is True, video
    path = (video.get("files") or {}).get("all")
    assert path and os.path.exists(path), path
    assert os.path.getsize(path) > 1000, (path, os.path.getsize(path))
    assert len(video.get("files") or {}) == 1, video.get("files")
    assert "subtitle_videos" in result.files
    if result.files.get("zip"):
        import zipfile

        with zipfile.ZipFile(result.files["zip"]) as zf:
            names = set(zf.namelist())
        video_names = [
            name for name in names
            if name.startswith("videos/") and name.endswith(".mp4")
        ]
        assert len(video_names) == 1, video_names
        assert any(name.endswith("_bad_and_all.mid") for name in names), names
        assert any(name.endswith("_percussion.mid") for name in names), names
    return (
        f"{result.duration:.1f}s 音频 / {render_stats['n_unique_units']} 单位 / "
        f"母带 {mix.get('loudness_db')}dB / 语音 {voices} / 字幕 {len(result.subtitles)} 行 / "
        f"绿幕视频 {video.get('width')}×{video.get('height')}（每次 {video.get('max_lines')} 行）"
    )


def test_video_all_track_allocation() -> str:
    """全部轨道合并成一条字幕时间线，槽不丢失 / 不重复，每次 1 行。"""
    from types import SimpleNamespace

    from hachimi.subtitles import build_group_subtitles
    from hachimi.video import build_video_timeline

    counts = [10, 8, 6, 4, 2]
    slots = []
    slot_id = 0
    for track, count in enumerate(counts):
        for i in range(count):
            slots.append(
                SimpleNamespace(
                    index=slot_id,
                    primary_track=track,
                    word="哈" if i % 2 == 0 else "基",
                    start=slot_id * 0.3,
                    end=slot_id * 0.3 + 0.25,
                    section="A",
                    track_ids=[track],
                )
            )
            slot_id += 1
    tracks = list(range(len(counts)))
    lines = build_group_subtitles(slots, tracks, max_len=20, hold_gap=0.0)
    ids = [
        int(ch["slot"])
        for line in lines
        for ch in line.chars
        if ch.get("slot") is not None
    ]
    assert sorted(ids) == list(range(len(slots))), "有歌词槽没有进入字幕或重复"
    assert all(line.text for line in lines), "出现空字幕"
    timeline = build_video_timeline(lines, max(10.0, lines[-1].end + 1.0))
    shown = [segment for segment in timeline if segment.text]
    assert len(shown) == len(lines), f"有 {len(lines) - len(shown)} 行被时间轴吞掉"
    layout = measure_layout([lines], max_lines=1, width=1920)
    assert layout.max_lines == 1, layout
    return (
        f"{len(slots)} 槽 → {len(lines)} 行全部上屏；单条视频、每次 {layout.max_lines} 行"
    )


def test_proactor_noise_filter() -> str:
    """只过滤 Windows Proactor 的已知 10054 收尾噪音，其他错误保留。"""
    import logging

    from hachimi.server import _WindowsProactorResetFilter

    reset = ConnectionResetError(10054, "远程主机强迫关闭了一个现有的连接。")
    reset.winerror = 10054
    benign = logging.LogRecord(
        "asyncio",
        logging.ERROR,
        __file__,
        1,
        "Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)",
        (),
        (ConnectionResetError, reset, None),
    )
    other = logging.LogRecord(
        "asyncio",
        logging.ERROR,
        __file__,
        1,
        "Exception in callback something_else",
        (),
        (ConnectionResetError, reset, None),
    )
    flt = _WindowsProactorResetFilter()
    assert flt.filter(benign) is False, "已知 10054 噪音应被过滤"
    assert flt.filter(other) is True, "其他 asyncio 错误不应被误杀"
    return "仅过滤 _call_connection_lost + WinError 10054"


def test_open_source_no_llm() -> str:
    """开源版不得包含本地 LLM 代码、入口或任何 API key。"""
    import re

    from hachimi.config import RenderOptions

    assert not os.path.exists(os.path.join(ROOT, "hachimi", "llm.py")), "llm.py 仍存在"
    assert "use_llm" not in RenderOptions.__dataclass_fields__, "RenderOptions 仍保留 use_llm"

    code_files = [
        os.path.join(ROOT, "hachimi", "config.py"),
        os.path.join(ROOT, "hachimi", "pipeline.py"),
        os.path.join(ROOT, "hachimi", "server.py"),
        os.path.join(ROOT, "main.py"),
        os.path.join(ROOT, "launcher.py"),
        os.path.join(ROOT, "web", "index.html"),
        os.path.join(ROOT, "web", "app.js"),
    ]
    forbidden = (
        "hachimi.llm",
        "use_llm",
        "llm_base_url",
        "llm_model",
        "/api/llm",
        "LLM_BASE_URL",
        "HACHIMI_LLM",
    )
    for path in code_files:
        text = open(path, encoding="utf-8").read()
        for token in forbidden:
            assert token not in text, f"{os.path.basename(path)} 仍包含 {token}"

    scan_roots = ("hachimi", "web", "scripts", "legacy")
    key_pattern = re.compile(r"sk-[A-Za-z0-9_\-]{16,}")
    scanned = 0
    for root_name in scan_roots:
        for dirpath, _dirnames, filenames in os.walk(os.path.join(ROOT, root_name)):
            for filename in filenames:
                if not filename.endswith((".py", ".js", ".html", ".md", ".json", ".txt")):
                    continue
                path = os.path.join(dirpath, filename)
                try:
                    text = open(path, encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                scanned += 1
                assert not key_pattern.search(text), f"{path} 含疑似 sk- API key"
    return f"无 llm.py / use_llm / /api/llm；扫描 {scanned} 个源码文件未发现 sk- key"


# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 72)
    print("Hachimi Sonata · 哈基米音律工坊 自检")
    print("=" * 72)
    if not os.path.exists(DEMO):
        print("生成演示 MIDI…")
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from make_demo_midi import build_demo

        build_demo(DEMO)

    check("和弦识别", test_chords)
    check("分解和弦槽（跨声部同字）", test_decomposed_slots)
    check("演示 MIDI 分析", test_analysis_demo)
    check("调性 / 段落计划 / 每段随机", test_key_and_section_plans)
    check("小提琴默认音量 1.5", test_violin_default_gain)
    check("跨声部同一和弦 → 同一歌词", test_cross_voice_same_word)
    check("同一动机 → 同一歌词", test_motif_same_lyrics)
    check("MIDI 力度 → 音量", test_velocity_gain)
    check("音源压缩 / 高音可听度", test_audio_loudness_and_extreme_pitch)
    check("齿音 / 清辅音衰减", test_sibilance_and_voiceless)
    check("时值拉伸（只拉中段）", test_note_stretch_middle_only)
    check("弱化音头 / 连奏过渡", test_legato_and_soft_attack)
    check("渐强渐弱 / 自动混音摆位", test_expression_and_auto_mix)
    check("混音总线压缩 / 自动补偿", test_mix_bus_compression)
    check("绿幕字幕视频（1920 / 单条 / 每次 1 行）", test_subtitle_video)
    check("全轨字幕合并（不丢失 / 不重复 / 单行）", test_video_all_track_allocation)
    check("实时字幕（SRT/LRC/HTML）", test_subtitles)
    check("标准句法 + 补充采样", test_syntax_and_supplement)
    check("第二套音源 + 低音声部", test_sample_set_and_bass)
    check("全声库调音 / 音头一致性", test_voice_tuning_consistency)
    check("GPU 混音一致性", test_gpu_mixing)
    check("发声校验的负向检测", test_verification_negative)
    check("断奏 gate 不被误判为漏音", test_gate_verification)
    check("辅音 / 爆破音衰减", test_consonant_plosive)
    check("不良音符判定 + 专用 MIDI", test_bad_notes_and_midi)
    check("补充 MIDI（_NG / 空白 / 全谱）", test_supplement_midi)
    check("打击乐 <unpitched> 解析 + 单独 MIDI", test_percussion_unpitched)
    check("工程参数包往返 / 旧 manifest", test_params_roundtrip)
    check("MusicXML 速度 / 力度 / 奏法", test_musicxml_analysis)
    check("MXL 压缩乐谱", test_musicxml_mxl)
    check("MusicXML 5 秒试听", test_musicxml_render_5s)
    check("分轨导出 + ZIP + 和弦发声", test_stems_zip_and_polyphony)
    check("端到端渲染", test_pipeline)
    check("开源版无 LLM / 无 API key", test_open_source_no_llm)
    check("Windows 连接收尾噪音过滤", test_proactor_noise_filter)

    print("-" * 72)
    print(f"通过 {PASS} 项，失败 {FAIL} 项")
    print("=" * 72)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
