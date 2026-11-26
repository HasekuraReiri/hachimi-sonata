from character_markov import *
from pitchshift import *
from reverb import *
from animate import draw
import pretty_midi
import mido
import soundfile as sf
import os

import glob
import json
import random
import numpy as np
from tqdm import tqdm

# ----------------------------------------------------------------------
# 以下所有函数和常量定义与原代码完全一致，未作任何修改
# ----------------------------------------------------------------------

SIMULTANEOUS_THRESHOLD = 0.05

def generate_lyrics(midi, models, start_time=0.0, end_time=None):
    note2word = []
    track_words_all = []

    valid_notes = []
    for inst in midi.instruments:
        for note in inst.notes:
            if note.end > start_time:
                if end_time is None or note.start < end_time:
                    valid_notes.append(note)

    pbar = tqdm(total=len(valid_notes), desc="编写歌词中：")

    for idx, inst in enumerate(midi.instruments):

        notes = sorted(inst.notes, key=lambda n: (n.start, n.end))
        if not notes:
            note2word.append({})
            track_words_all.append([])
            continue

        # ===== 和弦分组 =====
        groups = []
        current_group = [notes[0]]
        current_start = notes[0].start

        for note in notes[1:]:
            if note.start - current_start <= SIMULTANEOUS_THRESHOLD:
                current_group.append(note)
            else:
                groups.append(current_group)
                current_group = [note]
                current_start = note.start
        groups.append(current_group)

        model = models[idx] if idx < len(models) else models[0]
        if not model:
            note2word.append({})
            track_words_all.append([])
            continue

        current_token = random.choice(list(model.keys()))
        mapping = {}
        track_words = []

        for group in groups:
            word = current_token
            track_words.append(word)

            main_note = max(group, key=lambda n: n.pitch)

            pbar.update(len(group))

            mapping[main_note] = word

            # next token
            next_options = model.get(current_token, {})
            if next_options:
                current_token = random.choices(
                    list(next_options.keys()),
                    weights=list(next_options.values()),
                    k=1
                )[0]
            else:
                current_token = random.choice(list(model.keys()))

        note2word.append(mapping)
        track_words_all.append(track_words)

    return track_words_all, note2word

def synthesize_midi(preproc, note2word, midi, sr=44100, frame_period=5.0, start_time=0.0, end_time=60, octave_shift=None, volume_factor=None, ap_scale=0.05, sp_scale=1.0):
    n_tracks = len(midi.instruments)
    if octave_shift is None:
        octave_shift = [0] * n_tracks
    if volume_factor is None:
        volume_factor = [1.0] * n_tracks

    total_time = midi.get_end_time() if end_time is None else end_time
    n_samples = int(total_time * sr)
    y_stereo = np.zeros((n_samples, 2))

    valid_notes = []
    for idx, inst in enumerate(midi.instruments):
        if volume_factor[idx] == 0:
            continue

        for note in inst.notes:
            if note.end > start_time and note.start < end_time:
                valid_notes.append(note)

    pbar = tqdm(total=len(valid_notes), desc="Rendering MIDI")

    for idx, instrument in enumerate(midi.instruments):
        shift = octave_shift[idx] * 12
        vol = volume_factor[idx]

        if vol == 0:
            continue;

        for note in instrument.notes:
            if note.end <= start_time or note.start >= total_time:
                continue

            word = note2word[idx].get(note)
            if word is None:
                continue

            pbar.update(1)

            start = max(int((note.start-start_time)*sr), 0)
            end   = min(int((note.end-start_time)*sr), n_samples)
            dur   = end - start
            if dur <= 0:
                continue

            target_pitch = note.pitch + shift

            y_note = synthesize_note(
                preproc[word],
                target_pitch,
                dur,
                sr=sr,
                frame_period=frame_period,
                ap_scale=ap_scale,
                sp_scale=sp_scale
            )

            y_stereo[start:end, :] += y_note * vol

    max_amp = np.max(np.abs(y_stereo))
    if max_amp > 1.0:
        y_stereo /= max_amp

    return y_stereo

def build_events(note2word):
    """
    note2word: List[Dict[pretty_midi.Note, str]]

    return:
        events: List[Dict]
    """
    events = []

    for track_idx, track_map in enumerate(note2word):
        for note, word in track_map.items():
            events.append({
                "start": float(note.start),
                "end": float(note.end),
                "state": word,
                "track": track_idx,
                "pitch": int(note.pitch)
            })

    # 按时间排序（非常重要）
    events.sort(key=lambda e: (e["start"], e["end"]))

    return events

def sec_to_srt(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def make_track_srt(
    note2word,
    track_idx,
    output_path,
    reset_gap=1.0,
    max_duration=2.5,
    max_len=12
):
    items = [
        {
            "start": n.start,
            "end": n.end,
            "word": w
        }
        for n, w in note2word[track_idx].items()
    ]

    items.sort(key=lambda x: x["start"])

    subtitles = []

    text = ""
    seg_start = None
    prev_t = None
    idx = 1

    for i, it in enumerate(items):
        start = it["start"]
        word = it["word"]

        # reset segment（断句）
        if prev_t is not None and start - prev_t > reset_gap:
            text = ""
            seg_start = None

        # init segment start
        if seg_start is None:
            seg_start = start

        # accumulate text（token累积显示）
        text += word

        # max length protection（防爆）
        if len(text) > max_len:
            text = word
            seg_start = start

        # compute end time（关键：不重叠）
        if i + 1 < len(items):
            next_start = items[i + 1]["start"]
            end_time = min(next_start, start + max_duration)
        else:
            end_time = start + max_duration

        # 最小显示时长（防闪烁）
        if end_time - seg_start < 0.08:
            end_time = seg_start + 0.08

        subtitles.append((idx, seg_start, end_time, text))

        idx += 1
        prev_t = start

    # write SRT
    with open(output_path, "w", encoding="utf-8-sig") as f:
        for idx, s, e, t in subtitles:
            f.write(f"{idx}\n")
            f.write(f"{sec_to_srt(s)} --> {sec_to_srt(e)}\n")
            f.write(f"{t}\n\n")

# ----------------------------------------------------------------------
# 以下为原代码中的常量和默认配置（未改动）
# ----------------------------------------------------------------------

available_classes = ["哈基米", "哈基米only", "曼波only"]
cons_frames = {
    "哈": 3,
    "基": 8,
    "米": 8,
    "摸": 8,
    "南": 8,
    "北": 8,
    "绿": 8,
    "豆": 8,
    "阿": 8,
    "西": 30,
    "噶": 8,
    "呀": 8,
    "库": 8,
    "那": 8,
    "路": 8,
    "曼": 8,
    "波": 8
}

default_track_classes = [
    0,0,0,0,0,0,0,0,0,   # 0-8 静音
    0,                   # 9 Piano
    2,                   # 10 Violin I Synth
    2,                   # 11 Violin I Pizz
    2,                   # 12 Violin I Main
    2,                   # 13 Violin II Synth
    2,                   # 14 Violin II Main
    2,                   # 15 Violin II Pizz
    0,0,0,0,0,0,0,0      # 16-23 静音
]
default_octave_shift = [
    0,0,0,0,0,0,0,0,0,
    1,    # Piano
    -1,    # Vln I Synth
    -1,    # Vln I Pizz
    -1,    # Vln I Main
    -1,    # Vln II Synth
    -1,    # Vln II Main
    -1,    # Vln II Pizz
    0,0,0,0,0,0,0,0
]
default_volume_factor = [
    0,0,0,0,0,0,0,0,0,
    0.7,   # Piano
    1,   # Vln I Synth
    1,   # Vln I Pizz
    1,   # Vln I Main
    0,   # Vln II Synth
    0,   # Vln II Main
    0,   # Vln II Pizz
    0,0,0,0,0,0,0,0
]

# ----------------------------------------------------------------------
# 主程序：自动循环处理所有 MIDI 文件
# ----------------------------------------------------------------------

def process_midi_file(midi_path):
    """处理单个 MIDI 文件（无交互，全部使用默认配置）"""
    project_name = os.path.splitext(os.path.basename(midi_path))[0]
    output_dir = f"./output/{project_name}/"
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, f"{project_name}.wav")
    sample_path = "./assets/samples/"
    out_lyrics_path = os.path.join(output_dir, f"{project_name}_lyrics.json")
    out_model_path = "./models/character_markov/"
    os.makedirs(out_model_path, exist_ok=True)

    midi = pretty_midi.PrettyMIDI(midi_path)

    # --- 自动匹配轨道信息（仅打印，无交互）---
    n_tracks = len(midi.instruments)
    print(f"\n成功加载 MIDI: {midi_path}")
    print(f"检测到 {n_tracks} 个轨道:")
    for i, inst in enumerate(midi.instruments):
        name = inst.name if inst.name else "未命名轨道"
        notes_count = len(inst.notes)
        is_drum = " (打击乐)" if inst.is_drum else ""
        print(f"  [{i}] {name} - {notes_count} 个音符{is_drum}")

    # --- 轨道配置直接使用默认值（无交互）---
    track_classes = []
    octave_shift = []
    volume_factor = []

    for i, inst in enumerate(midi.instruments):
        name = inst.name if inst.name else "未命名轨道"
        is_drum = inst.is_drum

        if is_drum:
            # 打击乐默认静音
            track_classes.append(available_classes[0])
            octave_shift.append(0)
            volume_factor.append(0.0)
            continue

        # 从默认列表中取值，若索引超出则使用安全默认值
        if i < len(default_track_classes):
            cls_idx = default_track_classes[i]
        else:
            cls_idx = 0
        track_classes.append(available_classes[cls_idx])

        if i < len(default_octave_shift):
            octave_shift.append(default_octave_shift[i])
        else:
            octave_shift.append(0)

        if i < len(default_volume_factor):
            volume_factor.append(default_volume_factor[i])
        else:
            volume_factor.append(1.0)

    # 原有的其他参数
    start_time = 500
    midi_duration = midi.get_end_time()
    end_time = midi_duration + 2.0
    print(f"音频长度预计: {end_time:.2f} 秒")
    sr = 44100
    frame_period = 5.0

    # 训练模型
    track_models = []
    for cls in track_classes:
        model = train(allowed_classes=[cls])
        track_models.append(model)
        draw(model, save_path=out_model_path + f"{cls}.png")
        with open(f"{out_model_path}{cls}.json", "w", encoding="utf-8") as f:
            json.dump(model, f, ensure_ascii=False, indent=2)

    # 生成歌词映射
    track_words, note2word = generate_lyrics(midi, track_models)

    # 保存事件
    events = build_events(note2word)
    with open(out_lyrics_path, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)

    # 预处理样本
    preproc = {}
    for cls in track_classes:
        for sample in alphabet[cls]:
            if sample in preproc:
                continue
            path = sample_path + f"{sample}.wav"
            preproc[sample] = preprocess_sample(path, cons_frame=cons_frames.get(sample, 8))

    # 合成音频
    y_out = synthesize_midi(preproc, note2word, midi, start_time=start_time, end_time=end_time,
                            octave_shift=octave_shift, volume_factor=volume_factor,
                            ap_scale=0.05, sp_scale=1.0)

    # 后期效果（原样保留）
    y_out = vocal_low_cut(y_out, sr)
    y_out = stereo_spread(y_out.T, spread_amount=0.15).T
    y_out = multitap_delay(y_out, sr)
    y_out = hf_damping(y_out, sr, cutoff=10000)

    # 生成字幕（固定轨道9）
    make_track_srt(
        note2word,
        track_idx=9,
        output_path=output_dir + "track9.srt",
        reset_gap=1.0,
        max_len=12
    )

    # 音频保存（保持原样注释状态）
    # y_out = soft_normalize(y_out, 0.95)
    # sf.write(out_path, y_out, sr)
    # print(f"音频已保存至 {out_path}")

    print(f"处理完成: {project_name}\n")

# ----------------------------------------------------------------------
# 主入口：遍历所有 MIDI 文件并依次处理
# ----------------------------------------------------------------------

if __name__ == "__main__":
    midi_files = sorted(glob.glob("./assets/midi/**/*.mid", recursive=True))
    if not midi_files:
        print("未在 assets/midi 中找到任何 MIDI 文件。")
        exit(1)

    print(f"找到 {len(midi_files)} 个 MIDI 文件，开始自动处理...")
    for midi_path in midi_files:
        process_midi_file(midi_path)
    print("所有文件处理完毕。")
