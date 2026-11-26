from character_markov import *
from pitchshift import *
from reverb import *
from animate import draw
import pretty_midi
import soundfile as sf
import os
import glob
import re
import numpy as np
import json
import random
from tqdm import tqdm

# ========== 固定参数（均使用默认值）==========
available_classes = ["哈基米", "哈基米only", "曼波only"]
DEFAULT_CLASS = available_classes[0]   # "哈基米"
DEFAULT_OCTAVE_SHIFT = 0
DEFAULT_VOLUME = 1.0
LYRICS_PER_GROUP = 15                  # 每组分段歌词数
SIMULTANEOUS_THRESHOLD = 0.05
SR = 44100
FRAME_PERIOD = 5.0

# 采样对应的帧长度（用于预加重）
cons_frames = {
    "哈": 3, "基": 8, "米": 8, "摸": 8, "南": 8, "北": 8,
    "绿": 8, "豆": 8, "阿": 8, "西": 30, "噶": 8, "呀": 8,
    "库": 8, "那": 8, "路": 8, "曼": 8, "波": 8
}

# ========== 辅助函数（与原代码一致）==========
def sec_to_srt(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def make_track_srt(note2word, track_idx, output_path, reset_gap=1.0, max_duration=2.5, max_len=12):
    items = [
        {"start": n.start, "end": n.end, "word": w}
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

        if prev_t is not None and start - prev_t > reset_gap:
            text = ""
            seg_start = None

        if seg_start is None:
            seg_start = start

        text += word

        if len(text) > max_len:
            text = word
            seg_start = start

        if i + 1 < len(items):
            next_start = items[i + 1]["start"]
            end_time = min(next_start, start + max_duration)
        else:
            end_time = start + max_duration

        if end_time - seg_start < 0.08:
            end_time = seg_start + 0.08

        subtitles.append((idx, seg_start, end_time, text))
        idx += 1
        prev_t = start

    with open(output_path, "w", encoding="utf-8-sig") as f:
        for idx, s, e, t in subtitles:
            f.write(f"{idx}\n")
            f.write(f"{sec_to_srt(s)} --> {sec_to_srt(e)}\n")
            f.write(f"{t}\n\n")

def build_events(note2word):
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
    events.sort(key=lambda e: (e["start"], e["end"]))
    return events

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

def synthesize_midi(preproc, note2word, midi, sr=SR, frame_period=FRAME_PERIOD,
                    start_time=0.0, end_time=None, octave_shift=None,
                    volume_factor=None, ap_scale=0.05, sp_scale=1.0):
    n_tracks = len(midi.instruments)
    if octave_shift is None:
        octave_shift = [0] * n_tracks
    if volume_factor is None:
        volume_factor = [1.0] * n_tracks

    total_time = midi.get_end_time() if end_time is None else end_time
    n_samples = int(total_time * sr)
    y_stereo = np.zeros((n_samples, 2), dtype=np.float32)

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
            continue

        for note in instrument.notes:
            if note.end <= start_time or note.start >= total_time:
                continue

            word = note2word[idx].get(note)
            if word is None:
                continue

            pbar.update(1)

            start = max(int((note.start - start_time) * sr), 0)
            end = min(int((note.end - start_time) * sr), n_samples)
            dur = end - start
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

def render_group_audio(group_events, group_idx, src_notes, preproc_data,
                       sr, audio_dir, proj_name, midi_dir):
    """将一组歌词事件分割为独立MIDI并渲染音频，返回音频数组和该组原始起始时间"""
    if not group_events:
        print(f"  跳过第 {group_idx} 组（空事件）")
        return None, None

    group_events = sorted(group_events, key=lambda e: (e["start"], e["end"]))
    t0 = group_events[0]["start"]                  # 组内最早开始（原始绝对时间）
    t1 = max(evt["end"] for evt in group_events)    # 组内最晚结束

    sub_midi = pretty_midi.PrettyMIDI()
    sub_inst = pretty_midi.Instrument(program=0, is_drum=False, name=f"Piano_Part_{group_idx:03d}")
    sub_note2word = [{}]

    for evt in group_events:
        found = False
        for note in src_notes:
            if (note.pitch == evt["pitch"] and
                abs(note.start - evt["start"]) < 1e-5 and
                abs(note.end - evt["end"]) < 1e-5):
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start - t0,
                    end=note.end - t0
                )
                sub_inst.notes.append(new_note)
                sub_note2word[0][new_note] = evt["state"]
                found = True
                break
        if not found:
            print(f"  警告: 第 {group_idx} 组未匹配 pitch={evt['pitch']} start={evt['start']:.4f}")

    if not sub_inst.notes:
        print(f"  跳过第 {group_idx} 组（无匹配音符）")
        return None, None

    sub_inst.notes.sort(key=lambda n: (n.start, n.end))
    sub_midi.instruments.append(sub_inst)

    # MIDI 文件保存在项目根目录（非音频目录）
    part_mid_path = os.path.join(midi_dir, f"{proj_name}_part_{group_idx:03d}.mid")
    sub_midi.write(part_mid_path)

    sub_end = sub_midi.get_end_time()
    y_sub = synthesize_midi(preproc_data, sub_note2word, sub_midi,
                            start_time=0.0, end_time=sub_end,
                            octave_shift=[0], volume_factor=[1.0],
                            ap_scale=0.05, sp_scale=1.0)

    # 特效处理
    y_sub = vocal_low_cut(y_sub, sr)
    y_sub = stereo_spread(y_sub.T, spread_amount=0.15).T
    y_sub = multitap_delay(y_sub, sr)
    y_sub = hf_damping(y_sub, sr, cutoff=10000)

    # 音频文件保存在 audio_dir
    part_wav_path = os.path.join(audio_dir, f"{proj_name}_part_{group_idx:03d}.wav")
    sf.write(part_wav_path, y_sub, sr)
    print(f"  已保存 [{group_idx:03d}]: {part_mid_path} + {part_wav_path} (时长 {sub_end:.3f}s)")

    return y_sub, t0   # 返回音频数组和原始起始时间

# ========== 单个 MIDI 处理流程 ==========
def process_midi(midi_path):
    project_name = os.path.splitext(os.path.basename(midi_path))[0]
    # 修改输出根目录为 ./output/all/ 实现汇总
    output_dir = f"./output/all/{project_name}/"
    os.makedirs(output_dir, exist_ok=True)

    # 创建音频子目录
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    sample_path = "./assets/samples/"
    out_model_path = "./models/character_markov/"
    os.makedirs(out_model_path, exist_ok=True)

    midi = pretty_midi.PrettyMIDI(midi_path)

    # ---- 合并所有非打击乐轨道为单一钢琴轨道 ----
    piano = pretty_midi.Instrument(program=0, is_drum=False, name="Piano")
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            piano.notes.append(pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start,
                end=note.end
            ))
    piano.notes.sort(key=lambda n: (n.start, n.end))
    midi.instruments = [piano]

    n_tracks = len(midi.instruments)
    print(f"\n成功加载 MIDI: {midi_path}")
    print(f"合并后轨道数: {n_tracks}")

    # ---- 使用默认参数（不再交互） ----
    track_classes = [DEFAULT_CLASS]
    octave_shift = [DEFAULT_OCTAVE_SHIFT]
    volume_factor = [DEFAULT_VOLUME]

    # ---- 训练马尔可夫模型 ----
    track_models = []
    for cls in track_classes:
        model = train(allowed_classes=[cls])
        track_models.append(model)
        draw(model, save_path=out_model_path + f"{cls}.png")
        with open(f"{out_model_path}{cls}.json", "w", encoding="utf-8") as f:
            json.dump(model, f, ensure_ascii=False, indent=2)

    # ---- 生成歌词 ----
    start_time = 0.0
    midi_duration = midi.get_end_time()
    end_time = midi_duration + 2.0
    track_words, note2word = generate_lyrics(midi, track_models)
    events = build_events(note2word)

    # ---- 保存完整歌词 JSON ----
    out_lyrics_path = os.path.join(output_dir, f"{project_name}_lyrics.json")
    with open(out_lyrics_path, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)
    print(f"\n歌词事件已保存: {out_lyrics_path} ({len(events)} 个事件)")

    # ---- 预处理采样 ----
    preproc = {}
    for cls in track_classes:
        for sample in alphabet[cls]:
            if sample in preproc:
                continue
            path = sample_path + f"{sample}.wav"
            preproc[sample] = preprocess_sample(path, cons_frame=cons_frames.get(sample, 8))

    # ---- 按 15 个歌词事件分组并渲染每组 ----
    sorted_events = sorted(events, key=lambda e: (e["start"], e["end"]))
    total = len(sorted_events)
    num_groups = (total + LYRICS_PER_GROUP - 1) // LYRICS_PER_GROUP
    print(f"\n按歌词分割: 共 {total} 个事件 → {num_groups} 组（每 {LYRICS_PER_GROUP} 个一组）")

    src_notes = midi.instruments[0].notes
    group_audios = []   # 存储 (音频数组, 原始起始时间)
    for g in range(num_groups):
        group_events = sorted_events[g * LYRICS_PER_GROUP : (g + 1) * LYRICS_PER_GROUP]
        y_sub, t0 = render_group_audio(group_events, g + 1, src_notes, preproc,
                                       SR, audio_dir, project_name, output_dir)
        if y_sub is not None:
            group_audios.append((y_sub, t0))

    # ---- 按原始时间轴拼接完整音频（保留组间空白） ----
    print("\n正在按原始时间轴拼接完整音频...")
    if not group_audios:
        print("  无有效音频组，跳过拼接。")
        return

    total_samples = int(midi_duration * SR)
    full_audio = np.zeros((total_samples, 2), dtype=np.float32)

    for y_sub, t0 in group_audios:
        start_sample = int(t0 * SR)
        end_sample = start_sample + len(y_sub)
        if end_sample > total_samples:
            # 若超出总时长则截断（一般不会）
            y_sub = y_sub[:total_samples - start_sample]
            end_sample = total_samples
        full_audio[start_sample:end_sample] += y_sub

    # 归一化防溢出
    max_val = np.max(np.abs(full_audio))
    if max_val > 1.0:
        full_audio /= max_val

    full_wav_path = os.path.join(audio_dir, f"{project_name}_full.wav")
    sf.write(full_wav_path, full_audio, SR)
    print(f"完整音频已保存: {full_wav_path}")
    print(f"  总时长: {midi_duration:.3f}s (原始MIDI时长)")

    # ---- 生成字幕 ----
    make_track_srt(
        note2word,
        track_idx=0,
        output_path=os.path.join(output_dir, "track0.srt"),
        reset_gap=1.0,
        max_len=12
    )
    print(f"字幕已保存: {os.path.join(output_dir, 'track0.srt')}")

# ========== 主入口：批量处理所有 MIDI ==========
if __name__ == "__main__":
    # 从当前目录下的 midi 文件夹递归读取所有 .mid 文件
    midi_files = sorted(glob.glob("./midi/**/*.mid", recursive=True))
    if not midi_files:
        print("未在 midi 文件夹中找到任何 MIDI 文件。")
        exit(1)

    print(f"找到 {len(midi_files)} 个 MIDI 文件，开始批量处理...")
    for i, midi_path in enumerate(midi_files):
        print(f"\n========== 处理 [{i+1}/{len(midi_files)}]: {midi_path} ==========")
        try:
            process_midi(midi_path)
        except Exception as e:
            print(f"处理 {midi_path} 时出错: {e}")
            continue
    print("\n所有 MIDI 处理完成。")