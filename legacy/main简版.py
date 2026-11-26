from pitchshift import *
from reverb import *
import pretty_midi
import soundfile as sf
import os
import glob
import re
import json
import numpy as np
from tqdm import tqdm

# ========== 纯 CPU 版本（无需显卡）==========

def select_midi():
    midi_files = sorted(glob.glob("./assets/midi/**/*.mid", recursive=True))
    if not midi_files:
        print("未在 assets/midi 中找到 MIDI 文件。")
        exit(1)
    
    print("\n找到以下 MIDI 文件：")
    for i, path in enumerate(midi_files):
        print(f"[{i}] {path}")
    
    while True:
        try:
            choice = int(input("\n请输入要处理的 MIDI 文件编号: "))
            if 0 <= choice < len(midi_files):
                return midi_files[choice]
        except ValueError:
            pass
        print("输入无效，请输入正确的数字。")

def select_json(project_name):
    json_files = sorted(glob.glob(f"./output/{project_name}/*_lyrics.json"))
    if not json_files:
        print(f"未在 ./output/{project_name}/ 中找到歌词 JSON 文件。")
        exit(1)
    
    print(f"\n找到以下歌词 JSON 文件：")
    for i, path in enumerate(json_files):
        print(f"[{i}] {path}")
    
    while True:
        try:
            choice = int(input("\n请输入要处理的歌词 JSON 文件编号: "))
            if 0 <= choice < len(json_files):
                return json_files[choice]
        except ValueError:
            pass
        print("输入无效，请输入正确的数字。")

midi_path = select_midi()
project_name = os.path.splitext(os.path.basename(midi_path))[0]
output_dir = f"./output/{project_name}/"
os.makedirs(output_dir, exist_ok=True)

# ========== 读取歌词 JSON ==========
json_path = select_json(project_name)
with open(json_path, "r", encoding="utf-8") as f:
    events = json.load(f)

print(f"\n已读取歌词事件: {json_path} ({len(events)} 个事件)")

# ========== 读取完整 MIDI 并合并为单一钢琴轨道 ==========
midi = pretty_midi.PrettyMIDI(midi_path)

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

print(f"已加载 MIDI: {midi_path} ({len(piano.notes)} 个音符)")

sample_path = "./assets/samples/"
sr = 44100
frame_period = 5.0

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

# ========== 【新增】输入分割步数 ==========
DEFAULT_LYRICS_PER_GROUP = 15

print(f"\n--- 歌词分割设置 ---")
print(f"默认每 {DEFAULT_LYRICS_PER_GROUP} 个歌词事件分为一组，生成独立的 MIDI 和音频文件。")
group_input = input(f"请输入每组歌词个数（直接回车使用默认值 {DEFAULT_LYRICS_PER_GROUP}）: ")

if group_input.strip() == "":
    LYRICS_PER_GROUP = DEFAULT_LYRICS_PER_GROUP
else:
    try:
        LYRICS_PER_GROUP = max(1, int(group_input))
    except ValueError:
        print(f"输入无效，使用默认值 {DEFAULT_LYRICS_PER_GROUP}")
        LYRICS_PER_GROUP = DEFAULT_LYRICS_PER_GROUP

print(f"已设置: 每 {LYRICS_PER_GROUP} 个歌词事件为一组")

# ========== 预处理采样（只需一次） ==========
# 从 JSON 中提取所有用到的字
used_words = set(evt["state"] for evt in events)

preproc = {}
for word in used_words:
    if word in preproc:
        continue
    path = sample_path + f"{word}.wav"
    if not os.path.exists(path):
        print(f"警告: 采样文件不存在 {path}，跳过")
        continue
    preproc[word] = preprocess_sample(path, cons_frame=cons_frames.get(word, 8))

print(f"已预处理 {len(preproc)} 个采样")

# ========== 按歌词分组 → 分割MIDI + 每组单独渲染音频 ==========

def render_group_audio(group_events, group_idx, src_notes, preproc_data, sr, out_dir, proj_name):
    """将一组歌词事件分割为独立MIDI并渲染音频（去除所有多余前后空白）"""
    if not group_events:
        print(f"  跳过第 {group_idx} 组（空事件）")
        return
    
    group_events = sorted(group_events, key=lambda e: (e["start"], e["end"]))
    t0 = group_events[0]["start"]
    t1 = max(evt["end"] for evt in group_events)
    
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
        return
    
    sub_inst.notes.sort(key=lambda n: (n.start, n.end))
    sub_midi.instruments.append(sub_inst)
    
    part_mid_path = os.path.join(out_dir, f"{proj_name}_part_{group_idx:03d}.mid")
    sub_midi.write(part_mid_path)
    
    sub_end = sub_midi.get_end_time()
    y_sub = synthesize_midi(preproc_data, sub_note2word, sub_midi,
                            start_time=0.0, end_time=sub_end,
                            octave_shift=[0], volume_factor=[1.0],
                            ap_scale=0.05, sp_scale=1.0)
    
    y_sub = vocal_low_cut(y_sub, sr)
    y_sub = stereo_spread(y_sub.T, spread_amount=0.15).T
    y_sub = multitap_delay(y_sub, sr)
    y_sub = hf_damping(y_sub, sr, cutoff=10000)
    
    part_wav_path = os.path.join(out_dir, f"{proj_name}_part_{group_idx:03d}.wav")
    sf.write(part_wav_path, y_sub, sr)
    print(f"  已保存 [{group_idx:03d}]: {part_mid_path} + {part_wav_path} (时长 {sub_end:.3f}s)")

# 执行分组渲染
sorted_events = sorted(events, key=lambda e: (e["start"], e["end"]))
total = len(sorted_events)
num_groups = (total + LYRICS_PER_GROUP - 1) // LYRICS_PER_GROUP
print(f"\n按歌词分割: 共 {total} 个事件 → {num_groups} 组（每 {LYRICS_PER_GROUP} 个一组）")

src_notes = midi.instruments[0].notes
for g in range(num_groups):
    group_events = sorted_events[g * LYRICS_PER_GROUP : (g + 1) * LYRICS_PER_GROUP]
    render_group_audio(group_events, g + 1, src_notes, preproc, sr, output_dir, project_name)

# ========== 按顺序拼接所有 part WAV 为完整音频 ==========
print("\n正在拼接完整音频...")

part_wav_files = glob.glob(os.path.join(output_dir, f"{project_name}_part_*.wav"))

def _extract_index(path):
    m = re.search(rf"{re.escape(project_name)}_part_(\d+)\.wav$", os.path.basename(path))
    return int(m.group(1)) if m else 0

part_wav_files.sort(key=_extract_index)

if not part_wav_files:
    print("  未找到分段 WAV 文件，跳过拼接。")
else:
    chunks = []
    expected_sr = sr
    total_duration = 0.0
    for wav_path in part_wav_files:
        data, read_sr = sf.read(wav_path)
        if read_sr != expected_sr:
            print(f"  警告: {wav_path} 采样率 {read_sr} 与预期 {expected_sr} 不一致，跳过")
            continue
        chunks.append(data)
        total_duration += len(data) / expected_sr
        print(f"  读取: {os.path.basename(wav_path)} ({len(data)} 采样点, {len(data)/expected_sr:.3f}s)")
    
    if chunks:
        full_audio = np.concatenate(chunks, axis=0)
        full_wav_path = os.path.join(output_dir, f"{project_name}_full.wav")
        sf.write(full_wav_path, full_audio, expected_sr)
        print(f"\n完整音频已保存: {full_wav_path}")
        print(f"  总时长: {total_duration:.3f}s ({len(full_audio)} 采样点)")
    else:
        print("  无可拼接的有效音频块。")
