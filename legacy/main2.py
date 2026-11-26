from character_markov import *
from pitchshift import *
from reverb import *
from animate import draw
import pretty_midi
import mido
import soundfile as sf
import os
import glob
import re
import numpy as np

# ========== GPU: CUDA 支持 ==========
try:
    import torch
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"GPU 加速已启用: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device("cpu")
        print("CUDA 未检测到，使用 CPU 计算")
except ImportError:
    torch = None
    DEVICE = None
    print("PyTorch 未安装，使用 CPU 计算")

USE_GPU = (torch is not None) and (DEVICE.type == "cuda")
# ====================================

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

midi_path = select_midi()
project_name = os.path.splitext(os.path.basename(midi_path))[0]
output_dir = f"./output/{project_name}/"
os.makedirs(output_dir, exist_ok=True)

sample_path = "./assets/samples/"
out_model_path = "./models/character_markov/"
os.makedirs(out_model_path, exist_ok=True)

midi = pretty_midi.PrettyMIDI(midi_path)

# ========== 合并所有非打击乐轨道为单一钢琴轨道（索引 0）==========
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
# ================================================================

n_tracks = len(midi.instruments)
print(f"\n成功加载 MIDI: {midi_path}")
print(f"检测到 {n_tracks} 个轨道:")

for i, inst in enumerate(midi.instruments):
    name = inst.name if inst.name else "未命名轨道"
    notes_count = len(inst.notes)
    is_drum = " (打击乐)" if inst.is_drum else ""
    print(f"  [{i}] {name} - {notes_count} 个音符{is_drum}")

available_classes = ["哈基米", "哈基米only", "曼波only"]
print("\n可选角色:")
for i, cls in enumerate(available_classes):
    print(f"  [{i}] {cls}")

track_classes = []
octave_shift = []
volume_factor = []

default_track_classes = [0]
default_octave_shift = [0]
default_volume_factor = [1.0]

print("\n--- 请配置各轨道参数 (直接回车使用默认值) ---")
for i, inst in enumerate(midi.instruments):
    name = inst.name if inst.name else "未命名轨道"
    is_drum = inst.is_drum

    def_cls_idx = 0
    default_cls = available_classes[default_track_classes[i]]
    default_shift = default_octave_shift[i]
    default_vol = default_volume_factor[i]

    if is_drum:
        print(f"\n轨道 [{i}] {name}: 检测为打击乐，默认静音。")
        track_classes.append(available_classes[0])
        octave_shift.append(0)
        volume_factor.append(0.0)
        continue

    print(f"\n配置轨道 [{i}] {name}:")

    cls_input = input(
        f"  选择角色编号 (0-{len(available_classes)-1}, 默认 {def_cls_idx} [{default_cls}]): "
    )

    if cls_input.strip() == "":
        track_classes.append(default_cls)
    else:
        try:
            idx = int(cls_input)
            track_classes.append(
                available_classes[idx if 0 <= idx < len(available_classes) else def_cls_idx]
            )
        except:
            track_classes.append(default_cls)

    shift_input = input("  八度偏移 (例如 -1, 0, 1, 默认 0): ")
    if shift_input.strip() == "":
        octave_shift.append(default_shift)
    else:
        try:
            octave_shift.append(int(shift_input))
        except:
            octave_shift.append(default_shift)

    vol_input = input("  音量比例 (0.0-1.0, 默认 1.0): ")
    if vol_input.strip() == "":
        volume_factor.append(default_vol)
    else:
        try:
            volume_factor.append(float(vol_input))
        except:
            volume_factor.append(default_vol)

SIMULTANEOUS_THRESHOLD = 0.05

start_time = 0.0
midi_duration = midi.get_end_time()
end_time = midi_duration + 2.0 

print(f"音频起始: {start_time:.2f} 秒, 结束: {end_time:.2f} 秒")

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

def synthesize_midi(preproc, note2word, midi, sr=44100, frame_period=5.0, start_time=0.0, end_time=60, octave_shift=None, volume_factor=None, ap_scale=0.05, sp_scale=1.0):
    n_tracks = len(midi.instruments)
    if octave_shift is None:
        octave_shift = [0] * n_tracks
    if volume_factor is None:
        volume_factor = [1.0] * n_tracks

    total_time = midi.get_end_time() if end_time is None else end_time
    n_samples = int(total_time * sr)
    
    if USE_GPU:
        y_stereo = torch.zeros((n_samples, 2), device=DEVICE, dtype=torch.float32)
    else:
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

            if USE_GPU:
                if isinstance(y_note, np.ndarray):
                    y_note_t = torch.from_numpy(y_note).to(DEVICE, dtype=torch.float32)
                else:
                    y_note_t = y_note.to(DEVICE, dtype=torch.float32)
                y_stereo[start:end, :] += y_note_t * vol
            else:
                y_stereo[start:end, :] += y_note * vol

    if USE_GPU:
        max_amp = torch.max(torch.abs(y_stereo)).item()
        if max_amp > 1.0:
            y_stereo /= max_amp
        y_stereo = y_stereo.cpu().numpy()
    else:
        max_amp = np.max(np.abs(y_stereo))
        if max_amp > 1.0:
            y_stereo /= max_amp

    return y_stereo

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

# ========== 训练模型 ==========
track_models = []

for cls in track_classes:
    model = train(allowed_classes=[cls])
    track_models.append(model)
    draw(model, save_path=out_model_path + f"{cls}.png")
    with open(f"{out_model_path}{cls}.json", "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

# ========== 生成歌词 ==========
track_words, note2word = generate_lyrics(midi, track_models)
events = build_events(note2word)

# ========== 1. 只生成 1 个完整 JSON ==========
out_lyrics_path = os.path.join(output_dir, f"{project_name}_lyrics.json")
with open(out_lyrics_path, "w", encoding="utf-8") as f:
    json.dump(events, f, ensure_ascii=False, indent=2)
print(f"\n歌词事件已保存: {out_lyrics_path} ({len(events)} 个事件)")

# ========== 预处理采样（只需一次） ==========
preproc = {}
for cls in track_classes:
    for sample in alphabet[cls]:
        if sample in preproc:
            continue
        path = sample_path + f"{sample}.wav"
        preproc[sample] = preprocess_sample(path, cons_frame=cons_frames.get(sample, 8))

# ========== 2 & 3. 按歌词每15个分组 → 分割MIDI + 每组单独渲染音频 ==========
LYRICS_PER_GROUP = 15

def render_group_audio(group_events, group_idx, src_notes, preproc_data, sr, out_dir, proj_name):
    """将一组歌词事件分割为独立MIDI并渲染音频（去除所有多余前后空白）"""
    if not group_events:
        print(f"  跳过第 {group_idx} 组（空事件）")
        return
    
    # 按时间排序，计算该组时间范围
    group_events = sorted(group_events, key=lambda e: (e["start"], e["end"]))
    t0 = group_events[0]["start"]                  # 组内最早开始
    t1 = max(evt["end"] for evt in group_events)    # 组内最晚结束
    
    sub_midi = pretty_midi.PrettyMIDI()
    sub_inst = pretty_midi.Instrument(program=0, is_drum=False, name=f"Piano_Part_{group_idx:03d}")
    sub_note2word = [{}]  # 单轨道
    
    # 精确匹配原始音符，并平移到从0开始
    for evt in group_events:
        found = False
        for note in src_notes:
            if (note.pitch == evt["pitch"] and 
                abs(note.start - evt["start"]) < 1e-5 and 
                abs(note.end - evt["end"]) < 1e-5):
                # 【关键】创建平移后的新音符，消除前导空白
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
    
    # 保存子 MIDI（已平移，从0开始，无多余前导时间）
    part_mid_path = os.path.join(out_dir, f"{proj_name}_part_{group_idx:03d}.mid")
    sub_midi.write(part_mid_path)
    
    # 【关键】渲染精确到音符结束，不加 2 秒缓冲，消除尾部空白
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
    
    # 保存紧凑音频
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

# ========== 4. 按顺序拼接所有 part WAV 为完整音频 ==========
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

# 完整轨道字幕（可选）
make_track_srt(
    note2word,
    track_idx=0,
    output_path=os.path.join(output_dir, "track0.srt"),
    reset_gap=1.0,
    max_len=12
)
