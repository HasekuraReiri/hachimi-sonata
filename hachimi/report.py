"""乐器输出清单：记录哪些乐器导出了音频、对应文件名、以及未输出原因。

生成的 Markdown / CSV / JSON 会放进输出目录，并随 ZIP 一起打包。
"""

from __future__ import annotations

import csv
import json
import os
import time
from typing import Any, Mapping, Sequence


def _safe_cell(value: Any) -> str:
    text = str(value if value is not None else "").replace("|", "\\|")
    return text.replace("\r", " ").replace("\n", " ")


def _reason_for_unselected(track: Any, option: Mapping[str, Any] | None) -> str:
    option = option or {}
    suggested = option.get("suggested") or {}
    reason = str(suggested.get("reason") or option.get("reason") or "").strip()
    if getattr(track, "is_drum", False):
        return reason or "打击乐没有音高，未参与哈基米合成"
    if getattr(track, "role", "") == "bass":
        return reason or "低音声部默认不勾选，可按需手动勾选"
    if getattr(track, "n_notes", 0) == 0:
        return "空轨道"
    return reason or "未勾选参与生成"


def _reason_for_no_audio(
    track: Any,
    option: Mapping[str, Any] | None,
    *,
    selected: bool,
    audible: bool,
    write_stems: bool,
) -> str:
    if not selected:
        return _reason_for_unselected(track, option)
    if not audible:
        return "音量为 0（或轨道被静音），未输出"
    if not write_stems:
        return "已混入主混音，但未开启「每轨单独输出 MP3」"
    return "所选时间段内没有可渲染音符"


def build_instrument_report(
    *,
    project: str,
    mix_filename: str,
    analysis: Any,
    track_options: Mapping[int, Mapping[str, Any]],
    selected: Sequence[int],
    audible: Sequence[int],
    stems: Mapping[str, str],
    stem_entries: Sequence[Mapping[str, Any]],
    write_stems: bool,
    output_dir: str,
    mix_report: Mapping[str, Any] | None = None,
    video_report: Mapping[str, Any] | None = None,
    premix: Mapping[str, str] | None = None,
    premix_entries: Sequence[Mapping[str, Any]] | None = None,
    premix_zip: str = "",
    bad_notes: Mapping[str, Any] | None = None,
    bad_note_files: Mapping[str, str] | None = None,
    supplement_midi: Mapping[str, Any] | None = None,
    percussion_midi: Mapping[str, Any] | None = None,
    params_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """构造输出 / 未输出乐器清单（纯数据，便于测试）。"""
    selected_set = {int(i) for i in selected}
    audible_set = {int(i) for i in audible}
    rendered_stems = {int(i) for i in stems}
    premix = premix or {}
    premix_entries = premix_entries or []
    bad_notes = bad_notes or {}
    bad_note_files = bad_note_files or {}
    supplement_midi = supplement_midi or {}
    percussion_midi = percussion_midi or {}
    params_paths = params_paths or {}
    premix_by_index = {
        int(item.get("index", -1)): item
        for item in premix_entries
        if item.get("index") is not None
    }
    bad_by_index = {
        int(item.get("index", -1)): item
        for item in (bad_notes.get("tracks") or [])
        if item.get("index") is not None
    }
    entry_by_index = {
        int(item.get("index", -1)): item for item in stem_entries if item.get("index") is not None
    }
    placement_by_index = {
        int(item.get("index", -1)): item
        for item in (mix_report or {}).get("tracks", [])
        if item.get("index") is not None
    }
    supplement_by_index = {
        int(item.get("index", -1)): item
        for item in (supplement_midi.get("tracks") or [])
        if item.get("index") is not None
    }
    percussion_by_index = {
        int(item.get("index", -1)): item
        for item in (percussion_midi.get("tracks") or [])
        if item.get("index") is not None
    }

    outputs: list[dict[str, Any]] = []
    mix_only: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for track in analysis.tracks:
        idx = int(track.index)
        option = track_options.get(idx) or {}
        placement = placement_by_index.get(idx, {})
        base = {
            "index": idx,
            "name": track.name,
            "instrument": track.instrument,
            "role": getattr(track, "role_label", "") or getattr(track, "role", ""),
            "voice": "自动混合" if option.get("voice") == "auto" else option.get("voice", ""),
            "octave": option.get("octave", 0),
            "gain": option.get("gain", 1.0),
            "n_notes": track.n_notes,
            "midi_program": getattr(track, "program", ""),
            "pan": placement.get("pan", 0.0),
            "depth": placement.get("depth", 0.0),
            "auto_gain_db": placement.get("auto_gain_db", 0.0),
            "in_supplement_midi": idx in supplement_by_index,
            "supplement_midi_file": (
                f"bad_notes/{os.path.basename(str(supplement_midi.get('path') or ''))}"
                if supplement_midi.get("path")
                else ""
            ),
            "in_percussion_midi": idx in percussion_by_index,
            "percussion_midi_file": (
                os.path.basename(str(percussion_midi.get("path") or ""))
                if percussion_midi.get("path")
                else ""
            ),
        }
        premix_file = os.path.basename(str(premix.get(str(idx)) or premix.get(idx) or ""))
        bad_info = bad_by_index.get(idx, {})
        bad_midi = os.path.basename(str(bad_note_files.get(str(idx)) or ""))
        quality = {
            "bad_notes": int(bad_info.get("bad_notes", 0) or 0),
            "too_high": int(bad_info.get("too_high", 0) or 0),
            "too_low": int(bad_info.get("too_low", 0) or 0),
            "too_quiet": int(bad_info.get("too_quiet", 0) or 0),
            "expressive_soft": int(bad_info.get("expressive_soft", 0) or 0),
            "bad_midi_file": f"bad_notes/{bad_midi}" if bad_midi else "",
        }
        if idx in rendered_stems:
            entry = entry_by_index.get(idx, {})
            stem_file = os.path.basename(str(stems.get(str(idx)) or stems.get(idx) or ""))
            outputs.append(
                {
                    **base,
                    **quality,
                    "file": f"stems/{stem_file}" if stem_file else "",
                    "file_name": stem_file,
                    "premix_file": f"stems_premix/{premix_file}" if premix_file else "",
                    "premix_file_name": premix_file,
                    "stem_entry": entry,
                    "reason": "已导出独立分轨",
                }
            )
        elif idx in audible_set:
            reason = _reason_for_no_audio(
                track,
                option,
                selected=idx in selected_set,
                audible=True,
                write_stems=write_stems,
            )
            if write_stems:
                missing.append({**base, **quality, "reason": reason})
            else:
                mix_only.append({**base, **quality, "reason": reason})
        else:
            reason = _reason_for_no_audio(
                track,
                option,
                selected=idx in selected_set,
                audible=idx in audible_set,
                write_stems=write_stems,
            )
            missing.append({**base, **quality, "reason": reason})

    all_tempo = list(getattr(analysis, "tempo_changes", []) or [])
    try:
        tempo_rows, tempo_compressed, tempo_window = analysis.tempo_display(64)
    except Exception:  # pragma: no cover - 兼容旧的分析对象
        tempo_rows, tempo_compressed, tempo_window = all_tempo[:64], False, 0.0
    bpms = [float(item.get("bpm", 0) or 0) for item in all_tempo if float(item.get("bpm", 0) or 0) > 0]
    return {
        "project": project,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mix_file": mix_filename,
        "output_dir": output_dir,
        "outputs": outputs,
        "mix_only": mix_only,
        "missing": missing,
        "mix": dict(mix_report or {}),
        "video": dict(video_report or {}),
        "premix": {
            "n_tracks": len(premix),
            "zip": os.path.basename(premix_zip) if premix_zip else "",
            "entries": list(premix_entries),
        },
        "bad_notes": dict(bad_notes),
        "bad_note_files": {str(k): os.path.basename(v) for k, v in bad_note_files.items()},
        "supplement_midi": dict(supplement_midi),
        "percussion_midi": dict(percussion_midi),
        "params": {str(k): os.path.basename(v) for k, v in params_paths.items()},
        "score": {
            "source_format": getattr(analysis, "source_format", "midi"),
            "n_tempo_changes": len(all_tempo),
            "tempo_changes": tempo_rows,
            "tempo_changes_compressed": bool(tempo_compressed),
            "tempo_window": round(float(tempo_window), 2),
            "tempo_bpm_min": round(min(bpms), 2) if bpms else None,
            "tempo_bpm_max": round(max(bpms), 2) if bpms else None,
            "n_dynamics": int((getattr(analysis, "score_meta", {}) or {}).get("n_dynamics", 0) or 0),
            "n_articulations": int((getattr(analysis, "score_meta", {}) or {}).get("n_articulations", 0) or 0),
            "n_ties": int((getattr(analysis, "score_meta", {}) or {}).get("n_ties", 0) or 0),
            "n_grace": int((getattr(analysis, "score_meta", {}) or {}).get("n_grace", 0) or 0),
        },
        "summary": {
            "n_tracks": len(analysis.tracks),
            "n_outputs": len(outputs),
            "n_mix_only": len(mix_only),
            "n_missing": len(missing),
            "n_bad_notes": int(bad_notes.get("bad_notes", 0) or 0),
            "n_premix": len(premix),
            "n_supplement_tracks": int(
                supplement_midi.get("n_tracks") or len(supplement_by_index)
            ),
            "n_percussion_tracks": int(
                percussion_midi.get("n_tracks") or len(percussion_by_index)
            ),
            "n_percussion_notes": int(percussion_midi.get("n_notes", 0) or 0),
        },
    }


def write_instrument_report(
    report: Mapping[str, Any],
    out_dir: str,
    project: str,
) -> dict[str, str]:
    """写出 Markdown / CSV / JSON 三种清单，返回路径。"""
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, f"{project}_instrument_report.md")
    csv_path = os.path.join(out_dir, f"{project}_instrument_report.csv")
    json_path = os.path.join(out_dir, f"{project}_instrument_report.json")

    outputs = list(report.get("outputs") or [])
    mix_only = list(report.get("mix_only") or [])
    missing = list(report.get("missing") or [])
    mix_file = str(report.get("mix_file", ""))

    lines = [
        f"# 输出乐器清单 · {project}",
        "",
        f"- 生成时间：{report.get('generated_at', '')}",
        f"- 主混音文件：`{mix_file}`",
        f"- 输出目录：`{report.get('output_dir', '')}`",
        "",
        "## 一、输出乐器名单（含对应音频文件名）",
        "",
    ]
    if outputs:
        lines += [
            "| 轨道 | 轨道名 | 乐器 | 角色 | 语音 | 声像 / 深度 | 混音后分轨（ZIP 内路径） | 混音前干声（FLAC） | 不良音符 MIDI | 总音符数 | 音量 |",
            "| ---: | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |",
        ]
        for item in outputs:
            pan = float(item.get("pan", 0.0) or 0.0)
            depth = float(item.get("depth", 0.0) or 0.0)
            pan_text = "中" if abs(pan) < 0.05 else f"{'左' if pan < 0 else '右'}{abs(pan):.2f}"
            lines.append(
                "| {index} | {name} | {instrument} | {role} | {voice} | {pan} / {depth:.2f} | `{file}` | `{premix}` | `{bad_midi}` | {n_notes} | {gain} |".format(
                    index=item.get("index", ""),
                    name=_safe_cell(item.get("name", "")),
                    instrument=_safe_cell(item.get("instrument", "")),
                    role=_safe_cell(item.get("role", "")),
                    voice=_safe_cell(item.get("voice", "")),
                    pan=pan_text,
                    depth=depth,
                    file=_safe_cell(item.get("file", "")),
                    premix=_safe_cell(item.get("premix_file", "")) or "—",
                    bad_midi=_safe_cell(item.get("bad_midi_file", "")) or "—",
                    n_notes=item.get("n_notes", 0),
                    gain=item.get("gain", ""),
                )
            )
        lines += [
            "",
            "### 输出乐器 → 音频文件名对应表",
            "",
            "```text",
        ]
        for item in outputs:
            lines.append(
                f"{item.get('instrument', '')}（轨道 {item.get('index', '')} / {item.get('name', '')}）"
                f" → {item.get('file', '')}"
                + (f"｜混音前 {item.get('premix_file', '')}" if item.get("premix_file") else "")
                + (f"｜不良音 MIDI {item.get('bad_midi_file', '')}" if item.get("bad_midi_file") else "")
            )
        lines += ["```", ""]
    else:
        lines += ["本次没有导出任何独立分轨。", ""]

    if mix_only:
        lines += [
            "## 二、仅混入主混音、未单独导出的乐器",
            "",
            "| 轨道 | 轨道名 | 乐器 | 角色 | 原因 |",
            "| ---: | --- | --- | --- | --- |",
        ]
        for item in mix_only:
            lines.append(
                "| {index} | {name} | {instrument} | {role} | {reason} |".format(
                    index=item.get("index", ""),
                    name=_safe_cell(item.get("name", "")),
                    instrument=_safe_cell(item.get("instrument", "")),
                    role=_safe_cell(item.get("role", "")),
                    reason=_safe_cell(item.get("reason", "")),
                )
            )
        lines.append("")
    else:
        lines += [
            "## 二、仅混入主混音、未单独导出的乐器",
            "",
            "本次所有参与混音的乐器都导出了独立分轨。",
            "",
        ]

    lines += [
        "## 三、未输出名单（默认静音 / 未勾选 / 无音符）",
        "",
    ]
    if missing:
        lines += [
            "| 轨道 | 轨道名 | 乐器 | 角色 | 原因 |",
            "| ---: | --- | --- | --- | --- |",
        ]
        for item in missing:
            lines.append(
                "| {index} | {name} | {instrument} | {role} | {reason} |".format(
                    index=item.get("index", ""),
                    name=_safe_cell(item.get("name", "")),
                    instrument=_safe_cell(item.get("instrument", "")),
                    role=_safe_cell(item.get("role", "")),
                    reason=_safe_cell(item.get("reason", "")),
                )
            )
    else:
        lines.append("所有轨道均已输出。")
    lines += [
        "",
        "## 四、统计",
        "",
        f"- 轨道总数：{report.get('summary', {}).get('n_tracks', 0)}",
        f"- 独立分轨：{report.get('summary', {}).get('n_outputs', 0)}",
        f"- 仅混音：{report.get('summary', {}).get('n_mix_only', 0)}",
        f"- 未输出：{report.get('summary', {}).get('n_missing', 0)}",
        f"- 补充 MIDI 轨道：{report.get('summary', {}).get('n_supplement_tracks', 0)}",
        f"- 打击乐 MIDI：{report.get('summary', {}).get('n_percussion_tracks', 0)} 轨 / "
        f"{report.get('summary', {}).get('n_percussion_notes', 0)} 音",
        "",
    ]

    score = report.get("score") or {}
    tempo_changes = list(score.get("tempo_changes") or [])
    n_tempo = int(score.get("n_tempo_changes", len(tempo_changes)) or 0)
    if score.get("source_format") in ("musicxml", "mxl"):
        tempo_text = " → ".join(
            f"{float(item.get('bpm', 0)):g}" for item in tempo_changes[:16]
        ) or "—"
        lines += [
            "## 五、乐谱标记",
            "",
            f"- 来源格式：{'MXL（压缩 MusicXML）' if score.get('source_format') == 'mxl' else 'MusicXML'}",
            f"- 速度标记：{n_tempo} 处（{tempo_text} BPM）",
            f"- 力度标记：{score.get('n_dynamics', 0)} 处",
            f"- 奏法标记：{score.get('n_articulations', 0)} 个音符",
            f"- 连音线：{score.get('n_ties', 0)} 个标记（已合并为持续音）",
            f"- 装饰音：{score.get('n_grace', 0)} 个",
        ]
        if score.get("tempo_changes_compressed"):
            lines.append(
                f"- 速度显示：密集自动化，按约 {score.get('tempo_window', 0):g} 秒窗口取中位数，"
                f"报告中保留 {len(tempo_changes)} 个代表点；播放仍使用全部 {n_tempo} 个事件。"
            )
        lines.append("")
    else:
        lines += [
            "## 五、乐谱标记",
            "",
            "- 来源格式：MIDI（力度来自 velocity）",
            f"- 速度标记：{n_tempo} 处",
            "",
        ]

    mix = report.get("mix") or {}
    lines += [
        "## 六、自动混音摆位",
        "",
        f"- 自动混音：{'开启' if mix.get('enabled', True) else '关闭'}",
        f"- 母带响度：{mix.get('loudness_db', '—')} dB（目标 {mix.get('target_db', '—')} dB）",
        f"- 峰值：{mix.get('peak_db', '—')} dB / 峰值因数：{mix.get('crest_db', '—')} dB",
        (
            f"- 总线压缩：{mix.get('compression_ratio', '—')}:1 / "
            f"自动补偿 +{mix.get('makeup_db', 0)} dB / "
            f"平均衰减 {mix.get('gain_reduction_db', 0)} dB / "
            f"弱段提升 {mix.get('upward_gain_db', 0)} dB"
            if mix.get("enabled", True)
            else "- 总线压缩：关闭（传统后处理）"
        ),
        f"- 立体声相关度：{mix.get('correlation', '—')}",
        f"- 渐强 / 渐弱：{mix.get('n_dynamic_notes', 0)} 个音符",
        f"- 高音补偿：{mix.get('n_audibility_notes', 0)} 个音符",
        f"- Tremolo：{mix.get('n_tremolo_notes', 0)} 个音符",
        "",
    ]
    placements = list(mix.get("tracks") or [])
    if placements:
        lines += [
            "| 轨道 | 乐器 | 声像 | 深度 | 自动电平 | 早期反射 |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
        for item in placements:
            pan = float(item.get("pan", 0.0) or 0.0)
            pan_text = "中" if abs(pan) < 0.05 else f"{'左' if pan < 0 else '右'} {abs(pan):.2f}"
            lines.append(
                "| {index} | {instrument} | {pan} | {depth:.2f} | {gain:+.1f} dB | {ms:.1f} ms |".format(
                    index=item.get("index", ""),
                    instrument=_safe_cell(item.get("instrument", "")),
                    pan=pan_text,
                    depth=float(item.get("depth", 0.0) or 0.0),
                    gain=float(item.get("auto_gain_db", 0.0) or 0.0),
                    ms=float(item.get("early_reflection_ms", 0.0) or 0.0),
                )
            )
        lines.append("")
    else:
        lines += ["本次没有可用的自动混音摆位信息。", ""]

    video = report.get("video") or {}
    if video.get("available"):
        coverage = video.get("coverage") or {}
        lines += [
            "## 七、绿幕字幕视频",
            "",
            f"- 规格：单条视频，宽 {video.get('width', 1920)} × 高 {video.get('height', '—')}，"
            f"全部字幕每次 ≤ {max(1, int(video.get('max_lines') or 1))} 行；"
            f"{video.get('fps', 30)} fps，纯绿 `#00FF00`，"
            f"{video.get('text_color', '#FFFFFF')} {video.get('font', 'SimSun')} "
            f"{video.get('font_size', 8)} 号（{video.get('font_px', 80)} px），"
            f"{video.get('stroke_width', 2)} px 黑边，居中",
            f"- 全部轨道合并为一条连续时间线；歌词槽覆盖 "
            f"{coverage.get('covered_slots', '—')}/{coverage.get('input_slots', '—')}，"
            f"缺失 {coverage.get('missing_slots', '—')} 个",
            f"- 视频不带音轨，供 BB 素材 / Chroma Key 使用；生成耗时 "
            f"{video.get('elapsed_seconds', '—')} 秒",
            "",
            "| 字幕 | 轨道 | 乐器 | 句数 | 行数 | 文件 |",
            "| --- | ---: | --- | ---: | ---: | --- |",
        ]
        for key, label in (
            ("all", "全部"),
            ("melody", "主要（旧）"),
            ("accompaniment", "辅助（旧）"),
        ):
            item = (video.get("tracks") or {}).get(key) or {}
            if not item or not item.get("file_relative"):
                continue
            lines.append(
                "| {label} | {track} | {instrument} | {lines} | {rows} | `{file}` |".format(
                    label=label,
                    track=_safe_cell(item.get("track_label") or item.get("index", "—")),
                    instrument=_safe_cell(item.get("instrument", "")),
                    lines=item.get("lines", 0),
                    rows=item.get("max_lines", "—"),
                    file=_safe_cell(item.get("file_relative", "")),
                )
            )
        lines.append("")

    quality = report.get("bad_notes") or {}
    premix = report.get("premix") or {}
    params = report.get("params") or {}
    quality_no = 8 if video.get("available") else 7
    params_no = quality_no + 1
    lines += [
        f"## {'八' if quality_no == 8 else '七'}、混音前干声与不良音符",
        "",
        f"- 混音后分轨：{report.get('summary', {}).get('n_outputs', 0)} 条（`stems/`，已做声像 / 深度 / 自动电平）",
        f"- 混音前干声：{premix.get('n_tracks', 0)} 条 FLAC（`stems_premix/`），统一打包为 "
        f"`{premix.get('zip', '—')}`",
        f"- 不良音符：{quality.get('bad_notes', 0)} / {quality.get('total_notes', 0)} 个；"
        f"过高 {quality.get('too_high', 0)}、过低 {quality.get('too_low', 0)}、"
        f"音量过低 {quality.get('too_quiet', 0)}",
        "- 判定标准：按轨道八度移调后的实际音高超出 C2~C6，或偏离本轨中位数两个八度；"
        "音量明显低于本轨中位电平。MusicXML 的 p / pp / mp / 渐弱记号为作曲意图，"
        "默认不判为不良，仅对几乎听不见的极端弱音记录。",
        "",
        "| 轨道 | 乐器 | 总音符 | 不良 | 过高 | 过低 | 音量过低 | 谱面弱力度（仅记录） | 仅含不良音的 MIDI |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    quality_tracks = list(quality.get("tracks") or [])
    if quality_tracks:
        for item in quality_tracks:
            midi_name = os.path.basename(str((report.get("bad_note_files") or {}).get(str(item.get("index")), "")))
            lines.append(
                "| {index} | {instrument} | {total} | {bad} | {high} | {low} | {quiet} | {soft} | {midi} |".format(
                    index=item.get("index", ""),
                    instrument=_safe_cell(item.get("instrument", "")),
                    total=item.get("total_notes", 0),
                    bad=item.get("bad_notes", 0),
                    high=item.get("too_high", 0),
                    low=item.get("too_low", 0),
                    quiet=item.get("too_quiet", 0),
                    soft=item.get("expressive_soft", 0),
                    midi=f"`bad_notes/{_safe_cell(midi_name)}`" if midi_name else "—",
                )
            )
    else:
        lines.append("| — | 无可用轨道 | 0 | 0 | 0 | 0 | 0 | 0 | — |")

    supplement = report.get("supplement_midi") or {}
    supplement_tracks = list(supplement.get("tracks") or [])
    if supplement.get("path"):
        supplement_name = os.path.basename(str(supplement.get("path")))
        lines += [
            "",
            "### 补充 MIDI（不良音轨 → 空白轨 → 全谱）",
            "",
            f"- 文件：`bad_notes/{_safe_cell(supplement_name)}`",
            f"- 顺序：{supplement.get('n_bad_tracks', 0)} 条不良音轨（`*_NG`）→ "
            f"{supplement.get('n_spacer_tracks', 1)} 条空白分隔轨 → "
            f"{supplement.get('n_score_tracks', 0)} 条全谱轨；"
            f"共 {supplement.get('n_tracks', 0)} 轨 / {supplement.get('n_notes', 0)} 个音符。",
            f"- 不良音轨只含问题音 {supplement.get('bad_notes', 0)} 个；"
            f"全谱轨保留完整乐谱 {supplement.get('score_notes', 0)} 个音符；"
            f"其中未输出轨道的 {supplement.get('missing_notes', 0)} 个音符也完整保留。",
            "- MIDI 轨名全部使用英文，且与音频文件名不同：不良音轨为 `<name>_NG`，"
            "全谱轨为 `<name>`（如 `violin1_NG` / `violin1`）。",
            "",
            "| 段落 | 原轨 | MIDI 轨名 | 乐器 | 写入音符 | 其中不良 | 说明 |",
            "| --- | ---: | --- | --- | ---: | ---: | --- |",
        ]
        for item in supplement_tracks:
            section = str(item.get("section", ""))
            section_label = {
                "bad": "不良音",
                "spacer": "空白",
                "full": "全谱",
            }.get(section, section)
            lines.append(
                "| {section} | {index} | {track} | {instrument} | {notes} | {bad} | {reason} |".format(
                    section=section_label,
                    index=item.get("index", "—") if item.get("index") is not None else "—",
                    track=f"`{_safe_cell(item.get('midi_track_name'))}`" if item.get("midi_track_name") else "（空白）",
                    instrument=_safe_cell(item.get("instrument", "")),
                    notes=item.get("included_notes", 0),
                    bad=item.get("bad_notes", 0),
                    reason=_safe_cell(item.get("reason", "")),
                )
            )
        skipped = list(supplement.get("skipped") or [])
        if skipped:
            skipped_text = "、".join(
                f"轨道 {item.get('index')} {item.get('name', '')}".strip()
                for item in skipped
            )
            lines += ["", f"- 空轨道未写入：{skipped_text}（没有音符）。"]

    percussion = report.get("percussion_midi") or {}
    percussion_tracks = list(percussion.get("tracks") or [])
    if percussion.get("path"):
        percussion_name = os.path.basename(str(percussion.get("path")))
        lines += [
            "",
            "### 打击乐 MIDI（不参与音频合成）",
            "",
            f"- 文件：`{_safe_cell(percussion_name)}`（工程根目录）",
            f"- 共 {percussion.get('n_tracks', 0)} 轨 / "
            f"{percussion.get('n_notes', 0)} 个音符。",
            "- 打击乐没有可唱元音，默认不进入哈基米音频合成；轨名统一带 "
            "`_NOAUDIO`，可直接导入 DAW / 鼓采样器。",
            "- 音高来自 MusicXML `<midi-unpitched>`（1-based 转 0-based GM 鼓音高）；"
            "MIDI 输入则保留原音符。",
            "",
            "| 轨道 | 名称 | 乐器 | MIDI 轨名 | 音符数 |",
            "| ---: | --- | --- | --- | ---: |",
        ]
        for item in percussion_tracks:
            lines.append(
                "| {index} | {name} | {instrument} | `{track}` | {notes} |".format(
                    index=item.get("index", ""),
                    name=_safe_cell(item.get("name", "")),
                    instrument=_safe_cell(item.get("instrument", "")),
                    track=_safe_cell(item.get("midi_track_name", "")),
                    notes=item.get("n_notes", 0),
                )
            )
        percussion_skipped = list(percussion.get("skipped") or [])
        if percussion_skipped:
            lines += [
                "",
                "- 空打击乐轨未写入："
                + "、".join(
                    f"轨道 {item.get('index')} {item.get('name', '')}".strip()
                    for item in percussion_skipped
                ),
            ]
    lines += [
        "",
        f"## {'九' if params_no == 9 else '八'}、工程参数包",
        "",
        f"- 完整参数 JSON：`{params.get('json', '—')}`",
        f"- 可复用参数 ZIP：`{params.get('zip', '—')}`",
        "- 在网页左侧「工程参数复用」中导入任一文件，即可复制本工程的全局设置、"
        "轨道语音 / 八度 / 音量、段落编曲计划与字幕样式。",
        "",
    ]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    csv_fields = [
        "类别",
        "轨道号",
        "轨道名",
        "乐器",
        "角色",
        "语音",
        "八度",
        "音量",
        "声像",
        "深度",
        "自动电平",
        "音符数",
        "对应音频文件",
        "混音前干声",
        "不良音符数",
        "不良 MIDI",
        "补充 MIDI",
        "打击乐 MIDI",
        "主混音文件",
        "参数包",
        "原因",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for category, items in (("输出", outputs), ("仅混音", mix_only), ("未输出", missing)):
            for item in items:
                writer.writerow(
                    {
                        "类别": category,
                        "轨道号": item.get("index", ""),
                        "轨道名": item.get("name", ""),
                        "乐器": item.get("instrument", ""),
                        "角色": item.get("role", ""),
                        "语音": item.get("voice", ""),
                        "八度": item.get("octave", ""),
                        "音量": item.get("gain", ""),
                        "声像": item.get("pan", ""),
                        "深度": item.get("depth", ""),
                        "自动电平": item.get("auto_gain_db", ""),
                        "音符数": item.get("n_notes", 0),
                        "对应音频文件": item.get("file", ""),
                        "混音前干声": item.get("premix_file", ""),
                        "不良音符数": item.get("bad_notes", 0),
                        "不良 MIDI": item.get("bad_midi_file", ""),
                        "补充 MIDI": (
                            item.get("supplement_midi_file", "")
                            if item.get("in_supplement_midi")
                            else ""
                        ),
                        "打击乐 MIDI": (
                            item.get("percussion_midi_file", "")
                            if item.get("in_percussion_midi")
                            else ""
                        ),
                        "主混音文件": mix_file,
                        "参数包": params.get("zip", ""),
                        "原因": item.get("reason", ""),
                    }
                )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return {"markdown": md_path, "csv": csv_path, "json": json_path}


__all__ = ["build_instrument_report", "write_instrument_report"]
