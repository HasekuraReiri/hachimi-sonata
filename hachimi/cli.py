"""交互式 CLI：MIDI / MusicXML 选择、模型选择、轨道配置、分析摘要。"""

from __future__ import annotations

import glob
import os
import sys
from typing import Any, Sequence

from hachimi.analysis import AnalysisResult
from hachimi.config import (
    LOCAL_MUSIC_GLOB,
    MUSIC_EXTS,
    MUSIC_GLOB,
    TEST_MUSIC_GLOB,
    VOICE_LABELS,
    VOICE_MODES,
)


def setup_console() -> None:
    """让 Windows 控制台正确显示中文与符号。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


def iter_midi_files() -> list[str]:
    """收集项目内所有 MIDI / MusicXML（assets/midi 与 midi 目录）。

    函数名保留 ``iter_midi_files`` 以兼容旧脚本；返回列表里也包含
    ``.xml`` / ``.musicxml`` / ``.mxl``。
    """
    files: set[str] = set()
    for pattern in (MUSIC_GLOB, LOCAL_MUSIC_GLOB, TEST_MUSIC_GLOB):
        for path in glob.glob(pattern, recursive=True):
            if os.path.isfile(path) and os.path.splitext(path)[1].lower() in MUSIC_EXTS:
                files.add(path)
    return sorted(files)


def select_midi(files: Sequence[str] | None = None, show_limit: int = 60) -> str:
    """交互式选择 MIDI / MusicXML 文件。"""
    files = list(files) if files is not None else iter_midi_files()
    if not files:
        print("未找到 MIDI / 乐谱文件。请把 .mid 或 .xml 放到 assets/midi/ 或 midi/ 目录。")
        raise SystemExit(1)

    print(f"\n找到 {len(files)} 个 MIDI / 乐谱文件：")
    for i, path in enumerate(files[:show_limit]):
        try:
            rel = os.path.relpath(path)
        except ValueError:
            rel = path
        print(f"  [{i:>3}] {rel}")
    if len(files) > show_limit:
        print(f"  … 其余 {len(files) - show_limit} 个未显示（可输入更大编号）")

    while True:
        raw = input(f"\n请输入 MIDI / 乐谱编号 (0-{len(files) - 1}): ").strip()
        try:
            idx = int(raw)
        except ValueError:
            print("输入无效，请输入数字。")
            continue
        if 0 <= idx < len(files):
            return files[idx]
        print("编号超出范围。")


def select_model_kind(default: str = "markov") -> str:
    """交互式选择歌词生成模型。"""
    print("\n可选歌词生成模型：")
    print("  [0] markov       字符马尔科夫链（快、稳定，推荐）")
    print("  [1] transformer  GPT-2 式小模型（更连贯，首次训练较慢）")
    while True:
        choice = input(f"请选择模型编号 (直接回车 = {default}): ").strip()
        if choice == "":
            return default
        if choice in ("0", "markov"):
            return "markov"
        if choice in ("1", "transformer"):
            return "transformer"
        print("输入无效，请输入 0 或 1。")


def _ask(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw or default


def configure_tracks(
    tracks: Sequence[dict[str, Any]],
    *,
    interactive: bool = True,
) -> list[dict[str, Any]]:
    """配置每条轨道的参与状态、语音模式、八度与音量。

    Args:
        tracks: :func:`hachimi.pipeline.resolve_track_options` 得到的轨道列表。
        interactive: ``False`` 时直接使用分析建议（相当于全部回车）。

    Returns:
        更新后的轨道列表。
    """
    result = [dict(t) for t in tracks]
    if not interactive:
        return result

    print("\n可选语音模式：")
    for i, mode in enumerate(VOICE_MODES):
        print(f"  [{i}] {mode:<10} {VOICE_LABELS[mode]}")

    print(
        "\n--- 配置各轨道（直接回车 = 使用方括号中的默认值；"
        "输入 a = 其余轨道全部采用默认） ---"
    )
    accept_rest = False
    for t in result:
        idx = t["index"]
        name = t["name"]
        print(
            f"\n轨道 [{idx}] {name} | {t['instrument']} | {t['role_label']} | "
            f"{t['n_notes']} 音符"
        )
        print(
            f"  建议：{'参与' if t['selected'] else '不参与'} / 语音 {t['voice']} / "
            f"八度 {t['octave']:+d} / 音量 {t['gain']:.2f}"
        )

        if accept_rest:
            continue

        raw = input(
            "  参与生成? (y/n/a=其余全部默认) [{}]: ".format("Y" if t["selected"] else "n")
        ).strip().lower()
        if raw == "a":
            accept_rest = True
            continue
        if raw in ("n", "no", "0", "f", "false"):
            t["selected"] = False
            continue
        if raw in ("y", "yes", "1", "t", "true"):
            t["selected"] = True

        # 语音
        default_mode = t["voice"] if t["voice"] in VOICE_MODES else "auto"
        raw_mode = _ask(
            "  语音模式编号 (0=auto,1=哈基米,2=哈基米only,3=曼波,4=曼波only)",
            default_mode,
        )
        if raw_mode.isdigit():
            modes = list(VOICE_MODES)
            k = int(raw_mode)
            t["voice"] = modes[k] if 0 <= k < len(modes) else default_mode
        elif raw_mode in VOICE_MODES:
            t["voice"] = raw_mode
        else:
            t["voice"] = default_mode

        # 八度
        raw_oct = _ask("  八度偏移 (-3 ~ 3)", str(t["octave"]))
        try:
            t["octave"] = max(-3, min(3, int(raw_oct)))
        except ValueError:
            pass

        # 音量
        raw_gain = _ask("  轨道音量 (0 ~ 2)", f"{t['gain']:.2f}")
        try:
            t["gain"] = max(0.0, min(2.0, float(raw_gain)))
        except ValueError:
            pass

    return result


def print_analysis_summary(analysis: AnalysisResult, max_chords: int = 16) -> None:
    """在终端打印分析结果摘要。"""
    print("\n" + "=" * 72)
    source = {"musicxml": "MusicXML", "mxl": "MXL"}.get(analysis.source_format, "MIDI")
    print(
        f"曲目：{analysis.title}    时长：{analysis.duration:.1f}s    "
        f"速度：{analysis.tempo or 0:.0f} BPM    拍号：{analysis.time_signature}    来源：{source}"
    )
    if len(analysis.tempo_changes) > 1:
        changes = " → ".join(
            f"{float(item.get('bpm', 0)):g}" for item in analysis.tempo_changes
        )
        print(f"速度地图：{len(analysis.tempo_changes)} 处（{changes} BPM）")
    if analysis.key:
        print(
            f"调性：{analysis.key.get('label_cn') or analysis.key.get('label')}  "
            f"（置信度 {analysis.key.get('confidence', 0):.2f}，Krumhansl 轮廓）"
        )
    print("-" * 72)
    print(f"{'#':>3}  {'轨道名':<24} {'乐器':<22} {'角色':<6} {'音符':>6} {'复调':>5}")
    for t in analysis.tracks:
        print(
            f"{t.index:>3}  {t.name[:24]:<24} {t.instrument[:22]:<22} "
            f"{t.role_label:<6} {t.n_notes:>6} {t.polyphony:>5.2f}"
        )
    print("-" * 72)
    sections = "  ".join(
        f"{s.label}[{s.start:.0f}-{s.end:.0f}s"
        + (f" {s.key}" if s.key else "")
        + (f" {'/'.join(s.progression[:4])}" if s.progression else "")
        + "]"
        for s in analysis.sections[:12]
    )
    print(f"结构：{len(analysis.sections)} 段 / {len(analysis.phrases)} 个乐句   {sections}")
    labels = [c.label for c in analysis.chords[:max_chords]]
    print(f"和弦：{len(analysis.chords)} 个窗口   前 {len(labels)} 个：{' '.join(labels)}")
    print(f"分解和弦：{sum(1 for c in analysis.chords if c.is_decomposed)} 个")
    print(f"动机：{len(analysis.motifs)} 个")
    for m in analysis.motifs[:6]:
        print(
            f"  {m.id}: {m.length} 音 × {len(m.occurrences)} 次  "
            f"音程 {m.intervals}  首次 @ {m.occurrences[0]['start']:.1f}s"
        )
    print("=" * 72 + "\n")


__all__ = [
    "configure_tracks",
    "iter_midi_files",
    "print_analysis_summary",
    "select_midi",
    "select_model_kind",
    "setup_console",
]
