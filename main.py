#!/usr/bin/env python
"""Hachimi Sonata · 哈基米音律工坊 —— 命令行入口。

常用命令
--------
::

    python main.py                     # 交互式：选择 MIDI / 乐谱 / 模型 / 轨道后生成
    python main.py song.mid            # 直接生成（自动选择轨道）
    python main.py song.musicxml       # 直接读取 MusicXML / MXL
    python main.py song.mid --yes      # 全部使用默认参数，不询问
    python main.py analyze song.xml    # 只分析：轨道 / 速度 / 和弦 / 结构 / 动机
    python main.py web                 # 启动网页界面并自动打开浏览器
    python main.py serve --port 9000   # 只启动服务
    python main.py launch              # 启动图形化 Python 启动器

渲染结果输出到 ``output/<曲名>/``：MP3、歌词 JSON、SRT/LRC/JSON/Markdown 字幕、
可直接双击打开的实时字幕 HTML、分析 JSON 等。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from typing import Any

# 允许从任意工作目录运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hachimi.analysis import analyze
from hachimi.cli import (
    configure_tracks,
    iter_midi_files,
    print_analysis_summary,
    select_midi,
    select_model_kind,
    setup_console,
)
from hachimi.config import (
    APP_NAME,
    APP_NAME_CN,
    APP_SLUG,
    DEFAULT_VOICE,
    MODEL_KINDS,
    VELOCITY_GAMMA,
    VOICE_MODES,
    RenderOptions,
    ensure_dirs,
)
from hachimi.pipeline import render_midi, resolve_track_options

logger = logging.getLogger("hachimi")


# ---------------------------------------------------------------------------
# 控制台进度
# ---------------------------------------------------------------------------


class ConsoleProgress:
    def __init__(self) -> None:
        self.last_pct = -10.0
        self.last_stage = ""
        self.start = time.time()

    def __call__(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "stage":
            pct = float(event.get("pct", 0))
            stage = str(event.get("stage", ""))
            if pct - self.last_pct >= 3 or stage != self.last_stage or pct >= 100:
                self.last_pct = pct
                self.last_stage = stage
                msg = str(event.get("message", ""))
                print(f"\r[{pct:5.1f}%] {msg[:90]:<90}", end="" if pct < 100 else "\n", flush=True)
        elif kind == "log":
            print(f"\n  · {event.get('message', '')}", flush=True)
        elif kind == "subtitle":
            lines = event.get("lines") or []
            print(f"\n  · 已生成 {len(lines)} 行实时字幕（可下载 SRT/LRC/HTML）", flush=True)


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=APP_SLUG,
        description=f"{APP_NAME} · {APP_NAME_CN}：基于 MIDI / MusicXML 自动生成哈基米音乐（歌词 + 人声 + 实时字幕）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("target", nargs="?", help="MIDI / MusicXML 文件路径，或子命令 web / serve / launch / analyze")
    parser.add_argument("midi_arg", nargs="?", help="analyze 子命令使用的 MIDI / 乐谱路径")
    parser.add_argument("--midi", "-m", help="MIDI / MusicXML 文件路径")
    parser.add_argument("--out", "-o", help="输出目录（默认 output/<曲名>）")
    parser.add_argument("--project", help="项目名（默认 MIDI 文件名）")
    parser.add_argument("--model", choices=MODEL_KINDS, help="歌词模型（默认 markov）")
    parser.add_argument("--voice", choices=list(VOICE_MODES), help="全局语音模式（默认 auto 自动混合）")
    parser.add_argument("--sample-set", type=int, choices=[1, 2], default=1, help="哈基米音源：1=音源/哈基米，2=音源/哈基米第二版")
    parser.add_argument(
        "--tracks",
        help="参与生成的轨道，例如 0,1,2 或 all（默认使用分析建议）",
    )
    parser.add_argument("--seed", type=int, default=1337, help="随机种子（默认 1337）")
    parser.add_argument("--syntax-weight", type=float, default=0.3, help="标准句法权重 0~1（默认 0.3）")
    parser.add_argument("--gamma", type=float, default=VELOCITY_GAMMA, help="力度曲线指数（默认 1.6）")
    parser.add_argument("--workers", type=int, default=0, help="渲染进程数，0=自动")
    parser.add_argument("--audio-format", choices=["mp3", "wav"], default="mp3", help="音频输出格式（默认 mp3）")
    parser.add_argument("--gpu", choices=["auto", "on", "off"], default="auto", help="GPU 加速（默认 auto）")
    parser.add_argument("--gpu-memory-mb", type=int, default=2048, help="GPU 显存预算 MB（默认 2048，8G 显卡只用 2G）")
    parser.add_argument("--max-seconds", type=float, help="只渲染前 N 秒（快速试听）")
    parser.add_argument("--tail-seconds", type=float, help="混响尾巴秒数（默认 3；试听可设 0）")
    parser.add_argument("--no-decomposed", action="store_true", help="关闭分解和弦合并（不推荐）")
    parser.add_argument("--no-auto-mix", action="store_true", help="关闭自动混音（真实摆位 / 相位 / 自动电平）")
    parser.add_argument("--no-stems", action="store_true", help="不输出混音后的每轨 MP3")
    parser.add_argument("--no-premix", action="store_true", help="不输出混音前 FLAC 干声包")
    parser.add_argument(
        "--no-bad-notes",
        action="store_true",
        help="不导出不良音符 / 未输出轨道合并 MIDI",
    )
    parser.add_argument("--no-zip", action="store_true", help="不打包 ZIP")
    parser.add_argument("--no-videos", action="store_true", help="不生成绿幕字幕视频（默认单条包含全部字幕、每次 1 行）")
    parser.add_argument("--yes", "-y", action="store_true", help="全部使用默认参数，不交互")
    parser.add_argument("--list", action="store_true", help="列出可用 MIDI / MusicXML 文件")
    parser.add_argument("--json", help="analyze 子命令：把分析结果写入 JSON")
    parser.add_argument("--host", default=None, help="Web 服务监听地址")
    parser.add_argument("--port", type=int, default=None, help="Web 服务端口")
    parser.add_argument("--no-browser", action="store_true", help="启动 Web 时不自动打开浏览器")
    parser.add_argument("--verbose", "-v", action="store_true", help="输出调试日志")
    return parser


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------


def _parse_track_selection(spec: str | None, tracks: list[dict[str, Any]]) -> set[int] | None:
    if not spec:
        return None
    spec = spec.strip().lower()
    if spec in ("all", "*"):
        return {t["index"] for t in tracks if t["n_notes"] > 0 and t["role"] != "percussion"}
    selected: set[int] = set()
    for part in spec.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                a, b = part.split("-", 1)
                selected.update(range(int(a), int(b) + 1))
                continue
            except ValueError:
                pass
        try:
            selected.add(int(part))
        except ValueError:
            print(f"忽略无效的轨道编号：{part}")
    return selected


def cmd_render(args: argparse.Namespace) -> int:
    midi_path = args.midi or (args.target if args.target and not args.target.endswith(("/", "\\")) else None)
    if not midi_path:
        midi_path = select_midi()

    ensure_dirs()

    # 读取 MIDI / 乐谱元信息用于展示 / 轨道配置
    analysis = analyze(midi_path)
    print_analysis_summary(analysis)
    if not analysis.tracks or not analysis.notes:
        print("该文件没有可用音符，无法生成。")
        return 2

    options = RenderOptions(
        model_kind=args.model or "markov",
        voice=args.voice or DEFAULT_VOICE,
        sample_set=args.sample_set,
        seed=args.seed,
        syntax_weight=max(0.0, min(1.0, args.syntax_weight)),
        velocity_gamma=args.gamma,
        workers=args.workers,
        decomposed_merge=not args.no_decomposed,
        auto_mix=not args.no_auto_mix,
        write_stems=not args.no_stems,
        write_premix=not args.no_premix,
        write_bad_note_midis=not args.no_bad_notes,
        write_zip=not args.no_zip,
        write_subtitle_videos=not args.no_videos,
        audio_format=args.audio_format,
        gpu=args.gpu,
        gpu_memory_mb=args.gpu_memory_mb,
        max_seconds=args.max_seconds,
    )
    if args.tail_seconds is not None:
        options.tail_seconds = max(0.0, float(args.tail_seconds))

    # 交互选择模型
    if not args.model and not args.yes and sys.stdin.isatty():
        options.model_kind = select_model_kind(options.model_kind)

    tracks = resolve_track_options(analysis, options)
    selected_spec = _parse_track_selection(args.tracks, tracks)
    if selected_spec is not None:
        for t in tracks:
            t["selected"] = t["index"] in selected_spec

    interactive = not args.yes and sys.stdin.isatty()
    if interactive:
        tracks = configure_tracks(tracks, interactive=True)

    options.tracks = [
        {
            "index": t["index"],
            "selected": bool(t["selected"]),
            "voice": t["voice"],
            "octave": t["octave"],
            "gain": t["gain"],
        }
        for t in tracks
    ]

    chosen = [t for t in options.tracks if t["selected"]]
    if not chosen:
        print("没有勾选任何轨道。")
        return 2

    print("即将生成：")
    for t in chosen:
        print(
            f"  轨道 {t['index']:>2}  {t['voice']:<10} 八度 {t['octave']:+d}  "
            f"音量 {t['gain']:.2f}"
        )
    print(
        f"  语音模式：{options.voice}    模型：{options.model_kind}    "
        f"力度曲线：{options.velocity_gamma}    进程：{options.workers or '自动'}    "
        f"自动混音：{'开' if options.auto_mix else '关'}"
    )

    print("\n开始渲染…（Ctrl+C 可中断）")
    progress = ConsoleProgress()
    try:
        result = render_midi(
            midi_path,
            options,
            output_dir=args.out,
            project_name=args.project,
            progress_cb=progress,
        )
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130
    except Exception as exc:
        logger.error("渲染失败：%s", exc)
        if args.verbose:
            traceback.print_exc()
        return 1

    print("\n" + "=" * 72)
    print("生成完成！")
    print(f"  音频：{result.audio_path}  ({result.duration:.1f}s)")
    print(f"  目录：{result.output_dir}")
    print(f"  渲染：{result.render_seconds:.1f}s（{result.stats['render']['realtime_factor']:.1f}× 实时）")
    mix_stats = result.stats.get("mix") or {}
    print(
        f"  混音：{'自动摆位' if mix_stats.get('enabled') else '统一后处理'}    "
        f"响度 {mix_stats.get('loudness_db', '—')} dB RMS    "
        f"峰值因数 {mix_stats.get('crest_db', '—')} dB    "
        f"压缩 {mix_stats.get('compression_ratio', '—')}:1    "
        f"相关度 {mix_stats.get('correlation', '—')}"
    )
    print("  字幕 / 资料：")
    for key, value in result.files.items():
        if isinstance(value, dict):
            for k2, v2 in value.items():
                print(f"    {key}[{k2}]: {v2}")
        else:
            print(f"    {key}: {value}")
    verification = result.verification or {}
    if verification:
        print(f"  发声校验：{verification.get('message', '')}")
    print("=" * 72)
    return 0


# ---------------------------------------------------------------------------
# 分析
# ---------------------------------------------------------------------------


def cmd_analyze(args: argparse.Namespace) -> int:
    midi_path = args.midi or args.midi_arg or (args.target if args.target != "analyze" else None)
    if not midi_path:
        midi_path = select_midi()
    result = analyze(midi_path)
    print_analysis_summary(result)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"分析结果已写入 {args.json}")
    return 0


# ---------------------------------------------------------------------------
# Web / 启动器
# ---------------------------------------------------------------------------


def cmd_web(args: argparse.Namespace) -> int:
    from hachimi.server import run_server

    return run_server(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


def cmd_launch(args: argparse.Namespace) -> int:
    from hachimi.launcher import launch

    return launch(port=args.port, host=args.host, open_browser=not args.no_browser)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    setup_console()
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    for noisy in ("matplotlib", "PIL", "numba", "librosa", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.list:
        files = iter_midi_files()
        print(f"共 {len(files)} 个 MIDI / MusicXML：")
        for i, p in enumerate(files):
            print(f"[{i:>3}] {p}")
        return 0

    target = (args.target or "").lower()
    if target in ("web", "serve"):
        return cmd_web(args)
    if target in ("launch", "gui", "launcher"):
        return cmd_launch(args)
    if target == "analyze":
        return cmd_analyze(args)
    return cmd_render(args)


if __name__ == "__main__":
    sys.exit(main())
