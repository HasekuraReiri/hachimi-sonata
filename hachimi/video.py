"""纯绿幕字幕视频：全部字幕合并为一条，供 BB 素材抠像。

设计取舍
--------
* **音频完成后再生成**：音频合成已经吃满多核 CPU，ffmpeg 的 x264 同样是
  CPU 密集型；并发会互相抢核、拉长总时间，还会让内存峰值叠加。字幕视频
  只用最终时间轴，不需要等音频之外的数据，因此串行生成是时间最优方案。
* **逐句 PNG + concat，而不是逐帧 drawtext**：每句只排版一次，ffmpeg 只需
  把静态画面按时间拼接；814 秒的歌也只有几百张图，不会每帧重新排版文字。
* **固定宽 1920，单条视频、每次最多 1 行**：全部声部的歌词槽合并成一条
  连续时间线；任何时刻最多只有一行字幕，不横向压缩、不裁切。必要时自动
  缩小字号，保证这句仍是一行。
* **纯绿 #00FF00 + 默认 8 号白色宋体 + 2px 黑边**：背景没有渐变 / 阴影，
  黑边只在文字自身，方便 Chroma Key；字体 / 字号 / 颜色可在网页左侧调整。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from hachimi.config import (
    PROJECT_ROOT,
    SUBTITLE_VIDEO_CRF,
    SUBTITLE_VIDEO_FPS,
    SUBTITLE_VIDEO_FONT_FAMILY,
    SUBTITLE_VIDEO_FONT_SIZE,
    SUBTITLE_VIDEO_FONT_UNIT_PX,
    SUBTITLE_VIDEO_GREEN,
    SUBTITLE_VIDEO_LINE_SPACING,
    SUBTITLE_VIDEO_MAX_LINES,
    SUBTITLE_VIDEO_PADDING_RATIO,
    SUBTITLE_VIDEO_TEXT_COLOR,
    SUBTITLE_VIDEO_TEXT_STROKE_COLOR,
    SUBTITLE_VIDEO_TEXT_STROKE_WIDTH,
    SUBTITLE_VIDEO_WIDTH,
    TEMP_DIR,
)
from hachimi.subtitles import SubtitleLine, write_srt

logger = logging.getLogger(__name__)

#: 与用户要求一致的纯绿 / 纯白；不要改成带灰度的“演播室绿”。
GREEN_HEX = SUBTITLE_VIDEO_GREEN.upper()
WHITE_HEX = SUBTITLE_VIDEO_TEXT_COLOR.upper()
STROKE_HEX = SUBTITLE_VIDEO_TEXT_STROKE_COLOR.upper()


def _hex_rgb(value: str) -> tuple[int, int, int]:
    text = value.lstrip("#")
    return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _normalize_hex(value: str, fallback: str = WHITE_HEX) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"#?[0-9a-fA-F]{6}", text):
        return ("#" + text.lstrip("#")).upper()
    return fallback


GREEN_RGB = _hex_rgb(GREEN_HEX)
WHITE_RGB = _hex_rgb(WHITE_HEX)

#: 可选字体：默认宋体；其余用于 UI 下拉 / 自定义。
FONT_FAMILIES: dict[str, tuple[tuple[str, str], ...]] = {
    "SimSun": (("simsun.ttc", "SimSun"),),
    "Microsoft YaHei": (
        ("msyh.ttc", "Microsoft YaHei"),
        ("msyhbd.ttc", "Microsoft YaHei Bold"),
    ),
    "SimHei": (("simhei.ttf", "SimHei"),),
    "Noto Sans SC": (("NotoSansSC-VF.ttf", "Noto Sans SC"),),
    "DengXian": (("Deng.ttf", "DengXian"),),
}
DEFAULT_FONT_FAMILY = SUBTITLE_VIDEO_FONT_FAMILY if SUBTITLE_VIDEO_FONT_FAMILY in FONT_FAMILIES else "SimSun"

FFMPEG_CANDIDATES: tuple[str, ...] = (
    "ffmpeg.exe",
    r"C:\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
)


@dataclass(frozen=True)
class VideoLayout:
    width: int
    height: int
    font_path: str
    font_name: str
    font_family: str
    font_size: int
    font_units: float
    text_rgb: tuple[int, int, int]
    text_hex: str
    stroke_width: int
    stroke_rgb: tuple[int, int, int]
    stroke_hex: str
    max_lines: int
    line_height: int
    padding_x: int
    padding_y: int
    usable_width: int


@dataclass(frozen=True)
class TimelineSegment:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _windows_fonts_dir() -> str:
    return os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")


def resolve_font_family(family: str | None = None) -> str:
    """把界面上的字体名归一化到 FONT_FAMILIES 的键。"""
    text = str(family or "").strip()
    if text in FONT_FAMILIES:
        return text
    lowered = text.lower()
    aliases = {
        "宋体": "SimSun",
        "simsun": "SimSun",
        "微软雅黑": "Microsoft YaHei",
        "microsoft yahei": "Microsoft YaHei",
        "msyh": "Microsoft YaHei",
        "黑体": "SimHei",
        "simhei": "SimHei",
        "noto sans sc": "Noto Sans SC",
        "noto": "Noto Sans SC",
        "等线": "DengXian",
        "dengxian": "DengXian",
    }
    if lowered in aliases:
        return aliases[lowered]
    return DEFAULT_FONT_FAMILY


def find_video_font(family: str | None = None) -> tuple[str, str] | None:
    """按字体族返回 (字体文件, 字体名)；找不到时回退宋体 / 雅黑。"""
    env_path = os.environ.get("HACHIMI_SUBTITLE_FONT", "").strip()
    if env_path and os.path.isfile(env_path):
        return env_path, Path(env_path).stem

    requested = resolve_font_family(family)
    keys = [requested] + [k for k in ("SimSun", "Microsoft YaHei", "SimHei", "Noto Sans SC") if k != requested]
    fonts_dir = _windows_fonts_dir()
    for key in keys:
        for filename, name in FONT_FAMILIES.get(key, ()):
            for base in (fonts_dir, os.path.join(str(PROJECT_ROOT), "assets", "fonts")):
                path = os.path.join(base, filename)
                if path and os.path.isfile(path):
                    return path, name
    return None


def find_ffmpeg() -> str | None:
    """定位 ffmpeg：环境变量 → PATH → 常见安装位置。"""
    for key in ("HACHIMI_FFMPEG", "FFMPEG_BIN"):
        value = os.environ.get(key, "").strip().strip('"')
        if value and os.path.isfile(value):
            return value
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in FFMPEG_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    # WinGet Links 是当前机器上常见的位置
    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        candidate = os.path.join(local, "Microsoft", "WinGet", "Links", "ffmpeg.exe")
        if os.path.isfile(candidate):
            return candidate
    return None


def _font_for(layout: VideoLayout) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(layout.font_path, layout.font_size, index=0)
    except OSError:
        # TTC 个别索引不可用时退回默认索引
        return ImageFont.truetype(layout.font_path, layout.font_size)


def _line_text(line: Any) -> str:
    if isinstance(line, Mapping):
        return str(line.get("text", "") or "")
    return str(getattr(line, "text", "") or "")


def _line_time(line: Any, name: str) -> float:
    if isinstance(line, Mapping):
        return float(line.get(name, 0.0) or 0.0)
    return float(getattr(line, name, 0.0) or 0.0)


def _text_width(
    font: ImageFont.FreeTypeFont,
    text: str,
    stroke_width: int = 0,
) -> float:
    """文字像素宽度，包含外侧描边；用于换行时避免黑边被裁切。"""
    try:
        return float(font.getlength(text)) + max(0, int(stroke_width)) * 2
    except Exception:
        return 0.0


def _wrap_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    usable_width: int,
    max_lines: int = 2,
    stroke_width: int = 0,
) -> list[str]:
    """把一句字幕排版成最多 ``max_lines`` 行。

    ``max_lines=1`` 时绝不拆行；若太宽，由 :func:`measure_layout` 逐级
    缩小字号，保证最终仍是一行。
    """
    text = " ".join(str(text or "").split())
    if not text:
        return []
    if _text_width(font, text, stroke_width) <= usable_width:
        return [text]
    if max_lines <= 1:
        return [text]

    if max_lines >= 2 and len(text) > 1:
        best: tuple[float, str, str] | None = None
        for i in range(1, len(text)):
            left = text[:i].strip()
            right = text[i:].strip()
            if not left or not right:
                continue
            lw = _text_width(font, left, stroke_width)
            rw = _text_width(font, right, stroke_width)
            if lw <= usable_width and rw <= usable_width:
                # 先保证两行都放得下，再让两行长度尽量接近。
                score = max(lw, rw) + abs(lw - rw) * 0.2
                if best is None or score < best[0]:
                    best = (score, left, right)
        if best is not None:
            return [best[1], best[2]]

    # 放不下两行时先按字符贪心切分，measure_layout 会据此缩小字号。
    lines: list[str] = []
    current = ""
    for ch in text:
        candidate = current + ch
        if current and _text_width(font, candidate, stroke_width) > usable_width:
            lines.append(current)
            current = ch
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [text]


def measure_layout(
    line_sets: Iterable[Sequence[Any]],
    *,
    width: int = SUBTITLE_VIDEO_WIDTH,
    font_family: str = DEFAULT_FONT_FAMILY,
    font_size: float | None = SUBTITLE_VIDEO_FONT_SIZE,
    text_color: str = SUBTITLE_VIDEO_TEXT_COLOR,
    font_unit_px: float = SUBTITLE_VIDEO_FONT_UNIT_PX,
    max_lines: int = SUBTITLE_VIDEO_MAX_LINES,
    stroke_width: int = SUBTITLE_VIDEO_TEXT_STROKE_WIDTH,
    stroke_color: str = SUBTITLE_VIDEO_TEXT_STROKE_COLOR,
    min_font_size: int = 48,
) -> VideoLayout:
    """固定宽 1920；高度按实际排版行数计算。

    字号参数以“号”为单位（默认 8 号），按 ``font_unit_px`` 换算成像素。
    ``max_lines=1`` 时，全部字幕严格保持一行；若某句太宽，逐级缩小字号
    直到整句放进画面，绝不拆成第二行。``max_lines>=2`` 时保留原先的
    “能一行就一行、否则均分两行”行为。默认带 2px 黑色细描边。
    """
    family = resolve_font_family(font_family)
    found = find_video_font(family)
    if found is None:
        raise RuntimeError(
            "找不到中文字体（宋体 / 微软雅黑 / 黑体 / Noto Sans SC）。"
            "可设置环境变量 HACHIMI_SUBTITLE_FONT 指向一个 ttf/ttc 字体文件。"
        )
    font_path, font_name = found
    width = max(640, int(width))
    width += width % 2
    padding_x = max(32, int(round(width * SUBTITLE_VIDEO_PADDING_RATIO)))
    usable_width = max(120, width - padding_x * 2)
    max_lines = max(1, int(max_lines))
    color = _normalize_hex(text_color)
    text_rgb = _hex_rgb(color)
    stroke = _normalize_hex(stroke_color, STROKE_HEX)
    stroke_rgb = _hex_rgb(stroke)
    stroke_width = max(0, min(12, int(stroke_width)))
    size = int(round(float(font_size or SUBTITLE_VIDEO_FONT_SIZE) * max(1.0, float(font_unit_px))))
    size = max(12, min(480, size))

    texts: list[str] = []
    seen: set[str] = set()
    for lines in line_sets:
        for line in lines:
            text = _line_text(line).strip()
            if text and text not in seen:
                seen.add(text)
                texts.append(text)

    wraps: dict[str, list[str]] = {}
    if max_lines <= 1:
        # 单行模式：不换行，只缩字号。20 个全角字在 8 号（80px）下约
        # 1600px，正常不会触发；超长句最多缩到 24px，保证仍是一行。
        floor = max(12, min(int(min_font_size), 24))
        too_wide = False
        while True:
            font = ImageFont.truetype(font_path, size, index=0)
            too_wide = any(
                _text_width(font, text, stroke_width) > usable_width
                for text in texts
            )
            if not too_wide or size <= floor:
                break
            size = max(floor, size - 4)
        if too_wide:
            raise RuntimeError(
                f"有字幕在 {size}px 下仍宽于 {usable_width}px，无法保持一行；"
                "请缩短该句歌词或增大视频宽度。"
            )
        wraps = {text: [text] for text in texts}
    else:
        # 正常字幕（≤24 字）在默认 8 号（≈80px）下会落在 1~2 行；
        # 只有外部直接传入超长句、或用户把字号调得很大时才触发字号回退。
        while True:
            font = ImageFont.truetype(font_path, size, index=0)
            wraps = {
                text: _wrap_text(
                    text,
                    font,
                    usable_width,
                    max_lines=max_lines,
                    stroke_width=stroke_width,
                )
                for text in texts
            }
            if all(len(lines) <= max_lines for lines in wraps.values()):
                break
            if size <= min_font_size:
                break
            size = max(min_font_size, size - 4)

    used_lines = max([len(lines) for lines in wraps.values()] or [1])
    if used_lines > max_lines:
        raise RuntimeError(
            f"有字幕在 {size}px 下仍超过 {max_lines} 行，无法完成排版；"
            "请缩短该句歌词或增大视频宽度。"
        )
    used_lines = max(1, used_lines)
    line_height = max(1, int(round(size * SUBTITLE_VIDEO_LINE_SPACING)))
    padding_y = max(24, int(round(size * 0.38)) + stroke_width * 2)
    height = used_lines * line_height + padding_y * 2
    height = max(160, height)
    # yuv420p 要求偶数高度
    height += height % 2
    return VideoLayout(
        width=width,
        height=height,
        font_path=font_path,
        font_name=font_name,
        font_family=family,
        font_size=size,
        font_units=round(size / max(1.0, float(font_unit_px)), 2),
        text_rgb=text_rgb,
        text_hex=color,
        stroke_width=stroke_width,
        stroke_rgb=stroke_rgb,
        stroke_hex=stroke,
        max_lines=used_lines,
        line_height=line_height,
        padding_x=padding_x,
        padding_y=padding_y,
        usable_width=usable_width,
    )


def _relative_lines(
    lines: Sequence[Any],
    *,
    time_offset: float,
    duration: float,
) -> list[SubtitleLine]:
    """把绝对 MIDI 时间换算成视频内部的 0~duration 时间（侧车 SRT 用）。"""
    out: list[SubtitleLine] = []
    offset = max(0.0, float(time_offset))
    duration = max(0.05, float(duration))
    for i, line in enumerate(lines, start=1):
        text = _line_text(line).strip()
        if not text:
            continue
        start = max(0.0, _line_time(line, "start") - offset)
        end = min(duration, _line_time(line, "end") - offset)
        if end <= start + 1e-4:
            continue
        chars = []
        raw_chars = line.get("chars", []) if isinstance(line, Mapping) else getattr(line, "chars", [])
        for ch in raw_chars or []:
            try:
                cs = max(0.0, float(ch.get("s", 0.0)) - offset)
                ce = min(duration, float(ch.get("e", cs)) - offset)
            except (TypeError, ValueError):
                continue
            if ce > cs:
                chars.append({**dict(ch), "s": round(cs, 3), "e": round(ce, 3)})
        out.append(
            SubtitleLine(
                index=i,
                start=round(start, 3),
                end=round(end, 3),
                text=text,
                chars=chars,
                section=str(getattr(line, "section", "") or ""),
                voice=str(getattr(line, "voice", "") or ""),
            )
        )
    return out


def build_video_timeline(
    lines: Sequence[Any],
    duration: float,
    *,
    time_offset: float = 0.0,
    min_gap: float = 0.01,
) -> list[TimelineSegment]:
    """把字幕行换算成 [0, duration] 的时间轴；空档用空白绿幕补齐。"""
    duration = max(0.05, float(duration))
    offset = max(0.0, float(time_offset))
    raw: list[tuple[float, float, str]] = []
    for line in lines:
        text = _line_text(line).strip()
        if not text:
            continue
        start = max(0.0, _line_time(line, "start") - offset)
        end = min(duration, _line_time(line, "end") - offset)
        if end <= start + 1e-4:
            continue
        raw.append((start, end, text))
    raw.sort(key=lambda item: (item[0], item[1]))

    segments: list[TimelineSegment] = []

    def add_blank(start: float, end: float) -> None:
        if end - start <= min_gap:
            return
        if segments and not segments[-1].text:
            segments[-1] = TimelineSegment(segments[-1].start, end, "")
        else:
            segments.append(TimelineSegment(start, end, ""))

    cursor = 0.0
    for start, end, text in raw:
        if end <= cursor + 1e-4:
            continue
        start = max(start, cursor)
        if start - cursor > min_gap:
            add_blank(cursor, start)
        end = min(duration, max(start + 0.04, end))
        if end <= start:
            continue
        segments.append(TimelineSegment(start, end, text))
        cursor = end
    if cursor < duration - min_gap:
        add_blank(cursor, duration)
    if not segments:
        segments.append(TimelineSegment(0.0, duration, ""))
    return segments


def _draw_frame(path: str, text: str, layout: VideoLayout) -> None:
    img = Image.new("RGB", (layout.width, layout.height), GREEN_RGB)
    if text:
        font = _font_for(layout)
        draw = ImageDraw.Draw(img)
        lines = _wrap_text(
            text,
            font,
            layout.usable_width,
            max_lines=layout.max_lines,
            stroke_width=layout.stroke_width,
        )
        if lines:
            block_height = len(lines) * layout.line_height
            top = (layout.height - block_height) / 2.0
            for i, line in enumerate(lines):
                bbox = draw.textbbox(
                    (0, 0),
                    line,
                    font=font,
                    stroke_width=layout.stroke_width,
                )
                text_w = max(1, bbox[2] - bbox[0])
                text_h = max(1, bbox[3] - bbox[1])
                x = (layout.width - text_w) / 2.0 - bbox[0]
                y = top + i * layout.line_height + (layout.line_height - text_h) / 2.0 - bbox[1]
                # 默认白字 + 2px 黑边；纯绿底上仍保持清晰，抠像后边缘自然。
                draw.text(
                    (x, y),
                    line,
                    font=font,
                    fill=layout.text_rgb,
                    stroke_width=layout.stroke_width,
                    stroke_fill=layout.stroke_rgb,
                )
    # compress_level=1 比默认快很多；纯色 PNG 本来就很小的。
    img.save(path, "PNG", compress_level=1)


def _concat_entry(path: str, duration: float) -> str:
    # ffmpeg concat demuxer 的路径使用正斜杠，单引号需要转义。
    safe = path.replace("\\", "/").replace("'", "'\\''")
    return f"file '{safe}'\nduration {duration:.6f}\n"


def _encode_with_ffmpeg(
    ffmpeg: str,
    concat_path: str,
    output_path: str,
    *,
    duration: float,
    fps: int,
    crf: int,
) -> None:
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_path,
        # 注意：ffmpeg 9 已移除 -vsync；而只写 -r 30 不会补齐 concat
        # 末尾的静帧，长空档会被截短。用 fps 滤镜可稳定输出完整定位。
        "-vf",
        f"fps={max(1, int(fps))}",
        "-t",
        f"{max(0.05, float(duration)):.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(max(0, int(crf))),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(
            "ffmpeg 生成字幕视频失败"
            + (f"：{stderr[-1200:]}" if stderr else f"（退出码 {proc.returncode}）")
        )
    if not os.path.isfile(output_path) or os.path.getsize(output_path) <= 0:
        raise RuntimeError("ffmpeg 没有写出有效的视频文件")
    logger.info(
        "字幕视频完成：%s（%.1f 秒，%.2f MB，%.1f 秒编码）",
        os.path.basename(output_path),
        duration,
        os.path.getsize(output_path) / 1048576,
        time.perf_counter() - started,
    )


def _render_one(
    *,
    ffmpeg: str,
    layout: VideoLayout,
    lines: Sequence[Any],
    output_path: str,
    duration: float,
    fps: int,
    crf: int,
    time_offset: float,
    frame_dir: str,
    frame_cache: dict[str, str],
) -> dict[str, Any]:
    started = time.perf_counter()
    segments = build_video_timeline(lines, duration, time_offset=time_offset)
    if not segments:
        raise RuntimeError("没有可输出的字幕行")

    blank_path = frame_cache.get("")
    if blank_path is None:
        blank_path = os.path.join(frame_dir, "blank.png")
        _draw_frame(blank_path, "", layout)
        frame_cache[""] = blank_path

    concat_lines: list[str] = []
    cursor = 0.0
    for segment in segments:
        # build_video_timeline 在相邻行之间会留 0.01s 防重叠空隙；
        # concat 必须把这些空隙显式补成空白绿幕，否则整条视频会逐句提前。
        if segment.start > cursor + 1e-4:
            concat_lines.append(_concat_entry(blank_path, segment.start - cursor))
        text = segment.text
        frame = frame_cache.get(text)
        if frame is None:
            digest = hashlib.md5(text.encode("utf-8", "replace")).hexdigest()[:12]
            frame = os.path.join(frame_dir, f"line_{digest}.png")
            _draw_frame(frame, text, layout)
            frame_cache[text] = frame
        concat_lines.append(_concat_entry(frame, max(0.04, segment.duration)))
        cursor = max(cursor, segment.end)
    # concat demuxer 默认在最后一张图后停止；重复最后一张可保证 -t 覆盖完整时长。
    last_frame = frame_cache.get(segments[-1].text) or blank_path
    concat_lines.append(_concat_entry(last_frame, 1.0 / max(1, fps)))

    concat_path = os.path.join(frame_dir, f"concat_{hashlib.md5(output_path.encode('utf-8')).hexdigest()[:8]}.txt")
    with open(concat_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(concat_lines)

    _encode_with_ffmpeg(
        ffmpeg,
        concat_path,
        output_path,
        duration=duration,
        fps=fps,
        crf=crf,
    )
    return {
        "file": output_path,
        "name": os.path.basename(output_path),
        "duration": round(duration, 3),
        "segments": len(segments),
        "text_segments": sum(1 for s in segments if s.text),
        "bytes": os.path.getsize(output_path),
        "encode_seconds": round(time.perf_counter() - started, 2),
    }


def _write_meta(path: str, payload: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(payload), fh, ensure_ascii=False, indent=2)


def render_subtitle_video(
    lines: Sequence[Any],
    *,
    out_dir: str,
    project: str,
    duration: float,
    time_offset: float = 0.0,
    width: int = SUBTITLE_VIDEO_WIDTH,
    fps: int = SUBTITLE_VIDEO_FPS,
    font_family: str = DEFAULT_FONT_FAMILY,
    font_size: float | None = SUBTITLE_VIDEO_FONT_SIZE,
    text_color: str = SUBTITLE_VIDEO_TEXT_COLOR,
    stroke_width: int = SUBTITLE_VIDEO_TEXT_STROKE_WIDTH,
    stroke_color: str = SUBTITLE_VIDEO_TEXT_STROKE_COLOR,
    crf: int = SUBTITLE_VIDEO_CRF,
    track_info: Mapping[str, Mapping[str, Any]] | None = None,
    coverage: Mapping[str, Any] | None = None,
    progress_cb=None,
) -> dict[str, Any]:
    """生成一条包含全部字幕的绿幕视频，任何时刻最多一行。

    字幕行由 pipeline 先把全部参与轨道汇总成一条连续时间线，本函数只负责
    排版、逐句 PNG 与编码。侧车 SRT 与视频使用同一条序列化时间轴，因此
    SRT 也不会出现同一时刻两行。

    Returns:
        dict：``available`` / ``files`` / ``srt`` / ``tracks`` / 规格信息。
        没有 ffmpeg 或字体时返回 ``available=False``，不抛异常，避免影响音频输出。
    """
    started = time.perf_counter()
    raw_lines = list(lines or [])
    key = "all"
    if not raw_lines:
        return {
            "available": False,
            "error": "没有可用的字幕行",
            "files": {},
            "srt": {},
            "tracks": {},
        }

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return {
            "available": False,
            "error": "未找到 ffmpeg；字幕视频已跳过，音频与字幕文件不受影响",
            "files": {},
            "srt": {},
            "tracks": {},
        }

    try:
        layout = measure_layout(
            [raw_lines],
            width=width,
            font_family=font_family,
            font_size=font_size,
            text_color=text_color,
            max_lines=1,
            stroke_width=stroke_width,
            stroke_color=stroke_color,
        )
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "files": {},
            "srt": {},
            "tracks": {},
        }

    video_dir = os.path.join(out_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    try:
        os.makedirs(TEMP_DIR, exist_ok=True)
        frame_dir = tempfile.mkdtemp(prefix="hachimi_subtitle_", dir=TEMP_DIR)
    except OSError:
        frame_dir = tempfile.mkdtemp(prefix="hachimi_subtitle_")

    files: dict[str, str] = {}
    srt_files: dict[str, str] = {}
    track_results: dict[str, Any] = {}
    errors: list[str] = []
    info: dict[str, Any] = {}
    if track_info:
        candidate = track_info.get(key) if key in track_info else track_info
        if isinstance(candidate, Mapping):
            info = dict(candidate)

    try:
        # 视频时间轴从 0 开始；侧车 SRT 也必须是相对时间，
        # 不能把源 MIDI 的 2:29 原样写进去。
        relative = _relative_lines(raw_lines, time_offset=time_offset, duration=duration)
        if not relative:
            track_results[key] = {**info, "lines": 0, "skipped": "没有可输出的字幕行"}
        else:
            output_path = os.path.join(video_dir, f"subtitle_{key}_green.mp4")
            srt_path = os.path.join(video_dir, f"subtitle_{key}.srt")
            # 与视频共用 build_video_timeline：前一行的结束时间被压到下
            # 一行开始前，SRT 也严格保持“每次最多一行”。
            timeline = build_video_timeline(relative, duration, time_offset=0.0)
            srt_lines = [
                SubtitleLine(
                    index=i,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                )
                for i, segment in enumerate(
                    (s for s in timeline if s.text),
                    start=1,
                )
            ]
            write_srt(srt_lines, srt_path)
            srt_files[key] = srt_path
            if progress_cb:
                progress_cb(0, 1, f"渲染全部字幕视频（{len(relative)} 句，单行）")
            video_frame_dir = os.path.join(frame_dir, key)
            os.makedirs(video_frame_dir, exist_ok=True)
            try:
                result = _render_one(
                    ffmpeg=ffmpeg,
                    layout=layout,
                    lines=relative,
                    output_path=output_path,
                    duration=duration,
                    fps=fps,
                    crf=crf,
                    time_offset=0.0,
                    frame_dir=video_frame_dir,
                    frame_cache={},
                )
            except Exception as exc:
                logger.warning("全部字幕视频生成失败：%s", exc)
                errors.append(str(exc))
                track_results[key] = {
                    **info,
                    "lines": len(relative),
                    "error": str(exc),
                }
            else:
                files[key] = output_path
                track_results[key] = {
                    **info,
                    "lines": len(relative),
                    "srt": os.path.basename(srt_path),
                    "segments": result["segments"],
                    "text_segments": result["text_segments"],
                    "encode_seconds": result["encode_seconds"],
                    "bytes": result["bytes"],
                    "height": layout.height,
                    "max_lines": layout.max_lines,
                    "font_size": layout.font_units,
                    "font_px": layout.font_size,
                    "stroke_width": layout.stroke_width,
                }
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)

    layout_meta = {
        key: {
            "width": layout.width,
            "height": layout.height,
            "max_lines": layout.max_lines,
            "font_size": layout.font_units,
            "font_px": layout.font_size,
            "stroke_width": layout.stroke_width,
            "stroke_color": layout.stroke_hex,
        }
    }
    common = {
        "width": layout.width,
        "height": layout.height,
        "max_lines": layout.max_lines,
        "single_video": True,
        "fps": int(fps),
        "font": layout.font_name,
        "font_family": layout.font_family,
        "font_size": layout.font_units,
        "font_px": layout.font_size,
        "background": GREEN_HEX,
        "text_color": layout.text_hex,
        "stroke_width": layout.stroke_width,
        "stroke_color": layout.stroke_hex,
        "layouts": layout_meta,
        "coverage": dict(coverage or {}),
        "duration": round(float(duration), 3),
    }

    meta_path = os.path.join(video_dir, "subtitle_videos.json")
    meta = {
        "project": project,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "single_video_after_audio_png_concat",
        **common,
        "font_file": os.path.basename(layout.font_path),
        "tracks": track_results,
        "files": {k: os.path.relpath(v, out_dir).replace("\\", "/") for k, v in files.items()},
        "errors": errors,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    _write_meta(meta_path, meta)

    if not files:
        return {
            "available": False,
            "error": "；".join(errors) or "字幕视频没有生成",
            **common,
            "files": {},
            "srt": srt_files,
            "tracks": track_results,
            "meta_path": meta_path,
        }
    return {
        "available": True,
        **common,
        "files": files,
        "srt": srt_files,
        "tracks": track_results,
        "meta_path": meta_path,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "errors": errors,
    }


def probe_video(path: str) -> dict[str, Any]:
    """用 ffprobe 读取视频规格；没有 ffprobe 时返回空 dict。"""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {}
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return {}
        data = json.loads(proc.stdout or "{}")
        streams = data.get("streams") or [{}]
        fmt = data.get("format") or {}
        return {
            "width": streams[0].get("width"),
            "height": streams[0].get("height"),
            "fps": streams[0].get("r_frame_rate"),
            "frames": streams[0].get("nb_frames"),
            "duration": float(fmt.get("duration") or 0.0),
            "size": int(fmt.get("size") or 0),
        }
    except Exception:
        return {}


__all__ = [
    "GREEN_HEX",
    "GREEN_RGB",
    "WHITE_HEX",
    "WHITE_RGB",
    "DEFAULT_FONT_FAMILY",
    "FONT_FAMILIES",
    "TimelineSegment",
    "VideoLayout",
    "build_video_timeline",
    "find_ffmpeg",
    "find_video_font",
    "measure_layout",
    "probe_video",
    "render_subtitle_video",
]
