"""工程参数包：保存、读取与复用一次工程的全部生成设置。

每个作品除了音频 / 字幕以外，还会额外生成：

* ``<工程>_params.json``：完整参数与轨道设置；
* ``<工程>_params.zip``：只含参数 JSON 的小压缩包，便于单独发给别人复用。

参数包同时兼容旧作品的 ``manifest.json``：旧工程没有参数文件时，服务端可从
manifest 的 ``tracks`` / ``analysis.render_tracks`` 还原可复用的部分设置。
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from datetime import datetime
from typing import Any, Mapping, Sequence

from hachimi.config import (
    APP_VERSION,
    PARAMS_SCHEMA,
    PARAMS_SCHEMA_VERSION,
    RenderOptions,
)

MAX_IMPORT_BYTES = 32 * 1024 * 1024

_TRACK_KEYS = ("index", "name", "instrument", "selected", "voice", "octave", "gain", "role")


def _sha256(path: str) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def _clean_track(track: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _TRACK_KEYS:
        if key in track:
            out[key] = track[key]
    if "index" in out:
        try:
            out["index"] = int(out["index"])
        except (TypeError, ValueError):
            pass
    if "octave" in out:
        try:
            out["octave"] = max(-3, min(3, int(out["octave"])))
        except (TypeError, ValueError):
            pass
    if "gain" in out:
        try:
            out["gain"] = max(0.0, min(2.0, float(out["gain"])))
        except (TypeError, ValueError):
            pass
    return out


def build_project_params(
    *,
    project: str,
    midi_path: str,
    options: RenderOptions,
    track_options: Mapping[int, Mapping[str, Any]],
    selected: Sequence[int],
    audible: Sequence[int],
    analysis: Any,
    outputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构造一个可序列化、可跨工程复用的参数包。"""
    tracks = [
        _clean_track(track_options[int(i)])
        for i in sorted(track_options)
        if int(i) in track_options
    ]
    option_dict = options.to_dict()
    option_dict["tracks"] = tracks
    source_name = os.path.basename(midi_path)
    try:
        source_size = os.path.getsize(midi_path)
    except OSError:
        source_size = 0
    stats = {}
    try:
        stats = analysis.stats() or {}
    except Exception:
        pass
    key = {}
    try:
        key = analysis.key or {}
    except Exception:
        pass
    return {
        "schema": PARAMS_SCHEMA,
        "schema_version": PARAMS_SCHEMA_VERSION,
        "app_version": APP_VERSION,
        "project": project,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": {
            "file_name": source_name,
            "file_size": source_size,
            "sha256": _sha256(midi_path),
            "format": getattr(analysis, "source_format", "midi"),
            "duration": round(float(getattr(analysis, "duration", 0.0) or 0.0), 3),
            "key": key.get("label_cn") or key.get("label") or "",
            "n_tracks": int(stats.get("n_tracks", 0) or 0),
            "n_notes": int(stats.get("n_notes", 0) or 0),
            "n_sections": int(stats.get("n_sections", 0) or 0),
            "n_chords": int(stats.get("n_chords", 0) or 0),
        },
        "options": option_dict,
        "tracks": tracks,
        "selected": [int(i) for i in selected],
        "audible": [int(i) for i in audible],
        "outputs": dict(outputs or {}),
    }


def write_project_params(
    params: Mapping[str, Any],
    out_dir: str,
    project: str,
) -> dict[str, str]:
    """写出参数 JSON 与小 ZIP，返回 ``{json, zip}`` 路径。"""
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{project}_params.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(dict(params), fh, ensure_ascii=False, indent=2)

    zip_path = os.path.join(out_dir, f"{project}_params.zip")
    readme = (
        "Hachimi Sonata 工程参数包\n"
        "=========================\n"
        "把本 ZIP 或其中的 *_params.json 导入网页的「工程参数复用」栏目，\n"
        "即可复制该工程的全部生成设置、轨道语音 / 八度 / 音量、段落编曲计划与字幕样式。\n"
        "参数包不包含音频、MIDI 或任何个人信息。\n"
    ).encode("utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(json_path, os.path.basename(json_path))
        zf.writestr("README.txt", readme)
    return {"json": json_path, "zip": zip_path}


def params_from_manifest(manifest: Mapping[str, Any], *, source: str = "") -> dict[str, Any]:
    """从旧作品 manifest 还原可复用参数（不要求存在 _params.json）。"""
    analysis = manifest.get("analysis") or {}
    raw_tracks = manifest.get("tracks") or analysis.get("render_tracks") or []
    tracks = [_clean_track(t) for t in raw_tracks if isinstance(t, Mapping)]
    tracks = [t for t in tracks if "index" in t]
    plans = analysis.get("section_plans") or []
    if not plans:
        plans = (manifest.get("stats") or {}).get("lyrics", {}).get("section_plans") or []
    stats = manifest.get("stats") or {}
    mix = stats.get("mix") or manifest.get("mix_report") or {}
    video = stats.get("video") or {}
    options: dict[str, Any] = {
        "tracks": tracks,
        "section_plans": plans,
        "auto_mix": bool(mix.get("enabled", True)),
        "write_subtitle_videos": bool(video.get("available", True)),
        "subtitle_video_font_size": video.get("font_size", 8),
        "subtitle_video_font_family": video.get("font", "SimSun"),
        "subtitle_video_text_color": video.get("text_color", "#FFFFFF"),
        "subtitle_video_width": video.get("width", 1920),
        "subtitle_video_fps": video.get("fps", 30),
    }
    for item in tracks:
        if item.get("voice"):
            options["voice"] = item["voice"]
            break
    return {
        "schema": PARAMS_SCHEMA,
        "schema_version": PARAMS_SCHEMA_VERSION,
        "app_version": APP_VERSION,
        "legacy": True,
        "project": manifest.get("project", ""),
        "created_at": "",
        "source": {
            "file_name": manifest.get("midi_path", ""),
            "format": analysis.get("source_format", "midi"),
            "duration": manifest.get("duration", 0),
            "key": (analysis.get("key") or {}).get("label_cn", ""),
            "n_tracks": len(tracks),
            "n_notes": (analysis.get("stats") or {}).get("n_notes", 0),
        },
        "options": options,
        "tracks": tracks,
        "imported_from": source,
        "note": "旧作品 manifest 缺少部分全局参数；已恢复轨道设置与段落计划。",
    }


def _unwrap_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("参数包不是 JSON 对象")
    data = dict(payload)
    if isinstance(data.get("params"), Mapping):
        data = dict(data["params"])
    # 兼容有人把参数包再包一层 { "schema":..., "options":... }
    return data


def _read_json_bytes(raw: bytes) -> Any:
    text = raw.decode("utf-8-sig")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"参数 JSON 解析失败：{exc}") from exc


def load_params_bytes(raw: bytes, filename: str = "params.zip") -> dict[str, Any]:
    """从 ZIP / JSON 字节读取参数包；ZIP 内没有参数时回退到 manifest。"""
    if not raw:
        raise ValueError("参数包为空")
    if len(raw) > MAX_IMPORT_BYTES:
        raise ValueError("参数包超过 32 MB，已拒绝")
    lower = (filename or "").lower()
    if lower.endswith(".zip") or raw[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                infos = [i for i in zf.infolist() if not i.is_dir()]
                if len(infos) > 200:
                    raise ValueError("参数包内文件过多")
                if sum(int(i.file_size) for i in infos) > 64 * 1024 * 1024:
                    raise ValueError("参数包解压后超过 64 MB，已拒绝")
                names = [i.filename for i in infos]
                param_names = [n for n in names if n.lower().endswith("_params.json")]
                if not param_names:
                    param_names = [n for n in names if os.path.basename(n).lower() == "params.json"]
                if param_names:
                    payload = _unwrap_payload(_read_json_bytes(zf.read(param_names[0])))
                    payload.setdefault("imported_from", filename)
                    return payload
                manifests = [n for n in names if os.path.basename(n) == "manifest.json"]
                if manifests:
                    manifest = _read_json_bytes(zf.read(manifests[0]))
                    if isinstance(manifest, Mapping):
                        return params_from_manifest(manifest, source=filename)
                raise ValueError("ZIP 内没有 *_params.json 或 manifest.json")
        except zipfile.BadZipFile as exc:
            raise ValueError("ZIP 参数包损坏") from exc
    payload = _unwrap_payload(_read_json_bytes(raw))
    if "options" not in payload and "project" in payload and "analysis" in payload:
        return params_from_manifest(payload, source=filename)
    payload.setdefault("imported_from", filename)
    return payload


def load_params_file(path: str) -> dict[str, Any]:
    with open(path, "rb") as fh:
        raw = fh.read(MAX_IMPORT_BYTES + 1)
    return load_params_bytes(raw, os.path.basename(path))


def params_summary(params: Mapping[str, Any]) -> dict[str, Any]:
    """给网页列表用的摘要。"""
    source = params.get("source") or {}
    options = params.get("options") or {}
    tracks = params.get("tracks") or options.get("tracks") or []
    return {
        "project": params.get("project", ""),
        "created_at": params.get("created_at", ""),
        "app_version": params.get("app_version", ""),
        "legacy": bool(params.get("legacy", False)),
        "file_name": source.get("file_name", ""),
        "duration": source.get("duration", 0),
        "key": source.get("key", ""),
        "tracks": len(tracks),
        "voice": options.get("voice", ""),
        "font_size": options.get("subtitle_video_font_size", 8),
    }


__all__ = [
    "MAX_IMPORT_BYTES",
    "build_project_params",
    "load_params_bytes",
    "load_params_file",
    "params_from_manifest",
    "params_summary",
    "write_project_params",
]
