"""FastAPI 网页后端：上传/分析 MIDI 或 MusicXML、启动渲染、SSE 实时进度与字幕、下载结果。

启动方式::

    python main.py web            # 自动选择端口并打开浏览器
    python main.py serve          # 只启动服务
    python launcher.py            # 图形化启动器

设计要点
--------
* **零额外依赖上传**：前端直接用 ``fetch(body=file)`` 发送原始字节，
  服务端读 ``request.body()``，无需 python-multipart。
* **任务线程 + SSE**：渲染在后台线程执行，进度/日志/字幕通过
  ``asyncio.Queue`` 实时推送给浏览器；同时保留轮询接口兜底。
* 会话（上传的 MIDI + 分析结果）与任务（渲染结果）都带 TTL 清理。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import socket
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from hachimi.analysis import AnalysisResult, analyze
from hachimi.arrangement import build_section_plans
from hachimi.config import (
    APP_NAME,
    APP_NAME_CN,
    APP_SLUG,
    APP_VERSION,
    AUTO_MIX,
    DEFAULT_MARKOV_RATIO,
    DEFAULT_SOURCE_WEIGHTS,
    DEFAULT_VOICE,
    DIR_FALLBACKS,
    GPU_MEMORY_MB,
    HOST,
    JOB_TTL_SECONDS,
    MAX_UPLOAD_MB,
    MODEL_KINDS,
    MODEL_LABELS,
    MUSIC_EXTS,
    OUTPUT_DIR,
    PORT,
    PROJECT_ROOT,
    SAMPLE_SET,
    SAMPLE_SETS,
    SOURCE_DIR,
    TEMP_DIR,
    UPLOAD_DIR,
    VOICE_LABELS,
    VOICE_MODES,
    WEB_DIR,
    RenderOptions,
    ensure_dirs,
    local_url,
    public_url,
)
from hachimi.gpu import gpu_info
from hachimi.params import (
    MAX_IMPORT_BYTES,
    load_params_bytes,
    load_params_file,
    params_from_manifest,
    params_summary,
)
from hachimi.pipeline import RenderCancelled, render_midi

logger = logging.getLogger(__name__)


class _WindowsProactorResetFilter(logging.Filter):
    """屏蔽 Windows Proactor 在客户端断开后的已知无害收尾异常。

    Python 3.10 的 ``_ProactorBasePipeTransport._call_connection_lost`` 会在
    对端已 RST 的 socket 上再次 ``shutdown``，抛 WinError 10054 并由 asyncio
    以 ERROR 打印。任务此时已经完成，实际输出不受影响；这里只过滤这一种
    精确的日志，其他 asyncio / 渲染错误仍然正常显示。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not record.name.startswith("asyncio"):
            return True
        exc = record.exc_info[1] if record.exc_info else None
        if not isinstance(exc, ConnectionResetError):
            return True
        if getattr(exc, "winerror", None) != 10054:
            return True
        return "_call_connection_lost" not in record.getMessage()


_PROACTOR_RESET_FILTER = _WindowsProactorResetFilter()


def quiet_windows_proactor_noise() -> None:
    """在 Windows 上安装上面的窄过滤器；重复调用不会叠加。"""
    if os.name != "nt":
        return
    target = logging.getLogger("asyncio")
    if not any(isinstance(item, _WindowsProactorResetFilter) for item in target.filters):
        target.addFilter(_PROACTOR_RESET_FILTER)


APP = FastAPI(title=f"{APP_NAME} · {APP_NAME_CN}", version=APP_VERSION)


# ---------------------------------------------------------------------------
# 会话 / 任务
# ---------------------------------------------------------------------------


@dataclass
class Session:
    id: str
    path: str
    filename: str
    analysis: AnalysisResult
    created: float = field(default_factory=time.time)


@dataclass
class Job:
    id: str
    session_id: str
    loop: asyncio.AbstractEventLoop
    status: str = "queued"  # queued / running / done / error / cancelled
    pct: float = 0.0
    stage: str = ""
    message: str = ""
    result: dict[str, Any] | None = None
    error: str = ""
    created: float = field(default_factory=time.time)
    cancel: bool = False
    subtitles: list[dict[str, Any]] = field(default_factory=list)
    queues: list[asyncio.Queue] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)

    # ---- 事件 ----
    def emit(self, event: dict[str, Any]) -> None:
        etype = event.get("type")
        if etype == "stage":
            self.pct = float(event.get("pct", self.pct))
            self.stage = str(event.get("stage", self.stage))
            self.message = str(event.get("message", self.message))
            if self.status == "queued":
                self.status = "running"
        elif etype == "log":
            msg = str(event.get("message", ""))
            self.log_lines.append(msg)
            self.log_lines = self.log_lines[-200:]
        elif etype == "subtitle":
            self.subtitles = event.get("lines") or []
        elif etype == "done":
            self.status = "done"
            self.result = event.get("result")
            self.pct = 100.0
        elif etype == "error":
            self.status = "error"
            self.error = str(event.get("message", "未知错误"))
        elif etype == "cancelled":
            self.status = "cancelled"

        for q in list(self.queues):
            try:
                self.loop.call_soon_threadsafe(q.put_nowait, event)
            except RuntimeError:  # 事件循环已关闭
                pass

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self.queues.remove(q)
        except ValueError:
            pass

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": "snapshot",
            "job_id": self.id,
            "status": self.status,
            "pct": round(self.pct, 1),
            "stage": self.stage,
            "message": self.message,
            "error": self.error,
            "log": self.log_lines[-60:],
            "subtitles": self.subtitles,
            "result": self.result,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "session_id": self.session_id,
            "status": self.status,
            "pct": round(self.pct, 1),
            "stage": self.stage,
            "message": self.message,
            "error": self.error,
            "result": self.result,
            "log": self.log_lines[-60:],
            "subtitles": self.subtitles,
        }


SESSIONS: dict[str, Session] = {}
JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()


def _prune() -> None:
    now = time.time()
    with _LOCK:
        for sid, s in list(SESSIONS.items()):
            if now - s.created > JOB_TTL_SECONDS:
                SESSIONS.pop(sid, None)
        for jid, j in list(JOBS.items()):
            if now - j.created > JOB_TTL_SECONDS and j.status in ("done", "error", "cancelled"):
                JOBS.pop(jid, None)


def _safe_filename(name: str) -> str:
    name = os.path.basename(name or "upload.mid")
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    if not name.lower().endswith(MUSIC_EXTS):
        name += ".mid"
    return name[:120]


# ---------------------------------------------------------------------------
# 静态页面
# ---------------------------------------------------------------------------


@APP.get("/")
def index():
    path = os.path.join(WEB_DIR, "index.html")
    if not os.path.exists(path):
        return JSONResponse({"error": "web/index.html 不存在"}, status_code=500)
    return FileResponse(
        path,
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@APP.get("/favicon.ico")
def favicon():
    root_icon = os.path.join(PROJECT_ROOT, "HachimiSonata.ico")
    if os.path.exists(root_icon):
        return FileResponse(root_icon, media_type="image/x-icon")
    for name, media in (("favicon.ico", "image/x-icon"), ("favicon.svg", "image/svg+xml")):
        path = os.path.join(WEB_DIR, name)
        if os.path.exists(path):
            return FileResponse(path, media_type=media)
    return JSONResponse({}, status_code=204)


APP.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@APP.get("/api/config")
def api_config():
    ensure_dirs()
    return {
        "app": APP_NAME,
        "app_cn": APP_NAME_CN,
        "slug": APP_SLUG,
        "version": APP_VERSION,
        "domain": public_url(PORT).split("://", 1)[-1].rsplit(":", 1)[0],
        "voices": [{"value": k, "label": VOICE_LABELS[k]} for k in VOICE_MODES],
        "sample_sets": SAMPLE_SETS,
        "models": [{"value": k, "label": MODEL_LABELS.get(k, k)} for k in MODEL_KINDS],
        "music_formats": list(MUSIC_EXTS),
        "defaults": {
            "voice": DEFAULT_VOICE,
            "model": "markov",
            "sample_set": SAMPLE_SET,
            "velocity_gamma": 1.6,
            "workers": 0,
            "normalize": "peak",
            "auto_mix": AUTO_MIX,
            "decomposed_merge": True,
            "syntax_weight": 0.3,
            "markov_ratio": DEFAULT_MARKOV_RATIO,
            "source_weights": DEFAULT_SOURCE_WEIGHTS,
            "gpu": "auto",
            "gpu_memory_mb": GPU_MEMORY_MB,
            "write_subtitle_videos": True,
            "write_premix": True,
            "write_bad_note_midis": True,
            "subtitle_video_width": 1920,
            "subtitle_video_fps": 30,
            "subtitle_video_font_family": "SimSun",
            "subtitle_video_font_size": 8,
            "subtitle_video_text_color": "#FFFFFF",
            "subtitle_video_text_stroke_width": 2,
        },
        "gpu": gpu_info(),
        "audio_format": "mp3",
        "syntax_weight": 0.3,
        "markov_ratio": DEFAULT_MARKOV_RATIO,
        "source_weights": DEFAULT_SOURCE_WEIGHTS,
        "gpu_memory_mb": GPU_MEMORY_MB,
        "source_dir": SOURCE_DIR,
        "max_upload_mb": MAX_UPLOAD_MB,
        "paths": {
            "output": OUTPUT_DIR,
            "uploads": UPLOAD_DIR,
            "temp": TEMP_DIR,
        },
        "dir_fallbacks": dict(DIR_FALLBACKS),
        "output_fallback": str(Path(OUTPUT_DIR).resolve())
        != str((Path(PROJECT_ROOT) / "output").resolve()),
        "musicxml": {
            "supported": [".xml", ".musicxml", ".mxl"],
            "tempo": "读取全部 sound tempo / metronome，并按谱面位置换算",
            "dynamics": "sound dynamics / dynamics / wedge / cresc. / dim.",
            "articulations": ["staccato", "tenuto", "accent", "marcato", "tremolo"],
            "ties": "合并跨小节连音线",
        },
    }


@APP.get("/api/samples")
def api_samples():
    """返回本地音源目录、文件数量与当前签名。"""
    from hachimi.synthesis import bank_signature, available_sample_chars

    return {
        "source_dir": SOURCE_DIR,
        "signature": bank_signature(),
        "files": len(available_sample_chars()),
        "hint": "修改/替换 音源 目录中的 wav 后，点击刷新或直接重新生成即可生效",
    }


@APP.post("/api/samples/refresh")
def api_samples_refresh():
    """手动刷新本地音源（清除内存单位缓存；磁盘预处理按文件签名自动失效）。"""
    from hachimi.render import clear_unit_cache
    from hachimi.synthesis import available_sample_chars, bank_signature, refresh_sample_bank

    refresh_sample_bank(force=True)
    clear_unit_cache()
    return {
        "ok": True,
        "source_dir": SOURCE_DIR,
        "signature": bank_signature(),
        "files": len(available_sample_chars()),
    }


# ---------------------------------------------------------------------------
# 分析
# ---------------------------------------------------------------------------


def _analysis_overview(result: AnalysisResult, buckets: int = 480) -> dict[str, Any]:
    """压缩出用于网页 canvas 的旋律轮廓 / 密度数据。"""
    if not result.notes:
        return {"pitch": [], "density": []}
    duration = max(1.0, result.duration)
    bin_size = duration / buckets
    pitch_sum = [0.0] * buckets
    count = [0] * buckets
    for n in result.notes:
        idx = min(buckets - 1, int(n.start / bin_size))
        pitch_sum[idx] += n.pitch
        count[idx] += 1
    pitch = [
        round(pitch_sum[i] / count[i], 2) if count[i] else None
        for i in range(buckets)
    ]
    density = [count[i] for i in range(buckets)]
    return {"bin": round(bin_size, 4), "pitch": pitch, "density": density}


def _session_payload(session: Session) -> dict[str, Any]:
    analysis = session.analysis
    return {
        "session_id": session.id,
        "filename": session.filename,
        "path": session.path,
        "analysis": analysis.to_dict(),
        "overview": _analysis_overview(analysis),
        "section_plans": build_section_plans(
            analysis.sections,
            base_markov=DEFAULT_MARKOV_RATIO,
            base_seed=1337,
            duration=analysis.duration,
        ),
        "default_selected": analysis.selected_track_indices(),
    }


@APP.post("/api/analyze")
async def api_analyze(request: Request, name: str = "upload.mid"):
    _prune()
    ensure_dirs()
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="空文件")
    if len(body) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"文件超过 {MAX_UPLOAD_MB} MB")

    filename = _safe_filename(name)
    sid = uuid.uuid4().hex
    path = os.path.join(UPLOAD_DIR, f"{sid}_{filename}")
    try:
        with open(path, "wb") as f:
            f.write(body)
    except OSError as exc:
        # 只读 / Low integrity 安装目录下回退到临时目录，网页仍可正常分析
        logger.warning("uploads 目录不可写，改用临时目录：%s", exc)
        fallback_path = ""
        try:
            os.makedirs(TEMP_DIR, exist_ok=True)
            fallback_path = os.path.join(TEMP_DIR, f"hachimi_upload_{sid}_{filename}")
            with open(fallback_path, "wb") as f:
                f.write(body)
        except OSError as exc2:
            logger.warning("临时目录仍不可写，改用系统临时目录：%s", exc2)
            tmp_dir = tempfile.mkdtemp(prefix="hachimi_upload_")
            fallback_path = os.path.join(tmp_dir, filename)
            with open(fallback_path, "wb") as f:
                f.write(body)
        path = fallback_path

    try:
        analysis = analyze(path)
    except Exception as exc:
        try:
            os.remove(path)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail=f"文件解析失败：{exc}") from exc

    if not analysis.tracks or not analysis.notes:
        try:
            os.remove(path)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail="该文件没有可用音符")

    session = Session(id=sid, path=path, filename=filename, analysis=analysis)
    with _LOCK:
        SESSIONS[sid] = session
    logger.info("分析完成：%s (%d 音符)", filename, len(analysis.notes))
    return _session_payload(session)


@APP.post("/api/analyze-path")
async def api_analyze_path(request: Request):
    """分析服务器本地已有的 MIDI / MusicXML（方便直接使用项目曲库）。"""
    _prune()
    data = await request.json()
    path = os.path.abspath(str(data.get("path", "")))
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="文件不存在")
    try:
        analysis = analyze(path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"文件解析失败：{exc}") from exc
    sid = uuid.uuid4().hex
    session = Session(id=sid, path=path, filename=os.path.basename(path), analysis=analysis)
    with _LOCK:
        SESSIONS[sid] = session
    return _session_payload(session)


@APP.get("/api/midi/list")
def api_midi_list(limit: int = 200):
    from hachimi.cli import iter_midi_files

    files = iter_midi_files()
    return {
        "count": len(files),
        "files": [
            {
                "path": p,
                "name": os.path.basename(p),
                "size": os.path.getsize(p),
                "rel": os.path.relpath(p),
            }
            for p in files[:limit]
        ],
    }


# ---------------------------------------------------------------------------
# 渲染任务
# ---------------------------------------------------------------------------


def _start_job(session: Session, options: RenderOptions) -> Job:
    loop = asyncio.get_running_loop()
    job = Job(id=uuid.uuid4().hex, session_id=session.id, loop=loop)
    with _LOCK:
        JOBS[job.id] = job

    def run() -> None:
        job.status = "running"
        job.emit({"type": "stage", "stage": "start", "pct": 0.5, "message": "任务已启动"})
        project = os.path.splitext(session.filename)[0] or "hachimi"
        try:
            result = render_midi(
                session.path,
                options,
                project_name=project,
                progress_cb=job.emit,
                cancel_cb=lambda: job.cancel,
            )
            payload = result.to_dict()
            # 转成前端可直接使用的下载信息
            files: dict[str, Any] = {}
            for key, value in payload["files"].items():
                if isinstance(value, dict):
                    files[key] = {k: _file_info(v, result.output_dir) for k, v in value.items()}
                else:
                    files[key] = _file_info(value, result.output_dir)
            payload["files"] = files
            payload["audio_url"] = f"/api/jobs/{job.id}/audio"
            payload["duration"] = round(result.duration, 2)
            job.emit({"type": "done", "result": payload})
        except RenderCancelled:
            job.emit({"type": "cancelled", "message": "任务已取消"})
        except Exception as exc:  # pragma: no cover - 后台线程
            logger.exception("渲染任务失败")
            job.emit({"type": "error", "message": str(exc)})

    thread = threading.Thread(target=run, name=f"render-{job.id[:8]}", daemon=True)
    thread.start()
    return job


def _file_info(path: str, base: str) -> dict[str, Any]:
    from urllib.parse import quote

    name = os.path.basename(path)
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    return {
        "name": name,
        "path": path,
        "size": size,
        "url": f"/api/download?path={quote(path)}",
    }


@APP.post("/api/render")
async def api_render(request: Request):
    _prune()
    data = await request.json()
    sid = str(data.get("session_id", ""))
    with _LOCK:
        session = SESSIONS.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="会话已过期，请重新上传 MIDI / 乐谱")

    raw_options = data.get("options") or {}
    options = RenderOptions.from_dict(raw_options)
    options.tracks = list(data.get("tracks") or raw_options.get("tracks") or [])
    if not options.tracks:
        raise HTTPException(status_code=400, detail="请至少勾选一个轨道")
    if not any(bool(t.get("selected")) for t in options.tracks):
        raise HTTPException(status_code=400, detail="请至少勾选一个轨道")
    if options.model_kind not in MODEL_KINDS:
        options.model_kind = "markov"
    if options.voice not in VOICE_MODES:
        options.voice = DEFAULT_VOICE
    if options.max_seconds is not None and options.max_seconds < 1:
        options.max_seconds = None

    job = _start_job(session, options)
    return {"job_id": job.id, "status": job.status}


@APP.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    with _LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job.to_dict()


@APP.get("/api/jobs/{job_id}/events")
async def api_job_events(job_id: str):
    with _LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    async def stream():
        q = job.subscribe()
        try:
            yield _sse(job.snapshot())
            if job.status in ("done", "error", "cancelled"):
                yield _sse({"type": "end", "status": job.status})
                return
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=20.0)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                yield _sse(event)
                if event.get("type") in ("done", "error", "cancelled"):
                    break
            yield _sse({"type": "end", "status": job.status})
        finally:
            job.unsubscribe(q)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@APP.post("/api/jobs/{job_id}/cancel")
def api_job_cancel(job_id: str):
    with _LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    job.cancel = True
    return {"ok": True}


@APP.get("/api/jobs/{job_id}/audio")
def api_job_audio(job_id: str):
    with _LOCK:
        job = JOBS.get(job_id)
    if job is None or not job.result:
        raise HTTPException(status_code=404, detail="音频尚未生成")
    path = job.result.get("audio_path")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="音频文件不存在")
    media = "audio/mpeg" if path.lower().endswith(".mp3") else "audio/wav"
    return FileResponse(path, media_type=media, filename=os.path.basename(path))


def _output_bases() -> list[Path]:
    """实际输出目录 + 项目原有 output（回退时仍能访问旧作品）。"""
    bases = [Path(OUTPUT_DIR).resolve()]
    legacy = (Path(PROJECT_ROOT) / "output").resolve()
    if legacy not in bases:
        bases.append(legacy)
    return bases


@APP.get("/api/download")
def api_download(path: str):
    """只允许下载输出目录下的文件，避免任意路径读取。"""
    candidate = Path(path).resolve()
    allowed = False
    for base in _output_bases():
        try:
            candidate.relative_to(base)
            allowed = True
            break
        except ValueError:
            continue
    if not allowed:
        raise HTTPException(status_code=403, detail="仅允许下载输出目录中的文件")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    suffix = candidate.suffix.lower()
    media = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".json": "application/json; charset=utf-8",
        ".srt": "text/plain; charset=utf-8",
        ".lrc": "text/plain; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
        ".html": "text/html; charset=utf-8",
        ".mid": "audio/midi",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".zip": "application/zip",
    }.get(suffix, "application/octet-stream")
    return FileResponse(candidate, media_type=media, filename=candidate.name)


@APP.get("/api/library")
def api_library(limit: int = 60):
    """列出 output/ 下已有作品（按 manifest.json 修改时间倒序）。"""
    ensure_dirs()
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for base in _output_bases():
        if not base.exists():
            continue
        for manifest in base.glob("*/manifest.json"):
            # 自检 / 临时探针目录不属于“最近作品”
            if manifest.parent.name.startswith(("_", ".")):
                continue
            key = str(manifest.resolve())
            if key in seen:
                continue
            seen.add(key)
            try:
                mtime = manifest.stat().st_mtime
                with open(manifest, "r", encoding="utf-8") as f:
                    data = json.load(f)
                items.append(
                    {
                        "project": data.get("project", manifest.parent.name),
                        "dir": str(manifest.parent),
                        "duration": data.get("duration"),
                        "audio": data.get("audio_path"),
                        "audio_url": f"/api/download?path={quote(str(data.get('audio_path')))}",
                        "mtime": mtime,
                    }
                )
            except Exception:
                continue
    items.sort(key=lambda x: x.get("mtime") or 0, reverse=True)
    return {"items": items[:limit]}


# ---------------------------------------------------------------------------
# 工程参数包：读取旧工程 / 导入参数 ZIP
# ---------------------------------------------------------------------------


def _param_path_allowed(path: str) -> Path | None:
    candidate = Path(path).resolve()
    for base in _output_bases():
        try:
            candidate.relative_to(base)
            return candidate
        except ValueError:
            continue
    return None


@APP.get("/api/params/list")
def api_params_list(limit: int = 80):
    """列出已有作品的参数包；旧作品没有参数文件时回退到 manifest。"""
    ensure_dirs()
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for base in _output_bases():
        if not base.exists():
            continue
        for manifest in base.glob("*/manifest.json"):
            if manifest.parent.name.startswith(("_", ".")):
                continue
            folder = manifest.parent
            key = str(folder.resolve())
            if key in seen:
                continue
            seen.add(key)
            param_jsons = sorted(folder.glob("*_params.json"))
            param_zips = sorted(folder.glob("*_params.zip"))
            param_json = param_jsons[0] if param_jsons else None
            param_zip = param_zips[0] if param_zips else None
            path = param_json or param_zip or manifest
            summary: dict[str, Any] = {}
            try:
                if param_json is not None and param_json.exists():
                    summary = params_summary(load_params_file(str(param_json)))
                elif param_zip is not None and param_zip.exists():
                    summary = params_summary(load_params_file(str(param_zip)))
                else:
                    with open(manifest, "r", encoding="utf-8") as fh:
                        manifest_data = json.load(fh)
                    summary = params_summary(params_from_manifest(manifest_data, source=str(manifest)))
                summary["legacy_only"] = path == manifest
                summary["path"] = str(path)
                summary["dir"] = str(folder)
                summary["mtime"] = path.stat().st_mtime
                summary["url"] = f"/api/download?path={quote(str(path))}"
                summary["read_url"] = f"/api/params/read?path={quote(str(path))}"
                items.append(summary)
            except Exception:
                continue
    items.sort(key=lambda x: x.get("mtime") or 0, reverse=True)
    return {"items": items[: max(1, min(500, limit))]}


@APP.get("/api/params/read")
def api_params_read(path: str):
    candidate = _param_path_allowed(path)
    if candidate is None or not candidate.is_file():
        raise HTTPException(status_code=404, detail="参数包不存在或不在输出目录中")
    try:
        if candidate.name == "manifest.json":
            with open(candidate, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            params = params_from_manifest(manifest, source=str(candidate))
        else:
            params = load_params_file(str(candidate))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"读取参数包失败：{exc}") from exc
    return {"ok": True, "source": candidate.name, "params": params}


@APP.post("/api/params/import")
async def api_params_import(request: Request, name: str = "params.zip"):
    raw = await request.body()
    if len(raw) > MAX_IMPORT_BYTES:
        raise HTTPException(status_code=413, detail="参数包超过 32 MB")
    try:
        params = load_params_bytes(raw, name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "source": name, "params": params}


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------


def find_free_port(host: str = HOST, start: int = PORT) -> int:
    for port in range(start, start + 60):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def run_server(
    host: str | None = None,
    port: int | None = None,
    open_browser: bool = True,
    log_level: str = "info",
) -> int:
    import uvicorn

    quiet_windows_proactor_noise()
    ensure_dirs()
    host = host or HOST
    port = port or find_free_port(host)
    url = local_url(port, host)
    domain_url = public_url(port, host)

    print("=" * 64)
    print(f"  {APP_NAME} · {APP_NAME_CN}  v{APP_VERSION}")
    print(f"  本机地址：{url}")
    print(f"  品牌域名：{domain_url}（浏览器支持时可用）")
    print("  Ctrl+C 停止服务")
    print("=" * 64)

    if open_browser and os.environ.get("HACHIMI_NO_BROWSER", "").lower() not in ("1", "true", "yes"):
        from hachimi.browser import open_when_ready

        def _open_browser() -> None:
            ok, message = open_when_ready(url, timeout=30.0)
            print(("[浏览器] " if ok else "[浏览器失败] ") + message, flush=True)

        threading.Thread(target=_open_browser, name="open-browser", daemon=True).start()

    uvicorn.run(APP, host=host, port=port, log_level=log_level)
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(run_server())
