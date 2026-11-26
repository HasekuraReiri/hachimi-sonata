"""跨平台打开浏览器，Windows 下优先保证 Edge 可用。

启动器与 ``main.py web`` 共用这里的逻辑，避免三处各自调用
``webbrowser.open`` 时行为不一致。Windows 上按以下顺序尝试：

1. 若系统默认浏览器是 Edge，或显式要求优先 Edge：直接启动 ``msedge.exe``；
2. 系统文件关联（``os.startfile``）；
3. ``webbrowser`` 模块；
4. ``cmd /c start``。

每一步失败都会继续下一层，最终返回 ``(是否成功, 说明)`` 供日志记录。
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

logger = logging.getLogger(__name__)

_EDGE_ENV = "HACHIMI_EDGE_PATH"


def _windows() -> bool:
    return sys.platform.startswith("win")


def _registry_value(root: int, path: str, name: str) -> str:
    try:
        import winreg

        with winreg.OpenKey(root, path) as key:
            value, _ = winreg.QueryValueEx(key, name)
            return str(value)
    except Exception:
        return ""


def find_edge() -> str | None:
    """返回本机 Edge 可执行文件路径；找不到返回 ``None``。"""
    override = os.environ.get(_EDGE_ENV, "").strip().strip('"')
    if override and Path(override).is_file():
        return override

    candidates: list[Path] = []
    for env_name in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA"):
        base = os.environ.get(env_name)
        if base:
            candidates.append(Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe")
    candidates.extend(
        [
            Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        ]
    )
    for path in candidates:
        try:
            if path.is_file():
                return str(path)
        except OSError:
            continue

    which = shutil.which("msedge") or shutil.which("msedge.exe")
    if which:
        return which

    if _windows():
        try:
            import winreg

            for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                value = _registry_value(
                    root,
                    r"Software\Microsoft\Windows\CurrentVersion\App Paths\msedge.exe",
                    "",
                )
                if value:
                    value = value.strip().strip('"')
                    if Path(value).is_file():
                        return value
        except Exception:
            pass
    return None


def default_browser_progid() -> str:
    """读取 Windows 默认浏览器 ProgId（读不到返回空串）。"""
    if not _windows():
        return ""
    try:
        import winreg

        for scheme in ("http", "https"):
            value = _registry_value(
                winreg.HKEY_CURRENT_USER,
                rf"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\{scheme}\UserChoice",
                "ProgId",
            )
            if value:
                return value
    except Exception:
        pass
    return ""


def is_edge_default() -> bool:
    progid = default_browser_progid().lower()
    return "edge" in progid or "msedge" in progid


def _launch_edge(url: str, edge: str) -> tuple[bool, str]:
    try:
        subprocess.Popen(
            [edge, url],
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True, f"Microsoft Edge：{edge}"
    except Exception as exc:
        return False, f"启动 Edge 失败：{exc}"


def open_url(url: str, *, prefer_edge: bool | None = None) -> tuple[bool, str]:
    """用系统浏览器打开 URL，Windows 下优先/回退到 Edge。

    Args:
        url: 要打开的地址。
        prefer_edge: ``None`` = 默认浏览器是 Edge 时优先 Edge；
            ``True`` = 总是优先 Edge；``False`` = 先走系统默认关联。
    """
    if prefer_edge is None:
        prefer_edge = is_edge_default()

    edge = find_edge() if _windows() else None
    errors: list[str] = []

    if _windows() and prefer_edge and edge:
        ok, message = _launch_edge(url, edge)
        if ok:
            return True, message
        errors.append(message)

    if _windows():
        try:
            os.startfile(url)  # type: ignore[attr-defined]
            return True, "系统默认浏览器"
        except Exception as exc:
            errors.append(f"系统关联打开失败：{exc}")

    try:
        if webbrowser.open(url, new=2):
            return True, "webbrowser 默认浏览器"
    except Exception as exc:
        errors.append(f"webbrowser 失败：{exc}")

    if _windows() and edge:
        ok, message = _launch_edge(url, edge)
        if ok:
            return True, message
        errors.append(message)

    try:
        subprocess.Popen(
            ["cmd", "/c", "start", "", url],
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True, "cmd start"
    except Exception as exc:
        errors.append(f"cmd start 失败：{exc}")

    return False, "；".join(dict.fromkeys(errors)) or "没有可用的浏览器启动方式"


def wait_until_ready(url: str, timeout: float = 30.0, interval: float = 0.35) -> bool:
    """轮询本机网页，直到服务真正可访问。"""
    # 服务绑定 127.0.0.1；避免某些环境把 localhost 优先解析成 ::1
    parts = urllib.parse.urlsplit(url)
    probe_url = url
    if parts.hostname == "localhost":
        netloc = parts.netloc.replace("localhost", "127.0.0.1", 1)
        probe_url = urllib.parse.urlunsplit(
            (parts.scheme, netloc, parts.path or "/", parts.query, parts.fragment)
        )
    deadline = time.time() + max(1.0, timeout)
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(probe_url, timeout=1.0) as response:
                if int(getattr(response, "status", 200)) == 200:
                    return True
        except Exception:
            pass
        time.sleep(max(0.1, interval))
    return False


def open_when_ready(
    url: str,
    *,
    timeout: float = 30.0,
    prefer_edge: bool | None = None,
) -> tuple[bool, str]:
    """等网页服务就绪后再打开，避免浏览器先打开一个「无法连接」页。"""
    if not wait_until_ready(url, timeout=timeout):
        return False, f"等待服务就绪超时（{timeout:.0f}s）：{url}"
    return open_url(url, prefer_edge=prefer_edge)


__all__ = [
    "default_browser_progid",
    "find_edge",
    "is_edge_default",
    "open_url",
    "open_when_ready",
    "wait_until_ready",
]
