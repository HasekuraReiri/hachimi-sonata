#!/usr/bin/env python
"""图形化 Python 启动器（tkinter，零第三方依赖）。

功能
----
* 一键启动网页后端并自动打开浏览器；
* 显示 / 复制访问地址，选择端口，查看运行日志；
* 检测 Python 依赖状态；
* 支持无 GUI 模式：``python launcher.py --no-gui [--port 9000]``。
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hachimi.browser import open_url as open_browser_url
from hachimi.browser import open_when_ready
from hachimi.config import (
    APP_NAME,
    APP_NAME_CN,
    APP_VERSION,
    HOST,
    PORT,
    ensure_dirs,
    local_url,
    public_url,
)
from hachimi.server import APP, find_free_port, quiet_windows_proactor_noise

logger = logging.getLogger("launcher")

REQUIRED = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("soundfile", "soundfile"),
    ("pyworld", "pyworld"),
    ("pretty_midi", "pretty_midi"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("tqdm", "tqdm"),
]
OPTIONAL = [("matplotlib", "matplotlib"), ("networkx", "networkx")]


def check_dependencies() -> tuple[list[str], list[str]]:
    missing, optional_missing = [], []
    for label, module in REQUIRED:
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(label)
    for label, module in OPTIONAL:
        try:
            importlib.import_module(module)
        except Exception:
            optional_missing.append(label)
    return missing, optional_missing


class QueueLogHandler(logging.Handler):
    def __init__(self, q: "queue.Queue[str]") -> None:
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.q.put_nowait(self.format(record))
        except Exception:
            pass


class ServerController:
    """在后台线程中运行 uvicorn，可随时停止。"""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.server = None
        self.thread: threading.Thread | None = None
        self.url = local_url(port, host)
        self.display_url = public_url(port, host)

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self) -> None:
        import uvicorn

        quiet_windows_proactor_noise()
        config = uvicorn.Config(
            APP,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=False,
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, name="uvicorn", daemon=True)
        self.thread.start()

    def wait_ready(self, timeout: float = 30.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._probe():
                return True
            time.sleep(0.2)
        return False

    def _probe(self) -> bool:
        host = self.host if self.host not in ("0.0.0.0", "::", "localhost") else "127.0.0.1"
        try:
            with urllib.request.urlopen(f"http://{host}:{self.port}/", timeout=0.8) as r:
                return int(getattr(r, "status", 200)) == 200
        except Exception:
            return False

    def stop(self) -> None:
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout=5)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------


def run_gui(default_port: int | None = None, open_browser: bool = True) -> int:
    import tkinter as tk
    from tkinter import messagebox, scrolledtext, ttk

    log_queue: "queue.Queue[str]" = queue.Queue()
    ui_queue: "queue.Queue" = queue.Queue()

    def post(fn) -> None:
        """把 UI 更新排到主线程执行（tkinter 不是线程安全的）。"""
        ui_queue.put(fn)
    handler = QueueLogHandler(log_queue)
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    for name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.addHandler(handler)
        lg.setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    root = tk.Tk()
    root.title(f"{APP_NAME} · 启动器")
    root.geometry("820x600")
    root.minsize(720, 520)
    root.configure(bg="#14121f")
    # 任务栏 / 窗口图标
    icon_ico = ROOT / "HachimiSonata.ico"
    try:
        if icon_ico.exists():
            root.iconbitmap(default=str(icon_ico))
    except Exception:
        pass
    try:
        icon_png = ROOT / "HachimiSonata.png"
        if icon_png.exists():
            _icon_img = tk.PhotoImage(file=str(icon_png))
            root.iconphoto(True, _icon_img)
            root._icon_img = _icon_img  # type: ignore[attr-defined]
    except Exception:
        pass

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("TFrame", background="#14121f")
    style.configure("TLabel", background="#14121f", foreground="#e8e2f7", font=("Microsoft YaHei UI", 10))
    style.configure("Muted.TLabel", foreground="#9a8fbb", font=("Microsoft YaHei UI", 9))
    style.configure("Title.TLabel", font=("Microsoft YaHei UI", 18, "bold"), foreground="#ffffff")
    style.configure("Accent.TButton", font=("Microsoft YaHei UI", 11, "bold"), padding=8)
    style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=6)
    style.configure("TEntry", padding=4)

    controller = ServerController(HOST, default_port or find_free_port(HOST, PORT))

    # ---- 顶部 ----
    header = ttk.Frame(root, padding=(22, 18, 22, 6))
    header.pack(fill="x")
    ttk.Label(header, text=f"{APP_NAME}  v{APP_VERSION}", style="Title.TLabel").pack(anchor="w")
    ttk.Label(
        header,
        text=f"{APP_NAME_CN} · MIDI / MusicXML → 哈基米歌声 · 速度 / 力度 / 奏法 · 实时字幕",
        style="Muted.TLabel",
    ).pack(anchor="w", pady=(2, 0))

    if open_browser and os.environ.get("HACHIMI_NO_BROWSER", "").lower() not in ("1", "true", "yes"):
        auto_open = tk.BooleanVar(value=True)
    else:
        auto_open = tk.BooleanVar(value=False)

    body = ttk.Frame(root, padding=(22, 8))
    body.pack(fill="both", expand=True)

    # ---- 状态卡片 ----
    status_frame = tk.Frame(body, bg="#1d1930", highlightbackground="#3a3157", highlightthickness=1)
    status_frame.pack(fill="x", pady=(4, 12))
    status_inner = tk.Frame(status_frame, bg="#1d1930", padx=16, pady=12)
    status_inner.pack(fill="x")

    url_var = tk.StringVar(value="服务未启动")
    url_label = tk.Label(
        status_inner, textvariable=url_var, bg="#1d1930", fg="#c9b6ff",
        font=("Consolas", 12, "bold"), anchor="w",
    )
    url_label.pack(fill="x")
    url_label.configure(cursor="hand2")
    url_label.bind("<Button-1>", lambda _e: _open_url(controller.url) if controller.running else None)

    deps_missing, opt_missing = check_dependencies()
    dep_text = "依赖检查：全部就绪" if not deps_missing else f"缺少依赖：{', '.join(deps_missing)}"
    dep_color = "#7ee787" if not deps_missing else "#ff7b72"
    dep_label = tk.Label(
        status_inner, text=dep_text, bg="#1d1930", fg=dep_color,
        font=("Microsoft YaHei UI", 9), anchor="w",
    )
    dep_label.pack(fill="x", pady=(4, 0))

    if opt_missing:
        tk.Label(
            status_inner, text=f"可选依赖缺失（不影响主流程）：{', '.join(opt_missing)}",
            bg="#1d1930", fg="#d2a8ff", font=("Microsoft YaHei UI", 9), anchor="w",
        ).pack(fill="x", pady=(2, 0))

    # ---- 控制区 ----
    controls = ttk.Frame(body)
    controls.pack(fill="x", pady=(0, 10))

    ttk.Label(controls, text="端口").pack(side="left")
    port_var = tk.StringVar(value=str(controller.port))
    port_entry = ttk.Entry(controls, textvariable=port_var, width=8)
    port_entry.pack(side="left", padx=(6, 14))

    ttk.Checkbutton(controls, text="启动后打开浏览器", variable=auto_open).pack(side="left", padx=(0, 14))

    start_btn = ttk.Button(controls, text="启动网页界面", style="Accent.TButton")
    start_btn.pack(side="left", padx=(0, 8))
    open_btn = ttk.Button(controls, text="打开网页", state="disabled")
    open_btn.pack(side="left", padx=(0, 8))
    copy_btn = ttk.Button(controls, text="复制地址", state="disabled")
    copy_btn.pack(side="left", padx=(0, 8))
    stop_btn = ttk.Button(controls, text="停止服务", state="disabled")
    stop_btn.pack(side="left", padx=(0, 8))
    folder_btn = ttk.Button(controls, text="输出目录")
    folder_btn.pack(side="left", padx=(0, 8))
    cli_btn = ttk.Button(controls, text="命令行生成")
    cli_btn.pack(side="left")

    # ---- 说明 ----
    tips = tk.Label(
        body,
        text=(
            "使用流程：启动网页界面 → 拖拽 MIDI 或 MusicXML → 勾选要参与生成的轨道 → 选择语音模式 → 开始生成。\n"
            "生成完成后可在网页内试听、查看实时字幕，并下载 WAV / SRT / LRC / HTML 字幕文档。\n"
            "也可以直接双击 start.bat，或在命令行运行：python main.py"
        ),
        bg="#14121f", fg="#8f83ae", justify="left", font=("Microsoft YaHei UI", 9),
    )
    tips.pack(fill="x", pady=(0, 10))

    # ---- 日志 ----
    log_box = scrolledtext.ScrolledText(
        body, height=14, bg="#0e0c16", fg="#c8c0dd", insertbackground="#c8c0dd",
        font=("Consolas", 9), relief="flat", borderwidth=0,
    )
    log_box.pack(fill="both", expand=True)
    log_box.configure(state="disabled")

    def append_log(text: str) -> None:
        log_box.configure(state="normal")
        log_box.insert("end", text + "\n")
        log_box.see("end")
        log_box.configure(state="disabled")

    def drain_logs() -> None:
        while True:
            try:
                append_log(log_queue.get_nowait())
            except queue.Empty:
                break
        while True:
            try:
                fn = ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except Exception:
                pass
        root.after(250, drain_logs)

    root.after(250, drain_logs)

    # ---- 行为 ----
    def set_running(running: bool) -> None:
        start_btn.configure(state="disabled" if running else "normal")
        stop_btn.configure(state="normal" if running else "disabled")
        open_btn.configure(state="normal" if running else "disabled")
        copy_btn.configure(state="normal" if running else "disabled")
        port_entry.configure(state="disabled" if running else "normal")

    def start_server() -> None:
        if controller.running:
            return
        # Tk 变量只能在主线程读取；先取快照，后台线程不再直接访问 tkinter
        should_open = bool(auto_open.get())
        try:
            port = int(port_var.get())
        except ValueError:
            messagebox.showerror("端口错误", "请输入有效端口号")
            return
        controller.port = port
        controller.url = local_url(port, controller.host)
        controller.display_url = public_url(port, controller.host)
        url_var.set(f"正在启动…  {controller.url}")
        append_log(f"启动服务：{controller.url}")
        append_log(f"品牌域名（可选）：{controller.display_url}")
        try:
            controller.start()
        except Exception as exc:
            url_var.set("启动失败")
            append_log(f"启动失败：{exc}")
            messagebox.showerror("启动失败", str(exc))
            return

        def on_ready() -> None:
            ok = controller.wait_ready(timeout=30.0)
            if ok:
                def _ready_ui() -> None:
                    url_var.set(f"{controller.url}   （域名：{controller.display_url}）")
                    try:
                        root.clipboard_clear()
                        root.clipboard_append(controller.url)
                    except Exception:
                        pass

                post(_ready_ui)
                post(lambda: append_log("服务已就绪；本机地址已复制到剪贴板"))
                print(f"[启动器] 服务已就绪：{controller.url}，自动打开={should_open}", flush=True)
                post(lambda: set_running(True))
                if should_open:
                    # 必须回到 tkinter 主线程再打开浏览器：
                    # 后台线程读 Tk 变量、弹 messagebox 都不安全
                    post(lambda: _open_url(controller.url))
            else:
                post(lambda: url_var.set("启动超时，请查看日志"))
                post(lambda: set_running(False))
                post(lambda: append_log("启动超时：端口可能被占用，或防火墙阻止了本机访问"))
                post(lambda: messagebox.showerror(
                    "网页启动失败",
                    f"无法访问 {controller.url}\n\n"
                    "请查看启动器日志；也可换一个端口后点击「启动网页界面」重试。",
                ))

        threading.Thread(target=on_ready, daemon=True).start()

    def _open_url(url: str) -> None:
        ok, message = open_browser_url(url)
        text = ("已打开浏览器：" if ok else "无法打开浏览器：") + message
        append_log(text)
        try:
            print(f"[浏览器] {text}", flush=True)
        except Exception:
            pass
        if not ok:
            messagebox.showerror("无法打开浏览器", f"{url}\n\n{message}")

    def stop_server() -> None:
        controller.stop()
        url_var.set("服务已停止")
        append_log("服务已停止")
        set_running(False)

    def open_ui() -> None:
        if controller.running:
            _open_url(controller.url)

    def copy_url() -> None:
        root.clipboard_clear()
        root.clipboard_append(controller.url)
        append_log("地址已复制到剪贴板")

    def open_output() -> None:
        ensure_dirs()
        from hachimi.config import OUTPUT_DIR

        os.startfile(OUTPUT_DIR)  # type: ignore[attr-defined]

    def open_cli() -> None:
        try:
            import subprocess

            flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
            subprocess.Popen(
                [sys.executable, str(ROOT / "main.py")],
                cwd=str(ROOT),
                creationflags=flags,
            )
        except Exception as exc:
            messagebox.showerror("无法打开", str(exc))

    start_btn.configure(command=start_server)
    stop_btn.configure(command=stop_server)
    open_btn.configure(command=open_ui)
    copy_btn.configure(command=copy_url)
    folder_btn.configure(command=open_output)
    cli_btn.configure(command=open_cli)

    def on_close() -> None:
        controller.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    append_log("一键启动器已就绪，正在自动启动本地服务…")

    # 双击启动器即自动启动服务并打开网页；无需再点按钮
    root.after(450, start_server)

    root.mainloop()
    return 0


def launch(port: int | None = None, host: str | None = None, open_browser: bool = True) -> int:
    if host:
        raise SystemExit("图形启动器暂不支持自定义 host，请使用 python main.py serve --host")
    try:
        import tkinter  # noqa: F401
    except Exception:
        print("未安装 tkinter，改用无界面模式启动服务。")
        return run_headless(port or PORT, open_browser)

    missing, _ = check_dependencies()
    if missing:
        print("缺少依赖：" + ", ".join(missing))
        print("请先运行：pip install -r requirements.txt")
        try:
            import tkinter as tk
            from tkinter import messagebox

            r = tk.Tk()
            r.withdraw()
            messagebox.showerror(
                "缺少依赖",
                "缺少以下依赖：\n" + "\n".join(missing) + "\n\n请运行：pip install -r requirements.txt",
            )
            r.destroy()
        except Exception:
            pass
    return run_gui(default_port=port, open_browser=open_browser)


def run_headless(port: int | None = None, open_browser: bool = True) -> int:
    import uvicorn

    quiet_windows_proactor_noise()
    ensure_dirs()
    host = HOST
    port = port or find_free_port(host, PORT)
    url = local_url(port, host)
    print(f"本机网页：{url}")
    print(f"品牌域名：{public_url(port, host)}")
    if open_browser and os.environ.get("HACHIMI_NO_BROWSER", "").lower() not in ("1", "true", "yes"):
        def _open() -> None:
            ok, message = open_when_ready(url, timeout=30.0)
            print(("[浏览器] " if ok else "[浏览器失败] ") + message, flush=True)

        threading.Thread(target=_open, name="open-browser", daemon=True).start()
    uvicorn.run(APP, host=host, port=port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    # pythonw 没有控制台：把 stdout/stderr 写入日志，避免 GUI 静默退出
    if sys.stderr is None or sys.stdout is None:
        try:
            _log = open(ROOT / "launcher_error.log", "a", encoding="utf-8", buffering=1)
            sys.stdout = _log
            sys.stderr = _log
        except Exception:
            pass
    # 保证 Windows 控制台 / 日志中的中文正常显示
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass

    parser = argparse.ArgumentParser(description=f"{APP_NAME} · {APP_NAME_CN} 启动器")
    parser.add_argument("--no-gui", action="store_true", help="不打开 GUI，直接启动服务")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check", action="store_true", help="只检查依赖")
    args = parser.parse_args(argv)

    if args.check:
        missing, opt_missing = check_dependencies()
        print("Python:", sys.version.split()[0], sys.executable)
        print("必需依赖:", "OK" if not missing else "缺少 " + ", ".join(missing))
        print("可选依赖:", "OK" if not opt_missing else "缺少 " + ", ".join(opt_missing))
        try:
            from hachimi.gpu import gpu_info

            info = gpu_info()
            print(
                "GPU:",
                f"{info['name']} / {info['total_mb']} MB（可用 {info['free_mb']} MB）"
                if info.get("available")
                else "不可用（将使用 CPU）",
            )
        except Exception as exc:
            print("GPU: 检测失败", exc)
        return 0 if not missing else 1

    if args.no_gui:
        return run_headless(args.port, not args.no_browser)
    return launch(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    sys.exit(main())
