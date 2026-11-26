"""图形启动器（实现位于项目根目录 ``launcher.py``，这里提供包内入口）。

这样既保证 ``python launcher.py`` / ``start.bat`` 可用，也支持::

    python main.py launch
    python -m hachimi.main launch
    from hachimi.launcher import launch
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_root_launcher() -> Any:
    root = Path(__file__).resolve().parents[1]
    entry = root / "launcher.py"
    spec = importlib.util.spec_from_file_location("_hachimi_root_launcher", entry)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"找不到启动器：{entry}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def launch(port: int | None = None, host: str | None = None, open_browser: bool = True) -> int:
    """启动图形界面（无 tkinter 时自动退化为无界面服务）。"""
    return int(_load_root_launcher().launch(port=port, host=host, open_browser=open_browser))


def main(argv: list[str] | None = None) -> int:
    return int(_load_root_launcher().main(argv))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
