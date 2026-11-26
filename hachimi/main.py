"""``python -m hachimi.main`` 入口：等价于项目根目录的 ``python main.py``。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    entry = root / "main.py"
    spec = importlib.util.spec_from_file_location("_hachimi_root_main", entry)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"找不到入口脚本：{entry}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.main(argv))


if __name__ == "__main__":
    sys.exit(main())
