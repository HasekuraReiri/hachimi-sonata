"""GPU 探测与显存预算（CUDA + PyTorch，可选）。

重要：探测本身**不导入 torch**，改用系统 ``nvidia-smi``。否则仅打开网页 /
启动 CPU 渲染就会加载 PyTorch 和 CUDA 运行时，白白占用数百 MB 到 1GB 内存。
真正走 GPU 混音时，``hachimi.render`` 才导入 torch。

原则
----
* 只在检测到可用 NVIDIA GPU 时启用；
* **绝不**吃满显卡：默认预算 2GB，且不超过总显存的 1/4；
* WORLD 合成（pyworld）本身是 CPU 库，无法搬到 GPU；GPU 用于音符合成后的
  批量混音落位。
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)

_CACHE: dict | None = None


def _nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    for candidate in (
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "System32", "nvidia-smi.exe"),
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def gpu_info() -> dict:
    """返回 GPU 信息（不可用时 available=False），不加载 torch。"""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    info = {
        "available": False,
        "name": "",
        "total_mb": 0,
        "free_mb": 0,
        "torch": "",
        "error": "",
    }
    exe = _nvidia_smi()
    if not exe:
        info["error"] = "未找到 nvidia-smi"
        _CACHE = info
        return info
    try:
        raw = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=4,
            check=True,
        ).stdout.strip().splitlines()
        if not raw:
            raise RuntimeError("nvidia-smi 没有返回 GPU")
        name, total, free = [part.strip() for part in raw[0].split(",")[:3]]
        info.update(
            available=True,
            name=name,
            total_mb=int(float(total)),
            free_mb=int(float(free)),
        )
    except Exception as exc:  # pragma: no cover
        info["error"] = str(exc)
    _CACHE = info
    return info


def gpu_enabled(mode: str = "auto", memory_mb: int = 2048) -> bool:
    """根据模式判断是否启用 GPU（不导入 torch）。"""
    mode = (mode or "auto").lower()
    if mode == "off":
        return False
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        return False
    info = gpu_info()
    if not info.get("available"):
        return False
    # 显存预算不能超过总显存的一半（8G 卡默认 2G）
    budget = min(int(memory_mb or 2048), max(512, info["total_mb"] // 2))
    return budget >= 256 and info.get("free_mb", 0) >= 256


def gpu_budget_bytes(memory_mb: int = 2048) -> int:
    info = gpu_info()
    total = info.get("total_mb") or 0
    budget_mb = int(memory_mb or 2048)
    if total:
        budget_mb = min(budget_mb, max(512, total // 4))
    return max(128, budget_mb) * 1024 * 1024


__all__ = ["gpu_budget_bytes", "gpu_enabled", "gpu_info"]
