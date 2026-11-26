"""兼容旧导入：音频后处理效果。

新代码请使用 :mod:`hachimi.audio`。
"""

from hachimi.audio import (  # noqa: F401
    apply_post_chain,
    cathedral_reverb,
    fade_edges,
    hf_damping,
    limit_peak,
    multitap_delay,
    normalize_rms,
    peak_normalize,
    soft_normalize,
    stereo_spread,
    vocal_low_cut,
)

__all__ = [
    "apply_post_chain",
    "cathedral_reverb",
    "fade_edges",
    "hf_damping",
    "limit_peak",
    "multitap_delay",
    "normalize_rms",
    "peak_normalize",
    "soft_normalize",
    "stereo_spread",
    "vocal_low_cut",
]
