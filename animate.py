"""兼容旧导入：马尔科夫图与动画。

新代码请使用 :mod:`hachimi.animate`。
"""

from hachimi.animate import (  # noqa: F401
    animate_markov_track,
    draw_markov_graph,
    mux_video_audio,
    slice_events,
)

# 旧名字
draw = draw_markov_graph

__all__ = [
    "animate_markov_track",
    "draw",
    "draw_markov_graph",
    "mux_video_audio",
    "slice_events",
]
