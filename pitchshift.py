"""兼容旧导入：WORLD 合成相关函数。

新代码请使用 :mod:`hachimi.synthesis`。
"""

from hachimi.synthesis import (  # noqa: F401
    available_sample_chars,
    get_preprocessed,
    preprocess_chars,
    preprocess_sample,
    preproc_cache_path,
    sample_path_for,
    synthesize_note,
    synthesize_note_from_char,
)

__all__ = [
    "available_sample_chars",
    "get_preprocessed",
    "preprocess_chars",
    "preprocess_sample",
    "preproc_cache_path",
    "sample_path_for",
    "synthesize_note",
    "synthesize_note_from_char",
]
