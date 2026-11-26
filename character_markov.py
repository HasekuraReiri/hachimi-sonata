"""兼容旧导入：``from character_markov import train, alphabet``。

新代码请使用 :mod:`hachimi.markov` / :mod:`hachimi.base`。
"""

from hachimi.config import ALPHABET as alphabet  # noqa: F401
from hachimi.markov import MarkovModel, generate, train  # noqa: F401

data_path = __import__("hachimi.config", fromlist=["DATA_PATH"]).DATA_PATH

__all__ = ["alphabet", "train", "generate", "MarkovModel", "data_path"]
