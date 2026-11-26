"""标准句法生成：把音乐槽位填成「哈基米 / 叮咚鸡 / 曼巴」的标准句子。

句法来源
--------
* ``补充/补充.txt`` 要点一（哈基米三种正确句法）
* 要点二（叮咚鸡、曼巴标准句法）

设计要点
--------
* 每个语音有一组“句法单元”（词块）。单个词块内部不允许拆开，保证逗号
  连接的两部分连续；整句允许按词块自由组合 / 省略。
* :func:`generate_phrase` 按词块循环选取，尽量填满目标槽数；剩余 1~2 个
  槽时才用单字补足。
* 主题动机、长音由歌词层强制使用本模块；普通旋律按 ``syntax_weight``
  与马尔科夫链混合。
"""

from __future__ import annotations

import random
from typing import Iterable

from hachimi.config import ALPHABET

#: 每种语音的“句法单元”。顺序即标准句法顺序，可循环使用。
SYNTAX_BANKS: dict[str, list[str]] = {
    # 要点一：哈基米摸，南北绿豆；阿西噶哈，呀库那路。
    # 也允许第三种 2+2 分句（哈基，基米。南北，绿豆……）
    "哈基米": [
        "哈基米摸", "南北绿豆", "阿西噶哈", "呀库那路",
        "哈基", "基米", "南北", "绿豆", "阿西", "噶哈", "呀库", "那路",
    ],
    "哈基米only": ["哈基米", "哈基", "基米"],
    # 要点二：叮咚鸡，大狗叫。楼上的下来搞核酸。
    "叮咚鸡": [
        "叮咚鸡", "大狗叫", "楼上的下来搞核酸",
        "叮咚鸡", "叮咚鸡", "大狗大狗", "叫", "叫", "叫",
    ],
    # 要点二：曼1巴1，曼2巴2。欧嘛吉利，曼3巴3。
    "曼巴": ["曼巴", "曼巴", "欧嘛吉利", "曼巴"],
    # 旧语音保持简单句法
    "曼波only": ["曼波"],
    "曼波": ["曼波", "欧马吉利", "哇夏达不耶"],
}

#: 循环时优先使用的标准句（不含拆分的 2 字块）
CANONICAL_CYCLES: dict[str, list[str]] = {
    "哈基米": ["哈基米摸", "南北绿豆", "阿西噶哈", "呀库那路"],
    "哈基米only": ["哈基米"],
    "叮咚鸡": ["叮咚鸡", "大狗叫", "楼上的下来搞核酸", "叮咚鸡", "大狗大狗", "叫", "叫", "叫"],
    "曼巴": ["曼巴", "曼巴", "欧嘛吉利", "曼巴"],
    "曼波only": ["曼波"],
    "曼波": ["曼波", "欧马吉利", "哇夏达不耶"],
}


def available_voices() -> list[str]:
    return list(SYNTAX_BANKS)


def _valid_chars(voice: str) -> set[str]:
    return set(ALPHABET.get(voice, ""))


def generate_phrase(voice: str, length: int, rng: random.Random | None = None) -> str:
    """生成一个长度为 ``length`` 的标准句法片段。

    词块优先、整块拼接；只有剩余槽位不足一个词块时才拆单字补足。
    """
    if length <= 0:
        return ""
    rng = rng or random
    units = SYNTAX_BANKS.get(voice)
    if not units:
        # 未知语音：从字符集里取
        chars = list(ALPHABET.get("哈基米", "哈基米"))
        return "".join(rng.choice(chars) for _ in range(length))

    cycle = CANONICAL_CYCLES.get(voice) or units
    out: list[str] = []
    total = 0
    start = rng.randrange(len(cycle))
    guard = 0
    while total < length and guard < 128:
        guard += 1
        # 从当前位置向后找可放下的词块，优先选更长的（减少单字补足）
        candidates: list[tuple[int, str]] = []
        for step in range(len(cycle)):
            unit = cycle[(start + step) % len(cycle)]
            if total + len(unit) > length:
                continue
            if out and unit == out[-1] and rng.random() < 0.65:
                continue
            candidates.append((step, unit))
        if candidates:
            longest = max(len(u) for _, u in candidates)
            step, unit = rng.choice([(s, u) for s, u in candidates if len(u) == longest])
            out.append(unit)
            total += len(unit)
            start = (start + step + 1) % len(cycle)
            continue
        # 循环里没有合适长度：从所有词块（含 2 字块）里找能放下的
        fits = [u for u in units if total + len(u) <= length]
        if fits:
            unit = rng.choice(fits)
            out.append(unit)
            total += len(unit)
            continue
        break

    text = "".join(out)
    if len(text) < length:
        chars = [c for c in ALPHABET.get(voice, "哈基米") if c in _valid_chars(voice)]
        if not chars:
            chars = list("哈基米")
        while len(text) < length:
            text += rng.choice(chars)
    return text[:length]


def split_units(voice: str, text: str) -> list[str]:
    """把生成文本按句法词块尽量切分（用于测试 / 展示）。"""
    units = sorted(SYNTAX_BANKS.get(voice, []), key=len, reverse=True)
    out: list[str] = []
    i = 0
    while i < len(text):
        for unit in units:
            if text.startswith(unit, i):
                out.append(unit)
                i += len(unit)
                break
        else:
            out.append(text[i])
            i += 1
    return out


def phrase_preview(voice: str, length: int, seed: int = 1) -> str:
    return generate_phrase(voice, length, random.Random(seed))


__all__ = [
    "CANONICAL_CYCLES",
    "SYNTAX_BANKS",
    "available_voices",
    "generate_phrase",
    "phrase_preview",
    "split_units",
]
