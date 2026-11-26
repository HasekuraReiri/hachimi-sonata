"""段落级编曲计划：马尔科夫比例、随机种子与音源配比。

网页会把每个段落的计划保存在 ``RenderOptions.section_plans`` 中；CLI / 旧调用
不传时，本模块会按段落能量、调性、重复关系自动生成一套可复现的计划。

设计原则
--------
* **默认 70% 马尔科夫链**：``markov_ratio=0.70``，对应标准句法 30%；
* **哈基米绝对主导**：自动配比中哈基米始终占多数（默认约 76%），
  叮咚鸡是主要辅助音色，曼巴 / 曼波只占少量色彩；
* **每段可变**：段落比例、种子与配比都由稳定哈希在基线附近轻微变化，
  同一 MIDI + 同一随机种子可复现；
* **手动优先**：网页提交的段落值优先于自动值，手动值不会被后端覆盖。
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

#: 默认马尔科夫链占比（剩余为标准句法）
DEFAULT_MARKOV_RATIO = 0.70
#: 自动配比的基线：哈基米绝对主导，叮咚鸡辅助
DEFAULT_SOURCE_WEIGHTS: dict[str, float] = {
    "哈基米": 0.76,
    "叮咚鸡": 0.16,
    "曼巴": 0.05,
    "曼波only": 0.03,
}
#: 自动模式下哈基米的最低占比（避免随机抖动削弱主导地位）
MIN_HACHIMI_WEIGHT = 0.68
SOURCE_KEYS: tuple[str, ...] = ("哈基米", "叮咚鸡", "曼巴", "曼波only")
SOURCE_LABELS: dict[str, str] = {
    "哈基米": "哈基米",
    "叮咚鸡": "叮咚鸡",
    "曼巴": "曼巴",
    "曼波only": "曼波",
}
MARKOV_RANGE = (0.0, 1.0)
#: 自动计划的范围：允许段落之间形成明显对比，而不是所有段都贴着 70%。
#: 主题 / 长音仍强制标准句法，因此这里的自由度只影响普通旋律句。
AUTO_MARKOV_RANGE = (0.38, 0.94)
#: 段落对比强度：哈希抖动 + 能量 + 织体 + 小节位置的组合。
SECTION_CONTRAST = 0.18
#: 自动段至少偏离基线这么多，避免哈希偶然落在 70% 附近导致“听不出区别”。
MIN_SECTION_DELTA = 0.08


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _stable_int(*parts: Any) -> int:
    """跨进程稳定的整数哈希（Python 内置 hash 带随机盐，不能用于复现）。"""
    text = "|".join(str(p) for p in parts)
    digest = hashlib.blake2s(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def normalize_source_weights(
    weights: Mapping[str, Any] | None,
    *,
    fallback: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """把任意音源配比归一化为 ``{哈基米: ..., 叮咚鸡: ...}``。

    ``fallback`` 用于手动配比全部为 0 的情况；只保留四个已知音源。
    """
    clean: dict[str, float] = {}
    source = weights or {}
    aliases = {
        "曼波": "曼波only",
        "曼波only": "曼波only",
        "哈基米only": "哈基米",
    }
    for key, value in source.items():
        name = aliases.get(str(key), str(key))
        if name not in SOURCE_KEYS:
            continue
        clean[name] = max(0.0, _float(value, 0.0))
    if sum(clean.values()) <= 1e-9:
        base = fallback or DEFAULT_SOURCE_WEIGHTS
        clean = {k: max(0.0, _float(base.get(k, 0.0), 0.0)) for k in SOURCE_KEYS}
    total = sum(clean.values())
    if total <= 1e-9:
        clean = dict(DEFAULT_SOURCE_WEIGHTS)
        total = sum(clean.values())
    return {k: clean.get(k, 0.0) / total for k in SOURCE_KEYS}


def _section_value(section: Any, name: str, default: Any = None) -> Any:
    if isinstance(section, Mapping):
        return section.get(name, default)
    return getattr(section, name, default)


def _section_payload(section: Any, index: int, duration: float) -> dict[str, Any]:
    start = _float(_section_value(section, "start", 0.0))
    end = _float(_section_value(section, "end", duration), duration)
    if end <= start:
        end = min(duration, start + 0.25)
    key = _section_value(section, "key", "") or ""
    progression = list(_section_value(section, "progression", []) or [])
    return {
        "section_index": index,
        "label": str(_section_value(section, "label", chr(65 + index % 26)) or chr(65 + index % 26)),
        "start": round(start, 3),
        "end": round(end, 3),
        "key": str(key),
        "energy": round(_float(_section_value(section, "energy", 0.0)), 3),
        "n_notes": int(_float(_section_value(section, "n_notes", 0), 0)),
        "progression": progression[:12],
        "repeat_of": _section_value(section, "repeat_of", None),
        "character": str(_section_value(section, "character", "") or ""),
        "phrase_count": int(_float(_section_value(section, "phrase_count", 0), 0)),
    }


def _auto_markov(
    index: int,
    *,
    base_markov: float,
    base_seed: int,
    payload: Mapping[str, Any],
) -> float:
    jitter = (_stable_int("markov", base_seed, index) % 1601) / 800.0 - 1.0
    energy = _float(payload.get("energy"), 0.5)
    if index == 0:
        # 第一段作为全曲的「默认段」，严格落在给定基线（默认 70%），
        # 后续段落才做可复现的明显对比。
        return round(_clamp(base_markov, *MARKOV_RANGE), 3)

    # 0.18 的哈希抖动 + 能量项 + 织体/重复项，使相邻段落真正听得出差别。
    # 全部由稳定哈希与段落特征决定，同一 MIDI + 种子仍可复现。
    ratio = base_markov + SECTION_CONTRAST * jitter + 0.10 * (0.5 - energy)
    character = str(payload.get("character", "") or "")
    if "minor" in str(payload.get("key", "")).lower():
        # 小调段落和声张力更高，略多保留马尔科夫的自由度。
        ratio += 0.015
    if payload.get("repeat_of"):
        # 再现段换一种句式，避免 A/A′ 听起来完全一样。
        ratio += 0.035
    if any(word in character for word in ("强奏", "激烈", "高潮", "密集")):
        ratio += 0.025
    elif any(word in character for word in ("轻奏", "安静", "留白")):
        ratio -= 0.025
    # 三拍一组的段落级对比，防止哈希值偶然全部挤在基线附近。
    if index % 3 == 0:
        ratio += 0.030
    elif index % 3 == 1:
        ratio -= 0.030
    # 哈希抖动偶尔会刚好抵消能量/织体项；此时强制拉出一个最小对比，
    # 保证自动段不会全部听起来像同一个 70% 基线。
    if abs(ratio - base_markov) < MIN_SECTION_DELTA:
        direction = 1.0 if jitter >= 0 else -1.0
        ratio = base_markov + direction * MIN_SECTION_DELTA
    low = min(AUTO_MARKOV_RANGE[0], base_markov)
    high = max(AUTO_MARKOV_RANGE[1], base_markov)
    return round(_clamp(ratio, low, high), 3)


def _auto_weights(
    index: int,
    *,
    base_seed: int,
    payload: Mapping[str, Any],
) -> dict[str, float]:
    if index == 0:
        return normalize_source_weights(DEFAULT_SOURCE_WEIGHTS)
    jitter = (_stable_int("voices", base_seed, index) % 1001) / 500.0 - 1.0
    energy = _float(payload.get("energy"), 0.5)
    # 音源配比也拉开：哈基米仍在 58%~90% 主导，但不同段落的
    # 叮咚鸡 / 曼巴 / 曼波比例会明显换色，而不是每段都接近默认值。
    hachimi = 0.75 + 0.14 * jitter + 0.08 * (0.5 - energy)
    hachimi = round(_clamp(hachimi, max(0.56, MIN_HACHIMI_WEIGHT - 0.12), 0.90), 4)
    aux = 1.0 - hachimi
    # 辅助声部中叮咚鸡仍是主力，但曼巴 / 曼波可以在一部分段落里更突出。
    dd_share = _clamp(0.62 + 0.20 * jitter + 0.06 * (energy - 0.5), 0.42, 0.82)
    mb_share = _clamp(0.22 - 0.10 * jitter + 0.06 * (0.5 - energy), 0.10, 0.34)
    manbo_share = max(0.02, 1.0 - dd_share - mb_share)
    total_aux = dd_share + mb_share + manbo_share
    return normalize_source_weights(
        {
            "哈基米": hachimi,
            "叮咚鸡": aux * dd_share / total_aux,
            "曼巴": aux * mb_share / total_aux,
            "曼波only": aux * manbo_share / total_aux,
        }
    )


def _auto_seed(index: int, base_seed: int) -> int:
    if index == 0:
        return max(0, min(2_147_483_647, int(base_seed)))
    return int((int(base_seed) + _stable_int("section-seed", base_seed, index) % 900_000) % 2_147_483_647)


def auto_section_plan(
    section: Any,
    index: int,
    *,
    base_markov: float = DEFAULT_MARKOV_RATIO,
    base_seed: int = 1337,
    duration: float = 0.0,
) -> dict[str, Any]:
    """生成单个段落的自动计划（可复现）。"""
    payload = _section_payload(section, index, duration)
    markov_ratio = _auto_markov(
        index,
        base_markov=base_markov,
        base_seed=base_seed,
        payload=payload,
    )
    weights = _auto_weights(index, base_seed=base_seed, payload=payload)
    return {
        **payload,
        "markov_ratio": markov_ratio,
        "syntax_weight": round(1.0 - markov_ratio, 3),
        "seed": _auto_seed(index, base_seed),
        "source_weights": {k: round(v, 4) for k, v in weights.items()},
        # 给网页一个直观的“本段与基线差多少”的指标（不影响生成）。
        "auto_contrast": round(
            abs(markov_ratio - base_markov) + (max(weights.values()) - min(weights.values())) * 0.25,
            3,
        ),
        "auto": True,
        "source": "auto",
    }


def _find_override(overrides: Sequence[Mapping[str, Any]], index: int, label: str) -> Mapping[str, Any] | None:
    for item in overrides:
        if not isinstance(item, Mapping):
            continue
        raw_index = item.get("section_index", item.get("index"))
        if raw_index is not None:
            try:
                if int(raw_index) == int(index):
                    return item
            except (TypeError, ValueError):
                pass
        elif str(item.get("label", "")) == str(label):
            return item
    return None


def _apply_override(plan: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """应用网页提交的段落值。

    网页显示什么就渲染什么：即使 ``auto=True``，只要提交了数值也按数值执行。
    ``auto`` 只用于 UI 标记 / 后续「全局比例重建」，不再作为忽略用户值的开关。
    """
    markov = override.get("markov_ratio")
    if markov is None and override.get("syntax_weight") is not None:
        markov = 1.0 - _float(override.get("syntax_weight"), 0.3)
    if markov is not None:
        plan["markov_ratio"] = round(_clamp(_float(markov, 0.7), *MARKOV_RANGE), 3)
        plan["syntax_weight"] = round(1.0 - plan["markov_ratio"], 3)
    if override.get("seed") is not None:
        try:
            plan["seed"] = max(0, min(2_147_483_647, int(override["seed"])))
        except (TypeError, ValueError):
            pass
    weights = override.get("source_weights") or override.get("weights")
    if not isinstance(weights, Mapping):
        # 也接受扁平字段
        weights = {
            key: override[key]
            for key in SOURCE_KEYS
            if override.get(key) is not None
        }
    if isinstance(weights, Mapping) and weights:
        plan["source_weights"] = {
            k: round(v, 4)
            for k, v in normalize_source_weights(weights, fallback=plan["source_weights"]).items()
        }
    plan["auto"] = bool(override.get("auto", False))
    plan["source"] = str(override.get("source") or ("auto" if plan["auto"] else "manual"))
    return plan


def build_section_plans(
    sections: Iterable[Any],
    *,
    base_markov: float = DEFAULT_MARKOV_RATIO,
    base_seed: int = 1337,
    duration: float = 0.0,
    overrides: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """生成全部段落的编曲计划。

    Args:
        sections: :class:`hachimi.analysis.Section` 或同名字典。
        base_markov: 自动计划使用的马尔科夫基线（默认 0.70）。
        base_seed: 全局随机种子。
        duration: 没有段落时的虚拟段落时长。
        overrides: 网页提交的段落计划。``auto=False`` 的项按手动值覆盖。
    """
    base_markov = _clamp(_float(base_markov, DEFAULT_MARKOV_RATIO), *MARKOV_RANGE)
    base_seed = int(_float(base_seed, 1337))
    items = list(sections)
    if not items:
        items = [
            {
                "index": 0,
                "label": "A",
                "start": 0.0,
                "end": duration if duration > 0 else 0.0,
                "energy": 0.0,
                "n_notes": 0,
            }
        ]
    overrides = list(overrides or [])
    plans: list[dict[str, Any]] = []
    for index, section in enumerate(items):
        plan = auto_section_plan(
            section,
            index,
            base_markov=base_markov,
            base_seed=base_seed,
            duration=duration,
        )
        override = _find_override(overrides, index, plan["label"])
        if override is not None:
            plan = _apply_override(plan, override)
        plans.append(plan)
    return plans


def plan_by_section(plans: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for index, plan in enumerate(plans):
        try:
            key = int(plan.get("section_index", index))
        except (TypeError, ValueError):
            key = index
        out[key] = dict(plan)
    return out


def section_index_at(sections: Sequence[Any], t: float) -> int:
    """返回时间点所在段落索引；找不到时回退到最近段落。"""
    for index, section in enumerate(sections):
        start = _float(_section_value(section, "start", 0.0))
        end = _float(_section_value(section, "end", 0.0))
        if start <= t < end:
            return index
    if not sections:
        return 0
    return min(
        range(len(sections)),
        key=lambda i: abs(_float(_section_value(sections[i], "start", 0.0)) - t),
    )


def plan_summary(plans: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """给 manifest / 网页结果使用的精简摘要。"""
    out = []
    for plan in plans:
        out.append(
            {
                "section": plan.get("label", ""),
                "section_index": plan.get("section_index", 0),
                "start": plan.get("start", 0.0),
                "end": plan.get("end", 0.0),
                "key": plan.get("key", ""),
                "markov_ratio": plan.get("markov_ratio", DEFAULT_MARKOV_RATIO),
                "seed": plan.get("seed", 0),
                "source_weights": plan.get("source_weights", {}),
                "auto": bool(plan.get("auto", True)),
            }
        )
    return out


__all__ = [
    "DEFAULT_MARKOV_RATIO",
    "DEFAULT_SOURCE_WEIGHTS",
    "MIN_HACHIMI_WEIGHT",
    "AUTO_MARKOV_RANGE",
    "MARKOV_RANGE",
    "SOURCE_KEYS",
    "SOURCE_LABELS",
    "auto_section_plan",
    "build_section_plans",
    "normalize_source_weights",
    "plan_by_section",
    "plan_summary",
    "section_index_at",
]
