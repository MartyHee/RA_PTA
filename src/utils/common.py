"""通用工具函数：安全解析字符串化 list/JSON 字段"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


def safe_parse_list(value: Any) -> Any:
    """尝试安全解析字符串化 list，如 '[a, b, c]'。

    不破坏原始值，解析失败时返回原值。
    """
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            # 先尝试标准 JSON 解析
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            # 非标准格式（如单引号或无引号），返回原值
            return value
    return value


def safe_parse_json(value: Any) -> Any:
    """尝试安全解析字符串化 JSON，如 '{"key": "val"}'。

    不破坏原始值，解析失败时返回原值。
    """
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def _sample_non_null(df: pd.DataFrame, col: str, max_samples: int = 100) -> pd.Series:
    """对指定列采样非空值，最多取 max_samples 个。"""
    non_null = df[col].dropna()
    if len(non_null) == 0:
        return non_null
    return non_null.sample(n=min(max_samples, len(non_null)), random_state=42)


def detect_list_like_columns(
    df: pd.DataFrame, threshold: float = 0.5, max_samples: int = 100
) -> list[str]:
    """检测 DataFrame 中哪些列的字符串值看起来像字符串化 list。

    判断逻辑：对每列采样非空值，若超过 threshold 比例的样本
    以 '[' 开头且以 ']' 结尾，则标记为 list-like。

    Args:
        df: 输入 DataFrame
        threshold: 判定阈值，默认 0.5
        max_samples: 每列最多采样数

    Returns:
        list-like 列名列表
    """
    list_like = []
    for col in df.columns:
        samples = _sample_non_null(df, col, max_samples)
        if len(samples) == 0:
            continue
        str_samples = samples[samples.apply(lambda x: isinstance(x, str))]
        if len(str_samples) == 0:
            continue
        match_ratio = str_samples.apply(
            lambda x: x.strip().startswith("[") and x.strip().endswith("]")
        ).mean()
        if match_ratio >= threshold:
            list_like.append(col)
    return list_like


def detect_json_like_columns(
    df: pd.DataFrame, threshold: float = 0.5, max_samples: int = 100
) -> list[str]:
    """检测 DataFrame 中哪些列的字符串值看起来像字符串化 JSON 对象。

    判断逻辑：对每列采样非空值，若超过 threshold 比例的样本
    以 '{' 开头且以 '}' 结尾，则标记为 json-like。

    Args:
        df: 输入 DataFrame
        threshold: 判定阈值，默认 0.5
        max_samples: 每列最多采样数

    Returns:
        json-like 列名列表
    """
    json_like = []
    for col in df.columns:
        samples = _sample_non_null(df, col, max_samples)
        if len(samples) == 0:
            continue
        str_samples = samples[samples.apply(lambda x: isinstance(x, str))]
        if len(str_samples) == 0:
            continue
        match_ratio = str_samples.apply(
            lambda x: x.strip().startswith("{") and x.strip().endswith("}")
        ).mean()
        if match_ratio >= threshold:
            json_like.append(col)
    return json_like


def detect_id_columns(df: pd.DataFrame) -> list[str]:
    """检测 DataFrame 中可能为 ID 的列。

    规则：字段名包含 id / video_id / author_id / music_id / hashtag_id /
    comment_id / crawl_batch_id 的列。

    Returns:
        ID 列名列表
    """
    id_keywords = ["video_id", "author_id", "music_id", "hashtag_id",
                   "comment_id", "crawl_batch_id", "group_id", "tag_id",
                   "source_video_id", "related_video_id"]
    detected = []
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in id_keywords):
            detected.append(col)
    return detected


def detect_raw_columns(df: pd.DataFrame) -> list[str]:
    """检测 DataFrame 中 '_raw' 结尾的原始字段列。"""
    return [col for col in df.columns if col.lower().endswith("_raw")]