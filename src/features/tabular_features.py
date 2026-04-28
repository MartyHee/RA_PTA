"""表格模型可复用的特征构建函数。

本模块提供：
1. 文本统计特征（长度、词数）
2. 聚合特征（计数、均值、拼接等）
3. 时间戳特征提取
4. 持续桶化（duration bucket）
5. 缺失值统计辅助
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# 文本统计特征
# =============================================================================


def build_text_stat_features(
    df: pd.DataFrame, text_col: str, out_len: str = None, out_wc: str = None
) -> pd.DataFrame:
    """从文本列生成长度和词数特征。

    Args:
        df: 输入 DataFrame
        text_col: 文本列名
        out_len: 输出长度列名，默认 f'{text_col}_length'
        out_wc: 输出词数列名，默认 f'{text_col}_word_count'

    Returns:
        新增列的 DataFrame（原地修改）
    """
    out_len = out_len or f"{text_col}_length"
    out_wc = out_wc or f"{text_col}_word_count"

    # 将 NaN 转为空字符串
    text_series = df[text_col].fillna("").astype(str)

    df[out_len] = text_series.str.len()
    # 非空格字符数（对中文文本更有意义）
    df[out_wc] = text_series.str.replace(" ", "", regex=False).str.len()
    return df


# =============================================================================
# 聚合特征
# =============================================================================


def aggregate_count(
    df: pd.DataFrame, group_col: str, count_col: str, out_name: str
) -> pd.DataFrame:
    """按 group_col 分组，统计 count_col 的非空行数。"""
    agg = df.groupby(group_col)[count_col].count().reset_index(name=out_name)
    return agg


def aggregate_max(
    df: pd.DataFrame, group_col: str, value_col: str, out_name: str
) -> pd.DataFrame:
    """按 group_col 分组，计算 value_col 的最大值。"""
    agg = df.groupby(group_col)[value_col].max().reset_index(name=out_name)
    return agg


def aggregate_mean_str_len(
    df: pd.DataFrame, group_col: str, text_col: str, out_name: str
) -> pd.DataFrame:
    """按 group_col 分组，计算 text_col 的字符串平均长度。"""
    str_len = df[text_col].fillna("").astype(str).str.len()
    temp = df[[group_col]].copy()
    temp["_str_len"] = str_len
    agg = temp.groupby(group_col)["_str_len"].mean().reset_index(name=out_name)
    return agg


def aggregate_join_unique(
    df: pd.DataFrame, group_col: str, text_col: str, out_name: str, sep: str = " | "
) -> pd.DataFrame:
    """按 group_col 分组，将 text_col 的去重值用 sep 拼接。"""
    agg = (
        df.groupby(group_col)[text_col]
        .apply(lambda x: sep.join(x.dropna().astype(str).unique()))
        .reset_index(name=out_name)
    )
    return agg


def aggregate_join_all(
    df: pd.DataFrame, group_col: str, text_col: str, out_name: str, sep: str = " | "
) -> pd.DataFrame:
    """按 group_col 分组，将所有 text_col 值用 sep 拼接（不去重）。"""
    agg = (
        df.groupby(group_col)[text_col]
        .apply(lambda x: sep.join(x.dropna().astype(str)))
        .reset_index(name=out_name)
    )
    return agg


def build_aggregated_features(
    df_source: pd.DataFrame,
    group_col: str,
    agg_configs: list[dict[str, Any]],
) -> pd.DataFrame:
    """根据聚合配置列表，批量生成聚合特征表。

    Args:
        df_source: 源 DataFrame（如 raw_hashtag）
        group_col: 分组列名（如 video_id）
        agg_configs: 每项包含 name / method / column / (sep)

    Returns:
        聚合后的 DataFrame，列为 [group_col, ...agg_features]
    """
    results = {}
    for cfg in agg_configs:
        method = cfg["method"]
        col = cfg["column"]
        out = cfg["name"]
        sep = cfg.get("sep", " | ")

        if method == "count":
            result = aggregate_count(df_source, group_col, col, out)
        elif method == "max":
            result = aggregate_max(df_source, group_col, col, out)
        elif method == "mean_str_len":
            result = aggregate_mean_str_len(df_source, group_col, col, out)
        elif method == "join_unique":
            result = aggregate_join_unique(df_source, group_col, col, out, sep)
        elif method == "join_all":
            result = aggregate_join_all(df_source, group_col, col, out, sep)
        else:
            continue
        results[out] = result

    # 从第一个聚合结果开始 merge
    first = True
    merged = None
    for out_name, result_df in results.items():
        if first:
            merged = result_df
            first = False
        else:
            merged = merged.merge(result_df, on=group_col, how="outer")

    if merged is None:
        merged = pd.DataFrame({group_col: []})

    return merged


# =============================================================================
# 时间戳特征
# =============================================================================


def extract_timestamp_features(
    df: pd.DataFrame, ts_col: str, prefix: str = None
) -> pd.DataFrame:
    """从 Unix 秒级时间戳提取时间特征。

    Args:
        df: 输入 DataFrame
        ts_col: 时间戳列名
        prefix: 输出列名前缀，默认 f'{ts_col}_'

    Returns:
        新增列的 DataFrame（原地修改）
    """
    prefix = prefix or f"{ts_col}_"

    # 只对非空值操作
    valid = df[ts_col].notna()
    ts = pd.to_datetime(df.loc[valid, ts_col], unit="s", errors="coerce")

    df.loc[valid, f"{prefix}hour"] = ts.dt.hour
    df.loc[valid, f"{prefix}day_of_week"] = ts.dt.dayofweek
    df.loc[valid, f"{prefix}month"] = ts.dt.month

    # 原始时间戳也保留为数值特征（归一化前可能有用）
    # 不额外创建新列，直接保留原始 ts_col

    return df


# =============================================================================
# Duration 桶化
# =============================================================================


def build_duration_bucket(
    df: pd.DataFrame,
    duration_col: str = "duration_ms",
    out_col: str = "duration_bucket",
    bins: list[int] = None,
    labels: list[str] = None,
) -> pd.DataFrame:
    """将视频时长（毫秒）划分为离散桶。

    默认桶划分（毫秒）：
        - short:        < 60s (60000ms)
        - medium:       60s ~ 5min (60000 ~ 300000ms)
        - long:         5min ~ 10min (300000 ~ 600000ms)
        - very_long:    >= 10min (600000ms)
    """
    if bins is None:
        bins = [0, 60_000, 300_000, 600_000, float("inf")]
    if labels is None:
        labels = ["short", "medium", "long", "very_long"]

    df[out_col] = pd.cut(
        df[duration_col].fillna(0),
        bins=bins,
        labels=labels,
        right=False,
    ).astype(str)
    df.loc[df[duration_col].isna(), out_col] = "unknown"
    return df


# =============================================================================
# 缺失值统计
# =============================================================================


def compute_missing_summary(df: pd.DataFrame) -> dict[str, Any]:
    """计算 DataFrame 每列的缺失值统计。"""
    summary = {}
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_rate = round(missing_count / len(df), 4) if len(df) > 0 else 0.0
        summary[col] = {
            "missing_count": missing_count,
            "missing_rate": missing_rate,
        }
    return summary