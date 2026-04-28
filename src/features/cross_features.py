"""Wide & Deep 交叉特征候选生成函数。

本模块提供字符串拼接级别的交叉特征，用于 Wide 侧的线性变换。
注意：当前只生成原始字符串拼接候选，不做复杂编码或哈希。
"""

from __future__ import annotations

from typing import Any

import pandas as pd


# =============================================================================
# 交叉特征生成
# =============================================================================


def build_cross_feature_pair(
    df: pd.DataFrame,
    col_left: str,
    col_right: str,
    out_name: str = None,
    sep: str = "_x_",
) -> pd.DataFrame:
    """生成两列的字符串交叉特征。

    Args:
        df: 输入 DataFrame
        col_left: 左列名
        col_right: 右列名
        out_name: 输出列名，默认 f'{col_left}{sep}{col_right}'
        sep: 拼接分隔符

    Returns:
        新增列的 DataFrame（原地修改）
    """
    out_name = out_name or f"{col_left}{sep}{col_right}"
    # 将两列转为字符串，NaN 转为 "NA"
    left_str = df[col_left].fillna("NA").astype(str)
    right_str = df[col_right].fillna("NA").astype(str)
    df[out_name] = left_str + sep + right_str
    return df


def build_cross_features(
    df: pd.DataFrame,
    cross_configs: list[dict[str, Any]],
) -> pd.DataFrame:
    """根据交叉特征配置列表批量生成交叉特征。

    Args:
        df: 输入 DataFrame
        cross_configs: 每项包含 name / left / right / (sep)

    Returns:
        新增交叉特征列的 DataFrame（原地修改）
    """
    for cfg in cross_configs:
        out_name = cfg.get("name")
        left = cfg["left"]
        right = cfg["right"]
        sep = cfg.get("sep", "_x_")

        # 检查所需列是否存在
        if left not in df.columns:
            continue
        if right not in df.columns:
            continue

        build_cross_feature_pair(
            df, col_left=left, col_right=right,
            out_name=out_name, sep=sep,
        )
    return df