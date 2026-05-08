"""A/B 模拟分组与指标统计函数。

提供分组分配（hash/random）、各组指标计算、组间 lift 计算等核心逻辑。
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from evaluation.metrics import compute_classification_metrics
from evaluation.ranking_metrics import compute_precision_at_k, compute_recall_at_k


def assign_groups_hash(
    df: pd.DataFrame,
    group_key: str,
    treatment_ratio: float = 0.5,
    group_labels: dict[str, str] | None = None,
) -> pd.Series:
    """基于 group_key 的稳定 hash 分配 A/B 组。

    Args:
        df: 输入 DataFrame。
        group_key: 用于 hash 的字段名。
        treatment_ratio: 实验组比例（默认 0.5）。
        group_labels: 组标签映射，如 {'control': 'A', 'treatment': 'B'}。

    Returns:
        group_role Series: 每行对应的组角色（control / treatment）。
    """
    if group_labels is None:
        group_labels = {"control": "A", "treatment": "B"}

    if group_key not in df.columns:
        raise ValueError(f"group_key '{group_key}' 不在 DataFrame 列中: {list(df.columns)}")

    unique_keys = df[group_key].unique()
    key_to_group: dict[Any, str] = {}

    for key in unique_keys:
        h = hashlib.md5(str(key).encode("utf-8")).hexdigest()
        hash_val = int(h, 16) / (2**128)
        key_to_group[key] = "treatment" if hash_val < treatment_ratio else "control"

    return df[group_key].map(key_to_group)


def assign_groups_random(
    df: pd.DataFrame,
    treatment_ratio: float = 0.5,
    random_seed: int = 2026,
) -> pd.Series:
    """基于随机种子分配 A/B 组。

    Args:
        df: 输入 DataFrame。
        treatment_ratio: 实验组比例（默认 0.5）。
        random_seed: 随机种子。

    Returns:
        group_role Series: control / treatment。
    """
    rng = np.random.default_rng(random_seed)
    n = len(df)
    indices = rng.permutation(n)
    split_idx = int(n * treatment_ratio)
    groups = np.array(["control"] * n)
    groups[indices[:split_idx]] = "treatment"
    return pd.Series(groups, index=df.index)


def compute_group_metrics(
    df: pd.DataFrame,
    group_role: str,
    k_values: list[int] | None = None,
    score_col: str = "score",
    label_col: str = "label",
    pred_col: str = "pred",
) -> dict[str, Any]:
    """计算单个 A/B 组的指标。

    Args:
        df: 该组的样本 DataFrame（至少包含 score, label, pred）。
        group_role: 组角色名称（control / treatment）。
        k_values: Top-K K 值列表。
        score_col, label_col, pred_col: 列名。

    Returns:
        指标字典。
    """
    if k_values is None:
        k_values = [5, 10, 20]

    y_true = df[label_col].values.astype(float)
    y_score = df[score_col].values.astype(float)
    y_pred = df[pred_col].values.astype(int) if pred_col in df.columns else (y_score >= 0.5).astype(int)

    result: dict[str, Any] = {
        "group": group_role,
        "group_role": group_role,
        "sample_count": len(df),
        "positive_count": int(y_true.sum()),
        "negative_count": len(df) - int(y_true.sum()),
        "label_positive_rate": float(y_true.mean()),
        "score_mean": float(y_score.mean()),
        "score_std": float(y_score.std()),
        "score_median": float(np.median(y_score)),
        "score_min": float(y_score.min()),
        "score_max": float(y_score.max()),
        "pred_positive_count": int(y_pred.sum()),
        "pred_positive_rate": float(y_pred.mean()),
    }

    # 分类指标
    cls_metrics, cls_warnings = compute_classification_metrics(y_true, y_score, y_pred)
    for k in ("auc", "accuracy", "precision", "recall", "f1"):
        result[k] = cls_metrics.get(k)

    # Top-K 指标（对组内样本重新排序）
    pk_metrics, pk_warnings = compute_precision_at_k(y_true, y_score, k_values)
    rk_metrics, rk_warnings = compute_recall_at_k(y_true, y_score, k_values)
    for k in k_values:
        effective_k = min(k, len(df))
        result[f"precision_at_{k}"] = pk_metrics.get(f"precision_at_{k}")
        result[f"recall_at_{k}"] = rk_metrics.get(f"recall_at_{k}")

    # 收集 warnings
    all_warnings = cls_warnings + pk_warnings + rk_warnings
    if all_warnings:
        result["warnings"] = all_warnings

    return result


def compute_lift(
    treatment_metrics: dict[str, Any],
    control_metrics: dict[str, Any],
) -> dict[str, Any]:
    """计算 treatment 相对于 control 的 lift。

    Args:
        treatment_metrics: 实验组指标字典。
        control_metrics: 对照组指标字典。

    Returns:
        lift 字典，包含绝对差和相对 lift。
    """
    lift: dict[str, Any] = {}

    # 绝对差: treatment - control
    for key in ("score_mean", "label_positive_rate", "pred_positive_rate"):
        c_val = control_metrics.get(key)
        t_val = treatment_metrics.get(key)
        if c_val is not None and t_val is not None and c_val != 0:
            lift[f"treatment_minus_control_{key}"] = t_val - c_val
        elif c_val is not None and t_val is not None:
            lift[f"treatment_minus_control_{key}"] = t_val - c_val
        else:
            lift[f"treatment_minus_control_{key}"] = None

    # 相对 lift: (treatment - control) / control * 100%
    for key in ("score_mean", "label_positive_rate", "pred_positive_rate"):
        c_val = control_metrics.get(key)
        t_val = treatment_metrics.get(key)
        diff = None
        if c_val is not None and t_val is not None:
            diff = t_val - c_val
        if diff is not None and c_val is not None and c_val != 0:
            lift[f"relative_lift_{key}"] = diff / c_val * 100.0
        else:
            lift[f"relative_lift_{key}"] = None

    # 绝对差: treatment - control for AUC/Precision/Recall/F1
    for key in ("auc", "precision", "recall", "f1"):
        c_val = control_metrics.get(key)
        t_val = treatment_metrics.get(key)
        if c_val is not None and t_val is not None:
            lift[f"treatment_minus_control_{key}"] = t_val - c_val
        else:
            lift[f"treatment_minus_control_{key}"] = None

    return lift


def compute_score_distribution(
    df: pd.DataFrame,
    group_role: str,
    score_col: str = "score",
    label_col: str = "label",
    pred_col: str = "pred",
) -> dict[str, Any]:
    """计算单组的分数分布统计。

    Args:
        df: 该组的样本 DataFrame。
        group_role: 组角色名称。
        score_col, label_col, pred_col: 列名。

    Returns:
        分布统计字典。
    """
    scores = df[score_col].values.astype(float)
    labels = df[label_col].values.astype(float)
    preds = df[pred_col].values.astype(int) if pred_col in df.columns else (scores >= 0.5).astype(int)

    return {
        "group": group_role,
        "group_role": group_role,
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_median": float(np.median(scores)),
        "label_positive_rate": float(labels.mean()),
        "pred_positive_rate": float(preds.mean()),
    }