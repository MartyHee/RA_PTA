"""排序指标计算"""

import numpy as np


def compute_precision_at_k(y_true, y_score, k_values=None):
    """计算 Precision@K，返回 (metrics_dict, warnings_list)。"""
    if k_values is None:
        k_values = [5, 10, 20]

    metrics = {}
    warnings = []

    n_samples = len(y_true)
    if n_samples == 0:
        for k in k_values:
            metrics[f"precision_at_{k}"] = None
        warnings.append("样本数为 0，无法计算排序指标")
        return metrics, warnings

    sorted_indices = np.argsort(y_score)[::-1]
    sorted_labels = np.array(y_true)[sorted_indices]

    for k in k_values:
        effective_k = min(k, n_samples)
        if effective_k <= 0:
            metrics[f"precision_at_{k}"] = None
            continue
        top_k_labels = sorted_labels[:effective_k]
        metrics[f"precision_at_{k}"] = float(np.mean(top_k_labels))

    return metrics, warnings


def compute_recall_at_k(y_true, y_score, k_values=None):
    """计算 Recall@K，返回 (metrics_dict, warnings_list)。"""
    if k_values is None:
        k_values = [5, 10, 20]

    metrics = {}
    warnings = []

    n_samples = len(y_true)
    n_pos = int(np.sum(y_true))

    if n_samples == 0:
        for k in k_values:
            metrics[f"recall_at_{k}"] = None
        warnings.append("样本数为 0，无法计算排序指标")
        return metrics, warnings

    sorted_indices = np.argsort(y_score)[::-1]
    sorted_labels = np.array(y_true)[sorted_indices]

    for k in k_values:
        effective_k = min(k, n_samples)
        if effective_k <= 0:
            metrics[f"recall_at_{k}"] = None
            continue
        top_k_labels = sorted_labels[:effective_k]
        n_hits = int(np.sum(top_k_labels))
        if n_pos == 0:
            metrics[f"recall_at_{k}"] = None
            if "正样本数为 0，无法计算 Recall@K" not in warnings:
                warnings.append("正样本数为 0，无法计算 Recall@K")
        else:
            metrics[f"recall_at_{k}"] = float(n_hits / n_pos)

    return metrics, warnings