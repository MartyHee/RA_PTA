"""分类指标计算"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(y_true, y_score, y_pred, threshold=0.5):
    """计算分类指标，返回 (metrics_dict, warnings_list)。"""
    metrics = {}
    warnings = []

    # AUC
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
    except Exception as e:
        metrics["auc"] = None
        warnings.append(f"AUC 无法计算: {e}")

    # Accuracy
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    # Precision
    try:
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    except Exception as e:
        metrics["precision"] = None
        warnings.append(f"Precision 无法计算: {e}")

    # Recall
    try:
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    except Exception as e:
        metrics["recall"] = None
        warnings.append(f"Recall 无法计算: {e}")

    # F1
    try:
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    except Exception as e:
        metrics["f1"] = None
        warnings.append(f"F1 无法计算: {e}")

    return metrics, warnings