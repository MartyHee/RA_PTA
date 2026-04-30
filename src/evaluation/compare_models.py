"""模型对比逻辑：读取预测、校验质量、汇总指标、Top-K 重校验、分数分布"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── 路径设置 ──────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
sys.path.insert(0, os.path.join(_project_root, "src"))

from evaluation.ranking_metrics import (  # noqa: E402
    compute_precision_at_k,
    compute_recall_at_k,
)


def check_predictions_quality(
    df: pd.DataFrame,
    required_cols: list[str],
    model_name: str,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    """检查 predictions.csv 质量，返回检查结果字典。"""
    result: dict[str, Any] = {
        "file_exists": True,
        "model_name": model_name,
        "expected_run_id": expected_run_id,
        "n_rows": len(df),
        "required_cols_present": True,
        "missing_required_cols": [],
        "score_in_range": True,
        "score_min": None,
        "score_max": None,
        "pred_is_01": True,
        "label_is_01": True,
        "split_is_eval": True,
        "split_values": [],
        "model_name_match": True,
        "model_name_values": [],
        "run_id_match": True,
        "run_id_values": [],
        "has_nan": False,
        "has_inf": False,
        "duplicate_sample_id": 0,
        "duplicate_video_id": 0,
        "warnings": [],
        "errors": [],
    }

    # required cols
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        result["required_cols_present"] = False
        result["missing_required_cols"] = missing
        result["errors"].append(f"缺少必需列: {missing}")
        return result

    # score range
    if "score" in df.columns:
        score_col = pd.to_numeric(df["score"], errors="coerce")
        result["score_min"] = float(score_col.min()) if not score_col.isna().all() else None
        result["score_max"] = float(score_col.max()) if not score_col.isna().all() else None
        if result["score_min"] is not None and (result["score_min"] < 0 or result["score_max"] > 1):
            result["score_in_range"] = False
            result["warnings"].append(
                f"score 范围 [{result['score_min']:.4f}, {result['score_max']:.4f}]，部分超出 [0,1]"
            )

    # pred 0/1
    if "pred" in df.columns:
        uniq = df["pred"].dropna().unique()
        if not set(uniq).issubset({0, 1}):
            result["pred_is_01"] = False
            result["warnings"].append(f"pred 取值 {sorted(uniq)}，不是纯 0/1")

    # label 0/1
    if "label" in df.columns:
        uniq = df["label"].dropna().unique()
        if not set(uniq).issubset({0.0, 1.0, 0, 1}):
            result["label_is_01"] = False
            result["warnings"].append(f"label 取值 {sorted(uniq)}，不是纯 0/1")

    # split
    if "split" in df.columns:
        result["split_values"] = df["split"].dropna().unique().tolist()
        if set(result["split_values"]) != {"eval"}:
            result["split_is_eval"] = False
            result["warnings"].append(f"split 值 {result['split_values']}，不完全是 eval")

    # model_name consistency
    if "model_name" in df.columns:
        result["model_name_values"] = df["model_name"].dropna().unique().tolist()
        if model_name not in result["model_name_values"]:
            result["model_name_match"] = False
            result["warnings"].append(
                f"model_name 列值 {result['model_name_values']} 与期望 {model_name} 不一致"
            )

    # run_id (compare as strings to handle int/str type mismatch)
    if "run_id" in df.columns and expected_run_id:
        run_id_vals = df["run_id"].dropna().astype(str).unique().tolist()
        result["run_id_values"] = run_id_vals
        if str(expected_run_id) not in run_id_vals:
            result["run_id_match"] = False
            result["warnings"].append(
                f"run_id 列值 {run_id_vals} 与期望 {expected_run_id} 不一致"
            )

    # NaN / inf
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            result["has_nan"] = True
            result["warnings"].append(f"列 {col} 包含 NaN")
        if np.isinf(df[col]).any():
            result["has_inf"] = True
            result["warnings"].append(f"列 {col} 包含 inf")

    # duplicates
    if "sample_id" in df.columns:
        result["duplicate_sample_id"] = int(df["sample_id"].duplicated().sum())
    if "video_id" in df.columns:
        result["duplicate_video_id"] = int(df["video_id"].duplicated().sum())

    return result


def collect_model_metrics(
    model_key: str,
    model_cfg: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    """读取单个模型的 metrics.json 和 run_meta.json，合并返回。"""
    model_name = model_cfg["model_name"]
    run_id = model_cfg["run_id"]
    output_dir = project_root / model_cfg["output_dir"]

    summary: dict[str, Any] = {
        "model_name": model_name,
        "run_id": run_id,
        "output_dir": str(output_dir),
    }

    # 读 metrics.json
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        summary["error"] = f"metrics.json 不存在: {metrics_path}"
        return summary

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    summary["sample_count"] = metrics.get("sample_count")
    summary["positive_count"] = metrics.get("positive_count")
    summary["negative_count"] = metrics.get("negative_count")

    # 分类指标
    for k in ("auc", "accuracy", "precision", "recall", "f1"):
        summary[k] = metrics.get(k)

    # 展开 precision_at_k / recall_at_k
    pk = metrics.get("precision_at_k", {})
    rk = metrics.get("recall_at_k", {})
    for k, v in pk.items():
        summary[k] = v
    for k, v in rk.items():
        summary[k] = v

    summary["eval_loss"] = metrics.get("eval_loss")
    summary["threshold"] = metrics.get("threshold")
    summary["warnings"] = metrics.get("warnings", [])

    # 读 run_meta.json
    meta_path = output_dir / "run_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        summary["best_epoch"] = meta.get("best_epoch")
        summary["best_eval_loss"] = meta.get("best_eval_loss")
        summary["device"] = meta.get("device")
        summary["run_meta_notes"] = meta.get("notes", [])
    else:
        summary["best_epoch"] = None
        summary["best_eval_loss"] = None
        summary["device"] = None

    return summary


def compute_topk_comparison(
    df: pd.DataFrame,
    model_name: str,
    run_id: str,
    k_values: list[int],
) -> list[dict[str, Any]]:
    """基于 predictions 重新计算 Top-K 指标。"""
    y_true = df["label"].values.astype(float)
    y_score = df["score"].values.astype(float)

    pk_metrics, pk_warnings = compute_precision_at_k(y_true, y_score, k_values)
    rk_metrics, rk_warnings = compute_recall_at_k(y_true, y_score, k_values)

    n_pos = int(y_true.sum())
    n_samples = len(y_true)
    all_warnings = pk_warnings + rk_warnings

    rows = []
    for k in k_values:
        effective_k = min(k, n_samples)
        row = {
            "model_name": model_name,
            "run_id": run_id,
            "k": k,
            "effective_k": effective_k,
            "precision_at_k": pk_metrics.get(f"precision_at_{k}"),
            "recall_at_k": rk_metrics.get(f"recall_at_{k}"),
            "topk_positive_count": None,
            "eval_positive_count": n_pos,
            "sample_count": n_samples,
        }
        # compute actual positive count in top-k
        if effective_k > 0:
            sorted_indices = np.argsort(y_score)[::-1]
            sorted_labels = y_true[sorted_indices]
            row["topk_positive_count"] = int(sorted_labels[:effective_k].sum())
        rows.append(row)

    if all_warnings:
        rows[0]["_warnings"] = all_warnings

    return rows


def compute_score_distribution(
    df: pd.DataFrame,
    model_name: str,
    run_id: str,
) -> dict[str, Any]:
    """计算分数分布统计。"""
    scores = df["score"].values.astype(float)
    labels = df["label"].values.astype(float)
    preds = df["pred"].values.astype(int) if "pred" in df.columns else (scores >= 0.5).astype(int)

    result = {
        "model_name": model_name,
        "run_id": run_id,
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_median": float(np.median(scores)),
        "pred_positive_count": int(preds.sum()),
        "pred_positive_rate": float(preds.mean()),
        "label_positive_count": int(labels.sum()),
        "label_positive_rate": float(labels.mean()),
    }
    return result