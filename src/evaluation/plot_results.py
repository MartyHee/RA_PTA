"""图表生成模块"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("plot_results")

# matplotlib 是否可用
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use("Agg")  # 非交互后端
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # 尝试设置 CJK 字体（Windows 环境）
    _cjk_font_set = False
    for _font_name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei"]:
        try:
            plt.rcParams["font.sans-serif"] = [_font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            _cjk_font_set = True
            break
        except Exception:
            continue
    if not _cjk_font_set:
        logger.warning("未找到 CJK 字体，图表中文可能显示为方框")

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logger.warning("matplotlib 不可用，图表将跳过")

# 颜色方案
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def _ensure_plt():
    """确保 matplotlib 已导入。"""
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib 不可用，无法生成图表")


def plot_metric_bar(
    summary_df: "pd.DataFrame",
    metric_col: str,
    title: str,
    save_path: Path,
    ylabel: str | None = None,
) -> dict[str, Any]:
    """生成各模型某指标的柱状图。"""
    result: dict[str, Any] = {"metric": metric_col, "save_path": str(save_path), "success": False}
    try:
        _ensure_plt()
        import pandas as pd

        # 转换 metric_df
        if not isinstance(summary_df, pd.DataFrame):
            summary_df = pd.DataFrame(summary_df)

        vals = summary_df[["model_name", metric_col]].dropna()
        if vals.empty:
            result["error"] = f"{metric_col} 无有效值"
            return result

        model_names = vals["model_name"].tolist()
        metric_vals = vals[metric_col].tolist()

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(model_names))
        bars = ax.bar(x, metric_vals, color=COLORS[: len(model_names)], width=0.5)

        for bar, val in zip(bars, metric_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=10)
        ax.set_title(title, fontsize=13)
        ax.set_ylabel(ylabel or metric_col, fontsize=11)
        ax.set_ylim(0, max(metric_vals) * 1.15 + 0.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)

        result["success"] = True
        logger.info(f"图表已保存: {save_path}")
    except Exception as e:
        result["error"] = str(e)
        logger.warning(f"图表生成失败 ({metric_col}): {e}")

    return result


def plot_precision_recall_bar(
    summary_df: "pd.DataFrame",
    save_path: Path,
) -> dict[str, Any]:
    """生成各模型 Precision / Recall 分组柱状图。"""
    result: dict[str, Any] = {"save_path": str(save_path), "success": False}
    try:
        _ensure_plt()
        import pandas as pd

        if not isinstance(summary_df, pd.DataFrame):
            summary_df = pd.DataFrame(summary_df)

        vals = summary_df[["model_name", "precision", "recall"]].dropna()
        if vals.empty:
            result["error"] = "Precision/Recall 无有效值"
            return result

        model_names = vals["model_name"].tolist()
        precisions = vals["precision"].tolist()
        recalls = vals["recall"].tolist()

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(model_names))
        width = 0.35

        bars1 = ax.bar(x - width / 2, precisions, width, label="Precision", color=COLORS[0])
        bars2 = ax.bar(x + width / 2, recalls, width, label="Recall", color=COLORS[1])

        for bar, val in zip(bars1, precisions):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        for bar, val in zip(bars2, recalls):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=10)
        ax.set_title("各模型 Precision / Recall 对比", fontsize=13)
        ax.set_ylabel("Score", fontsize=11)
        ax.legend(fontsize=10)

        all_vals = precisions + recalls
        ax.set_ylim(0, max(all_vals) * 1.2 + 0.05)

        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)

        result["success"] = True
        logger.info(f"图表已保存: {save_path}")
    except Exception as e:
        result["error"] = str(e)
        logger.warning(f"Precision/Recall 对比图生成失败: {e}")

    return result


def plot_score_distribution(
    score_data: list[dict[str, Any]],
    save_path: Path,
) -> dict[str, Any]:
    """生成各模型 score 分布图（使用直方图或箱线图）。"""
    result: dict[str, Any] = {"save_path": str(save_path), "success": False}
    try:
        _ensure_plt()
        if not score_data:
            result["error"] = "无分数数据"
            return result

        model_names = [s["model_name"] for s in score_data]
        score_lists = []
        for s in score_data:
            # 如果没有原始分数列表，用分布统计模拟
            # 实际操作中由调用方传入 scores 列表
            scores = s.get("_scores", None)
            if scores is not None and len(scores) > 0:
                score_lists.append(scores)
            else:
                score_lists.append([])

        # 过滤掉空列表
        valid_indices = [i for i, sl in enumerate(score_lists) if len(sl) > 0]
        if not valid_indices:
            result["error"] = "无有效分数数据"
            return result

        fig, ax = plt.subplots(figsize=(8, 5))

        valid_names = [model_names[i] for i in valid_indices]
        valid_scores = [score_lists[i] for i in valid_indices]

        bp = ax.boxplot(valid_scores, labels=valid_names, patch_artist=True)

        for patch, color in zip(bp["boxes"], COLORS[: len(valid_names)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title("各模型 Score 分布对比", fontsize=13)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="Threshold=0.5")
        ax.legend(fontsize=9)

        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)

        result["success"] = True
        logger.info(f"Score 分布图已保存: {save_path}")
    except Exception as e:
        result["error"] = str(e)
        logger.warning(f"Score 分布图生成失败: {e}")

    return result


def plot_all(
    summary_df: "pd.DataFrame",
    score_data: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """生成所有图表，返回成功/失败结果。"""
    plot_results: dict[str, Any] = {}

    if not MATPLOTLIB_AVAILABLE:
        plot_results["_warning"] = "matplotlib 不可用，所有图表已跳过"
        return plot_results

    # 1. AUC bar
    plot_results["metric_bar_auc"] = plot_metric_bar(
        summary_df, "auc", "各模型 AUC 对比", output_dir / "metric_bar_auc.png"
    )

    # 2. F1 bar
    plot_results["metric_bar_f1"] = plot_metric_bar(
        summary_df, "f1", "各模型 F1 对比", output_dir / "metric_bar_f1.png"
    )

    # 3. Precision / Recall bar
    plot_results["metric_bar_precision_recall"] = plot_precision_recall_bar(
        summary_df, output_dir / "metric_bar_precision_recall.png"
    )

    # 4. Score distribution
    plot_results["model_score_distribution"] = plot_score_distribution(
        score_data, output_dir / "model_score_distribution.png"
    )

    return plot_results