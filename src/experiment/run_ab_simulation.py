#!/usr/bin/env python
"""离线 A/B 模拟主程序。

基于模型 predictions.csv 模拟简单 A/B 分组逻辑和指标统计方法，
输出分组结果、统计结果、分数分布图和 Markdown 报告。

用法:
    python src/experiment/run_ab_simulation.py --config configs/ab_test/ab_base.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── 项目路径 ──────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
sys.path.insert(0, os.path.join(_project_root, "src"))

from experiment.ab_metrics import (
    assign_groups_hash,
    assign_groups_random,
    compute_group_metrics,
    compute_lift,
    compute_score_distribution,
)

# ── YAML 加载工具 ─────────────────────────────────────────
try:
    import yaml
except ImportError:
    yaml = None


def load_config(config_path: str) -> dict[str, Any]:
    """加载 YAML 配置文件。"""
    if yaml is None:
        raise ImportError("需要 PyYAML: pip install pyyaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def resolve_path(path: str, project_root: str) -> str:
    """将相对路径解析为绝对路径。"""
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(Path(project_root) / path)


def safe_json_serialize(obj: Any) -> Any:
    """递归将 numpy 类型转为基础 Python 类型。"""
    if isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json_serialize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def try_set_cjk_font() -> bool:
    """尝试设置 matplotlib 中文字体，返回是否成功。"""
    import matplotlib.pyplot as plt

    try:
        from matplotlib.font_manager import FontProperties

        # Windows 常见中文字体候选
        candidates = [
            "Microsoft YaHei",
            "SimHei",
            "SimSun",
            "FangSong",
            "KaiTi",
        ]
        for font in candidates:
            try:
                fp = FontProperties(family=font)
                if fp.get_name() != "sans-serif":
                    plt.rcParams["font.sans-serif"] = [font]
                    plt.rcParams["axes.unicode_minus"] = False
                    return True
            except Exception:
                continue
        # 检查已安装字体
        from matplotlib.font_manager import findfont, FontProperties

        for font in candidates:
            try:
                findfont(FontProperties(family=font))
                plt.rcParams["font.sans-serif"] = [font]
                plt.rcParams["axes.unicode_minus"] = False
                return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def plot_score_distribution(
    df: pd.DataFrame,
    output_path: str,
    score_col: str = "score",
    group_col: str = "group_role",
    group_labels: dict[str, str] | None = None,
) -> bool:
    """生成 A/B 两组 score 分布箱线图。

    Returns:
        是否成功生成。
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    font_ok = try_set_cjk_font()

    groups = df[group_col].unique()
    data = []
    labels = []
    for g in sorted(groups):
        scores = df[df[group_col] == g][score_col].values.astype(float)
        data.append(scores)
        label = group_labels.get(g, g) if group_labels else g
        labels.append(f"{label} ({g})")

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    # 着色
    colors = ["#4ECDC4", "#FF6B6B"]
    for patch, color in zip(bp["boxes"], colors[: len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    title = "A/B Group Score Distribution" if not font_ok else "A/B 两组分数分布"
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return True


def generate_report(config: dict[str, Any], total: dict[str, Any],
                    group_summary: list[dict[str, Any]],
                    lift: dict[str, Any], ab_run_id: str,
                    output_dir: str, warnings_list: list[str],
                    baseline_comparison: dict[str, Any] | None = None) -> str:
    """生成 A/B 模拟 Markdown 报告（数据驱动，基于 config 内容）。"""
    treatment = next((g for g in group_summary if g["group"] == "treatment"), {})
    control = next((g for g in group_summary if g["group"] == "control"), {})

    dataset_name = config.get("dataset_name", "unknown")
    model_name = config["model_name"]
    group_labels = config.get("group_labels", {"control": "A", "treatment": "B"})
    k_values = config.get("k_values", [5, 10, 20])
    notes = config.get("notes", [])
    data_split = config.get("data_split", "eval")

    lines = []
    lines.append(f"# 离线 A/B 模拟报告（{dataset_name}）\n")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"> A/B Run ID：{ab_run_id}\n")

    # ── 1. 模拟目标 ──
    lines.append("## 1. 模拟目标\n")
    lines.append("本报告基于模型 predictions.csv 进行离线 A/B 分组和指标统计。")
    lines.append("当前仅基于同一批预测结果做分组统计差异分析，不涉及真实线上策略差异。\n")
    lines.append(f"- **数据集**: {dataset_name}")
    lines.append(f"- **主模型**: {model_name} (run_id: {config.get('model_run_id', 'N/A')})")
    if baseline_comparison:
        lines.append(f"- **Baseline 参考**: {config.get('baseline_model_name', 'dnn')} (run_id: {config.get('baseline_run_id', 'N/A')})")
    lines.append(f"- **Comparison Run ID**: {config.get('comparison_run_id', 'N/A')}")
    lines.append(f"- **样本**: {total.get('total_sample_count', 'N/A')} 条 {data_split} split\n")
    dataset_desc = config.get("dataset_description", "")
    if dataset_desc:
        lines.append(f"> **数据说明**: {dataset_desc}\n")

    # ── 2. 输入数据 ──
    lines.append("## 2. 输入数据\n")
    lines.append(f"- **模型**: {model_name}")
    lines.append(f"- **Model Run ID**: {config.get('model_run_id', 'N/A')}")
    lines.append(f"- **Comparison Run ID**: {config.get('comparison_run_id', 'N/A')}")
    lines.append(f"- **Predictions 路径**: {config.get('input_predictions_path', 'N/A')}")
    if baseline_comparison:
        lines.append(f"- **Baseline Predictions 路径**: {config.get('baseline_predictions_path', 'N/A')}")
    lines.append(f"- **总样本数**: {total.get('total_sample_count', 'N/A')}")
    lines.append(f"- **正样本数 / 负样本数**: {total.get('total_positive_count', 'N/A')} / "
                 f"{total.get('total_negative_count', 'N/A')}")
    lines.append(f"- **标签正类率**: {total.get('total_label_positive_rate', 'N/A'):.2%}")
    lines.append("")

    # ── 3. 分组逻辑 ──
    lines.append("## 3. 分组逻辑\n")
    lines.append(f"- **分组方法**: {config['group_method']}")
    lines.append(f"- **分组键**: {config['group_key']}")
    lines.append(f"- **实验组比例 (treatment_ratio)**: {config['treatment_ratio']}")
    if config.get("group_method") == "hash":
        lines.append("  - hash 分组基于 group_key（video_id）的 MD5 值进行稳定分配，")
        lines.append("    同一 video_id 多次运行分组结果一致。")
    else:
        lines.append(f"  - random 分组基于随机种子 {config.get('random_seed', 'N/A')} 进行洗牌分配。")
    lines.append("")

    # ── 4. 分组结果 ──
    lines.append("## 4. 分组结果\n")
    lines.append(f"| 组 | 角色 | 样本数 | 正样本数 | 负样本数 | 标签正类率 | 预测正类率 | 平均分 | 中位数分 |")
    lines.append(f"|---|---|---|---|---|---|---|---|---|")
    for g in group_summary:
        n_neg = g["sample_count"] - g["positive_count"]
        lines.append(
            f"| {group_labels.get(g['group'], g['group'])} | {g['group']} | "
            f"{g['sample_count']} | {g['positive_count']} | {n_neg} | "
            f"{g['label_positive_rate']:.2%} | {g['pred_positive_rate']:.2%} | "
            f"{g['score_mean']:.4f} | {g['score_median']:.4f} |"
        )
    lines.append("")

    # 分组均衡性检查
    n_control = next((g["sample_count"] for g in group_summary if g["group"] == "control"), 0)
    n_treatment = next((g["sample_count"] for g in group_summary if g["group"] == "treatment"), 0)
    lines.append(f"> 分组均衡性: Control={n_control}, Treatment={n_treatment}, "
                 f"合计={n_control + n_treatment}。")
    if abs(n_control - n_treatment) > 10:
        lines.append("> ⚠️ 两组样本量差异超过 10，hash 分组在小样本下出现一定波动。\n")
    else:
        lines.append("> 两组样本量基本均衡。\n")

    # ── 5. 指标统计 ──
    lines.append("## 5. 指标统计\n")

    # 5.1 分类指标
    lines.append("### 5.1 分类指标\n")
    lines.append(f"| 组 | AUC | Accuracy | Precision | Recall | F1 |")
    lines.append(f"|---|---|---|---|---|---|")
    for g in group_summary:
        grp_label = group_labels.get(g['group'], g['group'])
        auc_str = f"{g['auc']:.4f}" if g.get('auc') is not None else "N/A"
        acc_str = f"{g['accuracy']:.4f}" if g.get('accuracy') is not None else "N/A"
        prec_str = f"{g['precision']:.4f}" if g.get('precision') is not None else "N/A"
        rec_str = f"{g['recall']:.4f}" if g.get('recall') is not None else "N/A"
        f1_str = f"{g['f1']:.4f}" if g.get('f1') is not None else "N/A"
        lines.append(f"| {grp_label} | {auc_str} | {acc_str} | {prec_str} | {rec_str} | {f1_str} |")
    lines.append("")

    # 5.2 排序指标
    lines.append("### 5.2 排序指标\n")
    k_headers = " | ".join([f"Precision@{k}" for k in k_values] + [f"Recall@{k}" for k in k_values])
    lines.append(f"| 组 | {k_headers} |")
    sep_headers = " | ".join(["---"] * (1 + len(k_values) * 2))
    lines.append(f"|{sep_headers}|")
    for g in group_summary:
        p_vals = " | ".join([
            f"{g.get(f'precision_at_{k}', 'N/A'):.4f}" if g.get(f'precision_at_{k}') is not None else "N/A"
            for k in k_values
        ])
        r_vals = " | ".join([
            f"{g.get(f'recall_at_{k}', 'N/A'):.4f}" if g.get(f'recall_at_{k}') is not None else "N/A"
            for k in k_values
        ])
        lines.append(f"| {group_labels.get(g['group'], g['group'])} | {p_vals} | {r_vals} |")
    lines.append("")

    # 5.3 分数分布
    lines.append("### 5.3 分数分布\n")
    lines.append(f"| 组 | Score Mean | Score Std | Score Median | Score Min | Score Max |")
    lines.append(f"|---|---|---|---|---|---|")
    for g in group_summary:
        grp_label = group_labels.get(g['group'], g['group'])
        smin_str = f"{g['score_min']:.4f}" if g.get('score_min') is not None else "N/A"
        smax_str = f"{g['score_max']:.4f}" if g.get('score_max') is not None else "N/A"
        lines.append(
            f"| {grp_label} | {g['score_mean']:.4f} | {g['score_std']:.4f} | "
            f"{g['score_median']:.4f} | {smin_str} | {smax_str} |"
        )
    lines.append("")

    # ── 6. Lift 计算 ──
    lines.append("## 6. Lift 计算\n")
    lines.append("以下 lift 为离线分组统计差异，不是线上因果收益。\n")
    lines.append("计算公式：")
    lines.append("- 绝对差 = treatment - control")
    lines.append("- 相对 lift = (treatment - control) / control × 100%\n")
    lines.append(f"| 指标 | Control | Treatment | 绝对差 | 相对 Lift |")
    lines.append(f"|---|---|---|---|---|")
    for metric_name, display_name in [
        ("score_mean", "平均预测分"),
        ("label_positive_rate", "标签正类率"),
        ("pred_positive_rate", "预测正类率"),
    ]:
        c_val = control.get(metric_name, "N/A")
        t_val = treatment.get(metric_name, "N/A")
        diff = lift.get(f"treatment_minus_control_{metric_name}")
        rel = lift.get(f"relative_lift_{metric_name}")
        c_str = f"{c_val:.4f}" if isinstance(c_val, (int, float)) else str(c_val)
        t_str = f"{t_val:.4f}" if isinstance(t_val, (int, float)) else str(t_val)
        d_str = f"{diff:+.4f}" if diff is not None else "N/A"
        r_str = f"{rel:+.2f}%" if rel is not None else "N/A"
        lines.append(f"| {display_name} | {c_str} | {t_str} | {d_str} | {r_str} |")
    lines.append("")
    lines.append("> **重要说明**: 以上 lift 只是同一模型预测结果在两组间的统计差异，")
    lines.append("> 不反映任何策略干预的因果效应。在当前离线 A/B 模拟中，")
    lines.append("> control 和 treatment 两组使用完全相同的模型预测结果，")
    lines.append("> 分组仅基于 group_key 的 hash 值，不属于真实线上 A/B 测试。\n")

    # ── 7. Baseline 对比（可选） ──
    if baseline_comparison:
        lines.append("## 7. Baseline 模型分数对比（参考）\n")
        lines.append(f"以下为 DNN (baseline) 与 {model_name} (主模型) 在各组的分数对比，仅供参考。\n")
        lines.append(f"| 组 | DNN Score Mean | DNN Score Std | {model_name} Score Mean | Score Delta Mean |")
        lines.append(f"|---|---|---|---|---|")
        for role in ["control", "treatment"]:
            if role in baseline_comparison:
                bc = baseline_comparison[role]
                dnn_mean = bc.get("dnn_score_mean", "N/A")
                dnn_std = bc.get("dnn_score_std", "N/A")
                group_metrics = next((g for g in group_summary if g["group"] == role), {})
                mm_mean = group_metrics.get("score_mean", "N/A")
                delta = bc.get("score_delta_mean", "N/A")
                dnn_mean_s = f"{dnn_mean:.4f}" if isinstance(dnn_mean, (int, float)) else str(dnn_mean)
                dnn_std_s = f"{dnn_std:.4f}" if isinstance(dnn_std, (int, float)) else str(dnn_std)
                mm_mean_s = f"{mm_mean:.4f}" if isinstance(mm_mean, (int, float)) else str(mm_mean)
                delta_s = f"{delta:+.4f}" if isinstance(delta, (int, float)) else str(delta)
                grp_label = group_labels.get(role, role)
                lines.append(f"| {grp_label} | {dnn_mean_s} | {dnn_std_s} | {mm_mean_s} | {delta_s} |")
        lines.append("")
        lines.append("> **注意**: DNN 分数仅作为参考 baseline，不代表真实 control 策略。")
        lines.append("> 两个模型在同一批数据上评分，差异反映模型行为差异而非策略收益。\n")

    # ── 8. 主要发现 ──
    lines.append("## 8. 主要发现\n")
    for g in group_summary:
        grp_label = group_labels.get(g['group'], g['group'])
        auc_str = f"{g.get('auc', 'N/A'):.4f}" if g.get('auc') is not None else "N/A"
        prec_str = f"{g.get('precision', 'N/A'):.4f}" if g.get('precision') is not None else "N/A"
        rec_str = f"{g.get('recall', 'N/A'):.4f}" if g.get('recall') is not None else "N/A"
        f1_str = f"{g.get('f1', 'N/A'):.4f}" if g.get('f1') is not None else "N/A"
        lines.append(f"- **{grp_label} 组 (n={g['sample_count']})**: AUC={auc_str}, "
                     f"Precision={prec_str}, Recall={rec_str}, F1={f1_str}")
    lines.append("")

    # ── 9. 局限性 ──
    lines.append("## 9. 局限性\n")
    lines.append("1. **没有真实用户曝光日志**: 当前仅基于模型预测 score，无真实曝光数据。")
    lines.append("2. **没有真实点击/转化/完播标签**: 当前 label 为 interaction_score 分位数伪标签。")
    lines.append("3. **没有真实 control/treatment 策略差异**: 两组使用完全相同的模型预测，无策略差异。")
    lines.append(f"4. **样本量有限**: 当前基于 {data_split} split {total.get('total_sample_count', 'N/A')} 条样本，"
                 f"A/B 分组后各组样本更少，hash 分组可能存在一定波动。")
    lines.append("5. **lift 不是因果收益**: 所有 lift 仅为分组统计差异，不代表任何策略干预效果。")
    lines.append("6. **离线 A/B 不等于线上 A/B**: 离线模拟没有流量干预、没有用户行为反馈、没有时间维度。")
    lines.append("7. **样本来源**: 数据来自抖音公开网页端，不代表平台内部完整数据。")
    lines.append("")

    # ── 10. 后续真实 A/B 测试建议 ──
    lines.append("## 10. 后续真实 A/B 测试建议\n")
    lines.append("1. **实验单位**: 明确以 user_id 或 device_id 为实验单位，确保同一用户始终在同一组。")
    lines.append("2. **随机化方式**: 使用稳定的 hash 分桶（如 user_id mod 100），确保分组可复现。")
    lines.append("3. **流量比例**: 根据实验风险确定 treatment 流量比例（如 1%/5%/10%/50%）。")
    lines.append("4. **主要指标**: 明确核心业务指标如 CTR、CVR、完播率、人均观看时长等。")
    lines.append("5. **护栏指标**: 设定护栏指标如 DAU、刷新频率、负反馈率等确保实验安全。")
    lines.append("6. **样本量**: 确保足够样本量使指标达到统计显著性（建议实验前做 power analysis）。")
    lines.append("7. **实验周期**: 保证足够的实验周期（至少 7-14 天）以覆盖周间效应。")
    lines.append("8. **完整链路**: 记录曝光、点击、播放、完播、互动等完整用户行为链路。")
    lines.append("9. **显著性检验**: 使用 t-test 或 bootstrap 计算置信区间和 p-value。")
    lines.append("10. **AA 测试**: 上线前先做 AA 测试验证分组无偏性。")
    lines.append("")

    # ── 11. 结论 ──
    lines.append("## 11. 结论\n")
    lines.append(f"✅ **已跑通离线 A/B 模拟流程**: 基于 {dataset_name} {data_split} split "
                 f"({total.get('total_sample_count', 'N/A')} 条) 完成分组、指标统计、lift 计算。")
    lines.append("")
    lines.append("❌ **不支持正式线上收益判断**: 当前结果基于离线预测和 hash 分组，")
    lines.append("    不具备统计显著性，不能代表真实线上 A/B 测试结论。")
    lines.append("")

    # ── 12. 输出文件清单 ──
    section_num = 12
    lines.append(f"## {section_num}. 输出文件清单\n")
    output_files_text = [
        f"- `{output_dir}/ab_run_meta.json`",
        f"- `{output_dir}/ab_group_assignment.csv`",
        f"- `{output_dir}/ab_metrics_summary.csv`",
        f"- `{output_dir}/ab_metrics_summary.json`",
        f"- `{output_dir}/ab_score_distribution.csv`",
        f"- `{output_dir}/ab_score_distribution.png`",
        f"- `{output_dir}/ab_simulation_report.md`",
    ]
    lines.extend(output_files_text)
    lines.append("")

    if warnings_list:
        lines.append(f"## {section_num + 1}. 运行警告\n")
        for w in warnings_list:
            lines.append(f"- ⚠️ {w}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="离线 A/B 模拟")
    parser.add_argument("--config", default="configs/ab_test/ab_base.yaml",
                        help="配置文件路径")
    args = parser.parse_args()

    # ── 加载配置 ─────────────────────────────────────────
    config_path = args.config
    if not os.path.isfile(config_path):
        print(f"[ERROR] 配置文件不存在: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    print(f"[INFO] 已加载配置: {config_path}")

    output_root = config.get("output_root", "outputs/ab_test")
    output_root_abs = resolve_path(output_root, _project_root)

    # ── 生成 ab_run_id ───────────────────────────────────
    ab_run_id = datetime.now().strftime("%Y%m%d%H%M")
    output_dir_abs = os.path.join(output_root_abs, ab_run_id)
    os.makedirs(output_dir_abs, exist_ok=True)
    print(f"[INFO] ab_run_id = {ab_run_id}")
    print(f"[INFO] 输出目录: {output_dir_abs}")

    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 加载 predictions ─────────────────────────────────
    input_path = config["input_predictions_path"]
    input_path_abs = resolve_path(input_path, _project_root)
    print(f"[INFO] 加载 predictions: {input_path_abs}")

    if not os.path.isfile(input_path_abs):
        print(f"[ERROR] Predictions 文件不存在: {input_path_abs}")
        sys.exit(1)

    df = pd.read_csv(input_path_abs)
    print(f"[INFO] 加载完毕: {len(df)} 行, {list(df.columns)}")

    all_warnings: list[str] = []

    # ── 检查必需列 ─────────────────────────────────────
    required_cols = config.get("required_prediction_columns", [])
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[WARNING] 缺少必需列: {missing}")
        all_warnings.append(f"缺少必需列: {missing}")

    score_col = config.get("score_col", "score")
    label_col = config.get("label_col", "label")
    pred_col = config.get("pred_col", "pred")

    # ── 可选 DNN baseline 加载 ────────────────────────────
    baseline_path = config.get("baseline_predictions_path")
    if baseline_path:
        baseline_path_abs = resolve_path(baseline_path, _project_root)
        if os.path.isfile(baseline_path_abs):
            baseline_df = pd.read_csv(baseline_path_abs)
            df = df.merge(
                baseline_df[["video_id", "score"]].rename(columns={"score": "dnn_score"}),
                on="video_id", how="left"
            )
            n_missing = df["dnn_score"].isna().sum()
            if n_missing > 0:
                all_warnings.append(f"DNN baseline merge: {n_missing} 条未匹配到 dnn_score")
            print(f"[INFO] DNN baseline merged: {len(df)} rows, {n_missing} missing dnn_score")
        else:
            print(f"[WARNING] DNN baseline 文件不存在: {baseline_path_abs}")
            all_warnings.append(f"DNN baseline 文件不存在: {baseline_path_abs}")

    # ── 分组分配 ───────────────────────────────────────
    group_method = config.get("group_method", "hash")
    group_key = config.get("group_key", "video_id")
    treatment_ratio = config.get("treatment_ratio", 0.5)
    random_seed = config.get("random_seed", 2026)
    group_labels = config.get("group_labels", {"control": "A", "treatment": "B"})

    # 检查 group_key 是否存在
    if group_key not in df.columns:
        all_warnings.append(
            f"group_key '{group_key}' 不存在，回退到 sample_id"
        )
        group_key = "sample_id"
        if group_key not in df.columns:
            print(f"[ERROR] group_key 和 sample_id 均不存在")
            sys.exit(1)

    print(f"[INFO] 分组方法: {group_method}, 分组键: {group_key}, ratio: {treatment_ratio}")

    if group_method == "hash":
        group_series = assign_groups_hash(df, group_key, treatment_ratio, group_labels)
    elif group_method == "random":
        group_series = assign_groups_random(df, treatment_ratio, random_seed)
    else:
        print(f"[ERROR] 未知分组方法: {group_method}")
        sys.exit(1)

    df["group"] = group_series
    df["group_role"] = df["group"].map(group_labels)
    df["group_method"] = group_method
    df["group_key"] = group_key
    df["model_name"] = config.get("model_name", "unknown")
    df["model_run_id"] = config.get("model_run_id", "unknown")
    df["ab_run_id"] = ab_run_id

    # 检查分组是否均衡
    group_counts = df["group"].value_counts()
    print(f"[INFO] 分组结果: {group_counts.to_dict()}")
    for g in ["control", "treatment"]:
        if g not in group_counts.index:
            all_warnings.append(f"组 '{g}' 无样本分配到")

    # ── 整体统计 ───────────────────────────────────────
    total_positive = int(df[label_col].sum())
    total_sample = len(df)
    total_negative = total_sample - total_positive
    total_label_rate = float(df[label_col].mean())
    total_score_mean = float(df[score_col].mean())
    total_pred_rate = float(df[pred_col].mean()) if pred_col in df.columns else None

    total_summary = {
        "total_sample_count": total_sample,
        "total_positive_count": total_positive,
        "total_negative_count": total_negative,
        "total_label_positive_rate": total_label_rate,
        "total_score_mean": total_score_mean,
        "total_pred_positive_rate": total_pred_rate,
    }
    print(f"[INFO] 总计: {total_sample} 样本, {total_positive} 正, {total_negative} 负")
    print(f"[INFO] 标签正类率: {total_label_rate:.2%}")

    # ── 各组指标 ───────────────────────────────────────
    k_values = config.get("k_values", [5, 10])

    group_summary = []
    group_distributions = []
    for group_role in ["control", "treatment"]:
        gdf = df[df["group"] == group_role].copy()
        if len(gdf) == 0:
            all_warnings.append(f"组 '{group_role}' 无样本，跳过指标计算")
            continue

        metrics = compute_group_metrics(
            gdf, group_role, k_values, score_col, label_col, pred_col
        )
        group_summary.append(metrics)
        group_distributions.append(compute_score_distribution(
            gdf, group_role, score_col, label_col, pred_col
        ))
        print(f"[INFO] 组 {group_role}: {len(gdf)} 样本, "
              f"score_mean={metrics['score_mean']:.4f}, "
              f"precision={metrics.get('precision', 'N/A')}, "
              f"recall={metrics.get('recall', 'N/A')}")

    # ── Lift 计算 ──────────────────────────────────────
    treatment_metrics = next((g for g in group_summary if g["group"] == "treatment"), None)
    control_metrics = next((g for g in group_summary if g["group"] == "control"), None)
    lift_summary = {}
    if treatment_metrics is not None and control_metrics is not None:
        lift_summary = compute_lift(treatment_metrics, control_metrics)
        print(f"[INFO] Lift - score_mean: {lift_summary.get('relative_lift_score_mean', 'N/A')}")
    else:
        all_warnings.append("缺少 treatment 或 control 组，无法计算 lift")

    # ── DNN baseline 分组统计 ─────────────────────────────
    baseline_comparison: dict[str, Any] = {}
    if "dnn_score" in df.columns:
        for group_role in ["control", "treatment"]:
            gdf = df[df["group"] == group_role]
            dnn_scores = gdf["dnn_score"].values.astype(float)
            mm_scores = gdf[score_col].values.astype(float)
            baseline_comparison[group_role] = {
                "model_name": config.get("baseline_model_name", "dnn"),
                "dnn_score_mean": float(dnn_scores.mean()),
                "dnn_score_std": float(dnn_scores.std()),
                "dnn_score_median": float(np.median(dnn_scores)),
                "score_delta_mean": float((mm_scores - dnn_scores).mean()),
            }
        print(f"[INFO] DNN baseline comparison computed for {len(baseline_comparison)} groups")

    # ── 写出 ab_group_assignment.csv ───────────────────
    assignment_path = os.path.join(output_dir_abs, "ab_group_assignment.csv")
    assignment_cols = [
        "sample_id", "video_id", label_col, score_col, pred_col,
        "group", "group_role", "group_method", "group_key",
        "model_name", "model_run_id", "ab_run_id",
    ]
    # 保留额外存在的列（如 node_id, author_id）
    extra_cols = [c for c in df.columns if c not in assignment_cols and c not in (
        label_col, score_col, pred_col)]
    final_cols = assignment_cols + [c for c in extra_cols
                                     if c in df.columns and c not in assignment_cols]
    # 确保 label/score/pred 在正确位置
    output_cols = []
    for c in assignment_cols:
        if c == label_col or c == score_col or c == pred_col:
            output_cols.append(c)
        elif c in df.columns:
            output_cols.append(c)
    # 添加不在 assignment_cols 中的额外列
    for c in df.columns:
        if c not in output_cols:
            output_cols.append(c)

    df[output_cols].to_csv(assignment_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 分组结果已写入: {assignment_path}")

    # ── 写出 ab_metrics_summary.csv ────────────────────
    metrics_csv_path = os.path.join(output_dir_abs, "ab_metrics_summary.csv")
    metrics_csv_cols = [
        "group", "group_role", "sample_count", "positive_count", "negative_count",
        "label_positive_rate", "score_mean", "score_std", "score_median",
        "score_min", "score_max",
        "pred_positive_count", "pred_positive_rate",
        "precision", "recall", "f1",
    ]
    for k in k_values:
        metrics_csv_cols.append(f"precision_at_{k}")
    for k in k_values:
        metrics_csv_cols.append(f"recall_at_{k}")

    with open(metrics_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_csv_cols, extrasaction="ignore")
        writer.writeheader()
        for g in group_summary:
            writer.writerow(g)
    print(f"[INFO] 指标汇总已写入: {metrics_csv_path}")

    # ── 写出 ab_metrics_summary.json ───────────────────
    metrics_json_path = os.path.join(output_dir_abs, "ab_metrics_summary.json")
    metrics_json = safe_json_serialize({
        "ab_run_id": ab_run_id,
        "input_predictions_path": input_path_abs,
        "model_name": config["model_name"],
        "model_run_id": config["model_run_id"],
        "comparison_run_id": config.get("comparison_run_id"),
        "comparison_output_dir": config.get("comparison_output_dir"),
        "group_method": group_method,
        "group_key": group_key,
        "treatment_ratio": treatment_ratio,
        "random_seed": random_seed,
        "total_summary": total_summary,
        "group_summary": group_summary,
        "lift_summary": lift_summary,
        "baseline_comparison": baseline_comparison if baseline_comparison else None,
        "warnings": all_warnings,
        "notes": config.get("notes", []),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 指标 JSON 已写入: {metrics_json_path}")

    # ── 写出 ab_score_distribution.csv ─────────────────
    dist_csv_path = os.path.join(output_dir_abs, "ab_score_distribution.csv")
    dist_cols = [
        "group", "group_role", "score_min", "score_max",
        "score_mean", "score_std", "score_median",
        "label_positive_rate", "pred_positive_rate",
    ]
    with open(dist_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=dist_cols, extrasaction="ignore")
        writer.writeheader()
        for d in group_distributions:
            writer.writerow(d)
    print(f"[INFO] 分数分布已写入: {dist_csv_path}")

    # ── 生成 ab_score_distribution.png ─────────────────
    dist_png_path = os.path.join(output_dir_abs, "ab_score_distribution.png")
    plot_ok = plot_score_distribution(
        df, dist_png_path, score_col, "group_role", group_labels
    )
    if plot_ok:
        print(f"[INFO] 分数分布图已生成: {dist_png_path}")
    else:
        all_warnings.append("matplotlib 不可用，跳过分数分布图生成")
        print(f"[WARNING] matplotlib 不可用，跳过图片生成")

    # ── 生成 ab_simulation_report.md ────────────────────
    report = generate_report(
        config=config,
        total=total_summary,
        group_summary=group_summary,
        lift=lift_summary,
        ab_run_id=ab_run_id,
        output_dir=output_dir_abs,
        warnings_list=all_warnings,
        baseline_comparison=baseline_comparison if baseline_comparison else None,
    )
    report_path = os.path.join(output_dir_abs, "ab_simulation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[INFO] 报告已生成: {report_path}")

    # ── 写出 ab_run_meta.json ──────────────────────────
    finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_files = [
        "ab_run_meta.json",
        "ab_group_assignment.csv",
        "ab_metrics_summary.csv",
        "ab_metrics_summary.json",
        "ab_score_distribution.csv",
        "ab_simulation_report.md",
    ]
    if plot_ok:
        output_files.append("ab_score_distribution.png")

    meta = safe_json_serialize({
        "ab_run_id": ab_run_id,
        "output_dir": output_dir_abs,
        "started_at": started_at,
        "finished_at": finished_at,
        "config_path": config_path,
        "input_predictions_path": input_path_abs,
        "model_name": config["model_name"],
        "model_run_id": config["model_run_id"],
        "comparison_run_id": config.get("comparison_run_id"),
        "comparison_output_dir": config.get("comparison_output_dir"),
        "group_method": group_method,
        "group_key": group_key,
        "treatment_ratio": treatment_ratio,
        "random_seed": random_seed,
        "output_files": output_files,
        "warnings": all_warnings,
        "notes": config.get("notes", []),
    })
    meta_path = os.path.join(output_dir_abs, "ab_run_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 运行元信息已写入: {meta_path}")

    # ── 汇总 ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"离线 A/B 模拟完成!")
    print(f"  A/B Run ID: {ab_run_id}")
    print(f"  输出目录: {output_dir_abs}")
    print(f"  总样本: {total_sample}, 正样本: {total_positive}")
    print(f"  Control 组: {group_counts.get('control', 0)} 样本")
    print(f"  Treatment 组: {group_counts.get('treatment', 0)} 样本")
    if lift_summary:
        print(f"  Score Mean Lift: {lift_summary.get('relative_lift_score_mean', 'N/A')}")
    print(f"  生成文件数: {len(output_files)}")
    if all_warnings:
        print(f"  Warnings ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"    - {w}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()