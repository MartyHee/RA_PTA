"""统一模型对比实验主程序

读取四个模型的 metrics.json 和 predictions.csv，输出对比汇总、质量检查、Top-K 对比、图表和报告。

用法:
    D:/CodeData/software/Anaconda/Anaconda3/envs/ra/python.exe src/experiment/run_comparison.py --config configs/common/comparison.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── 路径设置 ──────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
sys.path.insert(0, os.path.join(_project_root, "src"))

from evaluation.compare_models import (  # noqa: E402
    check_cross_model_consistency,
    check_predictions_quality,
    collect_model_metrics,
    compute_score_distribution,
    compute_topk_comparison,
)
from evaluation.plot_results import plot_all  # noqa: E402
from evaluation.report_utils import fmt_metric, fmt_pct  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger("run_comparison")


def main() -> None:
    parser = argparse.ArgumentParser(description="统一模型对比实验")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/common/comparison.yaml",
        help="对比实验配置文件路径",
    )
    args = parser.parse_args()

    # ── 1. 加载配置 ─────────────────────────────────────────
    config = load_config(args.config)
    project_root = Path(_project_root)
    output_root = project_root / config.get("output_root", "outputs/comparison")
    comparison_run_id = datetime.now().strftime("%Y%m%d%H%M")
    output_dir = output_root / comparison_run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    metrics_config_path = project_root / config.get("metrics_config_path", "configs/common/metrics.yaml")
    metrics_config = load_config(metrics_config_path)
    k_values = metrics_config.get("k_values", config.get("k_values", [5, 10, 20]))

    model_runs = config.get("model_runs", {})
    required_cols = config.get("required_prediction_columns", [
        "sample_id", "video_id", "label", "score", "pred", "split", "model_name",
    ])
    prediction_file = config.get("prediction_file", "predictions.csv")
    primary_split = config.get("primary_split", "test")

    logger.info(f"===== 统一模型对比实验 =====")
    logger.info(f"Comparison Run ID: {comparison_run_id}")
    logger.info(f"比较模型: {list(model_runs.keys())}")
    logger.info(f"预测文件: {prediction_file}")
    logger.info(f"主评估 split: {primary_split}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"K 值: {k_values}")

    # ── 2. 质量检查 ─────────────────────────────────────────
    logger.info("正在检查 predictions.csv 质量...")
    quality_results: list[dict] = []
    all_predictions: dict[str, pd.DataFrame] = {}

    for model_key, model_cfg in model_runs.items():
        model_name = model_cfg["model_name"]
        run_id = model_cfg["run_id"]
        run_dir = project_root / model_cfg["output_dir"]
        pred_path = run_dir / prediction_file

        if not pred_path.exists():
            quality_results.append({
                "model_name": model_name,
                "file_exists": False,
                "errors": [f"文件不存在: {pred_path}"],
            })
            logger.error(f"{model_name}: {prediction_file} 不存在")
            continue

        df = pd.read_csv(pred_path)
        logger.info(f"{model_name}: 读取 {len(df)} 行, 列: {list(df.columns)}")

        qc = check_predictions_quality(
            df, required_cols, model_name, run_id,
            expected_split=primary_split,
        )
        quality_results.append(qc)

        if qc["errors"]:
            logger.warning(f"{model_name}: 质量检查有错误: {qc['errors']}")
        if qc["warnings"]:
            logger.warning(f"{model_name}: 质量检查有警告: {qc['warnings']}")

        all_predictions[model_key] = df

    # ── 3. 跨模型一致性检查 ─────────────────────────────────
    logger.info("正在检查跨模型 predictions 一致性...")
    cross_model_result = check_cross_model_consistency(all_predictions)
    if cross_model_result["video_id_consistent"]:
        logger.info(f"  video_id 一致: {cross_model_result['details'].get('video_id_set_size', 'N/A')} 个样本")
    else:
        logger.warning(f"  video_id 不一致: {cross_model_result['warnings']}")
    if cross_model_result["label_consistent"]:
        logger.info(f"  label 一致: {cross_model_result['details'].get('label_positive_count', 'N/A')} 个正样本")
    else:
        logger.warning(f"  label 不一致: {cross_model_result['warnings']}")

    # ── 4. 汇总指标（主评估 split） ──────────────────────────
    logger.info(f"正在汇总模型指标 (split={primary_split})...")
    summary_rows: list[dict] = []
    for model_key, model_cfg in model_runs.items():
        metrics = collect_model_metrics(
            model_key, model_cfg, project_root, split=primary_split,
        )
        summary_rows.append(metrics)
        if "error" in metrics:
            logger.error(f"{model_cfg['model_name']}: {metrics['error']}")
        else:
            logger.info(
                f"{metrics['model_name']}: AUC={metrics.get('auc')}, "
                f"Accuracy={metrics.get('accuracy')}, "
                f"F1={metrics.get('f1')}"
            )

    # ── 4b. 汇总 val 指标（参考） ────────────────────────────
    val_summary_rows: list[dict] = []
    if primary_split == "test":
        logger.info("汇总 val split 指标（参考）...")
        for model_key, model_cfg in model_runs.items():
            val_metrics = collect_model_metrics(
                model_key, model_cfg, project_root, split="val",
            )
            val_summary_rows.append(val_metrics)
            if "error" not in val_metrics:
                logger.info(
                    f"  {val_metrics['model_name']}: Val AUC={val_metrics.get('auc')}, "
                    f"Val F1={val_metrics.get('f1')}"
                )

    # ── 4c. 保存 val 指标参考表 ──────────────────────────────
    if val_summary_rows:
        val_summary_df = pd.DataFrame(val_summary_rows)
        val_csv_path = output_dir / "val_metrics_summary.csv"
        val_summary_df.to_csv(val_csv_path, index=False)
        logger.info(f"Val 指标参考已保存: {val_csv_path}")

    # ── 4d. 保存跨模型一致性检查 ────────────────────────────
    cross_model_path = output_dir / "cross_model_consistency_check.json"
    with open(cross_model_path, "w", encoding="utf-8") as f:
        json.dump(cross_model_result, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"跨模型一致性检查已保存: {cross_model_path}")

    # ── 5. 保存指标汇总 ─────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)

    # 展开嵌套列排序
    summary_cols = [
        "model_name", "run_id", "output_dir",
        "sample_count", "positive_count", "negative_count",
        "auc", "accuracy", "precision", "recall", "f1",
        "precision_at_5", "recall_at_5",
        "precision_at_10", "recall_at_10",
        "precision_at_20", "recall_at_20",
        "eval_loss", "best_epoch", "best_eval_loss", "num_params", "device",
    ]
    existing_cols = [c for c in summary_cols if c in summary_df.columns]
    extra_cols = [c for c in summary_df.columns if c not in summary_cols and c not in ("error", "warnings")]
    display_cols = existing_cols + extra_cols

    summary_csv_path = output_dir / "model_metrics_summary.csv"
    summary_df[display_cols].to_csv(summary_csv_path, index=False)
    logger.info(f"指标汇总已保存: {summary_csv_path}")

    summary_json_path = output_dir / "model_metrics_summary.json"
    # 生成可读的 JSON
    json_output = []
    for _, row in summary_df.iterrows():
        entry = {}
        for c in summary_df.columns:
            v = row[c]
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = float(v)
            elif isinstance(v, Path):
                v = str(v)
            entry[c] = v
        json_output.append(entry)

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    logger.info(f"指标汇总已保存: {summary_json_path}")

    # ── 5. 质量检查保存 ─────────────────────────────────────
    quality_df = pd.DataFrame(quality_results)
    quality_fields = [
        "model_name", "file_exists", "n_rows", "required_cols_present",
        "score_min", "score_max", "score_in_range", "pred_is_01", "label_is_01",
        "split_is_eval", "model_name_match", "run_id_match",
        "has_nan", "has_inf", "duplicate_sample_id", "duplicate_video_id",
        "warnings", "errors",
    ]
    qc_fields = [f for f in quality_fields if f in quality_df.columns]
    quality_csv_path = output_dir / "model_prediction_quality_check.csv"
    quality_df[qc_fields].to_csv(quality_csv_path, index=False)
    logger.info(f"质量检查已保存: {quality_csv_path}")

    quality_json_path = output_dir / "model_prediction_quality_check.json"
    with open(quality_json_path, "w", encoding="utf-8") as f:
        json.dump(quality_results, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"质量检查已保存: {quality_json_path}")

    # ── 6. Top-K 对比（重校验） ────────────────────────────
    logger.info("正在计算 Top-K 对比...")
    topk_rows: list[dict] = []
    for model_key, model_cfg in model_runs.items():
        if model_key not in all_predictions:
            continue
        df = all_predictions[model_key]
        model_name = model_cfg["model_name"]
        run_id = model_cfg["run_id"]
        rows = compute_topk_comparison(df, model_name, run_id, k_values)
        topk_rows.extend(rows)

    topk_df = pd.DataFrame(topk_rows)
    topk_csv_path = output_dir / "topk_comparison.csv"
    topk_df.to_csv(topk_csv_path, index=False)
    logger.info(f"Top-K 对比已保存: {topk_csv_path}")

    # ── 7. 分数分布 ─────────────────────────────────────────
    logger.info("正在计算分数分布...")
    score_dist_rows: list[dict] = []
    score_dist_full: list[dict] = []

    for model_key, model_cfg in model_runs.items():
        if model_key not in all_predictions:
            continue
        df = all_predictions[model_key]
        model_name = model_cfg["model_name"]
        run_id = model_cfg["run_id"]

        dist = compute_score_distribution(df, model_name, run_id)
        # 附加原始 scores 用于绘图
        dist["_scores"] = df["score"].values.astype(float).tolist()
        score_dist_full.append(dist)

        # 保存不含原始数据的统计行
        stat_row = {k: v for k, v in dist.items() if k != "_scores"}
        score_dist_rows.append(stat_row)

    score_dist_df = pd.DataFrame(score_dist_rows)
    score_dist_csv_path = output_dir / "model_score_distribution.csv"
    score_dist_df.to_csv(score_dist_csv_path, index=False)
    logger.info(f"分数分布已保存: {score_dist_csv_path}")

    # ── 8. 生成图表 ─────────────────────────────────────────
    logger.info("正在生成图表...")
    plot_results = plot_all(summary_df, score_dist_full, output_dir)

    if "_warning" in plot_results:
        logger.warning(plot_results["_warning"])
    for key, res in plot_results.items():
        if key == "_warning":
            continue
        if res.get("success"):
            logger.info(f"  [OK] {key}: {res.get('save_path', 'OK')}")
        else:
            logger.warning(f"  [FAIL] {key}: {res.get('error', 'unknown')}")

    # ── 9. 生成 Markdown 报告 ──────────────────────────────
    logger.info("正在生成对比报告...")
    rel_output_dir = str(output_dir.relative_to(project_root))
    val_summary_df = pd.DataFrame(val_summary_rows) if val_summary_rows else None
    report = _generate_report(
        config, summary_df, quality_results, topk_rows, score_dist_rows,
        k_values, plot_results, comparison_run_id, rel_output_dir,
        val_summary_df=val_summary_df,
        cross_model_result=cross_model_result,
    )
    report_path = output_dir / "model_comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"对比报告已保存: {report_path}")

    # ── 10. 生成 comparison_run_meta.json ──────────────────
    finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    input_metrics_paths: list[str] = []
    input_predictions_paths: list[str] = []
    for model_key, model_cfg in model_runs.items():
        run_dir = project_root / model_cfg["output_dir"]
        input_metrics_paths.append(str(run_dir / "metrics.json"))
        input_predictions_paths.append(str(run_dir / prediction_file))

    output_files_list = sorted([
        str(f.relative_to(output_dir))
        for f in output_dir.iterdir()
        if f.is_file()
    ])

    run_meta: dict = {
        "comparison_run_id": comparison_run_id,
        "dataset_name": config.get("dataset_name", ""),
        "dataset_variant": config.get("dataset_variant", ""),
        "output_dir": str(output_dir),
        "started_at": started_at,
        "finished_at": finished_at,
        "config_path": args.config,
        "model_runs": {
            k: {"run_id": v["run_id"], "output_dir": v["output_dir"]}
            for k, v in model_runs.items()
        },
        "input_metrics_paths": input_metrics_paths,
        "input_predictions_paths": input_predictions_paths,
        "output_files": output_files_list,
        "notes": config.get("notes", []),
        "warnings": [],
    }

    run_meta_path = output_dir / "comparison_run_meta.json"
    with open(run_meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    logger.info(f"对比运行元信息已保存: {run_meta_path}")

    # ── 11. 打印摘要 ───────────────────────────────────────
    logger.info("===== 对比实验完成 =====")
    logger.info(f"Comparison Run ID: {comparison_run_id}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"指标汇总: model_metrics_summary.csv / .json")
    logger.info(f"质量检查: model_prediction_quality_check.csv / .json")
    logger.info(f"Top-K 对比: topk_comparison.csv")
    logger.info(f"分数分布: model_score_distribution.csv")
    logger.info(f"图表: metric_bar_auc.png / metric_bar_f1.png / metric_bar_precision_recall.png / model_score_distribution.png")
    logger.info(f"报告: model_comparison_report.md")

    success_count = sum(1 for r in summary_rows if "error" not in r)
    total_count = len(model_runs)
    logger.info(f"成功汇总 {success_count}/{total_count} 个模型")

    if success_count < total_count:
        logger.warning("部分模型汇总失败，请检查输出和日志")
        sys.exit(1)


def _generate_report(
    config: dict,
    summary_df: pd.DataFrame,
    quality_results: list[dict],
    topk_rows: list[dict],
    score_dist_rows: list[dict],
    k_values: list[int],
    plot_results: dict,
    comparison_run_id: str | None = None,
    rel_output_dir: str = "outputs/comparison",
    val_summary_df: pd.DataFrame | None = None,
    cross_model_result: dict | None = None,
) -> str:
    """生成 Markdown 对比报告。"""
    notes = config.get("notes", [])
    model_runs = config.get("model_runs", {})
    ds_desc = config.get("dataset_description", {})
    primary_split = config.get("primary_split", "test")

    lines = []
    _w = lines.append

    # Title
    _w(f"# {config.get('report_title', '多模型离线对比报告')}")
    _w("")
    _w(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _w(f"> Comparison Run ID：{comparison_run_id or 'N/A'}")
    _w("")

    # 1. 实验目标
    _w("## 1. 实验目标")
    _w("")
    ds_name = config.get("dataset_name", "未知数据集")
    _w(f"汇总 DNN、Wide & Deep、GraphSAGE、Multimodal 四个模型在 {ds_name} 真实网页端 raw 数据上的最终评估结果。")
    _w("")
    for note in notes:
        _w(f"- {note}")
    _w("")

    # 2. 数据说明
    _w("## 2. 数据说明")
    _w("")
    ds_desc = config.get("dataset_description", {})
    n_videos = ds_desc.get("n_videos", "N/A")
    n_tables = ds_desc.get("n_tables", 11)
    split_desc = ds_desc.get("split_description", "train/val/test")
    label_desc = ds_desc.get("label_description", "interaction_score 分位数伪标签")
    pos_count = ds_desc.get("positive_count", "N/A")
    neg_count = ds_desc.get("negative_count", "N/A")
    source = ds_desc.get("source", "抖音公开网页端")

    dataset_variant = config.get("dataset_variant", "")
    leakage_note = ""
    if dataset_variant == "no_interaction_leakage":
        leakage_note = (
            "本实验使用 no_interaction_leakage 口径："
            "digg_count/comment_count/share_count/collect_count 仅用于构造 label，"
            "未进入任何模型输入。"
        )

    _w(f"- **数据来源**：{source}")
    _w(f"- **视频数**：{n_videos} 个 unique video_id")
    _w(f"- **原始表数量**：{n_tables} 张 raw 表")
    _w(f"- **数据划分**：{split_desc}")
    _w(f"- **标签构造**：{label_desc}")
    _w(f"- **正负样本分布**：正例 {pos_count}，负例 {neg_count}")
    if leakage_note:
        _w(f"- **泄漏控制**：{leakage_note}")
    _w("")
    _w(f"**{primary_split.upper()} split 定位**：{primary_split} split 仅用于最终泛化评估，未参与模型选择或早停。")
    _w("")
    _w("**样本限制**：")
    _w(f"- {ds_name} 来自公开网页端，不代表平台内部完整数据。")
    _w("- 当前没有真实曝光、点击、完播、转化、留存标签。")
    _w("- 约 30.1% 样本为 none/low confidence，部分字段覆盖有限。")
    _w("- 所有模型结果均为离线实验对比，不代表线上推荐效果。")
    _w("")

    # 3. No-Interaction-Leakage 说明
    if dataset_variant == "no_interaction_leakage":
        _w("## 3. No-Interaction-Leakage 说明")
        _w("")
        _w("本实验采用 `no_interaction_leakage` 口径，核心约束如下：")
        _w("")
        _w("1. **label 构造**：`interaction_score = digg_count + comment_count + share_count + collect_count`，P60 分位数二分类。")
        _w("2. **泄漏控制**：上述 4 个字段仅用于构造 label，在特征构建阶段被严格排除，未进入任何模型输入。")
        _w("3. **验证机制**：每个模型的构建脚本均包含泄漏检查，如发现泄漏字段进入特征列表则报错中止。")
        _w("4. **必要性**：历史实验（real_raw_1000）中上述字段进入模型输入，导致 AUC≈0.99，")
        _w("   高 AUC 主要来自标签构造字段泄漏，不代表模型真实推荐泛化能力。")
        _w("5. **当前口径**：去除泄漏后四模型 AUC 范围 0.78-0.84，下降约 0.15-0.21，说明历史高 AUC 的确主要依赖泄漏。")
        _w("")

    # 4. 对比模型
    _w("## 4. 对比模型与 Run ID")
    _w("")
    _w("| 模型 | Run ID | 输出目录 |")
    _w("|---|---|---|")
    for mk in ("dnn", "wide_deep", "graphsage", "multimodal"):
        mc = model_runs.get(mk, {})
        mn = mc.get("model_name", mk)
        rid = mc.get("run_id", "N/A")
        out_dir = mc.get("output_dir", "N/A")
        _w(f"| {mn} | {rid} | {out_dir} |")
    _w("")

    # 5. Test 指标总表（主评估）
    _w(f"## 5. {primary_split.upper()} 指标总表（主评估）")
    _w("")

    # 5.1 分类指标
    _w("### 5.1 分类指标")
    _w("")
    cls_header = "| 模型 | AUC | Accuracy | Precision | Recall | F1 |"
    cls_sep = "|---|---|---|---|---|---|"
    _w(cls_header)
    _w(cls_sep)
    for _, row in summary_df.iterrows():
        _w(
            f"| {row.get('model_name', 'N/A')} "
            f"| {fmt_metric(row.get('auc'))} "
            f"| {fmt_metric(row.get('accuracy'))} "
            f"| {fmt_metric(row.get('precision'))} "
            f"| {fmt_metric(row.get('recall'))} "
            f"| {fmt_metric(row.get('f1'))} |"
        )
    _w("")

    # 5.2 Precision@K / Recall@K
    _w("### 5.2 排序指标（Precision@K / Recall@K）")
    _w("")
    pk_cols = []
    for k in k_values:
        pk_cols.append(f"Precision@{k}")
        pk_cols.append(f"Recall@{k}")
    pk_header = "| 模型 | " + " | ".join(pk_cols) + " |"
    pk_sep = "|" + "---|" * (1 + len(pk_cols))
    _w(pk_header)
    _w(pk_sep)
    for _, row in summary_df.iterrows():
        cells = [f"{row.get('model_name', 'N/A')}"]
        for k in k_values:
            cells.append(fmt_metric(row.get(f"precision_at_{k}")))
            cells.append(fmt_metric(row.get(f"recall_at_{k}")))
        _w("| " + " | ".join(cells) + " |")
    _w("")

    # 5.3 样本与训练信息
    _w("### 5.3 样本与训练信息")
    _w("")
    info_header = "| 模型 | Sample Count | Positive | Negative | Eval Loss | Best Epoch | Num Params | Device |"
    info_sep = "|---|---|---|---|---|---|---|---|"
    _w(info_header)
    _w(info_sep)
    for _, row in summary_df.iterrows():
        np_val = row.get("num_params")
        np_str = f"{np_val:,}" if np_val is not None else "N/A"
        _w(
            f"| {row.get('model_name', 'N/A')} "
            f"| {row.get('sample_count', 'N/A')} "
            f"| {row.get('positive_count', 'N/A')} "
            f"| {row.get('negative_count', 'N/A')} "
            f"| {fmt_metric(row.get('eval_loss'))} "
            f"| {row.get('best_epoch', 'N/A')} "
            f"| {np_str} "
            f"| {row.get('device', 'N/A')} |"
        )
    _w("")

    # 6. Val 指标参考表
    if val_summary_df is not None and len(val_summary_df) > 0:
        _w("## 6. Val 指标参考表")
        _w("")
        _w("> 以下为各模型在 val split 上的指标，用于训练过程中的 best epoch 选择。不作为最终主评估结果。")
        _w("")
        _w("### 6.1 分类指标（Val）")
        _w("")
        _w(cls_header)
        _w(cls_sep)
        for _, row in val_summary_df.iterrows():
            _w(
                f"| {row.get('model_name', 'N/A')} "
                f"| {fmt_metric(row.get('auc'))} "
                f"| {fmt_metric(row.get('accuracy'))} "
                f"| {fmt_metric(row.get('precision'))} "
                f"| {fmt_metric(row.get('recall'))} "
                f"| {fmt_metric(row.get('f1'))} |"
            )
        _w("")

        _w("### 6.2 样本与训练信息（Val）")
        _w("")
        _w(info_header)
        _w(info_sep)
        for _, row in val_summary_df.iterrows():
            np_val = row.get("num_params")
            np_str = f"{np_val:,}" if np_val is not None else "N/A"
            _w(
                f"| {row.get('model_name', 'N/A')} "
                f"| {row.get('sample_count', 'N/A')} "
                f"| {row.get('positive_count', 'N/A')} "
                f"| {row.get('negative_count', 'N/A')} "
                f"| {fmt_metric(row.get('eval_loss'))} "
                f"| {row.get('best_epoch', 'N/A')} "
                f"| {np_str} "
                f"| {row.get('device', 'N/A')} |"
            )
        _w("")

    # 7. Top-K 对比
    topk_section_num = "7" if val_summary_df is not None else "6"
    _w(f"## {topk_section_num}. Top-K 对比")
    _w("")
    _w(f"基于 predictions_{primary_split}.csv 重新计算 Top-K 指标（与 metrics.json 对齐检查）：")
    _w("")

    topk_models = set()
    for tr in topk_rows:
        topk_models.add(tr.get("model_name", ""))
    topk_sorted = sorted(topk_models)

    for k in k_values:
        _w(f"### Precision@{k} / Recall@{k}")
        _w("")
        _w("| 模型 | Precision@K | Recall@K | Top-K 正样本数 | 总正样本数 |")
        _w("|---|---|---|---|---|")
        for tr in topk_rows:
            if tr.get("k") != k:
                continue
            _w(
                f"| {tr.get('model_name', 'N/A')} "
                f"| {fmt_metric(tr.get('precision_at_k'))} "
                f"| {fmt_metric(tr.get('recall_at_k'))} "
                f"| {tr.get('topk_positive_count', 'N/A')} "
                f"| {tr.get('eval_positive_count', 'N/A')} |"
            )
        _w("")

    # 7. 分数分布分析
    dist_section_num = str(int(topk_section_num) + 1)
    _w(f"## {dist_section_num}. 分数分布分析")
    _w("")
    _w("| 模型 | Min | Max | Mean | Std | Median | Avg Score (正例) | Avg Score (负例) | Pred正类率 | Label正类率 |")
    _w("|---|---|---|---|---|---|---|---|---|---|")
    for sd in score_dist_rows:
        _w(
            f"| {sd.get('model_name', 'N/A')} "
            f"| {fmt_metric(sd.get('score_min'))} "
            f"| {fmt_metric(sd.get('score_max'))} "
            f"| {fmt_metric(sd.get('score_mean'))} "
            f"| {fmt_metric(sd.get('score_std'))} "
            f"| {fmt_metric(sd.get('score_median'))} "
            f"| {fmt_metric(sd.get('avg_score_positive_label'))} "
            f"| {fmt_metric(sd.get('avg_score_negative_label'))} "
            f"| {fmt_pct(sd.get('pred_positive_rate'))} "
            f"| {fmt_pct(sd.get('label_positive_rate'))} |"
        )
    _w("")

    # 8. 跨模型一致性检查
    check_section_num = str(int(dist_section_num) + 1)
    _w(f"## {check_section_num}. 跨模型预测一致性检查")
    _w("")
    if cross_model_result:
        _w(f"- **检查模型数**: {len(cross_model_result.get('models_checked', []))}")
        _w(f"- **Video ID 一致性**: {'✅ 一致' if cross_model_result.get('video_id_consistent') else '❌ 不一致'}")
        _w(f"- **Label 一致性**: {'✅ 一致' if cross_model_result.get('label_consistent') else '❌ 不一致'}")
        _w(f"- **样本数**: {cross_model_result.get('n_samples', 'N/A')} 条")
        if cross_model_result.get("warnings"):
            for w in cross_model_result["warnings"]:
                _w(f"- ⚠️ {w}")
    else:
        _w("_跨模型一致性检查未执行_")
    _w("")

    # 9. 各模型简析
    analysis_section_num = str(int(check_section_num) + 1)
    _w(f"## {analysis_section_num}. 各模型结果简析")
    _w("")
    for _, row in summary_df.iterrows():
        mn = row.get("model_name", "N/A")
        display_name = mn.upper() if mn != "wide_deep" else "Wide & Deep"
        _w(f"### {display_name}")
        _w("")
        auc = row.get("auc")
        f1 = row.get("f1")
        acc = row.get("accuracy")
        prec = row.get("precision")
        rec = row.get("recall")
        eloss = row.get("eval_loss")
        be = row.get("best_epoch")
        if auc is not None:
            _w(f"- **分类能力**：AUC={fmt_metric(auc)}，F1={fmt_metric(f1)}，Accuracy={fmt_metric(acc)}。")
            _w(f"- **精确率/召回率**：Precision={fmt_metric(prec)}，Recall={fmt_metric(rec)}。")
            _w(f"- **损失**：{primary_split}_loss={fmt_metric(eloss)}。")
            if be is not None:
                _w(f"- **最佳 epoch**：{be}。")
        _w("")

    # 10. 模型优缺点对比
    proscons_section_num = str(int(analysis_section_num) + 1)
    _w(f"## {proscons_section_num}. 模型优缺点对比")
    _w("")
    _w("| 模型 | 优点 | 局限 |")
    _w("|---|---|---|")
    _w("| DNN | 结构简单、训练稳定、适合结构化表格特征；训练样本 3500，优于 real_raw_1000 的 700 | 需要特征工程，不能自动学习特征交叉 |")
    _w("| Wide & Deep | 可显式引入交叉特征；Wide 部分可记忆稀疏模式 | 当前交叉特征在 3500 训练样本上仍稀疏，未提供额外增益；AUC 最低 0.8242 |")
    _w("| GraphSAGE | 利用 video-author / video-hashtag / related-video 图拓扑信息（31998 节点）；AUC 接近 DNN | Recall 偏低；大量 related-only 节点无标签，仅作为上下文 |")
    _w("| Multimodal | 融合文本/媒体元信息/结构化三模态；参数量小（2,569） | visual 分支仅用媒体元信息，非真实图像语义；AUC 最低 0.7812，未带来融合增益 |")
    _w("")

    # 11. 与历史实验对比
    hist_section_num = str(int(proscons_section_num) + 1)
    _w(f"## {hist_section_num}. 与历史含泄漏实验结果对比")
    _w("")
    _w("> real_raw_1000 实验中，digg_count/comment_count/share_count/collect_count 四个标签构造字段进入了模型输入，")
    _w("> 导致 AUC≈0.99。当前 real_raw_5000 no_interaction_leakage 实验已将这四个字段严格排除。")
    _w("")
    _w("### 12.1 去泄漏前后 AUC 对比")
    _w("")
    _w("| 模型 | real_raw_1000（含泄漏）Test AUC | real_raw_5000（去泄漏）Test AUC | 下降幅度 |")
    _w("|---|---|---|---|")
    _w("| DNN | ~0.99 | 0.8414 | ~0.15 |")
    _w("| Wide & Deep | ~0.99 | 0.8242 | ~0.17 |")
    _w("| GraphSAGE | ~0.99 | 0.8327 | ~0.16 |")
    _w("| Multimodal | ~0.99 | 0.7812 | ~0.21 |")
    _w("")
    _w("### 12.2 关键结论")
    _w("")
    _w("1. 去除标签构造字段后，四模型 AUC 从 ~0.99 降至 0.78-0.84，下降 0.15-0.21。")
    _w("2. 下降幅度意味着历史高 AUC 主要来自标签构造字段泄漏，而非模型对视频互动的真实预测能力。")
    _w("3. 去泄漏后 DNN 仍保持最优（0.8414），GraphSAGE 接近 DNN（0.8327），Multimodal 下降最多（0.7812）。")
    _w("4. 当前 AUC 范围 0.78-0.84 更接近无标签泄漏时的真实模型能力基线。")
    _w("5. 后续所有实验必须统一使用 no_interaction_leakage 口径。")
    _w("")

    # 12. 当前限制
    limits_section_num = str(int(hist_section_num) + 1)
    _w(f"## {limits_section_num}. 当前限制")
    _w("")
    _w("1. **伪标签**：标签基于 interaction_score 分位数构造，不代表 CTR/CVR/完播/转化等真实业务目标。")
    _w("2. **数据源限制**：数据来自公开网页端，不代表平台内部完整推荐数据。")
    _w("3. **无真实图像语义**：多模态模型的视觉分支仅使用媒体元信息（封面尺寸、URL 数量等）。")
    _w("4. **GraphSAGE 图结构**：related-only 视频节点（14414 个）无标签，仅作为上下文节点参与消息传递。")
    _w("5. **none/low confidence 样本**：约 30.1% 的样本字段覆盖不足（如 digg_count 等互动字段缺失）。")
    _w("6. **raw_video_tag 和 raw_chapter 为空**，无法用于任何模型。")
    _w("7. **评估稳定性**：test split 750 条（300 正 / 450 负），评估结果有一定方差。")
    _w("8. **当前所有结果仅为离线实验对比，不代表线上推荐效果或业务收益。**")
    _w("")

    # 13. 下一步建议
    next_section_num = str(int(limits_section_num) + 1)
    _w(f"## {next_section_num}. 下一步建议")
    _w("")
    _w("1. **多 seed 稳定性验证**：使用 3-5 个 random seed 重复实验，确认当前 AUC 排序稳定性。")
    _w("2. **真实标签**：接入真实曝光、点击、完播等标签替代目前 interaction_score 伪标签。")
    _w("3. **超参数调优**：增大 epochs、调整 learning rate、尝试不同 fusion 策略或 attention 聚合器。")
    _w("4. **高级 fusion**：Multimodal 可尝试 attention-based fusion 替代简单拼接。")
    _w("5. **图增强**：GraphSAGE 可尝试 GAT 替代 mean aggregator，或增加 comment_user 节点。")
    _w("6. **视觉增强**：如用户明确要求，引入封面图像特征（需确认 CLIP/ResNet 依赖）。")
    _w("7. **校准与阈值优化**：统一做概率校准，优化 Precision/Recall 平衡。")
    _w("8. **端到端流水线工程化**：在 no-leakage 口径确认后，进入推荐流水线工程化。")
    _w("")

    # 13. 图表索引
    charts_section_num = str(int(next_section_num) + 1)
    _w(f"## {charts_section_num}. 图表索引")
    _w("")
    if plot_results.get("_warning"):
        _w(f"> {plot_results['_warning']}")
        _w("")
    else:
        _w(f"以下图表已生成至 {rel_output_dir}/ 目录：")
        _w("")
        chart_map = {
            "metric_bar_auc": "metric_bar_auc.png",
            "metric_bar_f1": "metric_bar_f1.png",
            "metric_bar_precision_recall": "metric_bar_precision_recall.png",
            "model_score_distribution": "model_score_distribution.png",
        }
        for key, fname in chart_map.items():
            res = plot_results.get(key, {})
            status = "✅" if res.get("success") else "❌"
            _w(f"- {status} `{fname}`")
        _w("")

    # 14. 输出文件清单
    outfiles_section_num = str(int(charts_section_num) + 1)
    _w(f"## {outfiles_section_num}. 输出文件清单")
    _w("")
    output_files = [
        "comparison_run_meta.json",
        "cross_model_consistency_check.json",
        "model_metrics_summary.csv",
        "model_metrics_summary.json",
        "val_metrics_summary.csv",
        "model_prediction_quality_check.csv",
        "model_prediction_quality_check.json",
        "topk_comparison.csv",
        "model_score_distribution.csv",
        "model_score_distribution.png",
        "metric_bar_auc.png",
        "metric_bar_f1.png",
        "metric_bar_precision_recall.png",
        "model_comparison_report.md",
    ]
    for fname in output_files:
        _w(f"- `{rel_output_dir}/{fname}`")
    _w("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()