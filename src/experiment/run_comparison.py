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

    logger.info(f"===== 统一模型对比实验 =====")
    logger.info(f"Comparison Run ID: {comparison_run_id}")
    logger.info(f"比较模型: {list(model_runs.keys())}")
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
        pred_path = run_dir / "predictions.csv"

        if not pred_path.exists():
            quality_results.append({
                "model_name": model_name,
                "file_exists": False,
                "errors": [f"文件不存在: {pred_path}"],
            })
            logger.error(f"{model_name}: predictions.csv 不存在")
            continue

        df = pd.read_csv(pred_path)
        logger.info(f"{model_name}: 读取 {len(df)} 行, 列: {list(df.columns)}")

        qc = check_predictions_quality(df, required_cols, model_name, run_id)
        quality_results.append(qc)

        if qc["errors"]:
            logger.warning(f"{model_name}: 质量检查有错误: {qc['errors']}")
        if qc["warnings"]:
            logger.warning(f"{model_name}: 质量检查有警告: {qc['warnings']}")

        all_predictions[model_key] = df

    # ── 3. 汇总指标 ─────────────────────────────────────────
    logger.info("正在汇总模型指标...")
    summary_rows: list[dict] = []
    for model_key, model_cfg in model_runs.items():
        metrics = collect_model_metrics(model_key, model_cfg, project_root)
        summary_rows.append(metrics)
        if "error" in metrics:
            logger.error(f"{model_cfg['model_name']}: {metrics['error']}")
        else:
            logger.info(
                f"{metrics['model_name']}: AUC={metrics.get('auc')}, "
                f"Accuracy={metrics.get('accuracy')}, "
                f"F1={metrics.get('f1')}"
            )

    # ── 4. 保存指标汇总 ─────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)

    # 展开嵌套列排序
    summary_cols = [
        "model_name", "run_id", "output_dir",
        "sample_count", "positive_count", "negative_count",
        "auc", "accuracy", "precision", "recall", "f1",
        "precision_at_5", "recall_at_5",
        "precision_at_10", "recall_at_10",
        "precision_at_20", "recall_at_20",
        "eval_loss", "best_epoch", "best_eval_loss", "device",
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
    report = _generate_report(
        config, summary_df, quality_results, topk_rows, score_dist_rows,
        k_values, plot_results, comparison_run_id, rel_output_dir,
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
        input_predictions_paths.append(str(run_dir / "predictions.csv"))

    output_files_list = sorted([
        str(f.relative_to(output_dir))
        for f in output_dir.iterdir()
        if f.is_file()
    ])

    run_meta: dict = {
        "comparison_run_id": comparison_run_id,
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
) -> str:
    """生成 Markdown 对比报告。"""
    notes = config.get("notes", [])
    model_runs = config.get("model_runs", {})

    lines = []
    _w = lines.append

    # Title
    _w("# 多模型离线对比报告（sample0427 流程验证）")
    _w("")
    _w(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _w("")

    # 1. 对比目标
    _w("## 1. 对比目标")
    _w("")
    _w("本报告汇总 DNN、Wide & Deep、GraphSAGE、多模态四个模型在 sample0427 样本数据上的 eval 评估结果。")
    _w("")
    for note in notes:
        _w(f"- {note}")
    _w("")

    # 2. 数据与标签说明
    _w("## 2. 数据与标签说明")
    _w("")
    _w("- **数据来源**：sample0427 样本数据（79 条主视频，11 张表）")
    _w("- **标签构造**：interaction_score = digg_count + comment_count + share_count + collect_count，60% 分位数阈值构造二分类伪标签")
    _w("- **标签含义**：当前标签为流程验证伪标签，不代表真实曝光、点击、完播、转化、留存等业务指标")
    _w("- **数据划分**：train/eval = 80/20（seed=2026），train 63 条、eval 16 条")
    _w("- **独立 test 集**：当前无独立 test 集，eval 仅用于流程验证和最小模型评估")
    _w("- **样本限制**：eval 仅 16 条（正例 6、负例 10），所有指标波动极大，仅支持工程流程验证")
    _w("")

    # 3. 对比模型
    _w("## 3. 对比模型与 Run ID")
    _w("")
    _w(f"> Comparison Run ID：{comparison_run_id or 'N/A'}")
    _w("")
    _w("| 模型 | Run ID | 输出目录 |")
    _w("|---|---|---|")
    _w("| DNN | {dnn_run} | outputs/dnn/{dnn_run} |".format(
        dnn_run=model_runs.get("dnn", {}).get("run_id", "N/A")
    ))
    _w("| Wide & Deep | {wd_run} | outputs/wide_deep/{wd_run} |".format(
        wd_run=model_runs.get("wide_deep", {}).get("run_id", "N/A")
    ))
    _w("| GraphSAGE | {gs_run} | outputs/graphsage/{gs_run} |".format(
        gs_run=model_runs.get("graphsage", {}).get("run_id", "N/A")
    ))
    _w("| Multimodal | {mm_run} | outputs/multimodal/{mm_run} |".format(
        mm_run=model_runs.get("multimodal", {}).get("run_id", "N/A")
    ))
    _w("")

    # 4. 指标汇总
    _w("## 4. 指标汇总")
    _w("")

    # 基本指标表
    _w("### 4.1 分类指标")
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

    # Top-K 指标表
    _w("### 4.2 排序指标（Precision@K / Recall@K）")
    _w("")
    pk_header = "| 模型 | Precision@5 | Recall@5 | Precision@10 | Recall@10 | Precision@20 | Recall@20 |"
    pk_sep = "|---|---|---|---|---|---|---|"
    _w(pk_header)
    _w(pk_sep)
    for _, row in summary_df.iterrows():
        _w(
            f"| {row.get('model_name', 'N/A')} "
            f"| {fmt_metric(row.get('precision_at_5'))} "
            f"| {fmt_metric(row.get('recall_at_5'))} "
            f"| {fmt_metric(row.get('precision_at_10'))} "
            f"| {fmt_metric(row.get('recall_at_10'))} "
            f"| {fmt_metric(row.get('precision_at_20'))} "
            f"| {fmt_metric(row.get('recall_at_20'))} |"
        )
    _w("")

    # 样本信息表
    _w("### 4.3 样本与训练信息")
    _w("")
    info_header = "| 模型 | Sample Count | Positive | Negative | Eval Loss | Best Epoch | Device |"
    info_sep = "|---|---|---|---|---|---|---|"
    _w(info_header)
    _w(info_sep)
    for _, row in summary_df.iterrows():
        _w(
            f"| {row.get('model_name', 'N/A')} "
            f"| {row.get('sample_count', 'N/A')} "
            f"| {row.get('positive_count', 'N/A')} "
            f"| {row.get('negative_count', 'N/A')} "
            f"| {fmt_metric(row.get('eval_loss'))} "
            f"| {row.get('best_epoch', 'N/A')} "
            f"| {row.get('device', 'N/A')} |"
        )
    _w("")

    # 5. 各模型结果简析
    _w("## 5. 各模型结果简析")
    _w("")
    for _, row in summary_df.iterrows():
        mn = row.get("model_name", "N/A")
        _w(f"### {mn.upper() if mn != 'wide_deep' else 'Wide & Deep'}")
        _w("")
        auc = row.get("auc")
        f1 = row.get("f1")
        acc = row.get("accuracy")
        if auc is not None:
            _w(f"- 在当前流程验证集上，AUC 为 {fmt_metric(auc)}，F1 为 {fmt_metric(f1)}，Accuracy 为 {fmt_metric(acc)}。")
        _w("")

    # 6. 模型优缺点对比
    _w("## 6. 模型优缺点对比")
    _w("")
    _w("| 模型 | 优点 | 局限 |")
    _w("|---|---|---|")
    _w("| DNN | 结构简单、易跑通、适合表格特征 | 不显式建模交叉、依赖特征工程 |")
    _w("| Wide & Deep | 能显式引入交叉特征 | 当前交叉特征样本少、容易偏正类 |")
    _w("| GraphSAGE | 能使用 video-author / video-hashtag / related-video 图关系 | 当前图中大量节点和边为规则补齐，图关系不代表真实推荐图谱 |")
    _w("| Multimodal | 能融合文本、媒体元信息、结构化特征 | 当前 visual_features 只是媒体元信息，不是真实图像语义 |")
    _w("")

    # 7. 主要限制
    _w("## 7. 主要限制")
    _w("")
    _w("1. **样本量小**：79 条主视频，16 条 eval，所有指标波动大，不具备统计显著性。")
    _w("2. **无独立 test 集**：当前仅使用 train/eval 切分，无法做最终泛化评估。")
    _w("3. **伪标签**：标签基于 interaction_score 分位数构造，不代表 CTR/CVR/完播/留存等真实业务目标。")
    _w("4. **部分字段规则生成**：5 张完全补齐表（raw_video_tag、raw_video_status_control、raw_chapter、raw_comment、raw_related_video）数据不代表真实分布。")
    _w("5. **多模态视觉分支**：visual_features 仅包含媒体元信息（封面尺寸、URL 存在性等），不包含真实图像语义特征。")
    _w("6. **指标不可用于正式业务结论**：当前所有对比结果仅支持流程级验证。")
    _w("")

    # 8. 后续改进假设
    _w("## 8. 后续改进假设")
    _w("")
    _w("1. **数据规模**：接入更大规模真实数据（1000+ 条），提升指标稳定性。")
    _w("2. **评估方式**：使用 train/val/test 三路切分或交叉验证，替代当前仅 train/eval 的方式。")
    _w("3. **Wide & Deep 交叉特征**：加强交叉特征选择，增加有效的 wide 侧信号。")
    _w("4. **GraphSAGE 图关系**：使用更真实的用户-视频、作者-视频、视频-视频关系，减少规则补齐边占比。")
    _w("5. **多模态视觉**：引入真实封面图像或视频帧特征（需用户明确要求并确认依赖）。")
    _w("6. **阈值选择**：统一做阈值选择与概率校准，改善 Precision/Recall 平衡。")
    _w("7. **在线验证**：后续开展在线或准在线 A/B 实验，验证模型在线收益。")
    _w("")

    # 9. 结论
    _w("## 9. 结论")
    _w("")
    _w("当前对比结果仅支持以下结论：")
    _w("")
    _w("1. ✅ **已跑通多模型流程**：DNN、Wide & Deep、GraphSAGE、Multimodal 四类模型均已实现最小可运行闭环。")
    _w("2. ✅ **已完成统一输出**：四个模型均按统一规范输出 metrics.json、predictions.csv、train_log.csv、model.pt 等文件。")
    _w("3. ✅ **已完成统一对比**：对比实验入口统一，可自动汇总指标、生成对比报告和图表。")
    _w("4. ❌ **不支持正式业务效果判断**：当前 sample0427 样本数据、伪标签和 16 条 eval 不足以支持任何正式推荐系统效果结论。")
    _w("")

    # 10. 图表索引
    _w("## 10. 图表索引")
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

    # 11. 分数分布简表
    _w("## 11. 分数分布概况")
    _w("")
    sd_header = "| 模型 | Score Min | Score Max | Score Mean | Score Std | Pred Positive Rate | Label Positive Rate |"
    sd_sep = "|---|---|---|---|---|---|---|"
    _w(sd_header)
    _w(sd_sep)
    for sd in score_dist_rows:
        _w(
            f"| {sd.get('model_name', 'N/A')} "
            f"| {fmt_metric(sd.get('score_min'))} "
            f"| {fmt_metric(sd.get('score_max'))} "
            f"| {fmt_metric(sd.get('score_mean'))} "
            f"| {fmt_metric(sd.get('score_std'))} "
            f"| {fmt_pct(sd.get('pred_positive_rate'))} "
            f"| {fmt_pct(sd.get('label_positive_rate'))} |"
        )
    _w("")

    # 12. 输出文件
    _w("## 12. 输出文件清单")
    _w("")
    output_dir_path = rel_output_dir
    output_files = [
        "model_metrics_summary.csv",
        "model_metrics_summary.json",
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
        _w(f"- `{output_dir_path}/{fname}`")
    _w("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()