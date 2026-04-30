"""模型结果收集脚本

读取四个模型的 metrics.json 和 predictions.csv，输出汇总表和 Top-K 二次校验。
不生成图表和报告（由 run_comparison.py 完成）。

用法:
    D:/CodeData/software/Anaconda/Anaconda3/envs/ra/python.exe src/experiment/collect_results.py --config configs/common/comparison.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

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
from utils.config import load_config  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger("collect_results")


def main() -> None:
    parser = argparse.ArgumentParser(description="模型结果收集")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/common/comparison.yaml",
        help="对比实验配置文件路径",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    project_root = Path(_project_root)
    output_root = project_root / config.get("output_root", "outputs/comparison")
    comparison_run_id = datetime.now().strftime("%Y%m%d%H%M")
    output_dir = output_root / comparison_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_config_path = project_root / config.get("metrics_config_path", "configs/common/metrics.yaml")
    metrics_config = load_config(metrics_config_path)
    k_values = metrics_config.get("k_values", config.get("k_values", [5, 10, 20]))

    model_runs = config.get("model_runs", {})
    required_cols = config.get("required_prediction_columns", [
        "sample_id", "video_id", "label", "score", "pred", "split", "model_name",
    ])

    logger.info("===== 模型结果收集 =====")

    # 1. 质量检查
    all_predictions: dict[str, pd.DataFrame] = {}
    quality_results = []
    for model_key, model_cfg in model_runs.items():
        model_name = model_cfg["model_name"]
        run_id = model_cfg["run_id"]
        pred_path = project_root / model_cfg["output_dir"] / "predictions.csv"
        if not pred_path.exists():
            quality_results.append({"model_name": model_name, "file_exists": False})
            continue
        df = pd.read_csv(pred_path)
        qc = check_predictions_quality(df, required_cols, model_name, run_id)
        quality_results.append(qc)
        all_predictions[model_key] = df

    quality_df = pd.DataFrame(quality_results)
    quality_df.to_csv(output_dir / "model_prediction_quality_check.csv", index=False)
    with open(output_dir / "model_prediction_quality_check.json", "w", encoding="utf-8") as f:
        json.dump(quality_results, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"质量检查完成: {len(quality_results)} 个模型")

    # 2. 指标汇总
    summary_rows = []
    for model_key, model_cfg in model_runs.items():
        metrics = collect_model_metrics(model_key, model_cfg, project_root)
        summary_rows.append(metrics)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "model_metrics_summary.csv", index=False)
    with open(output_dir / "model_metrics_summary.json", "w", encoding="utf-8") as f:
        json_summary = []
        for _, row in summary_df.iterrows():
            entry = {}
            for c in summary_df.columns:
                v = row[c]
                if isinstance(v, (pd.Int64Dtype,)):
                    v = int(v)
                elif isinstance(v, (pd.Float64Dtype,)):
                    v = float(v)
                elif isinstance(v, Path):
                    v = str(v)
                entry[c] = v
            json_summary.append(entry)
        json.dump(json_summary, f, ensure_ascii=False, indent=2)
    logger.info(f"指标汇总完成: {len(summary_rows)} 个模型")

    # 3. Top-K 对比
    topk_rows = []
    for model_key, model_cfg in model_runs.items():
        if model_key not in all_predictions:
            continue
        df = all_predictions[model_key]
        rows = compute_topk_comparison(df, model_cfg["model_name"], model_cfg["run_id"], k_values)
        topk_rows.extend(rows)

    topk_df = pd.DataFrame(topk_rows)
    topk_df.to_csv(output_dir / "topk_comparison.csv", index=False)
    logger.info(f"Top-K 对比完成: {len(topk_rows)} 行")

    # 4. 分数分布
    score_dist_rows = []
    for model_key, model_cfg in model_runs.items():
        if model_key not in all_predictions:
            continue
        dist = compute_score_distribution(all_predictions[model_key], model_cfg["model_name"], model_cfg["run_id"])
        score_dist_rows.append(dist)

    score_dist_df = pd.DataFrame(score_dist_rows)
    score_dist_df.to_csv(output_dir / "model_score_distribution.csv", index=False)
    logger.info(f"分数分布完成: {len(score_dist_rows)} 个模型")

    # ── 5. 生成 comparison_run_meta.json ──────────────────
    from datetime import datetime as _dt
    started_at = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    input_metrics_paths = [
        str(project_root / model_cfg["output_dir"] / "metrics.json")
        for model_cfg in model_runs.values()
    ]
    input_predictions_paths = [
        str(project_root / model_cfg["output_dir"] / "predictions.csv")
        for model_cfg in model_runs.values()
    ]
    output_files_list = sorted([
        str(f.relative_to(output_dir))
        for f in output_dir.iterdir() if f.is_file()
    ])
    run_meta = {
        "comparison_run_id": comparison_run_id,
        "output_dir": str(output_dir),
        "started_at": started_at,
        "finished_at": _dt.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": args.config,
        "model_runs": {k: {"run_id": v["run_id"], "output_dir": v["output_dir"]} for k, v in model_runs.items()},
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

    logger.info("===== 收集完成 =====")


if __name__ == "__main__":
    main()