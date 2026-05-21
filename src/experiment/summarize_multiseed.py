"""DNN 多 seed 验证汇总脚本。

读取多个 DNN run 的 metrics.json 和 run_meta.json，
输出汇总报告到 outputs/validation/dnn_multiseed/real_raw_5000/<validation_run_id>/。

用法:
    D:/CodeData/software/Anaconda/Anaconda3/envs/ra/python.exe src/experiment/summarize_multiseed.py
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "dnn" / "real_raw_5000"
VALIDATION_BASE = (
    PROJECT_ROOT / "outputs" / "validation" / "dnn_multiseed" / "real_raw_5000"
)

# 10 个 run 的清单
RUNS: list[dict[str, Any]] = [
    # Baseline
    {"seed": 2024, "config_type": "baseline", "run_id": "202605201605"},
    {"seed": 2025, "config_type": "baseline", "run_id": "202605201606"},
    {"seed": 2026, "config_type": "baseline", "run_id": "202605201607"},
    {"seed": 2027, "config_type": "baseline", "run_id": "202605201639"},
    {"seed": 2028, "config_type": "baseline", "run_id": "202605201646"},
    # Tuned
    {"seed": 2024, "config_type": "tuned", "run_id": "202605201650"},
    {"seed": 2025, "config_type": "tuned", "run_id": "202605201647"},
    {"seed": 2026, "config_type": "tuned", "run_id": "202605201652"},
    {"seed": 2027, "config_type": "tuned", "run_id": "202605201654"},
    {"seed": 2028, "config_type": "tuned", "run_id": "202605201656"},
]

METRICS_TO_EXTRACT = [
    "test_auc",
    "test_f1",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "val_auc",
    "val_f1",
    "val_loss",
]

META_TO_EXTRACT = ["best_epoch", "num_params"]


def extract_run(run_id: str) -> dict[str, Any] | None:
    """读取单个 run 的指标，失败返回 None。"""
    run_dir = OUTPUT_BASE / run_id
    metrics_path = run_dir / "metrics.json"
    meta_path = run_dir / "run_meta.json"

    if not run_dir.is_dir():
        return None
    if not metrics_path.is_file():
        return None
    if not meta_path.is_file():
        return None

    result: dict[str, Any] = {"run_id": run_id, "status": "completed"}

    # 读取 metrics.json
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    vm = metrics.get("val_metrics", {})
    tm = metrics.get("test_metrics", {})
    result["val_auc"] = vm.get("auc")
    result["val_f1"] = vm.get("f1")
    result["val_loss"] = vm.get("eval_loss")
    result["test_auc"] = tm.get("auc")
    result["test_f1"] = tm.get("f1")
    result["test_accuracy"] = tm.get("accuracy")
    result["test_precision"] = tm.get("precision")
    result["test_recall"] = tm.get("recall")

    # 读取 run_meta.json
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    result["best_epoch"] = meta.get("best_epoch")
    result["num_params"] = meta.get("num_params")
    result["best_val_loss"] = meta.get("best_val_loss")

    return result


def compute_summary(values: list[float]) -> dict[str, float]:
    """计算一组数值的描述性统计。"""
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "min": None, "max": None,
                "median": None, "count": 0}
    mean = sum(values) / n
    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = variance ** 0.5
    else:
        std = 0.0
    sorted_vals = sorted(values)
    median = sorted_vals[n // 2] if n % 2 == 1 else (
        sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
    return {
        "mean": round(mean, 6),
        "std": round(std, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "median": round(median, 6),
        "count": n,
    }


def main():
    validation_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = VALIDATION_BASE / validation_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 提取所有 run 的指标
    manifest = []
    for run in RUNS:
        seed = run["seed"]
        config_type = run["config_type"]
        run_id = run["run_id"]
        extracted = extract_run(run_id)
        if extracted is None:
            manifest.append({
                "config_type": config_type,
                "seed": seed,
                "run_id": run_id,
                "output_path": str(OUTPUT_BASE / run_id),
                "status": "failed",
                "val_auc": "",
                "val_f1": "",
                "val_loss": "",
                "test_auc": "",
                "test_f1": "",
                "test_accuracy": "",
                "test_precision": "",
                "test_recall": "",
                "best_epoch": "",
                "best_val_loss": "",
                "num_params": "",
            })
        else:
            manifest.append({
                "config_type": config_type,
                "seed": seed,
                "run_id": run_id,
                "output_path": str(OUTPUT_BASE / run_id),
                "status": "completed",
                "val_auc": extracted.get("val_auc", ""),
                "val_f1": extracted.get("val_f1", ""),
                "val_loss": extracted.get("val_loss", ""),
                "test_auc": extracted.get("test_auc", ""),
                "test_f1": extracted.get("test_f1", ""),
                "test_accuracy": extracted.get("test_accuracy", ""),
                "test_precision": extracted.get("test_precision", ""),
                "test_recall": extracted.get("test_recall", ""),
                "best_epoch": extracted.get("best_epoch", ""),
                "best_val_loss": extracted.get("best_val_loss", ""),
                "num_params": extracted.get("num_params", ""),
            })

    # 2. 输出 validation_config.json
    validation_config = {
        "validation_run_id": validation_run_id,
        "project_root": str(PROJECT_ROOT),
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "description": "DNN 多 seed 验证 — Batch 13B",
        "seeds": [2024, 2025, 2026, 2027, 2028],
        "configs": {
            "baseline": {
                "source": "configs/models/dnn.yaml",
                "hidden_units": [64, 32],
                "dropout": 0.3,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "batch_size": 128,
            },
            "tuned": {
                "source": "outputs/tuning/dnn/real_raw_5000/20260514_191432/best_config.yaml",
                "hidden_units": [256, 128],
                "dropout": 0.314,
                "learning_rate": 0.00808,
                "weight_decay": 0.000542,
                "batch_size": 256,
            },
        },
        "num_total_runs": len(RUNS),
        "num_completed": sum(1 for m in manifest if m["status"] == "completed"),
        "num_failed": sum(1 for m in manifest if m["status"] == "failed"),
    }

    vc_path = output_dir / "validation_config.json"
    with open(vc_path, "w", encoding="utf-8") as f:
        json.dump(validation_config, f, ensure_ascii=False, indent=2)
    print(f"validation_config.json -> {vc_path}")

    # 3. 输出 runs_manifest.csv
    manifest_fields = [
        "config_type", "seed", "run_id", "output_path", "status",
        "val_auc", "val_f1", "val_loss", "test_auc", "test_f1",
        "test_accuracy", "test_precision", "test_recall",
        "best_epoch", "best_val_loss", "num_params",
    ]
    manifest_path = output_dir / "runs_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        for row in manifest:
            writer.writerow(row)
    print(f"runs_manifest.csv -> {manifest_path}")

    # 4. 按 config_type 分组汇总
    completed_baseline = [m for m in manifest
                          if m["config_type"] == "baseline" and m["status"] == "completed"]
    completed_tuned = [m for m in manifest
                       if m["config_type"] == "tuned" and m["status"] == "completed"]

    summary_rows = []
    for config_type, group in [("baseline", completed_baseline), ("tuned", completed_tuned)]:
        for metric in METRICS_TO_EXTRACT:
            values = []
            for row in group:
                val = row.get(metric)
                if val != "" and val is not None:
                    values.append(float(val))
            stats = compute_summary(values)
            summary_rows.append({
                "config_type": config_type,
                "metric": metric,
                "mean": stats["mean"],
                "std": stats["std"],
                "min": stats["min"],
                "max": stats["max"],
                "median": stats["median"],
                "count": stats["count"],
            })

    # 5. 输出 multiseed_summary.csv
    summary_fields = ["config_type", "metric", "mean", "std", "min", "max", "median", "count"]
    summary_path = output_dir / "multiseed_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"multiseed_summary.csv -> {summary_path}")

    # 6. 判断结论
    bl_test_auc_vals = [
        float(m["test_auc"]) for m in completed_baseline
        if m.get("test_auc") != "" and m["test_auc"] is not None
    ]
    tn_test_auc_vals = [
        float(m["test_auc"]) for m in completed_tuned
        if m.get("test_auc") != "" and m["test_auc"] is not None
    ]

    bl_stats = compute_summary(bl_test_auc_vals)
    tn_stats = compute_summary(tn_test_auc_vals)

    # 判断逻辑
    conclusion = ""
    details = {}
    if tn_stats["count"] < 3 or bl_stats["count"] < 3:
        conclusion = "INSUFFICIENT_RUNS"
        details = {
            "reason": "completed runs < 3 for one or both groups",
            "baseline_completed": bl_stats["count"],
            "tuned_completed": tn_stats["count"],
        }
    elif tn_stats["mean"] <= bl_stats["mean"]:
        conclusion = "KEEP_BASELINE"
        details = {
            "reason": "tuned mean test_auc <= baseline mean test_auc",
            "baseline_mean": bl_stats["mean"],
            "tuned_mean": tn_stats["mean"],
        }
    else:
        if tn_stats["std"] > 2 * bl_stats["std"]:
            conclusion = "KEEP_BASELINE_TUNED_UNSTABLE"
            details = {
                "reason": "tuned std > 2x baseline std (unstable)",
                "baseline_mean": bl_stats["mean"],
                "baseline_std": bl_stats["std"],
                "tuned_mean": tn_stats["mean"],
                "tuned_std": tn_stats["std"],
            }
        else:
            lower_bound = tn_stats["mean"] - tn_stats["std"]
            upper_bound = bl_stats["mean"] + bl_stats["std"]
            if lower_bound > upper_bound:
                conclusion = "TUNED_BETTER"
                details = {
                    "reason": "tuned mean - std > baseline mean + std (beyond noise)",
                    "baseline_mean": bl_stats["mean"],
                    "baseline_std": bl_stats["std"],
                    "tuned_mean": tn_stats["mean"],
                    "tuned_std": tn_stats["std"],
                    "tuned_lower_bound": lower_bound,
                    "baseline_upper_bound": upper_bound,
                }
            else:
                conclusion = "KEEP_BASELINE_WITHIN_NOISE"
                details = {
                    "reason": "tuned mean > baseline mean but within noise range",
                    "baseline_mean": bl_stats["mean"],
                    "baseline_std": bl_stats["std"],
                    "tuned_mean": tn_stats["mean"],
                    "tuned_std": tn_stats["std"],
                    "tuned_lower_bound": lower_bound,
                    "baseline_upper_bound": upper_bound,
                }

    # 7. 输出 multiseed_report.json
    report = {
        "validation_run_id": validation_run_id,
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline": {
            "config_type": "baseline",
            "num_runs": len(completed_baseline),
            "config": {
                "source": "configs/models/dnn.yaml",
                "hidden_units": [64, 32],
                "dropout": 0.3,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "batch_size": 128,
            },
            "summary": {},
        },
        "tuned": {
            "config_type": "tuned",
            "num_runs": len(completed_tuned),
            "config": {
                "source": "outputs/tuning/dnn/real_raw_5000/20260514_191432/best_config.yaml",
                "hidden_units": [256, 128],
                "dropout": 0.314,
                "learning_rate": 0.00808,
                "weight_decay": 0.000542,
                "batch_size": 256,
            },
            "summary": {},
        },
        "comparison": {
            "primary_metric": "test_auc",
            "baseline_mean": bl_stats["mean"],
            "baseline_std": bl_stats["std"],
            "tuned_mean": tn_stats["mean"],
            "tuned_std": tn_stats["std"],
            "diff_mean": round(tn_stats["mean"] - bl_stats["mean"], 6) if (
                tn_stats["mean"] is not None and bl_stats["mean"] is not None) else None,
            "conclusion": conclusion,
            "details": details,
        },
        "warnings": [
            "所有指标基于离线代理标签（interaction_score 分位数二分类），不代表真实线上收益",
            "只使用了描述性统计，未做正式统计检验",
        ],
    }

    # 填写各配置的汇总
    for sr in summary_rows:
        ct = sr["config_type"]
        metric = sr["metric"]
        report[ct]["summary"][metric] = {
            "mean": sr["mean"],
            "std": sr["std"],
            "min": sr["min"],
            "max": sr["max"],
            "median": sr["median"],
            "count": sr["count"],
        }

    report_path = output_dir / "multiseed_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"multiseed_report.json -> {report_path}")

    # 8. 输出 multiseed_report.md
    md_lines = []
    md_lines.append("# DNN 多 seed 验证报告\n")
    md_lines.append(f"> validation_run_id: {validation_run_id}")
    md_lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append(f"> 项目根目录: `{PROJECT_ROOT}`\n")

    md_lines.append("## 1. 验证配置\n")
    md_lines.append(f"| 项目 | Baseline | Tuned |")
    md_lines.append(f"|------|----------|-------|")
    md_lines.append(f"| 配置来源 | `configs/models/dnn.yaml` | `best_config.yaml` |")
    md_lines.append(f"| hidden_units | [64, 32] | [256, 128] |")
    md_lines.append(f"| dropout | 0.3 | 0.314 |")
    md_lines.append(f"| learning_rate | 0.001 | 0.00808 |")
    md_lines.append(f"| weight_decay | 0.0001 | 0.000542 |")
    md_lines.append(f"| batch_size | 128 | 256 |")
    md_lines.append(f"| num_params | 24,581 | 65,861 |")
    md_lines.append(f"| seeds | 2024~2028 | 2024~2028 |")
    md_lines.append(f"| 计划训练次数 | 5 | 5 |")
    md_lines.append(f"| 完成次数 | {len(completed_baseline)} | {len(completed_tuned)} |")
    md_lines.append("")

    md_lines.append("## 2. Runs Manifest\n")
    md_lines.append("| config_type | seed | run_id | status | val_auc | val_loss | test_auc | test_f1 | best_epoch | num_params |")
    md_lines.append("|-------------|------|--------|--------|---------|----------|---------|----------|------------|------------|")
    for m in manifest:
        va = f"{float(m['val_auc']):.4f}" if m["val_auc"] != "" else "-"
        vl = f"{float(m['val_loss']):.6f}" if m["val_loss"] != "" else "-"
        ta = f"{float(m['test_auc']):.4f}" if m["test_auc"] != "" else "-"
        tf = f"{float(m['test_f1']):.4f}" if m["test_f1"] != "" else "-"
        be = str(m["best_epoch"]) if m["best_epoch"] != "" else "-"
        np_ = str(m["num_params"]) if m["num_params"] != "" else "-"
        md_lines.append(
            f"| {m['config_type']} | {m['seed']} | {m['run_id']} | {m['status']} | "
            f"{va} | {vl} | {ta} | {tf} | {be} | {np_} |"
        )
    md_lines.append("")

    md_lines.append("## 3. Baseline 汇总\n")
    md_lines.append("| 指标 | Mean | Std | Min | Max | Median |")
    md_lines.append("|------|------|-----|-----|-----|--------|")
    for sr in summary_rows:
        if sr["config_type"] == "baseline":
            m = sr["metric"]
            mn = f"{sr['mean']:.4f}" if sr["mean"] is not None else "-"
            sd = f"{sr['std']:.4f}" if sr["std"] is not None else "-"
            lo = f"{sr['min']:.4f}" if sr["min"] is not None else "-"
            hi = f"{sr['max']:.4f}" if sr["max"] is not None else "-"
            md = f"{sr['median']:.4f}" if sr["median"] is not None else "-"
            md_lines.append(f"| {m} | {mn} | {sd} | {lo} | {hi} | {md} |")
    md_lines.append("")

    md_lines.append("## 4. Tuned 汇总\n")
    md_lines.append("| 指标 | Mean | Std | Min | Max | Median |")
    md_lines.append("|------|------|-----|-----|-----|--------|")
    for sr in summary_rows:
        if sr["config_type"] == "tuned":
            m = sr["metric"]
            mn = f"{sr['mean']:.4f}" if sr["mean"] is not None else "-"
            sd = f"{sr['std']:.4f}" if sr["std"] is not None else "-"
            lo = f"{sr['min']:.4f}" if sr["min"] is not None else "-"
            hi = f"{sr['max']:.4f}" if sr["max"] is not None else "-"
            md_ = f"{sr['median']:.4f}" if sr["median"] is not None else "-"
            md_lines.append(f"| {m} | {mn} | {sd} | {lo} | {hi} | {md_} |")
    md_lines.append("")

    md_lines.append("## 5. Baseline vs Tuned Test AUC 对比\n")
    md_lines.append("| | Mean | Std | Min | Max | Median |")
    md_lines.append("|--|------|-----|-----|-----|--------|")
    bl_ta = next((sr for sr in summary_rows
                  if sr["config_type"] == "baseline" and sr["metric"] == "test_auc"), None)
    tn_ta = next((sr for sr in summary_rows
                  if sr["config_type"] == "tuned" and sr["metric"] == "test_auc"), None)

    if bl_ta:
        md_lines.append(
            f"| Baseline | {bl_ta['mean']:.4f} | {bl_ta['std']:.4f} | "
            f"{bl_ta['min']:.4f} | {bl_ta['max']:.4f} | {bl_ta['median']:.4f} |"
        )
    if tn_ta:
        md_lines.append(
            f"| Tuned | {tn_ta['mean']:.4f} | {tn_ta['std']:.4f} | "
            f"{tn_ta['min']:.4f} | {tn_ta['max']:.4f} | {tn_ta['median']:.4f} |"
        )
    diff = report["comparison"]["diff_mean"]
    if diff is not None:
        md_lines.append(f"| 差异 (tuned - baseline) | {diff:.4f} | — | — | — | — |")
    md_lines.append("")

    md_lines.append("## 6. 判断结论\n")
    conclusion_texts = {
        "INSUFFICIENT_RUNS": "❌ 完成次数不足，无法判断",
        "KEEP_BASELINE": "✅ 保留 baseline — tuned mean test_auc <= baseline mean test_auc",
        "KEEP_BASELINE_WITHIN_NOISE": "✅ 保留 baseline — tuned 均值略高但在随机波动范围内",
        "KEEP_BASELINE_TUNED_UNSTABLE": "✅ 保留 baseline — tuned 配置更不稳定 (std > 2×)",
        "TUNED_BETTER": "🔄 tuned 配置稳定优于 baseline — 可考虑替换 baseline",
    }
    md_lines.append(f"**结论：{conclusion_texts.get(conclusion, conclusion)}**\n")
    md_lines.append("### 判断详情\n")
    for k, v in details.items():
        md_lines.append(f"- **{k}**: {v}")
    md_lines.append("")

    md_lines.append("## 7. 注意事项\n")
    md_lines.append("1. 所有指标基于离线代理标签（interaction_score 分位数二分类），不代表真实线上收益。")
    md_lines.append("2. 只使用了描述性统计（mean/std/min/max/median），未做正式统计检验。")
    md_lines.append("3. 样本量较小（每组 5 个 run），统计效力有限。")
    md_lines.append("4. 如果 tuned 配置被判定为更好，仍需在实际推荐场景中验证。")
    md_lines.append("")

    md_lines.append("---\n")
    md_lines.append(f"*报告由 summarize_multiseed.py 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    md_report_path = output_dir / "multiseed_report.md"
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"multiseed_report.md -> {md_report_path}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("汇总完成")
    print(f"  validation_run_id: {validation_run_id}")
    print(f"  输出目录: {output_dir}")
    print(f"  Baseline completed: {len(completed_baseline)}/5")
    print(f"  Tuned completed: {len(completed_tuned)}/5")
    print(f"  结论: {conclusion_texts.get(conclusion, conclusion)}")
    print("=" * 60)


if __name__ == "__main__":
    main()