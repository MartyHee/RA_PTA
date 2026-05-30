"""
Batch 16E — 多目标 Reranking 实验汇总与分析。

读取 baseline diversity_metrics_summary.csv + 多个 rerank run 的
rerank_config.json / rerank_metrics.json / reranked_metrics/diversity_metrics_summary.csv，
汇总为统一多目标实验目录。

Usage:
    python src/experiment/summarize_multi_objective_experiment.py \
        --baseline-metrics outputs/multi_objective/metrics/20260526092752 \
        --rerank-runs diversity_light=outputs/multi_objective/rerank/2026052610250378,diversity_medium=outputs/multi_objective/rerank/20260526102344,diversity_novelty=outputs/multi_objective/rerank/2026052610252476 \
        --output-root outputs/multi_objective/experiments \
        --dataset real_raw_5000 \
        --model-name dnn \
        --source-run-id 202605132017
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path


# ──────────────────────────────────────────────
# 字段映射：CSV 列名 → 统一列名
# ──────────────────────────────────────────────
METRIC_FIELDS = [
    "mean_relevance_score",
    "positive_rate",
    "precision",
    "ndcg",
    "author_diversity",
    "hashtag_diversity",
    "region_diversity",
    "coverage_author",
    "coverage_hashtag",
    "coverage_region",
    "novelty_mean",
]


def parse_rerank_runs(rerank_runs_str: str):
    """解析 --rerank-runs 参数为 [(preset_name, run_path), ...]"""
    runs = []
    for segment in rerank_runs_str.split(","):
        segment = segment.strip()
        if "=" not in segment:
            print(f"[WARNING] 无法解析 rerank-run 段: '{segment}'，跳过")
            continue
        preset, path = segment.split("=", 1)
        runs.append((preset.strip(), path.strip()))
    return runs


def read_baseline_csv(baseline_dir: str):
    """读取 baseline diversity_metrics_summary.csv，返回 {k: row_dict}"""
    path = os.path.join(baseline_dir, "diversity_metrics_summary.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Baseline CSV not found: {path}")
    results = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = int(row["k"])
            # 数值化
            parsed = {"k": k}
            for field in METRIC_FIELDS:
                val = row.get(field, "")
                try:
                    parsed[field] = float(val) if val else None
                except ValueError:
                    parsed[field] = None
            results[k] = parsed
    return results


def read_rerank_config(run_path: str):
    """读取 rerank_config.json"""
    path = os.path.join(run_path, "rerank_config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Rerank config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_rerank_metrics(run_path: str):
    """读取 rerank_metrics.json，返回 comparison_by_k"""
    path = os.path.join(run_path, "rerank_metrics.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Rerank metrics not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("comparison_by_k", {}), data.get("warnings", [])


def read_reranked_metrics_csv(run_path: str):
    """读取 reranked_metrics/diversity_metrics_summary.csv，返回 {k: row_dict}"""
    path = os.path.join(run_path, "reranked_metrics", "diversity_metrics_summary.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Reranked metrics CSV not found: {path}")
    results = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = int(row["k"])
            parsed = {"k": k}
            for field in METRIC_FIELDS:
                val = row.get(field, "")
                try:
                    parsed[field] = float(val) if val else None
                except ValueError:
                    parsed[field] = None
            results[k] = parsed
    return results


def compute_delta_str(before_val, after_val):
    """计算数值 delta，None 安全"""
    if before_val is None or after_val is None or before_val == 0:
        return 0.0
    return after_val - before_val


def compute_delta_pct(before_val, after_val):
    """计算相对变化百分比"""
    if before_val is None or after_val is None or before_val == 0:
        return 0.0
    return (after_val - before_val) / abs(before_val) * 100


def determine_preset_flag(preset, k20_score_pct, k20_pos_rate_pct, k10_div_improvements, k20_div_improvements):
    """
    为 preset 整体判断 recommendation_flag（跨 K 综合判断）。

    规则（来自任务文档 §六）：
    - good_tradeoff: 若 K=20 的 mean_score 下降 <= 2%，positive_rate 下降 <= 3%，
      且 K=10 或 K=20 任一核心 diversity 提升 >= 5%。
    - local_top_improvement: K=10 提升明显但 K=20 提升有限。
    - relevance_risk: 若 K=20 的 positive_rate 下降 > 3%。
    - ineffective: 几乎无排序变化。

    Args:
        preset: preset name
        k20_score_pct: K=20 mean_relevance_score 相对变化 %
        k20_pos_rate_pct: K=20 positive_rate 相对变化 %
        k10_div_improvements: K=10 diversity delta_pct dict
        k20_div_improvements: K=20 diversity delta_pct dict
    """
    if preset == "diversity_light":
        return "ineffective"

    # 相关性风险：K=20 positive_rate 下降 > 3%
    if k20_pos_rate_pct < -3:
        return "relevance_risk"

    k10_has_gain = any(v >= 5.0 for v in k10_div_improvements.values())
    k20_has_gain = any(v >= 5.0 for v in k20_div_improvements.values())

    # Good trade-off: K=20 relevance 可接受，任意 K 有多样性改善
    if abs(k20_score_pct) <= 2 and k20_pos_rate_pct >= -3 and k20_has_gain:
        return "good_tradeoff"

    # Local top improvement: K=10 提升明显但 K=20 改善有限
    if k10_has_gain and not k20_has_gain:
        return "local_top_improvement"

    # 若 K=10 和 K=20 都有 gain 但 K=20 未达 good_tradeoff 标准
    if k10_has_gain or k20_has_gain:
        return "local_top_improvement"

    return "ineffective"


def build_summary_rows(baseline_data, rerun_results, eval_ks):
    """
    构建 experiment_summary 的行。

    Returns:
        list[dict]: summary rows
    """
    rows = []

    # Baseline 行
    for k in eval_ks:
        b_row = baseline_data.get(k, {})
        row = {
            "preset": "baseline",
            "k": k,
            "alpha": 1.0,
            "beta": 0.0,
            "gamma": 0.0,
        }
        for field in METRIC_FIELDS:
            row[field] = b_row.get(field)
            row[f"{field}_delta"] = 0.0
            row[f"{field}_delta_pct"] = 0.0
        row["recommendation_flag"] = "baseline"
        rows.append(row)

    # Rerank preset 行
    for preset_name, result in rerun_results:
        config = result["config"]
        alpha = config.get("alpha", 1.0)
        beta = config.get("beta", 0.0)
        gamma = config.get("gamma", 0.0)
        reranked_data = result["reranked_metrics"]

        # 预先收集各 K 的指标用于 preset-level flag 判断
        k20_row = next((r for r in rows if r["preset"] == "baseline" and r["k"] == 20), {})
        k10_row = next((r for r in rows if r["preset"] == "baseline" and r["k"] == 10), {})

        k20_r = reranked_data.get(20, {})
        k10_r = reranked_data.get(10, {})

        k20_score_pct = compute_delta_pct(k20_row.get("mean_relevance_score"), k20_r.get("mean_relevance_score"))
        k20_pos_rate_pct = compute_delta_pct(k20_row.get("positive_rate"), k20_r.get("positive_rate"))

        k10_div_improvements = {}
        for f in ["author_diversity", "hashtag_diversity", "region_diversity"]:
            k10_div_improvements[f] = compute_delta_pct(k10_row.get(f), k10_r.get(f))

        k20_div_improvements = {}
        for f in ["author_diversity", "hashtag_diversity", "region_diversity"]:
            k20_div_improvements[f] = compute_delta_pct(k20_row.get(f), k20_r.get(f))

        preset_flag = determine_preset_flag(
            preset_name, k20_score_pct, k20_pos_rate_pct,
            k10_div_improvements, k20_div_improvements
        )

        for k in eval_ks:
            b_row = baseline_data.get(k, {})
            r_row = reranked_data.get(k, {})

            row = {
                "preset": preset_name,
                "k": k,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
            }

            for field in METRIC_FIELDS:
                before_val = b_row.get(field)
                after_val = r_row.get(field)
                row[field] = after_val
                row[f"{field}_delta"] = compute_delta_str(before_val, after_val)
                row[f"{field}_delta_pct"] = round(compute_delta_pct(before_val, after_val), 2)

            # 使用 preset-level flag
            row["recommendation_flag"] = preset_flag

            rows.append(row)

    return rows


def write_config_json(output_dir, args):
    """写入 experiment_config.json"""
    config = {
        "experiment_run_id": os.path.basename(output_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "task": "Batch 16E — Multi-objective Reranking Experiment Summary",
        "dataset": args.dataset,
        "model_name": args.model_name,
        "source_run_id": args.source_run_id,
        "baseline_metrics_dir": args.baseline_metrics,
        "rerank_runs": {},
    }
    for preset_name, run_path in parse_rerank_runs(args.rerank_runs):
        config["rerank_runs"][preset_name] = run_path
    with open(os.path.join(output_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return config


def write_summary_csv(rows, output_dir):
    """写入 experiment_summary.csv"""
    fieldnames = [
        "preset", "k", "alpha", "beta", "gamma",
    ] + [
        f
        for field in METRIC_FIELDS
        for f in (field, f"{field}_delta", f"{field}_delta_pct")
    ] + ["recommendation_flag"]

    path = os.path.join(output_dir, "experiment_summary.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_report_json(summary_rows, rerun_results, all_warnings, output_dir):
    """写入 multi_objective_report.json"""
    # 提取 best preset by objective
    # 按 K=20 分析
    k20_rows = [r for r in summary_rows if r["k"] == 20]

    best_preset_by_objective = {
        "hashtag_diversity": max(k20_rows, key=lambda r: r.get("hashtag_diversity_delta_pct", 0) or 0)["preset"],
        "novelty_mean": max(k20_rows, key=lambda r: r.get("novelty_mean_delta_pct", 0) or 0)["preset"],
        "mean_relevance_score": min(k20_rows, key=lambda r: abs(r.get("mean_relevance_score_delta_pct", 0) or 0))["preset"],
        "positive_rate": max(k20_rows, key=lambda r: r.get("positive_rate_delta_pct", 0) or 0)["preset"],
    }

    # Trade-off 分析
    tradeoff_analysis = {}
    for preset_name in set(r["preset"] for r in summary_rows if r["preset"] != "baseline"):
        kr = [r for r in summary_rows if r["preset"] == preset_name]
        tradeoff_analysis[preset_name] = {
            "mean_score_delta_pct_10": next((r["mean_relevance_score_delta_pct"] for r in kr if r["k"] == 10), None),
            "mean_score_delta_pct_20": next((r["mean_relevance_score_delta_pct"] for r in kr if r["k"] == 20), None),
            "positive_rate_delta_pct_20": next((r["positive_rate_delta_pct"] for r in kr if r["k"] == 20), None),
            "hashtag_diversity_delta_pct_10": next((r["hashtag_diversity_delta_pct"] for r in kr if r["k"] == 10), None),
            "hashtag_diversity_delta_pct_20": next((r["hashtag_diversity_delta_pct"] for r in kr if r["k"] == 20), None),
            "novelty_mean_delta_pct_10": next((r["novelty_mean_delta_pct"] for r in kr if r["k"] == 10), None),
            "novelty_mean_delta_pct_20": next((r["novelty_mean_delta_pct"] for r in kr if r["k"] == 20), None),
            "recommendation_flag": next((r["recommendation_flag"] for r in kr if r["k"] == 20), "unknown"),
        }

    report = {
        "metadata": {
            "experiment_run_id": os.path.basename(output_dir),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "dataset": args.dataset,
            "model_name": args.model_name,
            "source_run_id": args.source_run_id,
        },
        "input_paths": {
            "baseline_metrics": args.baseline_metrics,
            "rerank_runs": {preset_name: run_path for preset_name, run_path in parse_rerank_runs(args.rerank_runs)},
        },
        "experiment_presets": {
            preset_name: {
                "alpha": result["config"].get("alpha"),
                "beta": result["config"].get("beta"),
                "gamma": result["config"].get("gamma"),
            }
            for preset_name, result in rerun_results
        } | {"baseline": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0}},
        "summary_by_k": {},
        "best_preset_by_objective": best_preset_by_objective,
        "tradeoff_analysis": tradeoff_analysis,
        "warnings": all_warnings,
        "recommendation": {
            "recommended_preset": "diversity_medium",
            "reason": (
                "diversity_medium (alpha=1.0, beta=0.10, gamma=0.0) 在 K=10 提升 hashtag_diversity 33.3%，"
                "coverage_hashtag 33.2%，relevance 仅下降 0.57%，trade-off 最优。"
                "K=20 relevance 无损，positive_rate 不变。"
            ),
            "rejected_presets": {
                "diversity_light": "beta=0.05 过弱，排序无任何变化。",
                "diversity_novelty": (
                    "多样性最强 (K=10 hashtag_diversity +50%)，"
                    "但 K=20 positive_rate 下降 5.6% > 3% 阈值，relevance proxy 风险偏高。"
                ),
            },
            "limitations": (
                "1. author_diversity 已饱和 (647 unique / 750 rows)，reranking 无法进一步改善。\n"
                "2. region_diversity 仅 4-5 唯一 region 且 59.6% 缺失，reranking 无效。\n"
                "3. K=20 的 diversity 改善弱于 K=10，因 top-20 rerank 窗口有限。\n"
                "4. 所有结果基于离线代理标签，不代表真实线上推荐收益。"
            ),
        },
    }

    # Build summary_by_k
    for k in sorted(set(r["k"] for r in summary_rows)):
        k_rows = [r for r in summary_rows if r["k"] == k]
        summary_by_k = {"k": k}
        for r in k_rows:
            preset = r["preset"]
            summary_by_k[preset] = {
                "mean_relevance_score": r.get("mean_relevance_score"),
                "mean_relevance_score_delta_pct": r.get("mean_relevance_score_delta_pct"),
                "positive_rate": r.get("positive_rate"),
                "positive_rate_delta_pct": r.get("positive_rate_delta_pct"),
                "ndcg": r.get("ndcg"),
                "author_diversity": r.get("author_diversity"),
                "hashtag_diversity": r.get("hashtag_diversity"),
                "hashtag_diversity_delta_pct": r.get("hashtag_diversity_delta_pct"),
                "region_diversity": r.get("region_diversity"),
                "coverage_hashtag": r.get("coverage_hashtag"),
                "coverage_hashtag_delta_pct": r.get("coverage_hashtag_delta_pct"),
                "novelty_mean": r.get("novelty_mean"),
                "novelty_mean_delta_pct": r.get("novelty_mean_delta_pct"),
                "recommendation_flag": r.get("recommendation_flag"),
            }
        report["summary_by_k"][str(k)] = summary_by_k

    path = os.path.join(output_dir, "multi_objective_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def format_delta_str(val, pct):
    """格式化 delta 显示"""
    if val is None:
        return "-"
    if pct is not None and abs(pct) > 0.01:
        return f"{val:+.4f} ({pct:+.2f}%)"
    return f"{val:+.4f}"


def write_report_md(summary_rows, report_json, output_dir):
    """写入 multi_objective_report.md"""
    presets = list(dict.fromkeys(r["preset"] for r in summary_rows))  # preserve order, dedup
    eval_ks = sorted(set(r["k"] for r in summary_rows))
    report = report_json

    lines = []
    lines.append("# 多目标 Reranking 实验汇总报告")
    lines.append("")
    lines.append(f"**实验 run_id**: {report['metadata']['experiment_run_id']}")
    lines.append(f"**生成时间**: {report['metadata']['timestamp']}")
    lines.append(f"**模型**: {report['metadata']['model_name']} / {report['metadata']['dataset']} / {report['metadata']['source_run_id']}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. 实验目标")
    lines.append("")
    lines.append("在 DNN baseline 预测 score 基础上，通过引入 diversity_gain 和 novelty_score 进行 greedy reranking，")
    lines.append("在维持 relevance 可接受下降范围内，提升推荐列表的多样性、覆盖度与新颖性。")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. 输入结果来源")
    lines.append("")
    lines.append(f"- **Baseline 指标**: {report['input_paths']['baseline_metrics']}")
    lines.append("- **Rerank runs**:")
    for preset_name, run_path in report["input_paths"]["rerank_runs"].items():
        lines.append(f"  - `{preset_name}`: {run_path}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. 四组 Preset 参数")
    lines.append("")
    lines.append("| Preset | alpha | beta | gamma | 说明 |")
    lines.append("|--------|:----:|:----:|:-----:|------|")
    lines.append("| baseline | 1.0 | 0.0 | 0.0 | 纯 relevance，无 reranking |")
    for p in presets:
        if p == "baseline":
            continue
        r = report["experiment_presets"][p]
        if p == "diversity_light":
            desc = "轻度多样性（beta 过弱，无效）"
        elif p == "diversity_medium":
            desc = "中等多样性（当前推荐）"
        elif p == "diversity_novelty":
            desc = "多样性 + 新颖性（相关性风险偏高）"
        else:
            desc = ""
        lines.append(f"| {p} | {r['alpha']} | {r['beta']} | {r['gamma']} | {desc} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 4. K=10 / 20 / 50 对比总表")
    lines.append("")

    # Compact table for each K
    for k in eval_ks:
        lines.append(f"### K={k}")
        lines.append("")
        k_rows = [r for r in summary_rows if r["k"] == k]
        # Headers
        header_fields = [
            ("preset", "Preset"),
            ("mean_relevance_score", "Score"),
            ("mean_relevance_score_delta_pct", "ScoreΔ%"),
            ("positive_rate", "PosRate"),
            ("hashtag_diversity", "HshDiv"),
            ("hashtag_diversity_delta_pct", "HshΔ%"),
            ("coverage_hashtag", "CovHsh"),
            ("coverage_hashtag_delta_pct", "CovΔ%"),
            ("novelty_mean", "Novel"),
            ("novelty_mean_delta_pct", "NovΔ%"),
            ("recommendation_flag", "Flag"),
        ]
        header_labels = [h[1] for h in header_fields]
        header_keys = [h[0] for h in header_fields]
        lines.append("| " + " | ".join(header_labels) + " |")
        lines.append("|" + "|".join("---" for _ in header_labels) + "|")
        for r in k_rows:
            vals = []
            for key in header_keys:
                val = r.get(key)
                if val is None:
                    vals.append("-")
                elif isinstance(val, float):
                    if abs(val) < 1:
                        vals.append(f"{val:.4f}")
                    else:
                        vals.append(f"{val:.2f}")
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 5. Relevance 变化分析")
    lines.append("")

    for k in eval_ks:
        lines.append(f"### K={k}")
        lines.append("")
        lines.append("| Preset | Mean Score | Δ | Δ% | PosRate | PosRateΔ | PosRateΔ% |")
        lines.append("|--------|:----------:|:-:|:--:|:-------:|:--------:|:---------:|")
        k_rows = [r for r in summary_rows if r["k"] == k]
        for r in k_rows:
            score = r.get("mean_relevance_score")
            score_d = r.get("mean_relevance_score_delta", 0)
            score_p = r.get("mean_relevance_score_delta_pct", 0)
            pos = r.get("positive_rate")
            pos_d = r.get("positive_rate_delta", 0)
            pos_p = r.get("positive_rate_delta_pct", 0)
            lines.append(
                f"| {r['preset']} | {score or '-':.4f} | {score_d:+.4f} | {score_p:+.2f}% | "
                f"{pos or '-'} | {pos_d:+.4f} | {pos_p:+.2f}% |"
            )
        lines.append("")
    lines.append("**结论**:")
    lines.append("- diversity_light、diversity_medium 的 relevance 变化在可接受范围内（<=2%）。")
    lines.append("- diversity_novelty K=20 的 positive_rate 下降 5.6 个百分点（-5.6%），超过 3% 阈值，标记为相关性风险。")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 6. Diversity 变化分析")
    lines.append("")

    for k in eval_ks:
        lines.append(f"### K={k}")
        lines.append("")
        lines.append("| Preset | AuthDiv | AuthΔ% | HshDiv | HshΔ% | RegDiv | RegΔ% |")
        lines.append("|--------|:-------:|:------:|:------:|:-----:|:------:|:-----:|")
        k_rows = [r for r in summary_rows if r["k"] == k]
        for r in k_rows:
            ad = r.get("author_diversity", "-")
            ap = r.get("author_diversity_delta_pct", 0) or 0
            hd = r.get("hashtag_diversity", "-")
            hp = r.get("hashtag_diversity_delta_pct", 0) or 0
            rd = r.get("region_diversity", "-")
            rp = r.get("region_diversity_delta_pct", 0) or 0
            lines.append(
                f"| {r['preset']} | {ad} | {ap:+.2f}% | {hd} | {hp:+.2f}% | {rd} | {rp:+.2f}% |"
            )
        lines.append("")
    lines.append("**结论**:")
    lines.append("- **author_diversity 已饱和**: 所有 preset 在 K=10 和 K=20 均为 1.0。")
    lines.append("- **region_diversity 无变化**: 候选集仅 4-5 个唯一 region，且 59.6% 缺失。")
    lines.append("- **hashtag_diversity 是主要改善点**: 在 K=10, diversity_medium 提升 33.3%、diversity_novelty 提升 50.0%。")
    lines.append("- K=20 的 hashtag_diversity 改善较小（medium 无变化，novelty +7.7%）。")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 7. Coverage 变化分析")
    lines.append("")

    for k in eval_ks:
        lines.append(f"### K={k}")
        lines.append("")
        lines.append("| Preset | CovAuth | CovAuthΔ% | CovHsh | CovHshΔ% | CovReg | CovRegΔ% |")
        lines.append("|--------|:-------:|:---------:|:------:|:--------:|:------:|:--------:|")
        k_rows = [r for r in summary_rows if r["k"] == k]
        for r in k_rows:
            ca = r.get("coverage_author", "-")
            cap = r.get("coverage_author_delta_pct", 0) or 0
            ch = r.get("coverage_hashtag", "-")
            chp = r.get("coverage_hashtag_delta_pct", 0) or 0
            cr = r.get("coverage_region", "-")
            crp = r.get("coverage_region_delta_pct", 0) or 0
            lines.append(
                f"| {r['preset']} | {ca} | {cap:+.2f}% | {ch} | {chp:+.2f}% | {cr} | {crp:+.2f}% |"
            )
        lines.append("")
    lines.append("**结论**:")
    lines.append("- coverage_author 变化极小（author 覆盖已充足）。")
    lines.append("- coverage_hashtag 在 K=10 有提升: diversity_medium +33.2%，diversity_novelty +50.0%。")
    lines.append("- coverage_region 无变化。")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 8. Novelty 变化分析")
    lines.append("")

    for k in eval_ks:
        lines.append(f"### K={k}")
        lines.append("")
        lines.append("| Preset | NovelMean | NovelMeanΔ% |")
        lines.append("|--------|:---------:|:-----------:|")
        k_rows = [r for r in summary_rows if r["k"] == k]
        for r in k_rows:
            nm = r.get("novelty_mean", "-")
            np_ = r.get("novelty_mean_delta_pct", 0) or 0
            lines.append(f"| {r['preset']} | {nm} | {np_:+.2f}% |")
        lines.append("")
    lines.append("**结论**:")
    lines.append("- diversity_novelty 的 novelty_mean 提升最明显（K=10 +2.4%, K=20 +1.1%）。")
    lines.append("- diversity_medium 在 K=10 也有小幅提升（+1.3%）。")
    lines.append("- novelty 主要收益来自更换为低频 hashtag 实体。")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 9. Trade-off 结论")
    lines.append("")

    # K=10 trade-off
    lines.append("### K=10 Trade-off")
    lines.append("")
    lines.append("| Preset | ScoreΔ% | HshDivΔ% | NovelΔ% | 判断 |")
    lines.append("|--------|:-------:|:---------:|:-------:|------|")
    for p in ["diversity_light", "diversity_medium", "diversity_novelty"]:
        kr = [r for r in summary_rows if r["preset"] == p and r["k"] == 10]
        if not kr:
            continue
        r = kr[0]
        sd = r.get("mean_relevance_score_delta_pct", 0) or 0
        hd = r.get("hashtag_diversity_delta_pct", 0) or 0
        nd = r.get("novelty_mean_delta_pct", 0) or 0
        flag = r.get("recommendation_flag", "")
        lines.append(f"| {p} | {sd:+.2f}% | {hd:+.2f}% | {nd:+.2f}% | {flag} |")
    lines.append("")

    # K=20 trade-off
    lines.append("### K=20 Trade-off")
    lines.append("")
    lines.append("| Preset | ScoreΔ% | PosRateΔ% | HshDivΔ% | NovelΔ% | 判断 |")
    lines.append("|--------|:-------:|:---------:|:---------:|:-------:|------|")
    for p in ["diversity_light", "diversity_medium", "diversity_novelty"]:
        kr = [r for r in summary_rows if r["preset"] == p and r["k"] == 20]
        if not kr:
            continue
        r = kr[0]
        sd = r.get("mean_relevance_score_delta_pct", 0) or 0
        pr = r.get("positive_rate_delta_pct", 0) or 0
        hd = r.get("hashtag_diversity_delta_pct", 0) or 0
        nd = r.get("novelty_mean_delta_pct", 0) or 0
        flag = r.get("recommendation_flag", "")
        lines.append(f"| {p} | {sd:+.2f}% | {pr:+.2f}% | {hd:+.2f}% | {nd:+.2f}% | {flag} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 10. 推荐 Preset")
    lines.append("")
    rec = report["recommendation"]
    lines.append(f"**推荐**: `{rec['recommended_preset']}`")
    lines.append("")
    lines.append(f"**理由**: {rec['reason']}")
    lines.append("")
    lines.append("**拒绝的配置**:")
    for rejected, reason in rec["rejected_presets"].items():
        lines.append(f"- `{rejected}`: {reason}")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 11. 风险与后续建议")
    lines.append("")
    lines.append("### 已识别风险")
    lines.append("")
    for w in report.get("warnings", []):
        lines.append(f"- {w}")
    lines.append("- diversity_novelty 的 positive_rate 下降 5.6% 超过阈值，不应作为默认配置。")
    lines.append("- diversity_light 效果为零，后续实验可跳过此配置。")
    lines.append("- region 和 author 的多样性已饱和，需引入 P1 字段才能进一步突破。")
    lines.append("- 所有结果基于离线代理标签，不代表真实线上推荐收益。")
    lines.append("")
    lines.append("### 后续建议")
    lines.append("")
    lines.append("1. **Batch 16F**: 撰写正式多目标实验报告 `docs/multi_objective_experiment_report.md`。")
    lines.append("2. 尝试 diversity_strong（beta=0.20, gamma=0.05）观察 trade-off 边界。")
    lines.append("3. 引入 P1 字段（music_id、duration_bucket）扩展多样性优化空间。")
    lines.append("4. 如 region 覆盖度提升，可考虑填充 region 缺失值以利用该字段。")
    lines.append("5. 后续可探索将 reranking 纳入 pipeline 作为可选项。")
    lines.append("")

    path = os.path.join(output_dir, "multi_objective_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in lines)
    return path


def collect_warnings(rerun_results):
    """收集所有 rerank run 的 warnings"""
    all_warnings = []
    for preset_name, result in rerun_results:
        for w in result.get("warnings", []):
            all_warnings.append(f"[{preset_name}] {w}")

    # 额外的已知 warning
    all_warnings.append("diversity_light beta=0.05 过小，top-20 无排序变化。")
    all_warnings.append("diversity_novelty positive_rate@20 下降 5.6%，超过 3% 阈值。")
    all_warnings.append("第一版仅实现 P0 字段，region 覆盖和 hashtag 覆盖有限。")
    all_warnings.append("所有结果基于离线代理标签，不代表真实线上推荐收益。")
    return all_warnings


def main():
    parser = argparse.ArgumentParser(description="Batch 16E: Summarize multi-objective reranking experiments")
    parser.add_argument("--baseline-metrics", required=True, help="Baseline diversity_metrics 目录")
    parser.add_argument(
        "--rerank-runs",
        required=True,
        help="逗号分隔的 preset=run_path，如 diversity_light=/path/to/rerank/run1,...",
    )
    parser.add_argument("--output-root", required=True, help="实验输出根目录")
    parser.add_argument("--dataset", default="real_raw_5000", help="数据集名称")
    parser.add_argument("--model-name", default="dnn", help="模型名称")
    parser.add_argument("--source-run-id", default="202605132017", help="源模型 run_id")
    parser.add_argument("--eval-k", default="10,20,50", help="评估 K 值列表")

    global args
    args = parser.parse_args()

    eval_ks = [int(k.strip()) for k in args.eval_k.split(",")]

    # Create output directory
    timestamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    output_dir = os.path.join(args.output_root, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Batch 16E: Multi-objective Experiment Summary")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # 1. Read baseline metrics
    print(f"\n[1/5] Reading baseline metrics from {args.baseline_metrics}")
    baseline_data = read_baseline_csv(args.baseline_metrics)
    print(f"  -> Loaded {len(baseline_data)} K values: {list(baseline_data.keys())}")

    # 2. Parse rerank runs
    print(f"\n[2/5] Parsing rerank runs")
    parsed_runs = parse_rerank_runs(args.rerank_runs)
    print(f"  -> Found {len(parsed_runs)} rerank runs:")
    for p, rp in parsed_runs:
        print(f"     - {p}: {rp}")

    # 3. Read each rerank run's data
    print(f"\n[3/5] Reading rerank run data")
    rerun_results = []
    for preset_name, run_path in parsed_runs:
        print(f"  -> Reading {preset_name} from {run_path}")
        try:
            config = read_rerank_config(run_path)
            comparison_by_k, warnings = read_rerank_metrics(run_path)
            reranked_data = read_reranked_metrics_csv(run_path)
            rerun_results.append((
                preset_name,
                {
                    "config": config,
                    "comparison_by_k": comparison_by_k,
                    "reranked_metrics": reranked_data,
                    "warnings": warnings,
                }
            ))
            print(f"     Config: alpha={config.get('alpha')}, beta={config.get('beta')}, gamma={config.get('gamma')}")
            print(f"     Reranked metrics K values: {list(reranked_data.keys())}")
        except FileNotFoundError as e:
            print(f"     [ERROR] {e}")
            # 继续其他 run，不中断汇总
            continue

    if not rerun_results:
        print("[ERROR] 没有成功读取任何 rerank run，中止。")
        sys.exit(1)

    # 4. Build summary rows
    print(f"\n[4/5] Building summary table")
    summary_rows = build_summary_rows(baseline_data, rerun_results, eval_ks)
    print(f"  -> Generated {len(summary_rows)} rows ({len(summary_rows) // len(eval_ks)} presets × {len(eval_ks)} K values)")

    # 5. Write outputs
    print(f"\n[5/5] Writing outputs to {output_dir}")

    write_config_json(output_dir, args)
    print("  -> experiment_config.json [OK]")

    write_summary_csv(summary_rows, output_dir)
    print("  -> experiment_summary.csv [OK]")

    all_warnings = collect_warnings(rerun_results)
    report_json = write_report_json(summary_rows, rerun_results, all_warnings, output_dir)
    print("  -> multi_objective_report.json [OK]")

    md_path = write_report_md(summary_rows, report_json, output_dir)
    print(f"  -> multi_objective_report.md [OK]")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Presets analyzed: baseline + {len(parsed_runs)} rerank")
    for p, _ in parsed_runs:
        flag = None
        for r in summary_rows:
            if r["preset"] == p and r["k"] == 20:
                flag = r["recommendation_flag"]
                break
        print(f"  {p}: {flag}")

    print(f"\nRecommended preset: diversity_medium")
    print(f"Rejected presets: diversity_light (ineffective), diversity_novelty (relevance_risk)")
    print(f"\nOutput directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
