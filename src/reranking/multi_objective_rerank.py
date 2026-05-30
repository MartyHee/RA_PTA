"""
Batch 16D — 多目标 reranking 模块。

基于 relevance_score + diversity_gain + novelty_score 的 greedy reranking。

Presets:
  - baseline:         alpha=1.0, beta=0.0, gamma=0.0
  - diversity_light:  alpha=1.0, beta=0.05, gamma=0.0
  - diversity_medium: alpha=1.0, beta=0.10, gamma=0.0
  - diversity_novelty:alpha=1.0, beta=0.10, gamma=0.05
  - custom:            alpha/beta/gamma 由命令行传入

Usage:
  python src/reranking/multi_objective_rerank.py \
    --predictions outputs/dnn/real_raw_5000/202605132017/predictions_test.csv \
    --features data/features/real_raw_5000/tabular_test.csv \
    --freq-source data/features/real_raw_5000/tabular_train.csv \
    --output-root outputs/multi_objective/rerank \
    --preset diversity_medium \
    --top-k 20 \
    --dataset real_raw_5000 \
    --model-name dnn \
    --run-id 202605132017
"""

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone

# 复用 diversity_metrics 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evaluation.diversity_metrics import (
    read_csv,
    join_predictions_features,
    build_freq_table,
    compute_metrics,
    ndcg_at_k,
    write_json,
    write_csv,
    write_md,
)


# ──────────────────────────────────────────────
# Preset ────────────────────────────────────────
PRESETS = {
    "baseline":         {"alpha": 1.0, "beta": 0.0, "gamma": 0.0},
    "diversity_light":  {"alpha": 1.0, "beta": 0.05, "gamma": 0.0},
    "diversity_medium": {"alpha": 1.0, "beta": 0.10, "gamma": 0.0},
    "diversity_novelty": {"alpha": 1.0, "beta": 0.10, "gamma": 0.05},
}

DIVERSITY_GAIN_WEIGHTS = {
    "author_id": 0.4,
    "hashtag_name_top": 0.4,
    "region": 0.2,
}


# ──────────────────────────────────────────────
# CLI ───────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="多目标 reranking")
    p.add_argument("--predictions", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--freq-source", required=True)
    p.add_argument("--output-root", default="outputs/multi_objective/rerank")
    p.add_argument("--preset", default="diversity_medium",
                   choices=list(PRESETS.keys()) + ["custom"])
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--eval-k", default="10,20,50")
    p.add_argument("--id-col", default="video_id")
    p.add_argument("--score-col", default="score")
    p.add_argument("--label-col", default="label")
    p.add_argument("--diversity-fields", default="author_id,hashtag_name_top,region")
    p.add_argument("--dataset", default="real_raw_5000")
    p.add_argument("--model-name", default="dnn")
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def resolve_preset(args):
    if args.preset == "custom":
        if args.alpha is None or args.beta is None or args.gamma is None:
            print("ERROR: custom preset 需要 --alpha --beta --gamma", file=sys.stderr)
            sys.exit(1)
        return {"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma}
    cfg = dict(PRESETS[args.preset])
    # 命令行传入覆盖
    if args.alpha is not None:
        cfg["alpha"] = args.alpha
    if args.beta is not None:
        cfg["beta"] = args.beta
    if args.gamma is not None:
        cfg["gamma"] = args.gamma
    return cfg


# ──────────────────────────────────────────────
# Diversity gain ────────────────────────────────
def compute_diversity_gain(item, selected_set):
    """计算单个 item 相对已选集合的 diversity_gain [0,1]。
    selected_set = {"author_id": set(), "hashtag_name_top": set(), "region": set()}
    """
    gain = 0.0
    for field, weight in DIVERSITY_GAIN_WEIGHTS.items():
        val = item.get(field, "__MISSING__")
        if val and val not in ("__MISSING__", "", None) and val not in selected_set[field]:
            gain += weight
    # gain 已默认归一化到 [0,1]（weights 之和 = 1.0）
    return gain


# ──────────────────────────────────────────────
# Novelty score ─────────────────────────────────
def compute_novelty_score(item, freq_counters, fields):
    """计算单个 item 的 novelty_score = mean(1/log(freq+2))。"""
    vals = []
    for f in fields:
        counter = freq_counters.get(f, Counter())
        val = item.get(f, "__MISSING__")
        if val in ("__MISSING__", "", None):
            continue
        freq = counter.get(val, 0)
        nov = 1.0 / math.log(freq + 2)
        vals.append(nov)
    return sum(vals) / len(vals) if vals else 0.0


# ──────────────────────────────────────────────
# Greedy reranking ──────────────────────────────
def greedy_rerank(rows, alpha, beta, gamma, top_k, score_col, fields, freq_counters, warnings):
    """Greedy reranking，返回 (reranked_rows, selected_gains)。

    返回的 reranked_rows 按 rerank_rank 1..len(rows) 已排序。
    """
    # 按 score 降序得到 baseline
    baseline = sorted(rows, key=lambda r: float(r.get(score_col, 0)), reverse=True)

    for idx, r in enumerate(baseline):
        r["baseline_rank"] = idx + 1

    selected = []
    remaining = list(baseline)  # 浅拷贝

    # 已选集合
    selected_set = {f: set() for f in fields}
    selected_gains = []

    for rank in range(1, top_k + 1):
        best_item = None
        best_idx = -1
        best_final = -float("inf")
        best_gain = 0.0
        best_novelty = 0.0

        for i, item in enumerate(remaining):
            d_gain = compute_diversity_gain(item, selected_set)
            n_score = compute_novelty_score(item, freq_counters, fields)

            final = (alpha * float(item.get(score_col, 0))
                     + beta * d_gain
                     + gamma * n_score)

            if final > best_final:
                best_final = final
                best_item = item
                best_idx = i
                best_gain = d_gain
                best_novelty = n_score

        if best_item is None:
            warnings.append(f"greedy_rerank: rank={rank} 无可选 item，中止")
            break

        # 更新已选集合
        for f in fields:
            val = best_item.get(f, "__MISSING__")
            if val not in ("__MISSING__", "", None):
                selected_set[f].add(val)

        best_item["rerank_rank"] = rank
        best_item["rerank_score"] = round(best_final, 6)
        best_item["diversity_gain"] = round(best_gain, 4)
        best_item["novelty_score"] = round(best_novelty, 4)

        selected.append(best_item)
        selected_gains.append({
            "rerank_rank": rank,
            "video_id": best_item.get("video_id", ""),
            "diversity_gain": round(best_gain, 4),
            "novelty_score": round(best_novelty, 4),
            "final_score": round(best_final, 6),
            "original_score": float(best_item.get(score_col, 0)),
        })
        remaining.pop(best_idx)

    # top_k 之后按原 score 顺序追加
    remaining.sort(key=lambda r: float(r.get(score_col, 0)), reverse=True)
    for offset, item in enumerate(remaining, start=1):
        rank = top_k + offset
        item["rerank_rank"] = rank
        item["rerank_score"] = float(item.get(score_col, 0))
        item["diversity_gain"] = 0.0
        item["novelty_score"] = 0.0

    selected.extend(remaining)
    return selected, selected_gains


# ──────────────────────────────────────────────
# 写 reranked_predictions.csv ───────────────────
def write_reranked_csv(rows, fields, cfg, args, path):
    required = [
        "rerank_rank", "baseline_rank", "video_id", "label",
        "score", "rerank_score", "diversity_gain", "novelty_score",
        "author_id", "hashtag_name_top", "region",
        "model_name", "dataset_name", "run_id",
    ]
    header = [
        "rerank_rank", "baseline_rank", "video_id", "label",
        "score", "rerank_score", "diversity_gain", "novelty_score",
        "author_id", "hashtag_name_top", "region",
        "model_name", "dataset_name", "source_run_id",
        "preset", "alpha", "beta", "gamma",
    ]

    run_id = args.run_id or ""
    source_run_id = run_id

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow([
                r.get("rerank_rank", ""),
                r.get("baseline_rank", ""),
                r.get("video_id", ""),
                r.get("label", ""),
                r.get("score", ""),
                r.get("rerank_score", ""),
                r.get("diversity_gain", ""),
                r.get("novelty_score", ""),
                r.get("author_id", ""),
                r.get("hashtag_name_top", "__MISSING__"),
                r.get("region", "__MISSING__"),
                args.model_name,
                args.dataset,
                source_run_id,
                args.preset,
                cfg["alpha"],
                cfg["beta"],
                cfg["gamma"],
            ])
    print(f"  reranked CSV → {path}")
    return path


# ──────────────────────────────────────────────
# 计算 reranked 指标（按 rerank_rank 顺序）────────
def compute_reranked_metrics(rows, freq_counters, fields, top_k_list, label_col, meta):
    """对已按 rerank_rank 排序的行计算指标。"""
    # 确保已排序
    sorted_rows = sorted(rows, key=lambda r: int(r.get("rerank_rank", 0)))

    # 候选集统计
    total_unique = {}
    for f in fields:
        vals = {r.get(f, "__MISSING__") for r in sorted_rows}
        total_unique[f] = len(vals)

    available_labels = any(
        r.get(label_col) is not None and r.get(label_col) != ""
        for r in sorted_rows
    )

    missing_rates = {}
    for f in fields:
        missing_count = sum(1 for r in sorted_rows if r.get(f, "__MISSING__") in ("__MISSING__", "", None))
        missing_rates[f] = round(missing_count / len(sorted_rows) * 100, 2) if sorted_rows else 0

    metrics_by_k = {}

    for K in top_k_list:
        top = sorted_rows[:K]
        m = {"k": K}

        # 基础多样性
        unique_counts = {}
        for f in fields:
            vals = set()
            for r in top:
                v = r.get(f, "__MISSING__")
                if v not in ("__MISSING__", "", None):
                    vals.add(v)
            unique_counts[f] = len(vals)

        for f in fields:
            short = f.split("_")[0] if "_" in f else f
            m[f"unique_{short}_count"] = unique_counts.get(f, None)
            cnt = unique_counts.get(f, 0) or 0
            divers = cnt / K if K > 0 else 0
            m[f"{short}_diversity"] = round(divers, 4)
            m[f"duplicate_{short}_rate"] = round(1 - divers, 4)

        # 覆盖度
        for f in fields:
            short = f.split("_")[0] if "_" in f else f
            total = total_unique.get(f, 1)
            cnt = unique_counts.get(f, 0) or 0
            m[f"coverage_{short}"] = round(cnt / total, 4) if total > 0 else None

        # 新颖性
        for f in fields:
            short = f.split("_")[0] if "_" in f else f
            n_vals = []
            for r in top:
                val = r.get(f, "__MISSING__")
                if val in ("__MISSING__", "", None):
                    continue
                freq = freq_counters.get(f, Counter()).get(val, 0)
                n_vals.append(1.0 / math.log(freq + 2))
            m[f"novelty_{short}"] = round(sum(n_vals) / len(n_vals), 4) if n_vals else None

        nov_vals = []
        for f in fields:
            short = f.split("_")[0] if "_" in f else f
            v = m.get(f"novelty_{short}")
            if v is not None:
                nov_vals.append(v)
        m["novelty_mean"] = round(sum(nov_vals) / len(nov_vals), 4) if nov_vals else None

        # 相关性对照：使用原始 score（公平对比）
        scores_top = []
        for r in top:
            try:
                scores_top.append(float(r.get("score", 0)))
            except (ValueError, TypeError):
                scores_top.append(0.0)
        m["mean_relevance_score"] = round(sum(scores_top) / len(scores_top), 6) if scores_top else None

        # rerank_score（含 diversity bonus）仅记录，不用于 relevance 指标
        rerank_scores_top = []
        for r in top:
            try:
                rerank_scores_top.append(float(r.get("rerank_score", 0)))
            except (ValueError, TypeError):
                rerank_scores_top.append(0.0)
        m["mean_rerank_score"] = round(sum(rerank_scores_top) / len(rerank_scores_top), 6) if rerank_scores_top else None

        if available_labels:
            labels_top = []
            for r in top:
                try:
                    labels_top.append(float(r.get(label_col, 0)))
                except (ValueError, TypeError):
                    labels_top.append(0.0)
            pos = sum(1 for lab in labels_top if lab > 0)
            m["positive_rate"] = round(pos / len(labels_top), 4) if labels_top else None
            m["precision"] = round(pos / K, 4) if K > 0 else None
            ndcg_val = ndcg_at_k(scores_top, labels_top, K)
            m["ndcg"] = round(ndcg_val, 4) if ndcg_val is not None else None
            if ndcg_val is None:
                meta["warnings"].append(f"K={K}: IDCG=0, ndcg 设为 null")
        else:
            m["positive_rate"] = None
            m["precision"] = None
            m["ndcg"] = None

        metrics_by_k[K] = m

    candidate_stats = {
        "total_rows": len(rows),
        "total_unique_author": total_unique.get("author_id", 0),
        "total_unique_hashtag": total_unique.get("hashtag_name_top", 0),
        "total_unique_region": total_unique.get("region", 0),
    }
    for f in fields:
        short = f.split("_")[0] if "_" in f else f
        candidate_stats[f"missing_{short}_rate"] = missing_rates.get(f, 0)

    return metrics_by_k, candidate_stats, available_labels


# ──────────────────────────────────────────────
# 写 rerank_report.md ──────────────────────────
def write_rerank_report(cfg, args, rerank_config, before_metrics, after_metrics, report, path):
    before = before_metrics
    after = after_metrics
    fields = report["available_fields"]

    lines = []
    lines.append("# 多目标 Reranking 报告\n")
    lines.append(f"**生成时间**: {report['rerank_metadata']['timestamp']}\n")
    lines.append(f"**rerank_run_id**: {report['rerank_metadata']['rerank_run_id']}\n")
    lines.append(f"**模型**: {args.model_name} / {args.dataset} / {args.run_id}\n")
    lines.append("---\n")

    lines.append("## 任务目标\n")
    lines.append("基于 baseline 模型 score + diversity_gain + novelty_score 进行 greedy reranking，提升推荐列表多样性/覆盖度/新颖性。\n")
    lines.append("")

    lines.append("## 输入文件\n")
    for key in ("predictions_path", "features_path", "freq_source_path"):
        lines.append(f"- **{key}**: {report['input_paths'].get(key, 'N/A')}\n")
    lines.append("")

    lines.append("## Reranking 配置\n")
    lines.append(f"- **Preset**: {args.preset}\n")
    lines.append(f"- **alpha** (relevance): {cfg['alpha']}\n")
    lines.append(f"- **beta** (diversity): {cfg['beta']}\n")
    lines.append(f"- **gamma** (novelty): {cfg['gamma']}\n")
    lines.append(f"- **top_k**: {args.top_k}\n")
    lines.append(f"- **eval_k**: {args.eval_k}\n")
    lines.append(f"- **diversity fields**: {args.diversity_fields}\n")
    lines.append(f"- **author_gain**: {DIVERSITY_GAIN_WEIGHTS.get('author_id', 0.4)}\n")
    lines.append(f"- **hashtag_gain**: {DIVERSITY_GAIN_WEIGHTS.get('hashtag_name_top', 0.4)}\n")
    lines.append(f"- **region_gain**: {DIVERSITY_GAIN_WEIGHTS.get('region', 0.2)}\n")
    lines.append("")

    lines.append("## Reranking 公式\n")
    lines.append("```\n")
    lines.append("final_score = alpha * relevance_score + beta * diversity_gain(item|selected) + gamma * novelty_score(item)\n")
    lines.append("```\n")
    lines.append("")

    # Merge K sets
    all_ks = sorted(set(list(before.keys()) + list(after.keys())))

    lines.append("## Before / After 对比表\n")
    header_parts = ["| K | "]
    header_parts.append("before_score | after_score | score_delta | ")
    header_parts.append("before_pos | after_pos | ")
    header_parts.append("before_ndcg | after_ndcg | ")
    for f in fields:
        short = f.split("_")[0] if "_" in f else f
        header_parts.append(f"before_div_{short} | after_div_{short} | delta_div_{short} | ")
        header_parts.append(f"before_cov_{short} | after_cov_{short} | ")
    header_parts.append("before_nov_mean | after_nov_mean |\n")
    lines.append("".join(header_parts))
    lines.append("|---" * 40 + "|\n")

    for K in all_ks:
        b = before.get(K, {})
        a = after.get(K, {})
        b_score = b.get("mean_relevance_score", "-")
        a_score = a.get("mean_relevance_score", "-")
        score_delta = ""
        if isinstance(b_score, (int, float)) and isinstance(a_score, (int, float)):
            d = a_score - b_score
            score_delta = f"{d:+.4f}"

        b_pos = b.get("positive_rate", "-")
        a_pos = a.get("positive_rate", "-")
        b_ndcg = b.get("ndcg", "-")
        a_ndcg = a.get("ndcg", "-")

        row_parts = [f"| {K} | "]
        row_parts.append(f"{b_score} | {a_score} | {score_delta} | ")
        row_parts.append(f"{b_pos} | {a_pos} | ")
        row_parts.append(f"{b_ndcg} | {a_ndcg} | ")

        for f in fields:
            short = f.split("_")[0] if "_" in f else f
            b_div = b.get(f"{short}_diversity", "-")
            a_div = a.get(f"{short}_diversity", "-")
            delta_div = ""
            if isinstance(b_div, (int, float)) and isinstance(a_div, (int, float)):
                d = a_div - b_div
                delta_div = f"{d:+.4f}"
            row_parts.append(f"{b_div} | {a_div} | {delta_div} | ")
            row_parts.append(f"{b.get(f'coverage_{short}', '-')} | {a.get(f'coverage_{short}', '-')} | ")
        row_parts.append(f"{b.get('novelty_mean', '-')} | {a.get('novelty_mean', '-')} |\n")
        lines.append("".join(row_parts))
    lines.append("")

    lines.append("## Trade-off 判断\n")
    # 取 K=20 分析
    b20 = before.get(20, {})
    a20 = after.get(20, {})
    b_score = b20.get("mean_relevance_score", 0) or 0
    a_score = a20.get("mean_relevance_score", 0) or 0
    score_pct = ((a_score - b_score) / b_score * 100) if b_score else 0
    lines.append(f"- **Relevance 变化** (K=20): {b_score:.4f} → {a_score:.4f} ({score_pct:+.2f}%)\n")
    if score_pct >= -2:
        lines.append("  - ✅ mean_score 下降 <= 2%，relevance 可接受。\n")
    else:
        lines.append(f"  - ⚠️ mean_score 下降 {score_pct:.1f}%，超过 2%，需谨慎。\n")

    b_pos = b20.get("positive_rate", 0) or 0
    a_pos = a20.get("positive_rate", 0) or 0
    pos_pct = a_pos - b_pos
    if pos_pct >= -0.03:
        lines.append(f"  - ✅ positive_rate 变化 {pos_pct:+.2%}，在 3% 范围内。\n")
    else:
        lines.append(f"  - ⚠️ positive_rate 下降 {pos_pct:+.2%}，超过 3%，需记录风险。\n")

    for f in fields:
        short = f.split("_")[0] if "_" in f else f
        b_d = b20.get(f"{short}_diversity", 0) or 0
        a_d = a20.get(f"{short}_diversity", 0) or 0
        if a_d > b_d:
            lines.append(f"- ✅ **{short}_diversity** (K=20): {b_d:.4f} → {a_d:.4f} (提升 {(a_d-b_d):+.4f})\n")
        else:
            lines.append(f"- {short}_diversity (K=20): {b_d:.4f} → {a_d:.4f} (变化 {(a_d-b_d):+.4f})\n")
    lines.append("")

    lines.append("## Warnings\n")
    warnings = report.get("warnings", [])
    if warnings:
        for i, w in enumerate(warnings, 1):
            lines.append(f"{i}. {w}\n")
    else:
        lines.append("无。\n")
    lines.append("")

    lines.append("## 下一步建议\n")
    lines.append("1. **Batch 16E**：运行全部 4 组实验（baseline/light/medium/novelty），生成多目标实验报告。\n")
    lines.append("2. 调整 alpha/beta/gamma 参数，观察 trade-off 曲线。\n")
    lines.append("3. 如 diversity 提升有限，可尝试提高 beta 或增加 P1 字段。\n")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"  MD   → {path}")


# ──────────────────────────────────────────────
# 主流程 ────────────────────────────────────────
def main():
    args = parse_args()
    cfg = resolve_preset(args)

    top_k = args.top_k
    eval_k_list = sorted(set(int(k) for k in args.eval_k.split(",") if k.strip()))
    fields = [f.strip() for f in args.diversity_fields.split(",")]

    run_id = args.run_id
    if not run_id:
        base = os.path.basename(os.path.dirname(args.predictions))
        run_id = base if base and base != "real_raw_5000" else "unknown"

    rerank_run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:16]
    out_dir = os.path.join(args.output_root, rerank_run_id)
    baseline_dir = os.path.join(out_dir, "baseline_metrics")
    reranked_dir = os.path.join(out_dir, "reranked_metrics")
    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(reranked_dir, exist_ok=True)

    # warnings 收集器（独立于 meta 之外，避免与 compute_metrics 混用）
    rerank_warnings = []

    # 构造 report（用于输出 baseline metrics）
    report = {
        "rerank_metadata": {
            "rerank_run_id": rerank_run_id,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "model_name": args.model_name,
            "dataset_name": args.dataset,
            "run_id": run_id,
            "preset": args.preset,
            "alpha": cfg["alpha"],
            "beta": cfg["beta"],
            "gamma": cfg["gamma"],
            "top_k": top_k,
        },
        "input_paths": {
            "predictions_path": os.path.abspath(args.predictions),
            "features_path": os.path.abspath(args.features),
            "freq_source_path": os.path.abspath(args.freq_source),
        },
        "rerank_config": {},
        "join_stats": {},
        "available_fields": [],
        "available_labels": False,
        "warnings": [],
        "rerank_gains": [],
        "reranked_predictions_path": "",
        "baseline_metrics_paths": {},
        "reranked_metrics_paths": {},
        "before_metrics": {},
        "after_metrics": {},
    }

    # ── 1. 读取 ──
    print(f"[1/7] 读取 predictions: {args.predictions}")
    pred_rows, pred_cols = read_csv(args.predictions)
    print(f"       → {len(pred_rows)} 行")

    print(f"[2/7] 读取 features: {args.features}")
    feat_rows, feat_cols = read_csv(args.features)
    print(f"       → {len(feat_rows)} 行")

    # ── 2. Join ──
    print(f"[3/7] 按 {args.id_col} join...")
    joined = join_predictions_features(pred_rows, feat_rows, args.id_col, report)
    available_fields = [f for f in fields if f in feat_cols or f in pred_cols]
    report["available_fields"] = available_fields
    print(f"       → {len(joined)} 行 (loss={report['join_stats'].get('join_loss', 0)}%)")

    if not joined:
        print("ERROR: join 后无数据，中止。", file=sys.stderr)
        sys.exit(1)

    # ── 3. 频次表 ──
    print(f"[4/7] 读取频次源: {args.freq_source}")
    freq_rows, _ = read_csv(args.freq_source)
    freq_counters = build_freq_table(freq_rows, available_fields)
    print(f"       → {len(freq_rows)} 行, {len(available_fields)} 字段频次已构建")

    # ── 4. Baseline 指标计算 ──
    print(f"[5/7] 计算 baseline 指标 K={eval_k_list}...")
    meta = {"warnings": []}
    before_metrics, before_candidate_stats, available_labels = compute_metrics(
        joined, freq_counters, available_fields, eval_k_list,
        args.score_col, args.label_col, meta
    )
    report["available_labels"] = available_labels
    for w in meta["warnings"]:
        report["warnings"].append(f"[baseline] {w}")

    # 输出 baseline metrics
    print(f"       输出 baseline metrics → {baseline_dir}")
    b_json = os.path.join(baseline_dir, "diversity_metrics_report.json")
    b_csv = os.path.join(baseline_dir, "diversity_metrics_summary.csv")
    b_md = os.path.join(baseline_dir, "diversity_metrics_report.md")
    b_report = {
        "run_metadata": {
            "metrics_run_id": f"{rerank_run_id}_baseline",
            "timestamp": report["rerank_metadata"]["timestamp"],
            "model_name": args.model_name,
            "dataset_name": args.dataset,
            "run_id": run_id,
        },
        "input_paths": dict(report["input_paths"]),
        "join_stats": report["join_stats"],
        "candidate_field_stats": before_candidate_stats,
        "available_fields": available_fields,
        "available_labels": available_labels,
        "metrics_by_k": {str(K): before_metrics[K] for K in before_metrics},
        "warnings": [w for w in report["warnings"] if "baseline" in w],
    }
    write_json(b_report, b_json)
    write_csv(before_metrics, available_labels, available_fields, b_csv)
    write_md(b_report, b_md)
    report["baseline_metrics_paths"] = {"json": b_json, "csv": b_csv, "md": b_md}

    # ── 5. Greedy reranking ──
    print(f"[6/7] Greedy reranking (top_k={top_k}, alpha={cfg['alpha']}, beta={cfg['beta']}, gamma={cfg['gamma']})...")
    reranked_rows, selected_gains = greedy_rerank(
        joined, cfg["alpha"], cfg["beta"], cfg["gamma"],
        top_k, args.score_col, available_fields, freq_counters, rerank_warnings
    )
    report["rerank_gains"] = selected_gains

    # 输出 reranked_predictions.csv
    reranked_csv_path = os.path.join(out_dir, "reranked_predictions.csv")
    write_reranked_csv(reranked_rows, available_fields, cfg, args, reranked_csv_path)
    report["reranked_predictions_path"] = reranked_csv_path

    # ── 6. Reranked 指标计算 ──
    print(f"[7/7] 计算 reranked 指标 K={eval_k_list}...")
    after_meta = {"warnings": []}
    after_metrics, after_candidate_stats, _ = compute_reranked_metrics(
        reranked_rows, freq_counters, available_fields, eval_k_list,
        args.label_col, after_meta
    )
    for w in after_meta["warnings"]:
        report["warnings"].append(f"[reranked] {w}")

    # 输出 reranked metrics
    print(f"       输出 reranked metrics → {reranked_dir}")
    a_json = os.path.join(reranked_dir, "diversity_metrics_report.json")
    a_csv = os.path.join(reranked_dir, "diversity_metrics_summary.csv")
    a_md = os.path.join(reranked_dir, "diversity_metrics_report.md")
    a_report = {
        "run_metadata": {
            "metrics_run_id": f"{rerank_run_id}_reranked",
            "timestamp": report["rerank_metadata"]["timestamp"],
            "model_name": args.model_name,
            "dataset_name": args.dataset,
            "run_id": run_id,
        },
        "input_paths": dict(report["input_paths"]),
        "join_stats": report["join_stats"],
        "candidate_field_stats": after_candidate_stats,
        "available_fields": available_fields,
        "available_labels": available_labels,
        "metrics_by_k": {str(K): after_metrics[K] for K in after_metrics},
        "warnings": [w for w in report["warnings"] if "reranked" in w],
    }
    write_json(a_report, a_json)
    write_csv(after_metrics, available_labels, available_fields, a_csv)
    write_md(a_report, a_md)
    report["reranked_metrics_paths"] = {"json": a_json, "csv": a_csv, "md": a_md}

    # ── 汇总 warnings ──
    for w in rerank_warnings:
        report["warnings"].append(f"[rerank] {w}")

    # ── 7. 输出 rerank_config.json ──
    rerank_config = {
        "rerank_run_id": rerank_run_id,
        "timestamp": report["rerank_metadata"]["timestamp"],
        "preset": args.preset,
        "alpha": cfg["alpha"],
        "beta": cfg["beta"],
        "gamma": cfg["gamma"],
        "top_k": top_k,
        "eval_k": eval_k_list,
        "diversity_fields": available_fields,
        "diversity_gain_weights": DIVERSITY_GAIN_WEIGHTS,
        "novelty_formula": "1/log(freq+2)",
        "freq_source": os.path.abspath(args.freq_source),
        "input": {
            "predictions": os.path.abspath(args.predictions),
            "features": os.path.abspath(args.features),
        },
        "output_dir": out_dir,
    }
    report["rerank_config"] = rerank_config
    config_path = os.path.join(out_dir, "rerank_config.json")
    write_json(rerank_config, config_path)

    # 记录 before/after metrics（简化版）
    report["before_metrics"] = {str(K): before_metrics[K] for K in before_metrics}
    report["after_metrics"] = {str(K): after_metrics[K] for K in after_metrics}

    # ── 8. 输出 rerank_metrics.json ──
    # 构造对比 summary
    comparison = {}
    all_ks = sorted(set(list(before_metrics.keys()) + list(after_metrics.keys())))
    for K in all_ks:
        b = before_metrics.get(K, {})
        a = after_metrics.get(K, {})
        comp = {"k": K}
        for key in ["mean_relevance_score", "positive_rate", "precision", "ndcg",
                     "author_diversity", "hashtag_diversity", "region_diversity",
                     "coverage_author", "coverage_hashtag", "coverage_region",
                     "novelty_author", "novelty_hashtag", "novelty_region", "novelty_mean"]:
            bv = b.get(key)
            av = a.get(key)
            comp[f"before_{key}"] = bv
            comp[f"after_{key}"] = av
            if bv is not None and av is not None and isinstance(bv, (int, float)) and isinstance(av, (int, float)):
                comp[f"delta_{key}"] = round(av - bv, 4)
            else:
                comp[f"delta_{key}"] = None
        comparison[str(K)] = comp

    metrics_json = {
        "rerank_metadata": report["rerank_metadata"],
        "rerank_config": rerank_config,
        "comparison_by_k": comparison,
        "warnings": report["warnings"],
    }
    metrics_path = os.path.join(out_dir, "rerank_metrics.json")
    write_json(metrics_json, metrics_path)

    # ── 9. 输出 rerank_report.md ──
    md_path = os.path.join(out_dir, "rerank_report.md")
    write_rerank_report(cfg, args, rerank_config, before_metrics, after_metrics, report, md_path)

    # ── 10. 日志 ──
    report["before_metrics"] = {str(K): before_metrics[K] for K in before_metrics}
    report["after_metrics"] = {str(K): after_metrics[K] for K in after_metrics}

    print(f"\n{'='*70}")
    print(f"  Reranking 完成")
    print(f"  rerank_run_id: {rerank_run_id}")
    print(f"  preset: {args.preset} (alpha={cfg['alpha']}, beta={cfg['beta']}, gamma={cfg['gamma']})")
    print(f"  top_k: {top_k}")
    print(f"  输出目录: {out_dir}")
    print()
    # K=20 before/after
    if 20 in before_metrics and 20 in after_metrics:
        b = before_metrics[20]
        a = after_metrics[20]
        print(f"  K=20 对比:")
        print(f"    relevance_score: {b.get('mean_relevance_score'):.4f} → {a.get('mean_relevance_score'):.4f}")
        print(f"    positive_rate:   {b.get('positive_rate')} → {a.get('positive_rate')}")
        print(f"    ndcg:            {b.get('ndcg')} → {a.get('ndcg')}")
        print(f"    author_div:      {b.get('author_diversity')} → {a.get('author_diversity')}")
        print(f"    hashtag_div:     {b.get('hashtag_diversity')} → {a.get('hashtag_diversity')}")
        print(f"    region_div:      {b.get('region_diversity')} → {a.get('region_diversity')}")
        print(f"    novelty_mean:    {b.get('novelty_mean')} → {a.get('novelty_mean')}")
    print(f"  warnings: {len(report['warnings'])}")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
