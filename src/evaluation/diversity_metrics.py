"""
Batch 16C — 多样性与覆盖度指标计算脚本。

对推荐列表 top-K 计算：
  - 基础多样性：unique_author/hashtag/region count & ratio
  - 覆盖度：coverage_author/hashtag/region
  - 新颖性：novelty_author/hashtag/region (基于训练集频次)
  - 相关性对照：mean_relevance_score / positive_rate / precision / ndcg

输出 JSON / CSV / MD 报告到 outputs/multi_objective/metrics/<metrics_run_id>/。

Usage:
  python src/evaluation/diversity_metrics.py \
    --predictions outputs/dnn/real_raw_5000/202605132017/predictions_test.csv \
    --features data/features/real_raw_5000/tabular_test.csv \
    --freq-source data/features/real_raw_5000/tabular_train.csv \
    --output-root outputs/multi_objective/metrics \
    --top-k 10,20,50
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone


# ──────────────────────────────────────────────
# 解析 ──────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="多样性与覆盖度指标计算")
    p.add_argument("--predictions", required=True, help="模型预测文件 CSV")
    p.add_argument("--features", required=True, help="特征 CSV（补充 author/hashtag/region 等）")
    p.add_argument("--freq-source", required=True, help="频次来源 CSV（训练集）")
    p.add_argument("--output-root", default="outputs/multi_objective/metrics")
    p.add_argument("--top-k", default="10,20,50", help="逗号分隔的 K 值")
    p.add_argument("--id-col", default="video_id")
    p.add_argument("--score-col", default="score")
    p.add_argument("--label-col", default="label")
    p.add_argument("--diversity-fields", default="author_id,hashtag_name_top,region")
    p.add_argument("--dataset", default="real_raw_5000")
    p.add_argument("--model-name", default="dnn")
    p.add_argument("--run-id", default=None)
    return p.parse_args()


# ──────────────────────────────────────────────
# CSV 读取  ─────────────────────────────────────
def read_csv(path):
    """读取 CSV，返回 (list[dict], list[str])：行列表、列名。"""
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows, (reader.fieldnames or [])


def join_predictions_features(pred_rows, feat_rows, id_col, meta):
    """按 id_col 做 inner join，输出 joined 列表并记录 stats。"""
    feat_map = {}
    for r in feat_rows:
        k = r.get(id_col)
        if k is not None:
            feat_map[k] = r

    joined = []
    missing_id = 0
    for r in pred_rows:
        k = r.get(id_col)
        if k is None:
            missing_id += 1
            continue
        f = feat_map.get(k)
        if f is None:
            continue
        # merge: pred 在前，feat 补充
        merged = dict(r)
        for col, val in f.items():
            if col not in merged or merged[col] == "":
                merged[col] = val
        joined.append(merged)

    meta.setdefault("join_stats", {})
    js = meta["join_stats"]
    js["pred_rows"] = len(pred_rows)
    js["feat_rows"] = len(feat_rows)
    js["joined_rows"] = len(joined)
    js["missing_id_in_predictions"] = missing_id
    if joined:
        js["join_loss"] = round(
            (len(pred_rows) - len(joined)) / len(pred_rows) * 100, 2
        )
    else:
        js["join_loss"] = 100.0
    if js["join_loss"] > 0:
        meta.setdefault("warnings", []).append(
            f"join 后行数减少 {js['join_loss']}% "
            f"({len(pred_rows)} → {len(joined)})"
        )
    return joined


# ──────────────────────────────────────────────
# 频次计算 ──────────────────────────────────────
def build_freq_table(rows, fields):
    """从频次源数据构建 {field: Counter}。"""
    counters = {}
    for f in fields:
        counters[f] = Counter()
    for r in rows:
        for f in fields:
            val = r.get(f, "__MISSING__")
            counters[f][val] += 1
    return counters


# ──────────────────────────────────────────────
# NDCG ─────────────────────────────────────────
def ndcg_at_k(scores, labels, k):
    """binary label NDCG@K。"""
    top = list(zip(scores, labels))[:k]
    dcg = 0.0
    for i, (_, lab) in enumerate(top):
        lab = float(lab)
        if lab > 0:
            dcg += lab / math.log2(i + 2)
    # IDCG: 理想排序
    ideal = sorted(top, key=lambda x: float(x[1]), reverse=True)
    idcg = 0.0
    for i, (_, lab) in enumerate(ideal):
        lab = float(lab)
        if lab > 0:
            idcg += lab / math.log2(i + 2)
    if idcg <= 0:
        return None
    return dcg / idcg


# ──────────────────────────────────────────────
# 核心指标 ──────────────────────────────────────
def compute_metrics(joined, freq_counters, fields, top_k_list, score_col, label_col, meta):
    """对每个 K 计算指标。"""
    # 候选集统计（所有 join 后的数据）
    total_unique = {}
    for f in fields:
        vals = {r.get(f, "__MISSING__") for r in joined}
        total_unique[f] = len(vals)

    # 按 score 降序排序
    try:
        sorted_rows = sorted(joined, key=lambda r: float(r.get(score_col, 0)), reverse=True)
    except (ValueError, TypeError) as e:
        print(f"ERROR: 无法解析 score 列 '{score_col}': {e}", file=sys.stderr)
        sys.exit(1)

    available_labels = any(
        r.get(label_col) is not None and r.get(label_col) != ""
        for r in sorted_rows
    )

    # 缺失率统计
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

        m["unique_author_count"] = unique_counts.get("author_id", None)
        m["unique_hashtag_count"] = unique_counts.get("hashtag_name_top", None)
        m["unique_region_count"] = unique_counts.get("region", None)

        for f in fields:
            short = f.split("_")[0] if "_" in f else f
            m[f"unique_{short}_count"] = unique_counts.get(f, None)

        # 多样性比率
        for f in fields:
            short = f.split("_")[0] if "_" in f else f
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
            n_values = []
            for r in top:
                val = r.get(f, "__MISSING__")
                if val in ("__MISSING__", "", None):
                    continue
                freq = freq_counters.get(f, Counter()).get(val, 0)
                n_values.append(1.0 / math.log(freq + 2))
            if n_values:
                m[f"novelty_{short}"] = round(sum(n_values) / len(n_values), 4)
            else:
                m[f"novelty_{short}"] = None

        # 新颖性均值（skip unavailable）
        nov_vals = []
        for f in fields:
            short = f.split("_")[0] if "_" in f else f
            v = m.get(f"novelty_{short}")
            if v is not None:
                nov_vals.append(v)
        m["novelty_mean"] = round(sum(nov_vals) / len(nov_vals), 4) if nov_vals else None

        # 相关性对照
        scores_top = []
        for r in top:
            try:
                scores_top.append(float(r.get(score_col, 0)))
            except (ValueError, TypeError):
                scores_top.append(0.0)
        m["mean_relevance_score"] = round(sum(scores_top) / len(scores_top), 6) if scores_top else None

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

    # 候选集全量统计
    candidate_stats = {
        "total_rows": len(joined),
        "total_unique_author": total_unique.get("author_id", 0),
        "total_unique_hashtag": total_unique.get("hashtag_name_top", 0),
        "total_unique_region": total_unique.get("region", 0),
    }
    candidate_stats["missing_author_rate"] = missing_rates.get("author_id", 0)
    candidate_stats["missing_hashtag_rate"] = missing_rates.get("hashtag_name_top", 0)
    candidate_stats["missing_region_rate"] = missing_rates.get("region", 0)

    return metrics_by_k, candidate_stats, available_labels


# ──────────────────────────────────────────────
# 输出 ─────────────────────────────────────────
def write_json(report, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  JSON → {path}")


def write_csv(metrics_by_k, available_labels, fields, path):
    field_shorts = []
    for f in fields:
        short = f.split("_")[0] if "_" in f else f
        field_shorts.append(short)

    header = [
        "k",
        "mean_relevance_score",
    ]
    if available_labels:
        header += ["positive_rate", "precision", "ndcg"]
    for short in field_shorts:
        header += [
            f"{short}_diversity",
            f"coverage_{short}",
            f"novelty_{short}",
        ]
    header.append("novelty_mean")
    header.append("warning_count")

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for K in sorted(metrics_by_k.keys()):
            m = metrics_by_k[K]
            row = [m["k"], m.get("mean_relevance_score", "")]
            if available_labels:
                row += [
                    m.get("positive_rate", ""),
                    m.get("precision", ""),
                    m.get("ndcg", ""),
                ]
            for short in field_shorts:
                row += [
                    m.get(f"{short}_diversity", ""),
                    m.get(f"coverage_{short}", ""),
                    m.get(f"novelty_{short}", ""),
                ]
            row.append(m.get("novelty_mean", ""))
            row.append(m.get("warning_count", 0))
            writer.writerow(row)
    print(f"  CSV  → {path}")


def write_md(report, path):
    lines = []
    lines.append("# 多样性指标报告\n")
    lines.append(f"**生成时间**: {report['run_metadata']['timestamp']}\n")
    lines.append(f"**metrics_run_id**: {report['run_metadata']['metrics_run_id']}\n")
    lines.append(f"**模型**: {report['run_metadata']['model_name']} / {report['run_metadata']['dataset_name']} / {report['run_metadata']['run_id']}\n")
    lines.append("---\n")

    lines.append("## 输入文件\n")
    for key in ("predictions_path", "features_path", "freq_source_path"):
        lines.append(f"- **{key}**: {report['input_paths'].get(key, 'N/A')}\n")
    lines.append("")

    lines.append("## Join 结果\n")
    js = report["join_stats"]
    lines.append(f"- 预测行数: {js['pred_rows']}\n")
    lines.append(f"- 特征行数: {js['feat_rows']}\n")
    lines.append(f"- Join 后行数: {js['joined_rows']}\n")
    lines.append(f"- Join 丢失率: {js.get('join_loss', 'N/A')}%\n")
    lines.append("")

    lines.append("## 候选集字段统计\n")
    cs = report["candidate_field_stats"]
    lines.append(f"- 总行数: {cs['total_rows']}\n")
    lines.append(f"- 唯一 author_id: {cs['total_unique_author']}\n")
    lines.append(f"- 唯一 hashtag_name_top: {cs['total_unique_hashtag']}\n")
    lines.append(f"- 唯一 region: {cs['total_unique_region']}\n")
    lines.append(f"- author_id 缺失率: {cs['missing_author_rate']}%\n")
    lines.append(f"- hashtag_name_top 缺失率: {cs['missing_hashtag_rate']}%\n")
    lines.append(f"- region 缺失率: {cs['missing_region_rate']}%\n")
    lines.append("")

    lines.append("## Top-K 指标表\n")
    header_parts = ["| K | mean_score | "]
    if report.get("available_labels", False):
        header_parts.append("pos_rate | prec | ndcg | ")
    for f in report.get("available_fields", []):
        short = f.split("_")[0] if "_" in f else f
        header_parts.append(f"div_{short} | cov_{short} | nov_{short} | ")
    header_parts.append("nov_mean | warnings |\n")
    lines.append("".join(header_parts))
    lines.append("|---" * 30 + "|\n")

    for K in sorted(report["metrics_by_k"].keys()):
        m = report["metrics_by_k"][K]
        wc = m.get("warning_count", 0)
        row_parts = [f"| {m['k']} | {m.get('mean_relevance_score', '-')} | "]
        if report.get("available_labels", False):
            row_parts.append(f"{m.get('positive_rate', '-')} | {m.get('precision', '-')} | {m.get('ndcg', '-')} | ")
        for f in report.get("available_fields", []):
            short = f.split("_")[0] if "_" in f else f
            row_parts.append(
                f"{m.get(f'{short}_diversity', '-')} | "
                f"{m.get(f'coverage_{short}', '-')} | "
                f"{m.get(f'novelty_{short}', '-')} | "
            )
        row_parts.append(f"{m.get('novelty_mean', '-')} | {wc} |\n")
        lines.append("".join(row_parts))
    lines.append("")

    lines.append("## Warnings\n")
    warnings = report.get("warnings", [])
    if warnings:
        for i, w in enumerate(warnings, 1):
            lines.append(f"{i}. {w}\n")
    else:
        lines.append("无。\n")
    lines.append("")

    lines.append("## 解读\n")
    lines.append("### 多样性\n")
    lines.append("`author_diversity` / `hashtag_diversity` / `region_diversity` 反映推荐列表各 K 值下实体独特性。值越接近 1.0 表示列表越多样。\n")
    lines.append("")

    lines.append("### 覆盖度\n")
    lines.append("`coverage_author` / `coverage_hashtag` / `coverage_region` 反映推荐列表占候选集实体比例。候选集越大，覆盖度通常越低。\n")
    lines.append("")

    lines.append("### 新颖性\n")
    lines.append("`novelty_mean` 反映推荐列表实体在训练集中的长尾程度。值越高，表示越倾向于推荐训练集中低频的实体。\n")
    lines.append("")

    lines.append("### 相关性对照\n")
    lines.append("`mean_relevance_score` / `positive_rate` / `precision` / `ndcg` 作为 relevance proxy 对照指标，辅助判断多样性提升是否过度牺牲相关性。\n")
    lines.append("")

    lines.append("## 下一步建议\n")
    lines.append("1. **Batch 16D**：实现多目标 reranking 模块 `src/reranking/multi_objective_rerank.py`。\n")
    lines.append("2. 基于本报告的 baseline 指标与 rerank 后指标做对比。\n")
    lines.append("3. 调整 alpha/beta/gamma 参数观察 trade-off。\n")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"  MD   → {path}")


# ──────────────────────────────────────────────
# 主流程 ────────────────────────────────────────
def main():
    args = parse_args()

    # 解析 top_k
    top_k_list = sorted(set(int(k) for k in args.top_k.split(",") if k.strip()))
    fields = [f.strip() for f in args.diversity_fields.split(",")]

    # 推断 run_id
    run_id = args.run_id
    if not run_id:
        # 从 predictions 路径尝试提取
        base = os.path.basename(os.path.dirname(args.predictions))
        if base and base != "real_raw_5000":
            run_id = base
        else:
            run_id = "unknown"

    # metrics_run_id
    metrics_run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    out_dir = os.path.join(args.output_root, metrics_run_id)
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "predictions_path": os.path.abspath(args.predictions),
        "features_path": os.path.abspath(args.features),
        "freq_source_path": os.path.abspath(args.freq_source),
    }

    report = {
        "run_metadata": {
            "metrics_run_id": metrics_run_id,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "model_name": args.model_name,
            "dataset_name": args.dataset,
            "run_id": run_id,
        },
        "input_paths": dict(meta),
        "join_stats": {},
        "candidate_field_stats": {},
        "available_fields": [],
        "available_labels": False,
        "metrics_by_k": {},
        "warnings": [],
    }

    # 1. 读取
    print(f"[1/5] 读取 predictions: {args.predictions}")
    pred_rows, pred_cols = read_csv(args.predictions)
    print(f"       → {len(pred_rows)} 行")

    print(f"[2/5] 读取 features: {args.features}")
    feat_rows, feat_cols = read_csv(args.features)
    print(f"       → {len(feat_rows)} 行")

    # 2. Join
    print(f"[3/5] 按 {args.id_col} join...")
    joined = join_predictions_features(pred_rows, feat_rows, args.id_col, report)
    available_fields = [f for f in fields if f in feat_cols or f in pred_cols]
    report["available_fields"] = available_fields
    print(f"       → {len(joined)} 行 (loss={report['join_stats'].get('join_loss', 0)}%)")

    if not joined:
        print("ERROR: join 后无数据，中止。", file=sys.stderr)
        sys.exit(1)

    # 3. 频次表
    print(f"[4/5] 读取频次源: {args.freq_source}")
    freq_rows, _ = read_csv(args.freq_source)
    freq_counters = build_freq_table(freq_rows, available_fields)
    print(f"       → {len(freq_rows)} 行, {len(available_fields)} 字段频次已构建")

    # 4. 计算指标
    print(f"[5/5] 计算指标 K={top_k_list}...")
    metrics_by_k, candidate_stats, available_labels = compute_metrics(
        joined, freq_counters, available_fields, top_k_list,
        args.score_col, args.label_col, report
    )
    report["metrics_by_k"] = {str(K): metrics_by_k[K] for K in metrics_by_k}
    report["candidate_field_stats"] = candidate_stats
    report["available_labels"] = available_labels

    # 汇总每个 K 的 warning 数
    for K, m in metrics_by_k.items():
        m["warning_count"] = len(report["warnings"])

    # 5. 输出
    json_path = os.path.join(out_dir, "diversity_metrics_report.json")
    csv_path = os.path.join(out_dir, "diversity_metrics_summary.csv")
    md_path = os.path.join(out_dir, "diversity_metrics_report.md")

    print(f"\n输出目录: {out_dir}")
    write_json(report, json_path)
    write_csv(metrics_by_k, available_labels, available_fields, csv_path)
    write_md(report, md_path)

    # 6. 摘要
    print(f"\n{'='*60}")
    print(f"  多样性指标计算完成")
    print(f"  metrics_run_id: {metrics_run_id}")
    print(f"  输出目录: {out_dir}")
    for K in sorted(metrics_by_k.keys()):
        m = metrics_by_k[K]
        print(f"  K={K}: score={m.get('mean_relevance_score'):.4f}  "
              f"div_auth={m.get('author_diversity'):.4f}  "
              f"div_hash={m.get('hashtag_diversity'):.4f}  "
              f"div_reg={m.get('region_diversity'):.4f}  "
              f"novel={m.get('novelty_mean')}  "
              f"ndcg={m.get('ndcg')}")
    print(f"  warnings: {len(report['warnings'])}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
