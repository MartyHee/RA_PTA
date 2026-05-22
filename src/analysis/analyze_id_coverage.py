"""
Batch 14I-coverage: High-cardinality ID coverage/OOV analysis for real_raw_5000.

Analyzes author_id and music_id coverage, OOV rates, frequency distribution,
and long-tail ratios across train/val/test splits.

Usage:
    D:/CodeData/software/Anaconda/Anaconda3/envs/ra/python.exe src/analysis/analyze_id_coverage.py --dataset real_raw_5000

Output:
    outputs/analysis/id_coverage/real_raw_5000/<run_id>/
    ├── id_coverage_report.json
    ├── id_coverage_summary.csv
    ├── id_frequency_top.csv
    └── id_coverage_report.md
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("D:/CodeData/Program Coding/ByteDance/RA_PTA")
PYTHON = "D:/CodeData/software/Anaconda/Anaconda3/envs/ra/python.exe"


def parse_args():
    parser = argparse.ArgumentParser(description="High-cardinality ID coverage/OOV analysis")
    parser.add_argument("--dataset", default="real_raw_5000", help="Dataset name")
    parser.add_argument("--output-root", default="outputs/analysis/id_coverage", help="Output root")
    return parser.parse_args()


def run_id() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


def load_tabular_splits(dataset: str):
    """Load tabular train/val/test CSVs and return dict of DataFrames."""
    base = PROJECT_ROOT / "data" / "features" / dataset
    splits = {}
    for split in ["train", "val", "test"]:
        path = base / f"tabular_{split}.csv"
        print(f"  Loading {path} ...")
        df = pd.read_csv(path)
        assert "video_id" in df.columns, f"{path} missing video_id"
        assert "author_id" in df.columns, f"{path} missing author_id"
        splits[split] = df
    return splits


def load_music_mapping(dataset: str) -> pd.DataFrame:
    """Load raw_music table and return video_id -> music_id mapping."""
    config_path = PROJECT_ROOT / "configs" / "datasets.yaml"
    # Look up raw_music filename from dataset config
    music_path = (
        PROJECT_ROOT / "douyin_data_project" / "data" / "interim" / dataset / f"raw_music_{dataset}.csv"
    )
    # Also try alternative path
    if not music_path.exists():
        music_path = (
            PROJECT_ROOT / "douyin_data_project" / "data" / "interim" / "real_raw_5000" / f"raw_music_{dataset}.csv"
        )
    if not music_path.exists():
        # Fallback: list files in directory
        interim_dir = PROJECT_ROOT / "douyin_data_project" / "data" / "interim" / dataset
        if interim_dir.exists():
            files = list(interim_dir.glob("raw_music_*.csv"))
            if files:
                music_path = files[0]

    print(f"  Loading raw music table from {music_path} ...")
    music_df = pd.read_csv(str(music_path))

    # Normalize video_id to string for consistent merging
    music_df["video_id"] = music_df["video_id"].astype(str)
    music_df["music_id"] = music_df["music_id"].astype(str)

    # Mark NaN/empty music_id as missing
    music_df["music_id"] = music_df["music_id"].replace(["nan", "", "None"], "__MISSING__")

    return music_df[["video_id", "music_id"]]


def compute_split_stats(ser: pd.Series) -> dict:
    """Compute basic frequency statistics for a Series of IDs."""
    freq = ser.value_counts()
    n_unique = len(freq)
    n_total = len(ser)
    freq_values = freq.values
    return {
        "count": n_total,
        "unique": n_unique,
        "freq_mean": float(freq.mean()),
        "freq_median": float(freq.median()),
        "freq_min": int(freq.min()),
        "freq_max": int(freq.max()),
        "freq_std": float(freq.std()),
    }


def compute_oov(train_ser, val_ser, test_ser) -> dict:
    """Compute OOV rates for val and test relative to train IDs."""
    train_ids = set(train_ser.unique())
    val_ids = set(val_ser.unique())
    test_ids = set(test_ser.unique())

    val_oov = val_ids - train_ids
    test_oov = test_ids - train_ids

    result = {
        "train_unique": len(train_ids),
        "val_unique": len(val_ids),
        "test_unique": len(test_ids),
        "val_oov_count": len(val_oov),
        "val_oov_rate": round(len(val_oov) / len(val_ids) * 100, 2) if len(val_ids) > 0 else 0.0,
        "test_oov_count": len(test_oov),
        "test_oov_rate": round(len(test_oov) / len(test_ids) * 100, 2) if len(test_ids) > 0 else 0.0,
        "val_test_oov_total": len(val_oov.union(test_oov)),
        "val_test_union_unique": len(val_ids.union(test_ids)),
        "val_test_oov_rate": round(len(val_oov.union(test_oov)) / len(val_ids.union(test_ids)) * 100, 2)
        if len(val_ids.union(test_ids)) > 0 else 0.0,
    }
    return result


def compute_freq_distribution(train_ser: pd.Series) -> dict:
    """Compute frequency distribution (long-tail) statistics."""
    freq = train_ser.value_counts()
    freq_values = freq.values
    n_unique = len(freq)
    n_total = len(train_ser)

    singleton = int((freq_values == 1).sum())
    freq_ge_2 = int((freq_values >= 2).sum())
    freq_ge_3 = int((freq_values >= 3).sum())
    freq_ge_5 = int((freq_values >= 5).sum())
    freq_ge_10 = int((freq_values >= 10).sum())

    return {
        "singleton_id_count": singleton,
        "singleton_id_ratio": round(singleton / n_unique * 100, 2) if n_unique > 0 else 0.0,
        "freq_ge_2_count": freq_ge_2,
        "freq_ge_2_ratio": round(freq_ge_2 / n_unique * 100, 2) if n_unique > 0 else 0.0,
        "freq_ge_3_count": freq_ge_3,
        "freq_ge_3_ratio": round(freq_ge_3 / n_unique * 100, 2) if n_unique > 0 else 0.0,
        "freq_ge_5_count": freq_ge_5,
        "freq_ge_5_ratio": round(freq_ge_5 / n_unique * 100, 2) if n_unique > 0 else 0.0,
        "freq_ge_10_count": freq_ge_10,
        "freq_ge_10_ratio": round(freq_ge_10 / n_unique * 100, 2) if n_unique > 0 else 0.0,
        "avg_samples_per_id": round(n_total / n_unique, 2) if n_unique > 0 else 0.0,
    }


def compute_topk_coverage(train_ser: pd.Series, ks=None) -> dict:
    """Compute top-K ID coverage proportions in train split."""
    if ks is None:
        ks = [10, 50, 100, 200, 500, 1000]
    freq = train_ser.value_counts()
    n_total = len(train_ser)
    result = {}
    for k in ks:
        if k <= len(freq):
            coverage = int(freq.head(k).sum())
            result[f"top{k}_coverage"] = round(coverage / n_total * 100, 2)
        else:
            result[f"top{k}_coverage"] = 100.0
    return result


def make_decision(oov_rate, avg_samples, singleton_ratio, missing_rate) -> str:
    """
    Determine pass/caution/fail based on thresholds from the design doc.
    Pass thresholds:
      - OOV rate < 20%
      - avg_samples_per_id >= 2.0
      - singleton_id_ratio < 50%
      - missing_rate < 5%
    """
    failures = []
    if oov_rate >= 20:
        failures.append(f"oov={oov_rate}>=20")
    if avg_samples < 2.0:
        failures.append(f"avg_samples={avg_samples}<2.0")
    if singleton_ratio >= 50:
        failures.append(f"singleton={singleton_ratio}>=50")
    if missing_rate >= 5:
        failures.append(f"missing={missing_rate}>=5")

    if len(failures) == 0:
        return "pass"
    elif len(failures) <= 2:
        return "caution"
    else:
        return "fail"


def overall_recommendation(author_decision, music_decision,
                           author_oov, music_oov,
                           author_avg, music_avg) -> str:
    """Determine overall recommendation."""
    if author_decision == "fail" and music_decision in ("fail", "unavailable"):
        return "do_not_implement_id_embedding"

    if author_decision == "pass" and music_decision == "pass":
        return "proceed_to_impl"

    if author_decision in ("pass", "caution") and music_decision in ("fail", "unavailable"):
        return "proceed_author_only"

    if author_decision in ("fail",) and music_decision == "pass":
        return "proceed_music_only"

    # caution / mixed
    return "need_data_fix"


def build_frequency_top_df(train_ser, val_ser, test_ser, field_name, top_n=500) -> pd.DataFrame:
    """Build a DataFrame of top-N most frequent IDs with per-split counts."""
    train_freq = train_ser.value_counts().head(top_n).reset_index()
    train_freq.columns = ["id_value", "train_count"]

    # Add val/test counts
    val_freq = val_ser.value_counts().to_dict()
    test_freq = test_ser.value_counts().to_dict()
    train_freq["val_count"] = train_freq["id_value"].map(val_freq).fillna(0).astype(int)
    train_freq["test_count"] = train_freq["id_value"].map(test_freq).fillna(0).astype(int)
    train_freq["total_count"] = train_freq["train_count"] + train_freq["val_count"] + train_freq["test_count"]
    train_freq["in_train"] = True
    train_freq["in_val"] = train_freq["id_value"].isin(val_freq.keys())
    train_freq["in_test"] = train_freq["id_value"].isin(test_freq.keys())
    train_freq["field"] = field_name

    return train_freq.reset_index(drop=True)


def analyze_field(field_name: str, train_ser, val_ser, test_ser, missing_rate: float,
                  source: str):
    """Run full analysis for one field and return results dict."""
    result = {"field": field_name, "source": source, "missing_rate": missing_rate}

    # Split basic stats (store all stats for train, count+unique for val/test)
    for split_name, ser in [("train", train_ser), ("val", val_ser), ("test", test_ser)]:
        stats = compute_split_stats(ser)
        result[f"{split_name}_count"] = stats["count"]
        result[f"{split_name}_unique"] = stats["unique"]
        if split_name == "train":
            result["freq_mean"] = stats["freq_mean"]
            result["freq_median"] = stats["freq_median"]
            result["freq_min"] = stats["freq_min"]
            result["freq_max"] = stats["freq_max"]
            result["freq_std"] = stats["freq_std"]

    # OOV analysis
    oov = compute_oov(train_ser, val_ser, test_ser)
    result.update(oov)

    # Train frequency distribution
    freq_dist = compute_freq_distribution(train_ser)
    result.update(freq_dist)

    # Top-K coverage in train
    topk = compute_topk_coverage(train_ser)
    result.update(topk)

    # Decision
    decision = make_decision(
        oov_rate=result.get("val_test_oov_rate", 0),
        avg_samples=freq_dist["avg_samples_per_id"],
        singleton_ratio=freq_dist["singleton_id_ratio"],
        missing_rate=missing_rate,
    )
    result["decision"] = decision

    return result


def main():
    args = parse_args()
    dataset = args.dataset
    rid = run_id()
    print(f"=== Batch 14I-coverage: {dataset} ID coverage analysis ===\n")
    print(f"Run ID: {rid}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n1. Loading tabular splits ...")
    splits = load_tabular_splits(dataset)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    # Convert video_id to string for merging
    for df in [train_df, val_df, test_df]:
        df["video_id"] = df["video_id"].astype(str)

    # ------------------------------------------------------------------
    # 2. Music ID: load from raw_music table and merge
    # ------------------------------------------------------------------
    print("\n2. Loading music_id from raw_music table ...")
    music_mapping = load_music_mapping(dataset)

    # Check merge coverage
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        merged = df[["video_id"]].merge(music_mapping, on="video_id", how="left")
        music_missing = (merged["music_id"] == "__MISSING__").sum()
        print(f"  {split_name}: {len(merged)} samples, music_id missing={music_missing} ({music_missing/len(merged)*100:.1f}%)")

        if music_missing == len(merged):
            print(f"  WARNING: All music_id are missing for {split_name}!")
        df["music_id"] = merged["music_id"].values

    # ------------------------------------------------------------------
    # 3. Extract IDs
    # ------------------------------------------------------------------
    print("\n3. Extracting ID series ...")
    author_train = train_df["author_id"].astype(str)
    author_val = val_df["author_id"].astype(str)
    author_test = test_df["author_id"].astype(str)

    music_train = train_df["music_id"].astype(str)
    music_val = val_df["music_id"].astype(str)
    music_test = test_df["music_id"].astype(str)

    # Check author_id missing values
    for split_name, ser in [("train", author_train), ("val", author_val), ("test", author_test)]:
        missing = (ser == "__MISSING__").sum()
        print(f"  author_id {split_name}: {len(ser)} total, {missing} __MISSING__")

    # Check music_id missing values
    for split_name, ser in [("train", music_train), ("val", music_val), ("test", music_test)]:
        missing = (ser == "__MISSING__").sum()
        total = len(ser)
        print(f"  music_id {split_name}: {total} total, {missing} __MISSING__ ({missing/total*100:.1f}%)")

    # ------------------------------------------------------------------
    # 4. Compute statistics
    # ------------------------------------------------------------------
    print("\n4. Computing statistics ...")

    author_missing_rate = 0.0  # No missing author_id in tabular CSV
    music_missing_rate = float((music_train == "__MISSING__").sum()) / len(music_train) * 100
    music_missing_rate_val = float((music_val == "__MISSING__").sum()) / len(music_val) * 100
    music_missing_rate_test = float((music_test == "__MISSING__").sum()) / len(music_test) * 100

    # Filter out __MISSING__ for frequency/OOV analysis (missing is separate)
    author_train_clean = author_train[author_train != "__MISSING__"]
    author_val_clean = author_val[author_val != "__MISSING__"]
    author_test_clean = author_test[author_test != "__MISSING__"]

    music_train_clean = music_train[music_train != "__MISSING__"]
    music_val_clean = music_val[music_val != "__MISSING__"]
    music_test_clean = music_test[music_test != "__MISSING__"]

    author_result = analyze_field(
        field_name="author_id",
        train_ser=author_train_clean,
        val_ser=author_val_clean,
        test_ser=author_test_clean,
        missing_rate=author_missing_rate,
        source="tabular_train/val/test.csv (built-in column)",
    )
    author_result["missing_rate_train"] = 0.0
    author_result["missing_rate_val"] = 0.0
    author_result["missing_rate_test"] = 0.0

    # Determine music_id availability
    n_music_valid = len(music_train_clean)
    music_available = n_music_valid > 0

    if music_available:
        music_result = analyze_field(
            field_name="music_id",
            train_ser=music_train_clean,
            val_ser=music_val_clean,
            test_ser=music_test_clean,
            missing_rate=music_missing_rate,
            source="raw_music_real_raw_5000.csv (joined by video_id)",
        )
        music_result["missing_rate_train"] = round(music_missing_rate, 2)
        music_result["missing_rate_val"] = round(music_missing_rate_val, 2)
        music_result["missing_rate_test"] = round(music_missing_rate_test, 2)

        # Recompute music_id decision including missing rate
        music_result["decision"] = make_decision(
            oov_rate=music_result.get("val_test_oov_rate", 0),
            avg_samples=music_result.get("avg_samples_per_id", 0),
            singleton_ratio=music_result.get("singleton_id_ratio", 0),
            missing_rate=music_missing_rate,  # Use train missing rate for decision
        )
    else:
        music_result = {
            "field": "music_id",
            "source": "raw_music_real_raw_5000.csv",
            "missing_rate": 100.0,
            "missing_rate_train": 100.0,
            "missing_rate_val": 100.0,
            "missing_rate_test": 100.0,
            "train_count": 0,
            "val_count": 0,
            "test_count": 0,
            "train_unique": 0,
            "val_unique": 0,
            "test_unique": 0,
            "val_oov_rate": None,
            "test_oov_rate": None,
            "val_test_oov_rate": None,
            "avg_samples_per_id": 0,
            "singleton_id_ratio": None,
            "decision": "unavailable",
            "note": "All music_id values are missing in raw_music table for train split"
        }

    # ------------------------------------------------------------------
    # 5. Overall recommendation
    # ------------------------------------------------------------------
    overall_rec = overall_recommendation(
        author_decision=author_result["decision"],
        music_decision=music_result["decision"],
        author_oov=author_result.get("val_test_oov_rate", 100),
        music_oov=music_result.get("val_test_oov_rate", 100) if music_available else 100,
        author_avg=author_result.get("avg_samples_per_id", 0),
        music_avg=music_result.get("avg_samples_per_id", 0) if music_available else 0,
    )

    # Build pass/fail reason strings for readability
    def decision_detail(result, field_label):
        d = result["decision"]
        if d == "pass":
            return f"PASS: OOV={result.get('val_test_oov_rate','N/A')}%, avg_samples={result.get('avg_samples_per_id','N/A')}, singleton={result.get('singleton_id_ratio','N/A')}%, missing={result.get('missing_rate', 'N/A')}%"
        elif d == "unavailable":
            return "SKIP: music_id unavailable (all missing)"
        else:
            return f"{d.upper()}: OOV={result.get('val_test_oov_rate','N/A')}%, avg_samples={result.get('avg_samples_per_id','N/A')}, singleton={result.get('singleton_id_ratio','N/A')}%, missing={result.get('missing_rate', 'N/A')}%"

    overall = {
        "author_id": decision_detail(author_result, "author_id"),
        "music_id": decision_detail(music_result, "music_id"),
        "overall_recommendation": overall_rec,
    }

    # ------------------------------------------------------------------
    # 6. Build frequency top CSVs
    # ------------------------------------------------------------------
    print("\n5. Building frequency top tables ...")
    author_top_df = build_frequency_top_df(author_train_clean, author_val_clean, author_test_clean, "author_id")

    if music_available:
        music_top_df = build_frequency_top_df(music_train_clean, music_val_clean, music_test_clean, "music_id")
        top_df = pd.concat([author_top_df, music_top_df], ignore_index=True)
    else:
        top_df = author_top_df

    # Build summary CSV
    summaries = []
    for r in [author_result, music_result]:
        summaries.append({
            "field": r["field"],
            "source": r["source"],
            "train_unique": r.get("train_unique", 0),
            "val_unique": r.get("val_unique", 0),
            "test_unique": r.get("test_unique", 0),
            "val_oov_rate": r.get("val_oov_rate", None),
            "test_oov_rate": r.get("test_oov_rate", None),
            "train_freq_mean": r.get("freq_mean", None),
            "train_freq_median": r.get("freq_median", None),
            "singleton_ratio": r.get("singleton_id_ratio", None),
            "top100_coverage": r.get("top100_coverage", None),
            "missing_rate_train": r.get("missing_rate_train", r.get("missing_rate", 0)),
            "decision": r["decision"],
        })
    summary_df = pd.DataFrame(summaries)

    # ------------------------------------------------------------------
    # 7. Write output files
    # ------------------------------------------------------------------
    output_dir = PROJECT_ROOT / args.output_root / dataset / rid
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n6. Writing output to {output_dir} ...")

    # JSON report
    report_json = {
        "run_id": rid,
        "dataset": dataset,
        "analysis_type": "id_coverage",
        "batch": "14I-coverage",
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "author_id": author_result,
        "music_id": music_result,
        "overall": overall,
        "design_doc_reference": "docs/multimodal_high_cardinality_id_embedding_design.md",
        "pass_thresholds": {
            "oov_rate_max": 20,
            "avg_samples_per_id_min": 2.0,
            "singleton_id_ratio_max": 50,
            "missing_rate_max": 5,
        },
        "note": "All analysis is based on tabular CSVs and raw_music table. Results are for coverage assessment only, not model training."
    }

    json_path = output_dir / "id_coverage_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, ensure_ascii=False, indent=2)
    print(f"  Wrote {json_path}")

    # Summary CSV
    summary_path = output_dir / "id_coverage_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"  Wrote {summary_path}")

    # Frequency top CSV
    freq_path = output_dir / "id_frequency_top.csv"
    top_df.to_csv(freq_path, index=False, encoding="utf-8")
    print(f"  Wrote {freq_path}")

    # ------------------------------------------------------------------
    # 8. Markdown report
    # ------------------------------------------------------------------
    def fmt(v, suffix=""):
        if v is None or v == "N/A":
            return "N/A"
        if isinstance(v, float):
            return f"{v:.2f}{suffix}"
        return str(v)

    md_lines = [
        f"# ID Coverage Analysis Report — {dataset}",
        f"",
        f"> Batch: 14I-coverage",
        f"> Run ID: {rid}",
        f"> Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> Design Reference: `docs/multimodal_high_cardinality_id_embedding_design.md`",
        f"",
        f"---",
        f"",
        f"## 1. 分析目标",
        f"",
        f"对 `author_id` 和 `music_id` 在 `{dataset}` 的 train/val/test 中的覆盖率、OOV 率和长尾分布进行分析，",
        f"判断是否具备进入 Multimodal categorical embedding 实现的基础条件。",
        f"",
        f"## 2. 数据来源",
        f"",
        f"| 字段 | 来源 | 说明 |",
        f"|------|------|------|",
        f"| author_id | `data/features/{dataset}/tabular_*.csv` | tabular CSV 内置列 |",
        f"| music_id | `douyin_data_project/data/interim/{dataset}/raw_music_{dataset}.csv` | 按 video_id join，不在 tabular CSV 中 |",
        f"",
        f"## 3. author_id 分析",
        f"",
        f"### 3.1 基础统计",
        f"",
        f"| Split | 样本数 | Unique ID | 缺失数 | 缺失率 |",
        f"|-------|--------|-----------|--------|--------|",
        f"| train | {author_result['train_count']} | {author_result['train_unique']} | 0 | 0.0% |",
        f"| val | {author_result['val_count']} | {author_result['val_unique']} | 0 | 0.0% |",
        f"| test | {author_result['test_count']} | {author_result['test_unique']} | 0 | 0.0% |",
        f"",
        f"### 3.2 OOV 分析",
        f"",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| val OOV count | {author_result['val_oov_count']} |",
        f"| val OOV rate | {fmt(author_result['val_oov_rate'], '%')} |",
        f"| test OOV count | {author_result['test_oov_count']} |",
        f"| test OOV rate | {fmt(author_result['test_oov_rate'], '%')} |",
        f"| val+test OOV rate | {fmt(author_result['val_test_oov_rate'], '%')} |",
        f"",
        f"### 3.3 频次分布 (train)",
        f"",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| 平均样本数/ID | {author_result['avg_samples_per_id']} |",
        f"| 中位数样本数/ID | {author_result['freq_median']} |",
        f"| 最小样本数/ID | {author_result['freq_min']} |",
        f"| 最大样本数/ID | {author_result['freq_max']} |",
        f"| 标准差 | {fmt(author_result['freq_std'])} |",
        f"| singleton ID 数 | {author_result['singleton_id_count']} |",
        f"| singleton ID 比例 | {fmt(author_result['singleton_id_ratio'], '%')} |",
        f"| freq >= 2 比例 | {fmt(author_result['freq_ge_2_ratio'], '%')} |",
        f"| freq >= 3 比例 | {fmt(author_result['freq_ge_3_ratio'], '%')} |",
        f"| freq >= 5 比例 | {fmt(author_result['freq_ge_5_ratio'], '%')} |",
        f"| freq >= 10 比例 | {fmt(author_result['freq_ge_10_ratio'], '%')} |",
        f"",
        f"### 3.4 Top-K 覆盖率 (train)",
        f"",
        f"| K | 覆盖样本比例 |",
        f"|---|-------------|",
        f"| 10 | {fmt(author_result['top10_coverage'], '%')} |",
        f"| 50 | {fmt(author_result['top50_coverage'], '%')} |",
        f"| 100 | {fmt(author_result['top100_coverage'], '%')} |",
        f"| 200 | {fmt(author_result['top200_coverage'], '%')} |",
        f"| 500 | {fmt(author_result['top500_coverage'], '%')} |",
        f"| 1000 | {fmt(author_result.get('top1000_coverage', 'N/A'), '%')} |",
        f"",
        f"### 3.5 决策: {author_result['decision'].upper()}",
        f"",
        f"{decision_detail(author_result, 'author_id')}",
        f"",
        f"## 4. music_id 分析",
        f"",
    ]

    if music_available:
        md_lines += [
            f"### 4.1 基础统计",
            f"",
            f"| Split | 样本数 | Unique ID | 缺失数 | 缺失率 |",
            f"|-------|--------|-----------|--------|--------|",
            f"| train | {music_result['train_count']} | {music_result['train_unique']} | {music_missing_rate:.0f}% of {len(music_train)} | {music_missing_rate:.1f}% |",
            f"| val | {music_result['val_count']} | {music_result['val_unique']} | {music_missing_rate_val:.0f}% of {len(music_val)} | {music_missing_rate_val:.1f}% |",
            f"| test | {music_result['test_count']} | {music_result['test_unique']} | {music_missing_rate_test:.0f}% of {len(music_test)} | {music_missing_rate_test:.1f}% |",
            f"",
            f"### 4.2 OOV 分析",
            f"",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| val OOV count | {music_result['val_oov_count']} |",
            f"| val OOV rate | {fmt(music_result['val_oov_rate'], '%')} |",
            f"| test OOV count | {music_result['test_oov_count']} |",
            f"| test OOV rate | {fmt(music_result['test_oov_rate'], '%')} |",
            f"| val+test OOV rate | {fmt(music_result['val_test_oov_rate'], '%')} |",
            f"",
            f"### 4.3 频次分布 (train)",
            f"",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 平均样本数/ID | {music_result['avg_samples_per_id']} |",
            f"| 中位数样本数/ID | {music_result['freq_median']} |",
            f"| 最小样本数/ID | {music_result['freq_min']} |",
            f"| 最大样本数/ID | {music_result['freq_max']} |",
            f"| 标准差 | {fmt(music_result['freq_std'])} |",
            f"| singleton ID 数 | {music_result['singleton_id_count']} |",
            f"| singleton ID 比例 | {fmt(music_result['singleton_id_ratio'], '%')} |",
            f"| freq >= 2 比例 | {fmt(music_result['freq_ge_2_ratio'], '%')} |",
            f"| freq >= 3 比例 | {fmt(music_result['freq_ge_3_ratio'], '%')} |",
            f"| freq >= 5 比例 | {fmt(music_result['freq_ge_5_ratio'], '%')} |",
            f"| freq >= 10 比例 | {fmt(music_result['freq_ge_10_ratio'], '%')} |",
            f"",
            f"### 4.4 Top-K 覆盖率 (train, 仅有效 music_id)",
            f"",
            f"| K | 覆盖样本比例 |",
            f"|---|-------------|",
            f"| 10 | {fmt(music_result['top10_coverage'], '%')} |",
            f"| 50 | {fmt(music_result['top50_coverage'], '%')} |",
            f"| 100 | {fmt(music_result['top100_coverage'], '%')} |",
            f"| 200 | {fmt(music_result['top200_coverage'], '%')} |",
            f"| 500 | {fmt(music_result['top500_coverage'], '%')} |",
            f"",
            f"### 4.5 决策: {music_result['decision'].upper()}",
            f"",
            f"{decision_detail(music_result, 'music_id')}",
            f"",
        ]
    else:
        md_lines += [
            f"**music_id 不可分析。**",
            f"",
            f"所有 video_id 在 raw_music 表中的 music_id 均为空。",
            f"",
            f"### 4.5 决策: UNAVAILABLE",
            f"",
            f"music_id 无法用于 categorical embedding（全体缺失）。",
            f"",
        ]

    # Overall
    md_lines += [
        f"---",
        f"",
        f"## 5. 总体判断",
        f"",
        f"| 条件 | 阈值 | author_id | music_id |",
        f"|------|------|-----------|----------|",
        f"| OOV rate | < 20% | {fmt(author_result.get('val_test_oov_rate'), '%')} | {fmt(music_result.get('val_test_oov_rate'), '%') if music_available else 'N/A'} |",
        f"| avg samples/ID | >= 2.0 | {author_result.get('avg_samples_per_id')} | {music_result.get('avg_samples_per_id', 'N/A') if music_available else 'N/A'} |",
        f"| singleton ratio | < 50% | {fmt(author_result.get('singleton_id_ratio'), '%')} | {fmt(music_result.get('singleton_id_ratio'), '%') if music_available else 'N/A'} |",
        f"| missing rate | < 5% | {fmt(author_result.get('missing_rate', 0), '%')} | {fmt(music_result.get('missing_rate', 100), '%') if music_available else 'N/A'} |",
        f"",
        f"**author_id 决策: {author_result['decision'].upper()}**",
        f"",
        f"**music_id 决策: {music_result['decision'].upper()}**",
        f"",
        f"**总体建议: {overall_rec}**",
        f"",
        f"### 5.1 原因分析",
        f"",
    ]

    # author_id analysis
    author_issues = []
    if author_result.get("avg_samples_per_id", 2) < 2.0:
        author_issues.append(f"平均样本数/ID = {author_result['avg_samples_per_id']} < 2.0，大量 ID 仅出现 1 次")
    if author_result.get("singleton_id_ratio", 0) >= 50:
        author_issues.append(f"singleton ID 占比 = {author_result['singleton_id_ratio']}% >= 50%，长尾极长")
    if author_result.get("val_test_oov_rate", 0) >= 20:
        author_issues.append(f"OOV 率 = {author_result['val_test_oov_rate']}% >= 20%，val/test 中过多未见 ID")

    if author_issues:
        md_lines.append(f"**author_id 问题:**")
        for issue in author_issues:
            md_lines.append(f"- {issue}")
        md_lines.append(f"")
    else:
        md_lines.append(f"**author_id 通过所有阈值检查。**")
        md_lines.append(f"")

    # music_id analysis
    music_issues = []
    if music_available:
        if music_result.get("missing_rate", 0) >= 5:
            music_issues.append(f"缺失率 = {music_result['missing_rate']}% >= 5%，大量样本无 music_id")
        if music_result.get("avg_samples_per_id", 2) < 2.0:
            music_issues.append(f"平均样本数/ID = {music_result['avg_samples_per_id']} < 2.0")
        if music_result.get("singleton_id_ratio", 0) >= 50:
            music_issues.append(f"singleton ID 占比 = {music_result['singleton_id_ratio']}% >= 50%")
        if music_result.get("val_test_oov_rate", 0) >= 20:
            music_issues.append(f"OOV 率 = {music_result['val_test_oov_rate']}% >= 20%")
    else:
        music_issues.append("music_id 全体缺失（raw_music 表中 1361/5000 行 music_id 为空），无法用于 embedding")

    if music_issues:
        md_lines.append(f"**music_id 问题:**")
        for issue in music_issues:
            md_lines.append(f"- {issue}")
        md_lines.append(f"")
    else:
        md_lines.append(f"**music_id 通过所有阈值检查。**")
        md_lines.append(f"")

    # Final recommendation
    md_lines += [
        f"### 5.2 推荐结论",
        f"",
    ]

    if overall_rec == "proceed_to_impl":
        md_lines += [
            f"**建议进入 ID embedding 实现。** 两个字段均通过覆盖率/OOV 检查。",
            f"",
        ]
    elif overall_rec == "do_not_implement_id_embedding":
        md_lines += [
            f"**不建议实现 ID embedding。** author_id 和 music_id 均不满足基本条件。",
            f"",
            f"**建议转向替代方向:**",
            f"1. **Batch 14J: 文本分支增强** — per-field TF-IDF/SVD, SVD 维度 32→64/128",
            f"2. **融合策略优化** — attention-based fusion, late fusion",
            f"",
        ]
    elif overall_rec == "proceed_author_only":
        md_lines += [
            f"**建议仅接入 author_id。** music_id 不满足条件。",
            f"",
        ]
    elif overall_rec == "proceed_music_only":
        md_lines += [
            f"**建议仅接入 music_id。** author_id 不满足条件。",
            f"",
        ]
    elif overall_rec == "need_data_fix":
        md_lines += [
            f"**需要数据修复或额外评估。** 当前不满足直接进入实现的条件。",
            f"",
        ]

    md_lines += [
        f"---",
        f"",
        f"## 6. 输出文件",
        f"",
        f"| 文件 | 路径 |",
        f"|------|------|",
        f"| JSON 报告 | `id_coverage_report.json` |",
        f"| 汇总 CSV | `id_coverage_summary.csv` |",
        f"| 频次 Top CSV | `id_frequency_top.csv` |",
        f"| 本报告 | `id_coverage_report.md` |",
        f"",
    ]

    md_path = output_dir / "id_coverage_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"  Wrote {md_path}")

    # ------------------------------------------------------------------
    # 9. Print summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"  Run ID:              {rid}")
    print(f"  Output dir:          {output_dir}")
    print(f"  author_id decision:  {author_result['decision'].upper()}")
    print(f"    unique(train):     {author_result['train_unique']}")
    print(f"    avg/ID:            {author_result['avg_samples_per_id']}")
    print(f"    singleton ratio:   {author_result['singleton_id_ratio']}%")
    print(f"    val OOV rate:      {author_result['val_oov_rate']}%")
    print(f"    test OOV rate:     {author_result['test_oov_rate']}%")
    print(f"  music_id decision:   {music_result['decision'].upper()}")
    if music_available:
        print(f"    missing rate:      {music_result['missing_rate']}%")
        print(f"    unique(train):     {music_result['train_unique']}")
        print(f"    avg/ID:            {music_result['avg_samples_per_id']}")
        print(f"    singleton ratio:   {music_result['singleton_id_ratio']}%")
        print(f"    val OOV rate:      {music_result['val_oov_rate']}%")
        print(f"    test OOV rate:     {music_result['test_oov_rate']}%")
    else:
        print(f"    (unavailable)")
    print(f"  Overall:             {overall_rec}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
