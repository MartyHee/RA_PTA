"""
build_real_raw_5000_package.py

正式 real_raw_5000 数据包构建 + high-confidence 辅助文件生成。

功能：
1. 确认 candidate 文件已到位（不做复制，只验证）
2. 合并 part1-part5 的 real_web_video_meta → high_confidence_web_video_meta
3. 输出 high_confidence_video_ids.txt
4. 输出 high_confidence_filter_report.json

使用方法：
  python src/processing/build_real_raw_5000_package.py
"""

import csv
import json
import os
from datetime import datetime

PROJECT_ROOT = "D:/CodeData/Program Coding/ByteDance/RA_PTA/douyin_data_project"

RUN_IDS = [
    "20260509_212100",
    "20260510_151052",
    "20260510_183114",
    "20260510_210600",
    "20260511_084923",
]

BATCH_DIRS = [os.path.join(PROJECT_ROOT, "data/interim", rid) for rid in RUN_IDS]

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data/processed/real_raw_5000")

CANDIDATE_DIR = os.path.join(PROJECT_ROOT, "data/interim/real_raw_5000_candidate")
FORMAL_DIR = os.path.join(PROJECT_ROOT, "data/interim/real_raw_5000")

HIGH_CONF_META_NAME = "high_confidence_web_video_meta_real_raw_5000.csv"
VIDEO_IDS_NAME = "high_confidence_video_ids.txt"
REPORT_NAME = "high_confidence_filter_report.json"


def get_meta_path(batch_dir):
    run_id = os.path.basename(batch_dir)
    return os.path.join(batch_dir, f"real_web_video_meta_{run_id}.csv")


def count_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        return sum(1 for _ in reader) - 1  # minus header


def merge_meta_files():
    """合并所有 batch 的 real_web_video_meta，按 video_id 去重，优先保留 exact+high。"""
    all_rows = []
    fieldnames = None
    stats = {
        "input_run_ids": RUN_IDS,
        "total_raw_rows": 0,
        "after_dedup_rows": 0,
        "high_confidence_rows": 0,
        "high_confidence_video_ids": [],
    }

    for batch_dir in BATCH_DIRS:
        meta_path = get_meta_path(batch_dir)
        if not os.path.exists(meta_path):
            print(f"[WARN] Missing: {meta_path}")
            continue
        lines_before = count_lines(meta_path)
        stats["total_raw_rows"] += lines_before
        print(f"  {os.path.basename(batch_dir)}: {lines_before} rows")

        with open(meta_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                all_rows.append(row)

    # Dedup by video_id: keep first occurrence (batches are ordered, earlier = better)
    seen = set()
    deduped = []
    for row in all_rows:
        vid = row.get("video_id", "").strip()
        if vid in seen:
            continue
        seen.add(vid)
        deduped.append(row)
    stats["after_dedup_rows"] = len(deduped)
    stats["unique_video_ids"] = len(seen)

    print(f"\n  Total raw rows: {stats['total_raw_rows']}")
    print(f"  After dedup: {stats['after_dedup_rows']}")

    # Filter high-confidence: match_type=exact AND confidence=high
    high_conf_rows = []
    for row in deduped:
        mt = row.get("match_type", "").strip().lower()
        cf = row.get("confidence", "").strip().lower()
        if mt == "exact" and cf == "high":
            high_conf_rows.append(row)
    stats["high_confidence_rows"] = len(high_conf_rows)
    stats["high_confidence_ratio"] = round(
        len(high_conf_rows) / len(deduped), 4
    ) if deduped else 0
    stats["high_confidence_video_ids"] = [
        r["video_id"] for r in high_conf_rows
    ]

    print(f"  High-confidence: {len(high_conf_rows)} / {len(deduped)} ({stats['high_confidence_ratio']:.1%})")

    return fieldnames, high_conf_rows, deduped, stats


def write_high_conf_meta(fieldnames, high_conf_rows):
    """写入 high_confidence_web_video_meta CSV。"""
    out_path = os.path.join(PROCESSED_DIR, HIGH_CONF_META_NAME)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(high_conf_rows)
    print(f"\n  Written: {out_path} ({len(high_conf_rows)} rows)")
    return out_path


def write_video_ids_file(video_ids):
    """写入 high_confidence_video_ids.txt，每行一个 video_id。"""
    out_path = os.path.join(PROCESSED_DIR, VIDEO_IDS_NAME)
    with open(out_path, "w", encoding="utf-8") as f:
        for vid in video_ids:
            f.write(vid + "\n")
    print(f"  Written: {out_path} ({len(video_ids)} IDs)")
    return out_path


def write_filter_report(stats):
    """写入 high_confidence_filter_report.json。"""
    report = {
        "build_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_run_ids": stats["input_run_ids"],
        "total_raw_rows": stats["total_raw_rows"],
        "unique_video_ids": stats["unique_video_ids"],
        "high_confidence": {
            "count": stats["high_confidence_rows"],
            "ratio": stats["high_confidence_ratio"],
            "description": "match_type=exact AND confidence=high",
        },
        "filter_criteria": {
            "match_type": "exact",
            "confidence": "high",
        },
        "output_files": {
            "high_confidence_meta": os.path.join(PROCESSED_DIR, HIGH_CONF_META_NAME),
            "video_ids": os.path.join(PROCESSED_DIR, VIDEO_IDS_NAME),
        },
        "data_package_relation": {
            "full_data": "data/interim/real_raw_5000/ (11 raw tables, 5000 video_ids)",
            "high_confidence_filter": "data/processed/real_raw_5000/ (exact+high subset)",
            "note": "high_confidence_video_ids.txt 可用于上层特征构建时过滤样本。默认标准输入仍是 data/interim/real_raw_5000/ 下的 11 张 raw 表。",
        },
    }
    out_path = os.path.join(PROCESSED_DIR, REPORT_NAME)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Written: {out_path}")
    return out_path


def verify_formal_tables():
    """验证正式目录中文件完整性。"""
    EXPECTED_TABLES = [
        "raw_video_detail_real_raw_5000.csv",
        "raw_author_real_raw_5000.csv",
        "raw_music_real_raw_5000.csv",
        "raw_hashtag_real_raw_5000.csv",
        "raw_video_tag_real_raw_5000.csv",
        "raw_video_media_real_raw_5000.csv",
        "raw_video_status_control_real_raw_5000.csv",
        "raw_chapter_real_raw_5000.csv",
        "raw_comment_real_raw_5000.csv",
        "raw_related_video_real_raw_5000.csv",
        "raw_crawl_log_real_raw_5000.csv",
        "merge_report.json",
        "quality_audit.json",
        "table_summary.csv",
    ]
    print(f"\n=== Verifying formal tables ===")
    all_ok = True
    for fname in EXPECTED_TABLES:
        path = os.path.join(FORMAL_DIR, fname)
        if not os.path.exists(path):
            print(f"  [FAIL] Missing: {fname}")
            all_ok = False
        else:
            size = os.path.getsize(path)
            print(f"  [OK]   {fname} ({size:,} bytes)")
    return all_ok


def main():
    print("=" * 60)
    print("real_raw_5000 数据包构建 + high-confidence 辅助文件生成")
    print("=" * 60)

    # Step 1: Verify candidate tables (no modifications)
    print("\n[1/4] 验证 candidate 数据完整性...")
    if not os.path.exists(CANDIDATE_DIR):
        print(f"[ERROR] Candidate dir not found: {CANDIDATE_DIR}")
        return

    # Step 2: Verify formal tables already placed
    print("\n[2/4] 验证正式目录文件...")
    verify_formal_tables()

    # Step 3: Merge meta files → high confidence
    print("\n[3/4] 合并 real_web_video_meta → high-confidence 过滤...")
    fieldnames, high_conf_rows, all_deduped, stats = merge_meta_files()
    write_high_conf_meta(fieldnames, high_conf_rows)
    write_video_ids_file(stats["high_confidence_video_ids"])
    write_filter_report(stats)

    # Step 4: Final summary
    print("\n[4/4] 构建完成")
    print(f"\n  Formal package: {FORMAL_DIR}")
    print(f"  Processed aux:  {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
