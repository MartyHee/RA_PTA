"""
sample0427 schema 校验与数据概览

基于 sample_data_dictionary.md 整理的 schema_sample0427.yaml，
检查 11 张表的字段完整性、缺失情况、关键关联可用性。

用法:
    python src/data/validate_schema.py
    python src/data/validate_schema.py --config configs/common/data_paths.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.io import read_csv_safe

# =============================================================================
# 默认路径
# =============================================================================
DEFAULT_DATA_PATH_CONFIG = str(
    PROJECT_ROOT / "configs" / "common" / "data_paths.yaml"
)
DEFAULT_SCHEMA_CONFIG = str(
    PROJECT_ROOT / "configs" / "common" / "schema_sample0427.yaml"
)

# =============================================================================
# CSV 路径映射（与 load_sample0427.py 保持一致）
# =============================================================================
CSV_FILE_MAP: dict[str, str] = {
    "raw_video_detail": "sample0427_raw_video_detail.csv",
    "raw_author": "sample0427_raw_author.csv",
    "raw_music": "sample0427_raw_music.csv",
    "raw_hashtag": "sample0427_raw_hashtag.csv",
    "raw_video_tag": "sample0427_raw_video_tag.csv",
    "raw_video_media": "sample0427_raw_video_media.csv",
    "raw_video_status_control": "sample0427_raw_video_status_control.csv",
    "raw_chapter": "sample0427_raw_chapter.csv",
    "raw_comment": "sample0427_raw_comment.csv",
    "raw_related_video": "sample0427_raw_related_video.csv",
    "raw_crawl_log": "sample0427_raw_crawl_log.csv",
}


def _list_to_str(items: list[str] | None, sep: str = "; ") -> str:
    """将列表转为分隔符字符串。"""
    if not items:
        return ""
    return sep.join(str(x) for x in items)


def check_table_schema(
    df: pd.DataFrame,
    table_cfg: dict[str, Any],
    table_name: str,
) -> dict[str, Any]:
    """检查单张表的 schema 完整性，返回校验结果。"""
    expected_cols: list[str] = table_cfg.get("expected_columns", [])
    important_cols: list[str] = table_cfg.get("important_columns", [])
    gen_placeholder_cols: list[str] = table_cfg.get(
        "generated_or_placeholder_columns", []
    )
    actual_cols: list[str] = list(df.columns)

    set_expected = set(expected_cols)
    set_actual = set(actual_cols)

    missing_cols = sorted(set_expected - set_actual)
    extra_cols = sorted(set_actual - set_expected)

    # 字段匹配状态
    if not missing_cols and not extra_cols:
        column_match_status = "full_match"
    elif missing_cols and not extra_cols:
        column_match_status = "has_missing"
    elif not missing_cols and extra_cols:
        column_match_status = "has_extra"
    else:
        column_match_status = "has_missing_and_extra"

    # 缺失统计
    total_cells = df.size
    missing_cells = int(df.isnull().sum().sum())
    missing_rate_overall = (
        round(missing_cells / total_cells, 6) if total_cells > 0 else 0.0
    )
    duplicate_rows = int(df.duplicated().sum())

    # 全空字段和高缺失字段
    all_null_cols: list[str] = []
    high_missing_cols: list[str] = []
    per_col_missing: list[dict[str, Any]] = []

    for col in actual_cols:
        miss_count = int(df[col].isnull().sum())
        miss_rate = round(miss_count / len(df), 6) if len(df) > 0 else 0.0
        is_all_null = miss_rate >= 1.0
        is_high_missing = miss_rate >= 0.8
        is_important = col in important_cols
        is_gen_or_placeholder = col in gen_placeholder_cols

        if is_all_null:
            all_null_cols.append(col)
        if is_high_missing:
            high_missing_cols.append(col)

        # 字段备注
        notes_parts = []
        if is_gen_or_placeholder:
            notes_parts.append("规则生成/占位字段")

        # 从 schema notes 中获取已知的空字段信息
        table_notes = table_cfg.get("notes", "")
        if col in table_notes:
            if "整列全空" in table_notes and col in [
                c for c in actual_cols if c not in gen_placeholder_cols
            ]:
                pass  # 在表级 notes 中已涵盖

        per_col_missing.append({
            "table_name": table_name,
            "column_name": col,
            "missing_count": miss_count,
            "missing_rate": miss_rate,
            "is_all_null": is_all_null,
            "is_high_missing": is_high_missing,
            "is_important_column": is_important,
            "is_generated_or_placeholder": is_gen_or_placeholder,
            "notes": "; ".join(notes_parts),
        })

    # 检查关键 ID 字段是否存在
    key_id_fields = [
        "video_id", "author_id", "music_id", "hashtag_id",
        "comment_id", "related_video_id", "crawl_batch_id",
    ]
    missing_key_fields = sorted(
        kf for kf in key_id_fields if kf in expected_cols and kf not in actual_cols
    )

    # 确定表级 schema 状态
    if missing_key_fields:
        schema_status = "fail"
    elif missing_cols:
        schema_status = "warning"
    else:
        schema_status = "pass"

    # 构建表级 notes
    notes_parts = []
    if gen_placeholder_cols:
        notes_parts.append(
            f"{len(gen_placeholder_cols)} 个字段为规则生成/占位"
        )
    if all_null_cols:
        notes_parts.append(f"{len(all_null_cols)} 个字段全空")
    if extra_cols:
        notes_parts.append(f"{len(extra_cols)} 个额外字段")
    if missing_cols:
        notes_parts.append(f"{len(missing_cols)} 个缺失字段")

    result: dict[str, Any] = {
        "table_name": table_name,
        "file_exists": True,
        "row_count": len(df),
        "column_count": len(actual_cols),
        "expected_column_count": len(expected_cols),
        "actual_column_count": len(actual_cols),
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
        "missing_key_fields": missing_key_fields,
        "column_match_status": column_match_status,
        "schema_status": schema_status,
        "all_null_columns": all_null_cols,
        "high_missing_columns": high_missing_cols,
        "duplicate_rows": duplicate_rows,
        "missing_cells_total": missing_cells,
        "missing_rate_overall": missing_rate_overall,
        "notes": "; ".join(notes_parts),
        "per_column_missing": per_col_missing,
    }
    return result


def check_key_relation(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    source_key: str,
    target_key: str,
    source_table: str,
    target_table: str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """检查两张表之间的关键关联可用性。"""
    if source_key not in source_df.columns:
        return {
            "source_table": source_table,
            "source_key": source_key,
            "target_table": target_table,
            "target_key": target_key,
            "source_non_null_count": 0,
            "matched_count": 0,
            "unmatched_count": 0,
            "match_rate": 0.0,
            "relation_status": "fail",
            "notes": f"源表 {source_table} 中缺少键列 {source_key}",
        }
    if target_key not in target_df.columns:
        return {
            "source_table": source_table,
            "source_key": source_key,
            "target_table": target_table,
            "target_key": target_key,
            "source_non_null_count": 0,
            "matched_count": 0,
            "unmatched_count": 0,
            "match_rate": 0.0,
            "relation_status": "fail",
            "notes": f"目标表 {target_table} 中缺少键列 {target_key}",
        }

    source_vals = source_df[source_key].dropna()
    source_non_null_count = len(source_vals)

    if source_non_null_count == 0:
        return {
            "source_table": source_table,
            "source_key": source_key,
            "target_table": target_table,
            "target_key": target_key,
            "source_non_null_count": 0,
            "matched_count": 0,
            "unmatched_count": 0,
            "match_rate": 0.0,
            "relation_status": "fail",
            "notes": f"源表 {source_table} 中 {source_key} 全部为空，无法关联",
        }

    target_vals = set(target_df[target_key].dropna().unique())
    matched = source_vals.isin(target_vals).sum()
    unmatched = source_non_null_count - matched
    match_rate = round(matched / source_non_null_count, 6)

    if match_rate >= threshold:
        relation_status = "pass"
        notes = f"关联正常（匹配率 {match_rate:.2%}）"
    elif match_rate > 0:
        relation_status = "warning"
        notes = f"部分匹配（匹配率 {match_rate:.2%}），{unmatched} 条无对应目标"
    else:
        relation_status = "fail"
        notes = f"无匹配（匹配率 0%），所有 {source_key} 值在目标表中不存在"

    return {
        "source_table": source_table,
        "source_key": source_key,
        "target_table": target_table,
        "target_key": target_key,
        "source_non_null_count": int(source_non_null_count),
        "matched_count": int(matched),
        "unmatched_count": int(unmatched),
        "match_rate": match_rate,
        "relation_status": relation_status,
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="sample0427 schema 校验与数据概览"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_DATA_PATH_CONFIG,
        help="数据路径配置文件路径",
    )
    parser.add_argument(
        "--schema-config",
        type=str,
        default=DEFAULT_SCHEMA_CONFIG,
        help="schema 配置文件路径",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. 加载配置
    # ------------------------------------------------------------------
    data_cfg = load_config(args.config)
    schema_cfg = load_config(args.schema_config)

    sample0427_dir = Path(data_cfg["sample0427_dir"])
    data_check_dir = Path(data_cfg["data_check_dir"])

    print(f"[validate_schema] 数据目录: {sample0427_dir}")
    print(f"[validate_schema] schema 配置: {args.schema_config}")
    print(f"[validate_schema] 输出目录: {data_check_dir}")
    data_check_dir.mkdir(parents=True, exist_ok=True)

    if not sample0427_dir.exists():
        print(f"[validate_schema] 错误: 数据目录不存在: {sample0427_dir}")
        sys.exit(1)

    global_cfg = schema_cfg.get("global", {})
    high_missing_threshold = global_cfg.get("high_missing_threshold", 0.8)
    relation_threshold = global_cfg.get("key_relation_match_threshold", 0.5)

    tables_cfg: dict[str, Any] = schema_cfg.get("tables", {})
    key_relations_cfg: list[dict[str, Any]] = schema_cfg.get(
        "key_relations", []
    )

    # ------------------------------------------------------------------
    # 2. 读取所有 CSV 并逐表校验
    # ------------------------------------------------------------------
    all_dfs: dict[str, pd.DataFrame] = {}
    schema_results: list[dict[str, Any]] = []
    field_missing_rows: list[dict[str, Any]] = []

    missing_tables: list[str] = []
    failed_tables: list[str] = []
    tables_with_missing_columns: list[str] = []
    tables_with_extra_columns: list[str] = []
    tables_with_all_null_columns: list[str] = []
    generated_field_summary: dict[str, list[str]] = {}

    print(f"\n[validate_schema] 开始校验 {len(tables_cfg)} 张表...\n")

    for table_name in tables_cfg:
        table_cfg = tables_cfg[table_name]

        if table_name not in CSV_FILE_MAP:
            print(f"  [{table_name}] 跳过: CSV_FILE_MAP 中未定义")
            continue

        csv_name = CSV_FILE_MAP[table_name]
        csv_path = sample0427_dir / csv_name
        print(f"  [{table_name}] 检查 {csv_name} ...", end=" ")

        if not csv_path.exists():
            print("文件不存在")
            missing_tables.append(table_name)
            schema_results.append({
                "table_name": table_name,
                "file_exists": False,
                "row_count": 0,
                "column_count": 0,
                "expected_column_count": len(
                    table_cfg.get("expected_columns", [])
                ),
                "actual_column_count": 0,
                "missing_columns": table_cfg.get("expected_columns", []),
                "extra_columns": [],
                "missing_key_fields": [],
                "column_match_status": "missing_file",
                "schema_status": "fail",
                "all_null_columns": [],
                "high_missing_columns": [],
                "duplicate_rows": 0,
                "missing_cells_total": 0,
                "missing_rate_overall": 0.0,
                "notes": "文件不存在",
                "per_column_missing": [],
            })
            continue

        try:
            df, encoding = read_csv_safe(str(csv_path))
            all_dfs[table_name] = df
            result = check_table_schema(df, table_cfg, table_name)

            # 收集表级问题
            if result["schema_status"] == "fail":
                failed_tables.append(table_name)
            if result["missing_columns"]:
                tables_with_missing_columns.append(table_name)
            if result["extra_columns"]:
                tables_with_extra_columns.append(table_name)
            if result["all_null_columns"]:
                tables_with_all_null_columns.append(table_name)

            # 收集生成/占位字段信息
            gen_cols = table_cfg.get("generated_or_placeholder_columns", [])
            if gen_cols:
                generated_field_summary[table_name] = gen_cols

            # 收集字段级缺失信息
            field_missing_rows.extend(result.pop("per_column_missing"))

            schema_results.append(result)
            print(
                f"OK | {result['row_count']} 行 x {result['column_count']} 列 "
                f"| 状态: {result['schema_status']}"
            )
            if result["missing_columns"]:
                print(
                    f"    [W] 缺失字段: {_list_to_str(result['missing_columns'])}"
                )
            if result["extra_columns"]:
                print(
                    f"    [W] 额外字段: {_list_to_str(result['extra_columns'])}"
                )
            if result["all_null_columns"]:
                print(
                    f"    [W] 全空字段: {_list_to_str(result['all_null_columns'][:5])}"
                    f"{'...' if len(result['all_null_columns']) > 5 else ''}"
                )

        except Exception as e:
            print(f"读取失败: {e}")
            failed_tables.append(table_name)
            schema_results.append({
                "table_name": table_name,
                "file_exists": True,
                "row_count": 0,
                "column_count": 0,
                "expected_column_count": len(
                    table_cfg.get("expected_columns", [])
                ),
                "actual_column_count": 0,
                "missing_columns": table_cfg.get("expected_columns", []),
                "extra_columns": [],
                "missing_key_fields": [],
                "column_match_status": "read_failed",
                "schema_status": "fail",
                "all_null_columns": [],
                "high_missing_columns": [],
                "duplicate_rows": 0,
                "missing_cells_total": 0,
                "missing_rate_overall": 0.0,
                "notes": f"读取失败: {e}",
                "per_column_missing": [],
            })

    # ------------------------------------------------------------------
    # 3. 关键关联检查
    # ------------------------------------------------------------------
    print(f"\n[validate_schema] 开始关键关联检查 ({len(key_relations_cfg)} 项)...\n")
    relation_results: list[dict[str, Any]] = []

    for rel in key_relations_cfg:
        src_table = rel["source_table"]
        src_key = rel["source_key"]
        tgt_table = rel["target_table"]
        tgt_key = rel["target_key"]
        desc = rel.get("description", f"{src_table}.{src_key} → {tgt_table}.{tgt_key}")

        print(f"  [{desc}] ...", end=" ")

        if src_table not in all_dfs:
            print(f"跳过: 源表 {src_table} 未成功读取")
            relation_results.append({
                "source_table": src_table,
                "source_key": src_key,
                "target_table": tgt_table,
                "target_key": tgt_key,
                "source_non_null_count": 0,
                "matched_count": 0,
                "unmatched_count": 0,
                "match_rate": 0.0,
                "relation_status": "fail",
                "notes": f"源表 {src_table} 未成功读取",
            })
            continue

        if tgt_table not in all_dfs:
            print(f"跳过: 目标表 {tgt_table} 未成功读取")
            relation_results.append({
                "source_table": src_table,
                "source_key": src_key,
                "target_table": tgt_table,
                "target_key": tgt_key,
                "source_non_null_count": 0,
                "matched_count": 0,
                "unmatched_count": 0,
                "match_rate": 0.0,
                "relation_status": "fail",
                "notes": f"目标表 {tgt_table} 未成功读取",
            })
            continue

        result = check_key_relation(
            source_df=all_dfs[src_table],
            target_df=all_dfs[tgt_table],
            source_key=src_key,
            target_key=tgt_key,
            source_table=src_table,
            target_table=tgt_table,
            threshold=relation_threshold,
        )
        relation_results.append(result)
        print(
            f"匹配率: {result['match_rate']:.2%} "
            f"({result['matched_count']}/{result['source_non_null_count']}) "
            f"| 状态: {result['relation_status']}"
        )

    # ------------------------------------------------------------------
    # 4. raw_related_video 额外检查
    # ------------------------------------------------------------------
    print(f"\n[validate_schema] 额外检查: raw_related_video...")
    related_video_extra: dict[str, Any] = {"table": "raw_related_video"}
    if "raw_related_video" in all_dfs:
        rv_df = all_dfs["raw_related_video"]
        has_source = "source_video_id" in rv_df.columns
        has_related = "related_video_id" in rv_df.columns
        related_video_extra["has_source_video_id"] = has_source
        related_video_extra["has_related_video_id"] = has_related
        related_video_extra["can_serve_as_edge_source"] = has_source and has_related
        if has_source:
            related_video_extra["source_non_null"] = int(
                rv_df["source_video_id"].notna().sum()
            )
        if has_related:
            related_video_extra["related_non_null"] = int(
                rv_df["related_video_id"].notna().sum()
            )
        related_video_extra["notes"] = (
            "可用于 GraphSAGE video-video 边构建（流程验证级别）"
            if (has_source and has_related)
            else "缺少关键边构建字段"
        )
        print(f"    has_source_video_id: {has_source}, "
              f"has_related_video_id: {has_related}")
    else:
        related_video_extra["has_source_video_id"] = False
        related_video_extra["has_related_video_id"] = False
        related_video_extra["can_serve_as_edge_source"] = False
        related_video_extra["notes"] = "raw_related_video 未成功读取"
        print("    未成功读取，跳过")

    # ------------------------------------------------------------------
    # 5. 确定总体状态
    # ------------------------------------------------------------------
    relation_fails = [
        r for r in relation_results if r["relation_status"] == "fail"
    ]
    relation_warnings = [
        r for r in relation_results if r["relation_status"] == "warning"
    ]
    schema_fails = [r for r in schema_results if r["schema_status"] == "fail"]
    schema_warnings = [
        r for r in schema_results if r["schema_status"] == "warning"
    ]

    if missing_tables or schema_fails:
        overall_status = "fail"
    elif schema_warnings or relation_fails or relation_warnings:
        overall_status = "warning"
    elif tables_with_all_null_columns or generated_field_summary:
        # 存在全空字段或生成/占位字段 → 结构完整但数据质量有限 → warning
        overall_status = "warning"
    else:
        overall_status = "pass"

    print(f"\n[validate_schema] 总体状态: {overall_status}")

    # ------------------------------------------------------------------
    # 6. 输出 schema_validation_report.csv
    # ------------------------------------------------------------------
    csv_report_path = data_check_dir / "schema_validation_report.csv"
    csv_rows = []
    for r in schema_results:
        csv_rows.append({
            "table_name": r["table_name"],
            "file_exists": r["file_exists"],
            "row_count": r["row_count"],
            "column_count": r["column_count"],
            "expected_column_count": r["expected_column_count"],
            "actual_column_count": r["actual_column_count"],
            "missing_columns": _list_to_str(r["missing_columns"]),
            "extra_columns": _list_to_str(r["extra_columns"]),
            "missing_key_fields": _list_to_str(r.get("missing_key_fields", [])),
            "all_null_columns": _list_to_str(r["all_null_columns"]),
            "high_missing_columns": _list_to_str(r["high_missing_columns"]),
            "duplicate_rows": r["duplicate_rows"],
            "schema_status": r["schema_status"],
            "notes": r["notes"],
        })
    pd.DataFrame(csv_rows).to_csv(
        csv_report_path, index=False, encoding="utf-8-sig"
    )
    print(f"\n[validate_schema] 表级校验报告已保存: {csv_report_path}")

    # ------------------------------------------------------------------
    # 7. 输出 field_missing_report.csv
    # ------------------------------------------------------------------
    field_report_path = data_check_dir / "field_missing_report.csv"
    field_rows_sorted = sorted(
        field_missing_rows,
        key=lambda x: (x["table_name"], x["missing_rate"]),
        reverse=True,
    )
    pd.DataFrame(field_rows_sorted).to_csv(
        field_report_path, index=False, encoding="utf-8-sig"
    )
    print(f"[validate_schema] 字段缺失报告已保存: {field_report_path}")

    # ------------------------------------------------------------------
    # 8. 输出 key_relation_check_report.csv
    # ------------------------------------------------------------------
    relation_report_path = data_check_dir / "key_relation_check_report.csv"
    pd.DataFrame(relation_results).to_csv(
        relation_report_path, index=False, encoding="utf-8-sig"
    )
    print(f"[validate_schema] 关键关联报告已保存: {relation_report_path}")

    # ------------------------------------------------------------------
    # 9. 输出 schema_validation_report.json
    # ------------------------------------------------------------------
    # 构建生成/占位字段摘要
    gen_field_summary_list = []
    for tbl, cols in generated_field_summary.items():
        gen_field_summary_list.append({
            "table_name": tbl,
            "generated_or_placeholder_count": len(cols),
            "columns": cols,
        })

    json_report = {
        "input_dir": str(sample0427_dir),
        "schema_config_path": args.schema_config,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "expected_table_count": len(tables_cfg),
        "actual_loaded_table_count": len(all_dfs),
        "missing_tables": missing_tables,
        "failed_tables": failed_tables,
        "tables_with_missing_columns": tables_with_missing_columns,
        "tables_with_extra_columns": tables_with_extra_columns,
        "tables_with_all_null_columns": tables_with_all_null_columns,
        "overall_status": overall_status,
        "relation_check_summary": {
            "total_relations": len(relation_results),
            "passed": sum(
                1 for r in relation_results if r["relation_status"] == "pass"
            ),
            "warning": sum(
                1 for r in relation_results if r["relation_status"] == "warning"
            ),
            "failed": sum(
                1 for r in relation_results if r["relation_status"] == "fail"
            ),
            "results": relation_results,
        },
        "generated_or_placeholder_field_summary": {
            "total_tables_with_generated_fields": len(generated_field_summary),
            "details": gen_field_summary_list,
        },
        "extra_checks": {
            "raw_related_video": related_video_extra,
        },
        "overall_notes": [
            "当前校验基于 sample0427 样本数据，所有结果为流程级验证。",
            "5 张完全补齐表（raw_video_tag / raw_video_status_control / "
            "raw_chapter / raw_comment / raw_related_video）的所有数据为样本补齐值。",
            "约 30 个字段整列全空，已在 field_missing_report.csv 中详细列出。",
            "部分 ID 字段为规则生成（sec_uid / unique_id / music_id / "
            "hashtag_id / related_video_id / tag_id），不可用于外部关联。",
            "关键关联检查验证了 9 组表间关联的覆盖度。",
            "overall_status=warning 不完全代表数据不可用，而是提示部分字段为占位值或全空。",
        ],
    }

    json_report_path = data_check_dir / "schema_validation_report.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, ensure_ascii=False, indent=2)
    print(f"[validate_schema] 总校验报告已保存: {json_report_path}")

    # ------------------------------------------------------------------
    # 10. 打印汇总
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Schema 校验汇总")
    print("=" * 60)
    print(f"  期望表数:               {len(tables_cfg)}")
    print(f"  成功读取:               {len(all_dfs)}")
    print(f"  缺失文件:               {len(missing_tables)}")
    print(f"  读取失败:               {len(failed_tables)}")
    print(f"  表级 fail:              {len(schema_fails)}")
    print(f"  表级 warning:           {len(schema_warnings)}")
    print(f"  缺失字段的表:           {len(tables_with_missing_columns)}")
    print(f"  含额外字段的表:         {len(tables_with_extra_columns)}")
    print(f"  含全空字段的表:         {len(tables_with_all_null_columns)}")
    print(f"  关联检查通过/警告/失败: "
          f"{json_report['relation_check_summary']['passed']}/"
          f"{json_report['relation_check_summary']['warning']}/"
          f"{json_report['relation_check_summary']['failed']}")
    print(f"  含生成/占位字段的表数:  {len(generated_field_summary)}")
    print(f"  overall_status:         {overall_status}")
    print("=" * 60)

    # 如果总体状态不是 pass，打印说明
    if overall_status != "pass":
        print(f"\n[validate_schema] 说明: overall_status={overall_status}。")
        if missing_tables or failed_tables:
            print(
                "  - 存在缺失或读取失败的表，请检查数据目录。"
            )
        if schema_warnings:
            print(
                "  - 部分表存在字段缺失/全空情况，详见报告。"
            )
        if tables_with_all_null_columns:
            print(
                "  - 部分表存在整列全空字段，这些字段在样本中无数据，不应作为特征输入。"
            )
        if generated_field_summary:
            print(
                "  - 部分表包含规则生成/占位字段，为 sample0427 样本数据的预期行为。"
            )

    if overall_status == "fail":
        sys.exit(1)
    else:
        print(f"\n[validate_schema] Schema 校验完成。")


if __name__ == "__main__":
    main()