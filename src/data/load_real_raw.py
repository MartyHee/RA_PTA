"""
real_raw_1000 数据读取与读取检查

读取 11 张 raw CSV 表，输出表级摘要和读取报告。

用法:
    D:/CodeData/software/Anaconda/Anaconda3/envs/ra/python.exe src/data/load_real_raw.py --config configs/common/real_raw_1000.yaml
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


def build_table_info(
    table_key: str,
    file_path: Path,
    read_success: bool,
    encoding_used: str | None,
    df: pd.DataFrame | None,
    error_message: str | None,
    allowed_empty: bool = False,
) -> dict:
    """构建单张表的读取摘要。"""
    info: dict = {
        "table_key": table_key,
        "table_name": f"raw_{table_key}",
        "file_name": file_path.name,
        "file_path": str(file_path),
        "file_exists": file_path.exists(),
        "read_success": read_success,
        "encoding_used": encoding_used,
        "row_count": None,
        "column_count": None,
        "columns": [],
        "is_empty": None,
        "allowed_empty": allowed_empty,
        "error_message": error_message,
    }
    if df is not None:
        info["row_count"] = len(df)
        info["column_count"] = len(df.columns)
        info["columns"] = list(df.columns)
        info["is_empty"] = len(df) == 0
    return info


def check_key_field(
    df: pd.DataFrame, field: str, table_label: str
) -> dict:
    """检查一个关键字段的非空值、unique 数量。"""
    result = {
        "table": table_label,
        "field": field,
        "non_null": None,
        "total": None,
        "non_null_rate": None,
        "unique_count": None,
    }
    if df is None or field not in df.columns:
        result["error"] = f"字段 {field} 不存在"
        return result
    result["non_null"] = int(df[field].notna().sum())
    result["total"] = len(df)
    result["non_null_rate"] = round(result["non_null"] / result["total"], 4) if result["total"] > 0 else 0.0
    result["unique_count"] = int(df[field].nunique())
    return result


def check_relation(
    source_df: pd.DataFrame,
    source_key: str,
    source_label: str,
    target_df: pd.DataFrame,
    target_key: str,
    target_label: str,
) -> dict:
    """检查 source 表的 key 能否匹配到 target 表。"""
    result = {
        "source_table": source_label,
        "source_key": source_key,
        "target_table": target_label,
        "target_key": target_key,
    }
    if source_df is None or target_df is None:
        result["status"] = "skipped"
        result["note"] = "源表或目标表未成功读取"
        return result
    if source_key not in source_df.columns:
        result["status"] = "skipped"
        result["note"] = f"源表缺少字段 {source_key}"
        return result
    if target_key not in target_df.columns:
        result["status"] = "skipped"
        result["note"] = f"目标表缺少字段 {target_key}"
        return result

    source_keys = source_df[source_key].dropna().unique()
    target_keys = set(target_df[target_key].dropna().unique())
    matched = sum(1 for k in source_keys if k in target_keys)
    match_rate = round(matched / len(source_keys), 4) if len(source_keys) > 0 else 0.0
    result["source_unique_keys"] = len(source_keys)
    result["matched_keys"] = matched
    result["match_rate"] = match_rate
    result["status"] = "pass" if match_rate == 1.0 else "warning"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="real_raw_1000 数据读取与读取检查")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "common" / "real_raw_1000.yaml"),
        help="real_raw_1000 配置文件路径",
    )
    args = parser.parse_args()

    # =========================================================================
    # 1. 加载配置
    # =========================================================================
    config_path = Path(args.config)
    print(f"[load_real_raw] 加载配置文件: {config_path}")
    cfg = load_config(str(config_path))

    dataset_name = cfg["dataset_name"]
    run_id = cfg["run_id"]
    root_rel = cfg["root"]
    delivery_doc = cfg["delivery_doc"]
    tables_cfg: dict[str, str] = cfg["tables"]
    critical_tables: list[str] = cfg.get("critical_tables", [])
    allowed_empty: list[str] = cfg.get("allowed_empty_tables", [])
    key_field_checks_cfg: dict[str, list[str]] = cfg.get("key_field_checks", {})
    relation_checks_cfg: list[dict[str, str]] = cfg.get("relation_checks", [])

    # 解析绝对路径
    root_dir = (PROJECT_ROOT / root_rel).resolve()
    print(f"[load_real_raw] 数据根目录: {root_dir}")

    if not root_dir.exists():
        print(f"[load_real_raw] 错误: 数据目录不存在: {root_dir}")
        sys.exit(1)

    # 输出目录
    outputs_dir = PROJECT_ROOT / "outputs"
    data_check_dir = outputs_dir / "data_check"
    data_check_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 2. 逐表读取
    # =========================================================================
    table_keys = list(tables_cfg.keys())
    loaded_dfs: dict[str, pd.DataFrame] = {}
    tables_info: list[dict] = []
    loaded_tables: list[str] = []
    empty_tables: list[str] = []
    missing_files: list[str] = []
    failed_tables: list[str] = []
    total_rows = 0
    total_columns = 0

    print(f"\n[load_real_raw] 开始读取 {len(tables_cfg)} 张 raw 表...\n")

    for table_key in table_keys:
        file_name = tables_cfg[table_key]
        file_path = root_dir / file_name
        is_allowed_empty = table_key in allowed_empty

        print(f"  [raw_{table_key}] 读取 {file_name} ...", end=" ")

        if not file_path.exists():
            print("文件不存在")
            missing_files.append(file_name)
            info = build_table_info(
                table_key=table_key,
                file_path=file_path,
                read_success=False,
                encoding_used=None,
                df=None,
                error_message="文件不存在",
                allowed_empty=is_allowed_empty,
            )
            tables_info.append(info)
            continue

        try:
            df, encoding_used = read_csv_safe(str(file_path))
            info = build_table_info(
                table_key=table_key,
                file_path=file_path,
                read_success=True,
                encoding_used=encoding_used,
                df=df,
                error_message=None,
                allowed_empty=is_allowed_empty,
            )

            loaded_dfs[table_key] = df
            loaded_tables.append(table_key)
            total_rows += len(df)
            total_columns += len(df.columns)

            # 空表处理：允许的空表不视为失败
            if len(df) == 0:
                empty_tables.append(table_key)
                if is_allowed_empty:
                    print(f"OK | 0 行（允许空表）| 编码: {encoding_used}")
                else:
                    print(f"OK | 0 行（非预期空表）| 编码: {encoding_used}")
            else:
                print(
                    f"OK | {len(df)} 行 x {len(df.columns)} 列 "
                    f"| 编码: {encoding_used}"
                )
        except Exception as e:
            print(f"失败: {e}")
            failed_tables.append(table_key)
            info = build_table_info(
                table_key=table_key,
                file_path=file_path,
                read_success=False,
                encoding_used=None,
                df=None,
                error_message=str(e),
            )

        tables_info.append(info)

    # =========================================================================
    # 3. 关键字段检查
    # =========================================================================
    print("\n[load_real_raw] 关键字段检查...")
    key_field_results: list[dict] = []
    for table_key, fields in key_field_checks_cfg.items():
        df = loaded_dfs.get(table_key)
        table_label = f"raw_{table_key}"
        for field in fields:
            result = check_key_field(df, field, table_label)
            key_field_results.append(result)
            if result.get("error"):
                print(f"  [{table_label}.{field}] 错误: {result['error']}")
            else:
                print(
                    f"  [{table_label}.{field}] "
                    f"非空: {result['non_null']}/{result['total']} "
                    f"({result['non_null_rate']*100:.1f}%), "
                    f"unique: {result['unique_count']}"
                )

    # video_id 特有检查：1000 unique
    if "video_detail" in loaded_dfs:
        vd = loaded_dfs["video_detail"]
        if "video_id" in vd.columns:
            n_unique_video_id = int(vd["video_id"].nunique())
            print(f"\n[load_real_raw] raw_video_detail.video_id unique: {n_unique_video_id}")
            if n_unique_video_id != 1000:
                print(f"[load_real_raw] 警告: video_id unique 数 ({n_unique_video_id}) 不等于 1000")

    # =========================================================================
    # 4. 跨表关联检查
    # =========================================================================
    print("\n[load_real_raw] 跨表关联检查...")
    relation_results: list[dict] = []
    for rc in relation_checks_cfg:
        src_key = rc["source"]
        src_field = rc["source_key"]
        tgt_key = rc["target"]
        tgt_field = rc["target_key"]
        result = check_relation(
            source_df=loaded_dfs.get(src_key),
            source_key=src_field,
            source_label=f"raw_{src_key}",
            target_df=loaded_dfs.get(tgt_key),
            target_key=tgt_field,
            target_label=f"raw_{tgt_key}",
        )
        relation_results.append(result)
        if result["status"] == "pass":
            print(f"  [{result['source_table']}.{src_field}] -> "
                  f"[{result['target_table']}.{tgt_field}] "
                  f"匹配率: {result['match_rate']*100:.1f}% ({result['matched_keys']}/{result['source_unique_keys']}) [OK]")
        elif result["status"] == "warning":
            print(f"  [{result['source_table']}.{src_field}] -> "
                  f"[{result['target_table']}.{tgt_field}] "
                  f"匹配率: {result['match_rate']*100:.1f}% [WARN]")
        else:
            print(f"  [{result['source_table']}.{src_field}] -> "
                  f"[{result['target_table']}.{tgt_field}] "
                  f"跳过: {result.get('note', '')}")

    # =========================================================================
    # 5. 输出表级摘要 CSV
    # =========================================================================
    summary_csv_path = data_check_dir / "real_raw_1000_table_summary.csv"
    summary_rows = []
    for info in tables_info:
        summary_rows.append({
            "table_key": info["table_key"],
            "table_name": info["table_name"],
            "file_name": info["file_name"],
            "file_exists": info["file_exists"],
            "read_success": info["read_success"],
            "encoding_used": info["encoding_used"],
            "row_count": info["row_count"],
            "column_count": info["column_count"],
            "is_empty": info["is_empty"],
            "allowed_empty": info["allowed_empty"],
            "error_message": info["error_message"] or "",
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[load_real_raw] 表级摘要已保存: {summary_csv_path}")

    # =========================================================================
    # 6. 输出读取报告 JSON
    # =========================================================================
    known_limitations = [
        "raw_video_tag 当前 0 行，详情页响应未发现 video_tag 结构。",
        "raw_chapter 当前 0 行，详情页响应未发现 chapter_list 结构。",
        "none-match 样本占 21.2%，部分互动和媒体字段覆盖率低。",
        "当前不包含真实曝光、点击、完播、转化标签。",
        "当前来自公开网页端数据，不代表平台内部完整数据。",
    ]

    report: dict[str, Any] = {
        "dataset_name": dataset_name,
        "run_id": run_id,
        "root_path": str(root_dir),
        "delivery_doc": str(root_dir / delivery_doc),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "table_status": {
            info["table_name"]: {
                "file_name": info["file_name"],
                "read_success": info["read_success"],
                "row_count": info["row_count"],
                "column_count": info["column_count"],
                "is_empty": info["is_empty"],
                "allowed_empty": info["allowed_empty"],
            }
            for info in tables_info
        },
        "loaded_table_count": len(loaded_tables),
        "empty_tables": [f"raw_{k}" for k in empty_tables],
        "missing_files": missing_files,
        "failed_tables": [f"raw_{k}" for k in failed_tables],
        "key_field_checks": key_field_results,
        "relation_checks": relation_results,
        "known_limitations": known_limitations,
        "next_step": "完成读取检查后，下一步可基于 real_raw_1000 构建 tabular 数据集。",
    }

    report_json_path = data_check_dir / "real_raw_1000_load_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[load_real_raw] 读取报告已保存: {report_json_path}")

    # =========================================================================
    # 7. 打印汇总
    # =========================================================================
    print("\n" + "=" * 60)
    print("读取汇总")
    print("=" * 60)
    print(f"  数据集:           {dataset_name}")
    print(f"  run_id:           {run_id}")
    print(f"  期望表数:         {len(tables_cfg)}")
    print(f"  成功读取:         {len(loaded_tables)}")
    print(f"  缺失文件:         {len(missing_files)}")
    print(f"  读取失败:         {len(failed_tables)}")
    print(f"  空表:             {len(empty_tables)} ({', '.join(empty_tables) if empty_tables else '无'})")
    print(f"  总行数:           {total_rows}")
    print(f"  总列数:           {total_columns}")
    if missing_files:
        print(f"  缺失文件列表:     {', '.join(missing_files)}")
    if failed_tables:
        print(f"  失败表列表:       {', '.join(failed_tables)}")
    print("=" * 60)

    # 检查关键表
    print("\n关键表非空检查:")
    critical_ok = True
    for ct in critical_tables:
        df = loaded_dfs.get(ct)
        table_label = f"raw_{ct}"
        if df is not None and len(df) > 0:
            print(f"  [OK] {table_label}: {len(df)} 行")
        elif df is not None and len(df) == 0:
            print(f"  [FAIL] {table_label}: 空表（预期非空）")
            critical_ok = False
        else:
            print(f"  [FAIL] {table_label}: 未读取")
            critical_ok = False

    if critical_ok:
        print("[load_real_raw] 所有关键表非空检查通过。")
    else:
        print("[load_real_raw] 部分关键表为空或未读取，请检查。")

    # 退出码
    if len(failed_tables) > 0 or len(missing_files) > 0:
        print(f"\n[load_real_raw] 部分表读取失败，请检查。")
        sys.exit(1)
    else:
        print(f"\n[load_real_raw] 全部读取完成。")


if __name__ == "__main__":
    main()