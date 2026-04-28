"""
sample0427 统一读取层

从 sample0427 目录读取 11 张 CSV 表，生成表级读取摘要和总读取报告。

用法:
    python src/data/load_sample0427.py
    python src/data/load_sample0427.py --config configs/common/data_paths.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# 将项目根目录加入 sys.path，确保 src 可导入
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.io import read_csv_safe
from src.utils.common import (
    detect_id_columns,
    detect_json_like_columns,
    detect_list_like_columns,
    detect_raw_columns,
)

# =============================================================================
# 预期文件列表
# =============================================================================
# 映射关系：table_name -> file_name
EXPECTED_FILES: dict[str, str] = {
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


def build_table_info(
    table_name: str,
    file_path: Path,
    read_success: bool,
    encoding_used: str | None,
    df: pd.DataFrame | None,
    error_message: str | None,
) -> dict:
    """构建单张表的读取摘要。"""
    info: dict = {
        "table_name": table_name,
        "file_name": file_path.name,
        "file_path": str(file_path),
        "read_success": read_success,
        "encoding_used": encoding_used,
        "row_count": None,
        "column_count": None,
        "columns": [],
        "dtypes": {},
        "missing_cells_total": None,
        "missing_rate_overall": None,
        "duplicate_rows": None,
        "likely_id_columns": [],
        "list_like_columns": [],
        "json_like_columns": [],
        "raw_columns": [],
        "error_message": error_message,
    }
    if df is not None:
        info["row_count"] = len(df)
        info["column_count"] = len(df.columns)
        info["columns"] = list(df.columns)
        info["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        info["missing_cells_total"] = int(missing_cells)
        info["missing_rate_overall"] = round(
            float(missing_cells / total_cells), 6
        ) if total_cells > 0 else 0.0
        info["duplicate_rows"] = int(df.duplicated().sum())
        info["likely_id_columns"] = detect_id_columns(df)
        info["list_like_columns"] = detect_list_like_columns(df)
        info["json_like_columns"] = detect_json_like_columns(df)
        info["raw_columns"] = detect_raw_columns(df)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="sample0427 统一读取层"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "common" / "data_paths.yaml"),
        help="数据路径配置文件路径",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. 加载配置
    # ------------------------------------------------------------------
    config_path = Path(args.config)
    print(f"[load_sample0427] 加载配置文件: {config_path}")
    cfg = load_config(str(config_path))

    sample0427_dir = Path(cfg["sample0427_dir"])
    outputs_dir = Path(cfg["outputs_dir"])
    data_check_dir = outputs_dir / "data_check"

    print(f"[load_sample0427] sample0427 目录: {sample0427_dir}")
    print(f"[load_sample0427] 输出目录: {data_check_dir}")

    if not sample0427_dir.exists():
        print(f"[load_sample0427] 错误: sample0427 目录不存在: {sample0427_dir}")
        sys.exit(1)

    # 确保输出目录存在
    data_check_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. 逐表读取
    # ------------------------------------------------------------------
    tables_info: list[dict] = []
    loaded_tables: list[str] = []
    failed_tables: list[str] = []
    missing_files: list[str] = []
    total_rows = 0
    total_columns = 0

    print(f"\n[load_sample0427] 开始读取 {len(EXPECTED_FILES)} 张表...\n")

    for table_name, file_name in EXPECTED_FILES.items():
        file_path = sample0427_dir / file_name
        print(f"  [{table_name}] 读取 {file_name} ...", end=" ")

        if not file_path.exists():
            print("文件不存在")
            missing_files.append(file_name)
            info = build_table_info(
                table_name=table_name,
                file_path=file_path,
                read_success=False,
                encoding_used=None,
                df=None,
                error_message="文件不存在",
            )
            tables_info.append(info)
            continue

        try:
            df, encoding_used = read_csv_safe(str(file_path))
            info = build_table_info(
                table_name=table_name,
                file_path=file_path,
                read_success=True,
                encoding_used=encoding_used,
                df=df,
                error_message=None,
            )
            loaded_tables.append(table_name)
            total_rows += len(df)
            total_columns += len(df.columns)
            print(
                f"OK | {len(df)} 行 x {len(df.columns)} 列 "
                f"| 编码: {encoding_used}"
            )
        except Exception as e:
            print(f"失败: {e}")
            failed_tables.append(table_name)
            info = build_table_info(
                table_name=table_name,
                file_path=file_path,
                read_success=False,
                encoding_used=None,
                df=None,
                error_message=str(e),
            )

        tables_info.append(info)

    # ------------------------------------------------------------------
    # 3. 输出表级摘要 CSV
    # ------------------------------------------------------------------
    summary_csv_path = data_check_dir / "sample0427_table_summary.csv"
    summary_rows = []
    for info in tables_info:
        summary_rows.append({
            "table_name": info["table_name"],
            "file_name": info["file_name"],
            "read_success": info["read_success"],
            "encoding_used": info["encoding_used"],
            "row_count": info["row_count"],
            "column_count": info["column_count"],
            "missing_cells_total": info["missing_cells_total"],
            "missing_rate_overall": info["missing_rate_overall"],
            "duplicate_rows": info["duplicate_rows"],
            "likely_id_columns": (
                "; ".join(info["likely_id_columns"])
                if info["likely_id_columns"]
                else ""
            ),
            "list_like_columns": (
                "; ".join(info["list_like_columns"])
                if info["list_like_columns"]
                else ""
            ),
            "json_like_columns": (
                "; ".join(info["json_like_columns"])
                if info["json_like_columns"]
                else ""
            ),
            "raw_columns": (
                "; ".join(info["raw_columns"]) if info["raw_columns"] else ""
            ),
            "error_message": info["error_message"] or "",
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[load_sample0427] 表级摘要已保存: {summary_csv_path}")

    # ------------------------------------------------------------------
    # 4. 输出总读取报告 JSON
    # ------------------------------------------------------------------
    report = {
        "input_dir": str(sample0427_dir),
        "expected_tables": list(EXPECTED_FILES.keys()),
        "expected_file_count": len(EXPECTED_FILES),
        "loaded_tables": loaded_tables,
        "loaded_count": len(loaded_tables),
        "missing_files": missing_files,
        "missing_count": len(missing_files),
        "failed_tables": failed_tables,
        "failed_count": len(failed_tables),
        "total_rows": int(total_rows),
        "total_columns": int(total_columns),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": [
            "当前结果仅反映 sample0427 样本数据的读取状态。",
            "list_like_columns 和 json_like_columns 基于采样检测，"
            "非严格类型判定。",
            "所有原始字段保留不变，未做类型转换或覆盖。",
            "未做 schema 严格校验，仅做读取层可用性验证。",
        ],
    }
    report_json_path = data_check_dir / "sample0427_load_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[load_sample0427] 总读取报告已保存: {report_json_path}")

    # ------------------------------------------------------------------
    # 5. 打印汇总
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("读取汇总")
    print("=" * 60)
    print(f"  期望表数:      {report['expected_file_count']}")
    print(f"  成功读取:      {report['loaded_count']}")
    print(f"  缺失文件:      {report['missing_count']}")
    print(f"  读取失败:      {report['failed_count']}")
    print(f"  总行数:        {report['total_rows']}")
    print(f"  总列数:        {report['total_columns']}")
    if missing_files:
        print(f"  缺失文件列表:  {', '.join(missing_files)}")
    if failed_tables:
        print(f"  失败表列表:    {', '.join(failed_tables)}")
    print("=" * 60)

    # 如果所有表都成功读取，exit code 为 0
    if report["loaded_count"] == report["expected_file_count"]:
        print("[load_sample0427] 所有表读取成功。")
    else:
        print(
            f"[load_sample0427] 部分表读取失败 "
            f"({report['failed_count']} 失败, "
            f"{report['missing_count']} 缺失)。"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()