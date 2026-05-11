"""
real_raw_5000 Tabular 数据集构建脚本 (no_interaction_leakage)

基于 real_raw_5000 的 11 张 raw 表构建 DNN / Wide & Deep 可用的 tabular 输入，
使用 train / val / test 三路切分。

核心设计:
  1. 标签用 interaction_score = digg+comment+share+collect 分位数。
  2. 标签构造完成后，四个 label source 字段必须从特征中删除。
  3. 潜在泄漏字段（max_comment_digg_count 等）不进入模型特征。
  4. 泄漏检查在输出前强制执行，发现问题报错中止。
  5. 质量过滤 mode=full（5000 样本）或 high_confidence（3493 样本）。

用法:
    D:/CodeData/software/Anaconda/Anaconda3/envs/ra/python.exe src/data/build_tabular_real_raw_5000.py
        --config configs/common/feature_tabular_real_raw_5000.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.io import read_csv_safe
from src.features.tabular_features import (
    build_aggregated_features,
    build_duration_bucket,
    build_text_stat_features,
    compute_missing_summary,
)
from src.features.cross_features import build_cross_features


# =============================================================================
# 辅助函数
# =============================================================================

def load_all_tables(root_dir: Path, tables_cfg: dict[str, str]) -> dict[str, pd.DataFrame]:
    """读取所有 CSV 表并返回 {table_key: df} 字典。"""
    tables: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    for key, file_name in tables_cfg.items():
        file_path = root_dir / file_name
        try:
            df, encoding = read_csv_safe(str(file_path))
            tables[key] = df
            print(f"  [{key}] {file_name} -> {len(df)} 行 x {len(df.columns)} 列 | {encoding}")
        except Exception as e:
            print(f"  [{key}] 读取失败: {e}")
            failed.append(key)
    if failed:
        print(f"\n  [W] 以下表读取失败: {failed}")
    return tables


def analyze_field_quality(df: pd.DataFrame, label: str) -> dict[str, Any]:
    """分析 DataFrame 每列的质量：非空率、unique 数、是否常量。"""
    info = {}
    for col in df.columns:
        non_null = int(df[col].notna().sum())
        total = len(df)
        rate = round(non_null / total, 4) if total > 0 else 0.0
        nunique = int(df[col].nunique(dropna=False))
        is_constant = nunique <= 1
        info[col] = {
            "non_null": non_null,
            "total": total,
            "non_null_rate": rate,
            "nunique": nunique,
            "is_constant": is_constant,
        }
    return info


def drop_low_quality_fields(
    df: pd.DataFrame,
    quality: dict[str, Any],
    table_name: str,
    min_non_null_rate: float = 0.01,
    drop_constant: bool = True,
    always_keep: list[str] | None = None,
) -> pd.DataFrame:
    """根据字段质量信息排除低质量列。"""
    always_keep = always_keep or []
    to_drop = []
    reasons = {}

    for col, info in quality.items():
        if col in always_keep:
            continue
        if drop_constant and info["is_constant"]:
            to_drop.append(col)
            reasons[col] = "constant"
        elif info["non_null_rate"] < min_non_null_rate:
            to_drop.append(col)
            reasons[col] = f"low_coverage ({info['non_null_rate']:.1%})"

    if to_drop:
        df = df.drop(columns=[c for c in to_drop if c in df.columns])
        print(f"    {table_name}: 排除 {len(to_drop)} 个低质量字段")
        for col in to_drop:
            print(f"      - {col}: {reasons[col]}")
    else:
        print(f"    {table_name}: 无低质量字段需排除")
    return df


def deduplicate_table(
    df: pd.DataFrame, key: str, table_label: str
) -> pd.DataFrame:
    """按 key 去重，保留第一个出现的行。"""
    before = len(df)
    df = df.drop_duplicates(subset=[key], keep="first")
    after = len(df)
    dup_count = before - after
    if dup_count > 0:
        print(f"    {table_label}: {dup_count} 行重复 {key}，去重后 {after} 行")
    return df


def build_label(
    df: pd.DataFrame,
    components: list[str],
    default_quantile: float = 0.60,
    fallback_quantile: float = 0.50,
    imbalance_threshold: float = 0.20,
    label_col: str = "label",
    score_col: str = "interaction_score",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """构造交互伪标签。

    策略:
        1. interaction_score = sum(components)
        2. 以 default_quantile 分位数为阈值构造二分类
        3. 若 minority class < imbalance_threshold，回退到 fallback_quantile
    """
    df = df.copy()
    valid_components = [c for c in components if c in df.columns]
    missing = [c for c in components if c not in df.columns]
    if missing:
        print(f"  [W] 标签组件列缺失: {missing}")

    # 缺失互动分量的样本 score=0
    df[score_col] = df[valid_components].sum(axis=1).fillna(0).astype(float)

    threshold = df[score_col].quantile(default_quantile)
    df[label_col] = (df[score_col] >= threshold).astype(int)

    pos_ratio = df[label_col].mean()
    neg_ratio = 1 - pos_ratio
    minority = min(pos_ratio, neg_ratio)

    used_quantile = default_quantile
    if minority < imbalance_threshold:
        print(
            f"  [W] 标签分布不均衡 (minority={minority:.3f} < {imbalance_threshold}), "
            f"回退到 {fallback_quantile} 分位数"
        )
        threshold = df[score_col].quantile(fallback_quantile)
        df[label_col] = (df[score_col] >= threshold).astype(int)
        used_quantile = fallback_quantile

    label_info = {
        "label_col": label_col,
        "score_col": score_col,
        "components": components,
        "valid_components": valid_components,
        "missing_components": missing,
        "method": "interaction_binary",
        "quantile_used": used_quantile,
        "threshold": float(threshold),
        "pos_count": int(df[label_col].sum()),
        "neg_count": int((df[label_col] == 0).sum()),
        "pos_ratio": round(float(df[label_col].mean()), 4),
        "neg_ratio": round(float(1 - df[label_col].mean()), 4),
    }
    print(f"  标签: {label_col}, threshold={threshold:.2f}, "
          f"pos={label_info['pos_count']}/{label_info['neg_count']} "
          f"({label_info['pos_ratio']:.1%})")
    return df, label_info


def train_val_test_split(
    df: pd.DataFrame,
    seed: int = 2026,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    label_col: str = "label",
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """按 video_id 维度三路切分 train / val / test。

    支持按 label 分层抽样。
    """
    df = df.copy()
    rng = np.random.RandomState(seed)

    if stratify and label_col in df.columns:
        # 分层抽样：按 label 分组，各自比例抽取
        pos_df = df[df[label_col] == 1]
        neg_df = df[df[label_col] == 0]

        def _split_group(group_df: pd.DataFrame, label_name: str) -> tuple[set, set, set]:
            """对一个 label 组内的 video_id 做三路切分。"""
            vids = group_df["video_id"].unique()
            rng.shuffle(vids)
            n = len(vids)
            n_train = max(1, int(n * train_ratio))
            n_val = max(0, int(n * val_ratio))
            train_ids = set(vids[:n_train])
            val_ids = set(vids[n_train:n_train + n_val])
            test_ids = set(vids[n_train + n_val:])
            return train_ids, val_ids, test_ids

        pos_train, pos_val, pos_test = _split_group(pos_df, "pos")
        neg_train, neg_val, neg_test = _split_group(neg_df, "neg")

        train_vids = pos_train | neg_train
        val_vids = pos_val | neg_val
        test_vids = pos_test | neg_test
        method_detail = f"stratified_by_{label_col}"
    else:
        vids = df["video_id"].unique()
        rng.shuffle(vids)
        n = len(vids)
        n_train = max(1, int(n * train_ratio))
        n_val = max(0, int(n * val_ratio))
        train_vids = set(vids[:n_train])
        val_vids = set(vids[n_train:n_train + n_val])
        test_vids = set(vids[n_train + n_val:])
        method_detail = "random"

    df["split"] = df["video_id"].apply(
        lambda x: "train" if x in train_vids else ("val" if x in val_vids else "test")
    )

    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    def _dist(d: pd.DataFrame, name: str) -> dict:
        pos = int(d[label_col].sum())
        neg = len(d) - pos
        return {
            f"{name}_size": len(d),
            f"{name}_pos": pos,
            f"{name}_neg": neg,
            f"{name}_pos_ratio": round(pos / len(d), 4) if len(d) > 0 else 0,
        }

    split_info = {
        "method": method_detail,
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "total_size": len(df),
        **_dist(df_train, "train"),
        **_dist(df_val, "val"),
        **_dist(df_test, "test"),
    }

    print(f"  切分: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
    print(f"  train label: pos={int(df_train[label_col].sum())}, "
          f"neg={len(df_train)-int(df_train[label_col].sum())}")
    print(f"  val   label: pos={int(df_val[label_col].sum())}, "
          f"neg={len(df_val)-int(df_val[label_col].sum())}")
    print(f"  test  label: pos={int(df_test[label_col].sum())}, "
          f"neg={len(df_test)-int(df_test[label_col].sum())}")

    return df, df_train, df_val, df_test, split_info


def check_split_exclusivity(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """检查各 split 间 video_id 是否互斥。"""
    splits = df["split"].unique()
    result = {"exclusive": True, "details": {}}
    split_sets = {}
    for sp in splits:
        vids = set(df[df["split"] == sp]["video_id"].unique())
        split_sets[sp] = vids

    for sp1 in splits:
        for sp2 in splits:
            if sp1 >= sp2:
                continue
            overlap = split_sets[sp1] & split_sets[sp2]
            if overlap:
                result["exclusive"] = False
                result["details"][f"{sp1}_x_{sp2}"] = {
                    "overlap_count": len(overlap),
                    "overlap_ids": sorted(overlap)[:10],
                }
    return result


def validate_leakage(
    df: pd.DataFrame,
    feature_info_features: list[str],
    excluded_label_source_features: list[str],
    excluded_potential_leakage_cols: list[str],
    audit_id_cols: list[str],
    label_col: str,
    score_col: str,
) -> dict[str, Any]:
    """验证最终数据集无泄漏。返回检查报告；发现问题则报错中止。"""
    errors: list[str] = []
    warnings: list[str] = []

    # 检查四个 label source 字段是否仍在 CSV 列中
    leaked_in_csv = [c for c in excluded_label_source_features if c in df.columns]
    if leaked_in_csv:
        errors.append(f"四个 label source 字段仍存在于最终 CSV 中: {leaked_in_csv}")

    # 检查四个 label source 字段是否在 feature_info 特征列表中
    leaked_in_features = [c for c in excluded_label_source_features if c in feature_info_features]
    if leaked_in_features:
        errors.append(f"四个 label source 字段仍存在于 feature_info 特征列表中: {leaked_in_features}")

    # 检查 interaction_score 不在模型特征中
    id_and_label = set(audit_id_cols + [label_col])
    if score_col in feature_info_features:
        errors.append(f"interaction_score 出现在模型特征列表中，应仅用于审计")

    # 检查 audit_id_cols 不在模型特征中
    for col in audit_id_cols:
        if col in feature_info_features:
            errors.append(f"审计列 {col} 出现在模型特征列表中，应仅用于标识")

    # 检查潜在泄漏字段是否出现在特征列表中
    leaked_potential = [c for c in excluded_potential_leakage_cols if c in feature_info_features]
    if leaked_potential:
        warnings.append(f"潜在泄漏字段出现在特征列表中: {leaked_potential}")

    report = {
        "leakage_check_passed": len(errors) == 0,
        "excluded_label_source_features": {
            "defined": excluded_label_source_features,
            "found_in_csv": leaked_in_csv,
            "found_in_feature_info": leaked_in_features,
            "status": "ERROR" if leaked_in_csv or leaked_in_features else "OK",
        },
        "interaction_score_in_features": {
            "status": "ERROR" if score_col in feature_info_features else "OK",
        },
        "audit_id_cols_in_features": {
            "defined": audit_id_cols,
            "found_in_feature_info": [c for c in audit_id_cols if c in feature_info_features],
        },
        "excluded_potential_leakage_cols": {
            "defined": excluded_potential_leakage_cols,
            "found_in_feature_info": leaked_potential,
            "status": "WARNING" if leaked_potential else "OK",
        },
        "errors": errors,
        "warnings": warnings,
    }

    if errors:
        error_msg = "\n".join(errors)
        print(f"\n[leakage_check] 泄漏检查失败:\n{error_msg}")
        sys.exit(1)

    print(f"\n[leakage_check] 泄漏检查通过: 未发现泄漏字段")
    if warnings:
        for w in warnings:
            print(f"  [W] {w}")
    return report


# =============================================================================
# 主流程
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="real_raw_5000 Tabular 数据集构建 (no_interaction_leakage)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "common" / "feature_tabular_real_raw_5000.yaml"),
        help="特征构建配置文件路径",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("real_raw_5000 Tabular 数据集构建 (no_interaction_leakage)")
    print("=" * 60)

    # =========================================================================
    # 1. 加载配置
    # =========================================================================
    cfg_path = Path(args.config)
    print(f"\n[1/11] 加载配置...")
    cfg = load_config(str(cfg_path))

    dataset_name = cfg["dataset_name"]
    run_id = cfg["run_id"]
    data_root = PROJECT_ROOT / cfg["data_root"]
    tables_cfg: dict[str, str] = cfg["tables"]
    allowed_empty: list[str] = cfg.get("allowed_empty_tables", [])
    agg_features_cfg: dict[str, list[dict]] = cfg.get("aggregation_features", {})
    label_cfg: dict = cfg.get("label_config", {})
    split_cfg: dict = cfg.get("split", {})
    quality_cfg: dict = cfg.get("field_quality", {})
    leak_cfg: dict = cfg.get("leakage_control", {})
    min_non_null_rate = quality_cfg.get("min_non_null_rate", 0.01)
    drop_constant = quality_cfg.get("drop_constant", True)
    fill_na_num = quality_cfg.get("fill_na_numeric", 0)
    fill_na_cat = quality_cfg.get("fill_na_categorical", "__MISSING__")

    quality_filter_cfg = cfg.get("quality_filter", {})
    qf_mode = quality_filter_cfg.get("mode", "full")
    high_conf_path_rel = quality_filter_cfg.get("high_confidence_video_ids", "")

    # 泄漏控制参数
    excluded_label_source = leak_cfg.get("excluded_label_source_features", [])
    excluded_potential_leakage = leak_cfg.get("excluded_potential_leakage_cols", [])
    audit_id_cols = leak_cfg.get("audit_id_cols", [])
    enforce_strict = leak_cfg.get("enforce_strict_deletion", True)

    output_paths_cfg = cfg.get("output_paths", {})
    features_dir = PROJECT_ROOT / output_paths_cfg.get("features_dir", "data/features/real_raw_5000")
    check_dir = PROJECT_ROOT / output_paths_cfg.get("check_dir", "outputs/data_check/real_raw_5000")

    features_dir.mkdir(parents=True, exist_ok=True)
    check_dir.mkdir(parents=True, exist_ok=True)

    print(f"  数据集: {dataset_name}")
    print(f"  run_id: {run_id}")
    print(f"  数据根目录: {data_root}")
    print(f"  质量过滤模式: {qf_mode}")
    print(f"  特征输出: {features_dir}")
    print(f"  报告输出: {check_dir}")

    # =========================================================================
    # 2. 加载所有表
    # =========================================================================
    print(f"\n[2/11] 加载所有表 ({len(tables_cfg)} 张)...")
    tables = load_all_tables(data_root, tables_cfg)

    # =========================================================================
    # 3. 质量过滤（可选：high_confidence 模式）
    # =========================================================================
    print(f"\n[3/11] 质量过滤 (mode={qf_mode})...")

    # 先应用质量过滤到 raw_video_detail 以确定主表样本集
    vd_df = tables["video_detail"].copy()
    original_count = len(vd_df)

    if qf_mode == "full":
        print(f"  full 模式: 使用全部 {len(vd_df)} 条主视频样本")
        valid_video_ids = set(vd_df["video_id"].unique())
    elif qf_mode == "high_confidence":
        hc_path = PROJECT_ROOT / high_conf_path_rel
        if not hc_path.exists():
            print(f"  [ERROR] high_confidence_video_ids 文件不存在: {hc_path}")
            print(f"  回退到 full 模式")
            valid_video_ids = set(vd_df["video_id"].unique())
        else:
            with open(hc_path, "r") as f:
                hc_ids = {line.strip() for line in f if line.strip()}
            valid_video_ids = hc_ids & set(vd_df["video_id"].unique())
            print(f"  high_confidence 模式: {len(valid_video_ids)} 条样本 "
                  f"(过滤掉 {len(vd_df) - len(valid_video_ids)} 条 none-match 样本)")
    else:
        print(f"  [W] 未知质量过滤模式 '{qf_mode}'，使用 full")
        valid_video_ids = set(vd_df["video_id"].unique())

    # 对各表按 valid_video_ids 过滤
    for key in list(tables.keys()):
        df = tables[key]
        if "video_id" in df.columns:
            before = len(df)
            df = df[df["video_id"].isin(valid_video_ids)].copy()
            after = len(df)
            if before != after:
                print(f"  {key}: {before} -> {after} 行 (过滤 {before-after} 行)")
            tables[key] = df

    vd_df = tables["video_detail"].copy()
    final_count = len(vd_df)
    print(f"  主表最终样本数: {final_count}")

    # =========================================================================
    # 4. 构建主表 + 字段质量分析 + 低质量字段排除
    # =========================================================================
    print(f"\n[4/11] 构建主表并排除低质量字段...")

    main_df = vd_df.copy()
    print(f"  raw_video_detail 原始列数: {len(main_df.columns)}")

    # 创建 sample_id
    main_df["sample_id"] = main_df["video_id"].astype(str)

    # 删除明显非特征列
    non_feature_cols = ["page_url", "crawl_time", "sec_item_id", "group_id",
                        "comment_gid", "primary_source_key"]
    existing_non_feature = [c for c in non_feature_cols if c in main_df.columns]
    if existing_non_feature:
        main_df = main_df.drop(columns=existing_non_feature)
        print(f"    - 排除非特征列: {existing_non_feature}")

    # 字段质量分析
    vd_quality = analyze_field_quality(
        main_df, "raw_video_detail",
    )
    main_df = drop_low_quality_fields(
        main_df, vd_quality, "raw_video_detail",
        min_non_null_rate=min_non_null_rate,
        drop_constant=drop_constant,
        always_keep=["video_id", "author_id", "sample_id"],
    )

    # =========================================================================
    # 5. 多表 Join / 聚合
    # =========================================================================
    print(f"\n[5/11] 多表 Join / 聚合...")

    field_quality_report: dict[str, Any] = {}
    excluded_fields: dict[str, list[str]] = {}

    # 5a. raw_author — 先去重，再 LEFT JOIN
    print("  raw_author: 去重后 LEFT JOIN on author_id")
    author_df = tables["author"].copy()
    for c in ["crawl_time", "sec_uid", "unique_id", "short_id", "avatar_thumb_url_list"]:
        if c in author_df.columns:
            excluded_fields.setdefault("raw_author", []).append(c)
            author_df = author_df.drop(columns=[c])
    author_df = deduplicate_table(author_df, "author_id", "raw_author")

    au_quality = analyze_field_quality(author_df, "raw_author")
    field_quality_report["raw_author"] = au_quality
    author_df = drop_low_quality_fields(
        author_df, au_quality, "raw_author",
        min_non_null_rate=min_non_null_rate,
        drop_constant=drop_constant,
    )
    main_df = main_df.merge(author_df, on="author_id", how="left", suffixes=("", "_author_dup"))
    dup_cols = [c for c in main_df.columns if c.endswith("_author_dup")]
    if dup_cols:
        main_df = main_df.drop(columns=dup_cols)
    print(f"    merge 后主表: {len(main_df)} 行")

    # 5b. raw_music — 直接 LEFT JOIN
    print("  raw_music: LEFT JOIN on video_id")
    music_df = tables["music"].copy()
    for c in ["crawl_time", "music_mid", "music_id", "music_owner_id"]:
        if c in music_df.columns:
            excluded_fields.setdefault("raw_music", []).append(c)
            music_df = music_df.drop(columns=[c])
    mu_quality = analyze_field_quality(music_df, "raw_music")
    field_quality_report["raw_music"] = mu_quality
    music_df = drop_low_quality_fields(
        music_df, mu_quality, "raw_music",
        min_non_null_rate=min_non_null_rate,
        drop_constant=drop_constant,
    )
    main_df = main_df.merge(music_df, on="video_id", how="left", suffixes=("", "_music_dup"))
    dup_cols = [c for c in main_df.columns if c.endswith("_music_dup")]
    if dup_cols:
        main_df = main_df.drop(columns=dup_cols)

    # 5c. raw_video_media — 直接 LEFT JOIN
    print("  raw_video_media: LEFT JOIN on video_id")
    media_df = tables["video_media"].copy()
    for c in ["crawl_time", "cover_uri", "origin_cover_uri", "dynamic_cover_uri",
              "video_format", "video_ratio", "bit_rate_raw", "big_thumbs_raw", "video_meta_raw"]:
        if c in media_df.columns:
            excluded_fields.setdefault("raw_video_media", []).append(c)
            media_df = media_df.drop(columns=[c])
    md_quality = analyze_field_quality(media_df, "raw_video_media")
    field_quality_report["raw_video_media"] = md_quality
    media_df = drop_low_quality_fields(
        media_df, md_quality, "raw_video_media",
        min_non_null_rate=min_non_null_rate,
        drop_constant=drop_constant,
    )
    main_df = main_df.merge(media_df, on="video_id", how="left", suffixes=("", "_media_dup"))
    dup_cols = [c for c in main_df.columns if c.endswith("_media_dup")]
    if dup_cols:
        main_df = main_df.drop(columns=dup_cols)

    # 5d. raw_video_status_control — 直接 LEFT JOIN
    print("  raw_video_status_control: LEFT JOIN on video_id")
    sc_df = tables["video_status_control"].copy()
    sc_df = sc_df.drop(columns=[c for c in ["crawl_time"] if c in sc_df.columns])
    sc_quality = analyze_field_quality(sc_df, "raw_video_status_control")
    field_quality_report["raw_video_status_control"] = sc_quality
    sc_df = drop_low_quality_fields(
        sc_df, sc_quality, "raw_video_status_control",
        min_non_null_rate=min_non_null_rate,
        drop_constant=drop_constant,
    )
    if len(sc_df.columns) > 1:
        main_df = main_df.merge(sc_df, on="video_id", how="left", suffixes=("", "_sc_dup"))
        dup_cols = [c for c in main_df.columns if c.endswith("_sc_dup")]
        if dup_cols:
            main_df = main_df.drop(columns=dup_cols)
    else:
        print("    raw_video_status_control: 无有效特征，跳过 merge")

    # 5e. raw_hashtag — 聚合到 video_id
    print("  raw_hashtag: 聚合到 video_id")
    hashtag_df = tables["hashtag"].copy()
    hashtag_agg = build_aggregated_features(
        hashtag_df, "video_id",
        agg_features_cfg.get("raw_hashtag", []),
    )
    top_hashtag = (
        hashtag_df.groupby("video_id")["hashtag_name"]
        .first()
        .reset_index(name="hashtag_name_top")
    )
    hashtag_agg = hashtag_agg.merge(top_hashtag, on="video_id", how="left")
    print(f"    聚合特征: {list(hashtag_agg.columns)}")
    main_df = main_df.merge(hashtag_agg, on="video_id", how="left")

    # 5f. raw_video_tag — 空表，记录跳过
    print("  raw_video_tag: 聚合到 video_id")
    tag_df = tables.get("video_tag")
    if tag_df is not None and len(tag_df) > 0:
        tag_agg = build_aggregated_features(
            tag_df, "video_id",
            agg_features_cfg.get("raw_video_tag", []),
        )
        print(f"    聚合特征: {list(tag_agg.columns)}")
        main_df = main_df.merge(tag_agg, on="video_id", how="left")
    else:
        print("    raw_video_tag 为空表，跳过聚合")
        main_df["video_tag_count"] = 0

    # 5g. raw_comment — 聚合到 video_id
    print("  raw_comment: 聚合到 video_id")
    comment_df = tables["comment"].copy()
    # 排除低质量列
    comment_df = drop_low_quality_fields(
        comment_df,
        analyze_field_quality(comment_df, "raw_comment"),
        "raw_comment",
        min_non_null_rate=min_non_null_rate,
        drop_constant=drop_constant,
        always_keep=["video_id", "comment_text", "comment_user_id"],
    )
    comment_agg = build_aggregated_features(
        comment_df, "video_id",
        agg_features_cfg.get("raw_comment", []),
    )
    print(f"    聚合特征: {list(comment_agg.columns)}")
    main_df = main_df.merge(comment_agg, on="video_id", how="left")

    # 5h. raw_chapter — 空表，记录跳过
    print("  raw_chapter: 聚合到 video_id")
    chapter_df = tables.get("chapter")
    if chapter_df is not None and len(chapter_df) > 0:
        chapter_agg = build_aggregated_features(
            chapter_df, "video_id",
            agg_features_cfg.get("raw_chapter", []),
        )
        print(f"    聚合特征: {list(chapter_agg.columns)}")
        main_df = main_df.merge(chapter_agg, on="video_id", how="left")
    else:
        print("    raw_chapter 为空表，跳过聚合")
        main_df["chapter_count"] = 0

    # 5i. raw_related_video — 聚合到 source_video_id
    print("  raw_related_video: 聚合到 source_video_id")
    related_df = tables["related_video"].copy()
    related_agg = build_aggregated_features(
        related_df, "source_video_id",
        agg_features_cfg.get("raw_related_video", []),
    )
    related_agg = related_agg.rename(columns={"source_video_id": "video_id"})
    print(f"    聚合特征: {list(related_agg.columns)}")
    main_df = main_df.merge(related_agg, on="video_id", how="left")

    print(f"\n  合并后主表: {len(main_df)} 行 x {len(main_df.columns)} 列")

    # =========================================================================
    # 6. 特征工程
    # =========================================================================
    print(f"\n[6/11] 特征工程...")

    # 6a. 文本统计特征
    print("  文本统计特征:")
    if "desc" in main_df.columns:
        main_df = build_text_stat_features(main_df, "desc")
        print("    desc -> desc_length, desc_word_count")
    if "caption" in main_df.columns:
        main_df = build_text_stat_features(main_df, "caption")
        print("    caption -> caption_length")
    if "signature" in main_df.columns:
        main_df = build_text_stat_features(main_df, "signature")
        print("    signature -> signature_length")
    if "nickname" in main_df.columns:
        main_df = build_text_stat_features(main_df, "nickname")
        print("    nickname -> nickname_length")
    if "music_title" in main_df.columns:
        main_df = build_text_stat_features(main_df, "music_title")
        print("    music_title -> music_title_length")

    # 6b. 媒体 URL 存在性特征
    print("  媒体 URL 存在性特征:")
    for col in ["cover_url_list", "origin_cover_url_list", "dynamic_cover_url_list"]:
        if col in main_df.columns:
            count_col = col.replace("_list", "_count")
            main_df[count_col] = main_df[col].apply(
                lambda x: len(str(x).split("|")) if pd.notna(x) and str(x).strip() else 0
            )
            print(f"    {col} -> {count_col}")

    # 6c. Duration 桶化
    if "duration_ms" in main_df.columns:
        main_df = build_duration_bucket(main_df)
        print("  duration_ms -> duration_bucket (short/medium/long/very_long)")

    # 6d. 时间戳特征
    if "create_time" in main_df.columns:
        valid = main_df["create_time"].notna()
        ts = pd.to_datetime(main_df.loc[valid, "create_time"], unit="s", errors="coerce")
        main_df["publish_hour"] = np.nan
        main_df["publish_weekday"] = np.nan
        main_df.loc[valid, "publish_hour"] = ts.dt.hour.astype(float)
        main_df.loc[valid, "publish_weekday"] = ts.dt.dayofweek.astype(float)
        print("  create_time -> publish_hour, publish_weekday")

    # 6e. 缺失值填充
    print("  缺失值填充:")
    numeric_cols_in_df = main_df.select_dtypes(include=[np.number]).columns.tolist()
    id_exceptions = set(audit_id_cols + ["label"])
    numeric_to_fill = [c for c in numeric_cols_in_df if c not in id_exceptions]
    before_fill = main_df[numeric_to_fill].isna().sum().sum()
    main_df[numeric_to_fill] = main_df[numeric_to_fill].fillna(fill_na_num)
    after_fill = main_df[numeric_to_fill].isna().sum().sum()
    filled = before_fill - after_fill
    if filled > 0:
        print(f"    {int(filled)} 个数值 NaN 已填充为 {fill_na_num}")

    cat_cols_in_df = main_df.select_dtypes(include=["object"]).columns.tolist()
    id_exceptions_cat = set(audit_id_cols + ["split", "label"])
    cat_to_fill = [c for c in cat_cols_in_df if c not in id_exceptions_cat]
    before_fill_cat = main_df[cat_to_fill].isna().sum().sum()
    main_df[cat_to_fill] = main_df[cat_to_fill].fillna(fill_na_cat)
    after_fill_cat = main_df[cat_to_fill].isna().sum().sum()
    filled_cat = before_fill_cat - after_fill_cat
    if filled_cat > 0:
        print(f"    {int(filled_cat)} 个类别 NaN 已填充为 {fill_na_cat}")

    # 6f. 交叉特征
    print("  交叉特征:")
    cross_configs = cfg.get("wide_cross_features", [])
    cross_features_built = []
    for cc in cross_configs:
        left = cc["left"]
        right = cc["right"]
        name = cc.get("name", f"{left}_x_{right}")
        if left in main_df.columns and right in main_df.columns:
            main_df = build_cross_features(main_df, [cc])
            cross_features_built.append(name)
            print(f"    {name}: {left} x {right}")
        else:
            missing = [x for x in [left, right] if x not in main_df.columns]
            print(f"    [W] 跳过 {name}: 缺少 {missing}")

    # =========================================================================
    # 7. 标签构造
    # =========================================================================
    print(f"\n[7/11] 标签构造...")
    components = label_cfg.get("interaction_components", [])
    main_df, label_info = build_label(
        main_df,
        components=components,
        default_quantile=label_cfg.get("default_quantile", 0.60),
        fallback_quantile=label_cfg.get("fallback_quantile", 0.50),
        imbalance_threshold=label_cfg.get("imbalance_threshold", 0.20),
        label_col=label_cfg.get("label_col", "label"),
        score_col=label_cfg.get("score_col", "interaction_score"),
    )

    # =========================================================================
    # 8. 泄漏控制：删除 label_source 字段和潜在泄漏字段
    # =========================================================================
    print(f"\n[8/11] 泄漏控制: 删除标签源字段和潜在泄漏字段...")

    # 记录删除前的字段
    before_drop_cols = set(main_df.columns)

    # 删除四个 label source 字段
    label_source_to_drop = [c for c in excluded_label_source if c in main_df.columns]
    if label_source_to_drop:
        main_df = main_df.drop(columns=label_source_to_drop)
        print(f"  删除 label source 字段: {label_source_to_drop}")

    # 删除潜在泄漏字段
    potential_leakage_to_drop = [c for c in excluded_potential_leakage if c in main_df.columns]
    if potential_leakage_to_drop:
        main_df = main_df.drop(columns=potential_leakage_to_drop)
        print(f"  删除潜在泄漏字段: {potential_leakage_to_drop}")

    # 确保 interaction_score 不在模型中，但保留在 DataFrame 中作为审计列
    # 它将在列序整理时被排除在特征列表外

    after_drop_cols = set(main_df.columns)
    removed_cols = before_drop_cols - after_drop_cols
    print(f"  泄漏控制共删除 {len(removed_cols)} 个字段")

    # =========================================================================
    # 9. Train / Val / Test 切分
    # =========================================================================
    print(f"\n[9/11] Train/Val/Test 切分...")
    stratify = split_cfg.get("stratify", True)
    main_df, df_train, df_val, df_test, split_info = train_val_test_split(
        main_df,
        seed=split_cfg.get("seed", 2026),
        train_ratio=split_cfg.get("train_ratio", 0.70),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        test_ratio=split_cfg.get("test_ratio", 0.15),
        label_col=label_cfg.get("label_col", "label"),
        stratify=stratify,
    )

    # 互斥检查
    exclusivity = check_split_exclusivity(main_df)
    if exclusivity["exclusive"]:
        print("  video_id 互斥检查: [OK] 各 split 间无重叠")
    else:
        print(f"  [W] video_id 重叠: {exclusivity['details']}")

    # =========================================================================
    # 10. 整理列序
    # =========================================================================
    print(f"\n[10/11] 整理列序...")
    id_cols = audit_id_cols  # ["sample_id", "video_id", "author_id"]
    label_col = label_cfg.get("label_col", "label")
    score_col = label_cfg.get("score_col", "interaction_score")
    split_col = "split"

    # 收集已构建的特征列
    numeric_feats_raw: list[str] = []
    for _table_name, cols in cfg.get("numeric_feature_candidates", {}).items():
        for col in cols:
            if col in main_df.columns and col not in numeric_feats_raw and col not in id_cols:
                numeric_feats_raw.append(col)

    # 动态收集数值类型特征（排除审计列和标签列）
    numeric_feats = []
    for col in numeric_feats_raw:
        if col in main_df.columns and col not in id_cols + [label_col, score_col, split_col]:
            numeric_feats.append(col)

    # 聚合特征（已不包含泄漏字段，因为配置中已排除）
    agg_count_features = [
        "hashtag_count", "video_tag_count", "comment_table_count",
        "chapter_count", "related_video_count",
        "comment_user_count", "cover_url_count", "origin_cover_url_count",
        "dynamic_cover_url_count",
    ]
    agg_count_features = [c for c in agg_count_features if c in main_df.columns]

    # 文本统计特征
    text_stat_cols = [
        c for c in cfg.get("text_stat_features", [])
        if c in main_df.columns
    ]

    # 类别特征
    cat_cols = [
        c for c in cfg.get("categorical_feature_candidates", [])
        if c in main_df.columns
    ]

    # 交叉特征
    wide_cross_cols = cross_features_built

    # merge 或聚合过程中自动新增的其他数值特征
    extra_numeric = [
        "publish_hour", "publish_weekday",
    ]
    extra_numeric = [c for c in extra_numeric if c in main_df.columns]

    # 最终特征集合
    explicit_features = list(dict.fromkeys(
        numeric_feats + extra_numeric + text_stat_cols + cat_cols
        + agg_count_features + wide_cross_cols
    ))
    explicit_features = [c for c in explicit_features if c in main_df.columns]

    column_order = list(dict.fromkeys(
        id_cols
        + [c for c in explicit_features if c not in id_cols]
        + [score_col, label_col, split_col]
    ))
    column_order = [c for c in column_order if c in main_df.columns]

    main_df = main_df[column_order]
    df_train = df_train[column_order]
    df_val = df_val[column_order]
    df_test = df_test[column_order]

    # =========================================================================
    # 11. 泄漏验证 + 输出文件
    # =========================================================================
    print(f"\n[11/11] 泄漏验证 + 输出文件...")

    # 泄漏验证
    leakage_report = validate_leakage(
        df=main_df,
        feature_info_features=explicit_features,
        excluded_label_source_features=excluded_label_source,
        excluded_potential_leakage_cols=excluded_potential_leakage,
        audit_id_cols=audit_id_cols,
        label_col=label_col,
        score_col=score_col,
    )

    # 11a. train.csv / val.csv / test.csv
    train_path = features_dir / "tabular_train.csv"
    val_path = features_dir / "tabular_val.csv"
    test_path = features_dir / "tabular_test.csv"
    df_train.to_csv(train_path, index=False, encoding="utf-8-sig")
    df_val.to_csv(val_path, index=False, encoding="utf-8-sig")
    df_test.to_csv(test_path, index=False, encoding="utf-8-sig")
    print(f"  train.csv: {train_path} ({len(df_train)} 行)")
    print(f"  val.csv:   {val_path} ({len(df_val)} 行)")
    print(f"  test.csv:  {test_path} ({len(df_test)} 行)")

    # 11b. feature_info.json
    actual_numeric = [c for c in explicit_features
                      if c in main_df.columns
                      and main_df[c].dtype in (np.int64, np.float64, np.int32, np.float32)
                      and c not in text_stat_cols
                      and c not in cat_cols
                      and c not in id_cols
                      and c not in agg_count_features]
    actual_cat = [c for c in cat_cols if c in main_df.columns]
    actual_text_stat = [c for c in text_stat_cols if c in main_df.columns]
    actual_agg = [c for c in agg_count_features + extra_numeric
                  if c in main_df.columns
                  and c not in actual_numeric
                  and c not in actual_text_stat]
    actual_wide = [c for c in wide_cross_cols if c in main_df.columns]
    actual_numeric = list(dict.fromkeys(actual_numeric + actual_agg))

    total_features = len(explicit_features)
    missing_summary = compute_missing_summary(main_df)

    label_definition = (
        f"离线实验伪标签: interaction_score ("
        + "+".join(label_info["valid_components"])
        + f") >= {label_info['threshold']:.2f} (P{int(label_info['quantile_used']*100)} 分位数)"
    )

    n_none_match = original_count - final_count
    n_none_ratio = n_none_match / original_count if original_count > 0 else 0

    warnings: list[str] = []
    warnings.append(
        f"none-match 样本 {n_none_match} 条 ({n_none_ratio*100:.1f}%) "
        f"中互动字段覆盖率较低。"
    )
    # play_count 检查
    if "play_count" in main_df.columns:
        pc_zero = (main_df["play_count"] == 0).sum()
        warnings.append(f"play_count: {pc_zero}/{len(main_df)} 条为 0 "
                        f"({pc_zero/len(main_df)*100:.1f}%)")
    else:
        warnings.append("play_count 已被排除（潜在泄漏字段）。")

    if label_info["pos_ratio"] < 0.2 or label_info["neg_ratio"] < 0.2:
        warnings.append(
            f"标签分布不均衡: pos={label_info['pos_ratio']:.1%}, "
            f"neg={label_info['neg_ratio']:.1%}"
        )
    warnings.append(
        f"当前样本量 {final_count} 条，train/val/test 三路切分后 "
        f"test 约 {int(final_count * 0.15)} 条。"
    )
    for tbl_key in allowed_empty:
        tbl_name = f"raw_{tbl_key}"
        warnings.append(f"{tbl_name} 当前为空表，不作为特征来源。")
    warnings.append(
        "digg_count/comment_count/share_count/collect_count 已从特征中排除 "
        "(leakage control)。"
    )
    warnings.append(
        "max_comment_digg_count/avg_related_digg_count/avg_related_comment_count 等 "
        "潜在泄漏字段已从聚合特征中排除。"
    )
    warnings.append(
        "interaction_score 仅作为审计列保留，不进入模型特征。"
    )

    feature_info = {
        "dataset_name": dataset_name,
        "run_id": run_id,
        "quality_filter_mode": qf_mode,
        "label_col": label_col,
        "label_definition": label_definition,
        "label_threshold": label_info["threshold"],
        "id_cols": id_cols,
        "numeric_features": actual_numeric,
        "categorical_features": actual_cat,
        "text_stat_features": actual_text_stat,
        "aggregation_features": actual_agg,
        "wide_cross_features": actual_wide,
        "excluded_fields": excluded_fields,
        "excluded_label_source_features": excluded_label_source,
        "excluded_potential_leakage_cols": excluded_potential_leakage,
        "field_quality_report": {
            k: {col: info for col, info in v.items() if info.get("is_constant") or info.get("non_null_rate", 1) < 0.5}
            for k, v in field_quality_report.items()
        },
        "total_features": total_features,
        "split_info": {
            "method": split_info["method"],
            "seed": split_info["seed"],
            "train_size": split_info["train_size"],
            "val_size": split_info["val_size"],
            "test_size": split_info["test_size"],
            "total_size": split_info["total_size"],
            "train_pos": split_info["train_pos"],
            "train_neg": split_info["train_neg"],
            "val_pos": split_info["val_pos"],
            "val_neg": split_info["val_neg"],
            "test_pos": split_info["test_pos"],
            "test_neg": split_info["test_neg"],
        },
        "label_distribution_total": {
            "pos": label_info["pos_count"],
            "neg": label_info["neg_count"],
            "pos_ratio": label_info["pos_ratio"],
        },
        "label_distribution_train": {
            "pos": split_info["train_pos"],
            "neg": split_info["train_neg"],
            "pos_ratio": split_info["train_pos_ratio"],
        },
        "label_distribution_val": {
            "pos": split_info["val_pos"],
            "neg": split_info["val_neg"],
            "pos_ratio": split_info["val_pos_ratio"],
        },
        "label_distribution_test": {
            "pos": split_info["test_pos"],
            "neg": split_info["test_neg"],
            "pos_ratio": split_info["test_pos_ratio"],
        },
        "split_exclusivity": exclusivity,
        "test_split_note": (
            "test split 仅用于最终泛化评估，不得用于调参或早停。"
            "val split 用于训练过程中选 best epoch、早停和超参数调优。"
        ),
        "source_tables_used": list(tables_cfg.keys()),
        "join_keys_used": ["video_id", "author_id", "source_video_id"],
        "leakage_control": {
            "excluded_label_source_features": excluded_label_source,
            "excluded_potential_leakage_cols": excluded_potential_leakage,
            "audit_id_cols": audit_id_cols,
            "leakage_check_passed": leakage_report["leakage_check_passed"],
        },
        "warnings": warnings,
        "known_limitations": [
            "real_raw_5000 来自公开网页端，不代表平台内部完整数据。",
            "当前没有真实曝光、点击、完播、转化、留存标签。",
            "标签为 interaction_score 分位数伪标签，仅用于离线多模型对比。",
            "raw_video_tag 当前为空。",
            "raw_chapter 当前为空。",
            f"none-match 样本占 {n_none_ratio*100:.1f}%，部分互动字段覆盖率低。",
            "评论和相关推荐不是每个视频都触发。",
            "所有模型结果仍属于离线实验结果，不代表线上业务收益。",
        ],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": [
            f"本数据集基于 real_raw_5000 真实网页端 raw 数据构建 (quality_filter_mode={qf_mode})。",
            "标签为 interaction_score 分位数伪标签，仅供离线多模型对比实验使用。",
            "train/val/test 三路切分，按 video_id 分层。",
            "已实施 no_interaction_leakage 控制: 四个 label source 字段已从特征中删除。",
            "潜在泄漏字段（comment_digg/related_digg 等）已从聚合特征中排除。",
            f"共包含来自 {len(tables_cfg)} 张源表的特征。",
        ],
    }

    feature_info_path = features_dir / "tabular_feature_info.json"
    with open(feature_info_path, "w", encoding="utf-8") as f:
        json.dump(feature_info, f, ensure_ascii=False, indent=2)
    print(f"  feature_info.json: {feature_info_path}")

    # 11c. tabular_dataset_report.json
    report = {
        "dataset_name": dataset_name,
        "run_id": run_id,
        "input_dir": str(data_root),
        "quality_filter_mode": qf_mode,
        "output_train_path": str(train_path),
        "output_val_path": str(val_path),
        "output_test_path": str(test_path),
        "output_feature_info_path": str(feature_info_path),
        "original_total": original_count,
        "filtered_total": final_count,
        "train_rows": len(df_train),
        "val_rows": len(df_val),
        "test_rows": len(df_test),
        "total_columns": len(column_order),
        "total_features": total_features,
        "numeric_feature_count": len(actual_numeric),
        "categorical_feature_count": len(actual_cat),
        "text_stat_feature_count": len(actual_text_stat),
        "aggregation_feature_count": len(actual_agg),
        "wide_cross_feature_count": len(actual_wide),
        "excluded_fields": excluded_fields,
        "excluded_label_source_features": excluded_label_source,
        "excluded_potential_leakage_cols": excluded_potential_leakage,
        "label_summary": {
            "method": label_info["method"],
            "threshold": label_info["threshold"],
            "pos_count": label_info["pos_count"],
            "neg_count": label_info["neg_count"],
            "pos_ratio": label_info["pos_ratio"],
            "label_definition": label_definition,
        },
        "split_summary": {
            "method": split_info["method"],
            "seed": split_info["seed"],
            "train_size": split_info["train_size"],
            "val_size": split_info["val_size"],
            "test_size": split_info["test_size"],
        },
        "split_exclusivity": exclusivity,
        "test_split_note": feature_info["test_split_note"],
        "empty_tables_handled": allowed_empty,
        "missing_summary": {
            col: info
            for col, info in missing_summary.items()
            if info["missing_count"] > 0
        },
        "warnings": warnings,
        "generated_at": feature_info["generated_at"],
        "known_limitations": feature_info["known_limitations"],
    }

    report_path = check_dir / "real_raw_5000_tabular_dataset_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  tabular_dataset_report.json: {report_path}")

    # 11d. preview CSV
    preview_path = check_dir / "real_raw_5000_tabular_dataset_preview.csv"
    main_df.head(20).to_csv(preview_path, index=False, encoding="utf-8-sig")
    print(f"  tabular_dataset_preview.csv: {preview_path}")

    # 11e. leakage_check_report.json
    leakage_report_path = check_dir / "real_raw_5000_leakage_check_report.json"
    with open(leakage_report_path, "w", encoding="utf-8") as f:
        json.dump(leakage_report, f, ensure_ascii=False, indent=2)
    print(f"  leakage_check_report.json: {leakage_report_path}")

    # =========================================================================
    # 汇总
    # =========================================================================
    print("\n" + "=" * 60)
    print("构建完成")
    print("=" * 60)
    print(f"  数据集:           {dataset_name}")
    print(f"  质量过滤模式:     {qf_mode}")
    print(f"  总样本数:         {final_count} (过滤前 {original_count})")
    print(f"  Train:            {len(df_train)} (pos={split_info['train_pos']}, neg={split_info['train_neg']})")
    print(f"  Val:              {len(df_val)} (pos={split_info['val_pos']}, neg={split_info['val_neg']})")
    print(f"  Test:             {len(df_test)} (pos={split_info['test_pos']}, neg={split_info['test_neg']})")
    print(f"  总特征列数:       {len(explicit_features)}")
    print(f"  - 数值:           {len(actual_numeric)}")
    print(f"  - 类别:           {len(actual_cat)}")
    print(f"  - 文本统计:       {len(actual_text_stat)}")
    print(f"  - 交叉:           {len(actual_wide)}")
    print(f"  正样本:           {label_info['pos_count']} ({label_info['pos_ratio']:.1%})")
    print(f"  负样本:           {label_info['neg_count']} ({label_info['neg_ratio']:.1%})")
    print(f"  video_id 互斥:    {'OK' if exclusivity['exclusive'] else '重叠!'}")
    print(f"  泄漏检查:         {'通过' if leakage_report['leakage_check_passed'] else '失败!'}")
    print("=" * 60)


if __name__ == "__main__":
    main()