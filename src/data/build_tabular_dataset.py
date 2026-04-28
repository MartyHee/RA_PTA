"""
DNN / Wide & Deep 共用 tabular 数据集构建脚本。

基于 sample0427 的 11 张表，生成表格模型可直接使用的训练集、评估集和特征说明文件。

用法:
    python src/data/build_tabular_dataset.py
    python src/data/build_tabular_dataset.py --config configs/common/data_paths.yaml
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
# 文件映射（与 load_sample0427.py 保持一致）
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


# =============================================================================
# 全空字段列表（来自 schema 校验输出）
# =============================================================================
ALL_NULL_COLS: dict[str, list[str]] = {
    "raw_video_detail": ["sec_item_id", "share_url", "preview_title", "item_title", "shoot_way"],
    "raw_author": ["short_id", "custom_verify", "enterprise_verify_reason"],
    "raw_music": ["music_mid", "music_owner_id"],
    "raw_video_media": ["cover_uri", "video_format", "video_ratio", "bit_rate_raw", "big_thumbs_raw", "video_meta_raw"],
    "raw_chapter": ["chapter_cover_url"],
    "raw_comment": ["label_text", "comment_user_sec_uid", "comment_user_unique_id"],
    "raw_related_video": ["related_author_sec_uid", "related_music_title", "related_text_extra_raw", "related_chapter_abstract"],
}

# 数值类型的占位字段（全 -1），应排除
PLACEHOLDER_NUMERIC_COLS: list[str] = [
    "cover_width", "cover_height",
    "dynamic_cover_width", "dynamic_cover_height",
    "video_width", "video_height",
    "is_h265", "is_long_video",
    "music_duration", "music_shoot_duration", "music_collect_count",
    "caption_start", "caption_end",
    # status_control 字段虽不是 -1，但全部为常量，排除
    "can_comment", "can_forward", "can_share", "can_show_comment",
    "allow_download", "allow_duet", "allow_music", "allow_record", "allow_stitch",
    "private_status", "is_delete", "is_prohibited", "in_reviewing",
    "review_status", "comment_permission_status",
]

# 所有 status_control 字段（全部常量）
STATUS_CONTROL_COLS: list[str] = [
    "can_comment", "can_forward", "can_share", "can_show_comment",
    "allow_download", "allow_duet", "allow_music", "allow_record", "allow_stitch",
    "private_status", "is_delete", "is_prohibited", "in_reviewing",
    "review_status", "comment_permission_status",
]

# 样本中全部为 -1 的字段，应排除
ALL_MINUS_ONE_COLS: list[str] = [
    "favoriting_count",
    "following_count",
]


def load_all_tables(sample0427_dir: Path) -> dict[str, pd.DataFrame]:
    """读取所有 CSV 表并返回 {table_name: df} 字典。"""
    tables: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    for table_name, file_name in CSV_FILE_MAP.items():
        file_path = sample0427_dir / file_name
        try:
            df, encoding = read_csv_safe(str(file_path))
            tables[table_name] = df
            print(f"  [{table_name}] {file_name} -> {len(df)} 行 x {len(df.columns)} 列 | {encoding}")
        except Exception as e:
            print(f"  [{table_name}] 读取失败: {e}")
            failed.append(table_name)
    if failed:
        print(f"\n[W] 以下表读取失败: {failed}")
    return tables


def drop_all_null_columns(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """排除指定表的全空字段。"""
    to_drop = [c for c in ALL_NULL_COLS.get(table_name, []) if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"    - 排除全空字段: {to_drop}")
    return df


def drop_placeholder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """排除数值占位字段和全常量状态控制字段。"""
    to_drop = [c for c in PLACEHOLDER_NUMERIC_COLS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"    - 排除占位/常量字段: {to_drop}")
    # 排除全 -1 字段
    minus_one_drop = [c for c in ALL_MINUS_ONE_COLS if c in df.columns]
    if minus_one_drop:
        df = df.drop(columns=minus_one_drop)
        print(f"    - 排除全 -1 字段: {minus_one_drop}")
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

    Returns:
        (df_with_label, label_info_dict)
    """
    df = df.copy()
    # 确保所有组件列存在
    valid_components = [c for c in components if c in df.columns]
    missing = [c for c in components if c not in df.columns]
    if missing:
        print(f"  [W] 标签组件列缺失: {missing}")

    df[score_col] = df[valid_components].sum(axis=1).fillna(0).astype(float)

    # 尝试 default_quantile
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


def train_eval_split(
    df: pd.DataFrame,
    seed: int = 2026,
    train_ratio: float = 0.8,
    eval_ratio: float = 0.2,
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """按 video_id 维度随机切分 train/eval。

    Returns:
        (df_with_split, df_train, split_info)
    """
    df = df.copy()
    np.random.seed(seed)

    video_ids = df["video_id"].unique()
    n_total = len(video_ids)
    n_train = max(1, int(n_total * train_ratio))

    shuffled = np.random.permutation(video_ids)
    train_ids = set(shuffled[:n_train])
    eval_ids = set(shuffled[n_train:])

    df["split"] = df["video_id"].apply(lambda x: "train" if x in train_ids else "eval")

    df_train = df[df["split"] == "train"].copy()
    df_eval = df[df["split"] == "eval"].copy()

    train_pos = df_train[label_col].sum()
    train_neg = len(df_train) - train_pos
    eval_pos = df_eval[label_col].sum()
    eval_neg = len(df_eval) - eval_pos

    split_info = {
        "method": "random",
        "seed": seed,
        "train_ratio": train_ratio,
        "eval_ratio": eval_ratio,
        "train_size": len(df_train),
        "eval_size": len(df_eval),
        "total_size": len(df),
        "train_pos": int(train_pos),
        "train_neg": int(train_neg),
        "eval_pos": int(eval_pos),
        "eval_neg": int(eval_neg),
        "train_pos_ratio": round(float(train_pos / len(df_train)), 4) if len(df_train) > 0 else 0,
        "eval_pos_ratio": round(float(eval_pos / len(df_eval)), 4) if len(df_eval) > 0 else 0,
    }
    print(f"  切分: train={len(df_train)}, eval={len(df_eval)}")
    print(f"  train label dist: pos={int(train_pos)}, neg={int(train_neg)}")
    print(f"  eval label dist:  pos={int(eval_pos)}, neg={int(eval_neg)}")

    return df, df_train, df_eval, split_info


def build_feature_info(
    df: pd.DataFrame,
    feature_config: dict[str, Any],
    label_info: dict[str, Any],
    split_info: dict[str, Any],
) -> dict[str, Any]:
    """构建 tabular_feature_info.json 所需的特征说明字典。"""
    excluded_all_null = []
    for table_name, cols in ALL_NULL_COLS.items():
        excluded_all_null.extend(cols)

    excluded_placeholder = PLACEHOLDER_NUMERIC_COLS.copy()

    id_cols = ["sample_id", "video_id", "author_id"]
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    text_stat_cols: list[str] = []
    wide_cross_cols: list[str] = []

    for table_name, cols in feature_config.get("numeric_feature_candidates", {}).items():
        for col in cols:
            if col in df.columns and col not in id_cols:
                numeric_cols.append(col)

    agg_count_names = ["hashtag_count", "video_tag_count", "comment_table_count",
                       "max_comment_digg_count", "chapter_count", "related_video_count"]
    for col in agg_count_names:
        if col in df.columns and col not in numeric_cols and col not in id_cols:
            numeric_cols.append(col)

    cat_candidates = feature_config.get("categorical_feature_candidates", [])
    for col in cat_candidates:
        if col in df.columns:
            categorical_cols.append(col)

    text_stat_candidates = feature_config.get("text_stat_features", [])
    for col in text_stat_candidates:
        if col in df.columns:
            text_stat_cols.append(col)

    cross_configs = feature_config.get("wide_cross_features", [])
    for cfg in cross_configs:
        name = cfg.get("name")
        if name and name in df.columns:
            wide_cross_cols.append(name)

    # 构建 warnings
    warnings: list[str] = []
    # 全 -1 字段
    warnings.append("favoriting_count / following_count 在 sample0427 中全部为 -1，已从特征中排除")
    # 缺失情况
    if "music_author" in df.columns:
        ma_missing = df["music_author"].isna().sum()
        warnings.append(f"music_author 缺失 {ma_missing}/{len(df)}，{ma_missing/len(df):.1%}")
    if "hashtag_count" in df.columns:
        hc_missing = df["hashtag_count"].isna().sum()
        warnings.append(f"hashtag_count 缺失 {hc_missing}/{len(df)} ({hc_missing/len(df):.1%}) 的视频无 hashtag")
    # 样本量
    if len(df) < 100:
        warnings.append(f"样本量仅 {len(df)} 条，train/eval split 结果不稳定，后续需扩大样本至 1000+")
    # play_count 全为 0
    if "play_count" in df.columns and df["play_count"].nunique() == 1:
        warnings.append("play_count 全部为 0（公开页面可能返回 0），无特征区分度")

    return {
        "label_col": label_info["label_col"],
        "label_definition": "流程验证伪标签: interaction_score >= threshold",
        "label_threshold": label_info["threshold"],
        "id_cols": id_cols,
        "numeric_cols": [c for c in numeric_cols if c not in text_stat_cols and c not in id_cols],
        "categorical_cols": categorical_cols,
        "text_stat_cols": text_stat_cols,
        "wide_cross_cols": wide_cross_cols,
        "excluded_all_null_cols": excluded_all_null,
        "excluded_placeholder_cols": excluded_placeholder,
        "generated_or_placeholder_cols_used": [],
        "train_size": split_info["train_size"],
        "eval_size": split_info["eval_size"],
        "total_size": split_info["total_size"],
        "label_distribution_total": {
            "pos": label_info["pos_count"],
            "neg": label_info["neg_count"],
            "pos_ratio": label_info["pos_ratio"],
        },
        "label_distribution_train": {
            "pos": split_info["train_pos"],
            "neg": split_info["train_neg"],
        },
        "label_distribution_eval": {
            "pos": split_info["eval_pos"],
            "neg": split_info["eval_neg"],
        },
        "source_tables_used": list(CSV_FILE_MAP.keys()),
        "join_keys_used": ["video_id", "author_id", "source_video_id"],
        "warnings": warnings,
        "notes": [
            "当前标签为流程验证伪标签，基于 interaction_score (digg+comment+share+collect) 构造。",
            "不代表真实曝光/点击/转化目标。当前不能用于正式效果结论。",
            "部分类别特征（如 tag_id, hashtag_id, music_id）为规则生成ID，已在数值特征中排除。",
            "status_control 全部字段为固定默认值，已从特征中排除。",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="tabular 数据集构建")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "common" / "data_paths.yaml"),
        help="数据路径配置文件路径",
    )
    parser.add_argument(
        "--feature-config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "common" / "feature_tabular.yaml"),
        help="特征构建配置文件路径",
    )
    parser.add_argument(
        "--split-config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "common" / "split.yaml"),
        help="切分配置文件路径",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Tabular 数据集构建")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. 加载配置
    # ------------------------------------------------------------------
    cfg_path = Path(args.config)
    feat_cfg_path = Path(args.feature_config)
    split_cfg_path = Path(args.split_config)

    print(f"\n[1/8] 加载配置...")
    path_cfg = load_config(str(cfg_path))
    feature_cfg = load_config(str(feat_cfg_path))
    split_cfg = load_config(str(split_cfg_path))

    sample0427_dir = Path(path_cfg["sample0427_dir"])
    features_dir = Path(path_cfg["features_dir"])
    data_check_dir = Path(path_cfg["data_check_dir"])

    # 从 split.yaml 读取切分参数（允许被代码中默认值覆盖）
    split_seed = split_cfg.get("random_seed", 2026)
    split_train_ratio = split_cfg.get("train_ratio", 0.8)

    # 使用 train_ratio 作为训练比例，剩余为 eval（无 test 集）
    eval_ratio = round(1.0 - split_train_ratio, 2)

    features_dir.mkdir(parents=True, exist_ok=True)
    data_check_dir.mkdir(parents=True, exist_ok=True)

    output_paths = feature_cfg.get("output_paths", {})
    print(f"  数据路径: {sample0427_dir}")
    print(f"  特征输出: {features_dir}")
    print(f"  报告输出: {data_check_dir}")

    # ------------------------------------------------------------------
    # 2. 加载所有表
    # ------------------------------------------------------------------
    print(f"\n[2/8] 加载所有表...")
    tables = load_all_tables(sample0427_dir)

    # ------------------------------------------------------------------
    # 3. 构建主表 (raw_video_detail) -> 清理全空/占位字段
    # ------------------------------------------------------------------
    print(f"\n[3/8] 构建主表 + 清理字段...")
    main_df = tables["raw_video_detail"].copy()
    print(f"  raw_video_detail 原始列数: {len(main_df.columns)}")
    main_df = drop_all_null_columns(main_df, "raw_video_detail")
    print(f"  清理后列数: {len(main_df.columns)}")

    # 创建 sample_id
    main_df["sample_id"] = main_df["video_id"].astype(str)

    # ------------------------------------------------------------------
    # 4. 多表 Join / 聚合
    # ------------------------------------------------------------------
    print(f"\n[4/8] 多表 Join / 聚合...")

    # 4a. raw_author (直接 LEFT JOIN)
    print("  raw_author: LEFT JOIN on author_id")
    author_df = tables["raw_author"].copy()
    author_df = drop_all_null_columns(author_df, "raw_author")
    # 排除占位列（avatar_thumb_url_list, cover_url_list 为空列表占位，sec_uid/unique_id 为规则生成）
    # 保留 follower_count, total_favorited, favoriting_count, following_count, verification_type
    # author_status, author_secret, author_prevent_download, signature 等
    author_cols_to_drop = ["sec_uid", "unique_id", "avatar_thumb_url_list", "cover_url_list", "favoriting_count", "following_count"]
    author_df = author_df.drop(columns=[c for c in author_cols_to_drop if c in author_df.columns])
    print(f"    raw_author 清理后列: {list(author_df.columns)}")
    main_df = main_df.merge(author_df, on="author_id", how="left", suffixes=("", "_author_dup"))
    # 删除重复列
    dup_cols = [c for c in main_df.columns if c.endswith("_author_dup")]
    if dup_cols:
        main_df = main_df.drop(columns=dup_cols)
    print(f"    merge 后主表行数: {len(main_df)}")

    # 4b. raw_music (直接 LEFT JOIN)
    print("  raw_music: LEFT JOIN on video_id")
    music_df = tables["raw_music"].copy()
    music_df = drop_all_null_columns(music_df, "raw_music")
    music_df = drop_placeholder_columns(music_df)
    # music_id 是规则生成 ID，排除
    if "music_id" in music_df.columns:
        music_df = music_df.drop(columns=["music_id"])
    # 保留 music_title, music_author, is_original_sound, is_commerce_music
    print(f"    raw_music 清理后列: {list(music_df.columns)}")
    main_df = main_df.merge(music_df, on="video_id", how="left", suffixes=("", "_music_dup"))
    dup_cols = [c for c in main_df.columns if c.endswith("_music_dup")]
    if dup_cols:
        main_df = main_df.drop(columns=dup_cols)

    # 4c. raw_video_media (直接 LEFT JOIN)
    print("  raw_video_media: LEFT JOIN on video_id")
    media_df = tables["raw_video_media"].copy()
    media_df = drop_all_null_columns(media_df, "raw_video_media")
    media_df = drop_placeholder_columns(media_df)
    print(f"    raw_video_media 清理后列: {list(media_df.columns)}")
    main_df = main_df.merge(media_df, on="video_id", how="left", suffixes=("", "_media_dup"))
    dup_cols = [c for c in main_df.columns if c.endswith("_media_dup")]
    if dup_cols:
        main_df = main_df.drop(columns=dup_cols)

    # 4d. raw_video_status_control — 全部为常量，不参与特征
    print("  raw_video_status_control: 全部常量，跳过特征 (仅用于记录)")
    # 不进行 merge，避免常量列污染特征

    # 4e. raw_hashtag — 聚合到 video_id
    print("  raw_hashtag: 聚合到 video_id")
    hashtag_df = tables["raw_hashtag"].copy()
    hashtag_agg = build_aggregated_features(
        hashtag_df, "video_id",
        feature_cfg["aggregation_features"]["raw_hashtag"],
    )
    # 取第一个 hashtag_name 作为 top_hashtag_name
    top_hashtag = (
        hashtag_df.groupby("video_id")["hashtag_name"]
        .first()
        .reset_index(name="hashtag_name_top")
    )
    hashtag_agg = hashtag_agg.merge(top_hashtag, on="video_id", how="left")
    print(f"    聚合特征: {list(hashtag_agg.columns)}")
    main_df = main_df.merge(hashtag_agg, on="video_id", how="left")

    # 4f. raw_video_tag — 聚合到 video_id
    print("  raw_video_tag: 聚合到 video_id（完全补齐表）")
    tag_df = tables["raw_video_tag"].copy()
    tag_agg = build_aggregated_features(
        tag_df, "video_id",
        feature_cfg["aggregation_features"]["raw_video_tag"],
    )
    print(f"    聚合特征: {list(tag_agg.columns)}")
    main_df = main_df.merge(tag_agg, on="video_id", how="left")

    # 4g. raw_comment — 聚合到 video_id
    print("  raw_comment: 聚合到 video_id（完全补齐表）")
    comment_df = tables["raw_comment"].copy()
    # 排除全空字段
    comment_df = drop_all_null_columns(comment_df, "raw_comment")
    comment_agg = build_aggregated_features(
        comment_df, "video_id",
        feature_cfg["aggregation_features"]["raw_comment"],
    )
    print(f"    聚合特征: {list(comment_agg.columns)}")
    main_df = main_df.merge(comment_agg, on="video_id", how="left")

    # 4h. raw_chapter — 聚合到 video_id
    print("  raw_chapter: 聚合到 video_id（完全补齐表）")
    chapter_df = tables["raw_chapter"].copy()
    chapter_agg = build_aggregated_features(
        chapter_df, "video_id",
        feature_cfg["aggregation_features"]["raw_chapter"],
    )
    print(f"    聚合特征: {list(chapter_agg.columns)}")
    main_df = main_df.merge(chapter_agg, on="video_id", how="left")

    # 4i. raw_related_video — 聚合到 source_video_id
    print("  raw_related_video: 聚合到 source_video_id（完全补齐表）")
    related_df = tables["raw_related_video"].copy()
    related_agg = build_aggregated_features(
        related_df, "source_video_id",
        feature_cfg["aggregation_features"]["raw_related_video"],
    )
    related_agg = related_agg.rename(columns={"source_video_id": "video_id"})
    print(f"    聚合特征: {list(related_agg.columns)}")
    main_df = main_df.merge(related_agg, on="video_id", how="left")

    print(f"\n  合并后主表: {len(main_df)} 行 x {len(main_df.columns)} 列")

    # ------------------------------------------------------------------
    # 5. 特征工程
    # ------------------------------------------------------------------
    print(f"\n[5/8] 特征工程...")

    # 5a. 文本统计特征
    print("  文本统计特征:")
    if "desc" in main_df.columns:
        main_df = build_text_stat_features(main_df, "desc")
        print(f"    desc -> desc_length, desc_word_count")
    if "signature" in main_df.columns:
        main_df = build_text_stat_features(main_df, "signature")
        print(f"    signature -> signature_length")
    if "caption" in main_df.columns and "desc" not in main_df.columns:
        main_df = build_text_stat_features(main_df, "caption")

    # 5b. Duration 桶化 (用于交叉特征)
    if "duration_ms" in main_df.columns:
        main_df = build_duration_bucket(main_df)
        print("  duration_ms -> duration_bucket (short/medium/long/very_long)")

    # 5c. 处理 numeric 特征中的 -1 占位值
    # raw_author 中的 favoriting_count, following_count 部分为 -1
    for col in ["favoriting_count", "following_count"]:
        if col in main_df.columns:
            neg_one_mask = main_df[col] == -1
            if neg_one_mask.any():
                n = neg_one_mask.sum()
                # 将 -1 替换为 NaN，后续填充
                main_df.loc[neg_one_mask, col] = np.nan
                print(f"    {col}: {n} 个 -1 已替换为 NaN")

    # 5d. 处理 join 后可能出现的 NaN (fill values for numeric features)
    # 先记录缺失情况
    missing_before = main_df.isna().sum()
    missing_cols = missing_before[missing_before > 0].index.tolist()
    if missing_cols:
        print(f"  Join 后含 NaN 的列: {missing_cols}")

    # 5e. 交叉特征
    print("  交叉特征:")
    cross_configs = feature_cfg.get("wide_cross_features", [])
    cross_features_built = []
    for cfg in cross_configs:
        left = cfg["left"]
        right = cfg["right"]
        name = cfg.get("name", f"{left}_x_{right}")
        if left in main_df.columns and right in main_df.columns:
            main_df = build_cross_features(main_df, [cfg])
            cross_features_built.append(name)
            print(f"    {name}: {left} x {right}")
        else:
            missing_side = []
            if left not in main_df.columns:
                missing_side.append(left)
            if right not in main_df.columns:
                missing_side.append(right)
            print(f"    [W] 跳过 {name}: 缺少列 {missing_side}")

    # ------------------------------------------------------------------
    # 6. 标签构造
    # ------------------------------------------------------------------
    print(f"\n[6/8] 标签构造...")
    label_config = feature_cfg.get("label_config", {})
    components = label_config.get("interaction_components", [])
    main_df, label_info = build_label(
        main_df,
        components=components,
        default_quantile=label_config.get("default_quantile", 0.60),
        fallback_quantile=label_config.get("fallback_quantile", 0.50),
        imbalance_threshold=label_config.get("imbalance_threshold", 0.20),
        label_col=label_config.get("label_col", "label"),
        score_col=label_config.get("score_col", "interaction_score"),
    )

    # ------------------------------------------------------------------
    # 7. Train / Eval 切分
    # ------------------------------------------------------------------
    print(f"\n[7/8] Train/Eval 切分...")
    main_df, df_train, df_eval, split_info = train_eval_split(
        main_df,
        seed=split_seed,
        train_ratio=split_train_ratio,
        eval_ratio=eval_ratio,
        label_col=label_config.get("label_col", "label"),
    )

    # 按 split 排序整理列顺序 — 只保留明确选择的特征列
    id_cols = ["sample_id", "video_id", "author_id"]
    label_col = label_config.get("label_col", "label")
    score_col = label_config.get("score_col", "interaction_score")
    split_col = "split"

    # 从 config 中收集明确选择的特征列
    selected_numeric: list[str] = []
    for table_name, cols in feature_cfg.get("numeric_feature_candidates", {}).items():
        for col in cols:
            if col not in selected_numeric:
                selected_numeric.append(col)

    # 移除被排除的字段
    selected_numeric = [
        c for c in selected_numeric
        if c not in PLACEHOLDER_NUMERIC_COLS
        and c not in ALL_MINUS_ONE_COLS
        and c not in ALL_NULL_COLS.get("raw_author", [])
        and c not in ALL_NULL_COLS.get("raw_video_detail", [])
    ]

    # 文本统计特征
    text_stat_cols = [
        c for c in feature_cfg.get("text_stat_features", [])
        if c in main_df.columns
    ]

    # 类别特征
    cat_cols = [
        c for c in feature_cfg.get("categorical_feature_candidates", [])
        if c in main_df.columns
    ]

    # wide_cross
    wide_cross_cols = cross_features_built

    # 聚合计数特征（从聚合列表自动收集）
    agg_count_features = [
        "hashtag_count", "video_tag_count", "comment_table_count",
        "max_comment_digg_count", "chapter_count", "related_video_count",
    ]
    agg_count_features = [c for c in agg_count_features if c in main_df.columns]

    # 最终的明确特征列集合
    explicit_features = list(dict.fromkeys(
        selected_numeric + text_stat_cols + cat_cols + agg_count_features + wide_cross_cols
    ))
    explicit_features = [c for c in explicit_features if c in main_df.columns]

    # 构建输出列序
    column_order = (
        id_cols
        + [c for c in explicit_features if c not in id_cols]
        + [score_col, label_col, split_col]
    )
    column_order = [c for c in column_order if c in main_df.columns]

    main_df = main_df[column_order]
    df_train = df_train[column_order]
    df_eval = df_eval[column_order]

    # ------------------------------------------------------------------
    # 8. 输出
    # ------------------------------------------------------------------
    print(f"\n[8/8] 输出文件...")

    # 8a. 输出 train.csv 和 eval.csv
    train_path = features_dir / "tabular_train.csv"
    eval_path = features_dir / "tabular_eval.csv"
    df_train.to_csv(train_path, index=False, encoding="utf-8-sig")
    df_eval.to_csv(eval_path, index=False, encoding="utf-8-sig")
    print(f"  train.csv: {train_path} ({len(df_train)} 行)")
    print(f"  eval.csv:  {eval_path} ({len(df_eval)} 行)")

    # 8b. 构建 feature_info.json
    feature_info = build_feature_info(main_df, feature_cfg, label_info, split_info)
    feature_info_path = features_dir / "tabular_feature_info.json"
    with open(feature_info_path, "w", encoding="utf-8") as f:
        json.dump(feature_info, f, ensure_ascii=False, indent=2)
    print(f"  feature_info.json: {feature_info_path}")

    # 8c. tabular_dataset_report.json
    missing_summary = compute_missing_summary(main_df)
    report = {
        "input_dir": str(sample0427_dir),
        "output_train_path": str(train_path),
        "output_eval_path": str(eval_path),
        "output_feature_info_path": str(feature_info_path),
        "total_rows": len(main_df),
        "train_rows": len(df_train),
        "eval_rows": len(df_eval),
        "total_columns": len(column_order),
        "numeric_feature_count": len(feature_info["numeric_cols"]),
        "categorical_feature_count": len(feature_info["categorical_cols"]),
        "text_stat_feature_count": len(feature_info["text_stat_cols"]),
        "wide_cross_feature_count": len(feature_info["wide_cross_cols"]),
        "excluded_all_null_cols": feature_info["excluded_all_null_cols"],
        "excluded_placeholder_cols": feature_info["excluded_placeholder_cols"],
        "label_summary": {
            "method": label_info["method"],
            "threshold": label_info["threshold"],
            "pos_count": label_info["pos_count"],
            "neg_count": label_info["neg_count"],
            "pos_ratio": label_info["pos_ratio"],
        },
        "split_summary": {
            "method": "random_by_video_id",
            "seed": split_info["seed"],
            "train_size": split_info["train_size"],
            "eval_size": split_info["eval_size"],
        },
        "missing_summary": {
            col: info
            for col, info in missing_summary.items()
            if info["missing_count"] > 0
        },
        "warnings": feature_info["warnings"],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": [
            "当前数据集基于 sample0427 样本数据构建，仅用于流程级验证。",
            "标签为交互伪标签 (interaction_score 分位数)，不代表真实曝光/点击/转化目标。",
            "5 张完全补齐表 (raw_video_tag/raw_video_status_control/raw_chapter/raw_comment/raw_related_video) 的数据不代表真实分布。",
            "约 27 个全空字段已被排除。",
            "status_control 全部字段为固定默认值，已从特征中排除。",
            "raw_crawl_log 不作为模型特征来源。",
        ],
    }
    report_path = data_check_dir / "tabular_dataset_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  tabular_dataset_report.json: {report_path}")

    # 8d. 输出 preview CSV（前 20 行）
    preview_path = data_check_dir / "tabular_dataset_preview.csv"
    main_df.head(20).to_csv(preview_path, index=False, encoding="utf-8-sig")
    print(f"  tabular_dataset_preview.csv: {preview_path}")

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("构建完成")
    print("=" * 60)
    print(f"  总样本数:     {len(main_df)}")
    print(f"  Train:        {len(df_train)}")
    print(f"  Eval:         {len(df_eval)}")
    print(f"  总特征列数:   {len(column_order)}")
    print(f"  - 数值:       {feature_info['numeric_cols']}")
    print(f"  - 类别:       {feature_info['categorical_cols']}")
    print(f"  - 文本统计:   {feature_info['text_stat_cols']}")
    print(f"  - 交叉:       {feature_info['wide_cross_cols']}")
    print(f"  正样本:       {label_info['pos_count']} ({label_info['pos_ratio']:.1%})")
    print(f"  负样本:       {label_info['neg_count']} ({label_info['neg_ratio']:.1%})")
    print("=" * 60)


if __name__ == "__main__":
    main()