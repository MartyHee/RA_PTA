"""多模态数据集构建主程序

基于 sample0427 原始表和已生成的 tabular train/eval 数据，
构建文本、视觉（媒体元信息）、结构化三模态的 npz 输入文件，
为下一步多模态模型训练做准备。

用法:
    python src/data/build_multimodal_dataset.py --config configs/multimodal/multimodal_base.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.text_features import (  # noqa: E402
    build_combined_text,
    fit_text_vectorizer,
    transform_text,
)
from src.features.image_features import build_visual_features  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import read_csv_safe  # noqa: E402

logger = print  # 轻量日志

# ── 预期 CSV 文件映射 ───────────────────────────────────────────────────
SAMPLE_TABLES: dict[str, str] = {
    "raw_video_detail": "sample0427_raw_video_detail.csv",
    "raw_chapter": "sample0427_raw_chapter.csv",
    "raw_comment": "sample0427_raw_comment.csv",
    "raw_hashtag": "sample0427_raw_hashtag.csv",
    "raw_music": "sample0427_raw_music.csv",
    "raw_author": "sample0427_raw_author.csv",
    "raw_video_media": "sample0427_raw_video_media.csv",
}


def load_sample_table(sample_dir: Path, table_name: str, filename: str) -> pd.DataFrame:
    """加载单张 sample0427 表。"""
    path = sample_dir / filename
    if not path.exists():
        logger(f"  [WARN] 文件不存在: {path} → 返回空 DataFrame")
        return pd.DataFrame()
    df, enc = read_csv_safe(str(path))
    logger(f"  [{table_name}] {len(df)} 行 x {len(df.columns)} 列 (编码: {enc})")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="多模态数据集构建")
    parser.add_argument(
        "--config",
        type=str,
        default=str(_PROJECT_ROOT / "configs/multimodal/multimodal_base.yaml"),
        help="多模态配置文件路径",
    )
    args = parser.parse_args()

    # ── 1. 加载配置 ─────────────────────────────────────────────────────
    logger(f"[build_multimodal] 加载配置: {args.config}")
    config = load_config(args.config)

    project_root = _PROJECT_ROOT
    sample0427_dir = project_root / "douyin_data_project" / "data" / "sample0427"
    multimodal_dir = project_root / "data" / "multimodal"
    features_dir = project_root / "data" / "features"
    data_check_dir = project_root / "outputs" / "data_check"

    multimodal_dir.mkdir(parents=True, exist_ok=True)
    data_check_dir.mkdir(parents=True, exist_ok=True)

    text_dim = config.get("text_dim", 32)
    random_seed = config.get("random_seed", 2026)
    notes: list[str] = []
    warnings: list[str] = []
    used_fallback = False

    # ── 2. 加载 tabular train/eval 基准 ────────────────────────────────
    logger("[build_multimodal] 加载 tabular train/eval 基准...")
    train_path = features_dir / "tabular_train.csv"
    eval_path = features_dir / "tabular_eval.csv"
    feature_info_path = features_dir / "tabular_feature_info.json"
    quality_check_path = data_check_dir / "tabular_dataset_quality_check.json"

    if not train_path.exists() or not eval_path.exists():
        logger(f"[ERROR] tabular train/eval 文件不存在: {train_path} / {eval_path}")
        sys.exit(1)

    tabular_train = pd.read_csv(train_path, encoding="utf-8-sig")
    tabular_eval = pd.read_csv(eval_path, encoding="utf-8-sig")
    tabular_all = pd.concat([tabular_train, tabular_eval], ignore_index=True)

    train_ids = set(tabular_train["video_id"].unique())
    eval_ids = set(tabular_eval["video_id"].unique())

    logger(f"  Train 样本: {len(tabular_train)} (video_ids: {len(train_ids)})")
    logger(f"  Eval 样本: {len(tabular_eval)} (video_ids: {len(eval_ids)})")

    # 加载 feature_info
    with open(feature_info_path, "r", encoding="utf-8") as f:
        tabular_feature_info = json.load(f)

    # 加载 quality check
    excluded_all_minus_one = []
    excluded_all_null = []
    excluded_placeholder = []
    flagged_zero_cols = []
    flagged_constant_cols = []
    if quality_check_path.exists():
        with open(quality_check_path, "r", encoding="utf-8") as f:
            qc = json.load(f)
        excluded_all_minus_one = qc.get("excluded_fields", {}).get("excluded_all_minus_one_cols", [])
        excluded_all_null = qc.get("excluded_fields", {}).get("excluded_all_null_cols", [])
        excluded_placeholder = qc.get("excluded_fields", {}).get("excluded_placeholder_cols", [])
        flagged_zero_cols = qc.get("flagged_fields", {}).get("all_zero_cols", [])
        flagged_constant_cols = [
            c["col"] for c in qc.get("flagged_fields", {}).get("constant_cols", [])
        ]

    # ── 3. 加载 sample0427 原始表 ──────────────────────────────────────
    logger("[build_multimodal] 加载 sample0427 原始表...")
    loaded: dict[str, pd.DataFrame] = {}
    for name, fname in SAMPLE_TABLES.items():
        loaded[name] = load_sample_table(sample0427_dir, name, fname)

    video_detail = loaded["raw_video_detail"]
    chapter = loaded["raw_chapter"]
    comment = loaded["raw_comment"]
    hashtag = loaded["raw_hashtag"]
    music = loaded["raw_music"]
    author = loaded["raw_author"]
    media = loaded["raw_video_media"]

    # ── 4. 构建 Text Features ───────────────────────────────────────────
    logger("[build_multimodal] 构建 text_features...")

    # 4a. 合并文本
    text_df = build_combined_text(
        video_detail=video_detail,
        chapter=chapter if not chapter.empty else None,
        comment=comment if not comment.empty else None,
        hashtag=hashtag if not hashtag.empty else None,
        music=music if not music.empty else None,
        author=author if not author.empty else None,
    )
    logger(f"  combined_text 覆盖 video_ids: {len(text_df)}")

    # 4b. 分离 train/eval text
    text_train = text_df[text_df["video_id"].isin(train_ids)].copy()
    text_eval = text_df[text_df["video_id"].isin(eval_ids)].copy()

    # 补充缺失的 video_id
    missing_train_ids = train_ids - set(text_train["video_id"])
    missing_eval_ids = eval_ids - set(text_eval["video_id"])
    if missing_train_ids:
        logger(f"  [WARN] train 中 {len(missing_train_ids)} 个 video_id 无文本 → 填空")
        for vid in missing_train_ids:
            text_train = pd.concat(
                [text_train, pd.DataFrame({"video_id": [vid], "combined_text": [""]})],
                ignore_index=True,
            )
    if missing_eval_ids:
        logger(f"  [WARN] eval 中 {len(missing_eval_ids)} 个 video_id 无文本 → 填空")
        for vid in missing_eval_ids:
            text_eval = pd.concat(
                [text_eval, pd.DataFrame({"video_id": [vid], "combined_text": [""]})],
                ignore_index=True,
            )

    # 按 video_id 排序对齐
    text_train = text_train.sort_values("video_id").reset_index(drop=True)
    text_eval = text_eval.sort_values("video_id").reset_index(drop=True)

    # 4c. 拟合 vectorizer
    logger("  拟合 text vectorizer (TF-IDF + SVD / HashingVectorizer 回退)...")
    vectorizer, svd, text_info = fit_text_vectorizer(
        text_train["combined_text"], text_dim=text_dim, random_seed=random_seed
    )
    logger(f"  text_feature_method: {text_info.get('method', 'unknown')}")
    if text_info.get("tfidf_failed"):
        logger(f"  TF-IDF 失败: {text_info.get('tfidf_error')}")
    if text_info.get("method", "").startswith("stats_fallback"):
        used_fallback = True
        warnings.append("文本特征使用统计特征回退（非 TF-IDF/SVD）")

    # 4d. 转换
    train_text_vec = transform_text(
        text_train["combined_text"], vectorizer, svd, text_dim
    )
    eval_text_vec = transform_text(
        text_eval["combined_text"], vectorizer, svd, text_dim
    )
    actual_text_dim = train_text_vec.shape[1]
    logger(f"  text_features shape: train={train_text_vec.shape}, eval={eval_text_vec.shape}")

    # ── 5. 构建 Visual Features ──────────────────────────────────────────
    logger("[build_multimodal] 构建 visual_features (媒体元信息)...")
    all_video_ids = sorted(train_ids | eval_ids)
    vis_df, vis_cols, vis_info = build_visual_features(media, video_ids=all_video_ids)
    logger(f"  visual_dim: {vis_info['visual_dim']}, rows: {len(vis_df)}")

    # 补充缺失 video_id
    vis_by_vid = dict(zip(vis_df["video_id"].astype(np.int64), range(len(vis_df))))
    vis_np = np.zeros((len(all_video_ids), vis_info["visual_dim"]), dtype=np.float32)
    for i, vid in enumerate(all_video_ids):
        if vid in vis_by_vid:
            vis_np[i] = vis_df.iloc[vis_by_vid[vid]][vis_cols].values.astype(np.float32)

    # 分离 train/eval
    train_vis = np.array(
        [vis_np[all_video_ids.index(vid)] for vid in sorted(train_ids)]
    )
    eval_vis = np.array(
        [vis_np[all_video_ids.index(vid)] for vid in sorted(eval_ids)]
    )
    logger(f"  visual_features shape: train={train_vis.shape}, eval={eval_vis.shape}")

    # 检查缺失
    vis_has_nan = np.isnan(train_vis).any() or np.isnan(eval_vis).any()
    if vis_has_nan:
        train_vis = np.nan_to_num(train_vis, nan=0.0)
        eval_vis = np.nan_to_num(eval_vis, nan=0.0)
        warnings.append("visual_features 中存在 NaN，已填充为 0")

    # ── 6. 构建 Structured Features ─────────────────────────────────────
    logger("[build_multimodal] 构建 structured_features (复用 tabular 数值特征)...")

    # 6a. 确定可用数值列
    numeric_candidates = tabular_feature_info.get("numeric_cols", [])
    text_stat_candidates = tabular_feature_info.get("text_stat_cols", [])
    id_cols = tabular_feature_info.get("id_cols", ["sample_id", "video_id", "author_id"])
    exclude_cols = set(
        id_cols
        + ["label", "split", "interaction_score"]
        + excluded_all_minus_one
        + excluded_all_null
        + excluded_placeholder
        + flagged_zero_cols
        + flagged_constant_cols
    )

    # 从实际 CSV 列中筛选
    available_cols = [c for c in tabular_train.columns if c not in exclude_cols]
    structured_cols = [
        c for c in available_cols if c in numeric_candidates or c in text_stat_candidates
    ]
    logger(f"  structured 候选: {len(numeric_candidates)} numeric + {len(text_stat_candidates)} text_stat")
    logger(f"  排除 {len(exclude_cols)} 列, 最终使用 {len(structured_cols)} 列")

    # 6b. 提取 train/eval 结构化矩阵
    train_struct_raw = tabular_train[structured_cols].copy().astype(np.float32)
    eval_struct_raw = tabular_eval[structured_cols].copy().astype(np.float32)

    # 6c. 缺失值填充 (train median → train+eval)
    struct_imputation = {}
    for col in structured_cols:
        median_val = train_struct_raw[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        struct_imputation[col] = float(median_val)
        train_struct_raw[col] = train_struct_raw[col].fillna(median_val)
        eval_struct_raw[col] = eval_struct_raw[col].fillna(median_val)

    # 6d. 标准化 (z-score, fit on train)
    struct_mean = {}
    struct_std = {}
    train_struct_scaled = train_struct_raw.copy().values
    eval_struct_scaled = eval_struct_raw.copy().values

    for i, col in enumerate(structured_cols):
        col_mean = float(train_struct_raw[col].mean())
        col_std = float(train_struct_raw[col].std())
        if col_std < 1e-8:
            col_std = 1.0
        struct_mean[col] = col_mean
        struct_std[col] = col_std
        train_struct_scaled[:, i] = (train_struct_raw[col].values - col_mean) / col_std
        eval_struct_scaled[:, i] = (eval_struct_raw[col].values - col_mean) / col_std

    structured_dim = train_struct_scaled.shape[1]
    logger(f"  structured_features shape: train={train_struct_scaled.shape}, eval={eval_struct_scaled.shape}")

    # ── 7. 对齐样本并保存 npz ──────────────────────────────────────────
    logger("[build_multimodal] 对齐样本并保存 npz...")

    def build_npz(
        video_ids_sorted: list[int],
        text_vec: np.ndarray,
        vis_vec: np.ndarray,
        struct_vec: np.ndarray,
        split_label: str,
    ) -> dict[str, Any]:
        """构建单个 npz 的内容字典。"""
        tabular_subset = tabular_all[tabular_all["video_id"].isin(video_ids_sorted)].copy()
        tabular_subset = tabular_subset.set_index("video_id").loc[video_ids_sorted].reset_index()

        labels = tabular_subset["label"].values.astype(np.float32)
        sample_ids = tabular_subset["sample_id"].values.astype(np.int64)
        video_ids_arr = tabular_subset["video_id"].values.astype(np.int64)
        author_ids = tabular_subset["author_id"].values.astype(str)

        return {
            "sample_id": sample_ids,
            "video_id": video_ids_arr,
            "author_id": author_ids,
            "label": labels,
            "text_features": text_vec.astype(np.float32),
            "visual_features": vis_vec.astype(np.float32),
            "structured_features": struct_vec.astype(np.float32),
            "split": np.array([split_label] * len(video_ids_sorted), dtype=object),
        }

    train_vids_sorted = sorted(train_ids)
    eval_vids_sorted = sorted(eval_ids)

    train_data = build_npz(train_vids_sorted, train_text_vec, train_vis, train_struct_scaled, "train")
    eval_data = build_npz(eval_vids_sorted, eval_text_vec, eval_vis, eval_struct_scaled, "eval")

    # 保存
    train_npz_path = multimodal_dir / "multimodal_train.npz"
    eval_npz_path = multimodal_dir / "multimodal_eval.npz"

    np.savez_compressed(train_npz_path, **train_data)
    np.savez_compressed(eval_npz_path, **eval_data)
    logger(f"  Train npz: {train_npz_path} (samples={len(train_vids_sorted)})")
    logger(f"  Eval npz: {eval_npz_path} (samples={len(eval_vids_sorted)})")

    # ── 8. 输出 multimodal_feature_info.json ────────────────────────────
    logger("[build_multimodal] 输出 multimodal_feature_info.json...")

    excluded_cols_applied = {
        "excluded_all_minus_one_cols": excluded_all_minus_one,
        "excluded_all_null_cols": excluded_all_null,
        "excluded_placeholder_cols": excluded_placeholder,
        "excluded_all_zero_cols": flagged_zero_cols,
        "excluded_constant_cols": flagged_constant_cols,
        "excluded_id_label_split_interaction": list(exclude_cols),
    }

    feature_info: dict[str, Any] = {
        "label_col": "label",
        "label_definition": tabular_feature_info.get(
            "label_definition",
            "流程验证伪标签: interaction_score >= threshold (继承自 tabular)",
        ),
        "train_size": len(train_vids_sorted),
        "eval_size": len(eval_vids_sorted),
        # text
        "text_feature_method": text_info.get("method", "unknown"),
        "text_dim": actual_text_dim,
        "text_source_fields": {
            "video_detail": ["caption", "desc"],
            "chapter": ["chapter_abstract", "chapter_desc"],
            "comment": ["comment_text"],
            "hashtag": ["hashtag_name"],
            "music": ["music_title", "music_author"],
            "author": ["signature"],
        },
        "text_vectorizer_info": text_info,
        # visual
        "visual_feature_method": "media_metadata_only",
        "visual_dim": vis_info["visual_dim"],
        "visual_feature_columns": vis_cols,
        # structured
        "structured_feature_method": "tabular_numeric_scaled",
        "structured_dim": structured_dim,
        "structured_feature_columns": structured_cols,
        "structured_imputation_values": struct_imputation,
        "structured_scaler_mean": struct_mean,
        "structured_scaler_scale": struct_std,
        # excluded
        "excluded_cols_applied": excluded_cols_applied,
        # sources
        "source_tables_used": list(SAMPLE_TABLES.keys()),
        "join_keys_used": ["video_id", "author_id"],
        "train_npz_path": str(train_npz_path),
        "eval_npz_path": str(eval_npz_path),
        # flags
        "no_image_download": True,
        "no_external_api": True,
        "no_large_pretrained_model": True,
        # misc
        "warnings": warnings,
        "notes": notes + [
            "当前多模态数据集仅基于 sample0427 构建，仅用于流程级验证。",
            "标签为 interaction_score 伪标签，不代表真实曝光/点击/转化目标。",
            "text_features 基于 TF-IDF+SVD 或 HashingVectorizer，非大型语言模型。",
            "visual_features 仅使用媒体元信息（URL 存在性、尺寸、水印），未下载图片。",
            "structured_features 复用 tabular 数值+文本统计特征，经 z-score 标准化。",
            "未下载图片，未调用外部 API，未使用大型预训练模型。",
        ],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    info_json_path = multimodal_dir / "multimodal_feature_info.json"
    with open(info_json_path, "w", encoding="utf-8") as f:
        json.dump(feature_info, f, ensure_ascii=False, indent=2)
    logger(f"  feature_info: {info_json_path}")

    # ── 9. 输出 multimodal_dataset_report.json ─────────────────────────
    logger("[build_multimodal] 输出 multimodal_dataset_report.json...")

    # label distribution
    train_labels = train_data["label"]
    eval_labels_eval = eval_data["label"]
    train_pos = int((train_labels == 1).sum())
    train_neg = int((train_labels == 0).sum())
    eval_pos = int((eval_labels_eval == 1).sum())
    eval_neg = int((eval_labels_eval == 0).sum())

    # missing counts
    train_text_empty = int((train_text_vec.sum(axis=1) == 0).sum())
    eval_text_empty = int((eval_text_vec.sum(axis=1) == 0).sum())
    train_vis_nan_count = int(np.isnan(train_vis).any(axis=1).sum())
    eval_vis_nan_count = int(np.isnan(eval_vis).any(axis=1).sum())

    report: dict[str, Any] = {
        "input_paths": {
            "sample0427_dir": str(sample0427_dir),
            "tabular_train": str(train_path),
            "tabular_eval": str(eval_path),
            "tabular_feature_info": str(feature_info_path),
            "quality_check": str(quality_check_path),
        },
        "output_paths": {
            "train_npz": str(train_npz_path),
            "eval_npz": str(eval_npz_path),
            "feature_info": str(info_json_path),
        },
        "train_size": len(train_vids_sorted),
        "eval_size": len(eval_vids_sorted),
        "label_distribution_train": {"pos": train_pos, "neg": train_neg},
        "label_distribution_eval": {"pos": eval_pos, "neg": eval_neg},
        "text_feature_shape_train": list(train_text_vec.shape),
        "text_feature_shape_eval": list(eval_text_vec.shape),
        "visual_feature_shape_train": list(train_vis.shape),
        "visual_feature_shape_eval": list(eval_vis.shape),
        "structured_feature_shape_train": list(train_struct_scaled.shape),
        "structured_feature_shape_eval": list(eval_struct_scaled.shape),
        "missing_text_count_train": train_text_empty,
        "missing_text_count_eval": eval_text_empty,
        "missing_visual_count_train": train_vis_nan_count,
        "missing_visual_count_eval": eval_vis_nan_count,
        "used_fallback_method": used_fallback,
        "no_image_download_confirmed": True,
        "no_external_api_confirmed": True,
        "no_large_pretrained_model_confirmed": True,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "warnings": warnings,
        "notes": [
            "当前多模态数据集仅用于流程级验证，不表示正式推荐系统效果。",
            "text_features 基于 combined_text（caption + desc + chapter + comment + hashtag + music + signature）。",
            "visual_features 仅使用 raw_video_media 元信息，未下载任何图片。",
            "structured_features 从 tabular 数值特征筛选，经 z-score 标准化。",
        ],
    }

    report_json_path = data_check_dir / "multimodal_dataset_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger(f"  report: {report_json_path}")

    # ── 10. 输出 multimodal_dataset_preview.csv ─────────────────────────
    logger("[build_multimodal] 输出 multimodal_dataset_preview.csv...")

    preview_rows: list[dict[str, Any]] = []
    for split_name, vid_list, tvec, vvec, svec in [
        ("train", train_vids_sorted, train_text_vec, train_vis, train_struct_scaled),
        ("eval", eval_vids_sorted, eval_text_vec, eval_vis, eval_struct_scaled),
    ]:
        for i, vid in enumerate(vid_list):
            tab_row = tabular_all[tabular_all["video_id"] == vid]
            if tab_row.empty:
                continue
            preview_rows.append({
                "sample_id": int(tab_row.iloc[0].get("sample_id", vid)),
                "video_id": int(vid),
                "author_id": str(tab_row.iloc[0].get("author_id", "0")),
                "label": int(tab_row.iloc[0].get("label", -1)),
                "split": split_name,
                "combined_text_length": int(len(str(text_train["combined_text"].iloc[i])))
                if split_name == "train" and i < len(text_train)
                else int(len(str(text_eval["combined_text"].iloc[i])))
                if split_name == "eval" and i < len(text_eval)
                else 0,
                "visual_non_null_count": int(np.sum(vvec[i] != 0)) if i < len(vvec) else 0,
                "structured_non_null_count": int(np.sum(~np.isnan(svec[i]))) if i < len(svec) else 0,
            })

    preview_df = pd.DataFrame(preview_rows).head(20)
    preview_csv_path = data_check_dir / "multimodal_dataset_preview.csv"
    preview_df.to_csv(preview_csv_path, index=False, encoding="utf-8-sig")
    logger(f"  preview: {preview_csv_path} ({len(preview_df)} 行)")

    # ── 11. 摘要 ─────────────────────────────────────────────────────────
    logger("")
    logger("=" * 60)
    logger("多模态数据集构建完成")
    logger("=" * 60)
    logger(f"  Train 样本: {len(train_vids_sorted)}")
    logger(f"  Eval 样本: {len(eval_vids_sorted)}")
    logger(f"  text_dim: {actual_text_dim}")
    logger(f"  visual_dim: {vis_info['visual_dim']}")
    logger(f"  structured_dim: {structured_dim}")
    logger(f"  文本方法: {text_info.get('method', 'unknown')}")
    logger(f"  视觉方法: media_metadata_only (未下载图片)")
    logger(f"  结构化方法: tabular_numeric_scaled ({len(structured_cols)} 列)")
    logger(f"  使用 fallback: {used_fallback}")
    if warnings:
        logger(f"  Warnings: {warnings}")
    logger(f"  Train npz: {train_npz_path}")
    logger(f"  Eval npz: {eval_npz_path}")
    logger(f"  Feature info: {info_json_path}")
    logger(f"  Report: {report_json_path}")
    logger(f"  Preview: {preview_csv_path}")
    logger("=" * 60)


if __name__ == "__main__":
    main()