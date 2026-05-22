"""Multimodal 数据集构建

基于真实网页端 raw 数据，结合 tabular train/val/test split，
构建 text / visual / structured 三模态 npz 输入文件。
数据集名称、路径、表名由配置文件驱动。

用法:
    python src/data/build_multimodal_real_raw.py --config configs/multimodal/multimodal_real_raw_5000.yaml

输出:
    data/multimodal/<dataset_name>/
        multimodal_train.npz
        multimodal_val.npz
        multimodal_test.npz
        multimodal_feature_info.json
    outputs/data_check/<dataset_name>/
        multimodal_dataset_report.json
        multimodal_dataset_preview.csv
        multimodal_leakage_check_report.json
"""

from __future__ import annotations

import argparse
import json
import pickle
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
    TEXT_PROFILES,
    build_combined_text,
    build_fieldwise_texts,
    fit_fieldwise_text_vectorizers,
    fit_text_vectorizer,
    transform_fieldwise_text,
    transform_text,
)
from src.features.image_features import build_visual_features  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import read_csv_safe  # noqa: E402

logger = print  # 轻量日志


def load_raw_table(raw_dir: Path, filename: str, table_label: str) -> pd.DataFrame:
    """加载单张 raw 表，空表返回空 DataFrame。"""
    path = raw_dir / filename
    if not path.exists():
        logger(f"  [WARN] 文件不存在: {path} → 返回空 DataFrame")
        return pd.DataFrame()
    if path.stat().st_size == 0:
        logger(f"  [WARN] 文件为空: {path} → 返回空 DataFrame")
        return pd.DataFrame()
    try:
        df, enc = read_csv_safe(str(path))
    except pd.errors.EmptyDataError:
        logger(f"  [WARN] 文件无解析列: {path} → 返回空 DataFrame")
        return pd.DataFrame()
    logger(f"  [{table_label}] {len(df)} 行 x {len(df.columns)} 列 (编码: {enc})")
    return df


def validate_multimodal_leakage(
    structured_cols: list[str],
    text_source_fields: dict[str, list[str]],
    leakage_cfg: dict,
    dataset_name: str,
) -> dict:
    """验证多模态特征中不包含泄漏字段。

    Checks:
      1. excluded_label_source_features 不在 structured_features 中
      2. interaction_score 不在 structured_features 中
      3. excluded_label_source_features 不在 text_source_fields 中

    Returns:
        leakage_check_report dict
    """
    excluded = leakage_cfg.get("excluded_label_source_features", [])
    audit_id_cols = leakage_cfg.get("audit_id_cols", [])

    errors: list[str] = []
    warnings: list[str] = []

    # Check 1: excluded_label_source_features in structured_cols
    found_in_struct = [f for f in excluded if f in structured_cols]
    if found_in_struct:
        errors.append(
            f"泄漏字段 {found_in_struct} 出现在 structured_features 中"
        )

    # Check 2: interaction_score in structured_cols
    if "interaction_score" in structured_cols:
        errors.append("interaction_score 出现在 structured_features 中")

    # Check 3: audit_id_cols (除 interaction_score 外) in structured_cols
    audit_in_struct = [
        c for c in audit_id_cols if c != "interaction_score" and c in structured_cols
    ]
    if audit_in_struct:
        errors.append(f"审计列 {audit_in_struct} 出现在 structured_features 中")

    # Check 4: excluded_label_source_features in text_source_fields
    found_in_text = {}
    for field in excluded:
        sources = []
        for table, cols in text_source_fields.items():
            if field in cols:
                sources.append(f"{table}.{field}")
        if sources:
            found_in_text[field] = sources
    if found_in_text:
        errors.append(
            f"泄漏字段出现在 text 源字段中: {found_in_text}"
        )

    report = {
        "dataset_name": dataset_name,
        "leakage_check_passed": len(errors) == 0,
        "excluded_label_source_features": {
            "defined": excluded,
            "found_in_structured_features": found_in_struct,
            "found_in_text_sources": found_in_text,
            "status": "ERROR" if found_in_struct or found_in_text else "OK",
        },
        "interaction_score_in_structured_features": {
            "found": "interaction_score" in structured_cols,
            "status": "ERROR" if "interaction_score" in structured_cols else "OK",
        },
        "audit_id_cols_in_structured_features": {
            "defined": audit_id_cols,
            "found_in_structured_features": audit_in_struct,
            "status": "ERROR" if audit_in_struct else "OK",
        },
        "structured_feature_columns": structured_cols,
        "text_source_fields": text_source_fields,
        "errors": errors,
        "warnings": warnings,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal 数据集构建")
    parser.add_argument(
        "--config",
        type=str,
        default=str(_PROJECT_ROOT / "configs/multimodal/multimodal_real_raw_5000.yaml"),
        help="配置文件路径",
    )
    parser.add_argument(
        "--text-profile",
        type=str,
        default=None,
        help="text_profile 名称（覆盖配置文件中的值），可选: merged_text_v1 / fieldwise_text_v1 / fieldwise_text_v2",
    )
    args = parser.parse_args()

    # ── 1. 加载配置 ─────────────────────────────────────────────────────
    logger(f"[build_multimodal_real_raw] 加载配置: {args.config}")
    config = load_config(args.config)

    project_root = _PROJECT_ROOT
    raw_root = project_root / config["raw_data_root"]
    tabular_dir = project_root / config["tabular_dir"]
    multimodal_dir = project_root / config["multimodal_output_dir"]
    data_check_dir = project_root / config["data_check_output_dir"]

    multimodal_dir.mkdir(parents=True, exist_ok=True)
    data_check_dir.mkdir(parents=True, exist_ok=True)

    text_dim = config.get("text_dim", 32)
    random_seed = config.get("random_seed", 2026)
    leakage_cfg = config.get("leakage_control", {})

    # ── text_profile 解析 ──────────────────────────────────────
    text_profile_name = args.text_profile or config.get("text_profile", "merged_text_v1")
    if text_profile_name not in TEXT_PROFILES:
        logger(f"[ERROR] 未知 text_profile: {text_profile_name}")
        logger(f"  可选: {list(TEXT_PROFILES.keys())}")
        sys.exit(1)
    text_profile = TEXT_PROFILES[text_profile_name]
    logger(f"  text_profile: {text_profile_name} (mode={text_profile['mode']}, total_dim={text_profile['total_dim']})")

    # 使用 text_profile 的 total_dim 覆盖 config 的 text_dim（如果 profile 有定义）
    # 使 merged 模式能通过 text_profile 控制总维度，而非只依赖 config 中的 text_dim
    if text_profile.get("total_dim") and text_profile["total_dim"] != text_dim:
        logger(f"  text_dim 从配置值 {text_dim} 更新为 text_profile 的 {text_profile['total_dim']}")
        text_dim = text_profile["total_dim"]

    notes: list[str] = []
    warnings: list[str] = []
    used_fallback = False

    tables_cfg = config["tables"]

    # ── 2. 加载 tabular train / val / test 基准 ─────────────────────────
    logger("[build_multimodal_real_raw] 加载 tabular train/val/test 基准...")

    train_path = tabular_dir / config["tabular_train"]
    val_path = tabular_dir / config["tabular_val"]
    test_path = tabular_dir / config["tabular_test"]
    feature_info_path = tabular_dir / config["tabular_feature_info"]

    for p in [train_path, val_path, test_path, feature_info_path]:
        if not p.exists():
            logger(f"[ERROR] 文件不存在: {p}")
            sys.exit(1)

    tabular_train = pd.read_csv(train_path, encoding="utf-8-sig")
    tabular_val = pd.read_csv(val_path, encoding="utf-8-sig")
    tabular_test = pd.read_csv(test_path, encoding="utf-8-sig")
    tabular_all = pd.concat([tabular_train, tabular_val, tabular_test], ignore_index=True)

    train_ids = set(tabular_train["video_id"].unique())
    val_ids = set(tabular_val["video_id"].unique())
    test_ids = set(tabular_test["video_id"].unique())

    # 验证三路互斥
    assert len(train_ids & val_ids) == 0, "train/val video_id 重叠"
    assert len(train_ids & test_ids) == 0, "train/test video_id 重叠"
    assert len(val_ids & test_ids) == 0, "val/test video_id 重叠"

    logger(f"  Train 样本: {len(tabular_train)} (video_ids: {len(train_ids)})")
    logger(f"  Val   样本: {len(tabular_val)} (video_ids: {len(val_ids)})")
    logger(f"  Test  样本: {len(tabular_test)} (video_ids: {len(test_ids)})")

    # 加载 tabular_feature_info
    with open(feature_info_path, "r", encoding="utf-8") as f:
        tabular_feature_info = json.load(f)

    label_definition = tabular_feature_info.get(
        "label_definition",
        "离线实验伪标签: interaction_score >= P60 (18147.80), 继承自 tabular",
    )

    # ── 3. 加载原始表 ────────────────────────────────────
    logger(f"[build_multimodal_real_raw] 加载 {config['dataset_name']} 原始表...")

    video_detail = load_raw_table(raw_root, tables_cfg["video_detail"], "video_detail")
    author = load_raw_table(raw_root, tables_cfg["author"], "author")
    music = load_raw_table(raw_root, tables_cfg["music"], "music")
    hashtag = load_raw_table(raw_root, tables_cfg["hashtag"], "hashtag")
    media = load_raw_table(raw_root, tables_cfg["video_media"], "video_media")
    comment = load_raw_table(raw_root, tables_cfg["comment"], "comment")
    chapter = load_raw_table(raw_root, tables_cfg["chapter"], "chapter")
    video_tag = load_raw_table(raw_root, tables_cfg["video_tag"], "video_tag")

    if chapter.empty:
        logger("  [INFO] raw_chapter 为空表，不参与文本特征")
    if video_tag.empty:
        logger("  [INFO] raw_video_tag 为空表，不参与文本特征")

    # ── 4. 构建 Text Features ────────────────────────────────────────────
    logger("[build_multimodal_real_raw] 构建 text_features...")

    text_profile_mode = text_profile["mode"]
    field_vectorizers_dir = multimodal_dir / "field_vectorizers"

    if text_profile_mode == "merged_tfidf_svd":
        # ── 4M. Merged 模式（当前默认方案） ─────────────────────────────
        logger("  使用 merged_text_v1 模式（所有字段合并 → TF-IDF → SVD）")

        text_df = build_combined_text(
            video_detail=video_detail,
            chapter=chapter if not chapter.empty else None,
            comment=comment if not comment.empty else None,
            hashtag=hashtag if not hashtag.empty else None,
            music=music if not music.empty else None,
            author=author if not author.empty else None,
        )
        logger(f"  combined_text 覆盖 video_ids: {len(text_df)}")

        text_train = text_df[text_df["video_id"].isin(train_ids)].copy()
        text_val = text_df[text_df["video_id"].isin(val_ids)].copy()
        text_test = text_df[text_df["video_id"].isin(test_ids)].copy()

        for label, ids, subset in [
            ("train", train_ids, text_train),
            ("val", val_ids, text_val),
            ("test", test_ids, text_test),
        ]:
            missing = ids - set(subset["video_id"])
            if missing:
                logger(f"  [WARN] {label} 中 {len(missing)} 个 video_id 无文本 → 填空")
                for vid in missing:
                    subset = pd.concat(
                        [subset, pd.DataFrame({"video_id": [vid], "combined_text": [""]})],
                        ignore_index=True,
                    )
            if label == "train":
                text_train = subset.sort_values("video_id").reset_index(drop=True)
            elif label == "val":
                text_val = subset.sort_values("video_id").reset_index(drop=True)
            else:
                text_test = subset.sort_values("video_id").reset_index(drop=True)

        text_train = text_train.sort_values("video_id").reset_index(drop=True)
        text_val = text_val.sort_values("video_id").reset_index(drop=True)
        text_test = text_test.sort_values("video_id").reset_index(drop=True)
        logger(f"  text_train: {len(text_train)}, text_val: {len(text_val)}, text_test: {len(text_test)}")

        logger("  拟合 text vectorizer (TF-IDF + SVD)...")
        vectorizer, svd, text_info = fit_text_vectorizer(
            text_train["combined_text"], text_dim=text_dim, random_seed=random_seed
        )
        logger(f"  text_feature_method: {text_info.get('method', 'unknown')}")
        if text_info.get("tfidf_failed"):
            logger(f"  TF-IDF 失败: {text_info.get('tfidf_error')}")
        if text_info.get("method", "").startswith("stats_fallback"):
            used_fallback = True
            warnings.append("文本特征使用统计特征回退（非 TF-IDF/SVD）")

        train_text_vec = transform_text(
            text_train["combined_text"], vectorizer, svd, text_dim
        )
        val_text_vec = transform_text(
            text_val["combined_text"], vectorizer, svd, text_dim
        )
        test_text_vec = transform_text(
            text_test["combined_text"], vectorizer, svd, text_dim
        )
        actual_text_dim = train_text_vec.shape[1]

        text_info_for_feature_info = text_info

    elif text_profile_mode == "fieldwise_tfidf_svd":
        # ── 4F. Fieldwise 模式（每字段独立 TF-IDF + SVD → concat） ─────
        logger(f"  使用 fieldwise 模式 ({text_profile_name})：7 字段独立编码 → concat({text_profile['total_dim']})")

        field_texts = build_fieldwise_texts(
            video_detail=video_detail,
            comment=comment if not comment.empty else None,
            hashtag=hashtag if not hashtag.empty else None,
            music=music if not music.empty else None,
            author=author if not author.empty else None,
        )

        # Fit on train only
        logger("  拟合每字段 TF-IDF + SVD（fit on train only）...")
        f_vectorizers, f_svds, f_field_info = fit_fieldwise_text_vectorizers(
            field_texts=field_texts,
            profile=text_profile,
            train_video_ids=train_ids,
            random_seed=random_seed,
        )

        # 记录各字段 explained_variance
        for fn, info in f_field_info.items():
            evr = info.get("explained_variance_ratio", 0)
            eff = info.get("effective_samples", 0)
            skipped = info.get("skipped", False)
            status = "SKIPPED" if skipped else "OK"
            logger(f"    {fn}: dim={info['dim']}, effective={eff}, "
                   f"explained_var={evr:.4f}, status={status}")

        # Transform train/val/test
        train_vids_sorted = sorted(train_ids)
        val_vids_sorted = sorted(val_ids)
        test_vids_sorted = sorted(test_ids)

        train_text_vec = transform_fieldwise_text(
            field_texts, f_vectorizers, f_svds, text_profile, train_vids_sorted
        )
        val_text_vec = transform_fieldwise_text(
            field_texts, f_vectorizers, f_svds, text_profile, val_vids_sorted
        )
        test_text_vec = transform_fieldwise_text(
            field_texts, f_vectorizers, f_svds, text_profile, test_vids_sorted
        )
        actual_text_dim = train_text_vec.shape[1]

        # Save field vectorizers
        field_vectorizers_dir.mkdir(parents=True, exist_ok=True)
        for fn in f_vectorizers:
            v_path = field_vectorizers_dir / f"{fn}_vectorizer.pkl"
            with open(v_path, "wb") as f:
                pickle.dump(f_vectorizers[fn], f)
        for fn in f_svds:
            s_path = field_vectorizers_dir / f"{fn}_svd.pkl"
            with open(s_path, "wb") as f:
                pickle.dump(f_svds[fn], f)
        logger(f"  field vectorizers 已保存: {field_vectorizers_dir}")

        # 构建 text_info
        text_info = {
            "method": "fieldwise_tfidf_svd",
            "text_profile_name": text_profile_name,
            "total_dim": text_profile["total_dim"],
            "field_info": f_field_info,
        }

    else:
        logger(f"[ERROR] 未知 text_profile mode: {text_profile_mode}")
        sys.exit(1)

    logger(f"  text_features shape: train={train_text_vec.shape}, val={val_text_vec.shape}, test={test_text_vec.shape}")

    # ── 5. 构建 Visual Features ───────────────────────────────────────────
    logger("[build_multimodal_real_raw] 构建 visual_features (媒体元信息)...")
    all_video_ids = sorted(train_ids | val_ids | test_ids)
    vis_df, vis_cols, vis_info = build_visual_features(media, video_ids=all_video_ids)
    logger(f"  visual_dim: {vis_info['visual_dim']}, rows: {len(vis_df)}, cols: {vis_cols}")

    # 补齐缺失 video_id
    vis_by_vid = dict(zip(vis_df["video_id"].astype(np.int64), range(len(vis_df))))
    vis_np = np.zeros((len(all_video_ids), vis_info["visual_dim"]), dtype=np.float32)
    for i, vid in enumerate(all_video_ids):
        if vid in vis_by_vid:
            vis_np[i] = vis_df.iloc[vis_by_vid[vid]][vis_cols].values.astype(np.float32)

    # 分离 train/val/test
    train_vis = np.array([vis_np[all_video_ids.index(vid)] for vid in sorted(train_ids)])
    val_vis = np.array([vis_np[all_video_ids.index(vid)] for vid in sorted(val_ids)])
    test_vis = np.array([vis_np[all_video_ids.index(vid)] for vid in sorted(test_ids)])
    logger(f"  visual_features shape: train={train_vis.shape}, val={val_vis.shape}, test={test_vis.shape}")

    # 检查 / 填充 NaN
    for label, arr in [("train", train_vis), ("val", val_vis), ("test", test_vis)]:
        if np.isnan(arr).any():
            logger(f"  [WARN] visual_features ({label}) 中存在 NaN，已填充为 0")
            arr = np.nan_to_num(arr, nan=0.0)
    vis_has_nan = np.isnan(train_vis).any() or np.isnan(val_vis).any() or np.isnan(test_vis).any()
    if vis_has_nan:
        train_vis = np.nan_to_num(train_vis, nan=0.0)
        val_vis = np.nan_to_num(val_vis, nan=0.0)
        test_vis = np.nan_to_num(test_vis, nan=0.0)
        warnings.append("visual_features 中存在 NaN，已填充为 0")

    # ── 6. 构建 Structured Features ──────────────────────────────────────
    logger("[build_multimodal_real_raw] 构建 structured_features (复用 tabular 数值特征)...")

    # 6a. 确定可用数值列
    numeric_features = tabular_feature_info.get("numeric_features", [])
    text_stat_features = tabular_feature_info.get("text_stat_features", [])
    id_cols = tabular_feature_info.get("id_cols", ["sample_id", "video_id", "author_id"])
    exclude_cols = set(id_cols + ["label", "split", "interaction_score"])

    available_cols = [c for c in tabular_train.columns if c not in exclude_cols]
    structured_cols = [
        c for c in available_cols if c in numeric_features or c in text_stat_features
    ]
    logger(f"  structured 候选: {len(numeric_features)} numeric + {len(text_stat_features)} text_stat")
    logger(f"  排除 {len(exclude_cols)} 列, 最终使用 {len(structured_cols)} 列")

    # 6b. 提取 train/val/test 结构化矩阵
    train_struct_raw = tabular_train[structured_cols].copy().astype(np.float32)
    val_struct_raw = tabular_val[structured_cols].copy().astype(np.float32)
    test_struct_raw = tabular_test[structured_cols].copy().astype(np.float32)

    # 6c. 缺失值填充（train median → 全部三个 split）
    struct_imputation = {}
    for col in structured_cols:
        median_val = train_struct_raw[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        struct_imputation[col] = float(median_val)
        train_struct_raw[col] = train_struct_raw[col].fillna(median_val)
        val_struct_raw[col] = val_struct_raw[col].fillna(median_val)
        test_struct_raw[col] = test_struct_raw[col].fillna(median_val)

    # 6d. 标准化（z-score, fit on train only, 不污染 val/test）
    struct_mean = {}
    struct_std = {}
    train_struct_scaled = train_struct_raw.copy().values
    val_struct_scaled = val_struct_raw.copy().values
    test_struct_scaled = test_struct_raw.copy().values

    for i, col in enumerate(structured_cols):
        col_mean = float(train_struct_raw[col].mean())
        col_std = float(train_struct_raw[col].std())
        if col_std < 1e-8:
            col_std = 1.0
        struct_mean[col] = col_mean
        struct_std[col] = col_std
        train_struct_scaled[:, i] = (train_struct_raw[col].values - col_mean) / col_std
        val_struct_scaled[:, i] = (val_struct_raw[col].values - col_mean) / col_std
        test_struct_scaled[:, i] = (test_struct_raw[col].values - col_mean) / col_std

    structured_dim = train_struct_scaled.shape[1]
    logger(f"  structured_features shape: train={train_struct_scaled.shape}, val={val_struct_scaled.shape}, test={test_struct_scaled.shape}")

    # ── 6.5 构建 Categorical Features ──────────────────────────────────
    logger("[build_multimodal_real_raw] 构建 categorical_features...")

    categorical_features = tabular_feature_info.get("categorical_features", [])
    logger(f"  categorical_features 列表: {categorical_features}")

    if not categorical_features:
        logger("  [WARN] tabular_feature_info 中无 categorical_features")
        cat_train_arr = np.empty((len(train_ids), 0), dtype=np.int64)
        cat_val_arr = np.empty((len(val_ids), 0), dtype=np.int64)
        cat_test_arr = np.empty((len(test_ids), 0), dtype=np.int64)
        cat_vocabs = {}
        cat_embed_dims: list[list[int]] = []
    else:
        # 从训练集构建 vocab（与 DNNDataProcessor.fit() 一致）
        cat_vocabs: dict[str, dict] = {}
        cat_embed_dims: list[list[int]] = []
        for col in categorical_features:
            raw_series = tabular_train[col].fillna("__MISSING__")
            unique_vals = sorted(raw_series.unique().tolist())
            vocab: dict = {"__UNK__": 0}
            for i, v in enumerate(unique_vals):
                vocab[v] = i + 1  # 0 保留给 UNK
            # 确保 __MISSING__ 在 vocab 中
            if "__MISSING__" not in vocab and "__MISSING__" not in unique_vals:
                # 缺失值不在训练数据中，但可能出现在 val/test
                pass  # 缺失值会映射到 __UNK__ (0)
            cat_vocabs[col] = vocab
            embed_dim = min(16, max(4, int(len(vocab) ** 0.5) + 1))
            cat_embed_dims.append([len(vocab), embed_dim])
            logger(f"    {col}: vocab_size={len(vocab)}, embed_dim={embed_dim}")

        # 将原始值映射为索引
        def map_cat_to_indices(
            df: pd.DataFrame, cat_vocabs: dict, cat_features: list[str]
        ) -> np.ndarray:
            indices_list = []
            for col in cat_features:
                raw = df[col].fillna("__MISSING__").values
                indices = [cat_vocabs[col].get(v, 0) for v in raw]
                indices_list.append(indices)
            # (num_cat, N) → (N, num_cat)
            return np.array(indices_list, dtype=np.int64).T

        cat_train_arr = map_cat_to_indices(
            tabular_train.set_index("video_id").loc[sorted(train_ids)].reset_index(),
            cat_vocabs, categorical_features,
        )
        cat_val_arr = map_cat_to_indices(
            tabular_val.set_index("video_id").loc[sorted(val_ids)].reset_index(),
            cat_vocabs, categorical_features,
        )
        cat_test_arr = map_cat_to_indices(
            tabular_test.set_index("video_id").loc[sorted(test_ids)].reset_index(),
            cat_vocabs, categorical_features,
        )

        logger(f"  categorical_features shape: "
               f"train={cat_train_arr.shape}, val={cat_val_arr.shape}, test={cat_test_arr.shape}")
        # 检查 UNK 比例
        for label, arr in [("train", cat_train_arr), ("val", cat_val_arr), ("test", cat_test_arr)]:
            if arr.size > 0:
                unk_count = int((arr == 0).sum())
                unk_pct = unk_count / arr.size * 100
                if unk_pct > 0:
                    logger(f"  [INFO] {label} UNK 占比: {unk_pct:.1f}% ({unk_count}/{arr.size})")

    # ── 7. 对齐样本并保存 npz ───────────────────────────────────────────
    logger("[build_multimodal_real_raw] 对齐样本并保存 npz...")

    def build_npz(
        video_ids_sorted: list[int],
        text_vec: np.ndarray,
        vis_vec: np.ndarray,
        struct_vec: np.ndarray,
        cat_arr: np.ndarray,
        split_label: str,
    ) -> dict[str, Any]:
        """构建单个 npz 的内容字典。"""
        tabular_subset = tabular_all[tabular_all["video_id"].isin(video_ids_sorted)].copy()
        tabular_subset = tabular_subset.set_index("video_id").loc[video_ids_sorted].reset_index()

        labels = tabular_subset["label"].values.astype(np.float32)
        sample_ids = tabular_subset["sample_id"].values.astype(np.int64)
        video_ids_arr = tabular_subset["video_id"].values.astype(np.int64)
        author_ids = tabular_subset["author_id"].values.astype(str)

        result = {
            "sample_id": sample_ids,
            "video_id": video_ids_arr,
            "author_id": author_ids,
            "label": labels,
            "text_features": text_vec.astype(np.float32),
            "visual_features": vis_vec.astype(np.float32),
            "structured_features": struct_vec.astype(np.float32),
            "split": np.array([split_label] * len(video_ids_sorted), dtype=object),
        }
        if cat_arr.shape[1] > 0:
            result["categorical_features"] = cat_arr
        return result

    train_vids_sorted = sorted(train_ids)
    val_vids_sorted = sorted(val_ids)
    test_vids_sorted = sorted(test_ids)

    train_data = build_npz(train_vids_sorted, train_text_vec, train_vis, train_struct_scaled, cat_train_arr, "train")
    val_data = build_npz(val_vids_sorted, val_text_vec, val_vis, val_struct_scaled, cat_val_arr, "val")
    test_data = build_npz(test_vids_sorted, test_text_vec, test_vis, test_struct_scaled, cat_test_arr, "test")

    # 保存
    train_npz_path = multimodal_dir / "multimodal_train.npz"
    val_npz_path = multimodal_dir / "multimodal_val.npz"
    test_npz_path = multimodal_dir / "multimodal_test.npz"

    np.savez_compressed(train_npz_path, **train_data)
    np.savez_compressed(val_npz_path, **val_data)
    np.savez_compressed(test_npz_path, **test_data)
    logger(f"  Train npz: {train_npz_path} (samples={len(train_vids_sorted)})")
    logger(f"  Val npz: {val_npz_path} (samples={len(val_vids_sorted)})")
    logger(f"  Test npz: {test_npz_path} (samples={len(test_vids_sorted)})")

    # ── 8. 输出 multimodal_feature_info.json ─────────────────────────────
    logger("[build_multimodal_real_raw] 输出 multimodal_feature_info.json...")

    # 缺失计数
    train_labels_arr = train_data["label"]
    val_labels_arr = val_data["label"]
    test_labels_arr = test_data["label"]
    train_pos = int((train_labels_arr == 1).sum())
    train_neg = int((train_labels_arr == 0).sum())
    val_pos = int((val_labels_arr == 1).sum())
    val_neg = int((val_labels_arr == 0).sum())
    test_pos = int((test_labels_arr == 1).sum())
    test_neg = int((test_labels_arr == 0).sum())

    # 缺失文本 / 视觉计数
    train_text_empty = int((train_text_vec.sum(axis=1) == 0).sum())
    val_text_empty = int((val_text_vec.sum(axis=1) == 0).sum())
    test_text_empty = int((test_text_vec.sum(axis=1) == 0).sum())

    text_source_fields = {
        "video_detail": ["caption", "desc"],
        "comment": ["comment_text"],
        "hashtag": ["hashtag_name"],
        "music": ["music_title", "music_author"],
        "author": ["signature"],
    }

    # 构建 text_profile 元信息
    text_profile_info = {
        "name": text_profile_name,
        "mode": text_profile_mode,
        "total_dim": text_profile.get("total_dim", text_dim),
    }
    if text_profile_mode == "fieldwise_tfidf_svd":
        field_info_list = []
        for fc in text_profile.get("fields", []):
            fn = fc["name"]
            fi = text_info.get("field_info", {}).get(fn, {})
            field_info_list.append({
                "name": fn,
                "dim": fc["dim"],
                "max_features": fc.get("max_features", 5000),
                "svd_explained_variance_ratio": fi.get("explained_variance_ratio", 0),
                "effective_samples": fi.get("effective_samples", 0),
                "svd_reduced": fi.get("svd_reduced", False),
                "skipped": fi.get("skipped", False),
            })
        text_profile_info["fields"] = field_info_list

    feature_info: dict[str, Any] = {
        "dataset_name": config["dataset_name"],
        "label_col": "label",
        "label_definition": label_definition,
        "train_size": len(train_vids_sorted),
        "val_size": len(val_vids_sorted),
        "test_size": len(test_vids_sorted),
        "label_distribution_train": {"pos": train_pos, "neg": train_neg},
        "label_distribution_val": {"pos": val_pos, "neg": val_neg},
        "label_distribution_test": {"pos": test_pos, "neg": test_neg},
        # text
        "text_profile": text_profile_info,
        "text_feature_method": text_info.get("method", "unknown"),
        "text_dim": actual_text_dim,
        "text_source_fields": text_source_fields,
        "text_vectorizer_info": text_info,
        "text_empty_count_train": train_text_empty,
        "text_empty_count_val": val_text_empty,
        "text_empty_count_test": test_text_empty,
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
        # sources
        "source_tables_used": list(tables_cfg.keys()),
        "join_keys_used": ["video_id", "author_id"],
        "output_files": {
            "train_npz": str(train_npz_path),
            "val_npz": str(val_npz_path),
            "test_npz": str(test_npz_path),
        },
        # flags
        "no_image_download": True,
        "no_external_api": True,
        "no_large_pretrained_model": True,
        "fit_text_on_train_only": True,
        "fit_struct_scaler_on_train_only": True,
        # categorical
        "categorical_features_available": len(categorical_features) > 0,
        "categorical_features": categorical_features,
        "categorical_dim": len(categorical_features),
        "cat_vocabs": cat_vocabs,
        "cat_embed_dims": cat_embed_dims,
        "future_categorical_candidates": ["author_id", "music_id"],
        "categorical_vocab_source": "data/features/real_raw_5000/tabular_feature_info.json",
        # misc
        "warnings": warnings,
        "notes": notes
        + [
            f"本数据集基于 {config['dataset_name']} 真实网页端 raw 数据构建。",
            f"text_features 使用 {text_profile_name} 编码 ({text_profile_mode})，维度={actual_text_dim}。",
            "标签为 interaction_score 分位数伪标签，仅供离线多模型对比实验使用。",
            "visual_features 仅使用媒体元信息（URL 存在性、尺寸、水印），未下载图片。",
            "structured_features 复用 tabular 数值+文本统计特征，经 z-score 标准化（fit on train only）。",
            "未下载图片，未调用外部 API，未使用大型预训练模型。",
            "raw_video_tag 为空表，未参与文本特征。",
            "raw_chapter 为空表，未参与文本特征。",
        ],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    info_json_path = multimodal_dir / "multimodal_feature_info.json"
    with open(info_json_path, "w", encoding="utf-8") as f:
        json.dump(feature_info, f, ensure_ascii=False, indent=2)
    logger(f"  feature_info: {info_json_path}")

    # ── 8.5 泄漏验证 ─────────────────────────────────────────────────────
    logger("[build_multimodal_real_raw] 执行多模态泄漏检查...")

    leakage_report = validate_multimodal_leakage(
        structured_cols=structured_cols,
        text_source_fields=text_source_fields,
        leakage_cfg=leakage_cfg,
        dataset_name=config["dataset_name"],
    )

    if leakage_report["leakage_check_passed"]:
        logger("  泄漏检查: 通过 OK")
    else:
        logger("  泄漏检查: 失败 FAILED")
        for err in leakage_report["errors"]:
            logger(f"    ERROR: {err}")
        if leakage_cfg.get("enforce_strict_deletion", False):
            logger("  enforce_strict_deletion=True → 终止")
            sys.exit(1)

    leakage_json_path = data_check_dir / "multimodal_leakage_check_report.json"
    with open(leakage_json_path, "w", encoding="utf-8") as f:
        json.dump(leakage_report, f, ensure_ascii=False, indent=2)
    logger(f"  泄漏检查报告: {leakage_json_path}")

    # ── 9. 输出 multimodal_dataset_report.json ──────────────────────────
    logger("[build_multimodal_real_raw] 输出 multimodal_dataset_report.json...")

    report: dict[str, Any] = {
        "input_paths": {
            "raw_data_root": str(raw_root),
            "tabular_train": str(train_path),
            "tabular_val": str(val_path),
            "tabular_test": str(test_path),
            "tabular_feature_info": str(feature_info_path),
        },
        "output_paths": {
            "train_npz": str(train_npz_path),
            "val_npz": str(val_npz_path),
            "test_npz": str(test_npz_path),
            "feature_info": str(info_json_path),
        },
        "train_size": len(train_vids_sorted),
        "val_size": len(val_vids_sorted),
        "test_size": len(test_vids_sorted),
        "label_distribution_train": {"pos": train_pos, "neg": train_neg},
        "label_distribution_val": {"pos": val_pos, "neg": val_neg},
        "label_distribution_test": {"pos": test_pos, "neg": test_neg},
        "text_feature_shape_train": list(train_text_vec.shape),
        "text_feature_shape_val": list(val_text_vec.shape),
        "text_feature_shape_test": list(test_text_vec.shape),
        "visual_feature_shape_train": list(train_vis.shape),
        "visual_feature_shape_val": list(val_vis.shape),
        "visual_feature_shape_test": list(test_vis.shape),
        "structured_feature_shape_train": list(train_struct_scaled.shape),
        "structured_feature_shape_val": list(val_struct_scaled.shape),
        "structured_feature_shape_test": list(test_struct_scaled.shape),
        "categorical_features_available": len(categorical_features) > 0,
        "categorical_features": categorical_features,
        "categorical_shape_train": list(cat_train_arr.shape),
        "categorical_shape_val": list(cat_val_arr.shape),
        "categorical_shape_test": list(cat_test_arr.shape),
        "missing_text_count_train": train_text_empty,
        "missing_text_count_val": val_text_empty,
        "missing_text_count_test": test_text_empty,
        "used_fallback_method": used_fallback,
        "no_image_download_confirmed": True,
        "no_external_api_confirmed": True,
        "no_large_pretrained_model_confirmed": True,
        "leakage_check": {
            "passed": leakage_report["leakage_check_passed"],
            "report_path": str(leakage_json_path),
        },
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "warnings": warnings,
        "notes": [
            f"当前多模态数据集基于 {config['dataset_name']} 真实网页端 raw 数据构建。",
            f"text_features 使用 {text_profile_name} 编码 ({text_profile_mode})，维度={actual_text_dim}。",
            "visual_features 仅使用 raw_video_media 元信息，未下载任何图片。",
            "structured_features 从 tabular 数值特征筛选，经 z-score 标准化（fit on train only）。",
            "split 与 tabular 完全一致。",
        ],
    }

    report_json_path = data_check_dir / "multimodal_dataset_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger(f"  report: {report_json_path}")

    # ── 10. 输出 multimodal_dataset_preview.csv ─────────────────────────
    logger("[build_multimodal_real_raw] 输出 multimodal_dataset_preview.csv...")

    # fieldwise 模式下 text_train/val/test 未定义，用 None 占位
    text_train_ref = locals().get("text_train", None)
    text_val_ref = locals().get("text_val", None)
    text_test_ref = locals().get("text_test", None)

    preview_rows: list[dict[str, Any]] = []
    for split_name, vid_list, tvec, vvec, svec, tdf in [
        ("train", train_vids_sorted, train_text_vec, train_vis, train_struct_scaled, text_train_ref),
        ("val", val_vids_sorted, val_text_vec, val_vis, val_struct_scaled, text_val_ref),
        ("test", test_vids_sorted, test_text_vec, test_vis, test_struct_scaled, text_test_ref),
    ]:
        for i, vid in enumerate(vid_list):
            tab_row = tabular_all[tabular_all["video_id"] == vid]
            if tab_row.empty:
                continue
            combined_text_len = 0
            if tdf is not None and i < len(tdf):
                try:
                    combined_text_len = int(len(str(tdf.iloc[i]["combined_text"])))
                except (KeyError, TypeError):
                    combined_text_len = 0
            preview_rows.append({
                "sample_id": int(tab_row.iloc[0].get("sample_id", vid)),
                "video_id": int(vid),
                "author_id": str(tab_row.iloc[0].get("author_id", "0")),
                "label": int(tab_row.iloc[0].get("label", -1)),
                "split": split_name,
                "combined_text_length": combined_text_len,
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
    logger(f"Multimodal 数据集构建完成 — {config['dataset_name']}")
    logger("=" * 60)
    logger(f"  Train 样本: {len(train_vids_sorted)}")
    logger(f"  Val   样本: {len(val_vids_sorted)}")
    logger(f"  Test  样本: {len(test_vids_sorted)}")
    logger(f"  text_profile: {text_profile_name}")
    logger(f"  text_dim: {actual_text_dim}")
    logger(f"  visual_dim: {vis_info['visual_dim']}")
    logger(f"  structured_dim: {structured_dim}")
    cat_dim_str = f"categorical_dim={len(categorical_features)}" if categorical_features else "no categorical"
    logger(f"  categorical: {cat_dim_str}")
    logger(f"  文本方法: {text_info.get('method', 'unknown')}")
    logger(f"  视觉方法: media_metadata_only (未下载图片)")
    logger(f"  结构化方法: tabular_numeric_scaled ({len(structured_cols)} 列)")
    logger(f"  使用 fallback: {used_fallback}")
    if warnings:
        logger(f"  Warnings: {warnings}")
    logger(f"  Train npz: {train_npz_path}")
    logger(f"  Val npz: {val_npz_path}")
    logger(f"  Test npz: {test_npz_path}")
    logger(f"  泄漏检查: {'通过 OK' if leakage_report['leakage_check_passed'] else '失败 FAILED'}")
    logger(f"  Feature info: {info_json_path}")
    logger(f"  Report: {report_json_path}")
    logger(f"  Preview: {preview_csv_path}")
    logger(f"  Leakage report: {leakage_json_path}")
    logger("=" * 60)


if __name__ == "__main__":
    main()