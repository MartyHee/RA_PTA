"""视觉/媒体元信息特征构建

当前仅使用本地 CSV 中已有的媒体元信息和 URL 存在性特征。
不下载图片，不调用外部 API，不使用大型预训练模型。
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def build_visual_features(
    media_df: pd.DataFrame,
    video_ids: list[int] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """从 raw_video_media 表提取视觉/媒体元信息特征。

    特征列表：
      1. has_origin_cover       — origin_cover_uri 是否存在
      2. has_dynamic_cover      — dynamic_cover_uri 是否存在
      3. origin_cover_width     — 原始封面宽度（NaN 填 -1）
      4. origin_cover_height    — 原始封面高度（NaN 填 -1）
      5. origin_cover_wh_ratio  — 宽高比（分母<=0 时置 0）
      6. has_watermark          — 是否有水印 (bool → float)
      7. has_cover_url          — cover_url_list 是否有值
      8. has_origin_cover_url   — origin_cover_url_list 是否有值
      9. has_dynamic_cover_url  — dynamic_cover_url_list 是否有值
     10. origin_cover_url_count — origin_cover_url_list 中 URL 数量

    Args:
        media_df: raw_video_media DataFrame
        video_ids: 可选的 video_id 列表，用于筛选和对齐

    Returns:
        (feature_df, column_names, info_dict)
    """
    info: dict[str, Any] = {
        "method": "media_metadata_only",
        "no_image_download": True,
        "no_external_api": True,
        "no_large_pretrained_model": True,
    }

    df = media_df.copy()

    if video_ids is not None:
        df = df[df["video_id"].isin(video_ids)].copy()

    df = df.set_index("video_id")

    # 1. has_origin_cover
    df["has_origin_cover"] = df["origin_cover_uri"].notna().astype(np.float32)

    # 2. has_dynamic_cover
    df["has_dynamic_cover"] = df["dynamic_cover_uri"].notna().astype(np.float32)

    # 3. origin_cover_width
    df["origin_cover_width"] = pd.to_numeric(
        df["origin_cover_width"], errors="coerce"
    ).fillna(-1).astype(np.float32)

    # 4. origin_cover_height
    df["origin_cover_height"] = pd.to_numeric(
        df["origin_cover_height"], errors="coerce"
    ).fillna(-1).astype(np.float32)

    # 5. origin_cover_wh_ratio
    ratio = np.where(
        df["origin_cover_height"].values > 0,
        df["origin_cover_width"].values / df["origin_cover_height"].values,
        0.0,
    )
    df["origin_cover_wh_ratio"] = ratio.astype(np.float32)

    # 6. has_watermark
    df["has_watermark"] = (
        df["has_watermark"].fillna(False).astype(bool).astype(np.float32)
    )

    # 7-10. URL 存在性特征
    def _parse_url_count(val: Any) -> int:
        """安全解析 URL 列表字段，返回 URL 数量。"""
        if pd.isna(val):
            return 0
        if not isinstance(val, str):
            return 1 if val else 0
        s = val.strip()
        if s == "[]" or s == "":
            return 0
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                return len(parsed) if isinstance(parsed, list) else 1
            except (json.JSONDecodeError, TypeError):
                return 1
        return 1

    df["has_cover_url"] = df["cover_url_list"].apply(
        lambda x: 1.0 if _parse_url_count(x) > 0 else 0.0
    )
    df["has_origin_cover_url"] = df["origin_cover_url_list"].apply(
        lambda x: 1.0 if _parse_url_count(x) > 0 else 0.0
    )
    df["has_dynamic_cover_url"] = df["dynamic_cover_url_list"].apply(
        lambda x: 1.0 if _parse_url_count(x) > 0 else 0.0
    )
    df["origin_cover_url_count"] = df["origin_cover_url_list"].apply(
        lambda x: float(_parse_url_count(x))
    )

    # ── 扩展特征（向后兼容，缺列则跳过）────────────────────────────────
    # 11. cover_url_count（数值计数，非仅二值）
    if "cover_url_list" in df.columns:
        df["cover_url_count"] = df["cover_url_list"].apply(
            lambda x: float(_parse_url_count(x))
        )

    # 12. dynamic_cover_url_count
    if "dynamic_cover_url_list" in df.columns:
        df["dynamic_cover_url_count"] = df["dynamic_cover_url_list"].apply(
            lambda x: float(_parse_url_count(x))
        )

    # 13-14. video_width / video_height
    for col in ["video_width", "video_height"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32)

    # 15. video_wh_ratio
    if "video_width" in df.columns and "video_height" in df.columns:
        v_ratio = np.where(
            df["video_height"].values > 0,
            df["video_width"].values / df["video_height"].values,
            0.0,
        )
        df["video_wh_ratio"] = v_ratio.astype(np.float32)

    # 16-17. dynamic_cover_width / dynamic_cover_height
    for col in ["dynamic_cover_width", "dynamic_cover_height"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1.0).astype(np.float32)

    # 18. dynamic_cover_wh_ratio
    if "dynamic_cover_width" in df.columns and "dynamic_cover_height" in df.columns:
        d_ratio = np.where(
            df["dynamic_cover_height"].values > 0,
            df["dynamic_cover_width"].values / df["dynamic_cover_height"].values,
            0.0,
        )
        df["dynamic_cover_wh_ratio"] = d_ratio.astype(np.float32)

    feature_cols = [
        "has_origin_cover",
        "has_dynamic_cover",
        "origin_cover_width",
        "origin_cover_height",
        "origin_cover_wh_ratio",
        "has_watermark",
        "has_cover_url",
        "has_origin_cover_url",
        "has_dynamic_cover_url",
        "origin_cover_url_count",
    ]

    # 扩展特征列（仅包含实际存在的列）
    extra_cols = [
        "cover_url_count", "dynamic_cover_url_count",
        "video_width", "video_height", "video_wh_ratio",
        "dynamic_cover_width", "dynamic_cover_height", "dynamic_cover_wh_ratio",
    ]
    for col in extra_cols:
        if col in df.columns:
            feature_cols.append(col)

    result = df[feature_cols].reset_index()  # video_id 列恢复为数据列
    info["visual_dim"] = len(feature_cols)
    info["feature_columns"] = feature_cols
    info["rows"] = len(result)

    return result, feature_cols, info