"""轻量文本特征构建：TF-IDF + TruncatedSVD / HashingVectorizer 回退

当前仅使用本地已有文本字段，不调用外部 API，不联网下载资源。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# ── scikit-learn 可用 ────────────────────────────────────────────────────
_SKLEARN_AVAILABLE = True
try:
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
except ImportError:
    _SKLEARN_AVAILABLE = False


def build_combined_text(
    video_detail: pd.DataFrame,
    chapter: pd.DataFrame | None = None,
    comment: pd.DataFrame | None = None,
    hashtag: pd.DataFrame | None = None,
    music: pd.DataFrame | None = None,
    author: pd.DataFrame | None = None,
    video_id_col: str = "video_id",
) -> pd.DataFrame:
    """将多张表的文本字段按 video_id 聚合，拼接为 combined_text。

    Args:
        video_detail: raw_video_detail DataFrame（必需，含 caption/desc）
        chapter: raw_chapter DataFrame（可选）
        comment: raw_comment DataFrame（可选）
        hashtag: raw_hashtag DataFrame（可选）
        music: raw_music DataFrame（可选）
        author: raw_author DataFrame（可选）
        video_id_col: video_id 列名

    Returns:
        每 video_id 一行的 DataFrame，含 combined_text 列
    """
    texts: dict[int, list[str]] = {}

    def _add(vid: Any, text: str | None) -> None:
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            return
        vid_int = int(vid)
        if vid_int not in texts:
            texts[vid_int] = []
        texts[vid_int].append(text.strip())

    # 1. video_detail caption / desc
    for _, row in video_detail.iterrows():
        vid = row[video_id_col]
        _add(vid, row.get("caption"))
        _add(vid, row.get("desc"))

    # 2. chapter: chapter_abstract / chapter_desc
    if chapter is not None and not chapter.empty:
        grp = chapter.groupby(video_id_col)
        for vid, group in grp:
            for _, r in group.iterrows():
                _add(vid, r.get("chapter_abstract"))
                _add(vid, r.get("chapter_desc"))

    # 3. comment: comment_text
    if comment is not None and not comment.empty:
        grp = comment.groupby(video_id_col)
        for vid, group in grp:
            texts_piece: list[str] = []
            for _, r in group.iterrows():
                t = r.get("comment_text")
                if pd.notna(t) and isinstance(t, str) and t.strip():
                    texts_piece.append(t.strip())
            if texts_piece:
                _add(vid, ". ".join(texts_piece))

    # 4. hashtag: hashtag_name
    if hashtag is not None and not hashtag.empty:
        grp = hashtag.groupby(video_id_col)
        for vid, group in grp:
            tags = [
                r.get("hashtag_name")
                for _, r in group.iterrows()
                if pd.notna(r.get("hashtag_name"))
                and isinstance(r.get("hashtag_name"), str)
            ]
            if tags:
                _add(vid, " ".join(tags))

    # 5. music: music_title + music_author
    if music is not None and not music.empty:
        for _, row in music.iterrows():
            vid = row[video_id_col]
            _add(vid, row.get("music_title"))
            _add(vid, row.get("music_author"))

    # 6. author: signature
    if author is not None and not author.empty:
        auth_map: dict[Any, str] = {}
        for _, row in author.iterrows():
            aid = row["author_id"]
            sig = row.get("signature", "")
            if isinstance(sig, str) and sig.strip():
                auth_map[aid] = sig
        # 通过 video_detail 的 author_id 关联
        for _, row in video_detail.iterrows():
            aid = row.get("author_id")
            sig = auth_map.get(aid)
            if sig and isinstance(sig, str) and sig.strip():
                _add(row[video_id_col], sig)

    # 拼接
    rows = []
    for vid in sorted(texts.keys()):
        combined = " ".join(texts[vid])
        rows.append({"video_id": vid, "combined_text": combined})

    result = pd.DataFrame(rows)
    return result


def fit_text_vectorizer(
    texts: pd.Series,
    text_dim: int = 32,
    random_seed: int = 2026,
) -> tuple[Any, Any, dict[str, Any]]:
    """在训练文本上拟合 TF-IDF + TruncatedSVD（或 HashingVectorizer 回退）。

    Args:
        texts: 训练集 combined_text Series
        text_dim: 目标文本向量维度
        random_seed: 随机种子

    Returns:
        (vectorizer, svd_or_none, info_dict)
    """
    info: dict[str, Any] = {}
    info["text_dim"] = text_dim
    info["sklearn_available"] = _SKLEARN_AVAILABLE

    # 兜底：空文本或 sklearn 不可用
    if not _SKLEARN_AVAILABLE:
        info["method"] = "stats_fallback"
        info["fallback_reason"] = "sklearn_not_available"
        return None, None, info

    # 去除空值
    valid_texts = texts.fillna("").astype(str)
    valid_texts = valid_texts.replace("", " ")
    has_content = valid_texts.str.strip().str.len().sum() > 0
    if not has_content:
        info["method"] = "stats_fallback"
        info["fallback_reason"] = "all_texts_empty"
        return None, None, info

    svd = None

    try:
        # 方法 1: TfidfVectorizer + TruncatedSVD
        tfidf = TfidfVectorizer(
            max_features=min(256, max(2, len(valid_texts))),
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=1,
            max_df=1.0,
            sublinear_tf=True,
        )
        tfidf_matrix = tfidf.fit_transform(valid_texts)
        n_features = tfidf_matrix.shape[1]
        info["tfidf_features"] = n_features

        if n_features >= text_dim:
            n_components = min(text_dim, n_features - 1)
            svd = TruncatedSVD(n_components=n_components, random_state=random_seed)
            svd.fit(tfidf_matrix)
            info["method"] = "tfidf_svd"
            info["svd_components"] = n_components
            info["explained_variance_ratio"] = float(svd.explained_variance_ratio_.sum())
            return tfidf, svd, info
        else:
            # 维度不足，降维到 n_features 并 zero-pad
            n_components = max(1, n_features - 1) if n_features > 1 else 1
            svd = TruncatedSVD(n_components=n_components, random_state=random_seed)
            svd.fit(tfidf_matrix)
            info["method"] = "tfidf_svd_padded"
            info["svd_components"] = n_components
            info["explained_variance_ratio"] = float(svd.explained_variance_ratio_.sum())
            return tfidf, svd, info

    except Exception as e:
        info["tfidf_error"] = str(e)
        info["tfidf_failed"] = True

    # 方法 2: HashingVectorizer 回退
    try:
        hv = HashingVectorizer(
            n_features=text_dim,
            analyzer="char_wb",
            ngram_range=(2, 4),
            alternate_sign=False,
        )
        hv.fit(valid_texts)
        info["method"] = "hashing"
        info["hashing_n_features"] = text_dim
        return hv, None, info
    except Exception as e2:
        info["hashing_error"] = str(e2)
        info["hashing_failed"] = True

    # 方法 3: 纯统计特征回退
    info["method"] = "stats_fallback"
    info["fallback_reason"] = "all_vectorizers_failed"
    return None, None, info


def transform_text(
    texts: pd.Series,
    vectorizer: Any,
    svd: Any | None,
    text_dim: int = 32,
) -> np.ndarray:
    """使用拟合好的 vectorizer 转换文本为固定维度向量。

    Args:
        texts: 待转换文本 Series
        vectorizer: 拟合好的 TfidfVectorizer 或 HashingVectorizer
        svd: 拟合好的 TruncatedSVD 或 None
        text_dim: 目标维度

    Returns:
        [n_samples, text_dim] numpy array
    """
    valid_texts = texts.fillna("").astype(str)
    valid_texts = valid_texts.replace("", " ")

    if vectorizer is None:
        # 纯统计回退
        return _stats_fallback(valid_texts, text_dim)

    try:
        matrix = vectorizer.transform(valid_texts)

        if svd is not None:
            reduced = svd.transform(matrix)  # [n, svd_dim]
            n, d = reduced.shape
            if d >= text_dim:
                return reduced[:, :text_dim]
            else:
                # zero-pad 到 text_dim
                padded = np.zeros((n, text_dim), dtype=np.float32)
                padded[:, :d] = reduced[:, :d]
                return padded
        else:
            # HashingVectorizer
            if matrix.shape[1] >= text_dim:
                arr = matrix.toarray() if hasattr(matrix, "toarray") else np.array(matrix)
                return arr[:, :text_dim].astype(np.float32)
            else:
                arr = matrix.toarray() if hasattr(matrix, "toarray") else np.array(matrix)
                padded = np.zeros((arr.shape[0], text_dim), dtype=np.float32)
                padded[:, : matrix.shape[1]] = arr[:, : matrix.shape[1]]
                return padded

    except Exception:
        return _stats_fallback(valid_texts, text_dim)


def _stats_fallback(texts: pd.Series, text_dim: int = 32) -> np.ndarray:
    """纯统计特征回退：文本长度、词数等统计特征。

    当所有 vectorizer 都失败时的最终兜底。
    """
    n = len(texts)
    features = np.zeros((n, max(text_dim, 4)), dtype=np.float32)

    for i, t in enumerate(texts):
        if not isinstance(t, str) or not t.strip():
            continue
        words = t.split()
        features[i, 0] = len(t)          # 字符长度
        features[i, 1] = len(words)      # 词数
        features[i, 2] = len(set(words))  # 唯一词数
        features[i, 3] = features[i, 0] / max(features[i, 1], 1)  # 平均词长

    return features[:, :text_dim]