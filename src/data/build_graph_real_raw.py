"""GraphSAGE 图数据集构建脚本 — real_raw_1000

基于 real_raw_1000 原始 raw 表和 tabular 数据，构建以视频为核心的简化同构图。
所有节点映射到统一 node_id 空间，保留 node_type 字段。
所有边输出双向（正向 + 反向）。
使用 train/val/test 三路 mask 对齐 tabular split。

设计原则：
- 不修改 douyin_data_project 爬虫代码
- 不修改真实 raw CSV 文件
- 不覆盖 sample0427 旧图数据
- raw_video_tag / raw_chapter 空表跳过，不影响构图
- music_id 缺失时跳过对应 music 节点（不使用回退标识）
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import read_csv_safe  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.features.graph_features import (  # noqa: E402
    build_node_type_onehot,
    build_degree_features,
    build_video_tabular_features,
    compute_node_degrees,
    get_feature_columns,
    REAL_RAW_TABULAR_FEATURE_COLS,
)

# ============================================================================
# Constants
# ============================================================================
REAL_RAW_NODE_TYPES = ["video", "author", "music", "hashtag"]

RAW_FILE_MAP = {
    "raw_video_detail": "raw_video_detail_20260507_230322.csv",
    "raw_author": "raw_author_20260507_230322.csv",
    "raw_music": "raw_music_20260507_230322.csv",
    "raw_hashtag": "raw_hashtag_20260507_230322.csv",
    "raw_video_tag": "raw_video_tag_20260507_230322.csv",
    "raw_chapter": "raw_chapter_20260507_230322.csv",
    "raw_comment": "raw_comment_20260507_230322.csv",
    "raw_related_video": "raw_related_video_20260507_230322.csv",
}

ALLOWED_EMPTY_TABLES = {"raw_video_tag", "raw_chapter"}


def make_raw_key(entity_type: str, raw_id: str) -> str:
    """生成 raw_key: entity_type::raw_id"""
    return f"{entity_type}::{raw_id}"


def safe_int_str(value: Any) -> str:
    """将值安全转换为整数字符串。

    处理 float NaN、大整数、UUID 字符串等情况。
    UUID 字符串（如 f28b3db5-...）不会被 int() 转换，
    直接返回原始字符串。
    """
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        if value == value and value == int(value):
            return str(int(value))
        return str(value)
    try:
        return str(int(value))
    except (ValueError, TypeError):
        return str(value).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphSAGE 图数据集构建 — real_raw_1000")
    parser.add_argument(
        "--config",
        default="configs/graphsage/graphsage_real_raw_1000.yaml",
        help="GraphSAGE 配置文件路径",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    project_root = Path(config.get("project_root", PROJECT_ROOT))

    # Resolve directories
    raw_data_dir = Path(
        config.get(
            "raw_data_dir",
            project_root / "douyin_data_project" / "data" / "interim" / "20260507_230322",
        )
    )
    feature_dir = Path(
        config.get("features_dir", project_root / "data" / "features" / "real_raw_1000")
    )
    graph_data_dir = Path(
        config.get("graph_data_dir", project_root / "data" / "graph" / "real_raw_1000")
    )
    check_dir = Path(
        config.get("data_check_dir", project_root / "outputs" / "data_check" / "real_raw_1000")
    )

    # Create output directories
    graph_data_dir.mkdir(parents=True, exist_ok=True)
    check_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []

    # ========================================================================
    # 1. Load raw tables
    # ========================================================================
    print("[1/8] 加载原始数据表...")
    tables: dict[str, pd.DataFrame] = {}
    for name, fname in RAW_FILE_MAP.items():
        path = raw_data_dir / fname
        if not path.exists():
            if name in ALLOWED_EMPTY_TABLES:
                print(f"  {name}: 文件不存在（允许空表），使用空 DataFrame")
                tables[name] = pd.DataFrame()
                continue
            raise FileNotFoundError(f"关键表缺失: {path}")
        df, enc = read_csv_safe(path)
        tables[name] = df
        print(f"  {name}: {len(df)} rows, {len(df.columns)} cols, encoding={enc}")

    # ========================================================================
    # 2. Load tabular data (labels, split, numeric features)
    # ========================================================================
    print("[2/8] 加载 tabular train/val/test 数据...")
    tabular_train_path = feature_dir / "tabular_train.csv"
    tabular_val_path = feature_dir / "tabular_val.csv"
    tabular_test_path = feature_dir / "tabular_test.csv"

    tabular_train, _ = read_csv_safe(tabular_train_path)
    tabular_val, _ = read_csv_safe(tabular_val_path)
    tabular_test, _ = read_csv_safe(tabular_test_path)
    tabular_all = pd.concat([tabular_train, tabular_val, tabular_test], ignore_index=True)
    print(f"  tabular_train: {len(tabular_train)} rows")
    print(f"  tabular_val:   {len(tabular_val)} rows")
    print(f"  tabular_test:  {len(tabular_test)} rows")
    print(f"  tabular_all:   {len(tabular_all)} rows")

    # Build lookup dicts for labels and splits
    video_label: dict[str, Any] = {}
    video_split: dict[str, str] = {}
    for _, row in tabular_train.iterrows():
        vid = safe_int_str(row["video_id"])
        if vid:
            video_label[vid] = row.get("label", -1)
            video_split[vid] = "train"
    for _, row in tabular_val.iterrows():
        vid = safe_int_str(row["video_id"])
        if vid:
            video_label[vid] = row.get("label", -1)
            video_split[vid] = "val"
    for _, row in tabular_test.iterrows():
        vid = safe_int_str(row["video_id"])
        if vid:
            video_label[vid] = row.get("label", -1)
            video_split[vid] = "test"

    # Verify split counts
    split_counts = defaultdict(int)
    for sp in video_split.values():
        split_counts[sp] += 1
    print(f"  Labeled split counts: {dict(split_counts)}")
    expected_splits = {"train": 700, "val": 150, "test": 150}
    for sp, cnt in expected_splits.items():
        actual = split_counts.get(sp, 0)
        if actual != cnt:
            warnings.append(f"Expected {sp}={cnt}, got {actual}")
            print(f"  [WARN] {sp}: expected {cnt}, got {actual}")

    # ========================================================================
    # 3. Build node mapping
    # ========================================================================
    print("[3/8] 构建节点映射...")
    raw_key_to_node_id: dict[str, int] = {}
    node_id_to_raw_key: dict[str, str] = {}
    node_type_map: dict[int, str] = {}
    node_raw_id_map: dict[int, str] = {}

    def add_node(entity_type: str, raw_id: str) -> int:
        raw_key = make_raw_key(entity_type, raw_id)
        if raw_key not in raw_key_to_node_id:
            nid = len(raw_key_to_node_id)
            raw_key_to_node_id[raw_key] = nid
            node_id_to_raw_key[str(nid)] = raw_key
            node_type_map[nid] = entity_type
            node_raw_id_map[nid] = raw_id
            return nid
        return raw_key_to_node_id[raw_key]

    # --- 3a. Video: main video nodes from raw_video_detail ---
    rd = tables["raw_video_detail"]
    main_video_ids: set[str] = set()
    for vid in rd["video_id"].dropna().unique():
        vid_str = safe_int_str(vid)
        if not vid_str:
            continue
        add_node("video", vid_str)
        main_video_ids.add(vid_str)
    print(f"  Main video nodes: {len(main_video_ids)}")

    # --- 3b. Video: related-only video nodes from raw_related_video ---
    rv = tables["raw_related_video"]
    related_video_ids: set[str] = set()
    if len(rv) > 0:
        for rvid in rv["related_video_id"].dropna().unique():
            rvid_str = str(rvid).strip()
            if rvid_str and rvid_str not in main_video_ids:
                add_node("video", rvid_str)
                related_video_ids.add(rvid_str)
    print(f"  Related-only video nodes: {len(related_video_ids)}")

    # --- 3c. Author nodes ---
    ra = tables["raw_author"]
    author_ids_from_author: set[str] = set()
    for aid in ra["author_id"].dropna().unique():
        aid_str = safe_int_str(aid)
        if not aid_str:
            continue
        add_node("author", aid_str)
        author_ids_from_author.add(aid_str)
    print(f"  Author nodes from raw_author: {len(author_ids_from_author)}")

    # --- 3d. Music nodes (skip if music_id missing, no fallback) ---
    rm = tables["raw_music"]
    music_count = 0
    music_empty = 0
    for _, row in rm.iterrows():
        mid = row.get("music_id")
        if mid is not None and not (isinstance(mid, float) and pd.isna(mid)):
            mid_str = safe_int_str(mid)
            if mid_str:
                add_node("music", mid_str)
                music_count += 1
                continue
        music_empty += 1
    if music_empty > 0:
        msg = f"Skipped {music_empty} music node(s) due to missing music_id (no fallback used)"
        warnings.append(msg)
        print(f"  [INFO] {msg}")
    print(f"  Music nodes: {music_count}")

    # --- 3e. Hashtag nodes (prefer hashtag_id, fallback hashtag_name) ---
    rh = tables["raw_hashtag"]
    hashtag_count = 0
    hashtag_fallback = False
    for _, row in rh.iterrows():
        hid = row.get("hashtag_id")
        if hid is not None and not (isinstance(hid, float) and pd.isna(hid)):
            hid_str = safe_int_str(hid)
            if hid_str:
                add_node("hashtag", hid_str)
                hashtag_count += 1
            else:
                hname = row.get("hashtag_name")
                if hname and not (isinstance(hname, float) and pd.isna(hname)):
                    add_node("hashtag", str(hname).strip())
                    hashtag_count += 1
                    hashtag_fallback = True
        else:
            hname = row.get("hashtag_name")
            if hname and not (isinstance(hname, float) and pd.isna(hname)):
                add_node("hashtag", str(hname).strip())
                hashtag_count += 1
                hashtag_fallback = True
    if hashtag_fallback:
        warnings.append("Some hashtag nodes used fallback identifier (hashtag_name)")
    print(f"  Hashtag nodes: {hashtag_count}")

    # --- 3f. raw_video_tag is empty, skip tag nodes ---
    vt = tables.get("raw_video_tag", pd.DataFrame())
    if len(vt) == 0:
        print("  raw_video_tag: empty table, tag nodes skipped (expected)")

    # --- 3g. raw_chapter is empty, skip ---
    rc = tables.get("raw_chapter", pd.DataFrame())
    if len(rc) == 0:
        print("  raw_chapter: empty table, no chapter handling needed (expected)")

    num_nodes = len(raw_key_to_node_id)
    print(f"  Total nodes: {num_nodes}")

    # ========================================================================
    # 4. Build edges (bidirectional)
    # ========================================================================
    print("[4/8] 构建边（双向）...")
    edges: list[tuple] = []
    edge_type_counts: dict[str, int] = defaultdict(int)
    unmapped_edge_counts: dict[str, int] = defaultdict(int)

    def try_add_edge(src_key: str, tgt_key: str, etype: str) -> None:
        src_id = raw_key_to_node_id.get(src_key)
        tgt_id = raw_key_to_node_id.get(tgt_key)
        if src_id is None:
            unmapped_edge_counts[f"missing_src_{etype}"] += 1
            return
        if tgt_id is None:
            unmapped_edge_counts[f"missing_tgt_{etype}"] += 1
            return
        # Forward
        edges.append((src_id, tgt_id, src_key, tgt_key, etype, False))
        edge_type_counts[etype] += 1
        # Reverse (bidirectional)
        edges.append((tgt_id, src_id, tgt_key, src_key, etype, True))
        edge_type_counts[f"{etype}_reverse"] += 1

    # --- 4a. video -> author ---
    for _, row in rd.iterrows():
        vid = safe_int_str(row["video_id"])
        if not vid:
            continue
        aid = row["author_id"]
        if pd.notna(aid):
            aid_str = safe_int_str(aid)
            if aid_str:
                try_add_edge(
                    make_raw_key("video", vid),
                    make_raw_key("author", aid_str),
                    "video_author",
                )

    # --- 4b. video -> music (skip if music_id missing) ---
    for _, row in rm.iterrows():
        vid = safe_int_str(row["video_id"])
        if not vid:
            continue
        mid = row.get("music_id")
        if mid is not None and not (isinstance(mid, float) and pd.isna(mid)):
            mid_str = safe_int_str(mid)
            if mid_str:
                tgt_key = make_raw_key("music", mid_str)
            else:
                continue
        else:
            continue
        try_add_edge(make_raw_key("video", vid), tgt_key, "video_music")

    # --- 4c. video -> hashtag ---
    for _, row in rh.iterrows():
        vid = safe_int_str(row["video_id"])
        if not vid:
            continue
        hid = row.get("hashtag_id")
        if hid is not None and not (isinstance(hid, float) and pd.isna(hid)):
            hid_str = safe_int_str(hid)
            if hid_str:
                tgt_key = make_raw_key("hashtag", hid_str)
            else:
                hname = row.get("hashtag_name")
                if hname and not (isinstance(hname, float) and pd.isna(hname)):
                    tgt_key = make_raw_key("hashtag", str(hname).strip())
                else:
                    continue
        else:
            hname = row.get("hashtag_name")
            if hname and not (isinstance(hname, float) and pd.isna(hname)):
                tgt_key = make_raw_key("hashtag", str(hname).strip())
            else:
                continue
        try_add_edge(make_raw_key("video", vid), tgt_key, "video_hashtag")

    # --- 4d. source_video -> related_video ---
    if len(rv) > 0:
        for _, row in rv.iterrows():
            src_vid = row["source_video_id"]
            tgt_rvid = row["related_video_id"]
            if pd.notna(src_vid) and pd.notna(tgt_rvid) and str(tgt_rvid).strip():
                src_vid_str = safe_int_str(src_vid)
                if not src_vid_str:
                    continue
                tgt_rvid_str = str(tgt_rvid).strip()
                try_add_edge(
                    make_raw_key("video", src_vid_str),
                    make_raw_key("video", tgt_rvid_str),
                    "video_related_video",
                )

    num_edges = len(edges)
    print(f"  Total edges (bidirectional): {num_edges}")
    for etype, cnt in sorted(edge_type_counts.items()):
        print(f"    {etype}: {cnt}")

    for k, cnt in sorted(unmapped_edge_counts.items()):
        print(f"    [WARN] Unmapped {k}: {cnt}")
        warnings.append(f"Unmapped edges: {k}: {cnt}")

    edges_df = pd.DataFrame(
        edges,
        columns=[
            "source_node_id", "target_node_id",
            "source_raw_id", "target_raw_id",
            "edge_type", "is_reverse",
        ],
    )

    # ========================================================================
    # 5. Build node features
    # ========================================================================
    print("[5/8] 构建节点特征...")

    # 5a. One-hot node type (4 dims: video, author, music, hashtag)
    node_types = [node_type_map[i] for i in range(num_nodes)]
    type_onehot = build_node_type_onehot(node_types, type_list=REAL_RAW_NODE_TYPES)
    print(f"  Node type one-hot: {type_onehot.shape}")

    # 5b. Degree features (from forward edges only, to avoid double counting)
    forward_edges = edges_df[edges_df["is_reverse"] == False].copy()
    node_degrees = compute_node_degrees(forward_edges)
    deg_features = build_degree_features(list(range(num_nodes)), node_degrees)
    print(f"  Degree features: {deg_features.shape}")

    # 5c. Tabular numeric features for main video nodes
    video_key_to_nid: dict[str, int] = {}
    for vid_str in main_video_ids:
        raw_key = make_raw_key("video", vid_str)
        nid = raw_key_to_node_id.get(raw_key)
        if nid is not None:
            video_key_to_nid[vid_str] = nid

    # Use REAL_RAW_TABULAR_FEATURE_COLS (without video_tag_count, chapter_count)
    tabular_feat = build_video_tabular_features(
        tabular_all, video_key_to_nid, num_nodes,
        feature_cols=REAL_RAW_TABULAR_FEATURE_COLS,
    )
    print(f"  Tabular features: {tabular_feat.shape}")

    # Combine: [type_onehot | deg_features | tabular_feat]
    node_features = np.concatenate([type_onehot, deg_features, tabular_feat], axis=1)
    feature_dim = node_features.shape[1]

    # Build feature column names for metadata
    type_onehot_cols = [f"node_type_{t}" for t in REAL_RAW_NODE_TYPES]
    degree_cols = ["total_degree", "in_degree", "out_degree"]
    feature_columns = type_onehot_cols + degree_cols + list(REAL_RAW_TABULAR_FEATURE_COLS)
    print(f"  Total feature dim: {feature_dim}")
    print(f"  Feature columns: {feature_columns}")

    # ========================================================================
    # 6. Build labels and masks (train/val/test three-way)
    # ========================================================================
    print("[6/8] 构建标签和 mask（三路切分）...")
    labels = np.full(num_nodes, -1, dtype=np.float32)
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    for vid_str in main_video_ids:
        raw_key = make_raw_key("video", vid_str)
        nid = raw_key_to_node_id.get(raw_key)
        if nid is None:
            continue
        lbl = video_label.get(vid_str, -1)
        try:
            labels[nid] = float(lbl) if lbl is not None else -1.0
        except (ValueError, TypeError):
            labels[nid] = -1.0
        sp = video_split.get(vid_str, "unknown")
        if sp == "train":
            train_mask[nid] = True
        elif sp == "val":
            val_mask[nid] = True
        elif sp == "test":
            test_mask[nid] = True

    labeled_count = int((labels >= 0).sum())
    train_count = int(train_mask.sum())
    val_count = int(val_mask.sum())
    test_count = int(test_mask.sum())
    unlabeled_count = num_nodes - labeled_count
    print(f"  Labeled nodes: {labeled_count} (train={train_count}, val={val_count}, test={test_count})")
    print(f"  Unlabeled nodes: {unlabeled_count}")

    # Verify no overlap between masks
    overlap_tv = int((train_mask & val_mask).sum())
    overlap_tt = int((train_mask & test_mask).sum())
    overlap_vt = int((val_mask & test_mask).sum())
    total_overlap = overlap_tv + overlap_tt + overlap_vt
    if total_overlap > 0:
        warnings.append(
            f"Mask overlap: train-val={overlap_tv}, "
            f"train-test={overlap_tt}, val-test={overlap_vt}"
        )
        print(f"  [WARN] Mask overlap: train-val={overlap_tv}, "
              f"train-test={overlap_tt}, val-test={overlap_vt}")
    else:
        print("  [OK] All masks are mutually exclusive")

    # ========================================================================
    # 7. Output graph data files
    # ========================================================================
    print("[7/8] 输出图数据文件...")

    # --- 7a. nodes.csv ---
    nodes_rows = []
    for nid in range(num_nodes):
        raw_key = node_id_to_raw_key.get(str(nid), "")
        ntype = node_type_map.get(nid, "unknown")
        raw_id = node_raw_id_map.get(nid, "")
        is_main = "yes" if (ntype == "video" and raw_id in main_video_ids) else "no"
        is_related = "yes" if (ntype == "video" and raw_id in related_video_ids) else "no"
        has_lbl = "yes" if labels[nid] >= 0 else "no"
        lbl = int(labels[nid]) if labels[nid] >= 0 else -1
        if train_mask[nid]:
            sp = "train"
        elif val_mask[nid]:
            sp = "val"
        elif test_mask[nid]:
            sp = "test"
        else:
            sp = "none"
        nodes_rows.append({
            "node_id": nid,
            "raw_id": raw_id,
            "node_type": ntype,
            "is_main_video": is_main,
            "is_related_only_video": is_related,
            "has_label": has_lbl,
            "label": lbl,
            "split": sp,
        })
    nodes_df = pd.DataFrame(nodes_rows)
    nodes_path = graph_data_dir / "nodes.csv"
    nodes_df.to_csv(nodes_path, index=False, encoding="utf-8-sig")
    print(f"  nodes.csv: {len(nodes_df)} rows -> {nodes_path}")

    # --- 7b. edges.csv ---
    edges_path = graph_data_dir / "edges.csv"
    edges_df.to_csv(edges_path, index=False, encoding="utf-8-sig")
    print(f"  edges.csv: {len(edges_df)} rows -> {edges_path}")

    # --- 7c. node_features.npy ---
    features_path = graph_data_dir / "node_features.npy"
    np.save(str(features_path), node_features)
    print(f"  node_features.npy: {node_features.shape} -> {features_path}")

    # --- 7d. labels.npy ---
    labels_path = graph_data_dir / "labels.npy"
    np.save(str(labels_path), labels)
    print(f"  labels.npy: {labels.shape} -> {labels_path}")

    # --- 7e. train_mask.npy ---
    train_mask_path = graph_data_dir / "train_mask.npy"
    np.save(str(train_mask_path), train_mask)
    print(f"  train_mask.npy: {train_mask.shape} -> {train_mask_path}")

    # --- 7f. val_mask.npy ---
    val_mask_path = graph_data_dir / "val_mask.npy"
    np.save(str(val_mask_path), val_mask)
    print(f"  val_mask.npy: {val_mask.shape} -> {val_mask_path}")

    # --- 7g. test_mask.npy ---
    test_mask_path = graph_data_dir / "test_mask.npy"
    np.save(str(test_mask_path), test_mask)
    print(f"  test_mask.npy: {test_mask.shape} -> {test_mask_path}")

    # --- 7h. node_id_mapping.json ---
    mapping = {
        "raw_key_to_node_id": {
            k: v for k, v in sorted(raw_key_to_node_id.items(), key=lambda x: x[1])
        },
        "node_id_to_raw_key": {
            str(k): v for k, v in sorted(node_id_to_raw_key.items(), key=lambda x: int(x[0]))
        },
        "node_type_mapping": {
            str(k): v for k, v in sorted(node_type_map.items())
        },
    }
    mapping_path = graph_data_dir / "node_id_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"  node_id_mapping.json -> {mapping_path}")

    # --- 7i. graph_meta.json ---
    node_type_counts_dict: dict[str, int] = {}
    for nt in REAL_RAW_NODE_TYPES:
        node_type_counts_dict[nt] = sum(1 for t in node_type_map.values() if t == nt)

    forward_edge_type_counts: dict[str, int] = defaultdict(int)
    for _, row in edges_df[edges_df["is_reverse"] == False].iterrows():
        forward_edge_type_counts[row["edge_type"]] += 1

    graph_meta = {
        "dataset_name": "real_raw_1000",
        "num_nodes": num_nodes,
        "num_edges_forward": int(edges_df[edges_df["is_reverse"] == False].shape[0]),
        "num_edges_bidirectional": num_edges,
        "feature_dim": feature_dim,
        "node_type_counts": dict(node_type_counts_dict),
        "edge_type_counts": dict(forward_edge_type_counts),
        "main_video_count": len(main_video_ids),
        "related_only_video_count": len(related_video_ids),
        "labeled_node_count": labeled_count,
        "train_node_count": train_count,
        "val_node_count": val_count,
        "test_node_count": test_count,
        "unlabeled_node_count": unlabeled_count,
        "feature_columns": feature_columns,
        "label_definition": "离线实验伪标签: interaction_score >= P60 (18147.80), 继承自 tabular",
        "label_source": "data/features/real_raw_1000/tabular_train/val/test.csv",
        "split_definition": "train/val/test 三路切分（700/150/150），继承自 tabular，按 video_id 分层",
        "edge_direction": (
            "bidirectional: 每条边同时输出正向 (is_reverse=False) 和反向 (is_reverse=True)"
        ),
        "graph_type": "simplified homogeneous (all node types in single node_id space)",
        "source_tables_used": list(RAW_FILE_MAP.keys()),
        "tabular_data_used": [
            "data/features/real_raw_1000/tabular_train.csv",
            "data/features/real_raw_1000/tabular_val.csv",
            "data/features/real_raw_1000/tabular_test.csv",
        ],
        "edge_definitions": {
            "video_author": "raw_video_detail.video_id -> author_id",
            "video_music": (
                "raw_music.video_id -> music_id (skip if missing, no fallback)"
            ),
            "video_hashtag": (
                "raw_hashtag.video_id -> hashtag_id (or fallback hashtag_name)"
            ),
            "video_related_video": (
                "raw_related_video.source_video_id -> related_video_id"
            ),
        },
        "music_node_handling": (
            f"music_id missing for {music_empty} video(s); "
            "those nodes were skipped (no fallback identifier used)"
        ),
        "empty_tables_handled": [
            "raw_video_tag (0 rows, tag nodes skipped)",
            "raw_chapter (0 rows, no handling needed)",
        ],
        "node_types_used": REAL_RAW_NODE_TYPES,
        "tabular_feature_cols_used": REAL_RAW_TABULAR_FEATURE_COLS,
        "test_split_note": (
            "test_mask 仅用于最终泛化评估，不得用于训练、调参或特征标准化拟合"
        ),
        "warnings": warnings,
        "notes": [
            "当前图数据基于 real_raw_1000 真实网页端 raw 数据构建。",
            "related_video_id 来自真实搜索/推荐响应，非规则生成 ID。",
            "hashtag_id, music_id 来自真实页面响应。",
            "raw_video_tag 为空表，不包含 tag 节点和 video-tag 边。",
            "raw_chapter 为空表。",
            "music_id 缺失的视频已跳过对应 music 节点，未使用回退标识。",
            "所有边已输出双向 (bidirectional)，GraphSAGE 训练时可使用无向邻居采样。",
            "非主视频节点 (author/music/hashtag/related_only_video) label=-1, mask=False。",
            "特征以 raw 值保存，训练阶段在 GraphSAGE dataset 中基于 train_mask 做标准化。",
            "不包含 comment_user 节点，可在后续迭代中加入。",
        ],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = graph_data_dir / "graph_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(graph_meta, f, ensure_ascii=False, indent=2)
    print(f"  graph_meta.json -> {meta_path}")

    # ========================================================================
    # 8. Output check files
    # ========================================================================
    print("[8/8] 输出检查报告...")

    # Isolated nodes
    node_ids_with_edges: set[int] = set()
    for _, row in edges_df.iterrows():
        node_ids_with_edges.add(int(row["source_node_id"]))
        node_ids_with_edges.add(int(row["target_node_id"]))
    isolated_indices = [
        nid for nid in range(num_nodes) if nid not in node_ids_with_edges
    ]

    # Label distribution
    pos_count = int((labels == 1).sum())
    neg_count = int((labels == 0).sum())
    unlabeled_actual = int((labels == -1).sum())

    # --- 8a. graph_dataset_report.json ---
    report = {
        "build_success": True,
        "dataset_name": "real_raw_1000",
        "num_nodes": num_nodes,
        "num_edges_forward": int(edges_df[edges_df["is_reverse"] == False].shape[0]),
        "num_edges_bidirectional": num_edges,
        "node_type_counts": dict(node_type_counts_dict),
        "edge_type_counts": dict(forward_edge_type_counts),
        "labeled_node_count": labeled_count,
        "train_node_count": train_count,
        "val_node_count": val_count,
        "test_node_count": test_count,
        "unlabeled_node_count": unlabeled_count,
        "label_distribution": {
            "positive": pos_count,
            "negative": neg_count,
            "unlabeled": unlabeled_actual,
        },
        "mask_check": {
            "train_true": train_count,
            "val_true": val_count,
            "test_true": test_count,
            "overlap_train_val": overlap_tv,
            "overlap_train_test": overlap_tt,
            "overlap_val_test": overlap_vt,
            "neither_true": int((~train_mask & ~val_mask & ~test_mask).sum()),
        },
        "isolated_node_count": len(isolated_indices),
        "isolated_node_ids": isolated_indices[:30],
        "has_unmapped_edges": len(unmapped_edge_counts) > 0,
        "unmapped_edge_summary": dict(unmapped_edge_counts),
        "feature_dim": feature_dim,
        "feature_columns": feature_columns,
        "tabular_feature_cols_used": REAL_RAW_TABULAR_FEATURE_COLS,
        "node_types_used": REAL_RAW_NODE_TYPES,
        "empty_tables_handled": [
            "raw_video_tag (0 rows, skipped tag nodes)",
            "raw_chapter (0 rows, skipped)",
        ],
        "music_empty_skipped": music_empty,
        "warnings": warnings,
        "notes": [
            "当前图数据检查报告基于 real_raw_1000 真实网页端 raw 数据构建。",
            "孤立节点是预期行为（无关联边的节点不影响主视频监督训练）。",
            "所有边已输出双向。",
            "test_mask 仅用于最终泛化评估，不得用于训练或调参。",
        ],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    report_path = check_dir / "graph_dataset_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  graph_dataset_report.json -> {report_path}")

    # --- 8b. graph_node_preview.csv (first 20) ---
    preview_nodes_path = check_dir / "graph_node_preview.csv"
    nodes_df.head(20).to_csv(preview_nodes_path, index=False, encoding="utf-8-sig")
    print(f"  graph_node_preview.csv -> {preview_nodes_path}")

    # --- 8c. graph_edge_preview.csv (first 50) ---
    preview_edges_path = check_dir / "graph_edge_preview.csv"
    edges_df.head(50).to_csv(preview_edges_path, index=False, encoding="utf-8-sig")
    print(f"  graph_edge_preview.csv -> {preview_edges_path}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 60)
    print("  GraphSAGE 图数据集构建完成 (real_raw_1000)!")
    print("=" * 60)
    print(f"  总节点数:         {num_nodes}")
    print(f"    主视频:          {len(main_video_ids)}")
    print(f"    related_only:   {len(related_video_ids)}")
    print(f"    author:         {node_type_counts_dict.get('author', 0)}")
    print(f"    music:          {node_type_counts_dict.get('music', 0)}")
    print(f"    hashtag:        {node_type_counts_dict.get('hashtag', 0)}")
    print(f"  总边数 (双向):     {num_edges}")
    print(f"  特征维度:          {feature_dim}")
    print(f"  有标签节点:        {labeled_count}")
    print(f"    其中 train:     {train_count}")
    print(f"    其中 val:       {val_count}")
    print(f"    其中 test:      {test_count}")
    print(f"  无标签节点:        {unlabeled_count}")
    print(f"  孤立节点:          {len(isolated_indices)}")
    print(f"  Warnings:          {len(warnings)}")
    for w in warnings:
        print(f"    - {w}")
    print("=" * 60)


if __name__ == "__main__":
    main()