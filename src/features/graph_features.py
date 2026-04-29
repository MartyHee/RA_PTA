"""GraphSAGE 图特征可复用函数。

包含节点特征计算、度统计、特征拼接等工具函数，
供 build_graph_dataset.py 和后续 GraphSAGE 训练使用。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_NODE_TYPES = ["video", "author", "music", "hashtag", "tag"]
DEFAULT_TABULAR_FEATURE_COLS = [
    "duration_ms",
    "digg_count",
    "comment_count",
    "share_count",
    "collect_count",
    "follower_count",
    "hashtag_count",
    "video_tag_count",
    "comment_table_count",
    "chapter_count",
]


def build_node_type_onehot(
    node_types: list[str],
    type_list: list[str] | None = None,
) -> np.ndarray:
    """将节点类型列表转换为 one-hot 编码矩阵。

    Args:
        node_types: 每个节点的类型字符串列表
        type_list: 类型定义列表（顺序决定 one-hot 列顺序）

    Returns:
        shape (num_nodes, num_types) 的 one-hot 矩阵
    """
    if type_list is None:
        type_list = DEFAULT_NODE_TYPES
    type_to_idx = {t: i for i, t in enumerate(type_list)}
    num_nodes = len(node_types)
    num_types = len(type_list)
    onehot = np.zeros((num_nodes, num_types), dtype=np.float32)
    for i, nt in enumerate(node_types):
        idx = type_to_idx.get(nt)
        if idx is not None:
            onehot[i, idx] = 1.0
    return onehot


def compute_node_degrees(edges_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """根据边表计算每个节点的度。

    Args:
        edges_df: 包含 source_node_id 和 target_node_id 列的 DataFrame

    Returns:
        嵌套 dict: {node_id_str: {"total": int, "in": int, "out": int}}
    """
    from collections import defaultdict

    in_deg: dict[str | Any, int] = defaultdict(int)
    out_deg: dict[str | Any, int] = defaultdict(int)

    for _, row in edges_df.iterrows():
        src = row["source_node_id"]
        tgt = row["target_node_id"]
        out_deg[src] += 1
        in_deg[tgt] += 1

    all_nodes = set(in_deg.keys()) | set(out_deg.keys())
    result = {}
    for n in all_nodes:
        out_d = out_deg.get(n, 0)
        in_d = in_deg.get(n, 0)
        result[str(n)] = {"total": out_d + in_d, "in": in_d, "out": out_d}
    return result


def build_degree_features(
    node_ids: list[int],
    node_degrees: dict[str, dict[str, int]],
) -> np.ndarray:
    """为节点列表构建度特征矩阵。

    Args:
        node_ids: node_id 列表（int）
        node_degrees: compute_node_degrees 的返回结果

    Returns:
        shape (num_nodes, 3) 矩阵: [total_degree, in_degree, out_degree]
    """
    num_nodes = len(node_ids)
    deg_feat = np.zeros((num_nodes, 3), dtype=np.float32)
    for i, nid in enumerate(node_ids):
        deg_info = node_degrees.get(str(nid), {"total": 0, "in": 0, "out": 0})
        deg_feat[i, 0] = deg_info["total"]
        deg_feat[i, 1] = deg_info["in"]
        deg_feat[i, 2] = deg_info["out"]
    return deg_feat


def build_video_tabular_features(
    tabular_df: pd.DataFrame,
    video_id_to_node: dict[str, int],
    num_nodes: int,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    """从 tabular 数据为所有视频节点构建数值特征。

    非视频节点对应位置填 0；主视频节点取 tabular 对应值；
    缺失值填 0。related_only 视频节点的值为 0（无 tabular 数据）。

    Args:
        tabular_df: 合并后的 train + eval tabular DataFrame
        video_id_to_node: {video_id_str: node_id} 映射
        num_nodes: 总节点数
        feature_cols: 要提取的特征列名列表

    Returns:
        shape (num_nodes, len(feature_cols)) 的特征矩阵
    """
    if feature_cols is None:
        feature_cols = DEFAULT_TABULAR_FEATURE_COLS

    # 只保留 feature_cols 中实际存在的列
    available_cols = [c for c in feature_cols if c in tabular_df.columns]
    n_feat = len(available_cols)
    feat_mat = np.zeros((num_nodes, n_feat), dtype=np.float32)

    video_id_str = "video_id"

    for _, row in tabular_df.iterrows():
        vid = str(row.get(video_id_str, ""))
        if vid in video_id_to_node:
            node_id = video_id_to_node[vid]
            for j, col in enumerate(available_cols):
                val = row.get(col)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    feat_mat[node_id, j] = float(val)
                else:
                    feat_mat[node_id, j] = 0.0

    return feat_mat


def get_feature_columns(
    tabular_feature_cols: list[str] | None = None,
) -> list[str]:
    """获取节点特征列名列表（用于 graph_meta.json 记录）。

    Args:
        tabular_feature_cols: 使用的 tabular 数值特征列

    Returns:
        完整特征列名列表
    """
    if tabular_feature_cols is None:
        tabular_feature_cols = DEFAULT_TABULAR_FEATURE_COLS
    type_list = DEFAULT_NODE_TYPES
    type_onehot_cols = [f"node_type_{t}" for t in type_list]
    degree_cols = ["total_degree", "in_degree", "out_degree"]
    return type_onehot_cols + degree_cols + list(tabular_feature_cols)