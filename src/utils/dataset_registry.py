"""Dataset registry — 通过 dataset_name + variant 查询数据产物路径。

用法:
    from utils.dataset_registry import get_dataset_info, resolve_train_config

    info = get_dataset_info("real_raw_5000", variant="no_interaction_leakage")
    # info == {
    #     "dataset_name": "real_raw_5000",
    #     "dataset_variant": "no_interaction_leakage",
    #     "train_npz_path": "data/multimodal/real_raw_5000/multimodal_train.npz",
    #     "val_npz_path": ...,
    #     ...
    # }

    config = resolve_train_config(base_config_dict, info, extra_params={})
    # config = base_config + dataset paths + extra hyperparams
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY_PATH = _PROJECT_ROOT / "configs" / "datasets.yaml"


def load_registry() -> dict[str, Any]:
    """加载 configs/datasets.yaml 全文。"""
    with open(_REGISTRY_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def list_datasets() -> list[str]:
    """返回所有已注册的数据集名称列表。"""
    registry = load_registry()
    return sorted(registry.get("datasets", {}).keys())


def list_variants(dataset_name: str) -> list[str]:
    """返回指定数据集的所有 variant 名称列表。"""
    registry = load_registry()
    ds = registry.get("datasets", {}).get(dataset_name)
    if ds is None:
        raise KeyError(f"Dataset '{dataset_name}' not found in registry")
    return list(ds.get("variants", {}).keys())


def get_dataset_info(
    dataset_name: str,
    variant: str | None = None,
) -> dict[str, Any]:
    """获取指定数据集 + variant 的产物路径信息。

    Args:
        dataset_name: 数据集名称，如 "real_raw_5000"。
        variant: variant 名称，如 "no_interaction_leakage"。
                 为 None 时尝试 "default"，无 default 则取第一个 variant。

    Returns:
        包含 paths 的字典，始终含 dataset_name / dataset_variant。

    Raises:
        KeyError: 数据集或 variant 未注册。
    """
    registry = load_registry()
    datasets = registry.get("datasets", {})
    if dataset_name not in datasets:
        raise KeyError(
            f"Dataset '{dataset_name}' not found in registry. "
            f"Available: {list(datasets.keys())}"
        )

    ds = datasets[dataset_name]
    variants = ds.get("variants", {})
    if not variants:
        raise KeyError(f"Dataset '{dataset_name}' has no variants defined")

    # 确定 variant
    if variant is not None:
        if variant not in variants:
            raise KeyError(
                f"Variant '{variant}' not found for dataset '{dataset_name}'. "
                f"Available: {list(variants.keys())}"
            )
        chosen_variant = variant
    elif "default" in variants:
        chosen_variant = "default"
    else:
        chosen_variant = list(variants.keys())[0]

    info = dict(variants[chosen_variant])
    info["dataset_name"] = dataset_name
    info["dataset_variant"] = chosen_variant
    return info


def resolve_train_config(
    base_config: dict[str, Any],
    dataset_info: dict[str, Any],
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """将通用模型 base_config 与 dataset 路径信息合并为完整训练配置。

    合并顺序（后覆盖前）:
        1. base_config 的副本
        2. dataset_info 中的路径字段
        3. extra_params（如采样的超参数）

    Args:
        base_config: 通用模型基础配置字典（如 multimodal_base.yaml）。
        dataset_info: get_dataset_info() 返回值。
        extra_params: 额外覆盖参数，如 trial 采样的超参数。

    Returns:
        完整的训练配置字典，可直接写入临时 YAML 供 train.py 使用。
    """
    config = dict(base_config)

    # 注入 dataset 路径（覆盖 base_config 中的占位路径）
    path_keys = [
        "train_npz_path",
        "val_npz_path",
        "test_npz_path",
        "feature_info_path",
        "leakage_report_path",
    ]
    for key in path_keys:
        if key in dataset_info:
            config[key] = dataset_info[key]

    # 注入 dataset 标识
    for key in ["dataset_name", "dataset_variant"]:
        if key in dataset_info:
            config[key] = dataset_info[key]

    # 注入额外参数（超参数覆盖）
    if extra_params:
        config.update(extra_params)

    return config


def build_default_output_root(
    dataset_name: str,
    variant: str | None = None,
    base_dir: str = "outputs/tuning/multimodal",
) -> str:
    """生成默认的 tuning 输出根目录。

    格式: outputs/tuning/multimodal/<dataset_name>[_<variant>]
    """
    if variant and variant != "default":
        return f"{base_dir}/{dataset_name}_{variant}"
    return f"{base_dir}/{dataset_name}"