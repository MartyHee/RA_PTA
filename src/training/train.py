#!/usr/bin/env python
"""
统一训练入口 — Batch 6C (支持 DNN / Wide & Deep)

用法:
  python src/training/train.py --dataset real_raw_5000 --model dnn
  python src/training/train.py --dataset real_raw_5000 --model dnn --dry-run
  python src/training/train.py --dataset real_raw_5000 --model dnn --override epochs=30
  python src/training/train.py --dataset real_raw_5000 --model wide_deep
  python src/training/train.py --dataset real_raw_5000 --model wide_deep --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


# ── 路径 ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 泄漏检查 — 硬约束：以下字段不得出现在模型输入特征中
HARDCODED_EXCLUDED_LABEL_SOURCE = {
    "digg_count",
    "comment_count",
    "share_count",
    "collect_count",
}

# 统一字段名 → 模型 train.py config 字段名映射
FIELD_MAP = {
    "seed": "random_seed",
    "device": "device",
    "threshold": "threshold",
}

# 已接入的模型
SUPPORTED_MODELS = {"dnn", "wide_deep", "graphsage", "multimodal"}
MODEL_SCRIPTS = {
    "dnn": "src/models/dnn/train.py",
    "wide_deep": "src/models/wide_deep/train.py",
    "graphsage": "src/models/graphsage/train.py",
    "multimodal": "src/models/multimodal/train.py",
}

# Multimodal 字段名映射（统一格式 short_name → Multimodal train.py 期望的 _dim 全名）
MULTIMODAL_FIELD_MAP = {
    "text_hidden": "text_hidden_dim",
    "visual_hidden": "visual_hidden_dim",
    "structured_hidden": "structured_hidden_dim",
    "fusion_hidden": "fusion_hidden_dim",
}

# GraphSAGE 文件映射（路径键 → 文件名）
GRAPH_FILE_MAP = {
    "node_features_path": "node_features.npy",
    "labels_path": "labels.npy",
    "train_mask_path": "train_mask.npy",
    "val_mask_path": "val_mask.npy",
    "test_mask_path": "test_mask.npy",
    "edge_path": "edges.csv",
    "node_path": "nodes.csv",
    "graph_meta_path": "graph_meta.json",
}


def load_yaml(path: Path) -> dict:
    """加载 YAML 文件，返回字典。"""
    if not path.exists():
        print(f"[错误] 配置文件不存在: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data else {}


def parse_override_value(value_str: str):
    """解析 --override 的值到合适的 Python 类型。"""
    # JSON 格式（处理列表、字典等）
    if value_str.startswith(("[", "{")):
        try:
            return json.loads(value_str)
        except json.JSONDecodeError:
            pass
    # 整数
    try:
        return int(value_str)
    except ValueError:
        pass
    # 浮点数
    try:
        return float(value_str)
    except ValueError:
        pass
    # 布尔值
    if value_str.lower() in ("true", "false"):
        return value_str.lower() == "true"
    # 字符串
    return value_str


def resolve_tabular_dataset_paths(model_name: str, dataset_name: str) -> dict:
    """从 model_name 和 dataset_name 推导 tabular 数据路径。

    DNN 和 Wide & Deep 共享同一套 tabular CSV 文件，仅 output_root 不同。
    所有路径相对于 PROJECT_ROOT，与模型 train.py 的约定一致。
    """
    features_dir = f"data/features/{dataset_name}"
    return {
        "train_data_path": f"{features_dir}/tabular_train.csv",
        "val_data_path": f"{features_dir}/tabular_val.csv",
        "test_data_path": f"{features_dir}/tabular_test.csv",
        "feature_info_path": f"{features_dir}/tabular_feature_info.json",
        "leakage_report_path": (
            f"outputs/data_check/{dataset_name}/{dataset_name}_leakage_check_report.json"
        ),
        "output_root": f"outputs/{model_name}/{dataset_name}",
    }


def resolve_multimodal_dataset_paths(dataset_name: str, model_name: str) -> dict:
    """从 model_name 和 dataset_name 推导 multimodal NPZ 数据路径。

    Multimodal 使用 NPZ 文件输入（train/val/test），
    以及 feature_info JSON 和独立的 leakage 检查报告。
    路径模板: data/multimodal/{dataset_name}/multimodal_{split}.npz
    """
    multimodal_dir = f"data/multimodal/{dataset_name}"
    return {
        "train_npz_path": f"{multimodal_dir}/multimodal_train.npz",
        "val_npz_path": f"{multimodal_dir}/multimodal_val.npz",
        "test_npz_path": f"{multimodal_dir}/multimodal_test.npz",
        "feature_info_path": f"{multimodal_dir}/multimodal_feature_info.json",
        "leakage_report_path": (
            f"outputs/data_check/{dataset_name}/multimodal_leakage_check_report.json"
        ),
        "output_root": f"outputs/{model_name}/{dataset_name}",
    }


def resolve_graph_dataset_paths(dataset_name: str, model_name: str) -> dict:
    """从 dataset_name 和 model_name 推导 graph 数据路径。

    GraphSAGE train.py 读取逐文件路径（node_features.npy、edges.csv 等），
    并自动在 output_root 后追加 dataset_name 子目录，
    因此 output_root 设为 outputs/{model_name} 即可。
    所有路径相对于 PROJECT_ROOT。
    """
    graph_dir = f"data/graph/{dataset_name}"
    paths = {
        "graph_data_dir": graph_dir,
        "output_root": f"outputs/{model_name}",
    }
    for key, filename in GRAPH_FILE_MAP.items():
        paths[key] = f"{graph_dir}/{filename}"
    return paths


def check_leakage(dataset_name: str, model_name: str = "") -> bool:
    """训练前泄漏检查。不通过返回 False，不提供跳过选项。"""
    ok = True
    input_type = "tabular"  # default assumption

    # 判断 input_type：graph 模型走 graph_meta 检查
    if model_name == "graphsage":
        input_type = "graph"
    elif model_name == "multimodal":
        input_type = "multimodal"

    # ── 1. 检查 graph_meta（graph 模型） ─────────────────────
    if input_type == "graph":
        graph_meta_path = (
            PROJECT_ROOT / "data" / "graph" / dataset_name / "graph_meta.json"
        )
        if graph_meta_path.exists():
            with open(graph_meta_path, "r", encoding="utf-8") as f:
                graph_meta = json.load(f)
            lc = graph_meta.get("leakage_control", {})
            if lc.get("leakage_control_passed", False):
                excluded = lc.get("excluded_label_source_features", [])
                print(f"[leakage-check] graph_meta 泄漏检查通过: excluded={excluded}")
            else:
                print(
                    "[leakage-check] 错误: graph_meta leakage_control 未通过或不存在"
                )
                ok = False
                # 回退检查 feature_columns 中是否包含泄漏字段
                feature_columns = graph_meta.get("feature_columns", [])
                for col in HARDCODED_EXCLUDED_LABEL_SOURCE:
                    if col in feature_columns:
                        print(
                            f"[leakage-check] 错误: feature_columns 包含泄漏字段 '{col}'"
                        )
                        ok = False
                if "interaction_score" in feature_columns:
                    print(
                        "[leakage-check] 错误: feature_columns 包含 interaction_score"
                    )
                    ok = False
        else:
            print(f"[leakage-check] 警告: 未找到 graph_meta: {graph_meta_path}")

        if ok:
            print("[leakage-check] 全部检查通过，可以开始训练。")
        return ok

    # ── 2. multimodal 泄漏检查 ───────────────────────────────
    if input_type == "multimodal":
        multimodal_dir = PROJECT_ROOT / "data" / "multimodal" / dataset_name
        feature_info_path = multimodal_dir / "multimodal_feature_info.json"
        leakage_report_path = (
            PROJECT_ROOT / "outputs" / "data_check" / dataset_name
            / "multimodal_leakage_check_report.json"
        )

        # 2a. 检查 leakage_report_path
        if leakage_report_path.exists():
            with open(leakage_report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            if not report.get("leakage_check_passed", False):
                errors = report.get("errors", ["unknown"])
                print(f"[leakage-check] 错误: multimodal 泄漏检查未通过! errors={errors}")
                ok = False
            else:
                print(f"[leakage-check] 通过: {leakage_report_path}")
        else:
            print(f"[leakage-check] 警告: 未找到 multimodal 泄漏检查报告: {leakage_report_path}")

        # 2b. 检查 structured_feature_columns 不包含 label source 字段
        if feature_info_path.exists():
            with open(feature_info_path, "r", encoding="utf-8") as f:
                fi = json.load(f)
            struct_cols = fi.get("structured_feature_columns", [])
            for col in HARDCODED_EXCLUDED_LABEL_SOURCE:
                if col in struct_cols:
                    print(
                        f"[leakage-check] 错误: structured_feature_columns 包含泄漏字段 '{col}'"
                    )
                    ok = False
            if "interaction_score" in struct_cols:
                print(
                    "[leakage-check] 错误: interaction_score 出现在 "
                    "structured_feature_columns 中，只允许作为审计字段"
                )
                ok = False
        else:
            print(f"[leakage-check] 警告: 未找到 multimodal feature_info: {feature_info_path}")

        if ok:
            print("[leakage-check] 全部检查通过，可以开始训练。")
        return ok

    # ── 3. tabular 模型的泄漏检查 ─────────────────────────────
    # 2a. 检查 leakage_report_path
    leakage_report_path = (
        PROJECT_ROOT
        / "outputs"
        / "data_check"
        / dataset_name
        / f"{dataset_name}_leakage_check_report.json"
    )
    if leakage_report_path.exists():
        with open(leakage_report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        if not report.get("leakage_check_passed", False):
            errors = report.get("errors", ["unknown"])
            print(f"[leakage-check] 错误: 泄漏检查未通过! errors={errors}")
            ok = False
        else:
            print(f"[leakage-check] 通过: {leakage_report_path}")
    else:
        print(f"[leakage-check] 警告: 未找到泄漏检查报告: {leakage_report_path}")

    # 2b. 检查 feature_info 中不包含 label source 字段
    feature_info_path = (
        PROJECT_ROOT / "data" / "features" / dataset_name / "tabular_feature_info.json"
    )
    if feature_info_path.exists():
        with open(feature_info_path, "r", encoding="utf-8") as f:
            fi = json.load(f)
        for section in ["numeric_features", "categorical_features", "text_stat_features"]:
            for col in fi.get(section, []):
                if col in HARDCODED_EXCLUDED_LABEL_SOURCE:
                    print(
                        f"[leakage-check] 错误: feature_info.{section} 包含泄漏字段 '{col}'"
                    )
                    ok = False
        # 检查 interaction_score 是否在数值特征中
        if "interaction_score" in fi.get("numeric_features", []):
            print(
                "[leakage-check] 错误: interaction_score 出现在 numeric_features 中，"
                "只允许作为审计字段"
            )
            ok = False
    else:
        print(f"[leakage-check] 警告: 未找到 feature_info: {feature_info_path}")

    if ok:
        print("[leakage-check] 全部检查通过，可以开始训练。")
    return ok


def validate_paths(resolved: dict) -> bool:
    """验证 resolved config 中的数据路径是否存在。"""
    ok = True
    input_type = resolved.get("input_type", "tabular")

    if input_type == "graph":
        graph_keys = [
            "node_features_path", "labels_path", "train_mask_path",
            "val_mask_path", "edge_path", "node_path", "graph_meta_path",
        ]
        for key in graph_keys:
            path = PROJECT_ROOT / resolved[key]
            if not path.exists():
                print(f"[路径检查] 错误: {key} 不存在: {path}")
                ok = False
            else:
                print(f"[路径检查] 通过: {key} -> {path}")
        # test_mask_path 是可选字段
        if "test_mask_path" in resolved and resolved["test_mask_path"]:
            test_path = PROJECT_ROOT / resolved["test_mask_path"]
            if not test_path.exists():
                print(f"[路径检查] 错误: test_mask_path 不存在: {test_path}")
                ok = False
            else:
                print(f"[路径检查] 通过: test_mask_path -> {test_path}")
    elif input_type == "multimodal":
        for key in ["train_npz_path", "val_npz_path", "test_npz_path", "feature_info_path"]:
            path = PROJECT_ROOT / resolved[key]
            if not path.exists():
                print(f"[路径检查] 错误: {key} 不存在: {path}")
                ok = False
            else:
                print(f"[路径检查] 通过: {key} -> {path}")
    else:
        for key in ["train_data_path", "val_data_path", "test_data_path", "feature_info_path"]:
            path = PROJECT_ROOT / resolved[key]
            if not path.exists():
                print(f"[路径检查] 错误: {key} 不存在: {path}")
                ok = False
            else:
                print(f"[路径检查] 通过: {key} -> {path}")
    return ok


def build_resolved_config(
    dataset_name: str,
    model_name: str,
    model_config: dict,
) -> dict:
    """将统一模型配置 + dataset 路径解析合并为模型 train.py 接受的 resolved config。"""
    input_type = model_config.get("model", {}).get("input_type", "tabular")
    resolved = {
        # 模型身份
        "model_name": model_name,
        "dataset_name": dataset_name,
        "dataset_variant": "default",
        "input_type": input_type,
        # 通用字段（模型 train.py 可能读取也可能忽略）
        "activation": "relu",
        "optimizer": "adam",
        "loss": "bce",
    }

    # 从模型配置提取超参数（模型特有字段如 hidden_units / deep_hidden_units 等）
    default_params = model_config.get("model", {}).get("default_params", {})
    for key, value in default_params.items():
        resolved[key] = value

    # 从 training 块提取参数（通过 FIELD_MAP 映射字段名）
    training_cfg = model_config.get("training", {})
    for unified_key, config_key in FIELD_MAP.items():
        if unified_key in training_cfg:
            resolved[config_key] = training_cfg[unified_key]

    # 从 output 块提取 root_template 并展开
    output_cfg = model_config.get("output", {})
    root_template = output_cfg.get("root_template", "outputs/{model_name}/{dataset_name}")

    if input_type == "graph":
        # GraphSAGE train.py 自动追加 dataset_name 子目录，所以 output_root 只到模型级
        resolved["output_root"] = f"outputs/{model_name}"
        # graph 路径解析
        paths = resolve_graph_dataset_paths(dataset_name, model_name)
        resolved.update(paths)
        # 将 flat early_stopping_patience 转换为嵌套结构（GraphSAGE train.py 期望）
        patience = resolved.pop("early_stopping_patience", 10)
        resolved["early_stopping"] = {
            "enabled": True,
            "patience": patience,
        }
        # 加入 feature_normalization 嵌套配置
        fn_config = training_cfg.get("feature_normalization", {})
        if fn_config:
            resolved["feature_normalization"] = fn_config
    elif input_type == "multimodal":
        # Multimodal train.py 使用 output_root / run_id 输出
        resolved["output_root"] = root_template.format(model_name=model_name, dataset_name=dataset_name)
        # multimodal NPZ 路径解析
        paths = resolve_multimodal_dataset_paths(dataset_name, model_name)
        resolved.update(paths)
        # 字段名映射: text_hidden → text_hidden_dim 等
        for unified_key, config_key in MULTIMODAL_FIELD_MAP.items():
            if unified_key in resolved:
                resolved[config_key] = resolved.pop(unified_key)
    else:
        resolved["output_root"] = root_template.format(model_name=model_name, dataset_name=dataset_name)
        # dataset 路径解析（覆盖上面设置的 output_root，保持一致）
        paths = resolve_tabular_dataset_paths(model_name, dataset_name)
        resolved.update(paths)

    return resolved


def apply_overrides(resolved: dict, override_list: list[str]) -> dict:
    """应用 --override 参数到 resolved config。"""
    for kv in override_list:
        if "=" not in kv:
            print(f"[警告] 跳过无效 override 格式: '{kv}'（应为 key=value）")
            continue
        key, value_str = kv.split("=", 1)
        value = parse_override_value(value_str)
        old_val = resolved.get(key, "<未设置>")
        resolved[key] = value
        print(f"[override] {key}: {old_val} -> {value}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统一训练入口 — Batch 6E (支持 DNN / Wide & Deep / GraphSAGE / Multimodal)"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="数据集名称，如 real_raw_5000（必选）",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="模型名称，当前支持 dnn / wide_deep / graphsage / multimodal（必选）",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="自定义模型配置文件路径（覆盖 configs/models/<model>.yaml）",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="覆盖超参数，格式 key=value，可多次使用",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只解析配置、检查泄漏和路径，不执行训练",
    )
    args = parser.parse_args()

    # ── Step 1: 参数校验 ────────────────────────────────────
    model_name = args.model
    dataset_name = args.dataset

    if args.model not in SUPPORTED_MODELS:
        print(f"[错误] 当前仅支持 {'/'.join(sorted(SUPPORTED_MODELS))}，"
              f"收到 --model {args.model}")
        sys.exit(1)

    print(f"[Batch 6D] 数据集: {dataset_name}, 模型: {model_name}")

    # ── Step 2: 加载 datasets.yaml ──────────────────────────
    datasets_path = PROJECT_ROOT / "configs" / "datasets.yaml"
    datasets_cfg = load_yaml(datasets_path)

    if dataset_name not in datasets_cfg.get("datasets", {}):
        print(f"[错误] 数据集 '{dataset_name}' 未在 configs/datasets.yaml 中注册")
        available = list(datasets_cfg.get("datasets", {}).keys())
        print(f"  可用数据集: {available}")
        sys.exit(1)

    # ── Step 3: 加载模型配置 ────────────────────────────────
    if args.config:
        model_config_path = Path(args.config)
        if not model_config_path.is_absolute():
            model_config_path = PROJECT_ROOT / model_config_path
        print(f"[配置] 使用自定义配置: {model_config_path}")
    else:
        model_config_path = PROJECT_ROOT / "configs" / "models" / f"{model_name}.yaml"
        print(f"[配置] 使用默认配置: {model_config_path}")

    if not model_config_path.exists():
        print(f"[错误] 模型配置不存在: {model_config_path}")
        sys.exit(1)

    model_config = load_yaml(model_config_path)

    # ── Step 4: 构建 resolved config ────────────────────────
    resolved = build_resolved_config(dataset_name, model_name, model_config)

    # ── Step 5: 应用 --override ─────────────────────────────
    if args.override:
        print(f"[配置] 应用 {len(args.override)} 个 override 参数")
        resolved = apply_overrides(resolved, args.override)

    # ── Step 6: 泄漏检查（强制，不可跳过） ──────────────────
    print("\n" + "=" * 60)
    print("泄漏检查")
    print("=" * 60)
    if not check_leakage(dataset_name, model_name):
        print("\n[错误] 泄漏检查未通过，中止训练。")
        sys.exit(1)

    # ── Step 7: 路径验证 ────────────────────────────────────
    print("\n" + "=" * 60)
    print("路径检查")
    print("=" * 60)
    if not validate_paths(resolved):
        print("\n[错误] 数据路径不完整，中止训练。")
        sys.exit(1)

    # ── Step 8: 保存 resolved config ────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    resolved_config_dir = (
        PROJECT_ROOT
        / "outputs"
        / "training_configs"
        / model_name
        / dataset_name
    )
    resolved_config_path = resolved_config_dir / f"{timestamp}_resolved.yaml"
    resolved_config_dir.mkdir(parents=True, exist_ok=True)

    with open(resolved_config_path, "w", encoding="utf-8") as f:
        yaml.dump(resolved, f, default_flow_style=False, allow_unicode=True)
    print(f"\n[配置] Resolved config 已保存: {resolved_config_path}")

    # ── Step 9: Dry run? ────────────────────────────────────
    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN — 未执行训练")
        print("=" * 60)
        print(yaml.dump(resolved, default_flow_style=False, allow_unicode=True))
        print(f"\n运行以下命令可执行训练:")
        print(f"  python src/training/train.py "
              f"--dataset {dataset_name} --model {model_name}")
        return

    # ── Step 10: 调用模型训练脚本 ───────────────────────────
    train_script = PROJECT_ROOT / MODEL_SCRIPTS[model_name]
    if not train_script.exists():
        print(f"[错误] 训练脚本不存在: {train_script}")
        sys.exit(1)

    cmd = [
        sys.executable,
        str(train_script),
        "--config",
        str(resolved_config_path),
    ]

    print(f"\n[训练] 启动 {model_name} 训练...")
    print(f"[训练] 命令: {' '.join(cmd)}")
    print(f"[训练] Config: {resolved_config_path}")
    print()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print(f"\n[错误] {model_name} 训练失败，返回码: {result.returncode}")
        sys.exit(result.returncode)

    # ── Step 11: 输出摘要 ───────────────────────────────────
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"  Dataset:     {dataset_name}")
    print(f"  Model:       {model_name}")
    print(f"  Config:      {resolved_config_path}")
    print(f"  输出目录:    outputs/{model_name}/{dataset_name}/<run_id>/")
    print()
    print("检查输出文件:")
    print(f"  outputs/{model_name}/{dataset_name}/<run_id>/model.pt")
    print(f"  outputs/{model_name}/{dataset_name}/<run_id>/metrics.json")
    print(f"  outputs/{model_name}/{dataset_name}/<run_id>/predictions_val.csv")
    print(f"  outputs/{model_name}/{dataset_name}/<run_id>/predictions_test.csv")
    print(f"  outputs/{model_name}/{dataset_name}/<run_id>/train_log.csv")
    print(f"  outputs/{model_name}/{dataset_name}/<run_id>/feature_config_used.json")
    print(f"  outputs/{model_name}/{dataset_name}/<run_id>/run_meta.json")


if __name__ == "__main__":
    main()