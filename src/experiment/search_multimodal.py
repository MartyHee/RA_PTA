"""Multimodal 随机搜索超参调优 — 通用入口（数据集无关）。

本脚本通过 subprocess 调用现有 train.py（src/models/multimodal/train.py），
以 val AUC 为主排序指标。与 tune_multimodal_random_search.py 的区别：

  1. 不依赖 dataset-specific 调优配置文件。
  2. 数据集路径从 configs/datasets.yaml 注册表自动解析。
  3. 输出目录由 dataset_name + dataset_variant 自动生成。
  4. 新增数据集时只需更新 datasets.yaml，无需新增搜索脚本。

用法:
    cd D:/CodeData/Program Coding/ByteDance/RA_PTA
    D:/CodeData/software/Anaconda/Anaconda3/envs/ra/python.exe \\
        src/experiment/search_multimodal.py \\
        --config configs/experiments/multimodal_search.yaml \\
        --num-trials 20

参数:
    --config      实验配置文件路径（相对于项目根目录）
    --num-trials  搜索 trial 数（覆盖配置文件中的默认值）
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# ── 路径设置 ──────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
sys.path.insert(0, os.path.join(_project_root, "src"))

from utils.config import load_config  # noqa: E402
from utils.dataset_registry import (  # noqa: E402
    build_default_output_root,
    get_dataset_info,
    resolve_train_config,
)
from utils.logger import get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

logger = get_logger("search_multimodal")

# ── 训练脚本路径 ──────────────────────────────────────────
TRAIN_SCRIPT = os.path.join(_project_root, "src", "models", "multimodal", "train.py")


# ============================================================================
# 超参采样
# ============================================================================

def sample_hyperparams(
    search_space: dict,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """从搜索空间中随机采样一组超参数。"""
    params: dict[str, Any] = {}
    for name, space in search_space.items():
        if "values" in space:
            vals = list(space["values"])
            idx = rng.integers(0, len(vals))
            params[name] = vals[idx]
        elif "scale" in space:
            low = float(space["min"])
            high = float(space["max"])
            if space["scale"] == "log":
                log_val = rng.uniform(np.log(low), np.log(high))
                params[name] = float(np.exp(log_val))
            else:
                params[name] = float(rng.uniform(low, high))
        else:
            params[name] = float(rng.uniform(space["min"], space["max"]))
    return params


def _map_hidden_dim(hidden_dim: int) -> dict[str, int]:
    """将抽象的 hidden_dim 映射到各分支维度。"""
    return {
        "text_hidden_dim": hidden_dim,
        "visual_hidden_dim": max(8, hidden_dim // 2),
        "structured_hidden_dim": hidden_dim,
        "fusion_hidden_dim": hidden_dim * 2,
    }


def build_trial_extra_params(params: dict) -> dict:
    """将采样参数转换为训练配置可用的 extra_params。"""
    extra = {
        "learning_rate": float(params["learning_rate"]),
        "weight_decay": float(params["weight_decay"]),
        "dropout": float(params["dropout"]),
        "batch_size": int(params["batch_size"]),
    }
    hidden_dim = int(params["hidden_dim"])
    extra.update(_map_hidden_dim(hidden_dim))
    return extra


# ============================================================================
# Trial 输出读取
# ============================================================================

def read_trial_metrics(
    trial_output_root: Path,
) -> tuple[dict | None, Path | None]:
    """读取一个 trial 的输出 metrics.json。"""
    latest_txt = trial_output_root / "latest_run.txt"
    if not latest_txt.exists():
        logger.warning(f"  latest_run.txt not found: {latest_txt}")
        return None, None

    trial_run_id = latest_txt.read_text().strip()
    trial_run_dir = trial_output_root / trial_run_id
    metrics_path = trial_run_dir / "metrics.json"

    if not metrics_path.exists():
        logger.warning(f"  metrics.json not found: {metrics_path}")
        return None, trial_run_dir

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics, trial_run_dir


# ============================================================================
# 汇总输出
# ============================================================================

def _write_summary_md(
    path: Path,
    tune_run_id: str,
    experiment_config: dict,
    search_space: dict,
    num_trials: int,
    num_completed: int,
    num_failed: int,
    best_row: dict | None,
    trials_df: pd.DataFrame,
    dataset_name: str,
) -> None:
    """生成 tuning_summary.md。"""
    lines: list[str] = [
        f"# Multimodal 随机搜索调优摘要（{dataset_name}）",
        "",
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 调优 Run ID：{tune_run_id}",
        f"> Trial 数：{num_trials}",
        "",
        "## 1. 调优设置",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        f"| 实验配置 | `{experiment_config.get('experiment_name', '')}` |",
        f"| 数据集 | {dataset_name} |",
        f"| Variant | {experiment_config.get('dataset_variant', '')} |",
        f"| Trial 数 | {num_trials}（请求） / "
        f"完成 {num_completed}，失败 {num_failed} |",
        f"| 随机种子 | {experiment_config.get('random_seed', 2026)} |",
        f"| 排序指标（主） | {experiment_config.get('primary_metric', 'val_auc')} |",
        f"| 排序指标（次） | {experiment_config.get('secondary_metric', 'val_f1')} |",
        "",
        "### 搜索空间",
        "",
        "| 参数 | 范围 | 采样方式 |",
        "|---|---|---|",
    ]
    for name, space in search_space.items():
        if "values" in space:
            lines.append(f"| {name} | {space['values']} | 离散均匀 |")
        elif "scale" in space:
            lines.append(
                f"| {name} | [{space['min']}, {space['max']}] | {space['scale']} |"
            )
        else:
            lines.append(
                f"| {name} | [{space['min']}, {space['max']}] | uniform |"
            )

    lines += ["", "## 2. 最佳 Trial"]

    if best_row is not None:
        lines += [
            "",
            "| 项目 | 值 |",
            "|---|---|",
            f"| Trial ID | {int(best_row['trial_id'])} |",
            f"| 状态 | {best_row.get('status', 'unknown')} |",
        ]
        for key in ["learning_rate", "weight_decay", "dropout",
                     "hidden_dim", "batch_size"]:
            val = best_row.get(key)
            if isinstance(val, float):
                lines.append(f"| {key} | {val:.6f} |")
            else:
                lines.append(f"| {key} | {val} |")

        lines += [
            f"| Best Epoch | {best_row.get('best_epoch', 'N/A')} |",
            f"| Val AUC | {best_row.get('val_auc', 'N/A'):.4f} |",
            f"| Val F1 | {best_row.get('val_f1', 'N/A'):.4f} |",
            f"| Test AUC | {best_row.get('test_auc', 'N/A'):.4f} |",
            f"| Test F1 | {best_row.get('test_f1', 'N/A'):.4f} |",
            f"| 输出目录 | `{best_row.get('output_dir', '')}` |",
        ]
    else:
        lines += ["", "（无成功完成的 trial）"]

    lines += [
        "",
        "## 3. 全部 Trial 汇总",
        "",
    ]

    if not trials_df.empty:
        lines.append(
            "| Trial ID | Status | val_auc | val_f1 | "
            "test_auc | test_f1 | lr | wd | dropout | "
            "hidden_dim | batch_size |"
        )
        lines.append(
            "|----------|--------|---------|--------|"
            "----------|---------|-----|----|--------|"
            "-----------|------------|"
        )

        for _, row in trials_df.iterrows():
            status = row.get("status", "unknown")
            tid = int(row["trial_id"])
            lr = row.get("learning_rate", "N/A")
            wd = row.get("weight_decay", "N/A")
            dr = row.get("dropout", "N/A")
            hd = row.get("hidden_dim", "N/A")
            bs = row.get("batch_size", "N/A")

            if status == "completed":
                lines.append(
                    f"| {tid} | {status} "
                    f"| {row.get('val_auc', 'N/A'):.4f} "
                    f"| {row.get('val_f1', 'N/A'):.4f} "
                    f"| {row.get('test_auc', 'N/A'):.4f} "
                    f"| {row.get('test_f1', 'N/A'):.4f} "
                    f"| {lr:.6f} | {wd:.6f} | {dr:.3f} "
                    f"| {int(hd) if pd.notna(hd) else 'N/A'} "
                    f"| {int(bs) if pd.notna(bs) else 'N/A'} |"
                )
            else:
                lines.append(
                    f"| {tid} | {status} | - | - | - | - "
                    f"| {lr} | {wd} | {dr} | {hd} | {bs} |"
                )
    else:
        lines.append("（无 trial 记录）")

    lines += [
        "",
        "## 4. 主要限制",
        "",
        f"- 当前调优仅基于 {num_trials} trials，搜索不充分。",
        "- 标签为 interaction_score 伪标签，不代表真实业务目标。",
        "- 所有结果均为离线实验，不代表线上推荐效果。",
        "- Test 指标仅供参考，best trial 选择仅依据 val 指标。",
        "- 部分 trials 可能因配置不兼容失败，失败原因见 tuning_trials.csv。",
        "",
        "## 5. 下一步建议",
        "",
        "- 增加 trial 数以获得更充分的搜索覆盖。",
        "- 根据最佳 trial 的收敛位置缩小搜索空间。",
        "- 扩展搜索空间参数（如 epochs、early_stopping_patience、optimizer 类型）。",
        "- 验证最佳配置在 val 和 test 上的稳定性（多 seed 重复）。",
    ]

    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"调优报告: {path}")


# ============================================================================
# 主程序
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal 随机搜索超参调优（通用入口）"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="实验配置文件路径（相对于项目根目录），如 "
             "configs/experiments/multimodal_search.yaml",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="搜索 trial 数（覆盖配置文件中的默认值）",
    )
    args = parser.parse_args()

    # ── 1. 加载实验配置 ─────────────────────────────────────
    experiment_config = load_config(args.config)
    logger.info(f"实验配置加载完成: {args.config}")

    dataset_name = experiment_config["dataset_name"]
    dataset_variant = experiment_config.get("dataset_variant", "")
    base_model_config_path = experiment_config.get("base_model_config_path", "")
    search_space = experiment_config.get("search_space", {})
    num_trials = args.num_trials or experiment_config.get("num_trials", 10)
    random_seed = experiment_config.get("random_seed", 2026)
    primary_metric = experiment_config.get("primary_metric", "val_auc")
    secondary_metric = experiment_config.get("secondary_metric", "val_f1")
    model_defaults = experiment_config.get("model_defaults", {})

    logger.info(f"数据集: {dataset_name} (variant={dataset_variant})")
    logger.info(f"基础模型配置: {base_model_config_path}")
    logger.info(f"随机搜索: {num_trials} trials, seed={random_seed}")
    logger.info(f"搜索空间 keys: {list(search_space.keys())}")

    # ── 2. 从 registry 查询数据集路径 ───────────────────────
    logger.info("从 datasets.yaml 注册表查询数据集路径...")
    dataset_info = get_dataset_info(dataset_name, variant=dataset_variant)
    logger.info(f"  train_npz_path: {dataset_info.get('train_npz_path')}")
    logger.info(f"  val_npz_path: {dataset_info.get('val_npz_path')}")
    logger.info(f"  test_npz_path: {dataset_info.get('test_npz_path')}")
    logger.info(f"  feature_info_path: {dataset_info.get('feature_info_path')}")

    # ── 3. 加载基础模型配置 ─────────────────────────────────
    base_config = {}
    if base_model_config_path:
        base_config = load_config(base_model_config_path)
        logger.info(f"基础模型配置加载完成: {base_model_config_path}")

    # 合并 model_defaults（实验配置中的默认值覆盖 base_config）
    if model_defaults:
        base_config.update(model_defaults)

    # ── 4. 创建调优输出目录 ─────────────────────────────────
    tune_run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    tuning_output_root = build_default_output_root(
        dataset_name, variant=dataset_variant,
    )
    tuning_output_dir = Path(_project_root) / tuning_output_root / tune_run_id
    tuning_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"调优输出目录: {tuning_output_dir}")

    # ── 5. 随机数生成器 ─────────────────────────────────────
    set_seed(random_seed)
    rng = np.random.default_rng(random_seed)

    # ── 6. Trial 循环 ───────────────────────────────────────
    trials: list[dict[str, Any]] = []
    python_exe = sys.executable

    for trial_id in range(num_trials):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Trial {trial_id + 1} / {num_trials}")
        logger.info(f"{'=' * 60}")

        # 6a. 采样超参数
        params = sample_hyperparams(search_space, rng)
        logger.info(
            f"  超参数: "
            f"lr={params['learning_rate']:.6f}, "
            f"wd={params['weight_decay']:.6f}, "
            f"dropout={params['dropout']:.3f}, "
            f"hidden_dim={params['hidden_dim']}, "
            f"batch_size={params['batch_size']}"
        )

        # 6b. Trial 输出目录（相对于项目根目录）
        trial_output_root = str(
            tuning_output_dir / f"trial_{trial_id}"
        )

        # 6c. 构建 trial 完整配置
        trial_seed = random_seed + trial_id * 7 + 13
        extra_params = build_trial_extra_params(params)
        extra_params["random_seed"] = trial_seed
        extra_params["output_root"] = trial_output_root

        trial_config = resolve_train_config(base_config, dataset_info, extra_params)

        # 6d. 写入临时配置 YAML
        temp_config_path = tuning_output_dir / f"trial_{trial_id}_config.yaml"
        with open(temp_config_path, "w", encoding="utf-8") as f:
            yaml.dump(trial_config, f, default_flow_style=False, sort_keys=False)

        # 6e. 运行 train.py (subprocess)
        logger.info(f"  运行 train.py ...")
        cmd = [python_exe, TRAIN_SCRIPT, "--config", str(temp_config_path)]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=_project_root,
                timeout=600,  # 每个 trial 最大 10 分钟
            )

            # 保存 stdout/stderr 日志
            log_path = tuning_output_dir / f"trial_{trial_id}_log.txt"
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)

            if result.returncode != 0:
                logger.error(
                    f"  Trial {trial_id} 失败, returncode={result.returncode}"
                )
                stderr_tail = result.stderr[-500:] if result.stderr else ""
                logger.error(f"  stderr tail: {stderr_tail}")
                trials.append({
                    "trial_id": trial_id,
                    "status": "failed",
                    "error": stderr_tail,
                    **params,
                })
                continue

            logger.info(f"  Train.py 完成 (returncode=0)")

        except subprocess.TimeoutExpired:
            logger.error(f"  Trial {trial_id} 超时（>10 min）")
            trials.append({
                "trial_id": trial_id,
                "status": "timeout",
                "error": "subprocess timeout after 600s",
                **params,
            })
            continue
        except Exception as e:
            logger.error(f"  Trial {trial_id} 异常: {e}")
            traceback.print_exc()
            trials.append({
                "trial_id": trial_id,
                "status": "error",
                "error": str(e),
                **params,
            })
            continue

        # 6f. 读取 trial 结果
        trial_metrics, trial_run_dir = read_trial_metrics(
            Path(trial_output_root)
        )

        if trial_metrics is None:
            logger.warning(f"  Trial {trial_id} metrics 未找到，标记为失败")
            trials.append({
                "trial_id": trial_id,
                "status": "no_metrics",
                **params,
            })
            continue

        # 6g. 提取指标
        val_m = trial_metrics.get("val_metrics", {})
        test_m = trial_metrics.get("test_metrics", {})

        best_epoch: int | None = None
        run_meta_path = (
            trial_run_dir / "run_meta.json" if trial_run_dir else None
        )
        if run_meta_path and run_meta_path.exists():
            with open(run_meta_path, "r", encoding="utf-8") as f:
                run_meta = json.load(f)
            best_epoch = run_meta.get("best_epoch")

        trial_record: dict[str, Any] = {
            "trial_id": trial_id,
            "status": "completed",
            "learning_rate": params["learning_rate"],
            "weight_decay": params["weight_decay"],
            "dropout": params["dropout"],
            "hidden_dim": int(params["hidden_dim"]),
            "batch_size": int(params["batch_size"]),
            "best_epoch": best_epoch,
            "val_auc": val_m.get("auc"),
            "val_f1": val_m.get("f1"),
            "test_auc": test_m.get("auc"),
            "test_f1": test_m.get("f1"),
            "output_dir": str(trial_run_dir) if trial_run_dir else "",
        }
        trials.append(trial_record)

        logger.info(
            f"  val_auc={trial_record['val_auc']:.4f}, "
            f"val_f1={trial_record['val_f1']:.4f}, "
            f"test_auc={trial_record['test_auc']:.4f}, "
            f"test_f1={trial_record['test_f1']:.4f}"
        )

    # ── 7. 汇总结果 ─────────────────────────────────────────
    logger.info(f"\n{'=' * 60}")
    logger.info(f"调优完成，共 {len(trials)} trials")

    trials_df = pd.DataFrame(trials)

    completed_mask = trials_df["status"] == "completed"
    completed_df = trials_df[completed_mask].copy()
    failed_df = trials_df[~completed_mask].copy()

    if not completed_df.empty:
        completed_df = completed_df.sort_values("val_auc", ascending=False)

    trials_sorted = pd.concat(
        [completed_df, failed_df], ignore_index=True
    )
    trials_csv_path = tuning_output_dir / "tuning_trials.csv"
    trials_sorted.to_csv(trials_csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Trials 汇总: {trials_csv_path} ({len(trials_sorted)} rows)")

    # ── 8. 选择最佳 trial + 输出 best_config ────────────────
    best_row: dict | None = None
    if completed_df.empty:
        logger.error("没有成功完成的 trial，无法输出最佳配置")
        for fname in ["best_config.yaml", "best_metrics.json"]:
            (tuning_output_dir / fname).write_text("", encoding="utf-8")
    else:
        best_row = completed_df.iloc[0].to_dict()
        logger.info(
            f"最佳 trial: trial_id={best_row['trial_id']}, "
            f"val_auc={best_row['val_auc']:.4f}, "
            f"val_f1={best_row['val_f1']:.4f}"
        )

        # 最佳配置
        best_extra = build_trial_extra_params(best_row)
        best_extra["random_seed"] = random_seed
        best_config = resolve_train_config(base_config, dataset_info, best_extra)
        best_config_path = tuning_output_dir / "best_config.yaml"
        with open(best_config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                best_config, f, default_flow_style=False, sort_keys=False,
            )
        logger.info(f"最佳配置: {best_config_path}")

        # 最佳指标
        best_metrics: dict[str, Any] = {
            "tune_run_id": tune_run_id,
            "experiment_config": args.config,
            "dataset_name": dataset_name,
            "dataset_variant": dataset_variant,
            "best_trial_id": int(best_row["trial_id"]),
            "primary_metric": primary_metric,
            "secondary_metric": secondary_metric,
            "best_hyperparams": {
                "learning_rate": best_row["learning_rate"],
                "weight_decay": best_row["weight_decay"],
                "dropout": best_row["dropout"],
                "hidden_dim": int(best_row["hidden_dim"]),
                "batch_size": int(best_row["batch_size"]),
            },
            "val_metrics": {
                "auc": best_row.get("val_auc"),
                "f1": best_row.get("val_f1"),
            },
            "test_metrics": {
                "auc": best_row.get("test_auc"),
                "f1": best_row.get("test_f1"),
            },
            "num_completed_trials": len(completed_df),
            "num_failed_trials": len(failed_df),
            "warnings": [
                "Best trial 仅基于 val 指标选择，test 指标仅供参考。",
                "当前调优基于 interaction_score 伪标签，不代表真实业务目标。",
                "所有结果均为离线实验，不代表线上推荐效果。",
            ],
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        best_metrics_path = tuning_output_dir / "best_metrics.json"
        with open(best_metrics_path, "w", encoding="utf-8") as f:
            json.dump(best_metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"最佳指标: {best_metrics_path}")

    # ── 9. 保存搜索空间快照 ─────────────────────────────────
    search_space_path = tuning_output_dir / "search_space_used.json"
    with open(search_space_path, "w", encoding="utf-8") as f:
        json.dump(search_space, f, ensure_ascii=False, indent=2)

    # ── 10. 生成 tuning_summary.md ─────────────────────────
    _write_summary_md(
        path=tuning_output_dir / "tuning_summary.md",
        tune_run_id=tune_run_id,
        experiment_config=experiment_config,
        search_space=search_space,
        num_trials=num_trials,
        num_completed=len(completed_df),
        num_failed=len(trials) - len(completed_df),
        best_row=best_row,
        trials_df=trials_sorted,
        dataset_name=dataset_name,
    )

    # ── 11. 打印最终摘要 ───────────────────────────────────
    logger.info(f"\n{'=' * 60}")
    logger.info(f"随机搜索调优完成")
    logger.info(f"  Dataset:      {dataset_name} ({dataset_variant})")
    logger.info(f"  Run ID:       {tune_run_id}")
    logger.info(f"  输出目录:     {tuning_output_dir}")
    logger.info(f"  Trial 完成:   {len(completed_df)} / {len(trials)}")
    if best_row is not None:
        logger.info(f"  Best Trial:   {int(best_row['trial_id'])}")
        logger.info(f"  Val AUC:      {best_row.get('val_auc', 'N/A'):.4f}")
        logger.info(f"  Val F1:       {best_row.get('val_f1', 'N/A'):.4f}")
        logger.info(f"  Test AUC:     {best_row.get('test_auc', 'N/A'):.4f}")
        logger.info(f"  Test F1:      {best_row.get('test_f1', 'N/A'):.4f}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()