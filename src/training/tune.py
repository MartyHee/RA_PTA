#!/usr/bin/env python
"""
统一超参调优入口 — Batch 9B (第一版: DNN random search)

本脚本是 train.py 的上层调度器，不实现训练逻辑。
职责：搜索空间采样 → 生成 trial config → 调用 train.py → 汇总结果 → 选择最佳 trial。

用法:
    # dry-run 预览 3 个 trial 配置
    python src/training/tune.py --dataset real_raw_5000 --model dnn --num-trials 3 --dry-run

    # 真实搜索 3 trials（小规模验证）
    python src/training/tune.py --dataset real_raw_5000 --model dnn --num-trials 3

    # 完整搜索（默认 20 trials）
    python src/training/tune.py --dataset real_raw_5000 --model dnn

    # 自定义搜索空间 + 排序指标
    python src/training/tune.py --dataset real_raw_5000 --model dnn ^
        --search-config configs/tuning/dnn_random.yaml --num-trials 10 ^
        --metric val_loss --direction minimize
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
ALLOWED_DATASETS = frozenset({"real_raw_5000"})
ALLOWED_MODELS = frozenset({"dnn"})
ALLOWED_DIRECTIONS = frozenset({"maximize", "minimize"})

DEFAULT_PYTHON = "D:/CodeData/software/Anaconda/Anaconda3/envs/ra/python.exe"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# trials_summary.csv 字段顺序
CSV_FIELDS = [
    "trial_id",
    "run_id",
    "status",
    "val_auc",
    "val_loss",
    "val_accuracy",
    "val_f1",
    "val_precision",
    "val_recall",
    "test_auc",
    "test_f1",
    "learning_rate",
    "weight_decay",
    "dropout",
    "hidden_units",
    "batch_size",
    "epochs",
    "early_stopping_patience",
    "params_json",
    "output_dir",
    "error_message",
]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    """加载 YAML 文件，不存在则报错退出。"""
    if not path.exists():
        print(f"[错误] 配置文件不存在: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data else {}


def validate_search_config(cfg: dict, config_path: str) -> None:
    """验证搜索空间配置结构是否完整合法。"""
    if "search" not in cfg:
        print(f"[错误] 搜索配置缺少 'search' 块: {config_path}", file=sys.stderr)
        sys.exit(1)
    if "space" not in cfg:
        print(f"[错误] 搜索配置缺少 'space' 块: {config_path}", file=sys.stderr)
        sys.exit(1)

    search = cfg["search"]
    for key in ("method", "num_trials", "metric", "direction", "seed"):
        if key not in search:
            print(f"[错误] search 块缺少 '{key}': {config_path}", file=sys.stderr)
            sys.exit(1)

    if search["method"] != "random":
        print(f"[错误] 当前仅支持 method=random，收到: {search['method']}", file=sys.stderr)
        sys.exit(1)

    if search["direction"] not in ALLOWED_DIRECTIONS:
        print(f"[错误] direction 必须是 {sorted(ALLOWED_DIRECTIONS)}，收到: {search['direction']}", file=sys.stderr)
        sys.exit(1)

    space = cfg["space"]
    for param_name, param_cfg in space.items():
        if "type" not in param_cfg:
            print(f"[错误] space.{param_name} 缺少 'type' 字段", file=sys.stderr)
            sys.exit(1)
        ptype = param_cfg["type"]
        if ptype not in ("uniform", "loguniform", "choice", "fixed"):
            print(f"[错误] space.{param_name} 不支持的类型: {ptype}", file=sys.stderr)
            sys.exit(1)
        if ptype in ("uniform", "loguniform"):
            if "low" not in param_cfg or "high" not in param_cfg:
                print(f"[错误] space.{param_name} ({ptype}) 需要 low/high", file=sys.stderr)
                sys.exit(1)
        elif ptype == "choice":
            if "values" not in param_cfg:
                print(f"[错误] space.{param_name} (choice) 需要 values", file=sys.stderr)
                sys.exit(1)
        elif ptype == "fixed":
            if "value" not in param_cfg:
                print(f"[错误] space.{param_name} (fixed) 需要 value", file=sys.stderr)
                sys.exit(1)


def sample_param(param_name: str, param_cfg: dict, rng: random.Random):
    """从单个参数配置中采样一个值。"""
    ptype = param_cfg["type"]
    if ptype == "uniform":
        return rng.uniform(param_cfg["low"], param_cfg["high"])
    elif ptype == "loguniform":
        log_low = math.log(param_cfg["low"])
        log_high = math.log(param_cfg["high"])
        return math.exp(rng.uniform(log_low, log_high))
    elif ptype == "choice":
        return rng.choice(param_cfg["values"])
    elif ptype == "fixed":
        return param_cfg["value"]
    else:
        raise ValueError(f"不支持的采样类型: {ptype}")


def get_run_dirs(model: str, dataset: str) -> list[Path]:
    """获取 outputs/<model>/<dataset>/ 下已有的 run 目录。"""
    run_dir = PROJECT_ROOT / "outputs" / model / dataset
    if not run_dir.is_dir():
        return []
    return sorted(
        [d for d in run_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )


def find_new_run_id(before: list[Path], after: list[Path]) -> str | None:
    """对比前后目录列表，找出新增的 run_id（按 mtime 取最新）。"""
    before_names = {d.name for d in before}
    after_names = {d.name for d in after}
    new_names = after_names - before_names
    if not new_names:
        return None
    candidates = [d for d in after if d.name in new_names]
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return newest.name


def read_metrics(model: str, dataset: str, run_id: str) -> dict:
    """读取指定 run 的 metrics.json，返回扁平化的指标字典。"""
    metrics_path = (
        PROJECT_ROOT / "outputs" / model / dataset / run_id / "metrics.json"
    )
    if not metrics_path.exists():
        print(f"  [警告] metrics.json 不存在: {metrics_path}")
        return {}
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        print(f"  [警告] 读取 metrics.json 失败: {e}")
        return {}

    flat = {}
    # val 指标
    vm = m.get("val_metrics", {})
    if vm:
        flat["val_auc"] = vm.get("auc")
        flat["val_loss"] = vm.get("eval_loss")
        flat["val_accuracy"] = vm.get("accuracy")
        flat["val_f1"] = vm.get("f1")
        flat["val_precision"] = vm.get("precision")
        flat["val_recall"] = vm.get("recall")
    # test 指标
    tm = m.get("test_metrics", {})
    if tm:
        flat["test_auc"] = tm.get("auc")
        flat["test_f1"] = tm.get("f1")
    return flat


def param_to_str(value) -> str:
    """将参数值转换为 CSV 友好字符串。"""
    if isinstance(value, list):
        return json.dumps(value)
    if isinstance(value, float):
        # 避免过长小数
        return f"{value:.8g}"
    return str(value)


def format_value_for_override(value):
    """将采样值转换为 --override 可用的字符串。"""
    if isinstance(value, list):
        return json.dumps(value)
    if isinstance(value, float):
        return str(value)
    return str(value)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="统一超参调优入口 — Batch 9B (第一版: DNN random search)"
    )
    parser.add_argument("--dataset", default="real_raw_5000", help="数据集名称")
    parser.add_argument("--model", default="dnn", help="模型名称")
    parser.add_argument(
        "--search-config", default=None,
        help="搜索空间配置路径（默认 configs/tuning/{model}_random.yaml）",
    )
    parser.add_argument(
        "--num-trials", type=int, default=None,
        help="Trial 数量（覆盖搜索配置中的值）",
    )
    parser.add_argument("--metric", default=None, help="排序指标（覆盖搜索配置）")
    parser.add_argument("--direction", default=None, help="优化方向 maximize/minimize（覆盖搜索配置）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（覆盖搜索配置）")
    parser.add_argument("--dry-run", action="store_true", help="只生成 trial 配置，不执行训练")
    parser.add_argument(
        "--python", default=DEFAULT_PYTHON,
        help="Python 解释器路径",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. 参数校验
    # -----------------------------------------------------------------------
    if args.dataset not in ALLOWED_DATASETS:
        parser.error(
            f"当前只支持 dataset: {', '.join(sorted(ALLOWED_DATASETS))}，收到: {args.dataset}"
        )
    if args.model not in ALLOWED_MODELS:
        parser.error(
            f"当前只支持 model: {', '.join(sorted(ALLOWED_MODELS))}，收到: {args.model}"
        )
    if args.direction and args.direction not in ALLOWED_DIRECTIONS:
        parser.error(
            f"direction 必须是 {sorted(ALLOWED_DIRECTIONS)}，收到: {args.direction}"
        )

    # -----------------------------------------------------------------------
    # 2. 加载搜索空间配置
    # -----------------------------------------------------------------------
    search_config_path = args.search_config or f"configs/tuning/{args.model}_random.yaml"
    search_config_full = PROJECT_ROOT / search_config_path
    if not search_config_full.exists():
        parser.error(f"搜索空间配置不存在: {search_config_full}")

    print(f"[配置] 搜索空间配置: {search_config_full}")
    full_cfg = load_yaml(search_config_full)
    validate_search_config(full_cfg, str(search_config_full))

    search_cfg = full_cfg["search"]
    space_cfg = full_cfg["space"]
    base_cfg_block = full_cfg.get("base", {})

    # CLI 参数覆盖搜索配置
    num_trials = args.num_trials if args.num_trials is not None else search_cfg["num_trials"]
    metric = args.metric if args.metric is not None else search_cfg["metric"]
    direction = args.direction if args.direction is not None else search_cfg["direction"]
    seed = args.seed if args.seed is not None else search_cfg["seed"]

    # 验证 direction
    if direction not in ALLOWED_DIRECTIONS:
        print(f"[错误] direction 必须是 {sorted(ALLOWED_DIRECTIONS)}，收到: {direction}", file=sys.stderr)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 3. 加载 base model config
    # -----------------------------------------------------------------------
    model_config_rel = base_cfg_block.get("model_config", f"configs/models/{args.model}.yaml")
    model_config_path = PROJECT_ROOT / model_config_rel
    if not model_config_path.exists():
        print(f"[错误] 基础模型配置不存在: {model_config_path}", file=sys.stderr)
        sys.exit(1)
    model_config = load_yaml(model_config_path)
    print(f"[配置] 基础模型配置: {model_config_path}")

    # -----------------------------------------------------------------------
    # 4. 创建 tuning 输出目录
    # -----------------------------------------------------------------------
    tuning_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuning_output_dir = (
        PROJECT_ROOT / "outputs" / "tuning" / args.model / args.dataset / tuning_run_id
    )
    tuning_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[输出] Tuning 目录: {tuning_output_dir}")
    print()

    if not args.dry_run:
        print(f"[调参] 开始 {num_trials} trials {args.model} 随机搜索")
        print(f"[调参] 数据集: {args.dataset}")
        print(f"[调参] 排序指标: {metric} ({direction})")
        print(f"[调参] 种子: {seed}")
        print()

    # -----------------------------------------------------------------------
    # 5. 保存 tuning_config_used.yaml
    # -----------------------------------------------------------------------
    tuning_config_used = {
        "tuning_run_id": tuning_run_id,
        "model_name": args.model,
        "dataset_name": args.dataset,
        "search_config_path": str(search_config_full),
        "model_config_path": str(model_config_path),
        "num_trials": num_trials,
        "metric": metric,
        "direction": direction,
        "seed": seed,
        "dry_run": args.dry_run,
        "search": search_cfg,
        "space": space_cfg,
        "base": base_cfg_block,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    tuning_config_used_path = tuning_output_dir / "tuning_config_used.yaml"
    with open(tuning_config_used_path, "w", encoding="utf-8") as f:
        yaml.dump(tuning_config_used, f, default_flow_style=False, allow_unicode=True)
    print(f"[输出] Tuning 配置已保存: {tuning_config_used_path}")

    # -----------------------------------------------------------------------
    # 6. 执行 trials
    # -----------------------------------------------------------------------
    trials = []  # 每个 trial 的 dict 记录

    for trial_id in range(num_trials):
        print("-" * 60)
        print(f"  Trial {trial_id:03d}/{num_trials - 1:03d}")
        print("-" * 60)

        trial_dir = tuning_output_dir / f"trial_{trial_id:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # 6a. 采样参数
        trial_rng = random.Random(seed + trial_id * 7 + 13)
        trial_params = {}
        for param_name, param_cfg in space_cfg.items():
            trial_params[param_name] = sample_param(param_name, param_cfg, trial_rng)

        print(f"  params: {json.dumps(trial_params, default=str)}")

        # 6b. 添加训练 seed（每个 trial 独立）
        trial_seed = seed + trial_id * 7 + 13
        trial_params["random_seed"] = trial_seed

        # 6c. 保存 trial resolved.yaml
        trial_resolved = {
            "trial_id": trial_id,
            "tuning_run_id": tuning_run_id,
            "model_name": args.model,
            "dataset_name": args.dataset,
        }
        trial_resolved.update(trial_params)
        trial_resolved_path = trial_dir / "resolved.yaml"
        with open(trial_resolved_path, "w", encoding="utf-8") as f:
            yaml.dump(trial_resolved, f, default_flow_style=False, allow_unicode=True)

        # 6d. 构建 trial 记录
        trial_record = {
            "trial_id": trial_id,
            "run_id": None,
            "status": "dry_run" if args.dry_run else "pending",
            "val_auc": None,
            "val_loss": None,
            "val_accuracy": None,
            "val_f1": None,
            "val_precision": None,
            "val_recall": None,
            "test_auc": None,
            "test_f1": None,
        }
        # 将采样参数加入记录
        for k, v in trial_params.items():
            trial_record[k] = v
        trial_record["params_json"] = json.dumps(trial_params, default=str)
        trial_record["output_dir"] = str(trial_dir)
        trial_record["error_message"] = ""

        if args.dry_run:
            print(f"  [dry-run] 跳过训练")
            trials.append(trial_record)
            continue

        # 6e. 真实搜索：调用 train.py
        override_args = []
        for k, v in trial_params.items():
            override_args.append(f"{k}={format_value_for_override(v)}")

        cmd = [
            args.python,
            "src/training/train.py",
            "--dataset", args.dataset,
            "--model", args.model,
        ]
        for oa in override_args:
            cmd.append("--override")
            cmd.append(oa)

        print(f"  命令: python src/training/train.py --dataset {args.dataset} --model {args.model} --override ...")
        print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        t0 = time.time()

        # 记录训练前 run 目录
        before_dirs = get_run_dirs(args.model, args.dataset)

        try:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=7200,  # 2h 超时
            )
            elapsed = time.time() - t0
            status = "completed" if result.returncode == 0 else "failed"
            print(f"  结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  耗时: {elapsed:.1f}s")
            print(f"  返回码: {result.returncode}")

            if result.returncode != 0:
                # 截取 stderr 最后 500 字符
                stderr_tail = (result.stderr or "")[-500:]
                print(f"  stderr: {stderr_tail[:200]}")
                trial_record["status"] = "failed"
                trial_record["error_message"] = stderr_tail[:500]
            else:
                trial_record["status"] = "completed"

            # 识别 run_id
            after_dirs = get_run_dirs(args.model, args.dataset)
            new_run_id = find_new_run_id(before_dirs, after_dirs)
            if new_run_id:
                trial_record["run_id"] = new_run_id
                print(f"  捕获 run_id: {new_run_id}")
                # 读取指标
                metrics = read_metrics(args.model, args.dataset, new_run_id)
                for k, v in metrics.items():
                    trial_record[k] = v
                if metrics:
                    print(f"  val_auc={metrics.get('val_auc', 'N/A')}, val_loss={metrics.get('val_loss', 'N/A')}")
            else:
                print(f"  [警告] 未检测到新 run_dir (可能是 run_id 冲突)")
                # fallback: 尝试 latest_run.txt 或 stdout
                latest_txt = PROJECT_ROOT / "outputs" / args.model / args.dataset / "latest_run.txt"
                latest_txt_id = ""
                if latest_txt.exists():
                    latest_txt_id = latest_txt.read_text().strip()
                if result.returncode == 0 and latest_txt_id:
                    trial_record["run_id"] = latest_txt_id
                    trial_record["status"] = "completed"
                    print(f"  通过 latest_run.txt 捕获 run_id: {latest_txt_id}")
                    metrics = read_metrics(args.model, args.dataset, latest_txt_id)
                    for k, v in metrics.items():
                        trial_record[k] = v
                else:
                    err_msg = "未检测到新 run_dir"
                    if latest_txt_id:
                        err_msg += f" (latest_run.txt: {latest_txt_id})"
                    trial_record["error_message"] = (
                        trial_record.get("error_message", "") + "; " + err_msg
                    ).strip("; ")

        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"  超时 ({elapsed:.1f}s)")
            trial_record["status"] = "failed"
            trial_record["error_message"] = "训练超时 (>7200s)"
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  异常: {e}")
            trial_record["status"] = "failed"
            trial_record["error_message"] = str(e)[:500]

        print(f"  trial {trial_id:03d}: status={trial_record['status']}, run_id={trial_record['run_id']}")
        trials.append(trial_record)

        # 避免 run_id 冲突（DNN train.py 使用分钟级时间戳 %Y%m%d%H%M）
        if not args.dry_run and trial_id < num_trials - 1:
            current_min = datetime.now().strftime("%Y%m%d%H%M")
            while datetime.now().strftime("%Y%m%d%H%M") == current_min:
                time.sleep(1)
            print(f"  run_id 时序保护: 等待至下一分钟...")

    # -----------------------------------------------------------------------
    # 7. 写入 trials_summary.csv
    # -----------------------------------------------------------------------
    csv_path = tuning_output_dir / "trials_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for tr in trials:
            row = {}
            for field in CSV_FIELDS:
                val = tr.get(field)
                if isinstance(val, list):
                    val = json.dumps(val)
                elif val is None:
                    val = ""
                row[field] = val
            writer.writerow(row)
    print(f"[输出] Trials summary: {csv_path}")

    # 7b. 写入 trials.jsonl
    jsonl_path = tuning_output_dir / "trials.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for tr in trials:
            # 处理不可 JSON 序列化的类型
            clean = {}
            for k, v in tr.items():
                if isinstance(v, Path):
                    clean[k] = str(v)
                else:
                    clean[k] = v
            f.write(json.dumps(clean, default=str) + "\n")
    print(f"[输出] Trials JSONL: {jsonl_path}")

    # -----------------------------------------------------------------------
    # 8. 选择 Best Trial
    # -----------------------------------------------------------------------
    completed_trials = [t for t in trials if t["status"] == "completed"]
    best_trial = None
    if completed_trials:
        # 确定实际使用的指标
        actual_metric = metric
        actual_direction = direction

        # 检查 metric 在 completed trial 中是否可用
        metric_available = any(t.get(actual_metric) is not None for t in completed_trials)
        if not metric_available:
            print(f"[警告] 所有 completed trial 都缺少 '{metric}', 回退到 val_loss (minimize)")
            actual_metric = "val_loss"
            actual_direction = "minimize"

        # 排序
        reverse = (actual_direction == "maximize")
        valid_trials = [t for t in completed_trials if t.get(actual_metric) is not None]
        if valid_trials:
            valid_trials.sort(key=lambda t: t[actual_metric], reverse=reverse)
            best_trial = valid_trials[0]
            print(f"[Best] 最佳 trial: trial_{best_trial['trial_id']:03d}")
            print(f"[Best] 基于 {actual_metric} ({actual_direction})")
            print(f"[Best] {actual_metric} = {best_trial.get(actual_metric)}")
        else:
            print(f"[警告] 没有 trial 包含有效的 '{actual_metric}'")

    # -----------------------------------------------------------------------
    # 9. 输出 best_trial.json
    # -----------------------------------------------------------------------
    if best_trial:
        best_trial_json = {
            "trial_id": best_trial["trial_id"],
            "run_id": best_trial["run_id"],
            "tuning_run_id": tuning_run_id,
            "model_name": args.model,
            "dataset_name": args.dataset,
            "selected_metric": actual_metric,
            "selected_direction": actual_direction,
            "selected_metric_value": best_trial.get(actual_metric),
        }
        # 复制所有空间参数
        for k in space_cfg.keys():
            if k in best_trial:
                best_trial_json[k] = best_trial[k]
        best_trial_json["params_json"] = best_trial.get("params_json")
        best_trial_json["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        best_trial_path = tuning_output_dir / "best_trial.json"
        with open(best_trial_path, "w", encoding="utf-8") as f:
            json.dump(best_trial_json, f, indent=2, default=str, ensure_ascii=False)
        print(f"[输出] Best trial: {best_trial_path}")

    # -----------------------------------------------------------------------
    # 10. 输出 best_config.yaml
    # -----------------------------------------------------------------------
    if best_trial and not args.dry_run:
        best_config = {
            "model_name": args.model,
            "dataset_name": args.dataset,
            "tuning_run_id": tuning_run_id,
            "source_run_id": best_trial["run_id"],
            "source_trial_id": best_trial["trial_id"],
            "selected_metric": actual_metric,
            "selected_direction": actual_direction,
            "selected_metric_value": best_trial.get(actual_metric),
        }
        for k in space_cfg.keys():
            if k in best_trial:
                best_config[k] = best_trial[k]
        best_config["train_command"] = (
            f"python src/training/train.py --dataset {args.dataset} --model {args.model}"
        )
        best_config["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        best_config_path = tuning_output_dir / "best_config.yaml"
        with open(best_config_path, "w", encoding="utf-8") as f:
            yaml.dump(best_config, f, default_flow_style=False, allow_unicode=True)
        print(f"[输出] Best config: {best_config_path}")
    elif args.dry_run:
        print(f"[输出] dry-run 模式: 不输出 best_config.yaml")

    # -----------------------------------------------------------------------
    # 11. 输出 tuning_report.json
    # -----------------------------------------------------------------------
    num_completed = sum(1 for t in trials if t["status"] == "completed")
    num_failed = sum(1 for t in trials if t["status"] == "failed")

    report = {
        "tuning_run_id": tuning_run_id,
        "model_name": args.model,
        "dataset_name": args.dataset,
        "search_config": str(search_config_full),
        "num_trials": num_trials,
        "num_completed": num_completed,
        "num_failed": num_failed,
        "metric": metric,
        "direction": direction,
        "seed": seed,
        "dry_run": args.dry_run,
    }

    if best_trial:
        report["best_trial_id"] = best_trial["trial_id"]
        report["best_run_id"] = best_trial["run_id"]
        report["best_val_auc"] = best_trial.get("val_auc")
        report["best_val_f1"] = best_trial.get("val_f1")
        report["best_test_auc"] = best_trial.get("test_auc")
        report["best_test_f1"] = best_trial.get("test_f1")
        report["best_val_loss"] = best_trial.get("val_loss")
        report["best_params"] = {}
        for k in space_cfg.keys():
            if k in best_trial:
                report["best_params"][k] = best_trial[k]

    warnings = []
    if args.dry_run:
        warnings.append("Dry-run 模式，未执行实际训练。")
    if num_failed > 0:
        warnings.append(f"{num_failed} trials failed (详见 trials_summary.csv).")
    if best_trial and actual_metric != metric:
        warnings.append(
            f"指标回退: 原 '{metric}' 不可用，实际使用 '{actual_metric}'."
        )
    if not args.dry_run and num_completed == 0:
        warnings.append("所有 trial 均失败，未输出 best_config.yaml。")
    report["warnings"] = warnings
    report["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_path = tuning_output_dir / "tuning_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    print(f"[输出] Tuning report: {report_path}")

    # -----------------------------------------------------------------------
    # 12. 最终摘要
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  调参完成摘要")
    print("=" * 60)
    print(f"  Tuning Run ID:  {tuning_run_id}")
    print(f"  模型:            {args.model}")
    print(f"  数据集:          {args.dataset}")
    print(f"  Trial 总数:      {num_trials}")
    print(f"  Completed:       {num_completed}")
    print(f"  Failed:          {num_failed}")
    print(f"  排序指标:        {metric} ({direction})")
    if best_trial:
        print(f"  最佳 Trial:      trial_{best_trial['trial_id']:03d}")
        print(f"  最佳 run_id:     {best_trial['run_id']}")
        print(f"  最佳 val_auc:    {best_trial.get('val_auc', 'N/A')}")
        print(f"  最佳 test_auc:   {best_trial.get('test_auc', 'N/A')}")
    print(f"  输出目录:        {tuning_output_dir}")
    if args.dry_run:
        print(f"  模式:            dry-run (未执行训练)")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()