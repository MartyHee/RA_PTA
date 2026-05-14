"""RA_PTA 端到端推荐流水线轻量 orchestrator 第一版。

支持分阶段串联 DNN 主线：
    load → build_tabular → train(dnn) → batch_predict

使用示例：
    # dry-run 预览
    D:/CodeData/software/Anaconda/Anaconda3/envs/ra/python.exe src/pipeline/run_pipeline.py ^
        --dataset real_raw_5000 --model dnn --dry-run

    # 只执行推理（需指定已有 run_id）
    D:/CodeData/software/Anaconda/Anaconda3/envs/ra/python.exe src/pipeline/run_pipeline.py ^
        --dataset real_raw_5000 --model dnn --steps infer --run-id 202605132017

    # 全流程（load → tabular → train → infer）
    D:/CodeData/software/Anaconda/Anaconda3/envs/ra/python.exe src/pipeline/run_pipeline.py ^
        --dataset real_raw_5000 --model dnn --steps load,tabular,train,infer
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 允许的值
# ---------------------------------------------------------------------------
ALLOWED_DATASETS = frozenset({"real_raw_5000"})
ALLOWED_MODELS = frozenset({"dnn"})
ALLOWED_STEPS = frozenset({"load", "tabular", "train", "infer"})

DEFAULT_PYTHON = "D:/CodeData/software/Anaconda/Anaconda3/envs/ra/python.exe"

# ---------------------------------------------------------------------------
# Helper: 获取 outputs/<model>/<dataset>/ 下的所有 run 目录（按名称排序）
# ---------------------------------------------------------------------------

def _get_run_dirs(dataset: str, model: str) -> list[Path]:
    run_dir = Path("outputs") / model / dataset
    if not run_dir.is_dir():
        return []
    return sorted(
        [d for d in run_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )


def _find_new_run_id(before: list[Path], after: list[Path]) -> str | None:
    """对比训练前后的目录列表，找出新增的 run_id（按 mtime 取最新）。"""
    before_names = {d.name for d in before}
    after_names = {d.name for d in after}
    new_names = after_names - before_names
    if not new_names:
        return None
    candidates = [d for d in after if d.name in new_names]
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return newest.name


def _get_latest_inference_dir(run_id: str, dataset: str, model: str) -> Path | None:
    """扫描 outputs/inference/<model>/<dataset>/<run_id>/ 下最新的子目录。"""
    base = Path("outputs") / "inference" / model / dataset / run_id
    if not base.is_dir():
        return None
    subdirs = [d for d in base.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def _build_stage_command(
    python: str, step: str, dataset: str, model: str, run_id: str, input_csv: str
) -> str:
    """构建指定阶段的 shell 命令字符串。"""
    if step == "load":
        return f'"{python}" src/data/load_raw.py --dataset {dataset}'
    if step == "tabular":
        return f'"{python}" src/data/build_tabular.py --dataset {dataset}'
    if step == "train":
        return f'"{python}" src/training/train.py --dataset {dataset} --model {model}'
    if step == "infer":
        return (
            f'"{python}" src/inference/batch_predict.py '
            f"--model {model} --dataset {dataset} --run-id {run_id} --input \"{input_csv}\""
        )
    raise ValueError(f"未知阶段: {step}")


def _print_header(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _run_command(cmd: str, dry_run: bool, step_name: str) -> bool:
    """执行 shell 命令，或在 dry-run 模式下只打印。返回 True 表示成功。"""
    print()
    print(f"[{step_name}] 命令: {cmd}")
    if dry_run:
        print(f"[{step_name}] 模式: dry-run，跳过执行")
        return True

    print(f"[{step_name}] 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - t0
    status = "成功" if result.returncode == 0 else f"失败 (exit code={result.returncode})"
    print(f"[{step_name}] 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[{step_name}] 耗时: {elapsed:.1f}s")
    print(f"[{step_name}] 状态: {status}")
    return result.returncode == 0


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RA_PTA Pipeline Orchestrator（第一版只支持 DNN + real_raw_5000）"
    )
    parser.add_argument("--dataset", default="real_raw_5000", help="数据集名称")
    parser.add_argument("--model", default="dnn", help="模型名称")
    parser.add_argument(
        "--steps", default="load,tabular,train,infer",
        help="逗号分隔的阶段列表，可选: load,tabular,train,infer",
    )
    parser.add_argument("--run-id", default=None, help="推理阶段使用的 run_id")
    parser.add_argument(
        "--input", default=None,
        help="推理阶段输入 CSV（默认 data/features/{dataset}/tabular_test.csv）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印命令不执行",
    )
    parser.add_argument(
        "--python", default=DEFAULT_PYTHON,
        help="Python 解释器路径",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 参数校验
    # -----------------------------------------------------------------------
    if args.dataset not in ALLOWED_DATASETS:
        parser.error(
            f"当前只支持 dataset: {', '.join(sorted(ALLOWED_DATASETS))}，收到: {args.dataset}"
        )

    if args.model not in ALLOWED_MODELS:
        parser.error(
            f"当前只支持 model: {', '.join(sorted(ALLOWED_MODELS))}，收到: {args.model}"
        )

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = set(steps) - ALLOWED_STEPS
    if unknown:
        parser.error(
            f"未知阶段: {', '.join(sorted(unknown))}。"
            f"支持: {', '.join(sorted(ALLOWED_STEPS))}"
        )

    has_train = "train" in steps
    has_infer = "infer" in steps

    # infer 需要 run_id
    if has_infer and not has_train and not args.run_id:
        parser.error("infer 阶段需要 --run-id（未执行 train 时必须提供）")

    input_csv = args.input or f"data/features/{args.dataset}/tabular_test.csv"

    if has_infer and not args.dry_run and not os.path.isfile(input_csv):
        parser.error(f"输入 CSV 不存在: {input_csv}")

    # -----------------------------------------------------------------------
    # 打印概览
    # -----------------------------------------------------------------------
    print()
    print("#" * 60)
    print("#  RA_PTA Pipeline Orchestrator")
    print("#" * 60)
    print(f"  Dataset:   {args.dataset}")
    print(f"  Model:     {args.model}")
    print(f"  Steps:     {', '.join(steps)}")
    print(f"  Dry-run:   {args.dry_run}")
    print(f"  Python:    {args.python}")
    if args.run_id:
        print(f"  Run ID:    {args.run_id}")
    print(f"  Input CSV: {input_csv}")
    print("#" * 60)
    print()

    # -----------------------------------------------------------------------
    # 阶段执行
    # -----------------------------------------------------------------------
    run_id = args.run_id
    executed_steps: list[str] = []
    infer_output_dir: str | None = None

    for step in steps:
        # 对于 train 阶段，记录当前已有目录，便于训练后捕获 run_id
        if step == "train":
            before_dirs = _get_run_dirs(args.dataset, args.model)
            if not args.dry_run:
                print(f"[train] 训练前已有 {len(before_dirs)} 个 run 目录")

        _print_header({
            "load": "阶段 1/4: 数据读取 (load_raw)",
            "tabular": "阶段 2/4: Tabular 特征构建 (build_tabular)",
            "train": "阶段 3/4: 模型训练 (train)",
            "infer": "阶段 4/4: Batch 推理 (batch_predict)",
        }[step])

        # 构建命令
        cmd = _build_stage_command(
            python=args.python,
            step=step,
            dataset=args.dataset,
            model=args.model,
            run_id=run_id or "<run_id>",
            input_csv=input_csv,
        )

        # dry-run 时特殊处理 infer 的 run_id 提示
        if step == "infer" and args.dry_run:
            if has_train:
                print(f"[infer] dry-run: 将使用 train 阶段产生的新 run_id")
            if run_id:
                print(f"[infer] run_id: {run_id}")

        ok = _run_command(cmd, args.dry_run, step)

        if ok:
            executed_steps.append(step)
        elif not args.dry_run:
            print(f"错误: {step} 阶段失败，中止流水线", file=sys.stderr)
            sys.exit(1)

        # 训练后捕获 run_id
        if step == "train":
            if not args.dry_run:
                after_dirs = _get_run_dirs(args.dataset, args.model)
                new_id = _find_new_run_id(before_dirs, after_dirs)
                if new_id is None:
                    print("错误: 训练完成后无法检测到新 run_id", file=sys.stderr)
                    print("  outputs/ 目录可能没有新增 run 目录", file=sys.stderr)
                    sys.exit(1)
                run_id = new_id
                print(f"[train] 捕获到新 run_id: {run_id}")
            else:
                # dry-run 模式：设置占位 run_id 以便摘要展示
                if run_id is None:
                    run_id = "<训练将产生的 run_id>"

        # 推理完成后捕获输出目录
        if step == "infer" and not args.dry_run and run_id:
            latest = _get_latest_inference_dir(run_id, args.dataset, args.model)
            if latest is not None:
                infer_output_dir = str(latest)

    # -----------------------------------------------------------------------
    # 输出摘要
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  Pipeline 完成摘要")
    print("=" * 60)
    print(f"  数据集:     {args.dataset}")
    print(f"  模型:       {args.model}")
    print(f"  执行阶段:   {', '.join(executed_steps)}")
    if run_id:
        print(f"  Run ID:     {run_id}")
    if infer_output_dir:
        print(f"  推理输出:   {infer_output_dir}")
        pred_csv = Path(infer_output_dir) / "predictions.csv"
        meta_json = Path(infer_output_dir) / "inference_meta.json"
        if pred_csv.is_file():
            print(f"  predictions: {pred_csv}")
        if meta_json.is_file():
            print(f"  meta:        {meta_json}")

    # REST API 启动命令（不自动启动）
    if run_id:
        print()
        print("  REST API 启动命令（手动执行）:")
        print()
        if args.dry_run and "<" in str(run_id):
            print(f"    {args.python} src/serving/api.py"
                  f" --model {args.model}"
                  f" --dataset {args.dataset}"
                  f" --run-id <实际_run_id>")
        else:
            print(f"    {args.python} src/serving/api.py"
                  f" --model {args.model}"
                  f" --dataset {args.dataset}"
                  f" --run-id {run_id}")
        print()

    print("=" * 60)
    print(f"  状态: {'dry-run (未执行)' if args.dry_run else '完成'}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()