"""DNN 原始模型推理 benchmark CLI。

Batch 10B：第一版只支持 DNN 原始模型 benchmark，不做模型压缩。
复用 Predictor 的模型加载、输入校验、特征处理逻辑，只对模型 forward 做计时。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_model_dir(model: str, dataset: str, run_id: str) -> str:
    """根据 --model / --dataset / --run-id 定位模型目录。"""
    return str(Path("outputs") / model / dataset / run_id)


def generate_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_device(device: str) -> str:
    """解析设备参数，返回实际使用的设备名。"""
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "请求使用 CUDA，但当前环境 CUDA 不可用。"
                "请使用 --device cpu 或在 CUDA 可用环境运行。"
            )
        return "cuda"
    if device == "cpu":
        return "cpu"
    raise ValueError(f"无效 device: {device}，必须是 auto/cuda/cpu")


def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="RA_PTA 推理 Benchmark CLI（Batch 10B，第一版只支持 DNN 原始模型）"
    )
    parser.add_argument("--model", default="dnn", help="模型名称（第一版只支持 dnn）")
    parser.add_argument(
        "--dataset",
        default="real_raw_5000",
        help="数据集名称（第一版只支持 real_raw_5000）",
    )
    parser.add_argument("--run-id", required=True, help="模型训练 run_id")
    parser.add_argument(
        "--input",
        default="data/features/real_raw_5000/tabular_test.csv",
        help="输入 CSV 路径",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="推理设备（auto/cuda/cpu，默认 auto）",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=3,
        help="Warmup 轮次（默认 3）",
    )
    parser.add_argument(
        "--num-repeat",
        type=int,
        default=10,
        help="重复推理轮次（默认 10）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="批大小（默认 None，全量推理）",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="自定义输出目录（可选，默认自动生成）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅验证路径和模型加载，不执行重复 benchmark",
    )
    parser.add_argument(
        "--compressed-dir",
        default=None,
        help="压缩模型目录（可选），提供则同时 benchmark 压缩模型并输出对比",
    )
    args = parser.parse_args()

    # ================================================================
    # 参数校验
    # ================================================================
    if args.model != "dnn":
        parser.error(f"当前版本只支持 --model dnn，收到: {args.model}")
    if args.dataset != "real_raw_5000":
        parser.error(
            f"当前版本只支持 --dataset real_raw_5000，收到: {args.dataset}"
        )
    if args.num_warmup < 0:
        parser.error(f"--num-warmup 必须 >= 0，收到: {args.num_warmup}")
    if args.num_repeat < 1:
        parser.error(f"--num-repeat 必须 >= 1，收到: {args.num_repeat}")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error(f"--batch-size 必须 >= 1，收到: {args.batch_size}")
    valid_devices = {"auto", "cuda", "cpu"}
    if args.device not in valid_devices:
        parser.error(
            f"--device 必须是 auto/cuda/cpu 之一，收到: {args.device}"
        )

    # ================================================================
    # 路径解析
    # ================================================================
    model_dir = resolve_model_dir(args.model, args.dataset, args.run_id)
    if not os.path.isdir(model_dir):
        print(f"错误: 模型目录不存在: {model_dir}", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"错误: 输入 CSV 不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    # ================================================================
    # 设备解析
    # ================================================================
    try:
        device = resolve_device(args.device)
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"模型目录: {model_dir}")
    print(f"输入 CSV: {input_path}")
    print(f"请求设备: {args.device}")
    print(f"实际设备: {device}")
    print(f"Warmup: {args.num_warmup}, Repeat: {args.num_repeat}")
    if args.dry_run:
        print("模式: dry-run（仅验证）")

    # 延迟加载（避免 CLI --help 时加载 torch）
    import torch

    from src.inference.predictor import Predictor

    # ================================================================
    # 阶段 1: 模型加载（计时）
    # ================================================================
    print("\n[阶段 1] 加载模型...")
    t0 = time.perf_counter()
    predictor = Predictor(model_dir=model_dir, device=device)
    predictor.load()
    load_time = time.perf_counter() - t0
    load_time_ms = round(load_time * 1000, 2)
    print(f"  完成: {load_time_ms} ms")

    # 模型体积
    model_path = Path(model_dir) / "model.pt"
    model_size_bytes = model_path.stat().st_size
    model_size_mb = round(model_size_bytes / (1024 * 1024), 4)

    # ================================================================
    # 阶段 2: 读取 CSV
    # ================================================================
    print("\n[阶段 2] 读取输入数据...")
    df = pd.read_csv(input_path, encoding="utf-8-sig", skipinitialspace=True)
    num_samples = len(df)
    numeric_cols = predictor.feature_config.get("numeric_cols", [])
    categorical_cols = predictor.feature_config.get("categorical_cols", [])
    feature_count = len(numeric_cols) + len(categorical_cols)
    print(f"  样本数: {num_samples}, 特征数: {feature_count}")

    # ================================================================
    # 阶段 3: 特征预处理（计时）
    # ================================================================
    print("\n[阶段 3] 特征预处理...")
    t0 = time.perf_counter()
    predictor.validate_input(df)
    numeric_tensor, cat_tensor = predictor.transform(df)
    preprocessing_time = time.perf_counter() - t0
    preprocessing_time_ms = round(preprocessing_time * 1000, 2)
    print(f"  完成: {preprocessing_time_ms} ms")

    # 数据移到目标设备
    numeric_tensor = numeric_tensor.to(device)
    cat_tensor = cat_tensor.to(device)

    # ================================================================
    # Dry-run: 在此结束
    # ================================================================
    if args.dry_run:
        print("\n[Dry-run] 验证通过。")
        print(f"  模型目录:      [ok] {model_dir}")
        print(f"  输入 CSV:       [ok] {input_path}")
        print(f"  样本数:         {num_samples}")
        print(f"  特征数:         {feature_count}")
        print(f"  设备:           {device}")
        print(f"  模型加载:       {load_time_ms} ms")
        print(f"  预处理:         {preprocessing_time_ms} ms")
        print(f"  模型大小:       {model_size_mb} MB")
        print("\n如需运行完整 benchmark，去掉 --dry-run 即可。")
        return

    # ================================================================
    # 阶段 4: Warmup
    # ================================================================
    if args.num_warmup > 0:
        print(f"\n[阶段 4] Warmup ({args.num_warmup} 轮)...")
        for _ in range(args.num_warmup):
            with torch.no_grad():
                _ = predictor.model(numeric_tensor, cat_tensor)
            if device == "cuda":
                torch.cuda.synchronize()
        print("  Warmup 完成")

    # ================================================================
    # 阶段 5: Repeat 推理（计时）
    # ================================================================
    print(f"\n[阶段 5] 重复推理 ({args.num_repeat} 轮)...")
    repeat_times = []
    for i in range(args.num_repeat):
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = predictor.model(numeric_tensor, cat_tensor)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        repeat_times.append(elapsed)
        print(f"  第 {i+1:2d} 轮: {elapsed * 1000:.2f} ms")

    times_ms = np.array(repeat_times) * 1000
    inference_time_ms_mean = round(float(np.mean(times_ms)), 2)
    inference_time_ms_std = round(float(np.std(times_ms)), 2)
    inference_time_ms_min = round(float(np.min(times_ms)), 2)
    inference_time_ms_max = round(float(np.max(times_ms)), 2)
    inference_time_ms_per_sample = round(
        inference_time_ms_mean / num_samples, 4
    )
    throughput_samples_per_sec = round(
        num_samples / (inference_time_ms_mean / 1000), 2
    )

    print(f"\n  推理耗时统计:")
    print(f"    均值:   {inference_time_ms_mean} ms")
    print(f"    标准差: {inference_time_ms_std} ms")
    print(f"    最小值: {inference_time_ms_min} ms")
    print(f"    最大值: {inference_time_ms_max} ms")
    print(f"    单样本: {inference_time_ms_per_sample} ms")
    print(f"    吞吐:   {throughput_samples_per_sec} samples/sec")

    # ================================================================
    # 阶段 6: 最终 scores
    # ================================================================
    print("\n[阶段 6] 计算 scores...")
    with torch.no_grad():
        logits = predictor.model(numeric_tensor, cat_tensor)
    if device == "cuda":
        torch.cuda.synchronize()
    scores = torch.sigmoid(logits).cpu().numpy().flatten()

    score_min = round(float(np.min(scores)), 6)
    score_max = round(float(np.max(scores)), 6)
    print(f"  Score 范围: [{score_min}, {score_max}]")

    # 保存原模型数据用于对比
    original_scores = scores.copy()
    original_times_ms = times_ms.copy()

    # 生成 run_id 和输出目录（提前初始化以供 Phase 7 使用）
    benchmark_run_id = generate_run_id()
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            Path("outputs")
            / "optimization"
            / "benchmark"
            / args.model
            / args.dataset
            / args.run_id
            / benchmark_run_id
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # 阶段 7（可选）: 压缩模型 benchmark（当 --compressed-dir 提供时）
    # ================================================================
    is_compressed_run = args.compressed_dir is not None
    compressed_data = {}

    if is_compressed_run:
        compressed_dir = Path(args.compressed_dir)
        compressed_model_path = compressed_dir / "model_compressed.pt"
        if not compressed_model_path.is_file():
            print(f"\n错误: 压缩模型不存在: {compressed_model_path}", file=sys.stderr)
            sys.exit(1)

        print(f"\n[阶段 7] 压缩模型 benchmark...")
        print(f"  压缩模型: {compressed_model_path}")

        # 加载压缩模型（动态量化只能运行在 CPU）
        comp_device = "cpu"
        t0 = time.perf_counter()
        comp_model = torch.load(compressed_model_path, map_location=comp_device, weights_only=False)
        comp_model.eval()
        comp_load_time = time.perf_counter() - t0
        comp_load_time_ms = round(comp_load_time * 1000, 2)
        print(f"  加载: {comp_load_time_ms} ms")

        # 压缩模型体积
        comp_size_bytes = compressed_model_path.stat().st_size
        comp_size_mb = round(comp_size_bytes / (1024 * 1024), 4)
        print(f"  体积: {comp_size_mb} MB")

        # 将 tensors 移到 CPU
        num_cpu = numeric_tensor.cpu()
        cat_cpu = cat_tensor.cpu()

        # Warmup
        if args.num_warmup > 0:
            print(f"  Warmup ({args.num_warmup} 轮)...")
            for _ in range(args.num_warmup):
                with torch.no_grad():
                    _ = comp_model(num_cpu, cat_cpu)

        # Repeat
        print(f"  重复推理 ({args.num_repeat} 轮)...")
        comp_repeat_times = []
        for i in range(args.num_repeat):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = comp_model(num_cpu, cat_cpu)
            elapsed = time.perf_counter() - t0
            comp_repeat_times.append(elapsed)
            print(f"    第 {i+1:2d} 轮: {elapsed * 1000:.2f} ms")

        comp_times_ms = np.array(comp_repeat_times) * 1000
        comp_inference_ms_mean = round(float(np.mean(comp_times_ms)), 2)
        comp_inference_ms_std = round(float(np.std(comp_times_ms)), 2)
        comp_inference_ms_min = round(float(np.min(comp_times_ms)), 2)
        comp_inference_ms_max = round(float(np.max(comp_times_ms)), 2)
        comp_inference_ms_per_sample = round(
            comp_inference_ms_mean / num_samples, 4
        )
        comp_throughput = round(
            num_samples / (comp_inference_ms_mean / 1000), 2
        )

        print(f"\n  压缩模型推理耗时统计:")
        print(f"    均值:   {comp_inference_ms_mean} ms")
        print(f"    标准差: {comp_inference_ms_std} ms")
        print(f"    最小值: {comp_inference_ms_min} ms")
        print(f"    最大值: {comp_inference_ms_max} ms")
        print(f"    单样本: {comp_inference_ms_per_sample} ms")
        print(f"    吞吐:   {comp_throughput} samples/sec")

        # 压缩模型最终 scores
        print(f"\n  计算压缩模型 scores...")
        with torch.no_grad():
            comp_logits = comp_model(num_cpu, cat_cpu)
        comp_scores = torch.sigmoid(comp_logits).cpu().numpy().flatten()
        comp_score_min = round(float(np.min(comp_scores)), 6)
        comp_score_max = round(float(np.max(comp_scores)), 6)

        # Score 差异
        score_diff = original_scores - comp_scores
        max_abs_diff = round(float(np.max(np.abs(score_diff))), 8)
        mean_abs_diff = round(float(np.mean(np.abs(score_diff))), 8)

        # Pearson 相关系数
        try:
            corr_matrix = np.corrcoef(original_scores, comp_scores)
            pearson_corr = round(float(corr_matrix[0, 1]), 6)
        except Exception:
            pearson_corr = None

        # 速度对比
        speedup_ratio = round(
            inference_time_ms_mean / comp_inference_ms_mean, 4
        ) if comp_inference_ms_mean > 0 else 0

        print(f"\n  Score 对比:")
        print(f"    原模型 score 范围: [{score_min}, {score_max}]")
        print(f"    压缩模型 score 范围: [{comp_score_min}, {comp_score_max}]")
        print(f"    max_abs_diff: {max_abs_diff}")
        print(f"    mean_abs_diff: {mean_abs_diff}")
        print(f"    Pearson corr: {pearson_corr}")
        print(f"    速度提升: {speedup_ratio}x")

        compressed_data = {
            "is_compressed": True,
            "compressed_dir": str(compressed_dir.resolve()),
            "compressed_model_path": str(compressed_model_path.resolve()),
            "compressed_size_mb": comp_size_mb,
            "compressed_load_time_ms": comp_load_time_ms,
            "compressed_inference_time_ms_mean": comp_inference_ms_mean,
            "compressed_inference_time_ms_std": comp_inference_ms_std,
            "compressed_inference_time_ms_min": comp_inference_ms_min,
            "compressed_inference_time_ms_max": comp_inference_ms_max,
            "compressed_inference_time_ms_per_sample": comp_inference_ms_per_sample,
            "compressed_throughput_samples_per_sec": comp_throughput,
            "compressed_score_min": comp_score_min,
            "compressed_score_max": comp_score_max,
            "comparison_max_abs_score_diff": max_abs_diff,
            "comparison_mean_abs_score_diff": mean_abs_diff,
            "comparison_pearson_corr": pearson_corr,
            "comparison_speedup_ratio": speedup_ratio,
        }

        # 保存压缩模型 raw_scores.csv
        comp_scores_df = pd.DataFrame({"score": comp_scores})
        if "video_id" in df.columns:
            comp_scores_df["video_id"] = df["video_id"].values
        comp_scores_df.to_csv(
            Path(output_dir) / "raw_scores_compressed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(f"  raw_scores_compressed.csv [ok] ({len(comp_scores_df)} rows)")

    # ================================================================
    # 输出
    # ================================================================
    print(f"\n输出目录: {output_dir}")

    # Warnings
    warnings = []
    if device == "cpu" and torch.cuda.is_available():
        warnings.append(
            "CPU 推理（CUDA 可用但未使用），速度可能较慢。"
        )
    if is_compressed_run:
        warnings.append("动态量化模型仅支持 CPU 推理，速度对比基于原模型 device 与 CPU 压缩模型。")

    # 1. benchmark_report.json
    report = {
        "benchmark_run_id": benchmark_run_id,
        "model_name": args.model,
        "dataset_name": args.dataset,
        "source_run_id": args.run_id,
        "model_dir": str(Path(model_dir).resolve()),
        "input_path": str(input_path.resolve()),
        "num_samples": num_samples,
        "feature_count": feature_count,
        "device": device,
        "requested_device": args.device,
        "model_size_mb": model_size_mb,
        "load_time_ms": load_time_ms,
        "preprocessing_time_ms": preprocessing_time_ms,
        "num_warmup": args.num_warmup,
        "num_repeat": args.num_repeat,
        "inference_time_ms_mean": inference_time_ms_mean,
        "inference_time_ms_std": inference_time_ms_std,
        "inference_time_ms_min": inference_time_ms_min,
        "inference_time_ms_max": inference_time_ms_max,
        "inference_time_ms_per_sample": inference_time_ms_per_sample,
        "throughput_samples_per_sec": throughput_samples_per_sec,
        "score_min": score_min,
        "score_max": score_max,
        "is_compressed": is_compressed_run,
        "compressed_metrics": compressed_data if is_compressed_run else None,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "warnings": warnings,
    }
    save_json(report, output_dir / "benchmark_report.json")
    print(f"  benchmark_report.json [ok]")

    # 2. benchmark_summary.csv（单行 CSV）
    summary_path = output_dir / "benchmark_summary.csv"
    summary_fields = [
        "benchmark_run_id",
        "model_name",
        "dataset_name",
        "source_run_id",
        "num_samples",
        "feature_count",
        "device",
        "model_size_mb",
        "load_time_ms",
        "preprocessing_time_ms",
        "num_warmup",
        "num_repeat",
        "inference_time_ms_mean",
        "inference_time_ms_std",
        "inference_time_ms_min",
        "inference_time_ms_max",
        "inference_time_ms_per_sample",
        "throughput_samples_per_sec",
        "score_min",
        "score_max",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow({k: report.get(k, "") for k in summary_fields})
    print(f"  benchmark_summary.csv [ok]")

    # 3. raw_scores.csv
    raw_scores_df = pd.DataFrame({"score": scores})
    if "video_id" in df.columns:
        raw_scores_df["video_id"] = df["video_id"].values
    raw_scores_df.to_csv(
        output_dir / "raw_scores.csv", index=False, encoding="utf-8-sig"
    )
    print(f"  raw_scores.csv [ok] ({len(raw_scores_df)} rows)")

    # 4. benchmark_config_used.json
    config = {
        "cli_args": {
            "model": args.model,
            "dataset": args.dataset,
            "run_id": args.run_id,
            "input": str(args.input),
            "device": args.device,
            "num_warmup": args.num_warmup,
            "num_repeat": args.num_repeat,
            "batch_size": args.batch_size,
            "output_dir": args.output_dir,
            "dry_run": args.dry_run,
        },
        "resolved_paths": {
            "model_dir": str(Path(model_dir).resolve()),
            "input_path": str(input_path.resolve()),
            "output_dir": str(output_dir.resolve()),
        },
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_used": device,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dry_run": False,
    }
    save_json(config, output_dir / "benchmark_config_used.json")
    print(f"  benchmark_config_used.json [ok]")

    # 5. 压缩对比报告（仅当 --compressed-dir 提供时）
    if is_compressed_run:
        comparison_run_id = generate_run_id()
        comparison_dir = (
            Path("outputs")
            / "optimization"
            / "comparison"
            / args.model
            / args.dataset
            / args.run_id
            / comparison_run_id
        )
        comparison_dir.mkdir(parents=True, exist_ok=True)

        # 结论
        if max_abs_diff < 0.001:
            score_conclusion = "score 差异可忽略 (max_abs_diff < 0.001)"
        elif max_abs_diff < 0.01:
            score_conclusion = "score 差异较小 (max_abs_diff < 0.01)"
        elif max_abs_diff < 0.1:
            score_conclusion = "score 差异中等 (max_abs_diff < 0.1)，需关注"
        else:
            score_conclusion = "score 差异显著 (max_abs_diff >= 0.1)，不建议使用压缩模型"

        if speedup_ratio > 1.0:
            speed_conclusion = f"压缩后速度提升 {speedup_ratio:.2f}x"
        elif speedup_ratio > 0.9:
            speed_conclusion = f"压缩前后速度相当 ({speedup_ratio:.2f}x)"
        else:
            speed_conclusion = f"压缩后速度未提升 ({speedup_ratio:.2f}x)，动态量化对小模型可能变慢"

        comparison_json = {
            "comparison_run_id": comparison_run_id,
            "source_model_run_id": args.run_id,
            "model_name": args.model,
            "dataset_name": args.dataset,
            "source_model_size_mb": model_size_mb,
            "compressed_model_size_mb": comp_size_mb,
            "compression_ratio": round(comp_size_mb / model_size_mb, 4) if model_size_mb > 0 else 0,
            "size_reduction_percent": round((1 - comp_size_mb / model_size_mb) * 100, 2) if model_size_mb > 0 else 0,
            "original_inference_time_ms_mean": inference_time_ms_mean,
            "compressed_inference_time_ms_mean": comp_inference_ms_mean,
            "speedup_ratio": speedup_ratio,
            "original_throughput_samples_per_sec": throughput_samples_per_sec,
            "compressed_throughput_samples_per_sec": comp_throughput,
            "max_abs_score_diff": max_abs_diff,
            "mean_abs_score_diff": mean_abs_diff,
            "pearson_corr": pearson_corr,
            "score_conclusion": score_conclusion,
            "speed_conclusion": speed_conclusion,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "warnings": warnings,
        }
        save_json(comparison_json, comparison_dir / "compression_benchmark_comparison.json")
        print(f"  compression_benchmark_comparison.json [ok] ({comparison_dir})")

        # 对比 summary CSV
        comp_summary_fields = [
            "comparison_run_id",
            "source_model_run_id",
            "model_name",
            "dataset_name",
            "source_model_size_mb",
            "compressed_model_size_mb",
            "compression_ratio",
            "size_reduction_percent",
            "original_inference_time_ms_mean",
            "compressed_inference_time_ms_mean",
            "speedup_ratio",
            "original_throughput_samples_per_sec",
            "compressed_throughput_samples_per_sec",
            "max_abs_score_diff",
            "mean_abs_score_diff",
            "pearson_corr",
        ]
        comp_summary_path = comparison_dir / "compression_benchmark_summary.csv"
        with comp_summary_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=comp_summary_fields)
            writer.writeheader()
            writer.writerow({k: comparison_json.get(k, "") for k in comp_summary_fields})
        print(f"  compression_benchmark_summary.csv [ok]")

    # ================================================================
    # 汇总
    # ================================================================
    print(f"\n{'=' * 50}")
    print("Benchmark 完成！")
    print(f"  模型: {args.model}/{args.dataset}/{args.run_id}")
    print(f"  设备: {device}")
    print(
        f"  推理耗时: {inference_time_ms_mean} ms"
        f" ± {inference_time_ms_std} ms"
    )
    print(f"  吞吐: {throughput_samples_per_sec} samples/sec")
    if is_compressed_run:
        print(f"  [压缩模型] 体积: {comp_size_mb} MB, "
              f"耗时: {comp_inference_ms_mean} ms, "
              f"吞吐: {comp_throughput} samples/sec")
        print(f"  [对比] 加速比: {speedup_ratio}x, "
              f"max_abs_score_diff: {max_abs_diff}")
    print(f"  输出: {output_dir}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()