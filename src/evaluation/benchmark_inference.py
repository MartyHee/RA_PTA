"""Multimodal 模型推理性能 Benchmark 脚本

评估 baseline Multimodal 与正式 tuned Multimodal 的模型大小、参数量和推理时延。
统一使用 test split 单样本推理 (batch_size=1)。

用法:
    python src/evaluation/benchmark_inference.py \\
        --config configs/evaluation/inference_benchmark_real_raw_1000.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── 路径设置 ──────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
sys.path.insert(0, os.path.join(_project_root, "src"))

from models.multimodal.dataset import MultimodalDataset  # noqa: E402
from models.multimodal.fusion_model import MultimodalFusionModel  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

logger = get_logger("benchmark_inference")


def count_parameters(model: nn.Module) -> int:
    """返回模型可训练参数量。"""
    return sum(p.numel() for p in model.parameters())


def measure_model_size_mb(model_path: Path) -> float:
    """返回模型文件大小 (MB)。"""
    return os.path.getsize(model_path) / (1024 * 1024)


def build_model_from_config(model_config: dict, device: str) -> MultimodalFusionModel:
    """根据配置构建模型（不加载权重）。"""
    return MultimodalFusionModel(
        text_dim=model_config.get("text_dim", 32),
        visual_dim=model_config.get("visual_dim", 18),
        structured_dim=model_config.get("structured_dim", 38),
        text_hidden_dim=model_config.get("text_hidden_dim", 32),
        visual_hidden_dim=model_config.get("visual_hidden_dim", 16),
        structured_hidden_dim=model_config.get("structured_hidden_dim", 32),
        fusion_hidden_dim=model_config.get("fusion_hidden_dim", 64),
        dropout=model_config.get("dropout", 0.3),
    ).to(device)


def run_inference_benchmark(
    model: MultimodalFusionModel,
    dataset: MultimodalDataset,
    device: str,
    warmup_runs: int = 10,
    repeat_runs: int = 100,
) -> dict:
    """执行推理 benchmark。

    预热 warmup_runs 遍，然后计时 repeat_runs 遍，返回延迟统计。

    Args:
        model: 已加载权重的模型（eval mode 由本函数设置）。
        dataset: 测试集 Dataset。
        device: 推理设备。
        warmup_runs: 预热遍数。
        repeat_runs: 计时遍数。

    Returns:
        dict: 包含延迟统计的字典。
    """
    model.eval()
    n_samples = len(dataset)

    # 将所有数据预移到 device
    all_text: list[torch.Tensor] = []
    all_visual: list[torch.Tensor] = []
    all_struct: list[torch.Tensor] = []
    for i in range(n_samples):
        item = dataset[i]
        all_text.append(item["text"].unsqueeze(0).to(device))
        all_visual.append(item["visual"].unsqueeze(0).to(device))
        all_struct.append(item["structured"].unsqueeze(0).to(device))

    # 预热
    logger.info(f"  预热 {warmup_runs} 遍 × {n_samples} 样本 ...")
    for _ in range(warmup_runs):
        with torch.no_grad():
            for i in range(n_samples):
                _ = model(all_text[i], all_visual[i], all_struct[i])

    if device == "cuda":
        torch.cuda.synchronize()

    # 计时推理
    logger.info(f"  计时推理 {repeat_runs} 遍 × {n_samples} 样本 ...")
    all_latencies: list[float] = []
    for r in range(repeat_runs):
        for i in range(n_samples):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                logit = model(all_text[i], all_visual[i], all_struct[i])
                _ = torch.sigmoid(logit)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            all_latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(all_latencies)
    avg_latency = float(np.mean(latencies))
    p50_latency = float(np.median(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    total_time_s = float(np.sum(latencies) / 1000.0)
    throughput = float(n_samples * repeat_runs / total_time_s)

    return {
        "num_samples": n_samples,
        "warmup_runs": warmup_runs,
        "repeat_runs": repeat_runs,
        "total_inference_calls": n_samples * repeat_runs,
        "avg_latency_ms_per_sample": round(avg_latency, 6),
        "p50_latency_ms_per_sample": round(p50_latency, 6),
        "p95_latency_ms_per_sample": round(p95_latency, 6),
        "latency_std_ms": round(float(np.std(latencies)), 6),
        "latency_min_ms": round(float(np.min(latencies)), 6),
        "latency_max_ms": round(float(np.max(latencies)), 6),
        "total_benchmark_time_s": round(total_time_s, 4),
        "throughput_samples_per_sec": round(throughput, 2),
    }


def generate_report(
    results: list[dict],
    output_dir: Path,
    run_id: str,
    device: str,
    warmup_runs: int,
    repeat_runs: int,
    started_at: str,
    finished_at: str,
) -> None:
    """生成 benchmark 报告 markdown 文件。"""
    lines: list[str] = []
    lines.append("# Multimodal 模型推理性能 Benchmark 报告")
    lines.append("")
    lines.append(f"- **Run ID**: {run_id}")
    lines.append(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **设备**: {device}")
    lines.append(f"- **测试样本数**: {results[0]['num_samples']}")
    lines.append(f"- **Warmup**: {warmup_runs} passes")
    lines.append(f"- **Repeat**: {repeat_runs} passes")
    lines.append("")

    lines.append("## 重要声明")
    lines.append("")
    lines.append("1. **当前为离线本地推理计时，不代表线上服务延迟。**")
    lines.append("2. **结果受设备、batch_size、warmup/repeat 设置影响。**")
    lines.append("3. 所有模型基于 interaction_score 伪标签训练，不代表真实业务目标。")
    lines.append("4. 当前 benchmark 仅覆盖 Multimodal 模型，不包括 DNN / Wide & Deep / GraphSAGE。")
    lines.append("5. Tuned 模型来自 random search best trial (trial 16) 并已固化为正式 tuned run (202605091755)。")
    lines.append("")

    lines.append("## 模型版本")
    lines.append("")
    lines.append("| 版本 | 说明 |")
    lines.append("| --- | --- |")
    for r in results:
        lines.append(f"| {r['model_version']} | {r['description']} |")
    lines.append("")

    lines.append("## 模型大小与参数量")
    lines.append("")
    lines.append("| 模型版本 | 参数量 | 模型文件大小 (MB) |")
    lines.append("| --- | ---: | ---: |")
    for r in results:
        lines.append(
            f"| {r['model_version']} | {r['num_parameters']:,} | "
            f"{r['model_size_mb']:.4f} |"
        )
    lines.append("")

    lines.append("## 推理时延 (单样本, batch_size=1)")
    lines.append("")
    lines.append("| 模型版本 | Avg (ms) | P50 (ms) | P95 (ms) | Min (ms) | Max (ms) | Std (ms) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        lines.append(
            f"| {r['model_version']} | "
            f"{r['avg_latency_ms_per_sample']:.4f} | "
            f"{r['p50_latency_ms_per_sample']:.4f} | "
            f"{r['p95_latency_ms_per_sample']:.4f} | "
            f"{r['latency_min_ms']:.4f} | "
            f"{r['latency_max_ms']:.4f} | "
            f"{r['latency_std_ms']:.4f} |"
        )
    lines.append("")

    lines.append("## 吞吐量")
    lines.append("")
    lines.append("| 模型版本 | 吞吐量 (samples/sec) |")
    lines.append("| --- | ---: |")
    for r in results:
        lines.append(
            f"| {r['model_version']} | {r['throughput_samples_per_sec']:.2f} |"
        )
    lines.append("")

    # 对比 summary
    if len(results) == 2:
        r0, r1 = results[0], results[1]
        param_ratio = r1["num_parameters"] / r0["num_parameters"]
        size_ratio = r1["model_size_mb"] / r0["model_size_mb"]
        lat_ratio = r1["avg_latency_ms_per_sample"] / r0["avg_latency_ms_per_sample"]
        thr_ratio = r1["throughput_samples_per_sec"] / r0["throughput_samples_per_sec"]

        lines.append("## Tuned vs Baseline 对比")
        lines.append("")
        lines.append("| 指标 | Baseline | Tuned | Tuned/Baseline |")
        lines.append("| --- | ---: | ---: | ---: |")
        lines.append(
            f"| 参数量 | {r0['num_parameters']:,} | {r1['num_parameters']:,} | "
            f"{param_ratio:.2f}× |"
        )
        lines.append(
            f"| 模型文件大小 | {r0['model_size_mb']:.4f} MB | "
            f"{r1['model_size_mb']:.4f} MB | {size_ratio:.2f}× |"
        )
        lines.append(
            f"| Avg 延迟 | {r0['avg_latency_ms_per_sample']:.4f} ms | "
            f"{r1['avg_latency_ms_per_sample']:.4f} ms | {lat_ratio:.2f}× |"
        )
        lines.append(
            f"| P50 延迟 | {r0['p50_latency_ms_per_sample']:.4f} ms | "
            f"{r1['p50_latency_ms_per_sample']:.4f} ms | "
            f"{r1['p50_latency_ms_per_sample'] / r0['p50_latency_ms_per_sample']:.2f}× |"
        )
        lines.append(
            f"| P95 延迟 | {r0['p95_latency_ms_per_sample']:.4f} ms | "
            f"{r1['p95_latency_ms_per_sample']:.4f} ms | "
            f"{r1['p95_latency_ms_per_sample'] / r0['p95_latency_ms_per_sample']:.2f}× |"
        )
        lines.append(
            f"| 吞吐量 | {r0['throughput_samples_per_sec']:.2f} samples/s | "
            f"{r1['throughput_samples_per_sec']:.2f} samples/s | {thr_ratio:.2f}× |"
        )
        lines.append("")

    lines.append("## Benchmark 设置")
    lines.append("")
    lines.append(f"- **Warmup passes**: {warmup_runs}")
    lines.append(f"- **Repeat passes**: {repeat_runs}")
    lines.append(f"- **Batch size**: 1（单样本推理）")
    lines.append(f"- **Device**: {device}")
    lines.append(f"- **测试样本数**: {results[0]['num_samples']}")
    lines.append(f"- **起止时间**: {started_at} → {finished_at}")
    lines.append("")

    lines.append("## 结论")
    lines.append("")
    lines.append(
        "本次 benchmark 评估了 baseline Multimodal 与正式 tuned Multimodal "
        "两个版本在相同 test split (150 样本) 上的推理性能。"
    )
    lines.append("")
    lines.append(
        "所有延迟数据为离线单样本推理计时，不代表线上服务延迟。"
        "结果受设备、CUDA 驱动版本、PyTorch 版本和 benchmark 参数设置影响。"
    )
    lines.append("")

    report_path = output_dir / "benchmark_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Benchmark 报告已保存: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal 模型推理性能 benchmark"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/evaluation/inference_benchmark_real_raw_1000.yaml",
        help="benchmark 配置文件路径（相对于项目根目录）",
    )
    args = parser.parse_args()

    project_root = Path(_project_root)
    benchmark_config = load_config(args.config)

    # ── 生成 run_id ──────────────────────────────────────
    run_id = datetime.now().strftime("%Y%m%d%H%M")
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Benchmark Run ID: {run_id}")

    # ── 随机种子 ─────────────────────────────────────────
    set_seed(benchmark_config.get("random_seed", 42))

    # ── 输出目录 ─────────────────────────────────────────
    output_root = project_root / benchmark_config["output_root"]
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")

    # ── 加载 feature_info（获取输入维度）────────────────
    feature_info_path = project_root / benchmark_config["feature_info_path"]
    with open(feature_info_path, "r", encoding="utf-8") as f:
        feature_info = json.load(f)

    input_dims = {
        "text": feature_info.get("text_dim", 32),
        "visual": feature_info.get("visual_dim", 18),
        "structured": feature_info.get("structured_dim", 38),
    }
    logger.info(f"输入维度: {input_dims}")

    # ── Device ───────────────────────────────────────────
    device = benchmark_config.get("device", "cuda")
    device_fallback = False
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，回退到 CPU")
        device = "cpu"
        device_fallback = True
    logger.info(f"设备: {device}")

    # ── Test split ──────────────────────────────────────
    test_npz_path = project_root / benchmark_config["test_npz_path"]
    test_dataset = MultimodalDataset(test_npz_path, feature_info)
    logger.info(f"测试样本: {len(test_dataset)}")

    warmup_runs = benchmark_config.get("warmup_runs", 10)
    repeat_runs = benchmark_config.get("repeat_runs", 100)
    logger.info(f"Warmup: {warmup_runs} passes, Repeat: {repeat_runs} passes")

    # ── 对每个模型版本做 benchmark ─────────────────────
    results: list[dict] = []
    model_descriptions: dict[str, str] = {}

    for mv in benchmark_config["model_versions"]:
        name: str = mv["name"]
        description: str = mv.get("description", "")
        model_config_path = project_root / mv["config_path"]
        model_file = project_root / mv["model_path"]

        model_descriptions[name] = description

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Benchmark: {name} — {description}")
        logger.info(f"  配置: {model_config_path}")
        logger.info(f"  权重: {model_file}")

        if not model_file.exists():
            logger.error(f"  模型文件不存在，跳过: {model_file}")
            continue

        # 加载模型配置
        model_config = load_config(str(model_config_path))
        # 合并输入维度（feature_info 里的 text_dim 等）
        model_config.update(input_dims)

        # 构建模型
        model = build_model_from_config(model_config, device)
        n_params = count_parameters(model)
        logger.info(f"  参数量: {n_params}")

        model_size_mb_val = measure_model_size_mb(model_file)
        logger.info(f"  模型文件大小: {model_size_mb_val:.4f} MB")

        # 加载权重
        state_dict = torch.load(str(model_file), map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"  权重加载完成")

        # 推理 benchmark
        bench = run_inference_benchmark(
            model=model,
            dataset=test_dataset,
            device=device,
            warmup_runs=warmup_runs,
            repeat_runs=repeat_runs,
        )

        result = {
            "model_version": name,
            "description": description,
            "model_path": str(model_file),
            "num_parameters": n_params,
            "model_size_mb": round(model_size_mb_val, 4),
            "device": device,
            **bench,
        }
        results.append(result)

        # 打印摘要
        logger.info(f"  Avg 延迟: {result['avg_latency_ms_per_sample']:.4f} ms/sample")
        logger.info(f"  P50 延迟: {result['p50_latency_ms_per_sample']:.4f} ms/sample")
        logger.info(f"  P95 延迟: {result['p95_latency_ms_per_sample']:.4f} ms/sample")
        logger.info(
            f"  吞吐量: {result['throughput_samples_per_sec']:.2f} samples/sec"
        )

    if not results:
        logger.error("所有模型版本均失败，无 benchmark 结果")
        sys.exit(1)

    # ── 保存 inference_benchmark.csv ─────────────────────
    results_df = pd.DataFrame(results)
    csv_path = output_dir / "inference_benchmark.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Benchmark CSV 已保存: {csv_path}")

    # ── 保存 run_meta.json ──────────────────────────────
    finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_meta = {
        "task": "multimodal_inference_benchmark",
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "config_path": args.config,
        "test_npz_path": str(test_npz_path),
        "feature_info_path": str(feature_info_path),
        "test_samples": len(test_dataset),
        "warmup_runs": warmup_runs,
        "repeat_runs": repeat_runs,
        "device": device,
        "device_fallback": device_fallback,
        "model_versions": [mv["name"] for mv in benchmark_config["model_versions"]],
        "results": results,
    }
    with open(output_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    logger.info(f"Run meta 已保存: {output_dir / 'run_meta.json'}")

    # ── 生成 benchmark_report.md ────────────────────────
    generate_report(
        results=results,
        output_dir=output_dir,
        run_id=run_id,
        device=device,
        warmup_runs=warmup_runs,
        repeat_runs=repeat_runs,
        started_at=started_at,
        finished_at=finished_at,
    )

    # ── 汇总输出 ────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Benchmark 完成！")
    logger.info(f"输出目录: {output_dir}")
    for r in results:
        logger.info(
            f"  {r['model_version']}: "
            f"{r['num_parameters']:,} params, "
            f"{r['model_size_mb']:.2f} MB, "
            f"avg {r['avg_latency_ms_per_sample']:.4f} ms, "
            f"P50 {r['p50_latency_ms_per_sample']:.4f} ms, "
            f"P95 {r['p95_latency_ms_per_sample']:.4f} ms, "
            f"{r['throughput_samples_per_sec']:.0f} samples/sec"
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()