"""Multimodal 动态量化压缩实验脚本

对正式 tuned Multimodal (run_id=202605091755) 进行 PyTorch 动态量化，
评估量化前后离线指标、模型大小和推理时延变化。

用法:
    python src/models/multimodal/quantize.py \
        --config configs/multimodal/multimodal_quantization_real_raw_1000_tuned.yaml
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
_project_root = os.path.abspath(os.path.join(_script_dir, "..", "..", ".."))
sys.path.insert(0, os.path.join(_project_root, "src"))

from evaluation.metrics import compute_classification_metrics  # noqa: E402
from evaluation.ranking_metrics import (  # noqa: E402
    compute_precision_at_k,
    compute_recall_at_k,
)
from models.multimodal.dataset import MultimodalDataset  # noqa: E402
from models.multimodal.fusion_model import MultimodalFusionModel  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

logger = get_logger("multimodal_quantize")


# ── 工具函数 ──────────────────────────────────────────────


def count_parameters(model: nn.Module) -> int:
    """返回模型可训练参数量（量化前调用）。"""
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
        text_hidden_dim=model_config.get("text_hidden_dim", 16),
        visual_hidden_dim=model_config.get("visual_hidden_dim", 8),
        structured_hidden_dim=model_config.get("structured_hidden_dim", 16),
        fusion_hidden_dim=model_config.get("fusion_hidden_dim", 32),
        dropout=model_config.get("dropout", 0.123),
    ).to(device)


# ── 量化函数 ──────────────────────────────────────────────


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """对模型 Linear 层应用 PyTorch 动态量化。

    将 nn.Linear 替换为 dynamically quantized Linear（weights qint8，
    activations 动态量化）。量化后的模型在 CPU 上运行。

    Args:
        model: 原始 FP32 模型。

    Returns:
        量化后的模型（in-place 转换）。
    """
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear},
        dtype=torch.qint8,
    )
    return quantized_model


# ── 评估函数 ──────────────────────────────────────────────


def evaluate_model(
    model: nn.Module,
    dataset: MultimodalDataset,
    device: str,
    threshold: float,
) -> dict:
    """在 test split 上评估模型，返回指标和预测。

    Args:
        model: 模型（eval mode 由本函数设置）。
        dataset: 测试集 Dataset。
        device: 推理设备。
        threshold: 分类阈值。

    Returns:
        dict: 包含指标和预测结果。
    """
    model.eval()
    all_logits: list[float] = []
    all_scores: list[float] = []
    all_labels: list[float] = []
    all_preds: list[int] = []
    all_video_ids: list[int] = []

    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            text_t = item["text"].unsqueeze(0).to(device)
            visual_t = item["visual"].unsqueeze(0).to(device)
            struct_t = item["structured"].unsqueeze(0).to(device)

            logit = model(text_t, visual_t, struct_t)
            score = torch.sigmoid(logit)

            all_logits.append(logit.cpu().item())
            all_scores.append(score.cpu().item())
            all_labels.append(item["label"].item())
            all_video_ids.append(int(item["video_id"]))

    all_scores_arr = np.array(all_scores)
    all_labels_arr = np.array(all_labels)
    all_preds_arr = (all_scores_arr >= threshold).astype(int)

    # 分类指标
    cls_metrics, cls_warnings = compute_classification_metrics(
        all_labels_arr, all_scores_arr, all_preds_arr, threshold
    )

    # 排序指标
    k_values = [5, 10, 20]
    pk_metrics, pk_warnings = compute_precision_at_k(
        all_labels_arr, all_scores_arr, k_values
    )
    rk_metrics, rk_warnings = compute_recall_at_k(
        all_labels_arr, all_scores_arr, k_values
    )

    n_pos = int(all_labels_arr.sum())
    n_neg = int(len(all_labels_arr) - n_pos)

    # BCE loss
    all_logits_arr = np.array(all_logits)
    logits_tensor = torch.tensor(all_logits_arr)
    labels_tensor = torch.tensor(all_labels_arr)
    eval_loss = nn.BCEWithLogitsLoss()(logits_tensor, labels_tensor).item()

    return {
        "video_ids": all_video_ids,
        "scores": all_scores_arr,
        "preds": all_preds_arr,
        "labels": all_labels_arr,
        "eval_loss": eval_loss,
        "cls_metrics": cls_metrics,
        "pk_metrics": pk_metrics,
        "rk_metrics": rk_metrics,
        "warnings": cls_warnings + pk_warnings + rk_warnings,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


# ── Benchmark 函数 ────────────────────────────────────────


def run_inference_benchmark(
    model: nn.Module,
    dataset: MultimodalDataset,
    device: str,
    warmup_runs: int = 10,
    repeat_runs: int = 100,
) -> dict:
    """执行推理 benchmark（适配 CPU / CUDA）。

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
    for _ in range(repeat_runs):
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


# ── 报告生成 ──────────────────────────────────────────────


def generate_report(
    tuned_metrics: dict,
    quantized_eval: dict,
    quantized_bench: dict,
    fp32_bench: dict | None,
    source_bench: dict | None,
    output_dir: Path,
    source_model_run_id: str,
    source_benchmark_run_id: str,
    quantization_method: str,
    quantized_layers: list[str],
    device: str,
    warmup_runs: int,
    repeat_runs: int,
    started_at: str,
    finished_at: str,
) -> None:
    """生成压缩报告 markdown 文件。"""
    lines: list[str] = []
    lines.append("# Multimodal 动态量化压缩实验报告")
    lines.append("")
    lines.append(f"- **Run ID**: {output_dir.name}")
    lines.append(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **量化方法**: {quantization_method}")
    lines.append(f"- **量化层**: {', '.join(quantized_layers)}")
    lines.append(f"- **Benchmark 设备**: {device}")
    lines.append(f"- **测试样本数**: {quantized_eval['n_pos'] + quantized_eval['n_neg']}")
    lines.append(f"- **Warmup**: {warmup_runs} passes")
    lines.append(f"- **Repeat**: {repeat_runs} passes")
    lines.append("")

    lines.append("## 重要声明")
    lines.append("")
    lines.append("1. **当前为离线本地推理计时，不代表线上服务延迟。**")
    lines.append("2. 输入模型为正式 tuned Multimodal (run_id=%s)。" % source_model_run_id)
    lines.append("3. 使用 PyTorch 动态量化（`torch.quantization.quantize_dynamic`），仅量化 Linear 层。")
    lines.append("4. **当前模型参数量仅 2,649，延迟主要受框架开销主导，量化不一定降低延迟。**")
    lines.append("5. Quantized 模型运行在 CPU（动态量化 Linear 层仅支持 CPU），")
    lines.append("   FP32 CPU benchmark 用于同设备公平对比。")
    lines.append("6. 前置 CUDA benchmark (run_id=%s) 数据仅作为参考。" % source_benchmark_run_id)
    lines.append("7. test split 仅用于最终离线评估，不参与训练或超参选择。")
    lines.append("8. **当前结果不代表线上服务延迟或业务效果。**")
    lines.append("")

    lines.append("## 模型版本")
    lines.append("")
    lines.append("| 版本 | 说明 |")
    lines.append("| --- | --- |")
    lines.append("| tuned_multimodal_fp32_cuda | 正式 tuned FP32 CUDA（前置 benchmark 参考）|")
    lines.append("| tuned_multimodal_fp32_cpu | 正式 tuned FP32 CPU（同设备公平对比）|")
    lines.append("| tuned_multimodal_quantized | 动态量化后模型 CPU |")
    lines.append("")

    lines.append("## 模型大小与参数量")
    lines.append("")
    lines.append("| 模型版本 | 参数量 | 模型文件大小 (MB) |")
    lines.append("| --- | ---: | ---: |")
    lines.append(
        f"| tuned_multimodal (FP32) | {tuned_metrics['num_parameters']:,} | "
        f"{tuned_metrics['model_size_mb']:.4f} |"
    )
    lines.append(
        f"| tuned_multimodal_quantized | {quantized_eval['num_parameters']:,} | "
        f"{quantized_eval['model_size_mb']:.4f} |"
    )
    lines.append("")

    lines.append("## 离线评估指标 (test split)")
    lines.append("")
    lines.append("| 指标 | FP32 (原始) | Quantized |")
    lines.append("| --- | ---: | ---: |")
    lines.append(
        f"| AUC | {tuned_metrics['auc']:.6f} | "
        f"{quantized_eval['auc']:.6f} |"
    )
    lines.append(
        f"| Accuracy | {tuned_metrics['accuracy']:.6f} | "
        f"{quantized_eval['accuracy']:.6f} |"
    )
    lines.append(
        f"| Precision | {tuned_metrics['precision']:.6f} | "
        f"{quantized_eval['precision']:.6f} |"
    )
    lines.append(
        f"| Recall | {tuned_metrics['recall']:.6f} | "
        f"{quantized_eval['recall']:.6f} |"
    )
    lines.append(
        f"| F1 | {tuned_metrics['f1']:.6f} | "
        f"{quantized_eval['f1']:.6f} |"
    )
    lines.append("")
    lines.append(
        f"注: FP32 和 Quantized 均在 test split ({quantized_eval['n_pos']} 正 / "
        f"{quantized_eval['n_neg']} 负) 上评估。"
    )
    lines.append("")

    lines.append("## 推理时延对比")
    lines.append("")
    lines.append("| 模型版本 | Device | Avg (ms) | P50 (ms) | P95 (ms) | Min (ms) | Max (ms) |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")

    if source_bench:
        lines.append(
            f"| tuned_fp32 (参考) | CUDA | "
            f"{source_bench['avg_latency_ms_per_sample']:.4f} | "
            f"{source_bench['p50_latency_ms_per_sample']:.4f} | "
            f"{source_bench['p95_latency_ms_per_sample']:.4f} | "
            f"{source_bench['latency_min_ms']:.4f} | "
            f"{source_bench['latency_max_ms']:.4f} |"
        )
    if fp32_bench:
        lines.append(
            f"| tuned_fp32 (公平对比) | CPU | "
            f"{fp32_bench['avg_latency_ms_per_sample']:.4f} | "
            f"{fp32_bench['p50_latency_ms_per_sample']:.4f} | "
            f"{fp32_bench['p95_latency_ms_per_sample']:.4f} | "
            f"{fp32_bench['latency_min_ms']:.4f} | "
            f"{fp32_bench['latency_max_ms']:.4f} |"
        )
    lines.append(
        f"| tuned_quantized | CPU | "
        f"{quantized_bench['avg_latency_ms_per_sample']:.4f} | "
        f"{quantized_bench['p50_latency_ms_per_sample']:.4f} | "
        f"{quantized_bench['p95_latency_ms_per_sample']:.4f} | "
        f"{quantized_bench['latency_min_ms']:.4f} | "
        f"{quantized_bench['latency_max_ms']:.4f} |"
    )
    lines.append("")

    lines.append("## 吞吐量对比")
    lines.append("")
    lines.append("| 模型版本 | Device | 吞吐量 (samples/sec) |")
    lines.append("| --- | --- | ---: |")
    if source_bench:
        lines.append(
            f"| tuned_fp32 (参考) | CUDA | {source_bench['throughput_samples_per_sec']:.2f} |"
        )
    if fp32_bench:
        lines.append(
            f"| tuned_fp32 (公平对比) | CPU | {fp32_bench['throughput_samples_per_sec']:.2f} |"
        )
    lines.append(
        f"| tuned_quantized | CPU | "
        f"{quantized_bench['throughput_samples_per_sec']:.2f} |"
    )
    lines.append("")

    lines.append("## 量化前后对比总结")
    lines.append("")
    lines.append("| 对比项 | FP32 | Quantized | 变化 |")
    lines.append("| --- | ---: | ---: | ---: |")

    size_ratio = quantized_eval["model_size_mb"] / tuned_metrics["model_size_mb"]
    lines.append(
        f"| 模型大小 | {tuned_metrics['model_size_mb']:.4f} MB | "
        f"{quantized_eval['model_size_mb']:.4f} MB | "
        f"{size_ratio:.2f}× |"
    )

    lat_delta = quantized_bench["avg_latency_ms_per_sample"]
    fp32_lat = fp32_bench["avg_latency_ms_per_sample"] if fp32_bench else None
    if fp32_lat:
        lat_ratio = lat_delta / fp32_lat
        lines.append(
            f"| Avg 延迟 | {fp32_lat:.4f} ms | {lat_delta:.4f} ms | "
            f"{lat_ratio:.2f}× |"
        )

    auc_diff = quantized_eval["auc"] - tuned_metrics["auc"]
    lines.append(
        f"| AUC | {tuned_metrics['auc']:.6f} | "
        f"{quantized_eval['auc']:.6f} | "
        f"{auc_diff:+.6f} |"
    )

    f1_diff = quantized_eval["f1"] - tuned_metrics["f1"]
    lines.append(
        f"| F1 | {tuned_metrics['f1']:.6f} | "
        f"{quantized_eval['f1']:.6f} | "
        f"{f1_diff:+.6f} |"
    )
    lines.append("")

    lines.append("## 分析")
    lines.append("")
    lines.append(
        "1. **模型大小**: 动态量化将 Linear 权重从 FP32 转为 INT8，"
        "预期模型文件大小可降至约 1/4（仅 Linear 层权重压缩）。"
        "对于极小模型（2,649 参数），绝对节省有限。"
    )
    lines.append(
        "2. **离线指标**: 动态量化对 Linear 权重做 INT8 近似，"
        "通常会导致轻微精度损失。需要关注 AUC/F1 的变化幅度。"
    )
    lines.append(
        "3. **推理时延**: 量化模型运行在 CPU，FP32 基准也运行在 CPU 做公平对比。"
        "CPU 推理延迟显著高于 CUDA，这是设备差异而非量化代价。"
    )
    lines.append(
        "4. **框架开销**: 当前模型仅 2,649 参数，"
        "量化对计算量的减少被 PyTorch 框架调度开销淹没。"
    )
    lines.append("")

    lines.append("## Benchmark 设置")
    lines.append("")
    lines.append(f"- **Warmup passes**: {warmup_runs}")
    lines.append(f"- **Repeat passes**: {repeat_runs}")
    lines.append(f"- **Batch size**: 1（单样本推理）")
    lines.append(f"- **Device**: {device}")
    lines.append(f"- **测试样本数**: {quantized_eval['n_pos'] + quantized_eval['n_neg']}")
    lines.append(f"- **起止时间**: {started_at} → {finished_at}")
    lines.append("")

    lines.append("## 结论")
    lines.append("")
    lines.append(
        "本次实验对正式 tuned Multimodal (run_id=%s) 进行了 PyTorch 动态量化，"
        "评估了量化前后在 test split 上的离线指标、模型大小和推理时延。"
        % source_model_run_id
    )
    lines.append("")
    lines.append(
        "所有延迟数据为离线单样本推理计时，不代表线上服务延迟。"
        "结果受设备、CPU/GPU 架构、PyTorch 版本和 benchmark 参数设置影响。"
    )
    lines.append("")

    report_path = output_dir / "compression_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"压缩报告已保存: {report_path}")


# ── 主程序 ──────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal 动态量化压缩实验"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multimodal/multimodal_quantization_real_raw_1000_tuned.yaml",
        help="量化配置文件路径（相对于项目根目录）",
    )
    args = parser.parse_args()

    project_root = Path(_project_root)
    config = load_config(args.config)

    # ── 生成 run_id ──────────────────────────────────────
    run_id = datetime.now().strftime("%Y%m%d%H%M")
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"量化实验 Run ID: {run_id}")

    # ── 随机种子 ─────────────────────────────────────────
    set_seed(config.get("random_seed", 42))

    # ── 输出目录 ─────────────────────────────────────────
    output_root = project_root / config["output_root"]
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")

    # ── 配置参数 ─────────────────────────────────────────
    source_model_run_id = config["source_model_run_id"]
    source_model_config_path = config["source_model_config_path"]
    source_model_path = project_root / config["source_model_path"]
    source_benchmark_run_id = config["source_benchmark_run_id"]
    source_benchmark_dir = project_root / config["source_benchmark_dir"]

    quantization_config = config.get("quantization", {})
    quantization_method = quantization_config.get("method", "dynamic_quantization")
    quantized_layers_config = quantization_config.get("quantized_layers", ["nn.Linear"])

    test_npz_path = project_root / config["test_npz_path"]
    feature_info_path = project_root / config["feature_info_path"]
    warmup_runs = config.get("warmup_runs", 10)
    repeat_runs = config.get("repeat_runs", 100)
    device = config.get("device", "cpu")
    threshold = config.get("threshold", 0.5)

    # ── 加载 feature_info ────────────────────────────────
    with open(feature_info_path, "r", encoding="utf-8") as f:
        feature_info = json.load(f)

    input_dims = {
        "text": feature_info.get("text_dim", 32),
        "visual": feature_info.get("visual_dim", 18),
        "structured": feature_info.get("structured_dim", 38),
    }
    logger.info(f"输入维度: {input_dims}")

    # ── 加载 test dataset ────────────────────────────────
    test_dataset = MultimodalDataset(test_npz_path, feature_info)
    logger.info(f"测试样本: {len(test_dataset)}")

    # ── 1. 加载源模型配置和权重 ─────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"加载源模型: {source_model_path}")

    if not source_model_path.exists():
        logger.error(f"源模型文件不存在: {source_model_path}")
        sys.exit(1)

    model_config = load_config(str(project_root / source_model_config_path))
    model_config.update(input_dims)

    model = build_model_from_config(model_config, device)
    state_dict = torch.load(str(source_model_path), map_location=device)
    model.load_state_dict(state_dict)
    logger.info("源模型权重加载完成")

    n_params = count_parameters(model)
    logger.info(f"参数量（量化前）: {n_params}")

    original_size_mb = measure_model_size_mb(source_model_path)
    logger.info(f"源模型文件大小: {original_size_mb:.4f} MB")

    # ── 2. 评估原始 FP32 模型（test split）──────────────
    logger.info("")
    logger.info("─" * 40)
    logger.info("评估原始 FP32 模型（test split）...")
    fp32_eval = evaluate_model(model, test_dataset, device, threshold)
    logger.info(f"  FP32 AUC: {fp32_eval['cls_metrics']['auc']:.6f}")
    logger.info(f"  FP32 F1:  {fp32_eval['cls_metrics']['f1']:.6f}")

    # 保存 FP32 predictions
    fp32_pred_df = pd.DataFrame({
        "video_id": fp32_eval["video_ids"],
        "label": fp32_eval["labels"],
        "score": fp32_eval["scores"],
        "pred": fp32_eval["preds"],
        "split": "test",
        "model_name": "tuned_multimodal_fp32",
        "run_id": run_id,
    })
    fp32_pred_df.to_csv(
        output_dir / "predictions_fp32.csv", index=False, encoding="utf-8-sig"
    )
    logger.info(f"  FP32 predictions 已保存: {output_dir / 'predictions_fp32.csv'}")

    # ── 3. 应用动态量化 ─────────────────────────────────
    logger.info("")
    logger.info("─" * 40)
    logger.info("应用 PyTorch 动态量化...")
    quantized_model = apply_dynamic_quantization(model)

    # 量化的模型必须在 CPU 上运行
    quantized_model = quantized_model.cpu()
    logger.info("动态量化完成，量化模型移至 CPU")

    # 保存量化模型
    quantized_model_path = output_dir / "model_quantized.pt"
    torch.save(quantized_model.state_dict(), str(quantized_model_path))
    quantized_size_mb = measure_model_size_mb(quantized_model_path)
    logger.info(f"量化模型已保存: {quantized_model_path}")
    logger.info(f"量化模型文件大小: {quantized_size_mb:.4f} MB")

    # ── 4. 评估量化模型（CPU）────────────────────────────
    logger.info("")
    logger.info("─" * 40)
    logger.info("评估量化模型（CPU）...")
    quantized_eval = evaluate_model(
        quantized_model, test_dataset, "cpu", threshold
    )
    logger.info(f"  Quantized AUC: {quantized_eval['cls_metrics']['auc']:.6f}")
    logger.info(f"  Quantized F1:  {quantized_eval['cls_metrics']['f1']:.6f}")

    # 保存 quantized predictions
    quantized_pred_df = pd.DataFrame({
        "video_id": quantized_eval["video_ids"],
        "label": quantized_eval["labels"],
        "score": quantized_eval["scores"],
        "pred": quantized_eval["preds"],
        "split": "test",
        "model_name": "tuned_multimodal_quantized",
        "run_id": run_id,
    })
    quantized_pred_df.to_csv(
        output_dir / "predictions.csv", index=False, encoding="utf-8-sig"
    )
    logger.info(f"  Quantized predictions 已保存: {output_dir / 'predictions.csv'}")

    # ── 5. 量化模型 benchmark（CPU）─────────────────────
    logger.info("")
    logger.info("─" * 40)
    logger.info("量化模型推理 benchmark（CPU）...")
    quantized_bench = run_inference_benchmark(
        model=quantized_model,
        dataset=test_dataset,
        device="cpu",
        warmup_runs=warmup_runs,
        repeat_runs=repeat_runs,
    )
    logger.info(
        f"  Quantized Avg 延迟: {quantized_bench['avg_latency_ms_per_sample']:.4f} ms"
    )
    logger.info(
        f"  Quantized 吞吐量: {quantized_bench['throughput_samples_per_sec']:.2f} samples/s"
    )

    # ── 6. 原始 FP32 模型 benchmark（CPU，公平对比）─────
    logger.info("")
    logger.info("─" * 40)
    logger.info("原始 FP32 模型推理 benchmark（CPU，公平对比）...")

    # 重新构建并加载原始 FP32 模型（因为 model 已被 in-place 量化）
    fp32_model = build_model_from_config(model_config, "cpu")
    fp32_state_dict = torch.load(str(source_model_path), map_location="cpu")
    fp32_model.load_state_dict(fp32_state_dict)

    fp32_bench = run_inference_benchmark(
        model=fp32_model,
        dataset=test_dataset,
        device="cpu",
        warmup_runs=warmup_runs,
        repeat_runs=repeat_runs,
    )
    logger.info(
        f"  FP32 Avg 延迟: {fp32_bench['avg_latency_ms_per_sample']:.4f} ms"
    )
    logger.info(
        f"  FP32 吞吐量: {fp32_bench['throughput_samples_per_sec']:.2f} samples/s"
    )

    # ── 7. 加载前置 CUDA benchmark 结果（参考）──────────
    source_bench = None
    source_bench_path = source_benchmark_dir / "run_meta.json"
    if source_bench_path.exists():
        try:
            with open(source_bench_path, "r", encoding="utf-8") as f:
                source_meta = json.load(f)
            for r in source_meta.get("results", []):
                if r["model_version"] == "tuned_multimodal":
                    source_bench = r
                    break
            if source_bench:
                logger.info("前置 CUDA benchmark 数据已加载（参考）")
            else:
                logger.warning("前置 benchmark 中未找到 tuned_multimodal 结果")
        except Exception as e:
            logger.warning(f"前置 benchmark 数据加载失败: {e}")

    # ── 8. 保存 metrics.json（量化后指标）────────────────
    q_metrics = quantized_eval["cls_metrics"]
    q_pk = quantized_eval["pk_metrics"]
    q_rk = quantized_eval["rk_metrics"]
    metrics_data = {
        "model_name": "multimodal",
        "dataset_name": "real_raw_1000",
        "run_id": run_id,
        "split": "test",
        "source_model_run_id": source_model_run_id,
        "quantization_method": quantization_method,
        "sample_count": len(quantized_eval["labels"]),
        "positive_count": quantized_eval["n_pos"],
        "negative_count": quantized_eval["n_neg"],
        "eval_loss": quantized_eval["eval_loss"],
        "auc": q_metrics.get("auc"),
        "accuracy": q_metrics.get("accuracy"),
        "precision": q_metrics.get("precision"),
        "recall": q_metrics.get("recall"),
        "f1": q_metrics.get("f1"),
        "precision_at_k": {
            f"precision_at_{k.split('_')[-1]}": q_pk.get(k)
            for k in ["precision_at_5", "precision_at_10", "precision_at_20"]
            if k in q_pk
        },
        "recall_at_k": {
            f"recall_at_{k.split('_')[-1]}": q_rk.get(k)
            for k in ["recall_at_5", "recall_at_10", "recall_at_20"]
            if k in q_rk
        },
        "threshold": threshold,
        "num_parameters": n_params,
        "model_size_mb": round(quantized_size_mb, 4),
        "warnings": quantized_eval["warnings"],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, ensure_ascii=False, indent=2)
    logger.info(f"量化后 metrics 已保存: {output_dir / 'metrics.json'}")

    # ── 9. 保存 compression_summary.csv ─────────────────
    # 收集 FP32 原始指标（从评估结果获取）
    fp32_cls = fp32_eval["cls_metrics"]

    summary_rows = []
    # 行 1: 原始 FP32 (CUDA 参考)
    if source_bench:
        summary_rows.append({
            "model_version": "tuned_multimodal_fp32_cuda",
            "auc": source_bench.get("auc", fp32_cls.get("auc")),
            "accuracy": source_bench.get("accuracy", fp32_cls.get("accuracy")),
            "precision": source_bench.get("precision", fp32_cls.get("precision")),
            "recall": source_bench.get("recall", fp32_cls.get("recall")),
            "f1": source_bench.get("f1", fp32_cls.get("f1")),
            "num_parameters": n_params,
            "model_size_mb": round(original_size_mb, 4),
            "avg_latency_ms_per_sample": source_bench["avg_latency_ms_per_sample"],
            "p50_latency_ms_per_sample": source_bench["p50_latency_ms_per_sample"],
            "p95_latency_ms_per_sample": source_bench["p95_latency_ms_per_sample"],
            "throughput_samples_per_sec": source_bench["throughput_samples_per_sec"],
            "device": "cuda",
            "quantized": False,
        })

    # 行 2: 原始 FP32 (CPU，公平对比)
    if fp32_bench:
        summary_rows.append({
            "model_version": "tuned_multimodal_fp32_cpu",
            "auc": fp32_cls.get("auc"),
            "accuracy": fp32_cls.get("accuracy"),
            "precision": fp32_cls.get("precision"),
            "recall": fp32_cls.get("recall"),
            "f1": fp32_cls.get("f1"),
            "num_parameters": n_params,
            "model_size_mb": round(original_size_mb, 4),
            "avg_latency_ms_per_sample": fp32_bench["avg_latency_ms_per_sample"],
            "p50_latency_ms_per_sample": fp32_bench["p50_latency_ms_per_sample"],
            "p95_latency_ms_per_sample": fp32_bench["p95_latency_ms_per_sample"],
            "throughput_samples_per_sec": fp32_bench["throughput_samples_per_sec"],
            "device": "cpu",
            "quantized": False,
        })

    # 行 3: 量化模型
    summary_rows.append({
        "model_version": "tuned_multimodal_quantized",
        "auc": q_metrics.get("auc"),
        "accuracy": q_metrics.get("accuracy"),
        "precision": q_metrics.get("precision"),
        "recall": q_metrics.get("recall"),
        "f1": q_metrics.get("f1"),
        "num_parameters": n_params,
        "model_size_mb": round(quantized_size_mb, 4),
        "avg_latency_ms_per_sample": quantized_bench["avg_latency_ms_per_sample"],
        "p50_latency_ms_per_sample": quantized_bench["p50_latency_ms_per_sample"],
        "p95_latency_ms_per_sample": quantized_bench["p95_latency_ms_per_sample"],
        "throughput_samples_per_sec": quantized_bench["throughput_samples_per_sec"],
        "device": "cpu",
        "quantized": True,
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / "compression_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"压缩汇总表已保存: {summary_csv_path}")

    # ── 10. 保存 run_meta.json ──────────────────────────
    finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_meta = {
        "task": "multimodal_dynamic_quantization",
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "config_path": args.config,
        "source_model_run_id": source_model_run_id,
        "source_model_path": str(source_model_path),
        "source_model_config_path": str(project_root / source_model_config_path),
        "source_benchmark_run_id": source_benchmark_run_id,
        "source_benchmark_dir": str(source_benchmark_dir),
        "quantization_method": quantization_method,
        "quantized_dtype": quantization_config.get("dtype", "qint8"),
        "quantized_layers": quantized_layers_config,
        "device": device,
        "warmup_runs": warmup_runs,
        "repeat_runs": repeat_runs,
        "test_npz_path": str(test_npz_path),
        "feature_info_path": str(feature_info_path),
        "num_parameters": n_params,
        "model_size_fp32_mb": round(original_size_mb, 4),
        "model_size_quantized_mb": round(quantized_size_mb, 4),
        "size_reduction_ratio": round(quantized_size_mb / original_size_mb, 4),
        "fp32_eval": {
            "eval_loss": fp32_eval["eval_loss"],
            "auc": fp32_cls.get("auc"),
            "accuracy": fp32_cls.get("accuracy"),
            "precision": fp32_cls.get("precision"),
            "recall": fp32_cls.get("recall"),
            "f1": fp32_cls.get("f1"),
        },
        "quantized_eval": {
            "eval_loss": quantized_eval["eval_loss"],
            "auc": q_metrics.get("auc"),
            "accuracy": q_metrics.get("accuracy"),
            "precision": q_metrics.get("precision"),
            "recall": q_metrics.get("recall"),
            "f1": q_metrics.get("f1"),
        },
        "summary_rows": summary_rows,
    }
    with open(output_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    logger.info(f"Run meta 已保存: {output_dir / 'run_meta.json'}")

    # ── 11. 生成 compression_report.md ──────────────────
    # 构造 tuned_metrics dict 供报告使用
    tuned_metrics_for_report = {
        "num_parameters": n_params,
        "model_size_mb": original_size_mb,
        "auc": fp32_cls.get("auc"),
        "accuracy": fp32_cls.get("accuracy"),
        "precision": fp32_cls.get("precision"),
        "recall": fp32_cls.get("recall"),
        "f1": fp32_cls.get("f1"),
    }
    quantized_eval_for_report = {
        "num_parameters": n_params,
        "model_size_mb": quantized_size_mb,
        "auc": q_metrics.get("auc"),
        "accuracy": q_metrics.get("accuracy"),
        "precision": q_metrics.get("precision"),
        "recall": q_metrics.get("recall"),
        "f1": q_metrics.get("f1"),
        "n_pos": quantized_eval["n_pos"],
        "n_neg": quantized_eval["n_neg"],
    }

    generate_report(
        tuned_metrics=tuned_metrics_for_report,
        quantized_eval=quantized_eval_for_report,
        quantized_bench=quantized_bench,
        fp32_bench=fp32_bench,
        source_bench=source_bench,
        output_dir=output_dir,
        source_model_run_id=source_model_run_id,
        source_benchmark_run_id=source_benchmark_run_id,
        quantization_method=quantization_method,
        quantized_layers=quantized_layers_config,
        device=device,
        warmup_runs=warmup_runs,
        repeat_runs=repeat_runs,
        started_at=started_at,
        finished_at=finished_at,
    )

    # ── 汇总输出 ────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("动态量化实验完成！")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"  FP32 AUC: {fp32_cls.get('auc', 'N/A'):.6f}")
    logger.info(f"  Quantized AUC: {q_metrics.get('auc', 'N/A'):.6f}")
    logger.info(f"  FP32 模型大小: {original_size_mb:.4f} MB")
    logger.info(f"  Quantized 模型大小: {quantized_size_mb:.4f} MB")
    if fp32_bench:
        logger.info(
            f"  FP32 (CPU) Avg 延迟: {fp32_bench['avg_latency_ms_per_sample']:.4f} ms"
        )
    logger.info(
        f"  Quantized (CPU) Avg 延迟: "
        f"{quantized_bench['avg_latency_ms_per_sample']:.4f} ms"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()