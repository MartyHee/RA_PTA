"""DNN / Wide & Deep 动态量化压缩实验脚本

对上一阶段 real_raw_1000 的 DNN 和 Wide & Deep baseline 做同口径 benchmark
和动态量化，评估量化前后在 test split 上的离线指标、模型大小和推理时延。

用法:
    python src/experiment/quantize_tabular_models.py \
        --config configs/compression/tabular_quantization_real_raw_1000.yaml
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
from torch.utils.data import DataLoader, Dataset

# ── 路径设置 ──────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
sys.path.insert(0, os.path.join(_project_root, "src"))

from evaluation.metrics import compute_classification_metrics  # noqa: E402
from evaluation.ranking_metrics import (  # noqa: E402
    compute_precision_at_k,
    compute_recall_at_k,
)
from models.dnn.dataset import DNNDataProcessor, TabularDataset  # noqa: E402
from models.dnn.model import DNNModel  # noqa: E402
from models.wide_deep.dataset import WideDeepDataProcessor, WideDeepDataset  # noqa: E402
from models.wide_deep.model import WideDeepModel  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.io import read_csv_safe  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

logger = get_logger("tabular_quantize")


# ── 工具函数 ──────────────────────────────────────────────


def count_parameters(model: nn.Module) -> int:
    """返回模型可训练参数量。"""
    return sum(p.numel() for p in model.parameters())


def measure_model_size_mb(model_path: Path) -> float:
    """返回模型文件大小 (MB)。"""
    return os.path.getsize(model_path) / (1024 * 1024)


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """对模型 Linear 层应用 PyTorch 动态量化。"""
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear},
        dtype=torch.qint8,
    )
    return quantized_model


# ── 模型与处理器加载 ──────────────────────────────────────


def load_dnn_model_and_processor(
    source_dir: Path,
    device: str,
) -> tuple[nn.Module, DNNDataProcessor, dict]:
    """加载 DNN 模型和处理器，返回 (model, processor, feature_config)。"""
    feature_config_path = source_dir / "feature_config_used.json"
    if not feature_config_path.exists():
        logger.error(f"DNN 特征配置不存在: {feature_config_path}")
        sys.exit(1)
    with open(feature_config_path, "r", encoding="utf-8") as f:
        feature_config = json.load(f)

    processor = DNNDataProcessor.from_config(feature_config)

    numeric_dim = len(feature_config.get("numeric_cols", []))
    model = DNNModel(
        numeric_dim=numeric_dim,
        cat_embed_dims=processor.cat_embed_dims,
        hidden_units=feature_config.get("hidden_units", [64, 32]),
        dropout=feature_config.get("dropout", 0.3),
    ).to(device)

    model_path = source_dir / "model.pt"
    if not model_path.exists():
        logger.error(f"DNN 模型文件不存在: {model_path}")
        sys.exit(1)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()
    logger.info(f"DNN 模型加载完成: {model_path}")
    logger.info(f"DNN 参数量: {count_parameters(model)}")

    return model, processor, feature_config


def load_wde_model_and_processor(
    source_dir: Path,
    device: str,
) -> tuple[nn.Module, WideDeepDataProcessor, dict]:
    """加载 Wide & Deep 模型和处理器，返回 (model, processor, feature_config)。"""
    feature_config_path = source_dir / "feature_config_used.json"
    if not feature_config_path.exists():
        logger.error(f"Wide & Deep 特征配置不存在: {feature_config_path}")
        sys.exit(1)
    with open(feature_config_path, "r", encoding="utf-8") as f:
        feature_config = json.load(f)

    processor = WideDeepDataProcessor.from_config(feature_config)

    numeric_dim = len(feature_config.get("numeric_cols", []))
    model = WideDeepModel(
        numeric_dim=numeric_dim,
        cat_embed_dims=processor.cat_embed_dims,
        wide_vocab_sizes=processor.wide_vocab_sizes,
        deep_hidden_units=feature_config.get("deep_hidden_units", [64, 32]),
        dropout=feature_config.get("dropout", 0.3),
        wide_embedding_dim=feature_config.get("wide_embedding_dim", 1),
    ).to(device)

    model_path = source_dir / "model.pt"
    if not model_path.exists():
        logger.error(f"Wide & Deep 模型文件不存在: {model_path}")
        sys.exit(1)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()
    logger.info(f"Wide & Deep 模型加载完成: {model_path}")
    logger.info(f"Wide & Deep 参数量: {count_parameters(model)}")

    return model, processor, feature_config


# ── 评估函数（DataLoader-based，支持批量推理）─────────────


def evaluate_model(
    model: nn.Module,
    dataset: Dataset,
    device: str,
    threshold: float,
    batch_size: int = 64,
    ids_data: pd.DataFrame | None = None,
) -> dict:
    """在给定 Dataset 上评估模型，返回指标和预测结果。"""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_logits: list[float] = []
    all_scores: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for batch in loader:
            numeric_b = batch["numeric"].to(device)
            cat_b = batch["categorical"].to(device)
            labels_b = batch["label"].to(device)

            if "wide" in batch:
                wide_b = batch["wide"].to(device)
                logits = model(numeric_b, cat_b, wide_b)
            else:
                logits = model(numeric_b, cat_b)

            scores = torch.sigmoid(logits)
            all_logits.extend(logits.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels_b.cpu().numpy())

    all_logits_arr = np.array(all_logits)
    all_scores_arr = np.array(all_scores)
    all_labels_arr = np.array(all_labels)
    all_preds_arr = (all_scores_arr >= threshold).astype(int)

    eval_loss = nn.BCEWithLogitsLoss()(
        torch.tensor(all_logits_arr), torch.tensor(all_labels_arr)
    ).item()

    cls_metrics, cls_warnings = compute_classification_metrics(
        all_labels_arr, all_scores_arr, all_preds_arr, threshold
    )
    k_values = [5, 10, 20]
    pk_metrics, pk_warnings = compute_precision_at_k(
        all_labels_arr, all_scores_arr, k_values
    )
    rk_metrics, rk_warnings = compute_recall_at_k(
        all_labels_arr, all_scores_arr, k_values
    )

    n_pos = int(all_labels_arr.sum())
    n_neg = int(len(all_labels_arr) - n_pos)

    return {
        "logits": all_logits_arr,
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


# ── Benchmark 函数（per-sample 计时）──────────────────────


def run_inference_benchmark(
    model: nn.Module,
    dataset: Dataset,
    device: str,
    model_type: str,
    warmup_runs: int = 10,
    repeat_runs: int = 100,
) -> dict:
    """执行推理 benchmark，返回延迟统计。

    Args:
        model: 已加载权重的模型（eval mode 由本函数设置）。
        dataset: 测试集 Dataset。
        device: 推理设备。
        model_type: 'dnn' 或 'wide_deep'。
        warmup_runs: 预热遍数。
        repeat_runs: 计时遍数。

    Returns:
        dict: 包含延迟统计的字典。
    """
    model.eval()
    n_samples = len(dataset)

    # 预加载全部数据到 device
    all_numeric: list[torch.Tensor] = []
    all_cat: list[torch.Tensor] = []
    all_wide: list[torch.Tensor] | None = [] if model_type == "wide_deep" else None

    for i in range(n_samples):
        item = dataset[i]
        all_numeric.append(item["numeric"].unsqueeze(0).to(device))
        all_cat.append(item["categorical"].unsqueeze(0).to(device))
        if model_type == "wide_deep":
            all_wide.append(item["wide"].unsqueeze(0).to(device))

    # 预热
    logger.info(f"  预热 {warmup_runs} 遍 × {n_samples} 样本 ...")
    for _ in range(warmup_runs):
        with torch.no_grad():
            for i in range(n_samples):
                if model_type == "wide_deep":
                    _ = model(all_numeric[i], all_cat[i], all_wide[i])
                else:
                    _ = model(all_numeric[i], all_cat[i])

    # 计时推理
    logger.info(f"  计时推理 {repeat_runs} 遍 × {n_samples} 样本 ...")
    all_latencies: list[float] = []
    for _ in range(repeat_runs):
        for i in range(n_samples):
            t0 = time.perf_counter()
            with torch.no_grad():
                if model_type == "wide_deep":
                    logit = model(all_numeric[i], all_cat[i], all_wide[i])
                else:
                    logit = model(all_numeric[i], all_cat[i])
                _ = torch.sigmoid(logit)
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
    summary_rows: list[dict],
    output_dir: Path,
    source_dnn_run_id: str,
    source_wde_run_id: str,
    quantization_method: str,
    quantized_layers: list[str],
    device: str,
    warmup_runs: int,
    repeat_runs: int,
    threshold: float,
    started_at: str,
    finished_at: str,
) -> None:
    """生成压缩报告 markdown 文件（含 DNN + Wide & Deep 对照）。"""
    lines: list[str] = []
    lines.append("# Tabular 模型 (DNN / Wide & Deep) 动态量化压缩实验报告")
    lines.append("")
    lines.append(f"- **Run ID**: {output_dir.name}")
    lines.append(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **量化方法**: {quantization_method}")
    lines.append(f"- **量化层**: {', '.join(quantized_layers)}")
    lines.append(f"- **Benchmark 设备**: {device}")
    lines.append(f"- **Warmup**: {warmup_runs} passes")
    lines.append(f"- **Repeat**: {repeat_runs} passes")
    lines.append(f"- **分类阈值**: {threshold}")
    lines.append("")

    lines.append("## 重要声明")
    lines.append("")
    lines.append("1. **当前为离线本地推理计时，不代表线上服务延迟。**")
    lines.append("2. 输入模型为上一阶段 real_raw_1000 的 DNN (run_id=%s) 和" % source_dnn_run_id)
    lines.append("   Wide & Deep (run_id=%s) baseline 模型。" % source_wde_run_id)
    lines.append("3. 使用 PyTorch 动态量化（`torch.quantization.quantize_dynamic`），仅量化 Linear 层。")
    lines.append("4. Quantized 模型运行在 CPU（动态量化 Linear 层仅支持 CPU），")
    lines.append("   FP32 CPU benchmark 用于同设备公平对比。")
    lines.append("5. test split 仅用于最终离线评估，不参与训练或超参选择。")
    lines.append("6. **当前标签为 interaction_score 伪标签，不代表真实 CTR/CVR。**")
    lines.append("7. 本实验与 Multimodal tuned 量化结果（outputs/compression/multimodal/real_raw_1000_tuned/）")
    lines.append("   形成对照，考察动态量化在不同参数量级模型上的效果。")
    lines.append("8. **当前结果不代表线上服务延迟或业务效果。**")
    lines.append("")

    lines.append("## 模型版本说明")
    lines.append("")
    lines.append("| 版本 | 说明 |")
    lines.append("| --- | --- |")
    lines.append("| dnn_fp32_cpu | DNN FP32 CPU（同设备公平对比）|")
    lines.append("| dnn_quantized | DNN 动态量化后 CPU |")
    lines.append("| wide_deep_fp32_cpu | Wide & Deep FP32 CPU（同设备公平对比）|")
    lines.append("| wide_deep_quantized | Wide & Deep 动态量化后 CPU |")
    lines.append("")

    # ── 模型大小与参数量 ──
    lines.append("## 模型大小与参数量")
    lines.append("")
    lines.append("| 模型 | 参数量 | FP32 (MB) | Quantized (MB) | 压缩比 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")

    # Group by model
    dnn_fp32 = next(r for r in summary_rows if r["model_version"] == "dnn_fp32_cpu")
    dnn_q = next(r for r in summary_rows if r["model_version"] == "dnn_quantized")
    wde_fp32 = next(r for r in summary_rows if r["model_version"] == "wide_deep_fp32_cpu")
    wde_q = next(r for r in summary_rows if r["model_version"] == "wide_deep_quantized")

    dnn_ratio = dnn_q["model_size_mb"] / dnn_fp32["model_size_mb"]
    lines.append(
        f"| DNN | {dnn_fp32['num_parameters']:,} | "
        f"{dnn_fp32['model_size_mb']:.4f} | "
        f"{dnn_q['model_size_mb']:.4f} | "
        f"{dnn_ratio:.2f}× |"
    )
    wde_ratio = wde_q["model_size_mb"] / wde_fp32["model_size_mb"]
    lines.append(
        f"| Wide & Deep | {wde_fp32['num_parameters']:,} | "
        f"{wde_fp32['model_size_mb']:.4f} | "
        f"{wde_q['model_size_mb']:.4f} | "
        f"{wde_ratio:.2f}× |"
    )
    lines.append("")

    # ── 离线评估指标 ──
    lines.append("## 离线评估指标 (test split)")
    lines.append("")
    lines.append("| 模型 | 版本 | AUC | Accuracy | Precision | Recall | F1 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")

    for r in summary_rows:
        ver = r["model_version"]
        auc_s = f"{r['auc']:.6f}" if r["auc"] is not None else "N/A"
        acc_s = f"{r['accuracy']:.6f}" if r["accuracy"] is not None else "N/A"
        prec_s = f"{r['precision']:.6f}" if r["precision"] is not None else "N/A"
        rec_s = f"{r['recall']:.6f}" if r["recall"] is not None else "N/A"
        f1_s = f"{r['f1']:.6f}" if r["f1"] is not None else "N/A"
        lines.append(
            f"| {'DNN' if 'dnn' in ver else 'Wide&Deep'} | "
            f"{'Quantized' if 'quantized' in ver else 'FP32'} | "
            f"{auc_s} | {acc_s} | {prec_s} | {rec_s} | {f1_s} |"
        )
    lines.append("")
    lines.append(
        f"注: 所有指标在 test split ({dnn_fp32['num_samples']} 样本, "
        f"{dnn_fp32['num_positive']} 正 / {dnn_fp32['num_negative']} 负) 上评估。"
    )
    lines.append("")

    # ── 推理时延对比 ──
    lines.append("## 推理时延对比")
    lines.append("")
    lines.append("| 模型 | Device | Avg (ms) | P50 (ms) | P95 (ms) | Min (ms) | Max (ms) |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for r in summary_rows:
        ver = r["model_version"]
        label = "DNN FP32" if ver == "dnn_fp32_cpu" else (
            "DNN Quantized" if ver == "dnn_quantized" else (
                "W&D FP32" if ver == "wide_deep_fp32_cpu" else "W&D Quantized"
            )
        )
        lines.append(
            f"| {label} | CPU | "
            f"{r['avg_latency_ms_per_sample']:.4f} | "
            f"{r['p50_latency_ms_per_sample']:.4f} | "
            f"{r['p95_latency_ms_per_sample']:.4f} | "
            f"{r['latency_min_ms']:.4f} | "
            f"{r['latency_max_ms']:.4f} |"
        )
    lines.append("")

    # ── 吞吐量对比 ──
    lines.append("## 吞吐量对比")
    lines.append("")
    lines.append("| 模型 | Device | 吞吐量 (samples/sec) |")
    lines.append("| --- | --- | ---: |")
    for r in summary_rows:
        ver = r["model_version"]
        label = "DNN FP32" if ver == "dnn_fp32_cpu" else (
            "DNN Quantized" if ver == "dnn_quantized" else (
                "W&D FP32" if ver == "wide_deep_fp32_cpu" else "W&D Quantized"
            )
        )
        lines.append(
            f"| {label} | CPU | {r['throughput_samples_per_sec']:.2f} |"
        )
    lines.append("")

    # ── 量化前后对比总结 ──
    lines.append("## 量化前后对比总结")
    lines.append("")
    lines.append("| 对比项 | DNN FP32 | DNN Quantized | 变化 | W&D FP32 | W&D Quantized | 变化 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    # 模型大小
    lines.append(
        f"| 模型大小 | {dnn_fp32['model_size_mb']:.4f} MB | "
        f"{dnn_q['model_size_mb']:.4f} MB | {dnn_ratio:.2f}× | "
        f"{wde_fp32['model_size_mb']:.4f} MB | "
        f"{wde_q['model_size_mb']:.4f} MB | {wde_ratio:.2f}× |"
    )
    # 延迟
    dnn_lat_ratio = dnn_q["avg_latency_ms_per_sample"] / dnn_fp32["avg_latency_ms_per_sample"]
    wde_lat_ratio = wde_q["avg_latency_ms_per_sample"] / wde_fp32["avg_latency_ms_per_sample"]
    lines.append(
        f"| Avg 延迟 | {dnn_fp32['avg_latency_ms_per_sample']:.4f} ms | "
        f"{dnn_q['avg_latency_ms_per_sample']:.4f} ms | {dnn_lat_ratio:.2f}× | "
        f"{wde_fp32['avg_latency_ms_per_sample']:.4f} ms | "
        f"{wde_q['avg_latency_ms_per_sample']:.4f} ms | {wde_lat_ratio:.2f}× |"
    )
    # AUC
    dnn_auc_diff = dnn_q["auc"] - dnn_fp32["auc"] if dnn_q["auc"] is not None and dnn_fp32["auc"] is not None else 0
    wde_auc_diff = wde_q["auc"] - wde_fp32["auc"] if wde_q["auc"] is not None and wde_fp32["auc"] is not None else 0
    dnn_auc_s = f"{dnn_auc_diff:+.6f}" if dnn_q["auc"] is not None else "N/A"
    wde_auc_s = f"{wde_auc_diff:+.6f}" if wde_q["auc"] is not None else "N/A"
    lines.append(
        f"| AUC | {dnn_fp32['auc']:.6f} | "
        f"{dnn_q['auc']:.6f} | {dnn_auc_s} | "
        f"{wde_fp32['auc']:.6f} | "
        f"{wde_q['auc']:.6f} | {wde_auc_s} |"
    )
    # F1
    dnn_f1_diff = dnn_q["f1"] - dnn_fp32["f1"] if dnn_q["f1"] is not None and dnn_fp32["f1"] is not None else 0
    wde_f1_diff = wde_q["f1"] - wde_fp32["f1"] if wde_q["f1"] is not None and wde_fp32["f1"] is not None else 0
    dnn_f1_s = f"{dnn_f1_diff:+.6f}" if dnn_q["f1"] is not None else "N/A"
    wde_f1_s = f"{wde_f1_diff:+.6f}" if wde_q["f1"] is not None else "N/A"
    lines.append(
        f"| F1 | {dnn_fp32['f1']:.6f} | "
        f"{dnn_q['f1']:.6f} | {dnn_f1_s} | "
        f"{wde_fp32['f1']:.6f} | "
        f"{wde_q['f1']:.6f} | {wde_f1_s} |"
    )
    lines.append("")

    # ── 分析 ──
    lines.append("## 分析")
    lines.append("")
    lines.append(
        "1. **模型大小**: 动态量化将 Linear 权重从 FP32 转为 INT8。"
        "对于 DNN（41,177 参数）和 Wide & Deep（42,760 参数），"
        "模型文件压缩效果有限。"
    )
    lines.append(
        "2. **离线指标**: 动态量化对 Linear 权重做 INT8 近似，"
        "通常会导致轻微精度损失。需要关注 AUC/F1 的变化幅度。"
    )
    lines.append(
        "3. **推理时延**: 量化模型和 FP32 基准均在 CPU 上运行做公平对比。"
        "动态量化引入额外的量化/反量化开销，对于小模型可能导致延迟增加。"
    )
    lines.append(
        "4. **与 Multimodal 量化对照**: DNN（41,177 参数）和 Wide & Deep（42,760 参数）"
        "的参数量显著大于 Multimodal tuned（2,649 参数），"
        "量化开销相对更小，但动态量化的运行时开销可能仍然显著。"
    )
    lines.append("")

    lines.append("## Benchmark 设置")
    lines.append("")
    lines.append(f"- **Warmup passes**: {warmup_runs}")
    lines.append(f"- **Repeat passes**: {repeat_runs}")
    lines.append(f"- **Batch size**: 1（单样本推理）")
    lines.append(f"- **Device**: {device}")
    lines.append(f"- **测试样本数**: 150")
    lines.append(f"- **起止时间**: {started_at} → {finished_at}")
    lines.append("")

    lines.append("## 结论")
    lines.append("")
    lines.append(
        "本次实验对上一阶段 real_raw_1000 的 DNN 和 Wide & Deep baseline 模型"
        "进行了 PyTorch 动态量化，评估了量化前后在 test split 上的离线指标、"
        "模型大小和推理时延。"
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
        description="DNN / Wide & Deep 动态量化压缩实验"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/compression/tabular_quantization_real_raw_1000.yaml",
        help="量化配置文件路径（相对于项目根目录）",
    )
    args = parser.parse_args()

    project_root = Path(_project_root)
    config = load_config(args.config)

    # ── 生成 run_id ──────────────────────────────────────
    run_id = datetime.now().strftime("%Y%m%d%H%M")
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Tabular 模型量化实验 Run ID: {run_id}")

    # ── 随机种子 ─────────────────────────────────────────
    set_seed(config.get("random_seed", 42))

    # ── 输出目录 ─────────────────────────────────────────
    output_root = project_root / config["output_root"]
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")

    # ── 配置参数 ─────────────────────────────────────────
    dnn_cfg = config["dnn"]
    wde_cfg = config["wide_deep"]
    source_dnn_run_id = dnn_cfg["source_run_id"]
    source_wde_run_id = wde_cfg["source_run_id"]
    source_dnn_dir = project_root / dnn_cfg["source_output_dir"]
    source_wde_dir = project_root / wde_cfg["source_output_dir"]
    dnn_model_config_path = project_root / dnn_cfg["config_path"]
    wde_model_config_path = project_root / wde_cfg["config_path"]

    quantization_config = config.get("quantization", {})
    quantization_method = quantization_config.get("method", "dynamic_quantization")
    quantized_layers_config = quantization_config.get("quantized_layers", ["nn.Linear"])

    test_data_path = project_root / config["test_data_path"]
    warmup_runs = config.get("warmup_runs", 10)
    repeat_runs = config.get("repeat_runs", 100)
    batch_size = config.get("batch_size", 64)
    device = config.get("device", "cpu")
    threshold = config.get("threshold", 0.5)

    # ── 加载测试数据 ────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"加载测试数据: {test_data_path}")
    test_df, _ = read_csv_safe(str(test_data_path))
    logger.info(f"测试样本数: {len(test_df)}")

    # 获取 id 列
    with open(project_root / config["feature_info_path"], "r", encoding="utf-8") as f:
        feature_info = json.load(f)
    id_cols = feature_info.get("id_cols", [])
    logger.info(f"ID 列: {id_cols}")

    # ── 存储所有 summary rows ────────────────────────────
    summary_rows: list[dict] = []

    # ── 处理每个模型 ─────────────────────────────────────
    models_to_process = [
        {
            "name": "dnn",
            "label": "DNN",
            "source_dir": source_dnn_dir,
            "source_run_id": source_dnn_run_id,
            "model_config_path": dnn_model_config_path,
            "load_func": load_dnn_model_and_processor,
            "model_type": "dnn",
        },
        {
            "name": "wide_deep",
            "label": "Wide & Deep",
            "source_dir": source_wde_dir,
            "source_run_id": source_wde_run_id,
            "model_config_path": wde_model_config_path,
            "load_func": load_wde_model_and_processor,
            "model_type": "wide_deep",
        },
    ]

    for m_idx, m_info in enumerate(models_to_process):
        model_name = m_info["name"]
        model_label = m_info["label"]
        source_dir = m_info["source_dir"]
        source_run_id = m_info["source_run_id"]
        model_type = m_info["model_type"]
        load_func = m_info["load_func"]

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"处理模型 [{m_idx+1}/2]: {model_label} (source_run_id={source_run_id})")
        logger.info("=" * 60)

        # ── 1. 加载模型 + 处理器 ────────────────────────
        logger.info("")
        logger.info("─" * 40)
        logger.info("1. 加载源模型...")
        model, processor, feature_config = load_func(source_dir, device)
        n_params = count_parameters(model)

        source_model_path = source_dir / "model.pt"
        original_size_mb = measure_model_size_mb(source_model_path)

        # ── 2. 构建测试 Dataset ─────────────────────────
        test_data = processor.transform(test_df)
        if model_type == "wide_deep":
            test_dataset = WideDeepDataset(
                test_data["numeric"], test_data["categorical"],
                test_data["wide"], test_data["labels"],
            )
        else:
            test_dataset = TabularDataset(
                test_data["numeric"], test_data["categorical"],
                test_data["labels"],
            )

        ids_df = test_data.get("ids")

        # ── 3. 评估 FP32 ────────────────────────────────
        logger.info("")
        logger.info("─" * 40)
        logger.info("2. 评估 FP32 模型（CPU）...")
        fp32_eval = evaluate_model(
            model, test_dataset, device, threshold, batch_size
        )
        fp32_cls = fp32_eval["cls_metrics"]
        logger.info(f"  FP32 AUC: {fp32_cls.get('auc', 'N/A')}")
        logger.info(f"  FP32 F1:  {fp32_cls.get('f1', 'N/A')}")

        # 保存 FP32 predictions
        if ids_df is not None:
            fp32_pred_df = ids_df.reset_index(drop=True).copy()
        else:
            fp32_pred_df = pd.DataFrame()
        fp32_pred_df["label"] = fp32_eval["labels"]
        fp32_pred_df["score"] = fp32_eval["scores"]
        fp32_pred_df["pred"] = fp32_eval["preds"]
        fp32_pred_df["split"] = "test"
        fp32_pred_df["model_name"] = f"{model_name}_fp32"
        fp32_pred_df["run_id"] = run_id
        fp32_pred_path = output_dir / f"{model_name}_predictions_fp32.csv"
        fp32_pred_df.to_csv(fp32_pred_path, index=False, encoding="utf-8-sig")
        logger.info(f"  FP32 predictions 已保存: {fp32_pred_path}")

        # ── 4. 应用动态量化 ─────────────────────────────
        logger.info("")
        logger.info("─" * 40)
        logger.info("3. 应用 PyTorch 动态量化...")
        quantized_model = apply_dynamic_quantization(model)
        quantized_model = quantized_model.cpu()
        logger.info("  动态量化完成，量化模型移至 CPU")

        # 保存量化模型
        quantized_model_path = output_dir / f"{model_name}_model_quantized.pt"
        torch.save(quantized_model.state_dict(), str(quantized_model_path))
        quantized_size_mb = measure_model_size_mb(quantized_model_path)
        logger.info(f"  量化模型已保存: {quantized_model_path}")
        logger.info(f"  量化模型文件大小: {quantized_size_mb:.4f} MB")

        # ── 5. 评估量化模型 ─────────────────────────────
        logger.info("")
        logger.info("─" * 40)
        logger.info("4. 评估量化模型（CPU）...")
        quantized_eval = evaluate_model(
            quantized_model, test_dataset, "cpu", threshold, batch_size
        )
        q_metrics = quantized_eval["cls_metrics"]
        logger.info(f"  Quantized AUC: {q_metrics.get('auc', 'N/A')}")
        logger.info(f"  Quantized F1:  {q_metrics.get('f1', 'N/A')}")

        # 保存 quantized predictions
        if ids_df is not None:
            q_pred_df = ids_df.reset_index(drop=True).copy()
        else:
            q_pred_df = pd.DataFrame()
        q_pred_df["label"] = quantized_eval["labels"]
        q_pred_df["score"] = quantized_eval["scores"]
        q_pred_df["pred"] = quantized_eval["preds"]
        q_pred_df["split"] = "test"
        q_pred_df["model_name"] = f"{model_name}_quantized"
        q_pred_df["run_id"] = run_id
        q_pred_path = output_dir / f"{model_name}_predictions_quantized.csv"
        q_pred_df.to_csv(q_pred_path, index=False, encoding="utf-8-sig")
        logger.info(f"  Quantized predictions 已保存: {q_pred_path}")

        # ── 6. 量化模型 benchmark（CPU）────────────────
        logger.info("")
        logger.info("─" * 40)
        logger.info("5. 量化模型推理 benchmark（CPU）...")
        quantized_bench = run_inference_benchmark(
            model=quantized_model,
            dataset=test_dataset,
            device="cpu",
            model_type=model_type,
            warmup_runs=warmup_runs,
            repeat_runs=repeat_runs,
        )
        logger.info(
            f"  Quantized Avg 延迟: {quantized_bench['avg_latency_ms_per_sample']:.4f} ms"
        )
        logger.info(
            f"  Quantized 吞吐量: {quantized_bench['throughput_samples_per_sec']:.2f} samples/s"
        )

        # ── 7. FP32 模型 benchmark（CPU，公平对比）──────
        logger.info("")
        logger.info("─" * 40)
        logger.info("6. FP32 模型推理 benchmark（CPU，公平对比）...")

        # 重新构建并加载 FP32 模型（因为 model 已被 in-place 量化）
        if model_type == "wide_deep":
            fp32_model = WideDeepModel(
                numeric_dim=len(feature_config.get("numeric_cols", [])),
                cat_embed_dims=processor.cat_embed_dims,
                wide_vocab_sizes=processor.wide_vocab_sizes,
                deep_hidden_units=feature_config.get("deep_hidden_units", [64, 32]),
                dropout=feature_config.get("dropout", 0.3),
                wide_embedding_dim=feature_config.get("wide_embedding_dim", 1),
            ).to("cpu")
        else:
            fp32_model = DNNModel(
                numeric_dim=len(feature_config.get("numeric_cols", [])),
                cat_embed_dims=processor.cat_embed_dims,
                hidden_units=feature_config.get("hidden_units", [64, 32]),
                dropout=feature_config.get("dropout", 0.3),
            ).to("cpu")

        fp32_state_dict = torch.load(str(source_model_path), map_location="cpu")
        fp32_model.load_state_dict(fp32_state_dict)
        fp32_model.eval()

        fp32_bench = run_inference_benchmark(
            model=fp32_model,
            dataset=test_dataset,
            device="cpu",
            model_type=model_type,
            warmup_runs=warmup_runs,
            repeat_runs=repeat_runs,
        )
        logger.info(
            f"  FP32 Avg 延迟: {fp32_bench['avg_latency_ms_per_sample']:.4f} ms"
        )
        logger.info(
            f"  FP32 吞吐量: {fp32_bench['throughput_samples_per_sec']:.2f} samples/s"
        )

        # ── 8. 保存 per-model metrics ───────────────────
        q_pk = quantized_eval["pk_metrics"]
        q_rk = quantized_eval["rk_metrics"]
        metrics_data = {
            "model_name": model_name,
            "dataset_name": "real_raw_1000",
            "run_id": run_id,
            "split": "test",
            "source_run_id": source_run_id,
            "quantization_method": quantization_method,
            "sample_count": len(quantized_eval["labels"]),
            "positive_count": quantized_eval["n_pos"],
            "negative_count": quantized_eval["n_neg"],
            "eval_loss": quantized_eval["eval_loss"],
            "fp32_metrics": {
                "eval_loss": fp32_eval["eval_loss"],
                "auc": fp32_cls.get("auc"),
                "accuracy": fp32_cls.get("accuracy"),
                "precision": fp32_cls.get("precision"),
                "recall": fp32_cls.get("recall"),
                "f1": fp32_cls.get("f1"),
            },
            "quantized_metrics": {
                "eval_loss": quantized_eval["eval_loss"],
                "auc": q_metrics.get("auc"),
                "accuracy": q_metrics.get("accuracy"),
                "precision": q_metrics.get("precision"),
                "recall": q_metrics.get("recall"),
                "f1": q_metrics.get("f1"),
            },
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
            "model_size_fp32_mb": round(original_size_mb, 4),
            "model_size_quantized_mb": round(quantized_size_mb, 4),
            "size_reduction_ratio": round(quantized_size_mb / original_size_mb, 4),
            "fp32_benchmark": fp32_bench,
            "quantized_benchmark": quantized_bench,
            "warnings": quantized_eval["warnings"],
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        metrics_path = output_dir / f"{model_name}_metrics_quantized.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=2)
        logger.info(f"  Quantized metrics 已保存: {metrics_path}")

        # ── 9. 构建 summary rows ────────────────────────
        # FP32 row
        n_samples = len(fp32_eval["labels"])
        n_pos = fp32_eval["n_pos"]
        n_neg = fp32_eval["n_neg"]
        summary_rows.append({
            "model_name": model_name,
            "model_version": f"{model_name}_fp32_cpu",
            "auc": fp32_cls.get("auc"),
            "accuracy": fp32_cls.get("accuracy"),
            "precision": fp32_cls.get("precision"),
            "recall": fp32_cls.get("recall"),
            "f1": fp32_cls.get("f1"),
            "num_parameters": n_params,
            "model_size_mb": round(original_size_mb, 4),
            "num_samples": n_samples,
            "num_positive": n_pos,
            "num_negative": n_neg,
            "avg_latency_ms_per_sample": fp32_bench["avg_latency_ms_per_sample"],
            "p50_latency_ms_per_sample": fp32_bench["p50_latency_ms_per_sample"],
            "p95_latency_ms_per_sample": fp32_bench["p95_latency_ms_per_sample"],
            "latency_min_ms": fp32_bench["latency_min_ms"],
            "latency_max_ms": fp32_bench["latency_max_ms"],
            "throughput_samples_per_sec": fp32_bench["throughput_samples_per_sec"],
            "device": "cpu",
            "quantized": False,
        })

        # Quantized row
        summary_rows.append({
            "model_name": model_name,
            "model_version": f"{model_name}_quantized",
            "auc": q_metrics.get("auc"),
            "accuracy": q_metrics.get("accuracy"),
            "precision": q_metrics.get("precision"),
            "recall": q_metrics.get("recall"),
            "f1": q_metrics.get("f1"),
            "num_parameters": n_params,
            "model_size_mb": round(quantized_size_mb, 4),
            "num_samples": len(quantized_eval["labels"]),
            "num_positive": quantized_eval["n_pos"],
            "num_negative": quantized_eval["n_neg"],
            "avg_latency_ms_per_sample": quantized_bench["avg_latency_ms_per_sample"],
            "p50_latency_ms_per_sample": quantized_bench["p50_latency_ms_per_sample"],
            "p95_latency_ms_per_sample": quantized_bench["p95_latency_ms_per_sample"],
            "latency_min_ms": quantized_bench["latency_min_ms"],
            "latency_max_ms": quantized_bench["latency_max_ms"],
            "throughput_samples_per_sec": quantized_bench["throughput_samples_per_sec"],
            "device": "cpu",
            "quantized": True,
        })

        logger.info(f"  [{model_label}] 处理完成")

    # ── 保存 compression_summary.csv ──────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("保存汇总结果...")
    summary_columns = [
        "model_name", "model_version", "auc", "accuracy", "precision",
        "recall", "f1", "num_parameters", "model_size_mb",
        "num_samples", "num_positive", "num_negative",
        "avg_latency_ms_per_sample", "p50_latency_ms_per_sample",
        "p95_latency_ms_per_sample", "latency_min_ms", "latency_max_ms",
        "throughput_samples_per_sec", "device", "quantized",
    ]
    summary_df = pd.DataFrame(summary_rows, columns=summary_columns)
    summary_csv_path = output_dir / "compression_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"压缩汇总表已保存: {summary_csv_path}")

    # ── 保存 run_meta.json ──────────────────────────────
    finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_meta = {
        "task": "tabular_dynamic_quantization",
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "config_path": args.config,
        "source_dnn_run_id": source_dnn_run_id,
        "source_dnn_dir": str(source_dnn_dir),
        "source_wide_deep_run_id": source_wde_run_id,
        "source_wide_deep_dir": str(source_wde_dir),
        "quantization_method": quantization_method,
        "quantized_dtype": quantization_config.get("dtype", "qint8"),
        "quantized_layers": quantized_layers_config,
        "device": device,
        "warmup_runs": warmup_runs,
        "repeat_runs": repeat_runs,
        "batch_size": batch_size,
        "threshold": threshold,
        "test_data_path": str(test_data_path),
        "summary_rows": summary_rows,
    }
    with open(output_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    logger.info(f"Run meta 已保存: {output_dir / 'run_meta.json'}")

    # ── 生成 compression_report.md ──────────────────────
    generate_report(
        summary_rows=summary_rows,
        output_dir=output_dir,
        source_dnn_run_id=source_dnn_run_id,
        source_wde_run_id=source_wde_run_id,
        quantization_method=quantization_method,
        quantized_layers=quantized_layers_config,
        device=device,
        warmup_runs=warmup_runs,
        repeat_runs=repeat_runs,
        threshold=threshold,
        started_at=started_at,
        finished_at=finished_at,
    )

    # ── 汇总输出 ────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Tabular 模型动态量化实验完成！")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"Run ID: {run_id}")
    logger.info("")
    logger.info("DNN 结果:")
    dnn_fp32 = next(r for r in summary_rows if r["model_version"] == "dnn_fp32_cpu")
    dnn_q = next(r for r in summary_rows if r["model_version"] == "dnn_quantized")
    logger.info(f"  FP32 AUC: {dnn_fp32['auc']:.6f}, "
                f"Quantized AUC: {dnn_q['auc']:.6f}")
    logger.info(f"  FP32 模型大小: {dnn_fp32['model_size_mb']:.4f} MB, "
                f"Quantized: {dnn_q['model_size_mb']:.4f} MB")
    logger.info(f"  FP32 (CPU) Avg 延迟: {dnn_fp32['avg_latency_ms_per_sample']:.4f} ms, "
                f"Quantized: {dnn_q['avg_latency_ms_per_sample']:.4f} ms")
    logger.info("")
    logger.info("Wide & Deep 结果:")
    wde_fp32 = next(r for r in summary_rows if r["model_version"] == "wide_deep_fp32_cpu")
    wde_q = next(r for r in summary_rows if r["model_version"] == "wide_deep_quantized")
    logger.info(f"  FP32 AUC: {wde_fp32['auc']:.6f}, "
                f"Quantized AUC: {wde_q['auc']:.6f}")
    logger.info(f"  FP32 模型大小: {wde_fp32['model_size_mb']:.4f} MB, "
                f"Quantized: {wde_q['model_size_mb']:.4f} MB")
    logger.info(f"  FP32 (CPU) Avg 延迟: {wde_fp32['avg_latency_ms_per_sample']:.4f} ms, "
                f"Quantized: {wde_q['avg_latency_ms_per_sample']:.4f} ms")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()