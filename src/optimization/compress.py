"""DNN 模型压缩 CLI。

Batch 10C：第一版只支持 DNN 动态量化（dynamic_quantization）。
只做 CPU 量化，不替换原始 model.pt。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_model_dir(model: str, dataset: str, run_id: str) -> str:
    return str(Path("outputs") / model / dataset / run_id)


def generate_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_model_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 4)


def main():
    parser = argparse.ArgumentParser(
        description="RA_PTA 模型压缩 CLI（Batch 10C，第一版只支持 DNN 动态量化）"
    )
    parser.add_argument("--model", default="dnn", help="模型名称（第一版只支持 dnn）")
    parser.add_argument(
        "--dataset",
        default="real_raw_5000",
        help="数据集名称（第一版只支持 real_raw_5000）",
    )
    parser.add_argument("--run-id", required=True, help="模型训练 run_id")
    parser.add_argument(
        "--method",
        default="dynamic_quantization",
        help="压缩方法（第一版只支持 dynamic_quantization）",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="推理设备（动态量化仅支持 cpu）",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="自定义输出目录（可选，默认自动生成）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅验证路径和压缩流程，不保存输出",
    )
    args = parser.parse_args()

    # ================================================================
    # 参数校验
    # ================================================================
    if args.model != "dnn":
        parser.error(f"当前版本只支持 --model dnn，收到: {args.model}")
    if args.dataset != "real_raw_5000":
        parser.error(f"当前版本只支持 --dataset real_raw_5000，收到: {args.dataset}")
    if args.method != "dynamic_quantization":
        parser.error(
            f"当前版本只支持 --method dynamic_quantization，收到: {args.method}"
        )
    if args.device != "cpu":
        parser.error(
            f"动态量化仅支持 --device cpu，收到: {args.device}。"
            "PyTorch 动态量化只支持 CPU 推理。"
        )

    # ================================================================
    # 路径解析
    # ================================================================
    source_model_dir = resolve_model_dir(args.model, args.dataset, args.run_id)
    if not os.path.isdir(source_model_dir):
        print(f"错误: 模型目录不存在: {source_model_dir}", file=sys.stderr)
        sys.exit(1)

    source_model_path = Path(source_model_dir) / "model.pt"
    required_files = ["model.pt", "run_meta.json", "feature_config_used.json"]
    for fname in required_files:
        fpath = Path(source_model_dir) / fname
        if not fpath.is_file():
            print(f"错误: 缺少必需文件: {fpath}", file=sys.stderr)
            sys.exit(1)

    print(f"源模型目录: {source_model_dir}")
    print(f"源模型文件: {source_model_path}")
    print(f"压缩方法: {args.method}")
    print(f"设备: {args.device}")
    if args.dry_run:
        print("模式: dry-run（仅验证，不保存）")

    # 模型体积（压缩前）
    source_size_bytes = source_model_path.stat().st_size
    source_size_mb = round(source_size_bytes / (1024 * 1024), 4)
    print(f"原模型大小: {source_size_mb} MB ({source_size_bytes} bytes)")

    # ================================================================
    # 阶段 1: 加载原始模型
    # ================================================================
    print("\n[阶段 1] 加载原始模型...")
    from src.inference.predictor import Predictor

    t0 = time.perf_counter()
    predictor = Predictor(model_dir=source_model_dir, device="cpu")
    predictor.load()
    load_time = time.perf_counter() - t0
    print(f"  模型加载完成: {load_time * 1000:.2f} ms")

    # 获取模型并确保在 CPU 上
    model = predictor.model.cpu()

    # 获取模型结构信息
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {num_params:,}")

    # ================================================================
    # 阶段 2: 动态量化
    # ================================================================
    print(f"\n[阶段 2] 执行动态量化 ({args.method})...")
    t0 = time.perf_counter()
    compressed_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    quantize_time = time.perf_counter() - t0
    print(f"  量化完成: {quantize_time * 1000:.2f} ms")

    # ================================================================
    # 阶段 3: Dry-run 验证
    # ================================================================
    if args.dry_run:
        print("\n[阶段 3] Dry-run 验证...")

        # 加载少量数据验证 forward 正常
        import pandas as pd

        test_csv = Path("data/features") / args.dataset / "tabular_test.csv"
        if not test_csv.is_file():
            print(f"  [skip] 测试 CSV 不存在，跳过 forward 验证: {test_csv}")
        else:
            df = pd.read_csv(test_csv, encoding="utf-8-sig", skipinitialspace=True)
            df_sample = df.head(5)

            predictor.validate_input(df_sample)
            numeric_tensor, cat_tensor = predictor.transform(df_sample)

            with torch.no_grad():
                logits = compressed_model(numeric_tensor, cat_tensor)
                scores = torch.sigmoid(logits)
            print(f"  Forward 验证通过: scores = {scores.tolist()}")

        # 估算压缩后体积（通过 state_dict 的 numel 估算）
        compressed_state = compressed_model.state_dict()
        total_bytes_est = sum(
            v.numel() * (1 if v.element_size() == 1 else v.element_size())
            for v in compressed_state.values()
            if hasattr(v, "numel")
        ) + 4096  # overhead estimate
        total_bytes_est = max(total_bytes_est, 1024)
        est_compressed_mb = round(total_bytes_est / (1024 * 1024), 4)
        est_compressed_mb = round(total_bytes_est / (1024 * 1024), 4)
        est_ratio = round(est_compressed_mb / source_size_mb, 4) if source_size_mb > 0 else 0

        print(f"  原始体积: {source_size_mb} MB")
        print(f"  估算压缩后: ~{est_compressed_mb} MB")
        print(f"  估算压缩比: {est_ratio}")
        print(f"\nDry-run 验证通过。")
        print(f"如需执行正式压缩，去掉 --dry-run 即可。")
        return

    # ================================================================
    # 阶段 4: 保存压缩产物
    # ================================================================
    compression_run_id = generate_run_id()
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            Path("outputs")
            / "optimization"
            / "compression"
            / args.model
            / args.dataset
            / args.run_id
            / compression_run_id
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n输出目录: {output_dir}")

    # 4a. 保存压缩模型
    compressed_model_path = output_dir / "model_compressed.pt"
    torch.save(compressed_model, compressed_model_path)
    print(f"  压缩模型已保存: {compressed_model_path}")

    compressed_size_bytes = compressed_model_path.stat().st_size
    compressed_size_mb = round(compressed_size_bytes / (1024 * 1024), 4)

    # 4b. 复制原始元信息
    shutil.copy2(
        Path(source_model_dir) / "run_meta.json",
        output_dir / "source_run_meta.json",
    )
    print(f"  source_run_meta.json [ok]")

    shutil.copy2(
        Path(source_model_dir) / "feature_config_used.json",
        output_dir / "feature_config_used.json",
    )
    print(f"  feature_config_used.json [ok]")

    # 4c. size_report.json
    compression_ratio = round(compressed_size_mb / source_size_mb, 4) if source_size_mb > 0 else 0
    size_reduction_pct = round((1 - compressed_size_mb / source_size_mb) * 100, 2) if source_size_mb > 0 else 0

    size_report = {
        "source_model_path": str(Path(source_model_dir).resolve()),
        "compressed_model_path": str(compressed_model_path.resolve()),
        "source_model_size_bytes": source_size_bytes,
        "compressed_model_size_bytes": compressed_size_bytes,
        "source_model_size_mb": source_size_mb,
        "compressed_model_size_mb": compressed_size_mb,
        "compression_ratio": compression_ratio,
        "size_reduction_percent": size_reduction_pct,
    }
    save_json(size_report, output_dir / "size_report.json")
    print(f"  size_report.json [ok]")
    print(f"    原体积: {source_size_mb} MB")
    print(f"    压缩后: {compressed_size_mb} MB")
    print(f"    压缩比: {compression_ratio}")
    print(f"    缩小: {size_reduction_pct}%")

    # 4d. compression_meta.json
    warnings = []
    warnings.append("动态量化仅支持 CPU 推理。")
    warnings.append("Score 可能有微小差异，需通过 benchmark 验证。")
    if compression_ratio > 0.9:
        warnings.append("压缩比接近 1.0，压缩效果不显著。")

    compression_meta = {
        "compression_run_id": compression_run_id,
        "source_model_run_id": args.run_id,
        "model_name": args.model,
        "dataset_name": args.dataset,
        "method": args.method,
        "source_model_path": str(Path(source_model_dir).resolve()),
        "compressed_model_path": str(compressed_model_path.resolve()),
        "source_model_size_mb": source_size_mb,
        "compressed_model_size_mb": compressed_size_mb,
        "compression_ratio": compression_ratio,
        "size_reduction_percent": size_reduction_pct,
        "device": args.device,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "warnings": warnings,
    }
    save_json(compression_meta, output_dir / "compression_meta.json")
    print(f"  compression_meta.json [ok]")

    # ================================================================
    # 汇总
    # ================================================================
    print(f"\n{'=' * 50}")
    print("压缩完成！")
    print(f"  模型: {args.model}/{args.dataset}/{args.run_id}")
    print(f"  方法: {args.method}")
    print(f"  原体积: {source_size_mb} MB -> 压缩后: {compressed_size_mb} MB")
    print(f"  压缩比: {compression_ratio} ({size_reduction_pct}% 减小)")
    print(f"  输出: {output_dir}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()