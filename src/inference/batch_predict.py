"""DNN batch inference CLI 入口。

使用示例：
    D:/CodeData/software/Anaconda/Anaconda3/envs/ra/python.exe src/inference/batch_predict.py ^
        --model dnn ^
        --dataset real_raw_5000 ^
        --run-id 202605132017 ^
        --input data/features/real_raw_5000/tabular_test.csv

    D:/CodeData/software/Anaconda/Anaconda3/envs/ra/python.exe src/inference/batch_predict.py ^
        --model dnn ^
        --dataset real_raw_5000 ^
        --run-id 202605132017 ^
        --input data/features/real_raw_5000/tabular_test.csv ^
        --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，使 src 可作为模块导入
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_model_dir(model: str, dataset: str, run_id: str) -> str:
    """根据 --model / --dataset / --run-id 定位模型目录。"""
    return str(Path("outputs") / model / dataset / run_id)


def main():
    parser = argparse.ArgumentParser(
        description="RA_PTA Batch Inference CLI（第一版只支持 DNN）"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="模型名称（第一版只支持 dnn）",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="数据集名称（例如 real_raw_5000）",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="模型训练 run_id（例如 202605132017）",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="输入 CSV 路径（已构建好的 tabular 特征文件）",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="自定义输出目录（可选，默认自动生成）",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="推理设备（cuda / cpu，默认自动检测）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅验证输入和模型加载，不执行推理",
    )

    args = parser.parse_args()

    # 第一版只支持 DNN
    if args.model != "dnn":
        parser.error(f"当前版本只支持 --model dnn，收到: {args.model}")

    # 解析模型目录
    model_dir = resolve_model_dir(args.model, args.dataset, args.run_id)
    if not os.path.isdir(model_dir):
        print(f"错误: 模型目录不存在: {model_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"模型目录: {model_dir}")
    print(f"输入 CSV: {args.input}")
    print(f"设备: {args.device or ('cuda' if __import__('torch').cuda.is_available() else 'cpu')}")
    if args.dry_run:
        print("模式: dry-run（仅验证）")

    # 延迟 import，避免 CLI 解析参数时加载 torch 等重依赖
    from src.inference.predictor import Predictor

    try:
        predictor = Predictor(model_dir=model_dir, device=args.device)
        predictor.load()
        print("模型加载成功")

        meta = predictor.predict_csv(
            input_path=args.input,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )

        if args.dry_run:
            print(f"\nDry-run 完成。输入文件共有 {meta['num_rows']} 行。")
        else:
            print(f"\n推理完成。输出目录: {meta.get('output_path', '')}")

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()