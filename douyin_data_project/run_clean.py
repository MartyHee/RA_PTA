#!/usr/bin/env python3
"""
运行数据清洗脚本。

支持：
1. 清洗原始网页视频元数据
2. 输出清洗后的数据供特征工程使用
3. 使用mock模式（无需真实数据）
4. 质量检查报告
"""
import sys
import argparse
from pathlib import Path
from typing import List, Optional
import pandas as pd

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.processing.clean import DataCleaner, clean_from_dataframe
from src.processing.transform import DataTransformer, transform_dataframe
from src.processing.quality_check import DataQualityChecker, generate_quality_report
from src.utils.config_loader import load_config
from src.utils.logger import setup_logging
from src.utils.io_utils import read_parquet, write_parquet, read_jsonl, write_jsonl


def load_input_data(input_path: Optional[Path], config: dict) -> pd.DataFrame:
    """加载输入数据（支持多种格式）"""
    if input_path and input_path.exists():
        print(f"从文件加载数据: {input_path}")
        if input_path.suffix == '.parquet':
            return read_parquet(input_path)
        elif input_path.suffix == '.jsonl':
            return pd.DataFrame(read_jsonl(input_path))
        elif input_path.suffix == '.csv':
            return pd.read_csv(input_path)
        else:
            raise ValueError(f"不支持的格式: {input_path.suffix}")
    else:
        # 使用mock数据
        print("未提供输入文件，使用mock数据")
        return generate_mock_data(config)


def generate_mock_data(config: dict) -> pd.DataFrame:
    """生成mock数据用于测试"""
    import numpy as np
    from datetime import datetime, date, timedelta

    num_samples = config.get('settings', {}).get('processing', {}).get('mock_samples', 10)

    # 生成模拟数据
    data = {
        'video_id': [f'video_{i:06d}' for i in range(num_samples)],
        'page_url': [f'https://www.douyin.com/video/{i:06d}' for i in range(num_samples)],
        'author_id': [f'author_{i % 5:03d}' for i in range(num_samples)],
        'author_name': [f'用户{i % 5}' for i in range(num_samples)],
        'author_profile_url': [f'https://www.douyin.com/user/author_{i % 5:03d}' for i in range(num_samples)],
        'desc_text': [f'测试视频描述 #{i} 这是第{i}个视频 #美食 #旅行' for i in range(num_samples)],
        'publish_time_raw': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d %H:%M:%S') for i in range(num_samples)],
        'publish_time_std': [datetime.now() - timedelta(days=i) for i in range(num_samples)],
        'like_count_raw': [f'{np.random.randint(100, 10000)}' for _ in range(num_samples)],
        'comment_count_raw': [f'{np.random.randint(10, 1000)}' for _ in range(num_samples)],
        'share_count_raw': [f'{np.random.randint(5, 500)}' for _ in range(num_samples)],
        'like_count': [np.random.randint(100, 10000) for _ in range(num_samples)],
        'comment_count': [np.random.randint(10, 1000) for _ in range(num_samples)],
        'share_count': [np.random.randint(5, 500) for _ in range(num_samples)],
        'collect_count': [np.random.randint(0, 100) for _ in range(num_samples)],
        'hashtag_list': [['美食', '旅行'] if i % 2 == 0 else ['科技', '健身'] for i in range(num_samples)],
        'hashtag_count': [2 for _ in range(num_samples)],
        'cover_url': [f'https://example.com/cover_{i}.jpg' for i in range(num_samples)],
        'music_name': [f'音乐{i % 3}' for i in range(num_samples)],
        'duration_sec': [np.random.choice([15, 30, 60]) for _ in range(num_samples)],
        'source_entry': [np.random.choice(['search', 'topic', 'manual_url']) for _ in range(num_samples)],
        'crawl_time': [datetime.now() for _ in range(num_samples)]
    }

    return pd.DataFrame(data)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行抖音数据清洗')
    parser.add_argument('--input', type=Path, help='输入文件路径（支持parquet/jsonl/csv）')
    parser.add_argument('--output', type=Path, help='输出文件路径（默认：data/interim/cleaned_video_data.parquet）')
    parser.add_argument('--config', type=Path, default=None, help='配置文件路径')
    parser.add_argument('--mock', action='store_true', help='使用mock模式（无需真实数据）')
    parser.add_argument('--quality-check', action='store_true', help='运行质量检查并生成报告')
    parser.add_argument('--samples', type=int, default=10, help='mock数据样本数')

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.config)

    # 加载配置
    config = load_config(args.config)

    print("=" * 60)
    print("抖音数据清洗启动")
    print("=" * 60)

    # 加载数据
    if args.mock:
        config.setdefault('settings', {}).setdefault('processing', {})['mock_samples'] = args.samples
        df_raw = generate_mock_data(config)
        print(f"生成 {len(df_raw)} 条mock数据")
    else:
        df_raw = load_input_data(args.input, config)
        print(f"加载 {len(df_raw)} 条数据")

    if df_raw.empty:
        print("错误: 没有数据可处理")
        return 1

    print(f"数据列: {list(df_raw.columns)}")
    print("-" * 60)

    # 数据清洗
    print("开始数据清洗...")
    cleaner = DataCleaner(args.config)
    df_clean = cleaner.clean_dataframe(df_raw)
    print(f"清洗完成，保留 {len(df_clean)} 条数据")

    # 数据转换
    print("开始数据转换...")
    transformer = DataTransformer(args.config)
    df_processed = transformer.transform_dataframe(df_clean)
    print(f"转换完成，生成 {len(df_processed)} 条处理数据")

    # 质量检查（可选）
    if args.quality_check:
        print("运行质量检查...")
        checker = DataQualityChecker(args.config)
        report = checker.generate_quality_report(df_processed)

        # 输出检查结果
        print(f"\n质量检查结果:")
        print(f"  总体状态: {report['summary']['overall_status']}")
        print(f"  通过率: {report['summary']['pass_rate']:.1%}")

        if report['issues']:
            print(f"  发现 {len(report['issues'])} 个问题:")
            for issue in report['issues']:
                print(f"    - {issue['check']}: {issue['message']}")

        # 保存报告
        import json
        report_dir = Path(config.get('settings', {}).get('paths', {}).get('processed_data', './data/processed'))
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / 'quality_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"质量报告保存至: {report_path}")

    # 保存结果
    output_path = args.output
    if output_path is None:
        output_dir = Path(config.get('settings', {}).get('paths', {}).get('interim_data', './data/interim'))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'cleaned_video_data.parquet'

    write_parquet(output_path, df_processed)
    print(f"\n清洗后数据保存至: {output_path}")
    print(f"文件大小: {output_path.stat().st_size / (1024*1024):.2f} MB")

    # 保存CSV样本
    sample_dir = Path(config.get('settings', {}).get('paths', {}).get('samples', './data/samples'))
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / 'sample_cleaned_video_data.csv'
    df_processed.head(100).to_csv(sample_path, index=False, encoding='utf-8')
    print(f"样本数据保存至: {sample_path}")

    print("\n清洗完成!")
    return 0


if __name__ == '__main__':
    sys.exit(main())