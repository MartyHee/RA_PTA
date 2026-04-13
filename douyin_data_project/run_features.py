#!/usr/bin/env python3
"""
运行特征工程脚本。

支持：
1. 从清洗后数据生成特征
2. 支持多种特征类型（基础、文本、时间、交互、复合）
3. 使用mock模式（无需真实数据）
4. 特征重要性分析（基础版）
"""
import sys
import argparse
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.processing.feature_engineering import FeatureEngineer, create_features
from src.utils.config_loader import load_config
from src.utils.logger import setup_logging
from src.utils.io_utils import read_parquet, write_parquet, read_csv, write_csv


def load_input_data(input_path: Optional[Path], config: dict) -> pd.DataFrame:
    """加载输入数据"""
    if input_path and input_path.exists():
        print(f"从文件加载数据: {input_path}")
        if input_path.suffix == '.parquet':
            return read_parquet(input_path)
        elif input_path.suffix == '.csv':
            return pd.read_csv(input_path, parse_dates=['publish_date', 'crawl_date'])
        else:
            raise ValueError(f"不支持的格式: {input_path.suffix}")
    else:
        # 使用默认路径
        default_path = Path(config.get('settings', {}).get('paths', {}).get('interim_data', './data/interim')) / 'cleaned_video_data.parquet'
        if default_path.exists():
            print(f"使用默认路径: {default_path}")
            return read_parquet(default_path)
        else:
            # 生成mock数据
            print("未找到输入文件，使用mock数据")
            return generate_mock_data(config)


def generate_mock_data(config: dict) -> pd.DataFrame:
    """生成mock数据用于测试"""
    import numpy as np
    from datetime import datetime, date, timedelta

    num_samples = config.get('settings', {}).get('processing', {}).get('mock_samples', 20)

    # 生成模拟的清洗后数据
    base_date = date.today()
    data = {
        'video_id': [f'video_{i:06d}' for i in range(num_samples)],
        'author_id': [f'author_{i % 5:03d}' for i in range(num_samples)],
        'desc_clean': [f'测试视频描述 #{i} 这是第{i}个视频' for i in range(num_samples)],
        'text_length': [np.random.randint(10, 200) for _ in range(num_samples)],
        'publish_date': [base_date - timedelta(days=np.random.randint(0, 365)) for _ in range(num_samples)],
        'publish_hour': [np.random.randint(0, 24) for _ in range(num_samples)],
        'publish_weekday': [np.random.randint(0, 7) for _ in range(num_samples)],
        'is_weekend': [1 if d >= 5 else 0 for d in [np.random.randint(0, 7) for _ in range(num_samples)]],
        'hashtag_list': [['美食', '旅行'] if i % 2 == 0 else ['科技', '健身'] for i in range(num_samples)],
        'hashtag_count': [np.random.randint(0, 5) for _ in range(num_samples)],
        'like_count': [np.random.randint(0, 10000) for _ in range(num_samples)],
        'comment_count': [np.random.randint(0, 1000) for _ in range(num_samples)],
        'share_count': [np.random.randint(0, 500) for _ in range(num_samples)],
        'collect_count': [np.random.randint(0, 100) for _ in range(num_samples)],
        'engagement_score': [np.random.uniform(0, 50000) for _ in range(num_samples)],
        'source_entry': [np.random.choice(['search', 'topic', 'manual_url']) for _ in range(num_samples)],
        'crawl_date': [base_date for _ in range(num_samples)],
        'data_version': ['v0.1' for _ in range(num_samples)]
    }

    df = pd.DataFrame(data)

    # 确保日期类型
    date_cols = ['publish_date', 'crawl_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.date

    return df


def analyze_feature_importance(df_features: pd.DataFrame, target_col: str = 'engagement_score') -> pd.DataFrame:
    """分析特征重要性（基于相关系数）"""
    if target_col not in df_features.columns:
        print(f"目标列 {target_col} 不存在，跳过特征重要性分析")
        return pd.DataFrame()

    # 选择数值列
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target_col and df_features[col].nunique() > 1]

    if not numeric_cols:
        return pd.DataFrame()

    # 计算相关系数
    correlations = {}
    for col in numeric_cols:
        corr = df_features[col].corr(df_features[target_col])
        if not pd.isna(corr):
            correlations[col] = corr

    # 创建重要性DataFrame
    importance_df = pd.DataFrame({
        'feature': list(correlations.keys()),
        'correlation': list(correlations.values()),
        'abs_correlation': [abs(c) for c in correlations.values()]
    }).sort_values('abs_correlation', ascending=False)

    return importance_df


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行抖音特征工程')
    parser.add_argument('--input', type=Path, help='输入文件路径（清洗后数据）')
    parser.add_argument('--output', type=Path, help='输出文件路径（默认：data/processed/featured_video_data.parquet）')
    parser.add_argument('--config', type=Path, default=None, help='配置文件路径')
    parser.add_argument('--mock', action='store_true', help='使用mock模式（无需真实数据）')
    parser.add_argument('--feature-types', nargs='+',
                       default=['basic', 'text', 'time', 'interaction', 'composite'],
                       help='特征类型: basic/text/time/interaction/composite')
    parser.add_argument('--analyze-importance', action='store_true',
                       help='分析特征重要性（基于相关系数）')
    parser.add_argument('--target', type=str, default='engagement_score',
                       help='特征重要性分析的目标列')
    parser.add_argument('--samples', type=int, default=20, help='mock数据样本数')

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.config)

    # 加载配置
    config = load_config(args.config)

    print("=" * 60)
    print("抖音特征工程启动")
    print("=" * 60)

    # 加载数据
    if args.mock:
        config.setdefault('settings', {}).setdefault('processing', {})['mock_samples'] = args.samples
        df_input = generate_mock_data(config)
        print(f"生成 {len(df_input)} 条mock数据")
    else:
        df_input = load_input_data(args.input, config)
        print(f"加载 {len(df_input)} 条数据")

    if df_input.empty:
        print("错误: 没有数据可处理")
        return 1

    print(f"输入数据列: {list(df_input.columns)}")
    print(f"特征类型: {args.feature_types}")
    print("-" * 60)

    # 特征工程
    print("开始特征工程...")
    engineer = FeatureEngineer(args.config)
    df_features = engineer.create_features(df_input, args.feature_types)
    print(f"特征工程完成，生成 {len(df_features.columns)} 个特征")

    # 特征重要性分析（可选）
    if args.analyze_importance:
        print("\n分析特征重要性...")
        import numpy as np  # 确保numpy已导入

        importance_df = analyze_feature_importance(df_features, args.target)
        if not importance_df.empty:
            print("\n特征重要性排名（基于与目标列的相关系数）:")
            print("-" * 80)
            print(f"{'特征':<30} {'相关系数':>10} {'绝对值':>10}")
            print("-" * 80)
            for _, row in importance_df.head(20).iterrows():
                print(f"{row['feature']:<30} {row['correlation']:>10.4f} {row['abs_correlation']:>10.4f}")

            # 保存重要性结果
            output_dir = Path(config.get('settings', {}).get('paths', {}).get('processed_data', './data/processed'))
            output_dir.mkdir(parents=True, exist_ok=True)
            importance_path = output_dir / 'feature_importance.csv'
            importance_df.to_csv(importance_path, index=False, encoding='utf-8')
            print(f"\n特征重要性保存至: {importance_path}")
        else:
            print("无法计算特征重要性")

    # 保存结果
    output_path = args.output
    if output_path is None:
        output_dir = Path(config.get('settings', {}).get('paths', {}).get('processed_data', './data/processed'))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'featured_video_data.parquet'

    write_parquet(output_path, df_features)
    print(f"\n特征数据保存至: {output_path}")
    print(f"特征维度: {df_features.shape[1]} 列 x {df_features.shape[0]} 行")

    # 保存CSV样本
    sample_dir = Path(config.get('settings', {}).get('paths', {}).get('samples', './data/samples'))
    sample_dir.mkdir(parents=True, exist_ok=True)

    # 完整样本
    sample_path = sample_dir / 'sample_featured_video_data.csv'
    df_features.head(100).to_csv(sample_path, index=False, encoding='utf-8')
    print(f"特征样本保存至: {sample_path}")

    # 特征描述文件
    feature_desc = pd.DataFrame({
        'feature': df_features.columns,
        'dtype': [str(dtype) for dtype in df_features.dtypes],
        'non_null_count': df_features.count().values,
        'null_count': df_features.isnull().sum().values,
        'null_pct': (df_features.isnull().sum() / len(df_features) * 100).values,
        'unique_values': [df_features[col].nunique() if not (df_features[col].dtype == object and len(df_features[col]) > 0 and type(df_features[col].iloc[0]).__name__ in ('list', 'ndarray')) else None for col in df_features.columns]
    })
    feature_desc_path = sample_dir / 'feature_descriptions.csv'
    feature_desc.to_csv(feature_desc_path, index=False, encoding='utf-8')
    print(f"特征描述保存至: {feature_desc_path}")

    print("\n特征工程完成!")
    return 0


if __name__ == '__main__':
    sys.exit(main())