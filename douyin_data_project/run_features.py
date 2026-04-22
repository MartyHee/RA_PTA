#!/usr/bin/env python3
"""
运行特征工程脚本。

支持：
1. 从高置信度processed数据生成特征（新版pipeline）
2. 支持特征版本管理和离线特征存储
3. 使用mock模式（无需真实数据，向后兼容）
4. 特征重要性分析（基础版）

新版命令示例：
python run_features.py --input "data/processed/20260421_194709/high_confidence_web_video_meta_20260421_194709.csv" --output-dir "data/features/20260421_194709/" --feature-version "v1" --run-id "20260421_194709"

旧版命令仍支持（Mock模式）：
python run_features.py --mock --samples 20
"""
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.processing.feature_engineering import FeatureEngineer, create_features
from src.utils.config_loader import load_config
from src.utils.logger import setup_logging
from src.utils.io_utils import read_parquet, write_parquet, read_csv, write_csv

# 新特征pipeline
try:
    from src.features.feature_pipeline import FeaturePipeline
    from src.features.feature_storage import FeatureStorage
    FEATURES_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入新特征模块，仅支持旧版Mock模式: {e}")
    FEATURES_MODULE_AVAILABLE = False


def load_input_data(input_path: Optional[Path], config: dict) -> pd.DataFrame:
    """加载输入数据（旧版函数，用于向后兼容）"""
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
    """生成mock数据用于测试（向后兼容）"""
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


def run_new_pipeline(args, config: dict) -> Dict[str, Any]:
    """运行新版特征pipeline。

    Args:
        args: 命令行参数
        config: 配置字典

    Returns:
        运行结果字典
    """
    if not FEATURES_MODULE_AVAILABLE:
        raise ImportError("新特征模块不可用，请检查src/features目录")

    print("=" * 60)
    print("新版特征Pipeline启动")
    print("=" * 60)

    # 验证输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 创建FeaturePipeline实例
    pipeline = FeaturePipeline(
        feature_version=args.feature_version,
        output_dir=args.output_dir,
        verbose=args.verbose
    )

    # 验证输入文件
    validation = pipeline.validate_input(input_path)
    if not validation['valid']:
        print(f"输入文件验证失败: {validation['errors']}")
        if validation['warnings']:
            print(f"警告: {validation['warnings']}")

    # 运行pipeline
    result = pipeline.run(
        input_path=input_path,
        run_id=args.run_id,
        feature_version=args.feature_version,
        output_dir=args.output_dir,
        save_sample=args.save_sample
    )

    # 打印结果摘要
    print("\n" + "=" * 60)
    print("特征Pipeline运行完成")
    print("=" * 60)
    print(f"运行ID: {result['run_id']}")
    print(f"特征版本: {result['feature_version']}")
    print(f"样本数量: {result['num_samples']}")
    print(f"特征数量: {result['num_features']}")
    print(f"输出目录: {result['output_dir']}")

    # 特征重要性分析（可选）
    if args.analyze_importance:
        print("\n分析特征重要性...")
        df_features = pd.read_parquet(Path(result['output_dir']) / 'features.parquet')
        importance_df = analyze_feature_importance(df_features, args.target)

        if not importance_df.empty:
            print("\n特征重要性排名（基于与目标列的相关系数）:")
            print("-" * 80)
            print(f"{'特征':<30} {'相关系数':>10} {'绝对值':>10}")
            print("-" * 80)
            for _, row in importance_df.head(20).iterrows():
                print(f"{row['feature']:<30} {row['correlation']:>10.4f} {row['abs_correlation']:>10.4f}")

            # 保存重要性结果
            output_dir = Path(result['output_dir'])
            importance_path = output_dir / 'feature_importance.csv'
            importance_df.to_csv(importance_path, index=False, encoding='utf-8')
            print(f"\n特征重要性保存至: {importance_path}")
        else:
            print("无法计算特征重要性")

    return result


def run_legacy_pipeline(args, config: dict) -> int:
    """运行旧版特征pipeline（向后兼容）。

    Args:
        args: 命令行参数
        config: 配置字典

    Returns:
        退出码
    """
    print("=" * 60)
    print("旧版特征工程启动（Mock模式）")
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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行抖音特征工程')

    # 新版pipeline参数
    parser.add_argument('--input', type=Path, help='输入文件路径（高置信度processed数据）')
    parser.add_argument('--output-dir', type=Path, help='输出目录路径（特征存储目录）')
    parser.add_argument('--feature-version', type=str, default='v1', help='特征版本（如v1, v1.1）')
    parser.add_argument('--run-id', type=str, help='运行ID（如20260421_194709），如果未提供则从输入路径提取')
    parser.add_argument('--save-sample', action='store_true', default=True, help='保存CSV样本')

    # 旧版pipeline参数（向后兼容）
    parser.add_argument('--output', type=Path, help='输出文件路径（旧版参数，默认：data/processed/featured_video_data.parquet）')
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

    # 通用参数
    parser.add_argument('--verbose', action='store_true', help='详细日志输出')
    parser.add_argument('--legacy', action='store_true', help='强制使用旧版pipeline')

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.config)

    # 加载配置
    config = load_config(args.config)

    # 决定运行哪个pipeline
    use_new_pipeline = False

    if args.legacy:
        print("强制使用旧版pipeline")
        use_new_pipeline = False
    elif args.input and args.output_dir:
        # 如果提供了新版参数，使用新版pipeline
        use_new_pipeline = True
    elif args.mock or args.output:
        # 如果使用mock模式或指定了旧版output参数，使用旧版pipeline
        use_new_pipeline = False
    else:
        # 默认尝试使用新版pipeline，但需要检查模块可用性
        if FEATURES_MODULE_AVAILABLE:
            print("未指定pipeline模式，默认使用新版pipeline（需要--input和--output-dir参数）")
            print("使用--legacy参数强制使用旧版pipeline，或使用--mock参数运行Mock模式")
            return 1
        else:
            print("新特征模块不可用，使用旧版pipeline")
            use_new_pipeline = False

    try:
        if use_new_pipeline:
            if not args.input or not args.output_dir:
                print("错误: 新版pipeline需要--input和--output-dir参数")
                return 1

            result = run_new_pipeline(args, config)
            print("\n新版特征Pipeline执行成功!")
            return 0
        else:
            # 旧版pipeline
            if not args.mock and not args.input and not args.output:
                print("错误: 旧版pipeline需要--mock参数或--input参数")
                return 1

            return run_legacy_pipeline(args, config)

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())