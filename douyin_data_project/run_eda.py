#!/usr/bin/env python3
"""
运行探索性数据分析（EDA）脚本。

支持：
1. 基础统计分析
2. 数据分布可视化
3. 相关性分析
4. 时间模式分析
5. 文本特征分析
6. 互动指标分析
7. 生成综合报告
"""
import sys
import argparse
from pathlib import Path
from typing import List, Optional
import pandas as pd

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.analysis.eda import EDAAnalyzer, run_comprehensive_eda, run_basic_eda
from src.utils.config_loader import load_config
from src.utils.logger import setup_logging
from src.utils.io_utils import read_parquet, read_csv


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
        # 尝试加载特征数据
        feature_path = Path(config.get('settings', {}).get('paths', {}).get('processed_data', './data/processed')) / 'featured_video_data.parquet'
        if feature_path.exists():
            print(f"使用特征数据: {feature_path}")
            return read_parquet(feature_path)

        # 尝试加载清洗数据
        cleaned_path = Path(config.get('settings', {}).get('paths', {}).get('interim_data', './data/interim')) / 'cleaned_video_data.parquet'
        if cleaned_path.exists():
            print(f"使用清洗后数据: {cleaned_path}")
            return read_parquet(cleaned_path)

        # 生成mock数据
        print("未找到输入文件，使用mock数据")
        return generate_mock_data(config)


def generate_mock_data(config: dict) -> pd.DataFrame:
    """生成mock数据用于测试"""
    import numpy as np
    from datetime import datetime, date, timedelta

    num_samples = config.get('settings', {}).get('processing', {}).get('mock_samples', 30)

    # 生成模拟的特征数据
    base_date = date.today()
    np.random.seed(42)  # 可重复性

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
        'like_count': [np.random.poisson(1000) for _ in range(num_samples)],
        'comment_count': [np.random.poisson(100) for _ in range(num_samples)],
        'share_count': [np.random.poisson(50) for _ in range(num_samples)],
        'collect_count': [np.random.poisson(20) for _ in range(num_samples)],
        'engagement_score': [np.random.exponential(5000) for _ in range(num_samples)],
        'source_entry': [np.random.choice(['search', 'topic', 'manual_url'], p=[0.5, 0.3, 0.2]) for _ in range(num_samples)],
        'crawl_date': [base_date for _ in range(num_samples)],
        'data_version': ['v0.1' for _ in range(num_samples)]
    }

    df = pd.DataFrame(data)

    # 添加一些衍生特征
    df['like_log'] = np.log1p(df['like_count'])
    df['comment_log'] = np.log1p(df['comment_count'])
    df['share_log'] = np.log1p(df['share_count'])
    df['engagement_log'] = np.log1p(df['engagement_score'])
    df['has_hashtags'] = (df['hashtag_count'] > 0).astype(int)
    df['text_length_category'] = pd.cut(df['text_length'],
                                        bins=[0, 10, 50, 100, np.inf],
                                        labels=['very_short', 'short', 'medium', 'long'])

    # 确保日期类型
    date_cols = ['publish_date', 'crawl_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.date

    return df


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行抖音数据探索性分析（EDA）')
    parser.add_argument('--input', type=Path, help='输入文件路径')
    parser.add_argument('--config', type=Path, default=None, help='配置文件路径')
    parser.add_argument('--mock', action='store_true', help='使用mock模式（无需真实数据）')
    parser.add_argument('--samples', type=int, default=30, help='mock数据样本数')
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='保存可视化图表（默认：True）')
    parser.add_argument('--report-only', action='store_true',
                       help='只生成报告，不保存图表')
    parser.add_argument('--quick', action='store_true',
                       help='快速模式（只运行基础分析）')
    parser.add_argument('--output-format', choices=['json', 'html', 'both'],
                       default='both', help='报告输出格式')

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.config)

    # 加载配置
    config = load_config(args.config)

    print("=" * 60)
    print("抖音数据探索性分析（EDA）启动")
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

    print(f"数据维度: {df_input.shape[1]} 列 x {df_input.shape[0]} 行")
    print(f"数据列: {list(df_input.columns)}")
    print("-" * 60)

    # 运行EDA
    analyzer = EDAAnalyzer(args.config)

    if args.quick:
        print("运行快速EDA（基础分析）...")
        results = analyzer.basic_summary(df_input)

        # 输出基础统计
        print("\n基础统计摘要:")
        print(f"总记录数: {results['dataset_info']['total_records']}")
        print(f"总列数: {results['dataset_info']['total_columns']}")

        if results['source_distribution']:
            print("\n来源分布:")
            for source, pct in results['source_distribution']['percentages'].items():
                print(f"  {source}: {pct}%")

        if 'engagement_summary' in results:
            eng = results['engagement_summary']
            print(f"\n互动分数统计:")
            print(f"  均值: {eng['mean']:.2f}")
            print(f"  中位数: {eng['median']:.2f}")
            print(f"  标准差: {eng['std']:.2f}")

        # 保存基础报告
        import json
        output_dir = Path(config.get('settings', {}).get('paths', {}).get('processed_data', './data/processed'))
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / 'eda_basic_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n基础报告保存至: {report_path}")

    else:
        print("运行综合EDA...")
        save_plots = args.save_plots and not args.report_only
        report = analyzer.generate_report(df_input, save_all_plots=save_plots)

        # 输出关键发现
        print("\n" + "=" * 60)
        print("EDA关键发现")
        print("=" * 60)

        # 数据集信息
        print(f"数据集大小: {report['dataset_shape']['rows']} 行, {report['dataset_shape']['columns']} 列")

        # 来源分布
        if 'source_distribution' in report.get('basic_summary', {}):
            source_info = report['basic_summary']['source_distribution']
            if 'percentages' in source_info:
                print("\n来源分布:")
                for source, pct in source_info['percentages'].items():
                    print(f"  {source}: {pct}%")

        # 分布特征
        if 'distributions' in report:
            print("\n分布特征:")
            for col, stats in report['distributions'].items():
                skew = stats.get('skewness', 0)
                if abs(skew) > 2:
                    print(f"  {col}: 偏度 = {skew:.2f} ({'右偏' if skew > 0 else '左偏'})")

        # 相关性
        if 'correlations' in report and report['correlations']:
            print("\n强相关性发现 (|r| > 0.7):")
            # 查找强相关性
            strong_corrs = []
            if isinstance(report['correlations'], dict):
                for var1, corr_dict in report['correlations'].items():
                    if isinstance(corr_dict, dict):
                        for var2, corr in corr_dict.items():
                            if var1 != var2 and abs(corr) > 0.7:
                                strong_corrs.append((var1, var2, corr))

            for var1, var2, corr in strong_corrs[:5]:  # 最多显示5个
                print(f"  {var1} <-> {var2}: r = {corr:.2f}")

        # 时间模式
        if 'time_patterns' in report:
            time_info = report['time_patterns']
            if 'hourly_distribution' in time_info:
                hourly = time_info['hourly_distribution']
                if hourly:
                    peak_hour = max(hourly.items(), key=lambda x: x[1])[0]
                    print(f"\n发布高峰时段: {peak_hour}:00")

            if 'engagement_by_hour' in time_info:
                engagement_by_hour = time_info['engagement_by_hour']
                if engagement_by_hour:
                    best_hour = max(engagement_by_hour.items(), key=lambda x: x[1])[0]
                    print(f"互动最佳时段: {best_hour}:00")

        # 文本特征
        if 'text_features' in report:
            text_info = report['text_features']
            if 'text_length' in text_info:
                length_info = text_info['text_length']
                print(f"\n文案长度统计:")
                print(f"  均值: {length_info.get('mean', 0):.1f} 字符")
                print(f"  中位数: {length_info.get('median', 0):.1f} 字符")
                print(f"  无文案比例: {length_info.get('zero_pct', 0):.1f}%")

            if 'text_engagement_correlation' in text_info:
                corr = text_info['text_engagement_correlation']
                print(f"  文案长度与互动相关性: r = {corr:.2f}")

        # 互动分析
        if 'engagement' in report:
            eng_info = report['engagement']
            if 'engagement_distribution' in eng_info:
                dist = eng_info['engagement_distribution']
                print(f"\n互动分数统计:")
                print(f"  均值: {dist.get('mean', 0):.2f}")
                print(f"  中位数: {dist.get('median', 0):.2f}")
                print(f"  偏度: {dist.get('skew', 0):.2f}")

            if 'engagement_by_source' in eng_info:
                source_eng = eng_info['engagement_by_source']
                if 'mean' in source_eng:
                    best_source = max(source_eng['mean'].items(), key=lambda x: x[1])[0]
                    print(f"  最高平均互动来源: '{best_source}'")

        # 洞察建议
        if 'insights' in report and report['insights']:
            print("\n洞察与建议:")
            for i, insight in enumerate(report['insights'][:5], 1):  # 最多显示5个
                print(f"  {i}. {insight}")

        # 输出格式处理
        output_dir = Path(config.get('settings', {}).get('paths', {}).get('processed_data', './data/processed'))
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存JSON报告
        if args.output_format in ['json', 'both']:
            import json
            report_path = output_dir / 'eda_comprehensive_report.json'
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n详细报告保存至: {report_path}")

        # 生成HTML报告（简化版）
        if args.output_format in ['html', 'both']:
            try:
                _generate_html_report(report, output_dir)
            except Exception as e:
                print(f"HTML报告生成失败: {e}")

        print(f"\n可视化图表保存至: {output_dir}")

    print("\nEDA完成!")
    return 0


def _generate_html_report(report: dict, output_dir: Path):
    """生成HTML报告（简化版）"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>抖音数据EDA报告</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
            h1, h2, h3 { color: #333; }
            .section { margin-bottom: 30px; padding: 20px; background: #f9f9f9; border-radius: 5px; }
            .metric { display: inline-block; margin: 10px 20px 10px 0; padding: 10px; background: white; border-radius: 3px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .metric .value { font-size: 24px; font-weight: bold; color: #007acc; }
            .metric .label { font-size: 14px; color: #666; }
            .insight { background: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 10px 0; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>抖音数据探索性分析（EDA）报告</h1>
        <p>生成时间: {timestamp}</p>

        <div class="section">
            <h2>数据集概览</h2>
            <div class="metric"><div class="value">{row_count}</div><div class="label">总记录数</div></div>
            <div class="metric"><div class="value">{column_count}</div><div class="label">特征数量</div></div>
        </div>

        <div class="section">
            <h2>关键指标</h2>
            {metrics_html}
        </div>

        <div class="section">
            <h2>洞察与建议</h2>
            {insights_html}
        </div>

        <div class="section">
            <h2>可视化图表</h2>
            <p>图表已保存至以下文件：</p>
            <ul>
                <li>分布直方图: distribution_*.png</li>
                <li>相关性热图: correlation_heatmap.png</li>
                <li>时间序列图: *.png</li>
            </ul>
        </div>
    </body>
    </html>
    """

    # 填充数据
    metrics = []
    if 'engagement_distribution' in report.get('engagement', {}):
        eng = report['engagement']['engagement_distribution']
        metrics.append(f"<div class='metric'><div class='value'>{eng.get('mean', 0):.0f}</div><div class='label'>平均互动分数</div></div>")

    if 'text_length' in report.get('text_features', {}):
        text = report['text_features']['text_length']
        metrics.append(f"<div class='metric'><div class='value'>{text.get('mean', 0):.0f}</div><div class='label'>平均文案长度</div></div>")

    insights_html = ""
    if 'insights' in report:
        for insight in report['insights']:
            insights_html += f'<div class="insight">{insight}</div>'

    html_content = html_content.format(
        timestamp=report.get('timestamp', ''),
        row_count=report.get('dataset_shape', {}).get('rows', 0),
        column_count=report.get('dataset_shape', {}).get('columns', 0),
        metrics_html=''.join(metrics),
        insights_html=insights_html
    )

    html_path = output_dir / 'eda_report.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML报告保存至: {html_path}")


if __name__ == '__main__':
    sys.exit(main())