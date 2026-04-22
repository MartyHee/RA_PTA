#!/usr/bin/env python3
"""
准备LR baseline建模数据集。

功能：
1. 加载完整版特征数据（features.parquet）
2. 保存完整特征CSV（full_features.csv）
3. 构造engagement_score和label
4. 选择默认入模字段，删除默认不入模字段
5. 按发布时间升序排序，8:2划分训练集和测试集
6. 输出train/test CSV到LR_baseline目录

使用示例：
python prepare_lr_dataset.py \
    --input "data/features/20260421_194709/v1_fix5/features.parquet" \
    --train-output "D:\CodeData\Program Coding\ByteDance\RA_PTA\LR_baseline\data\train\train_v1_fix5.csv" \
    --test-output "D:\CodeData\Program Coding\ByteDance\RA_PTA\LR_baseline\data\test\test_v1_fix5.csv" \
    --split-ratio 0.8
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.io_utils import read_parquet, write_csv, ensure_dir
from src.utils.time_utils import parse_douyin_time
from src.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


class LRDatasetPreparer:
    """LR baseline数据集准备器。"""

    def __init__(self, verbose: bool = True):
        """初始化。

        Args:
            verbose: 是否输出详细信息
        """
        self.verbose = verbose

    def load_features(self, input_path: Path) -> pd.DataFrame:
        """加载特征数据。

        Args:
            input_path: 输入文件路径（Parquet或CSV）

        Returns:
            特征DataFrame
        """
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        logger.info(f"加载特征数据: {input_path}")

        if input_path.suffix == '.parquet':
            df = read_parquet(input_path)
        elif input_path.suffix == '.csv':
            df = pd.read_csv(input_path, low_memory=False)
        else:
            raise ValueError(f"不支持的格式: {input_path.suffix}")

        logger.info(f"加载 {len(df)} 条记录，{len(df.columns)} 个字段")

        if self.verbose:
            print(f"字段列表: {list(df.columns)}")
            print(f"数据形状: {df.shape}")

        return df

    def save_full_csv(self, df: pd.DataFrame, output_dir: Path) -> Path:
        """保存完整特征CSV。

        Args:
            df: 特征DataFrame
            output_dir: 输出目录

        Returns:
            保存的CSV文件路径
        """
        ensure_dir(output_dir)
        csv_path = output_dir / 'full_features.csv'

        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"完整特征CSV保存到: {csv_path}")

        return csv_path

    def parse_publish_time(self, df: pd.DataFrame) -> pd.Series:
        """解析发布时间，用于排序。

        Args:
            df: 特征DataFrame，包含publish_time_raw字段

        Returns:
            解析后的日期时间Series
        """
        if 'publish_time_raw' not in df.columns:
            logger.warning("publish_time_raw字段不存在，使用crawl_time排序")
            if 'crawl_time' in df.columns:
                return pd.to_datetime(df['crawl_time'], errors='coerce')
            else:
                raise ValueError("没有可用于排序的时间字段")

        # 解析publish_time_raw
        publish_times = []
        for raw_time in df['publish_time_raw']:
            if pd.isna(raw_time) or raw_time == '':
                publish_times.append(pd.NaT)
            else:
                # 尝试解析时间戳
                try:
                    # 先尝试直接转换数值
                    if isinstance(raw_time, (int, float, np.integer, np.floating)):
                        timestamp = float(raw_time)
                    else:
                        timestamp = float(str(raw_time).strip())

                    # 判断是秒还是毫秒
                    if timestamp > 1e12:  # 毫秒（大于2001年）
                        dt = pd.to_datetime(timestamp, unit='ms', errors='coerce')
                    else:  # 秒
                        dt = pd.to_datetime(timestamp, unit='s', errors='coerce')

                    publish_times.append(dt)
                except Exception as e:
                    logger.debug(f"解析时间失败 {raw_time}: {e}")
                    publish_times.append(pd.NaT)

        publish_dt = pd.Series(publish_times, index=df.index)

        # 填充缺失值（使用crawl_time）
        if 'crawl_time' in df.columns and publish_dt.isna().any():
            crawl_dt = pd.to_datetime(df['crawl_time'], errors='coerce')
            publish_dt = publish_dt.fillna(crawl_dt)

        logger.info(f"发布时间解析: {publish_dt.notna().sum()}/{len(df)} 有效")

        return publish_dt

    def calculate_engagement_score(self, df: pd.DataFrame) -> pd.Series:
        """计算互动分数。

        公式: engagement_score = like_count_num + 2*comment_count_num + 2*share_count_num + 2*collect_count

        Args:
            df: 特征DataFrame

        Returns:
            互动分数Series
        """
        required_cols = ['like_count_num', 'comment_count_num', 'share_count_num', 'collect_count']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"缺少计算engagement_score所需的字段: {missing_cols}")

        # 确保数值类型
        like = pd.to_numeric(df['like_count_num'], errors='coerce').fillna(0)
        comment = pd.to_numeric(df['comment_count_num'], errors='coerce').fillna(0)
        share = pd.to_numeric(df['share_count_num'], errors='coerce').fillna(0)
        collect = pd.to_numeric(df['collect_count'], errors='coerce').fillna(0)

        engagement_score = like + 2*comment + 2*share + 2*collect

        logger.info(f"engagement_score统计: min={engagement_score.min():.2f}, "
                   f"max={engagement_score.max():.2f}, mean={engagement_score.mean():.2f}")

        return engagement_score

    def create_label(self, engagement_score: pd.Series) -> pd.Series:
        """创建二分类标签。

        按engagement_score中位数划分：
        - 大于等于中位数为正样本，label = 1
        - 小于中位数为负样本，label = 0

        Args:
            engagement_score: 互动分数Series

        Returns:
            标签Series
        """
        median_score = engagement_score.median()
        label = (engagement_score >= median_score).astype(int)

        pos_count = label.sum()
        neg_count = len(label) - pos_count

        logger.info(f"标签分布: 正样本={pos_count} ({pos_count/len(label)*100:.1f}%), "
                   f"负样本={neg_count} ({neg_count/len(label)*100:.1f}%), "
                   f"中位数={median_score:.2f}")

        return label

    def get_field_categories(self, df: pd.DataFrame) -> Dict[str, str]:
        """获取字段分类。

        基于v1_fix5版本的字段分类规则。

        Args:
            df: 特征DataFrame

        Returns:
            字段到分类的映射字典
        """
        # 定义分类映射（与feature_storage.py一致）
        retention_fields = {
            'video_id', 'page_url', 'author_id', 'publish_time_raw', 'crawl_time',
            'source_entry', 'match_type', 'confidence'
        }

        model_fields = {
            'author_follower_count', 'author_total_favorited', 'hashtag_count',
            'duration_sec', 'publish_hour', 'publish_weekday', 'is_weekend',
            'days_since_publish', 'author_verification_type', 'desc_text_length',
            'has_desc_text', 'has_hashtag'
        }

        excluded_fields = {
            'like_count_num', 'comment_count_num', 'share_count_num', 'collect_count'
        }

        text_fields = {
            'desc_text', 'author_name', 'hashtag_list'
        }

        # 构建映射
        category_map = {}
        for field in df.columns:
            if field in retention_fields:
                category_map[field] = 'retention'
            elif field in model_fields:
                category_map[field] = 'model'
            elif field in excluded_fields:
                category_map[field] = 'excluded'
            elif field in text_fields:
                category_map[field] = 'text'
            else:
                # 未知字段，默认归为保留字段
                category_map[field] = 'retention'

        return category_map

    def select_modeling_features(self, df: pd.DataFrame, category_map: Dict[str, str]) -> pd.DataFrame:
        """选择建模特征。

        保留model类别字段，删除excluded类别字段。
        保留retention和text字段可选（默认不保留）。

        Args:
            df: 特征DataFrame
            category_map: 字段分类映射

        Returns:
            建模特征DataFrame
        """
        # 选择model类别字段
        model_fields = [field for field, category in category_map.items() if category == 'model']

        if not model_fields:
            logger.warning("没有找到model类别字段，使用所有数值字段")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            model_fields = list(numeric_cols)

        logger.info(f"默认入模字段 ({len(model_fields)}个): {model_fields}")

        # 排除excluded字段
        excluded_fields = [field for field, category in category_map.items() if category == 'excluded']
        if excluded_fields:
            logger.info(f"排除字段 ({len(excluded_fields)}个): {excluded_fields}")

        # 创建建模数据集
        df_model = df[model_fields].copy()

        # 添加标签列（如果存在）
        if 'label' in df.columns:
            df_model['label'] = df['label']

        if 'engagement_score' in df.columns:
            df_model['engagement_score'] = df['engagement_score']

        logger.info(f"建模数据集形状: {df_model.shape}")

        return df_model

    def split_train_test(self, df: pd.DataFrame, split_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """划分训练集和测试集。

        按发布时间升序排序，前split_ratio为训练集，后1-split_ratio为测试集。

        Args:
            df: 特征DataFrame（包含publish_time_raw）
            split_ratio: 训练集比例

        Returns:
            (训练集DataFrame, 测试集DataFrame)
        """
        if 'publish_time_raw' not in df.columns:
            logger.warning("publish_time_raw字段不存在，使用随机划分")
            # 随机划分
            train_size = int(len(df) * split_ratio)
            indices = np.random.permutation(len(df))
            train_idx = indices[:train_size]
            test_idx = indices[train_size:]

            df_train = df.iloc[train_idx].copy()
            df_test = df.iloc[test_idx].copy()
        else:
            # 解析发布时间用于排序
            publish_dt = self.parse_publish_time(df)

            # 按发布时间排序
            df_sorted = df.copy()
            df_sorted['_publish_dt'] = publish_dt

            # 删除无法解析时间的行
            valid_mask = df_sorted['_publish_dt'].notna()
            if not valid_mask.all():
                logger.warning(f"删除 {valid_mask.sum()}/{len(df)} 条无法解析时间的记录")
                df_sorted = df_sorted[valid_mask].copy()

            df_sorted = df_sorted.sort_values('_publish_dt')

            # 划分
            train_size = int(len(df_sorted) * split_ratio)
            df_train = df_sorted.iloc[:train_size].copy()
            df_test = df_sorted.iloc[train_size:].copy()

            # 删除临时列
            df_train = df_train.drop(columns=['_publish_dt'])
            df_test = df_test.drop(columns=['_publish_dt'])

        logger.info(f"数据集划分: 训练集={len(df_train)} ({len(df_train)/len(df)*100:.1f}%), "
                   f"测试集={len(df_test)} ({len(df_test)/len(df)*100:.1f}%)")

        return df_train, df_test

    def prepare_dataset(
        self,
        input_path: Path,
        train_output_path: Path,
        test_output_path: Path,
        split_ratio: float = 0.8,
        save_full_csv: bool = True
    ) -> Dict[str, Any]:
        """准备LR baseline数据集。

        Args:
            input_path: 输入特征文件路径
            train_output_path: 训练集输出路径
            test_output_path: 测试集输出路径
            split_ratio: 训练集比例
            save_full_csv: 是否保存完整特征CSV

        Returns:
            处理结果字典
        """
        logger.info("=" * 60)
        logger.info("开始准备LR baseline数据集")
        logger.info("=" * 60)

        # 1. 加载特征数据
        df = self.load_features(input_path)

        # 2. 保存完整特征CSV
        if save_full_csv:
            output_dir = input_path.parent
            full_csv_path = self.save_full_csv(df, output_dir)

        # 3. 计算engagement_score和label
        engagement_score = self.calculate_engagement_score(df)
        label = self.create_label(engagement_score)

        df = df.copy()
        df['engagement_score'] = engagement_score
        df['label'] = label

        # 4. 获取字段分类
        category_map = self.get_field_categories(df)

        # 统计字段分类
        category_counts = {}
        for category in category_map.values():
            category_counts[category] = category_counts.get(category, 0) + 1

        logger.info(f"字段分类统计: {category_counts}")

        # 5. 按发布时间划分训练集和测试集（使用原始df，包含publish_time_raw）
        df_train_full, df_test_full = self.split_train_test(df, split_ratio)

        # 6. 分别选择建模特征
        df_train_model = self.select_modeling_features(df_train_full, category_map)
        df_test_model = self.select_modeling_features(df_test_full, category_map)

        # 7. 确保输出目录存在
        ensure_dir(train_output_path.parent)
        ensure_dir(test_output_path.parent)

        # 8. 保存训练集和测试集
        df_train_model.to_csv(train_output_path, index=False, encoding='utf-8')
        df_test_model.to_csv(test_output_path, index=False, encoding='utf-8')

        logger.info(f"训练集保存到: {train_output_path}")
        logger.info(f"测试集保存到: {test_output_path}")

        # 9. 生成结果统计
        model_features = [col for col in df_train_model.columns
                         if col not in ['label', 'engagement_score']]

        result = {
            'input_path': str(input_path),
            'full_csv_path': str(full_csv_path) if save_full_csv else None,
            'train_output_path': str(train_output_path),
            'test_output_path': str(test_output_path),
            'total_samples': len(df),
            'positive_samples': int(label.sum()),
            'negative_samples': int(len(label) - label.sum()),
            'train_samples': len(df_train_model),
            'test_samples': len(df_test_model),
            'model_features': model_features,
            'excluded_features': [field for field, category in category_map.items()
                                 if category == 'excluded'],
            'engagement_score_median': float(engagement_score.median()),
            'split_ratio': split_ratio,
        }

        logger.info("=" * 60)
        logger.info("LR baseline数据集准备完成")
        logger.info("=" * 60)

        return result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='准备LR baseline建模数据集')

    parser.add_argument('--input', type=Path, required=True,
                       help='输入特征文件路径（Parquet或CSV）')
    parser.add_argument('--train-output', type=Path, required=True,
                       help='训练集输出路径')
    parser.add_argument('--test-output', type=Path, required=True,
                       help='测试集输出路径')
    parser.add_argument('--split-ratio', type=float, default=0.8,
                       help='训练集比例（默认0.8）')
    parser.add_argument('--no-full-csv', action='store_true',
                       help='不保存完整特征CSV')
    parser.add_argument('--verbose', action='store_true',
                       help='详细日志输出')

    args = parser.parse_args()

    # 设置日志
    setup_logging()

    # 创建准备器
    preparer = LRDatasetPreparer(verbose=args.verbose)

    try:
        # 准备数据集
        result = preparer.prepare_dataset(
            input_path=args.input,
            train_output_path=args.train_output,
            test_output_path=args.test_output,
            split_ratio=args.split_ratio,
            save_full_csv=not args.no_full_csv
        )

        # 打印结果摘要
        print("\n" + "=" * 60)
        print("LR baseline数据集准备结果")
        print("=" * 60)
        print(f"总样本数: {result['total_samples']}")
        print(f"正样本数: {result['positive_samples']} ({result['positive_samples']/result['total_samples']*100:.1f}%)")
        print(f"负样本数: {result['negative_samples']} ({result['negative_samples']/result['total_samples']*100:.1f}%)")
        print(f"训练集样本数: {result['train_samples']} ({result['train_samples']/result['total_samples']*100:.1f}%)")
        print(f"测试集样本数: {result['test_samples']} ({result['test_samples']/result['total_samples']*100:.1f}%)")
        print(f"默认入模字段数: {len(result['model_features'])}")
        print(f"排除字段数: {len(result['excluded_features'])}")
        print(f"engagement_score中位数: {result['engagement_score_median']:.2f}")
        print(f"训练集输出: {result['train_output_path']}")
        print(f"测试集输出: {result['test_output_path']}")

        if result['full_csv_path']:
            print(f"完整特征CSV: {result['full_csv_path']}")

        print("\n默认入模字段:")
        for i, feature in enumerate(result['model_features'], 1):
            print(f"  {i:2d}. {feature}")

        if result['excluded_features']:
            print("\n排除字段:")
            for i, feature in enumerate(result['excluded_features'], 1):
                print(f"  {i:2d}. {feature}")

        print("\n数据集准备成功!")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())