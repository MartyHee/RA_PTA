"""
特征存储管理。

负责特征产物的保存、元数据生成和版本管理。
"""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np

from ..utils.io_utils import write_parquet, write_csv, write_json, ensure_dir
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FeatureStorage:
    """特征存储管理器。"""

    def __init__(self, base_dir: Union[str, Path] = "data/features"):
        """初始化特征存储。

        Args:
            base_dir: 特征存储基础目录
        """
        self.base_dir = Path(base_dir)
        ensure_dir(self.base_dir)

    def get_feature_dir(self, run_id: str, feature_version: str) -> Path:
        """获取特征目录路径。

        Args:
            run_id: 运行ID（如20260421_194709）
            feature_version: 特征版本（如v1）

        Returns:
            特征目录路径
        """
        feature_dir = self.base_dir / run_id / feature_version
        ensure_dir(feature_dir)
        return feature_dir

    def save_features(
        self,
        df_features: pd.DataFrame,
        run_id: str,
        feature_version: str,
        input_data_path: Optional[Union[str, Path]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        save_sample: bool = True
    ) -> Dict[str, Path]:
        """保存特征产物。

        Args:
            df_features: 特征DataFrame
            run_id: 运行ID
            feature_version: 特征版本
            input_data_path: 输入数据路径（用于元数据记录）
            metadata: 额外元数据
            save_sample: 是否保存CSV样本

        Returns:
            保存的文件路径字典
        """
        # 获取特征目录
        feature_dir = self.get_feature_dir(run_id, feature_version)

        # 生成时间戳
        created_at = datetime.now().isoformat()

        # 基本元数据
        base_metadata = {
            'feature_version': feature_version,
            'created_at': created_at,
            'input_data_path': str(input_data_path) if input_data_path else None,
            'input_run_id': run_id,
            'num_samples': len(df_features),
            'num_features': len(df_features.columns),
            'feature_fields': list(df_features.columns),
            'field_categories': self._get_field_categories(list(df_features.columns), feature_version),
        }

        # 合并额外元数据
        if metadata:
            base_metadata.update(metadata)

        # 保存特征数据（Parquet格式）
        features_path = feature_dir / 'features.parquet'
        write_parquet(features_path, df_features)
        logger.info(f"特征数据保存到: {features_path}")

        # 保存特征列表
        feature_list_path = feature_dir / 'feature_list.json'
        feature_list = {
            'feature_version': feature_version,
            'created_at': created_at,
            'features': list(df_features.columns),
            'dtypes': {col: str(dtype) for col, dtype in df_features.dtypes.items()}
        }
        write_json(feature_list_path, feature_list)
        logger.info(f"特征列表保存到: {feature_list_path}")

        # 保存特征元数据
        metadata_path = feature_dir / 'feature_metadata.json'
        write_json(metadata_path, base_metadata)
        logger.info(f"特征元数据保存到: {metadata_path}")

        # 保存构建报告
        report_path = feature_dir / 'build_report.json'
        report = self._generate_build_report(df_features, base_metadata)
        write_json(report_path, report)
        logger.info(f"构建报告保存到: {report_path}")

        # 保存CSV样本（前100行）
        if save_sample and len(df_features) > 0:
            sample_path = feature_dir / 'sample_features.csv'
            sample_size = min(100, len(df_features))
            df_features.head(sample_size).to_csv(sample_path, index=False, encoding='utf-8')
            logger.info(f"特征样本保存到: {sample_path}")

        # 创建latest符号链接（仅限Unix系统，Windows下创建副本）
        self._create_latest_link(run_id, feature_version)

        # 返回文件路径
        return {
            'features': features_path,
            'feature_list': feature_list_path,
            'metadata': metadata_path,
            'build_report': report_path,
            'sample': feature_dir / 'sample_features.csv' if save_sample else None,
        }

    def _get_field_categories(self, feature_fields: List[str], feature_version: str) -> Dict[str, str]:
        """获取字段分类映射。

        Args:
            feature_fields: 特征字段列表
            feature_version: 特征版本

        Returns:
            字段到分类的映射字典
        """
        # v1版本的字段分类
        if feature_version == 'v1' or feature_version.startswith('v1'):
            # 定义分类映射
            category_map = {}

            # A. 保留字段（用于追溯，不默认入模）
            retention_fields = {
                'video_id', 'page_url', 'author_id', 'publish_time_raw', 'crawl_time',
                'source_entry', 'match_type', 'confidence'
            }

            # B. 可入模字段（第一版候选）
            model_fields = {
                'author_follower_count', 'author_total_favorited', 'hashtag_count',
                'duration_sec', 'publish_hour', 'publish_weekday', 'is_weekend',
                'days_since_publish', 'author_verification_type', 'desc_text_length',
                'has_desc_text', 'has_hashtag'
            }

            # C. 默认不入模字段（保留在特征表中，但不能作为当前高互动预测baseline的默认输入）
            excluded_fields = {
                'like_count_num', 'comment_count_num', 'share_count_num', 'collect_count'
            }

            # 文本字段（保留原样，不默认入模）
            text_fields = {
                'desc_text', 'author_name', 'hashtag_list'
            }

            # 构建映射
            for field in feature_fields:
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
        else:
            # 其他版本暂时全部归为保留字段
            return {field: 'retention' for field in feature_fields}

    def _generate_build_report(
        self,
        df_features: pd.DataFrame,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成构建报告。

        Args:
            df_features: 特征DataFrame
            metadata: 元数据

        Returns:
            构建报告字典
        """
        # 基本统计信息
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        categorical_cols = df_features.select_dtypes(include=['object', 'string']).columns

        numeric_stats = {}
        for col in numeric_cols:
            numeric_stats[col] = {
                'min': float(df_features[col].min()),
                'max': float(df_features[col].max()),
                'mean': float(df_features[col].mean()),
                'std': float(df_features[col].std()),
                'null_count': int(df_features[col].isnull().sum()),
                'null_pct': float(df_features[col].isnull().mean() * 100),
            }

        categorical_stats = {}
        for col in categorical_cols:
            value_counts = df_features[col].value_counts().head(10)
            categorical_stats[col] = {
                'unique_count': int(df_features[col].nunique()),
                'top_values': value_counts.to_dict(),
                'null_count': int(df_features[col].isnull().sum()),
                'null_pct': float(df_features[col].isnull().mean() * 100),
            }

        report = {
            'build_info': metadata,
            'dataset_stats': {
                'num_samples': len(df_features),
                'num_features': len(df_features.columns),
                'memory_usage_mb': float(df_features.memory_usage(deep=True).sum() / 1024 / 1024),
            },
            'feature_stats': {
                'numeric': numeric_stats,
                'categorical': categorical_stats,
            },
            'quality_metrics': {
                'completeness': float(1 - df_features.isnull().mean().mean()),
                'duplicate_rows': int(df_features.duplicated().sum()),
            }
        }

        return report

    def _create_latest_link(self, run_id: str, feature_version: str):
        """创建latest符号链接或副本。

        Args:
            run_id: 运行ID
            feature_version: 特征版本
        """
        run_dir = self.base_dir / run_id
        latest_path = run_dir / 'latest'

        try:
            # 移除现有的latest链接或目录
            if latest_path.exists():
                if latest_path.is_symlink():
                    latest_path.unlink()
                else:
                    shutil.rmtree(latest_path)

            # 创建符号链接（Unix）或目录副本（Windows）
            import os
            if os.name == 'posix':
                # Unix系统：创建符号链接
                latest_path.symlink_to(feature_version, target_is_directory=True)
                logger.info(f"创建符号链接: {latest_path} -> {feature_version}")
            else:
                # Windows系统：创建目录副本（或跳过）
                # 这里我们只记录信息，不实际创建副本
                logger.info(f"Windows系统，跳过符号链接创建。latest版本: {feature_version}")
                # 可以创建一个文本文件记录最新版本
                latest_version_file = run_dir / 'LATEST_VERSION'
                with open(latest_version_file, 'w', encoding='utf-8') as f:
                    f.write(feature_version)

        except Exception as e:
            logger.warning(f"创建latest链接失败: {e}")

    def load_features(self, run_id: str, feature_version: str = 'latest') -> pd.DataFrame:
        """加载特征数据。

        Args:
            run_id: 运行ID
            feature_version: 特征版本，默认为'latest'

        Returns:
            特征DataFrame
        """
        if feature_version == 'latest':
            # 查找latest链接或LATEST_VERSION文件
            run_dir = self.base_dir / run_id
            latest_path = run_dir / 'latest'

            if latest_path.exists() and latest_path.is_symlink():
                # 解析符号链接
                feature_version = latest_path.resolve().name
            else:
                # 检查LATEST_VERSION文件
                version_file = run_dir / 'LATEST_VERSION'
                if version_file.exists():
                    with open(version_file, 'r', encoding='utf-8') as f:
                        feature_version = f.read().strip()
                else:
                    # 查找最新的版本目录
                    version_dirs = []
                    for item in run_dir.iterdir():
                        if item.is_dir() and item.name != 'latest':
                            version_dirs.append(item.name)
                    if version_dirs:
                        feature_version = sorted(version_dirs)[-1]
                    else:
                        raise FileNotFoundError(f"在 {run_dir} 中找不到特征版本")

        # 加载特征数据
        features_path = self.base_dir / run_id / feature_version / 'features.parquet'
        if not features_path.exists():
            raise FileNotFoundError(f"特征文件不存在: {features_path}")

        df = pd.read_parquet(features_path)
        logger.info(f"从 {features_path} 加载 {len(df)} 条特征数据")
        return df

    def get_metadata(self, run_id: str, feature_version: str) -> Dict[str, Any]:
        """获取特征元数据。

        Args:
            run_id: 运行ID
            feature_version: 特征版本

        Returns:
            特征元数据
        """
        metadata_path = self.base_dir / run_id / feature_version / 'feature_metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        return metadata

    def list_versions(self, run_id: str) -> List[str]:
        """列出指定运行ID的所有特征版本。

        Args:
            run_id: 运行ID

        Returns:
            特征版本列表
        """
        run_dir = self.base_dir / run_id
        if not run_dir.exists():
            return []

        versions = []
        for item in run_dir.iterdir():
            if item.is_dir() and item.name != 'latest':
                versions.append(item.name)

        return sorted(versions)