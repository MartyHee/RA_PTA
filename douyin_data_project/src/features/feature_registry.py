"""
特征注册表。

管理特征定义、版本和转换逻辑。
"""
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import pandas as pd

from .feature_schema import FeatureSchema


class FeatureRegistry:
    """特征注册表。"""

    def __init__(self):
        """初始化特征注册表。"""
        self.feature_definitions = {}
        self.feature_versions = {}

    def register_feature(
        self,
        name: str,
        version: str = 'v1',
        description: str = '',
        feature_type: str = 'numeric',
        source_fields: Optional[List[str]] = None,
        transformation: Optional[Callable] = None,
        default_value: Any = None
    ):
        """注册一个特征。

        Args:
            name: 特征名称
            version: 特征版本
            description: 特征描述
            feature_type: 特征类型（numeric, categorical, text, datetime）
            source_fields: 源字段列表
            transformation: 转换函数
            default_value: 默认值
        """
        if version not in self.feature_versions:
            self.feature_versions[version] = {}

        self.feature_versions[version][name] = {
            'name': name,
            'version': version,
            'description': description,
            'feature_type': feature_type,
            'source_fields': source_fields or [],
            'transformation': transformation,
            'default_value': default_value,
            'registered_at': datetime.now().isoformat(),
        }

    def get_feature_definitions(self, version: str = 'v1') -> Dict[str, Dict[str, Any]]:
        """获取指定版本的特征定义。

        Args:
            version: 特征版本

        Returns:
            特征定义字典
        """
        return self.feature_versions.get(version, {})

    def get_feature_list(self, version: str = 'v1') -> List[str]:
        """获取指定版本的特征列表。

        Args:
            version: 特征版本

        Returns:
            特征名称列表
        """
        definitions = self.get_feature_definitions(version)
        return list(definitions.keys())

    def apply_transformations(
        self,
        df: pd.DataFrame,
        version: str = 'v1',
        verbose: bool = False
    ) -> pd.DataFrame:
        """应用特征转换。

        Args:
            df: 输入DataFrame
            version: 特征版本
            verbose: 是否输出详细信息

        Returns:
            转换后的DataFrame
        """
        definitions = self.get_feature_definitions(version)
        df_out = df.copy()

        for name, definition in definitions.items():
            transformation = definition.get('transformation')
            if transformation and callable(transformation):
                try:
                    # 检查源字段是否存在
                    source_fields = definition.get('source_fields', [])
                    missing_sources = [f for f in source_fields if f not in df.columns]
                    if missing_sources:
                        if verbose:
                            print(f"警告: 特征 {name} 缺少源字段: {missing_sources}")
                        continue

                    # 应用转换
                    df_out[name] = transformation(df_out)
                    if verbose:
                        print(f"应用转换: {name}")

                except Exception as e:
                    if verbose:
                        print(f"特征 {name} 转换失败: {e}")
                    # 使用默认值
                    default_value = definition.get('default_value')
                    if default_value is not None:
                        df_out[name] = default_value

        return df_out

    def register_v1_features(self):
        """注册v1版本的特征定义。"""
        # 原始保留字段
        self.register_feature(
            name='video_id',
            version='v1',
            description='视频唯一标识',
            feature_type='string',
            source_fields=['video_id'],
        )

        self.register_feature(
            name='page_url',
            version='v1',
            description='页面URL',
            feature_type='string',
            source_fields=['page_url'],
        )

        self.register_feature(
            name='author_id',
            version='v1',
            description='作者ID',
            feature_type='string',
            source_fields=['author_id'],
            default_value='',
        )

        self.register_feature(
            name='publish_time_raw',
            version='v1',
            description='原始发布时间文本',
            feature_type='string',
            source_fields=['publish_time_raw'],
            default_value='',
        )

        self.register_feature(
            name='crawl_time',
            version='v1',
            description='抓取时间',
            feature_type='datetime',
            source_fields=['crawl_time'],
        )

        # 关键数值特征
        self.register_feature(
            name='author_follower_count',
            version='v1',
            description='作者粉丝数',
            feature_type='numeric',
            source_fields=['author_follower_count'],
            default_value=-1,
        )

        self.register_feature(
            name='author_total_favorited',
            version='v1',
            description='作者总获赞数',
            feature_type='numeric',
            source_fields=['author_total_favorited'],
            default_value=-1,
        )

        self.register_feature(
            name='hashtag_count',
            version='v1',
            description='话题标签数量',
            feature_type='numeric',
            source_fields=['hashtag_count'],
            default_value=0,
        )

        self.register_feature(
            name='duration_sec',
            version='v1',
            description='视频时长（秒）',
            feature_type='numeric',
            source_fields=['duration_sec'],
            default_value=0,
        )

        self.register_feature(
            name='collect_count',
            version='v1',
            description='收藏数',
            feature_type='numeric',
            source_fields=['collect_count'],
            default_value=0,
        )

        # 计数转换特征（在pipeline中处理）
        self.register_feature(
            name='like_count_num',
            version='v1',
            description='点赞数（数值化）',
            feature_type='numeric',
            source_fields=['like_count_raw'],
            default_value=0,
        )

        self.register_feature(
            name='comment_count_num',
            version='v1',
            description='评论数（数值化）',
            feature_type='numeric',
            source_fields=['comment_count_raw'],
            default_value=0,
        )

        self.register_feature(
            name='share_count_num',
            version='v1',
            description='分享数（数值化）',
            feature_type='numeric',
            source_fields=['share_count_raw'],
            default_value=0,
        )

        # 时间派生特征（在pipeline中处理）
        self.register_feature(
            name='publish_hour',
            version='v1',
            description='发布时间小时',
            feature_type='numeric',
            source_fields=['publish_time_std'],
            default_value=-1,
        )

        self.register_feature(
            name='publish_weekday',
            version='v1',
            description='星期几',
            feature_type='numeric',
            source_fields=['publish_time_std'],
            default_value=-1,
        )

        self.register_feature(
            name='is_weekend',
            version='v1',
            description='是否周末',
            feature_type='numeric',
            source_fields=['publish_time_std'],
            default_value=0,
        )

        self.register_feature(
            name='days_since_publish',
            version='v1',
            description='发布天数',
            feature_type='numeric',
            source_fields=['publish_time_std', 'crawl_time'],
            default_value=-1,
        )

        # 基本类别特征
        self.register_feature(
            name='source_entry',
            version='v1',
            description='数据来源入口',
            feature_type='categorical',
            source_fields=['source_entry'],
            default_value='unknown',
        )

        self.register_feature(
            name='match_type',
            version='v1',
            description='匹配类型',
            feature_type='categorical',
            source_fields=['match_type'],
            default_value='',
        )

        self.register_feature(
            name='confidence',
            version='v1',
            description='置信度',
            feature_type='categorical',
            source_fields=['confidence'],
            default_value='',
        )

        self.register_feature(
            name='author_verification_type',
            version='v1',
            description='作者认证类型',
            feature_type='categorical',
            source_fields=['author_verification_type'],
            default_value='',
        )

        # 文本字段
        self.register_feature(
            name='desc_text',
            version='v1',
            description='视频描述文本',
            feature_type='text',
            source_fields=['desc_text'],
            default_value='',
        )

        self.register_feature(
            name='author_name',
            version='v1',
            description='作者名称',
            feature_type='text',
            source_fields=['author_name'],
            default_value='',
        )

        self.register_feature(
            name='hashtag_list',
            version='v1',
            description='话题标签列表（JSON格式）',
            feature_type='text',
            source_fields=['hashtag_list'],
            default_value='[]',
        )

        # 可选轻量派生特征
        self.register_feature(
            name='desc_text_length',
            version='v1',
            description='描述文本长度',
            feature_type='numeric',
            source_fields=['desc_text'],
            default_value=0,
        )

        self.register_feature(
            name='has_desc_text',
            version='v1',
            description='是否有描述文本',
            feature_type='numeric',
            source_fields=['desc_text'],
            default_value=0,
        )

        self.register_feature(
            name='has_hashtag',
            version='v1',
            description='是否有话题标签',
            feature_type='numeric',
            source_fields=['hashtag_list'],
            default_value=0,
        )


# 全局注册表实例
_registry = FeatureRegistry()
_registry.register_v1_features()


def get_registry() -> FeatureRegistry:
    """获取全局特征注册表实例。"""
    return _registry