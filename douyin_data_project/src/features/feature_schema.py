"""
特征模式定义。

定义第一版特征集合的字段结构、类型和转换规则。
"""
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, date
import pandas as pd
import numpy as np


class FeatureSchema:
    """特征模式管理类。"""

    # 第一版特征集合（简化版本）
    FEATURE_SET_V1 = {
        # 原始保留字段
        'video_id': {'dtype': 'string', 'required': True, 'description': '视频唯一标识'},
        'page_url': {'dtype': 'string', 'required': True, 'description': '页面URL'},
        'author_id': {'dtype': 'string', 'required': False, 'description': '作者ID'},
        'publish_time_raw': {'dtype': 'string', 'required': False, 'description': '原始发布时间文本'},
        'crawl_time': {'dtype': 'datetime', 'required': True, 'description': '抓取时间'},

        # 关键数值特征
        'author_follower_count': {'dtype': 'int', 'required': False, 'description': '作者粉丝数', 'default': -1},
        'author_total_favorited': {'dtype': 'int', 'required': False, 'description': '作者总获赞数', 'default': -1},
        'hashtag_count': {'dtype': 'int', 'required': False, 'description': '话题标签数量', 'default': 0},
        'duration_sec': {'dtype': 'int', 'required': False, 'description': '视频时长（秒）', 'default': 0},
        'collect_count': {'dtype': 'int', 'required': False, 'description': '收藏数', 'default': 0},
        'like_count_num': {'dtype': 'int', 'required': False, 'description': '点赞数（数值化）', 'default': 0},
        'comment_count_num': {'dtype': 'int', 'required': False, 'description': '评论数（数值化）', 'default': 0},
        'share_count_num': {'dtype': 'int', 'required': False, 'description': '分享数（数值化）', 'default': 0},

        # 时间派生特征
        'publish_hour': {'dtype': 'int', 'required': False, 'description': '发布时间小时', 'default': -1},
        'publish_weekday': {'dtype': 'int', 'required': False, 'description': '星期几', 'default': -1},
        'is_weekend': {'dtype': 'int', 'required': False, 'description': '是否周末', 'default': 0},
        'days_since_publish': {'dtype': 'int', 'required': False, 'description': '发布天数', 'default': -1},

        # 基本类别特征
        'source_entry': {'dtype': 'string', 'required': True, 'description': '数据来源入口'},
        'match_type': {'dtype': 'string', 'required': False, 'description': '匹配类型'},
        'confidence': {'dtype': 'string', 'required': False, 'description': '置信度'},
        'author_verification_type': {'dtype': 'string', 'required': False, 'description': '作者认证类型'},

        # 文本字段
        'desc_text': {'dtype': 'string', 'required': False, 'description': '视频描述文本'},
        'author_name': {'dtype': 'string', 'required': False, 'description': '作者名称'},
        'hashtag_list': {'dtype': 'string', 'required': False, 'description': '话题标签列表（JSON格式）'},

        # 可选轻量派生特征
        'desc_text_length': {'dtype': 'int', 'required': False, 'description': '描述文本长度', 'default': 0},
        'has_desc_text': {'dtype': 'int', 'required': False, 'description': '是否有描述文本', 'default': 0},
        'has_hashtag': {'dtype': 'int', 'required': False, 'description': '是否有话题标签', 'default': 0},
    }

    # 特征版本映射
    FEATURE_SETS = {
        'v1': FEATURE_SET_V1,
        'v1_fix': FEATURE_SET_V1,  # 修复版本使用相同的特征集合
        'v1_fix2': FEATURE_SET_V1,  # 修复版本2
        'v1_fix3': FEATURE_SET_V1,  # 修复版本3
        'v1_fix4': FEATURE_SET_V1,  # 修复版本4
        'v1_fix5': FEATURE_SET_V1,  # 修复版本5：明确时间特征生成优先级
    }

    @classmethod
    def get_feature_set(cls, version: str = 'v1') -> Dict[str, Dict[str, Any]]:
        """获取指定版本的特征集合。

        Args:
            version: 特征版本，默认为'v1'

        Returns:
            特征集合字典
        """
        if version not in cls.FEATURE_SETS:
            raise ValueError(f"未知的特征版本: {version}。可用版本: {list(cls.FEATURE_SETS.keys())}")
        return cls.FEATURE_SETS[version]

    @classmethod
    def get_feature_list(cls, version: str = 'v1') -> List[str]:
        """获取指定版本的特征字段列表。

        Args:
            version: 特征版本

        Returns:
            特征字段列表
        """
        feature_set = cls.get_feature_set(version)
        return list(feature_set.keys())

    @classmethod
    def get_feature_metadata(cls, version: str = 'v1') -> List[Dict[str, Any]]:
        """获取指定版本的特征元数据。

        Args:
            version: 特征版本

        Returns:
            特征元数据列表
        """
        feature_set = cls.get_feature_set(version)
        metadata = []
        for name, info in feature_set.items():
            metadata.append({
                'name': name,
                'dtype': info['dtype'],
                'required': info['required'],
                'description': info['description'],
                'default': info.get('default', None),
            })
        return metadata

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame, version: str = 'v1') -> Dict[str, Any]:
        """验证DataFrame是否符合特征模式。

        Args:
            df: 待验证的DataFrame
            version: 特征版本

        Returns:
            验证结果字典
        """
        feature_set = cls.get_feature_set(version)
        results = {
            'valid': True,
            'missing_required': [],
            'type_mismatches': [],
            'suggestions': [],
        }

        # 检查必需字段
        for name, info in feature_set.items():
            if info['required'] and name not in df.columns:
                results['missing_required'].append(name)
                results['valid'] = False

        # 检查字段类型（基本类型检查）
        for name, info in feature_set.items():
            if name in df.columns:
                # 简单的类型检查
                expected_type = info['dtype']
                actual_dtype = str(df[name].dtype)

                # 映射Pandas类型到我们的类型
                type_map = {
                    'string': ['object', 'string'],
                    'int': ['int64', 'int32', 'int16', 'int8'],
                    'float': ['float64', 'float32'],
                    'datetime': ['datetime64[ns]'],
                }

                if expected_type in type_map:
                    expected_pandas_types = type_map[expected_type]
                    if actual_dtype not in expected_pandas_types:
                        # 尝试转换
                        try:
                            if expected_type == 'int':
                                df[name] = pd.to_numeric(df[name], errors='coerce').fillna(info.get('default', 0)).astype('int64')
                            elif expected_type == 'float':
                                df[name] = pd.to_numeric(df[name], errors='coerce').fillna(info.get('default', 0.0)).astype('float64')
                            elif expected_type == 'string':
                                df[name] = df[name].astype('string')
                            results['suggestions'].append(f"字段 {name} 已从 {actual_dtype} 转换为 {expected_type}")
                        except Exception as e:
                            results['type_mismatches'].append(f"{name}: 期望 {expected_type}, 实际 {actual_dtype}, 转换失败: {e}")
                            results['valid'] = False

        return results

    @classmethod
    def apply_defaults(cls, df: pd.DataFrame, version: str = 'v1') -> pd.DataFrame:
        """应用默认值到缺失字段。

        Args:
            df: 输入DataFrame
            version: 特征版本

        Returns:
            应用默认值后的DataFrame
        """
        feature_set = cls.get_feature_set(version)
        df_out = df.copy()

        for name, info in feature_set.items():
            if name not in df_out.columns:
                default = info.get('default')
                if default is not None:
                    if info['dtype'] == 'string':
                        df_out[name] = str(default)
                    elif info['dtype'] == 'int':
                        df_out[name] = int(default)
                    elif info['dtype'] == 'float':
                        df_out[name] = float(default)
                    else:
                        df_out[name] = default
                else:
                    # 没有默认值，根据类型填充
                    if info['dtype'] == 'string':
                        df_out[name] = ''
                    elif info['dtype'] == 'int':
                        df_out[name] = 0
                    elif info['dtype'] == 'float':
                        df_out[name] = 0.0
                    else:
                        df_out[name] = None

        return df_out

    @classmethod
    def ensure_schema(cls, df: pd.DataFrame, version: str = 'v1') -> pd.DataFrame:
        """确保DataFrame符合特征模式。

        包括：应用默认值、类型转换、字段排序。

        Args:
            df: 输入DataFrame
            version: 特征版本

        Returns:
            符合模式的DataFrame
        """
        # 应用默认值
        df_filled = cls.apply_defaults(df, version)

        # 验证并尝试类型转换
        validation = cls.validate_dataframe(df_filled, version)

        if not validation['valid']:
            # 记录警告但继续
            print(f"警告: 特征模式验证发现问题: {validation}")

        # 按特征列表排序字段
        feature_list = cls.get_feature_list(version)
        existing_features = [f for f in feature_list if f in df_filled.columns]
        missing_features = [f for f in feature_list if f not in df_filled.columns]

        if missing_features:
            print(f"警告: 以下特征字段缺失，将使用默认值: {missing_features}")

        # 重新排序字段
        df_ordered = df_filled[existing_features].copy()

        return df_ordered