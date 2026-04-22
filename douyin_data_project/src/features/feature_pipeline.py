"""
特征Pipeline主类。

协调数据加载、转换、特征工程和存储。
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import pandas as pd
import numpy as np

from ..utils.io_utils import read_csv, ensure_dir
from ..utils.logger import get_logger
from ..utils.time_utils import parse_douyin_time, normalize_count_string
from ..utils.text_utils import clean_text

from .feature_schema import FeatureSchema
from .feature_storage import FeatureStorage
from .feature_registry import get_registry

logger = get_logger(__name__)


class FeaturePipeline:
    """特征Pipeline主类。"""

    def __init__(
        self,
        feature_version: str = 'v1',
        output_dir: Union[str, Path] = "data/features",
        verbose: bool = True
    ):
        """初始化特征Pipeline。

        Args:
            feature_version: 特征版本
            output_dir: 输出目录
            verbose: 是否输出详细信息
        """
        self.feature_version = feature_version
        self.output_dir = Path(output_dir)
        self.verbose = verbose

        self.feature_schema = FeatureSchema()
        self.feature_storage = FeatureStorage(output_dir)
        self.feature_registry = get_registry()

        ensure_dir(self.output_dir)

    def load_web_video_meta(self, input_path: Union[str, Path]) -> pd.DataFrame:
        """加载web_video_meta CSV文件。

        Args:
            input_path: 输入CSV文件路径

        Returns:
            web_video_meta DataFrame
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        logger.info(f"加载数据: {input_path}")

        # 读取CSV文件
        try:
            df = pd.read_csv(input_path, low_memory=False)
        except Exception as e:
            logger.error(f"读取CSV文件失败: {e}")
            # 尝试使用项目中的read_csv函数
            df = read_csv(input_path)

        logger.info(f"加载 {len(df)} 条记录，{len(df.columns)} 个字段")

        if self.verbose:
            print(f"加载字段: {list(df.columns)}")
            print(f"数据形状: {df.shape}")

        return df

    def transform_web_video_meta(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换web_video_meta数据为特征数据。

        Args:
            df: web_video_meta DataFrame

        Returns:
            转换后的特征DataFrame
        """
        if self.verbose:
            print("开始转换web_video_meta数据...")

        df_transformed = df.copy()

        # 1. 处理时间字段
        df_transformed = self._process_time_fields(df_transformed)

        # 2. 处理计数字段（like_count_raw等）
        df_transformed = self._process_count_fields(df_transformed)

        # 3. 处理文本字段
        df_transformed = self._process_text_fields(df_transformed)

        # 4. 处理列表字段（hashtag_list）
        df_transformed = self._process_list_fields(df_transformed)

        # 5. 应用特征模式
        df_features = self.feature_schema.ensure_schema(df_transformed, self.feature_version)

        if self.verbose:
            print(f"转换完成，特征字段: {list(df_features.columns)}")
            print(f"特征数据形状: {df_features.shape}")

        return df_features

    def _process_time_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理时间字段。

        时间特征生成优先级：
        1. 优先使用publish_time_std（标准化时间字符串）
        2. 对于publish_time_std解析失败的行，回退使用publish_time_raw（时间戳）
        3. 两者都失败时使用默认值（-1）

        Args:
            df: 输入DataFrame

        Returns:
            处理后的DataFrame
        """
        if self.verbose:
            print("处理时间字段...")

        # 确保crawl_time是datetime类型
        if 'crawl_time' in df.columns:
            df['crawl_time'] = pd.to_datetime(df['crawl_time'], errors='coerce')
            if self.verbose:
                null_crawl = df['crawl_time'].isna().sum()
                print(f"crawl_time: 总行数={len(df)}, 无效值={null_crawl}")

        # 初始化时间派生特征列，先全部填充默认值
        df['publish_hour'] = -1
        df['publish_weekday'] = -1
        df['is_weekend'] = 0
        df['days_since_publish'] = -1

        # 准备一个datetime列，用于存储解析后的发布时间
        publish_dt = pd.Series([pd.NaT] * len(df), index=df.index)

        # 首先尝试解析publish_time_std（优先级1：标准化时间字符串）
        if 'publish_time_std' in df.columns:
            if self.verbose:
                print("优先解析publish_time_std...")
            # 转换为字符串，然后尝试解析
            time_std_str = df['publish_time_std'].astype(str)
            # 尝试多种解析方式：自动检测、标准格式、带毫秒格式
            for fmt in [None, '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f']:
                try:
                    dt = pd.to_datetime(time_std_str, errors='coerce', format=fmt)
                    # 更新尚未解析的行
                    mask = dt.notna() & publish_dt.isna()
                    publish_dt[mask] = dt[mask]
                    if self.verbose and mask.any():
                        print(f"使用格式 {fmt} 解析了 {mask.sum()} 行")
                except Exception as e:
                    if self.verbose:
                        print(f"格式 {fmt} 解析失败: {e}")

            if self.verbose:
                parsed_by_std = publish_dt.notna().sum()
                print(f"通过publish_time_std解析了 {parsed_by_std} 行")

        # 对于仍未解析的行，尝试使用publish_time_raw（优先级2：时间戳回退）
        if 'publish_time_raw' in df.columns and publish_dt.isna().any():
            if self.verbose:
                print(f"回退解析publish_time_raw，剩余未解析行数: {publish_dt.isna().sum()}")
            time_raw = df['publish_time_raw'].astype(str)
            # 尝试解析时间戳（秒级或毫秒级）
            for idx in df.index[publish_dt.isna()]:
                val = time_raw[idx]
                if pd.isna(val) or val.strip() == '':
                    continue
                # 尝试转换为数字
                try:
                    # 移除非数字字符
                    num_str = ''.join(c for c in val if c.isdigit())
                    if not num_str:
                        continue
                    timestamp = int(num_str)
                    # 判断是秒级还是毫秒级（通常毫秒级长度>=13位）
                    if len(num_str) >= 13:  # 毫秒级
                        timestamp = timestamp / 1000.0
                    # 转换为datetime
                    dt = pd.to_datetime(timestamp, unit='s', errors='coerce')
                    if pd.notna(dt):
                        publish_dt[idx] = dt
                except Exception:
                    pass

            if self.verbose:
                # 记录解析前的已解析数量
                before_parse_count = publish_dt.notna().sum()
                # 解析publish_time_raw后
                after_parse_count = publish_dt.notna().sum()
                parsed_by_raw = after_parse_count - before_parse_count
                print(f"通过publish_time_raw解析了 {parsed_by_raw} 行")

        # 处理publish_time_raw（保留原样）
        if 'publish_time_raw' in df.columns:
            # 确保是字符串类型
            df['publish_time_raw'] = df['publish_time_raw'].astype(str)
        else:
            df['publish_time_raw'] = ''

        # 计算时间派生特征
        valid_mask = publish_dt.notna()
        if valid_mask.any():
            df.loc[valid_mask, 'publish_hour'] = publish_dt[valid_mask].dt.hour
            df.loc[valid_mask, 'publish_weekday'] = publish_dt[valid_mask].dt.weekday  # 0=Monday, 6=Sunday
            df.loc[valid_mask, 'is_weekend'] = (df.loc[valid_mask, 'publish_weekday'] >= 5).astype(int)  # 5=Saturday, 6=Sunday

            # 计算发布天数（相对于抓取时间）
            if 'crawl_time' in df.columns:
                valid_crawl_mask = df['crawl_time'].notna()
                both_valid_mask = valid_mask & valid_crawl_mask
                if both_valid_mask.any():
                    time_diff = (df.loc[both_valid_mask, 'crawl_time'] - publish_dt[both_valid_mask]).dt.days
                    # 处理负值（未来时间）
                    time_diff = time_diff.clip(lower=0)
                    df.loc[both_valid_mask, 'days_since_publish'] = time_diff

        # 确保数据类型为int
        time_int_cols = ['publish_hour', 'publish_weekday', 'is_weekend', 'days_since_publish']
        for col in time_int_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)

        if self.verbose:
            print(f"时间特征统计:")
            for col in time_int_cols:
                if col in df.columns:
                    valid = (df[col] != -1).sum() if col != 'is_weekend' else (df[col] >= 0).sum()
                    default = (df[col] == -1).sum() if col != 'is_weekend' else 0
                    print(f"  {col}: 有效值={valid}, 默认值={default}, 最小值={df[col].min()}, 最大值={df[col].max()}")
            print(f"总有效时间解析: {valid_mask.sum()}/{len(df)}")

        return df

    def _process_count_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理计数字段。

        Args:
            df: 输入DataFrame

        Returns:
            处理后的DataFrame
        """
        # 原始计数字段映射（使用新的派生命名）
        count_fields = {
            'like_count_raw': 'like_count_num',
            'comment_count_raw': 'comment_count_num',
            'share_count_raw': 'share_count_num',
        }

        for raw_field, clean_field in count_fields.items():
            if raw_field in df.columns:
                # 转换计数字符串（如"1.2w"）为数值
                df[clean_field] = df[raw_field].apply(self._parse_count_string)
            else:
                # 字段不存在，创建默认值
                df[clean_field] = 0

        # 处理其他数值字段的缺失值
        numeric_fields = [
            'author_follower_count',
            'author_total_favorited',
            'duration_sec',
            'collect_count',
            'hashtag_count',
        ]

        for field in numeric_fields:
            if field in df.columns:
                # 转换为数值，处理缺失值
                df[field] = pd.to_numeric(df[field], errors='coerce')
                # 填充缺失值
                if field in ['author_follower_count', 'author_total_favorited']:
                    df[field] = df[field].fillna(-1).astype(int)
                else:
                    df[field] = df[field].fillna(0).astype(int)
            else:
                # 字段不存在，创建默认值
                if field in ['author_follower_count', 'author_total_favorited']:
                    df[field] = -1
                else:
                    df[field] = 0

        return df

    def _parse_count_string(self, count_str):
        """解析计数字符串（如"1.2w"、"5k"）为整数。

        Args:
            count_str: 计数字符串

        Returns:
            整数值
        """
        if pd.isna(count_str):
            return 0

        # 如果是数值，直接返回
        if isinstance(count_str, (int, float)):
            return int(count_str)

        # 如果是字符串，尝试解析
        if isinstance(count_str, str):
            count_str = count_str.strip().lower()

            # 空字符串
            if not count_str:
                return 0

            # 尝试直接转换
            try:
                return int(float(count_str))
            except ValueError:
                pass

            # 处理"w"（万）和"k"（千）
            if 'w' in count_str:
                try:
                    num = float(count_str.replace('w', ''))
                    return int(num * 10000)
                except ValueError:
                    pass
            elif 'k' in count_str:
                try:
                    num = float(count_str.replace('k', ''))
                    return int(num * 1000)
                except ValueError:
                    pass

        # 无法解析，返回0
        return 0

    def _process_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理文本字段。

        Args:
            df: 输入DataFrame

        Returns:
            处理后的DataFrame
        """
        # desc_text字段
        if 'desc_text' in df.columns:
            # 复制原始列，用于检查NaN
            desc_series = df['desc_text'].copy()

            # 定义真实文本判断函数
            def is_real_text(val):
                # 检查是否为NaN或None
                if pd.isna(val):
                    return False
                # 转换为字符串
                if not isinstance(val, str):
                    val = str(val)
                # 去除首尾空白
                val_stripped = val.strip()
                # 检查是否为空字符串
                if not val_stripped:
                    return False
                # 检查是否为"nan"（字符串形式的NaN）
                if val_stripped.lower() == 'nan':
                    return False
                # 检查是否为"{}"（可能的占位值）
                if val_stripped == '{}':
                    return False
                # 其他情况视为真实文本
                return True

            # 计算是否有真实文本
            df['has_desc_text'] = desc_series.apply(is_real_text).astype(int)

            # 计算文本长度：对于真实文本，计算去除空白后的长度；否则为0
            def compute_text_length(val):
                if pd.isna(val):
                    return 0
                if not isinstance(val, str):
                    val = str(val)
                val_stripped = val.strip()
                # 检查占位值
                if val_stripped.lower() == 'nan' or val_stripped == '{}':
                    return 0
                return len(val_stripped)

            df['desc_text_length'] = desc_series.apply(compute_text_length)

            # 确保desc_text为字符串类型，NaN替换为空字符串
            df['desc_text'] = desc_series.fillna('').astype(str)
        else:
            # 字段不存在，创建默认值
            df['desc_text'] = ''
            df['desc_text_length'] = 0
            df['has_desc_text'] = 0

        # author_name字段
        if 'author_name' in df.columns:
            df['author_name'] = df['author_name'].astype(str)
        else:
            df['author_name'] = ''

        # author_signature字段（保留但不使用）
        if 'author_signature' in df.columns:
            df['author_signature'] = df['author_signature'].astype(str)

        # author_verification_type字段
        if 'author_verification_type' in df.columns:
            df['author_verification_type'] = df['author_verification_type'].astype(str)
        else:
            df['author_verification_type'] = ''

        return df

    def _process_list_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理列表字段。

        Args:
            df: 输入DataFrame

        Returns:
            处理后的DataFrame
        """
        # hashtag_list字段
        if 'hashtag_list' in df.columns:
            # 确保是字符串类型
            df['hashtag_list'] = df['hashtag_list'].astype(str)
        else:
            df['hashtag_list'] = ''

        # 是否有话题标签：基于hashtag_count > 0
        if 'hashtag_count' in df.columns:
            # hashtag_count已在_process_count_fields中处理为数值
            df['has_hashtag'] = (df['hashtag_count'] > 0).astype(int)
        else:
            # 回退到基于hashtag_list字符串判断
            df['has_hashtag'] = (df['hashtag_list'].str.strip() != '').astype(int)

        return df

    def run(
        self,
        input_path: Union[str, Path],
        run_id: Optional[str] = None,
        feature_version: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        save_sample: bool = True
    ) -> Dict[str, Any]:
        """运行特征Pipeline。

        Args:
            input_path: 输入CSV文件路径
            run_id: 运行ID（如20260421_194709），如果为None则从输入路径提取
            feature_version: 特征版本，如果为None则使用初始化时的版本
            output_dir: 输出目录，如果为None则使用初始化时的目录
            save_sample: 是否保存CSV样本

        Returns:
            运行结果字典
        """
        # 参数处理
        if feature_version is not None:
            self.feature_version = feature_version

        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.feature_storage = FeatureStorage(self.output_dir)

        # 提取run_id（从输入路径或时间戳）
        if run_id is None:
            run_id = self._extract_run_id(input_path)

        logger.info(f"开始特征Pipeline运行")
        logger.info(f"输入文件: {input_path}")
        logger.info(f"运行ID: {run_id}")
        logger.info(f"特征版本: {self.feature_version}")
        logger.info(f"输出目录: {self.output_dir}")

        # 1. 加载数据
        df_raw = self.load_web_video_meta(input_path)

        if df_raw.empty:
            raise ValueError("输入数据为空")

        # 2. 转换数据
        df_features = self.transform_web_video_meta(df_raw)

        if df_features.empty:
            raise ValueError("特征数据为空")

        # 3. 保存特征产物
        metadata = {
            'generation_script': 'FeaturePipeline',
            'pipeline_version': '1.0',
            'input_file_size_mb': Path(input_path).stat().st_size / 1024 / 1024,
        }

        saved_paths = self.feature_storage.save_features(
            df_features=df_features,
            run_id=run_id,
            feature_version=self.feature_version,
            input_data_path=input_path,
            metadata=metadata,
            save_sample=save_sample
        )

        # 4. 生成运行报告
        report = self._generate_run_report(df_features, run_id, saved_paths)

        logger.info(f"特征Pipeline运行完成")
        logger.info(f"生成特征: {len(df_features.columns)} 个")
        logger.info(f"样本数量: {len(df_features)}")
        logger.info(f"输出目录: {self.output_dir / run_id / self.feature_version}")

        return {
            'success': True,
            'run_id': run_id,
            'feature_version': self.feature_version,
            'num_samples': len(df_features),
            'num_features': len(df_features.columns),
            'output_dir': str(self.output_dir / run_id / self.feature_version),
            'saved_paths': saved_paths,
            'report': report,
        }

    def _extract_run_id(self, input_path: Union[str, Path]) -> str:
        """从输入路径提取run_id。

        Args:
            input_path: 输入文件路径

        Returns:
            run_id字符串
        """
        input_path = Path(input_path)
        # 尝试从路径中提取（如"20260421_194709"）
        for part in input_path.parts:
            if part.count('_') == 1 and part.replace('_', '').isdigit() and len(part) == 15:
                return part

        # 如果找不到，使用当前时间戳
        now = datetime.now()
        return now.strftime("%Y%m%d_%H%M%S")

    def _generate_run_report(
        self,
        df_features: pd.DataFrame,
        run_id: str,
        saved_paths: Dict[str, Path]
    ) -> Dict[str, Any]:
        """生成运行报告。

        Args:
            df_features: 特征DataFrame
            run_id: 运行ID
            saved_paths: 保存的文件路径

        Returns:
            运行报告字典
        """
        # 基本统计
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        categorical_cols = df_features.select_dtypes(include=['object', 'string']).columns

        report = {
            'run_id': run_id,
            'feature_version': self.feature_version,
            'timestamp': datetime.now().isoformat(),
            'dataset_stats': {
                'num_samples': len(df_features),
                'num_features': len(df_features.columns),
                'numeric_features': len(numeric_cols),
                'categorical_features': len(categorical_cols),
                'memory_usage_mb': float(df_features.memory_usage(deep=True).sum() / 1024 / 1024),
            },
            'saved_files': {k: str(v) for k, v in saved_paths.items() if v is not None},
            'feature_summary': {
                'numeric_features': list(numeric_cols),
                'categorical_features': list(categorical_cols),
            }
        }

        return report

    def validate_input(self, input_path: Union[str, Path]) -> Dict[str, Any]:
        """验证输入文件。

        Args:
            input_path: 输入文件路径

        Returns:
            验证结果字典
        """
        input_path = Path(input_path)
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'file_info': {},
        }

        # 检查文件是否存在
        if not input_path.exists():
            results['valid'] = False
            results['errors'].append(f"文件不存在: {input_path}")
            return results

        # 文件信息
        results['file_info'] = {
            'path': str(input_path),
            'size_mb': input_path.stat().st_size / 1024 / 1024,
            'extension': input_path.suffix,
        }

        # 检查文件扩展名
        if input_path.suffix.lower() != '.csv':
            results['warnings'].append(f"文件扩展名不是.csv: {input_path.suffix}")

        # 尝试读取文件头
        try:
            df_sample = pd.read_csv(input_path, nrows=5)
            results['file_info']['columns'] = list(df_sample.columns)
            results['file_info']['sample_rows'] = len(df_sample)

            # 检查必需字段
            required_fields = ['video_id', 'page_url', 'crawl_time']
            missing_fields = [f for f in required_fields if f not in df_sample.columns]
            if missing_fields:
                results['warnings'].append(f"缺少建议字段: {missing_fields}")

        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"读取文件失败: {e}")

        return results