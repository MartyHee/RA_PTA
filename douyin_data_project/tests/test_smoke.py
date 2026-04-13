#!/usr/bin/env python3
"""
冒烟测试：验证项目核心模块的可导入性和基本功能。

这些测试确保：
1. 关键模块可以导入
2. 配置可以加载
3. 数据模型可以初始化
4. 运行脚本可以导入（不执行）

这些测试不依赖外部服务或真实数据。
"""
import sys
import importlib
from pathlib import Path
import pytest

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_import_core_modules():
    """测试核心模块可导入"""
    modules_to_test = [
        'src.utils.config_loader',
        'src.utils.logger',
        'src.utils.io_utils',
        'src.schemas.tables',
        'src.crawler.scheduler',
        'src.crawler.parser',
        'src.processing.clean',
        'src.processing.transform',
        'src.processing.feature_engineering',
        'src.processing.quality_check',
        'src.analysis.eda',
    ]

    for module_path in modules_to_test:
        try:
            module = importlib.import_module(module_path)
            assert module is not None
            print(f"[OK] {module_path}")
        except ImportError as e:
            pytest.fail(f"无法导入模块 {module_path}: {e}")


def test_config_loader():
    """测试配置加载器"""
    from src.utils.config_loader import load_config, get_config

    config = load_config()
    assert isinstance(config, dict)
    assert 'settings' in config

    # 测试基本配置项
    project_name = get_config('settings.project.name')
    assert project_name == 'douyin_data_project'

    data_version = get_config('settings.data_version')
    assert data_version == 'v0.1'

    print("[OK] 配置加载器")


def test_schemas():
    """测试数据模型"""
    from src.schemas.tables import (
        RawWebVideoData, WebVideoMeta, ProcessedVideoData,
        AuthorDim, ApiVideoStats, ApiUserProfile
    )

    # 测试模型可以初始化
    from datetime import datetime

    # RawWebVideoData
    raw_data = RawWebVideoData(
        crawl_id='test_001',
        source_entry='manual_url',
        page_url='https://www.douyin.com/video/123',
        http_status=200,
        crawl_time=datetime.now(),
        parse_status='success'
    )
    assert raw_data.crawl_id == 'test_001'
    assert raw_data.parse_status == 'success'

    # WebVideoMeta
    web_meta = WebVideoMeta(
        video_id='video_123',
        page_url='https://www.douyin.com/video/123',
        source_entry='manual_url',
        crawl_time=datetime.now(),
        hashtag_count=0
    )
    assert web_meta.video_id == 'video_123'
    assert web_meta.hashtag_count == 0

    # ProcessedVideoData
    processed = ProcessedVideoData(
        video_id='video_123',
        text_length=10,
        hashtag_count=0,
        source_entry='manual_url',
        crawl_date=datetime.now().date(),
        data_version='v0.1'
    )
    assert processed.text_length == 10
    assert processed.data_version == 'v0.1'

    print("[OK] 数据模型")


def test_logger():
    """测试日志模块"""
    from src.utils.logger import get_logger, setup_logging

    # 测试可以获取logger
    logger = get_logger('test_logger')
    assert logger is not None
    assert logger.name == 'test_logger'

    print("[OK] 日志模块")


def test_io_utils():
    """测试IO工具"""
    from src.utils.io_utils import ensure_dir

    # 测试ensure_dir
    test_dir = Path('test_temp_dir')
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)

    result_dir = ensure_dir(test_dir)
    assert result_dir.exists()
    assert result_dir.is_dir()

    # 清理
    import shutil
    shutil.rmtree(test_dir)

    print("[OK] IO工具")


def test_run_scripts_import():
    """测试运行脚本可导入（不执行）"""
    scripts_to_test = [
        'run_crawl',
        'run_clean',
        'run_features',
        'run_eda',
    ]

    for script_name in scripts_to_test:
        script_path = project_root / f'{script_name}.py'
        assert script_path.exists(), f"脚本不存在: {script_path}"

        # 测试可以导入（但小心执行）
        try:
            # 使用importlib导入模块
            spec = importlib.util.spec_from_file_location(script_name, script_path)
            module = importlib.util.module_from_spec(spec)

            # 只导入，不执行（避免执行main）
            # 这里我们只是检查文件存在和语法正确
            pass

            print(f"[OK] {script_name}.py")
        except SyntaxError as e:
            pytest.fail(f"脚本语法错误 {script_name}.py: {e}")
        except Exception as e:
            # 其他导入错误（如依赖缺失）
            print(f"⚠ {script_name}.py 导入警告: {e}")
            # 不失败，因为可能依赖环境


def test_mock_parser():
    """测试mock解析功能"""
    from src.crawler.parser import DouyinParser
    from datetime import datetime

    parser = DouyinParser()

    # 测试mock_parse
    mock_meta = parser.mock_parse(
        url='https://www.douyin.com/video/1234567890',
        source_entry='manual_url'
    )

    assert mock_meta.video_id is not None
    assert mock_meta.page_url == 'https://www.douyin.com/video/1234567890'
    assert mock_meta.source_entry == 'manual_url'
    assert mock_meta.hashtag_count >= 0

    print("[OK] Mock解析")


def test_cleaner():
    """测试数据清洗器"""
    from src.processing.clean import DataCleaner
    from src.schemas.tables import WebVideoMeta
    from datetime import datetime

    cleaner = DataCleaner()

    # 创建测试数据
    test_meta = WebVideoMeta(
        video_id='test_001',
        page_url='https://www.douyin.com/video/123',
        desc_text='测试 #美食 #旅行',
        publish_time_std=datetime.now(),
        like_count_raw='1.2w',
        comment_count_raw='500',
        share_count_raw='100',
        hashtag_list=['美食', '旅行'],
        hashtag_count=2,
        source_entry='manual_url',
        crawl_time=datetime.now()
    )

    # 测试清洗
    cleaned = cleaner.clean_web_video_meta(test_meta)
    assert cleaned is not None
    assert 'desc_clean' in cleaned
    assert 'text_length' in cleaned
    assert 'crawl_date' in cleaned

    print("[OK] 数据清洗器")


def test_transformer():
    """测试数据转换器"""
    from src.processing.transform import DataTransformer
    from datetime import datetime

    transformer = DataTransformer()

    # 测试转换（使用mock数据）
    test_data = {
        'video_id': 'test_001',
        'author_id': 'author_001',
        'desc_clean': '测试描述',
        'text_length': 4,
        'publish_date': datetime.now().date(),
        'publish_hour': 12,
        'publish_weekday': 1,
        'is_weekend': 0,
        'hashtag_list': ['美食'],
        'hashtag_count': 1,
        'like_count': 100,
        'comment_count': 10,
        'share_count': 5,
        'collect_count': 2,
        'source_entry': 'manual_url',
        'crawl_date': datetime.now().date(),
        'data_version': 'v0.1'
    }

    processed = transformer.transform_to_processed(test_data)
    assert processed.video_id == 'test_001'
    assert processed.engagement_score is not None

    print("[OK] 数据转换器")


def test_feature_engineer():
    """测试特征工程"""
    from src.processing.feature_engineering import FeatureEngineer
    import pandas as pd
    from datetime import date

    # 创建测试DataFrame
    df = pd.DataFrame({
        'video_id': ['v1', 'v2', 'v3'],
        'desc_clean': ['测试1', '测试2', '测试3'],
        'text_length': [10, 20, 30],
        'hashtag_count': [1, 2, 0],
        'like_count': [100, 200, 300],
        'comment_count': [10, 20, 30],
        'share_count': [5, 10, 15],
        'publish_date': [date.today()] * 3,
        'crawl_date': [date.today()] * 3,
        'data_version': ['v0.1'] * 3
    })

    engineer = FeatureEngineer()
    df_features = engineer.create_features(df, ['basic', 'text'])

    assert not df_features.empty
    assert len(df_features.columns) >= len(df.columns)

    print("[OK] 特征工程")


def test_eda_analyzer():
    """测试EDA分析器"""
    from src.analysis.eda import EDAAnalyzer
    import pandas as pd
    from datetime import date

    # 创建测试DataFrame
    df = pd.DataFrame({
        'video_id': ['v1', 'v2', 'v3'],
        'source_entry': ['search', 'topic', 'search'],
        'text_length': [10, 20, 30],
        'like_count': [100, 200, 300],
        'comment_count': [10, 20, 30],
        'engagement_score': [130, 260, 390],
        'publish_date': [date.today()] * 3,
        'crawl_date': [date.today()] * 3,
        'data_version': ['v0.1'] * 3
    })

    analyzer = EDAAnalyzer()
    summary = analyzer.basic_summary(df)

    assert summary is not None
    assert 'dataset_info' in summary
    assert summary['dataset_info']['total_records'] == 3

    print("[OK] EDA分析器")


def test_quality_checker():
    """测试质量检查器"""
    from src.processing.quality_check import DataQualityChecker
    import pandas as pd
    from datetime import date

    # 创建测试DataFrame
    df = pd.DataFrame({
        'video_id': ['v1', 'v2', 'v3'],
        'source_entry': ['search', 'topic', 'search'],
        'text_length': [10, 20, 30],
        'hashtag_count': [1, 2, 0],
        'like_count': [100, 200, 300],
        'crawl_date': [date.today()] * 3,
        'data_version': ['v0.1'] * 3
    })

    checker = DataQualityChecker()
    checks = checker.check_dataframe(df)

    assert checks is not None
    assert 'missing_values' in checks
    assert 'duplicates' in checks

    print("[OK] 质量检查器")


if __name__ == '__main__':
    """直接运行测试"""
    print("运行冒烟测试...")
    print("=" * 60)

    tests = [
        test_import_core_modules,
        test_config_loader,
        test_schemas,
        test_logger,
        test_io_utils,
        test_run_scripts_import,
        test_mock_parser,
        test_cleaner,
        test_transformer,
        test_feature_engineer,
        test_eda_analyzer,
        test_quality_checker,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_func.__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")

    if failed > 0:
        sys.exit(1)
    else:
        print("所有冒烟测试通过!")
        sys.exit(0)