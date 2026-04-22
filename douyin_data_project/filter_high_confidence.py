#!/usr/bin/env python
"""
独立的高置信度样本筛选脚本。
基于现有 interim CSV 文件，筛选出高置信度样本并保存到 processed 目录。
复用 scheduler.py 中的筛选逻辑。
"""

import csv
import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_video_id_from_url(url: str) -> Optional[str]:
    """从抖音URL中提取视频ID。

    复用 scheduler.py 中的正则表达式模式。

    Args:
        url: 抖音视频URL

    Returns:
        提取的视频ID，如果未找到则返回None
    """
    patterns = [
        r'/video/([^/?]+)',
        r'video/([^/?]+)',
        r'item_id=([^&]+)',
        r'id=([^&]+)',
        r'modal_id=([^&]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def filter_high_confidence_samples(
    input_csv: Path,
    output_dir: Path,
    output_filename: Optional[str] = None
) -> Tuple[List[Dict], List[Dict], Dict]:
    """筛选高置信度样本并保存到输出目录。

    筛选条件（严格复用开发日志中的规则）：
    1. match_type = exact
    2. confidence = high
    3. video_id 与 page_url 中提取的目标ID一致

    Args:
        input_csv: 输入CSV文件路径
        output_dir: 输出目录路径
        output_filename: 输出文件名（可选），如果未提供则基于输入文件名生成

    Returns:
        tuple: (high_confidence_records, all_records, stats_dict)
    """
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取所有记录
    records = []
    try:
        with open(input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
    except Exception as e:
        logger.error(f"读取CSV文件失败: {e}")
        raise

    if not records:
        logger.warning("CSV文件中没有记录")
        return [], [], {}

    logger.info(f"总记录数: {len(records)}")

    # 筛选高置信度样本
    high_confidence_records = []
    for row in records:
        # 检查 match_type 和 confidence
        match_type = row.get('match_type')
        confidence = row.get('confidence')

        if match_type != 'exact' or confidence != 'high':
            continue

        # 检查 video_id 与 page_url 的一致性
        video_id = row.get('video_id')
        page_url = row.get('page_url')

        if not video_id or not page_url:
            continue

        target_id = extract_video_id_from_url(page_url)
        if target_id and video_id == target_id:
            high_confidence_records.append(row)
        else:
            # video_id 可能已经是目标ID（从URL提取的）
            # 如果提取失败，仍然接受 video_id 非空的情况
            # 但根据开发日志中的规则，要求一致，所以这里跳过
            logger.debug(f"视频ID不匹配: video_id={video_id}, target_id={target_id}")

    logger.info(f"高置信度样本数: {len(high_confidence_records)}")

    # 保存高置信度样本
    if high_confidence_records:
        if output_filename is None:
            # 基于输入文件名生成输出文件名
            input_stem = input_csv.stem
            # 如果输入文件名以 "real_" 开头，去掉前缀（符合项目命名约定）
            if input_stem.startswith("real_"):
                input_stem = input_stem[5:]  # 去掉 "real_"
            # 如果输入文件名包含 "web_video_meta"，替换为 "high_confidence_web_video_meta"
            if "web_video_meta" in input_stem:
                output_stem = input_stem.replace("web_video_meta", "high_confidence_web_video_meta")
            else:
                output_stem = f"high_confidence_{input_stem}"
            output_filename = f"{output_stem}.csv"

        output_path = output_dir / output_filename

        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=high_confidence_records[0].keys())
            writer.writeheader()
            writer.writerows(high_confidence_records)

        logger.info(f"高置信度样本已保存到: {output_path}")
    else:
        logger.info("没有高置信度样本可保存")
        output_path = None

    # 计算统计信息
    stats = calculate_statistics(records, high_confidence_records)

    return high_confidence_records, records, stats


def calculate_statistics(
    all_records: List[Dict],
    high_confidence_records: List[Dict]
) -> Dict:
    """计算统计信息。

    复用 scheduler.py 中的统计逻辑。

    Args:
        all_records: 所有记录
        high_confidence_records: 高置信度记录

    Returns:
        包含统计信息的字典
    """
    total_records = len(all_records)
    high_confidence_count = len(high_confidence_records)

    # 初始化计数器
    match_type_counts = {'exact': 0, 'partial': 0, 'none': 0, 'unknown': 0}
    confidence_counts = {'high': 0, 'medium': 0, 'low': 0, 'unknown': 0}
    video_id_consistent = 0
    video_id_mismatch = 0
    video_id_short_or_abnormal = 0

    for row in all_records:
        # 统计 match_type
        match_type = row.get('match_type')
        if match_type in match_type_counts:
            match_type_counts[match_type] += 1
        else:
            match_type_counts['unknown'] += 1

        # 统计 confidence
        confidence = row.get('confidence')
        if confidence in confidence_counts:
            confidence_counts[confidence] += 1
        else:
            confidence_counts['unknown'] += 1

        # 检查 video_id 与 page_url 的一致性
        video_id = row.get('video_id')
        page_url = row.get('page_url')
        if video_id and page_url:
            target_id = extract_video_id_from_url(page_url)
            if target_id:
                if video_id == target_id:
                    video_id_consistent += 1
                else:
                    video_id_mismatch += 1

        # 检测短/异常 video_id
        if video_id:
            # 检查 video_id 是否异常短（<= 3字符）且为数字
            if len(video_id) <= 3 and video_id.isdigit():
                video_id_short_or_abnormal += 1
            # 检查 video_id 是否异常短（<= 3字符）
            elif len(video_id) <= 3:
                video_id_short_or_abnormal += 1

    # 计算高置信度样本占比
    high_confidence_ratio = (high_confidence_count / total_records * 100) if total_records > 0 else 0

    # 计算 video_id 一致的比例
    video_id_consistent_ratio = (video_id_consistent / total_records * 100) if total_records > 0 else 0

    # 构建统计字典
    stats = {
        'total_records': total_records,
        'high_confidence_count': high_confidence_count,
        'high_confidence_ratio': high_confidence_ratio,
        'filtered_out_count': total_records - high_confidence_count,
        'match_type_counts': match_type_counts,
        'confidence_counts': confidence_counts,
        'video_id_consistent': video_id_consistent,
        'video_id_consistent_ratio': video_id_consistent_ratio,
        'video_id_mismatch': video_id_mismatch,
        'video_id_short_or_abnormal': video_id_short_or_abnormal,
    }

    return stats


def print_statistics(stats: Dict):
    """打印统计信息。

    Args:
        stats: 统计信息字典
    """
    print("=" * 80)
    print("高置信度样本筛选统计")
    print("=" * 80)
    print(f"输入总记录数: {stats['total_records']}")
    print(f"输出高置信度样本数: {stats['high_confidence_count']}")
    print(f"被过滤掉的记录数: {stats['filtered_out_count']}")
    print(f"高置信度样本占比: {stats['high_confidence_ratio']:.2f}%")
    print()

    print("Match Type 分布:")
    for mt in ['exact', 'partial', 'none', 'unknown']:
        count = stats['match_type_counts'][mt]
        percent = (count / stats['total_records'] * 100) if stats['total_records'] > 0 else 0
        print(f"  {mt}: {count} ({percent:.1f}%)")
    print()

    print("Confidence 分布:")
    for conf in ['high', 'medium', 'low', 'unknown']:
        count = stats['confidence_counts'][conf]
        percent = (count / stats['total_records'] * 100) if stats['total_records'] > 0 else 0
        print(f"  {conf}: {count} ({percent:.1f}%)")
    print()

    print(f"video_id 与 page_url 目标ID一致: {stats['video_id_consistent']} ({stats['video_id_consistent_ratio']:.1f}%)")
    print(f"video_id 不匹配: {stats['video_id_mismatch']}")
    print(f"短/异常 video_id (<=3字符): {stats['video_id_short_or_abnormal']}")
    print("=" * 80)


def print_field_non_null_stats(records: List[Dict]):
    """打印关键字段的非空率统计。

    Args:
        records: 记录列表
    """
    if not records:
        print("没有记录可统计字段非空率")
        return

    # 关键字段列表
    key_fields = [
        'video_id', 'page_url', 'author_id', 'author_name',
        'author_profile_url', 'desc_text', 'publish_time_raw',
        'like_count_raw', 'comment_count_raw', 'share_count_raw',
        'collect_count', 'hashtag_list', 'cover_url', 'music_name',
        'duration_sec', 'match_type', 'confidence'
    ]

    total = len(records)
    print("\n关键字段非空率统计:")
    print("-" * 60)
    print(f"{'字段':<25} {'非空数':<10} {'非空率':<10}")
    print("-" * 60)

    for field in key_fields:
        non_null_count = sum(1 for row in records if row.get(field) not in [None, '', '{}', '[]'])
        non_null_rate = (non_null_count / total * 100) if total > 0 else 0
        print(f"{field:<25} {non_null_count:<10} {non_null_rate:.1f}%")

    print("-" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='筛选高置信度样本')
    parser.add_argument('--input', required=True, help='输入CSV文件路径')
    parser.add_argument('--output-dir', required=True, help='输出目录路径')
    parser.add_argument('--output-filename', help='输出文件名（可选）')
    parser.add_argument('--verbose', action='store_true', help='详细日志输出')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        sys.exit(1)

    logger.info(f"输入文件: {input_path}")
    logger.info(f"输出目录: {output_dir}")

    # 执行筛选
    try:
        high_confidence_records, all_records, stats = filter_high_confidence_samples(
            input_path, output_dir, args.output_filename
        )

        # 打印统计信息
        print_statistics(stats)

        # 打印关键字段非空率统计
        print_field_non_null_stats(all_records)

        # 如果有高置信度样本，也打印其字段非空率
        if high_confidence_records:
            print("\n高置信度样本关键字段非空率统计:")
            print_field_non_null_stats(high_confidence_records)

        logger.info("筛选完成")

    except Exception as e:
        logger.error(f"筛选过程中出错: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()