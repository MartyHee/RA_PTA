#!/usr/bin/env python3
"""
运行网页爬虫脚本。

支持：
1. 从配置文件读取样本URL
2. 手动输入URL
3. 使用mock模式（无需网络）
4. 保存原始数据和解析结果
"""
import sys
import argparse
from pathlib import Path
from typing import List

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.crawler.scheduler import CrawlScheduler
from src.utils.config_loader import load_config
from src.utils.logger import setup_logging


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行抖音网页爬虫')
    parser.add_argument('--urls', nargs='+', help='要抓取的URL列表')
    parser.add_argument('--config', type=Path, default=None, help='配置文件路径')
    parser.add_argument('--source-entry', default='manual_url',
                       choices=['search', 'topic', 'rank', 'manual_url'],
                       help='数据入口类型')
    parser.add_argument('--mock', action='store_true', help='使用mock模式（无需网络）')
    parser.add_argument('--workers', type=int, default=1, help='爬虫工作线程数')
    parser.add_argument('--sample', action='store_true', help='使用配置文件中的样本URL')

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.config)

    # 加载配置
    config = load_config(args.config)

    print("=" * 60)
    print("抖音网页爬虫启动")
    print("=" * 60)

    # 确定要抓取的URL
    urls_to_crawl = []

    if args.urls:
        urls_to_crawl = args.urls
        print(f"使用命令行参数中的URL: {len(urls_to_crawl)}个")

    elif args.sample:
        # 从配置中获取样本URL
        sample_urls = config.get('sources', {}).get('web', {}).get('sample_urls', [])
        urls_to_crawl = sample_urls
        print(f"使用配置文件中的样本URL: {len(urls_to_crawl)}个")

    else:
        # 交互模式
        print("\n请输入要抓取的URL（每行一个，空行结束）:")
        while True:
            try:
                url = input().strip()
                if not url:
                    break
                if url.startswith('http'):
                    urls_to_crawl.append(url)
                else:
                    print(f"跳过无效URL: {url}")
            except EOFError:
                break

        if not urls_to_crawl:
            print("未输入URL，使用配置文件中的样本URL")
            sample_urls = config.get('sources', {}).get('web', {}).get('sample_urls', [])
            urls_to_crawl = sample_urls

    if not urls_to_crawl:
        print("错误: 没有可抓取的URL")
        return 1

    print(f"\n准备抓取 {len(urls_to_crawl)} 个URL")
    print(f"数据入口: {args.source_entry}")
    print(f"模式: {'Mock' if args.mock else '真实抓取'}")
    print(f"工作线程: {args.workers}")
    print("-" * 60)

    # 创建调度器
    scheduler = CrawlScheduler(config_path=args.config, use_mock=args.mock)

    # 添加任务
    for url in urls_to_crawl:
        scheduler.add_task(url, args.source_entry)

    # 启动爬虫
    try:
        if args.mock:
            print("使用mock模式运行...")
            scheduler.mock_run(urls_to_crawl, args.source_entry)
        else:
            print("开始真实抓取...")
            scheduler.start(num_workers=args.workers)

        # 打印摘要
        total_tasks = len(scheduler.completed_tasks) + len(scheduler.failed_tasks)
        if total_tasks > 0:
            success_rate = len(scheduler.completed_tasks) / total_tasks * 100
            print(f"\n抓取完成!")
            print(f"成功: {len(scheduler.completed_tasks)}")
            print(f"失败: {len(scheduler.failed_tasks)}")
            print(f"成功率: {success_rate:.1f}%")

            # 输出文件位置
            data_dir = Path(config.get('settings', {}).get('paths', {}).get('raw_data', './data/raw'))
            if data_dir.exists():
                print(f"\n数据保存位置:")
                for file in data_dir.glob('*'):
                    if file.is_file():
                        size_mb = file.stat().st_size / (1024 * 1024)
                        print(f"  {file.name}: {size_mb:.2f} MB")

        else:
            print("\n未处理任何任务")

        return 0

    except KeyboardInterrupt:
        print("\n用户中断，停止爬虫...")
        scheduler.stop()
        return 130  # SIGINT退出码

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())