#!/usr/bin/env python3
"""
批量 URL 发现器：从抖音精选页/分类页自动收集候选视频 URL。

输出为可供现有主抓取流程使用的 `/video/<id>` 列表。

使用浏览器模式（Playwright）滚动页面，发现视频链接。
"""

import sys
import argparse
import re
import time
import json
import csv
from pathlib import Path
from typing import List, Set, Dict, Optional
from datetime import datetime
from urllib.parse import urlparse, urljoin, parse_qs

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logging, get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)

# 默认来源页面列表
DEFAULT_SOURCES = [
    "https://www.douyin.com/jingxuan",
    "https://www.douyin.com/jingxuan/food",
    "https://www.douyin.com/jingxuan/acg",
    "https://www.douyin.com/jingxuan/travel",
    "https://www.douyin.com/jingxuan/fashion",
]

def extract_video_id_from_url(url: str) -> Optional[str]:
    """
    从URL中提取视频ID。

    支持格式：
    - /video/{video_id}
    - /jingxuan?...modal_id={video_id}
    - item_id={video_id}
    - id={video_id}
    """
    patterns = [
        r'/video/([^/?]+)',          # /video/{video_id}
        r'video/([^/?]+)',           # video/{video_id} (no leading slash)
        r'item_id=([^&]+)',          # item_id={video_id}
        r'id=([^&]+)',               # id={video_id}
        r'modal_id=([^&]+)',         # modal_id={video_id}
    ]
    for pattern in patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

def normalize_video_url(video_id: str) -> str:
    """将视频ID规范化为标准URL格式。"""
    return f"https://www.douyin.com/video/{video_id}"

def discover_urls_from_page(
    browser,
    page_url: str,
    max_scrolls: int = 5,
    scroll_delay: float = 2.0,
    headless: bool = True
) -> Set[str]:
    """
    从单个页面发现视频URL。

    参数:
        browser: Playwright browser 实例
        page_url: 要抓取的页面URL
        max_scrolls: 最大滚动次数
        scroll_delay: 每次滚动后的延迟（秒）
        headless: 是否无头模式

    返回:
        发现的视频URL集合（标准化格式）
    """
    logger.info(f"开始从页面发现URL: {page_url}")

    # 创建新页面
    context = browser.new_context(
        viewport={'width': 1920, 'height': 1080},
        user_agent=(
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        ),
        ignore_https_errors=True
    )
    page = context.new_page()

    discovered_urls = set()

    try:
        # 导航到页面
        logger.info(f"导航到: {page_url}")
        response = page.goto(page_url, timeout=60000)
        if not response or response.status >= 400:
            logger.warning(f"页面加载失败: {page_url}, 状态: {response.status if response else '无响应'}")
            return discovered_urls

        # 等待初始加载
        page.wait_for_load_state('domcontentloaded', timeout=30000)
        time.sleep(3)  # 额外等待JavaScript执行

        # 尝试等待视频卡片出现（抖音精选页常见选择器）
        try:
            page.wait_for_selector('a[href*="/video/"]', timeout=15000)
            logger.info("检测到视频链接，页面加载成功")
        except:
            logger.warning("未找到视频链接，继续处理")

        # 额外等待确保内容加载
        time.sleep(2.0)

        # 记录页面HTML长度用于调试
        html_length = len(page.content())
        logger.info(f"页面HTML长度: {html_length} 字符")

        # 滚动页面以加载更多内容
        for scroll_idx in range(max_scrolls):
            logger.info(f"滚动 {scroll_idx + 1}/{max_scrolls}")

            # 滚动到底部
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(scroll_delay)

            # 等待可能的懒加载
            try:
                page.wait_for_load_state('networkidle', timeout=5000)
                logger.debug("networkidle 状态达到")
            except Exception as e:
                logger.debug(f"等待 networkidle 超时，继续: {e}")
            time.sleep(1.0)  # 额外等待确保内容稳定

            # 获取当前页面所有链接
            links = page.eval_on_selector_all(
                'a[href]',
                'elements => elements.map(el => el.href)'
            )

            # 提取视频ID
            for link in links:
                if not link:
                    continue

                video_id = extract_video_id_from_url(link)
                if video_id:
                    normalized_url = normalize_video_url(video_id)
                    discovered_urls.add(normalized_url)

            # 打印当前进度
            logger.info(f"已发现 {len(discovered_urls)} 个唯一视频URL")

            # 检查是否还有更多内容可加载
            current_height = page.evaluate("document.body.scrollHeight")
            scroll_position = page.evaluate("window.scrollY + window.innerHeight")

            if scroll_position >= current_height - 100:  # 接近底部
                logger.info(f"页面已滚动到底部")
                break

        # 也检查页面HTML中可能嵌入的视频ID（如data-video-id属性）
        html_content = page.content()

        # 在HTML中查找视频ID模式 - 扩展模式
        html_patterns = [
            r'video[_-]id["\']?\s*[:=]\s*["\']?([0-9]+)["\']?',
            r'data[_-]video[_-]id=["\']([0-9]+)["\']',
            r'data[_-]aweme[_-]id=["\']([0-9]+)["\']',
            r'data[_-]item[_-]id=["\']([0-9]+)["\']',
            r'aweme[_-]id["\']?\s*[:=]\s*["\']?([0-9]+)["\']?',
            r'"video_id"\s*:\s*["\']?([0-9]+)["\']?',
            r'"aweme_id"\s*:\s*["\']?([0-9]+)["\']?',
            r'"id"\s*:\s*["\']?([0-9]+)["\']?\s*,\s*[^}]+"video"',
            r'"id"\s*:\s*["\']?([0-9]+)["\']?\s*,\s*[^}]+"aweme"',
            r'video/([0-9]+)',  # 直接匹配 video/数字
            r'/video/([0-9]+)', # 匹配 /video/数字
        ]
        for pattern in html_patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            for vid in matches:
                if len(vid) > 5:  # 简单过滤
                    normalized_url = normalize_video_url(vid)
                    discovered_urls.add(normalized_url)
                    logger.debug(f"从HTML模式匹配到视频ID: {vid}")

        # 额外：通过Playwright选择器查找具有data-video-id属性的元素
        try:
            video_elements = page.query_selector_all('[data-video-id], [data-aweme-id], [data-item-id]')
            for elem in video_elements:
                for attr in ['data-video-id', 'data-aweme-id', 'data-item-id']:
                    vid = elem.get_attribute(attr)
                    if vid and len(vid) > 5:
                        normalized_url = normalize_video_url(vid)
                        discovered_urls.add(normalized_url)
                        logger.debug(f"从属性 {attr} 提取到视频ID: {vid}")
        except Exception as e:
            logger.debug(f"通过选择器提取视频ID时出错: {e}")

        logger.info(f"页面 {page_url} 发现完成，共 {len(discovered_urls)} 个唯一视频URL")

    except Exception as e:
        logger.error(f"从页面 {page_url} 发现URL时出错: {e}")
    finally:
        # 关闭页面和上下文
        try:
            page.close()
            context.close()
        except:
            pass

    return discovered_urls

def save_urls_to_file(
    urls: List[str],
    output_dir: Path,
    base_filename: str,
    source_info: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Path]:
    """
    保存URL到文件。

    参数:
        urls: URL列表
        output_dir: 输出目录
        base_filename: 基础文件名（不带扩展名）
        source_info: 可选，映射视频ID到来源页面列表

    返回:
        字典，包含生成的文件路径
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 保存为文本文件（每行一个URL）
    txt_path = output_dir / f"{base_filename}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        for url in urls:
            f.write(f"{url}\n")
    logger.info(f"保存 {len(urls)} 个URL到文本文件: {txt_path}")

    # 2. 可选：保存为带来源信息的CSV
    csv_path = output_dir / f"{base_filename}.csv"
    if source_info:
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['video_id', 'page_url', 'source_page', 'discover_time'])

            for url in urls:
                video_id = extract_video_id_from_url(url)
                if not video_id:
                    continue

                sources = source_info.get(video_id, [])
                for source_page in sources:
                    writer.writerow([
                        video_id,
                        url,
                        source_page,
                        datetime.now().isoformat()
                    ])
        logger.info(f"保存带来源信息的CSV: {csv_path}")

    return {
        'txt': txt_path,
        'csv': csv_path if source_info else None
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='抖音批量URL发现器')
    parser.add_argument('--sources', nargs='+', default=DEFAULT_SOURCES,
                       help='来源页面URL列表，用空格分隔')
    parser.add_argument('--max-scrolls', type=int, default=5,
                       help='每个页面的最大滚动次数（默认: 5）')
    parser.add_argument('--scroll-delay', type=float, default=2.0,
                       help='每次滚动后的延迟（秒，默认: 2.0）')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='输出目录（默认: configs/）')
    parser.add_argument('--output-filename', type=str, default=None,
                       help='输出文件名（不带扩展名），默认使用时间戳')
    parser.add_argument('--headless', action='store_true', default=True,
                       help='使用无头浏览器模式（默认: True）')
    parser.add_argument('--no-headless', dest='headless', action='store_false',
                       help='禁用无头模式（显示浏览器窗口）')
    parser.add_argument('--save-csv', action='store_true', default=False,
                       help='同时保存带来源信息的CSV文件')
    parser.add_argument('--config', type=Path, default=None,
                       help='配置文件路径')

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.config)

    # 加载配置
    config = load_config(args.config)

    # 生成本次运行的 run_id (时间戳)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 确定输出目录
    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"使用用户指定的输出目录: {output_dir}")
    else:
        configs_dir = Path(config.get('settings', {}).get('paths', {}).get('configs', './configs'))
        # 在 configs 下创建 url_discovery/<run_id> 子目录
        output_dir = configs_dir / "url_discovery" / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"使用自动生成的输出目录: {output_dir}")

    # 确定输出文件名
    if args.output_filename:
        base_filename = args.output_filename
    else:
        # 文件名简化，因为时间戳已经在目录名中
        base_filename = "batch_urls"

    logger.info("=" * 60)
    logger.info("抖音批量URL发现器启动")
    logger.info(f"本次运行 ID: {run_id}")
    logger.info("=" * 60)
    logger.info(f"来源页面: {len(args.sources)} 个")
    for i, source in enumerate(args.sources, 1):
        logger.info(f"  {i}. {source}")
    logger.info(f"最大滚动次数: {args.max_scrolls}")
    logger.info(f"滚动延迟: {args.scroll_delay}秒")
    logger.info(f"无头模式: {args.headless}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"输出文件名: {base_filename}")
    logger.info(f"保存CSV: {args.save_csv}")
    logger.info("-" * 60)

    # 初始化Playwright
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.error("Playwright未安装。请安装: pip install playwright && playwright install")
        return 1

    all_urls = set()
    source_info = {}  # video_id -> [source_pages]

    with sync_playwright() as playwright:
        # 启动浏览器
        browser = playwright.chromium.launch(
            headless=args.headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
            ]
        )

        try:
            # 遍历所有来源页面
            for source_idx, source_url in enumerate(args.sources, 1):
                logger.info(f"处理来源页面 ({source_idx}/{len(args.sources)}): {source_url}")

                # 从页面发现URL
                page_urls = discover_urls_from_page(
                    browser=browser,
                    page_url=source_url,
                    max_scrolls=args.max_scrolls,
                    scroll_delay=args.scroll_delay,
                    headless=args.headless
                )

                # 更新来源信息
                for url in page_urls:
                    video_id = extract_video_id_from_url(url)
                    if video_id:
                        if video_id not in source_info:
                            source_info[video_id] = []
                        if source_url not in source_info[video_id]:
                            source_info[video_id].append(source_url)

                # 添加到总集合
                all_urls.update(page_urls)

                logger.info(f"当前累计唯一URL数: {len(all_urls)}")

                # 页面间延迟，避免请求过于频繁
                if source_idx < len(args.sources):
                    delay = 3.0
                    logger.info(f"等待 {delay} 秒后处理下一个页面...")
                    time.sleep(delay)

        finally:
            # 关闭浏览器
            browser.close()

    # 转换为列表并排序（按视频ID）
    sorted_urls = sorted(list(all_urls))

    # 保存结果
    logger.info(f"发现完成，共 {len(sorted_urls)} 个唯一视频URL")

    # 保存文件
    files = save_urls_to_file(
        urls=sorted_urls,
        output_dir=output_dir,
        base_filename=base_filename,
        source_info=source_info if args.save_csv else None
    )

    # 输出统计信息
    logger.info("=" * 60)
    logger.info("URL发现统计")
    logger.info("=" * 60)
    logger.info(f"来源页面数量: {len(args.sources)}")
    logger.info(f"总发现链接数（去重前）: N/A（每个页面独立计数）")
    logger.info(f"去重后URL数: {len(sorted_urls)}")
    logger.info(f"最终写入文件:")
    logger.info(f"  - 文本文件: {files['txt']}")
    if files['csv']:
        logger.info(f"  - CSV文件: {files['csv']}")

    # 输出样例
    logger.info(f"样例前10条URL:")
    for i, url in enumerate(sorted_urls[:10], 1):
        logger.info(f"  {i}. {url}")

    # 检查URL格式是否符合规范
    invalid_urls = []
    for url in sorted_urls:
        if not url.startswith('https://www.douyin.com/video/'):
            invalid_urls.append(url)

    if invalid_urls:
        logger.warning(f"发现 {len(invalid_urls)} 个不符合规范格式的URL:")
        for url in invalid_urls[:5]:
            logger.warning(f"  {url}")
    else:
        logger.info("所有URL均符合规范格式: /video/<id>")

    logger.info("完成!")

    return 0

if __name__ == '__main__':
    sys.exit(main())