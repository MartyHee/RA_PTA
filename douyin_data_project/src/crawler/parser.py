"""
HTML parser for Douyin video pages.

Extracts structured data from HTML responses using BeautifulSoup.
Supports both actual parsing and mock parsing for development.
"""
import json
import re
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging

from bs4 import BeautifulSoup

from ..schemas.tables import WebVideoMeta, RawWebVideoData
from ..utils.config_loader import load_config, get_config
from ..utils.logger import get_logger
from ..utils.text_utils import extract_hashtags, normalize_text

logger = get_logger(__name__)


class DouyinParser:
    """Parser for Douyin video page HTML."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize parser.

        Args:
            config_path: Path to config file. If None, loads default.
        """
        self.config = load_config(config_path) if config_path else load_config()
        self.field_config = get_config('fields.web_video_meta.fields', {})

    def parse_html(self, html: str, url: str, source_entry: str,
                   crawl_time: datetime) -> Dict[str, Any]:
        """Parse HTML and extract video metadata.

        Args:
            html: HTML content.
            url: Page URL.
            source_entry: Source entry type.
            crawl_time: Crawl timestamp.

        Returns:
            Dictionary with extracted fields.
        """
        try:
            soup = BeautifulSoup(html, 'lxml')
        except Exception:
            soup = BeautifulSoup(html, 'html.parser')

        result = {
            'page_url': url,
            'source_entry': source_entry,
            'crawl_time': crawl_time,
            'parse_status': 'success',
            'parse_error_msg': None
        }

        try:
            # Extract from JavaScript state
            script_data = self._extract_script_data(soup)
            if script_data:
                result.update(self._parse_script_data(script_data))

            # Extract from HTML elements
            html_data = self._extract_html_data(soup)
            result.update(html_data)

            # Extract video ID from URL if not found
            if not result.get('video_id'):
                result['video_id'] = self._extract_video_id_from_url(url)

            # Extract hashtags from description
            if result.get('desc_text'):
                hashtags = extract_hashtags(result['desc_text'])
                result['hashtag_list'] = hashtags
                result['hashtag_count'] = len(hashtags)

            # Normalize counts
            result.update(self._normalize_counts(result))

            logger.debug(f"Successfully parsed {url}")
            return result

        except Exception as e:
            logger.error(f"Failed to parse {url}: {e}")
            result['parse_status'] = 'fail'
            result['parse_error_msg'] = str(e)
            return result

    def _extract_script_data(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Extract data from JavaScript state.

        Args:
            soup: BeautifulSoup object.

        Returns:
            Extracted data dict or None.
        """
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string:
                content = script.string
                # Look for common patterns
                patterns = [
                    r'window\.__INITIAL_STATE__\s*=\s*({.*?});',
                    r'\"video\"\s*:\s*({.*?})',
                    r'\"itemInfo\"\s*:\s*({.*?})'
                ]
                for pattern in patterns:
                    match = re.search(pattern, content, re.DOTALL)
                    if match:
                        try:
                            data_str = match.group(1)
                            # Clean up JSON string
                            data_str = re.sub(r',\s*}', '}', data_str)
                            data_str = re.sub(r',\s*]', ']', data_str)
                            return json.loads(data_str)
                        except json.JSONDecodeError:
                            # Try to fix common issues
                            data_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', data_str)
                            try:
                                return json.loads(data_str)
                            except:
                                continue
        return None

    def _parse_script_data(self, data: Dict) -> Dict[str, Any]:
        """Parse data extracted from script.

        Args:
            data: Raw script data.

        Returns:
            Parsed fields.
        """
        result = {}

        # Navigate through common data structures
        video_data = None
        if 'video' in data:
            video_data = data['video']
        elif 'itemInfo' in data and 'itemStruct' in data['itemInfo']:
            video_data = data['itemInfo']['itemStruct']
        elif 'aweme' in data:
            video_data = data['aweme']

        if video_data:
            # Map fields
            field_mapping = {
                'id': 'video_id',
                'desc': 'desc_text',
                'createTime': 'publish_time_std',
                'author': 'author_data',
                'stats': 'stats_data',
                'music': 'music_data',
                'duration': 'duration_sec',
                'cover': 'cover_url'
            }

            for src_key, dest_key in field_mapping.items():
                if src_key in video_data:
                    result[dest_key] = video_data[src_key]

            # Extract author info
            if 'author_data' in result:
                author = result.pop('author_data')
                if isinstance(author, dict):
                    result['author_id'] = author.get('id') or author.get('uid')
                    result['author_name'] = author.get('nickname')
                    result['author_profile_url'] = author.get('profile_url')

            # Extract stats
            if 'stats_data' in result:
                stats = result.pop('stats_data')
                if isinstance(stats, dict):
                    result['like_count_raw'] = str(stats.get('diggCount', ''))
                    result['comment_count_raw'] = str(stats.get('commentCount', ''))
                    result['share_count_raw'] = str(stats.get('shareCount', ''))
                    result['collect_count'] = stats.get('collectCount')

            # Extract music
            if 'music_data' in result:
                music = result.pop('music_data')
                if isinstance(music, dict):
                    result['music_name'] = music.get('title')

        return result

    def _extract_html_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract data from HTML elements.

        Args:
            soup: BeautifulSoup object.

        Returns:
            Extracted fields.
        """
        result = {}

        # Extract description
        desc_elements = soup.find_all(['div', 'p'], class_=re.compile(r'desc|title|content'))
        if desc_elements:
            result['desc_text'] = normalize_text(desc_elements[0].get_text(strip=True))

        # Extract author info
        author_elements = soup.find_all('a', href=re.compile(r'/user/'))
        if author_elements:
            result['author_name'] = author_elements[0].get_text(strip=True)
            result['author_profile_url'] = author_elements[0].get('href', '')
            # Extract author ID from URL
            if result['author_profile_url']:
                match = re.search(r'/user/([^/?]+)', result['author_profile_url'])
                if match:
                    result['author_id'] = match.group(1)

        # Extract stats
        stats_patterns = {
            'like_count_raw': r'like|digg|赞',
            'comment_count_raw': r'comment|评论',
            'share_count_raw': r'share|分享'
        }

        for field, pattern in stats_patterns.items():
            elements = soup.find_all(['span', 'div'], text=re.compile(pattern))
            if elements:
                # Try to find sibling with count
                for elem in elements:
                    count_elem = elem.find_next(['span', 'div'])
                    if count_elem:
                        result[field] = count_elem.get_text(strip=True)
                        break

        # Extract cover image
        img_elements = soup.find_all('img', src=re.compile(r'\.(jpg|jpeg|png|webp)'))
        if img_elements:
            result['cover_url'] = img_elements[0].get('src', '')

        # Extract publish time
        time_elements = soup.find_all(['span', 'div'], class_=re.compile(r'time|date'))
        if time_elements:
            result['publish_time_raw'] = time_elements[0].get_text(strip=True)

        return result

    def _extract_video_id_from_url(self, url: str) -> Optional[str]:
        """Extract video ID from URL.

        Args:
            url: Page URL.

        Returns:
            Video ID or None.
        """
        patterns = [
            r'/video/([^/?]+)',
            r'video/([^/?]+)',
            r'item_id=([^&]+)',
            r'id=([^&]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def _normalize_counts(self, data: Dict) -> Dict[str, Any]:
        """Normalize count fields (e.g., '1.2w' -> 12000).

        Args:
            data: Data with raw count fields.

        Returns:
            Data with normalized count fields.
        """
        result = {}

        count_fields = ['like_count_raw', 'comment_count_raw', 'share_count_raw']
        for raw_field in count_fields:
            if raw_field in data and data[raw_field]:
                normalized = self._normalize_count_string(data[raw_field])
                field_name = raw_field.replace('_raw', '')
                result[field_name] = normalized

        return result

    def _normalize_count_string(self, count_str: str) -> Optional[int]:
        """Normalize count string like '1.2w' or '5k' to integer.

        Args:
            count_str: Raw count string.

        Returns:
            Normalized integer or None.
        """
        if not count_str:
            return None

        count_str = count_str.strip().lower()

        # Remove commas and other non-numeric characters (except decimal point and k/w)
        count_str = re.sub(r'[^\d.kw]', '', count_str)

        try:
            if 'w' in count_str:
                num = float(count_str.replace('w', ''))
                return int(num * 10000)
            elif 'k' in count_str:
                num = float(count_str.replace('k', ''))
                return int(num * 1000)
            elif '.' in count_str:
                return int(float(count_str))
            else:
                return int(count_str)
        except (ValueError, TypeError):
            return None

    def create_web_video_meta(self, parsed_data: Dict) -> Optional[WebVideoMeta]:
        """Create WebVideoMeta object from parsed data.

        Args:
            parsed_data: Parsed data dictionary.

        Returns:
            WebVideoMeta object or None if validation fails.
        """
        try:
            # Convert datetime strings if needed
            if 'publish_time_std' in parsed_data and isinstance(parsed_data['publish_time_std'], (int, float)):
                parsed_data['publish_time_std'] = datetime.fromtimestamp(parsed_data['publish_time_std'])

            # Ensure required fields
            if not parsed_data.get('video_id'):
                logger.warning("Missing video_id, cannot create WebVideoMeta")
                return None

            return WebVideoMeta(**parsed_data)
        except Exception as e:
            logger.error(f"Failed to create WebVideoMeta: {e}")
            return None

    def mock_parse(self, url: str, source_entry: str) -> WebVideoMeta:
        """Create mock WebVideoMeta for testing.

        Args:
            url: Page URL.
            source_entry: Source entry type.

        Returns:
            Mock WebVideoMeta object.
        """
        video_id = self._extract_video_id_from_url(url) or "mock_1234567890"
        now = datetime.now()

        return WebVideoMeta(
            video_id=video_id,
            page_url=url,
            author_id="author_mock_001",
            author_name="测试用户",
            author_profile_url=f"https://www.douyin.com/user/author_mock_001",
            desc_text="这是一个测试视频描述 #美食 #旅行",
            publish_time_raw="2023-01-01 12:00:00",
            publish_time_std=datetime(2023, 1, 1, 12, 0, 0),
            like_count_raw="1.2w",
            comment_count_raw="450",
            share_count_raw="120",
            like_count=12000,
            comment_count=450,
            share_count=120,
            collect_count=56,
            hashtag_list=["美食", "旅行"],
            hashtag_count=2,
            cover_url="https://example.com/cover.jpg",
            music_name="测试音乐",
            duration_sec=15,
            source_entry=source_entry,
            crawl_time=now
        )