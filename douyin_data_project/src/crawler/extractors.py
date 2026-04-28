"""
Field extractors for Douyin HTML/JSON content.

Provides modular extractors that can be replaced or extended.
Each extractor focuses on a specific field or data type.
"""
import re
import json
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
import logging

from bs4 import BeautifulSoup

from ..utils.text_utils import extract_hashtags, normalize_text
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BaseExtractor:
    """Base class for field extractors."""

    def extract(self, content: Any) -> Optional[Any]:
        """Extract field value from content.

        Args:
            content: Source content (HTML, JSON, etc.)

        Returns:
            Extracted value or None.
        """
        raise NotImplementedError


class VideoIdExtractor(BaseExtractor):
    """Extract video ID from URL or content."""

    def extract(self, content: Any) -> Optional[str]:
        """Extract video ID.

        Args:
            content: URL string or HTML content.

        Returns:
            Video ID or None.
        """
        if isinstance(content, str):
            # Try to extract from URL
            patterns = [
                r'/video/([^/?]+)',
                r'video/([^/?]+)',
                r'item_id=([^&]+)',
                r'id=([^&]+)',
                r'modal_id=([^&]+)'  # Support for jingxuan modal pages
            ]
            for pattern in patterns:
                match = re.search(pattern, content)
                if match:
                    return match.group(1)

            # Try to extract from HTML
            if '<script>' in content:
                # Look for video ID in script
                script_patterns = [
                    r'video_id["\']?\s*:\s*["\']([^"\']+)["\']',
                    r'"id"\s*:\s*"([^"]+)"',
                    r'itemId["\']?\s*:\s*["\']([^"\']+)["\']'
                ]
                for pattern in script_patterns:
                    match = re.search(pattern, content)
                    if match:
                        return match.group(1)

        return None


class DescriptionExtractor(BaseExtractor):
    """Extract video description/text."""

    def extract(self, content: Any) -> Optional[str]:
        """Extract description.

        Args:
            content: HTML content or BeautifulSoup object.

        Returns:
            Description text or None.
        """
        soup = self._ensure_soup(content)
        if not soup:
            return None

        # Try multiple selectors
        selectors = [
            'div[class*="desc"]',
            'div[class*="title"]',
            'p[class*="content"]',
            'span[class*="text"]',
            'div.desc',
            'div.title'
        ]

        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                text = elements[0].get_text(strip=True)
                if text and len(text) > 5:  # Minimum length
                    return normalize_text(text)

        return None

    def _ensure_soup(self, content: Any) -> Optional[BeautifulSoup]:
        """Ensure content is BeautifulSoup object."""
        if isinstance(content, BeautifulSoup):
            return content
        elif isinstance(content, str):
            try:
                return BeautifulSoup(content, 'lxml')
            except:
                return BeautifulSoup(content, 'html.parser')
        return None


class AuthorExtractor(BaseExtractor):
    """Extract author information."""

    def extract(self, content: Any) -> Dict[str, Optional[str]]:
        """Extract author info.

        Args:
            content: HTML content or BeautifulSoup object.

        Returns:
            Dictionary with author_id, author_name, author_page_url.
        """
        result = {
            'author_id': None,
            'author_name': None,
            'author_page_url': None
        }

        soup = self._ensure_soup(content)
        if not soup:
            return result

        # Find author links
        author_links = soup.find_all('a', href=re.compile(r'/user/'))
        if author_links:
            link = author_links[0]
            result['author_name'] = link.get_text(strip=True)
            result['author_page_url'] = link.get('href', '')

            # Extract author ID from URL
            if result['author_page_url']:
                match = re.search(r'/user/([^/?]+)', result['author_page_url'])
                if match:
                    result['author_id'] = match.group(1)

        # Also try script data
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string:
                patterns = [
                    r'author["\']?\s*:\s*{.*?"id"\s*:\s*["\']([^"\']+)["\']',
                    r'"author"\s*:\s*{.*?"uid"\s*:\s*"([^"]+)"',
                    r'"nickname"\s*:\s*"([^"]+)"'
                ]
                for pattern in patterns:
                    match = re.search(pattern, script.string, re.DOTALL)
                    if match:
                        if 'id' in pattern or 'uid' in pattern:
                            result['author_id'] = match.group(1)
                        elif 'nickname' in pattern:
                            result['author_name'] = match.group(1)

        return result

    def _ensure_soup(self, content: Any) -> Optional[BeautifulSoup]:
        """Ensure content is BeautifulSoup object."""
        if isinstance(content, BeautifulSoup):
            return content
        elif isinstance(content, str):
            try:
                return BeautifulSoup(content, 'lxml')
            except:
                return BeautifulSoup(content, 'html.parser')
        return None


class StatsExtractor(BaseExtractor):
    """Extract video statistics (likes, comments, shares)."""

    def extract(self, content: Any) -> Dict[str, Optional[str]]:
        """Extract stats.

        Args:
            content: HTML content or BeautifulSoup object.

        Returns:
            Dictionary with raw count strings.
        """
        result = {
            'digg_count': None,
            'comment_count_raw': None,
            'share_count_raw': None,
            'collect_count_raw': None
        }

        soup = self._ensure_soup(content)
        if not soup:
            return result

        # Look for stats in HTML elements
        stat_patterns = {
            'digg_count': [r'like', r'digg', r'赞', r'点赞'],
            'comment_count_raw': [r'comment', r'评论'],
            'share_count_raw': [r'share', r'分享'],
            'collect_count_raw': [r'collect', r'收藏']
        }

        for field, patterns in stat_patterns.items():
            for pattern in patterns:
                elements = soup.find_all(text=re.compile(pattern, re.IGNORECASE))
                for elem in elements:
                    # Look for nearby number
                    parent = elem.parent
                    if parent:
                        # Search in siblings
                        for sibling in parent.find_next_siblings():
                            text = sibling.get_text(strip=True)
                            if re.search(r'\d+', text):
                                result[field] = text
                                break

        # Also try script data
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string:
                patterns = [
                    r'stats["\']?\s*:\s*{.*?"diggCount"\s*:\s*(\d+)',
                    r'"diggCount"\s*:\s*(\d+)',
                    r'"commentCount"\s*:\s*(\d+)',
                    r'"shareCount"\s*:\s*(\d+)',
                    r'"collectCount"\s*:\s*(\d+)'
                ]
                for pattern in patterns:
                    match = re.search(pattern, script.string, re.DOTALL)
                    if match:
                        count = match.group(1)
                        if 'diggCount' in pattern:
                            result['digg_count'] = count
                        elif 'commentCount' in pattern:
                            result['comment_count_raw'] = count
                        elif 'shareCount' in pattern:
                            result['share_count_raw'] = count
                        elif 'collectCount' in pattern:
                            result['collect_count_raw'] = count

        return result

    def _ensure_soup(self, content: Any) -> Optional[BeautifulSoup]:
        """Ensure content is BeautifulSoup object."""
        if isinstance(content, BeautifulSoup):
            return content
        elif isinstance(content, str):
            try:
                return BeautifulSoup(content, 'lxml')
            except:
                return BeautifulSoup(content, 'html.parser')
        return None


class PublishTimeExtractor(BaseExtractor):
    """Extract publish time."""

    def extract(self, content: Any) -> Dict[str, Optional[str]]:
        """Extract publish time.

        Args:
            content: HTML content or BeautifulSoup object.

        Returns:
            Dictionary with raw and standardized times.
        """
        result = {
            'create_time': None,
            'publish_time_std': None
        }

        soup = self._ensure_soup(content)
        if not soup:
            return result

        # Look for time elements
        time_elements = soup.find_all(['span', 'div', 'time'], class_=re.compile(r'time|date|publish'))
        if time_elements:
            result['create_time'] = time_elements[0].get_text(strip=True)

        # Also try script data
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string:
                patterns = [
                    r'createTime["\']?\s*:\s*(\d+)',
                    r'"createTime"\s*:\s*(\d+)',
                    r'"publishTime"\s*:\s*(\d+)'
                ]
                for pattern in patterns:
                    match = re.search(pattern, script.string)
                    if match:
                        timestamp = int(match.group(1))
                        result['publish_time_std'] = datetime.fromtimestamp(timestamp).isoformat()
                        break

        return result

    def _ensure_soup(self, content: Any) -> Optional[BeautifulSoup]:
        """Ensure content is BeautifulSoup object."""
        if isinstance(content, BeautifulSoup):
            return content
        elif isinstance(content, str):
            try:
                return BeautifulSoup(content, 'lxml')
            except:
                return BeautifulSoup(content, 'html.parser')
        return None


class HashtagExtractor(BaseExtractor):
    """Extract hashtags from description."""

    def extract(self, content: Any) -> Dict[str, Any]:
        """Extract hashtags.

        Args:
            content: Description text or HTML.

        Returns:
            Dictionary with hashtag_list and hashtag_count.
        """
        result = {
            'hashtag_list': [],
            'hashtag_count': 0
        }

        if isinstance(content, str):
            hashtags = extract_hashtags(content)
            result['hashtag_list'] = hashtags
            result['hashtag_count'] = len(hashtags)

        return result


class CoverUrlExtractor(BaseExtractor):
    """Extract cover image URL."""

    def extract(self, content: Any) -> Optional[str]:
        """Extract cover URL.

        Args:
            content: HTML content or BeautifulSoup object.

        Returns:
            Cover URL or None.
        """
        soup = self._ensure_soup(content)
        if not soup:
            return None

        # Look for image tags
        img_tags = soup.find_all('img', src=re.compile(r'\.(jpg|jpeg|png|webp|gif)'))
        for img in img_tags:
            src = img.get('src', '')
            if src and ('cover' in src.lower() or 'video' in src.lower()):
                return src

        # Return first image if no cover-specific found
        if img_tags:
            return img_tags[0].get('src')

        return None

    def _ensure_soup(self, content: Any) -> Optional[BeautifulSoup]:
        """Ensure content is BeautifulSoup object."""
        if isinstance(content, BeautifulSoup):
            return content
        elif isinstance(content, str):
            try:
                return BeautifulSoup(content, 'lxml')
            except:
                return BeautifulSoup(content, 'html.parser')
        return None


# Factory for creating extractors
class ExtractorFactory:
    """Factory for creating and managing extractors."""

    @staticmethod
    def get_extractor(field_name: str) -> Optional[BaseExtractor]:
        """Get extractor for specific field.

        Args:
            field_name: Name of field to extract.

        Returns:
            Extractor instance or None.
        """
        extractors = {
            'video_id': VideoIdExtractor(),
            'desc_text': DescriptionExtractor(),
            'author_info': AuthorExtractor(),
            'stats': StatsExtractor(),
            'publish_time': PublishTimeExtractor(),
            'hashtags': HashtagExtractor(),
            'origin_cover_url': CoverUrlExtractor()
        }

        # Map field names to extractors
        field_to_extractor = {
            'video_id': 'video_id',
            'author_id': 'author_info',
            'author_name': 'author_info',
            'author_page_url': 'author_info',
            'desc_text': 'desc_text',
            'digg_count': 'stats',
            'comment_count_raw': 'stats',
            'share_count_raw': 'stats',
            'collect_count_raw': 'stats',
            'create_time': 'publish_time',
            'publish_time_std': 'publish_time',
            'hashtag_list': 'hashtags',
            'hashtag_count': 'hashtags',
            'origin_cover_url': 'origin_cover_url'
        }

        extractor_key = field_to_extractor.get(field_name)
        if extractor_key:
            return extractors.get(extractor_key)

        return None

    @staticmethod
    def extract_all(content: Any) -> Dict[str, Any]:
        """Extract all fields using appropriate extractors.

        Args:
            content: Source content.

        Returns:
            Dictionary with all extracted fields.
        """
        result = {}

        # Get all extractors
        extractors = {
            'video_id': VideoIdExtractor(),
            'desc_text': DescriptionExtractor(),
            'author_info': AuthorExtractor(),
            'stats': StatsExtractor(),
            'publish_time': PublishTimeExtractor(),
            'hashtags': HashtagExtractor(),
            'origin_cover_url': CoverUrlExtractor()
        }

        # Run each extractor
        for name, extractor in extractors.items():
            try:
                extracted = extractor.extract(content)
                if extracted:
                    if isinstance(extracted, dict):
                        result.update(extracted)
                    else:
                        result[name] = extracted
            except Exception as e:
                logger.warning(f"Extractor {name} failed: {e}")

        return result