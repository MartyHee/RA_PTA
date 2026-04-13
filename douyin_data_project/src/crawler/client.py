"""
HTTP client for Douyin web crawler.

Handles requests with retries, delays, and anti-blocking measures.
Supports mock mode for development without network access.
"""
import time
import random
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import logging

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..utils.config_loader import load_config, get_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DouyinClient:
    """HTTP client for Douyin with retry and delay support."""

    def __init__(self, config_path: Optional[Path] = None, use_mock: bool = False):
        """Initialize the client.

        Args:
            config_path: Path to config file. If None, loads default.
            use_mock: Whether to use mock mode (no actual requests).
        """
        self.config = load_config(config_path) if config_path else load_config()
        # Get mock mode from sources.web.mock.enabled or parameter
        sources_mock = get_config('sources.web.mock.enabled', False)
        self.use_mock = use_mock or sources_mock

        self.request_timeout = get_config('settings.crawler.request_timeout', 30)
        self.max_retries = get_config('settings.crawler.max_retries', 3)
        self.retry_delay = get_config('settings.crawler.retry_delay', 2)
        self.delay_between_requests = get_config('settings.crawler.delay_between_requests', 1.5)
        self.user_agent = get_config('settings.crawler.user_agent', '')
        self.headers = get_config('settings.crawler.headers', {})

        if self.user_agent:
            self.headers['User-Agent'] = self.user_agent

        self.session = requests.Session()
        self._setup_session()

        self.last_request_time = 0

    def _setup_session(self):
        """Setup session with retry strategy."""
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _respect_delay(self):
        """Respect delay between requests to avoid being blocked."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.delay_between_requests:
            sleep_time = self.delay_between_requests - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _add_jitter(self, delay: float) -> float:
        """Add random jitter to delay to make pattern less predictable."""
        jitter = random.uniform(0.8, 1.2)
        return delay * jitter

    def get(self, url: str, params: Optional[Dict] = None, **kwargs) -> Optional[requests.Response]:
        """Make a GET request with retries and delays.

        Args:
            url: Target URL.
            params: Query parameters.
            **kwargs: Additional arguments to requests.get.

        Returns:
            Response object or None if all retries failed.
        """
        if self.use_mock:
            logger.info(f"Mock mode: would GET {url}")
            return self._mock_response(url)

        self._respect_delay()

        headers = {**self.headers, **kwargs.pop('headers', {})}
        timeout = kwargs.pop('timeout', self.request_timeout)

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"GET {url} (attempt {attempt + 1}/{self.max_retries + 1})")
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                    **kwargs
                )
                response.raise_for_status()
                logger.debug(f"GET {url} succeeded with status {response.status_code}")
                return response

            except requests.RequestException as e:
                logger.warning(f"GET {url} failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    delay = self._add_jitter(self.retry_delay * (2 ** attempt))
                    logger.debug(f"Retrying in {delay:.2f}s")
                    time.sleep(delay)
                else:
                    logger.error(f"GET {url} failed after {self.max_retries + 1} attempts")
                    return None

    def post(self, url: str, data: Optional[Dict] = None, json: Optional[Dict] = None, **kwargs) -> Optional[requests.Response]:
        """Make a POST request with retries and delays.

        Args:
            url: Target URL.
            data: Form data.
            json: JSON data.
            **kwargs: Additional arguments to requests.post.

        Returns:
            Response object or None if all retries failed.
        """
        if self.use_mock:
            logger.info(f"Mock mode: would POST {url}")
            return self._mock_response(url)

        self._respect_delay()

        headers = {**self.headers, **kwargs.pop('headers', {})}
        timeout = kwargs.pop('timeout', self.request_timeout)

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"POST {url} (attempt {attempt + 1}/{self.max_retries + 1})")
                response = self.session.post(
                    url,
                    data=data,
                    json=json,
                    headers=headers,
                    timeout=timeout,
                    **kwargs
                )
                response.raise_for_status()
                logger.debug(f"POST {url} succeeded with status {response.status_code}")
                return response

            except requests.RequestException as e:
                logger.warning(f"POST {url} failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    delay = self._add_jitter(self.retry_delay * (2 ** attempt))
                    logger.debug(f"Retrying in {delay:.2f}s")
                    time.sleep(delay)
                else:
                    logger.error(f"POST {url} failed after {self.max_retries + 1} attempts")
                    return None

    def _mock_response(self, url: str) -> requests.Response:
        """Create a mock response for testing.

        Args:
            url: URL to mock.

        Returns:
            Mock response object.
        """
        # Create a mock response
        from unittest.mock import Mock
        response = Mock(spec=requests.Response)
        response.status_code = 200
        response.url = url
        response.headers = {'Content-Type': 'text/html'}
        response.text = self._generate_mock_html(url)
        response.content = response.text.encode('utf-8')
        return response

    def _generate_mock_html(self, url: str) -> str:
        """Generate mock HTML for testing.

        Args:
            url: URL to generate mock for.

        Returns:
            Mock HTML string.
        """
        video_id = "1234567890123456789"
        if "video/" in url:
            # Extract video ID from URL
            parts = url.split("video/")
            if len(parts) > 1:
                video_id = parts[1].split("?")[0]

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mock Douyin Video</title>
            <script>
                window.__INITIAL_STATE__ = {{
                    video: {{
                        id: "{video_id}",
                        desc: "这是一个测试视频描述 #美食 #旅行",
                        createTime: 1672531200,
                        author: {{
                            id: "author123",
                            nickname: "测试用户",
                            uniqueId: "testuser",
                            avatar: "https://example.com/avatar.jpg"
                        }},
                        stats: {{
                            diggCount: 12000,
                            commentCount: 450,
                            shareCount: 120,
                            collectCount: 56
                        }},
                        music: {{
                            title: "测试音乐"
                        }},
                        duration: 15000,
                        cover: "https://example.com/cover.jpg"
                    }}
                }};
            </script>
        </head>
        <body>
            <div class="video-info">
                <h1>Mock Douyin Video</h1>
                <p>Video ID: {video_id}</p>
                <p>This is a mock HTML response for development.</p>
                <div class="stats">
                    <span class="like">1.2w</span>
                    <span class="comment">450</span>
                    <span class="share">120</span>
                </div>
            </div>
        </body>
        </html>
        """

    def close(self):
        """Close the session."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()