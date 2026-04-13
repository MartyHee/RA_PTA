"""
API client for Douyin Open Platform.

Handles HTTP requests to Douyin API with authentication and rate limiting.
This is a placeholder implementation - real implementation requires
valid API credentials and proper endpoint integration.
"""
import time
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging

import requests

from .auth import get_auth_instance
from ..utils.config_loader import load_config, get_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DouyinAPIClient:
    """Client for Douyin Open Platform API."""

    def __init__(self, config_path: Optional[Path] = None, use_mock: bool = False):
        """Initialize API client.

        Args:
            config_path: Path to config file.
            use_mock: Whether to use mock mode (no real API calls).
        """
        self.config = load_config(config_path) if config_path else load_config()
        self.api_config = get_config('settings.api', {})
        self.use_mock = use_mock

        self.base_url = self.api_config.get('base_url', 'https://open.douyin.com')
        self.auth = get_auth_instance(config_path, use_mock)

        # Rate limiting - try sources.api.rate_limit first, then fallback
        sources_api = get_config('sources.api', {})
        rate_limit = sources_api.get('rate_limit', {})
        self.requests_per_minute = rate_limit.get('requests_per_minute', 60)
        self.burst_limit = rate_limit.get('burst_limit', 10)
        self.request_timestamps = []

        # Session
        self.session = requests.Session()
        self._setup_session()

    def _setup_session(self):
        """Setup session with default headers."""
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

    def _rate_limit(self):
        """Apply rate limiting."""
        now = time.time()
        # Remove timestamps older than 1 minute
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 60]

        # Check if we're at the limit
        if len(self.request_timestamps) >= self.requests_per_minute:
            # Wait until the oldest request is more than 1 minute old
            oldest = self.request_timestamps[0]
            wait_time = 60 - (now - oldest) + 0.1  # Small buffer
            if wait_time > 0:
                logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                time.sleep(wait_time)

        # Add current timestamp
        self.request_timestamps.append(now)

    def _add_auth_header(self, headers: Optional[Dict] = None) -> Dict:
        """Add authentication header.

        Args:
            headers: Existing headers.

        Returns:
            Updated headers.
        """
        if headers is None:
            headers = {}

        if self.auth.is_authenticated():
            access_token = self.auth.get_access_token()
            if access_token:
                headers['access-token'] = access_token
            else:
                logger.warning("No access token available")
        else:
            logger.warning("Not authenticated. Some endpoints may fail.")

        return headers

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Make HTTP request to API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint.
            **kwargs: Additional arguments to requests.

        Returns:
            Response JSON or None.
        """
        if self.use_mock:
            return self._mock_request(method, endpoint, **kwargs)

        # Apply rate limiting
        self._rate_limit()

        url = f"{self.base_url}{endpoint}"
        headers = self._add_auth_header(kwargs.pop('headers', {}))

        try:
            logger.debug(f"{method} {url}")
            response = self.session.request(method, url, headers=headers, **kwargs)
            response.raise_for_status()

            if response.status_code == 204:  # No content
                return {}

            return response.json()

        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return None

    def _mock_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Make mock request for development.

        Args:
            method: HTTP method.
            endpoint: API endpoint.
            **kwargs: Additional arguments.

        Returns:
            Mock response.
        """
        logger.info(f"Mock {method} {endpoint}")

        # Simulate network delay
        time.sleep(0.1)

        # Mock responses based on endpoint
        if '/video/data/' in endpoint:
            return self._mock_video_data_response(endpoint, **kwargs)
        elif '/user/info/' in endpoint:
            return self._mock_user_info_response(endpoint, **kwargs)
        else:
            return {
                'data': {},
                'message': 'Mock response',
                'code': 0
            }

    def _mock_video_data_response(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Mock video data response.

        Args:
            endpoint: API endpoint.
            **kwargs: Additional arguments.

        Returns:
            Mock response.
        """
        # Extract video ID from endpoint or params
        video_id = 'mock_video_123'
        if 'item_ids' in kwargs.get('params', {}):
            video_id = kwargs['params']['item_ids'].split(',')[0]

        return {
            'data': {
                'list': [
                    {
                        'item_id': video_id,
                        'title': 'Mock Video Title',
                        'cover': 'https://example.com/cover.jpg',
                        'create_time': int(time.time()) - 86400,
                        'statistics': {
                            'play_count': 10000,
                            'digg_count': 500,
                            'comment_count': 50,
                            'share_count': 20,
                            'collect_count': 10
                        }
                    }
                ]
            },
            'message': 'success',
            'code': 0
        }

    def _mock_user_info_response(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Mock user info response.

        Args:
            endpoint: API endpoint.
            **kwargs: Additional arguments.

        Returns:
            Mock response.
        """
        open_id = self.auth.get_open_id() or 'mock_open_id'

        return {
            'data': {
                'user': {
                    'open_id': open_id,
                    'nickname': 'Mock User',
                    'avatar': 'https://example.com/avatar.jpg',
                    'gender': 1,  # 1: male, 2: female, 0: unknown
                    'province': 'Beijing',
                    'city': 'Beijing',
                    'country': 'CN'
                }
            },
            'message': 'success',
            'code': 0
        }

    def get_video_data(self, item_ids: Union[str, List[str]]) -> Optional[Dict[str, Any]]:
        """Get video data.

        Args:
            item_ids: Video ID(s).

        Returns:
            Video data or None.
        """
        endpoint = self.api_config.get('video_data_endpoint', '/api/douyin/v1/video/data/')

        if isinstance(item_ids, list):
            item_ids = ','.join(item_ids)

        params = {
            'item_ids': item_ids
        }

        return self._make_request('GET', endpoint, params=params)

    def get_user_info(self, open_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get user information.

        Args:
            open_id: Open ID. If None, uses authenticated user's open_id.

        Returns:
            User info or None.
        """
        endpoint = self.api_config.get('user_info_endpoint', '/api/douyin/v1/user/info/')

        if open_id is None:
            open_id = self.auth.get_open_id()

        if not open_id:
            logger.error("No open_id provided")
            return None

        params = {
            'open_id': open_id
        }

        return self._make_request('GET', endpoint, params=params)

    def get_user_videos(self, open_id: Optional[str] = None, cursor: int = 0,
                        count: int = 20) -> Optional[Dict[str, Any]]:
        """Get user's video list.

        Args:
            open_id: Open ID.
            cursor: Pagination cursor.
            count: Number of videos per page.

        Returns:
            Video list or None.
        """
        # TODO: Implement when endpoint details are available
        logger.warning("get_user_videos not implemented - endpoint may vary")
        return None

    def search_videos(self, keyword: str, cursor: int = 0, count: int = 20) -> Optional[Dict[str, Any]]:
        """Search videos.

        Args:
            keyword: Search keyword.
            cursor: Pagination cursor.
            count: Number of results per page.

        Returns:
            Search results or None.
        """
        # TODO: Implement when endpoint details are available
        logger.warning("search_videos not implemented - may require different permissions")
        return None

    def test_connection(self) -> bool:
        """Test API connection.

        Returns:
            True if connection successful.
        """
        if self.use_mock:
            logger.info("Mock mode - connection test always passes")
            return True

        # Try a simple request
        try:
            response = self.session.get(self.base_url, timeout=5)
            return response.status_code < 500
        except requests.RequestException as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Convenience functions
def get_api_client(config_path: Optional[Path] = None, use_mock: bool = False) -> DouyinAPIClient:
    """Get API client instance.

    Args:
        config_path: Path to config file.
        use_mock: Whether to use mock mode.

    Returns:
        API client instance.
    """
    return DouyinAPIClient(config_path, use_mock)