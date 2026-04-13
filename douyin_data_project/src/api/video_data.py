"""
Video data module for Douyin API.

Handles video data retrieval and processing from Douyin Open Platform.
This module bridges API responses with our internal data schemas.
"""
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path
import logging

from .client import get_api_client
from ..schemas.tables import ApiVideoStats, ApiUserProfile
from ..utils.config_loader import load_config, get_config
from ..utils.logger import get_logger
from ..utils.io_utils import write_parquet, write_jsonl

logger = get_logger(__name__)


class DouyinVideoData:
    """Handles video data from Douyin API."""

    def __init__(self, config_path: Optional[Path] = None, use_mock: bool = False):
        """Initialize video data handler.

        Args:
            config_path: Path to config file.
            use_mock: Whether to use mock mode.
        """
        self.config = load_config(config_path) if config_path else load_config()
        self.use_mock = use_mock
        self.client = get_api_client(config_path, use_mock)

        self.output_dir = Path(get_config('settings.paths.processed_data', './data/processed'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_video_stats(self, video_ids: Union[str, List[str]]) -> List[ApiVideoStats]:
        """Fetch video statistics from API.

        Args:
            video_ids: Video ID(s) to fetch.

        Returns:
            List of ApiVideoStats objects.
        """
        if isinstance(video_ids, str):
            video_ids = [video_ids]

        if not video_ids:
            logger.warning("No video IDs provided")
            return []

        # Fetch from API
        response = self.client.get_video_data(video_ids)
        if not response:
            logger.error("Failed to fetch video data")
            return []

        # Parse response
        stats_list = []
        api_pull_time = datetime.now()

        try:
            data = response.get('data', {})
            video_list = data.get('list', [])

            for video_data in video_list:
                stats = self._parse_video_stats(video_data, api_pull_time)
                if stats:
                    stats_list.append(stats)

            logger.info(f"Fetched stats for {len(stats_list)} videos")

        except Exception as e:
            logger.error(f"Failed to parse video stats: {e}")

        return stats_list

    def _parse_video_stats(self, video_data: Dict[str, Any], api_pull_time: datetime) -> Optional[ApiVideoStats]:
        """Parse video data from API response.

        Args:
            video_data: Video data from API.
            api_pull_time: Time when data was pulled.

        Returns:
            ApiVideoStats object or None.
        """
        try:
            # Extract fields
            video_id = video_data.get('item_id', '')
            open_id = self.client.auth.get_open_id() or ''

            if not video_id:
                logger.warning("Video data missing item_id")
                return None

            # Parse statistics
            stats = video_data.get('statistics', {})
            create_time = video_data.get('create_time')

            # Convert create_time if it's a timestamp
            create_time_dt = None
            if create_time:
                try:
                    create_time_dt = datetime.fromtimestamp(create_time)
                except (TypeError, ValueError):
                    pass

            # Create ApiVideoStats object
            return ApiVideoStats(
                video_id=video_id,
                open_id=open_id,
                stat_time=api_pull_time,  # Using pull time as stat time
                play_count=stats.get('play_count'),
                digg_count=stats.get('digg_count'),
                comment_count=stats.get('comment_count'),
                share_count=stats.get('share_count'),
                cover_url=video_data.get('cover'),
                create_time=create_time_dt,
                api_pull_time=api_pull_time
            )

        except Exception as e:
            logger.error(f"Failed to parse video data: {e}")
            return None

    def fetch_user_profile(self, open_id: Optional[str] = None) -> Optional[ApiUserProfile]:
        """Fetch user profile from API.

        Args:
            open_id: Open ID. If None, uses authenticated user.

        Returns:
            ApiUserProfile object or None.
        """
        response = self.client.get_user_info(open_id)
        if not response:
            logger.error("Failed to fetch user profile")
            return None

        try:
            data = response.get('data', {})
            user_data = data.get('user', {})

            if not user_data:
                logger.warning("No user data in response")
                return None

            # Parse gender
            gender_map = {0: 'unknown', 1: 'male', 2: 'female'}
            gender_code = user_data.get('gender', 0)
            gender = gender_map.get(gender_code, 'unknown')

            return ApiUserProfile(
                open_id=user_data.get('open_id', ''),
                nickname=user_data.get('nickname'),
                avatar_url=user_data.get('avatar'),
                gender=gender,
                province=user_data.get('province'),
                city=user_data.get('city'),
                pull_time=datetime.now()
            )

        except Exception as e:
            logger.error(f"Failed to parse user profile: {e}")
            return None

    def save_video_stats(self, stats_list: List[ApiVideoStats], append: bool = True):
        """Save video stats to file.

        Args:
            stats_list: List of ApiVideoStats objects.
            append: Whether to append to existing file.
        """
        if not stats_list:
            logger.warning("No stats to save")
            return

        output_file = self.output_dir / 'api_video_stats.parquet'

        # Convert to dictionaries
        data = [stats.dict() for stats in stats_list]

        # Save
        write_parquet(output_file, data, mode='a' if append else 'w')
        logger.info(f"Saved {len(stats_list)} video stats to {output_file}")

    def save_user_profile(self, profile: ApiUserProfile, append: bool = True):
        """Save user profile to file.

        Args:
            profile: ApiUserProfile object.
            append: Whether to append to existing file.
        """
        if not profile:
            logger.warning("No profile to save")
            return

        output_file = self.output_dir / 'api_user_profile.parquet'
        data = [profile.dict()]

        write_parquet(output_file, data, mode='a' if append else 'w')
        logger.info(f"Saved user profile to {output_file}")

    def batch_fetch_and_save(self, video_ids: List[str], batch_size: int = 10):
        """Fetch and save video stats in batches.

        Args:
            video_ids: List of video IDs.
            batch_size: Number of videos per batch.
        """
        total = len(video_ids)
        logger.info(f"Starting batch fetch for {total} videos (batch size: {batch_size})")

        for i in range(0, total, batch_size):
            batch = video_ids[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}")

            try:
                stats_list = self.fetch_video_stats(batch)
                if stats_list:
                    self.save_video_stats(stats_list, append=(i > 0))
                else:
                    logger.warning(f"No stats fetched for batch starting at index {i}")

                # Rate limiting delay
                if not self.use_mock and i + batch_size < total:
                    time.sleep(1)  # Be respectful to API

            except Exception as e:
                logger.error(f"Failed to process batch {i}: {e}")
                continue

        logger.info("Batch fetch completed")

    def sync_with_web_data(self, web_video_ids: List[str], existing_api_ids: List[str] = None):
        """Sync API data with web data.

        Args:
            web_video_ids: Video IDs from web crawling.
            existing_api_ids: Video IDs already in API database.
        """
        if existing_api_ids is None:
            existing_api_ids = []

        # Find missing video IDs
        missing_ids = [vid for vid in web_video_ids if vid not in existing_api_ids]

        if not missing_ids:
            logger.info("All web videos already have API data")
            return

        logger.info(f"Found {len(missing_ids)} videos missing API data")

        # Fetch missing data
        self.batch_fetch_and_save(missing_ids)

    def generate_mock_data(self, count: int = 10):
        """Generate mock API data for development.

        Args:
            count: Number of mock records to generate.
        """
        if not self.use_mock:
            logger.warning("Not in mock mode. Real API data will be fetched.")
            # Fall through to use real API

        logger.info(f"Generating {count} mock API records")

        # Mock video stats
        stats_list = []
        for i in range(count):
            stats = ApiVideoStats(
                video_id=f'mock_video_{i:04d}',
                open_id='mock_open_id',
                stat_time=datetime.now(),
                play_count=1000 + i * 100,
                digg_count=50 + i * 5,
                comment_count=10 + i,
                share_count=5 + i,
                cover_url=f'https://example.com/cover_{i}.jpg',
                create_time=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
                api_pull_time=datetime.now()
            )
            stats_list.append(stats)

        self.save_video_stats(stats_list, append=False)

        # Mock user profile
        profile = ApiUserProfile(
            open_id='mock_open_id',
            nickname='Mock User',
            avatar_url='https://example.com/avatar.jpg',
            gender='male',
            province='Beijing',
            city='Beijing',
            pull_time=datetime.now()
        )

        self.save_user_profile(profile, append=False)

        logger.info("Mock API data generated")


# Convenience functions
def fetch_and_save_video_stats(video_ids: Union[str, List[str]], config_path: Optional[Path] = None,
                               use_mock: bool = False) -> List[ApiVideoStats]:
    """Fetch and save video stats.

    Args:
        video_ids: Video ID(s).
        config_path: Path to config.
        use_mock: Whether to use mock mode.

    Returns:
        List of ApiVideoStats objects.
    """
    handler = DouyinVideoData(config_path, use_mock)
    stats = handler.fetch_video_stats(video_ids)
    if stats:
        handler.save_video_stats(stats)
    return stats


def fetch_and_save_user_profile(open_id: Optional[str] = None, config_path: Optional[Path] = None,
                                use_mock: bool = False) -> Optional[ApiUserProfile]:
    """Fetch and save user profile.

    Args:
        open_id: Open ID.
        config_path: Path to config.
        use_mock: Whether to use mock mode.

    Returns:
        ApiUserProfile object or None.
    """
    handler = DouyinVideoData(config_path, use_mock)
    profile = handler.fetch_user_profile(open_id)
    if profile:
        handler.save_user_profile(profile)
    return profile