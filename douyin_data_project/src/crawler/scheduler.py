"""
Scheduler for managing crawl tasks.

Supports queueing multiple URLs, rate limiting, and task tracking.
"""
import time
import threading
import json
import csv
import re
from datetime import datetime
from typing import List, Optional, Dict, Any, Callable
from queue import Queue, Empty
from pathlib import Path
import logging

from .client import DouyinClient
from .browser_client import BrowserClient
from .parser import DouyinParser
from ..schemas.tables import RawWebVideoData, WebVideoMeta
from ..utils.config_loader import load_config, get_config
from ..utils.logger import get_logger
from ..utils.io_utils import write_jsonl, write_parquet, write_csv

logger = get_logger(__name__)


class CrawlTask:
    """Represents a single crawl task."""

    def __init__(self, original_url: str, source_entry: str, priority: int = 0,
                 canonical_video_url: Optional[str] = None, page_type: Optional[str] = None):
        self.original_url = original_url
        self.canonical_video_url = canonical_video_url
        self.page_type = page_type
        # Determine which URL to use for display
        self.url = canonical_video_url if canonical_video_url else original_url
        self.source_entry = source_entry
        self.priority = priority
        self.created_at = datetime.now()
        self.attempts = 0
        self.max_attempts = 3
        self.status = 'pending'  # pending, in_progress, completed, failed

    def mark_in_progress(self):
        """Mark task as in progress."""
        self.status = 'in_progress'
        self.attempts += 1

    def mark_completed(self):
        """Mark task as completed."""
        self.status = 'completed'

    def mark_failed(self):
        """Mark task as failed."""
        self.status = 'failed'

    def should_retry(self) -> bool:
        """Check if task should be retried."""
        return self.attempts < self.max_attempts and self.status != 'completed'

    def __lt__(self, other):
        # For priority queue (higher priority first)
        return self.priority > other.priority


class CrawlScheduler:
    """Manages crawl tasks with rate limiting."""

    def __init__(self, config_path: Optional[Path] = None, use_mock: bool = False, use_browser: bool = False):
        """Initialize scheduler.

        Args:
            config_path: Path to config file.
            use_mock: Whether to use mock mode.
            use_browser: Whether to use browser mode for JavaScript-rendered pages.
        """
        self.config = load_config(config_path) if config_path else load_config()
        self.use_mock = use_mock
        self.use_browser = use_browser

        self.task_queue = Queue()
        self.completed_tasks = []
        self.failed_tasks = []

        # Generate run ID and output prefix
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.use_mock:
            self.output_prefix = "mock_"
            self.file_suffix = self.run_id
        else:
            # Get prefix from config, default to "real_"
            self.output_prefix = get_config('sources.web.real_crawl.output_prefix', 'real_')
            self.file_suffix = self.run_id

        # Initialize appropriate client based on mode
        if use_mock:
            # Mock mode uses DouyinClient with mock responses
            self.client = DouyinClient(config_path, use_mock=True)
            self.browser_client = None
        elif use_browser:
            # Browser mode uses BrowserClient for JavaScript rendering
            self.client = None
            self.browser_client = BrowserClient(config_path, use_mock=False)
            # Set run ID for organizing debug output
            self.browser_client.set_run_id(self.run_id)
        else:
            # Default mode: real requests using DouyinClient
            self.client = DouyinClient(config_path, use_mock=False)
            self.browser_client = None

        self.parser = DouyinParser(config_path)

        self.max_workers = get_config('settings.crawler.max_workers', 1)
        self.max_queue_size = get_config('settings.crawler.max_queue_size', 1000)
        self.save_raw_html = get_config('settings.crawler.save_raw_html', True)

        self.output_dir = Path(get_config('settings.paths.raw_data', './data/raw'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Interim data directory for standardized output
        self.interim_dir = Path(get_config('settings.paths.interim_data', './data/interim'))
        self.interim_dir.mkdir(parents=True, exist_ok=True)

        # Processed data directory for high-confidence samples
        self.processed_dir = Path(get_config('settings.paths.processed_data', './data/processed'))
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Create run-specific subdirectories
        self.interim_run_dir = self.interim_dir / self.run_id
        self.interim_run_dir.mkdir(parents=True, exist_ok=True)
        self.processed_run_dir = self.processed_dir / self.run_id
        self.processed_run_dir.mkdir(parents=True, exist_ok=True)


        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Log run information
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Interim data directory: {self.interim_run_dir}")
        logger.info(f"Processed data directory: {self.processed_run_dir}")
        if self.use_browser and not self.use_mock:
            debug_dir = self.output_dir / "debug" / self.run_id
            rendered_dir = self.output_dir / "rendered_html" / self.run_id
            logger.info(f"Debug output directory: {debug_dir}")
            logger.info(f"Rendered HTML directory: {rendered_dir}")
        elif not self.use_mock:
            html_dir = self.output_dir / "html" / self.run_id
            logger.info(f"HTML output directory: {html_dir}")

    def add_task(self, url: str, source_entry: str = 'manual_url', priority: int = 0):
        """Add a task to the queue.

        Args:
            url: URL to crawl.
            source_entry: Source entry type.
            priority: Task priority (higher = earlier execution).
        """
        if self.task_queue.qsize() >= self.max_queue_size:
            logger.warning(f"Queue full ({self.max_queue_size}), dropping task: {url}")
            return

        # Normalize URL to get canonical video URL and page type
        normalized = self.parser.normalize_douyin_url(url)
        original_url = normalized['original_url']
        canonical_video_url = normalized['canonical_video_url']
        page_type = normalized['page_type']

        task = CrawlTask(
            original_url=original_url,
            source_entry=source_entry,
            priority=priority,
            canonical_video_url=canonical_video_url,
            page_type=page_type
        )
        self.task_queue.put(task)
        logger.debug(f"Added task: {original_url} -> {canonical_video_url} ({page_type}, {source_entry})")

    def add_tasks(self, urls: List[str], source_entry: str = 'manual_url', priority: int = 0):
        """Add multiple tasks to the queue.

        Args:
            urls: List of URLs to crawl.
            source_entry: Source entry type.
            priority: Task priority.
        """
        for url in urls:
            self.add_task(url, source_entry, priority)

    def worker(self, worker_id: int):
        """Worker function to process tasks.

        Args:
            worker_id: Worker identifier.
        """
        logger.info(f"Worker {worker_id} started")

        while not self._stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=1)
            except Empty:
                continue

            try:
                task.mark_in_progress()
                logger.info(f"Worker {worker_id} processing: {task.url}")

                # Crawl with priority: try canonical_video_url first, fallback to original_url
                crawl_time = datetime.now()
                fetched_url = None
                response = None
                html_content = None
                http_status = None
                response_headers = {}
                rendered_html_path = None
                browser_extracted_fields = None
                browser_extraction_summary = None

                # Determine which client to use
                if self.use_mock or not self.use_browser:
                    # Use regular HTTP client (DouyinClient)
                    client = self.client
                else:
                    # Use browser client
                    client = self.browser_client

                # Try canonical video URL first if available
                if task.canonical_video_url and task.canonical_video_url != task.original_url:
                    logger.info(f"Trying canonical video URL: {task.canonical_video_url}")
                    if self.use_browser and not self.use_mock:
                        # Browser client returns dict
                        browser_result = client.get(task.canonical_video_url)
                        if browser_result is not None:
                            response = browser_result  # Store dict as response
                            html_content = browser_result['html']
                            fetched_url = browser_result['url']
                            http_status = browser_result['status']
                            response_headers = browser_result['headers']
                            # Extract fields from browser runtime data if available
                            if 'extracted_fields' in browser_result:
                                browser_extracted_fields = browser_result['extracted_fields']
                                browser_extraction_summary = browser_result.get('extraction_summary')
                                logger.info(f"Browser extracted {len(browser_extracted_fields)} fields from runtime data")
                            logger.info(f"Successfully fetched canonical URL via browser: {fetched_url}")
                        else:
                            logger.warning(f"Canonical URL failed via browser, falling back to original: {task.original_url}")
                            response = None
                    else:
                        # Regular HTTP client
                        response = client.get(task.canonical_video_url)
                        if response is not None and response.status_code == 200:
                            fetched_url = response.url
                            html_content = response.text
                            http_status = response.status_code
                            response_headers = response.headers
                            logger.info(f"Successfully fetched canonical URL: {fetched_url}")
                        else:
                            logger.warning(f"Canonical URL failed, falling back to original: {task.original_url}")
                            response = None

                # If canonical failed or not available, try original URL
                if response is None:
                    logger.info(f"Fetching original URL: {task.original_url}")
                    if self.use_browser and not self.use_mock:
                        browser_result = client.get(task.original_url)
                        if browser_result is not None:
                            response = browser_result
                            html_content = browser_result['html']
                            fetched_url = browser_result['url']
                            http_status = browser_result['status']
                            response_headers = browser_result['headers']
                            # Extract fields from browser runtime data if available
                            if 'extracted_fields' in browser_result:
                                browser_extracted_fields = browser_result['extracted_fields']
                                browser_extraction_summary = browser_result.get('extraction_summary')
                                logger.info(f"Browser extracted {len(browser_extracted_fields)} fields from runtime data")
                        else:
                            response = None
                    else:
                        response = client.get(task.original_url)
                        if response is not None:
                            fetched_url = response.url
                            html_content = response.text
                            http_status = response.status_code
                            response_headers = response.headers

                if response is None:
                    logger.error(f"Failed to fetch {task.original_url}")
                    task.mark_failed()
                    with self._lock:
                        self.failed_tasks.append(task)
                    continue

                # Log crawl evidence
                if isinstance(response, dict):
                    # Browser response
                    content_length = len(html_content) if html_content else 0
                    content_encoding = response_headers.get('Content-Encoding', 'none')
                    final_url = fetched_url or task.url
                    logger.info(f"Crawl evidence - URL: {task.url}, "
                               f"HTTP Status: {http_status}, "
                               f"Final URL: {final_url}, "
                               f"Content length: {content_length} chars, "
                               f"Content-Encoding: {content_encoding}")
                else:
                    # Regular HTTP response
                    logger.info(f"Crawl evidence - URL: {task.url}, "
                               f"HTTP Status: {response.status_code}, "
                               f"Final URL: {response.url}, "
                               f"Content length: {len(response.text)} chars, "
                               f"Content-Encoding: {response.headers.get('Content-Encoding', 'none')}")

                # Parse HTML content
                logger.info(f"Starting HTML parsing for {task.url}, HTML length: {len(html_content) if html_content else 0}")
                parsed_data = self.parser.parse_html(
                    html=html_content if html_content else (response.text if response else ''),
                    url=task.url,
                    source_entry=task.source_entry,
                    crawl_time=crawl_time,
                    page_type=task.page_type
                )
                logger.info(f"HTML parsing completed, parsed_data keys: {list(parsed_data.keys())}")

                # Merge browser-extracted fields if available (higher priority than HTML parsing)
                if browser_extracted_fields:
                    logger.info(f"Merging {len(browser_extracted_fields)} browser-extracted fields into parsed data")
                    # Field mapping from browser_extracted_fields to parsed_data field names
                    field_mapping = {
                        'video_id': 'video_id',
                        'author_id': 'author_id',
                        'author_name': 'author_name',
                        'author_profile_url': 'author_profile_url',
                        'desc_text': 'desc_text',
                        'publish_time_raw': 'publish_time_raw',
                        'like_count_raw': 'like_count_raw',
                        'comment_count_raw': 'comment_count_raw',
                        'share_count_raw': 'share_count_raw',
                        'collect_count': 'collect_count',
                        'hashtag_list': 'hashtag_list',
                        'cover_url': 'cover_url',
                        'music_name': 'music_name',
                        'duration_sec': 'duration_sec',
                        # 新增主表字段
                        'author_follower_count': 'author_follower_count',
                        'author_total_favorited': 'author_total_favorited',
                        'author_signature': 'author_signature',
                        'author_verification_type': 'author_verification_type',
                        'video_cover_url': 'video_cover_url',
                        'dynamic_cover_url': 'dynamic_cover_url',
                        'origin_cover_url': 'origin_cover_url'
                    }

                    for browser_field, parsed_field in field_mapping.items():
                        if browser_field in browser_extracted_fields:
                            value = browser_extracted_fields[browser_field]
                            if value is not None:
                                # Type conversion for WebVideoMeta schema compatibility
                                if parsed_field in ['video_id', 'author_id', 'author_name', 'desc_text']:
                                    # Convert to string
                                    if not isinstance(value, str):
                                        value = str(value)
                                        logger.debug(f"Converted {parsed_field} to string")

                                elif parsed_field == 'cover_url':
                                    # Ensure string URL
                                    if isinstance(value, list):
                                        # List of dicts, try to extract URL from first item
                                        if value and isinstance(value[0], dict):
                                            first = value[0]
                                            if 'url_list' in first and isinstance(first['url_list'], list) and first['url_list']:
                                                value = first['url_list'][0]
                                            elif 'url' in first:
                                                value = first['url']
                                            elif 'cover_url' in first:
                                                value = first['cover_url']
                                            elif 'cover' in first:
                                                value = first['cover']
                                            else:
                                                logger.warning(f"cover_url list item doesn't contain url key: {first}")
                                                value = str(value)
                                        else:
                                            logger.warning(f"cover_url list doesn't contain dicts: {value}")
                                            value = str(value)
                                    elif isinstance(value, dict):
                                        # Try to extract URL from common keys
                                        if 'url' in value:
                                            value = value['url']
                                        elif 'cover_url' in value:
                                            value = value['cover_url']
                                        elif 'cover' in value:
                                            value = value['cover']
                                        else:
                                            logger.warning(f"cover_url dict doesn't contain url key: {value}")
                                            value = str(value)
                                    if not isinstance(value, str):
                                        value = str(value)
                                    logger.debug(f"Processed cover_url: {value[:100]}")

                                elif parsed_field in ['publish_time_raw', 'like_count_raw', 'comment_count_raw', 'share_count_raw']:
                                    # Convert to string (raw counts and timestamps should be strings)
                                    if not isinstance(value, str):
                                        value = str(value)
                                        logger.debug(f"Converted {parsed_field} to string")

                                elif parsed_field == 'hashtag_list':
                                    # Ensure list of strings
                                    if isinstance(value, str):
                                        # Try to parse JSON string
                                        try:
                                            parsed = json.loads(value)
                                            if isinstance(parsed, list):
                                                value = parsed
                                            else:
                                                logger.warning(f"hashtag_list JSON string is not a list: {value[:100]}")
                                                value = []
                                        except json.JSONDecodeError:
                                            logger.warning(f"hashtag_list is string but not JSON: {value[:100]}")
                                            value = [value]  # Treat as single hashtag
                                    if isinstance(value, list):
                                        # Convert list elements to strings, extract from dicts if needed
                                        processed = []
                                        for item in value:
                                            if isinstance(item, str):
                                                processed.append(item)
                                            elif isinstance(item, dict):
                                                # Try to extract hashtag name
                                                if 'hashtag_name' in item:
                                                    processed.append(str(item['hashtag_name']))
                                                elif 'name' in item:
                                                    processed.append(str(item['name']))
                                                else:
                                                    logger.warning(f"Unhandled dict item in hashtag_list: {item}")
                                            else:
                                                processed.append(str(item))
                                        value = processed
                                        logger.debug(f"Processed hashtag_list to list of strings, count={len(value)}")
                                    elif isinstance(value, dict):
                                        # Convert dict to list of keys or values? Probably not expected
                                        logger.warning(f"hashtag_list is dict, converting to list of keys")
                                        value = list(value.keys())
                                    else:
                                        logger.warning(f"Unhandled type for hashtag_list: {type(value)}, converting to empty list")
                                        value = []

                                elif parsed_field == 'author_profile_url':
                                    # Convert to string
                                    if not isinstance(value, str):
                                        value = str(value)
                                    logger.debug(f"Processed author_profile_url: {value[:100]}")

                                elif parsed_field == 'collect_count':
                                    # Convert to integer if possible
                                    if isinstance(value, (int, float)):
                                        value = int(value)
                                    elif isinstance(value, str):
                                        try:
                                            value = int(value)
                                        except ValueError:
                                            logger.warning(f"collect_count string cannot be converted to int: {value}")
                                            value = None
                                    else:
                                        logger.warning(f"Unhandled type for collect_count: {type(value)}, setting to None")
                                        value = None
                                    logger.debug(f"Processed collect_count: {value}")

                                elif parsed_field == 'music_name':
                                    # Convert to string
                                    if not isinstance(value, str):
                                        value = str(value)
                                    logger.debug(f"Processed music_name: {value[:100]}")

                                elif parsed_field == 'duration_sec':
                                    # Convert to integer if possible
                                    if isinstance(value, (int, float)):
                                        value = int(value)
                                    elif isinstance(value, str):
                                        try:
                                            value = int(value)
                                        except ValueError:
                                            logger.warning(f"duration_sec string cannot be converted to int: {value}")
                                            value = None
                                    else:
                                        logger.warning(f"Unhandled type for duration_sec: {type(value)}, setting to None")
                                        value = None
                                    logger.debug(f"Processed duration_sec: {value}")

                                # 新增字段类型转换
                                elif parsed_field in ['author_follower_count', 'author_total_favorited', 'author_verification_type']:
                                    # Convert to integer if possible
                                    if isinstance(value, (int, float)):
                                        value = int(value)
                                    elif isinstance(value, str):
                                        try:
                                            value = int(value)
                                        except ValueError:
                                            logger.warning(f"{parsed_field} string cannot be converted to int: {value}")
                                            value = None
                                    else:
                                        logger.warning(f"Unhandled type for {parsed_field}: {type(value)}, setting to None")
                                        value = None
                                    logger.debug(f"Processed {parsed_field}: {value}")

                                elif parsed_field in ['author_signature']:
                                    # Convert to string
                                    if not isinstance(value, str):
                                        value = str(value)
                                    logger.debug(f"Processed {parsed_field}: {value[:100]}")

                                elif parsed_field in ['video_cover_url', 'dynamic_cover_url', 'origin_cover_url']:
                                    # Similar processing as cover_url
                                    if isinstance(value, list):
                                        # List of dicts, try to extract URL from first item
                                        if value and isinstance(value[0], dict):
                                            first = value[0]
                                            if 'url_list' in first and isinstance(first['url_list'], list) and first['url_list']:
                                                value = first['url_list'][0]
                                            elif 'url' in first:
                                                value = first['url']
                                            elif 'cover_url' in first:
                                                value = first['cover_url']
                                            elif 'cover' in first:
                                                value = first['cover']
                                            else:
                                                logger.warning(f"{parsed_field} list item doesn't contain url key: {first}")
                                                value = str(value)
                                        else:
                                            logger.warning(f"{parsed_field} list doesn't contain dicts: {value}")
                                            value = str(value)
                                    elif isinstance(value, dict):
                                        # Try to extract URL from common keys
                                        if 'url' in value:
                                            value = value['url']
                                        elif 'cover_url' in value:
                                            value = value['cover_url']
                                        elif 'cover' in value:
                                            value = value['cover']
                                        elif 'uri' in value:
                                            # uri field, might need to construct URL
                                            value = value['uri']
                                            logger.debug(f"Extracted uri from {parsed_field}: {value}")
                                        elif 'url_list' in value and isinstance(value['url_list'], list) and value['url_list']:
                                            # Take first URL from url_list
                                            value = value['url_list'][0]
                                            logger.debug(f"Extracted first URL from url_list in {parsed_field}")
                                        else:
                                            logger.warning(f"{parsed_field} dict doesn't contain url key: {value}")
                                            value = str(value)
                                    if not isinstance(value, str):
                                        value = str(value)
                                    logger.debug(f"Processed {parsed_field}: {value[:100]}")

                                parsed_data[parsed_field] = value
                                logger.debug(f"Updated {parsed_field} from browser runtime data: {str(value)[:100]}")

                    # Update parse status to indicate browser data was used
                    parsed_data['parse_status'] = 'success'  # RawWebVideoData validator only allows 'success', 'partial_success', 'fail'
                    if browser_extraction_summary:
                        parsed_data['browser_extraction_summary'] = browser_extraction_summary
                        # Extract match metadata for WebVideoMeta schema
                        match_metadata_fields = ['match_type', 'confidence', 'selected_reason', 'is_primary_match', 'target_video_id', 'primary_source_key', 'matched_object_id']
                        for field in match_metadata_fields:
                            if field in browser_extraction_summary:
                                parsed_data[field] = browser_extraction_summary[field]
                                logger.debug(f"Added match metadata to parsed_data: {field} = {browser_extraction_summary[field]}")

                    # Log detailed field information after merging browser data
                    logger.info("Field details after merging browser data:")
                    target_fields = ['video_id', 'author_id', 'author_name', 'author_profile_url', 'desc_text',
                                    'publish_time_raw', 'like_count_raw', 'comment_count_raw',
                                    'share_count_raw', 'collect_count', 'hashtag_list', 'cover_url',
                                    'music_name', 'duration_sec']
                    for field in target_fields:
                        if field in parsed_data:
                            value = parsed_data[field]
                            logger.info(f"  {field}: value='{value}', type={type(value).__name__}")

                # Create raw data record
                raw_data = self._create_raw_data(
                    task=task,
                    response=response,
                    parsed_data=parsed_data,
                    crawl_time=crawl_time,
                    fetched_url=fetched_url,
                    html_content=html_content,
                    http_status=http_status,
                    response_headers=response_headers
                )

                # Create metadata record
                logger.info(f"Calling create_web_video_meta with parsed_data keys: {list(parsed_data.keys())}")
                meta_record = self.parser.create_web_video_meta(parsed_data)
                if meta_record:
                    logger.info(f"WebVideoMeta created successfully, video_id: {meta_record.video_id}")
                else:
                    logger.warning("create_web_video_meta returned None")

                # Save results
                if raw_data:
                    self._save_raw_data(raw_data)

                if meta_record:
                    self._save_metadata(meta_record)

                # Log parsing summary for evidence (regardless of meta_record creation)
                self._log_parsing_summary(task.url, parsed_data, meta_record)

                task.mark_completed()
                with self._lock:
                    self.completed_tasks.append(task)

                logger.info(f"Worker {worker_id} completed: {task.url}")

            except Exception as e:
                logger.error(f"Worker {worker_id} error processing {task.url}: {e}")
                task.mark_failed()
                with self._lock:
                    self.failed_tasks.append(task)
            finally:
                self.task_queue.task_done()

    def _create_raw_data(self, task: CrawlTask, response, parsed_data: Dict,
                         crawl_time: datetime, fetched_url: str,
                         html_content: Optional[str] = None,
                         http_status: Optional[int] = None,
                         response_headers: Optional[Dict] = None) -> Optional[RawWebVideoData]:
        """Create raw web video data record.

        Args:
            task: Crawl task.
            response: Response object.
            parsed_data: Parsed data.
            crawl_time: Crawl timestamp.
            fetched_url: The URL that was actually fetched.

        Returns:
            RawWebVideoData object or None.
        """
        try:
            # Generate crawl ID
            crawl_id = f"crawl_{crawl_time.strftime('%Y%m%d_%H%M%S')}_{hash(task.url) % 10000:04d}"

            # Determine HTML content
            html_to_save = html_content
            if html_to_save is None:
                if isinstance(response, dict):
                    html_to_save = response.get('html', '')
                else:
                    html_to_save = response.text if response else ''

            # Determine HTTP status
            status = http_status
            if status is None:
                if isinstance(response, dict):
                    status = response.get('status', 200)
                else:
                    status = response.status_code if response else 0

            # Save HTML if configured
            raw_html_path = None
            rendered_html_path = None
            if self.save_raw_html and html_to_save:
                if self.use_browser and not self.use_mock:
                    # Save rendered HTML for browser mode
                    rendered_dir = self.output_dir / "rendered_html" / self.run_id
                    rendered_dir.mkdir(parents=True, exist_ok=True)
                    rendered_file = rendered_dir / f"{crawl_id}.html"
                    rendered_file.write_text(html_to_save, encoding='utf-8')
                    rendered_html_path = rendered_file
                    logger.info(f"Rendered HTML saved to: {rendered_file}")
                else:
                    # Save raw HTML for regular mode
                    html_dir = self.output_dir / "html" / self.run_id
                    html_dir.mkdir(parents=True, exist_ok=True)
                    html_file = html_dir / f"{crawl_id}.html"
                    html_file.write_text(html_to_save, encoding='utf-8')
                    raw_html_path = html_file
                    logger.info(f"Raw HTML saved to: {html_file}")

            return RawWebVideoData(
                crawl_id=crawl_id,
                source_entry=task.source_entry,
                original_url=task.original_url,
                canonical_video_url=task.canonical_video_url,
                fetched_url=fetched_url,
                page_type=task.page_type,
                page_url=task.url,  # Display URL (canonical if available, else original)
                raw_html_path=raw_html_path,
                rendered_html_path=rendered_html_path,
                raw_json_blob=None,  # TODO: Extract JSON blob from HTML
                http_status=status,
                crawl_time=crawl_time,
                parse_status=parsed_data.get('parse_status', 'fail'),
                parse_error_msg=parsed_data.get('parse_error_msg')
            )
        except Exception as e:
            logger.error(f"Failed to create raw data: {e}")
            return None

    def _save_raw_data(self, raw_data: RawWebVideoData):
        """Save raw data to file.

        Args:
            raw_data: RawWebVideoData object.
        """
        try:
            raw_filename = f"{self.output_prefix}crawl_{self.file_suffix}.jsonl"
            raw_file = self.output_dir / raw_filename
            write_jsonl(raw_file, [raw_data.dict()], mode='a')
            logger.debug(f"Raw data saved to {raw_file}")
        except Exception as e:
            logger.error(f"Failed to save raw data: {e}")

    def _save_metadata(self, metadata: WebVideoMeta):
        """Save metadata to file.

        Args:
            metadata: WebVideoMeta object.
        """
        try:
            # Log metadata fields before saving
            logger.info("Metadata fields before saving to CSV:")
            data_dict = metadata.dict()
            for field, value in data_dict.items():
                logger.info(f"  {field}: value='{value}', type={type(value).__name__}")

            # Save to interim directory as CSV (primary output for real crawl)
            meta_filename = f"{self.output_prefix}web_video_meta_{self.file_suffix}.csv"
            meta_file = self.interim_run_dir / meta_filename
            write_csv(meta_file, [metadata.dict()], mode='a', index=False)
            logger.info(f"Metadata saved to {meta_file}")

            # Also save as Parquet for compatibility (optional)
            # Uncomment if needed
            # parquet_file = self.interim_dir / f"{self.output_prefix}web_video_meta_{self.file_suffix}.parquet"
            # write_parquet(parquet_file, [metadata.dict()], mode='a')
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def start(self, num_workers: Optional[int] = None):
        """Start crawling with specified number of workers.

        Args:
            num_workers: Number of worker threads. Uses max_workers if None.
        """
        if num_workers is None:
            num_workers = self.max_workers

        logger.info(f"Starting scheduler with {num_workers} workers")
        logger.info(f"Queue size: {self.task_queue.qsize()}")

        workers = []
        for i in range(num_workers):
            worker_thread = threading.Thread(
                target=self.worker,
                args=(i,),
                daemon=True
            )
            worker_thread.start()
            workers.append(worker_thread)

        # Wait for queue to be empty
        self.task_queue.join()

        # Stop workers
        self._stop_event.set()
        for worker in workers:
            worker.join(timeout=5)

        logger.info("Scheduler finished")
        self._print_summary()
        # Filter high-confidence samples after crawl completion
        self._filter_and_save_high_confidence_samples()
        # Generate quality report
        self._generate_quality_report()

    def stop(self):
        """Stop the scheduler."""
        self._stop_event.set()
        logger.info("Scheduler stopped")

    def close(self):
        """Close all clients and clean up resources."""
        try:
            if self.client:
                self.client.close()
            if self.browser_client:
                self.browser_client.close()
            logger.info("All clients closed")
        except Exception as e:
            logger.warning(f"Error closing clients: {e}")

    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        self.close()

    def _log_parsing_summary(self, url: str, parsed_data: Dict, meta_record):
        """Log parsing summary for evidence.

        Args:
            url: Page URL.
            parsed_data: Parsed data dictionary.
            meta_record: WebVideoMeta object.
        """
        # Fields requested by user for evidence
        evidence_fields = [
            'video_id', 'page_url', 'desc_text', 'author_name',
            'publish_time_raw', 'like_count_raw', 'comment_count_raw',
            'share_count_raw', 'hashtag_list', 'cover_url'
        ]

        summary_lines = [f"Parsing summary for {url}:"]
        for field in evidence_fields:
            value = parsed_data.get(field)
            if value is None or value == '':
                summary_lines.append(f"  {field}: null")
            else:
                # Truncate long values for readability
                if field == 'desc_text' and len(str(value)) > 100:
                    value = str(value)[:100] + "..."
                summary_lines.append(f"  {field}: {value}")

        # Add parse status
        parse_status = parsed_data.get('parse_status', 'unknown')
        summary_lines.append(f"  parse_status: {parse_status}")

        logger.info("\n".join(summary_lines))

    def _print_summary(self):
        """Print summary of crawl results."""
        total = len(self.completed_tasks) + len(self.failed_tasks)
        if total == 0:
            logger.info("No tasks processed")
            return

        success_rate = len(self.completed_tasks) / total * 100
        logger.info(f"Crawl summary: {len(self.completed_tasks)} completed, "
                   f"{len(self.failed_tasks)} failed ({success_rate:.1f}% success)")

    def _filter_and_save_high_confidence_samples(self):
        """Filter high-confidence samples from interim CSV and save to processed directory."""
        try:
            # Find the interim CSV file
            csv_filename = f"{self.output_prefix}web_video_meta_{self.file_suffix}.csv"
            csv_path = self.interim_run_dir / csv_filename

            if not csv_path.exists():
                logger.warning(f"Interim CSV file not found: {csv_path}")
                return

            logger.info(f"Filtering high-confidence samples from {csv_path}")

            # Read all records
            records = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(row)

            if not records:
                logger.info("No records to filter")
                return

            logger.info(f"Total records loaded: {len(records)}")

            # Helper function to extract video ID from URL
            def extract_video_id_from_url(url: str) -> Optional[str]:
                """Extract video ID from Douyin URL."""
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

            # Filter criteria
            high_confidence_records = []
            for row in records:
                # Check match_type and confidence
                match_type = row.get('match_type')
                confidence = row.get('confidence')

                if match_type != 'exact' or confidence != 'high':
                    continue

                # Check video_id consistency with page_url
                video_id = row.get('video_id')
                page_url = row.get('page_url')
                if not video_id or not page_url:
                    continue

                target_id = extract_video_id_from_url(page_url)
                if target_id and video_id == target_id:
                    high_confidence_records.append(row)
                else:
                    # video_id may already be target_id from URL extraction
                    # If extraction failed, still accept if video_id is not empty
                    # but log warning
                    logger.debug(f"Video ID mismatch: video_id={video_id}, target_id={target_id}")

            logger.info(f"High-confidence samples found: {len(high_confidence_records)}")

            # Save high-confidence samples to processed directory
            if high_confidence_records:
                output_filename = f"high_confidence_web_video_meta_{self.file_suffix}.csv"
                output_path = self.processed_run_dir / output_filename

                with open(output_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=high_confidence_records[0].keys())
                    writer.writeheader()
                    writer.writerows(high_confidence_records)

                logger.info(f"High-confidence samples saved to {output_path}")
            else:
                logger.info("No high-confidence samples to save")

        except Exception as e:
            logger.error(f"Failed to filter high-confidence samples: {e}")

    def _generate_quality_report(self):
        """Generate quality statistics report for the crawl run."""
        try:
            # Find the interim CSV file
            csv_filename = f"{self.output_prefix}web_video_meta_{self.file_suffix}.csv"
            csv_path = self.interim_run_dir / csv_filename

            if not csv_path.exists():
                logger.warning(f"Interim CSV file not found: {csv_path}")
                return

            logger.info(f"Generating quality report from {csv_path}")

            # Read all records
            records = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(row)

            if not records:
                logger.info("No records for quality report")
                return

            total_records = len(records)
            logger.info(f"Total records for quality analysis: {total_records}")

            # Helper function to extract video ID from URL
            def extract_video_id_from_url(url: str) -> Optional[str]:
                """Extract video ID from Douyin URL."""
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

            # Initialize counters
            match_type_counts = {'exact': 0, 'partial': 0, 'none': 0, 'unknown': 0}
            confidence_counts = {'high': 0, 'medium': 0, 'low': 0, 'unknown': 0}
            video_id_consistent = 0
            video_id_short_or_abnormal = 0
            video_id_mismatch = 0
            cross_page_mixing_suspected = 0  # Placeholder for cross-page mixing detection

            for row in records:
                # Count match_type
                match_type = row.get('match_type')
                if match_type in match_type_counts:
                    match_type_counts[match_type] += 1
                else:
                    match_type_counts['unknown'] += 1

                # Count confidence
                confidence = row.get('confidence')
                if confidence in confidence_counts:
                    confidence_counts[confidence] += 1
                else:
                    confidence_counts['unknown'] += 1

                # Check video_id consistency with page_url
                video_id = row.get('video_id')
                page_url = row.get('page_url')
                if video_id and page_url:
                    target_id = extract_video_id_from_url(page_url)
                    if target_id:
                        if video_id == target_id:
                            video_id_consistent += 1
                        else:
                            video_id_mismatch += 1
                    # else: cannot extract target_id, skip consistency check

                # Detect short or abnormal video_id (e.g., very short numeric values)
                if video_id:
                    # Check if video_id is suspiciously short (<= 3 chars) and numeric
                    if len(video_id) <= 3 and video_id.isdigit():
                        video_id_short_or_abnormal += 1
                    # Check if video_id contains non-digit characters but very short
                    elif len(video_id) <= 3:
                        video_id_short_or_abnormal += 1

                # Cross-page mixing detection (simplistic heuristic)
                # If video_id doesn't match target_id and match_type is none/low confidence,
                # might indicate cross-page mixing
                if match_type == 'none' and confidence == 'low':
                    cross_page_mixing_suspected += 1

            # Calculate high-confidence samples (exact + high)
            high_confidence_samples = 0
            for row in records:
                if row.get('match_type') == 'exact' and row.get('confidence') == 'high':
                    high_confidence_samples += 1

            # Calculate percentages
            def safe_percent(count, total):
                return count / total * 100 if total > 0 else 0.0

            # Generate report lines
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append(f"Quality Report for Run ID: {self.run_id}")
            report_lines.append("=" * 80)
            report_lines.append(f"Total URL processed: {self.task_queue.qsize() + len(self.completed_tasks) + len(self.failed_tasks)}")
            report_lines.append(f"Successfully crawled: {len(self.completed_tasks)}")
            report_lines.append(f"Failed to crawl: {len(self.failed_tasks)}")
            report_lines.append(f"Records generated: {total_records}")
            report_lines.append("")
            report_lines.append("Match Type Distribution:")
            for mt in ['exact', 'partial', 'none', 'unknown']:
                count = match_type_counts[mt]
                percent = safe_percent(count, total_records)
                report_lines.append(f"  {mt}: {count} ({percent:.1f}%)")
            report_lines.append("")
            report_lines.append("Confidence Distribution:")
            for conf in ['high', 'medium', 'low', 'unknown']:
                count = confidence_counts[conf]
                percent = safe_percent(count, total_records)
                report_lines.append(f"  {conf}: {count} ({percent:.1f}%)")
            report_lines.append("")
            report_lines.append(f"High-confidence samples (exact+high): {high_confidence_samples} ({safe_percent(high_confidence_samples, total_records):.1f}%)")
            report_lines.append(f"Video ID consistent with page_url: {video_id_consistent} ({safe_percent(video_id_consistent, total_records):.1f}%)")
            report_lines.append(f"Video ID mismatch: {video_id_mismatch} ({safe_percent(video_id_mismatch, total_records):.1f}%)")
            report_lines.append(f"Short/abnormal video_id (<=3 chars): {video_id_short_or_abnormal} ({safe_percent(video_id_short_or_abnormal, total_records):.1f}%)")
            report_lines.append(f"Cross-page mixing suspected: {cross_page_mixing_suspected} ({safe_percent(cross_page_mixing_suspected, total_records):.1f}%)")
            report_lines.append("")
            report_lines.append("Run Stability Summary:")
            report_lines.append(f"  URLs failed: {len(self.failed_tasks)}")
            report_lines.append(f"  Pages with no candidates: {match_type_counts['none']}")
            report_lines.append(f"  Pages with only low confidence: {confidence_counts['low']}")
            report_lines.append("")
            report_lines.append("Overall Assessment:")
            if high_confidence_samples / total_records > 0.7:
                report_lines.append("  ✅ High-confidence sample ratio is good (>70%)")
            else:
                report_lines.append("  ⚠️ High-confidence sample ratio is low (<70%)")
            if video_id_consistent / total_records > 0.8:
                report_lines.append("  ✅ Video ID consistency is good (>80%)")
            else:
                report_lines.append("  ⚠️ Video ID consistency needs improvement")
            if video_id_short_or_abnormal == 0:
                report_lines.append("  ✅ No short/abnormal video IDs detected")
            else:
                report_lines.append(f"  ⚠️ {video_id_short_or_abnormal} short/abnormal video IDs detected")
            if cross_page_mixing_suspected == 0:
                report_lines.append("  ✅ No obvious cross-page mixing detected")
            else:
                report_lines.append(f"  ⚠️ {cross_page_mixing_suspected} records suspected of cross-page mixing")
            report_lines.append("=" * 80)

            # Log report
            for line in report_lines:
                logger.info(line)

            # Save report to debug directory
            debug_dir = self.output_dir / "debug" / self.run_id
            debug_dir.mkdir(parents=True, exist_ok=True)
            report_file = debug_dir / f"quality_report_{self.run_id}.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(report_lines))
            logger.info(f"Quality report saved to {report_file}")

        except Exception as e:
            logger.error(f"Failed to generate quality report: {e}")

    def mock_run(self, urls: List[str], source_entry: str = 'manual_url'):
        """Run in mock mode (no network requests).

        Args:
            urls: List of URLs to mock.
            source_entry: Source entry type.
        """
        logger.info(f"Running in mock mode with {len(urls)} URLs")

        for url in urls:
            try:
                # Get page type for mock parsing
                normalized = self.parser.normalize_douyin_url(url)
                page_type = normalized['page_type']

                # Create mock metadata
                meta_record = self.parser.mock_parse(url, source_entry, page_type)

                # Save mock data
                if meta_record:
                    self._save_metadata(meta_record)
                    logger.info(f"Mock data saved for {url} (page_type: {page_type})")
                else:
                    logger.warning(f"Failed to create mock data for {url}")

            except Exception as e:
                logger.error(f"Error in mock run for {url}: {e}")

        logger.info("Mock run completed")
        # Filter high-confidence samples after mock run
        self._filter_and_save_high_confidence_samples()