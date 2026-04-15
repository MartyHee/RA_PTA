"""
Scheduler for managing crawl tasks.

Supports queueing multiple URLs, rate limiting, and task tracking.
"""
import time
import threading
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

        # Initialize appropriate client based on mode
        if use_mock:
            # Mock mode uses DouyinClient with mock responses
            self.client = DouyinClient(config_path, use_mock=True)
            self.browser_client = None
        elif use_browser:
            # Browser mode uses BrowserClient for JavaScript rendering
            self.client = None
            self.browser_client = BrowserClient(config_path, use_mock=False)
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

        # Generate run ID and output prefix
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.use_mock:
            self.output_prefix = "mock_"
            self.file_suffix = self.run_id
        else:
            # Get prefix from config, default to "real_"
            self.output_prefix = get_config('sources.web.real_crawl.output_prefix', 'real_')
            self.file_suffix = self.run_id

        self._lock = threading.Lock()
        self._stop_event = threading.Event()

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
                parsed_data = self.parser.parse_html(
                    html=html_content if html_content else (response.text if response else ''),
                    url=task.url,
                    source_entry=task.source_entry,
                    crawl_time=crawl_time,
                    page_type=task.page_type
                )

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
                meta_record = self.parser.create_web_video_meta(parsed_data)

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
                    rendered_dir = self.output_dir / "rendered_html"
                    rendered_dir.mkdir(exist_ok=True)
                    rendered_file = rendered_dir / f"{crawl_id}.html"
                    rendered_file.write_text(html_to_save, encoding='utf-8')
                    rendered_html_path = rendered_file
                    logger.info(f"Rendered HTML saved to: {rendered_file}")
                else:
                    # Save raw HTML for regular mode
                    html_dir = self.output_dir / "html"
                    html_dir.mkdir(exist_ok=True)
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
            # Save to interim directory as CSV (primary output for real crawl)
            meta_filename = f"{self.output_prefix}web_video_meta_{self.file_suffix}.csv"
            meta_file = self.interim_dir / meta_filename
            write_csv(meta_file, [metadata.dict()], mode='a', index=False)
            logger.debug(f"Metadata saved to {meta_file}")

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