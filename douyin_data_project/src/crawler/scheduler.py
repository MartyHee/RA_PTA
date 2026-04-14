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
from .parser import DouyinParser
from ..schemas.tables import RawWebVideoData, WebVideoMeta
from ..utils.config_loader import load_config, get_config
from ..utils.logger import get_logger
from ..utils.io_utils import write_jsonl, write_parquet, write_csv

logger = get_logger(__name__)


class CrawlTask:
    """Represents a single crawl task."""

    def __init__(self, url: str, source_entry: str, priority: int = 0):
        self.url = url
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

    def __init__(self, config_path: Optional[Path] = None, use_mock: bool = False):
        """Initialize scheduler.

        Args:
            config_path: Path to config file.
            use_mock: Whether to use mock mode.
        """
        self.config = load_config(config_path) if config_path else load_config()
        self.use_mock = use_mock

        self.task_queue = Queue()
        self.completed_tasks = []
        self.failed_tasks = []

        self.client = DouyinClient(config_path, use_mock)
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

        task = CrawlTask(url, source_entry, priority)
        self.task_queue.put(task)
        logger.debug(f"Added task: {url} ({source_entry})")

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

                # Crawl
                crawl_time = datetime.now()
                response = self.client.get(task.url)

                if response is None:
                    logger.error(f"Failed to fetch {task.url}")
                    task.mark_failed()
                    with self._lock:
                        self.failed_tasks.append(task)
                    continue

                # Log crawl evidence
                logger.info(f"Crawl evidence - URL: {task.url}, "
                           f"HTTP Status: {response.status_code}, "
                           f"Final URL: {response.url}, "
                           f"Content length: {len(response.text)} chars, "
                           f"Content-Encoding: {response.headers.get('Content-Encoding', 'none')}")

                # Parse
                parsed_data = self.parser.parse_html(
                    html=response.text,
                    url=task.url,
                    source_entry=task.source_entry,
                    crawl_time=crawl_time
                )

                # Create raw data record
                raw_data = self._create_raw_data(
                    task=task,
                    response=response,
                    parsed_data=parsed_data,
                    crawl_time=crawl_time
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
                         crawl_time: datetime) -> Optional[RawWebVideoData]:
        """Create raw web video data record.

        Args:
            task: Crawl task.
            response: Response object.
            parsed_data: Parsed data.
            crawl_time: Crawl timestamp.

        Returns:
            RawWebVideoData object or None.
        """
        try:
            # Generate crawl ID
            crawl_id = f"crawl_{crawl_time.strftime('%Y%m%d_%H%M%S')}_{hash(task.url) % 10000:04d}"

            # Save HTML if configured
            raw_html_path = None
            if self.save_raw_html:
                html_dir = self.output_dir / "html"
                html_dir.mkdir(exist_ok=True)
                html_file = html_dir / f"{crawl_id}.html"
                html_file.write_text(response.text, encoding='utf-8')
                raw_html_path = html_file
                logger.info(f"Raw HTML saved to: {html_file}")

            return RawWebVideoData(
                crawl_id=crawl_id,
                source_entry=task.source_entry,
                page_url=task.url,
                raw_html_path=raw_html_path,
                raw_json_blob=None,  # TODO: Extract JSON blob from HTML
                http_status=response.status_code,
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
                # Create mock metadata
                meta_record = self.parser.mock_parse(url, source_entry)

                # Save mock data
                if meta_record:
                    self._save_metadata(meta_record)
                    logger.info(f"Mock data saved for {url}")
                else:
                    logger.warning(f"Failed to create mock data for {url}")

            except Exception as e:
                logger.error(f"Error in mock run for {url}: {e}")

        logger.info("Mock run completed")