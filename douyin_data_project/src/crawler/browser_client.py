"""
Browser client for Douyin web crawler using Playwright.

Handles JavaScript-rendered pages by using a headless browser.
Supports both headless and headed modes for debugging.
"""
import time
import random
import json
import re
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import logging
from datetime import datetime

from ..utils.config_loader import load_config, get_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BrowserClient:
    """Browser client for Douyin using Playwright."""

    def __init__(self, config_path: Optional[Path] = None, use_mock: bool = False):
        """Initialize the browser client.

        Args:
            config_path: Path to config file. If None, loads default.
            use_mock: Whether to use mock mode (no actual browser).
        """
        self.config = load_config(config_path) if config_path else load_config()
        sources_mock = get_config('sources.web.mock.enabled', False)
        self.use_mock = use_mock or sources_mock

        self.request_timeout = get_config('settings.crawler.browser_timeout', 60)  # Longer timeout for browser
        self.max_retries = get_config('settings.crawler.max_retries', 2)  # Fewer retries due to longer runtime
        self.retry_delay = get_config('settings.crawler.retry_delay', 3)
        self.delay_between_requests = get_config('settings.crawler.delay_between_requests', 3.0)  # Longer delay for browser
        self.user_agent = get_config('settings.crawler.user_agent', '')

        # Browser-specific settings
        self.headless = get_config('settings.crawler.browser.headless', True)
        self.viewport_width = get_config('settings.crawler.browser.viewport_width', 1920)
        self.viewport_height = get_config('settings.crawler.browser.viewport_height', 1080)
        self.wait_for_selector = get_config('settings.crawler.browser.wait_for_selector', 'body')
        self.wait_timeout = get_config('settings.crawler.browser.wait_timeout', 30000)  # ms
        self.scroll_to_bottom = get_config('settings.crawler.browser.scroll_to_bottom', False)
        self.scroll_delay = get_config('settings.crawler.browser.scroll_delay', 1.0)

        self.last_request_time = 0
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

        # Debug data storage
        self._network_responses = []
        self._runtime_objects = {}
        self._debug_output_dir = None

        # Initialize only when needed
        self._initialized = False

    def _initialize_browser(self):
        """Initialize Playwright browser if not already initialized."""
        if self._initialized and self._browser and self._browser.is_connected():
            return

        try:
            import playwright.sync_api
            self._playwright = playwright.sync_api.sync_playwright().start()

            # Launch browser
            browser_type = get_config('settings.crawler.browser.type', 'chromium')
            launch_options = {
                'headless': self.headless,
                'args': [
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                ]
            }

            if browser_type == 'chromium':
                self._browser = self._playwright.chromium.launch(**launch_options)
            elif browser_type == 'firefox':
                self._browser = self._playwright.firefox.launch(**launch_options)
            elif browser_type == 'webkit':
                self._browser = self._playwright.webkit.launch(**launch_options)
            else:
                logger.warning(f"Unknown browser type: {browser_type}, using chromium")
                self._browser = self._playwright.chromium.launch(**launch_options)

            # Create context
            context_options = {
                'viewport': {'width': self.viewport_width, 'height': self.viewport_height},
                'user_agent': self.user_agent if self.user_agent else (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/120.0.0.0 Safari/537.36'
                ),
                'ignore_https_errors': True,
            }
            self._context = self._browser.new_context(**context_options)

            # Create page
            self._page = self._context.new_page()
            self._initialized = True
            logger.info("Browser initialized successfully")

        except ImportError:
            logger.error("Playwright not installed. Please install with: pip install playwright && playwright install")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            self._close_browser()
            raise

    def _close_browser(self):
        """Close browser resources."""
        try:
            if self._page:
                self._page.close()
                self._page = None
            if self._context:
                self._context.close()
                self._context = None
            if self._browser:
                self._browser.close()
                self._browser = None
            if self._playwright:
                self._playwright.stop()
                self._playwright = None
            self._initialized = False
            logger.debug("Browser closed")
        except Exception as e:
            logger.warning(f"Error closing browser: {e}")

    def _setup_debug_output(self, url: str):
        """Set up debug output directory for network and runtime data.

        Args:
            url: Target URL for naming.
        """
        try:
            # Create debug directory under raw data
            raw_data_dir = Path(get_config('settings.paths.raw_data', './data/raw'))
            debug_dir = raw_data_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            self._debug_output_dir = debug_dir

            # Clear previous debug data
            self._network_responses = []
            self._runtime_objects = {}

            logger.debug(f"Debug output directory: {debug_dir}")
        except Exception as e:
            logger.warning(f"Failed to set up debug output: {e}")
            self._debug_output_dir = None

    def _is_json_candidate_response(self, response) -> bool:
        """Check if a response is a candidate for JSON data extraction.

        Args:
            response: Playwright response object.

        Returns:
            True if response should be captured.
        """
        try:
            url = response.url
            content_type = response.headers.get('content-type', '').lower()

            # Priority keywords in URL
            priority_keywords = ['aweme', 'detail', 'video', 'item', 'feed', 'web/api', 'statistics']
            url_lower = url.lower()
            has_priority_keyword = any(keyword in url_lower for keyword in priority_keywords)

            # Check content type
            is_json_content = 'application/json' in content_type or 'text/json' in content_type

            # Also consider text/plain with JSON-like URLs
            is_text_content = 'text/plain' in content_type

            # Capture if: has priority keyword OR is JSON content
            return has_priority_keyword or is_json_content or is_text_content
        except Exception as e:
            logger.debug(f"Error checking response candidate: {e}")
            return False

    def _setup_network_listener(self):
        """Set up network response listener to capture JSON responses."""
        if not self._page:
            return

        def on_response(response):
            try:
                if not self._is_json_candidate_response(response):
                    return

                url = response.url
                status = response.status
                headers = response.headers

                # Try to get response body
                try:
                    body = response.text()
                    # Check if body looks like JSON
                    if body and (body.strip().startswith('{') or body.strip().startswith('[')):
                        # Try to parse to validate
                        json.loads(body)

                        # Store response info
                        response_data = {
                            'url': url,
                            'status': status,
                            'headers': dict(headers),
                            'body': body,
                            'timestamp': datetime.now().isoformat()
                        }
                        self._network_responses.append(response_data)
                        logger.debug(f"Captured JSON response from {url} ({status})")

                except json.JSONDecodeError:
                    # Not valid JSON, skip
                    pass
                except Exception as e:
                    logger.debug(f"Error processing response body from {url}: {e}")

            except Exception as e:
                logger.debug(f"Error in network listener: {e}")

        # Register listener
        self._page.on('response', on_response)
        logger.debug("Network listener set up")

    def _capture_runtime_objects(self):
        """Capture runtime JavaScript objects from the page."""
        if not self._page:
            return

        try:
            # Define objects to check
            target_objects = [
                'window.__INITIAL_STATE__',
                'window.__NEXT_DATA__',
                'window.SSR_RENDER_DATA',
                'window.RENDER_DATA',
                'window.__DATA__',
                'window.__NUXT__',
                'window.__REDUX_STATE__',
                'window.data',
                'window.videoData',
                'window.awemeData',
                'window.itemInfo'
            ]

            # Also check for any window property containing video/aweme/detail/state/store
            additional_check_script = """
            const objects = {};
            for (let key in window) {
                if (typeof window[key] === 'object' && window[key] !== null) {
                    const keyLower = key.toLowerCase();
                    if (keyLower.includes('video') || keyLower.includes('aweme') ||
                        keyLower.includes('detail') || keyLower.includes('state') ||
                        keyLower.includes('store') || keyLower.includes('data')) {
                        try {
                            objects[key] = window[key];
                        } catch(e) {
                            // Skip if cannot serialize
                        }
                    }
                }
            }
            return objects;
            """

            captured = {}

            # Check predefined objects
            for obj_path in target_objects:
                try:
                    result = self._page.evaluate(f"typeof {obj_path}")
                    if result != "undefined":
                        value = self._page.evaluate(f"JSON.stringify({obj_path})")
                        if value:
                            captured[obj_path] = json.loads(value)
                            logger.debug(f"Captured runtime object: {obj_path}")
                except Exception as e:
                    logger.debug(f"Error checking {obj_path}: {e}")

            # Check additional objects
            try:
                additional_objects = self._page.evaluate(additional_check_script)
                if additional_objects:
                    for key, value in additional_objects.items():
                        try:
                            captured[f"window.{key}"] = value
                        except:
                            pass
            except Exception as e:
                logger.debug(f"Error checking additional objects: {e}")

            self._runtime_objects = captured
            logger.info(f"Captured {len(captured)} runtime objects")

        except Exception as e:
            logger.warning(f"Failed to capture runtime objects: {e}")
            self._runtime_objects = {}

    def _save_debug_data(self, url: str) -> Optional[Dict[str, Any]]:
        """Save captured debug data to files.

        Args:
            url: Original URL for naming.

        Returns:
            Summary dictionary with field_extraction if successful, None otherwise.
        """
        if not self._debug_output_dir:
            logger.warning("No debug output directory set, skipping debug data save")
            return None

        try:
            # Create timestamp for file names
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            url_hash = hash(url) % 10000

            # Save network responses
            if self._network_responses:
                network_file = self._debug_output_dir / f"network_json_{url_hash}_{timestamp}.json"
                with open(network_file, 'w', encoding='utf-8') as f:
                    json.dump(self._network_responses, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(self._network_responses)} network responses to {network_file}")

            # Save runtime objects
            if self._runtime_objects:
                runtime_file = self._debug_output_dir / f"runtime_objects_{url_hash}_{timestamp}.json"
                with open(runtime_file, 'w', encoding='utf-8') as f:
                    json.dump(self._runtime_objects, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(self._runtime_objects)} runtime objects to {runtime_file}")

            # Also save combined analysis summary and return it
            return self._analyze_and_save_summary(url, timestamp, url_hash)

        except Exception as e:
            logger.warning(f"Failed to save debug data: {e}")
            return None

    def _analyze_and_save_summary(self, url: str, timestamp: str, url_hash: int) -> Dict[str, Any]:
        """Analyze captured data for video objects and save summary.

        Args:
            url: Original URL.
            timestamp: Timestamp string.
            url_hash: Hash of URL.

        Returns:
            Summary dictionary with field_extraction and other analysis results.
        """
        try:
            summary = {
                'url': url,
                'timestamp': timestamp,
                'network_responses_count': len(self._network_responses),
                'runtime_objects_count': len(self._runtime_objects),
                'video_object_candidates': [],
                'field_extraction': {},
                'field_sources': {}
            }

            # Combine all data sources for analysis
            all_data_sources = []

            # Add network responses
            for i, resp in enumerate(self._network_responses):
                try:
                    body = resp.get('body', '')
                    if body:
                        data = json.loads(body)
                        all_data_sources.append({
                            'type': 'network_response',
                            'index': i,
                            'url': resp.get('url'),
                            'data': data
                        })
                except:
                    pass

            # Add runtime objects
            for obj_path, obj_data in self._runtime_objects.items():
                all_data_sources.append({
                    'type': 'runtime_object',
                    'path': obj_path,
                    'data': obj_data
                })

            # Extract fields from each data source
            all_field_results = []
            video_candidates = []

            for source in all_data_sources:
                source_type = source['type']
                source_id = source.get('index', source.get('path', 'unknown'))
                data = source['data']

                # Extract fields using the new method
                field_mappings = self._extract_fields_from_data(data, base_path=f"{source_type}_{source_id}")

                # Check if any fields were found
                found_fields = {k: v for k, v in field_mappings.items() if v['value'] is not None}
                if found_fields:
                    all_field_results.append({
                        'source': f"{source_type}_{source_id}",
                        'fields': field_mappings
                    })

                # Also look for video objects using the old method for compatibility
                paths = self._find_video_objects(data, base_path=f"{source_type}_{source_id}")
                if paths:
                    video_candidates.extend(paths)

            summary['video_object_candidates'] = video_candidates

            # Merge field results - take the first non-null value for each field
            merged_fields = {
                'video_id': {'value': None, 'path': None},
                'author_id': {'value': None, 'path': None},
                'author_name': {'value': None, 'path': None},
                'desc_text': {'value': None, 'path': None},
                'publish_time_raw': {'value': None, 'path': None},
                'like_count_raw': {'value': None, 'path': None},
                'comment_count_raw': {'value': None, 'path': None},
                'share_count_raw': {'value': None, 'path': None},
                'hashtag_list': {'value': None, 'path': None},
                'cover_url': {'value': None, 'path': None}
            }

            field_sources = {}

            for result in all_field_results:
                for field_name, field_info in result['fields'].items():
                    if field_info['value'] is not None and merged_fields[field_name]['value'] is None:
                        merged_fields[field_name] = field_info
                        field_sources[field_name] = result['source']

            summary['field_extraction'] = merged_fields
            summary['field_sources'] = field_sources

            # Calculate extraction statistics
            extracted_count = sum(1 for field in merged_fields.values() if field['value'] is not None)
            summary['extracted_field_count'] = extracted_count

            # Save summary
            summary_file = self._debug_output_dir / f"video_analysis_{url_hash}_{timestamp}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved video analysis summary to {summary_file}")

            # Also save a human-readable summary
            self._save_human_readable_summary(url, timestamp, url_hash, summary)

            return summary

        except Exception as e:
            logger.warning(f"Failed to analyze and save summary: {e}")
            # Return empty summary on error
            return {
                'url': url,
                'timestamp': timestamp,
                'network_responses_count': 0,
                'runtime_objects_count': 0,
                'video_object_candidates': [],
                'field_extraction': {
                    'video_id': {'value': None, 'path': None},
                    'author_id': {'value': None, 'path': None},
                    'author_name': {'value': None, 'path': None},
                    'desc_text': {'value': None, 'path': None},
                    'publish_time_raw': {'value': None, 'path': None},
                    'like_count_raw': {'value': None, 'path': None},
                    'comment_count_raw': {'value': None, 'path': None},
                    'share_count_raw': {'value': None, 'path': None},
                    'hashtag_list': {'value': None, 'path': None},
                    'cover_url': {'value': None, 'path': None}
                },
                'field_sources': {},
                'extracted_field_count': 0
            }

    def _save_human_readable_summary(self, url: str, timestamp: str, url_hash: int, summary: dict):
        """Save a human-readable summary of the analysis.

        Args:
            url: Original URL.
            timestamp: Timestamp string.
            url_hash: Hash of URL.
            summary: Analysis summary dictionary.
        """
        try:
            lines = []
            lines.append("=" * 80)
            lines.append(f"Video Data Analysis Summary")
            lines.append("=" * 80)
            lines.append(f"URL: {url}")
            lines.append(f"Timestamp: {timestamp}")
            lines.append(f"Network responses: {summary['network_responses_count']}")
            lines.append(f"Runtime objects: {summary['runtime_objects_count']}")
            lines.append(f"Video object candidates: {len(summary['video_object_candidates'])}")
            lines.append("")

            lines.append("Field Extraction Results:")
            lines.append("-" * 40)
            for field_name, field_info in summary['field_extraction'].items():
                value = field_info['value']
                path = field_info['path']
                if value is not None:
                    # Truncate long values
                    value_str = str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:100] + "..."
                    lines.append(f"  {field_name}: {value_str}")
                    lines.append(f"    Path: {path}")
                else:
                    lines.append(f"  {field_name}: NOT FOUND")
            lines.append("")

            lines.append(f"Extracted {summary['extracted_field_count']} out of 10 target fields")
            lines.append("=" * 80)

            summary_text = "\n".join(lines)

            # Save to text file
            text_file = self._debug_output_dir / f"analysis_summary_{url_hash}_{timestamp}.txt"
            text_file.write_text(summary_text, encoding='utf-8')
            logger.info(f"Saved human-readable summary to {text_file}")

        except Exception as e:
            logger.warning(f"Failed to save human-readable summary: {e}")

    def _find_video_objects(self, data, base_path="", max_depth=10):
        """Recursively search for video-like objects in data.

        Args:
            data: Data to search (dict, list, or primitive).
            base_path: Current path in data structure.
            max_depth: Maximum recursion depth.

        Returns:
            List of paths to potential video objects.
        """
        if max_depth <= 0:
            return []

        video_paths = []

        # Check if current object looks like a video object
        if isinstance(data, dict):
            # Check for video indicators
            video_indicators = ['video', 'aweme', 'itemInfo', 'itemStruct', 'detail', 'feed']
            has_video_indicator = any(indicator in str(data).lower() for indicator in video_indicators)

            # Check for video fields
            video_fields = ['id', 'video_id', 'aweme_id', 'desc', 'author', 'statistics', 'create_time']
            has_video_field = any(field in data for field in video_fields)

            if has_video_indicator or has_video_field:
                video_paths.append(base_path)

            # Recursively search deeper
            for key, value in data.items():
                new_path = f"{base_path}.{key}" if base_path else key
                video_paths.extend(self._find_video_objects(value, new_path, max_depth - 1))

        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{base_path}[{i}]"
                video_paths.extend(self._find_video_objects(item, new_path, max_depth - 1))

        return video_paths

    def _extract_video_fields(self, video_obj_path: str):
        """Extract target fields from a video object path.

        Args:
            video_obj_path: Path to video object.

        Returns:
            Dictionary of extracted fields with their paths.
        """
        # This is a simplified version - in practice would need to locate the actual object
        # For now, return placeholder
        return {
            'video_id': {'value': None, 'path': None},
            'author_id': {'value': None, 'path': None},
            'author_name': {'value': None, 'path': None},
            'desc_text': {'value': None, 'path': None},
            'publish_time_raw': {'value': None, 'path': None},
            'like_count_raw': {'value': None, 'path': None},
            'comment_count_raw': {'value': None, 'path': None},
            'share_count_raw': {'value': None, 'path': None},
            'hashtag_list': {'value': None, 'path': None},
            'cover_url': {'value': None, 'path': None}
        }

    def _extract_fields_from_data(self, data, base_path=""):
        """Recursively extract target video fields from data with paths.

        Args:
            data: Data to search (dict, list, or primitive).
            base_path: Current path in data structure.

        Returns:
            Dictionary mapping field names to {'value': ..., 'path': ...}
        """
        field_mappings = {
            'video_id': {'value': None, 'path': None},
            'author_id': {'value': None, 'path': None},
            'author_name': {'value': None, 'path': None},
            'desc_text': {'value': None, 'path': None},
            'publish_time_raw': {'value': None, 'path': None},
            'like_count_raw': {'value': None, 'path': None},
            'comment_count_raw': {'value': None, 'path': None},
            'share_count_raw': {'value': None, 'path': None},
            'hashtag_list': {'value': None, 'path': None},
            'cover_url': {'value': None, 'path': None}
        }

        # Field mapping configurations: field_name -> list of possible keys
        field_configs = {
            'video_id': ['id', 'video_id', 'aweme_id', 'itemId', 'videoId', 'awemeId', 'vid'],
            'author_id': ['author_id', 'authorId', 'uid', 'user_id', 'userId', 'author.uid', 'author.id'],
            'author_name': ['author_name', 'nickname', 'author_name', 'authorName', 'author.nickname', 'user.nickname'],
            'desc_text': ['desc', 'description', 'title', 'content', 'desc_text', 'caption'],
            'publish_time_raw': ['create_time', 'publish_time', 'timestamp', 'createTime', 'publishTime', 'time'],
            'like_count_raw': ['like_count', 'digg_count', 'likeCount', 'diggCount', 'statistics.digg_count', 'stats.digg_count'],
            'comment_count_raw': ['comment_count', 'commentCount', 'statistics.comment_count', 'stats.comment_count'],
            'share_count_raw': ['share_count', 'shareCount', 'statistics.share_count', 'stats.share_count'],
            'hashtag_list': ['hashtags', 'tag_list', 'hashtag_list', 'challenges', 'text_extra'],
            'cover_url': ['cover', 'cover_url', 'coverUrl', 'thumbnail', 'video.cover', 'cover_image']
        }

        def search_in_dict(d, current_path):
            """Search for fields in dictionary."""
            for key, value in d.items():
                new_path = f"{current_path}.{key}" if current_path else key

                # Check each field
                for field_name, possible_keys in field_configs.items():
                    # Check exact key match
                    if key in possible_keys:
                        if field_mappings[field_name]['value'] is None:  # Take first found
                            field_mappings[field_name]['value'] = value
                            field_mappings[field_name]['path'] = new_path

                # Recursively search nested structures
                if isinstance(value, dict):
                    search_in_dict(value, new_path)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        search_in_dict({f'[{i}]': item}, new_path)

        if isinstance(data, dict):
            search_in_dict(data, base_path)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                search_in_dict({f'[{i}]': item}, base_path)

        return field_mappings

    def _respect_delay(self):
        """Respect delay between requests to avoid being blocked."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.delay_between_requests:
            sleep_time = self.delay_between_requests - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _add_jitter(self, delay: float) -> float:
        """Add random jitter to delay."""
        jitter = random.uniform(0.8, 1.2)
        return delay * jitter

    def get(self, url: str, wait_for_selector: Optional[str] = None,
            wait_timeout: Optional[int] = None, scroll_to_bottom: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        """Navigate to URL using browser and get rendered HTML.

        Args:
            url: Target URL.
            wait_for_selector: Selector to wait for (default: config value).
            wait_timeout: Timeout in ms (default: config value).
            scroll_to_bottom: Whether to scroll to bottom (default: config value).

        Returns:
            Dictionary with:
            - 'html': Rendered HTML content
            - 'url': Final URL after redirects
            - 'status': HTTP status code (if available)
            - 'headers': Response headers
            - 'screenshot_path': Path to screenshot if saved
            - 'extracted_fields': Dictionary of extracted video fields from browser runtime data
            - 'extraction_summary': Full analysis summary including field paths and sources
            Or None if failed.
        """
        if self.use_mock:
            logger.info(f"Mock mode: would browser GET {url}")
            return self._mock_response(url)

        self._respect_delay()

        # Setup debug output for this request
        self._setup_debug_output(url)

        # Initialize browser if needed
        if not self._initialized:
            self._initialize_browser()

        # Use provided values or config defaults
        wait_for_selector = wait_for_selector or self.wait_for_selector
        wait_timeout = wait_timeout or self.wait_timeout
        scroll_to_bottom = scroll_to_bottom if scroll_to_bottom is not None else self.scroll_to_bottom

        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"Browser navigating to {url} (attempt {attempt + 1}/{self.max_retries + 1})")

                # Setup network listener to capture JSON responses
                self._setup_network_listener()

                # Navigate to URL
                response = self._page.goto(url, timeout=self.request_timeout * 1000)  # Convert to ms

                # Wait for page to load
                if wait_for_selector:
                    self._page.wait_for_selector(wait_for_selector, timeout=wait_timeout)
                else:
                    self._page.wait_for_load_state('networkidle', timeout=wait_timeout)

                # Scroll to bottom if needed (for lazy-loaded content)
                if scroll_to_bottom:
                    self._scroll_page_to_bottom()

                # Get final URL and content
                final_url = self._page.url
                html_content = self._page.content()

                # Capture runtime JavaScript objects
                self._capture_runtime_objects()

                # Take screenshot for debugging
                screenshot_path = None
                if get_config('settings.crawler.browser.save_screenshots', False):
                    screenshot_dir = Path(get_config('settings.paths.raw_data', './data/raw')) / "screenshots"
                    screenshot_dir.mkdir(exist_ok=True)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    screenshot_path = screenshot_dir / f"screenshot_{timestamp}.png"
                    self._page.screenshot(path=str(screenshot_path))
                    logger.debug(f"Screenshot saved to {screenshot_path}")

                logger.info(f"Browser navigation succeeded: {final_url}, HTML length: {len(html_content)} chars")

                # Save debug data (network responses and runtime objects) and get extraction summary
                summary = self._save_debug_data(url)

                # Extract fields from summary for integration with scheduler
                extracted_fields = {}
                if summary and 'field_extraction' in summary:
                    for field_name, field_info in summary['field_extraction'].items():
                        if field_info['value'] is not None:
                            extracted_fields[field_name] = field_info['value']

                logger.info(f"Extracted {len(extracted_fields)} fields from browser runtime data")

                return {
                    'html': html_content,
                    'url': final_url,
                    'status': response.status if response else 200,
                    'headers': response.headers if response else {},
                    'screenshot_path': screenshot_path,
                    'extracted_fields': extracted_fields,
                    'extraction_summary': summary
                }

            except Exception as e:
                logger.warning(f"Browser navigation to {url} failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    delay = self._add_jitter(self.retry_delay * (2 ** attempt))
                    logger.info(f"Retrying in {delay:.2f}s")
                    time.sleep(delay)
                    # Refresh page or create new page if needed
                    try:
                        self._page.reload()
                    except:
                        self._close_browser()
                        self._initialize_browser()
                else:
                    logger.error(f"Browser navigation to {url} failed after {self.max_retries + 1} attempts")
                    return None

    def _scroll_page_to_bottom(self):
        """Scroll page to bottom to trigger lazy loading."""
        try:
            # Get initial scroll position
            scroll_position = self._page.evaluate("window.scrollY")
            viewport_height = self._page.evaluate("window.innerHeight")
            document_height = self._page.evaluate("document.body.scrollHeight")

            # Scroll in increments
            while scroll_position + viewport_height < document_height:
                scroll_position += viewport_height
                self._page.evaluate(f"window.scrollTo(0, {scroll_position})")
                time.sleep(self.scroll_delay)
                # Update heights
                document_height = self._page.evaluate("document.body.scrollHeight")

            logger.debug(f"Scrolled to bottom, final position: {scroll_position}")
        except Exception as e:
            logger.warning(f"Failed to scroll page: {e}")

    def _mock_response(self, url: str) -> Dict[str, Any]:
        """Create a mock response for testing.

        Args:
            url: URL to mock.

        Returns:
            Mock response dictionary.
        """
        video_id = "1234567890123456789"
        if "video/" in url:
            parts = url.split("video/")
            if len(parts) > 1:
                video_id = parts[1].split("?")[0]

        mock_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mock Douyin Video (Browser Mode)</title>
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
                <h1>Mock Douyin Video (Browser Mode)</h1>
                <p>Video ID: {video_id}</p>
                <p>This is a mock HTML response for browser mode development.</p>
                <div class="author">测试用户</div>
                <div class="desc">这是一个测试视频描述 #美食 #旅行</div>
                <div class="stats">
                    <span class="like">1.2w</span>
                    <span class="comment">450</span>
                    <span class="share">120</span>
                </div>
                <div class="time">2023-01-01 12:00:00</div>
                <img src="https://example.com/cover.jpg" class="cover">
            </div>
        </body>
        </html>
        """

        # Mock extracted fields from browser runtime data
        mock_extracted_fields = {
            'video_id': video_id,
            'author_id': 'author123',
            'author_name': '测试用户',
            'desc_text': '这是一个测试视频描述 #美食 #旅行',
            'publish_time_raw': 1672531200,
            'like_count_raw': 12000,
            'comment_count_raw': 450,
            'share_count_raw': 120,
            'hashtag_list': ['美食', '旅行'],
            'cover_url': 'https://example.com/cover.jpg'
        }

        mock_summary = {
            'url': url,
            'timestamp': '20230101_120000',
            'network_responses_count': 0,
            'runtime_objects_count': 0,
            'field_extraction': {
                field: {'value': value, 'path': f'mock.{field}'}
                for field, value in mock_extracted_fields.items()
            },
            'extracted_field_count': len(mock_extracted_fields)
        }

        return {
            'html': mock_html,
            'url': url,
            'status': 200,
            'headers': {'Content-Type': 'text/html'},
            'screenshot_path': None,
            'extracted_fields': mock_extracted_fields,
            'extraction_summary': mock_summary
        }

    def close(self):
        """Close the browser."""
        self._close_browser()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()