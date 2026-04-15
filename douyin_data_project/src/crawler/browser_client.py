"""
Browser client for Douyin web crawler using Playwright.

Handles JavaScript-rendered pages by using a headless browser.
Supports both headless and headed modes for debugging.
"""
import time
import random
from typing import Optional, Dict, Any, Tuple
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
            Or None if failed.
        """
        if self.use_mock:
            logger.info(f"Mock mode: would browser GET {url}")
            return self._mock_response(url)

        self._respect_delay()

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

                return {
                    'html': html_content,
                    'url': final_url,
                    'status': response.status if response else 200,
                    'headers': response.headers if response else {},
                    'screenshot_path': screenshot_path,
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

        return {
            'html': mock_html,
            'url': url,
            'status': 200,
            'headers': {'Content-Type': 'text/html'},
            'screenshot_path': None
        }

    def close(self):
        """Close the browser."""
        self._close_browser()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()