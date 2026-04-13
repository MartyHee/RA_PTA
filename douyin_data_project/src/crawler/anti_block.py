"""
Anti-blocking strategies for web crawler.

Includes proxy rotation, User-Agent rotation, request delays, and other
techniques to avoid being blocked by Douyin.
"""
import random
import time
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

from ..utils.config_loader import load_config, get_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AntiBlockManager:
    """Manages anti-blocking strategies."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize anti-block manager.

        Args:
            config_path: Path to config file.
        """
        self.config = load_config(config_path) if config_path else load_config()
        self.crawler_config = get_config('settings.crawler', {})

        self.user_agents = self._load_user_agents()
        self.proxies = self._load_proxies()
        self.cookies_pool = []

        self.current_ua_index = 0
        self.current_proxy_index = 0

        self.request_count = 0
        self.blocked_count = 0
        self.last_block_time = None

    def _load_user_agents(self) -> List[str]:
        """Load user agents from config or defaults.

        Returns:
            List of user agent strings.
        """
        default_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
        ]

        config_agents = self.crawler_config.get('user_agents', [])
        if config_agents:
            return config_agents
        return default_agents

    def _load_proxies(self) -> List[Dict[str, str]]:
        """Load proxies from config or return empty list.

        Returns:
            List of proxy dictionaries.
        """
        # TODO: Implement proxy loading from config or external source
        return []

    def get_next_user_agent(self) -> str:
        """Get next user agent (round-robin or random).

        Returns:
            User agent string.
        """
        if not self.user_agents:
            return self.crawler_config.get('user_agent', '')

        strategy = self.crawler_config.get('ua_strategy', 'round_robin')
        if strategy == 'random':
            return random.choice(self.user_agents)
        else:  # round_robin
            ua = self.user_agents[self.current_ua_index]
            self.current_ua_index = (self.current_ua_index + 1) % len(self.user_agents)
            return ua

    def get_next_proxy(self) -> Optional[Dict[str, str]]:
        """Get next proxy if available.

        Returns:
            Proxy dict or None.
        """
        if not self.proxies:
            return None

        strategy = self.crawler_config.get('proxy_strategy', 'round_robin')
        if strategy == 'random':
            return random.choice(self.proxies)
        else:  # round_robin
            proxy = self.proxies[self.current_proxy_index]
            self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxies)
            return proxy

    def should_delay(self) -> bool:
        """Check if we should add extra delay based on request patterns.

        Returns:
            True if extra delay is needed.
        """
        # Simple strategy: add random delay every N requests
        delay_every = self.crawler_config.get('delay_every_n', 10)
        if self.request_count % delay_every == 0 and self.request_count > 0:
            return True
        return False

    def get_delay(self) -> float:
        """Get delay time between requests.

        Returns:
            Delay in seconds.
        """
        base_delay = self.crawler_config.get('delay_between_requests', 1.5)

        # Add jitter
        jitter_factor = random.uniform(0.8, 1.2)
        delay = base_delay * jitter_factor

        # Add extra delay if needed
        if self.should_delay():
            extra_delay = random.uniform(2.0, 5.0)
            delay += extra_delay
            logger.debug(f"Adding extra delay: {extra_delay:.2f}s")

        return delay

    def record_request(self):
        """Record a successful request."""
        self.request_count += 1

    def record_block(self):
        """Record a block event."""
        self.blocked_count += 1
        self.last_block_time = time.time()
        logger.warning(f"Block detected (total: {self.blocked_count})")

    def is_cool_down_needed(self) -> bool:
        """Check if cooldown is needed due to blocks.

        Returns:
            True if cooldown is needed.
        """
        if self.blocked_count == 0:
            return False

        # If blocked recently, need cooldown
        if self.last_block_time:
            time_since_block = time.time() - self.last_block_time
            if time_since_block < 300:  # 5 minutes
                return True

        # If multiple blocks in short period
        if self.blocked_count >= 3:
            return True

        return False

    def get_cool_down_time(self) -> float:
        """Get cooldown time based on block history.

        Returns:
            Cooldown time in seconds.
        """
        if self.blocked_count <= 1:
            return 60.0  # 1 minute
        elif self.blocked_count <= 3:
            return 300.0  # 5 minutes
        else:
            return 1800.0  # 30 minutes

    def reset_block_count(self):
        """Reset block counter (e.g., after successful cooldown)."""
        self.blocked_count = 0
        self.last_block_time = None
        logger.info("Block counter reset")

    def get_headers(self) -> Dict[str, str]:
        """Get headers with anti-block measures.

        Returns:
            Headers dictionary.
        """
        base_headers = self.crawler_config.get('headers', {})
        headers = base_headers.copy()

        # Add rotating User-Agent
        headers['User-Agent'] = self.get_next_user_agent()

        # Add referer (optional)
        if 'Referer' not in headers:
            headers['Referer'] = 'https://www.douyin.com/'

        # Add other headers to mimic browser
        headers['Accept-Language'] = 'zh-CN,zh;q=0.9,en;q=0.8'
        headers['Accept-Encoding'] = 'gzip, deflate, br'
        headers['Connection'] = 'keep-alive'
        headers['Upgrade-Insecure-Requests'] = '1'

        return headers

    def apply_cooldown_if_needed(self):
        """Apply cooldown if blocks have been detected."""
        if self.is_cool_down_needed():
            cooldown = self.get_cool_down_time()
            logger.warning(f"Applying cooldown for {cooldown:.0f}s due to blocks")
            time.sleep(cooldown)
            self.reset_block_count()