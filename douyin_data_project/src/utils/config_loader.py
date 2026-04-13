"""
Configuration loader for Douyin data project.

Loads configuration from YAML files and environment variables.
Supports hierarchical configuration with defaults.
"""
import os
import sys
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and manages configuration."""

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize config loader.

        Args:
            config_dir: Directory containing config files. If None, uses default.
        """
        self.config_dir = config_dir or Path(__file__).parent.parent.parent / 'configs'
        self.config: Dict[str, Any] = {}
        self._load_all()

    def _load_all(self):
        """Load all configuration files."""
        # Load environment variables
        self._load_env()

        # Load YAML configs
        config_files = ['settings.yaml', 'sources.yaml', 'fields.yaml', 'logging.yaml']
        for file_name in config_files:
            self._load_yaml(file_name)

        # Apply environment variable overrides
        self._apply_env_overrides()

        # Expand paths
        self._expand_paths()

    def _load_env(self):
        """Load environment variables from .env file."""
        env_path = self.config_dir.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            logger.debug(f"Loaded environment from {env_path}")
        else:
            logger.debug("No .env file found, using system environment")

    def _load_yaml(self, file_name: str):
        """Load YAML configuration file.

        Args:
            file_name: Name of YAML file.
        """
        file_path = self.config_dir / file_name
        if not file_path.exists():
            logger.warning(f"Config file not found: {file_path}")
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            # Use filename without extension as key
            key = file_name.replace('.yaml', '').replace('.yml', '')

            # If config_data is a dict and contains a top-level key matching the filename,
            # merge its contents instead of nesting
            if isinstance(config_data, dict) and key in config_data:
                # The YAML file has top-level key matching filename (e.g., sources.yaml with 'sources:' top-level)
                # Merge the contents into self.config[key]
                if key not in self.config:
                    self.config[key] = {}
                self._merge_dict(self.config[key], config_data[key])
            else:
                # Normal assignment
                self.config[key] = config_data

            logger.debug(f"Loaded config: {file_name}")

        except Exception as e:
            logger.error(f"Failed to load config {file_name}: {e}")

    def _merge_dict(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Recursively merge source dictionary into target dictionary.

        Args:
            target: Target dictionary to merge into.
            source: Source dictionary to merge from.
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                self._merge_dict(target[key], value)
            else:
                # Replace or add the value
                target[key] = value

    def _apply_env_overrides(self):
        """Apply environment variable overrides to config."""
        # Project settings
        if 'PROJECT_ENV' in os.environ:
            self._set_nested(self.config, ['settings', 'env'], os.environ['PROJECT_ENV'])

        if 'DATA_VERSION' in os.environ:
            self._set_nested(self.config, ['settings', 'data_version'], os.environ['DATA_VERSION'])

        # Crawler settings - paths should be under settings.crawler
        crawler_env_vars = {
            'CRAWLER_DELAY': 'settings.crawler.delay_between_requests',
            'CRAWLER_MAX_RETRIES': 'settings.crawler.max_retries',
            'CRAWLER_TIMEOUT': 'settings.crawler.request_timeout',
            'CRAWLER_USER_AGENT': 'settings.crawler.user_agent'
        }

        for env_var, config_path in crawler_env_vars.items():
            if env_var in os.environ:
                self._set_nested(self.config, config_path.split('.'), os.environ[env_var])

        # API settings - under settings.api
        api_env_vars = {
            'DOUYIN_CLIENT_ID': 'settings.api.client_id',
            'DOUYIN_CLIENT_SECRET': 'settings.api.client_secret',
            'DOUYIN_REDIRECT_URI': 'settings.api.redirect_uri',
            'DOUYIN_ACCESS_TOKEN': 'settings.api.access_token',
            'DOUYIN_REFRESH_TOKEN': 'settings.api.refresh_token',
            'DOUYIN_OPEN_ID': 'settings.api.open_id'
        }

        for env_var, config_path in api_env_vars.items():
            if env_var in os.environ:
                self._set_nested(self.config, config_path.split('.'), os.environ[env_var])

        # Logging settings - under settings.logging
        logging_env_vars = {
            'LOG_LEVEL': 'settings.logging.level',
            'LOG_TO_FILE': 'settings.logging.file_log',
            'LOG_TO_CONSOLE': 'settings.logging.console_log'
        }

        for env_var, config_path in logging_env_vars.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                # Convert string to boolean if needed
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                self._set_nested(self.config, config_path.split('.'), value)

        # Processing settings - engagement weights
        processing_env_vars = {
            'ENGAGEMENT_LIKE_WEIGHT': 'settings.processing.engagement_score_weights.like',
            'ENGAGEMENT_COMMENT_WEIGHT': 'settings.processing.engagement_score_weights.comment',
            'ENGAGEMENT_SHARE_WEIGHT': 'settings.processing.engagement_score_weights.share'
        }

        for env_var, config_path in processing_env_vars.items():
            if env_var in os.environ:
                try:
                    value = float(os.environ[env_var])
                    self._set_nested(self.config, config_path.split('.'), value)
                except ValueError:
                    logger.warning(f"Invalid float value for {env_var}: {os.environ[env_var]}")

        # Feature flags
        if 'USE_MOCK_DATA' in os.environ:
            use_mock = os.environ['USE_MOCK_DATA'].lower() == 'true'
            self._set_nested(self.config, ['sources', 'web', 'mock', 'enabled'], use_mock)

        if 'ENABLE_API' in os.environ:
            enable_api = os.environ['ENABLE_API'].lower() == 'true'
            self._set_nested(self.config, ['sources', 'api', 'enabled'], enable_api)

    def _set_nested(self, config_dict: Dict[str, Any], path: list, value: Any):
        """Set nested value in dictionary.

        Args:
            config_dict: Dictionary to modify.
            path: List of keys representing path.
            value: Value to set.
        """
        current = config_dict
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def _expand_paths(self):
        """Expand path placeholders in config recursively."""
        project_root = self.config_dir.parent
        self._expand_paths_in_dict(self.config, project_root)

    def _expand_paths_in_dict(self, config_dict: Dict[str, Any], project_root: Path):
        """Recursively expand paths in a dictionary.

        Args:
            config_dict: Dictionary to process.
            project_root: Project root directory.
        """
        for key, value in list(config_dict.items()):
            if isinstance(value, dict):
                self._expand_paths_in_dict(value, project_root)
            elif isinstance(value, str):
                # Replace relative paths starting with ./ with absolute paths
                if value.startswith('./'):
                    abs_path = project_root / value[2:]
                    config_dict[key] = str(abs_path)
                # Expand environment variables
                elif '$' in value:
                    config_dict[key] = os.path.expandvars(value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        if item.startswith('./'):
                            abs_path = project_root / item[2:]
                            value[i] = str(abs_path)
                        elif '$' in item:
                            value[i] = os.path.expandvars(item)
                    elif isinstance(item, dict):
                        self._expand_paths_in_dict(item, project_root)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key.

        Args:
            key: Dot notation key (e.g., 'crawler.request_timeout').
            default: Default value if key not found.

        Returns:
            Configuration value.
        """
        keys = key.split('.')
        current = self.config

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        return current

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """Get nested configuration value.

        Args:
            *keys: Sequence of keys.
            default: Default value if key not found.

        Returns:
            Configuration value.
        """
        current = self.config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def set(self, key: str, value: Any):
        """Set configuration value.

        Args:
            key: Dot notation key.
            value: Value to set.
        """
        keys = key.split('.')
        self._set_nested(self.config, keys, value)

    def save(self, file_name: str = 'settings.yaml'):
        """Save configuration to YAML file.

        Args:
            file_name: Name of YAML file.
        """
        file_path = self.config_dir / file_name
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config.get('settings', {}), f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Saved config to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def print_summary(self):
        """Print configuration summary."""
        print("Configuration Summary:")
        print("=" * 50)

        # Project info
        settings = self.config.get('settings', {})
        project = settings.get('project', {})
        print(f"Project: {project.get('name')} v{project.get('version')}")
        print(f"Environment: {settings.get('env')}")
        print(f"Data Version: {settings.get('data_version')}")

        # Paths
        paths = settings.get('paths', {})
        print(f"\nPaths:")
        for key, path in paths.items():
            print(f"  {key}: {path}")

        # Crawler settings
        crawler = settings.get('crawler', {})
        print(f"\nCrawler:")
        print(f"  Timeout: {crawler.get('request_timeout')}s")
        print(f"  Max Retries: {crawler.get('max_retries')}")
        print(f"  Delay: {crawler.get('delay_between_requests')}s")

        # API settings
        api = settings.get('api', {})
        print(f"\nAPI:")
        print(f"  Base URL: {api.get('base_url')}")
        print(f"  Client ID: {'*' * 8 if api.get('client_id') else 'Not set'}")

        # Sources
        sources = self.config.get('sources', {})
        print(f"\nSources:")
        print(f"  Web enabled: {sources.get('web', {}).get('mock', {}).get('enabled')}")
        print(f"  API enabled: {sources.get('api', {}).get('enabled')}")


# Singleton instance
_config_loader: Optional[ConfigLoader] = None


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration.

    Args:
        config_path: Path to config directory.

    Returns:
        Configuration dictionary.
    """
    global _config_loader

    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)

    return _config_loader.config


def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value.

    Args:
        key: Dot notation key.
        default: Default value.

    Returns:
        Configuration value.
    """
    config = load_config()
    keys = key.split('.')
    current = config

    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return default

    return current


def reload_config(config_path: Optional[Path] = None):
    """Reload configuration.

    Args:
        config_path: Path to config directory.
    """
    global _config_loader
    _config_loader = ConfigLoader(config_path)
    logger.info("Configuration reloaded")