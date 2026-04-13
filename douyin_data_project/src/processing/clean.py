"""
Data cleaning functions for Douyin video data.

Includes text cleaning, missing value handling, and data validation.
"""
import re
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, date
from pathlib import Path
import logging

import pandas as pd
import numpy as np

from ..schemas.tables import WebVideoMeta, ProcessedVideoData
from ..utils.config_loader import load_config, get_config
from ..utils.logger import get_logger
from ..utils.text_utils import clean_text, remove_urls, remove_emojis, remove_special_chars

logger = get_logger(__name__)


class DataCleaner:
    """Cleans raw video data for processing."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize cleaner.

        Args:
            config_path: Path to config file.
        """
        self.config = load_config(config_path) if config_path else load_config()
        self.clean_config = get_config('settings.processing.text_clean', {})

    def clean_web_video_meta(self, data: WebVideoMeta) -> Dict[str, Any]:
        """Clean WebVideoMeta data.

        Args:
            data: WebVideoMeta object.

        Returns:
            Cleaned data dictionary.
        """
        cleaned = data.dict()

        # Clean description text
        if cleaned.get('desc_text'):
            cleaned['desc_clean'] = self._clean_text(cleaned['desc_text'])
            cleaned['text_length'] = len(cleaned['desc_clean'])
        else:
            cleaned['desc_clean'] = ''
            cleaned['text_length'] = 0

        # Parse publish time
        if cleaned.get('publish_time_std'):
            publish_dt = cleaned['publish_time_std']
            if isinstance(publish_dt, str):
                try:
                    publish_dt = datetime.fromisoformat(publish_dt.replace('Z', '+00:00'))
                except:
                    publish_dt = None
            if isinstance(publish_dt, datetime):
                cleaned['publish_date'] = publish_dt.date()
                cleaned['publish_hour'] = publish_dt.hour
                cleaned['publish_weekday'] = publish_dt.weekday()  # 0=Monday, 6=Sunday
                cleaned['is_weekend'] = 1 if publish_dt.weekday() >= 5 else 0

        # Ensure hashtag list is list
        if cleaned.get('hashtag_list') is None:
            cleaned['hashtag_list'] = []
        if not isinstance(cleaned['hashtag_list'], list):
            cleaned['hashtag_list'] = []

        # Clean hashtags
        cleaned['hashtag_list'] = [self._clean_hashtag(tag) for tag in cleaned['hashtag_list']]
        cleaned['hashtag_count'] = len(cleaned['hashtag_list'])

        # Add crawl date
        if cleaned.get('crawl_time'):
            crawl_dt = cleaned['crawl_time']
            if isinstance(crawl_dt, str):
                try:
                    crawl_dt = datetime.fromisoformat(crawl_dt.replace('Z', '+00:00'))
                except:
                    crawl_dt = datetime.now()
            cleaned['crawl_date'] = crawl_dt.date()

        # Add data version
        cleaned['data_version'] = self.config.get('data_version', 'v0.1')

        return cleaned

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataframe of web video metadata.

        Args:
            df: Input dataframe.

        Returns:
            Cleaned dataframe.
        """
        df_clean = df.copy()

        # Clean text columns
        text_columns = ['desc_text', 'author_name']
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(self._clean_text)

        # Create desc_clean column for transformation
        if 'desc_text' in df_clean.columns:
            df_clean['desc_clean'] = df_clean['desc_text']
        else:
            df_clean['desc_clean'] = ''

        # Calculate text length
        df_clean['text_length'] = df_clean['desc_clean'].apply(lambda x: len(x) if isinstance(x, str) else 0)

        # Convert date columns
        date_columns = ['publish_time_std', 'crawl_time']
        for col in date_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')

        # Parse publish time features
        if 'publish_time_std' in df_clean.columns:
            df_clean['publish_date'] = df_clean['publish_time_std'].dt.date
            df_clean['publish_hour'] = df_clean['publish_time_std'].dt.hour
            df_clean['publish_weekday'] = df_clean['publish_time_std'].dt.weekday
            df_clean['is_weekend'] = (df_clean['publish_weekday'] >= 5).astype(int)

        if 'crawl_time' in df_clean.columns:
            df_clean['crawl_date'] = df_clean['crawl_time'].dt.date

        # Clean hashtags
        if 'hashtag_list' in df_clean.columns:
            df_clean['hashtag_list'] = df_clean['hashtag_list'].apply(
                lambda x: [self._clean_hashtag(tag.strip()) for tag in
                          (x.split(';') if isinstance(x, str) and x else
                           (x if isinstance(x, list) else []))]
            )
            df_clean['hashtag_count'] = df_clean['hashtag_list'].apply(len)

        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)

        # Add data version
        df_clean['data_version'] = self.config.get('data_version', 'v0.1')

        # Ensure hashtag_list is Python list (not numpy array)
        if 'hashtag_list' in df_clean.columns:
            df_clean['hashtag_list'] = df_clean['hashtag_list'].apply(
                lambda x: list(x) if isinstance(x, np.ndarray) else x
            )

        return df_clean

    def _clean_text(self, text: Optional[str]) -> str:
        """Clean text using configured rules.

        Args:
            text: Input text.

        Returns:
            Cleaned text.
        """
        if not text:
            return ''

        cleaned = str(text)

        if self.clean_config.get('remove_urls', True):
            cleaned = remove_urls(cleaned)

        if self.clean_config.get('remove_emojis', True):
            cleaned = remove_emojis(cleaned)

        if self.clean_config.get('remove_special_chars', False):
            cleaned = remove_special_chars(cleaned)

        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Truncate if too long (optional)
        max_length = self.clean_config.get('max_length')
        if max_length and len(cleaned) > max_length:
            cleaned = cleaned[:max_length] + '...'

        return cleaned

    def _clean_hashtag(self, tag: str) -> str:
        """Clean a hashtag.

        Args:
            tag: Input hashtag.

        Returns:
            Cleaned hashtag.
        """
        if not tag:
            return ''

        # Remove # symbol if present
        tag = tag.strip()
        if tag.startswith('#'):
            tag = tag[1:]

        # Remove special characters
        tag = re.sub(r'[^\w\u4e00-\u9fff\-_]', '', tag)

        return tag

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in dataframe.

        Args:
            df: Input dataframe.

        Returns:
            Dataframe with handled missing values.
        """
        # Fill missing counts with 0
        count_columns = ['like_count', 'comment_count', 'share_count', 'collect_count']
        for col in count_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        # Fill missing text with empty string
        text_columns = ['desc_clean', 'author_name']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('')

        # Fill missing lists with empty list
        list_columns = ['hashtag_list']
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])

        return df

    def validate_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate cleaned data.

        Args:
            data: Cleaned data dictionary.

        Returns:
            Tuple of (is_valid, error_messages).
        """
        errors = []

        # Check required fields
        required_fields = ['video_id', 'source_entry', 'crawl_date', 'data_version']
        for field in required_fields:
            if field not in data or not data[field]:
                errors.append(f"Missing required field: {field}")

        # Check text length
        min_length = self.clean_config.get('min_text_length', 2)
        if data.get('text_length', 0) < min_length and data.get('desc_clean'):
            errors.append(f"Text too short: {data['text_length']} < {min_length}")

        # Check count ranges
        count_fields = ['like_count', 'comment_count', 'share_count', 'collect_count']
        for field in count_fields:
            if field in data and data[field] is not None:
                if data[field] < 0:
                    errors.append(f"Negative count: {field} = {data[field]}")
                if data[field] > 1e9:  # 1 billion
                    errors.append(f"Unrealistically large count: {field} = {data[field]}")

        # Check date ranges
        if data.get('publish_date'):
            publish_date = data['publish_date']
            if isinstance(publish_date, str):
                try:
                    publish_date = datetime.fromisoformat(publish_date).date()
                except:
                    pass
            if isinstance(publish_date, date):
                today = date.today()
                if publish_date > today:
                    errors.append(f"Future publish date: {publish_date}")
                # Douyin launched in 2016
                if publish_date < date(2016, 1, 1):
                    errors.append(f"Unrealistically old publish date: {publish_date}")

        return len(errors) == 0, errors


# Convenience functions
def clean_single_record(record: Dict[str, Any], config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Clean a single record.

    Args:
        record: Input record.
        config_path: Path to config.

    Returns:
        Cleaned record.
    """
    cleaner = DataCleaner(config_path)
    return cleaner.clean_web_video_meta(record)


def clean_batch_records(records: List[Dict[str, Any]], config_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Clean a batch of records.

    Args:
        records: List of input records.
        config_path: Path to config.

    Returns:
        List of cleaned records.
    """
    cleaner = DataCleaner(config_path)
    return [cleaner.clean_web_video_meta(record) for record in records]


def clean_from_dataframe(df: pd.DataFrame, config_path: Optional[Path] = None) -> pd.DataFrame:
    """Clean data from dataframe.

    Args:
        df: Input dataframe.
        config_path: Path to config.

    Returns:
        Cleaned dataframe.
    """
    cleaner = DataCleaner(config_path)
    return cleaner.clean_dataframe(df)