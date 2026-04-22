"""
Data transformation functions for Douyin video data.

Includes normalization, type conversion, and feature calculation.
"""
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, date, timedelta
from pathlib import Path
import logging
import math

import pandas as pd
import numpy as np

from ..schemas.tables import ProcessedVideoData
from ..utils.config_loader import load_config, get_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataTransformer:
    """Transforms cleaned data into processed features."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize transformer.

        Args:
            config_path: Path to config file.
        """
        self.config = load_config(config_path) if config_path else load_config()
        self.processing_config = get_config('settings.processing', {})

    def transform_to_processed(self, cleaned_data: Dict[str, Any]) -> ProcessedVideoData:
        """Transform cleaned data to ProcessedVideoData.

        Args:
            cleaned_data: Cleaned data dictionary.

        Returns:
            ProcessedVideoData object.
        """
        # Ensure required fields
        required = ['video_id', 'source_entry', 'crawl_date', 'data_version']
        for field in required:
            if field not in cleaned_data:
                raise ValueError(f"Missing required field: {field}")

        # Calculate engagement score
        engagement_score = self._calculate_engagement_score(cleaned_data)

        # Create processed data
        processed = ProcessedVideoData(
            video_id=cleaned_data['video_id'],
            author_id=cleaned_data.get('author_id'),
            desc_clean=cleaned_data.get('desc_clean', ''),
            text_length=cleaned_data.get('text_length', 0),
            publish_date=cleaned_data.get('publish_date'),
            publish_hour=cleaned_data.get('publish_hour'),
            publish_weekday=cleaned_data.get('publish_weekday'),
            is_weekend=cleaned_data.get('is_weekend'),
            hashtag_list=cleaned_data.get('hashtag_list', []),
            hashtag_count=cleaned_data.get('hashtag_count', 0),
            like_count=cleaned_data.get('like_count'),
            comment_count=cleaned_data.get('comment_count'),
            share_count=cleaned_data.get('share_count'),
            collect_count=cleaned_data.get('collect_count'),
            engagement_score=engagement_score,
            source_entry=cleaned_data['source_entry'],
            crawl_date=cleaned_data['crawl_date'],
            data_version=cleaned_data['data_version']
        )

        return processed

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe of cleaned data.

        Args:
            df: Cleaned dataframe.

        Returns:
            Transformed dataframe with processed features.
        """
        df_transformed = df.copy()

        # Calculate engagement score
        weights = self.processing_config.get('engagement_score_weights', {})
        like_weight = weights.get('like', 1.0)
        comment_weight = weights.get('comment', 2.0)
        share_weight = weights.get('share', 3.0)

        engagement_score = 0.0
        if 'like_count' in df_transformed.columns:
            engagement_score += like_weight * df_transformed['like_count'].fillna(0)
        if 'comment_count' in df_transformed.columns:
            engagement_score += comment_weight * df_transformed['comment_count'].fillna(0)
        if 'share_count' in df_transformed.columns:
            engagement_score += share_weight * df_transformed['share_count'].fillna(0)

        df_transformed['engagement_score'] = engagement_score

        # Calculate additional features
        df_transformed = self._calculate_additional_features(df_transformed)

        # Select and order columns for processed data
        processed_columns = [
            'video_id', 'author_id', 'desc_clean', 'text_length',
            'publish_date', 'publish_hour', 'publish_weekday', 'is_weekend',
            'hashtag_list', 'hashtag_count', 'like_count', 'comment_count',
            'share_count', 'collect_count', 'engagement_score', 'source_entry',
            'crawl_date', 'data_version'
        ]

        # Ensure all columns exist
        for col in processed_columns:
            if col not in df_transformed.columns:
                df_transformed[col] = None

        return df_transformed[processed_columns]

    def _calculate_engagement_score(self, data: Dict[str, Any]) -> Optional[float]:
        """Calculate engagement score.

        Args:
            data: Data dictionary.

        Returns:
            Engagement score or None.
        """
        weights = self.processing_config.get('engagement_score_weights', {})
        like_weight = weights.get('like', 1.0)
        comment_weight = weights.get('comment', 2.0)
        share_weight = weights.get('share', 3.0)

        score = 0.0
        has_data = False

        if data.get('like_count') is not None:
            score += like_weight * data['like_count']
            has_data = True

        if data.get('comment_count') is not None:
            score += comment_weight * data['comment_count']
            has_data = True

        if data.get('share_count') is not None:
            score += share_weight * data['share_count']
            has_data = True

        return score if has_data else None

    def _calculate_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional features.

        Args:
            df: Dataframe.

        Returns:
            Dataframe with additional features.
        """
        # Time-based features
        if 'publish_date' in df.columns and 'crawl_date' in df.columns:
            # Calculate age at crawl (days)
            # Convert to datetime for timedelta calculation
            df['days_since_publish'] = (pd.to_datetime(df['crawl_date']) - pd.to_datetime(df['publish_date'])).dt.days

            # Categorize age
            df['publish_recency'] = pd.cut(
                df['days_since_publish'],
                bins=[-1, 1, 7, 30, 365, float('inf')],
                labels=['1d', '7d', '30d', '1y', 'older']
            )

        # Interaction ratios (if we had view count)
        # TODO: Add when view count available from API

        # Text complexity features
        if 'desc_clean' in df.columns:
            df['has_hashtags'] = (df['hashtag_count'] > 0).astype(int)
            df['has_long_text'] = (df['text_length'] > 50).astype(int)

            # Calculate approximate word count (Chinese characters + spaces)
            df['word_count'] = df['desc_clean'].apply(
                lambda x: len(x.strip()) if isinstance(x, str) else 0
            )

        # Author features (if we had multiple videos per author)
        # TODO: Add when author dimension is populated

        # Popularity tiers
        if 'like_count' in df.columns:
            df['like_tier'] = pd.cut(
                df['like_count'].fillna(0),
                bins=[-1, 0, 100, 1000, 10000, 100000, float('inf')],
                labels=['0', '1-100', '101-1k', '1k-10k', '10k-100k', '100k+']
            )

        if 'engagement_score' in df.columns:
            df['engagement_tier'] = pd.qcut(
                df['engagement_score'].rank(method='first'),
                q=4,
                labels=['low', 'medium', 'high', 'very_high']
            )

        return df

    def normalize_counts(self, df: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
        """Normalize count columns.

        Args:
            df: Dataframe with count columns.
            method: Normalization method ('log', 'sqrt', 'minmax', 'standard').

        Returns:
            Dataframe with normalized counts.
        """
        df_norm = df.copy()

        count_columns = ['like_count', 'comment_count', 'share_count', 'collect_count']
        count_columns = [col for col in count_columns if col in df_norm.columns]

        for col in count_columns:
            values = df_norm[col].fillna(0).astype(float)

            if method == 'log':
                # Log transformation (add 1 to handle zeros)
                df_norm[f'{col}_log'] = np.log1p(values)

            elif method == 'sqrt':
                # Square root transformation
                df_norm[f'{col}_sqrt'] = np.sqrt(values)

            elif method == 'minmax':
                # Min-max scaling
                min_val = values.min()
                max_val = values.max()
                if max_val > min_val:
                    df_norm[f'{col}_minmax'] = (values - min_val) / (max_val - min_val)
                else:
                    df_norm[f'{col}_minmax'] = 0

            elif method == 'standard':
                # Standard scaling
                mean_val = values.mean()
                std_val = values.std()
                if std_val > 0:
                    df_norm[f'{col}_standard'] = (values - mean_val) / std_val
                else:
                    df_norm[f'{col}_standard'] = 0

        return df_norm

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features.

        Args:
            df: Dataframe with date columns.

        Returns:
            Dataframe with time features.
        """
        df_time = df.copy()

        if 'publish_hour' in df_time.columns:
            # Hour categories
            df_time['hour_category'] = pd.cut(
                df_time['publish_hour'],
                bins=[-1, 6, 12, 18, 24],
                labels=['night', 'morning', 'afternoon', 'evening']
            )

            # Peak hours (assuming 12-14, 18-22)
            df_time['is_peak_hour'] = (
                ((df_time['publish_hour'] >= 12) & (df_time['publish_hour'] <= 14)) |
                ((df_time['publish_hour'] >= 18) & (df_time['publish_hour'] <= 22))
            ).astype(int)

        if 'publish_weekday' in df_time.columns:
            # Weekday/weekend
            df_time['is_weekend'] = (df_time['publish_weekday'] >= 5).astype(int)

            # Day of week name
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            df_time['weekday_name'] = df_time['publish_weekday'].apply(
                lambda x: day_names[x] if 0 <= x <= 6 else 'Unknown'
            )

        if 'publish_date' in df_time.columns:
            # Season
            df_time['season'] = df_time['publish_date'].apply(self._get_season)

            # Is holiday (simplified - just weekends for now)
            # TODO: Add actual holiday calendar

        return df_time

    def _get_season(self, d: date) -> str:
        """Get season from date.

        Args:
            d: Date.

        Returns:
            Season name.
        """
        month = d.month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:  # 9, 10, 11
            return 'autumn'


# Convenience functions
def transform_single_record(record: Dict[str, Any], config_path: Optional[Path] = None) -> ProcessedVideoData:
    """Transform a single record.

    Args:
        record: Cleaned record.
        config_path: Path to config.

    Returns:
        ProcessedVideoData object.
    """
    transformer = DataTransformer(config_path)
    return transformer.transform_to_processed(record)


def transform_batch_records(records: List[Dict[str, Any]], config_path: Optional[Path] = None) -> List[ProcessedVideoData]:
    """Transform a batch of records.

    Args:
        records: List of cleaned records.
        config_path: Path to config.

    Returns:
        List of ProcessedVideoData objects.
    """
    transformer = DataTransformer(config_path)
    return [transformer.transform_to_processed(record) for record in records]


def transform_dataframe(df: pd.DataFrame, config_path: Optional[Path] = None) -> pd.DataFrame:
    """Transform dataframe.

    Args:
        df: Cleaned dataframe.
        config_path: Path to config.

    Returns:
        Transformed dataframe.
    """
    transformer = DataTransformer(config_path)
    return transformer.transform_dataframe(df)


def transform_web_video_meta_to_features(df: pd.DataFrame, config_path: Optional[Path] = None) -> pd.DataFrame:
    """Transform web_video_meta dataframe to feature dataframe.

    This function converts raw web_video_meta data (from high-confidence samples)
    to feature dataframe suitable for modeling.

    Args:
        df: web_video_meta dataframe (with fields like video_id, page_url,
             publish_time_std, like_count_raw, etc.)
        config_path: Path to config.

    Returns:
        Feature dataframe with standardized schema.
    """
    from ..features.feature_pipeline import FeaturePipeline

    # Create a temporary pipeline instance
    pipeline = FeaturePipeline(feature_version='v1', verbose=False)

    # Apply transformations
    df_features = pipeline.transform_web_video_meta(df)

    return df_features