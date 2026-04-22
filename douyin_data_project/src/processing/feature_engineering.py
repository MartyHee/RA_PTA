"""
Feature engineering for Douyin video data.

Creates features for recommendation modeling and analysis.
"""
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, date, timedelta
from pathlib import Path
import logging
import math

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ..utils.config_loader import load_config, get_config
from ..utils.logger import get_logger
from ..utils.text_utils import extract_hashtags, clean_text

logger = get_logger(__name__)


class FeatureEngineer:
    """Engineers features from processed data."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize feature engineer.

        Args:
            config_path: Path to config file.
        """
        self.config = load_config(config_path) if config_path else load_config()
        self.feature_config = get_config('settings.processing.feature_engineering', {})

        # Initialize transformers (lazy loading)
        self._text_vectorizer = None
        self._hashtag_vectorizer = None
        self._scaler = None

    def create_features(self, df: pd.DataFrame, feature_types: Optional[List[str]] = None) -> pd.DataFrame:
        """Create features from processed data.

        Args:
            df: Processed dataframe.
            feature_types: Types of features to create. If None, creates all.

        Returns:
            Dataframe with features.
        """
        if feature_types is None:
            feature_types = ['basic', 'text', 'time', 'interaction', 'composite']

        df_features = df.copy()

        # Basic features (already in processed data)
        if 'basic' in feature_types:
            df_features = self._create_basic_features(df_features)

        # Text features
        if 'text' in feature_types and 'desc_clean' in df_features.columns:
            df_features = self._create_text_features(df_features)

        # Time features
        if 'time' in feature_types:
            df_features = self._create_time_features(df_features)

        # Interaction features
        if 'interaction' in feature_types:
            df_features = self._create_interaction_features(df_features)

        # Composite features
        if 'composite' in feature_types:
            df_features = self._create_composite_features(df_features)

        # Author features (if author dimension available)
        if 'author_id' in df_features.columns:
            df_features = self._create_author_features(df_features)

        return df_features

    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic features.

        Args:
            df: Dataframe.

        Returns:
            Dataframe with basic features.
        """
        df_basic = df.copy()

        # Text length features
        if 'text_length' in df_basic.columns:
            df_basic['has_text'] = (df_basic['text_length'] > 0).astype(int)
            df_basic['text_length_category'] = pd.cut(
                df_basic['text_length'],
                bins=[-1, 0, 10, 50, 100, float('inf')],
                labels=['empty', 'short', 'medium', 'long', 'very_long']
            )

        # Hashtag features
        if 'hashtag_count' in df_basic.columns:
            df_basic['has_hashtags'] = (df_basic['hashtag_count'] > 0).astype(int)
            df_basic['hashtag_density'] = df_basic['hashtag_count'] / (df_basic['text_length'] + 1)

        return df_basic

    def _create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create text-based features.

        Args:
            df: Dataframe.

        Returns:
            Dataframe with text features.
        """
        df_text = df.copy()

        # Simple text statistics
        if 'desc_clean' in df_text.columns:
            # Character type counts (approximate for Chinese)
            df_text['chinese_char_count'] = df_text['desc_clean'].apply(
                lambda x: sum(1 for c in str(x) if '\u4e00' <= c <= '\u9fff')
            )
            df_text['digit_count'] = df_text['desc_clean'].apply(
                lambda x: sum(1 for c in str(x) if c.isdigit())
            )
            df_text['punctuation_count'] = df_text['desc_clean'].apply(
                lambda x: sum(1 for c in str(x) if c in '，。！？；：、')
            )

            # Word count (approximate for Chinese)
            df_text['word_count'] = df_text['desc_clean'].apply(
                lambda x: len(str(x).strip())
            )

            # Contains question mark
            df_text['contains_question'] = df_text['desc_clean'].str.contains(r'[？?]').fillna(False).astype(int)

            # Contains exclamation
            df_text['contains_exclamation'] = df_text['desc_clean'].str.contains(r'[！!]').fillna(False).astype(int)

            # Text complexity (simple)
            df_text['text_complexity'] = df_text['chinese_char_count'] / (df_text['word_count'] + 1)

        # Hashtag features
        if 'hashtag_list' in df_text.columns:
            # Hashtag diversity (unique hashtags across dataset)
            # This would require global statistics, so we'll compute simple ones
            df_text['hashtag_count'] = df_text['hashtag_list'].apply(
                lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
            )

            # Hashtag length statistics
            def avg_hashtag_length(tags):
                if tags is None or len(tags) == 0:
                    return 0
                # Handle both list and numpy array
                if isinstance(tags, (list, np.ndarray)):
                    lengths = [len(str(tag)) for tag in tags]
                    return np.mean(lengths) if lengths else 0
                return 0

            df_text['avg_hashtag_length'] = df_text['hashtag_list'].apply(avg_hashtag_length)

        return df_text

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features.

        Args:
            df: Dataframe.

        Returns:
            Dataframe with time features.
        """
        df_time = df.copy()

        if 'publish_date' in df_time.columns and 'crawl_date' in df_time.columns:
            # Days since publish
            # Convert to datetime for timedelta calculation
            df_time['days_since_publish'] = (pd.to_datetime(df_time['crawl_date']) - pd.to_datetime(df_time['publish_date'])).dt.days

            # Publish recency categories
            df_time['publish_recency'] = pd.cut(
                df_time['days_since_publish'],
                bins=[-1, 0, 1, 7, 30, 365, float('inf')],
                labels=['future', 'today', 'week', 'month', 'year', 'older']
            )

            # Is recent (within 7 days)
            df_time['is_recent'] = (df_time['days_since_publish'] <= 7).astype(int)

        if 'publish_hour' in df_time.columns:
            # Hour categories
            df_time['hour_category'] = pd.cut(
                df_time['publish_hour'],
                bins=[-1, 6, 9, 12, 14, 18, 21, 24],
                labels=['night', 'early_morning', 'morning', 'noon', 'afternoon', 'evening', 'late_night']
            )

            # Peak hours (assumed)
            df_time['is_peak_hour'] = (
                ((df_time['publish_hour'] >= 12) & (df_time['publish_hour'] <= 14)) |
                ((df_time['publish_hour'] >= 19) & (df_time['publish_hour'] <= 22))
            ).astype(int)

        if 'publish_weekday' in df_time.columns:
            # Weekday features
            df_time['is_weekend'] = (df_time['publish_weekday'] >= 5).astype(int)

            # Day of week
            df_time['weekday_sin'] = np.sin(2 * np.pi * df_time['publish_weekday'] / 7)
            df_time['weekday_cos'] = np.cos(2 * np.pi * df_time['publish_weekday'] / 7)

        if 'publish_date' in df_time.columns:
            # Seasonal features
            df_time['month'] = pd.to_datetime(df_time['publish_date']).dt.month
            df_time['season'] = df_time['month'].apply(self._get_season)

            # Month cyclical encoding
            df_time['month_sin'] = np.sin(2 * np.pi * df_time['month'] / 12)
            df_time['month_cos'] = np.cos(2 * np.pi * df_time['month'] / 12)

        return df_time

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction-based features.

        Args:
            df: Dataframe.

        Returns:
            Dataframe with interaction features.
        """
        df_interaction = df.copy()

        # Engagement features
        if 'engagement_score' in df_interaction.columns:
            # Log transform engagement
            df_interaction['engagement_log'] = np.log1p(df_interaction['engagement_score'].fillna(0))

            # Engagement tier
            df_interaction['engagement_tier'] = pd.qcut(
                df_interaction['engagement_score'].rank(method='first'),
                q=5,
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            )

        # Count features
        count_columns = ['like_count', 'comment_count', 'share_count', 'collect_count']
        for col in count_columns:
            if col in df_interaction.columns:
                # Log transform
                df_interaction[f'{col}_log'] = np.log1p(df_interaction[col].fillna(0))

                # Percentile
                df_interaction[f'{col}_percentile'] = df_interaction[col].rank(pct=True)

                # Binary (has interaction)
                df_interaction[f'has_{col}'] = (df_interaction[col] > 0).astype(int)

        # Interaction ratios
        if all(col in df_interaction.columns for col in ['like_count', 'comment_count']):
            df_interaction['comment_to_like_ratio'] = (
                df_interaction['comment_count'] / (df_interaction['like_count'] + 1)
            )

        if all(col in df_interaction.columns for col in ['share_count', 'like_count']):
            df_interaction['share_to_like_ratio'] = (
                df_interaction['share_count'] / (df_interaction['like_count'] + 1)
            )

        # Interaction velocity (if we had time-series data)
        # TODO: Add when we have multiple snapshots

        return df_interaction

    def _create_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite features.

        Args:
            df: Dataframe.

        Returns:
            Dataframe with composite features.
        """
        df_composite = df.copy()

        # Popularity score (weighted combination)
        weights = self.feature_config.get('popularity_weights', {})
        like_weight = weights.get('like', 1.0)
        comment_weight = weights.get('comment', 2.0)
        share_weight = weights.get('share', 3.0)

        if all(col in df_composite.columns for col in ['like_count', 'comment_count', 'share_count']):
            df_composite['popularity_score'] = (
                like_weight * df_composite['like_count'].fillna(0) +
                comment_weight * df_composite['comment_count'].fillna(0) +
                share_weight * df_composite['share_count'].fillna(0)
            )

            # Normalized popularity
            if df_composite['popularity_score'].max() > 0:
                df_composite['popularity_normalized'] = (
                    df_composite['popularity_score'] / df_composite['popularity_score'].max()
                )

        # Virality score (share to like ratio weighted by absolute numbers)
        if all(col in df_composite.columns for col in ['share_count', 'like_count']):
            df_composite['virality_score'] = (
                df_composite['share_count'].fillna(0) *
                (df_composite['share_count'].fillna(0) / (df_composite['like_count'].fillna(0) + 1))
            )

        # Content richness score
        richness_components = []

        if 'text_length' in df_composite.columns:
            richness_components.append(df_composite['text_length'] / df_composite['text_length'].max())

        if 'hashtag_count' in df_composite.columns:
            richness_components.append(df_composite['hashtag_count'] / (df_composite['hashtag_count'].max() + 1))

        if richness_components:
            df_composite['content_richness'] = np.mean(richness_components, axis=0)

        return df_composite

    def _create_author_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create author-based features.

        Args:
            df: Dataframe.

        Returns:
            Dataframe with author features.
        """
        df_author = df.copy()

        if 'author_id' in df_author.columns:
            # Author popularity (aggregate stats)
            author_stats = df_author.groupby('author_id').agg({
                'video_id': 'count',
                'like_count': 'sum',
                'comment_count': 'sum',
                'share_count': 'sum',
                'engagement_score': 'sum'
            }).add_prefix('author_').reset_index()

            # Rename columns
            author_stats = author_stats.rename(columns={'author_video_id': 'author_video_count'})

            # Merge back
            df_author = pd.merge(df_author, author_stats, on='author_id', how='left')

            # Author-level features
            if 'author_video_count' in df_author.columns:
                df_author['author_experience'] = np.log1p(df_author['author_video_count'])

            if 'author_engagement_score' in df_author.columns:
                df_author['author_popularity'] = np.log1p(df_author['author_engagement_score'])

            # Relative performance (video vs author average)
            if all(col in df_author.columns for col in ['engagement_score', 'author_engagement_score', 'author_video_count']):
                df_author['engagement_vs_author_avg'] = (
                    df_author['engagement_score'] /
                    (df_author['author_engagement_score'] / df_author['author_video_count'] + 1)
                )

        return df_author

    def _get_season(self, month: int) -> str:
        """Get season from month.

        Args:
            month: Month (1-12).

        Returns:
            Season name.
        """
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:  # 9, 10, 11
            return 'autumn'

    def create_text_embeddings(self, texts: List[str], method: str = 'tfidf',
                               max_features: int = 100) -> np.ndarray:
        """Create text embeddings.

        Args:
            texts: List of text strings.
            method: Embedding method ('tfidf', 'count').
            max_features: Maximum number of features.

        Returns:
            Embedding matrix.
        """
        if method == 'tfidf':
            if self._text_vectorizer is None:
                self._text_vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    min_df=2,
                    max_df=0.95,
                    stop_words=None  # Chinese stop words could be added
                )
            embeddings = self._text_vectorizer.fit_transform(texts)
        else:  # count
            if self._text_vectorizer is None:
                self._text_vectorizer = CountVectorizer(
                    max_features=max_features,
                    min_df=2,
                    max_df=0.95
                )
            embeddings = self._text_vectorizer.fit_transform(texts)

        return embeddings

    def create_hashtag_embeddings(self, hashtag_lists: List[List[str]],
                                  method: str = 'count') -> np.ndarray:
        """Create hashtag embeddings.

        Args:
            hashtag_lists: List of hashtag lists.
            method: Embedding method ('count', 'binary').

        Returns:
            Embedding matrix.
        """
        # Convert to space-separated strings
        hashtag_texts = [' '.join(tags) if isinstance(tags, list) else '' for tags in hashtag_lists]

        if self._hashtag_vectorizer is None:
            if method == 'binary':
                self._hashtag_vectorizer = CountVectorizer(binary=True)
            else:  # count
                self._hashtag_vectorizer = CountVectorizer()

        embeddings = self._hashtag_vectorizer.fit_transform(hashtag_texts)
        return embeddings

    def create_features_from_web_video_meta(self, df: pd.DataFrame, feature_types: Optional[List[str]] = None) -> pd.DataFrame:
        """Create features from web_video_meta data.

        This method adapts the existing feature engineering logic to work with
        web_video_meta data (high-confidence samples).

        Args:
            df: web_video_meta dataframe.
            feature_types: Types of features to create.

        Returns:
            Dataframe with features.
        """
        # First, transform web_video_meta to processed format
        from ..processing.transform import transform_web_video_meta_to_features
        df_processed = transform_web_video_meta_to_features(df)

        # Then apply existing feature engineering
        return self.create_features(df_processed, feature_types)


# Convenience functions
def create_features(df: pd.DataFrame, config_path: Optional[Path] = None,
                    feature_types: Optional[List[str]] = None) -> pd.DataFrame:
    """Create features from processed data.

    Args:
        df: Processed dataframe.
        config_path: Path to config.
        feature_types: Types of features to create.

    Returns:
        Dataframe with features.
    """
    engineer = FeatureEngineer(config_path)
    return engineer.create_features(df, feature_types)