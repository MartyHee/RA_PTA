"""
Exploratory Data Analysis (EDA) for Douyin video data.

Provides basic statistics, visualizations, and insights.
"""
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, date
from pathlib import Path
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.config_loader import load_config, get_config
from ..utils.logger import get_logger
from ..utils.io_utils import save_figure

logger = get_logger(__name__)


class EDAAnalyzer:
    """Performs exploratory data analysis on processed video data."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize EDA analyzer.

        Args:
            config_path: Path to config file.
        """
        self.config = load_config(config_path) if config_path else load_config()
        self.output_dir = Path(get_config('settings.paths.processed_data', './data/processed'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set plot style
        plt.style.use('seaborn-v0_8')
        self.figsize = (12, 8)
        self.colors = sns.color_palette('husl', 8)

    def basic_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic summary statistics.

        Args:
            df: Processed dataframe.

        Returns:
            Summary statistics.
        """
        summary = {
            'dataset_info': {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'date_range': {
                    'min_crawl_date': df['crawl_date'].min().isoformat() if 'crawl_date' in df.columns else None,
                    'max_crawl_date': df['crawl_date'].max().isoformat() if 'crawl_date' in df.columns else None,
                    'min_publish_date': df['publish_date'].min().isoformat() if 'publish_date' in df.columns else None,
                    'max_publish_date': df['publish_date'].max().isoformat() if 'publish_date' in df.columns else None
                }
            },
            'source_distribution': {},
            'numeric_summary': {},
            'categorical_summary': {}
        }

        # Source distribution
        if 'source_entry' in df.columns:
            source_counts = df['source_entry'].value_counts()
            summary['source_distribution'] = {
                'counts': source_counts.to_dict(),
                'percentages': (source_counts / len(df) * 100).round(2).to_dict()
            }

        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary['numeric_summary'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                '25%': float(df[col].quantile(0.25)),
                '50%': float(df[col].median()),
                '75%': float(df[col].quantile(0.75)),
                'max': float(df[col].max()),
                'missing': int(df[col].isnull().sum()),
                'missing_pct': float(df[col].isnull().sum() / len(df) * 100)
            }

        # Categorical columns summary
        categorical_cols = ['data_version', 'author_id', 'hashtag_count_category']
        categorical_cols = [col for col in categorical_cols if col in df.columns]

        for col in categorical_cols:
            value_counts = df[col].value_counts()
            summary['categorical_summary'][col] = {
                'unique_values': int(df[col].nunique()),
                'top_values': value_counts.head(10).to_dict()
            }

        # Engagement score summary
        if 'engagement_score' in df.columns:
            engagement = df['engagement_score']
            summary['engagement_summary'] = {
                'total': float(engagement.sum()),
                'mean': float(engagement.mean()),
                'median': float(engagement.median()),
                'std': float(engagement.std()),
                'skew': float(engagement.skew()),
                'kurtosis': float(engagement.kurtosis())
            }

        return summary

    def analyze_distributions(self, df: pd.DataFrame, save_plots: bool = True) -> Dict[str, Any]:
        """Analyze distributions of key variables.

        Args:
            df: Processed dataframe.
            save_plots: Whether to save plots to file.

        Returns:
            Distribution analysis results.
        """
        results = {}

        # Key numeric columns to analyze
        numeric_cols = ['like_count', 'comment_count', 'share_count', 'engagement_score', 'text_length']
        numeric_cols = [col for col in numeric_cols if col in df.columns]

        for col in numeric_cols:
            # Basic stats
            data = df[col].dropna()
            results[col] = {
                'skewness': float(data.skew()),
                'kurtosis': float(data.kurtosis()),
                'is_normal': self._is_approximately_normal(data)
            }

            # Create histogram
            if save_plots:
                self._plot_histogram(data, col, f'distribution_{col}')

        # Categorical distributions
        if 'source_entry' in df.columns:
            if save_plots:
                self._plot_bar_chart(df['source_entry'], 'source_entry', 'Source Entry Distribution')

        if 'hashtag_count' in df.columns:
            if save_plots:
                self._plot_histogram(df['hashtag_count'], 'hashtag_count', 'Hashtag Count Distribution')

        return results

    def analyze_correlations(self, df: pd.DataFrame, save_plots: bool = True) -> pd.DataFrame:
        """Analyze correlations between variables.

        Args:
            df: Processed dataframe.
            save_plots: Whether to save plots to file.

        Returns:
            Correlation matrix.
        """
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_cols = [col for col in numeric_cols if df[col].nunique() > 1]

        if len(correlation_cols) < 2:
            logger.warning("Not enough numeric columns for correlation analysis")
            return pd.DataFrame()

        # Calculate correlation matrix
        corr_matrix = df[correlation_cols].corr()

        # Plot heatmap
        if save_plots and len(correlation_cols) > 1:
            self._plot_correlation_heatmap(corr_matrix, 'correlation_heatmap')

        # Identify strong correlations
        strong_correlations = []
        for i in range(len(correlation_cols)):
            for j in range(i + 1, len(correlation_cols)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.7:
                    strong_correlations.append({
                        'var1': correlation_cols[i],
                        'var2': correlation_cols[j],
                        'correlation': float(corr)
                    })

        logger.info(f"Found {len(strong_correlations)} strong correlations (|r| > 0.7)")

        return corr_matrix

    def analyze_time_patterns(self, df: pd.DataFrame, save_plots: bool = True) -> Dict[str, Any]:
        """Analyze time-based patterns.

        Args:
            df: Processed dataframe.
            save_plots: Whether to save plots to file.

        Returns:
            Time pattern analysis results.
        """
        results = {}

        # Ensure datetime columns
        time_df = df.copy()
        if 'publish_date' in time_df.columns:
            time_df['publish_date'] = pd.to_datetime(time_df['publish_date'])
        if 'crawl_date' in time_df.columns:
            time_df['crawl_date'] = pd.to_datetime(time_df['crawl_date'])

        # Daily volume
        if 'publish_date' in time_df.columns:
            daily_counts = time_df.groupby(time_df['publish_date'].dt.date).size()
            results['daily_volume'] = {
                'mean': float(daily_counts.mean()),
                'std': float(daily_counts.std()),
                'min': float(daily_counts.min()),
                'max': float(daily_counts.max()),
                'trend': 'increasing' if daily_counts.iloc[-1] > daily_counts.iloc[0] else 'decreasing'
            }

            if save_plots:
                self._plot_time_series(daily_counts, 'publish_date', 'Daily Video Volume')

        # Hourly patterns
        if 'publish_hour' in time_df.columns:
            hourly_counts = time_df['publish_hour'].value_counts().sort_index()
            results['hourly_distribution'] = hourly_counts.to_dict()

            if save_plots:
                self._plot_bar_chart(time_df['publish_hour'], 'publish_hour', 'Video Publishing Hour Distribution')

        # Weekday patterns
        if 'publish_weekday' in time_df.columns:
            weekday_counts = time_df['publish_weekday'].value_counts().sort_index()
            weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            weekday_mapping = {i: weekday_names[i] for i in range(7)}
            weekday_counts.index = weekday_counts.index.map(weekday_mapping)
            results['weekday_distribution'] = weekday_counts.to_dict()

            if save_plots:
                self._plot_bar_chart(weekday_counts, 'publish_weekday', 'Video Publishing Weekday Distribution')

        # Engagement by time
        if 'publish_hour' in time_df.columns and 'engagement_score' in time_df.columns:
            engagement_by_hour = time_df.groupby('publish_hour')['engagement_score'].mean()
            results['engagement_by_hour'] = engagement_by_hour.to_dict()

            if save_plots:
                self._plot_line_chart(engagement_by_hour, 'publish_hour', 'engagement_score',
                                      'Average Engagement by Hour')

        return results

    def analyze_text_features(self, df: pd.DataFrame, save_plots: bool = True) -> Dict[str, Any]:
        """Analyze text features.

        Args:
            df: Processed dataframe.
            save_plots: Whether to save plots to file.

        Returns:
            Text analysis results.
        """
        results = {}

        if 'desc_clean' not in df.columns:
            logger.warning("No desc_clean column for text analysis")
            return results

        # Text length analysis
        if 'text_length' in df.columns:
            text_lengths = df['text_length']
            results['text_length'] = {
                'mean': float(text_lengths.mean()),
                'median': float(text_lengths.median()),
                'std': float(text_lengths.std()),
                'min': int(text_lengths.min()),
                'max': int(text_lengths.max()),
                'zero_count': int((text_lengths == 0).sum()),
                'zero_pct': float((text_lengths == 0).sum() / len(df) * 100)
            }

            if save_plots:
                self._plot_histogram(text_lengths, 'text_length', 'Text Length Distribution')

        # Hashtag analysis
        if 'hashtag_count' in df.columns:
            hashtag_counts = df['hashtag_count']
            results['hashtag_count'] = {
                'mean': float(hashtag_counts.mean()),
                'median': float(hashtag_counts.median()),
                'std': float(hashtag_counts.std()),
                'zero_count': int((hashtag_counts == 0).sum()),
                'zero_pct': float((hashtag_counts == 0).sum() / len(df) * 100),
                'max_hashtags': int(hashtag_counts.max())
            }

            # Top hashtags
            if 'hashtag_list' in df.columns:
                all_hashtags = []
                for tags in df['hashtag_list']:
                    if isinstance(tags, list):
                        all_hashtags.extend(tags)

                if all_hashtags:
                    from collections import Counter
                    top_hashtags = Counter(all_hashtags).most_common(20)
                    results['top_hashtags'] = dict(top_hashtags)

                    if save_plots:
                        self._plot_top_hashtags(top_hashtags, 'top_hashtags')

        # Text length vs engagement
        if 'text_length' in df.columns and 'engagement_score' in df.columns:
            correlation = df['text_length'].corr(df['engagement_score'])
            results['text_engagement_correlation'] = float(correlation)

            if save_plots:
                self._plot_scatter(df, 'text_length', 'engagement_score',
                                   'Text Length vs Engagement Score')

        return results

    def analyze_engagement(self, df: pd.DataFrame, save_plots: bool = True) -> Dict[str, Any]:
        """Analyze engagement metrics.

        Args:
            df: Processed dataframe.
            save_plots: Whether to save plots to file.

        Returns:
            Engagement analysis results.
        """
        results = {}

        # Engagement score distribution
        if 'engagement_score' in df.columns:
            engagement = df['engagement_score']
            results['engagement_distribution'] = {
                'mean': float(engagement.mean()),
                'median': float(engagement.median()),
                'std': float(engagement.std()),
                'skew': float(engagement.skew()),
                'kurtosis': float(engagement.kurtosis())
            }

            if save_plots:
                self._plot_histogram(engagement, 'engagement_score', 'Engagement Score Distribution', log_scale=True)

        # Component analysis
        components = ['like_count', 'comment_count', 'share_count']
        components = [col for col in components if col in df.columns]

        for col in components:
            if col in df.columns:
                data = df[col]
                results[col] = {
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'std': float(data.std()),
                    'zero_count': int((data == 0).sum()),
                    'zero_pct': float((data == 0).sum() / len(df) * 100)
                }

        # Ratios
        if all(col in df.columns for col in ['like_count', 'comment_count', 'share_count']):
            df_ratios = df.copy()
            df_ratios['comment_to_like'] = df_ratios['comment_count'] / (df_ratios['like_count'] + 1)
            df_ratios['share_to_like'] = df_ratios['share_count'] / (df_ratios['like_count'] + 1)

            results['ratios'] = {
                'comment_to_like_mean': float(df_ratios['comment_to_like'].mean()),
                'share_to_like_mean': float(df_ratios['share_to_like'].mean())
            }

            if save_plots:
                self._plot_boxplot(df_ratios[['comment_to_like', 'share_to_like']],
                                   'Engagement Ratios Distribution')

        # Engagement by source
        if 'source_entry' in df.columns and 'engagement_score' in df.columns:
            engagement_by_source = df.groupby('source_entry')['engagement_score'].agg(['mean', 'median', 'count'])
            results['engagement_by_source'] = engagement_by_source.to_dict()

            if save_plots:
                self._plot_bar_chart(engagement_by_source['mean'], 'source_entry', 'engagement_score',
                                     'Average Engagement by Source')

        return results

    def generate_report(self, df: pd.DataFrame, save_all_plots: bool = True) -> Dict[str, Any]:
        """Generate comprehensive EDA report.

        Args:
            df: Processed dataframe.
            save_all_plots: Whether to save all plots.

        Returns:
            Comprehensive EDA report.
        """
        logger.info("Starting comprehensive EDA...")

        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_shape': {'rows': len(df), 'columns': len(df.columns)},
            'basic_summary': self.basic_summary(df),
            'distributions': self.analyze_distributions(df, save_all_plots),
            'correlations': self.analyze_correlations(df, save_all_plots).to_dict() if not self.analyze_correlations(df, False).empty else {},
            'time_patterns': self.analyze_time_patterns(df, save_all_plots),
            'text_features': self.analyze_text_features(df, save_all_plots),
            'engagement': self.analyze_engagement(df, save_all_plots),
            'insights': []
        }

        # Generate insights
        insights = self._generate_insights(report)
        report['insights'] = insights

        # Save report
        report_path = self.output_dir / 'eda_report.json'
        import json
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"EDA report saved to {report_path}")

        return report

    def _generate_insights(self, report: Dict[str, Any]) -> List[str]:
        """Generate insights from EDA results.

        Args:
            report: EDA report.

        Returns:
            List of insights.
        """
        insights = []

        # Basic insights
        basic = report.get('basic_summary', {})
        if basic.get('dataset_info', {}).get('total_records', 0) < 100:
            insights.append("Dataset is relatively small (< 100 records). Consider collecting more data.")

        # Distribution insights
        distributions = report.get('distributions', {})
        for col, stats in distributions.items():
            if stats.get('skewness', 0) > 3:
                insights.append(f"{col} is highly right-skewed (skewness = {stats['skewness']:.2f}). Consider log transformation.")

        # Correlation insights
        correlations = report.get('correlations', {})
        if isinstance(correlations, dict):
            for var1, corr_dict in correlations.items():
                if isinstance(corr_dict, dict):
                    for var2, corr in corr_dict.items():
                        if var1 != var2 and abs(corr) > 0.8:
                            insights.append(f"Strong correlation between {var1} and {var2} (r = {corr:.2f}).")

        # Time pattern insights
        time_patterns = report.get('time_patterns', {})
        if 'hourly_distribution' in time_patterns:
            hourly = time_patterns['hourly_distribution']
            peak_hour = max(hourly.items(), key=lambda x: x[1])[0] if hourly else None
            if peak_hour:
                insights.append(f"Peak publishing hour is {peak_hour}:00.")

        # Engagement insights
        engagement = report.get('engagement', {})
        if 'engagement_by_source' in engagement:
            source_engagement = engagement['engagement_by_source']
            if 'mean' in source_engagement:
                best_source = max(source_engagement['mean'].items(), key=lambda x: x[1])[0] if source_engagement['mean'] else None
                if best_source:
                    insights.append(f"Highest average engagement from '{best_source}' source.")

        return insights

    # Plotting helper methods
    def _plot_histogram(self, data: pd.Series, xlabel: str, title: str, log_scale: bool = False):
        """Plot histogram."""
        plt.figure(figsize=self.figsize)
        if log_scale:
            data_to_plot = np.log1p(data)
            plt.hist(data_to_plot, bins=50, alpha=0.7, color=self.colors[0])
            plt.xlabel(f'Log({xlabel} + 1)')
        else:
            plt.hist(data, bins=50, alpha=0.7, color=self.colors[0])
            plt.xlabel(xlabel)
        plt.ylabel('Frequency')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        save_figure(plt, self.output_dir / f'{title.lower().replace(" ", "_")}.png')
        plt.close()

    def _plot_bar_chart(self, data: pd.Series, xlabel: str, title: str, ylabel: str = 'Count'):
        """Plot bar chart."""
        plt.figure(figsize=self.figsize)
        if isinstance(data, pd.Series):
            # If data appears to be categorical (non-numeric), compute value counts
            if not pd.api.types.is_numeric_dtype(data):
                data = data.value_counts().sort_index()
            data.plot(kind='bar', color=self.colors[0], alpha=0.7)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        save_figure(plt, self.output_dir / f'{title.lower().replace(" ", "_")}.png')
        plt.close()

    def _plot_correlation_heatmap(self, corr_matrix: pd.DataFrame, title: str):
        """Plot correlation heatmap."""
        plt.figure(figsize=(max(10, len(corr_matrix) * 0.8), max(8, len(corr_matrix) * 0.6)))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
        plt.title(title)
        save_figure(plt, self.output_dir / f'{title.lower().replace(" ", "_")}.png')
        plt.close()

    def _plot_time_series(self, data: pd.Series, xlabel: str, title: str):
        """Plot time series."""
        plt.figure(figsize=self.figsize)
        data.plot(kind='line', color=self.colors[0], marker='o', markersize=3)
        plt.xlabel(xlabel)
        plt.ylabel('Count')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        save_figure(plt, self.output_dir / f'{title.lower().replace(" ", "_")}.png')
        plt.close()

    def _plot_line_chart(self, data: pd.Series, xlabel: str, ylabel: str, title: str):
        """Plot line chart."""
        plt.figure(figsize=self.figsize)
        data.plot(kind='line', color=self.colors[0], marker='o', linewidth=2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        save_figure(plt, self.output_dir / f'{title.lower().replace(" ", "_")}.png')
        plt.close()

    def _plot_scatter(self, df: pd.DataFrame, xcol: str, ycol: str, title: str):
        """Plot scatter plot."""
        plt.figure(figsize=self.figsize)
        plt.scatter(df[xcol], df[ycol], alpha=0.5, color=self.colors[0])
        plt.xlabel(xcol)
        plt.ylabel(ycol)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        save_figure(plt, self.output_dir / f'{title.lower().replace(" ", "_")}.png')
        plt.close()

    def _plot_boxplot(self, data: pd.DataFrame, title: str):
        """Plot boxplot."""
        plt.figure(figsize=self.figsize)
        data.boxplot()
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        save_figure(plt, self.output_dir / f'{title.lower().replace(" ", "_")}.png')
        plt.close()

    def _plot_top_hashtags(self, top_hashtags: List[Tuple[str, int]], title: str):
        """Plot top hashtags."""
        hashtags, counts = zip(*top_hashtags)
        plt.figure(figsize=(12, 6))
        plt.barh(range(len(hashtags)), counts, color=self.colors[0])
        plt.yticks(range(len(hashtags)), hashtags)
        plt.xlabel('Count')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        save_figure(plt, self.output_dir / f'{title.lower().replace(" ", "_")}.png')
        plt.close()

    def _is_approximately_normal(self, data: pd.Series, threshold: float = 0.05) -> bool:
        """Check if data is approximately normal using skewness and kurtosis.

        Args:
            data: Data series.
            threshold: Threshold for skewness and excess kurtosis.

        Returns:
            True if approximately normal.
        """
        if len(data) < 20:
            return False

        skewness = abs(data.skew())
        kurtosis = abs(data.kurtosis())

        return skewness < threshold and kurtosis < threshold


# Convenience functions
def run_basic_eda(df: pd.DataFrame, config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Run basic EDA.

    Args:
        df: Processed dataframe.
        config_path: Path to config.

    Returns:
        Basic EDA results.
    """
    analyzer = EDAAnalyzer(config_path)
    return analyzer.basic_summary(df)


def run_comprehensive_eda(df: pd.DataFrame, config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Run comprehensive EDA.

    Args:
        df: Processed dataframe.
        config_path: Path to config.

    Returns:
        Comprehensive EDA report.
    """
    analyzer = EDAAnalyzer(config_path)
    return analyzer.generate_report(df)