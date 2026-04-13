"""
Data quality checks for Douyin video data.

Includes completeness, consistency, and validity checks.
"""
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, date, timedelta
from pathlib import Path
import logging

import pandas as pd
import numpy as np

from ..schemas.tables import ProcessedVideoData
from ..utils.config_loader import load_config, get_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataQualityChecker:
    """Checks data quality for processed video data."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize quality checker.

        Args:
            config_path: Path to config file.
        """
        self.config = load_config(config_path) if config_path else load_config()
        self.quality_config = get_config('settings.processing.quality_check', {})

    def check_single_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Check quality of a single record.

        Args:
            record: Data record.

        Returns:
            Dictionary with check results.
        """
        checks = {}

        # Required fields check
        checks['required_fields'] = self._check_required_fields(record)

        # Data type check
        checks['data_types'] = self._check_data_types(record)

        # Value range check
        checks['value_ranges'] = self._check_value_ranges(record)

        # Consistency check
        checks['consistency'] = self._check_consistency(record)

        # Business logic check
        checks['business_logic'] = self._check_business_logic(record)

        # Overall status
        all_passed = all(check.get('passed', False) for check in checks.values())
        checks['overall'] = {
            'passed': all_passed,
            'message': 'All checks passed' if all_passed else 'Some checks failed'
        }

        return checks

    def check_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check quality of a dataframe.

        Args:
            df: Dataframe.

        Returns:
            Dictionary with check results.
        """
        checks = {}

        # Basic statistics
        checks['basic_stats'] = self._get_basic_stats(df)

        # Missing values check
        checks['missing_values'] = self._check_missing_values(df)

        # Duplicates check
        checks['duplicates'] = self._check_duplicates(df)

        # Outliers check
        checks['outliers'] = self._check_outliers(df)

        # Distribution check
        checks['distributions'] = self._check_distributions(df)

        # Consistency check
        checks['consistency'] = self._check_dataframe_consistency(df)

        # Summary
        passed_checks = sum(1 for check in checks.values() if check.get('passed', False))
        total_checks = len(checks)
        checks['summary'] = {
            'passed_checks': passed_checks,
            'total_checks': total_checks,
            'pass_rate': passed_checks / total_checks if total_checks > 0 else 0
        }

        return checks

    def _check_required_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Check required fields.

        Args:
            record: Data record.

        Returns:
            Check result.
        """
        required_fields = ['video_id', 'source_entry', 'crawl_date', 'data_version']
        missing = [field for field in required_fields if field not in record or record[field] is None]

        return {
            'passed': len(missing) == 0,
            'missing_fields': missing,
            'message': f"Missing required fields: {missing}" if missing else "All required fields present"
        }

    def _check_data_types(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Check data types.

        Args:
            record: Data record.

        Returns:
            Check result.
        """
        issues = []

        # Check specific fields
        type_checks = [
            ('video_id', str),
            ('text_length', (int, np.integer)),
            ('hashtag_count', (int, np.integer)),
            ('engagement_score', (int, float, np.integer, np.floating)),
            ('crawl_date', (date, datetime, pd.Timestamp, str))
        ]

        for field, expected_type in type_checks:
            if field in record and record[field] is not None:
                if not isinstance(record[field], expected_type):
                    issues.append(f"{field}: expected {expected_type}, got {type(record[field])}")

        # Check list fields
        list_fields = ['hashtag_list']
        for field in list_fields:
            if field in record and record[field] is not None:
                if not isinstance(record[field], list):
                    issues.append(f"{field}: expected list, got {type(record[field])}")

        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'message': f"Data type issues: {issues}" if issues else "All data types correct"
        }

    def _check_value_ranges(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Check value ranges.

        Args:
            record: Data record.

        Returns:
            Check result.
        """
        issues = []

        # Text length
        if 'text_length' in record and record['text_length'] is not None:
            if record['text_length'] < 0:
                issues.append(f"text_length negative: {record['text_length']}")
            if record['text_length'] > 10000:  # Arbitrary large limit
                issues.append(f"text_length too large: {record['text_length']}")

        # Hashtag count
        if 'hashtag_count' in record and record['hashtag_count'] is not None:
            if record['hashtag_count'] < 0:
                issues.append(f"hashtag_count negative: {record['hashtag_count']}")
            if record['hashtag_count'] > 100:  # Arbitrary limit
                issues.append(f"hashtag_count too large: {record['hashtag_count']}")

        # Count fields
        count_fields = ['like_count', 'comment_count', 'share_count', 'collect_count']
        for field in count_fields:
            if field in record and record[field] is not None:
                if record[field] < 0:
                    issues.append(f"{field} negative: {record[field]}")
                if record[field] > 1e9:  # 1 billion
                    issues.append(f"{field} unrealistically large: {record[field]}")

        # Engagement score
        if 'engagement_score' in record and record['engagement_score'] is not None:
            if record['engagement_score'] < 0:
                issues.append(f"engagement_score negative: {record['engagement_score']}")
            if record['engagement_score'] > 1e12:  # Arbitrary large limit
                issues.append(f"engagement_score too large: {record['engagement_score']}")

        # Date fields
        if 'publish_date' in record and record['publish_date']:
            try:
                if isinstance(record['publish_date'], str):
                    publish_date = datetime.fromisoformat(record['publish_date'].replace('Z', '+00:00')).date()
                else:
                    publish_date = record['publish_date']
                    if isinstance(publish_date, datetime):
                        publish_date = publish_date.date()

                today = date.today()
                if publish_date > today:
                    issues.append(f"publish_date in future: {publish_date}")
                # Douyin launched in 2016
                if publish_date < date(2016, 1, 1):
                    issues.append(f"publish_date too old: {publish_date}")
            except (ValueError, TypeError, AttributeError):
                issues.append(f"publish_date invalid format: {record['publish_date']}")

        if 'crawl_date' in record and record['crawl_date']:
            try:
                if isinstance(record['crawl_date'], str):
                    crawl_date = datetime.fromisoformat(record['crawl_date'].replace('Z', '+00:00')).date()
                else:
                    crawl_date = record['crawl_date']
                    if isinstance(crawl_date, datetime):
                        crawl_date = crawl_date.date()

                today = date.today()
                if crawl_date > today:
                    issues.append(f"crawl_date in future: {crawl_date}")
                # Project start date (arbitrary)
                if crawl_date < date(2020, 1, 1):
                    issues.append(f"crawl_date too old: {crawl_date}")
            except (ValueError, TypeError, AttributeError):
                issues.append(f"crawl_date invalid format: {record['crawl_date']}")

        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'message': f"Value range issues: {issues}" if issues else "All values in valid ranges"
        }

    def _check_consistency(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Check internal consistency.

        Args:
            record: Data record.

        Returns:
            Check result.
        """
        issues = []

        # Hashtag count vs list
        if 'hashtag_count' in record and 'hashtag_list' in record:
            list_count = len(record['hashtag_list']) if isinstance(record['hashtag_list'], list) else 0
            if record['hashtag_count'] != list_count:
                issues.append(f"hashtag_count ({record['hashtag_count']}) != list length ({list_count})")

        # Text length vs actual text
        if 'text_length' in record and 'desc_clean' in record:
            actual_length = len(str(record['desc_clean']))
            if record['text_length'] != actual_length:
                issues.append(f"text_length ({record['text_length']}) != actual length ({actual_length})")

        # Publish date vs crawl date
        if 'publish_date' in record and 'crawl_date' in record:
            try:
                if isinstance(record['publish_date'], str):
                    publish_date = datetime.fromisoformat(record['publish_date'].replace('Z', '+00:00')).date()
                else:
                    publish_date = record['publish_date']
                    if isinstance(publish_date, datetime):
                        publish_date = publish_date.date()

                if isinstance(record['crawl_date'], str):
                    crawl_date = datetime.fromisoformat(record['crawl_date'].replace('Z', '+00:00')).date()
                else:
                    crawl_date = record['crawl_date']
                    if isinstance(crawl_date, datetime):
                        crawl_date = crawl_date.date()

                if publish_date > crawl_date:
                    issues.append(f"publish_date ({publish_date}) > crawl_date ({crawl_date})")
            except (ValueError, TypeError, AttributeError):
                pass  # Already caught in value ranges

        # Engagement score consistency
        if 'engagement_score' in record:
            weights = self.quality_config.get('engagement_weights', {'like': 1.0, 'comment': 2.0, 'share': 3.0})
            calculated = 0.0
            if 'like_count' in record and record['like_count']:
                calculated += weights['like'] * record['like_count']
            if 'comment_count' in record and record['comment_count']:
                calculated += weights['comment'] * record['comment_count']
            if 'share_count' in record and record['share_count']:
                calculated += weights['share'] * record['share_count']

            if calculated > 0 and record['engagement_score']:
                diff = abs(calculated - record['engagement_score'])
                if diff > 0.01:  # Allow small floating point differences
                    issues.append(f"engagement_score mismatch: recorded={record['engagement_score']}, calculated={calculated}")

        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'message': f"Consistency issues: {issues}" if issues else "All consistency checks passed"
        }

    def _check_business_logic(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Check business logic rules.

        Args:
            record: Data record.

        Returns:
            Check result.
        """
        issues = []

        # Like count should be >= comment count (typically)
        if 'like_count' in record and 'comment_count' in record:
            if record['like_count'] is not None and record['comment_count'] is not None:
                if record['like_count'] < record['comment_count']:
                    issues.append(f"like_count ({record['like_count']}) < comment_count ({record['comment_count']})")

        # Share count should be <= like count (typically)
        if 'like_count' in record and 'share_count' in record:
            if record['like_count'] is not None and record['share_count'] is not None:
                if record['share_count'] > record['like_count'] * 10:  # Allow some variance
                    issues.append(f"share_count ({record['share_count']}) > 10x like_count ({record['like_count']})")

        # Hashtag count should be reasonable
        if 'hashtag_count' in record and record['hashtag_count'] is not None:
            if record['hashtag_count'] > 50:  # Arbitrary limit
                issues.append(f"hashtag_count ({record['hashtag_count']}) > 50")

        # Text length vs hashtag count
        if 'text_length' in record and 'hashtag_count' in record:
            if record['text_length'] is not None and record['hashtag_count'] is not None:
                if record['text_length'] > 0 and record['hashtag_count'] > record['text_length']:
                    issues.append(f"hashtag_count ({record['hashtag_count']}) > text_length ({record['text_length']})")

        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'message': f"Business logic issues: {issues}" if issues else "Business logic checks passed"
        }

    def _get_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic statistics.

        Args:
            df: Dataframe.

        Returns:
            Statistics dictionary.
        """
        stats = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'dtypes': {col: str(df[col].dtype) for col in df.columns}
        }

        # Value counts for categorical columns
        categorical_cols = ['source_entry', 'data_version']
        for col in categorical_cols:
            if col in df.columns:
                stats[f'{col}_value_counts'] = df[col].value_counts().to_dict()

        return stats

    def _check_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check missing values.

        Args:
            df: Dataframe.

        Returns:
            Check result.
        """
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100

        issues = []
        for col in df.columns:
            if missing_counts[col] > 0:
                issues.append(f"{col}: {missing_counts[col]} missing ({missing_pct[col]:.1f}%)")

        return {
            'passed': len(issues) == 0,
            'missing_counts': missing_counts.to_dict(),
            'missing_pct': missing_pct.to_dict(),
            'issues': issues,
            'message': f"Missing values in {len(issues)} columns" if issues else "No missing values"
        }

    def _check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check duplicates.

        Args:
            df: Dataframe.

        Returns:
            Check result.
        """
        # Check duplicate rows
        duplicate_rows = df.duplicated().sum()
        # Check duplicate video_ids
        duplicate_video_ids = df['video_id'].duplicated().sum() if 'video_id' in df.columns else 0

        issues = []
        if duplicate_rows > 0:
            issues.append(f"{duplicate_rows} duplicate rows")
        if duplicate_video_ids > 0:
            issues.append(f"{duplicate_video_ids} duplicate video_ids")

        return {
            'passed': len(issues) == 0,
            'duplicate_rows': int(duplicate_rows),
            'duplicate_video_ids': int(duplicate_video_ids),
            'issues': issues,
            'message': f"Duplicate issues: {issues}" if issues else "No duplicates found"
        }

    def _check_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check outliers using IQR method.

        Args:
            df: Dataframe.

        Returns:
            Check result.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}

        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outlier_mask.sum()

            if outlier_count > 0:
                outliers[col] = {
                    'count': int(outlier_count),
                    'pct': outlier_count / len(df) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'min': df[col].min(),
                    'max': df[col].max()
                }

        issues = [f"{col}: {info['count']} outliers ({info['pct']:.1f}%)" for col, info in outliers.items()]

        return {
            'passed': len(issues) == 0,
            'outliers': outliers,
            'issues': issues,
            'message': f"Outliers in {len(issues)} columns" if issues else "No outliers found"
        }

    def _check_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data distributions.

        Args:
            df: Dataframe.

        Returns:
            Check result.
        """
        distributions = {}

        # Check numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            distributions[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                '25%': float(df[col].quantile(0.25)),
                '50%': float(df[col].median()),
                '75%': float(df[col].quantile(0.75)),
                'max': float(df[col].max()),
                'skew': float(df[col].skew()),
                'kurtosis': float(df[col].kurtosis())
            }

        # Check for zero-inflated columns
        issues = []
        for col, stats in distributions.items():
            if stats['min'] == stats['max']:
                issues.append(f"{col}: constant value {stats['min']}")
            elif stats['std'] == 0:
                issues.append(f"{col}: zero variance")
            elif abs(stats['skew']) > 10:  # Highly skewed
                issues.append(f"{col}: highly skewed (skew={stats['skew']:.2f})")

        return {
            'passed': len(issues) == 0,
            'distributions': distributions,
            'issues': issues,
            'message': f"Distribution issues: {issues}" if issues else "Distributions look reasonable"
        }

    def _check_dataframe_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check dataframe consistency.

        Args:
            df: Dataframe.

        Returns:
            Check result.
        """
        issues = []

        # Check hashtag_count vs hashtag_list
        if 'hashtag_count' in df.columns and 'hashtag_list' in df.columns:
            mismatched = df.apply(
                lambda row: len(row['hashtag_list']) if isinstance(row['hashtag_list'], list) else 0 != row['hashtag_count'],
                axis=1
            ).sum()
            if mismatched > 0:
                issues.append(f"{mismatched} rows with hashtag_count != hashtag_list length")

        # Check text_length vs desc_clean
        if 'text_length' in df.columns and 'desc_clean' in df.columns:
            mismatched = df.apply(
                lambda row: len(str(row['desc_clean'])) != row['text_length'],
                axis=1
            ).sum()
            if mismatched > 0:
                issues.append(f"{mismatched} rows with text_length != desc_clean length")

        # Check date order
        if 'publish_date' in df.columns and 'crawl_date' in df.columns:
            future_publish = (df['publish_date'] > df['crawl_date']).sum()
            if future_publish > 0:
                issues.append(f"{future_publish} rows with publish_date > crawl_date")

        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'message': f"Consistency issues: {issues}" if issues else "All consistency checks passed"
        }

    def generate_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive quality report.

        Args:
            df: Dataframe.

        Returns:
            Quality report.
        """
        checks = self.check_dataframe(df)

        # Summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(df),
            'dataset_columns': len(df.columns),
            'pass_rate': checks.get('summary', {}).get('pass_rate', 0),
            'overall_status': 'PASS' if checks.get('summary', {}).get('pass_rate', 0) > 0.8 else 'FAIL'
        }

        # Issues by severity
        issues = []
        for check_name, check_result in checks.items():
            if check_name != 'summary' and check_name != 'basic_stats':
                if not check_result.get('passed', False):
                    issues.append({
                        'check': check_name,
                        'message': check_result.get('message', ''),
                        'issues': check_result.get('issues', [])
                    })

        report = {
            'summary': summary,
            'checks': checks,
            'issues': issues,
            'recommendations': self._generate_recommendations(checks)
        }

        return report

    def _generate_recommendations(self, checks: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on check results.

        Args:
            checks: Check results.

        Returns:
            List of recommendations.
        """
        recommendations = []

        # Missing values
        missing_values = checks.get('missing_values', {})
        if missing_values.get('issues'):
            recommendations.append("Consider imputing missing values or removing rows with high missing rates.")

        # Duplicates
        duplicates = checks.get('duplicates', {})
        if duplicates.get('duplicate_rows', 0) > 0:
            recommendations.append("Remove duplicate rows to avoid data leakage.")
        if duplicates.get('duplicate_video_ids', 0) > 0:
            recommendations.append("Investigate duplicate video_ids - may need deduplication.")

        # Outliers
        outliers = checks.get('outliers', {})
        if outliers.get('issues'):
            recommendations.append("Review outliers - may need winsorization or transformation.")

        # Distributions
        distributions = checks.get('distributions', {})
        if distributions.get('issues'):
            recommendations.append("Consider transformations for highly skewed variables.")

        # Consistency
        consistency = checks.get('consistency', {})
        if consistency.get('issues'):
            recommendations.append("Fix consistency issues in data generation pipeline.")

        return recommendations


# Convenience functions
def check_data_quality(df: pd.DataFrame, config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Check data quality.

    Args:
        df: Dataframe.
        config_path: Path to config.

    Returns:
        Quality check results.
    """
    checker = DataQualityChecker(config_path)
    return checker.check_dataframe(df)


def generate_quality_report(df: pd.DataFrame, config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Generate quality report.

    Args:
        df: Dataframe.
        config_path: Path to config.

    Returns:
        Quality report.
    """
    checker = DataQualityChecker(config_path)
    return checker.generate_quality_report(df)