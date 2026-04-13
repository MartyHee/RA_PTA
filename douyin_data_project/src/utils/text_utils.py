"""
Text utilities for Douyin data project.

Includes text cleaning, hashtag extraction, and other text processing functions.
"""
import re
import string
from typing import List, Optional, Set, Tuple
import emoji
import unicodedata


def clean_text(text: Optional[str], remove_urls: bool = True,
               remove_emojis: bool = True, remove_special_chars: bool = False) -> str:
    """Clean text by removing URLs, emojis, and special characters.

    Args:
        text: Input text.
        remove_urls: Whether to remove URLs.
        remove_emojis: Whether to remove emojis.
        remove_special_chars: Whether to remove special characters.

    Returns:
        Cleaned text.
    """
    if not text:
        return ''

    # Convert to string
    text = str(text)

    # Remove URLs
    if remove_urls:
        text = remove_urls(text)

    # Remove emojis
    if remove_emojis:
        text = remove_emojis(text)

    # Remove special characters
    if remove_special_chars:
        text = remove_special_chars(text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def remove_urls(text: str) -> str:
    """Remove URLs from text.

    Args:
        text: Input text.

    Returns:
        Text without URLs.
    """
    # Match common URL patterns
    url_patterns = [
        r'https?://\S+',  # http/https URLs
        r'www\.\S+',      # www URLs
        r'\S+\.(com|cn|net|org|io|edu|gov)\S*',  # Domain extensions
        r't\.cn/\S+',     # Short URLs
        r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/\S*'     # General domain pattern
    ]

    for pattern in url_patterns:
        text = re.sub(pattern, '', text)

    return text


def remove_emojis(text: str) -> str:
    """Remove emojis from text.

    Args:
        text: Input text.

    Returns:
        Text without emojis.
    """
    # Remove emoji characters
    text = emoji.replace_emoji(text, replace='')

    # Remove other non-text characters
    # Keep Chinese characters, letters, numbers, and basic punctuation
    cleaned_chars = []
    for char in text:
        # Keep Chinese characters
        if '\u4e00' <= char <= '\u9fff':
            cleaned_chars.append(char)
        # Keep ASCII letters and digits
        elif char.isalnum():
            cleaned_chars.append(char)
        # Keep basic punctuation and whitespace
        elif char in '，。！？；：、,.;:!?()[]{}「」【】《》':
            cleaned_chars.append(char)
        elif char.isspace():
            cleaned_chars.append(char)
        # Remove everything else

    return ''.join(cleaned_chars)


def remove_special_chars(text: str, keep_chinese: bool = True) -> str:
    """Remove special characters from text.

    Args:
        text: Input text.
        keep_chinese: Whether to keep Chinese characters.

    Returns:
        Text without special characters.
    """
    if keep_chinese:
        # Keep Chinese characters, letters, digits, and basic punctuation
        pattern = r'[^\u4e00-\u9fff\w\s,.;:!?()\[\]{}\-]'
    else:
        # Keep only alphanumeric and basic punctuation
        pattern = r'[^\w\s,.;:!?()\[\]{}\-]'

    text = re.sub(pattern, '', text)
    return text


def extract_hashtags(text: str) -> List[str]:
    """Extract hashtags from text.

    Args:
        text: Input text.

    Returns:
        List of hashtags (without # symbol).
    """
    if not text:
        return []

    # Match hashtags with # symbol
    # Supports Chinese and English hashtags
    hashtag_pattern = r'#([^#\s\u200b]+)'
    matches = re.findall(hashtag_pattern, text)

    # Clean hashtags
    cleaned_tags = []
    for tag in matches:
        tag = tag.strip()
        if tag:
            # Remove trailing punctuation
            tag = re.sub(r'[，。！？；：、,.;:!?]+$', '', tag)
            cleaned_tags.append(tag)

    return cleaned_tags


def normalize_text(text: str) -> str:
    """Normalize text by standardizing whitespace and case.

    Args:
        text: Input text.

    Returns:
        Normalized text.
    """
    if not text:
        return ''

    # Convert to string
    text = str(text)

    # Normalize Unicode
    text = unicodedata.normalize('NFKC', text)

    # Remove zero-width spaces and other invisible characters
    text = re.sub(r'[\u200b\u200c\u200d\uFEFF]', '', text)

    # Standardize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Lowercase (optional for Chinese, but useful for English)
    # text = text.lower()

    return text


def split_hashtag_text(text: str) -> Tuple[str, List[str]]:
    """Split text into content and hashtags.

    Args:
        text: Input text.

    Returns:
        Tuple of (content_without_hashtags, list_of_hashtags).
    """
    hashtags = extract_hashtags(text)

    # Remove hashtags from text
    content = re.sub(r'#[^#\s]+', '', text)
    content = re.sub(r'\s+', ' ', content).strip()

    return content, hashtags


def calculate_text_stats(text: str) -> dict:
    """Calculate text statistics.

    Args:
        text: Input text.

    Returns:
        Dictionary with text statistics.
    """
    if not text:
        return {
            'length': 0,
            'char_count': 0,
            'chinese_char_count': 0,
            'digit_count': 0,
            'hashtag_count': 0,
            'word_count': 0
        }

    # Basic length
    length = len(text)

    # Character type counts
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    digits = sum(1 for c in text if c.isdigit())
    letters = sum(1 for c in text if c.isalpha() and not ('\u4e00' <= c <= '\u9fff'))

    # Hashtag count
    hashtags = extract_hashtags(text)
    hashtag_count = len(hashtags)

    # Word count (approximate for Chinese)
    # Count Chinese characters + words separated by spaces/punctuation
    chinese_word_count = chinese_chars
    non_chinese_parts = re.findall(r'[^\u4e00-\u9fff]+', text)
    other_word_count = sum(len(re.findall(r'\b\w+\b', part)) for part in non_chinese_parts)
    word_count = chinese_word_count + other_word_count

    return {
        'length': length,
        'char_count': length,
        'chinese_char_count': chinese_chars,
        'digit_count': digits,
        'letter_count': letters,
        'hashtag_count': hashtag_count,
        'word_count': word_count
    }


def contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters.

    Args:
        text: Input text.

    Returns:
        True if text contains Chinese characters.
    """
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def contains_english(text: str) -> bool:
    """Check if text contains English letters.

    Args:
        text: Input text.

    Returns:
        True if text contains English letters.
    """
    return bool(re.search(r'[a-zA-Z]', text))


def get_text_language(text: str) -> str:
    """Detect text language (simplified).

    Args:
        text: Input text.

    Returns:
        Language code: 'zh' (Chinese), 'en' (English), 'mixed', or 'unknown'.
    """
    has_chinese = contains_chinese(text)
    has_english = contains_english(text)

    if has_chinese and not has_english:
        return 'zh'
    elif has_english and not has_chinese:
        return 'en'
    elif has_chinese and has_english:
        return 'mixed'
    else:
        return 'unknown'


# Convenience functions for common cleaning tasks
def clean_douyin_text(text: Optional[str]) -> str:
    """Clean Douyin video text with default settings.

    Args:
        text: Input text.

    Returns:
        Cleaned text.
    """
    return clean_text(
        text,
        remove_urls=True,
        remove_emojis=True,
        remove_special_chars=False  # Keep Chinese punctuation
    )


def extract_douyin_hashtags(text: str) -> List[str]:
    """Extract hashtags from Douyin text.

    Args:
        text: Input text.

    Returns:
        List of hashtags.
    """
    return extract_hashtags(text)