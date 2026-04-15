#!/usr/bin/env python
"""
Test WebVideoMeta creation with browser-extracted fields.
"""
import sys
sys.path.insert(0, '.')

from datetime import datetime
from src.schemas.tables import WebVideoMeta

def test_with_browser_fields():
    """Test creating WebVideoMeta with typical browser-extracted fields."""
    # Simulate browser extracted fields (as they might come from browser_client)
    browser_extracted = {
        'video_id': 7624485550934914906,  # Might be int
        'author_id': '123456789',
        'author_name': '测试用户',
        'desc_text': '这是一个测试视频描述 #美食 #旅行',
        'publish_time_raw': 1672531200,  # Unix timestamp as int
        'like_count_raw': 12000,  # int
        'comment_count_raw': 450,  # int
        'share_count_raw': 120,  # int
        'hashtag_list': ['美食', '旅行'],  # list of strings
        'cover_url': 'https://example.com/cover.jpg'
    }

    # Additional required fields from parser
    parsed_data = {
        'page_url': 'https://www.douyin.com/video/7624485550934914906',
        'source_entry': 'manual_url',
        'crawl_time': datetime.now(),
        'parse_status': 'success_with_browser_data'
    }

    # Merge browser fields (simulating scheduler logic)
    parsed_data.update(browser_extracted)

    print("Parsed data fields:")
    for k, v in parsed_data.items():
        print(f"  {k}: {v!r} (type: {type(v).__name__})")

    # Try to create WebVideoMeta
    try:
        meta = WebVideoMeta(**parsed_data)
        print("\nWebVideoMeta created successfully!")
        print(f"Fields: {meta.dict().keys()}")
        # Check specific fields
        print(f"video_id: {meta.video_id} (type: {type(meta.video_id).__name__})")
        print(f"hashtag_list: {meta.hashtag_list} (type: {type(meta.hashtag_list).__name__})")
        print(f"like_count_raw: {meta.like_count_raw} (type: {type(meta.like_count_raw).__name__})")
    except Exception as e:
        print(f"\nFailed to create WebVideoMeta: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def test_with_converted_fields():
    """Test with fields after type conversion (as per scheduler logic)."""
    # Simulate browser fields after type conversion
    browser_extracted = {
        'video_id': '7624485550934914906',  # string
        'author_id': '123456789',
        'author_name': '测试用户',
        'desc_text': '这是一个测试视频描述 #美食 #旅行',
        'publish_time_raw': '1672531200',  # string
        'like_count_raw': '12000',  # string
        'comment_count_raw': '450',  # string
        'share_count_raw': '120',  # string
        'hashtag_list': ['美食', '旅行'],  # list of strings
        'cover_url': 'https://example.com/cover.jpg'
    }

    parsed_data = {
        'page_url': 'https://www.douyin.com/video/7624485550934914906',
        'source_entry': 'manual_url',
        'crawl_time': datetime.now(),
        'parse_status': 'success_with_browser_data'
    }
    parsed_data.update(browser_extracted)

    print("\n\nTest with converted fields:")
    for k, v in parsed_data.items():
        print(f"  {k}: {v!r} (type: {type(v).__name__})")

    try:
        meta = WebVideoMeta(**parsed_data)
        print("\nWebVideoMeta created successfully with converted fields!")
        return True
    except Exception as e:
        print(f"\nFailed with converted fields: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("Testing WebVideoMeta creation...")
    success1 = test_with_browser_fields()
    success2 = test_with_converted_fields()

    if success1 and success2:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed")
        sys.exit(1)