#!/usr/bin/env python
"""
Test WebVideoMeta creation with real browser-extracted fields from summary.
"""
import sys
sys.path.insert(0, '.')

from datetime import datetime
import json
from src.schemas.tables import WebVideoMeta

def test_real_fields():
    # Fields extracted from browser analysis summary
    # Values from analysis_summary_994_20260415_205130.txt
    browser_extracted = {
        'video_id': 16306146,  # int
        'author_id': '0e47c041-8858-c8d8-353d-00b3fee6de79',
        'author_name': '刘春歧',
        'desc_text': '- 抖音电商直播间带货榜是根据真实数据进行排名的直播间排行榜，希望帮助消费者通过榜单找到正在直播的好商品，让选择更简单。直播间排名越靠前曝光机会越大。',
        'publish_time_raw': 1776256936142,  # int, milliseconds
        'like_count_raw': 43364,  # int
        'comment_count_raw': 316,  # int
        'share_count_raw': 5571,  # int
        'hashtag_list': [  # list of dicts from text_extra
            {'caption_end': 28, 'caption_start': 25, 'end': 28, 'hashtag_id': '1653004172633091', 'hashtag_name': '抖音电商', 'is_commerce': False, 'start': 25, 'type': 1},
            # additional items may exist
        ],
        'cover_url': [  # list of dicts from author.cover_url
            {'height': 720, 'uri': 'c8510002be9a3a61aad2', 'url_list': ['https://p3-pc-sign.douyinpic.com/obj/c8510002be9a3a61aad2']}
        ]
    }

    # Simulate scheduler type conversion logic
    parsed_data = {
        'page_url': 'https://www.douyin.com/video/7624485550934914906',
        'source_entry': 'manual_url',
        'crawl_time': datetime.now(),
        'parse_status': 'success_with_browser_data'
    }

    # Apply type conversions (simulating scheduler logic)
    for field, value in browser_extracted.items():
        if field in ['video_id', 'author_id', 'author_name', 'desc_text']:
            if not isinstance(value, str):
                value = str(value)
                print(f"Converted {field} to string: {value}")
        elif field == 'cover_url':
            if isinstance(value, list) and value and isinstance(value[0], dict):
                first = value[0]
                if 'url_list' in first and isinstance(first['url_list'], list) and first['url_list']:
                    value = first['url_list'][0]
                elif 'url' in first:
                    value = first['url']
                elif 'cover_url' in first:
                    value = first['cover_url']
                elif 'cover' in first:
                    value = first['cover']
                else:
                    value = str(value)
            elif isinstance(value, dict):
                if 'url' in value:
                    value = value['url']
                elif 'cover_url' in value:
                    value = value['cover_url']
                elif 'cover' in value:
                    value = value['cover']
                else:
                    value = str(value)
            if not isinstance(value, str):
                value = str(value)
            print(f"Processed cover_url: {value}")
        elif field in ['publish_time_raw', 'like_count_raw', 'comment_count_raw', 'share_count_raw']:
            if not isinstance(value, str):
                value = str(value)
            print(f"Converted {field} to string: {value}")
        elif field == 'hashtag_list':
            if isinstance(value, list):
                processed = []
                for item in value:
                    if isinstance(item, str):
                        processed.append(item)
                    elif isinstance(item, dict):
                        if 'hashtag_name' in item:
                            processed.append(str(item['hashtag_name']))
                        elif 'name' in item:
                            processed.append(str(item['name']))
                        else:
                            print(f"Unhandled dict item in hashtag_list: {item}")
                    else:
                        processed.append(str(item))
                value = processed
                print(f"Processed hashtag_list to list of strings, count={len(value)}")
            else:
                print(f"Unhandled type for hashtag_list: {type(value)}")
                value = []
        parsed_data[field] = value

    print("\nParsed data after conversion:")
    for k, v in parsed_data.items():
        if k == 'hashtag_list' and isinstance(v, list):
            print(f"  {k}: {v} (type: {type(v).__name__}, len={len(v)})")
        else:
            print(f"  {k}: {v!r} (type: {type(v).__name__})")

    # Try to create WebVideoMeta
    try:
        meta = WebVideoMeta(**parsed_data)
        print("\nWebVideoMeta created successfully!")
        print(f"video_id: {meta.video_id}")
        print(f"hashtag_list: {meta.hashtag_list}")
        print(f"cover_url: {meta.cover_url}")
        print(f"like_count_raw: {meta.like_count_raw}")
        print(f"publish_time_raw: {meta.publish_time_raw}")
        return True
    except Exception as e:
        print(f"\nFailed to create WebVideoMeta: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_real_fields()
    sys.exit(0 if success else 1)