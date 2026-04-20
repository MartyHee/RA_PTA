"""
HTML parser for Douyin video pages.

Extracts structured data from HTML responses using BeautifulSoup.
Supports both actual parsing and mock parsing for development.
"""
import json
import re
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
import urllib.parse

from bs4 import BeautifulSoup

from ..schemas.tables import WebVideoMeta, RawWebVideoData
from ..utils.config_loader import load_config, get_config
from ..utils.logger import get_logger
from ..utils.text_utils import extract_hashtags, normalize_text

logger = get_logger(__name__)


class DouyinParser:
    """Parser for Douyin video page HTML."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize parser.

        Args:
            config_path: Path to config file. If None, loads default.
        """
        self.config = load_config(config_path) if config_path else load_config()
        self.field_config = get_config('fields.web_video_meta.fields', {})

    def parse_html(self, html: str, url: str, source_entry: str,
                   crawl_time: datetime, page_type: Optional[str] = None) -> Dict[str, Any]:
        """Parse HTML and extract video metadata.

        Args:
            html: HTML content.
            url: Page URL.
            source_entry: Source entry type.
            crawl_time: Crawl timestamp.
            page_type: Page type (video_detail, jingxuan_modal, etc.)

        Returns:
            Dictionary with extracted fields.
        """
        try:
            soup = BeautifulSoup(html, 'lxml')
        except Exception:
            soup = BeautifulSoup(html, 'html.parser')

        result = {
            'page_url': url,
            'source_entry': source_entry,
            'crawl_time': crawl_time,
            'parse_status': 'success',
            'parse_error_msg': None
        }

        try:
            # 1. Extract and analyze all data blocks (primary method)
            # Generate debug ID from URL and timestamp
            from datetime import datetime
            import hashlib
            debug_id = hashlib.md5(f"{url}_{crawl_time}".encode()).hexdigest()[:8]

            # Skip deep block analysis for very large HTML to improve performance
            # Browser mode already extracts fields from runtime data
            if len(html) > 500000:
                logger.info(f"HTML too large ({len(html)} chars), skipping deep block analysis")
                blocks_info = {'blocks_found': [], 'decoded_blocks': {}, 'video_object_paths': [], 'field_mappings': {}}
            else:
                blocks_info = self._extract_all_data_blocks(soup, debug_id)

            # Merge field mappings from blocks analysis
            for field, mapping in blocks_info.get('field_mappings', {}).items():
                if mapping.get('value') is not None and field not in result:
                    result[field] = mapping['value']

            # Log block analysis results
            logger.debug(f"Data block analysis for {url}: {len(blocks_info.get('blocks_found', []))} blocks found, "
                        f"{len(blocks_info.get('video_object_paths', []))} video objects, "
                        f"{len(blocks_info.get('field_mappings', {}))} fields extracted")

            # 2. Extract from JavaScript state (fallback)
            script_data = self._extract_script_data(soup)
            if script_data:
                result.update(self._parse_script_data(script_data))

            # 3. Extract from HTML elements (fallback)
            html_data = self._extract_html_data(soup)
            result.update(html_data)

            # 4. Extract video ID from URL if not found
            if not result.get('video_id'):
                result['video_id'] = self._extract_video_id_from_url(url)

            # 5. Extract hashtags from description
            if result.get('desc_text'):
                hashtags = extract_hashtags(result['desc_text'])
                result['hashtag_list'] = hashtags
                result['hashtag_count'] = len(hashtags)

            # 6. Normalize counts
            result.update(self._normalize_counts(result))

            logger.debug(f"Successfully parsed {url}")
            return result

        except Exception as e:
            logger.error(f"Failed to parse {url}: {e}")
            result['parse_status'] = 'fail'
            result['parse_error_msg'] = str(e)
            return result

    def _extract_script_data(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Extract data from JavaScript state.

        Args:
            soup: BeautifulSoup object.

        Returns:
            Extracted data dict or None.
        """
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string:
                content = script.string
                # Look for common patterns
                patterns = [
                    r'window\.__INITIAL_STATE__\s*=\s*({.*?});',
                    r'window\.SSR_RENDER_DATA\s*=\s*({.*?});',
                    r'\"video\"\s*:\s*({.*?})',
                    r'\"itemInfo\"\s*:\s*({.*?})',
                    r'\"RENDER_DATA\"\s*:\s*\"([^\"]+)\"',
                ]
                for pattern in patterns:
                    match = re.search(pattern, content, re.DOTALL)
                    if match:
                        try:
                            data_str = match.group(1)
                            # For RENDER_DATA, it might be URL-encoded
                            if pattern == r'\"RENDER_DATA\"\s*:\s*\"([^\"]+)\"':
                                data_str = urllib.parse.unquote(data_str)
                            # Clean up JSON string
                            data_str = re.sub(r',\s*}', '}', data_str)
                            data_str = re.sub(r',\s*]', ']', data_str)
                            return json.loads(data_str)
                        except json.JSONDecodeError:
                            # Try to fix common issues
                            data_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', data_str)
                            try:
                                return json.loads(data_str)
                            except:
                                continue

            # Check for RENDER_DATA script tag with type="application/json"
            if script.get('id') == 'RENDER_DATA' or script.get('type') == 'application/json':
                if script.string:
                    try:
                        # Try to decode URL-encoded JSON
                        decoded = urllib.parse.unquote(script.string)
                        return json.loads(decoded)
                    except:
                        # Try direct JSON parse
                        try:
                            return json.loads(script.string)
                        except:
                            pass

        return None

    def _extract_all_data_blocks(self, soup: BeautifulSoup, crawl_id: str = None) -> Dict[str, Any]:
        """Extract and analyze all candidate data blocks from HTML.

        Args:
            soup: BeautifulSoup object.
            crawl_id: Optional crawl ID for debug file naming.

        Returns:
            Dictionary with extracted blocks and analysis results.
        """
        import urllib.parse
        from pathlib import Path
        from datetime import datetime

        blocks_info = {
            'blocks_found': [],
            'decoded_blocks': {},
            'video_object_paths': [],
            'field_mappings': {},
            'debug_files': []
        }

        # 1. Extract all script tags
        scripts = soup.find_all('script')

        for i, script in enumerate(scripts):
            block_info = {
                'index': i,
                'type': 'unknown',
                'has_content': bool(script.string),
                'content_length': len(script.string) if script.string else 0,
                'attributes': dict(script.attrs),
                'decoded': False,
                'decoded_keys': [],
                'contains_video_keywords': False,
                'video_object_found': False,
                'video_object_path': None
            }

            # Determine block type
            if script.get('id') == 'RENDER_DATA':
                block_info['type'] = 'RENDER_DATA_id'
            elif script.get('type') == 'application/json':
                block_info['type'] = 'application/json'
            elif script.string:
                content = script.string
                # Check for patterns
                if 'SSR_RENDER_DATA' in content:
                    block_info['type'] = 'SSR_RENDER_DATA'
                elif 'RENDER_DATA' in content:
                    block_info['type'] = 'RENDER_DATA_ref'
                elif '%7B%22' in content[:100]:  # URL-encoded JSON start
                    block_info['type'] = 'url_encoded_json'
                elif '{"' in content[:100] or "'{" in content[:100]:
                    block_info['type'] = 'json_literal'
                elif 'video' in content.lower() or 'aweme' in content.lower() or 'itemInfo' in content.lower():
                    block_info['type'] = 'video_related'

            blocks_info['blocks_found'].append(block_info)

            # Try to decode if there's content
            if script.string and block_info['type'] != 'unknown':
                decoded_data = self._decode_and_analyze_block(
                    block_info['type'], script.string, block_info
                )
                if decoded_data:
                    block_key = f"{block_info['type']}_{i}"
                    blocks_info['decoded_blocks'][block_key] = decoded_data

        # 2. Also look for window.__INITIAL_STATE__ and other patterns in all script content
        for i, script in enumerate(scripts):
            if not script.string:
                continue

            content = script.string
            patterns = [
                (r'window\.__INITIAL_STATE__\s*=\s*({.*?});', 'window.__INITIAL_STATE__'),
                (r'window\.SSR_RENDER_DATA\s*=\s*({.*?});', 'window.SSR_RENDER_DATA'),
                (r'\"RENDER_DATA\"\s*:\s*\"([^\"]+)\"', 'RENDER_DATA_string'),
                (r'\"video\"\s*:\s*({.*?})', 'video_object'),
                (r'\"itemInfo\"\s*:\s*({.*?})', 'itemInfo_object'),
                (r'\"aweme\"\s*:\s*({.*?})', 'aweme_object'),
            ]

            for pattern, pattern_type in patterns:
                import re
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    block_info = {
                        'index': f'pattern_{i}',
                        'type': pattern_type,
                        'has_content': True,
                        'content_length': len(match.group(1)),
                        'attributes': {'pattern': pattern},
                        'decoded': False,
                        'decoded_keys': [],
                        'contains_video_keywords': True,
                        'video_object_found': False,
                        'video_object_path': None
                    }

                    blocks_info['blocks_found'].append(block_info)

                    # Try to decode
                    decoded_data = self._decode_and_analyze_block(
                        pattern_type, match.group(1), block_info
                    )
                    if decoded_data:
                        block_key = f"{pattern_type}_{i}"
                        blocks_info['decoded_blocks'][block_key] = decoded_data

        # 3. Search for video objects in decoded data
        for block_key, decoded_data in blocks_info['decoded_blocks'].items():
            if isinstance(decoded_data, dict):
                video_obj, path = self._find_video_object_with_path(decoded_data)
                if video_obj:
                    blocks_info['video_object_paths'].append({
                        'block': block_key,
                        'path': path,
                        'video_keys': list(video_obj.keys()) if isinstance(video_obj, dict) else []
                    })

                    # Extract fields from video object
                    field_mappings = self._extract_fields_from_video_object(video_obj, path)
                    for field, mapping in field_mappings.items():
                        if field not in blocks_info['field_mappings']:
                            blocks_info['field_mappings'][field] = mapping

        # 4. Output debug files if crawl_id provided
        if crawl_id:
            debug_files = self._output_debug_info(blocks_info, crawl_id)
            blocks_info['debug_files'] = debug_files

        return blocks_info

    def _decode_and_analyze_block(self, block_type: str, raw_content: str, block_info: Dict) -> Optional[Any]:
        """Decode and analyze a single data block.

        Args:
            block_type: Type of block.
            raw_content: Raw content string.
            block_info: Block info dict to update.

        Returns:
            Decoded data or None.
        """
        import json
        import urllib.parse
        import re

        try:
            decoded = None
            raw_content = raw_content.strip()

            # Handle URL-encoded JSON (common in Douyin RENDER_DATA)
            if block_type in ['RENDER_DATA_id', 'RENDER_DATA_string', 'url_encoded_json', 'RENDER_DATA_ref']:
                # For Douyin, RENDER_DATA is often URL-encoded JSON
                # Try multiple decoding strategies

                # Strategy 1: Direct URL decode then JSON parse
                try:
                    decoded_str = urllib.parse.unquote(raw_content)
                    decoded = json.loads(decoded_str)
                    logger.debug(f"Successfully decoded {block_type} via strategy 1")
                    block_info['decoding_strategy'] = 'url_decode_then_json'
                except:
                    # Strategy 2: Try to extract JSON from possible wrapper
                    # Look for common patterns like "RENDER_DATA":"...", window.RENDER_DATA = "...", etc.
                    patterns = [
                        r'\"RENDER_DATA\"\s*:\s*\"([^\"]+)\"',
                        r'RENDER_DATA\s*=\s*\"([^\"]+)\"',
                        r'\"data\"\s*:\s*\"([^\"]+)\"',
                        r'%7B%22(.*?)%22%7D',  # URL-encoded JSON fragment
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, raw_content, re.DOTALL)
                        if match:
                            try:
                                encoded_str = match.group(1)
                                decoded_str = urllib.parse.unquote(encoded_str)
                                decoded = json.loads(decoded_str)
                                logger.debug(f"Successfully decoded {block_type} via pattern {pattern}")
                                block_info['decoding_strategy'] = f'pattern_{pattern}'
                                break
                            except:
                                continue

                    # Strategy 3: Try to fix JSON string and parse
                    if decoded is None:
                        try:
                            # Clean the raw content
                            content = raw_content
                            # Remove any JavaScript assignment
                            content = re.sub(r'^[^{[]*', '', content)
                            content = re.sub(r'[^}\]]*$', '', content)
                            # Fix unquoted keys
                            content = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', content)
                            # Try URL decode again in case it's partially encoded
                            try:
                                content = urllib.parse.unquote(content)
                            except:
                                pass
                            decoded = json.loads(content)
                            logger.debug(f"Successfully decoded {block_type} via strategy 3 (cleaning)")
                            block_info['decoding_strategy'] = 'cleaning_and_json'
                        except:
                            pass

            # Handle JSON literals
            elif block_type in ['application/json', 'json_literal']:
                # Clean up JSON string
                content = raw_content
                # Remove surrounding quotes if present
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]

                try:
                    decoded = json.loads(content)
                    block_info['decoding_strategy'] = 'direct_json'
                except json.JSONDecodeError:
                    # Try to fix common JSON issues
                    content = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', content)
                    try:
                        decoded = json.loads(content)
                        block_info['decoding_strategy'] = 'fixed_json'
                    except:
                        pass

            # Handle SSR_RENDER_DATA and window variable assignments
            elif block_type in ['SSR_RENDER_DATA', 'window.__INITIAL_STATE__', 'window.SSR_RENDER_DATA',
                               'video_object', 'itemInfo_object', 'aweme_object']:
                # Extract JSON from JavaScript assignment
                patterns = [
                    r'=\s*({.*?})\s*;',
                    r'=\s*(\[.*?\])\s*;',
                    r'=\s*({.*?})$',
                    r'=\s*(\[.*?\])$',
                ]
                for pattern in patterns:
                    match = re.search(pattern, raw_content, re.DOTALL)
                    if match:
                        try:
                            json_str = match.group(1).strip()
                            # Fix unquoted keys
                            json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', json_str)
                            decoded = json.loads(json_str)
                            logger.debug(f"Successfully decoded {block_type} via pattern {pattern}")
                            block_info['decoding_strategy'] = f'js_assignment_{pattern}'
                            break
                        except:
                            continue

            # Handle video_related blocks (might contain JSON snippets)
            elif block_type == 'video_related':
                # Try to find JSON structures in the content
                json_patterns = [
                    r'(\{"video".*?\})',
                    r'(\{"aweme".*?\})',
                    r'(\{"itemInfo".*?\})',
                    r'(\{.*?"id".*?"desc".*?\})',
                ]
                for pattern in json_patterns:
                    matches = re.findall(pattern, raw_content, re.DOTALL)
                    for json_str in matches:
                        try:
                            # Try to parse as JSON
                            decoded = json.loads(json_str)
                            logger.debug(f"Found video JSON in {block_type} via pattern {pattern}")
                            block_info['decoding_strategy'] = f'video_json_{pattern}'
                            break
                        except:
                            # Try fixing keys
                            fixed = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', json_str)
                            try:
                                decoded = json.loads(fixed)
                                logger.debug(f"Found video JSON (fixed) in {block_type} via pattern {pattern}")
                                block_info['decoding_strategy'] = f'video_json_fixed_{pattern}'
                                break
                            except:
                                continue
                    if decoded:
                        break

            # Update block info
            if decoded:
                block_info['decoded'] = True
                if isinstance(decoded, dict):
                    block_info['decoded_keys'] = list(decoded.keys())
                    # Check for video keywords
                    video_keywords = ['video', 'aweme', 'item', 'author', 'desc', 'statistics', 'stats', 'digg', 'comment', 'share', 'createTime', 'nickname', 'cover', 'hashtag']
                    for key in block_info['decoded_keys']:
                        if any(kw in key.lower() for kw in video_keywords):
                            block_info['contains_video_keywords'] = True
                            break

                return decoded

        except Exception as e:
            logger.debug(f"Failed to decode block type {block_type}: {e}")

        return None

    def _find_video_object_with_path(self, data: Any, path_prefix: str = "", depth: int = 0, max_depth: int = 10) -> tuple[Optional[Dict], str]:
        """Recursively find video object and return full path.

        Args:
            data: Data to search.
            path_prefix: Current path prefix.
            depth: Current recursion depth.
            max_depth: Maximum recursion depth.

        Returns:
            Tuple of (video_object, path_string)
        """
        # Prevent infinite recursion
        if depth >= max_depth:
            return None, ""

        # First, try known Douyin video data paths
        if isinstance(data, dict):
            known_paths = [
                ('tccConfig', 'download_info', 'video'),
                ('tccConfig', 'video'),
                ('video',),
                ('aweme',),
                ('itemInfo',),
                ('itemStruct',),
                ('app', 'tccConfig', 'download_info', 'video'),
                ('app', 'video'),
                ('app', 'aweme'),
                ('app', 'itemInfo'),
            ]

            for path_parts in known_paths:
                current = data
                current_path = []
                found = True
                for part in path_parts:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                        current_path.append(part)
                    else:
                        found = False
                        break
                if found and isinstance(current, dict):
                    # Check if this looks like a video object
                    if any(key in current for key in ['id', 'video_id', 'aweme_id', 'item_id']):
                        full_path = '.'.join(current_path)
                        if path_prefix:
                            full_path = f"{path_prefix}.{full_path}"
                        return current, full_path

        # Then use original recursive search
        if isinstance(data, dict):
            # Check if this dict is a video object
            video_keywords = ['video', 'aweme', 'itemInfo', 'itemStruct']
            for key, value in data.items():
                if isinstance(value, dict) and any(kw in key.lower() for kw in video_keywords):
                    return value, f"{path_prefix}.{key}" if path_prefix else key

            # Check for direct video fields
            if 'id' in data and ('desc' in data or 'title' in data):
                return data, path_prefix if path_prefix else "root"

            # Recursively search with depth limit
            for key, value in data.items():
                result, path = self._find_video_object_with_path(
                    value,
                    f"{path_prefix}.{key}" if path_prefix else key,
                    depth + 1,
                    max_depth
                )
                if result:
                    return result, path

        elif isinstance(data, list):
            # Limit list traversal to first few items
            for i, item in enumerate(data[:5]):  # Only check first 5 items
                result, path = self._find_video_object_with_path(
                    item,
                    f"{path_prefix}[{i}]" if path_prefix else f"[{i}]",
                    depth + 1,
                    max_depth
                )
                if result:
                    return result, path

        return None, ""

    def _extract_fields_from_video_object(self, video_obj: Dict, source_path: str) -> Dict[str, Dict]:
        """Extract target fields from video object and record source paths.

        Args:
            video_obj: Video object dictionary.
            source_path: Path to video object.

        Returns:
            Dictionary mapping field names to extraction info.
        """
        field_mappings = {}

        # Define field extraction rules
        field_rules = {
            'video_id': ['id', 'video_id', 'item_id', 'aweme_id'],
            'author_id': ['author.uid', 'author.id', 'author_user_id', 'uid'],
            'author_name': ['author.nickname', 'author.name', 'nickname', 'author_name'],
            'desc_text': ['desc', 'title', 'description', 'content'],
            'publish_time_raw': ['create_time', 'createTime', 'publish_time', 'timestamp'],
            'like_count_raw': ['stats.digg_count', 'statistics.digg_count', 'digg_count', 'like_count'],
            'comment_count_raw': ['stats.comment_count', 'statistics.comment_count', 'comment_count'],
            'share_count_raw': ['stats.share_count', 'statistics.share_count', 'share_count'],
            'hashtag_list': ['hashtags', 'text_extra', 'topics'],
            'cover_url': ['video.cover', 'cover.url', 'cover_url', 'video_cover']
        }

        def get_value(obj, path):
            """Get value from nested dict using dot notation path."""
            if not path:
                return None

            parts = path.split('.')
            current = obj

            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None

            return current

        # Try each field
        for field, possible_paths in field_rules.items():
            value = None
            found_path = None

            for path in possible_paths:
                # Try direct path in video object
                val = get_value(video_obj, path)
                if val is not None:
                    value = val
                    found_path = path
                    break

            # If not found, search recursively
            if value is None:
                value, found_path = self._search_field_recursively(video_obj, field)

            if value is not None:
                # Convert to appropriate format
                if field == 'hashtag_list' and isinstance(value, list):
                    # Extract hashtag texts
                    hashtags = []
                    for item in value:
                        if isinstance(item, dict):
                            if 'hashtag_name' in item:
                                hashtags.append(item['hashtag_name'])
                            elif 'name' in item:
                                hashtags.append(item['name'])
                        elif isinstance(item, str):
                            hashtags.append(item)
                    value = hashtags

                # Record mapping
                field_mappings[field] = {
                    'value': value,
                    'source_path': f"{source_path}.{found_path}" if found_path else source_path,
                    'raw_value': str(value)[:100] + ('...' if len(str(value)) > 100 else '')
                }

        return field_mappings

    def _search_field_recursively(self, data: Any, field_name: str) -> tuple[Optional[Any], Optional[str]]:
        """Recursively search for field value.

        Args:
            data: Data to search.
            field_name: Field name to search for.

        Returns:
            Tuple of (value, path)
        """
        if isinstance(data, dict):
            # Check for field in this dict
            for key, value in data.items():
                if field_name in key.lower() or key.lower() in field_name:
                    return value, key

            # Search recursively
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    result, path = self._search_field_recursively(value, field_name)
                    if result is not None:
                        return result, f"{key}.{path}" if path else key

        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    result, path = self._search_field_recursively(item, field_name)
                    if result is not None:
                        return result, f"[{i}].{path}" if path else f"[{i}]"

        return None, None

    def _output_debug_info(self, blocks_info: Dict, crawl_id: str) -> List[str]:
        """Output debug information to files.

        Args:
            blocks_info: Blocks information dictionary.
            crawl_id: Crawl ID for file naming.

        Returns:
            List of created file paths.
        """
        import json
        from pathlib import Path
        from datetime import datetime

        debug_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create debug directory
        debug_dir = Path("data/raw/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save decoded blocks
        decoded_file = debug_dir / f"decoded_blocks_{crawl_id}_{timestamp}.json"
        try:
            # Convert to serializable format
            serializable_blocks = {}
            for key, value in blocks_info.get('decoded_blocks', {}).items():
                try:
                    # Try to serialize, limit depth
                    serializable_blocks[key] = json.loads(json.dumps(value, default=str, ensure_ascii=False))
                except:
                    serializable_blocks[key] = str(type(value))

            with open(decoded_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_blocks, f, ensure_ascii=False, indent=2)
            debug_files.append(str(decoded_file))
        except Exception as e:
            logger.error(f"Failed to save decoded blocks: {e}")

        # 2. Save block summary
        summary_file = debug_dir / f"block_summary_{crawl_id}_{timestamp}.txt"
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Block Analysis Summary - {timestamp}\n")
                f.write(f"Crawl ID: {crawl_id}\n")
                f.write("=" * 80 + "\n\n")

                # Blocks found
                f.write("BLOCKS FOUND:\n")
                f.write("-" * 40 + "\n")
                for block in blocks_info.get('blocks_found', []):
                    f.write(f"Index: {block.get('index')}\n")
                    f.write(f"Type: {block.get('type')}\n")
                    f.write(f"Content length: {block.get('content_length')}\n")
                    f.write(f"Decoded: {block.get('decoded')}\n")
                    if block.get('decoded_keys'):
                        f.write(f"Keys: {', '.join(block.get('decoded_keys', []))}\n")
                    f.write(f"Contains video keywords: {block.get('contains_video_keywords')}\n")
                    f.write("\n")

                # Video object paths
                f.write("\nVIDEO OBJECT PATHS:\n")
                f.write("-" * 40 + "\n")
                for path_info in blocks_info.get('video_object_paths', []):
                    f.write(f"Block: {path_info.get('block')}\n")
                    f.write(f"Path: {path_info.get('path')}\n")
                    f.write(f"Keys: {', '.join(path_info.get('video_keys', []))}\n")
                    f.write("\n")

                # Field mappings
                f.write("\nFIELD MAPPINGS:\n")
                f.write("-" * 40 + "\n")
                for field, mapping in blocks_info.get('field_mappings', {}).items():
                    f.write(f"{field}:\n")
                    f.write(f"  Source: {mapping.get('source_path')}\n")
                    f.write(f"  Value: {mapping.get('raw_value')}\n")
                    f.write("\n")

                # Statistics
                f.write("\nSTATISTICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total blocks found: {len(blocks_info.get('blocks_found', []))}\n")
                f.write(f"Blocks decoded: {sum(1 for b in blocks_info.get('blocks_found', []) if b.get('decoded'))}\n")
                f.write(f"Video objects found: {len(blocks_info.get('video_object_paths', []))}\n")
                f.write(f"Fields extracted: {len(blocks_info.get('field_mappings', {}))}\n")

            debug_files.append(str(summary_file))
        except Exception as e:
            logger.error(f"Failed to save block summary: {e}")

        return debug_files

    def _find_video_object(self, data: Any) -> Optional[Dict]:
        """Recursively find video object in parsed data.

        Args:
            data: Parsed JSON data.

        Returns:
            Video object dict or None.
        """
        if isinstance(data, dict):
            # Check if this dict contains video fields
            if 'video' in data and isinstance(data['video'], dict):
                return data['video']
            if 'itemInfo' in data and isinstance(data['itemInfo'], dict):
                return data['itemInfo']
            if 'aweme' in data and isinstance(data['aweme'], dict):
                return data['aweme']
            # Check for direct video fields
            if 'id' in data and 'desc' in data:
                return data
            # Recursively search
            for value in data.values():
                result = self._find_video_object(value)
                if result:
                    return result
        elif isinstance(data, list):
            for item in data:
                result = self._find_video_object(item)
                if result:
                    return result
        return None

    def _parse_script_data(self, data: Dict) -> Dict[str, Any]:
        """Parse data extracted from script.

        Args:
            data: Raw script data.

        Returns:
            Parsed fields.
        """
        result = {}

        # First try to find video object recursively
        video_data = self._find_video_object(data)

        # If not found, try common structures
        if not video_data:
            if 'video' in data:
                video_data = data['video']
            elif 'itemInfo' in data and 'itemStruct' in data['itemInfo']:
                video_data = data['itemInfo']['itemStruct']
            elif 'aweme' in data:
                video_data = data['aweme']

        # If video_data is a string, try to parse as JSON
        if isinstance(video_data, str):
            try:
                video_data = json.loads(video_data)
            except:
                video_data = None

        if video_data and isinstance(video_data, dict):
            # Map fields
            field_mapping = {
                'id': 'video_id',
                'desc': 'desc_text',
                'createTime': 'publish_time_std',
                'author': 'author_data',
                'stats': 'stats_data',
                'music': 'music_data',
                'duration': 'duration_sec',
                'cover': 'cover_url'
            }

            for src_key, dest_key in field_mapping.items():
                if src_key in video_data:
                    result[dest_key] = video_data[src_key]

            # Extract author info
            if 'author_data' in result:
                author = result.pop('author_data')
                if isinstance(author, dict):
                    result['author_id'] = author.get('id') or author.get('uid')
                    result['author_name'] = author.get('nickname')
                    result['author_profile_url'] = author.get('profile_url')

            # Extract stats
            if 'stats_data' in result:
                stats = result.pop('stats_data')
                if isinstance(stats, dict):
                    result['like_count_raw'] = str(stats.get('diggCount', ''))
                    result['comment_count_raw'] = str(stats.get('commentCount', ''))
                    result['share_count_raw'] = str(stats.get('shareCount', ''))
                    result['collect_count'] = stats.get('collectCount')

            # Extract music
            if 'music_data' in result:
                music = result.pop('music_data')
                if isinstance(music, dict):
                    result['music_name'] = music.get('title')

        return result

    def _extract_html_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract data from HTML elements.

        Args:
            soup: BeautifulSoup object.

        Returns:
            Extracted fields.
        """
        result = {}

        # Extract description
        desc_elements = soup.find_all(['div', 'p'], class_=re.compile(r'desc|title|content'))
        if desc_elements:
            result['desc_text'] = normalize_text(desc_elements[0].get_text(strip=True))

        # Extract author info
        author_elements = soup.find_all('a', href=re.compile(r'/user/'))
        if author_elements:
            result['author_name'] = author_elements[0].get_text(strip=True)
            result['author_profile_url'] = author_elements[0].get('href', '')
            # Extract author ID from URL
            if result['author_profile_url']:
                match = re.search(r'/user/([^/?]+)', result['author_profile_url'])
                if match:
                    result['author_id'] = match.group(1)

        # Extract stats
        stats_patterns = {
            'like_count_raw': r'like|digg|赞',
            'comment_count_raw': r'comment|评论',
            'share_count_raw': r'share|分享'
        }

        for field, pattern in stats_patterns.items():
            elements = soup.find_all(['span', 'div'], text=re.compile(pattern))
            if elements:
                # Try to find sibling with count
                for elem in elements:
                    count_elem = elem.find_next(['span', 'div'])
                    if count_elem:
                        result[field] = count_elem.get_text(strip=True)
                        break

        # Extract cover image
        img_elements = soup.find_all('img', src=re.compile(r'\.(jpg|jpeg|png|webp)'))
        if img_elements:
            result['cover_url'] = img_elements[0].get('src', '')

        # Extract publish time
        time_elements = soup.find_all(['span', 'div'], class_=re.compile(r'time|date'))
        if time_elements:
            result['publish_time_raw'] = time_elements[0].get_text(strip=True)

        return result

    def normalize_douyin_url(self, url: str) -> Dict[str, str]:
        """Normalize Douyin URL to canonical video URL when possible.

        Args:
            url: Input URL (could be jingxuan modal, video detail, etc.)

        Returns:
            Dictionary with:
            - original_url: original input URL
            - canonical_video_url: canonical video detail page URL if can be normalized
            - page_type: video_detail, jingxuan_modal, jingxuan_feed, or unknown
        """
        from urllib.parse import urlparse, parse_qs

        result = {
            'original_url': url,
            'canonical_video_url': None,
            'page_type': 'unknown'
        }

        try:
            parsed = urlparse(url)
            path = parsed.path
            query = parse_qs(parsed.query)

            # Determine page type
            if '/video/' in path:
                result['page_type'] = 'video_detail'
                result['canonical_video_url'] = url  # Already canonical
            elif '/jingxuan' in path:
                if 'modal_id' in query:
                    result['page_type'] = 'jingxuan_modal'
                    modal_id = query['modal_id'][0]
                    # Construct canonical video URL
                    result['canonical_video_url'] = f"https://www.douyin.com/video/{modal_id}"
                else:
                    result['page_type'] = 'jingxuan_feed'
            else:
                # Check if it's a short video URL (e.g., from share)
                if path and '/video/' not in path:
                    # Could be short form URL, but we can't normalize without ID
                    result['page_type'] = 'unknown'

        except Exception as e:
            logger.warning(f"Failed to normalize URL {url}: {e}")

        return result

    def _extract_video_id_from_url(self, url: str) -> Optional[str]:
        """Extract video ID from URL.

        Args:
            url: Page URL.

        Returns:
            Video ID or None.
        """
        patterns = [
            r'/video/([^/?]+)',
            r'video/([^/?]+)',
            r'item_id=([^&]+)',
            r'id=([^&]+)',
            r'modal_id=([^&]+)'  # Support for jingxuan pages
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                video_id = match.group(1)
                logger.debug(f"Extracted video_id '{video_id}' from URL using pattern '{pattern}'")
                return video_id
        logger.debug(f"No video_id found in URL: {url}")
        return None

    def _normalize_counts(self, data: Dict) -> Dict[str, Any]:
        """Normalize count fields (e.g., '1.2w' -> 12000).

        Args:
            data: Data with raw count fields.

        Returns:
            Data with normalized count fields.
        """
        result = {}

        count_fields = ['like_count_raw', 'comment_count_raw', 'share_count_raw']
        for raw_field in count_fields:
            if raw_field in data and data[raw_field]:
                normalized = self._normalize_count_string(data[raw_field])
                field_name = raw_field.replace('_raw', '')
                result[field_name] = normalized

        return result

    def _normalize_count_string(self, count_str: str) -> Optional[int]:
        """Normalize count string like '1.2w' or '5k' to integer.

        Args:
            count_str: Raw count string.

        Returns:
            Normalized integer or None.
        """
        if not count_str:
            return None

        count_str = count_str.strip().lower()

        # Remove commas and other non-numeric characters (except decimal point and k/w)
        count_str = re.sub(r'[^\d.kw]', '', count_str)

        try:
            if 'w' in count_str:
                num = float(count_str.replace('w', ''))
                return int(num * 10000)
            elif 'k' in count_str:
                num = float(count_str.replace('k', ''))
                return int(num * 1000)
            elif '.' in count_str:
                return int(float(count_str))
            else:
                return int(count_str)
        except (ValueError, TypeError):
            return None

    def create_web_video_meta(self, parsed_data: Dict) -> Optional[WebVideoMeta]:
        """Create WebVideoMeta object from parsed data.

        Args:
            parsed_data: Parsed data dictionary.

        Returns:
            WebVideoMeta object or None if validation fails.
        """
        try:
            # Convert datetime strings if needed
            if 'publish_time_std' in parsed_data and isinstance(parsed_data['publish_time_std'], (int, float)):
                parsed_data['publish_time_std'] = datetime.fromtimestamp(parsed_data['publish_time_std'])

            # Log detailed field information before creating WebVideoMeta
            logger.info("Field details before creating WebVideoMeta:")
            target_fields = [
                'video_id', 'page_url', 'author_id', 'author_name', 'author_profile_url',
                'desc_text', 'publish_time_raw', 'publish_time_std', 'like_count_raw',
                'comment_count_raw', 'share_count_raw', 'collect_count', 'hashtag_list', 'hashtag_count',
                'cover_url', 'music_name', 'duration_sec', 'source_entry', 'crawl_time'
            ]
            for field in target_fields:
                if field in parsed_data:
                    value = parsed_data[field]
                    logger.info(f"  {field}: value='{value}', type={type(value).__name__}")

            # Ensure required fields
            if not parsed_data.get('video_id'):
                logger.warning("Missing video_id, cannot create WebVideoMeta")
                return None

            # Try to create WebVideoMeta with detailed error logging
            logger.info("Attempting to create WebVideoMeta instance...")
            meta = WebVideoMeta(**parsed_data)
            logger.info(f"WebVideoMeta created successfully with {len(parsed_data)} fields")
            return meta
        except Exception as e:
            logger.error(f"Failed to create WebVideoMeta: {e}")
            return None

    def mock_parse(self, url: str, source_entry: str, page_type: Optional[str] = None) -> WebVideoMeta:
        """Create mock WebVideoMeta for testing.

        Args:
            url: Page URL.
            source_entry: Source entry type.
            page_type: Page type (optional).

        Returns:
            Mock WebVideoMeta object.
        """
        video_id = self._extract_video_id_from_url(url) or "mock_1234567890"
        now = datetime.now()

        return WebVideoMeta(
            video_id=video_id,
            page_url=url,
            author_id="author_mock_001",
            author_name="测试用户",
            author_profile_url=f"https://www.douyin.com/user/author_mock_001",
            desc_text="这是一个测试视频描述 #美食 #旅行",
            publish_time_raw="2023-01-01 12:00:00",
            publish_time_std=datetime(2023, 1, 1, 12, 0, 0),
            like_count_raw="1.2w",
            comment_count_raw="450",
            share_count_raw="120",
            like_count=12000,
            comment_count=450,
            share_count=120,
            collect_count=56,
            hashtag_list=["美食", "旅行"],
            hashtag_count=2,
            cover_url="https://example.com/cover.jpg",
            music_name="测试音乐",
            duration_sec=15,
            source_entry=source_entry,
            crawl_time=now
        )