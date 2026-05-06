"""
Build raw table records from parsed crawl data.

Converts the merged parsed_data dict (from scheduler worker) into structured
records for all raw tables using column ordering from src/schemas/raw_schema.py.

Phase 1 (existing): raw_video_detail, raw_author, raw_music, raw_video_media
Phase 2 (new):      raw_hashtag, raw_video_status_control, raw_crawl_log
Phase 5 (new):      raw_related_video

Usage:
    from src.processing.raw_table_builder import build_raw_tables

    tables = build_raw_tables(parsed_data, run_id="20260506_120000")
    # tables == {
    #     "raw_video_detail": {video_id: ..., page_url: ..., ...},
    #     "raw_author": {author_id: ..., ...},
    #     "raw_music": {video_id: ..., ...},
    #     "raw_video_media": {video_id: ..., ...},
    #     "raw_hashtag": [{video_id: ..., hashtag_name: ..., ...}, ...],
    #     "raw_video_status_control": {video_id: ..., can_comment: ..., ...},
    #     "raw_crawl_log": {crawl_batch_id: ..., target_url: ..., ...},
    #     "raw_video_tag": [{video_id: ..., tag_id: ..., ...}, ...],
    #     "raw_chapter": [{video_id: ..., chapter_index: ..., ...}, ...],
    #     "raw_comment": [{video_id: ..., comment_id: ..., ...}, ...],
    #     "raw_related_video": [{source_video_id: ..., related_video_id: ..., ...}, ...],
    # }
"""

import json
import logging
from datetime import datetime
from typing import Optional, Union

from ..schemas.raw_schema import get_table_columns, build_empty_raw_record

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Field mappings: parsed_data key -> schema field name
# Only non-identity mappings need to be listed; omitted keys use identity.
# ---------------------------------------------------------------------------

_TABLE_FIELD_MAP: dict[str, dict[str, str]] = {
    "raw_video_detail": {
        # identity mapping used for all fields
    },
    "raw_author": {
        # parsed_data stores as author_verification_type; schema calls it verification_type
        "author_verification_type": "verification_type",
    },
    "raw_music": {
        # identity for all music_* fields
    },
    "raw_video_media": {
        # identity for all media fields
    },
    "raw_hashtag": {
        # identity mapping for raw hashtag fields
    },
    "raw_video_status_control": {
        # identity for all status fields
    },
    "raw_crawl_log": {
        # identity for all crawl log fields
    },
}

# URL list fields that must be stored as JSON strings in CSV
_URL_LIST_FIELDS: set[str] = {
    "cover_url_list",
    "origin_cover_url_list",
    "dynamic_cover_url_list",
    "avatar_thumb_url_list",
    "music_cover_url_list",
    "music_play_url_list",
    "comment_user_avatar_url_list",
    "related_cover_url_list",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Default table list: Phase 1 (4 tables) + Phase 2 (3 tables) + Phase 3 (2 tables)
#                 + Phase 4 (1 table) + Phase 5 (1 table)
_ALL_TABLES = [
    "raw_video_detail",
    "raw_author",
    "raw_music",
    "raw_video_media",
    "raw_hashtag",
    "raw_video_status_control",
    "raw_crawl_log",
    "raw_video_tag",
    "raw_chapter",
    "raw_comment",
    "raw_related_video",
]


def build_raw_tables(
    parsed_data: dict,
    run_id: Optional[str] = None,
    crawl_context: Optional[dict] = None,
) -> dict[str, Union[dict, list]]:
    """Build all raw table records from merged parsed_data and crawl context.

    Args:
        parsed_data: The merged data dict from scheduler worker (contains
                     fields from HTML parsing + browser extraction).
        run_id: Optional run identifier for inclusion in crawl_log.
        crawl_context: Optional dict with crawl-level metadata for crawl_log.
                       Keys: run_id, target_url, source_page_type, request_url,
                       response_status, primary_source_key, match_type, confidence,
                       network_response_count, runtime_objects_count.

    Returns:
        dict mapping table_name -> record(s).
        For single-row tables: table_name -> {column: value}
        For multi-row tables (raw_hashtag): table_name -> [{column: value}, ...]
    """
    tables: dict[str, Union[dict, list]] = {}

    # Phase 1: single-record tables
    for table_name in ["raw_video_detail", "raw_author", "raw_music", "raw_video_media"]:
        tables[table_name] = _build_single_record(table_name, parsed_data)

    # Phase 2: raw_hashtag (multi-row)
    tables["raw_hashtag"] = _build_hashtag_records(parsed_data)

    # Phase 2: raw_video_status_control (single-row)
    tables["raw_video_status_control"] = _build_single_record("raw_video_status_control", parsed_data)

    # Phase 2: raw_crawl_log (single-row, needs crawl_context)
    tables["raw_crawl_log"] = _build_crawl_log_record(parsed_data, crawl_context=crawl_context)

    # Phase 3: raw_video_tag (multi-row)
    tables["raw_video_tag"] = _build_video_tag_records(parsed_data)

    # Phase 3: raw_chapter (multi-row)
    tables["raw_chapter"] = _build_chapter_records(parsed_data)

    # Phase 4: raw_comment (multi-row)
    tables["raw_comment"] = _build_comment_records(parsed_data, crawl_context=crawl_context)

    # Phase 5: raw_related_video (multi-row)
    tables["raw_related_video"] = _build_related_video_records(parsed_data, crawl_context=crawl_context)

    # Cross-table corrections
    _apply_cross_table_fixes(tables, parsed_data)

    return tables


# ---------------------------------------------------------------------------
# Internal helpers: single-record builder
# ---------------------------------------------------------------------------


def _build_single_record(table_name: str, parsed_data: dict) -> dict:
    """Build one raw table record from parsed_data using schema column order."""
    record = build_empty_raw_record(table_name)

    field_map = _TABLE_FIELD_MAP.get(table_name, {})
    reverse_map = {v: k for k, v in field_map.items()}

    for schema_col in list(record.keys()):
        if schema_col in reverse_map:
            data_key = reverse_map[schema_col]
        else:
            data_key = schema_col

        raw_val = parsed_data.get(data_key)
        if raw_val is not None:
            record[schema_col] = _convert_field_value(table_name, schema_col, raw_val)

    # Ensure crawl_time is present
    if record.get("crawl_time") is None:
        record["crawl_time"] = _to_datetime_str(parsed_data.get("crawl_time"))

    return record


def _convert_field_value(table_name: str, field_name: str, value) -> object:
    """Convert a parsed_data field value to raw-table format."""
    if field_name in _URL_LIST_FIELDS:
        return _ensure_json_list(value)
    return value


def _ensure_json_list(value) -> Optional[str]:
    """Convert value to a JSON-encoded list string."""
    if value is None:
        return None

    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return stripped
                return json.dumps([parsed], ensure_ascii=False)
            except json.JSONDecodeError:
                return json.dumps([stripped], ensure_ascii=False)
        return json.dumps([stripped], ensure_ascii=False)

    return json.dumps([str(value)], ensure_ascii=False)


def _to_datetime_str(value) -> Optional[str]:
    """Convert various datetime formats to ISO string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return str(value)


# ---------------------------------------------------------------------------
# raw_hashtag (multi-row builder)
# ---------------------------------------------------------------------------


def _build_hashtag_records(parsed_data: dict) -> list[dict]:
    """Build raw_hashtag records from text_extra_raw or hashtag_list.

    Each text_extra item becomes one row. Falls back to extracting hashtag names
    from the flat hashtag_list if text_extra_raw is unavailable.

    Returns:
        list of dict, each representing one raw_hashtag row.
        Empty list if no hashtag data is available.
    """
    video_id = parsed_data.get("video_id")
    if not video_id:
        return []

    crawl_time = _to_datetime_str(parsed_data.get("crawl_time"))
    records: list[dict] = []

    # Priority 1: raw text_extra dict array (has hashtag_id, positions, etc.)
    text_extra_raw = parsed_data.get("text_extra_raw")
    if isinstance(text_extra_raw, list) and text_extra_raw:
        for item in text_extra_raw:
            if not isinstance(item, dict):
                continue
            hashtag_name = item.get("hashtag_name")
            if not hashtag_name:
                continue
            record = build_empty_raw_record("raw_hashtag")
            record["video_id"] = video_id
            record["hashtag_name"] = str(hashtag_name)
            record["hashtag_id"] = str(item["hashtag_id"]) if item.get("hashtag_id") is not None else None
            record["hashtag_type"] = _to_int(item.get("type"))
            record["is_commerce"] = item.get("is_commerce")
            record["caption_start"] = _to_int(item.get("caption_start"))
            record["caption_end"] = _to_int(item.get("caption_end"))
            record["start"] = _to_int(item.get("start"))
            record["end"] = _to_int(item.get("end"))
            record["crawl_time"] = crawl_time
            records.append(record)

        if records:
            return records

    # Priority 2: flat hashtag_list (strings only, from HTML parsing fallback)
    hashtag_list = parsed_data.get("hashtag_list")
    if isinstance(hashtag_list, list) and hashtag_list:
        for hashtag_name in hashtag_list:
            if not hashtag_name:
                continue
            record = build_empty_raw_record("raw_hashtag")
            record["video_id"] = video_id
            record["hashtag_name"] = str(hashtag_name)
            record["crawl_time"] = crawl_time
            records.append(record)

    return records


# ---------------------------------------------------------------------------
# raw_video_tag (multi-row builder)
# ---------------------------------------------------------------------------


def _build_video_tag_records(parsed_data: dict) -> list[dict]:
    """Build raw_video_tag records from video_tag_raw or video_tag.

    Each video_tag item becomes one row. Items must have at least
    tag_id or tag_name to be included.

    Returns:
        list of dict, each representing one raw_video_tag row.
        Empty list if no video_tag data is available.
    """
    video_id = parsed_data.get("video_id")
    if not video_id:
        return []

    crawl_time = _to_datetime_str(parsed_data.get("crawl_time"))
    records: list[dict] = []

    # Try video_tag_raw first (saved by scheduler from browser extraction)
    tag_source = parsed_data.get("video_tag_raw") or parsed_data.get("video_tag")
    if isinstance(tag_source, list) and tag_source:
        for item in tag_source:
            if not isinstance(item, dict):
                continue
            tag_id = item.get("tag_id")
            tag_name = item.get("tag_name")
            if not tag_id and not tag_name:
                continue
            record = build_empty_raw_record("raw_video_tag")
            record["video_id"] = video_id
            record["tag_id"] = str(tag_id) if tag_id is not None else None
            record["tag_name"] = str(tag_name) if tag_name is not None else None
            record["tag_level"] = _to_int(item.get("level"))
            record["crawl_time"] = crawl_time
            records.append(record)

    return records


# ---------------------------------------------------------------------------
# raw_chapter (multi-row builder)
# ---------------------------------------------------------------------------


def _build_chapter_records(parsed_data: dict) -> list[dict]:
    """Build raw_chapter records from chapter_list_raw or chapter_list.

    Each chapter_list item becomes one row. Video-level fields
    (chapter_abstract, chapter_review_status, chapter_recommend_type)
    are included on each row.

    Returns:
        list of dict, each representing one raw_chapter row.
        Empty list if no chapter_list data is available.
    """
    video_id = parsed_data.get("video_id")
    if not video_id:
        return []

    crawl_time = _to_datetime_str(parsed_data.get("crawl_time"))
    records: list[dict] = []

    # Try chapter_list_raw first (saved by scheduler from browser extraction)
    chapter_source = parsed_data.get("chapter_list_raw") or parsed_data.get("chapter_list")
    if isinstance(chapter_source, list) and chapter_source:
        for idx, item in enumerate(chapter_source):
            if not isinstance(item, dict):
                continue
            record = build_empty_raw_record("raw_chapter")
            record["video_id"] = video_id
            record["chapter_index"] = idx
            record["chapter_desc"] = str(item["desc"]) if item.get("desc") is not None else None
            record["chapter_detail"] = str(item["detail"]) if item.get("detail") is not None else None
            record["chapter_timestamp"] = _to_int(item.get("timestamp", item.get("start_time")))
            record["chapter_cover_url"] = str(item["url"]) if item.get("url") is not None else None
            # Video-level fields included on each row
            record["chapter_abstract"] = str(parsed_data.get("chapter_abstract", "") or "") or None
            record["chapter_review_status"] = _to_int(parsed_data.get("chapter_review_status"))
            record["chapter_recommend_type"] = str(parsed_data.get("chapter_recommend_type", "") or "") or None
            record["crawl_time"] = crawl_time
            records.append(record)

    return records


# ---------------------------------------------------------------------------
# raw_comment (multi-row builder)
# ---------------------------------------------------------------------------


def _build_comment_records(parsed_data: dict, crawl_context: dict = None) -> list[dict]:
    """Build raw_comment records from comment response JSON.

    Parses the comment_response_raw string (captured from network response
    /aweme/v1/web/comment/list/) and extracts individual comment records.

    Returns:
        list of dict, each representing one raw_comment row.
        Empty list if no comment response data is available.
    """
    video_id = parsed_data.get("video_id")
    if not video_id:
        return []

    crawl_time = _to_datetime_str(parsed_data.get("crawl_time"))
    response_raw = parsed_data.get("comment_response_raw")
    if not response_raw:
        return []

    # Parse the comment response JSON
    try:
        if isinstance(response_raw, str):
            response_data = json.loads(response_raw)
        elif isinstance(response_raw, dict):
            response_data = response_raw
        else:
            logger.debug(f"comment_response_raw has unexpected type: {type(response_raw)}")
            return []
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to parse comment_response_raw JSON: {e}")
        return []

    if not isinstance(response_data, dict):
        return []

    comments = response_data.get("comments", [])
    if not isinstance(comments, list) or not comments:
        # Record whether cursor/has_more/total exist for completeness metrics
        if any(k in response_data for k in ("cursor", "has_more", "total")):
            logger.debug(f"Comment response has page fields but no comments array")
        return []

    records: list[dict] = []
    for item in comments:
        if not isinstance(item, dict):
            continue

        comment_id = _get_nested(item, "cid")
        if not comment_id:
            continue  # skip items without comment_id

        record = build_empty_raw_record("raw_comment")
        record["video_id"] = video_id
        record["comment_id"] = str(comment_id)
        record["crawl_time"] = crawl_time

        # Extract comment-level fields
        record["comment_text"] = _str_or_none(_get_nested(item, "text"))
        record["comment_create_time"] = _to_int(_get_nested(item, "create_time"))
        record["comment_digg_count"] = _to_int(_get_nested(item, "digg_count"))
        record["comment_status"] = _to_int(_get_nested(item, "status"))
        record["comment_reply_total"] = _to_int(_get_nested(item, "reply_comment_total"))
        record["reply_id"] = _str_or_none(_get_nested(item, "reply_id"))
        record["reply_to_reply_id"] = _str_or_none(_get_nested(item, "reply_to_reply_id"))
        record["is_hot"] = _bool_or_none(_get_nested(item, "is_hot"))
        record["is_author_digged"] = _bool_or_none(_get_nested(item, "is_author_digged"))
        record["stick_position"] = _to_int(_get_nested(item, "stick_position"))
        record["label_text"] = _str_or_none(_get_nested(item, "label_text"))
        record["label_type"] = _to_int(_get_nested(item, "label_type"))

        # Extract user sub-object fields
        user = item.get("user")
        if isinstance(user, dict):
            record["comment_user_id"] = _str_or_none(user.get("uid"))
            record["comment_user_sec_uid"] = _str_or_none(user.get("sec_uid"))
            record["comment_user_nickname"] = _str_or_none(user.get("nickname"))
            record["comment_user_unique_id"] = _str_or_none(user.get("unique_id"))
            record["comment_user_region"] = _str_or_none(user.get("region"))

            # Extract avatar URL list from user.avatar_thumb.url_list
            avatar_thumb = user.get("avatar_thumb")
            if isinstance(avatar_thumb, dict):
                url_list = avatar_thumb.get("url_list")
                if isinstance(url_list, list):
                    record["comment_user_avatar_url_list"] = json.dumps(url_list, ensure_ascii=False)

        records.append(record)

    # Extract page-level fields from the response (set on every row)
    if records:
        page_cursor = response_data.get("cursor")
        page_has_more = response_data.get("has_more")
        page_total = response_data.get("total")

        for record in records:
            if record.get("comment_cursor") is None and page_cursor is not None:
                record["comment_cursor"] = _to_int(page_cursor)
            if record.get("comment_has_more") is None and page_has_more is not None:
                record["comment_has_more"] = _bool_or_none(page_has_more)
            if record.get("comment_total") is None and page_total is not None:
                record["comment_total"] = _to_int(page_total)

    logger.info(f"Built {len(records)} raw_comment records for video_id={video_id}")
    return records


# ---------------------------------------------------------------------------
# raw_related_video (multi-row builder)
# ---------------------------------------------------------------------------


def _build_related_video_records(parsed_data: dict, crawl_context: dict = None) -> list[dict]:
    """Build raw_related_video records from related response JSON.

    Parses the related_response_raw string (captured from network response
    /aweme/v1/web/aweme/related/) and extracts individual related video records.

    Returns:
        list of dict, each representing one raw_related_video row.
        Empty list if no related response data is available.
    """
    video_id = parsed_data.get("video_id")
    if not video_id:
        return []

    crawl_time = _to_datetime_str(parsed_data.get("crawl_time"))
    response_raw = parsed_data.get("related_response_raw")
    if not response_raw:
        return []

    # Parse the related response JSON
    try:
        if isinstance(response_raw, str):
            response_data = json.loads(response_raw)
        elif isinstance(response_raw, dict):
            response_data = response_raw
        else:
            logger.debug(f"related_response_raw has unexpected type: {type(response_raw)}")
            return []
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to parse related_response_raw JSON: {e}")
        return []

    if not isinstance(response_data, dict):
        return []

    aweme_list = response_data.get("aweme_list", [])
    if not isinstance(aweme_list, list) or not aweme_list:
        logger.debug(f"Related response has no aweme_list array")
        return []

    records: list[dict] = []
    for rank_idx, item in enumerate(aweme_list):
        if not isinstance(item, dict):
            continue

        related_video_id = _get_nested(item, "aweme_id")
        if not related_video_id:
            continue  # skip items without aweme_id

        record = build_empty_raw_record("raw_related_video")
        record["source_video_id"] = video_id
        record["related_video_id"] = str(related_video_id)
        record["related_rank_position"] = rank_idx
        record["crawl_time"] = crawl_time

        # Extract author sub-object
        author = item.get("author")
        if isinstance(author, dict):
            record["related_author_id"] = _str_or_none(author.get("uid"))
            record["related_author_sec_uid"] = _str_or_none(author.get("sec_uid"))
            record["related_author_nickname"] = _str_or_none(author.get("nickname"))

        # Extract text fields
        record["related_caption"] = _str_or_none(item.get("desc", item.get("caption")))
        record["related_create_time"] = _to_int(item.get("create_time"))
        record["related_duration_ms"] = _to_int(item.get("duration"))
        record["related_media_type"] = _to_int(item.get("media_type"))

        # Extract statistics sub-object
        statistics = item.get("statistics")
        if isinstance(statistics, dict):
            record["related_digg_count"] = _to_int(statistics.get("digg_count"))
            record["related_comment_count"] = _to_int(statistics.get("comment_count"))
            record["related_share_count"] = _to_int(statistics.get("share_count"))
            record["related_collect_count"] = _to_int(statistics.get("collect_count"))
            record["related_play_count"] = _to_int(statistics.get("play_count"))

        # Extract cover URL list from video.cover.url_list
        video = item.get("video")
        if isinstance(video, dict):
            cover = video.get("cover")
            if isinstance(cover, dict):
                url_list = cover.get("url_list")
                if isinstance(url_list, list) and url_list:
                    record["related_cover_url_list"] = json.dumps(url_list, ensure_ascii=False)

        # Extract music sub-object
        music = item.get("music")
        if isinstance(music, dict):
            record["related_music_id"] = _str_or_none(music.get("id_str", music.get("id")))
            record["related_music_title"] = _str_or_none(music.get("title"))

        # Preserve text_extra as raw JSON if present
        text_extra = item.get("text_extra")
        if isinstance(text_extra, list) and text_extra:
            record["related_text_extra_raw"] = json.dumps(text_extra, ensure_ascii=False)

        # Preserve video_tag as raw JSON if present
        video_tag = item.get("video_tag")
        if isinstance(video_tag, list) and video_tag:
            record["related_video_tag_raw"] = json.dumps(video_tag, ensure_ascii=False)

        # Extract chapter_abstract if present
        record["related_chapter_abstract"] = _str_or_none(item.get("chapter_abstract"))

        records.append(record)

    logger.info(f"Built {len(records)} raw_related_video records for source_video_id={video_id}")
    return records


def _get_nested(obj: dict, dotted_key: str):
    """Get a nested value from a dict using dot-separated keys."""
    parts = dotted_key.split(".")
    current = obj
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


def _str_or_none(value) -> str:
    """Convert to string or None."""
    if value is None:
        return None
    return str(value)


def _bool_or_none(value):
    """Convert to bool or None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lower = value.lower().strip()
        if lower in ("true", "1", "yes"):
            return True
        if lower in ("false", "0", "no"):
            return False
    return None


# ---------------------------------------------------------------------------
# raw_crawl_log (single-record builder)
# ---------------------------------------------------------------------------


def _build_crawl_log_record(parsed_data: dict, crawl_context: Optional[dict] = None) -> dict:
    """Build a single raw_crawl_log record.

    Uses crawl_context for task-level metadata; parsed_data for quality fields.

    Returns:
        dict representing one raw_crawl_log row.
        Always has at least crawl_time set.
    """
    record = build_empty_raw_record("raw_crawl_log")

    ctx = crawl_context or {}

    # --- Required fields ---
    record["crawl_batch_id"] = ctx.get("run_id") or parsed_data.get("run_id")
    record["crawl_time"] = _to_datetime_str(parsed_data.get("crawl_time"))
    record["target_url"] = ctx.get("target_url") or parsed_data.get("page_url")
    record["source_page_type"] = ctx.get("source_page_type") or parsed_data.get("source_entry", "manual_url")

    # --- Optional fields ---
    record["request_url"] = ctx.get("request_url")
    record["response_status"] = _to_int(ctx.get("response_status"))
    record["response_timestamp"] = record["crawl_time"]

    # --- Quality fields (from browser_extraction_summary via parsed_data or ctx) ---
    record["primary_source_key"] = (
        ctx.get("primary_source_key")
        or parsed_data.get("primary_source_key")
    )
    record["match_type"] = (
        ctx.get("match_type")
        or parsed_data.get("match_type")
    )
    record["confidence"] = (
        ctx.get("confidence")
        or parsed_data.get("confidence")
    )
    record["network_response_count"] = _to_int(
        ctx.get("network_response_count")
        or parsed_data.get("network_response_count")
    )
    record["runtime_objects_count"] = _to_int(
        ctx.get("runtime_objects_count")
        or parsed_data.get("runtime_objects_count")
    )

    return record


# ---------------------------------------------------------------------------
# Cross-table corrections
# ---------------------------------------------------------------------------


def _apply_cross_table_fixes(
    tables: dict[str, Union[dict, list]],
    parsed_data: dict,
) -> None:
    """Fix up values that depend on cross-table knowledge or parsed_data extras.

    - raw_video_detail: ensure page_url and primary_source_key are set.
    """
    detail = tables.get("raw_video_detail", {})
    if isinstance(detail, dict):
        if detail.get("page_url") is None:
            detail["page_url"] = parsed_data.get("page_url")
        if detail.get("primary_source_key") is None:
            detail["primary_source_key"] = parsed_data.get("primary_source_key")


# ---------------------------------------------------------------------------
# Mini helpers
# ---------------------------------------------------------------------------


def _to_int(value) -> Optional[int]:
    """Safely convert value to int or None."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value == value else None  # NaN check
    if isinstance(value, str):
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    return None