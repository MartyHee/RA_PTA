# Deprecated Field Cleanup Report

**Date:** 2026-05-06
**Task:** Remove 13 deprecated old field names from crawler output pipeline
**Verification method:** Mock mode (2 URLs) — verified CSV headers and model output

## Removed Fields (13)

| # | Old Field | Replacement | Reason |
|---|-----------|-------------|--------|
| 1 | `duration_sec` | `duration_ms` | Response value is in milliseconds |
| 2 | `publish_time_raw` | `create_time` | Response field name is create_time |
| 3 | `like_count_raw` | `digg_count` | Response field name is statistics.digg_count |
| 4 | `comment_count_raw` | `comment_count` | Unified naming, removed _raw suffix |
| 5 | `share_count_raw` | `share_count` | Unified naming, removed _raw suffix |
| 6 | `cover_url` | `cover_url_list` | Explicitly a URL list field |
| 7 | `origin_cover_url` | `origin_cover_url_list` | Explicitly a URL list field |
| 8 | `dynamic_cover_url` | `dynamic_cover_url_list` | Explicitly a URL list field |
| 9 | `author_name` | `nickname` | Align with response field name |
| 10 | `author_signature` | `signature` | Simplified naming |
| 11 | `author_follower_count` | `follower_count` | Simplified naming |
| 12 | `author_total_favorited` | `total_favorited` | Simplified naming |
| 13 | `music_name` | `music_title` | Align with response field name |

## Files Modified

| File | Changes |
|------|---------|
| `src/schemas/tables.py` | WebVideoMeta: removed old fields, added new fields |
| `src/crawler/browser_client.py` | `_mock_response()`: updated field names in mock dict |
| `src/crawler/scheduler.py` | `field_mapping` dict, type conversion blocks, log target_fields |
| `src/crawler/parser.py` | `field_rules`, `_parse_script_data`, `_extract_html_data`, `_normalize_counts`, `mock_parse`, `create_web_video_meta` log |
| `src/crawler/extractors.py` | AuthorExtractor, StatsExtractor result keys; ExtractorFactory field_to_extractor |
| `configs/fields.yaml` | `web_video_meta.fields`: renamed fields; `availability`: updated field names; kept `field_deprecations` |

## Verification Results

**Mock mode:** ✅ Passed (2 URLs)
- CSV column headers verified — no old field names present
- Model output dict verified — all field names use new naming

**Remaining deprecated field references (intentionally kept):**
- `configs/fields.yaml` → `field_deprecations` section (documentation of migration)
- 13 old fields no longer appear in any active output, schema, or mapping

## Confidence

The cleanup scope covers the full extraction → mapping → parsing → schema → output pipeline.
All 13 fields from the removal list are confirmed absent from mock-mode CSV output.
