#!/usr/bin/env python
"""
merge_real_raw_runs.py

Merge part1-part5 crawl results (real_raw_5000) into a unified candidate dataset,
then perform comprehensive quality audit.

Usage:
    D:/CodeData/software/Anaconda/Anaconda3/envs/ra/python.exe src/processing/merge_real_raw_runs.py \
        --run-ids 20260509_212100 20260510_151052 20260510_183114 20260510_210600 20260511_084923 \
        --output-dir data/interim/real_raw_5000_candidate
"""
import argparse
import csv
import json
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Project root (2 levels up from src/processing/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INTERIM_DIR = os.path.join(PROJECT_ROOT, 'data', 'interim')

# Table names in output order
TABLE_NAMES = [
    'raw_video_detail',
    'raw_author',
    'raw_music',
    'raw_hashtag',
    'raw_video_tag',
    'raw_video_media',
    'raw_video_status_control',
    'raw_chapter',
    'raw_comment',
    'raw_related_video',
    'raw_crawl_log',
]

# Core fields for coverage check
CORE_FIELDS = [
    'create_time', 'duration_ms', 'digg_count',
    'comment_count', 'share_count', 'collect_count',
]

# Dedup configuration per table
#   key: column name or list of column names for dedup
#   priority: sort fields (highest priority first) applied before keep='first'
DEDUP_CONFIG = {
    'raw_video_detail': {
        'key': 'video_id',
        # Priority: exact+high > others, then by field completeness
        'priority_fields': ['match_type_order', 'confidence_order', '_completeness'],
    },
    'raw_author': {
        'key': 'author_id',
        'priority_fields': ['_completeness'],
    },
    'raw_music': {
        'key': 'video_id',
        'priority_fields': ['_completeness'],
    },
    'raw_video_media': {
        'key': 'video_id',
        'priority_fields': ['_completeness'],
    },
    'raw_video_status_control': {
        'key': 'video_id',
        'priority_fields': ['_completeness'],
    },
    'raw_hashtag': {
        'key': ['video_id', 'hashtag_name'],
        'priority_fields': [],
    },
    'raw_comment': {
        'key': 'comment_id',
        'priority_fields': [],
    },
    'raw_related_video': {
        'key': ['source_video_id', 'related_video_id'],
        'priority_fields': [],
    },
    'raw_crawl_log': {
        'key': None,  # keep all rows
        'priority_fields': [],
    },
    'raw_video_tag': {
        'key': None,
        'priority_fields': [],
    },
    'raw_chapter': {
        'key': None,
        'priority_fields': [],
    },
}

# Cross-table join checks: (left_table, left_key, right_table, right_key)
CROSS_TABLE_JOINS = [
    ('raw_author',           'author_id',       'raw_video_detail', 'author_id'),
    ('raw_music',            'video_id',        'raw_video_detail', 'video_id'),
    ('raw_video_media',      'video_id',        'raw_video_detail', 'video_id'),
    ('raw_video_status_control', 'video_id',    'raw_video_detail', 'video_id'),
    ('raw_hashtag',          'video_id',        'raw_video_detail', 'video_id'),
    ('raw_comment',          'video_id',        'raw_video_detail', 'video_id'),
    ('raw_related_video',    'source_video_id', 'raw_video_detail', 'video_id'),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge real_raw_5000 batch crawl results'
    )
    parser.add_argument(
        '--run-ids', nargs='+', required=True,
        help='List of run_ids to merge (e.g., 20260509_212100 20260510_151052 ...)'
    )
    parser.add_argument(
        '--output-dir', required=True,
        help='Output directory (relative to project root or absolute)'
    )
    return parser.parse_args()


def resolve_path(path):
    """Resolve path: if relative, prepend PROJECT_ROOT."""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def load_batch_csv(run_id, table_name):
    """Load a single table CSV from a batch run directory."""
    path = os.path.join(INTERIM_DIR, run_id, f'{table_name}_{run_id}.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        # Drop unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
        # Replace empty strings with NaN for numeric-like fields
        df = df.replace(r'^\s*$', np.nan, regex=True)
        return df
    except Exception as e:
        print(f'  [ERROR] Failed to read {path}: {e}', file=sys.stderr)
        return pd.DataFrame()


def compute_completeness(row, fields):
    """Count how many of the given fields are non-null in a row."""
    return sum(1 for f in fields if pd.notna(row.get(f)))


def check_video_id_consistency(df):
    """Return (consistent_count, total_checked, mismatch_count)."""
    if 'video_id' not in df.columns or 'page_url' not in df.columns:
        return None, 0, 0
    consistent = 0
    for _, row in df.iterrows():
        vid = str(row.get('video_id', ''))
        url = str(row.get('page_url', ''))
        if vid and vid in url:
            consistent += 1
    total = len(df)
    return consistent, total, total - consistent


# ---------------------------------------------------------------------------
#  Merge logic per table
# ---------------------------------------------------------------------------

def merge_video_detail(batches, run_ids, crawl_log_master):
    """Merge raw_video_detail with quality-aware dedup."""
    merged = pd.concat(batches, ignore_index=True)
    rows_before = len(merged)
    print(f'  Total rows before dedup: {rows_before}')

    # Attach match_type / confidence from crawl_log via page_url -> target_url
    if not crawl_log_master.empty and 'page_url' in merged.columns:
        log_cols = ['target_url', 'match_type', 'confidence']
        log_lookup = crawl_log_master[log_cols].drop_duplicates(subset=['target_url'])
        merged = merged.merge(
            log_lookup,
            left_on='page_url', right_on='target_url', how='left'
        )
        # Order mapping: lower = better
        merged['match_type_order'] = (
            merged['match_type']
            .map({'exact': 0, 'partial': 1, 'none': 2, 'unknown': 3})
            .fillna(99)
        )
        merged['confidence_order'] = (
            merged['confidence']
            .map({'high': 0, 'medium': 1, 'low': 2, 'unknown': 3})
            .fillna(99)
        )
        merged['_completeness'] = merged.apply(
            lambda r: compute_completeness(
                r, CORE_FIELDS + ['author_id', 'video_id', 'page_url']
            ),
            axis=1,
        )
        merged = merged.sort_values(
            ['match_type_order', 'confidence_order', '_completeness'],
            ascending=True,
        )
        # Drop temp columns after sort
        merged = merged.drop(
            columns=['target_url', 'match_type', 'confidence',
                     'match_type_order', 'confidence_order', '_completeness'],
            errors='ignore',
        )
    else:
        # Fallback: sort by field completeness only
        merged['_completeness'] = merged.apply(
            lambda r: compute_completeness(
                r, CORE_FIELDS + ['author_id', 'video_id', 'page_url']
            ),
            axis=1,
        )
        merged = merged.sort_values('_completeness', ascending=False)
        merged = merged.drop(columns=['_completeness'])

    before = len(merged)
    merged = merged.drop_duplicates(subset=['video_id'], keep='first')
    duplicates_removed = before - len(merged)
    print(f'  Rows after dedup: {len(merged)} (removed {duplicates_removed})')
    return merged, rows_before, duplicates_removed


def merge_simple_table(merged, config):
    """Generic merge for tables without external quality fields.
    `merged` is already a concatenated DataFrame from all batches."""
    rows_before = len(merged)

    dedup_key = config.get('key')
    if dedup_key is None:
        # Keep all rows
        print(f'  Total rows (no dedup): {rows_before}')
        return merged, rows_before, 0

    if isinstance(dedup_key, list):
        existing = [k for k in dedup_key if k in merged.columns]
    else:
        existing = [dedup_key] if dedup_key in merged.columns else []

    if not existing:
        print(f'  [WARN] Dedup key(s) not found, keeping all rows')
        return merged, rows_before, 0

    # Sort by completeness if configured
    priority = config.get('priority_fields', [])
    if '_completeness' in priority:
        # Use ALL columns for completeness
        merged['_completeness'] = merged.apply(
            lambda r: int(pd.notna(r).sum()), axis=1
        )
        merged = merged.sort_values('_completeness', ascending=False)
        merged = merged.drop(columns=['_completeness'])

    before = len(merged)
    merged = merged.drop_duplicates(subset=existing, keep='first')
    duplicates_removed = before - len(merged)
    print(f'  Rows after dedup: {len(merged)} (removed {duplicates_removed})')
    return merged, rows_before, duplicates_removed


# ---------------------------------------------------------------------------
#  Quality audit
# ---------------------------------------------------------------------------

def run_quality_audit(output_dir, run_ids, summary, crawl_log_master):
    """Run the full quality audit and write audit files."""
    print('\n' + '=' * 70)
    print('QUALITY AUDIT')
    print('=' * 70)

    audit = {
        'audit_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input_run_ids': run_ids,
        'output_dir': output_dir,
    }

    # Load merged tables
    loaded = {}
    for tn in TABLE_NAMES:
        path = os.path.join(output_dir, f'{tn}_real_raw_5000_candidate.csv')
        if os.path.exists(path):
            try:
                loaded[tn] = pd.read_csv(path, dtype=str, keep_default_na=False).replace(r'^\s*$', np.nan, regex=True)
            except Exception:
                loaded[tn] = pd.DataFrame()
        else:
            loaded[tn] = pd.DataFrame()

    detail = loaded.get('raw_video_detail', pd.DataFrame())

    # --- 1. Raw video detail stats ---
    if 'video_id' in detail.columns:
        unique_videos = detail['video_id'].nunique()
        total_detail = len(detail)
        dup_videos = total_detail - unique_videos
    else:
        unique_videos = 0
        total_detail = 0
        dup_videos = 0
    audit['unique_video_ids'] = int(unique_videos)
    audit['total_detail_rows'] = int(total_detail)
    audit['duplicated_video_ids'] = int(dup_videos)

    # --- 2. Valid samples ---
    valid = pd.Series(True, index=detail.index)
    for col in ['video_id', 'author_id', 'page_url']:
        if col in detail.columns:
            valid &= detail[col].notna()
    audit['valid_samples'] = int(valid.sum())
    audit['valid_sample_ratio'] = round(float(valid.sum() / len(detail)), 4) if len(detail) > 0 else 0.0

    # --- 3. Video ID consistency ---
    consistent, total_checked, mismatch = check_video_id_consistency(detail)
    audit['video_id_consistent'] = int(consistent) if consistent is not None else None
    audit['video_id_consistency_ratio'] = round(consistent / total_checked, 4) if total_checked > 0 else 0.0
    audit['video_id_mismatch'] = int(mismatch) if mismatch is not None else None

    # --- 4. Core field coverage ---
    coverage = {}
    for f in CORE_FIELDS:
        if f in detail.columns:
            nn = int(detail[f].notna().sum())
            cov = round(nn / len(detail), 4) if len(detail) > 0 else 0.0
            coverage[f] = {'non_null': nn, 'total': len(detail), 'coverage': cov}
        else:
            coverage[f] = {'non_null': 0, 'total': len(detail), 'coverage': 0.0}
    audit['core_field_coverage'] = coverage

    # --- 5. match_type / confidence (from crawl_log master) ---
    if not crawl_log_master.empty:
        match_dist = crawl_log_master['match_type'].value_counts().to_dict()
        conf_dist = crawl_log_master['confidence'].value_counts().to_dict()
        audit['match_type_distribution'] = {str(k): int(v) for k, v in match_dist.items()}
        audit['confidence_distribution'] = {str(k): int(v) for k, v in conf_dist.items()}
        none_count = int((crawl_log_master['match_type'] == 'none').sum())
        audit['none_match_count'] = none_count
        audit['none_match_ratio'] = round(none_count / len(crawl_log_master), 4)
        high = int(conf_dist.get('high', 0))
        audit['high_confidence_ratio'] = round(high / len(crawl_log_master), 4)

    # --- 6. Low-quality samples ---
    low_q = 0
    for _, row in detail.iterrows():
        is_low = False
        if pd.isna(row.get('video_id')):
            is_low = True
        if pd.isna(row.get('author_id')):
            is_low = True
        if pd.isna(row.get('page_url')):
            is_low = True
        all_core_empty = all(pd.isna(row.get(f)) for f in CORE_FIELDS)
        if all_core_empty:
            is_low = True
        if is_low:
            low_q += 1
    audit['low_quality_samples'] = {
        'count': low_q,
        'ratio': round(low_q / len(detail), 4) if len(detail) > 0 else 0.0,
    }

    # --- 7. Cross-table join checks ---
    joins = {}
    for lt, lk, rt, rk in CROSS_TABLE_JOINS:
        key = f'{lt}.{lk} -> {rt}.{rk}'
        if lt not in loaded or rt not in loaded:
            joins[key] = {'status': 'skipped', 'reason': 'table not loaded'}
            continue
        ldf = loaded[lt]
        rdf = loaded[rt]
        if lk not in ldf.columns or rk not in rdf.columns:
            joins[key] = {'status': 'skipped', 'reason': f'column {lk} or {rk} not found'}
            continue
        lkeys = set(ldf[lk].dropna().unique())
        rkeys = set(rdf[rk].dropna().unique())
        matched = len(lkeys & rkeys)
        left_total = len(lkeys)
        joins[key] = {
            'left_total': left_total,
            'matched': matched,
            'match_ratio': round(matched / left_total, 4) if left_total > 0 else 0.0,
        }
    audit['cross_table_joins'] = joins

    # --- 8. Trigger rates ---
    triggers = {}
    if 'raw_hashtag' in loaded and 'video_id' in loaded['raw_hashtag'].columns:
        v = loaded['raw_hashtag']['video_id'].dropna().nunique()
        triggers['hashtag'] = {
            'videos_with_hashtag': int(v),
            'trigger_rate': round(v / unique_videos, 4) if unique_videos > 0 else 0.0,
        }
    if 'raw_comment' in loaded and 'video_id' in loaded['raw_comment'].columns:
        v = loaded['raw_comment']['video_id'].dropna().nunique()
        triggers['comment'] = {
            'videos_with_comment': int(v),
            'trigger_rate': round(v / unique_videos, 4) if unique_videos > 0 else 0.0,
        }
    if 'raw_related_video' in loaded and 'source_video_id' in loaded['raw_related_video'].columns:
        v = loaded['raw_related_video']['source_video_id'].dropna().nunique()
        triggers['related_video'] = {
            'videos_with_related': int(v),
            'trigger_rate': round(v / unique_videos, 4) if unique_videos > 0 else 0.0,
        }
    audit['trigger_rates'] = triggers

    # --- 9. Table summary ---
    table_rows = []
    for tn in TABLE_NAMES:
        df = loaded.get(tn, pd.DataFrame())
        table_rows.append({
            'table': tn,
            'rows': len(df),
            'columns': len(df.columns) if not df.empty else 0,
        })
    audit['table_summary'] = table_rows

    # --- 10. Overall assessment ---
    meets_5000 = unique_videos >= 5000
    valid_ok = audit.get('valid_samples', 0) >= 5000
    cov_ok = all(v.get('coverage', 0) >= 0.5 for v in coverage.values())
    none_ok = audit.get('none_match_ratio', 1.0) <= 0.30
    cross_ok = all(
        v.get('match_ratio', 0) >= 0.5
        for v in joins.values() if isinstance(v, dict) and 'match_ratio' in v
    )

    audit['overall_assessment'] = {
        'unique_video_ids_meets_5000': meets_5000,
        'valid_samples_meets_5000': valid_ok,
        'core_field_coverage_acceptable': cov_ok,
        'none_match_ratio_acceptable': none_ok,
        'cross_table_join_acceptable': cross_ok,
    }

    # Recommend batch 6?
    reasons = []
    if not meets_5000:
        reasons.append(f'Unique video_ids ({unique_videos}) < 5000')
    if not valid_ok:
        reasons.append(f'Valid samples ({audit["valid_samples"]}) < 5000')
    if not cov_ok:
        low = [f for f, v in coverage.items() if v.get('coverage', 0) < 0.5]
        reasons.append(f'Low core field coverage: {low}')
    if not none_ok:
        reasons.append(f'None-match ratio ({audit["none_match_ratio"]:.2%}) > 30%')
    if not cross_ok:
        bad = [k for k, v in joins.items()
               if isinstance(v, dict) and v.get('match_ratio', 1) < 0.5]
        reasons.append(f'Poor cross-table joins: {bad}')

    suggest_b6 = bool(reasons) and not meets_5000
    # Only suggest batch 6 if we don't have 5000 unique IDs
    # (other issues can be fixed by filtering, not by more data)

    audit['recommend_batch_6'] = suggest_b6
    audit['recommend_batch_6_reasons'] = reasons if reasons else [
        'No blocking issues found'
    ]

    # Write audit files
    audit_path = os.path.join(output_dir, 'quality_audit.json')
    with open(audit_path, 'w', encoding='utf-8') as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f'  quality_audit.json written')

    tsv_path = os.path.join(output_dir, 'table_summary.csv')
    with open(tsv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['table', 'rows', 'columns'])
        w.writeheader()
        for row in table_rows:
            w.writerow(row)
    print(f'  table_summary.csv written')

    return audit


def print_audit_summary(audit):
    """Print a human-readable audit summary."""
    print('\n' + '=' * 70)
    print('AUDIT SUMMARY')
    print('=' * 70)
    print(f'  Unique video IDs:     {audit["unique_video_ids"]} / 5000')
    print(f'  Valid samples:        {audit["valid_samples"]} '
          f'({audit["valid_sample_ratio"]:.1%})')
    print(f'  Video ID consistency: {audit.get("video_id_consistency_ratio", "N/A"):.1%}')
    print(f'  None-match ratio:     {audit.get("none_match_ratio", 0):.1%}')
    print(f'  High confidence:      {audit.get("high_confidence_ratio", 0):.1%}')
    print(f'  Low quality samples:  {audit["low_quality_samples"]["count"]} '
          f'({audit["low_quality_samples"]["ratio"]:.1%})')
    print()
    print('  Core field coverage:')
    for f, v in audit['core_field_coverage'].items():
        print(f'    {f:20s}  {v["coverage"]:.1%}  ({v["non_null"]}/{v["total"]})')
    print()
    print('  Match type:')
    for k, v in audit.get('match_type_distribution', {}).items():
        print(f'    {k:15s}  {v}')
    print()
    print('  Confidence:')
    for k, v in audit.get('confidence_distribution', {}).items():
        print(f'    {k:15s}  {v}')
    print()
    print('  Trigger rates:')
    for k, v in audit.get('trigger_rates', {}).items():
        print(f'    {k:20s}  {v["trigger_rate"]:.1%}  ({v.get("videos_with_hashtag", v.get("videos_with_comment", v.get("videos_with_related", "?")))} videos)')
    print()
    print('  Cross-table joins:')
    for k, v in audit.get('cross_table_joins', {}).items():
        ratio = v.get('match_ratio', 'N/A')
        if isinstance(ratio, float):
            print(f'    {k:60s}  {ratio:.1%}  ({v["matched"]}/{v["left_total"]})')
        else:
            print(f'    {k:60s}  {v.get("status", "N/A")}')
    print()
    print('  Overall assessment:')
    for k, v in audit.get('overall_assessment', {}).items():
        print(f'    {k:45s}  {"PASS" if v else "FAIL"}')
    print()
    print(f'  Recommend batch 6:    {"YES" if audit["recommend_batch_6"] else "NO"}')
    if audit['recommend_batch_6']:
        for r in audit['recommend_batch_6_reasons']:
            print(f'    - {r}')
    else:
        print(f'    - {audit["recommend_batch_6_reasons"][0]}')


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    run_ids = args.run_ids
    output_dir = resolve_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 70)
    print(f'real_raw_5000 Merge & Quality Audit')
    print(f'  Run IDs:  {run_ids}')
    print(f'  Output:   {output_dir}')
    print(f'  Time:     {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 70)

    summary = {
        'merge_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input_run_ids': run_ids,
        'output_dir': output_dir,
        'tables': {},
    }

    # ------------------------------------------------------------------
    # Pre-load all crawl_logs for quality metadata
    # ------------------------------------------------------------------
    print('\n--- Loading crawl_logs for quality metadata ---')
    all_logs = []
    for rid in run_ids:
        df = load_batch_csv(rid, 'raw_crawl_log')
        if not df.empty:
            df['source_run_id'] = rid
            all_logs.append(df)
            print(f'  {rid}: {len(df)} rows')
    crawl_log_master = pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame()

    # ------------------------------------------------------------------
    # Merge each table
    # ------------------------------------------------------------------
    for tn in TABLE_NAMES:
        print(f'\n--- {tn} ---')
        config = DEDUP_CONFIG.get(tn, {})
        batches = []
        for rid in run_ids:
            df = load_batch_csv(rid, tn)
            if not df.empty:
                batches.append((rid, df))
                print(f'  {rid}: {len(df)} rows')

        if not batches:
            print(f'  No data — writing empty header')
            pd.DataFrame().to_csv(
                os.path.join(output_dir, f'{tn}_real_raw_5000_candidate.csv'),
                index=False,
            )
            summary['tables'][tn] = {
                'rows_before_merge': 0,
                'rows_after_merge': 0,
                'duplicates_removed': 0,
            }
            continue

        # Concat
        concat_df = pd.concat([df for _, df in batches], ignore_index=True)

        # Table-specific merge
        if tn == 'raw_video_detail':
            merged_df, before, dup = merge_video_detail(
                [df for _, df in batches], run_ids, crawl_log_master
            )
        elif tn in ('raw_video_tag', 'raw_chapter', 'raw_crawl_log'):
            merged_df, before, dup = merge_simple_table(concat_df, config)
        elif tn in ('raw_author', 'raw_music', 'raw_video_media',
                     'raw_video_status_control', 'raw_hashtag',
                     'raw_comment', 'raw_related_video'):
            merged_df, before, dup = merge_simple_table(concat_df, config)
        else:
            # Fallback: just concat
            merged_df = concat_df
            before = len(concat_df)
            dup = 0

        # Remove temp source_run_id
        if 'source_run_id' in merged_df.columns:
            merged_df = merged_df.drop(columns=['source_run_id'])

        out_path = os.path.join(output_dir, f'{tn}_real_raw_5000_candidate.csv')
        merged_df.to_csv(out_path, index=False, encoding='utf-8')
        print(f'  Written: {out_path}  ({len(merged_df)} rows, {len(merged_df.columns)} cols)')

        summary['tables'][tn] = {
            'rows_before_merge': before,
            'rows_after_merge': len(merged_df),
            'duplicates_removed': dup,
        }

    # Write merge report
    report_path = os.path.join(output_dir, 'merge_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'\nMerge report written to {report_path}')

    # ------------------------------------------------------------------
    # Quality audit
    # ------------------------------------------------------------------
    audit = run_quality_audit(output_dir, run_ids, summary, crawl_log_master)
    print_audit_summary(audit)
    print('\nDone.')


if __name__ == '__main__':
    main()
