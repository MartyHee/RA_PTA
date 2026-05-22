"""Multimodal 文本字段覆盖率、长度分布和可用性分析

分析 real_raw_5000 中各文本字段：
  - 字段来源识别（哪个 raw 表、哪个列）
  - 覆盖率与空值率（整体 + 按 split）
  - 长度分布（字符、简单 token）
  - 字符/词稀疏性（unique chars, simple terms, 数字/拉丁/中文占比）
  - split 分布一致性
  - 候选字段推荐

用法:
    python src/analysis/analyze_text_fields.py --dataset real_raw_5000

输出:
    outputs/analysis/text_fields/<dataset>/<run_id>/
        text_field_report.json
        text_field_summary.csv
        text_field_examples.csv
        text_field_report.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.io import read_csv_safe  # noqa: E402


# ── 文本字段定义 ──────────────────────────────────────────────────────────
# 与 build_multimodal_real_raw.py 中 text_source_fields 完全一致
TEXT_SOURCE_FIELDS: dict[str, dict[str, Any]] = {
    "desc": {
        "source_table": "raw_video_detail",
        "source_column": "desc",
        "field_group": "desc",
        "group_label": "视频描述",
        "join_key": "video_id",
        "aggregation": "first",  # 每个 video_id 一条
    },
    "caption": {
        "source_table": "raw_video_detail",
        "source_column": "caption",
        "field_group": "desc",
        "group_label": "视频标题",
        "join_key": "video_id",
        "aggregation": "first",
    },
    "hashtag_name": {
        "source_table": "raw_hashtag",
        "source_column": "hashtag_name",
        "field_group": "hashtag",
        "group_label": "话题标签",
        "join_key": "video_id",
        "aggregation": "concat_space",  # 多个时用空格拼接
    },
    "comment_text": {
        "source_table": "raw_comment",
        "source_column": "comment_text",
        "field_group": "comment",
        "group_label": "评论文本",
        "join_key": "video_id",
        "aggregation": "concat_dot",  # 多个时用 ". " 拼接
    },
    "music_title": {
        "source_table": "raw_music",
        "source_column": "music_title",
        "field_group": "music",
        "group_label": "音乐标题",
        "join_key": "video_id",
        "aggregation": "first",
    },
    "music_author": {
        "source_table": "raw_music",
        "source_column": "music_author",
        "field_group": "music",
        "group_label": "音乐作者",
        "join_key": "video_id",
        "aggregation": "first",
    },
    "signature": {
        "source_table": "raw_author",
        "source_column": "signature",
        "field_group": "author",
        "group_label": "作者签名",
        "join_key": "author_id",
        "aggregation": "first",
    },
}

# 当前 Multimodal 使用的文本字段
CURRENT_MULTIMODAL_FIELDS = [
    "desc", "caption", "hashtag_name", "comment_text",
    "music_title", "music_author", "signature",
]


def safe_str(val: Any) -> str:
    """安全的字符串转换，NaN / None → ""。"""
    if pd.isna(val) or val is None:
        return ""
    s = str(val).strip()
    return s


def load_table(raw_root: Path, filename: str, label: str) -> pd.DataFrame:
    """加载单张 raw 表。"""
    path = raw_root / filename
    if not path.exists() or path.stat().st_size == 0:
        print(f"  [WARN] {label} 不存在或为空: {path}")
        return pd.DataFrame()
    try:
        df, enc = read_csv_safe(str(path))
    except Exception as e:
        print(f"  [ERROR] 读取 {label} 失败: {e}")
        return pd.DataFrame()
    print(f"  [{label}] {len(df)} 行 x {len(df.columns)} 列 (编码: {enc})")
    df.columns = df.columns.str.strip()
    return df


def compute_field_coverage(
    field_name: str,
    field_cfg: dict[str, Any],
    video_detail: pd.DataFrame,
    author: pd.DataFrame,
    music: pd.DataFrame,
    hashtag: pd.DataFrame,
    comment: pd.DataFrame,
    all_video_ids: list[int],
    split_map: dict[int, str],  # video_id -> split
) -> dict[str, Any]:
    """对单个文本字段计算覆盖率和长度分布。"""
    col = field_cfg["source_column"]
    src_table = field_cfg["source_table"]
    join_key = field_cfg["join_key"]
    agg = field_cfg["aggregation"]

    # 提取该字段的文本 Series (video_id -> text)
    field_texts: dict[int, str] = {}

    # 根据源表提取
    if src_table == "raw_video_detail":
        for _, row in video_detail.iterrows():
            vid = int(row["video_id"])
            text = safe_str(row.get(col, ""))
            field_texts[vid] = text

    elif src_table == "raw_author":
        # signature 通过 author_id 关联到 video_detail
        author_map: dict[Any, str] = {}
        for _, row in author.iterrows():
            aid = row.get("author_id")
            sig = safe_str(row.get(col, ""))
            if pd.notna(aid):
                author_map[aid] = sig
        for _, row in video_detail.iterrows():
            vid = int(row["video_id"])
            aid = row.get("author_id")
            if pd.notna(aid) and aid in author_map:
                field_texts[vid] = author_map[aid]
            else:
                field_texts[vid] = ""

    elif src_table == "raw_music":
        # 每个 video_id 一条记录
        for _, row in music.iterrows():
            vid = int(row["video_id"])
            text = safe_str(row.get(col, ""))
            if vid not in field_texts:
                field_texts[vid] = text
            elif text and not field_texts[vid]:
                field_texts[vid] = text

    elif src_table == "raw_hashtag":
        # 每个 video_id 多条，拼接
        groups = hashtag.groupby("video_id")
        for vid, group in groups:
            vals = [
                safe_str(r.get(col, ""))
                for _, r in group.iterrows()
                if safe_str(r.get(col, ""))
            ]
            if vals:
                field_texts[int(vid)] = " ".join(vals)
        # 未覆盖的为空白

    elif src_table == "raw_comment":
        groups = comment.groupby("video_id")
        for vid, group in groups:
            vals = [
                safe_str(r.get(col, ""))
                for _, r in group.iterrows()
                if safe_str(r.get(col, ""))
            ]
            if vals:
                field_texts[int(vid)] = ". ".join(vals)

    # 填充所有 video_id
    result: dict[int, str] = {}
    for vid in all_video_ids:
        result[vid] = field_texts.get(vid, "")

    # ── 统计 ──────────────────────────────────────────────────────────────
    total = len(all_video_ids)
    texts = list(result.values())
    non_empty = [t for t in texts if len(t) > 0]
    non_empty_count = len(non_empty)
    empty_count = total - non_empty_count
    coverage_rate = round(non_empty_count / total * 100, 2) if total > 0 else 0.0
    missing_rate = round(empty_count / total * 100, 2) if total > 0 else 0.0

    unique_texts = len(set(non_empty)) if non_empty else 0
    duplicate_rate = (
        round((non_empty_count - unique_texts) / non_empty_count * 100, 2)
        if non_empty_count > 0
        else 0.0
    )

    # 长度分布
    char_lens = [len(t) for t in non_empty] if non_empty else [0]
    token_lens = [len(t.split()) for t in non_empty] if non_empty else [0]

    char_len_mean = round(float(np.mean(char_lens)), 2)
    char_len_median = round(float(np.median(char_lens)), 2)
    char_len_p90 = round(float(np.percentile(char_lens, 90)), 2)
    char_len_p95 = round(float(np.percentile(char_lens, 95)), 2)
    char_len_max = int(max(char_lens))
    char_len_min = int(min(char_lens))
    token_len_mean = round(float(np.mean(token_lens)), 2)
    very_short_ratio = round(
        sum(1 for l in char_lens if l <= 2) / non_empty_count * 100, 2
    ) if non_empty_count > 0 else 0.0
    long_text_ratio = round(
        sum(1 for l in char_lens if l >= 100) / non_empty_count * 100, 2
    ) if non_empty_count > 0 else 0.0

    # 字符集多样性
    all_chars: list[str] = []
    for t in non_empty:
        all_chars.extend(list(t))
    unique_chars = len(set(all_chars)) if all_chars else 0

    # 字符类别比例
    total_chars = len(all_chars) if all_chars else 1
    digit_chars = sum(1 for c in all_chars if c.isdigit())
    latin_chars = sum(1 for c in all_chars if c.isascii() and c.isalpha())
    chinese_chars = sum(1 for c in all_chars if "一" <= c <= "鿿")
    emoji_chars = sum(
        1
        for c in all_chars
        if ord(c) >= 0x1F300 or (0x2600 <= ord(c) <= 0x27BF) or (0xFE00 <= ord(c) <= 0xFE0F)
    )

    digit_ratio = round(digit_chars / total_chars * 100, 2)
    latin_ratio = round(latin_chars / total_chars * 100, 2)
    chinese_char_ratio = round(chinese_chars / total_chars * 100, 2)
    emoji_ratio = round(emoji_chars / total_chars * 100, 2)

    # Top 10 characters
    char_counter = Counter(all_chars)
    top_chars = {k: v for k, v in char_counter.most_common(10)}

    # Top 10 terms by simple split
    all_terms: list[str] = []
    for t in non_empty:
        all_terms.extend(t.split())
    term_counter = Counter(all_terms)
    top_terms = {k: v for k, v in term_counter.most_common(10)}

    # ── split 分布 ────────────────────────────────────────────────────────
    split_stats: dict[str, dict[str, Any]] = {}
    for split_name in ["train", "val", "test"]:
        split_vids = [vid for vid in all_video_ids if split_map.get(vid) == split_name]
        split_texts = [result[vid] for vid in split_vids]
        split_non_empty = [t for t in split_texts if len(t) > 0]
        split_total = len(split_vids)
        split_cov = round(len(split_non_empty) / split_total * 100, 2) if split_total > 0 else 0.0
        split_lens = [len(t) for t in split_non_empty] or [0]
        split_stats[split_name] = {
            "total": split_total,
            "non_empty": len(split_non_empty),
            "coverage_rate": split_cov,
            "char_len_mean": round(float(np.mean(split_lens)), 2),
            "unique_texts": len(set(split_non_empty)),
        }

    # ── 推荐判断 ──────────────────────────────────────────────────────────
    recommendation, reason = _recommend_field(
        coverage_rate, char_len_median, duplicate_rate, very_short_ratio, field_name
    )

    return {
        "field_name": field_name,
        "field_group": field_cfg["field_group"],
        "group_label": field_cfg["group_label"],
        "source_table": src_table,
        "source_column": col,
        "used_in_current_multimodal": field_name in CURRENT_MULTIMODAL_FIELDS,
        "total_rows": total,
        "non_empty_count": non_empty_count,
        "empty_count": empty_count,
        "coverage_rate": coverage_rate,
        "missing_rate": missing_rate,
        "unique_text_count": unique_texts,
        "duplicate_rate": duplicate_rate,
        "char_len_mean": char_len_mean,
        "char_len_median": char_len_median,
        "char_len_min": char_len_min,
        "char_len_p90": char_len_p90,
        "char_len_p95": char_len_p95,
        "char_len_max": char_len_max,
        "token_len_mean": token_len_mean,
        "very_short_ratio": very_short_ratio,
        "long_text_ratio": long_text_ratio,
        "unique_char_count": unique_chars,
        "digit_ratio": digit_ratio,
        "latin_ratio": latin_ratio,
        "chinese_char_ratio": chinese_char_ratio,
        "emoji_ratio": emoji_ratio,
        "top_chars": top_chars,
        "top_terms": top_terms,
        "split_stats": split_stats,
        "recommendation": recommendation,
        "recommendation_reason": reason,
    }


def _recommend_field(
    coverage_rate: float,
    char_len_median: float,
    duplicate_rate: float,
    very_short_ratio: float,
    field_name: str,
) -> tuple[str, str]:
    """根据统计判断字段推荐类型。"""
    reasons: list[str] = []

    if coverage_rate >= 70.0 and char_len_median >= 5:
        if field_name == "desc":
            return "use_as_primary_text", "高覆盖率(≥70%)，中位长度≥5，适合作为主文本字段"
        return "use_as_auxiliary_text", "高覆盖率(≥70%)，中位长度≥5，适合作为辅助文本字段"

    if coverage_rate >= 30.0 and char_len_median >= 3:
        if field_name == "hashtag_name":
            return "use_as_auxiliary_text", "中等覆盖率(30-70%)，话题标签明确，适合辅助文本"
        return "use_as_auxiliary_text", "中等覆盖率(30-70%)，适合辅助文本"

    if coverage_rate < 20.0:
        return "exclude", f"覆盖率({coverage_rate}%)低于20%，信息含量不足"

    if char_len_median < 3 and coverage_rate < 50:
        return "exclude_or_optional", f"中位长度({char_len_median})短且覆盖率({coverage_rate}%)低"

    if duplicate_rate > 50.0 and coverage_rate < 50:
        return "exclude_or_optional", f"重复率({duplicate_rate}%)高且覆盖率({coverage_rate}%)低"

    return "need_analysis", "需进一步分析"


def generate_examples(
    field_name: str,
    field_cfg: dict[str, Any],
    video_detail: pd.DataFrame,
    author: pd.DataFrame,
    music: pd.DataFrame,
    hashtag: pd.DataFrame,
) -> list[str]:
    """从源表提取最多 3 条非空示例文本。"""
    col = field_cfg["source_column"]
    src_table = field_cfg["source_table"]
    examples: list[str] = []

    if src_table == "raw_video_detail":
        vals = video_detail[col].dropna().astype(str).str.strip()
        vals = vals[vals.str.len() > 0].head(5)
        examples = [v[:80] for v in vals]

    elif src_table == "raw_author":
        vals = author[col].dropna().astype(str).str.strip()
        vals = vals[vals.str.len() > 0].head(5)
        examples = [v[:80] for v in vals]

    elif src_table == "raw_music":
        vals = music[col].dropna().astype(str).str.strip()
        vals = vals[vals.str.len() > 0].head(5)
        examples = [v[:80] for v in vals]

    elif src_table == "raw_hashtag":
        vals = hashtag[col].dropna().astype(str).str.strip()
        vals = vals[vals.str.len() > 0].head(5)
        examples = [v[:80] for v in vals]

    return examples[:3]


def generate_markdown_report(
    all_results: dict[str, dict[str, Any]],
    run_id: str,
) -> str:
    """生成 Markdown 分析报告。"""
    lines: list[str] = []
    lines.append(f"# Multimodal 文本字段分析报告 — real_raw_5000")
    lines.append(f"")
    lines.append(f"> Run ID: {run_id}")
    lines.append(f"> 分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 设计参考: `docs/multimodal_text_branch_improvement_design.md`")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 1. 分析目标")
    lines.append(f"")
    lines.append(f"对 Multimodal 文本分支使用的各文本字段进行覆盖率、长度分布、空值率和字符/词稀疏性分析，")
    lines.append(f"判断哪些字段适合进入分字段 TF-IDF + SVD 编码，为 Batch 14J-impl 提供数据支撑。")
    lines.append(f"")
    lines.append(f"## 2. 数据来源")
    lines.append(f"")
    lines.append(f"| 字段 | 源表 | 源列 | 当前 Multimodal | 分组 |")
    lines.append(f"|------|------|------|:---------------:|------|")
    for fn, r in all_results.items():
        used = "✅" if r["used_in_current_multimodal"] else "❌"
        lines.append(f"| {fn} | {r['source_table']} | {r['source_column']} | {used} | {r['group_label']} |")
    lines.append(f"")
    lines.append(f"## 3. 覆盖率与空值率")
    lines.append(f"")
    lines.append(f"| 字段 | 总数 | 非空 | 空值 | 覆盖率 | 缺失率 | 唯一文本数 | 重复率 |")
    lines.append(f"|------|-----:|-----:|-----:|-------:|-------:|----------:|-------:|")
    for fn, r in all_results.items():
        lines.append(
            f"| {fn} | {r['total_rows']} | {r['non_empty_count']} | {r['empty_count']} "
            f"| {r['coverage_rate']:.1f}% | {r['missing_rate']:.1f}% "
            f"| {r['unique_text_count']} | {r['duplicate_rate']:.1f}% |"
        )
    lines.append(f"")
    lines.append(f"## 4. 长度分布")
    lines.append(f"")
    lines.append(f"| 字段 | 均值 | 中位数 | P90 | P95 | 最大 | 最小 | Token均值 | 极短比 | 长文比 |")
    lines.append(f"|------|-----:|-------:|----:|----:|-----:|-----:|---------:|-------:|-------:|")
    for fn, r in all_results.items():
        lines.append(
            f"| {fn} | {r['char_len_mean']} | {r['char_len_median']} "
            f"| {r['char_len_p90']} | {r['char_len_p95']} | {r['char_len_max']} "
            f"| {r['char_len_min']} | {r['token_len_mean']} "
            f"| {r['very_short_ratio']:.1f}% | {r['long_text_ratio']:.1f}% |"
        )
    lines.append(f"")
    lines.append(f"## 5. 字符组成")
    lines.append(f"")
    lines.append(f"| 字段 | Unique Chars | 中文占比 | 拉丁占比 | 数字占比 | Emoji占比 | Top Chars |")
    lines.append(f"|------|-------------:|---------:|---------:|---------:|----------:|----------|")
    for fn, r in all_results.items():
        tc = "".join(list(r["top_chars"].keys())[:5]) if r["top_chars"] else "-"
        lines.append(
            f"| {fn} | {r['unique_char_count']} "
            f"| {r['chinese_char_ratio']:.1f}% | {r['latin_ratio']:.1f}% "
            f"| {r['digit_ratio']:.1f}% | {r['emoji_ratio']:.1f}% | {tc} |"
        )
    lines.append(f"")
    lines.append(f"## 6. Split 分布一致性")
    lines.append(f"")
    for fn, r in all_results.items():
        lines.append(f"### {fn}")
        lines.append(f"")
        lines.append(f"| Split | 总数 | 非空 | 覆盖率 | 长度均值 | 唯一文本数 |")
        lines.append(f"|-------|-----:|-----:|-------:|---------:|----------:|")
        for sp, st in r["split_stats"].items():
            lines.append(
                f"| {sp} | {st['total']} | {st['non_empty']} | {st['coverage_rate']:.1f}% "
                f"| {st['char_len_mean']} | {st['unique_texts']} |"
            )
        lines.append(f"")
    lines.append(f"## 7. 字段质量诊断")
    lines.append(f"")
    for fn, r in all_results.items():
        issues: list[str] = []
        if r["coverage_rate"] < 20:
            issues.append(f"⚠️ 覆盖率仅 {r['coverage_rate']:.1f}%（<20%），大部分样本无此字段")
        elif r["coverage_rate"] < 50:
            issues.append(f"⚠️ 覆盖率 {r['coverage_rate']:.1f}%（20-50%），约半数样本缺失")
        elif r["coverage_rate"] < 70:
            issues.append(f"ℹ️ 覆盖率 {r['coverage_rate']:.1f}%（50-70%），中等覆盖")
        if r["char_len_median"] < 3:
            issues.append(f"⚠️ 中位长度仅 {r['char_len_median']}，过短")
        if r["very_short_ratio"] > 30:
            issues.append(f"⚠️ 极短文本占比 {r['very_short_ratio']:.1f}%（>30%）")
        if r["duplicate_rate"] > 50:
            issues.append(f"⚠️ 重复率 {r['duplicate_rate']:.1f}%（>50%）")
        if r["chinese_char_ratio"] < 10:
            issues.append(f"ℹ️ 中文占比仅 {r['chinese_char_ratio']:.1f}%，非中文文本为主")
        lines.append(f"- **{fn}** ({r['group_label']}): {'; '.join(issues) if issues else '✅ 无明显质量问题'}")
    lines.append(f"")
    lines.append(f"## 8. 候选字段推荐")
    lines.append(f"")
    lines.append(f"| 字段 | 推荐 | 原因 |")
    lines.append(f"|------|------|------|")
    for fn, r in all_results.items():
        label_map = {
            "use_as_primary_text": "✅ 主文本",
            "use_as_auxiliary_text": "✅ 辅助文本",
            "exclude": "❌ 排除",
            "exclude_or_optional": "⚠️ 排除或可选",
            "need_analysis": "❓ 需进一步分析",
        }
        rec_label = label_map.get(r["recommendation"], r["recommendation"])
        lines.append(f"| {fn} | {rec_label} | {r['recommendation_reason']} |")
    lines.append(f"")
    lines.append(f"## 9. 整体判断")
    lines.append(f"")
    lines.append(f"基于上述分析，目标字段如下：")
    lines.append(f"")
    primary_fields = [fn for fn, r in all_results.items() if r["recommendation"] == "use_as_primary_text"]
    aux_fields = [fn for fn, r in all_results.items() if r["recommendation"] == "use_as_auxiliary_text"]
    exclude_fields = [fn for fn, r in all_results.items() if "exclude" in r.get("recommendation", "")]

    lines.append(f"- **主文本字段（{len(primary_fields)} 个）**：{', '.join(primary_fields) if primary_fields else '无'}")
    lines.append(f"- **辅助文本字段（{len(aux_fields)} 个）**：{', '.join(aux_fields) if aux_fields else '无'}")
    lines.append(f"- **排除或可选（{len(exclude_fields)} 个）**：{', '.join(exclude_fields) if exclude_fields else '无'}")
    lines.append(f"")
    lines.append(f"## 10. 下一步建议")
    lines.append(f"")
    lines.append(f"1. **Batch 14J-impl：实现分字段 TF-IDF + SVD 编码**")
    lines.append(f"   - 对所有推荐字段进行独立 TF-IDF + SVD 编码")
    lines.append(f"   - SVD 维度搜索（32/64/128）")
    lines.append(f"   - 验证 fieldwise 方案是否优于当前 merged 方案")
    lines.append(f"2. **字段消融实验**（量化各字段贡献）")
    lines.append(f"3. **最终只保留有效字段，简化模型**")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 输出文件")
    lines.append(f"")
    lines.append(f"| 文件 | 路径 |")
    lines.append(f"|------|------|")
    lines.append(f"| JSON 报告 | `text_field_report.json` |")
    lines.append(f"| 汇总 CSV | `text_field_summary.csv` |")
    lines.append(f"| 示例 CSV | `text_field_examples.csv` |")
    lines.append(f"| 本报告 | `text_field_report.md` |")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal 文本字段分析")
    parser.add_argument(
        "--dataset",
        type=str,
        default="real_raw_5000",
        help="数据集名称",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/analysis/text_fields",
        help="输出根目录",
    )
    args = parser.parse_args()

    dataset = args.dataset
    project_root = _PROJECT_ROOT
    output_root = project_root / args.output_root / dataset
    raw_root = project_root / "douyin_data_project/data/interim" / (
        "real_raw_5000" if dataset == "real_raw_5000" else dataset
    )
    tabular_dir = project_root / "data/features" / dataset

    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Multimodal 文本字段分析 ===")
    print(f"Dataset: {dataset}")
    print(f"Output: {out_dir}")
    print(f"")

    # ── 1. 加载数据 ──────────────────────────────────────────────────────
    print(f"[加载数据] raw 表...")
    video_detail = load_table(raw_root, "raw_video_detail_real_raw_5000.csv", "video_detail")
    author = load_table(raw_root, "raw_author_real_raw_5000.csv", "author")
    music = load_table(raw_root, "raw_music_real_raw_5000.csv", "music")
    hashtag = load_table(raw_root, "raw_hashtag_real_raw_5000.csv", "hashtag")
    comment = load_table(raw_root, "raw_comment_real_raw_5000.csv", "comment")

    print(f"[加载数据] tabular train/val/test...")
    tabular_train = pd.read_csv(
        tabular_dir / "tabular_train.csv", encoding="utf-8-sig"
    )
    tabular_val = pd.read_csv(
        tabular_dir / "tabular_val.csv", encoding="utf-8-sig"
    )
    tabular_test = pd.read_csv(
        tabular_dir / "tabular_test.csv", encoding="utf-8-sig"
    )

    # 构建 split map
    split_map: dict[int, str] = {}
    for vid in tabular_train["video_id"]:
        split_map[int(vid)] = "train"
    for vid in tabular_val["video_id"]:
        split_map[int(vid)] = "val"
    for vid in tabular_test["video_id"]:
        split_map[int(vid)] = "test"

    all_video_ids = sorted(
        set(
            list(tabular_train["video_id"].unique())
            + list(tabular_val["video_id"].unique())
            + list(tabular_test["video_id"].unique())
        )
    )
    print(f"  总 video_ids: {len(all_video_ids)} (train={len(tabular_train)}, "
          f"val={len(tabular_val)}, test={len(tabular_test)})")

    # ── 2. 分析各字段 ────────────────────────────────────────────────────
    print(f"[分析] 计算各文本字段统计...")
    all_results: dict[str, dict[str, Any]] = {}
    examples_list: list[dict[str, Any]] = []

    for field_name, field_cfg in TEXT_SOURCE_FIELDS.items():
        print(f"  - {field_name}...")
        result = compute_field_coverage(
            field_name, field_cfg,
            video_detail, author, music, hashtag, comment,
            all_video_ids, split_map,
        )
        all_results[field_name] = result

        # 收集示例
        exs = generate_examples(
            field_name, field_cfg,
            video_detail, author, music, hashtag,
        )
        for i, ex in enumerate(exs):
            examples_list.append({
                "field_name": field_name,
                "source_table": result["source_table"],
                f"example_{i+1}": ex,
            })

    # ── 3. 输出 JSON ─────────────────────────────────────────────────────
    json_path = out_dir / "text_field_report.json"
    json_output = {
        "run_id": run_id,
        "dataset": dataset,
        "analysis_type": "text_fields",
        "batch": "14J-coverage",
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fields": all_results,
        "summary": {
            "total_fields_analyzed": len(all_results),
            "recommended_for_fieldwise_encoding": [
                fn for fn, r in all_results.items()
                if r["recommendation"] in ("use_as_primary_text", "use_as_auxiliary_text")
            ],
            "not_recommended": [
                fn for fn, r in all_results.items()
                if r["recommendation"] not in ("use_as_primary_text", "use_as_auxiliary_text")
            ],
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    print(f"  [JSON] {json_path}")

    # ── 4. 输出 CSV 汇总 ─────────────────────────────────────────────────
    csv_rows: list[dict[str, Any]] = []
    for fn, r in all_results.items():
        csv_rows.append({
            "field_name": fn,
            "source_table": r["source_table"],
            "source_column": r["source_column"],
            "field_group": r["field_group"],
            "used_in_current_multimodal": r["used_in_current_multimodal"],
            "total_rows": r["total_rows"],
            "non_empty_count": r["non_empty_count"],
            "coverage_rate": r["coverage_rate"],
            "missing_rate": r["missing_rate"],
            "unique_text_count": r["unique_text_count"],
            "duplicate_rate": r["duplicate_rate"],
            "char_len_mean": r["char_len_mean"],
            "char_len_median": r["char_len_median"],
            "char_len_p90": r["char_len_p90"],
            "very_short_ratio": r["very_short_ratio"],
            "recommendation": r["recommendation"],
        })
    csv_df = pd.DataFrame(csv_rows)
    csv_path = out_dir / "text_field_summary.csv"
    csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  [CSV] {csv_path}")

    # ── 5. 输出示例 CSV ──────────────────────────────────────────────────
    ex_df = pd.DataFrame(examples_list)
    ex_path = out_dir / "text_field_examples.csv"
    ex_df.to_csv(ex_path, index=False, encoding="utf-8-sig")
    print(f"  [CSV Examples] {ex_path}")

    # ── 6. 输出 Markdown 报告 ────────────────────────────────────────────
    md_content = generate_markdown_report(all_results, run_id)
    md_path = out_dir / "text_field_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"  [MD] {md_path}")

    # ── 7. 摘要 ──────────────────────────────────────────────────────────
    print(f"")
    print(f"=== 分析完成 ===")
    print(f"输出目录: {out_dir}")
    print(f"分析字段数: {len(all_results)}")
    recommended = json_output["summary"]["recommended_for_fieldwise_encoding"]
    not_recommended = json_output["summary"]["not_recommended"]
    print(f"推荐分字段编码: {recommended}")
    print(f"不推荐: {not_recommended}")
    print(f"输出文件:")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
