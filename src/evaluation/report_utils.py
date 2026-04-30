"""报告拼装工具函数"""

from __future__ import annotations

from typing import Any


def fmt_metric(val: Any, decimals: int = 4) -> str:
    """格式化指标值为字符串，None 显示为 N/A。"""
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def fmt_pct(val: Any, decimals: int = 2) -> str:
    """格式化比例为百分比字符串。"""
    if val is None:
        return "N/A"
    try:
        return f"{float(val) * 100:.{decimals}f}%"
    except (ValueError, TypeError):
        return str(val)


def make_table_row(values: list[str], widths: list[int]) -> str:
    """生成等宽 markdown 表格行。"""
    parts = []
    for v, w in zip(values, widths):
        parts.append(f" {v:<{w}} ")
    return "|" + "|".join(parts) + "|"


def make_separator(widths: list[int]) -> str:
    """生成 markdown 表格分隔行。"""
    parts = []
    for w in widths:
        parts.append(f" {'-' * w} ")
    return "|" + "|".join(parts) + "|"


def metric_table(
    rows: list[dict[str, Any]],
    columns: list[str],
    col_headers: dict[str, str] | None = None,
    fmt_overrides: dict[str, str] | None = None,
) -> str:
    """将指标字典列表渲染为 markdown 表格。

    Args:
        rows: 指标字典列表，每个字典对应一行。
        columns: 要显示的列名列表。
        col_headers: 列显示名称覆盖。
        fmt_overrides: 列格式化方式覆盖，'pct'=百分比，'raw'=原样，默认=4位小数。
    """
    if not rows:
        return "_无数据_"

    headers = [col_headers.get(c, c) if col_headers else c for c in columns]
    widths = [len(h) for h in headers]

    formatted: list[list[str]] = []
    for row in rows:
        vals: list[str] = []
        for c in columns:
            raw = row.get(c)
            if fmt_overrides and c in fmt_overrides:
                if fmt_overrides[c] == "pct":
                    vals.append(fmt_pct(raw))
                elif fmt_overrides[c] == "raw":
                    vals.append(str(raw) if raw is not None else "N/A")
                else:
                    vals.append(fmt_metric(raw))
            else:
                vals.append(fmt_metric(raw))
        widths = [max(w, len(v)) for w, v in zip(widths, vals)]
        formatted.append(vals)

    lines = [make_table_row(headers, widths), make_separator(widths)]
    for vals in formatted:
        lines.append(make_table_row(vals, widths))

    return "\n".join(lines)