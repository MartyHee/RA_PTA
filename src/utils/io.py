"""文件 I/O 工具函数"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


ENCODING_PRIORITY = ["utf-8-sig", "utf-8", "gbk"]


def read_csv_safe(file_path: str | Path, **kwargs: Any) -> tuple[pd.DataFrame, str]:
    """尝试以多种编码读取 CSV 文件，返回 (DataFrame, 实际使用的编码)。

    编码尝试优先级：utf-8-sig → utf-8 → gbk。
    首次成功后即返回，不再尝试后续编码。

    Args:
        file_path: CSV 文件路径
        **kwargs: 传递给 pd.read_csv 的额外参数

    Returns:
        (DataFrame, encoding_used)

    Raises:
        ValueError: 所有编码尝试均失败
    """
    errors = []
    for enc in ENCODING_PRIORITY:
        try:
            df = pd.read_csv(file_path, encoding=enc, **kwargs)
            return df, enc
        except (UnicodeDecodeError, UnicodeError) as e:
            errors.append(f"{enc}: {e}")
            continue
    raise ValueError(
        f"无法读取文件 {file_path}，已尝试编码 {ENCODING_PRIORITY}。\n"
        f"详细错误:\n" + "\n".join(errors)
    )