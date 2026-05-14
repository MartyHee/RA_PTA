"""从 tabular_test.csv 抽取前 3 条样本，打印 JSON 格式（用于 API 测试）。

运行：
    D:/CodeData/software/Anaconda/Anaconda3/envs/ra/python.exe outputs/inference_api_test/request_3_items.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import Predictor

# 读取前 3 行
csv_path = Path("data/features/real_raw_5000/tabular_test.csv")
df = pd.read_csv(csv_path, encoding="utf-8-sig", skipinitialspace=True)
df3 = df.head(3)

# 构建 items（排除泄漏字段和 label/split/互动位字段）
LEAKAGE_COLS = {"digg_count", "comment_count", "share_count", "collect_count"}
exclude = LEAKAGE_COLS | {"label", "split", "interaction_score"}
cols_to_keep = [c for c in df3.columns if c not in exclude]
df3_filtered = df3[cols_to_keep]

# 替换 NaN 为 null（JSON 序列化时自动处理）
items = df3_filtered.replace({pd.NA: None, float('nan'): None}).to_dict(orient="records")

# 打印 JSON
request = {"items": items}
print(json.dumps(request, ensure_ascii=False, indent=2))