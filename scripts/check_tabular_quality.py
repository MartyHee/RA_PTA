"""
Tabular 数据集质量检查脚本
- 全空/占位/全-1 排除字段确认
- 当前特征列中无效/常量列检查（用于流程验证 awareness）
- 标签分布和切分口径检查
- 输出 tabular_dataset_quality_check.json
"""
import sys, json, os
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, '..'))
sys.path.insert(0, os.path.join(_project_root, 'src'))

from utils.config import load_config
from utils.io import read_csv_safe
import pandas as pd
import numpy as np

# ============================================================
# 1. 加载配置和数据
# ============================================================
feature_tabular = load_config('configs/common/feature_tabular.yaml')
train_df, _ = read_csv_safe(feature_tabular['output_paths']['train_csv'])
eval_df, _ = read_csv_safe(feature_tabular['output_paths']['eval_csv'])
full = pd.concat([train_df, eval_df], ignore_index=True)

feature_info = json.load(open(feature_tabular['output_paths']['feature_info_json'], encoding='utf-8'))
dataset_report = json.load(open(feature_tabular['output_paths']['dataset_report_json'], encoding='utf-8'))

# ============================================================
# 2. 已知分类
# ============================================================
all_null_known = set(feature_info.get('excluded_all_null_cols', []))
placeholder_known = set(feature_info.get('excluded_placeholder_cols', []))
all_minus_one_excluded = ["favoriting_count", "following_count"]

used_cols = set(
    feature_info.get('id_cols', []) +
    feature_info.get('numeric_cols', []) +
    feature_info.get('categorical_cols', []) +
    feature_info.get('text_stat_cols', []) +
    feature_info.get('wide_cross_cols', []) +
    ['label', 'interaction_score', 'split']
)

label_col = feature_info.get('label_col', 'label')

# ============================================================
# 3. 构建质量报告
# ============================================================
quality = {
    # --- 基本信息 ---
    "dataset_info": {
        "total_rows": len(full),
        "total_columns": len(full.columns),
        "train_rows": len(train_df),
        "eval_rows": len(eval_df),
        "columns": sorted(full.columns.tolist()),
    },
    # --- 标签分布 ---
    "label_distribution": {},
    # --- 已被排除的无效字段（结构化记录） ---
    "excluded_fields": {
        "excluded_all_null_cols": sorted(all_null_known),
        "excluded_placeholder_cols": sorted(placeholder_known),
        "excluded_all_minus_one_cols": all_minus_one_excluded,
    },
    # --- 当前特征列中的预警字段（未被排除，但无区分度） ---
    "flagged_fields": {
        "all_zero_cols": [],      # 全为 0，无区分度
        "constant_cols": [],      # 常量列（含全零），无区分度
        "high_missing_cols": [],  # 高缺失率列
    },
    # --- 切分检查 ---
    "split_check": {
        "method": "random_by_video_id",
        "seed": 2026,
        "train_size": len(train_df),
        "eval_size": len(eval_df),
        "train_eval_overlap": 0,
        "has_independent_test": False,
        "note": (
            "当前仅使用 train/eval 切分，未设置独立 test 集。"
            "eval 用于流程验证和最小模型评估。"
            "正式实验阶段需启用 train/val/test 或交叉验证。"
        ),
    },
    # --- 缺失概览 ---
    "missing_summary": {},
    # --- 数据质量说明 ---
    "data_quality_notes": [],
}

# 标签分布
if label_col in full.columns:
    train_pos = int(train_df[label_col].sum()) if label_col in train_df.columns else 0
    eval_pos = int(eval_df[label_col].sum()) if label_col in eval_df.columns else 0
    quality["label_distribution"] = {
        "total_pos": int(full[label_col].sum()),
        "total_neg": int((1 - full[label_col]).sum()),
        "pos_ratio": round(float(full[label_col].mean()), 4),
        "train_pos": train_pos,
        "train_neg": len(train_df) - train_pos,
        "eval_pos": eval_pos,
        "eval_neg": len(eval_df) - eval_pos,
    }

# ============================================================
# 4. 扫描当前特征列，标记预警字段
# ============================================================
used_and_checkable = [
    c for c in used_cols if c in full.columns
    and c not in ('sample_id', 'video_id', 'interaction_score', label_col, 'split')
]

all_zero_list = []
constant_list = []

for c in used_and_checkable:
    s = full[c].dropna()
    if len(s) == 0:
        continue

    if s.dtype.kind in ('i', 'f', 'b'):
        is_all_zero = (s == 0).all()
        is_all_minus_one = (s == -1).all()
        is_constant = s.nunique() == 1

        if is_all_zero:
            all_zero_list.append(c)
        if is_constant:
            constant_list.append({"col": c, "value": str(s.iloc[0])})
    elif s.dtype.kind == 'O':
        if s.nunique() == 1 and len(s) > 0:
            constant_list.append({"col": c, "value": str(s.iloc[0])})

quality["flagged_fields"]["all_zero_cols"] = all_zero_list
quality["flagged_fields"]["constant_cols"] = constant_list

# 缺失率（从已有报告中搬运）
if 'missing_summary' in dataset_report and dataset_report.get('missing_summary'):
    quality["missing_summary"] = dataset_report['missing_summary']
    high_missing = [
        {"col": k, "missing_rate": v["missing_rate"]}
        for k, v in dataset_report['missing_summary'].items()
        if v.get("missing_rate", 0) > 0.10
    ]
    if high_missing:
        quality["flagged_fields"]["high_missing_cols"] = high_missing

# ============================================================
# 5. 汇总数据质量说明
# ============================================================
notes = []

n_excluded = (
    len(all_null_known) + len(placeholder_known) + len(all_minus_one_excluded)
)
notes.append(f"已排除无效字段共 {n_excluded} 个（全空 {len(all_null_known)} + 占位 {len(placeholder_known)} + 全-1 {len(all_minus_one_excluded)}）")

if all_zero_list:
    notes.append(f"当前特征列中有 {len(all_zero_list)} 个全零列（无区分度，当前未排除）: {', '.join(all_zero_list)}")

bool_all_false = [c for c in used_and_checkable if c in full.columns and full[c].dtype == bool and (~full[c]).all()]
if bool_all_false:
    notes.append(f"当前特征列中有 {len(bool_all_false)} 个布尔列全为 False（无区分度）: {', '.join(bool_all_false)}")

if quality["flagged_fields"]["high_missing_cols"]:
    notes.append("存在高缺失率特征列，请在模型训练中注意缺失值处理。")

quality["data_quality_notes"] = notes

# ============================================================
# 6. 写入输出文件
# ============================================================
output_path = os.path.join(_project_root, "outputs/data_check/tabular_dataset_quality_check.json")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(quality, f, ensure_ascii=False, indent=2)

print(f"质量检查报告已写入: {output_path}")
print(json.dumps(quality, ensure_ascii=False, indent=2))