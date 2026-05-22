# ID Coverage Analysis Report — real_raw_5000

> Batch: 14I-coverage
> Run ID: 20260521203608
> Analysis Date: 2026-05-21 20:36:08
> Design Reference: `docs/multimodal_high_cardinality_id_embedding_design.md`

---

## 1. 分析目标

对 `author_id` 和 `music_id` 在 `real_raw_5000` 的 train/val/test 中的覆盖率、OOV 率和长尾分布进行分析，
判断是否具备进入 Multimodal categorical embedding 实现的基础条件。

## 2. 数据来源

| 字段 | 来源 | 说明 |
|------|------|------|
| author_id | `data/features/real_raw_5000/tabular_*.csv` | tabular CSV 内置列 |
| music_id | `douyin_data_project/data/interim/real_raw_5000/raw_music_real_raw_5000.csv` | 按 video_id join，不在 tabular CSV 中 |

## 3. author_id 分析

### 3.1 基础统计

| Split | 样本数 | Unique ID | 缺失数 | 缺失率 |
|-------|--------|-----------|--------|--------|
| train | 3500 | 2595 | 0 | 0.0% |
| val | 750 | 650 | 0 | 0.0% |
| test | 750 | 647 | 0 | 0.0% |

### 3.2 OOV 分析

| 指标 | 值 |
|------|-----|
| val OOV count | 490 |
| val OOV rate | 75.38% |
| test OOV count | 489 |
| test OOV rate | 75.58% |
| val+test OOV rate | 78.74% |

### 3.3 频次分布 (train)

| 指标 | 值 |
|------|-----|
| 平均样本数/ID | 1.35 |
| 中位数样本数/ID | 1.0 |
| 最小样本数/ID | 1 |
| 最大样本数/ID | 32 |
| 标准差 | 1.80 |
| singleton ID 数 | 2305 |
| singleton ID 比例 | 88.82% |
| freq >= 2 比例 | 11.18% |
| freq >= 3 比例 | 4.86% |
| freq >= 5 比例 | 2.04% |
| freq >= 10 比例 | 1.16% |

### 3.4 Top-K 覆盖率 (train)

| K | 覆盖样本比例 |
|---|-------------|
| 10 | 6.29% |
| 50 | 17.49% |
| 100 | 22.54% |
| 200 | 29.00% |
| 500 | 40.14% |
| 1000 | 54.43% |

### 3.5 决策: FAIL

FAIL: OOV=78.74%, avg_samples=1.35, singleton=88.82%, missing=0.0%

## 4. music_id 分析

### 4.1 基础统计

| Split | 样本数 | Unique ID | 缺失数 | 缺失率 |
|-------|--------|-----------|--------|--------|
| train | 2549 | 2538 | 27% of 3500 | 27.2% |
| val | 538 | 538 | 28% of 750 | 28.3% |
| test | 552 | 550 | 26% of 750 | 26.4% |

### 4.2 OOV 分析

| 指标 | 值 |
|------|-----|
| val OOV count | 534 |
| val OOV rate | 99.26% |
| test OOV count | 548 |
| test OOV rate | 99.64% |
| val+test OOV rate | 99.54% |

### 4.3 频次分布 (train)

| 指标 | 值 |
|------|-----|
| 平均样本数/ID | 1.0 |
| 中位数样本数/ID | 1.0 |
| 最小样本数/ID | 1 |
| 最大样本数/ID | 3 |
| 标准差 | 0.08 |
| singleton ID 数 | 2530 |
| singleton ID 比例 | 99.68% |
| freq >= 2 比例 | 0.32% |
| freq >= 3 比例 | 0.12% |
| freq >= 5 比例 | 0.00% |
| freq >= 10 比例 | 0.00% |

### 4.4 Top-K 覆盖率 (train, 仅有效 music_id)

| K | 覆盖样本比例 |
|---|-------------|
| 10 | 0.82% |
| 50 | 2.39% |
| 100 | 4.35% |
| 200 | 8.28% |
| 500 | 20.05% |

### 4.5 决策: FAIL

FAIL: OOV=99.54%, avg_samples=1.0, singleton=99.68%, missing=27.171428571428574%

---

## 5. 总体判断

| 条件 | 阈值 | author_id | music_id |
|------|------|-----------|----------|
| OOV rate | < 20% | 78.74% | 99.54% |
| avg samples/ID | >= 2.0 | 1.35 | 1.0 |
| singleton ratio | < 50% | 88.82% | 99.68% |
| missing rate | < 5% | 0.00% | 27.17% |

**author_id 决策: FAIL**

**music_id 决策: FAIL**

**总体建议: do_not_implement_id_embedding**

### 5.1 原因分析

**author_id 问题:**
- 平均样本数/ID = 1.35 < 2.0，大量 ID 仅出现 1 次
- singleton ID 占比 = 88.82% >= 50%，长尾极长
- OOV 率 = 78.74% >= 20%，val/test 中过多未见 ID

**music_id 问题:**
- 缺失率 = 27.171428571428574% >= 5%，大量样本无 music_id
- 平均样本数/ID = 1.0 < 2.0
- singleton ID 占比 = 99.68% >= 50%
- OOV 率 = 99.54% >= 20%

### 5.2 推荐结论

**不建议实现 ID embedding。** author_id 和 music_id 均不满足基本条件。

**建议转向替代方向:**
1. **Batch 14J: 文本分支增强** — per-field TF-IDF/SVD, SVD 维度 32→64/128
2. **融合策略优化** — attention-based fusion, late fusion

---

## 6. 输出文件

| 文件 | 路径 |
|------|------|
| JSON 报告 | `id_coverage_report.json` |
| 汇总 CSV | `id_coverage_summary.csv` |
| 频次 Top CSV | `id_frequency_top.csv` |
| 本报告 | `id_coverage_report.md` |
