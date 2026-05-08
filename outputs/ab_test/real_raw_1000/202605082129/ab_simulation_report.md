# 离线 A/B 模拟报告（real_raw_1000）

> 生成时间：2026-05-08 21:29:41

> A/B Run ID：202605082129

## 1. 模拟目标

本报告基于模型 predictions.csv 进行离线 A/B 分组和指标统计。
当前仅基于同一批预测结果做分组统计差异分析，不涉及真实线上策略差异。

- **数据集**: real_raw_1000
- **主模型**: multimodal (run_id: 202605081927)
- **Baseline 参考**: dnn (run_id: 202605081636)
- **Comparison Run ID**: 202605082029
- **样本**: 150 条 test split

> **数据说明**: real_raw_1000 真实网页端数据，1000 条视频 URL
train/val/test = 700/150/150，当前 A/B 模拟基于 test split 150 条
标签为 interaction_score P60 分位数二分类伪标签


## 2. 输入数据

- **模型**: multimodal
- **Model Run ID**: 202605081927
- **Comparison Run ID**: 202605082029
- **Predictions 路径**: outputs/multimodal/real_raw_1000/202605081927/predictions_test.csv
- **Baseline Predictions 路径**: outputs/dnn/real_raw_1000/202605081636/predictions_test.csv
- **总样本数**: 150
- **正样本数 / 负样本数**: 60 / 90
- **标签正类率**: 40.00%

## 3. 分组逻辑

- **分组方法**: hash
- **分组键**: video_id
- **实验组比例 (treatment_ratio)**: 0.5
  - hash 分组基于 group_key（video_id）的 MD5 值进行稳定分配，
    同一 video_id 多次运行分组结果一致。

## 4. 分组结果

| 组 | 角色 | 样本数 | 正样本数 | 负样本数 | 标签正类率 | 预测正类率 | 平均分 | 中位数分 |
|---|---|---|---|---|---|---|---|---|
| A | control | 74 | 26 | 48 | 35.14% | 27.03% | 0.3077 | 0.0781 |
| B | treatment | 76 | 34 | 42 | 44.74% | 42.11% | 0.4206 | 0.1699 |

> 分组均衡性: Control=74, Treatment=76, 合计=150。
> 两组样本量基本均衡。

## 5. 指标统计

### 5.1 分类指标

| 组 | AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| A | 0.9952 | 0.9189 | 1.0000 | 0.7692 | 0.8696 |
| B | 0.9622 | 0.9211 | 0.9375 | 0.8824 | 0.9091 |

### 5.2 排序指标

| 组 | Precision@5 | Precision@10 | Precision@20 | Recall@5 | Recall@10 | Recall@20 |
|--- | --- | --- | --- | --- | --- | ---|
| A | 1.0000 | 1.0000 | 1.0000 | 0.1923 | 0.3846 | 0.7692 |
| B | 1.0000 | 1.0000 | 1.0000 | 0.1471 | 0.2941 | 0.5882 |

### 5.3 分数分布

| 组 | Score Mean | Score Std | Score Median | Score Min | Score Max |
|---|---|---|---|---|---|
| A | 0.3077 | 0.3871 | 0.0781 | 0.0000 | 1.0000 |
| B | 0.4206 | 0.4270 | 0.1699 | 0.0000 | 1.0000 |

## 6. Lift 计算

以下 lift 为离线分组统计差异，不是线上因果收益。

计算公式：
- 绝对差 = treatment - control
- 相对 lift = (treatment - control) / control × 100%

| 指标 | Control | Treatment | 绝对差 | 相对 Lift |
|---|---|---|---|---|
| 平均预测分 | 0.3077 | 0.4206 | +0.1129 | +36.68% |
| 标签正类率 | 0.3514 | 0.4474 | +0.0960 | +27.33% |
| 预测正类率 | 0.2703 | 0.4211 | +0.1508 | +55.79% |

> **重要说明**: 以上 lift 只是同一模型预测结果在两组间的统计差异，
> 不反映任何策略干预的因果效应。在当前离线 A/B 模拟中，
> control 和 treatment 两组使用完全相同的模型预测结果，
> 分组仅基于 group_key 的 hash 值，不属于真实线上 A/B 测试。

## 7. Baseline 模型分数对比（参考）

以下为 DNN (baseline) 与 multimodal (主模型) 在各组的分数对比，仅供参考。

| 组 | DNN Score Mean | DNN Score Std | multimodal Score Mean | Score Delta Mean |
|---|---|---|---|---|
| A | 0.2894 | 0.4048 | 0.3077 | +0.0183 |
| B | 0.3803 | 0.4324 | 0.4206 | +0.0402 |

> **注意**: DNN 分数仅作为参考 baseline，不代表真实 control 策略。
> 两个模型在同一批数据上评分，差异反映模型行为差异而非策略收益。

## 8. 主要发现

- **A 组 (n=74)**: AUC=0.9952, Precision=1.0000, Recall=0.7692, F1=0.8696
- **B 组 (n=76)**: AUC=0.9622, Precision=0.9375, Recall=0.8824, F1=0.9091

## 9. 局限性

1. **没有真实用户曝光日志**: 当前仅基于模型预测 score，无真实曝光数据。
2. **没有真实点击/转化/完播标签**: 当前 label 为 interaction_score 分位数伪标签。
3. **没有真实 control/treatment 策略差异**: 两组使用完全相同的模型预测，无策略差异。
4. **样本量有限**: 当前基于 test split 150 条样本，A/B 分组后各组样本更少，hash 分组可能存在一定波动。
5. **lift 不是因果收益**: 所有 lift 仅为分组统计差异，不代表任何策略干预效果。
6. **离线 A/B 不等于线上 A/B**: 离线模拟没有流量干预、没有用户行为反馈、没有时间维度。
7. **样本来源**: 数据来自抖音公开网页端，不代表平台内部完整数据。

## 10. 后续真实 A/B 测试建议

1. **实验单位**: 明确以 user_id 或 device_id 为实验单位，确保同一用户始终在同一组。
2. **随机化方式**: 使用稳定的 hash 分桶（如 user_id mod 100），确保分组可复现。
3. **流量比例**: 根据实验风险确定 treatment 流量比例（如 1%/5%/10%/50%）。
4. **主要指标**: 明确核心业务指标如 CTR、CVR、完播率、人均观看时长等。
5. **护栏指标**: 设定护栏指标如 DAU、刷新频率、负反馈率等确保实验安全。
6. **样本量**: 确保足够样本量使指标达到统计显著性（建议实验前做 power analysis）。
7. **实验周期**: 保证足够的实验周期（至少 7-14 天）以覆盖周间效应。
8. **完整链路**: 记录曝光、点击、播放、完播、互动等完整用户行为链路。
9. **显著性检验**: 使用 t-test 或 bootstrap 计算置信区间和 p-value。
10. **AA 测试**: 上线前先做 AA 测试验证分组无偏性。

## 11. 结论

✅ **已跑通离线 A/B 模拟流程**: 基于 real_raw_1000 test split (150 条) 完成分组、指标统计、lift 计算。

❌ **不支持正式线上收益判断**: 当前结果基于离线预测和 hash 分组，
    不具备统计显著性，不能代表真实线上 A/B 测试结论。

## 12. 输出文件清单

- `d:\CodeData\Program Coding\ByteDance\RA_PTA\outputs\ab_test\real_raw_1000\202605082129/ab_run_meta.json`
- `d:\CodeData\Program Coding\ByteDance\RA_PTA\outputs\ab_test\real_raw_1000\202605082129/ab_group_assignment.csv`
- `d:\CodeData\Program Coding\ByteDance\RA_PTA\outputs\ab_test\real_raw_1000\202605082129/ab_metrics_summary.csv`
- `d:\CodeData\Program Coding\ByteDance\RA_PTA\outputs\ab_test\real_raw_1000\202605082129/ab_metrics_summary.json`
- `d:\CodeData\Program Coding\ByteDance\RA_PTA\outputs\ab_test\real_raw_1000\202605082129/ab_score_distribution.csv`
- `d:\CodeData\Program Coding\ByteDance\RA_PTA\outputs\ab_test\real_raw_1000\202605082129/ab_score_distribution.png`
- `d:\CodeData\Program Coding\ByteDance\RA_PTA\outputs\ab_test\real_raw_1000\202605082129/ab_simulation_report.md`
