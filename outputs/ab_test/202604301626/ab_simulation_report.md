# 离线 A/B 模拟报告（sample0427 流程验证）

> 生成时间：2026-04-30 16:26:03

> A/B Run ID：202604301626

## 1. 模拟目标

本报告演示基于模型 predictions.csv 的离线 A/B 分组和指标统计方法。
当前仅基于同一批预测结果做分组统计差异分析，不涉及真实线上策略差异。

## 2. 输入数据

- **模型**: graphsage
- **Model Run ID**: 202604291958
- **Comparison Run ID**: 202604301609
- **Predictions 路径**: outputs/graphsage/202604291958/predictions.csv
- **总样本数**: 16
- **正样本数 / 负样本数**: 6 / 10

## 3. 分组逻辑

- **分组方法**: hash
- **分组键**: video_id
- **实验组比例 (treatment_ratio)**: 0.5
- **随机种子 (random_seed)**: 2026
  - hash 分组基于 group_key（video_id）的 MD5 值进行稳定分配，
    同一 video_id 多次运行分组结果一致。

## 4. 分组结果

| 组 | 角色 | 样本数 | 正样本数 | 负样本数 | 标签正类率 | 预测正类率 |
|---|---|---|---|---|---|---|
| A | control | 8 | 2 | 6 | 25.00% | 25.00% |
| B | treatment | 8 | 4 | 4 | 50.00% | 37.50% |

## 5. 指标统计

### 5.1 分类指标

| 组 | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| A | 1.0000
| B | 0.6875

### 5.2 排序指标

| 组 | Precision@5 | Precision@10 | Recall@5 | Recall@10 |
|--- | --- | --- | --- | ---|
| A | 0.4000 | 0.2500 | 1.0000 | 1.0000 |
| B | 0.6000 | 0.5000 | 0.7500 | 1.0000 |

### 5.3 分数分布

| 组 | Score Mean | Score Std | Score Median | Score Min | Score Max |
|---|---|---|---|---|---|
| A | 0.4655 | 0.1613 | 0.4140 | 0.3052
| B | 0.5692 | 0.2320 | 0.4421 | 0.3814

## 6. Lift 计算

以下 lift 为离线分组统计差异，不是线上因果收益。

计算公式：
- 绝对差 = treatment - control
- 相对 lift = (treatment - control) / control × 100%

| 指标 | Control | Treatment | 绝对差 | 相对 Lift |
|---|---|---|---|---|
| 平均预测分 | 0.4655 | 0.5692 | +0.1038 | +22.29% |
| 标签正类率 | 0.2500 | 0.5000 | +0.2500 | +100.00% |
| 预测正类率 | 0.2500 | 0.3750 | +0.1250 | +50.00% |

> **重要说明**: 以上 lift 只是同一模型预测结果在两组间的统计差异，
> 不反映任何策略干预的因果效应。在当前离线 A/B 模拟中，
> control 和 treatment 两组使用完全相同的模型预测结果，
> 分组仅基于 group_key 的 hash 值，不属于真实线上 A/B 测试。

## 7. 局限性

1. **没有真实用户曝光日志**: 当前仅基于模型预测 score，无真实曝光数据。
2. **没有真实点击/转化/完播标签**: 当前 label 为 interaction_score 分位数伪标签。
3. **没有真实 control/treatment 策略差异**: 两组使用完全相同的模型预测，无策略差异。
4. **样本量极小**: 当前 eval 仅 16 条样本，A/B 分组后各组样本更少，统计稳定性差。
5. **当前 label 是伪标签**: 不代表真实 CTR/CVR/完播/留存等业务指标。
6. **当前结果不能代表真实线上 A/B 测试**: 分组统计差异可能完全来自随机波动。

## 8. 后续真实 A/B 测试建议

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
11. **样本污染防护**: 防止同一用户出现在不同组，避免实验组间干扰。

## 9. 结论

✅ **已跑通离线 A/B 模拟流程**: 分组逻辑、指标统计、lift 计算、报告生成均已实现。

❌ **不支持正式线上收益判断**: 当前结果基于 sample0427 样本数据（16 条 eval），
    不具备统计显著性，不能代表真实线上 A/B 测试结论。

## 10. 输出文件清单

- `D:\CodeData\Program Coding\ByteDance\RA_PTA\outputs\ab_test\202604301626/ab_group_assignment.csv`
- `D:\CodeData\Program Coding\ByteDance\RA_PTA\outputs\ab_test\202604301626/ab_metrics_summary.csv`
- `D:\CodeData\Program Coding\ByteDance\RA_PTA\outputs\ab_test\202604301626/ab_metrics_summary.json`
- `D:\CodeData\Program Coding\ByteDance\RA_PTA\outputs\ab_test\202604301626/ab_run_meta.json`
- `D:\CodeData\Program Coding\ByteDance\RA_PTA\outputs\ab_test\202604301626/ab_score_distribution.csv`
- `D:\CodeData\Program Coding\ByteDance\RA_PTA\outputs\ab_test\202604301626/ab_score_distribution.png`
- `D:\CodeData\Program Coding\ByteDance\RA_PTA\outputs\ab_test\202604301626/ab_simulation_report.md`
