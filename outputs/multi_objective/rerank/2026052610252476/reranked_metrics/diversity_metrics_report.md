# 多样性指标报告
**生成时间**: 2026-05-26 10:25:24 UTC
**metrics_run_id**: 2026052610252476_reranked
**模型**: dnn / real_raw_5000 / 202605132017
---
## 输入文件
- **predictions_path**: D:\CodeData\Program Coding\ByteDance\RA_PTA\outputs\dnn\real_raw_5000\202605132017\predictions_test.csv
- **features_path**: D:\CodeData\Program Coding\ByteDance\RA_PTA\data\features\real_raw_5000\tabular_test.csv
- **freq_source_path**: D:\CodeData\Program Coding\ByteDance\RA_PTA\data\features\real_raw_5000\tabular_train.csv
## Join 结果
- 预测行数: 750
- 特征行数: 750
- Join 后行数: 750
- Join 丢失率: 0.0%
## 候选集字段统计
- 总行数: 750
- 唯一 author_id: 647
- 唯一 hashtag_name_top: 316
- 唯一 region: 5
- author_id 缺失率: 0.0%
- hashtag_name_top 缺失率: 49.87%
- region 缺失率: 59.6%
## Top-K 指标表
| K | mean_score | pos_rate | prec | ndcg | div_author | cov_author | nov_author | div_hashtag | cov_hashtag | nov_hashtag | div_region | cov_region | nov_region | nov_mean | warnings |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.970325 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0155 | 1.301 | 0.9 | 0.0285 | 1.2121 | 0.1 | 0.2 | 0.1387 | 0.8839 | 0 |
| 20 | 0.940773 | 0.85 | 0.85 | 0.9965 | 1.0 | 0.0309 | 1.3452 | 0.7 | 0.0443 | 1.2564 | 0.05 | 0.2 | 0.1387 | 0.9134 | 0 |
| 50 | 0.850078 | 0.8 | 0.8 | 0.9757 | 0.96 | 0.0742 | 1.3196 | 0.76 | 0.1203 | 1.1056 | 0.02 | 0.2 | 0.1387 | 0.8546 | 0 |
## Warnings
无。
## 解读
### 多样性
`author_diversity` / `hashtag_diversity` / `region_diversity` 反映推荐列表各 K 值下实体独特性。值越接近 1.0 表示列表越多样。
### 覆盖度
`coverage_author` / `coverage_hashtag` / `coverage_region` 反映推荐列表占候选集实体比例。候选集越大，覆盖度通常越低。
### 新颖性
`novelty_mean` 反映推荐列表实体在训练集中的长尾程度。值越高，表示越倾向于推荐训练集中低频的实体。
### 相关性对照
`mean_relevance_score` / `positive_rate` / `precision` / `ndcg` 作为 relevance proxy 对照指标，辅助判断多样性提升是否过度牺牲相关性。
## 下一步建议
1. **Batch 16D**：实现多目标 reranking 模块 `src/reranking/multi_objective_rerank.py`。
2. 基于本报告的 baseline 指标与 rerank 后指标做对比。
3. 调整 alpha/beta/gamma 参数观察 trade-off。
