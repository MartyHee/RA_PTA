# 多目标 Reranking 报告
**生成时间**: 2026-05-26 10:25:24 UTC
**rerank_run_id**: 2026052610252476
**模型**: dnn / real_raw_5000 / 202605132017
---
## 任务目标
基于 baseline 模型 score + diversity_gain + novelty_score 进行 greedy reranking，提升推荐列表多样性/覆盖度/新颖性。
## 输入文件
- **predictions_path**: D:\CodeData\Program Coding\ByteDance\RA_PTA\outputs\dnn\real_raw_5000\202605132017\predictions_test.csv
- **features_path**: D:\CodeData\Program Coding\ByteDance\RA_PTA\data\features\real_raw_5000\tabular_test.csv
- **freq_source_path**: D:\CodeData\Program Coding\ByteDance\RA_PTA\data\features\real_raw_5000\tabular_train.csv
## Reranking 配置
- **Preset**: diversity_novelty
- **alpha** (relevance): 1.0
- **beta** (diversity): 0.1
- **gamma** (novelty): 0.05
- **top_k**: 20
- **eval_k**: 10,20,50
- **diversity fields**: author_id,hashtag_name_top,region
- **author_gain**: 0.4
- **hashtag_gain**: 0.4
- **region_gain**: 0.2
## Reranking 公式
```
final_score = alpha * relevance_score + beta * diversity_gain(item|selected) + gamma * novelty_score(item)
```
## Before / After 对比表
| K | before_score | after_score | score_delta | before_pos | after_pos | before_ndcg | after_ndcg | before_div_author | after_div_author | delta_div_author | before_cov_author | after_cov_author | before_div_hashtag | after_div_hashtag | delta_div_hashtag | before_cov_hashtag | after_cov_hashtag | before_div_region | after_div_region | delta_div_region | before_cov_region | after_cov_region | before_nov_mean | after_nov_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.980826 | 0.970325 | -0.0105 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | +0.0000 | 0.0155 | 0.0155 | 0.6 | 0.9 | +0.3000 | 0.019 | 0.0285 | 0.1 | 0.1 | +0.0000 | 0.2 | 0.2 | 0.8632 | 0.8839 |
| 20 | 0.943782 | 0.940773 | -0.0030 | 0.9 | 0.85 | 0.9974 | 0.9965 | 1.0 | 1.0 | +0.0000 | 0.0309 | 0.0309 | 0.65 | 0.7 | +0.0500 | 0.0411 | 0.0443 | 0.05 | 0.05 | +0.0000 | 0.2 | 0.2 | 0.9038 | 0.9134 |
| 50 | 0.850078 | 0.850078 | +0.0000 | 0.8 | 0.8 | 0.978 | 0.9757 | 0.96 | 0.96 | +0.0000 | 0.0742 | 0.0742 | 0.76 | 0.76 | +0.0000 | 0.1203 | 0.1203 | 0.02 | 0.02 | +0.0000 | 0.2 | 0.2 | 0.8546 | 0.8546 |
## Trade-off 判断
- **Relevance 变化** (K=20): 0.9438 → 0.9408 (-0.32%)
  - ✅ mean_score 下降 <= 2%，relevance 可接受。
  - ⚠️ positive_rate 下降 -5.00%，超过 3%，需记录风险。
- author_diversity (K=20): 1.0000 → 1.0000 (变化 +0.0000)
- ✅ **hashtag_diversity** (K=20): 0.6500 → 0.7000 (提升 +0.0500)
- region_diversity (K=20): 0.0500 → 0.0500 (变化 +0.0000)
## Warnings
无。
## 下一步建议
1. **Batch 16E**：运行全部 4 组实验（baseline/light/medium/novelty），生成多目标实验报告。
2. 调整 alpha/beta/gamma 参数，观察 trade-off 曲线。
3. 如 diversity 提升有限，可尝试提高 beta 或增加 P1 字段。
