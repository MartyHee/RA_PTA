# 多模型离线对比报告（real_raw_1000 真实网页端数据）

> 生成时间：2026-05-08 20:29:29
> Comparison Run ID：202605082029

## 1. 实验目标

汇总 DNN、Wide & Deep、GraphSAGE、Multimodal 四个模型在 real_raw_1000 真实网页端 raw 数据上的最终评估结果。

- 当前对比基于 real_raw_1000 真实网页端 raw 数据。
- 标签为 interaction_score 伪标签，不代表真实 CTR/CVR/完播/转化。
- test split 仅用于最终评估，未参与模型选择或早停。
- 所有模型结果均为离线实验对比，不代表线上推荐效果。
- 当前数据来自公开网页端，不代表平台内部完整数据。
- sample0427 是历史流程验证数据，不是当前主实验数据。

## 2. 数据说明

- **数据来源**：抖音公开网页端 20260507_230322 数据包，1000 条视频 URL
- **视频数**：1000 个 unique video_id
- **原始表数量**：11 张 raw 表
- **数据划分**：train 700 / val 150 / test 150（video_id 互斥，按 label 分层）
- **标签构造**：interaction_score = digg_count + comment_count + share_count + collect_count，P60 分位数（18147.80）构造二分类伪标签
- **正负样本分布**：正例 400，负例 600

**TEST split 定位**：test split 仅用于最终泛化评估，未参与模型选择或早停。

**样本限制**：
- real_raw_1000 来自公开网页端，不代表平台内部完整数据。
- 当前没有真实曝光、点击、完播、转化、留存标签。
- none-match 样本占 21.2%，部分字段覆盖有限。
- 所有模型结果均为离线实验对比，不代表线上推荐效果。

## 3. 对比模型与 Run ID

| 模型 | Run ID | 输出目录 |
|---|---|---|
| dnn | 202605081636 | outputs/dnn/real_raw_1000/202605081636 |
| wide_deep | 202605081746 | outputs/wide_deep/real_raw_1000/202605081746 |
| graphsage | 202605081828 | outputs/graphsage/real_raw_1000/202605081828 |
| multimodal | 202605081927 | outputs/multimodal/real_raw_1000/202605081927 |

## 4. TEST 指标总表（主评估）

### 4.1 分类指标

| 模型 | AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| dnn | 0.9717 | 0.9000 | 0.9592 | 0.7833 | 0.8624 |
| wide_deep | 0.9511 | 0.8867 | 0.9388 | 0.7667 | 0.8440 |
| graphsage | 0.9611 | 0.8800 | 1.0000 | 0.7000 | 0.8235 |
| multimodal | 0.9746 | 0.9200 | 0.9615 | 0.8333 | 0.8929 |

### 4.2 排序指标（Precision@K / Recall@K）

| 模型 | Precision@5 | Recall@5 | Precision@10 | Recall@10 | Precision@20 | Recall@20 |
|---|---|---|---|---|---|---|
| dnn | 1.0000 | 0.0833 | 1.0000 | 0.1667 | 1.0000 | 0.3333 |
| wide_deep | 1.0000 | 0.0833 | 1.0000 | 0.1667 | 1.0000 | 0.3333 |
| graphsage | 1.0000 | 0.0833 | 1.0000 | 0.1667 | 1.0000 | 0.3333 |
| multimodal | 1.0000 | 0.0833 | 1.0000 | 0.1667 | 1.0000 | 0.3333 |

### 4.3 样本与训练信息

| 模型 | Sample Count | Positive | Negative | Eval Loss | Best Epoch | Num Params | Device |
|---|---|---|---|---|---|---|---|
| dnn | 150 | 60 | 90 | 0.2361 | 33 | 41,177 | cuda |
| wide_deep | 150 | 60 | 90 | 0.2788 | 22 | 42,760 | cuda |
| graphsage | 150 | 60 | 90 | 0.2928 | 80 | 10,433 | cuda |
| multimodal | 150 | 60 | 90 | 0.1950 | 50 | 7,857 | cuda |

## 5. Val 指标参考表

> 以下为各模型在 val split 上的指标，用于训练过程中的 best epoch 选择。不作为最终主评估结果。

### 5.1 分类指标（Val）

| 模型 | AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| dnn | 0.9889 | 0.9133 | 0.9796 | 0.8000 | 0.8807 |
| wide_deep | 0.9772 | 0.9133 | 0.9796 | 0.8000 | 0.8807 |
| graphsage | 0.9711 | 0.8733 | 0.9556 | 0.7167 | 0.8190 |
| multimodal | 0.9894 | 0.9333 | 1.0000 | 0.8333 | 0.9091 |

### 5.2 样本与训练信息（Val）

| 模型 | Sample Count | Positive | Negative | Eval Loss | Best Epoch | Num Params | Device |
|---|---|---|---|---|---|---|---|
| dnn | 150 | 60 | 90 | 0.1747 | 33 | 41,177 | cuda |
| wide_deep | 150 | 60 | 90 | 0.2130 | 22 | 42,760 | cuda |
| graphsage | 150 | 60 | 90 | 0.2728 | 80 | 10,433 | cuda |
| multimodal | 150 | 60 | 90 | 0.1543 | 50 | 7,857 | cuda |

## 6. Top-K 对比

基于 predictions_test.csv 重新计算 Top-K 指标（与 metrics.json 对齐检查）：

### Precision@5 / Recall@5

| 模型 | Precision@K | Recall@K | Top-K 正样本数 | 总正样本数 |
|---|---|---|---|---|
| dnn | 1.0000 | 0.0833 | 5 | 60 |
| wide_deep | 1.0000 | 0.0833 | 5 | 60 |
| graphsage | 1.0000 | 0.0833 | 5 | 60 |
| multimodal | 1.0000 | 0.0833 | 5 | 60 |

### Precision@10 / Recall@10

| 模型 | Precision@K | Recall@K | Top-K 正样本数 | 总正样本数 |
|---|---|---|---|---|
| dnn | 1.0000 | 0.1667 | 10 | 60 |
| wide_deep | 1.0000 | 0.1667 | 10 | 60 |
| graphsage | 1.0000 | 0.1667 | 10 | 60 |
| multimodal | 1.0000 | 0.1667 | 10 | 60 |

### Precision@20 / Recall@20

| 模型 | Precision@K | Recall@K | Top-K 正样本数 | 总正样本数 |
|---|---|---|---|---|
| dnn | 1.0000 | 0.3333 | 20 | 60 |
| wide_deep | 1.0000 | 0.3333 | 20 | 60 |
| graphsage | 1.0000 | 0.3333 | 20 | 60 |
| multimodal | 1.0000 | 0.3333 | 20 | 60 |

## 7. 分数分布分析

| 模型 | Min | Max | Mean | Std | Median | Avg Score (正例) | Avg Score (负例) | Pred正类率 | Label正类率 |
|---|---|---|---|---|---|---|---|---|---|
| dnn | 0.0000 | 1.0000 | 0.3355 | 0.4215 | 0.0586 | 0.7714 | 0.0448 | 32.67% | 40.00% |
| wide_deep | 0.0000 | 1.0000 | 0.3566 | 0.4113 | 0.0978 | 0.7595 | 0.0880 | 32.67% | 40.00% |
| graphsage | 0.0001 | 1.0000 | 0.3629 | 0.3406 | 0.2460 | 0.6934 | 0.1425 | 28.00% | 40.00% |
| multimodal | 0.0000 | 1.0000 | 0.3649 | 0.4117 | 0.1001 | 0.8072 | 0.0700 | 34.67% | 40.00% |

## 8. 跨模型预测一致性检查

- **检查模型数**: 4
- **Video ID 一致性**: ✅ 一致
- **Label 一致性**: ✅ 一致
- **样本数**: 150 条

## 9. 各模型结果简析

### DNN

- **分类能力**：AUC=0.9717，F1=0.8624，Accuracy=0.9000。
- **精确率/召回率**：Precision=0.9592，Recall=0.7833。
- **损失**：test_loss=0.2361。
- **最佳 epoch**：33。

### Wide & Deep

- **分类能力**：AUC=0.9511，F1=0.8440，Accuracy=0.8867。
- **精确率/召回率**：Precision=0.9388，Recall=0.7667。
- **损失**：test_loss=0.2788。
- **最佳 epoch**：22。

### GRAPHSAGE

- **分类能力**：AUC=0.9611，F1=0.8235，Accuracy=0.8800。
- **精确率/召回率**：Precision=1.0000，Recall=0.7000。
- **损失**：test_loss=0.2928。
- **最佳 epoch**：80。

### MULTIMODAL

- **分类能力**：AUC=0.9746，F1=0.8929，Accuracy=0.9200。
- **精确率/召回率**：Precision=0.9615，Recall=0.8333。
- **损失**：test_loss=0.1950。
- **最佳 epoch**：50。

## 10. 模型优缺点对比

| 模型 | 优点 | 局限 |
|---|---|---|
| DNN | 结构简单、训练稳定、适合结构化表格特征 | 需要特征工程，不能自动学习特征交叉 |
| Wide & Deep | 可显式引入交叉特征 | 当前交叉特征在 700 条训练样本上稀疏度高，未提供额外增益 |
| GraphSAGE | 利用 video-author / video-hashtag / related-video 图拓扑信息 | 7257 个无标签节点；Recall 偏低（保守预测）|
| Multimodal | 融合文本/媒体元信息/结构化三模态；参数量小（7,857） | visual 分支仅用媒体元信息，非真实图像语义 |

## 11. 当前限制

1. **样本量有限**：1000 条视频，val/test 各 150 条，评估稳定性有限。
2. **伪标签**：标签基于 interaction_score 分位数构造，不代表 CTR/CVR/完播/转化等真实业务目标。
3. **数据源限制**：真实网页端数据不代表平台内部完整数据。
4. **无真实图像语义**：多模态模型的视觉分支仅使用媒体元信息（封面尺寸、URL 数量等）。
5. **GraphSAGE 图结构**：related-only 视频节点无标签，仅作为上下文节点。
6. **none-match 样本**：21.2% 的样本为 none/low confidence，部分字段覆盖不足。
7. **raw_video_tag 和 raw_chapter 为空**，无法用于任何模型。
8. **当前所有结果仅为离线实验对比，不代表线上推荐效果或业务收益。**

## 12. 下一步建议

1. **增大数据量**：继续采集 URL（当前仅 1000 条），提升模型泛化能力。
2. **真实标签**：接入真实曝光、点击、完播等标签替代目前 interaction_score 伪标签。
3. **超参数调优**：增大 epochs、调整 learning rate、尝试不同 fusion 策略或 attention 聚合器。
4. **高级 fusion**：Multimodal 可尝试 attention-based fusion 替代简单拼接。
5. **图增强**：GraphSAGE 可尝试 GAT 替代 mean aggregator，或增加 comment_user 节点。
6. **视觉增强**：如用户明确要求，引入封面图像特征（需确认 CLIP/ResNet 依赖）。
7. **校准与阈值优化**：统一做概率校准，优化 Precision/Recall 平衡。
8. **离线 A/B 模拟**：对各模型预测结果做离线分组统计，比较模型间差异。

## 13. 图表索引

以下图表已生成至 outputs\comparison\202605082029/ 目录：

- ✅ `metric_bar_auc.png`
- ✅ `metric_bar_f1.png`
- ✅ `metric_bar_precision_recall.png`
- ✅ `model_score_distribution.png`

## 14. 输出文件清单

- `outputs\comparison\202605082029/comparison_run_meta.json`
- `outputs\comparison\202605082029/cross_model_consistency_check.json`
- `outputs\comparison\202605082029/model_metrics_summary.csv`
- `outputs\comparison\202605082029/model_metrics_summary.json`
- `outputs\comparison\202605082029/val_metrics_summary.csv`
- `outputs\comparison\202605082029/model_prediction_quality_check.csv`
- `outputs\comparison\202605082029/model_prediction_quality_check.json`
- `outputs\comparison\202605082029/topk_comparison.csv`
- `outputs\comparison\202605082029/model_score_distribution.csv`
- `outputs\comparison\202605082029/model_score_distribution.png`
- `outputs\comparison\202605082029/metric_bar_auc.png`
- `outputs\comparison\202605082029/metric_bar_f1.png`
- `outputs\comparison\202605082029/metric_bar_precision_recall.png`
- `outputs\comparison\202605082029/model_comparison_report.md`
