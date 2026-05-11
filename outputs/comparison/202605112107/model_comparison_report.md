# 多模型离线对比报告（real_raw_5000 真实网页端数据，no-interaction-leakage）

> 生成时间：2026-05-11 21:07:29
> Comparison Run ID：202605112107

## 1. 实验目标

汇总 DNN、Wide & Deep、GraphSAGE、Multimodal 四个模型在 real_raw_5000 真实网页端 raw 数据上的最终评估结果。

- 当前对比基于 real_raw_5000 真实网页端 raw 数据。
- 标签为 interaction_score 伪标签，不代表真实 CTR/CVR/完播/转化。
- digg_count/comment_count/share_count/collect_count 仅用于构造 label，未进入任何模型输入。
- test split 仅用于最终评估，未参与模型选择或早停。
- 所有模型结果均为离线实验对比，不代表线上推荐效果。
- 当前数据来自公开网页端，不代表平台内部完整数据。
- 历史 real_raw_1000 实验 AUC≈0.99 主要来自标签构造字段泄漏，当前 no-leakage 口径 AUC≈0.78-0.84。

## 2. 数据说明

- **数据来源**：抖音公开网页端 5 批 1000-URL 合并数据包，real_raw_5000
- **视频数**：5000 个 unique video_id
- **原始表数量**：11 张 raw 表
- **数据划分**：train 3500 / val 750 / test 750（video_id 互斥，按 label 分层）
- **标签构造**：interaction_score = digg_count + comment_count + share_count + collect_count，P60 分位数（10039.60）构造二分类伪标签
- **正负样本分布**：正例 2000，负例 3000
- **泄漏控制**：本实验使用 no_interaction_leakage 口径：digg_count/comment_count/share_count/collect_count 仅用于构造 label，未进入任何模型输入。

**TEST split 定位**：test split 仅用于最终泛化评估，未参与模型选择或早停。

**样本限制**：
- real_raw_5000 来自公开网页端，不代表平台内部完整数据。
- 当前没有真实曝光、点击、完播、转化、留存标签。
- 约 30.1% 样本为 none/low confidence，部分字段覆盖有限。
- 所有模型结果均为离线实验对比，不代表线上推荐效果。

## 3. No-Interaction-Leakage 说明

本实验采用 `no_interaction_leakage` 口径，核心约束如下：

1. **label 构造**：`interaction_score = digg_count + comment_count + share_count + collect_count`，P60 分位数二分类。
2. **泄漏控制**：上述 4 个字段仅用于构造 label，在特征构建阶段被严格排除，未进入任何模型输入。
3. **验证机制**：每个模型的构建脚本均包含泄漏检查，如发现泄漏字段进入特征列表则报错中止。
4. **必要性**：历史实验（real_raw_1000）中上述字段进入模型输入，导致 AUC≈0.99，
   高 AUC 主要来自标签构造字段泄漏，不代表模型真实推荐泛化能力。
5. **当前口径**：去除泄漏后四模型 AUC 范围 0.78-0.84，下降约 0.15-0.21，说明历史高 AUC 的确主要依赖泄漏。

## 4. 对比模型与 Run ID

| 模型 | Run ID | 输出目录 |
|---|---|---|
| dnn | 202605111858 | outputs/dnn/real_raw_5000/202605111858 |
| wide_deep | 202605111921 | outputs/wide_deep/real_raw_5000/202605111921 |
| graphsage | 202605112028 | outputs/graphsage/real_raw_5000/202605112028 |
| multimodal | 202605111759 | outputs/multimodal/real_raw_5000/202605111759 |

## 5. TEST 指标总表（主评估）

### 5.1 分类指标

| 模型 | AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| dnn | 0.8414 | 0.7533 | 0.6478 | 0.8400 | 0.7315 |
| wide_deep | 0.8242 | 0.7387 | 0.6605 | 0.7133 | 0.6859 |
| graphsage | 0.8327 | 0.7640 | 0.6742 | 0.7933 | 0.7289 |
| multimodal | 0.7812 | 0.7000 | 0.6154 | 0.6667 | 0.6400 |

### 5.2 排序指标（Precision@K / Recall@K）

| 模型 | Precision@5 | Recall@5 | Precision@10 | Recall@10 | Precision@20 | Recall@20 | Precision@50 | Recall@50 |
|---|---|---|---|---|---|---|---|---|
| dnn | 1.0000 | 0.0167 | 1.0000 | 0.0333 | 0.9000 | 0.0600 | 0.8000 | 0.1333 |
| wide_deep | 1.0000 | 0.0167 | 1.0000 | 0.0333 | 0.8500 | 0.0567 | 0.7600 | 0.1267 |
| graphsage | 1.0000 | 0.0167 | 0.9000 | 0.0300 | 0.9000 | 0.0600 | 0.7400 | 0.1233 |
| multimodal | 0.8000 | 0.0133 | 0.7000 | 0.0233 | 0.7000 | 0.0467 | 0.6600 | 0.1100 |

### 5.3 样本与训练信息

| 模型 | Sample Count | Positive | Negative | Eval Loss | Best Epoch | Num Params | Device |
|---|---|---|---|---|---|---|---|
| dnn | 750 | 300 | 450 | 0.4536 | 7 | 24,581 | cuda |
| wide_deep | 750 | 300 | 450 | 0.4929 | 11 | 25,966 | cuda |
| graphsage | 750 | 300 | 450 | 0.4621 | 53 | 13,505 | cuda |
| multimodal | 750 | 300 | 450 | 0.4960 | 23 | 2,569 | cuda |

## 6. Val 指标参考表

> 以下为各模型在 val split 上的指标，用于训练过程中的 best epoch 选择。不作为最终主评估结果。

### 6.1 分类指标（Val）

| 模型 | AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| dnn | 0.8337 | 0.7453 | 0.6394 | 0.8333 | 0.7236 |
| wide_deep | 0.8251 | 0.7373 | 0.6585 | 0.7133 | 0.6848 |
| graphsage | 0.8242 | 0.7347 | 0.6361 | 0.7867 | 0.7034 |
| multimodal | 0.8340 | 0.7373 | 0.6614 | 0.7033 | 0.6817 |

### 6.2 样本与训练信息（Val）

| 模型 | Sample Count | Positive | Negative | Eval Loss | Best Epoch | Num Params | Device |
|---|---|---|---|---|---|---|---|
| dnn | 750 | 300 | 450 | 0.4562 | 7 | 24,581 | cuda |
| wide_deep | 750 | 300 | 450 | 0.4888 | 11 | 25,966 | cuda |
| graphsage | 750 | 300 | 450 | 0.4702 | 53 | 13,505 | cuda |
| multimodal | 750 | 300 | 450 | 0.4576 | 23 | 2,569 | cuda |

## 7. Top-K 对比

基于 predictions_test.csv 重新计算 Top-K 指标（与 metrics.json 对齐检查）：

### Precision@5 / Recall@5

| 模型 | Precision@K | Recall@K | Top-K 正样本数 | 总正样本数 |
|---|---|---|---|---|
| dnn | 1.0000 | 0.0167 | 5 | 300 |
| wide_deep | 1.0000 | 0.0167 | 5 | 300 |
| graphsage | 1.0000 | 0.0167 | 5 | 300 |
| multimodal | 0.8000 | 0.0133 | 4 | 300 |

### Precision@10 / Recall@10

| 模型 | Precision@K | Recall@K | Top-K 正样本数 | 总正样本数 |
|---|---|---|---|---|
| dnn | 1.0000 | 0.0333 | 10 | 300 |
| wide_deep | 1.0000 | 0.0333 | 10 | 300 |
| graphsage | 0.9000 | 0.0300 | 9 | 300 |
| multimodal | 0.7000 | 0.0233 | 7 | 300 |

### Precision@20 / Recall@20

| 模型 | Precision@K | Recall@K | Top-K 正样本数 | 总正样本数 |
|---|---|---|---|---|
| dnn | 0.9000 | 0.0600 | 18 | 300 |
| wide_deep | 0.8500 | 0.0567 | 17 | 300 |
| graphsage | 0.9000 | 0.0600 | 18 | 300 |
| multimodal | 0.7000 | 0.0467 | 14 | 300 |

### Precision@50 / Recall@50

| 模型 | Precision@K | Recall@K | Top-K 正样本数 | 总正样本数 |
|---|---|---|---|---|
| dnn | 0.8000 | 0.1333 | 40 | 300 |
| wide_deep | 0.7600 | 0.1267 | 38 | 300 |
| graphsage | 0.7400 | 0.1233 | 37 | 300 |
| multimodal | 0.6600 | 0.1100 | 33 | 300 |

## 8. 分数分布分析

| 模型 | Min | Max | Mean | Std | Median | Avg Score (正例) | Avg Score (负例) | Pred正类率 | Label正类率 |
|---|---|---|---|---|---|---|---|---|---|
| dnn | 0.0005 | 0.9999 | 0.4080 | 0.2915 | 0.5241 | 0.6178 | 0.2681 | 51.87% | 40.00% |
| wide_deep | 0.0002 | 1.0000 | 0.3924 | 0.3123 | 0.4320 | 0.6035 | 0.2516 | 43.20% | 40.00% |
| graphsage | 0.0006 | 0.9868 | 0.4091 | 0.2842 | 0.4720 | 0.6099 | 0.2753 | 47.07% | 40.00% |
| multimodal | 0.0000 | 0.8075 | 0.3969 | 0.2592 | 0.4636 | 0.5599 | 0.2881 | 43.33% | 40.00% |

## 9. 跨模型预测一致性检查

- **检查模型数**: 4
- **Video ID 一致性**: ✅ 一致
- **Label 一致性**: ❌ 不一致
- **样本数**: 750 条
- ⚠️ 模型 multimodal 的 label 序列与 dnn 不一致: 差异 360 行

## 10. 各模型结果简析

### DNN

- **分类能力**：AUC=0.8414，F1=0.7315，Accuracy=0.7533。
- **精确率/召回率**：Precision=0.6478，Recall=0.8400。
- **损失**：test_loss=0.4536。
- **最佳 epoch**：7。

### Wide & Deep

- **分类能力**：AUC=0.8242，F1=0.6859，Accuracy=0.7387。
- **精确率/召回率**：Precision=0.6605，Recall=0.7133。
- **损失**：test_loss=0.4929。
- **最佳 epoch**：11。

### GRAPHSAGE

- **分类能力**：AUC=0.8327，F1=0.7289，Accuracy=0.7640。
- **精确率/召回率**：Precision=0.6742，Recall=0.7933。
- **损失**：test_loss=0.4621。
- **最佳 epoch**：53。

### MULTIMODAL

- **分类能力**：AUC=0.7812，F1=0.6400，Accuracy=0.7000。
- **精确率/召回率**：Precision=0.6154，Recall=0.6667。
- **损失**：test_loss=0.4960。
- **最佳 epoch**：23。

## 11. 模型优缺点对比

| 模型 | 优点 | 局限 |
|---|---|---|
| DNN | 结构简单、训练稳定、适合结构化表格特征；训练样本 3500，优于 real_raw_1000 的 700 | 需要特征工程，不能自动学习特征交叉 |
| Wide & Deep | 可显式引入交叉特征；Wide 部分可记忆稀疏模式 | 当前交叉特征在 3500 训练样本上仍稀疏，未提供额外增益；AUC 最低 0.8242 |
| GraphSAGE | 利用 video-author / video-hashtag / related-video 图拓扑信息（31998 节点）；AUC 接近 DNN | Recall 偏低；大量 related-only 节点无标签，仅作为上下文 |
| Multimodal | 融合文本/媒体元信息/结构化三模态；参数量小（2,569） | visual 分支仅用媒体元信息，非真实图像语义；AUC 最低 0.7812，未带来融合增益 |

## 12. 与历史含泄漏实验结果对比

> real_raw_1000 实验中，digg_count/comment_count/share_count/collect_count 四个标签构造字段进入了模型输入，
> 导致 AUC≈0.99。当前 real_raw_5000 no_interaction_leakage 实验已将这四个字段严格排除。

### 12.1 去泄漏前后 AUC 对比

| 模型 | real_raw_1000（含泄漏）Test AUC | real_raw_5000（去泄漏）Test AUC | 下降幅度 |
|---|---|---|---|
| DNN | ~0.99 | 0.8414 | ~0.15 |
| Wide & Deep | ~0.99 | 0.8242 | ~0.17 |
| GraphSAGE | ~0.99 | 0.8327 | ~0.16 |
| Multimodal | ~0.99 | 0.7812 | ~0.21 |

### 12.2 关键结论

1. 去除标签构造字段后，四模型 AUC 从 ~0.99 降至 0.78-0.84，下降 0.15-0.21。
2. 下降幅度意味着历史高 AUC 主要来自标签构造字段泄漏，而非模型对视频互动的真实预测能力。
3. 去泄漏后 DNN 仍保持最优（0.8414），GraphSAGE 接近 DNN（0.8327），Multimodal 下降最多（0.7812）。
4. 当前 AUC 范围 0.78-0.84 更接近无标签泄漏时的真实模型能力基线。
5. 后续所有实验必须统一使用 no_interaction_leakage 口径。

## 13. 当前限制

1. **伪标签**：标签基于 interaction_score 分位数构造，不代表 CTR/CVR/完播/转化等真实业务目标。
2. **数据源限制**：数据来自公开网页端，不代表平台内部完整推荐数据。
3. **无真实图像语义**：多模态模型的视觉分支仅使用媒体元信息（封面尺寸、URL 数量等）。
4. **GraphSAGE 图结构**：related-only 视频节点（14414 个）无标签，仅作为上下文节点参与消息传递。
5. **none/low confidence 样本**：约 30.1% 的样本字段覆盖不足（如 digg_count 等互动字段缺失）。
6. **raw_video_tag 和 raw_chapter 为空**，无法用于任何模型。
7. **评估稳定性**：test split 750 条（300 正 / 450 负），评估结果有一定方差。
8. **当前所有结果仅为离线实验对比，不代表线上推荐效果或业务收益。**

## 14. 下一步建议

1. **多 seed 稳定性验证**：使用 3-5 个 random seed 重复实验，确认当前 AUC 排序稳定性。
2. **真实标签**：接入真实曝光、点击、完播等标签替代目前 interaction_score 伪标签。
3. **超参数调优**：增大 epochs、调整 learning rate、尝试不同 fusion 策略或 attention 聚合器。
4. **高级 fusion**：Multimodal 可尝试 attention-based fusion 替代简单拼接。
5. **图增强**：GraphSAGE 可尝试 GAT 替代 mean aggregator，或增加 comment_user 节点。
6. **视觉增强**：如用户明确要求，引入封面图像特征（需确认 CLIP/ResNet 依赖）。
7. **校准与阈值优化**：统一做概率校准，优化 Precision/Recall 平衡。
8. **端到端流水线工程化**：在 no-leakage 口径确认后，进入推荐流水线工程化。

## 15. 图表索引

以下图表已生成至 outputs\comparison\202605112107/ 目录：

- ✅ `metric_bar_auc.png`
- ✅ `metric_bar_f1.png`
- ✅ `metric_bar_precision_recall.png`
- ✅ `model_score_distribution.png`

## 16. 输出文件清单

- `outputs\comparison\202605112107/comparison_run_meta.json`
- `outputs\comparison\202605112107/cross_model_consistency_check.json`
- `outputs\comparison\202605112107/model_metrics_summary.csv`
- `outputs\comparison\202605112107/model_metrics_summary.json`
- `outputs\comparison\202605112107/val_metrics_summary.csv`
- `outputs\comparison\202605112107/model_prediction_quality_check.csv`
- `outputs\comparison\202605112107/model_prediction_quality_check.json`
- `outputs\comparison\202605112107/topk_comparison.csv`
- `outputs\comparison\202605112107/model_score_distribution.csv`
- `outputs\comparison\202605112107/model_score_distribution.png`
- `outputs\comparison\202605112107/metric_bar_auc.png`
- `outputs\comparison\202605112107/metric_bar_f1.png`
- `outputs\comparison\202605112107/metric_bar_precision_recall.png`
- `outputs\comparison\202605112107/model_comparison_report.md`
