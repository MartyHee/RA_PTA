# 多模型离线对比报告（sample0427 流程验证）

> 生成时间：2026-04-30 16:09:24

## 1. 对比目标

本报告汇总 DNN、Wide & Deep、GraphSAGE、多模态四个模型在 sample0427 样本数据上的 eval 评估结果。

- 当前对比仅基于 sample0427 样本数据，仅用于流程级验证。
- 不表示正式推荐系统效果结论。
- 标签为 interaction_score 伪标签，不代表真实曝光/点击/转化目标。
- 当前无独立 test 集，eval 仅用于流程验证和最小模型评估。
- eval 只有 16 条样本，所有指标波动大，不可用于正式泛化评估。

## 2. 数据与标签说明

- **数据来源**：sample0427 样本数据（79 条主视频，11 张表）
- **标签构造**：interaction_score = digg_count + comment_count + share_count + collect_count，60% 分位数阈值构造二分类伪标签
- **标签含义**：当前标签为流程验证伪标签，不代表真实曝光、点击、完播、转化、留存等业务指标
- **数据划分**：train/eval = 80/20（seed=2026），train 63 条、eval 16 条
- **独立 test 集**：当前无独立 test 集，eval 仅用于流程验证和最小模型评估
- **样本限制**：eval 仅 16 条（正例 6、负例 10），所有指标波动极大，仅支持工程流程验证

## 3. 对比模型与 Run ID

> Comparison Run ID：202604301609

| 模型 | Run ID | 输出目录 |
|---|---|---|
| DNN | 202604301440 | outputs/dnn/202604301440 |
| Wide & Deep | 202604301557 | outputs/wide_deep/202604301557 |
| GraphSAGE | 202604291958 | outputs/graphsage/202604291958 |
| Multimodal | 202604301557 | outputs/multimodal/202604301557 |

## 4. 指标汇总

### 4.1 分类指标

| 模型 | AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| dnn | 0.9667 | 0.6875 | 0.5455 | 1.0000 | 0.7059 |
| wide_deep | 0.8000 | 0.4375 | 0.4000 | 1.0000 | 0.5714 |
| graphsage | 0.8500 | 0.8125 | 0.8000 | 0.6667 | 0.7273 |
| multimodal | 0.8500 | 0.7500 | 0.7500 | 0.5000 | 0.6000 |

### 4.2 排序指标（Precision@K / Recall@K）

| 模型 | Precision@5 | Recall@5 | Precision@10 | Recall@10 | Precision@20 | Recall@20 |
|---|---|---|---|---|---|---|
| dnn | 0.8000 | 0.6667 | 0.6000 | 1.0000 | 0.3750 | 1.0000 |
| wide_deep | 0.8000 | 0.6667 | 0.5000 | 0.8333 | 0.3750 | 1.0000 |
| graphsage | 0.8000 | 0.6667 | 0.5000 | 0.8333 | 0.3750 | 1.0000 |
| multimodal | 0.6000 | 0.5000 | 0.6000 | 1.0000 | 0.3750 | 1.0000 |

### 4.3 样本与训练信息

| 模型 | Sample Count | Positive | Negative | Eval Loss | Best Epoch | Device |
|---|---|---|---|---|---|---|
| dnn | 16 | 6 | 10 | 0.6427 | 20 | cuda |
| wide_deep | 16 | 6 | 10 | 0.6521 | 20 | cuda |
| graphsage | 16 | 6 | 10 | 0.4871 | 20 | cuda |
| multimodal | 16 | 6 | 10 | 0.6003 | 10 | cuda |

## 5. 各模型结果简析

### DNN

- 在当前流程验证集上，AUC 为 0.9667，F1 为 0.7059，Accuracy 为 0.6875。

### Wide & Deep

- 在当前流程验证集上，AUC 为 0.8000，F1 为 0.5714，Accuracy 为 0.4375。

### GRAPHSAGE

- 在当前流程验证集上，AUC 为 0.8500，F1 为 0.7273，Accuracy 为 0.8125。

### MULTIMODAL

- 在当前流程验证集上，AUC 为 0.8500，F1 为 0.6000，Accuracy 为 0.7500。

## 6. 模型优缺点对比

| 模型 | 优点 | 局限 |
|---|---|---|
| DNN | 结构简单、易跑通、适合表格特征 | 不显式建模交叉、依赖特征工程 |
| Wide & Deep | 能显式引入交叉特征 | 当前交叉特征样本少、容易偏正类 |
| GraphSAGE | 能使用 video-author / video-hashtag / related-video 图关系 | 当前图中大量节点和边为规则补齐，图关系不代表真实推荐图谱 |
| Multimodal | 能融合文本、媒体元信息、结构化特征 | 当前 visual_features 只是媒体元信息，不是真实图像语义 |

## 7. 主要限制

1. **样本量小**：79 条主视频，16 条 eval，所有指标波动大，不具备统计显著性。
2. **无独立 test 集**：当前仅使用 train/eval 切分，无法做最终泛化评估。
3. **伪标签**：标签基于 interaction_score 分位数构造，不代表 CTR/CVR/完播/留存等真实业务目标。
4. **部分字段规则生成**：5 张完全补齐表（raw_video_tag、raw_video_status_control、raw_chapter、raw_comment、raw_related_video）数据不代表真实分布。
5. **多模态视觉分支**：visual_features 仅包含媒体元信息（封面尺寸、URL 存在性等），不包含真实图像语义特征。
6. **指标不可用于正式业务结论**：当前所有对比结果仅支持流程级验证。

## 8. 后续改进假设

1. **数据规模**：接入更大规模真实数据（1000+ 条），提升指标稳定性。
2. **评估方式**：使用 train/val/test 三路切分或交叉验证，替代当前仅 train/eval 的方式。
3. **Wide & Deep 交叉特征**：加强交叉特征选择，增加有效的 wide 侧信号。
4. **GraphSAGE 图关系**：使用更真实的用户-视频、作者-视频、视频-视频关系，减少规则补齐边占比。
5. **多模态视觉**：引入真实封面图像或视频帧特征（需用户明确要求并确认依赖）。
6. **阈值选择**：统一做阈值选择与概率校准，改善 Precision/Recall 平衡。
7. **在线验证**：后续开展在线或准在线 A/B 实验，验证模型在线收益。

## 9. 结论

当前对比结果仅支持以下结论：

1. ✅ **已跑通多模型流程**：DNN、Wide & Deep、GraphSAGE、Multimodal 四类模型均已实现最小可运行闭环。
2. ✅ **已完成统一输出**：四个模型均按统一规范输出 metrics.json、predictions.csv、train_log.csv、model.pt 等文件。
3. ✅ **已完成统一对比**：对比实验入口统一，可自动汇总指标、生成对比报告和图表。
4. ❌ **不支持正式业务效果判断**：当前 sample0427 样本数据、伪标签和 16 条 eval 不足以支持任何正式推荐系统效果结论。

## 10. 图表索引

以下图表已生成至 outputs\comparison\202604301609/ 目录：

- ✅ `metric_bar_auc.png`
- ✅ `metric_bar_f1.png`
- ✅ `metric_bar_precision_recall.png`
- ✅ `model_score_distribution.png`

## 11. 分数分布概况

| 模型 | Score Min | Score Max | Score Mean | Score Std | Pred Positive Rate | Label Positive Rate |
|---|---|---|---|---|---|---|
| dnn | 0.4844 | 0.6995 | 0.5325 | 0.0561 | 68.75% | 37.50% |
| wide_deep | 0.3869 | 0.7834 | 0.5506 | 0.0806 | 93.75% | 37.50% |
| graphsage | 0.3052 | 0.9968 | 0.5174 | 0.2064 | 31.25% | 37.50% |
| multimodal | 0.2352 | 0.6589 | 0.3574 | 0.1492 | 25.00% | 37.50% |

## 12. 输出文件清单

- `outputs\comparison\202604301609/model_metrics_summary.csv`
- `outputs\comparison\202604301609/model_metrics_summary.json`
- `outputs\comparison\202604301609/model_prediction_quality_check.csv`
- `outputs\comparison\202604301609/model_prediction_quality_check.json`
- `outputs\comparison\202604301609/topk_comparison.csv`
- `outputs\comparison\202604301609/model_score_distribution.csv`
- `outputs\comparison\202604301609/model_score_distribution.png`
- `outputs\comparison\202604301609/metric_bar_auc.png`
- `outputs\comparison\202604301609/metric_bar_f1.png`
- `outputs\comparison\202604301609/metric_bar_precision_recall.png`
- `outputs\comparison\202604301609/model_comparison_report.md`
