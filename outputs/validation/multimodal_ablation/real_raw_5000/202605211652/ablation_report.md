# Multimodal 核心模态消融实验报告

> 项目根目录：`D:/CodeData/Program Coding/ByteDance/RA_PTA/`
> 数据集：`real_raw_5000`
> 批次：Batch 14C
> 消融 Run ID：202605211652
> 执行日期：2026-05-21
> 实验设计参考：`docs/multimodal_ablation_design.md`

---

## 一、实验目的

通过逐一禁用 Multimodal 的模态分支，诊断 text / media_metadata / structured 三类模态在当前实现下的独立贡献和组合贡献，定位 Multimodal（当前 Test AUC=0.7812）弱于 DNN（Test AUC=0.8414）的主要原因。

---

## 二、实验设计

| 编号 | 变体 | enabled_modalities | 预期作用 |
|------|------|-------------------|----------|
| A | structured_only | structured | 确认结构化特征独立能力，与 DNN 对齐 |
| B | text_only | text | 确认文本分支独立预测能力 |
| C | media_only | media | 确认媒体元信息独立预测能力 |
| D | structured_text | structured, text | 确认文本对结构化的补充效果 |
| G | all_modalities | structured, text, media | 全模态融合对照（当前 baseline） |

未运行：E (structured_media)、F (text_media) — 按 Batch 14C 设计暂不执行。

---

## 三、指标汇总

| ablation_name | enabled_modalities | num_params | val_auc | val_f1 | test_auc | test_f1 | best_epoch |
|--------------|-------------------|-----------|---------|--------|----------|---------|------------|
| **structured_only** | structured | 1,121 | **0.5353** | 0.0000 | **0.5082** | 0.0000 | 7 |
| **text_only** | text | 1,105 | **0.8243** | 0.7085 | **0.7707** | 0.6607 | 46 |
| **media_only** | media | 473 | **0.7038** | 0.7151 | **0.6831** | 0.7034 | 2 |
| **structured_text** | structured, text | 2,161 | **0.8192** | 0.6787 | **0.7655** | 0.6476 | 40 |
| **all_modalities** | structured, text, media | 2,569 | **0.8340** | 0.6817 | **0.7812** | 0.6400 | 23 |

### 详细指标

| ablation_name | test_accuracy | test_precision | test_recall |
|--------------|--------------|---------------|-------------|
| structured_only | 0.6000 | 0.0000 | 0.0000 |
| text_only | 0.6973 | 0.5989 | 0.7367 |
| media_only | 0.6627 | 0.5425 | 1.0000 |
| structured_text | 0.6880 | 0.5907 | 0.7167 |
| all_modalities | 0.7000 | 0.6154 | 0.6667 |

---

## 四、各变体评估

### A — structured_only（Val AUC=0.5353, Test AUC=0.5082）

- 接近随机（AUC≈0.51），说明无 categorical embedding 的 structured 分支在当前 Multimodal 架构下无独立预测能力。
- 参数量 1,121，best_epoch=7（早停于 epoch 15）。
- **结论**：当前 33 维结构化特征（仅数值 + 文本统计，无 categorical embedding）无法独立预测 interaction_score 伪标签。
- 与 DNN（AUC=0.8414, 含 categorical embedding）的巨大差距主要由缺失类别特征 embedding 导致。

### B — text_only（Val AUC=0.8243, Test AUC=0.7707）

- 显著高于随机，是当前 Multimodal 的最强单模态。
- 参数量 1,105，与 structured_only（1,121）几乎相同但 AUC 高出约 0.29——文本特征的信息密度远超数值结构化特征。
- 训练至 epoch 46（接近满 50），val_auc 持续上升至约 0.82 后趋于平台。
- **结论**：TF-IDF+SVD 32 维文本表征具有强独立预测能力。这是当前 Multimodal 的关键信号来源。

### C — media_only（Val AUC=0.7038, Test AUC=0.6831）

- 中等独立预测能力，参数量仅 473。
- 早停于 epoch 2（验证集 loss 很快回升），说明 18 维媒体元信息信号有限。
- recall=1.0, precision≈0.54：模型倾向于将所有样本预测为正类（因为正类 300/750=40%，全部预测为正即可得到 60% accuracy）。
- **结论**：18 维媒体元信息含有限但非零的信号。但 recall=1.0 表明分类边界极弱。该结果不代表真实视觉语义特征的能力。

### D — structured_text（Val AUC=0.8192, Test AUC=0.7655）

- 与 text_only（0.8243）基本持平甚至略低——结构化分支在融合中未提供额外价值。
- 参数量 2,161（structured: 1,121 + text: 1,105 - 部分重叠），约等于两者之和。
- **结论**：当前结构化特征（无 categorical embedding）未对文本表征形成有效补充。

### G — all_modalities（Val AUC=0.8340, Test AUC=0.7812）

- 全模态融合：val_auc 最高但比 text_only 仅高出 0.0097（边际增益）。
- test_auc=0.7812，匹配历史 baseline（0.7812）。
- best_epoch=23（早停于 epoch 31），融合多模态后收敛更快。
- **结论**：全模态融合提供微小边际增益，主要贡献来自 text 分支。

---

## 五、诊断结论

### 5.1 structured_only 是否接近随机？

**是。** Val AUC=0.5353, Test AUC=0.5082，接近随机（0.5）。当前 33 维结构化特征（数值 + 文本统计，无 categorical embedding）在 Multimodal 的小容量架构下无独立预测能力。

### 5.2 text_only 是否有独立预测能力？

**有且很强。** Val AUC=0.8243，是当前 Multimodal 最强模态。当前 TF-IDF+SVD 32 维文本编码已捕获较强的预测信号。

### 5.3 media_only 是否有独立预测能力？

**有，中等。** Val AUC=0.7038，明显高于随机但 recall=1.0 表明预测几乎全部为正类，分类边界弱。参数量仅 473，信号有限。

### 5.4 structured_text 是否优于 structured_only？

**显著优于（+0.2839 val_auc）**，但增益几乎全部来自 text 分支。结构化分支未贡献额外信息。

### 5.5 all_modalities 是否优于 structured_only / structured_text？

- vs structured_only：显著优于（+0.2987 val_auc）
- vs text_only：边际优于（+0.0097 val_auc）
- vs structured_text：小幅优于（+0.0148 val_auc）

全模态融合提供边际增益，但主要贡献来自 text 分支。

### 5.6 当前 Multimodal 的主要短板

**结构化分支缺少 categorical embedding 是最主要的短板。**

诊断依据：
1. structured_only 接近随机（0.5353）→ 无 categorical embedding 的结构化分支几乎没有预测能力
2. text_only（0.8243）已接近 all_modalities（0.8340）→ 当前 Multimodal 主要依赖文本分支
3. 与 DNN（0.8414, 含 author_id/music_id embedding）的差距主要在结构化信息利用

### 5.7 下一步应优先做什么？

**Batch 14F：结构化分支加入 categorical embedding。**

理由：
1. structured_only（0.5353）与 DNN（0.8414）的差距最大（-0.3061）
2. text_only（0.8243）已相对较强，短期文本增强收益可能有限
3. 加入 embedding 后，structured 分支有望从"接近随机"提升到"有预测能力"，从而显著提升全模态融合效果

次优方向：
- Batch 14D：文本分支增强（分字段编码、提高 SVD 维度）
- 注意：不要同时做多个改动。先结构化增强（Batch 14F），再评估是否需要文本增强和融合改进。

---

## 六、对改进方案的反馈

### Batch 14D（文本分支增强）预期

text_only 已达 AUC=0.8243，文本改进（分字段编码、提高维度）可能带来进一步提升，但边际收益可能小于结构化增强。建议先做结构化增强，再用改进后的全模态 baseline 评估文本改进的增量价值。

### Batch 14E（融合策略改进）预期

all_modalities 仅比 text_only 高 +0.0097，说明当前融合策略未能有效利用多模态互补信息。但融合改进应优先在结构化增强后进行——当前 structured 分支几乎无贡献，改进融合策略也无法解决"输入信号弱"的根本问题。

### Batch 14F（结构化 categorical embedding）预期

这是当前最关键的改进方向。structured_only（0.5353）→ 预期加入 categorical embedding 后可达到或接近 DNN 水平（0.8414）。这将直接影响 Multimodal 的整体表现。

---

## 七、局限性

1. **单次消融，未多 seed 验证** — 重要结论需 3 seed 验证后才能确认。
2. **所有结论基于离线代理标签** — interaction_score P60 分位数伪标签，不是 CTR/CVR/完播/留存标签。
3. **media 为 media_metadata** — 18 维媒体元信息（封面尺寸、URL 存在性等），非真实图像/视频语义特征。
4. **media_only recall=1.0** — 该变体的 precision/F1/accuracy 不可靠，AUC 是较可靠的指标。
5. **structured_only 无 categorical embedding** — 不代表结构化分支的全部潜力。加入 embedding 后表现可能完全不同。
6. **val/test 趋势一致** — 所有变体的 val_auc 排名与 test_auc 排名一致（G > B > D > C > A），降低了单次方差风险。

---

## 八、各变体 Run ID 速查

| 变体 | Run ID | 输出目录 |
|------|--------|----------|
| structured_only (A) | 202605211639 | [outputs/multimodal/real_raw_5000/202605211639/](../../../multimodal/real_raw_5000/202605211639/) |
| text_only (B) | 202605211647 | [outputs/multimodal/real_raw_5000/202605211647/](../../../multimodal/real_raw_5000/202605211647/) |
| media_only (C) | 202605211648 | [outputs/multimodal/real_raw_5000/202605211648/](../../../multimodal/real_raw_5000/202605211648/) |
| structured_text (D) | 202605211649 | [outputs/multimodal/real_raw_5000/202605211649/](../../../multimodal/real_raw_5000/202605211649/) |
| all_modalities (G) | 202605211651 | [outputs/multimodal/real_raw_5000/202605211651/](../../../multimodal/real_raw_5000/202605211651/) |

---

## 九、汇总产物

| 文件 | 路径 |
|------|------|
| ablation_config.json | [ablation_config.json](./ablation_config.json) |
| ablation_runs_manifest.csv | [ablation_runs_manifest.csv](./ablation_runs_manifest.csv) |
| ablation_summary.csv | [ablation_summary.csv](./ablation_summary.csv) |
| ablation_report.json | [ablation_report.json](./ablation_report.json) |
| ablation_report.md | [ablation_report.md](./ablation_report.md) |
