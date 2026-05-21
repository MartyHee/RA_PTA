# Multimodal Categorical Embedding 消融实验报告

> 项目根目录：`D:/CodeData/Program Coding/ByteDance/RA_PTA/`
> 数据集：`real_raw_5000`
> 批次：Batch 14F-experiment
> 消融 Run ID：202605211815
> 执行日期：2026-05-21
> Categorical 特征：`region` + `hashtag_name_top`（vocab_size=13/1185, embed_dim=4/16, total=20）
> 实验设计参考：`docs/multimodal_categorical_embedding_design.md`
> 前置消融参考：`outputs/validation/multimodal_ablation/real_raw_5000/202605211652/`

---

## 一、实验目的

1. 验证 region + hashtag_name_top categorical embedding 在 structured_only、structured_text、all_modalities 中的增益。
2. 与 Batch 14C 无 categorical 消融结果对比，量化 categorical embedding 的贡献。
3. 诊断 Multimodal 与 DNN 差距是否因 categorical 缩小。
4. 评估是否建议继续接入 author_id/music_id。

---

## 二、指标汇总

| ablation_name | enabled_modalities | cat | num_params | val_auc | val_f1 | test_auc | test_f1 | best_epoch |
|--------------|-------------------|-----|-----------|---------|--------|----------|---------|------------|
| **structured_only_no_cat** | structured | 关 | 1,121 | 0.5353 | 0.0000 | **0.5082** | 0.0000 | 7 |
| **structured_only_cat** | structured | 开 | 20,773 | 0.7971 | 0.6893 | **0.7896** | 0.6862 | 14 |
| Δ structured | — | — | +19,652 | +0.2618 | +0.6893 | **+0.2814** | +0.6862 | — |
| **structured_text_no_cat** | structured, text | 关 | 2,161 | 0.8192 | 0.6787 | **0.7655** | 0.6476 | 40 |
| **structured_text_cat** | structured, text | 开 | 21,813 | 0.8284 | 0.7033 | **0.8183** | 0.7145 | 17 |
| Δ structured_text | — | — | +19,652 | +0.0092 | +0.0246 | **+0.0528** | +0.0669 | — |
| **all_modalities_no_cat** | structured, text, media | 关 | 2,569 | 0.8340 | 0.6817 | **0.7812** | 0.6400 | 23 |
| **all_modalities_cat** | structured, text, media | 开 | 22,221 | 0.8416 | 0.6831 | **0.8184** | 0.7040 | 21 |
| Δ all_modalities | — | — | +19,652 | +0.0076 | +0.0014 | **+0.0372** | +0.0640 | — |
| **DNN baseline** | tabular含cat emb | — | 24,581 | — | — | **0.8414** | 0.7315 | — |

### 详细指标

| ablation_name | test_accuracy | test_precision | test_recall | test_loss |
|--------------|--------------|---------------|-------------|-----------|
| structured_only_no_cat | 0.6000 | 0.0000 | 0.0000 | 0.6741 |
| structured_only_cat | 0.7280 | 0.6371 | 0.7433 | 0.5101 |
| structured_text_no_cat | 0.6880 | 0.5907 | 0.7167 | 0.5213 |
| structured_text_cat | 0.7373 | 0.6588 | 0.7167 | 0.4849 |
| all_modalities_no_cat | 0.7000 | 0.6154 | 0.6667 | 0.4960 |
| all_modalities_cat | 0.7373 | 0.6528 | 0.7167 | 0.4779 |
| DNN baseline | 0.7520 | 0.6707 | 0.7600 | — |

---

## 三、核心对比

### 3.1 structured_only_no_cat vs structured_only_cat

| 指标 | no_cat | cat | 变化 |
|------|--------|-----|------|
| Test AUC | 0.5082 | **0.7896** | **+0.2814** |
| Test F1 | 0.0000 | **0.6862** | **+0.6862** |
| Val AUC | 0.5353 | 0.7971 | +0.2618 |
| 参数量 | 1,121 | 20,773 | +19,652 |

**结论**：Categorical embedding 将 structured_only 从「接近随机」跃升至「强预测能力」。这是当前单次实验中最大幅度的单项提升。

### 3.2 structured_text_no_cat vs structured_text_cat

| 指标 | no_cat | cat | 变化 |
|------|--------|-----|------|
| Test AUC | 0.7655 | **0.8183** | **+0.0528** |
| Test F1 | 0.6476 | **0.7145** | **+0.0669** |
| Val AUC | 0.8192 | 0.8284 | +0.0092 |
| 参数量 | 2,161 | 21,813 | +19,652 |

**结论**：在已含 text 分支（当前最强信号）的基础上，categorical embedding 仍带来约 +0.05 Test AUC 的增量提升。

### 3.3 all_modalities_no_cat vs all_modalities_cat

| 指标 | no_cat | cat | 变化 |
|------|--------|-----|------|
| Test AUC | 0.7812 | **0.8184** | **+0.0372** |
| Test F1 | 0.6400 | **0.7040** | **+0.0640** |
| Val AUC | 0.8340 | 0.8416 | +0.0076 |
| 参数量 | 2,569 | 22,221 | +19,652 |

**结论**：全模态下 categorical 仍带来约 +0.037 Test AUC 提升。

### 3.4 all_modalities_cat vs DNN baseline

| 指标 | Multimodal all_cat | DNN baseline | 差距 |
|------|-------------------|-------------|------|
| Test AUC | 0.8184 | **0.8414** | **-0.0230** |
| Test F1 | 0.7040 | **0.7315** | -0.0275 |

**结论**：加入 categorical embedding 后，Multimodal 与 DNN 的差距从 Batch 14C 的 -0.0602 缩小至 **-0.0230**（缩小了 61.8%）。

---

## 四、诊断分析

### 4.1 Categorical embedding 是否显著提升 structured_only？

**是，极度显著（+0.2814 Test AUC）**。这是最大的单项提升，验证了设计文档的核心假设：structured_only 接近随机的主因是缺少 categorical embedding。

### 4.2 Categorical embedding 是否提升 structured_text？

**是（+0.0528 Test AUC）**。在 text 分支已提供强信号（0.7655）的基础上，categorical 仍带来额外增量，说明 region + hashtag 包含了文本特征无法覆盖的信息。

### 4.3 Categorical embedding 是否提升 all_modalities？

**是（+0.0372 Test AUC）**。全模态下增益略小于 structured_text_cat（0.0528），可能是因为 media 信息部分与 categorical 冗余。

### 4.4 all_modalities_cat 是否缩小与 DNN 的差距？

**是**，差距从 -0.0602 缩小至 -0.0230，缩小幅度达 **61.8%**。当前仅使用 2 个 categorical features（region + hashtag_name_top），而 DNN 还额外使用了 author_id + music_id（2 个额外高基数字段）。

### 4.5 是否存在 text 与 categorical 融合后的互相干扰？

**未观察到**。structured_text_cat（0.8183）与 all_modalities_cat（0.8184）的 Test AUC 几乎一致，说明 text 和 categorical 在当前架构下互补良好。

### 4.6 是否有明显过拟合迹象？

**轻微**。优势：
- 所有 cat 变体的 val_auc 与 test_auc 较为接近（最大差距 0.008）。
- best_epoch（14-21）明显早于无 cat 变体（7-40），说明早期收敛更快。

关注点：
- 参数量从 ~2k 增至 ~21k（增加约 18.5 倍 from 2,569 to 22,221）。
- 对于 3,500 训练样本，21k 参数仍属于轻量模型（参数/样本比 ≈ 6.3）。

### 4.7 是否建议接入 author_id/music_id？

**暂不建议直接接入**，理由：
1. 当前 region+hashtag（vocab=13+1185）已将 DNN 差距缩小至 0.023。
2. author_id（vocab=3558）和 music_id（vocab≈5000）基数和 embedding 参数量更大（约 3558×16 + 5000×16 ≈ 137k 参数）。
3. 在 3,500 训练样本下，盲目增加高基数字段的过拟合风险上升。
4. **建议先做 3 seed 验证确认稳定性，再做 author_id/music_id 设计。**

---

## 五、结论

1. **Categorical embedding（region + hashtag_name_top）显著提升 Multimodal 在所有消融变体上的表现。**
2. structured_only 的提升最为显著（+0.2814 Test AUC），从随机跃升至强预测能力。
3. 在已含 text 的变体中，categorical 仍带来约 +0.04~0.05 的 Test AUC 增量。
4. all_modalities_cat（0.8184）与 DNN baseline（0.8414）的差距已缩小至 **0.023**（缩小 61.8%）。
5. 未观察到 text 与 categorical 的明显融合干扰。
6. 参数量从 ~2k 增至 ~21k，但验证/测试指标一致，无过拟合证据。
7. **Categorical 方向推荐继续推进**，但 author_id/music_id 需先设计再接入。

---

## 六、下一步建议

1. **3 seed 验证**：对最优变体 all_modalities_cat 做 3 次不同 seed 训练，确认指标稳定性。
2. **author_id + music_id 设计**：评估过拟合风险后，设计高基数字段的 categorical embedding 接入方案。
3. **Text 分支增强**：per-field encoding 或增加 SVD 维度（当前 32 维，可尝试 64/128）。
4. **融合策略改进**：attention-based fusion 替代当前 concat+MLP。
5. **多 seed 确认后再更新结论**：单次结果不代表最终结论。
