# Multimodal Categorical Embedding 多 seed 验证报告

> 项目根目录：`D:/CodeData/Program Coding/ByteDance/RA_PTA/`
> 数据集：`real_raw_5000`
> 批次：Batch 14G
> 验证 Run ID：202605211854
> 执行日期：2026-05-21
> Categorical 特征：`region` + `hashtag_name_top`（vocab_size=13/1185, embed_dim=4/16, total=20）
> 实验设计参考：`docs/multimodal_categorical_embedding_design.md`
> 前置消融参考：`outputs/validation/multimodal_categorical_ablation/real_raw_5000/202605211815/`

---

## 一、实验目的

1. 对 all_modalities_cat（region+hashtag_name_top categorical embedding）做 3 seed（2025/2026/2027）验证。
2. 对 all_modalities_no_cat（无 categorical embedding）做 3 seed（2025/2026/2027）作为公平对照。
3. 验证 categorical embedding 的 Test AUC 提升是否稳定。
4. 评估与 DNN baseline（0.8414）的差距。

---

## 二、验证设计

| 配置类型 | categorical_enabled | enabled_modalities | seeds |
|---------|-------------------|-------------------|-------|
| all_modalities_cat | true | structured, text, media | 2025, 2026, 2027 |
| all_modalities_no_cat | false | structured, text, media | 2025, 2026, 2027 |

---

## 三、逐 seed 结果

### 3.1 all_modalities_cat

| seed | run_id | val_auc | test_auc | test_f1 | best_epoch | num_params |
|------|--------|---------|----------|---------|------------|------------|
| 2025 | 202605211846 | 0.8385 | **0.8155** | 0.7061 | 15 | 22221 |
| 2026 | 202605211847 | 0.8416 | **0.8184** | 0.7040 | 21 | 22221 |
| 2027 | 202605211849 | 0.8313 | **0.8053** | 0.7008 | 14 | 22221 |

### 3.2 all_modalities_no_cat

| seed | run_id | val_auc | test_auc | test_f1 | best_epoch | num_params |
|------|--------|---------|----------|---------|------------|------------|
| 2025 | 202605211850 | 0.8318 | **0.7755** | 0.6398 | 22 | 2569 |
| 2026 | 202605211851 | 0.8340 | **0.7812** | 0.6400 | 23 | 2569 |
| 2027 | 202605211853 | 0.8222 | **0.7678** | 0.6677 | 24 | 2569 |

---

## 四、统计汇总

### 4.1 Test AUC

| 配置 | mean | std | min | max | median |
|------|------|-----|-----|-----|--------|
| all_modalities_cat | **0.8131** | 0.0056 | 0.8053 | 0.8184 | 0.8155 |
| all_modalities_no_cat | **0.7749** | 0.0055 | 0.7678 | 0.7812 | 0.7755 |
| delta (cat - nocat) | **+0.0382** | -- | -- | -- | -- |

### 4.2 Val AUC

| 配置 | mean | std | min | max |
|------|------|-----|-----|-----|
| all_modalities_cat | 0.8371 | 0.0043 | 0.8313 | 0.8416 |
| all_modalities_no_cat | 0.8293 | 0.0051 | 0.8222 | 0.8340 |

### 4.3 与 DNN baseline 对比

| 指标 | Cat mean | Cat best | DNN baseline | 差距(mean) | 差距(best) |
|------|----------|----------|-------------|-------------|-------------|
| Test AUC | 0.8131 | 0.8184 | 0.8414 | **-0.0283** | **-0.0230** |

---

## 五、核心对比

### 5.1 Cat vs No-cat: 稳定性判断

Cat 3 seed Test AUC 范围：**[0.8053, 0.8184]**
No-cat 3 seed Test AUC 范围：**[0.7678, 0.7812]**

**结论：cat 和 no-cat 的 Test AUC 范围完全无重叠**。Cat 最低 seed（0.8053）仍高于 no-cat 最高 seed（0.7812），提升稳定。

### 5.2 Cat vs DNN baseline

Cat mean Test AUC = 0.8131，DNN = 0.8414，差距 = -0.0283。
Cat best seed Test AUC = 0.8184，DNN = 0.8414，差距 = -0.0230。

### 5.3 标准差分析

Cat Test AUC std = 0.0056，No-cat Test AUC std = 0.0055。
两个配置的标准差接近，categorical embedding 未引入额外不稳定性。

---

## 六、结论

1. **Categorical embedding 提升稳定**：所有 3 seed 的 cat 均优于 no-cat，无 seed 出现反转。
2. **均值提升：+0.0382 Test AUC**（cat=0.8131 vs nocat=0.7749）。
3. **Cat 与 No-cat Test AUC 范围完全不重叠**，提升可视为统计显著。
4. **Cat best seed (0.8184) 已接近 DNN baseline (0.8414)**。
5. **Cat mean (0.8131) 与 DNN 差距 -0.0283**，还需优化。
6. **Categorical embedding 方向成立**，建议继续推进。
7. **仅离线代理标签结果**，不代表真实线上收益。

---

## 七、下一步建议

1. **author_id + music_id 接入设计**（需评估过拟合风险）。
2. **Text 分支增强**（per-field encoding, SVD 维度 32->64/128）。
3. **融合策略改进**（attention-based fusion）。
4. **充分调参**（当前使用默认 multimodal 配置）。
