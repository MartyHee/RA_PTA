# Multimodal Late Fusion 多 seed 验证报告

> 项目根目录：`D:/CodeData/Program Coding/ByteDance/RA_PTA/`
> 数据集：`real_raw_5000`
> 批次：Batch 14L-experiment
> 验证 Run ID：202605221546
> 执行日期：2026-05-22
> Fusion 类型：`late_fusion`（可学习加权 sum）
> 实验设计参考：`docs/multimodal_fusion_strategy_design.md`
> 前置实现参考：Batch 14L-impl

---

## 一、实验目的

1. 对 all_modalities_cat + late_fusion（weighted_sum）配置做 3 seed（2025/2026/2027）验证。
2. 与 concat_mlp all_modalities_cat 3 seed mean（0.8131）对比。
3. 判断 late_fusion 是否稳定优于 concat_mlp。
4. 分析 late_fusion 可学习权重的稳定性和分布。
5. 不修改代码/config/data。不运行 gated_fusion/dropout/residual。

---

## 二、实验配置

| 参数 | 值 |
|------|-----|
| dataset | real_raw_5000 |
| enabled_modalities | structured, text, media |
| categorical_enabled | true |
| categorical_features | region (vocab=13, dim=4) + hashtag_name_top (vocab=1185, dim=16) |
| fusion_type | late_fusion |
| late_fusion_mode | weighted_sum |
| late_fusion_modality_order | text, media, structured |
| text_profile | merged_text_v2_dim64 |
| seeds | 2025, 2026, 2027 |

---

## 三、逐 seed 结果

### 3.1 指标汇总

| seed | run_id | val_auc | test_auc | test_f1 | best_epoch | num_params | late_fusion_weights |
|------|--------|---------|----------|---------|------------|------------|---------------------|
| 2025 | 202605221545 | 0.8433 | **0.8223** | 0.7089 | 25 | 20814 | text=0.3369, media=0.2714, structured=0.3917 |
| 2026 | 202605221529 | 0.8398 | **0.8288** | 0.7191 | 19 | 20814 | text=0.3396, media=0.2894, structured=0.3711 |
| 2027 | 202605221538 | 0.8400 | **0.8258** | 0.7207 | 33 | 20814 | text=0.3473, media=0.2948, structured=0.3579 |

---

## 四、统计汇总

### 4.1 Test AUC

| 配置 | mean | std | min | max | median |
|------|------|-----|-----|-----|--------|
| all_modalities_cat_late_fusion | **0.8256** | 0.0027 | 0.8223 | 0.8288 | 0.8258 |

### 4.2 Val AUC

| 配置 | mean | std | min | max |
|------|------|-----|-----|-----|
| all_modalities_cat_late_fusion | 0.8410 | 0.0016 | 0.8398 | 0.8433 |

### 4.3 与 concat_mlp baseline 对比

| 指标 | Late fusion mean | Concat+MLP mean | Δ |
|------|-----------------|-----------------|----|
| Test AUC | 0.8256 | 0.8131 | **+0.0125** |

### 4.4 与 DNN baseline 对比

| 指标 | Late fusion mean | Late fusion max | DNN | 差距(mean) | 差距(max) |
|------|-----------------|-----------------|-----|------------|-----------|
| Test AUC | 0.8256 | 0.8288 | 0.8414 | **-0.0158** | **-0.0126** |

---

## 五、Late Fusion 权重分析

| seed | w_text | w_media | w_structured |
|------|--------|---------|-------------|
| seed=2025 | 0.3369 | 0.2714 | 0.3917 |
| seed=2026 | 0.3396 | 0.2894 | 0.3711 |
| seed=2027 | 0.3473 | 0.2948 | 0.3579 |

**权重统计：**

| 模态 | mean | std | min | max |
|------|------|-----|-----|-----|
| text | 0.3413 | 0.0044 | 0.3369 | 0.3473 |
| media | 0.2852 | 0.0100 | 0.2714 | 0.2948 |
| structured | 0.3736 | 0.0139 | 0.3579 | 0.3917 |

---

## 六、核心对比

### 6.1 Late fusion vs concat_mlp

Late fusion mean **0.8256** > concat_mlp mean **0.8131** (Δ=**+0.0125**)

判断标准：
- >= 0.8181 (cat mean + 0.005)：**稳定优于**
- > 0.8131 但 < 0.8181：**小幅提升但不充分**
- <= 0.8131：**未稳定提升**

**判定：Late fusion 3 seed mean (0.8256) >= cat mean + 0.005 (0.8181), 判定稳定优于 concat_mlp**

### 6.2 Late fusion vs DNN

Late fusion mean **0.8256** < DNN **0.8414** (Δ=**-0.0158**)

### 6.3 Weight 稳定性

权重分布稳定，最大偏差=0.0619，无单模态塌缩

---

## 七、结论

1. Late fusion 3 seed mean Test AUC = 0.8256 >= 0.8181，判定 late_fusion 稳定优于 concat_mlp。
2. 建议考虑将 late_fusion 作为 Multimodal 内部默认 fusion_type。
3. Late fusion weights 分布相对稳定 (text=0.341, media=0.285, structured=0.374)，未出现单模态塌缩。
4. Late fusion mean (0.8256) 与 DNN (0.8414) 仍有差距 (Δ=-0.0158)，不替换 DNN baseline。
5. 建议继续尝试 gated_fusion 作为下一个改进方向。

---

## 八、下一步建议

1. 考虑将 late_fusion 作为 Multimodal 内部新的默认 fusion_type。
2. 继续尝试 gated_fusion 看是否能进一步提升。
3. DNN 仍保留为推荐 baseline，直至 Multimodal 在 3 seed 上超过 0.8414。
4. 如后续验证充分，可写 Multimodal 融合策略阶段总结文档。

---

## 九、离线实验声明

1. 所有结果基于**离线代理标签**（interaction_score P60 分位数二分类），不代表真实线上推荐收益。
2. 当前为本地模拟实验，不包含真实用户行为、曝光、点击、完播、转化或留存数据。
3. 不要将本文档中的数值直接引用为线上推荐系统效果。
4. DNN 基线（0.8414）仍为推荐 baseline，直至 Multimodal 以多 seed 验证稳定超过该数值。
