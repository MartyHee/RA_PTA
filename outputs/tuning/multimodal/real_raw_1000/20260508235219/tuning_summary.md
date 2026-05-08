# Multimodal 随机搜索调优摘要（real_raw_1000）

> 生成时间：2026-05-08 23:52:40
> 调优 Run ID：20260508235219
> Trial 数：1

## 1. 调优设置

| 项目 | 值 |
|---|---|
| 基础配置 | `configs/multimodal/multimodal_real_raw_1000.yaml` |
| 数据集 | real_raw_1000 |
| Trial 数 | 1（请求） / 完成 1，失败 0 |
| 随机种子 | 2026 |
| 排序指标（主） | val_auc |
| 排序指标（次） | val_f1 |

### 搜索空间

| 参数 | 范围 | 采样方式 |
|---|---|---|
| learning_rate | [1e-05, 0.01] | log |
| weight_decay | [1e-06, 0.01] | log |
| dropout | [0.1, 0.5] | uniform |
| hidden_dim | [16, 32, 64] | 离散均匀 |
| batch_size | [32, 64, 128] | 离散均匀 |

## 2. 最佳 Trial

| 项目 | 值 |
|---|---|
| Trial ID | 0 |
| 状态 | completed |
| learning_rate | 0.000034 |
| weight_decay | 0.000363 |
| dropout | 0.286907 |
| hidden_dim | 16 |
| batch_size | 64 |
| Best Epoch | 50 |
| Val AUC | 0.4494 |
| Val F1 | 0.2162 |
| Test AUC | 0.4594 |
| Test F1 | 0.2542 |
| 输出目录 | `outputs\tuning\multimodal\real_raw_1000\20260508235219\trial_0\202605082352` |

## 3. 全部 Trial 汇总

| Trial ID | Status | val_auc | val_f1 | test_auc | test_f1 | lr | wd | dropout | hidden_dim | batch_size |
|----------|--------|---------|--------|----------|---------|-----|----|--------|-----------|------------|
| 0 | completed | 0.4494 | 0.2162 | 0.4594 | 0.2542 | 0.000034 | 0.000363 | 0.287 | 16 | 64 |

## 4. 主要限制

- 当前调优仅基于 1 trials，搜索不充分。
- 标签为 interaction_score 伪标签，不代表真实业务目标。
- 所有结果均为离线实验，不代表线上推荐效果。
- Test 指标仅供参考，best trial 选择仅依据 val 指标。
- 部分 trials 可能因配置不兼容失败，失败原因见 tuning_trials.csv。

## 5. 下一步建议

- 增加 trial 数（建议 20～50）以获得更充分的搜索覆盖。
- 根据最佳 trial 的收敛位置缩小搜索空间。
- 扩展搜索空间参数（如 epochs、early_stopping_patience、optimizer 类型）。
- 验证最佳配置在 val 和 test 上的稳定性（多 seed 重复）。

> **⚠️ 1-trial smoke test**：当前仅运行了 1 个 trial，不构成调优结论。
> 以上结果为最小可运行检查，仅供流程验证。
