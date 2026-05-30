# 多目标 Reranking 实验汇总报告

**实验 run_id**: 20260526110629
**生成时间**: 2026-05-26 11:06:29 UTC
**模型**: dnn / real_raw_5000 / 202605132017

---

## 1. 实验目标

在 DNN baseline 预测 score 基础上，通过引入 diversity_gain 和 novelty_score 进行 greedy reranking，
在维持 relevance 可接受下降范围内，提升推荐列表的多样性、覆盖度与新颖性。

---

## 2. 输入结果来源

- **Baseline 指标**: outputs/multi_objective/metrics/20260526092752
- **Rerank runs**:
  - `diversity_light`: outputs/multi_objective/rerank/2026052610250378
  - `diversity_medium`: outputs/multi_objective/rerank/20260526102344
  - `diversity_novelty`: outputs/multi_objective/rerank/2026052610252476

---

## 3. 四组 Preset 参数

| Preset | alpha | beta | gamma | 说明 |
|--------|:----:|:----:|:-----:|------|
| baseline | 1.0 | 0.0 | 0.0 | 纯 relevance，无 reranking |
| diversity_light | 1.0 | 0.05 | 0.0 | 轻度多样性（beta 过弱，无效） |
| diversity_medium | 1.0 | 0.1 | 0.0 | 中等多样性（当前推荐） |
| diversity_novelty | 1.0 | 0.1 | 0.05 | 多样性 + 新颖性（相关性风险偏高） |

---

## 4. K=10 / 20 / 50 对比总表

### K=10

| Preset | Score | ScoreΔ% | PosRate | HshDiv | HshΔ% | CovHsh | CovΔ% | Novel | NovΔ% | Flag |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.9808 | 0.0000 | 1.00 | 0.6000 | 0.0000 | 0.0190 | 0.0000 | 0.8632 | 0.0000 | baseline |
| diversity_light | 0.9808 | 0.0000 | 1.00 | 0.6000 | 0.0000 | 0.0190 | 0.0000 | 0.8632 | 0.0000 | ineffective |
| diversity_medium | 0.9752 | -0.5800 | 1.00 | 0.8000 | 33.33 | 0.0253 | 33.16 | 0.8743 | 1.29 | good_tradeoff |
| diversity_novelty | 0.9703 | -1.07 | 1.00 | 0.9000 | 50.00 | 0.0285 | 50.00 | 0.8839 | 2.40 | good_tradeoff |

### K=20

| Preset | Score | ScoreΔ% | PosRate | HshDiv | HshΔ% | CovHsh | CovΔ% | Novel | NovΔ% | Flag |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.9438 | 0.0000 | 0.9000 | 0.6500 | 0.0000 | 0.0411 | 0.0000 | 0.9038 | 0.0000 | baseline |
| diversity_light | 0.9438 | 0.0000 | 0.9000 | 0.6500 | 0.0000 | 0.0411 | 0.0000 | 0.9038 | 0.0000 | ineffective |
| diversity_medium | 0.9438 | 0.0000 | 0.9000 | 0.6500 | 0.0000 | 0.0411 | 0.0000 | 0.9038 | 0.0000 | ineffective |
| diversity_novelty | 0.9408 | -0.3200 | 0.8500 | 0.7000 | 7.69 | 0.0443 | 7.79 | 0.9134 | 1.06 | relevance_risk |

### K=50

| Preset | Score | ScoreΔ% | PosRate | HshDiv | HshΔ% | CovHsh | CovΔ% | Novel | NovΔ% | Flag |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.8501 | 0.0000 | 0.8000 | 0.7600 | 0.0000 | 0.1203 | 0.0000 | 0.8546 | 0.0000 | baseline |
| diversity_light | 0.8501 | 0.0000 | 0.8000 | 0.7600 | 0.0000 | 0.1203 | 0.0000 | 0.8546 | 0.0000 | ineffective |
| diversity_medium | 0.8501 | 0.0000 | 0.8000 | 0.7600 | 0.0000 | 0.1203 | 0.0000 | 0.8546 | 0.0000 | ineffective |
| diversity_novelty | 0.8501 | 0.0000 | 0.8000 | 0.7600 | 0.0000 | 0.1203 | 0.0000 | 0.8546 | 0.0000 | ineffective |

---

## 5. Relevance 变化分析

### K=10

| Preset | Mean Score | Δ | Δ% | PosRate | PosRateΔ | PosRateΔ% |
|--------|:----------:|:-:|:--:|:-------:|:--------:|:---------:|
| baseline | 0.9808 | +0.0000 | +0.00% | 1.0 | +0.0000 | +0.00% |
| diversity_light | 0.9808 | +0.0000 | +0.00% | 1.0 | +0.0000 | +0.00% |
| diversity_medium | 0.9752 | -0.0056 | -0.58% | 1.0 | +0.0000 | +0.00% |
| diversity_novelty | 0.9703 | -0.0105 | -1.07% | 1.0 | +0.0000 | +0.00% |

### K=20

| Preset | Mean Score | Δ | Δ% | PosRate | PosRateΔ | PosRateΔ% |
|--------|:----------:|:-:|:--:|:-------:|:--------:|:---------:|
| baseline | 0.9438 | +0.0000 | +0.00% | 0.9 | +0.0000 | +0.00% |
| diversity_light | 0.9438 | +0.0000 | +0.00% | 0.9 | +0.0000 | +0.00% |
| diversity_medium | 0.9438 | +0.0000 | +0.00% | 0.9 | +0.0000 | +0.00% |
| diversity_novelty | 0.9408 | -0.0030 | -0.32% | 0.85 | -0.0500 | -5.56% |

### K=50

| Preset | Mean Score | Δ | Δ% | PosRate | PosRateΔ | PosRateΔ% |
|--------|:----------:|:-:|:--:|:-------:|:--------:|:---------:|
| baseline | 0.8501 | +0.0000 | +0.00% | 0.8 | +0.0000 | +0.00% |
| diversity_light | 0.8501 | +0.0000 | +0.00% | 0.8 | +0.0000 | +0.00% |
| diversity_medium | 0.8501 | +0.0000 | +0.00% | 0.8 | +0.0000 | +0.00% |
| diversity_novelty | 0.8501 | +0.0000 | +0.00% | 0.8 | +0.0000 | +0.00% |

**结论**:
- diversity_light、diversity_medium 的 relevance 变化在可接受范围内（<=2%）。
- diversity_novelty K=20 的 positive_rate 下降 5.6 个百分点（-5.6%），超过 3% 阈值，标记为相关性风险。

---

## 6. Diversity 变化分析

### K=10

| Preset | AuthDiv | AuthΔ% | HshDiv | HshΔ% | RegDiv | RegΔ% |
|--------|:-------:|:------:|:------:|:-----:|:------:|:-----:|
| baseline | 1.0 | +0.00% | 0.6 | +0.00% | 0.1 | +0.00% |
| diversity_light | 1.0 | +0.00% | 0.6 | +0.00% | 0.1 | +0.00% |
| diversity_medium | 1.0 | +0.00% | 0.8 | +33.33% | 0.1 | +0.00% |
| diversity_novelty | 1.0 | +0.00% | 0.9 | +50.00% | 0.1 | +0.00% |

### K=20

| Preset | AuthDiv | AuthΔ% | HshDiv | HshΔ% | RegDiv | RegΔ% |
|--------|:-------:|:------:|:------:|:-----:|:------:|:-----:|
| baseline | 1.0 | +0.00% | 0.65 | +0.00% | 0.05 | +0.00% |
| diversity_light | 1.0 | +0.00% | 0.65 | +0.00% | 0.05 | +0.00% |
| diversity_medium | 1.0 | +0.00% | 0.65 | +0.00% | 0.05 | +0.00% |
| diversity_novelty | 1.0 | +0.00% | 0.7 | +7.69% | 0.05 | +0.00% |

### K=50

| Preset | AuthDiv | AuthΔ% | HshDiv | HshΔ% | RegDiv | RegΔ% |
|--------|:-------:|:------:|:------:|:-----:|:------:|:-----:|
| baseline | 0.96 | +0.00% | 0.76 | +0.00% | 0.02 | +0.00% |
| diversity_light | 0.96 | +0.00% | 0.76 | +0.00% | 0.02 | +0.00% |
| diversity_medium | 0.96 | +0.00% | 0.76 | +0.00% | 0.02 | +0.00% |
| diversity_novelty | 0.96 | +0.00% | 0.76 | +0.00% | 0.02 | +0.00% |

**结论**:
- **author_diversity 已饱和**: 所有 preset 在 K=10 和 K=20 均为 1.0。
- **region_diversity 无变化**: 候选集仅 4-5 个唯一 region，且 59.6% 缺失。
- **hashtag_diversity 是主要改善点**: 在 K=10, diversity_medium 提升 33.3%、diversity_novelty 提升 50.0%。
- K=20 的 hashtag_diversity 改善较小（medium 无变化，novelty +7.7%）。

---

## 7. Coverage 变化分析

### K=10

| Preset | CovAuth | CovAuthΔ% | CovHsh | CovHshΔ% | CovReg | CovRegΔ% |
|--------|:-------:|:---------:|:------:|:--------:|:------:|:--------:|
| baseline | 0.0155 | +0.00% | 0.019 | +0.00% | 0.2 | +0.00% |
| diversity_light | 0.0155 | +0.00% | 0.019 | +0.00% | 0.2 | +0.00% |
| diversity_medium | 0.0155 | +0.00% | 0.0253 | +33.16% | 0.2 | +0.00% |
| diversity_novelty | 0.0155 | +0.00% | 0.0285 | +50.00% | 0.2 | +0.00% |

### K=20

| Preset | CovAuth | CovAuthΔ% | CovHsh | CovHshΔ% | CovReg | CovRegΔ% |
|--------|:-------:|:---------:|:------:|:--------:|:------:|:--------:|
| baseline | 0.0309 | +0.00% | 0.0411 | +0.00% | 0.2 | +0.00% |
| diversity_light | 0.0309 | +0.00% | 0.0411 | +0.00% | 0.2 | +0.00% |
| diversity_medium | 0.0309 | +0.00% | 0.0411 | +0.00% | 0.2 | +0.00% |
| diversity_novelty | 0.0309 | +0.00% | 0.0443 | +7.79% | 0.2 | +0.00% |

### K=50

| Preset | CovAuth | CovAuthΔ% | CovHsh | CovHshΔ% | CovReg | CovRegΔ% |
|--------|:-------:|:---------:|:------:|:--------:|:------:|:--------:|
| baseline | 0.0742 | +0.00% | 0.1203 | +0.00% | 0.2 | +0.00% |
| diversity_light | 0.0742 | +0.00% | 0.1203 | +0.00% | 0.2 | +0.00% |
| diversity_medium | 0.0742 | +0.00% | 0.1203 | +0.00% | 0.2 | +0.00% |
| diversity_novelty | 0.0742 | +0.00% | 0.1203 | +0.00% | 0.2 | +0.00% |

**结论**:
- coverage_author 变化极小（author 覆盖已充足）。
- coverage_hashtag 在 K=10 有提升: diversity_medium +33.2%，diversity_novelty +50.0%。
- coverage_region 无变化。

---

## 8. Novelty 变化分析

### K=10

| Preset | NovelMean | NovelMeanΔ% |
|--------|:---------:|:-----------:|
| baseline | 0.8632 | +0.00% |
| diversity_light | 0.8632 | +0.00% |
| diversity_medium | 0.8743 | +1.29% |
| diversity_novelty | 0.8839 | +2.40% |

### K=20

| Preset | NovelMean | NovelMeanΔ% |
|--------|:---------:|:-----------:|
| baseline | 0.9038 | +0.00% |
| diversity_light | 0.9038 | +0.00% |
| diversity_medium | 0.9038 | +0.00% |
| diversity_novelty | 0.9134 | +1.06% |

### K=50

| Preset | NovelMean | NovelMeanΔ% |
|--------|:---------:|:-----------:|
| baseline | 0.8546 | +0.00% |
| diversity_light | 0.8546 | +0.00% |
| diversity_medium | 0.8546 | +0.00% |
| diversity_novelty | 0.8546 | +0.00% |

**结论**:
- diversity_novelty 的 novelty_mean 提升最明显（K=10 +2.4%, K=20 +1.1%）。
- diversity_medium 在 K=10 也有小幅提升（+1.3%）。
- novelty 主要收益来自更换为低频 hashtag 实体。

---

## 9. Trade-off 结论

### K=10 Trade-off

| Preset | ScoreΔ% | HshDivΔ% | NovelΔ% | 判断 |
|--------|:-------:|:---------:|:-------:|------|
| diversity_light | +0.00% | +0.00% | +0.00% | ineffective |
| diversity_medium | -0.58% | +33.33% | +1.29% | good_tradeoff |
| diversity_novelty | -1.07% | +50.00% | +2.40% | good_tradeoff |

### K=20 Trade-off

| Preset | ScoreΔ% | PosRateΔ% | HshDivΔ% | NovelΔ% | 判断 |
|--------|:-------:|:---------:|:---------:|:-------:|------|
| diversity_light | +0.00% | +0.00% | +0.00% | +0.00% | ineffective |
| diversity_medium | +0.00% | +0.00% | +0.00% | +0.00% | ineffective |
| diversity_novelty | -0.32% | -5.56% | +7.69% | +1.06% | relevance_risk |

---

## 10. 推荐 Preset

**推荐**: `diversity_medium`

**理由**: diversity_medium (alpha=1.0, beta=0.10, gamma=0.0) 在 K=10 提升 hashtag_diversity 33.3%，coverage_hashtag 33.2%，relevance 仅下降 0.57%，trade-off 最优。K=20 relevance 无损，positive_rate 不变。

**拒绝的配置**:
- `diversity_light`: beta=0.05 过弱，排序无任何变化。
- `diversity_novelty`: 多样性最强 (K=10 hashtag_diversity +50%)，但 K=20 positive_rate 下降 5.6% > 3% 阈值，relevance proxy 风险偏高。

---

## 11. 风险与后续建议

### 已识别风险

- diversity_light beta=0.05 过小，top-20 无排序变化。
- diversity_novelty positive_rate@20 下降 5.6%，超过 3% 阈值。
- 第一版仅实现 P0 字段，region 覆盖和 hashtag 覆盖有限。
- 所有结果基于离线代理标签，不代表真实线上推荐收益。
- diversity_novelty 的 positive_rate 下降 5.6% 超过阈值，不应作为默认配置。
- diversity_light 效果为零，后续实验可跳过此配置。
- region 和 author 的多样性已饱和，需引入 P1 字段才能进一步突破。
- 所有结果基于离线代理标签，不代表真实线上推荐收益。

### 后续建议

1. **Batch 16F**: 撰写正式多目标实验报告 `docs/multi_objective_experiment_report.md`。
2. 尝试 diversity_strong（beta=0.20, gamma=0.05）观察 trade-off 边界。
3. 引入 P1 字段（music_id、duration_bucket）扩展多样性优化空间。
4. 如 region 覆盖度提升，可考虑填充 region 缺失值以利用该字段。
5. 后续可探索将 reranking 纳入 pipeline 作为可选项。

