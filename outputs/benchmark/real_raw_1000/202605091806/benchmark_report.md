# Multimodal 模型推理性能 Benchmark 报告

- **Run ID**: 202605091806
- **生成时间**: 2026-05-09 18:06:41
- **设备**: cuda
- **测试样本数**: 150
- **Warmup**: 10 passes
- **Repeat**: 100 passes

## 重要声明

1. **当前为离线本地推理计时，不代表线上服务延迟。**
2. **结果受设备、batch_size、warmup/repeat 设置影响。**
3. 所有模型基于 interaction_score 伪标签训练，不代表真实业务目标。
4. 当前 benchmark 仅覆盖 Multimodal 模型，不包括 DNN / Wide & Deep / GraphSAGE。
5. Tuned 模型来自 random search best trial (trial 16) 并已固化为正式 tuned run (202605091755)。

## 模型版本

| 版本 | 说明 |
| --- | --- |
| baseline_multimodal | 原始 Multimodal baseline (run_id=202605081927) |
| tuned_multimodal | 正式 tuned Multimodal (run_id=202605091755, best trial 16) |

## 模型大小与参数量

| 模型版本 | 参数量 | 模型文件大小 (MB) |
| --- | ---: | ---: |
| baseline_multimodal | 7,857 | 0.0334 |
| tuned_multimodal | 2,649 | 0.0136 |

## 推理时延 (单样本, batch_size=1)

| 模型版本 | Avg (ms) | P50 (ms) | P95 (ms) | Min (ms) | Max (ms) | Std (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_multimodal | 0.6807 | 0.5612 | 1.4713 | 0.2177 | 10.9791 | 0.4527 |
| tuned_multimodal | 0.6902 | 0.5710 | 1.5084 | 0.2194 | 10.7524 | 0.4459 |

## 吞吐量

| 模型版本 | 吞吐量 (samples/sec) |
| --- | ---: |
| baseline_multimodal | 1469.08 |
| tuned_multimodal | 1448.80 |

## Tuned vs Baseline 对比

| 指标 | Baseline | Tuned | Tuned/Baseline |
| --- | ---: | ---: | ---: |
| 参数量 | 7,857 | 2,649 | 0.34× |
| 模型文件大小 | 0.0334 MB | 0.0136 MB | 0.41× |
| Avg 延迟 | 0.6807 ms | 0.6902 ms | 1.01× |
| P50 延迟 | 0.5612 ms | 0.5710 ms | 1.02× |
| P95 延迟 | 1.4713 ms | 1.5084 ms | 1.03× |
| 吞吐量 | 1469.08 samples/s | 1448.80 samples/s | 0.99× |

## Benchmark 设置

- **Warmup passes**: 10
- **Repeat passes**: 100
- **Batch size**: 1（单样本推理）
- **Device**: cuda
- **测试样本数**: 150
- **起止时间**: 2026-05-09 18:06:17 → 2026-05-09 18:06:41

## 结论

本次 benchmark 评估了 baseline Multimodal 与正式 tuned Multimodal 两个版本在相同 test split (150 样本) 上的推理性能。

所有延迟数据为离线单样本推理计时，不代表线上服务延迟。结果受设备、CUDA 驱动版本、PyTorch 版本和 benchmark 参数设置影响。
