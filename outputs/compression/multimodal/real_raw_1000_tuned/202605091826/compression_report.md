# Multimodal 动态量化压缩实验报告

- **Run ID**: 202605091826
- **生成时间**: 2026-05-09 18:26:23
- **量化方法**: dynamic_quantization
- **量化层**: nn.Linear
- **Benchmark 设备**: cpu
- **测试样本数**: 150
- **Warmup**: 10 passes
- **Repeat**: 100 passes

## 重要声明

1. **当前为离线本地推理计时，不代表线上服务延迟。**
2. 输入模型为正式 tuned Multimodal (run_id=202605091755)。
3. 使用 PyTorch 动态量化（`torch.quantization.quantize_dynamic`），仅量化 Linear 层。
4. **当前模型参数量仅 2,649，延迟主要受框架开销主导，量化不一定降低延迟。**
5. Quantized 模型运行在 CPU（动态量化 Linear 层仅支持 CPU），
   FP32 CPU benchmark 用于同设备公平对比。
6. 前置 CUDA benchmark (run_id=202605091806) 数据仅作为参考。
7. test split 仅用于最终离线评估，不参与训练或超参选择。
8. **当前结果不代表线上服务延迟或业务效果。**

## 模型版本

| 版本 | 说明 |
| --- | --- |
| tuned_multimodal_fp32_cuda | 正式 tuned FP32 CUDA（前置 benchmark 参考）|
| tuned_multimodal_fp32_cpu | 正式 tuned FP32 CPU（同设备公平对比）|
| tuned_multimodal_quantized | 动态量化后模型 CPU |

## 模型大小与参数量

| 模型版本 | 参数量 | 模型文件大小 (MB) |
| --- | ---: | ---: |
| tuned_multimodal (FP32) | 2,649 | 0.0136 |
| tuned_multimodal_quantized | 2,649 | 0.0099 |

## 离线评估指标 (test split)

| 指标 | FP32 (原始) | Quantized |
| --- | ---: | ---: |
| AUC | 0.992778 | 0.991852 |
| Accuracy | 0.933333 | 0.940000 |
| Precision | 0.980769 | 0.981132 |
| Recall | 0.850000 | 0.866667 |
| F1 | 0.910714 | 0.920354 |

注: FP32 和 Quantized 均在 test split (60 正 / 90 负) 上评估。

## 推理时延对比

| 模型版本 | Device | Avg (ms) | P50 (ms) | P95 (ms) | Min (ms) | Max (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| tuned_fp32 (参考) | CUDA | 0.6902 | 0.5710 | 1.5084 | 0.2194 | 10.7524 |
| tuned_fp32 (公平对比) | CPU | 0.1134 | 0.0807 | 0.2427 | 0.0555 | 4.4691 |
| tuned_quantized | CPU | 0.9258 | 0.6783 | 2.2520 | 0.2666 | 39.8889 |

## 吞吐量对比

| 模型版本 | Device | 吞吐量 (samples/sec) |
| --- | --- | ---: |
| tuned_fp32 (参考) | CUDA | 1448.80 |
| tuned_fp32 (公平对比) | CPU | 8821.59 |
| tuned_quantized | CPU | 1080.19 |

## 量化前后对比总结

| 对比项 | FP32 | Quantized | 变化 |
| --- | ---: | ---: | ---: |
| 模型大小 | 0.0136 MB | 0.0099 MB | 0.73× |
| Avg 延迟 | 0.1134 ms | 0.9258 ms | 8.17× |
| AUC | 0.992778 | 0.991852 | -0.000926 |
| F1 | 0.910714 | 0.920354 | +0.009640 |

## 分析

1. **模型大小**: 动态量化将 Linear 权重从 FP32 转为 INT8，预期模型文件大小可降至约 1/4（仅 Linear 层权重压缩）。对于极小模型（2,649 参数），绝对节省有限。
2. **离线指标**: 动态量化对 Linear 权重做 INT8 近似，通常会导致轻微精度损失。需要关注 AUC/F1 的变化幅度。
3. **推理时延**: 量化模型运行在 CPU，FP32 基准也运行在 CPU 做公平对比。CPU 推理延迟显著高于 CUDA，这是设备差异而非量化代价。
4. **框架开销**: 当前模型仅 2,649 参数，量化对计算量的减少被 PyTorch 框架调度开销淹没。

## Benchmark 设置

- **Warmup passes**: 10
- **Repeat passes**: 100
- **Batch size**: 1（单样本推理）
- **Device**: cpu
- **测试样本数**: 150
- **起止时间**: 2026-05-09 18:26:05 → 2026-05-09 18:26:23

## 结论

本次实验对正式 tuned Multimodal (run_id=202605091755) 进行了 PyTorch 动态量化，评估了量化前后在 test split 上的离线指标、模型大小和推理时延。

所有延迟数据为离线单样本推理计时，不代表线上服务延迟。结果受设备、CPU/GPU 架构、PyTorch 版本和 benchmark 参数设置影响。
