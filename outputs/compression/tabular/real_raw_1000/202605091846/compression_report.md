# Tabular 模型 (DNN / Wide & Deep) 动态量化压缩实验报告

- **Run ID**: 202605091846
- **生成时间**: 2026-05-09 18:47:26
- **量化方法**: dynamic_quantization
- **量化层**: nn.Linear
- **Benchmark 设备**: cpu
- **Warmup**: 10 passes
- **Repeat**: 100 passes
- **分类阈值**: 0.5

## 重要声明

1. **当前为离线本地推理计时，不代表线上服务延迟。**
2. 输入模型为上一阶段 real_raw_1000 的 DNN (run_id=202605081636) 和
   Wide & Deep (run_id=202605081746) baseline 模型。
3. 使用 PyTorch 动态量化（`torch.quantization.quantize_dynamic`），仅量化 Linear 层。
4. Quantized 模型运行在 CPU（动态量化 Linear 层仅支持 CPU），
   FP32 CPU benchmark 用于同设备公平对比。
5. test split 仅用于最终离线评估，不参与训练或超参选择。
6. **当前标签为 interaction_score 伪标签，不代表真实 CTR/CVR。**
7. 本实验与 Multimodal tuned 量化结果（outputs/compression/multimodal/real_raw_1000_tuned/）
   形成对照，考察动态量化在不同参数量级模型上的效果。
8. **当前结果不代表线上服务延迟或业务效果。**

## 模型版本说明

| 版本 | 说明 |
| --- | --- |
| dnn_fp32_cpu | DNN FP32 CPU（同设备公平对比）|
| dnn_quantized | DNN 动态量化后 CPU |
| wide_deep_fp32_cpu | Wide & Deep FP32 CPU（同设备公平对比）|
| wide_deep_quantized | Wide & Deep 动态量化后 CPU |

## 模型大小与参数量

| 模型 | 参数量 | FP32 (MB) | Quantized (MB) | 压缩比 |
| --- | ---: | ---: | ---: | ---: |
| DNN | 41,177 | 0.1607 | 0.1379 | 0.86× |
| Wide & Deep | 42,760 | 0.1678 | 0.1453 | 0.87× |

## 离线评估指标 (test split)

| 模型 | 版本 | AUC | Accuracy | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DNN | FP32 | 0.971667 | 0.900000 | 0.959184 | 0.783333 | 0.862385 |
| DNN | Quantized | 0.967407 | 0.886667 | 0.938776 | 0.766667 | 0.844037 |
| Wide&Deep | FP32 | 0.951111 | 0.886667 | 0.938776 | 0.766667 | 0.844037 |
| Wide&Deep | Quantized | 0.941296 | 0.866667 | 0.884615 | 0.766667 | 0.821429 |

注: 所有指标在 test split (150 样本, 60 正 / 90 负) 上评估。

## 推理时延对比

| 模型 | Device | Avg (ms) | P50 (ms) | P95 (ms) | Min (ms) | Max (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DNN FP32 | CPU | 0.1753 | 0.1264 | 0.4222 | 0.0822 | 3.6906 |
| DNN Quantized | CPU | 0.7707 | 0.5974 | 1.7862 | 0.2331 | 20.4168 |
| W&D FP32 | CPU | 0.2788 | 0.2167 | 0.6052 | 0.1261 | 4.2739 |
| W&D Quantized | CPU | 0.9428 | 0.7147 | 2.2421 | 0.2865 | 29.7789 |

## 吞吐量对比

| 模型 | Device | 吞吐量 (samples/sec) |
| --- | --- | ---: |
| DNN FP32 | CPU | 5705.78 |
| DNN Quantized | CPU | 1297.55 |
| W&D FP32 | CPU | 3587.12 |
| W&D Quantized | CPU | 1060.67 |

## 量化前后对比总结

| 对比项 | DNN FP32 | DNN Quantized | 变化 | W&D FP32 | W&D Quantized | 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 模型大小 | 0.1607 MB | 0.1379 MB | 0.86× | 0.1678 MB | 0.1453 MB | 0.87× |
| Avg 延迟 | 0.1753 ms | 0.7707 ms | 4.40× | 0.2788 ms | 0.9428 ms | 3.38× |
| AUC | 0.971667 | 0.967407 | -0.004259 | 0.951111 | 0.941296 | -0.009815 |
| F1 | 0.862385 | 0.844037 | -0.018349 | 0.844037 | 0.821429 | -0.022608 |

## 分析

1. **模型大小**: 动态量化将 Linear 权重从 FP32 转为 INT8。对于 DNN（41,177 参数）和 Wide & Deep（42,760 参数），模型文件压缩效果有限。
2. **离线指标**: 动态量化对 Linear 权重做 INT8 近似，通常会导致轻微精度损失。需要关注 AUC/F1 的变化幅度。
3. **推理时延**: 量化模型和 FP32 基准均在 CPU 上运行做公平对比。动态量化引入额外的量化/反量化开销，对于小模型可能导致延迟增加。
4. **与 Multimodal 量化对照**: DNN（41,177 参数）和 Wide & Deep（42,760 参数）的参数量显著大于 Multimodal tuned（2,649 参数），量化开销相对更小，但动态量化的运行时开销可能仍然显著。

## Benchmark 设置

- **Warmup passes**: 10
- **Repeat passes**: 100
- **Batch size**: 1（单样本推理）
- **Device**: cpu
- **测试样本数**: 150
- **起止时间**: 2026-05-09 18:46:50 → 2026-05-09 18:47:26

## 结论

本次实验对上一阶段 real_raw_1000 的 DNN 和 Wide & Deep baseline 模型进行了 PyTorch 动态量化，评估了量化前后在 test split 上的离线指标、模型大小和推理时延。

所有延迟数据为离线单样本推理计时，不代表线上服务延迟。结果受设备、CPU/GPU 架构、PyTorch 版本和 benchmark 参数设置影响。
