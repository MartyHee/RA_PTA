# RA_PTA 多模型对比实验项目说明

## 1. 项目概述

本项目用于基于抖音公开网页端数据进行多模型对比实验，并在实验口径明确后推进端到端推荐流水线工程化。核心工作包括：

- raw 数据读取与质量检查
- tabular / graph / multimodal 输入构建
- DNN / Wide & Deep / GraphSAGE / Multimodal 多模型训练
- 统一离线评估
- 多模型指标对比
- 离线 A/B 模拟或分组统计
- 实验报告整理
- 后续训练与推理流水线工程化

当前项目根目录：

```text
D:/CodeData/Program Coding/ByteDance/RA_PTA/
```

上游爬虫与 raw 数据项目目录：

```text
D:/CodeData/Program Coding/ByteDance/RA_PTA/douyin_data_project/
```

---

## 2. 当前阶段状态

### 2.1 已完成：real_raw_1000 多模型对比实验

已完成基于 `real_raw_1000` 真实网页端 raw 数据的完整多模型对比实验：

1. real_raw_1000 数据读取与质量检查
2. tabular 输入构建
3. graph 输入构建
4. multimodal 输入构建
5. DNN / Wide & Deep / GraphSAGE / Multimodal 训练与评估
6. 统一多模型对比
7. 离线 A/B 模拟
8. 最终报告覆盖到 `reports/model_comparison_report.md`

该阶段发现一个关键问题：历史 tabular 输入中包含了 `digg_count/comment_count/share_count/collect_count`，而 label 又由这些字段求和构造。因此高 AUC 更可能来自当前数据和伪标签构造下的强可分性，存在明显标签构造字段泄漏风险，不能视为可靠推荐泛化能力。

### 2.2 已完成：real_raw_5000 数据扩充与正式数据包整理

已完成 5 批 1000-URL 采集，并合并整理为正式数据包：

```text
real_raw_5000
```

正式 raw 数据包目录：

```text
douyin_data_project/data/interim/real_raw_5000/
```

交付说明：

```text
douyin_data_project/data/interim/real_raw_5000/real_raw_5000_delivery.md
```

高置信辅助目录：

```text
douyin_data_project/data/processed/real_raw_5000/
```

其中包含：

```text
high_confidence_web_video_meta_real_raw_5000.csv
high_confidence_video_ids.txt
high_confidence_filter_report.json
```

重要口径：

- 上层实验默认读取 `data/interim/real_raw_5000/` 下的 11 张 raw 表。
- `data/processed/real_raw_5000/` 是 high-confidence 辅助筛选信息，不替代 11 张 raw 表。
- 后续特征构建可通过配置选择 `full` 或 `high_confidence`。

### 2.3 当前状态：real_raw_5000 全流程适配完成

已完成 `real_raw_5000` 数据集的全流程适配与四模型统一训练入口：

| 阶段 | 状态 |
|------|------|
| `real_raw_5000` 数据读取适配 | ✅ 完成 |
| `real_raw_5000` tabular 输入构建（no-leakage 强制） | ✅ 完成 |
| `real_raw_5000` graph 输入构建（no-leakage 强制） | ✅ 完成 |
| `real_raw_5000` multimodal 输入构建（no-leakage 强制） | ✅ 完成 |
| DNN / Wide & Deep / GraphSAGE / Multimodal 统一训练入口 | ✅ 完成 |
| DNN wrapper run（202605132017） | ✅ Test AUC=0.8414, F1=0.7315 |
| Wide & Deep wrapper run（202605132026） | ✅ Test AUC=0.8242, F1=0.6859 |
| GraphSAGE wrapper run（202605132107） | ✅ Test AUC=0.8327, F1=0.7289 |
| Multimodal wrapper run（202605132210） | ✅ Test AUC=0.7812, F1=0.6400 |
| 端到端 DNN pipeline orchestrator 第一版 | ✅ 完成 |
| pipeline orchestrator infer-only 验证 | ✅ 输出: outputs/inference/dnn/real_raw_5000/202605132017/20260514_171959/ |
| 统一调参入口 tune.py 第一版（DNN random search） | ✅ 完成 |
| REST API 本地模拟服务（FastAPI） | ✅ 已完成，尚未部署上线 |
| 系统架构文档（docs/system_architecture.md） | ✅ 完成 |
| 本地端到端推荐流水线第一版 | ✅ 完成 |
| 线上模拟环境搭建与 A/B 测试设计（Stage 2） | ✅ 完成 |
| 阶段二交付文档（delivery document） | ✅ 完成 |

**当前推荐 baseline：DNN**（Test AUC=0.8414, F1=0.7315）。

**Multimodal 当前最佳：** all_modalities_cat + late_fusion（3 seed mean Test AUC=0.8256）。
- 原始 Multimodal no-cat：0.7812
- +categorical embedding：0.8131（3 seed mean）
- +late_fusion：0.8256（3 seed mean）← **Multimodal 当前 preferred config**
- 与 DNN 差距从 -0.0602 缩小至 -0.0158（缩小 73.8%）
- Multimodal 尚未超过 DNN，DNN 保持全项目推荐 baseline

**最新 tuning best trial：** trial_000（run_id=202605141914, val_auc=0.8413, test_auc=0.8348）。
详情见 `outputs/tuning/dnn/real_raw_5000/20260514_191432/`。

### 2.4 已完成：阶段二（线上模拟环境搭建与 A/B 测试设计 + 模型改进）

阶段二已完整结束，涵盖以下 7 个方向：

| 方向 | 批次 | 关键产出 |
|------|------|---------|
| FastAPI online simulation service 扩展设计 | Batch 12B | `docs/online_simulation_design.md` |
| A/B 测试方案设计 | Batch 12C | `docs/ab_test_design.md` |
| FastAPI online simulation service 扩展实现 | Batch 12D | `src/serving/api.py` 扩展（`/models`, `/recommend`, `/ab/recommend`） |
| A/B 日志与指标计算 | Batch 12E | `src/serving/ab_metrics.py`, `outputs/online_simulation/20260518_183823/` |
| A/B 测试实施方案完善 | Batch 12F | `docs/ab_test_implementation_plan.md` |
| DNN 多 seed 验证与 20-trial 调参 | Batch 13A–13C | baseline 5 seed mean=0.8386, 20-trial 未超越 baseline |
| Multimodal 改进（消融 → categorical → 文本 → late_fusion） | Batch 14A–14L-summary | 从 0.7812 提升至 0.8256（late_fusion 3 seed mean） |

阶段二交付总结文档：

```text
docs/stage2_online_ab_model_improvement_delivery.md
```

#### 阶段二核心结论

- **API 仍为本地 FastAPI simulation，未部署上线。**
- **A/B 测试为本地模拟，没有真实线上点击/留存事件。**
- **DNN（0.8414）仍是全项目推荐 baseline。**
- **Multimodal 内部 baseline：all_modalities_cat + late_fusion（3 seed mean=0.8256），尚未超过 DNN（Δ=-0.0158）。**
- **所有结果基于离线代理标签（interaction_score P60 分位数二分类），不代表真实线上推荐收益。**

### 2.5 当前任务：最终交付材料整理

阶段二所有实验与工程任务已完成。当前默认不继续新实验，优先任务为：

1. 整理最终报告和演示材料（如需要）。
2. 归档阶段一和阶段二的关键结论、文档、输出产物。
3. 明确后续是进入最终交付还是启动新实验阶段（需用户明确指示）。

> **注意：默认不要启动新实验。如需继续实验（如 gated_fusion、DNN 更充分调参），必须由用户明确要求。**

---

## 3. 当前主数据源：real_raw_5000

### 3.1 数据包基本信息

| 项目                 | 内容                                                                                                    |
| -------------------- | ------------------------------------------------------------------------------------------------------- |
| 数据包名称           | `real_raw_5000`                                                                                       |
| 数据来源             | 抖音网页端公开视频页面                                                                                  |
| 输入采集批次         | 5 个 1000-URL run 合并                                                                                  |
| 输入 run_id          | `20260509_212100`, `20260510_151052`, `20260510_183114`, `20260510_210600`, `20260511_084923` |
| full unique video_id | 5000                                                                                                    |
| low_quality_count    | 0                                                                                                       |
| 正式 raw 数据目录    | `douyin_data_project/data/interim/real_raw_5000/`                                                     |
| 高置信辅助目录       | `douyin_data_project/data/processed/real_raw_5000/`                                                   |
| 交付说明             | `douyin_data_project/data/interim/real_raw_5000/real_raw_5000_delivery.md`                            |

### 3.2 输出 raw 表清单

| 序号 | 文件                                           |  行数 | 用途                 |
| ---: | ---------------------------------------------- | ----: | -------------------- |
|    1 | `raw_video_detail_real_raw_5000.csv`         |  5000 | 主视频详情表         |
|    2 | `raw_author_real_raw_5000.csv`               |  3558 | 作者信息表           |
|    3 | `raw_music_real_raw_5000.csv`                |  5000 | 音乐信息表           |
|    4 | `raw_hashtag_real_raw_5000.csv`              | 10097 | 话题标签表           |
|    5 | `raw_video_tag_real_raw_5000.csv`            |     0 | 平台标签表，当前为空 |
|    6 | `raw_video_media_real_raw_5000.csv`          |  5000 | 媒体元信息表         |
|    7 | `raw_video_status_control_real_raw_5000.csv` |  5000 | 状态权限表           |
|    8 | `raw_chapter_real_raw_5000.csv`              |     0 | 章节表，当前为空     |
|    9 | `raw_comment_real_raw_5000.csv`              |  9334 | 评论明细表           |
|   10 | `raw_related_video_real_raw_5000.csv`        | 22657 | 相关推荐边表         |
|   11 | `raw_crawl_log_real_raw_5000.csv`            |  5000 | 采集日志表           |

### 3.3 质量摘要

| 指标                              |          值 |
| --------------------------------- | ----------: |
| full unique video_id              |        5000 |
| create_time 覆盖率                |        100% |
| duration_ms 覆盖率                |       73.0% |
| digg/comment/share/collect 覆盖率 |    约 72.9% |
| exact + high                      | 3493，69.9% |
| none + low                        | 1507，30.1% |
| 跨表关联                          |        100% |
| hashtag 触发率                    |       50.8% |
| comment 触发率                    |       37.5% |
| related_video 触发率              |       45.3% |

### 3.4 high-confidence 辅助文件

高置信辅助文件位于：

```text
douyin_data_project/data/processed/real_raw_5000/
```

包括：

| 文件                                                 | 说明                             |
| ---------------------------------------------------- | -------------------------------- |
| `high_confidence_web_video_meta_real_raw_5000.csv` | exact + high 样本 meta，3493 行  |
| `high_confidence_video_ids.txt`                    | 3493 个 high-confidence video_id |
| `high_confidence_filter_report.json`               | 过滤统计报告                     |

该目录仅用于质量筛选和子集实验。默认标准输入仍然是 `douyin_data_project/data/interim/real_raw_5000/` 下的 11 张 raw 表。

---

## 4. 标签与泄漏控制口径

当前没有真实曝光、点击、完播、转化、留存标签。

当前可构造的离线标签仍是互动代理标签，例如：

```text
interaction_score = digg_count + comment_count + share_count + collect_count
label = interaction_score 的分位数二分类标签
```

必须明确：该 label 不是 CTR、CVR、完播率、转化率或真实推荐收益标签。

### 4.1 严禁进入模型输入的字段

以下字段只允许用于构造 `interaction_score` 和 `label`：

```text
digg_count
comment_count
share_count
collect_count
```

label 生成完成后，它们必须从最终建模输入中删除。

硬约束：

1. 这 4 个字段不得出现在 `tabular_train.csv`、`tabular_val.csv`、`tabular_test.csv` 中。
2. 这 4 个字段不得出现在 `tabular_feature_info.json` 的模型特征列表中。
3. 这 4 个字段不得进入 graph `node_features.npy`。
4. 这 4 个字段不得进入 multimodal `structured_features`。
5. 这 4 个字段不得进入任何模型的 `feature_config_used.json`。
6. `interaction_score` 不得作为模型输入特征。
7. 构建脚本必须生成 `leakage_check_report.json`。
8. 如发现泄漏字段进入最终 CSV 或 feature_info，脚本必须报错中止。

---

## 5. 当前建议开发顺序

### 5.1 已完成

- `real_raw_5000` 数据读取适配 — ✅
- `real_raw_5000` tabular 输入构建（no-leakage 强制） — ✅
- `real_raw_5000` graph 输入构建（no-leakage 强制） — ✅
- `real_raw_5000` multimodal 输入构建（no-leakage 强制） — ✅
- Multimodal tuned 迁移验证 — ✅
- DNN / Wide & Deep / GraphSAGE / Multimodal 统一训练入口 — ✅
- 统一调参入口 tune.py 第一版（DNN random search） — ✅
- Multimodal 模态消融实验（Batch 14C）— ✅
- Multimodal categorical embedding 接入与多 seed 验证（Batch 14F/14G）— ✅
- Multimodal 文本分支探索（fieldwise/dim64，Batch 14K）— ✅
- Multimodal late_fusion 实现与 3 seed 验证（Batch 14L）— ✅
- Multimodal 改进阶段总结文档（Batch 14L-summary）— ✅

### 5.2 阶段二（线上模拟与 A/B + 模型改进）— ✅ 已完成

阶段二已完整结束，涵盖：

| 方向 | 状态 |
|------|:----:|
| FastAPI online simulation service 扩展设计（Batch 12B） | ✅ |
| A/B 测试方案设计（Batch 12C） | ✅ |
| FastAPI online simulation service 扩展实现（Batch 12D） | ✅ |
| A/B 日志与指标计算（Batch 12E） | ✅ |
| A/B 测试实施方案完善（Batch 12F） | ✅ |
| DNN 多 seed 验证与 20-trial 调参（Batch 13A–13C） | ✅ |
| Multimodal 改进（消融 → categorical → 文本 → late_fusion）（Batch 14A–14L-summary） | ✅ |
| 阶段二交付总结文档（Batch 15A） | ✅ |

阶段二交付文档：`docs/stage2_online_ab_model_improvement_delivery.md`

### 5.3 工程化阶段（已完成）

端到端推荐流水线工程化阶段已完成：

1. 模块化数据预处理与特征工程脚本 — ✅（现有 build_tabular/build_graph/build_multimodal）
2. 可配置、可复现训练入口 — ✅（统一训练入口 train.py）
3. REST API 推理服务 — ✅（本地模拟服务，尚未部署上线）
4. 推理 batch 入口 — ✅（batch_predict.py）
5. Pipeline orchestrator — ✅（第一版，仅 DNN）
6. 统一调参入口 — ✅（tune.py 第一版，仅 DNN）
7. 推理 benchmark 与模型压缩 — ✅（已验证 dynamic_quantization 不推荐）
8. 流水线设计文档和系统架构文档 — ✅

### 5.4 当前阶段：最终交付材料整理

阶段一（数据 + 工程化）和阶段二（线上模拟/A/B + 模型改进）已全部结束。当前默认不启动新实验，优先任务：

1. **整理最终报告/演示材料**（如需要，由用户明确指示）。
2. **归档关键结论**：README.md 已同步，CLAUDE.md 已同步。
3. **等待用户指示**：后续方向（最终交付 / 继续实验 / 生产部署）由用户决定。

> **非用户明确要求，不启动任何新实验、新代码、新运行。**

---

## 6. 项目目录结构

```text
RA_PTA/
├── README.md
├── CLAUDE.md
├── development_log.md
├── configs/
│   ├── common/
│   │   ├── real_raw_5000.yaml
│   │   └── feature_tabular_real_raw.yaml
│   ├── dnn/
│   ├── wide_deep/
│   ├── graphsage/
│   ├── multimodal/
│   ├── experiments/          # 通用实验配置文件（数据集无关）
│   ├── ab_test/
│   └── datasets.yaml         # 数据集中心注册表
├── data/
│   ├── features/
│   │   └── real_raw_5000/
│   ├── graph/
│   │   └── real_raw_5000/
│   └── multimodal/
│       └── real_raw_5000/
├── douyin_data_project/
│   └── data/
│       ├── interim/
│       │   └── real_raw_5000/
│       └── processed/
│           └── real_raw_5000/
├── src/
│   ├── data/
│   │   ├── load_raw.py
│   │   ├── build_tabular.py
│   │   ├── build_graph_real_raw.py
│   │   └── build_multimodal_real_raw.py
│   ├── features/
│   ├── models/
│   ├── optimization/            # 模型压缩与推理 benchmark
│   │   ├── benchmark.py         # 推理耗时/吞吐/score 一致性测试
│   │   └── compress.py          # 模型压缩（当前仅 DNN dynamic_quantization）
│   ├── evaluation/
│   ├── experiment/
│   └── pipeline/
├── outputs/
├── reports/
└── docs/
```

正式训练、评估、对比和 A/B 模拟入口必须放在 `src/` 下。`notebooks/` 只能用于人工 sanity check。

---

## 7. 主要运行入口

### 数据读取与特征构建

| 脚本                                           | 用途                                  |
| ---------------------------------------------- | ------------------------------------- |
| `src/data/load_raw.py`                   | 统一数据读取入口（通过 --dataset 指定数据集）            |
| `src/data/build_tabular.py`              | 统一 tabular 数据集构建（no_interaction_leakage 强制） |
| `src/data/build_graph_real_raw.py`       | no-leakage 图节点、边、特征构建                         |
| `src/data/build_multimodal_real_raw.py`  | no-leakage 多模态数据集构建                             |

### 模型训练与评估（统一训练入口）

推荐使用统一训练入口 `src/training/train.py`：

| 命令 | 用途 |
|------|------|
| `python src/training/train.py --dataset real_raw_5000 --model dnn` | DNN 训练与评估 |
| `python src/training/train.py --dataset real_raw_5000 --model wide_deep` | Wide & Deep 训练与评估 |
| `python src/training/train.py --dataset real_raw_5000 --model graphsage` | GraphSAGE 训练与评估 |
| `python src/training/train.py --dataset real_raw_5000 --model multimodal` | Multimodal 训练与评估 |

支持 `--dry-run` 预览 resolved config：

```bash
python src/training/train.py --dataset real_raw_5000 --model dnn --dry-run
python src/training/train.py --dataset real_raw_5000 --model multimodal --dry-run
```

支持 `--override` 临时覆盖超参数：

```bash
python src/training/train.py --dataset real_raw_5000 --model dnn --override epochs=30
```

底层模型入口（`src/models/<model>/train.py`）仍保留，但推荐使用统一训练入口。

#### 模型默认配置

| 模型 | 默认配置路径 |
|------|-------------|
| DNN | `configs/models/dnn.yaml` |
| Wide & Deep | `configs/models/wide_deep.yaml` |
| GraphSAGE | `configs/models/graphsage.yaml` |
| Multimodal | `configs/models/multimodal.yaml` |

数据集注册：`configs/datasets.yaml`
Resolved config 自动输出到 `outputs/training_configs/<model>/<dataset>/<timestamp>_resolved.yaml`。

### 调参入口（第一版）

统一超参调优入口 `src/training/tune.py`，第一版只支持 DNN + real_raw_5000 随机搜索：

| 命令 | 用途 |
|------|------|
| `python src/training/tune.py --dataset real_raw_5000 --model dnn --num-trials 3 --dry-run` | Dry-run：只生成 trial 配置，不训练 |
| `python src/training/tune.py --dataset real_raw_5000 --model dnn --num-trials 3` | 小规模真实搜索 |
| `python src/training/tune.py --dataset real_raw_5000 --model dnn --num-trials 20` | 完整搜索（需确认耗时） |

默认搜索空间配置：`configs/tuning/dnn_random.yaml`

**说明：**
- 第一版只支持 `model=dnn` 和 `dataset=real_raw_5000`。
- 调参会重复训练并生成多个 `outputs/dnn/real_raw_5000/<run_id>/`。
- best trial 选择基于 val 指标，test 指标只记录。
- Wide & Deep / GraphSAGE / Multimodal 调参暂未接入。

### Optimization 入口

| 命令 | 用途 |
|------|------|
| `python src/optimization/benchmark.py --model dnn --dataset real_raw_5000 --run-id <run_id> --input data/features/real_raw_5000/tabular_test.csv --device cpu --num-warmup 3 --num-repeat 10` | 原始 DNN 模型 CPU benchmark |
| `python src/optimization/compress.py --model dnn --dataset real_raw_5000 --run-id <run_id> --method dynamic_quantization --device cpu` | DNN 动态量化压缩（dry-run 加 `--dry-run`） |
| `python src/optimization/benchmark.py --model dnn --dataset real_raw_5000 --run-id <run_id> --input data/features/real_raw_5000/tabular_test.csv --device cpu --num-warmup 3 --num-repeat 10 --compressed-dir <compression_output_dir>` | 压缩模型 benchmark（与原模型自动对比） |

**说明：**
- 第一版只支持 `model=dnn` 和 `dataset=real_raw_5000`。
- dynamic_quantization 只支持 CPU，不支持 GPU。
- 压缩模型输出到独立目录，不覆盖原始 model.pt。
- 当前 DNN baseline 不推荐使用 dynamic_quantization（已验证变慢约 6 倍）。
- 详细使用说明见 `docs/optimization_usage.md`。

### 实验入口

| 脚本                                            | 用途                               |
| ----------------------------------------------- | ---------------------------------- |
| `src/experiment/run_comparison.py`             | 统一多模型对比实验                 |
| `src/experiment/run_ab_simulation.py`          | 离线 A/B 模拟或分组统计            |
| `src/experiment/search_multimodal.py`          | Multimodal 随机搜索（通用数据集入口） |
| `src/experiment/tune_multimodal_random_search.py` | Multimodal 随机搜索（旧版，数据集专属） |

### 后续流水线入口

| 脚本                               | 用途                  |
| ---------------------------------- | --------------------- |
| `src/pipeline/preprocess.py`     | 数据预处理入口        |
| `src/pipeline/build_features.py` | 特征工程入口          |
| `src/pipeline/train.py`          | 可配置训练入口        |
| `src/pipeline/evaluate.py`       | 评估入口              |
| `src/pipeline/predict.py`        | 批量推理入口          |
| `src/pipeline/serve.py`          | REST API 推理服务入口 |

---

## 7.5 配置化入口原则

后续新增数据集时，优先通过配置而非新增脚本来接入。

### 7.5.1 数据集中心注册表

`configs/datasets.yaml` 是所有数据集的单一事实来源。每个数据集在此注册
其 multimodal npz 产物路径、feature_info 路径等。新增数据集时只需更新此文件。

### 7.5.2 通用实验入口

`src/experiment/search_multimodal.py` 是 Multimodal 随机搜索的通用入口。
它从 `configs/datasets.yaml` 读取数据集路径，从 `configs/experiments/multimodal_search.yaml`
读取搜索空间和实验参数，不再需要为每个数据集创建专属调优配置或脚本。

### 7.5.3 禁止行为

- 禁止为每个新增数据集复制一套同构的 build_*_real_raw_X000.py 脚本。
- 禁止为每个新增数据集创建专属的 search/调优脚本。
- 如果现有通用脚本无法支持，必须说明为什么无法通过配置解决。
- 仅当数据集的输入数据结构与现有数据集有本质差异时（如新增模态），
  才允许新增数据集专属脚本。

### 7.5.4 如何接入新数据集

1. 在 `configs/datasets.yaml` 中注册新 dataset_name + variant。
2. 确保 multimodal npz 产物路径存在（由 build_* 脚本生成）。
3. 运行通用实验入口，指定 dataset_name：
   ```
   D:/CodeData/software/Anaconda/Anaconda3/envs/ra/python.exe \\
       src/experiment/search_multimodal.py \\
       --config configs/experiments/multimodal_search.yaml \\
       --num-trials 20
   ```
4. 在实验配置 `configs/experiments/multimodal_search.yaml` 中修改
   `dataset_name` 和 `dataset_variant` 以指向新数据集。

### 7.6 Pipeline Orchestrator（第一版）

轻量端到端 pipeline orchestrator，串联 DNN 主线四个阶段：

```bash
# 预览完整 DNN 主线命令，不执行
python src/pipeline/run_pipeline.py --dataset real_raw_5000 --model dnn --dry-run

# 使用已有 DNN run 做 batch inference
python src/pipeline/run_pipeline.py --dataset real_raw_5000 --model dnn --steps infer --run-id 202605132017

# 全流程命令（会重新训练，谨慎执行）
python src/pipeline/run_pipeline.py --dataset real_raw_5000 --model dnn --steps load,tabular,train,infer
```

**说明：**

- 第一版只支持 `dataset=real_raw_5000` 和 `model=dnn`。
- `--dry-run` 只打印命令，不执行。
- `--steps infer` 必须提供 `--run-id`，除非同一次 pipeline 包含 train 阶段。
- pipeline 不自动启动 REST API，只打印启动命令。
- 如果需要启动服务，手动执行：
  ```bash
  python src/serving/api.py --model dnn --dataset real_raw_5000 --run-id <run_id>
  ```

**后续调参、压缩、benchmark 规划：**

| 阶段 | 入口 | 当前状态 |
|------|------|----------|
| 统一调参 | `src/training/tune.py` | ✅ 第一版已完成（仅支持 DNN） |
| 模型压缩 | `src/optimization/compress.py` | ✅ 第一版已完成（仅支持 DNN dynamic_quantization） |
| 推理 benchmark | `src/optimization/benchmark.py` | ✅ 已完成，支持 DNN 原模型和压缩模型 benchmark |

**当前 optimization 关键结论：**

| 指标 | 值 |
|------|-----|
| 原模型大小 | 0.0967 MB（DNN baseline，run_id=202605132017） |
| 压缩后大小 | 0.0851 MB |
| 体积缩减 | 12.0% |
| 原模型 CPU 推理均值 | 0.19 ms |
| 压缩模型 CPU 推理均值 | 1.15 ms |
| 加速比 | 0.165x（压缩后约慢 6 倍） |
| max_abs_score_diff | 0.1216 |
| 结论 | **dynamic_quantization 不推荐用于当前 DNN baseline** |

**说明：** 对于 DNN baseline（24K 参数，0.097 MB），PyTorch dynamic_quantization 的运行时开销（packing/unpacking）超过了小模型的计算节省，导致推理变慢约 6 倍。体积收益仅 12%。压缩模型不得覆盖原始 model.pt。

### 7.7 主要文档索引

| 文档 | 路径 | 说明 |
|------|------|------|
| 系统架构文档 | `docs/system_architecture.md` | 项目整体模块结构、数据流转、各层职责 |
| Pipeline 设计文档 | `docs/batch8_pipeline_design.md` | Pipeline orchestrator 设计说明 |
| Pipeline 使用说明 | `docs/pipeline_usage.md` | Pipeline orchestrator 使用指南 |
| 调参使用说明 | `docs/tuning_usage.md` | 统一调参入口 tune.py 使用指南 |
| Optimization 使用说明 | `docs/optimization_usage.md` | 推理 benchmark 与模型压缩使用指南 |
| REST API 设计文档 | `docs/batch7_rest_api_design.md` | 服务模拟层设计说明 |

---

## 8. 统一评估口径

多模型对比实验优先统一以下指标：

### 分类指标

- AUC
- Accuracy
- Precision
- Recall
- F1

### 排序指标

- Precision@K
- Recall@K

### 损失与样本统计

- loss
- num_samples
- num_positive
- num_negative
- warnings

如果某个模型无法计算某项指标，对应值可为 `null`，但必须在 `warnings` 中说明。

---

## 9. 统一输出格式

每个模型训练评估完成后，至少输出：

```text
outputs/<model_name>/<dataset_name>/<run_id>/metrics.json
outputs/<model_name>/<dataset_name>/<run_id>/predictions_val.csv
outputs/<model_name>/<dataset_name>/<run_id>/predictions_test.csv
outputs/<model_name>/<dataset_name>/<run_id>/train_log.csv
outputs/<model_name>/<dataset_name>/<run_id>/model.pt
outputs/<model_name>/<dataset_name>/<run_id>/run_meta.json
outputs/<model_name>/<dataset_name>/<run_id>/feature_config_used.json
```

其中 `<model_name>` 固定为：

```text
dnn
wide_deep
graphsage
multimodal
```

`predictions_val.csv` 和 `predictions_test.csv` 至少包含：

```text
video_id 或 sample_id
label
score
pred
split
model_name
dataset_name
run_id
```

`metrics.json` 至少包含：

```text
model_name
run_id
dataset_name
val_metrics
test_metrics
warnings
```

---

## 10. 当前已知限制

1. 数据来自公开网页端，不代表平台内部完整数据。
2. 当前没有真实曝光、点击、完播、转化、留存标签。
3. 当前 label 是由公开互动统计构造的离线代理标签。
4. 历史 real_raw_1000 实验存在标签构造字段进入模型输入的风险。
5. `raw_video_tag` 当前为空。
6. `raw_chapter` 当前为空。
7. 评论和相关推荐不是每个视频都触发。
8. none/low confidence 样本约 30.1%。
9. 所有模型结果仍属于离线实验结果，不代表线上业务收益。
10. 在完成复验前，不应将模型效果固化进端到端推荐流水线。

---

## 13. 历史产物说明

### 13.1 real_raw_1000 历史实验产物

| 模块          | Run ID       | 输出目录                                           |
| ------------- | ------------ | -------------------------------------------------- |
| DNN           | 202605081636 | `outputs/dnn/real_raw_1000/202605081636/`        |
| Wide & Deep   | 202605081746 | `outputs/wide_deep/real_raw_1000/202605081746/`  |
| GraphSAGE     | 202605081828 | `outputs/graphsage/real_raw_1000/202605081828/`  |
| Multimodal    | 202605081927 | `outputs/multimodal/real_raw_1000/202605081927/` |
| 统一对比实验  | 202605082029 | `outputs/comparison/202605082029/`               |
| 离线 A/B 模拟 | 202605082129 | `outputs/ab_test/real_raw_1000/202605082129/`    |
| 报告          | —           | `reports/model_comparison_report.md`             |

这些产物用于历史对照。由于已发现标签构造字段泄漏风险，后续结论应以 `real_raw_5000` 复验为准。

### 13.2 real_raw_5000 当前数据产物

| 产物                     | 路径                                                                         |
| ------------------------ | ---------------------------------------------------------------------------- |
| 正式 raw 数据包          | `douyin_data_project/data/interim/real_raw_5000/`                          |
| high-confidence 辅助文件 | `douyin_data_project/data/processed/real_raw_5000/`                        |
| 交付说明                 | `douyin_data_project/data/interim/real_raw_5000/real_raw_5000_delivery.md` |

---

## 14. 下一阶段产出要求

在完成 `real_raw_5000` 复验后，进入流水线工程化阶段，产出包括：

1. 端到端推荐流水线代码； ✅ 已完成（第一版）
2. 模块化数据预处理与特征工程脚本；
3. 支持参数配置与可复现训练过程的训练脚本； ✅ 已完成（统一训练入口）
4. 批量推理入口； ✅ 已完成（batch_predict）
5. REST API 推理服务，模拟线上推荐流程； ✅ 已完成（serving/api）
6. 流水线设计文档； ✅ 已完成（docs/batch8_pipeline_design.md）
7. 使用说明文档； ✅ 已完成（docs/pipeline_usage.md）
8. 系统架构文档。 ✅ 已完成（docs/system_architecture.md）

在完成去泄漏复验前，不要急于固化 pipeline。
