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

### 2.3 当前下一步：

1. `real_raw_5000` 读取适配；
2. 构建 `real_raw_5000` tabular 数据；
3. 保证 `digg_count/comment_count/share_count/collect_count` 只用于构造 label，不进入任何模型输入；
4. 用 tuned Multimodal 配置做迁移验证；
5. 再视情况重训四类模型并做统一对比。

在标签与特征泄漏问题确认后，进入端到端推荐流水线工程化：

- 设计模块化数据预处理和特征工程脚本
- 实现训练脚本支持参数配置，保证可复现训练过程
- 建立推理服务接口（如 REST API）模拟线上推荐流程
- 编写流水线使用说明和系统架构文档

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

### 5.1 立即任务

1. 新增 `real_raw_5000` 读取配置。
2. 读取并检查 `data/interim/real_raw_5000/` 下的 11 张 raw 表。
3. 构建 `real_raw_5000_no_leakage` tabular 数据集。
4. 严格删除 `digg_count/comment_count/share_count/collect_count` 四个 label source 字段。
5. 生成 `leakage_check_report.json`。
6. 使用 tuned Multimodal 配置先做迁移验证。

### 5.2 实验复验

7. 构建 no-leakage graph 输入。
8. 构建 no-leakage multimodal 输入。
9. 重训 DNN / Wide & Deep / GraphSAGE / Multimodal。
10. 统一多模型对比。
11. 对比 real_raw_1000 历史结果与 real_raw_5000 no-leakage 结果。
12. 判断历史高 AUC 是否主要来自标签构造字段泄漏。

### 5.3 工程化阶段

在完成 no-leakage 复验后，再进入端到端流水线工程化：

1. 设计模块化数据预处理和特征工程脚本；
2. 实现训练脚本支持参数配置，保证可复现训练过程；
3. 建立推理服务接口，例如 REST API，模拟线上推荐流程；
4. 编写流水线使用说明和系统架构文档；
5. 产出端到端推荐流水线代码。

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
│   │   └── feature_tabular_real_raw_5000.yaml
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
│   │   ├── load_real_raw_5000.py
│   │   ├── build_tabular_real_raw_5000.py
│   │   ├── build_graph_real_raw_5000.py
│   │   └── build_multimodal_real_raw_5000.py
│   ├── features/
│   ├── models/
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
| `src/data/load_real_raw_5000.py`             | real_raw_5000 真实 raw 数据读取与检查 |
| `src/data/build_tabular_real_raw_5000.py`    | no-leakage tabular 数据集构建         |
| `src/data/build_graph_real_raw_5000.py`      | no-leakage 图节点、边、特征构建       |
| `src/data/build_multimodal_real_raw_5000.py` | no-leakage 多模态数据集构建           |

### 模型训练与评估

| 脚本                                                 | 用途                   |
| ---------------------------------------------------- | ---------------------- |
| `src/models/dnn/train.py` / `evaluate.py`        | DNN 训练与评估         |
| `src/models/wide_deep/train.py` / `evaluate.py`  | Wide & Deep 训练与评估 |
| `src/models/graphsage/train.py` / `evaluate.py`  | GraphSAGE 训练与评估   |
| `src/models/multimodal/train.py` / `evaluate.py` | Multimodal 训练与评估  |

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

1. 端到端推荐流水线代码；
2. 模块化数据预处理与特征工程脚本；
3. 支持参数配置与可复现训练过程的训练脚本；
4. 批量推理入口；
5. REST API 推理服务，模拟线上推荐流程；
6. 流水线设计文档；
7. 使用说明文档；
8. 系统架构文档。

在完成去泄漏复验前，不要急于固化 pipeline。
