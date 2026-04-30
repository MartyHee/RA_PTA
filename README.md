# RA_PTA 模型实验项目说明

---

## 1. 当前阶段状态

已完成流程级验证，所有模型与实验流程均已跑通：

- ✅ sample0427 数据读取与 schema 校验
- ✅ tabular 数据集构建
- ✅ graph 数据集构建
- ✅ multimodal 数据集构建
- ✅ DNN 模型（最小训练、评估、预测闭环）
- ✅ Wide & Deep 模型（最小训练、评估、预测闭环）
- ✅ GraphSAGE 模型（最小训练、评估、预测闭环）
- ✅ Multimodal 融合模型（最小训练、评估、预测闭环）
- ✅ 统一模型对比实验（4 模型汇总、质量检查、图表、对比报告）
- ✅ 离线 A/B 模拟（hash 分组、组内指标统计、lift 计算、报告）
- ✅ 最终多模型对比实验报告（`reports/model_comparison_report.md`）

> **重要说明**：当前所有结果基于 sample0427 样本数据（79 条主视频，16 条 eval），仅用于流程级验证。所有指标波动极大，不具备统计显著性。标签为 interaction_score 伪标签，不代表真实 CTR/CVR/完播/留存等业务指标。**不证明任何模型在真实推荐场景下更优，不支持任何线上收益判断。**

---

## 2. 最新关键输出目录

| 模块         | Run ID       | 输出目录                               |
| ------------ | ------------ | -------------------------------------- |
| DNN          | 202604301440 | `outputs/dnn/202604301440/`          |
| Wide & Deep  | 202604301557 | `outputs/wide_deep/202604301557/`    |
| GraphSAGE    | 202604291958 | `outputs/graphsage/202604291958/`    |
| Multimodal   | 202604301557 | `outputs/multimodal/202604301557/`   |
| 统一对比实验 | 202604301609 | `outputs/comparison/202604301609/`   |
| A/B 模拟     | 202604301630 | `outputs/ab_test/202604301630/`      |
| 最终报告     | —           | `reports/model_comparison_report.md` |

各模型输出包含 `metrics.json`、`predictions.csv`、`train_log.csv`、`model.pt`、`run_meta.json`、`feature_config_used.json`。

对比实验输出包含 12 个文件（对比报告、指标汇总、质量检查、Top-K 校验、分数分布、图表）。

A/B 模拟输出包含 7 个文件（分组分配、指标汇总、分数分布、模拟报告）。

---

## 3. 当前推荐阅读顺序

```
1. README.md                           — 项目总览（本文件）
2. reports/model_comparison_report.md  — 最终多模型对比实验报告
3. outputs/comparison/202604301609/model_comparison_report.md  — 对比实验详细报告
4. outputs/ab_test/202604301630/ab_simulation_report.md         — A/B 模拟详细报告
```

---

## 4. 当前使用的数据

当前用于实验的数据并不是正式完整 raw 爬虫数据，而是**流程验证用样本数据**，路径如下：

`D:/CodeData/Program Coding/ByteDance/RA_PTA/douyin_data_project/data/sample0427/`

其中最重要的说明文件是：

`D:/CodeData/Program Coding/ByteDance/RA_PTA/douyin_data_project/data/sample0427/sample_data_dictionary.md`

使用数据前必须明确以下事实：

- `sample0427` 是用于实验流程验证的样本数据
- 它和正式 `data_dictionary.md` 不是严格等价
- 部分字段来自规则补齐、结构模拟或样本级占位
- 当前数据适合做模型输入输出、训练流程、实验框架、图构造、多模态融合等"流程验证"
- 当前数据不适合直接作为正式效果结论的最终依据
- 后续正式实验仍应优先使用真实抓取数据

当前样本目录包含 11 张表：

```text
sample0427_raw_video_detail.csv
sample0427_raw_author.csv
sample0427_raw_music.csv
sample0427_raw_hashtag.csv
sample0427_raw_video_tag.csv
sample0427_raw_video_media.csv
sample0427_raw_video_status_control.csv
sample0427_raw_chapter.csv
sample0427_raw_comment.csv
sample0427_raw_related_video.csv
sample0427_raw_crawl_log.csv
```

---

## 5. 项目目录结构

```text
RA_PTA/
├── README.md
├── development_log.md
├── configs/
│   ├── common/
│   │   ├── data_paths.yaml
│   │   ├── metrics.yaml
│   │   └── split.yaml
│   ├── dnn/
│   │   └── dnn_base.yaml
│   ├── wide_deep/
│   │   └── wide_deep_base.yaml
│   ├── graphsage/
│   │   └── graphsage_base.yaml
│   ├── multimodal/
│   │   └── multimodal_base.yaml
│   └── ab_test/
│       └── ab_base.yaml
├── data/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   ├── features/
│   ├── graph/
│   ├── multimodal/
│   └── experiment_inputs/
├── src/
│   ├── data/
│   │   ├── load_sample0427.py
│   │   ├── validate_schema.py
│   │   ├── build_tabular_dataset.py
│   │   ├── build_graph_dataset.py
│   │   └── build_multimodal_dataset.py
│   ├── features/
│   │   ├── tabular_features.py
│   │   ├── cross_features.py
│   │   ├── text_features.py
│   │   ├── image_features.py
│   │   └── graph_features.py
│   ├── models/
│   │   ├── dnn/
│   │   │   ├── model.py
│   │   │   ├── train.py
│   │   │   ├── evaluate.py
│   │   │   └── predict.py
│   │   ├── wide_deep/
│   │   │   ├── model.py
│   │   │   ├── train.py
│   │   │   ├── evaluate.py
│   │   │   └── predict.py
│   │   ├── graphsage/
│   │   │   ├── model.py
│   │   │   ├── train.py
│   │   │   ├── evaluate.py
│   │   │   └── predict.py
│   │   └── multimodal/
│   │       ├── text_encoder.py
│   │       ├── image_encoder.py
│   │       ├── fusion_model.py
│   │       ├── train.py
│   │       ├── evaluate.py
│   │       └── predict.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── ranking_metrics.py
│   │   ├── compare_models.py
│   │   ├── plot_results.py
│   │   └── report_utils.py
│   ├── experiment/
│   │   ├── run_comparison.py
│   │   ├── run_ab_simulation.py
│   │   └── ab_metrics.py
│   └── utils/
│       ├── seed.py
│       ├── io.py
│       ├── logger.py
│       ├── config.py
│       └── common.py
├── outputs/
│   ├── data_check/
│   ├── dnn/
│   ├── wide_deep/
│   ├── graphsage/
│   ├── multimodal/
│   ├── comparison/
│   ├── ab_test/
│   └── figures/
├── reports/
│   ├── model_comparison_report.md
│   └── figures/
└── scripts/
    └── check_tabular_quality.py
```

---

## 6. 目录设计说明

### 6.1 根目录文件

#### `README.md`

当前项目总说明文件。必须保持更新，说明当前任务、目录结构、数据来源、模型边界、运行方式和实验约定。

#### `development_log.md`

开发日志。每次运行、修改、实验或新增脚本后都要维护，保证可追溯。

---

### 6.2 `configs/`

用于保存统一配置文件，避免把路径、超参数、实验选项写死在代码里。

分为：

- `common/`：公共路径、切分、评估指标
- `dnn/`：DNN 模型配置
- `wide_deep/`：Wide & Deep 配置
- `graphsage/`：GraphSAGE 配置
- `multimodal/`：多模态模型配置
- `ab_test/`：A/B 模拟配置

---

### 6.3 `data/`

用于承接数据中间产物，不直接修改 `douyin_data_project` 原始文件。

约定：

- `external/`：外部预训练模型、静态资源等。当前阶段默认不使用外部下载资源
- `interim/`：中间处理结果
- `processed/`：清洗后可直接训练的数据
- `features/`：表格特征输入
- `graph/`：图节点、边、图特征
- `multimodal/`：文本、媒体、结构化融合输入
- `experiment_inputs/`：统一实验输入快照

---

### 6.4 `src/data/`

负责读取 `sample0427`、做 schema 对齐检查、构建 tabular / graph / multimodal 输入数据。

必须做到：

- 不直接修改原始 sample0427 文件
- 所有派生产物写入当前项目自己的 `data/` 目录
- 所有输入路径可由配置文件管理
- CSV 中字符串化的 list、JSON、ARRAY 字段要按存储类型和逻辑类型分别处理
- `*_raw` 字段原则上保留原始值，不要在数据读取层直接覆盖

---

### 6.5 `src/features/`

负责实验所需特征处理，不同模型可复用。

至少应包括：

- 表格特征
- 交叉特征
- 文本特征
- 视觉或媒体元信息特征
- 图节点/边特征

---

### 6.6 `src/models/`

每个模型独立一个子目录。

要求：

- 每个模型目录至少有 `model.py`、`train.py`、`evaluate.py`
- 如需要推理脚本，可加 `predict.py`
- GraphSAGE 的图数据构建逻辑放在 `src/data/build_graph_dataset.py`，模型目录只保留模型、训练、评估、预测逻辑
- 多模态的数据集构建逻辑放在 `src/data/build_multimodal_dataset.py`，模型目录保留 encoder、fusion、训练、评估、预测逻辑

---

### 6.7 `src/evaluation/`

负责统一评估逻辑。

必须统一：

- 分类指标
- 排序指标
- 多模型对比逻辑
- 图表输出逻辑
- 报告拼装逻辑

不要把评估代码散落在各模型训练脚本里。

---

### 6.8 `src/experiment/`

负责主程序入口。

当前已实现的实验入口：

- `run_comparison.py` — 统一读取各模型结果并出对比报告
- `run_ab_simulation.py` — 简单 A/B 分组模拟和指标统计
- `ab_metrics.py` — A/B 分组与指标统计函数（被 `run_ab_simulation.py` 调用）

---

### 6.9 `outputs/`

按模型和任务分类保存输出。

已产出：

- 数据检查输出目录（`outputs/data_check/`）
- 每个模型自己的输出目录（带 run_id 时间戳子目录）
- 对比实验目录（带 comparison_run_id 时间戳子目录）
- A/B 模拟目录（带 ab_run_id 时间戳子目录）

---

### 6.10 `notebooks/`

`notebooks/` 仅用于人工 sanity check、调试和结果查看，不作为正式训练、评估、对比或 A/B 模拟入口。正式流程必须通过 `src/experiment/` 下的脚本执行。

---

### 6.11 `reports/`

用于保存汇总后的 Markdown 报告或分析报告。

当前已产出：

- `reports/model_comparison_report.md` — 最终多模型对比实验报告
- `reports/figures/` — 报告所用图表

---

## 7. 当前的统一评估口径

当前样本数据主要用于流程验证，所有模型统一以下指标。

### 7.1 分类指标

- AUC
- Accuracy
- Precision
- Recall
- F1

### 7.2 排序指标

- Precision@K
- Recall@K

### 7.3 图模型与多模态模型

如果最终仍输出二分类分数或排序分数，也应优先保持与上面指标一致，方便横向比较。

### 7.4 A/B 模拟

建议至少统计：

- 分组样本数
- 平均预测分
- 平均标签值
- Top-K 命中情况
- 简单 lift

如果样本太小、标签单一或某些指标无法计算，必须在 `metrics.json`、报告和 `development_log.md` 中记录 warning，不要静默跳过。

---

## 8. 统一输出格式

为了后续统一对比实验，每个模型训练评估完成后，至少输出以下文件：

```text
outputs/<model_name>/metrics.json
outputs/<model_name>/predictions.csv
outputs/<model_name>/train_log.csv
outputs/<model_name>/model.pt
```

其中 `<model_name>` 使用以下固定名称：

```text
dnn
wide_deep
graphsage
multimodal
```

`predictions.csv` 至少包含以下列：

```text
sample_id 或 video_id
label
score
pred
split
model_name
```

字段含义：

- `sample_id` 或 `video_id`：样本标识。优先使用 `video_id`
- `label`：流程验证标签
- `score`：模型输出的正类概率或排序分数
- `pred`：按默认阈值产生的预测类别
- `split`：train / eval / test 等数据划分标识
- `model_name`：模型名称，必须与输出目录一致

`metrics.json` 至少包含：

```text
model_name
auc
accuracy
precision
recall
f1
precision_at_k
recall_at_k
num_samples
num_positive
num_negative
warnings
```

如果某个指标无法计算，对应值可以为 `null`，但必须在 `warnings` 中说明原因。

---

## 9. 当前限制

### 9.1 数据限制

1. **sample0427 样本量小**：主视频仅 79 条，train=63，eval=16，指标波动极大。
2. **无独立 test 集**：当前仅有 train/eval 切分，无法做最终泛化评估。
3. **伪标签**：label 为 interaction_score（digg_count+comment_count+share_count+collect_count）60% 分位数构造，不代表真实 CTR/CVR/完播/留存。
4. **无真实推荐标签**：当前无真实曝光、点击、完播、转化、留存标签。
5. **部分字段规则生成**：5 张完全补齐表（raw_video_tag、raw_video_status_control、raw_chapter、raw_comment、raw_related_video）数据不代表真实分布。
6. **27 个全空字段**：这些字段在 schema 中有定义但样本中无数据。

### 9.2 模型限制

7. **GraphSAGE 图关系不代表真实推荐图谱**：图中的 video-author、video-hashtag、video-related_video 等关系来自样本级补齐，不代表真实推荐图谱。
8. **Multimodal 视觉分支仅使用媒体元信息**：当前 visual_features 仅包含封面尺寸、URL 存在性等本地元信息，不是真实图像/视频语义特征。未下载图片、未调用外部 API、未使用大型预训练模型。

### 9.3 实验限制

9. **当前指标不能作为正式业务效果结论**：所有结果基于 sample0427 样本数据，仅用于流程级验证。
10. **A/B 模拟是离线分组统计，不是真实线上 A/B 测试**：两组使用完全相同的模型预测结果，分组仅基于 video_id 的 hash 值，lift 无因果含义。
11. **不证明任何模型效果更优**：当前指标差异可能完全来自样本偏差和随机波动。

---

## 10. 当前主要运行入口

### 数据与特征构建

| 脚本                                     | 用途                  |
| ---------------------------------------- | --------------------- |
| `src/data/load_sample0427.py`          | 统一读取 11 张 CSV 表 |
| `src/data/validate_schema.py`          | Schema 校验与关联检查 |
| `src/data/build_tabular_dataset.py`    | Tabular 数据集构建    |
| `src/data/build_graph_dataset.py`      | 图节点/边/特征构建    |
| `src/data/build_multimodal_dataset.py` | 三模态数据集构建      |

### 模型训练与评估

| 脚本                                                 | 用途                   |
| ---------------------------------------------------- | ---------------------- |
| `src/models/dnn/train.py` / `evaluate.py`        | DNN 训练与评估         |
| `src/models/wide_deep/train.py` / `evaluate.py`  | Wide & Deep 训练与评估 |
| `src/models/graphsage/train.py` / `evaluate.py`  | GraphSAGE 训练与评估   |
| `src/models/multimodal/train.py` / `evaluate.py` | Multimodal 训练与评估  |

### 实验入口

| 脚本                                    | 用途                                                  |
| --------------------------------------- | ----------------------------------------------------- |
| `src/experiment/run_comparison.py`    | 统一对比实验（汇总 4 模型指标、质量检查、图表、报告） |
| `src/experiment/run_ab_simulation.py` | 离线 A/B 模拟（分组、指标、lift、报告）               |

---

## 11. 后续建议

### 数据层面

1. **扩大数据规模**：接入更大规模真实抓取数据，提升指标稳定性。
2. **构造真实标签**：使用曝光、点击、完播、互动、转化等真实业务指标作为标签。
3. **完善评估方式**：建立 train / validation / test 三路切分，或使用交叉验证。

### 模型层面

4. **DNN 优化**：特征筛选、阈值校准、概率校准（Platt scaling / Isotonic regression）。
5. **Wide & Deep 优化**：更稳定的交叉特征设计（如 FM / FMFM 风格交互）、hash trick、正则化。
6. **GraphSAGE 增强**：接入真实用户行为边、作者关系边、视频共现边，减少规则补齐边占比。
7. **多模态增强**：接入封面图、视频帧、OCR、ASR、标题文本等更真实语义，使用预训练视觉编码器。

### 分析与实验层面

8. **分析增强**：增加校准曲线、PR 曲线、AUC 置信区间、特征重要性分析。
9. **在线实验**：设计正式线上或准线上 A/B 实验，明确实验单位、流量分配、核心指标和护栏指标。
10. **护栏指标**：增加推荐多样性、重复率、低质内容比例等非效果类指标。

---
