# RA_PTA W4 模型实验项目说明

## 1. 当前任务概述

本目录用于承接推荐算法项目第 4 周（W4）的模型搭建与对比实验工作。当前目标不是立即完成正式大规模实验，而是**先基于当前样本数据跑通多模型流程**，建立统一的数据输入、训练、评估、对比、A/B 模拟和实验记录框架。

当前 W4 的工作内容包括：

- 实现 DNN 模型和 Wide & Deep 模型，重点关注特征交叉处理和深层结构设计
- 简单实现图神经网络模型（GraphSAGE），用于捕捉图结构关系
- 实现最小可运行的多模态模型，融合文本、媒体元信息和结构化特征
- 搭建离线对比实验，统一输入数据、切分方式、评估指标与结果输出格式
- 模拟简单 A/B 测试分组逻辑和指标统计方法
- 汇总对比各模型优缺点，并提出后续改进假设

当前阶段的核心目标是：

1. 先把 W4 所需的模型与实验流程跑通
2. 建立统一规范的代码目录、配置方式、输出方式和开发日志
3. 后续再逐步从样本数据切换到更真实、更大规模的数据

必须注意：当前所有实验结果只能表述为“流程级验证结果”，不能表述为“正式推荐系统效果结论”。

---

## 2. 当前使用的数据

当前用于 W4 的数据并不是正式完整 raw 数据，而是**流程验证用样本数据**，路径如下：

`D:/CodeData/Program Coding/ByteDance/RA_PTA/douyin_data_project/data/sample0427/`

其中最重要的说明文件是：

`D:/CodeData/Program Coding/ByteDance/RA_PTA/douyin_data_project/data/sample0427/sample_data_dictionary.md`

使用数据前必须明确以下事实：

- `sample0427` 是用于 W4 流程验证的样本数据
- 它和正式 `data_dictionary.md` 不是严格等价
- 部分字段来自规则补齐、结构模拟或样本级占位
- 当前数据适合做模型输入输出、训练流程、实验框架、图构造、多模态融合等“流程验证”
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

如果涉及字段含义、字段可用性、规则生成字段、字符串化 JSON/ARRAY 字段，必须以 `sample_data_dictionary.md` 为准。

---

## 3. 建议的项目目录结构

建议在 `D:/CodeData/Program Coding/ByteDance/RA_PTA/` 下按如下结构组织 W4 项目：

```text
RA_PTA/
├── README.md
├── CLAUDE.md
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
│   │   ├── run_dnn.py
│   │   ├── run_wide_deep.py
│   │   ├── run_graphsage.py
│   │   ├── run_multimodal.py
│   │   ├── run_all_experiments.py
│   │   ├── run_comparison.py
│   │   └── run_ab_simulation.py
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
├── notebooks/
│   ├── sanity_check.ipynb
│   ├── graph_debug.ipynb
│   └── result_analysis.ipynb
└── reports/
    ├── experiment_report.md
    ├── comparison_report.md
    └── ab_test_report.md
```

命名约定：

- `src/data/build_tabular_dataset.py` 负责生成表格模型可直接使用的数据集文件
- `src/data/build_graph_dataset.py` 负责生成图节点、边、节点特征、标签等图数据集文件
- `src/data/build_multimodal_dataset.py` 负责生成多模态模型输入文件
- `src/features/` 下的脚本只放可复用特征函数，不作为正式实验主入口
- `src/experiment/` 下的脚本才是正式训练、评估、对比和 A/B 模拟入口

---

## 4. 目录设计说明

### 4.1 根目录文件

#### `README.md`
当前项目总说明文件。必须保持更新，说明当前任务、目录结构、数据来源、模型边界、运行方式和实验约定。

#### `CLAUDE.md`
提供给智能体的工作约束文件。每次开始任务前都应先阅读。

#### `development_log.md`
开发日志。每次运行、修改、实验或新增脚本后都要维护，保证可追溯。

---

### 4.2 `configs/`

用于保存统一配置文件，避免把路径、超参数、实验选项写死在代码里。

建议分为：

- `common/`：公共路径、切分、评估指标
- `dnn/`：DNN 模型配置
- `wide_deep/`：Wide & Deep 配置
- `graphsage/`：GraphSAGE 配置
- `multimodal/`：多模态模型配置
- `ab_test/`：A/B 模拟配置

---

### 4.3 `data/`

用于承接 W4 自己的数据中间产物，不直接修改 `douyin_data_project` 原始文件。

建议约定：

- `external/`：外部预训练模型、静态资源等。当前阶段默认不使用外部下载资源
- `interim/`：中间处理结果
- `processed/`：清洗后可直接训练的数据
- `features/`：表格特征输入
- `graph/`：图节点、边、图特征
- `multimodal/`：文本、媒体、结构化融合输入
- `experiment_inputs/`：统一实验输入快照

---

### 4.4 `src/data/`

负责读取 `sample0427`、做 schema 对齐检查、构建 tabular / graph / multimodal 输入数据。

必须做到：

- 不直接修改原始 sample0427 文件
- 所有派生产物写入当前项目自己的 `data/` 目录
- 所有输入路径可由配置文件管理
- CSV 中字符串化的 list、JSON、ARRAY 字段要按存储类型和逻辑类型分别处理
- `*_raw` 字段原则上保留原始值，不要在数据读取层直接覆盖

---

### 4.5 `src/features/`

负责 W4 所需特征处理，不同模型可复用。

至少应包括：

- 表格特征
- 交叉特征
- 文本特征
- 视觉或媒体元信息特征
- 图节点/边特征

---

### 4.6 `src/models/`

每个模型独立一个子目录，不要混写。

要求：

- 每个模型目录至少有 `model.py`、`train.py`、`evaluate.py`
- 如需要推理脚本，可加 `predict.py`
- GraphSAGE 的图数据构建逻辑放在 `src/data/build_graph_dataset.py`，模型目录只保留模型、训练、评估、预测逻辑
- 多模态的数据集构建逻辑放在 `src/data/build_multimodal_dataset.py`，模型目录保留 encoder、fusion、训练、评估、预测逻辑

---

### 4.7 `src/evaluation/`

负责统一评估逻辑。

必须统一：

- 分类指标
- 排序指标
- 多模型对比逻辑
- 图表输出逻辑
- 报告拼装逻辑

不要把评估代码散落在各模型训练脚本里。

---

### 4.8 `src/experiment/`

负责主程序入口。

建议至少有：

- `run_dnn.py`
- `run_wide_deep.py`
- `run_graphsage.py`
- `run_multimodal.py`
- `run_all_experiments.py`
- `run_comparison.py`
- `run_ab_simulation.py`

其中：

- `run_all_experiments.py` 用于串联所有模型实验，但只有在单模型流程全部跑通后再使用
- `run_comparison.py` 用于统一读取各模型结果并出对比报告
- `run_ab_simulation.py` 用于做简单 A/B 分组模拟和指标统计

---

### 4.9 `outputs/`

按模型和任务分类保存输出。

至少要分：

- 数据检查输出目录
- 每个模型自己的输出目录
- 对比实验目录
- A/B 模拟目录
- 图表目录

不要把所有结果直接堆在一个目录里。

---

### 4.10 `notebooks/`

`notebooks/` 仅用于人工 sanity check、调试和结果查看，不作为正式训练、评估、对比或 A/B 模拟入口。正式流程必须通过 `src/experiment/` 下的脚本执行。

---

### 4.11 `reports/`

用于保存汇总后的 Markdown 报告或分析报告。

建议至少有：

- `experiment_report.md`
- `comparison_report.md`
- `ab_test_report.md`

后续周报、答辩材料、阶段总结都可以从这里整理。

---

## 5. 当前阶段建议的开发顺序

考虑到当前使用的是 `sample0427` 样本数据，建议按以下顺序推进。每次交给智能体时只做其中一个窄任务，不要合并成大任务。

1. 工程目录初始化与配置文件检查
2. `sample0427` 统一读取
3. schema 校验与数据概览
4. tabular 数据集构建
5. DNN 最小训练、评估、预测闭环
6. Wide & Deep 最小训练、评估、预测闭环
7. GraphSAGE 图数据构建
8. GraphSAGE 最小训练、评估、预测闭环
9. 多模态数据集构建
10. 多模态模型最小训练、评估、预测闭环
11. 统一离线对比实验
12. A/B 模拟
13. W4 最终报告整理

### 5.1 表格模型优先

先做 DNN 和 Wide & Deep，原因是：

- 实现成本低
- 可以较快验证特征输入、标签构造、评估流程是否正常
- 后续 GraphSAGE 和多模态也可以复用 tabular 阶段的标签和部分特征

### 5.2 GraphSAGE

GraphSAGE 建议基于以下表构造简化图：

- `raw_video_detail`
- `raw_author`
- `raw_hashtag`
- `raw_music`
- `raw_video_tag`
- `raw_related_video`

先构造一个以视频节点为核心的简化图，完成最小可运行版本。不要一开始追求复杂异构图建模。

### 5.3 多模态模型

当前多模态阶段默认只使用本地已有字段和轻量特征：

- 文本：视频描述、章节文本、评论文本的统计特征或轻量向量
- 媒体：封面 URL 是否存在、媒体字段统计、视频尺寸、时长等元信息
- 结构化：复用 tabular 特征

除非用户明确要求，不联网下载图片，不调用外部 API，不新增大型预训练模型依赖，不默认引入 CLIP、ResNet、BERT 等重依赖。

---

## 6. 当前推荐的统一评估口径

当前样本数据主要用于流程验证，因此建议所有模型优先统一以下指标。

### 6.1 分类指标

- AUC
- Accuracy
- Precision
- Recall
- F1

### 6.2 排序指标

- Precision@K
- Recall@K

### 6.3 图模型与多模态模型

如果最终仍输出二分类分数或排序分数，也应优先保持与上面指标一致，方便横向比较。

### 6.4 A/B 模拟

建议至少统计：

- 分组样本数
- 平均预测分
- 平均标签值
- Top-K 命中情况
- 简单 lift

如果样本太小、标签单一或某些指标无法计算，必须在 `metrics.json`、报告和 `development_log.md` 中记录 warning，不要静默跳过。

---

## 7. 统一输出格式要求

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

## 8. 当前重要约束

1. 当前使用的数据是 `sample0427`，不是正式真实数据全集
2. `sample0427` 主要用于 W4 流程验证，不用于最终强结论
3. 每次运行前必须先阅读 `README.md`
4. 每次运行后必须维护 `development_log.md`
5. 不要直接修改 `douyin_data_project` 下的原始 sample0427 文件
6. 所有新产生的数据、特征、图、实验结果都写在本项目目录下
7. 各模型必须分目录实现，不要混成一个大脚本
8. 对比实验必须有单独主程序，不要人工拼结果
9. A/B 模拟必须有单独主程序，不要散落在 notebook 里
10. 当前目标是先把 W4 流程跑通，再逐步替换成真实数据
11. 每次任务只做用户本次明确要求的模块，不得顺手扩展到其他模型、其他实验、其他数据源或其他目录
12. 如发现相关问题，可以在最终汇报的“下一步建议”中提出，但不要擅自实现

---

## 9. 当前推荐的首批待实现脚本

建议优先落地以下脚本。

### 9.1 数据层

- `src/data/load_sample0427.py`
- `src/data/validate_schema.py`
- `src/data/build_tabular_dataset.py`
- `src/data/build_graph_dataset.py`
- `src/data/build_multimodal_dataset.py`

### 9.2 特征层

- `src/features/tabular_features.py`
- `src/features/cross_features.py`
- `src/features/text_features.py`
- `src/features/image_features.py`
- `src/features/graph_features.py`

### 9.3 模型层

- `src/models/dnn/model.py`
- `src/models/wide_deep/model.py`
- `src/models/graphsage/model.py`
- `src/models/multimodal/fusion_model.py`

### 9.4 实验入口

- `src/experiment/run_dnn.py`
- `src/experiment/run_wide_deep.py`
- `src/experiment/run_graphsage.py`
- `src/experiment/run_multimodal.py`
- `src/experiment/run_comparison.py`
- `src/experiment/run_ab_simulation.py`

---

## 10. 当前阶段的正确口径

当前阶段不应对外表述为：

- 已完成正式推荐系统实验
- 已得到可靠的真实效果对比
- 已验证模型在线收益

当前更准确的表述应是：

- 已进入 W4 多模型流程搭建阶段
- 当前基于 `sample0427` 样本数据进行流程级验证
- 目标是完成 DNN / Wide & Deep / GraphSAGE / 多模态 / 离线对比 / A/B 模拟的最小可运行原型
- 后续会基于更真实、更完整的数据继续完善实验

---

## 11. 开发日志要求

项目根目录必须维护：

`D:/CodeData/Program Coding/ByteDance/RA_PTA/development_log.md`

每次开发后至少记录：

- 日期时间
- 本次任务目标
- 修改文件列表
- 运行命令
- 输入数据路径
- 输出结果路径
- 当前结果
- 是否跑通
- 遇到的问题
- 下一步建议

该日志是当前 W4 阶段的重要追溯文件，必须持续维护。即使本次任务失败，也必须记录失败原因和未完成项。

---

## 12. 运行命令约定

所有 Python 运行命令默认使用以下解释器：

`D:/CodeData/software/Anaconda/Anaconda3/envs/ra/python.exe`

示例：

```text
D:/CodeData/software/Anaconda/Anaconda3/envs/ra/python.exe src/data/load_sample0427.py --config configs/common/data_paths.yaml
```

如果因为环境缺依赖导致运行失败，应记录：

- 失败命令
- 报错摘要
- 缺失依赖或疑似原因
- 建议补充的依赖

不要在没有用户确认的情况下大规模改环境或安装重依赖。
