# RA_PTA 多模型对比实验项目说明

## 1. 项目概述

本项目用于基于抖音公开网页端数据进行多模型对比实验。核心工作包括：

- raw 数据读取与质量检查
- tabular / graph / multimodal 输入构建
- DNN / Wide & Deep / GraphSAGE / Multimodal 多模型训练
- 统一离线评估
- 多模型指标对比
- 离线 A/B 模拟或分组统计
- 实验报告整理

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

### 2.1 已完成：sample0427 流程验证

此前已经基于 `sample0427` 跑通过完整流程：

- sample0427 数据读取与 schema 校验
- tabular 数据集构建
- graph 数据集构建
- multimodal 数据集构建
- DNN 模型最小训练、评估、预测闭环
- Wide & Deep 模型最小训练、评估、预测闭环
- GraphSAGE 模型最小训练、评估、预测闭环
- Multimodal 融合模型最小训练、评估、预测闭环
- 统一模型对比实验
- 离线 A/B 模拟
- 多模型对比实验报告

这些结果证明工程链路已经跑通，但由于 sample0427 样本量小且存在规则补齐字段，不能作为真实模型效果结论。

### 2.2 已完成：真实 raw 数据采集

已完成 1000 条公开视频 URL 的真实网页端 raw 数据采集，并形成数据包：

```text
real_raw_1000_20260507_230322
```

当前真实 raw 数据目录：

```text
douyin_data_project/data/interim/20260507_230322/
```

数据包交付说明：

```text
douyin_data_project/data/interim/20260507_230322/real_raw_data_delivery_20260507_230322.md
```

### 2.3 当前下一步

当前即将从 sample0427 切换到 `real_raw_1000`：

1. 新增真实 raw 数据读取配置。
2. 实现 `real_raw_1000` 读取检查。
3. 基于真实 raw 数据重新构建 tabular / graph / multimodal 输入。
4. 重新训练 DNN / Wide & Deep / GraphSAGE / Multimodal。
5. 重新进行多模型对比实验。
6. 生成基于真实 raw 数据的多模型对比报告。

---

## 3. 当前主数据源：real_raw_1000

### 3.1 数据包基本信息

| 项目 | 内容 |
|---|---|
| 数据包名称 | `real_raw_1000_20260507_230322` |
| 采集 run_id | `20260507_230322` |
| 数据来源 | 抖音网页端公开视频页面 |
| 输入 URL 文件 | `configs/batch_urls_1000_from_2000.txt` |
| 输入 URL 数 | 1000 |
| unique video_id | 1000 |
| 成功 URL | 1000 |
| failed URL | 0 |
| 输出目录 | `douyin_data_project/data/interim/20260507_230322/` |
| 交付说明 | `douyin_data_project/data/interim/20260507_230322/real_raw_data_delivery_20260507_230322.md` |
| 采集方式 | Playwright browser 模式，每 URL 独立 page，restart-every=10，workers=1 |

### 3.2 输出文件清单

| 序号 | 文件 | 行数 | 字段数 | 用途 |
|---:|---|---:|---:|---|
| 1 | `real_web_video_meta_20260507_230322.csv` | 1000 | 68 | 完整 meta 大宽表 |
| 2 | `raw_video_detail_20260507_230322.csv` | 1000 | 30 | 主视频详情表 |
| 3 | `raw_author_20260507_230322.csv` | 1000 | 20 | 作者信息表 |
| 4 | `raw_music_20260507_230322.csv` | 1000 | 16 | 音乐信息表 |
| 5 | `raw_hashtag_20260507_230322.csv` | 2420 | 10 | 话题标签表 |
| 6 | `raw_video_tag_20260507_230322.csv` | 0 | 5 | 平台标签表，当前为空 |
| 7 | `raw_video_media_20260507_230322.csv` | 1000 | 24 | 媒体元信息表 |
| 8 | `raw_video_status_control_20260507_230322.csv` | 1000 | 17 | 状态权限表 |
| 9 | `raw_chapter_20260507_230322.csv` | 0 | 10 | 章节表，当前为空 |
| 10 | `raw_comment_20260507_230322.csv` | 1852 | 24 | 评论明细表 |
| 11 | `raw_related_video_20260507_230322.csv` | 4482 | 22 | 相关推荐边表 |
| 12 | `raw_crawl_log_20260507_230322.csv` | 1000 | 14 | 采集日志表 |

---

## 4. 数据质量摘要

### 4.1 总体质量

| 指标 | 值 |
|---|---:|
| 总 URL | 1000 |
| 成功 URL | 1000，100% |
| failed URL | 0，0% |
| unique video_id | 1000，100% |
| video_id 与 page_url 一致率 | 1000 / 1000，100% |
| exact + high | 788，78.8% |
| none + low | 212，21.2% |

none-match 样本仍可获得 video_id 与 author_id，但部分 aweme_detail 内的互动和媒体字段覆盖不足。后续建模时可以基于 `raw_crawl_log.match_type` 和 `confidence` 选择是否过滤低置信样本。

### 4.2 关键字段覆盖

| 表 | 字段 | 非空率 |
|---|---|---:|
| raw_video_detail | video_id | 1000 / 1000，100% |
| raw_video_detail | author_id | 1000 / 1000，100%，822 unique |
| raw_video_detail | create_time | 1000 / 1000，100% |
| raw_video_detail | duration_ms | 822 / 1000，82.2% |
| raw_video_detail | digg_count | 822 / 1000，82.2% |
| raw_author | author_id | 822 / 1000，82.2%，822 unique |
| raw_music | video_id | 1000 / 1000，100% |
| raw_video_media | video_id | 1000 / 1000，100% |
| raw_video_status_control | video_id | 1000 / 1000，100% |
| raw_hashtag | video_id / hashtag_name | 2420 / 2420，100% |
| raw_comment | comment_id | 1852 / 1852，100% |
| raw_comment | comment_text | 1819 / 1852，98.2% |
| raw_related_video | source_video_id / related_video_id | 4482 / 4482，100% |
| raw_crawl_log | target_url / match_type / confidence | 1000 / 1000，100% |

### 4.3 跨表关联

| 关联 | 匹配率 |
|---|---:|
| raw_video_detail.author_id -> raw_author.author_id | 822 / 822，100% |
| raw_music.video_id -> raw_video_detail.video_id | 1000 / 1000，100% |
| raw_video_media.video_id -> raw_video_detail.video_id | 1000 / 1000，100% |
| raw_video_status_control.video_id -> raw_video_detail.video_id | 1000 / 1000，100% |
| raw_hashtag.video_id -> raw_video_detail.video_id | 603 / 603，100% |
| raw_comment.video_id -> raw_video_detail.video_id | 374 / 374，100% |
| raw_related_video.source_video_id -> raw_video_detail.video_id | 449 / 449，100% |

### 4.4 触发率

| 表 | 行数 | 触发视频数 | 触发率 |
|---|---:|---:|---:|
| raw_hashtag | 2420 | 603 | 60.3% |
| raw_comment | 1852 | 374 | 37.4% |
| raw_related_video | 4482 | 449 | 44.9% |

---

## 5. 表级使用建议

### raw_video_detail

主视频详情表。建议作为后续 tabular 输入主表，包含 `video_id`、`author_id`、文本内容、发布时间、时长、互动统计等字段。

注意：`duration_ms`、`digg_count` 等字段在 none-match 样本中覆盖不全，建模时需按非空率筛选或填充。

### raw_author

作者信息表。可通过 `raw_video_detail.author_id` 关联。当前有 822 个 unique author_id。

### raw_music

音乐信息表。通过 `video_id` 关联主表。可用于音乐标题、音乐作者、原声标记等特征。

### raw_hashtag

话题标签表。可聚合为 `hashtag_count`、`hashtag_name_joined`，也可用于构造 video-hashtag 图边。

### raw_video_tag

当前为空表。详情页响应未发现 `video_tag` 结构，暂不作为模型特征。后续可从搜索页或推荐流响应补充。

### raw_video_media

媒体元信息表。包含封面 URL 列表、动态封面、原始封面、视频宽高等字段。`*_url_list` 字段为 JSON 列表字符串。

### raw_video_status_control

状态权限表。可用于权限、下载、评论、分享、审核状态等过滤或特征。

### raw_chapter

当前为空表。详情页响应未发现 `chapter_list` 结构，暂不作为模型特征。

### raw_comment

评论明细表。374/1000 视频触发评论响应，共 1852 条评论。可聚合评论数量、评论文本长度、评论点赞等特征。

### raw_related_video

相关推荐边表。449/1000 视频触发相关推荐，共 4482 条边。可直接用于 GraphSAGE 的 video-video 边。

### raw_crawl_log

采集日志表。用于过滤低置信样本、追踪 `match_type`、`confidence`、状态码和采集质量。

---

## 6. sample0427 历史说明

`sample0427` 路径：

```text
douyin_data_project/data/sample0427/
```

`sample0427` 的当前定位是：

- 历史流程验证数据
- 快速回归测试数据
- schema 与表结构参考数据
- 小样本调试数据

不建议再作为当前主实验数据。不得覆盖或删除 sample0427。若使用 sample0427，应明确说明它只用于流程测试或回归验证。

---

## 7. 项目目录结构

```text
RA_PTA/
├── README.md
├── CLAUDE.md
├── development_log.md
├── configs/
│   ├── common/
│   │   ├── data_paths.yaml
│   │   ├── metrics.yaml
│   │   ├── split.yaml
│   │   └── real_raw_1000.yaml        # 建议新增
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
├── douyin_data_project/
│   └── data/
│       ├── sample0427/               # 历史流程验证数据
│       └── interim/
│           └── 20260507_230322/      # 当前真实 raw 数据包
├── src/
│   ├── data/
│   │   ├── load_sample0427.py
│   │   ├── load_real_raw.py           # 建议新增
│   │   ├── validate_schema.py
│   │   ├── build_tabular_dataset.py
│   │   ├── build_graph_dataset.py
│   │   └── build_multimodal_dataset.py
│   ├── features/
│   ├── models/
│   │   ├── dnn/
│   │   ├── wide_deep/
│   │   ├── graphsage/
│   │   └── multimodal/
│   ├── evaluation/
│   ├── experiment/
│   └── utils/
├── outputs/
│   ├── data_check/
│   ├── dnn/
│   ├── wide_deep/
│   ├── graphsage/
│   ├── multimodal/
│   ├── comparison/
│   └── ab_test/
├── reports/
│   ├── model_comparison_report.md
│   └── figures/
└── scripts/
```

---

## 8. 主要运行入口

### 数据读取与特征构建

| 脚本 | 用途 |
|---|---|
| `src/data/load_sample0427.py` | sample0427 历史样本读取 |
| `src/data/load_real_raw.py` | real_raw_1000 真实 raw 数据读取，建议新增 |
| `src/data/validate_schema.py` | schema 校验与关联检查 |
| `src/data/build_tabular_dataset.py` | tabular 数据集构建 |
| `src/data/build_graph_dataset.py` | 图节点、边、特征构建 |
| `src/data/build_multimodal_dataset.py` | 多模态数据集构建 |

### 模型训练与评估

| 脚本 | 用途 |
|---|---|
| `src/models/dnn/train.py` / `evaluate.py` | DNN 训练与评估 |
| `src/models/wide_deep/train.py` / `evaluate.py` | Wide & Deep 训练与评估 |
| `src/models/graphsage/train.py` / `evaluate.py` | GraphSAGE 训练与评估 |
| `src/models/multimodal/train.py` / `evaluate.py` | Multimodal 训练与评估 |

### 实验入口

| 脚本 | 用途 |
|---|---|
| `src/experiment/run_comparison.py` | 统一多模型对比实验 |
| `src/experiment/run_ab_simulation.py` | 离线 A/B 模拟或分组统计 |

---

## 9. 统一评估口径

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

- eval_loss
- num_samples
- num_positive
- num_negative
- warnings

如果某个模型无法计算某项指标，对应值可为 `null`，但必须在 `warnings` 中说明。

---

## 10. 统一输出格式

每个模型训练评估完成后，至少输出：

```text
outputs/<model_name>/<run_id>/metrics.json
outputs/<model_name>/<run_id>/predictions.csv
outputs/<model_name>/<run_id>/train_log.csv
outputs/<model_name>/<run_id>/model.pt
outputs/<model_name>/<run_id>/run_meta.json
outputs/<model_name>/<run_id>/feature_config_used.json
```

其中 `<model_name>` 固定为：

```text
dnn
wide_deep
graphsage
multimodal
```

`predictions.csv` 至少包含：

```text
video_id 或 sample_id
label
score
pred
split
model_name
run_id
```

`metrics.json` 至少包含：

```text
model_name
run_id
eval_loss
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

---

## 11. 当前已知限制

1. `raw_video_tag` 当前为空，详情页响应未发现 video_tag 结构。
2. `raw_chapter` 当前为空，详情页响应未发现 chapter_list 结构。
3. none-match 样本占 21.2%，部分互动和媒体字段覆盖不足。
4. raw_comment 触发率为 37.4%，不是所有视频都有评论响应。
5. raw_related_video 触发率为 44.9%，不是所有视频都有相关推荐响应。
6. 当前数据来自公开网页端，不代表平台内部完整数据。
7. 当前不包含真实曝光、点击、完播、转化、留存标签。
8. 后续标签仍需基于互动指标构造，不能称为真实 CTR/CVR 标签。
9. 当前离线实验结果不能代表线上推荐效果或业务收益。

---

## 12. 下一步开发计划

1. 新增 `real_raw_1000` 数据路径配置。
2. 实现真实 raw 数据读取检查。
3. 基于真实 raw 数据重新构建 tabular 数据集。
4. 基于真实 raw 数据重新构建 graph 数据集。
5. 基于真实 raw 数据重新构建 multimodal 数据集。
6. 使用真实 raw 数据重新训练 DNN。
7. 使用真实 raw 数据重新训练 Wide & Deep。
8. 使用真实 raw 数据重新训练 GraphSAGE。
9. 使用真实 raw 数据重新训练 Multimodal。
10. 重新进行统一多模型对比实验。
11. 生成真实 raw 数据上的多模型对比实验报告。
12. 如需要更大样本，可继续采集剩余 1060 URL 或使用 URL 发现器扩充数据。
13. 后续单独增强 raw_video_tag、raw_chapter、评论触发策略和相关推荐触发策略。

---

## 13. 实验结果表述规范

可以使用：

- “多模型对比实验”
- “流程验证结果”
- “真实网页端 raw 数据上的离线实验结果”
- “基于互动指标构造的离线标签”

不要使用：

- “正式线上推荐效果”
- “真实业务收益”
- “CTR/CVR 真实标签”
- “线上 A/B 实验结论”

如果报告中引用 sample0427 结果，应称为“sample0427 流程验证结果”。如果报告中引用 real_raw_1000 结果，应称为“真实网页端 raw 数据上的离线实验结果”。

---

## 14. 当前推荐阅读顺序

```text
1. README.md
2. CLAUDE.md
3. development_log.md
4. douyin_data_project/data/interim/20260507_230322/real_raw_data_delivery_20260507_230322.md
5. reports/model_comparison_report.md                 # sample0427 历史流程验证报告
6. outputs/comparison/202604301609/model_comparison_report.md
7. outputs/ab_test/202604301630/ab_simulation_report.md
```

---

## 15. 历史产物说明

历史 sample0427 流程验证产物保留在：

| 模块 | Run ID | 输出目录 |
|---|---|---|
| DNN | 202604301440 | `outputs/dnn/202604301440/` |
| Wide & Deep | 202604301557 | `outputs/wide_deep/202604301557/` |
| GraphSAGE | 202604291958 | `outputs/graphsage/202604291958/` |
| Multimodal | 202604301557 | `outputs/multimodal/202604301557/` |
| 统一对比实验 | 202604301609 | `outputs/comparison/202604301609/` |
| A/B 模拟 | 202604301630 | `outputs/ab_test/202604301630/` |
| 报告 | — | `reports/model_comparison_report.md` |

这些产物用于历史对照和流程回归，不再作为当前主实验结果。
