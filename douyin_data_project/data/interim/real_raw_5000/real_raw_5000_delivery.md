# 真实 raw 数据包交付说明

## 交付版本

| 项目 | 内容 |
|------|------|
| 数据包名称 | real_raw_5000 |
| 数据来源 | 抖音网页端公开视频页面，part1-part5 分批采集结果合并 |
| 交付日期 | 2026-05-11 |
| 采集方式 | Playwright browser 模式，每 URL 独立 page，分批采集 |
| 采集入口 | manual_url（手动输入 URL 列表） |

---

## 一、数据包基本信息

| 指标 | 值 |
|------|-----|
| 输入 run_id 列表 | 20260509_212100, 20260510_151052, 20260510_183114, 20260510_210600, 20260511_084923 |
| 输出目录 | `data/interim/real_raw_5000/` |
| full unique video_id | **5000** |
| high-confidence video_id | **3493** (69.9%) |
| low_quality_count | 0 |
| 跨批重复 video_id | 0（五批无跨批重复，分批设计有效） |

### full 数据包路径

```
data/interim/real_raw_5000/
├── 11 张 raw 表 (*_real_raw_5000.csv)
├── merge_report.json
├── quality_audit.json
├── table_summary.csv
└── real_raw_5000_delivery.md
```

### high-confidence 辅助文件路径

```
data/processed/real_raw_5000/
├── high_confidence_web_video_meta_real_raw_5000.csv
├── high_confidence_video_ids.txt
└── high_confidence_filter_report.json
```

---

## 二、11 张 raw 表行数

| 序号 | 表名 | 行数（不含表头） | 字段数 |
|------|------|----------------:|--------:|
| 1 | raw_video_detail | 5000 | 30 |
| 2 | raw_author | 3558 | 20 |
| 3 | raw_music | 5000 | 16 |
| 4 | raw_hashtag | 10097 | 10 |
| 5 | raw_video_tag | 0 | 0 |
| 6 | raw_video_media | 5000 | 24 |
| 7 | raw_video_status_control | 5000 | 17 |
| 8 | raw_chapter | 0 | 0 |
| 9 | raw_comment | 9334 | 24 |
| 10 | raw_related_video | 22657 | 22 |
| 11 | raw_crawl_log | 5000 | 14 |

---

## 三、质量摘要

### 3.1 核心指标

| 指标 | 值 |
|------|-----|
| unique video_id | 5000 / 5000 |
| 有效样本（video_id+author_id+page_url 非空） | 5000 (100.0%) |
| video_id 与 page_url 一致率 | 100.0% |
| none-match 样本 | 1507 (30.1%) |
| high confidence | 3493 (69.9%) |
| 低质量样本 | 0 (0.0%) |

### 3.2 核心字段覆盖率

| 字段 | 覆盖率 |
|------|--------|
| create_time | 100.0% (5000/5000) |
| duration_ms | 73.0% (3651/5000) |
| digg_count | 72.9% (3645/5000) |
| comment_count | 72.9% (3646/5000) |
| share_count | 72.9% (3646/5000) |
| collect_count | 72.9% (3645/5000) |

### 3.3 match_type / confidence 分布

| match_type | 数量 | 占比 |
|-----------|-----:|-----:|
| exact + high | 3493 | 69.9% |
| none + low | 1507 | 30.1% |

### 3.4 跨表关联

| 关联 | 匹配率 |
|-----|--------:|
| raw_author.author_id → raw_video_detail.author_id | 100% |
| raw_music.video_id → raw_video_detail.video_id | 100% |
| raw_video_media.video_id → raw_video_detail.video_id | 100% |
| raw_video_status_control.video_id → raw_video_detail.video_id | 100% |
| raw_hashtag.video_id → raw_video_detail.video_id | 100% |
| raw_comment.video_id → raw_video_detail.video_id | 100% |
| raw_related_video.source_video_id → raw_video_detail.video_id | 100% |

### 3.5 触发率

| 表 | 触发视频数 | 触发率 |
|----|----------:|-------:|
| raw_hashtag | 2541 | 50.8% |
| raw_comment | 1874 | 37.5% |
| raw_related_video | 2266 | 45.3% |

---

## 四、high-confidence 辅助文件说明

### 4.1 文件列表

| 文件 | 路径 | 内容 |
|------|------|------|
| high_confidence_web_video_meta_real_raw_5000.csv | `data/processed/real_raw_5000/` | 3485 行，69 列（含 match_type/confidence），仅保留 exact+high 样本 |
| high_confidence_video_ids.txt | `data/processed/real_raw_5000/` | 每行一个 video_id，共 3493 个 ID |
| high_confidence_filter_report.json | `data/processed/real_raw_5000/` | 过滤统计报告 |

### 4.2 过滤规则

- `match_type == "exact"` AND `confidence == "high"`
- 5 个 batch 的 `real_web_video_meta_*.csv` 合并后按 video_id 去重，然后按上述条件筛选

### 4.3 与 full 数据包的关系

1. **high-confidence 辅助文件是质量筛选工具**，并非默认输入。
2. **默认标准输入** 仍是 `data/interim/real_raw_5000/` 下的 11 张 raw 表（含全部 5000 video_id）。
3. `high_confidence_video_ids.txt` 可在上层特征构建阶段用于过滤样本（即只在 exact+high 样本上训练或评估）。
4. `high_confidence_web_video_meta_real_raw_5000.csv` 提供宽表格式的 high-confidence 样本视图，便于快速探索和调试。
5. 后续实验可通过配置切换 full / high_confidence 模式：

```yaml
# 使用全量 5000 样本
data_mode: full
# 或仅使用 exact+high 样本
data_mode: high_confidence
```

---

## 五、标签泄漏风险说明

### 5.1 当前标签构造来源

当前 `real_raw_5000` 数据包中可用于构造离线标签的字段包括：

- `digg_count`
- `comment_count`
- `share_count`
- `collect_count`
- `play_count`

### 5.2 必须做的 no_interaction_leakage 复验

后续建模时，**必须优先做 no_interaction_leakage 复验**，具体要求：

1. **标签构造字段不得进入模型特征。** 例如，如果以 `digg_count` 构造二分类标签，则 `digg_count` 本身以及与其高相关的其他互动字段不能作为模型输入特征。
2. **特征来源应限于**：
   - 视频内容元信息：`duration_ms`、`create_time`、`aweme_type`、`media_type` 等
   - 作者信息：`follower_count`、`total_favorited`、`verification_type` 等
   - 文本内容统计：`desc` 长度、`hashtag_count` 等
   - 媒体元信息：视频宽高、封面宽高等
   - 图结构信息：话题共现、相关推荐连接等
3. **避不开的字段**：`hashtag_count`、`desc_length` 等纯内容派生字段不与互动标签直接相关，可作为特征。
4. **严格区分的字段**：任何聚合统计（如作者平均互动量）需要来自不包含当前样本的训练集统计，防止 data leakage。
5. **建议方案**：
   - 方案 A：只使用 exact+high 样本构造标签，确保互动字段真实有效
   - 方案 B：过滤掉 none-match 样本中互动字段为空的样本
   - 方案 C：使用多层特征过滤，在特征配置中显式声明哪些字段排除

---

## 六、表级使用建议

### raw_video_detail（主视频详情表，5000 行，30 列）

**用途**：建模主表。每条视频一行，包含视频基础信息、文本内容、发布时间、时长、互动统计。

**建议**：
- 作为 DNN / Wide & Deep 主输入表
- duration_ms、digg_count 等字段在 none-match 样本中覆盖不全（~27% 缺失）
- author_id 可关联到 raw_author

### raw_author（作者信息表，3558 行，20 列）

**用途**：存储作者基础信息与公开统计。每条视频对应一行作者快照。

**建议**：
- 3558 unique author_id，可通过 author_id 关联 raw_video_detail
- 可用于 author 维度特征聚合

### raw_music（音乐信息表，5000 行，16 列）

**用途**：存储音乐元信息。每个视频-音乐一行（通过 video_id 关联）。

**建议**：
- 100% 覆盖率
- 可用于 music 维度特征

### raw_hashtag（话题标签表，10097 行，10 列）

**用途**：话题拆行表，用于构造 video-hashtag 异构图边。

**建议**：
- 2541/5000 视频触发（50.8%）
- 可聚合为 hashtag_count / hashtag_name_joined 等特征

### raw_video_tag（平台标签表，0 行）

**当前状态**：空表。详情页响应未发现 video_tag 结构。

### raw_video_media（媒体元信息表，5000 行，24 列）

**用途**：封面 URL、动态封面 URL、原始封面 URL、视频宽高等。

**建议**：
- 100% 覆盖
- cover_url_list 等字段为 JSON 列表，需解析使用

### raw_video_status_control（状态权限表，5000 行，17 列）

**用途**：评论、分享、下载等权限控制字段。

**建议**：
- 100% 覆盖

### raw_chapter（章节表，0 行）

**当前状态**：空表。详情页响应未发现 chapter_list 结构。

### raw_comment（评论明细表，9334 行，24 列）

**用途**：公开评论区评论明细。

**建议**：
- 1874/5000 视频触发（37.5%）
- 可聚合为评论统计特征

### raw_related_video（相关推荐边表，22657 行，22 列）

**用途**：相关推荐列表，每个源视频-推荐视频一行。

**建议**：
- 2266/5000 视频触发（45.3%）
- 可直接用于 GraphSAGE 图构建

### raw_crawl_log（采集日志表，5000 行，14 列）

**用途**：采集质量日志。

**建议**：
- 用于过滤低置信样本（仅保留 exact+high）

---

## 七、已知限制

1. **none/low confidence 占比约 30.1%**：1507/5000 视频无法从 aweme_detail JSON 主响应提取全量字段。但 video_id 与 page_url 一致率 100%，author_id 100% 可获取。互动指标在 none-match 样本中覆盖率偏低。

2. **raw_video_tag 当前为空**：0 行。当前详情页响应未发现 video_tag 结构。

3. **raw_chapter 当前为空**：0 行。当前详情页响应未发现 chapter_list 结构。

4. **评论和相关推荐不是全量触发**：raw_comment 触发率 37.5%，raw_related_video 触发率 45.3%。

5. **数据来自公开网页端**：不代表平台内部完整数据。不包含平台内部真实曝光、点击、完播、转化日志。

6. **当前没有真实曝光、点击、完播、转化标签**：所有标签只能基于互动字段构造，属于离线伪标签。

7. **作者表含重复快照**：同一作者在不同视频中有多行快照，不是去重后的唯一作者表。

---

## 八、下一步建议

1. **no_interaction_leakage 复验**：在构建 tabular 特征前，先定义标签构造方案，明确哪些字段不能进入特征。

2. **特征构建**：基于 11 张 raw 表构建 tabular / graph / multimodal 输入。

3. **多模型实验**：使用 real_raw_5000 重新训练 DNN / Wide & Deep / GraphSAGE / Multimodal。

4. **high-confidence 子集实验**：对比 full 5000 与 high-confidence 3493 子集上的模型表现差异。

5. **raw_video_tag / raw_chapter 补充**：待后续从其他响应中补充。

---

## 附录：数据字典引用

字段级定义以以下文档为准：

- `docs/data_dictionary.md`：字段名、类型、样例值、说明
- `docs/Data_description.md`：数据建设方案、表设计、字段可得性

数据包访问路径：

```
douyin_data_project/data/interim/real_raw_5000/
```
