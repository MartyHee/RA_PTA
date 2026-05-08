# 真实 raw 数据包交付说明

## 交付版本

| 项目 | 内容 |
|------|------|
| 数据包名称 | real_raw_1000_20260507_230322 |
| 采集 run_id | 20260507_230322 |
| 交付日期 | 2026-05-08 |
| 数据来源 | 抖音网页端公开视频页面 |
| 采集方式 | Playwright browser 模式，每 URL 独立 page，restart-every=10 |
| 采集入口 | manual_url（手动输入 URL 列表） |

---

## 一、数据包基本信息

| 指标 | 值 |
|------|-----|
| 输入 URL 文件 | `configs/batch_urls_1000_from_2000.txt` |
| 输入 URL 数 | 1000 |
| unique video_id | 1000 |
| 成功 URL | 1000 |
| failed URL | 0 |
| 输出目录 | `data/interim/20260507_230322/` |
| 运行耗时 | 约 102 分钟 |
| browser restart-every | 10（约重启 100 次） |
| workers | 1 |
| 运行命令 | `D:/CodeData/software/Anaconda/Anaconda3/envs/ra/python.exe run_crawl.py --browser --url-file configs/batch_urls_1000_from_2000.txt --source-entry manual_url --workers 1 --restart-every 10` |

---

## 二、输出文件清单

共 12 个文件（10 张非空 raw 表 + 1 张 meta 表 + 1 张采集日志表）：

| 序号 | 文件 | 行数（不含表头） | 字段数 | 用途 |
|------|------|------|--------|------|
| 1 | `real_web_video_meta_20260507_230322.csv` | 1000 | 68 | 完整原始 meta 输出，含所有解析字段的大宽表 |
| 2 | `raw_video_detail_20260507_230322.csv` | 1000 | 30 | 主视频详情表：基础信息、内容文本、互动统计、元信息 |
| 3 | `raw_author_20260507_230322.csv` | 1000 | 20 | 作者信息表：昵称、签名、粉丝数、认证等 |
| 4 | `raw_music_20260507_230322.csv` | 1000 | 16 | 音乐信息表：标题、作者、时长、封面、原声标记 |
| 5 | `raw_hashtag_20260507_230322.csv` | 2420 | 10 | 话题标签表：视频中话题拆行，用于 video-hashtag 图边 |
| 6 | `raw_video_tag_20260507_230322.csv` | 0 | 5 | 平台标签表：当前为空 |
| 7 | `raw_video_media_20260507_230322.csv` | 1000 | 24 | 媒体元信息表：封面 URL、视频尺寸、码率等 |
| 8 | `raw_video_status_control_20260507_230322.csv` | 1000 | 17 | 状态权限表：评论/分享/下载/私密/删除/审核状态 |
| 9 | `raw_chapter_20260507_230322.csv` | 0 | 10 | 章节表：当前为空 |
| 10 | `raw_comment_20260507_230322.csv` | 1852 | 24 | 评论明细表：评论文本、时间、点赞、回复、热评标记 |
| 11 | `raw_related_video_20260507_230322.csv` | 4482 | 22 | 相关推荐边表：源视频-推荐视频，用于 content graph 边 |
| 12 | `raw_crawl_log_20260507_230322.csv` | 1000 | 14 | 采集日志表：URL、状态码、match_type、confidence、耗时等 |

---

## 三、核心质量指标

### 3.1 总体指标

| 指标 | 值 |
|------|-----|
| 总 URL | 1000 |
| 成功 | 1000 (100%) |
| 失败 | 0 (0%) |
| unique video_id | 1000 (100%) |
| video_id 与 page_url 一致率 | 1000/1000 (100%) |

### 3.2 match_type / confidence 分布

| 类型 | 数量 | 占比 |
|------|------|------|
| exact + high | 788 | 78.8% |
| none + low | 212 | 21.2% |

none-match 样本的 video_id 和 author_id 仍为 100% 可获取，缺少的是 aweme_detail JSON 中的互动和媒体字段。

### 3.3 关键字段覆盖

| 表 | 字段 | 非空率 |
|----|------|--------|
| raw_video_detail | video_id | 1000/1000 (100%), 1000 unique |
| raw_video_detail | author_id | 1000/1000 (100%), 822 unique |
| raw_video_detail | create_time | 1000/1000 (100%) |
| raw_video_detail | duration_ms | 822/1000 (82.2%) |
| raw_video_detail | digg_count | 822/1000 (82.2%) |
| raw_author | author_id | 822/1000 (82.2%), 822 unique |
| raw_music | video_id | 1000/1000 (100%) |
| raw_video_media | video_id | 1000/1000 (100%) |
| raw_video_status_control | video_id | 1000/1000 (100%) |
| raw_hashtag | video_id | 2420/2420 (100%) |
| raw_hashtag | hashtag_name | 2420/2420 (100%) |
| raw_comment | video_id | 1852/1852 (100%) |
| raw_comment | comment_id | 1852/1852 (100%) |
| raw_comment | comment_text | 1819/1852 (98.2%) |
| raw_related_video | source_video_id | 4482/4482 (100%) |
| raw_related_video | related_video_id | 4482/4482 (100%) |
| raw_crawl_log | target_url | 1000/1000 (100%) |
| raw_crawl_log | match_type | 1000/1000 (100%) |
| raw_crawl_log | confidence | 1000/1000 (100%) |

### 3.4 跨表关联检查

| 关联 | 匹配率 |
|-----|--------|
| detail.author_id -> author.author_id | 822/822 (100%) |
| music.video_id -> detail.video_id | 1000/1000 (100%) |
| media.video_id -> detail.video_id | 1000/1000 (100%) |
| status_control.video_id -> detail.video_id | 1000/1000 (100%) |
| hashtag.video_id -> detail.video_id | 603/603 (100%) |
| comment.video_id -> detail.video_id | 374/374 (100%) |
| related.source_video_id -> detail.video_id | 449/449 (100%) |

所有跨表关联 100% 可匹配。

### 3.5 触发率统计

| 表 | 行数 | 触发视频数 | 触发率 |
|----|------|-----------|--------|
| raw_hashtag | 2420 | 603 | 60.3% |
| raw_related_video | 4482 | 449 | 44.9% |
| raw_comment | 1852 | 374 | 37.4% |

---

## 四、表级使用建议

### raw_video_detail（主视频详情表）

**用途**：建模主表。每条视频一行，包含视频基础信息（video_id、group_id、sec_item_id）、文本内容（desc、title）、发布时间（create_time）、时长（duration_ms）、互动统计（digg_count、comment_count、share_count、collect_count、play_count）以及额外 author_id 外键。

**建议**：
- 作为 DNN / Wide & Deep 主输入表
- duration_ms、digg_count 等字段在 none-match 样本中覆盖不全（~17%），建模时需按实际非空率筛选或填充
- author_id 可关联到 raw_author

### raw_author（作者信息表）

**用途**：存储作者基础信息与公开统计。每条视频对应一行作者快照（同一作者不同视频可能有不同快照）。

**建议**：
- 通过 author_id 关联 raw_video_detail
- 822 unique author_id，可用于 author 维度特征聚合
- 字段含 nickname、signature、follower_count、following_count、total_favorited、verification_type 等

### raw_music（音乐信息表）

**用途**：存储音乐元信息。每个视频-音乐一行。

**建议**：
- 通过 video_id 关联主表
- 100% 覆盖率
- 可用于 music 维度特征，如 is_original_sound 判断是否为原声

### raw_hashtag（话题标签表）

**用途**：话题拆行表，每个视频-话题一行。用于构造 video-hashtag 异构图边。

**建议**：
- 603/1000 视频触发（60.3%）
- 可聚合为 hashtag_count / hashtag_name_joined 等特征
- hashtag_id 和 hashtag_name 均为 100% 非空

### raw_video_tag（平台标签表）

**当前状态**：空表（0 行）。

**原因**：当前详情页网络响应中未发现 video_tag 结构。

**建议**：暂不作为模型特征输入。后续可从搜索页响应或推荐流响应中补充。

### raw_video_media（媒体元信息表）

**用途**：存储封面 URL、动态封面 URL、原始封面 URL、视频宽高、封面宽高等媒体字段。

**建议**：
- 100% 覆盖
- cover_url_list / origin_cover_url_list / dynamic_cover_url_list 为 JSON 列表，需解析使用
- 多模态阶段可用封面 URL 列表作为图像特征入口

### raw_video_status_control（状态权限表）

**用途**：存储视频评论、分享、下载等权限控制字段，以及删除、审核、私密状态。

**建议**：
- 100% 覆盖
- 可用于过滤权限受限或已删除视频

### raw_chapter（章节表）

**当前状态**：空表（0 行）。

**原因**：当前详情页网络响应中未发现 chapter_list 结构。

**建议**：暂不作为模型特征输入。

### raw_comment（评论明细表）

**用途**：存储公开评论区可见的评论明细。每条评论一行。

**建议**：
- 374/1000 视频触发评论响应（37.4%）
- 1852 条评论，comment_text 非空率 98.2%
- 可聚合为 comment_count、avg_comment_length、avg_digg_count 等特征
- 评论文本可用于文本特征或 NLP 分析

### raw_related_video（相关推荐边表）

**用途**：存储相关推荐列表，每个源视频-推荐视频一行。用于构造 video-video content graph 边。

**建议**：
- 449/1000 视频触发相关推荐（44.9%）
- 4482 条边，每条含 related_rank_position（推荐位次）
- 可直接用于 GraphSAGE 图构建
- 推荐视频的 author_id、digg_count 等字段可用于 edge feature

### raw_crawl_log（采集日志表）

**用途**：采集质量日志，记录每次请求的 URL、状态码、耗时、match_type、confidence、matched_object_id 等。

**建议**：
- 100% 覆盖
- 用于过滤低置信样本（如仅保留 exact+high 样本）
- 用于追踪失败原因和采集质量

---

## 五、已知限制

1. **raw_video_tag 为空**：当前 0 行。详情页响应未发现 video_tag 结构，不由当前版本爬虫提供。

2. **raw_chapter 为空**：当前 0 行。详情页响应未发现 chapter_list 结构，不由当前版本爬虫提供。

3. **none-match 占 21.2%**：212/1000 视频无法从 aweme_detail JSON 主响应提取全量字段。但 video_id 与 page_url 一致率 100%，author_id 100% 可获取。互动指标（duration_ms、digg_count 等）在 none-match 样本中覆盖率仅 ~17%。

4. **评论和相关推荐非全触发**：raw_comment 触发率 37.4%，raw_related_video 触发率 44.9%。不是每个视频都有评论或相关推荐响应，属于平台端的正常限制。

5. **部分字段低覆盖**：sec_uid、signature、cover_url_list 等字段在 none-match 样本覆盖率偏低。建模时应按实际非空率筛选特征字段。

6. **数据来源为公开网页端**：当前数据来自抖音公开视频页面的网页端采集，不包含：
   - 平台内部完整数据（如真实曝光、点击、完播、转化日志）
   - 需要用户额外授权才能访问的数据（如粉丝列表、关注列表）
   - 非公开或已删除视频的私有数据

7. **不含推荐标签**：当前数据不包含真实曝光/点击/完播/转化标签。推荐实验应以互动指标（digg_count、comment_count、share_count、collect_count、play_count）作为回归或分类目标。

8. **作者表含重复快照**：同一作者在不同视频中有多行快照，不是去重后的唯一作者表。使用时需按需去重或取最新快照。

---

## 六、上层实验输入切换建议

### 当前状况

上层实验项目位于：

```
D:\CodeData\Program Coding\ByteDance\RA_PTA\
```

当前使用的样本数据：

```
douyin_data_project/data/sample0427/
```

新真实 raw 数据：

```
douyin_data_project/data/interim/20260507_230322/
```

### 切换策略

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 新增配置路径 | 在 `configs/common/data_paths.yaml` 中新增 `real_raw_1000` 数据路径，指向 `data/interim/20260507_230322/` |
| 2 | 保留 sample0427 | 不覆盖 sample0427，保留作为流程测试数据和结构参考 |
| 3 | 数据读取适配 | 上层实验项目新增真实 raw CSV 读取入口，对齐 11 张表 schema |
| 4 | 数据集构建 | 使用真实数据重新生成 tabular / graph / multimodal 输入 |
| 5 | 重新训练 | 使用真实数据重新跑 DNN / Wide & Deep / GraphSAGE / Multimodal |
| 6 | 对比分析 | 对比 sample0427 流程验证结果与 real_raw_1000 结果，评估数据质量影响 |

### 配置示例

```yaml
# configs/common/data_paths.yaml 中建议增加
real_raw_1000:
  root: data/interim/20260507_230322/
  video_detail: raw_video_detail_20260507_230322.csv
  author: raw_author_20260507_230322.csv
  music: raw_music_20260507_230322.csv
  hashtag: raw_hashtag_20260507_230322.csv
  video_tag: raw_video_tag_20260507_230322.csv
  video_media: raw_video_media_20260507_230322.csv
  video_status_control: raw_video_status_control_20260507_230322.csv
  chapter: raw_chapter_20260507_230322.csv
  comment: raw_comment_20260507_230322.csv
  related_video: raw_related_video_20260507_230322.csv
  crawl_log: raw_crawl_log_20260507_230322.csv
```

### 注意事项

- sample0427 与 real_raw_1000 的 schema 不完全一致，读取适配应分别处理
- real_raw_1000 中 raw_video_tag 和 raw_chapter 为空，上层依赖这两个表的模块需要增加空表保护
- none-match 样本占 21.2%，上层可按 crawl_log.match_type 过滤只使用 exact+high 样本

---

## 七、下一步建议

1. **上层数据读取适配**：在上层实验项目中新增 `real_raw_1000` 数据集加载路径，实现 11 张 raw 表的统一读取接口。

2. **重新生成特征数据**：基于真实数据重新构建 tabular 特征、cross features、text features、graph dataset、multimodal dataset。

3. **重新跑模型流程**：使用真实数据依次重新训练 DNN → Wide & Deep → GraphSAGE → Multimodal，验证模型在真实数据上的表现。

4. **对比实验结果**：对比 sample0427 与 real_raw_1000 在各模型上的指标差异，评估样本数据与真实数据的分布偏移。

5. **数据扩充**：如需要更大规模数据（>2000 unique video_id），可从 `configs/batch_urls_2000.txt` 剩余 1060 条 URL 继续采集，或使用 `run_discover_urls.py` 发现更多 URL。

6. **raw_video_tag / raw_chapter 补充**：待后续从搜索页响应或推荐流响应中补充解析逻辑。

7. **评论触发策略增强**：当前评论触发率 37.4%，后续可通过扩展触发窗口或延迟加载提升覆盖率。

---

## 附录：数据字典引用

字段级定义以以下文档为准：

- `docs/data_dictionary.md`：字段名、类型、样例值、说明
- `docs/Data_description.md`：数据建设方案、表设计、字段可得性

数据包访问路径：

```
douyin_data_project/data/interim/20260507_230322/
```