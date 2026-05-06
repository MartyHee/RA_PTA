# 抖音公开数据采集与处理项目

## 1. 项目简介

本项目是 `RA_PTA` 推荐算法项目的数据工程子项目，负责从抖音网页端公开视频页面及相关网络响应中采集、解析、整理和输出原生数据。

前期项目已经完成了小批量抓取、字段修正、浏览器稳定性增强、`sample0427` 样本数据构建，以及基于样本数据的 W4 多模型流程验证。接下来本项目的重点是：

- 按 `docs/data_dictionary.md` 扩充真实可抓取原生字段；
- 从单表过渡到多表 raw schema；
- 进行大批量真实采集；
- 为后续 W4 正式模型实验提供真实数据，而不是继续依赖样本补齐数据。

当前主线仍然是：

**网页端公开视频采集为主，OpenAPI 仅作为可选辅助。**

---

## 2. 当前阶段目标

当前阶段的主要目标是建设真实大规模 raw 数据集。

具体目标包括：

1. 扩充视频详情页主响应字段；
2. 解析评论列表响应；
3. 解析相关推荐响应；
4. 输出接近最终设计的 11 张 raw 表；
5. 累计采集至少 1000 条不同公开视频；
6. 生成字段覆盖率、失败 URL、质量报告和采集批次报告；
7. 为上层 W4 模型项目提供真实数据输入。

当前阶段不再以 `sample0427` 为目标数据。`sample0427` 只保留为流程验证样本和结构参考。

---

## 3. 重要文档

| 文件                         | 说明                                             |
| ---------------------------- | ------------------------------------------------ |
| `docs/Data_description.md` | 数据建设方案、字段可得性、W4 任务与数据映射      |
| `docs/data_dictionary.md`  | 当前字段扩充与最终 raw schema 的核心依据         |
| `docs/development_log.md`  | 开发日志，记录历史修改、运行命令、输出和验证结果 |
| `README.md`                | 项目说明，面向用户和开发者                       |
| `CLAUDE.md`                | 智能体工作规范，面向 cc/Claude 使用              |

开发者阅读项目时，建议顺序是：

1. 先读 `README.md`
2. 再读 `docs/Data_description.md`
3. 再读 `docs/data_dictionary.md`
4. 如需了解历史问题，阅读 `docs/development_log.md`

---

## 4. 项目目录结构

```text
douyin_data_project/
├── docs/
│   ├── Data_description.md
│   ├── data_dictionary.md
│   └── development_log.md
├── configs/
│   ├── settings.yaml
│   ├── sources.yaml
│   ├── fields.yaml
│   ├── logging.yaml
│   └── url_discovery/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── features/
│   ├── sample0427/
│   └── logs/
├── src/
│   ├── crawler/
│   │   ├── browser_client.py
│   │   ├── parser.py
│   │   ├── scheduler.py
│   │   ├── client.py
│   │   ├── anti_block.py
│   │   └── extractors.py
│   ├── processing/
│   ├── features/
│   ├── analysis/
│   ├── api/
│   ├── schemas/
│   └── utils/
├── tests/
├── run_crawl.py
├── run_discover_urls.py
├── run_features.py
├── run_clean.py
├── run_eda.py
├── README.md
└── CLAUDE.md
```

---

## 5. 核心模块说明

### 5.1 `src/crawler/`

负责网页端采集与字段解析。

重点文件：

| 文件                  | 功能                                                        |
| --------------------- | ----------------------------------------------------------- |
| `browser_client.py` | Playwright 浏览器抓取、网络响应监听、页面隔离、browser 重启 |
| `parser.py`         | HTML / 网络响应解析与字段提取                               |
| `scheduler.py`      | 批量调度、结果落盘、高置信度筛选、报告生成                  |
| `extractors.py`     | 字段提取器                                                  |
| `anti_block.py`     | 频率控制与基础反反爬策略                                    |

### 5.2 `src/schemas/`

保存数据结构定义。后续字段扩充应优先检查和更新这里。

### 5.3 `data/`

保存采集、处理和特征产物。

当前推荐后续 raw 多表输出结构：

```text
data/raw/<run_id>/
├── raw_video_detail.parquet
├── raw_author.parquet
├── raw_music.parquet
├── raw_hashtag.parquet
├── raw_video_tag.parquet
├── raw_video_media.parquet
├── raw_video_status_control.parquet
├── raw_chapter.parquet
├── raw_comment.parquet
├── raw_related_video.parquet
├── raw_crawl_log.parquet
├── failed_urls.csv
└── raw_build_report.json
```

初期仍可输出 CSV，但字段名必须与 raw schema 对齐，并生成字段覆盖率报告。

---

## 6. 目标 raw 表

后续真实采集应逐步输出 11 张 raw 表：

| 表名                         | 粒度                    | 作用                                                   |
| ---------------------------- | ----------------------- | ------------------------------------------------------ |
| `raw_video_detail`         | 每条视频一行            | 主视频基础信息、文本、发布时间、互动统计、主响应元信息 |
| `raw_author`               | 每个作者一行或多快照    | 作者 ID、安全 ID、昵称、签名、粉丝数、认证信息         |
| `raw_music`                | 每个视频-音乐一行       | 音乐 ID、标题、作者、时长、封面、原声标记              |
| `raw_hashtag`              | 每个视频-话题一行       | 文案话题标签及其位置                                   |
| `raw_video_tag`            | 每个视频-平台标签一行   | 平台识别的视频内容分类标签                             |
| `raw_video_media`          | 每条视频一行            | 封面、动态封面、原始封面、宽高、码率、缩略图           |
| `raw_video_status_control` | 每条视频一行            | 评论、分享、下载、私密、删除、审核状态                 |
| `raw_chapter`              | 每个视频-章节一行       | 章节标题、详情、时间戳、摘要、封面                     |
| `raw_comment`              | 每个评论一行            | 评论文本、时间、点赞数、回复数、热评标记               |
| `raw_related_video`        | 每个源视频-推荐视频一行 | 相关推荐边、推荐位次、推荐视频基础字段                 |
| `raw_crawl_log`            | 每次请求/采集一行       | 采集批次、请求 URL、响应状态、匹配质量和追溯信息       |

字段定义以 `docs/data_dictionary.md` 为准。

---

## 7. 字段命名约定

raw 层应尽量保留网页响应中的原生字段语义，并统一使用当前数据字典中的字段名。

特别注意：

| 应使用                     | 不再使用              | 说明               |
| -------------------------- | --------------------- | ------------------ |
| `duration_ms`            | `duration_sec`      | raw 层统一保存毫秒 |
| `create_time`            | `publish_time_raw`  | 使用响应原字段名   |
| `digg_count`             | `like_count_raw`    | 使用响应原字段名   |
| `cover_url_list`         | `cover_url`         | 视频封面 URL 列表  |
| `origin_cover_url_list`  | `origin_cover_url`  | 原始封面 URL 列表  |
| `dynamic_cover_url_list` | `dynamic_cover_url` | 动态封面 URL 列表  |

---

## 8. 批量采集能力

当前浏览器抓取已经具备以下能力：

- URL 文件批量输入；
- browser 模式；
- 每个 URL 使用独立 page；
- 每个 URL 完成后关闭 page；
- 定期重启 browser/context；
- 默认关闭高强度 debug 输出；
- 支持 `--restart-every`；
- 支持 `--debug-mode`；
- 按 run_id 输出；
- 生成质量报告；
- 支持高置信度样本筛选。

示例命令：

```bash
D:\CodeData\software\Anaconda\Anaconda3\envs\ra\python.exe run_crawl.py --browser --url-file configs/batch_urls.txt --source-entry manual_url --workers 1 --restart-every 20
```

大批量采集建议从 `workers=1` 起步，默认不开 `--debug-mode`。

---

## 9. 推荐执行路线

### 第一阶段：字段扩充

目标是让现有爬虫从视频详情主响应、评论响应、相关推荐响应中提取更多原生字段，并逐步落成多表 raw schema。

建议先用 5-10 条 URL 验证。

### 第二阶段：小批量验证

使用 20-50 条 URL 检查：

- 表是否生成；
- 字段名是否对齐；
- 字段非空率是否合理；
- 主键是否能关联；
- 评论表和相关推荐边表是否真实形成；
- 质量报告是否生成。

### 第三阶段：中批量采集

使用 200-300 条 URL 检查：

- browser 长批次稳定性；
- 字段覆盖率；
- 重复率；
- 失败 URL；
- 输出目录结构。

### 第四阶段：大批量采集

目标是累计采集至少 1000 条不同视频。

验收重点：

- `raw_video_detail.video_id` 去重数量 ≥ 1000；
- 11 张 raw 表尽量齐全；
- 字段覆盖率报告完整；
- 失败 URL 可追溯；
- run_id、命令、输入、输出都记录到开发日志。

---

## 10. 质量验收

每次字段扩充或批量采集后，应至少检查：

1. 输出了哪些表；
2. 每张表行数和字段数；
3. `video_id` 是否非空和唯一；
4. `author_id`、`music_id`、`hashtag_id`、`related_video_id` 是否可关联；
5. `raw_comment` 是否形成评论表；
6. `raw_related_video` 是否形成边表；
7. `duration_ms`、`create_time`、`digg_count` 命名是否正确；
8. URL 列表字段是否可解析；
9. 字段非空率；
10. 失败 URL 和失败原因；
11. 质量报告是否生成；
12. `docs/development_log.md` 是否更新。

---

## 11. 与模型项目的关系

上层模型项目位于：

`D:/CodeData/Program Coding/ByteDance/RA_PTA/`

本项目负责真实数据建设。模型项目负责：

- DNN；
- Wide & Deep；
- GraphSAGE；
- 多模态；
- 离线对比实验；
- A/B 模拟。

此前 `sample0427` 已经用于跑通实验流程。接下来本项目输出的真实大规模 raw 数据将用于替换样本数据，支撑正式实验。
