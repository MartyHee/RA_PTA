# W4 开发日志

> 项目根目录：`D:/CodeData/Program Coding/ByteDance/RA_PTA/`
> 当前数据：`douyin_data_project/data/sample0427/`

---

## 2026-04-28 工程目录初始化与配置文件检查

### 任务目标
完成 W4 阶段工程目录初始化与基础配置文件创建，为后续 DNN、Wide & Deep、GraphSAGE、多模态模型、离线对比实验和 A/B 模拟搭建最小项目骨架。

### 修改或新增文件列表

**新增配置文件（8 个）：**
- `configs/common/data_paths.yaml` — 公共路径配置
- `configs/common/metrics.yaml` — 公共评估指标配置
- `configs/common/split.yaml` — 公共数据切分配置
- `configs/dnn/dnn_base.yaml` — DNN 模型基础配置
- `configs/wide_deep/wide_deep_base.yaml` — Wide & Deep 模型基础配置
- `configs/graphsage/graphsage_base.yaml` — GraphSAGE 模型基础配置
- `configs/multimodal/multimodal_base.yaml` — 多模态模型基础配置（含当前约束说明）
- `configs/ab_test/ab_base.yaml` — A/B 模拟基础配置

**新增占位文件（9 个）：**
- `data/external/.gitkeep`
- `data/interim/.gitkeep`
- `data/processed/.gitkeep`
- `data/features/.gitkeep`
- `data/graph/.gitkeep`
- `data/multimodal/.gitkeep`
- `data/experiment_inputs/.gitkeep`
- `outputs/data_check/.gitkeep`
- `notebooks/.gitkeep`
- `reports/.gitkeep`

**新增日志文件（1 个）：**
- `development_log.md` — 开发日志

### 创建或确认的目录列表

**configs/ 及其子目录（7 个）：**
- `configs/common/`
- `configs/dnn/`
- `configs/wide_deep/`
- `configs/graphsage/`
- `configs/multimodal/`
- `configs/ab_test/`

**data/ 及其子目录（7 个）：**
- `data/external/`
- `data/interim/`
- `data/processed/`
- `data/features/`
- `data/graph/`
- `data/multimodal/`
- `data/experiment_inputs/`

**src/ 及其子目录（12 个）：**
- `src/data/`
- `src/features/`
- `src/models/dnn/`
- `src/models/wide_deep/`
- `src/models/graphsage/`
- `src/models/multimodal/`
- `src/evaluation/`
- `src/experiment/`
- `src/utils/`

**outputs/ 及其子目录（8 个）：**
- `outputs/data_check/`
- `outputs/dnn/`
- `outputs/wide_deep/`
- `outputs/graphsage/`
- `outputs/multimodal/`
- `outputs/comparison/`
- `outputs/ab_test/`
- `outputs/figures/`

**notebooks/、reports/：**
- `notebooks/`
- `reports/`

### 运行命令
未运行训练或数据处理命令。本次仅为目录与配置文件创建，不涉及 Python 脚本执行。

### 输入路径
- 项目根目录：`D:/CodeData/Program Coding/ByteDance/RA_PTA/`
- sample0427 数据目录：`D:/CodeData/Program Coding/ByteDance/RA_PTA/douyin_data_project/data/sample0427/`

### 输出路径
- 配置文件目录：`configs/`
- 数据中间产物目录：`data/`
- 源代码目录：`src/`
- 模型输出目录：`outputs/`
- 报告目录：`reports/`
- 笔记本目录：`notebooks/`

### 当前结果
✅ W4 基础目录结构已全部创建
✅ 8 个基础配置文件已创建
✅ 占位文件已添加
✅ `README.md`、`CLAUDE.md` 已存在，未重复写入

### 是否跑通
本次仅为工程初始化，不涉及模型运行，无跑通概念。目录与配置文件创建已完成。

### 存在的问题
1. `configs/multimodal/multimodal_base.yaml` 中 `text_input_dim`、`visual_input_dim`、`tabular_input_dim` 设置为 `null`，后续多模态数据集构建完成后需更新。
2. `configs/wide_deep/wide_deep_base.yaml` 中 `wide_cross_features` 为空列表，后续特征工程阶段需填充具体交叉特征。
3. `configs/ab_test/ab_base.yaml` 中 `group_key` 为 `null`，后续实验阶段可根据需要设置为具体分组键。
4. `configs/common/split.yaml` 中 `label_col` 暂设为 `label`，后续 tabular 数据构建阶段需正式确定。

### 下一步建议
建议进入下一阶段：sample0427 统一读取层实现（`src/data/load_sample0427.py`）。

---

## 2026-04-28 sample0427 统一读取层实现

### 任务目标
实现 sample0427 统一读取层，稳定读取 11 张 CSV 表，生成表级读取摘要和总读取报告，为后续 schema 校验、tabular 特征构建、图构建和多模态输入构建做准备。

### 修改或新增文件列表

**新增文件（4 个）：**
- `src/utils/config.py` — YAML 配置加载工具
- `src/utils/common.py` — 安全解析字符串化 list/JSON 的通用函数
- `src/utils/io.py` — 带编码回退的 CSV 读取工具
- `src/data/load_sample0427.py` — sample0427 统一读取主脚本

**修改文件（1 个）：**
- `configs/common/data_paths.yaml` — 新增 `data_check_dir` 字段

**输出文件（2 个）：**
- `outputs/data_check/sample0427_table_summary.csv` — 表级读取摘要
- `outputs/data_check/sample0427_load_report.json` — 总读取报告

### 运行命令
```text
D:/CodeData/software/Anaconda/Anaconda3/envs/ra/python.exe src/data/load_sample0427.py
```

### 输入数据路径
`D:/CodeData/Program Coding/ByteDance/RA_PTA/douyin_data_project/data/sample0427/`

### 输出文件路径
- `outputs/data_check/sample0427_table_summary.csv`
- `outputs/data_check/sample0427_load_report.json`

### 读取成功的表

| 表名 | 行数 | 列数 | 编码 |
|---|---|---|---|
| raw_video_detail | 79 | 30 | utf-8-sig |
| raw_author | 78 | 20 | utf-8-sig |
| raw_music | 79 | 16 | utf-8-sig |
| raw_hashtag | 264 | 10 | utf-8-sig |
| raw_video_tag | 162 | 5 | utf-8-sig |
| raw_video_media | 79 | 24 | utf-8-sig |
| raw_video_status_control | 79 | 17 | utf-8-sig |
| raw_chapter | 169 | 10 | utf-8-sig |
| raw_comment | 250 | 24 | utf-8-sig |
| raw_related_video | 474 | 22 | utf-8-sig |
| raw_crawl_log | 79 | 14 | utf-8-sig |
| **合计** | **1792** | **192** | |

### 缺失或读取失败的表
无。11 张表全部读取成功。

### 识别出的 list-like / json-like / raw 字段概况

**list-like 字段（基于采样检测，字符串以 `[` 开头并以 `]` 结尾）：**
- `raw_author`: avatar_thumb_url_list, cover_url_list
- `raw_music`: music_cover_url_list, music_play_url_list
- `raw_video_media`: origin_cover_url_list, dynamic_cover_url_list
- `raw_comment`: comment_user_avatar_url_list
- `raw_related_video`: related_cover_url_list, related_video_tag_raw

**json-like 字段：** 本次未检测出纯 JSON 对象类字段（所有 JSON 数组被归入 list-like）。

**raw 字段（以 `_raw` 结尾）：**
- `raw_video_media`: bit_rate_raw, big_thumbs_raw, video_meta_raw
- `raw_related_video`: related_text_extra_raw, related_video_tag_raw
- `raw_crawl_log`: response_headers_raw

### 当前结果
✅ 11 张 CSV 全部使用 utf-8-sig 编码成功读取
✅ 表级摘要 CSV 已生成（统计行数、列数、缺失值、重复行、ID 列、list-like/json-like/raw 字段等）
✅ 总读取报告 JSON 已生成
✅ 所有原始字段保留不变，未做类型转换或覆盖
✅ 字符串化字段仅做识别检测，未强制解析覆盖原值

### 是否跑通
✅ 全部 11 张表读取成功，无缺失、无失败

### 存在的问题
1. Python 3.9 不支持 `str | Path` 等 PEP 604 语法，需添加 `from __future__ import annotations` 解决。
2. 当前仅做了读取层可用性验证，未做 schema 严格校验（留待下一步 `validate_schema.py`）。
3. `related_video_tag_raw` 虽是 JSON 字符串但被检测为 list-like（因其以 `[` 开头），后续使用时需注意其实际为 JSON 数组字符串。
4. sample0427 数据字典说明部分字段整列全空（约 30 个），实际读取后这些字段被 Pandas 推断为 float64(NaN)，当前已体现在 missing_rate 中。

### 下一步建议
建议进入下一阶段：schema 校验与数据概览（`src/data/validate_schema.py`），基于 sample_data_dictionary.md 做字段级一致性检查。

---

## 2026-04-28 sample0427 schema 校验与数据概览

### 任务目标
实现 sample0427 schema 校验与数据概览，基于 sample_data_dictionary.md 和实际 CSV 读取结果，建立可维护的 schema 校验配置，输出 schema 校验报告、字段缺失报告和关键关联检查报告。

### 修改或新增文件列表

**新增文件（2 个）：**
- `configs/common/schema_sample0427.yaml` — sample0427 schema 校验配置（含 11 张表的 expected_columns、primary_key_candidates、foreign_key_candidates、important_columns、generated_or_placeholder_columns、raw_columns、list_like_columns、json_like_columns、suggested_usage、notes，以及 9 组 key_relations 和 raw_related_video 额外检查）
- `src/data/validate_schema.py` — schema 校验主脚本，复用已有 config/io/common 工具

**输出文件（4 个）：**
- `outputs/data_check/schema_validation_report.csv` — 表级校验报告
- `outputs/data_check/schema_validation_report.json` — 总校验报告
- `outputs/data_check/field_missing_report.csv` — 字段级缺失报告
- `outputs/data_check/key_relation_check_report.csv` — 关键关联检查报告

### 实际运行命令
```text
D:/CodeData/software/Anaconda/Anaconda3/envs/ra/python.exe src/data/validate_schema.py
```

### 输入数据路径
`D:/CodeData/Program Coding/ByteDance/RA_PTA/douyin_data_project/data/sample0427/`

### 输出文件路径
- `outputs/data_check/schema_validation_report.csv`
- `outputs/data_check/schema_validation_report.json`
- `outputs/data_check/field_missing_report.csv`
- `outputs/data_check/key_relation_check_report.csv`

### Schema 校验总体状态
**overall_status = warning**

### 11 张表字段校验概况

| 表名 | 行数 | 列数 | 期望列数 | 缺失列 | 额外列 | 全空字段数 | 高缺失字段数 | schema_status |
|---|---|---|---|---|---|---|---|---|
| raw_video_detail | 79 | 30 | 30 | 0 | 0 | 5 | 5 | pass |
| raw_author | 78 | 20 | 20 | 0 | 0 | 3 | 3 | pass |
| raw_music | 79 | 16 | 16 | 0 | 0 | 2 | 2 | pass |
| raw_hashtag | 264 | 10 | 10 | 0 | 0 | 0 | 0 | pass |
| raw_video_tag | 162 | 5 | 5 | 0 | 0 | 0 | 0 | pass |
| raw_video_media | 79 | 24 | 24 | 0 | 0 | 6 | 6 | pass |
| raw_video_status_control | 79 | 17 | 17 | 0 | 0 | 0 | 0 | pass |
| raw_chapter | 169 | 10 | 10 | 0 | 0 | 1 | 1 | pass |
| raw_comment | 250 | 24 | 24 | 0 | 0 | 3 | 3 | pass |
| raw_related_video | 474 | 22 | 22 | 0 | 0 | 4 | 4 | pass |
| raw_crawl_log | 79 | 14 | 14 | 0 | 0 | 3 | 3 | pass |
| **合计** | **1792** | **192** | **192** | **0** | **0** | **27** | **27** | |

### 缺失字段 / 额外字段概况
- 0 张表存在缺失字段（所有表实际字段与预期字段完全匹配）
- 0 张表存在额外字段（所有表无多余字段）
- 所有列的 expected_columns 与 actual_columns 完全一致

### 高缺失字段 / 全空字段概况
8 张表存在全空字段，共 27 个字段整列全空（均来自 sample_data_dictionary.md 已说明的空字段）：

- **raw_video_detail（5 个）**：sec_item_id, share_url, preview_title, item_title, shoot_way
- **raw_author（3 个）**：short_id, custom_verify, enterprise_verify_reason
- **raw_music（2 个）**：music_mid, music_owner_id
- **raw_video_media（6 个）**：cover_uri, video_format, video_ratio, bit_rate_raw, big_thumbs_raw, video_meta_raw
- **raw_chapter（1 个）**：chapter_cover_url
- **raw_comment（3 个）**：label_text, comment_user_sec_uid, comment_user_unique_id
- **raw_related_video（4 个）**：related_author_sec_uid, related_music_title, related_text_extra_raw, related_chapter_abstract
- **raw_crawl_log（3 个）**：request_url, request_cursor, response_headers_raw

这些空字段在 Pandas 中被推断为 float64(NaN)，均为 schema 中有定义但样本中无数据的字段。

### 关键关联检查结果

| 源表 | 源键 | 目标表 | 目标键 | 匹配率 | 状态 |
|---|---|---|---|---|---|
| raw_video_detail | author_id | raw_author | author_id | 100.00% | pass |
| raw_music | video_id | raw_video_detail | video_id | 100.00% | pass |
| raw_hashtag | video_id | raw_video_detail | video_id | 100.00% | pass |
| raw_video_tag | video_id | raw_video_detail | video_id | 100.00% | pass |
| raw_video_media | video_id | raw_video_detail | video_id | 100.00% | pass |
| raw_video_status_control | video_id | raw_video_detail | video_id | 100.00% | pass |
| raw_chapter | video_id | raw_video_detail | video_id | 100.00% | pass |
| raw_comment | video_id | raw_video_detail | video_id | 100.00% | pass |
| raw_related_video | source_video_id | raw_video_detail | video_id | 100.00% | pass |

所有 9 组关键关联均通过，匹配率 100%。
注意：raw_video_detail 本身不含 music_id 列，因此原计划检查的 video_detail.music_id → music.music_id 改为 raw_music.video_id → raw_video_detail.video_id。

### raw_related_video 额外检查
- has_source_video_id: True（474 条非空）
- has_related_video_id: True（474 条非空）
- can_serve_as_edge_source: True
- 结论：可用于 GraphSAGE video-video 边构建（流程验证级别）

### 规则生成或占位字段说明
所有 11 张表均包含规则生成或占位字段，总 83 个字段。详见表级分类：

- **raw_video_detail（1 个）**：author_id（额外补充字段）
- **raw_author（4 个）**：sec_uid, unique_id（规则生成 ID）, avatar_thumb_url_list, cover_url_list（空列表占位）
- **raw_music（6 个）**：music_id（规则生成）, music_duration, music_shoot_duration, music_collect_count（-1 占位）, music_cover_url_list, music_play_url_list（空列表占位）
- **raw_hashtag（3 个）**：hashtag_id（规则生成）, caption_start, caption_end（-1 占位）
- **raw_video_tag（3 个）**：tag_id, tag_name, tag_level（完全补齐表，所有字段为样本补齐）
- **raw_video_media（8 个）**：cover_width, cover_height, dynamic_cover_width, dynamic_cover_height, video_width, video_height, is_h265, is_long_video（-1 占位）
- **raw_video_status_control（15 个）**：全部布尔/状态字段（完全补齐表，所有字段为固定默认值）
- **raw_chapter（6 个）**：chapter_desc, chapter_detail, chapter_timestamp, chapter_abstract（从 caption 派生）, chapter_review_status, chapter_recommend_type（样本补齐）
- **raw_comment（19 个）**：comment_id 至 comment_total 除 video_id, label_text 外全部字段（完全补齐表，所有数据为模拟值）
- **raw_related_video（16 个）**：related_video_id 至 related_video_tag_raw 共 16 个字段（完全补齐表，所有数据为模拟值）
- **raw_crawl_log（2 个）**：network_response_count, runtime_objects_count（-1 占位）

5 张完全补齐表：raw_video_tag, raw_video_status_control, raw_chapter, raw_comment, raw_related_video。

### 当前结果
✅ 11 张表全部字段校验通过（0 缺失列、0 额外列）
✅ 9 组关键关联全部通过（100% 匹配率）
✅ 已识别 27 个全空字段和 83 个规则生成/占位字段
✅ raw_related_video 可用于 GraphSAGE 边构建
✅ 已输出 4 份报告（表级校验、字段缺失、关联检查、总报告）
✅ 所有样本数据的预期限制已在报告中明确标注
✅ 未修改任何原始 CSV 文件

### 是否跑通
✅ Schema 校验全部跑通，overall_status = warning（预期行为，表示结构完整但数据质量有限）

### 存在的问题
1. **原始任务说明中的 `raw_video_detail.music_id` 不存在**：raw_video_detail 不含 music_id 列，关联应通过 raw_music.video_id → raw_video_detail.video_id。已在 key_relations 中修正。
2. **27 个全空字段**：这些字段在 schema 中有定义但样本中无数据，后续 tabular 特征构建时需主动排除。
3. **5 张完全补齐表**：所有数据均为样本补齐值或固定占位值，不代表真实分布。W4 模型训练中仅可用于流程跑通，不应视为有效特征信号。
4. **Windows 终端编码问题**：print 语句中的 ⚠️（U+26A0）字符在 gbk 编码终端中导致 UnicodeEncodeError，已将特殊字符替换为 `[W]` 避免异常。

### 下一步建议
建议进入下一阶段：tabular 数据集构建（`src/data/build_tabular_dataset.py`）：
- 合并表格数据，构建 label
- 排除 27 个全空字段
- 对生成/占位字段做标注
- 为 DNN 和 Wide & Deep 模型准备统一训练输入

---

## 2026-04-28 tabular 数据集构建

### 任务目标
基于 sample0427 的 11 张表，构建 DNN 和 Wide & Deep 共用的 tabular 训练/评估数据集。当前只做数据集构建，不训练模型。

### 修改或新增文件列表

**新增文件（4 个）：**
- `configs/common/feature_tabular.yaml` — 特征构建配置（源表、join 计划、特征候选、标签配置、输出路径）
- `src/features/tabular_features.py` — 可复用特征函数（文本统计、聚合、时间戳、duration 桶化）
- `src/features/cross_features.py` — 交叉特征生成函数
- `src/data/build_tabular_dataset.py` — tabular 数据集构建主脚本

**修改文件（3 个）：**
- `configs/common/split.yaml` — 更新 train_ratio=0.8, random_seed=2026
- `configs/dnn/dnn_base.yaml` — 新增 train_data_path / eval_data_path / feature_info_path
- `configs/wide_deep/wide_deep_base.yaml` — 同上，并预填 wide_cross_features 列表

**输出文件（5 个）：**
- `data/features/tabular_train.csv` — 63 行 × 49 列
- `data/features/tabular_eval.csv` — 16 行 × 49 列
- `data/features/tabular_feature_info.json` — 特征说明
- `outputs/data_check/tabular_dataset_report.json` — 数据集报告
- `outputs/data_check/tabular_dataset_preview.csv` — 前 20 行预览

### 实际运行命令
```text
D:\CodeData\software\Anaconda\Anaconda3\envs\ra\python.exe src\data\build_tabular_dataset.py
```

### 输入数据路径
`D:/CodeData/Program Coding/ByteDance/RA_PTA/douyin_data_project/data/sample0427/`

### 输出路径
- `data/features/tabular_train.csv`
- `data/features/tabular_eval.csv`
- `data/features/tabular_feature_info.json`
- `outputs/data_check/tabular_dataset_report.json`
- `outputs/data_check/tabular_dataset_preview.csv`

### 使用的源表和关联方式

| 源表 | 关联方式 | 行数 | 是否特征来源 |
|---|---|---|---|
| raw_video_detail（主表） | — | 79 | 数值+文本+标签 |
| raw_author | LEFT JOIN on author_id（79→78，1行作者为空） | 78 | 数值（follower_count 等） |
| raw_music | LEFT JOIN on video_id | 79 | 数值+类别 |
| raw_video_media | LEFT JOIN on video_id | 79 | 数值（media size, has_watermark） |
| raw_video_status_control | 不参与特征（全部常量） | 79 | 排除 |
| raw_hashtag | 聚合到 video_id | 264→79 | 计数+top话题名 |
| raw_video_tag | 聚合到 video_id | 162→79 | 计数+标签名拼接 |
| raw_comment | 聚合到 video_id | 250→79 | 计数+avg文本长度 |
| raw_chapter | 聚合到 video_id | 169→79 | 计数+文本拼接 |
| raw_related_video | 聚合到 source_video_id | 474→79 | 计数 |
| raw_crawl_log | 不作为特征 | 79 | 排除 |

### 聚合特征说明

- **hashtag**: hashtag_count, hashtag_name_joined, hashtag_name_top
- **video_tag**: video_tag_count, video_tag_list_str
- **comment**: comment_table_count, avg_comment_text_length, max_comment_digg_count, comment_text_joined
- **chapter**: chapter_count, chapter_text_joined
- **related_video**: related_video_count

### 标签构造方法与标签分布

使用交互伪标签：
- interaction_score = digg_count + comment_count + share_count + collect_count
- 阈值：60% 分位数（34302）
- minority=40.5% > 20%，无需回退
- 标签分布：正样本 32（40.5%），负样本 47（59.5%）

### Train/Eval 切分结果

- 方法：按 video_id 随机切分（seed=2026）
- train/eval 比例：80/20
- Train：63 样本（26 正 / 37 负）
- Eval：16 样本（6 正 / 10 负）

### 特征列概况

| 类别 | 数量 | 列表 |
|---|---|---|
| ID 列 | 3 | sample_id, video_id, author_id |
| 数值特征 | 30 | duration_ms, digg_count, comment_count, share_count, collect_count, play_count, recommend_count, admire_count, is_top, is_ads, is_life_item, original, aweme_type, media_type, create_time, follower_count, total_favorited, verification_type, author_status, author_secret, author_prevent_download, origin_cover_width, origin_cover_height, has_watermark + 聚合计数（hashtag_count, video_tag_count, comment_table_count, max_comment_digg_count, chapter_count, related_video_count） |
| 类别特征 | 6 | author_id, region, music_title, music_author, hashtag_name_top, video_tag_list_str |
| 文本统计 | 4 | desc_length, desc_word_count, signature_length, avg_comment_text_length |
| Wide 交叉 | 4 | author_id_x_top_hashtag, author_id_x_music_title, duration_bucket_x_top_hashtag, has_watermark_x_duration_bucket |
| 总计 | 49（含 score+label+split） | |

### 排除字段说明

**全空字段（24 个）**：sec_item_id, share_url, preview_title, item_title, shoot_way, short_id, custom_verify, enterprise_verify_reason, music_mid, music_owner_id, cover_uri, video_format, video_ratio, bit_rate_raw, big_thumbs_raw, video_meta_raw, chapter_cover_url, label_text, comment_user_sec_uid, comment_user_unique_id, related_author_sec_uid, related_music_title, related_text_extra_raw, related_chapter_abstract

**占位/常量字段（29 个）**：cover_width, cover_height, dynamic_cover_width, dynamic_cover_height, video_width, video_height, is_h265, is_long_video, music_duration, music_shoot_duration, music_collect_count, caption_start, caption_end + 15 个 status_control 字段

**全 -1 排除字段（2 个）**：favoriting_count, following_count（样本中全部为 -1）

**不可用字段说明**：play_count 全部为 0（无区分度），generated ID 字段（music_id, hashtag_id, tag_id 等）不进入数值特征

### Warnings

1. favoriting_count / following_count 全部为 -1，已排除
2. music_author 缺失 23/79（29.1%）
3. hashtag_count 缺失 11/79（13.9%）的视频无 hashtag
4. 样本量仅 79 条，split 结果不稳定
5. play_count 全部为 0，无特征区分度

### 是否跑通
✅ 全部跑通。79 条视频成功构建 49 列表格特征数据集，已输出 train.csv / eval.csv / feature_info.json / report.json / preview.csv。

### 存在的问题
1. 样本量太小（79 条），train/eval split 稳定性差，后续需扩大至 1000+
2. 5 张完全补齐表的数据不代表真实分布，feature 仅可用于流程验证
3. 部分类别特征（hashtag_name_top, music_author）缺失率较高（14%~29%）
4. raw_author 表仅 78 条，1 条视频无 author 数据
5. raw_video_media 的 cover_url_list 存储格式与正式 schema 不一致（单条 URL 而非 JSON 数组）
6. play_count 全部为 0，可能为 API 限制导致，不是有效特征

### 下一步建议
建议进入下一阶段：DNN 最小训练、评估、预测闭环（`src/models/dnn/train.py` + `src/models/dnn/evaluate.py`）。

---

## 2026-04-29 tabular 数据集质量检查与无效字段记录补充

### 任务目标
对上一轮 tabular 数据集构建结果做轻量检查和必要补充，不修改特征逻辑、不重新切分数据、不进入模型训练。具体包括：
1. 补充 tabular_feature_info.json 和 tabular_dataset_report.json 中缺失的无效字段排除记录（excluded_all_minus_one_cols）
2. 补充 tabular_dataset_report.json 中 test split 口径说明
3. 新增 tabular_dataset_quality_check.json 质量检查报告
4. 更新 split.yaml 切分口径文档说明

### 修改或新增文件列表

**新增文件（1 个）：**
- `outputs/data_check/tabular_dataset_quality_check.json` — 结构化质量检查报告

**新增辅助脚本（1 个）：**
- `scripts/check_tabular_quality.py` — 质量检查脚本（扫描 CSV 标注常量/全零/全-1/高缺失列）

**修改文件（4 个）：**
- `data/features/tabular_feature_info.json` — 新增 `excluded_all_minus_one_cols` 字段（favoriting_count, following_count）
- `outputs/data_check/tabular_dataset_report.json` — 新增 `excluded_all_minus_one_cols` 和 `split_summary.test_split_note`/`has_independent_test`
- `configs/common/split.yaml` — 更新切分口径注释（当前无 test、eval 定位、正式阶段需启用 train/val/test 或 CV）
- `development_log.md` — 本次开发日志

### 实际运行命令
```text
D:/CodeData/software/Anaconda/Anaconda3/envs/ra/python.exe scripts/check_tabular_quality.py
```

### 输入数据路径
- `data/features/tabular_train.csv`
- `data/features/tabular_eval.csv`
- `data/features/tabular_feature_info.json`
- `outputs/data_check/tabular_dataset_report.json`
- `configs/common/feature_tabular.yaml`

### 输出文件路径
- `outputs/data_check/tabular_dataset_quality_check.json`

### 无效字段排除记录

检查前：tabular_feature_info.json 和 tabular_dataset_report.json 中已有：

| 字段类别 | 字段数 | 是否已记录 |
|---|---|---|
| excluded_all_null_cols | 24 | ✅ 已记录 |
| excluded_placeholder_cols | 28 | ✅ 已记录 |

检查后补充：

| 字段类别 | 字段数 | 字段列表 | 是否已记录 |
|---|---|---|---|
| excluded_all_minus_one_cols | 2 | favoriting_count, following_count | ⬆️ 本次补充 |
| test_split_note | - | 当前无独立 test 集 | ⬆️ 本次补充 |

新增 quality_check 报告还标记了以下**当前未排除但无区分度的特征列**（供模型训练 awareness）：

- **全零列（11 个）**：play_count, recommend_count, admire_count, is_top, is_ads, is_life_item, original, aweme_type, author_secret, author_prevent_download, has_watermark
- **非零常量列（5 个）**：media_type=4, author_status=1, related_video_count=6, region=CN + 4 个已在前面的全零列中
- **高缺失列（3 个）**：music_author (29.1%), hashtag_name_top (13.9%), hashtag_count (13.9%)

### Train/Eval/Test 切分口径确认

- **当前切分方式**：random_by_video_id, seed=2026, 80/20
- **当前 test 集**：无。当前仅使用 train/eval 切分
- **eval 定位**：用于流程验证和最小模型评估，不作为最终测试集
- **正式阶段建议**：需启用 train/val/test 三路切分或交叉验证
- **配置口径**：split.yaml 已更新注释说明，tabular_dataset_report.json 已添加 test_split_note

### 未做事项确认

本次严格遵守任务边界，以下内容未做：

1. ✅ 未修改 sample0427 原始 CSV
2. ✅ 未修改 douyin_data_project 抓取逻辑
3. ✅ 未重构 tabular 特征
4. ✅ 未重新切分 train/eval
5. ✅ 未新增 test split
6. ✅ 未实现或训练 DNN
7. ✅ 未实现或训练 Wide & Deep
8. ✅ 未构建 GraphSAGE 或多模态数据
9. ✅ 未做离线对比实验
10. ✅ 未做 A/B 模拟
11. ✅ 未把当前伪标签解释为正式业务标签
12. ✅ 未删除或移动上一轮已生成的文件

### 当前结果
✅ tabular_feature_info.json 和 tabular_dataset_report.json 的无效字段排除记录已补充完整
✅ tabular_dataset_quality_check.json 已生成（含排除字段、已标记预警字段、切分口径、缺失率）
✅ split.yaml 切分口径文档已明确（当前无 test，eval 为流程验证，正式阶段需 CV）
✅ 所有五类排除字段（all_null / placeholder / all_minus_one / constant / all_zero）在报告中均有结构化记录

### 是否跑通
✅ 全部跑通。

### 存在的问题
1. 当前特征列中有 11 个全零列和 5 个非零常量列无区分度，模型训练时需注意这些列不会提供有效信号。
2. 常量列在后续 DNN 训练中可能会导致 embedding 层或 batch norm 计算异常，建议模型训练前确认是否需进一步过滤。
3. music_author 缺失率 29.1%，需在模型训练中正确处理缺失值。

### 下一步建议
确认上述报告补充无误后，建议进入下一阶段：DNN 最小训练、评估、预测闭环（`src/models/dnn/model.py` + `src/models/dnn/train.py` + `src/models/dnn/evaluate.py`）。
