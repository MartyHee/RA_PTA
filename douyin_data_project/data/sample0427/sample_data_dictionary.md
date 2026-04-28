# sample0427 样本数据字典

## 1. 文档目的

本文档专门描述 `data/sample0427/` 样本数据集的表结构、字段定义、存储格式及与正式 `docs/data_dictionary.md` 的差异。

**用途定位：**
- 用于当前 W4 流程（DNN / Wide & Deep / GraphSAGE / 多模态 / 离线实验）的**流程验证与方案试跑**
- 是**样本级数据**，不完全等同于正式 raw 数据字典
- 部分字段来源于**样本补齐**、**规则补全**或**结构模拟**，不代表真实抖音 API 返回值
- 后续正式实验应优先使用真实抓取数据

## 2. 当前样本表清单

| 表名 | 行数 | 列数 | 数据来源 | 说明 |
|---|---|---|---|---|
| `sample0427_raw_video_detail.csv` | 79 | 30 | 真实抓取 + 字段补齐 | 79 条完全来自真实抓取高置信度样本，`author_id` 为额外补充字段 |
| `sample0427_raw_author.csv` | 78 | 20 | 真实派生 + 字段补齐 | 从视频表拆出去重，`sec_uid` / `unique_id` 来自规则补齐 |
| `sample0427_raw_music.csv` | 79 | 16 | 真实派生 + 字段补齐 | `music_id` 来自规则补齐 |
| `sample0427_raw_hashtag.csv` | 264 | 10 | 真实派生 + 字段补齐 | `hashtag_id` 来自规则补齐 |
| `sample0427_raw_video_tag.csv` | 162 | 5 | **完全补齐** | 所有字段均为样本补齐值 |
| `sample0427_raw_video_media.csv` | 79 | 24 | 真实派生 | 部分字段有真实 URL，部分为占位值 |
| `sample0427_raw_video_status_control.csv` | 79 | 17 | **完全补齐** | 所有字段均为合理默认值 |
| `sample0427_raw_chapter.csv` | 169 | 10 | **完全补齐** | 从 caption 派生的章节模拟数据 |
| `sample0427_raw_comment.csv` | 250 | 24 | **完全补齐** | 所有字段均为模拟评论数据 |
| `sample0427_raw_related_video.csv` | 474 | 22 | **完全补齐** | 所有字段均为模拟推荐数据 |
| `sample0427_raw_crawl_log.csv` | 79 | 14 | 映射补齐 | 从源 CSV 入口字段映射，部分字段为占位值 |

## 3. 每张表的数据字典

### 3.1 `sample0427_raw_video_detail`

| 字段名 | CSV 存储类型 | 目标语义 | 与 `data_dictionary.md` 一致 | 备注 |
|---|---|---|---|---|
| `video_id` | int64 | 视频 ID，主键 | ⚠️ 类型不一致 | 正式定义类型为 string；Pandas 推断为 int64 |
| `sec_item_id` | float64 | 安全化视频 ID | ⚠️ 整列全空 | 正式定义类型为 string；当前样本中无数据 |
| `group_id` | int64 | 视频分组 ID | ✅ 一致 | |
| `comment_gid` | int64 | 评论关联视频 ID | ✅ 一致 | |
| `page_url` | str | 视频页面 URL | ✅ 一致 | |
| `share_url` | float64 | 分享链接 | ⚠️ 整列全空 | 正式定义类型为 string；当前样本中无数据 |
| `caption` | str | 视频描述文本 | ✅ 一致 | 77/79 行有数据 |
| `desc` | str | 视频描述（同 caption） | ✅ 一致 | 77/79 行有数据，与 caption 基本一致 |
| `preview_title` | float64 | 页面预览标题 | ⚠️ 整列全空 | 正式定义类型为 string；当前样本中无数据 |
| `item_title` | float64 | 视频标题 | ⚠️ 整列全空 | 正式定义类型为 string；当前样本中无数据 |
| `create_time` | int64 | 发布时间 Unix 秒级时间戳 | ✅ 一致 | |
| `duration_ms` | int64 | 视频时长（毫秒） | ✅ 一致 | 值域 381~351944，为真实毫秒值 |
| `aweme_type` | int64 | 视频/内容类型编码 | ✅ 一致 | |
| `media_type` | int64 | 媒体类型编码 | ✅ 一致 | |
| `region` | str | 内容地区 | ✅ 一致 | |
| `is_top` | int64 | 是否置顶 | ✅ 一致 | |
| `is_ads` | bool | 是否广告内容 | ✅ 一致 | CSV 中为 True/False |
| `is_life_item` | bool | 是否生活服务内容 | ✅ 一致 | CSV 中为 True/False |
| `original` | int64 | 是否原创标记 | ✅ 一致 | |
| `shoot_way` | float64 | 拍摄/发布方式 | ⚠️ 整列全空 | 正式定义类型为 string；当前样本中无数据 |
| `digg_count` | int64 | 点赞数 | ✅ 一致 | 真实值（min=0, max=1,681,414） |
| `comment_count` | int64 | 评论数 | ✅ 一致 | 真实值 |
| `share_count` | int64 | 分享数 | ✅ 一致 | 真实值 |
| `collect_count` | int64 | 收藏数 | ✅ 一致 | 真实值 |
| `play_count` | int64 | 播放数 | ✅ 一致 | 全为 0（公开页面可能返回 0） |
| `recommend_count` | int64 | 推荐计数 | ✅ 一致 | |
| `admire_count` | int64 | 赞赏计数 | ✅ 一致 | |
| `crawl_time` | str | 采集时间 | ✅ 一致 | |
| `primary_source_key` | int64 | 主数据来源键 | ⚠️ 类型不一致 | 正式定义类型为 string；Pandas 推断为 int64 |
| `author_id` | int64 | 作者 UID（额外字段） | ⚠️ 正式 schema 中未列此字段 | 为方便关联从源 CSV 映射补充，粒度与 `raw_video_detail` 一致 |

### 3.2 `sample0427_raw_author`

| 字段名 | CSV 存储类型 | 目标语义 | 与 `data_dictionary.md` 一致 | 备注 |
|---|---|---|---|---|
| `author_id` | int64 | 作者 UID | ⚠️ 类型不一致 | 正式定义类型为 string；Pandas 推断为 int64 |
| `author_user_id` | int64 | 作者用户 ID | ⚠️ 类型不一致 | 正式定义类型为 int/string；Pandas 推断为 int64 |
| `sec_uid` | str | 作者安全 ID | ⚠️ **样本补齐值** | 正式定义应来自 API 响应；当前为 SHA256 规则生成，非真实值 |
| `unique_id` | str | 作者抖音号 | ⚠️ **样本补齐值** | 正式定义应来自 API 响应；当前为 MD5 规则生成，非真实值 |
| `short_id` | float64 | 作者短 ID | ⚠️ 整列全空 | 正式定义类型为 string；当前样本中无数据 |
| `nickname` | str | 作者昵称 | ✅ 一致 | 真实值 |
| `signature` | str | 作者签名 | ✅ 一致 | 74/78 行有真实数据 |
| `follower_count` | int64 | 粉丝数 | ✅ 一致 | 真实值 |
| `total_favorited` | int64 | 总获赞数 | ✅ 一致 | 真实值 |
| `favoriting_count` | int64 | 喜欢计数 | ✅ 一致 | 有真实值，部分为 -1 占位 |
| `following_count` | int64 | 关注数 | ✅ 一致 | 有真实值，部分为 -1 占位 |
| `verification_type` | int64 | 认证类型 | ✅ 一致 | 真实值 |
| `custom_verify` | float64 | 个人认证文本 | ⚠️ 整列全空 | 正式定义类型为 string；当前样本中无数据 |
| `enterprise_verify_reason` | float64 | 企业认证原因 | ⚠️ 整列全空 | 正式定义类型为 string；当前样本中无数据 |
| `avatar_thumb_url_list` | str | 作者头像 URL 列表 | ⚠️ 存储格式差异 | 正式定义类型为 ARRAY\<STRING\>；CSV 中为字符串 `'[]'`（全空） |
| `cover_url_list` | str | 作者封面 URL 列表 | ⚠️ 存储格式差异 | 正式定义类型为 ARRAY\<STRING\>；CSV 中为字符串 `'[]'`（全空） |
| `author_status` | int64 | 账号状态 | ✅ 一致 | |
| `author_secret` | int64 | 私密状态 | ✅ 一致 | |
| `author_prevent_download` | bool | 是否限制下载 | ✅ 一致 | CSV 中为 True/False |
| `crawl_time` | str | 采集时间 | ✅ 一致 | |

### 3.3 `sample0427_raw_music`

| 字段名 | CSV 存储类型 | 目标语义 | 与 `data_dictionary.md` 一致 | 备注 |
|---|---|---|---|---|
| `video_id` | int64 | 视频 ID | ✅ 一致 | |
| `music_title` | str | 音乐名称 | ✅ 一致 | 78/79 行有真实数据 |
| `music_author` | str | 音乐作者 | ✅ 一致 | 56/79 行有数据 |
| `music_id` | int64 | 音乐 ID | ⚠️ **样本补齐值 + 类型不一致** | 正式定义类型为 string/int；当前为 MD5 规则生成 ID，非真实 API 值 |
| `music_mid` | float64 | 音乐 MID | ⚠️ 整列全空 | 正式定义类型为 string/int；当前样本中无数据 |
| `music_owner_id` | float64 | 音乐归属作者 ID | ⚠️ 整列全空 | 正式定义类型为 string；当前样本中无数据 |
| `music_owner_nickname` | str | 音乐归属作者昵称 | ✅ 一致 | 56/79 行有数据 |
| `music_duration` | int64 | 音乐时长（秒） | ⚠️ 值域为占位值 | 全部为 -1，不代表真实数据分布 |
| `music_shoot_duration` | int64 | 拍摄时长 | ⚠️ 值域为占位值 | 全部为 -1 |
| `is_original_sound` | bool | 是否原创声音 | ✅ 一致 | CSV 中为 True/False |
| `is_commerce_music` | bool | 是否商业音乐 | ✅ 一致 | CSV 中为 True/False |
| `music_status` | int64 | 音乐状态 | ✅ 一致 | 值域合理（均为 1） |
| `music_collect_count` | int64 | 音乐收藏计数 | ⚠️ 值域为占位值 | 全部为 -1 |
| `music_cover_url_list` | str | 音乐封面 URL 列表 | ⚠️ 存储格式差异 + 占位值 | 正式定义类型为 ARRAY\<STRING\>；CSV 中为字符串 `'[]'`（全空） |
| `music_play_url_list` | str | 音乐播放地址列表 | ⚠️ 存储格式差异 + 占位值 | 正式定义类型为 ARRAY\<STRING\>；CSV 中为字符串 `'[]'`（全空） |
| `crawl_time` | str | 采集时间 | ✅ 一致 | |

### 3.4 `sample0427_raw_hashtag`

| 字段名 | CSV 存储类型 | 目标语义 | 与 `data_dictionary.md` 一致 | 备注 |
|---|---|---|---|---|
| `video_id` | int64 | 视频 ID | ✅ 一致 | |
| `hashtag_id` | int64 | 话题 ID | ⚠️ **样本补齐值 + 类型不一致** | 正式定义类型为 string；当前为 MD5 规则生成 ID，非真实 API 值 |
| `hashtag_name` | str | 话题名称 | ✅ 一致 | 真实值 |
| `hashtag_type` | int64 | 话题类型 | ✅ 一致 | |
| `is_commerce` | bool | 是否商业话题 | ✅ 一致 | CSV 中为 True/False |
| `caption_start` | int64 | caption 中起始位置 | ⚠️ 值域为占位值 | 全部为 -1，不代表真实分布 |
| `caption_end` | int64 | caption 中结束位置 | ⚠️ 值域为占位值 | 全部为 -1 |
| `start` | int64 | 起始位置 | ✅ 一致 | |
| `end` | int64 | 结束位置 | ✅ 一致 | |
| `crawl_time` | str | 采集时间 | ✅ 一致 | |

### 3.5 `sample0427_raw_video_tag`

| 字段名 | CSV 存储类型 | 目标语义 | 与 `data_dictionary.md` 一致 | 备注 |
|---|---|---|---|---|
| `video_id` | int64 | 视频 ID | ✅ 一致 | |
| `tag_id` | str | 标签 ID | ⚠️ **标签格式不一致 + 样本补齐值** | 正式定义类型为 int（§4.4 中 `video_tag_id`）；当前为 `TAG_LIFESTYLE` 风格字符串，完全来自规则生成 |
| `tag_name` | str | 标签名称 | ⚠️ **样本补齐值** | 来自 12 种预定义标签类别，非真实 API 值 |
| `tag_level` | int64 | 标签层级 | ⚠️ **样本补齐值** | 来自规则生成 |
| `crawl_time` | str | 采集时间 | ✅ 一致 | |

### 3.6 `sample0427_raw_video_media`

| 字段名 | CSV 存储类型 | 目标语义 | 与 `data_dictionary.md` 一致 | 备注 |
|---|---|---|---|---|
| `video_id` | int64 | 视频 ID | ✅ 一致 | |
| `cover_uri` | float64 | 封面资源 URI | ⚠️ 整列全空 | 正式定义类型为 string；当前样本中无数据 |
| `cover_url_list` | str | 视频封面 URL | ⚠️ **存储格式偏差** | 正式定义类型为 ARRAY\<STRING\>；CSV 中为**单条 URL 字符串**（非 JSON 数组），而非 URL 列表格式 |
| `cover_width` | int64 | 封面宽度 | ⚠️ 值域为占位值 | 全部为 -1 |
| `cover_height` | int64 | 封面高度 | ⚠️ 值域为占位值 | 全部为 -1 |
| `origin_cover_uri` | str | 原始封面 URI | ✅ 一致 | 78/79 行有真实值 |
| `origin_cover_url_list` | str | 原始封面 URL 列表 | ⚠️ **存储格式差异** | 正式定义类型为 ARRAY\<STRING\>；CSV 中以 JSON 字符串 `["url1"]` 形式存储 |
| `origin_cover_width` | int64 | 原始封面宽度 | ✅ 一致 | 真实值 |
| `origin_cover_height` | int64 | 原始封面高度 | ✅ 一致 | 真实值 |
| `dynamic_cover_uri` | str | 动态封面 URI | ✅ 一致 | 77/79 行有数据 |
| `dynamic_cover_url_list` | str | 动态封面 URL 列表 | ⚠️ **存储格式差异** | 正式定义类型为 ARRAY\<STRING\>；CSV 中以 JSON 字符串 `["url1"]` 形式存储 |
| `dynamic_cover_width` | int64 | 动态封面宽度 | ⚠️ 值域为占位值 | 全部为 -1 |
| `dynamic_cover_height` | int64 | 动态封面高度 | ⚠️ 值域为占位值 | 全部为 -1 |
| `video_width` | int64 | 视频宽度 | ⚠️ 值域为占位值 | 全部为 -1 |
| `video_height` | int64 | 视频高度 | ⚠️ 值域为占位值 | 全部为 -1 |
| `video_format` | float64 | 视频格式 | ⚠️ 整列全空 | 正式定义类型为 string；当前样本中无数据 |
| `video_ratio` | float64 | 视频比例 | ⚠️ 整列全空 | 正式定义类型为 string；当前样本中无数据 |
| `is_h265` | int64 | 是否 H.265 编码 | ⚠️ 值域为占位值 | 全部为 -1 |
| `is_long_video` | int64 | 是否长视频 | ⚠️ 值域为占位值 | 全部为 -1 |
| `has_watermark` | bool | 是否有水印 | ✅ 一致 | |
| `bit_rate_raw` | float64 | 视频码率元信息 | ⚠️ 整列全空 | 正式定义类型为 STRING/JSON；当前样本中无数据 |
| `big_thumbs_raw` | float64 | 缩略图信息 | ⚠️ 整列全空 | 正式定义类型为 STRING/JSON；当前样本中无数据 |
| `video_meta_raw` | float64 | 媒体元信息 | ⚠️ 整列全空 | 正式定义类型为 STRING/JSON；当前样本中无数据 |
| `crawl_time` | str | 采集时间 | ✅ 一致 | |

### 3.7 `sample0427_raw_video_status_control`

| 字段名 | CSV 存储类型 | 目标语义 | 与 `data_dictionary.md` 一致 | 备注 |
|---|---|---|---|---|
| `video_id` | int64 | 视频 ID | ✅ 一致 | |
| `can_comment` | bool | 是否允许评论 | ⚠️ **样本补齐值** | 全部为 True（合理默认值），非真实 API 响应值 |
| `can_forward` | bool | 是否允许转发 | ⚠️ **样本补齐值** | 全部为 True |
| `can_share` | bool | 是否允许分享 | ⚠️ **样本补齐值** | 全部为 True |
| `can_show_comment` | bool | 是否展示评论 | ⚠️ **样本补齐值** | 全部为 True |
| `allow_download` | bool | 是否允许下载 | ⚠️ **样本补齐值** | 全部为 False |
| `allow_duet` | bool | 是否允许合拍 | ⚠️ **样本补齐值** | 全部为 False |
| `allow_music` | bool | 是否允许使用音乐 | ⚠️ **样本补齐值** | 全部为 True |
| `allow_record` | bool | 是否允许录制 | ⚠️ **样本补齐值** | 全部为 True |
| `allow_stitch` | bool | 是否允许拼接 | ⚠️ **样本补齐值** | 全部为 False |
| `private_status` | int64 | 私密状态 | ⚠️ **样本补齐值** | 全部为 0 |
| `is_delete` | bool | 是否删除 | ⚠️ **样本补齐值** | 全部为 False |
| `is_prohibited` | bool | 是否被禁止 | ⚠️ **样本补齐值** | 全部为 False |
| `in_reviewing` | bool | 是否审核中 | ⚠️ **样本补齐值** | 全部为 False |
| `review_status` | int64 | 审核状态 | ⚠️ **样本补齐值** | 全部为 1 |
| `comment_permission_status` | int64 | 评论权限状态 | ⚠️ **样本补齐值** | 全部为 1 |
| `crawl_time` | str | 采集时间 | ✅ 一致 | |

### 3.8 `sample0427_raw_chapter`

| 字段名 | CSV 存储类型 | 目标语义 | 与 `data_dictionary.md` 一致 | 备注 |
|---|---|---|---|---|
| `video_id` | int64 | 视频 ID | ✅ 一致 | |
| `chapter_index` | int64 | 章节序号 | ✅ 一致 | |
| `chapter_desc` | str | 章节标题 | ⚠️ **样本补齐值** | 从 caption 派生（"章节1", "章节2"） |
| `chapter_detail` | str | 章节详情 | ⚠️ **样本补齐值** | 从 caption 派生 |
| `chapter_timestamp` | int64 | 章节时间戳（毫秒） | ⚠️ **样本补齐值** | 从 caption 派生，值域不代表真实 API 返回 |
| `chapter_cover_url` | float64 | 章节封面 URL | ⚠️ 整列全空 | 正式定义类型为 string；当前样本中无数据 |
| `chapter_abstract` | str | 视频智能摘要 | ⚠️ **样本补齐值** | 165/169 行有数据，从 caption 派生 |
| `chapter_review_status` | int64 | 章节审核状态 | ⚠️ **样本补齐值** | 全部为 1 |
| `chapter_recommend_type` | str | 章节推荐类型 | ⚠️ **类型不一致 + 样本补齐值** | 正式定义类型为 string/int；当前为 `'default'` 字符串 |
| `crawl_time` | str | 采集时间 | ✅ 一致 | |

### 3.9 `sample0427_raw_comment`

| 字段名 | CSV 存储类型 | 目标语义 | 与 `data_dictionary.md` 一致 | 备注 |
|---|---|---|---|---|
| `video_id` | int64 | 视频 ID | ✅ 一致 | |
| `comment_id` | str | 评论 ID | ⚠️ **样本补齐值** | 规则生成 ID，非真实 API 值 |
| `comment_text` | str | 评论文本 | ⚠️ **样本补齐值** | 模板生成文本（"学到了，谢谢分享"等），不代表真实评论分布 |
| `comment_create_time` | int64 | 评论时间戳 | ⚠️ **样本补齐值** | 规则生成，值域接近合理但非真实 |
| `comment_digg_count` | int64 | 评论点赞数 | ⚠️ **样本补齐值** | 随机生成 |
| `comment_status` | int64 | 评论状态 | ⚠️ **样本补齐值** | 全部为 1 |
| `comment_reply_total` | int64 | 回复数 | ⚠️ **样本补齐值** | 随机生成 |
| `reply_id` | int64 | 回复 ID | ⚠️ **样本补齐值** | 全部为 0 |
| `reply_to_reply_id` | int64 | 被回复的回复 ID | ⚠️ **样本补齐值** | 全部为 0 |
| `is_hot` | bool | 是否热评 | ⚠️ **样本补齐值** | 全部为 False |
| `is_author_digged` | bool | 作者是否点赞 | ⚠️ **样本补齐值** | 全部为 False |
| `stick_position` | int64 | 置顶位置 | ⚠️ **样本补齐值** | 全部为 -1 |
| `label_text` | float64 | 用户标签文本 | ⚠️ 整列全空 | 正式定义类型为 string；当前样本中无数据 |
| `label_type` | int64 | 用户标签类型 | ⚠️ **样本补齐值** | 全部为 -1 |
| `comment_user_id` | int64 | 评论用户 ID | ⚠️ **样本补齐值** | 规则生成 |
| `comment_user_sec_uid` | float64 | 评论用户安全 ID | ⚠️ 整列全空 | 正式定义类型为 string；当前样本中无数据 |
| `comment_user_nickname` | str | 评论用户昵称 | ⚠️ **样本补齐值** | 模板生成（"用户2582"等），非真实 |
| `comment_user_unique_id` | float64 | 评论用户抖音号 | ⚠️ 整列全空 | 正式定义类型为 string；当前样本中无数据 |
| `comment_user_avatar_url_list` | str | 评论用户头像 | ⚠️ 存储格式差异 + 占位值 | 正式定义类型为 ARRAY\<STRING\>；CSV 中为字符串 `'[]'`（全空） |
| `comment_user_region` | str | 评论用户地区 | ⚠️ **样本补齐值** | 全部为 'CN' |
| `comment_cursor` | int64 | 分页游标 | ⚠️ **样本补齐值** | 全部为 0 |
| `comment_has_more` | bool | 是否还有更多评论 | ⚠️ **样本补齐值** | 全部为 False |
| `comment_total` | int64 | 评论总数 | ⚠️ **样本补齐值** | 每视频 2 条 |
| `crawl_time` | str | 采集时间 | ✅ 一致 | |

### 3.10 `sample0427_raw_related_video`

| 字段名 | CSV 存储类型 | 目标语义 | 与 `data_dictionary.md` 一致 | 备注 |
|---|---|---|---|---|
| `source_video_id` | int64 | 源视频 ID | ✅ 一致 | |
| `related_video_id` | str | 推荐视频 ID | ⚠️ **样本补齐值** | 规则生成 ID（`REL...`），非真实 API 值 |
| `related_rank_position` | int64 | 推荐位置 | ⚠️ **样本补齐值** | 规则生成 |
| `related_author_id` | int64 | 推荐视频作者 ID | ⚠️ **样本补齐值** | 规则生成 |
| `related_author_sec_uid` | float64 | 推荐视频作者安全 ID | ⚠️ 整列全空 | 正式定义类型为 string；当前样本中无数据 |
| `related_author_nickname` | str | 推荐视频作者昵称 | ⚠️ **样本补齐值** | 模板生成（"推荐作者_489"等） |
| `related_caption` | str | 推荐视频描述 | ⚠️ **样本补齐值** | 模板生成 |
| `related_create_time` | int64 | 推荐视频发布时间 | ⚠️ **样本补齐值** | 规则生成时间戳 |
| `related_duration_ms` | int64 | 推荐视频时长（毫秒） | ⚠️ **样本补齐值** | 全部为 1000（固定占位值，不代表真实分布） |
| `related_media_type` | int64 | 推荐媒体类型 | ⚠️ **样本补齐值** | 全部为 4 |
| `related_digg_count` | int64 | 推荐点赞数 | ⚠️ **样本补齐值** | 随机生成，不代表真实分布 |
| `related_comment_count` | int64 | 推荐评论数 | ⚠️ **样本补齐值** | 随机生成 |
| `related_share_count` | int64 | 推荐分享数 | ⚠️ **样本补齐值** | 随机生成 |
| `related_collect_count` | int64 | 推荐收藏数 | ⚠️ **样本补齐值** | 随机生成 |
| `related_play_count` | int64 | 推荐播放数 | ⚠️ **样本补齐值** | 随机生成 |
| `related_cover_url_list` | str | 推荐视频封面 URL 列表 | ⚠️ 存储格式差异 + 占位值 | 正式定义类型为 ARRAY\<STRING\>；CSV 中为字符串 `'[]'`（全空） |
| `related_music_id` | int64 | 推荐视频音乐 ID | ⚠️ **样本补齐值 + 类型不一致** | 正式定义类型为 string/int；当前从 `video_id→music_id` 映射规则补齐 |
| `related_music_title` | float64 | 推荐视频音乐名称 | ⚠️ 整列全空 | 正式定义类型为 string；当前样本中无数据 |
| `related_text_extra_raw` | float64 | 推荐视频话题列表 | ⚠️ 整列全空 | 正式定义类型为 STRING/JSON（list[object]）；当前样本中无数据 |
| `related_video_tag_raw` | str | 推荐视频平台分类标签 | ⚠️ **样本补齐值** | 正式定义类型为 STRING/JSON（list[object]）；当前为规则生成 JSON 字符串 |
| `related_chapter_abstract` | float64 | 推荐视频智能摘要 | ⚠️ 整列全空 | 正式定义类型为 string；当前样本中无数据 |
| `crawl_time` | str | 采集时间 | ✅ 一致 | |

### 3.11 `sample0427_raw_crawl_log`

| 字段名 | CSV 存储类型 | 目标语义 | 与 `data_dictionary.md` 一致 | 备注 |
|---|---|---|---|---|
| `crawl_batch_id` | str | 采集批次 ID | ✅ 一致 | |
| `crawl_time` | str | 采集时间 | ✅ 一致 | |
| `target_url` | str | 原始采集 URL | ✅ 一致 | |
| `source_page_type` | str | 来源页面类型 | ✅ 一致 | |
| `request_url` | float64 | 网络响应 URL | ⚠️ 整列全空 | 正式定义类型为 string；当前样本中无数据 |
| `request_cursor` | float64 | 分页游标 | ⚠️ 整列全空 | 正式定义类型为 int；当前样本中无数据 |
| `response_status` | int64 | HTTP 状态码 | ✅ 一致 | 全部为 200 |
| `response_timestamp` | str | 响应采集时间 | ✅ 一致 | |
| `response_headers_raw` | float64 | 响应头 | ⚠️ 整列全空 | 正式定义类型为 STRING/JSON；当前样本中无数据 |
| `primary_source_key` | int64 | 主数据来源键 | ⚠️ 类型不一致 | 正式定义类型为 string；Pandas 推断为 int64 |
| `match_type` | str | 主对象匹配类型 | ✅ 一致 | |
| `confidence` | str | 选择置信度 | ✅ 一致 | |
| `network_response_count` | int64 | 网络响应数量 | ⚠️ 值域为占位值 | 全部为 -1 |
| `runtime_objects_count` | int64 | 运行时对象数量 | ⚠️ 值域为占位值 | 全部为 -1 |

## 4. 与正式 `data_dictionary.md` 的差异说明

### 4.1 表级别对齐情况

| 表名 | 对齐等级 | 说明 |
|---|---|---|
| `raw_video_detail` | 部分一致 | 字段名基本对齐；但有 1 个额外字段（`author_id`）；5 个字段整列全空；少量类型偏差 |
| `raw_author` | 部分一致 | 字段名全部对齐；2 个关键字段（`sec_uid`, `unique_id`）为样本补齐值；2 个字段整列全空；2 个 ARRAY 字段为占位空串 |
| `raw_music` | 部分一致 | 字段名全部对齐；`music_id` 为样本补齐值；3 个字段整列全空；3 个数值字段为 -1 占位；2 个 ARRAY 字段为空 |
| `raw_hashtag` | 部分一致 | 字段名全部对齐；`hashtag_id` 为样本补齐值；2 个位置字段为 -1 占位 |
| `raw_video_tag` | 结构对齐 | 字段名对齐但**所有数据为样本补齐值**；`tag_id` 类型和值格式均偏离正式定义 |
| `raw_video_media` | 部分一致 | 字段名全部对齐；6 个字段整列全空；6 个数值字段为 -1 占位；`cover_url_list` 存储格式偏离正式定义 |
| `raw_video_status_control` | 结构对齐 | 字段名和类型全部对齐但**所有数据为样本补齐默认值** |
| `raw_chapter` | 结构对齐 | 字段名基本对齐；`chapter_recommend_type` 类型不一致；所有数据为样本补齐值 |
| `raw_comment` | 结构对齐 | 字段名全部对齐；4 个字段整列全空；**所有有数据的字段均为样本补齐值** |
| `raw_related_video` | 结构对齐 | 字段名全部对齐；5 个字段整列全空；**所有有数据的字段均为样本补齐值** |
| `raw_crawl_log` | 部分一致 | 字段名全部对齐；4 个字段整列全空；2 个字段为 -1 占位 |

### 4.2 字段类型差异汇总

| 差异类型 | 涉及字段 | 说明 |
|---|---|---|
| string → int64（Pandas 推断） | `video_id`, `author_id`, `author_user_id`, `hashtag_id`, `primary_source_key`, `comment_user_id` | 这些字段在正式 schema 中定义为 string/ID 类型，但 Pandas 读取 CSV 时将纯数字列推断为 int64 |
| string → float64（空列） | `share_url`, `preview_title`, `item_title`, `shoot_way`, `short_id`, `custom_verify`, `enterprise_verify_reason`, `music_mid`, `music_owner_id`, `cover_uri`, `video_format`, `video_ratio`, `bit_rate_raw`, `big_thumbs_raw`, `video_meta_raw`, `chapter_cover_url`, `related_author_sec_uid`, `related_music_title`, `related_text_extra_raw`, `related_chapter_abstract`, `comment_user_sec_uid`, `comment_user_unique_id`, `label_text`, `request_url`, `request_cursor`, `response_headers_raw` | 整列全空的 string 字段被 Pandas 读为 float64(NaN) |
| ARRAY\<STRING\> → STRING | `avatar_thumb_url_list`, `cover_url_list`（author）, `music_cover_url_list`, `music_play_url_list`, `origin_cover_url_list`, `dynamic_cover_url_list`, `related_cover_url_list`, `comment_user_avatar_url_list` | 逻辑类型为 ARRAY\<STRING\> 的字段在 CSV 中一律以字符串形式存储（空列表为 `'[]'`，有值时为 JSON 字符串） |
| STRING/JSON → STRING | `related_video_tag_raw`, `bit_rate_raw`, `big_thumbs_raw`, `video_meta_raw`, `response_headers_raw` | 逻辑类型为 JSON 的字段在 CSV 中以字符串形式存储 |
| list[object] → STRING | `related_video_tag_raw` | 逻辑上为对象数组，CSV 中为 JSON 字符串 |
| int → STRING（补齐风格） | `tag_id` | 正式定义应为 int 类型的标签 ID，样本中使用 `TAG_LIFESTYLE` 风格字符串 |
| int/string → STRING | `chapter_recommend_type` | 正式定义可为 int，样本中全部为 `'default'` 字符串 |

### 4.3 字段语义差异汇总

| 语义差异类型 | 涉及字段 | 说明 |
|---|---|---|
| **样本补齐值（规则生成，非真实 API）** | `sec_uid`, `unique_id`, `music_id`, `hashtag_id`, `related_music_id`, `related_video_tag_raw` | 这些字段通过哈希/规则生成 ID，虽格式接近真实但值无实际含义 |
| **完全补齐表（无真实数据）** | raw_video_tag, raw_video_status_control, raw_chapter, raw_comment, raw_related_video | 这 5 张表的全部字段均为结构性补齐，数据不代表真实 API 响应分布 |
| **占位默认值（-1 / 0 / False）** | `music_duration`, `music_collect_count`, `video_width`, `video_height`, `cover_width`, `cover_height` 等 | 多个数值字段全部为 -1，仅表示"有值"但无实际语义 |
| **全空字段（schema 有定义但无数据）** | 约 30 个字段（见 4.2 中 float64 空列） | 这些字段在 schema 中已定义，但当前样本中整列为空，不应作为特征输入 |
| **固定值字段** | `related_duration_ms`（全 1000）, `comment_user_region`（全 'CN'）, `source_page_type`（全 'manual_url'） | 这些字段仅有单一固定值，无区分度 |

## 5. 当前使用建议

### 5.1 可用范围

- ✅ **W4 流程验证**：可用于 DNN、Wide & Deep、GraphSAGE、多模态模型、离线实验的 pipeline 验证
- ✅ **Schema 兼容性测试**：字段名与正式 schema 基本对齐，可作为 ETL 流程测试输入
- ✅ **特征工程逻辑验证**：可验证特征派生逻辑是否正确
- ✅ **模型训练流程跑通**：可验证训练脚本能否正常执行
- ✅ **小规模原型实验**：79 条视频、474 条推荐边可用于原型验证

### 5.2 使用限制

- ❌ **不应直接当作严格真实 raw 数据全集用于建模**
- ❌ **补齐表（comment / related_video / video_tag / video_status_control / chapter）的值分布不代表真实数据分布**
- ❌ **规则补齐字段（sec_uid / unique_id / music_id / hashtag_id / related_music_id）的值不可用于去重、关联外部数据或特征交叉**
- ❌ **全空字段（约 30 个）在特征工程中应直接排除处理**
- ❌ **-1 占位值不应参与统计计算或归一化**
- ⚠️ **`duration_ms` 在 raw_video_detail 中为真实毫秒值，在 raw_related_video 中为固定占位 1000，使用时需区分来源表**
- ⚠️ **`digg_count` 等互动统计在真实表中为真实值，在补齐表中为随机值，不可混用分析**

### 5.3 后续建议

1. **用真实抓取逐步替换补齐表**：优先替换 raw_related_video（对 GraphSAGE 最关键）和 raw_comment
2. **填补全空字段**：在后续抓取中补充 `sec_item_id`、`share_url`、`preview_title`、`item_title` 等字段
3. **修复 `cover_url_list` 存储格式**：统一为 JSON 数组字符串或直接存储单条 URL
4. **对齐 Pandas 类型推断**：关键 ID 字段在读取时指定 dtype=str，避免 int64 截断
5. **正式实验时扩大样本量**：当前 79 条视频对模型训练过于稀疏，建议扩大到 1000+
