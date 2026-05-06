# Video Detail Field Extraction Audit Report

## 1. 本次修改目标

从视频详情页主响应（aweme_detail）中扩充真实可采集字段，优先覆盖
raw_video_detail、raw_author、raw_music、raw_video_media 四张表的 P0/P1
字段，使字段命名对齐 src/schemas/raw_tables.yaml。

## 2. 修改文件

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| src/crawler/browser_client.py | 修改 | 扩充 field_configs 和 merged_fields |
| src/crawler/scheduler.py | 修改 | 扩充字段映射和类型转换 |
| src/crawler/parser.py | 修改 | 扩充字段提取规则 |
| src/crawler/extractors.py | 修改 | 扩充提取器 |
| src/schemas/tables.py | 修改 | WebVideoMeta 新增字段 |
| configs/fields.yaml | 修改 | 更新提取状态 |
| audit/video_detail_extraction_report.md | 新增 | 当前文件 |
| audit/video_detail_field_check.csv | 新增 | 字段覆盖审计 |
| docs/development_log.md | 修改 | 追加本次记录 |

## 3. 当前视频详情解析入口

当前字段提取使用以下入口（按优先级）：

### 3.1 浏览器运行时响应（BrowserClient）

1. **Network response capture** (`_setup_network_listener`):
   - 拦截页面所有 JSON response
   - 通过 URL 关键词（aweme、detail、video、item、feed）筛选候选响应
   - 解析 JSON body 后递归查找 video 对象字段
   - `response.body` 中包含 `aweme_detail` 等完整 JSON 结构

2. **Runtime objects capture** (`_capture_runtime_objects`):
   - 从 `window.__INITIAL_STATE__`、`window.RENDER_DATA` 等全局对象提取
   - 通过 JavaScript evaluate 直接获取运行时对象

3. **字段合并** (`_analyze_and_save_summary`):
   - 依据 video_id match_type 选择 primary source
   - 从所有候选源合并字段，primary source 优先
   - post-process：author_page_url 构造、create_time 修正、origin_cover_url 修正

### 3.2 HTML 解析（DouyinParser）

1. **Data block analysis** (`_extract_all_data_blocks`):
   - 提取 script 标签中的 RENDER_DATA/SSR_RENDER_DATA
   - URL decode 后解析 JSON
   - 递归查找 video object

2. **Script data extraction** (`_extract_script_data` / `_parse_script_data`):
   - 从 `window.__INITIAL_STATE__` 等模式提取
   - 映射到目标字段

3. **HTML element extraction** (`_extract_html_data`):
   - 从 DOM 元素提取 desc、author、stats、cover 等

### 3.3 调度器合并（CrawlScheduler.worker）

1. 优先使用 browser_extracted_fields（浏览器运行时数据）
2. 补充 HTML 解析得到的字段
3. merged → 创建 WebVideoMeta → 保存 CSV

## 4. 当前字段覆盖情况

### 4.1 raw_video_detail 字段覆盖

| 字段 | Priority | 当前状态 | 来源路径 |
|------|----------|----------|----------|
| video_id | P0 | ✅ 已提取 | url_extraction / aweme_detail.aweme_id |
| page_url | P0 | ✅ 已提取 | 采集入口 URL |
| crawl_time | P0 | ✅ 已提取 | 采集程序生成 |
| sec_item_id | P1 | ❌ 未提取 | aweme_detail.sec_item_id |
| group_id | P1 | ❌ 未提取 | aweme_detail.group_id |
| comment_gid | P1 | ⚠️ 候选字段 | aweme_detail.comment_gid |
| share_url | P1 | ❌ 未提取 | aweme_detail.share_url |
| caption | P1 | ⚠️ 部分提取(as desc_text) | aweme_detail.caption |
| desc | P1 | ❌ 未提取 | aweme_detail.desc |
| create_time | P1 | ✅ 已提取 | aweme_detail.create_time |
| duration_ms | P1 | ✅ 已提取 | aweme_detail.duration |
| digg_count | P1 | ✅ 已提取 | aweme_detail.statistics.digg_count |
| comment_count | P1 | ⚠️ 旧名(comment_count_raw) | aweme_detail.statistics.comment_count |
| share_count | P1 | ⚠️ 旧名(share_count_raw) | aweme_detail.statistics.share_count |
| collect_count | P1 | ✅ 已提取 | aweme_detail.statistics.collect_count |
| play_count | P1 | ⚠️ 候选字段未输出 | aweme_detail.statistics.play_count |

### 4.2 raw_author 字段覆盖

| 字段 | Priority | 当前状态 | 来源路径 |
|------|----------|----------|----------|
| author_id | P0 | ✅ 已提取 | aweme_detail.author.uid |
| crawl_time | P0 | ✅ 已提取 | 采集程序生成 |
| sec_uid | P1 | ⚠️ 仅用于URL构造 | aweme_detail.author.sec_uid |
| nickname | P1 | ⚠️ 旧名(author_name) | aweme_detail.author.nickname |
| signature | P1 | ⚠️ 旧名(author_signature) | aweme_detail.author.signature |
| follower_count | P1 | ⚠️ 旧名(author_follower_count) | aweme_detail.author.follower_count |
| total_favorited | P1 | ⚠️ 旧名(author_total_favorited) | aweme_detail.author.total_favorited |

### 4.3 raw_music 字段覆盖

| 字段 | Priority | 当前状态 | 来源路径 |
|------|----------|----------|----------|
| video_id | P0 | ✅ 已提取 | 构造字段(通过video_detail) |
| crawl_time | P0 | ✅ 已提取 | 采集程序生成 |
| music_id | P1 | ❌ 未提取 | aweme_detail.music.id_str |
| music_title | P1 | ⚠️ 旧名(music_name) | aweme_detail.music.title |
| music_author | P1 | ❌ 未提取 | aweme_detail.music.author |
| music_duration | P1 | ❌ 未提取 | aweme_detail.music.duration |

### 4.4 raw_video_media 字段覆盖

| 字段 | Priority | 当前状态 | 来源路径 |
|------|----------|----------|----------|
| video_id | P0 | ✅ 已提取 | 构造字段(通过video_detail) |
| crawl_time | P0 | ✅ 已提取 | 采集程序生成 |
| cover_url_list | P1 | ✅ 已提取 | aweme_detail.video.cover.url_list |
| origin_cover_url_list | P1 | ⚠️ 旧名(origin_cover_url) | aweme_detail.video.origin_cover.url_list |
| dynamic_cover_url_list | P1 | ⚠️ 旧名(dynamic_cover_url) | aweme_detail.video.dynamic_cover.url_list |

## 5. 旧字段名处理情况

| 旧字段名 | 新字段名 | 当前代码状态 | 处理方式 |
|----------|----------|-------------|----------|
| duration_sec | duration_ms | ✅ 已使用 duration_ms | 无需修改 |
| publish_time_raw | create_time | ✅ 已使用 create_time | 无需修改 |
| like_count_raw | digg_count | ✅ 已使用 digg_count | 无需修改 |
| comment_count_raw | comment_count | ⚠️ 代码仍使用 comment_count_raw | 添加 comment_count 输出 |
| share_count_raw | share_count | ⚠️ 代码仍使用 share_count_raw | 添加 share_count 输出 |
| cover_url | cover_url_list | ✅ 已使用 cover_url_list | 无需修改 |
| origin_cover_url | origin_cover_url_list | ⚠️ 代码仍使用 origin_cover_url | 添加 origin_cover_url_list 输出 |
| dynamic_cover_url | dynamic_cover_url_list | ⚠️ 代码仍使用 dynamic_cover_url | 添加 dynamic_cover_url_list 输出 |
| desc_text | caption/desc | ⚠️ 代码仍使用 desc_text | 添加 caption/desc 输出 |

## 6. 无法确认字段

以下字段在 data_dictionary.md 中有定义路径，但浏览器响应中可能缺失或为占位值：

- `play_count`：网页端经常返回 0，需原样保存
- `preview_title`：样例中全空
- `item_title`：样例中全空
- `shoot_way`：样例中为 direct_shoot，可能稳定
- `recommend_count`：当前未添加提取路径
- `admire_count`：当前未添加提取路径

## 7. 下一步建议

1. 完成本次字段扩充后，切换到 11 张 raw 表多表输出
2. 对 comment_count / share_count 字段进行小样本非空率验证
3. 追加上游 playlist / comment list 等 API 的浏览器拦截
4. 对 media 字段（cover_url_list 等）验证 JSON 字符串化保存
