# 爬虫项目现状审计与字段差距分析

> 审计日期: 2026-05-06
> 审计方法: 静态代码阅读 + schema 交叉比对
> 审计范围: src/crawler/, src/schemas/tables.py, configs/, docs/

---

## 1. 功能特性支持现状

| 特性 | 状态 | 所在文件 | 说明 |
|------|------|----------|------|
| Browser 模式 | ✅ 已支持 | `browser_client.py` | Playwright 浏览器抓取，独立 page 隔离 |
| Mock 模式 | ✅ 已支持 | `browser_client.py` `_mock_response()` | 生成 10 字段 mock 数据 |
| URL 文件批量输入 | ✅ 已支持 | `run_crawl.py` `--url-file` | 支持 `N URL` 和裸 URL 格式 |
| URL 发现器 | ✅ 已支持 | `run_discover_urls.py` | 从精选页发现，5 个默认来源 |
| Run ID 时间戳目录 | ✅ 已支持 | `scheduler.py` | `%Y%m%d_%H%M%S` 格式 |
| 每 URL 独立 page | ✅ 已支持 | `browser_client.py` `get()` | `new_page()` + `finally: close()` |
| 定期重启 browser/context | ✅ 已支持 | `--restart-every` | 可配置重启间隔 |
| Debug 模式 | ✅ 已支持 | `--debug-mode` | 默认关闭，排查时开启 |
| 高置信度筛选 | ✅ 已支持 | `scheduler.py` `_filter_and_save_high_confidence_samples()` | match_type=exact + confidence=high + video_id 一致性 |
| 失败 URL 记录 | ✅ 已支持 | `scheduler.py` `failed_urls.csv` | 记录失败 URL 和原因 |
| 质量报告 | ✅ 已支持 | `scheduler.py` `_generate_quality_report()` | 匹配分布、置信度、video_id 一致性 |
| 匹配质量字段 | ✅ 已支持 | `browser_client.py` `merged_fields` | match_type, confidence, matched_object_id, is_primary_match |
| video_id 一致性检查 | ✅ 已支持 | `scheduler.py` `_check_video_id_consistency()` | 确保 video_id 与 URL 目标 ID 一致 |
| 防反爬机制 | ⚠️ 基础 | `anti_block.py` | UA 轮换 + 请求延迟 + 冷却，无代理池 |

---

## 2. 11 张 Raw 表支持现状

| 表名 | 状态 | 当前行数(最近批) | 目标字段数 | 当前字段数 | 字段覆盖率 | 说明 |
|------|------|-----------------|-----------|-----------|-----------|------|
| raw_video_detail | ⚠️ 部分支持 | 15 | ~29 | ~17 | ~59% | 独缺 sec_item_id, share_url, preview_title, item_title, play_count, recommend_count, admire_count, shoot_way |
| raw_author | ⚠️ 部分支持 | 派生在 video_detail 中 | ~15 | ~7 | ~47% | 缺少 short_id, sec_uid, unique_id, avatar_thumb_url_list, enterprise_verify_reason 等 |
| raw_music | ❌ 极少支持 | 仅 music_name | ~11 | ~1 | ~9% | 只提取了 music_name，缺少 music_id, music_author, music_duration 等 |
| raw_hashtag | ⚠️ 部分支持 | 264(sample) | ~8 | ~4 | ~50% | hashtag_list 已提取但未拆为多行，缺少 hashtag_id, hashtag_type |
| raw_video_tag | ❌ 不支持 | — | ~3 | 0 | 0% | 无任何提取逻辑 |
| raw_video_media | ⚠️ 部分支持 | 仅封面 URL | ~17 | ~3 | ~18% | cover_url_list, origin_cover_url, dynamic_cover_url 已提取，缺少宽高、URI、码率等 |
| raw_video_status_control | ❌ 不支持 | — | ~10 | 0 | 0% | 无任何提取逻辑 |
| raw_chapter | ❌ 不支持 | — | ~7 | 0 | 0% | 无任何提取逻辑 |
| raw_comment | ❌ 不支持 | — | ~17 | 0 | 0% | 无评论 API 响应监听与解析 |
| raw_related_video | ❌ 不支持 | — | ~16 | 0 | 0% | 无相关推荐 API 响应监听与解析 |
| raw_crawl_log | ⚠️ 部分支持 | 15 | ~11 | ~5 | ~45% | 已有 batch_id, crawl_time, target_url, match_type, confidence；缺少 request_url, response 明细 |

**当前输出结构**: `data/interim/<run_id>/real_web_video_meta_<run_id>.csv` (单表)
**目标输出结构**: `data/raw/<run_id>/raw_*.parquet` (11 表)

---

## 3. 字段命名合规性检查

对照 `docs/data_dictionary.md` 命名约定:

| 约定字段 | 禁用字段 | 当前代码状态 | 结论 |
|----------|----------|-------------|------|
| `duration_ms` | `duration_sec` | 代码已使用 `duration_ms` ✅ | 通过 |
| `create_time` | `publish_time_raw` | browser_client 已使用 `create_time` ✅; parser.py 仍用 `publish_time_std` ⚠️ | 基本通过 |
| `digg_count` | `like_count_raw` | 代码已使用 `digg_count` ✅ | 通过 |
| `comment_count` | `comment_count_raw` | 代码仍用 `comment_count_raw` ⚠️ | 待修正 |
| `share_count` | `share_count_raw` | 代码仍用 `share_count_raw` ⚠️ | 待修正 |
| `cover_url_list` | `cover_url` | browser_client 已使用 `cover_url_list` ✅ | 通过 |
| `origin_cover_url_list` | `origin_cover_url` | browser_client 仍用 `origin_cover_url` ⚠️ | 待修正 |
| `dynamic_cover_url_list` | `dynamic_cover_url` | browser_client 仍用 `dynamic_cover_url` ⚠️ | 待修正 |
| `author_page_url` | `author_profile_url` | 代码已使用 `author_page_url` ✅; fields.yaml 仍用 `author_profile_url` ⚠️ | 代码通过 |

**`configs/fields.yaml` 严重滞后**: 该文件仍大量使用旧命名，与当前代码已不一致。

---

## 4. 代码路径 vs 数据字典字段交叉分析

### 4.1 `browser_client.py` `merged_fields` (实际提取字段集, line ~680)

当前仅 ~21 字段在合并集中:

```
video_id, author_id, author_name, author_page_url, desc_text,
create_time, digg_count, comment_count_raw, share_count_raw,
collect_count, hashtag_list, origin_cover_url, music_name, duration_ms,
author_follower_count, author_total_favorited, author_signature,
author_verification_type, cover_url_list, dynamic_cover_url, origin_cover_url
```

### 4.2 缺失的高优先级字段

从数据字典 §4.1 raw_video_detail 中缺失的高优先级字段:

| 字段 | 期望来源路径 | 优先级 | 缺失原因 |
|------|------------|--------|---------|
| `sec_item_id` | `aweme_detail.video.sec_item_id` | 高 | merged_fields 未包含 |
| `group_id` | `aweme_detail.group_id` | 高 | merged_fields 未包含 |
| `share_url` | `aweme_detail.share_url` | 中 | merged_fields 未包含 |
| `play_count` | `aweme_detail.statistics.play_count` | 高 | 网页端可能为 0 |
| `aweme_type` | `aweme_detail.aweme_type` | 中 | merged_fields 未包含 |
| `media_type` | `aweme_detail.media_type` | 中 | merged_fields 未包含 |
| `region` | `aweme_detail.region` | 中 | merged_fields 未包含 |
| `is_ads` | `aweme_detail.is_ads` | 中 | merged_fields 未包含 |
| `music_id` | `aweme_detail.music.id_str` | 高 | 仅提取 music_name |

### 4.3 评论/相关推荐/章节/标签/状态控制

**完全缺失**。当前代码无任何以下响应监听或解析逻辑:
- 评论列表 API (`/aweme/v1/web/comment/list/`)
- 相关推荐 API (`/aweme/v1/web/related/`)  
- 视频标签 `aweme_detail.video_tag[]`
- 章节 `aweme_detail.chapter[]`
- 状态控制 `aweme_detail.status`, `aweme_detail.download_status` 等

---

## 5. 配置与代码不一致问题

| 文件 | 问题 | 影响 |
|------|------|------|
| `configs/fields.yaml` | 使用 `publish_time_raw`, `like_count_raw`, `duration_sec`, `cover_url`, `author_profile_url` | schema 定义与代码实际输出不匹配 |
| `configs/fields.yaml` | 包含 `like_count`, `comment_count`, `share_count` (无后缀版本) | 与当前字段名 `digg_count`, `comment_count_raw` 等冲突 |
| `parser.py` `mock_parse()` | 仍输出 `publish_time_std` | browser_client 已改为 `create_time`，不一致 |
| `parser.py` `_parse_script_data()` | 提取映射使用 `diggCount`, `commentCount`, `shareCount` 内部键名 | 中间层映射键名，不完全对齐最终输出名 |
| `src/schemas/tables.py` | `WebVideoMeta` 字段不完整 | 缺少 11 表 raw schema 定义 |
| `src/schemas/tables.py` | `duration_ms` validator 有误 (毫秒判定阈值 360000) | 部分时长 >6min 的视频会被错误转换 |

---

## 6. 阶段就绪度评估

| 阶段 | 就绪度 | 阻塞点 |
|------|--------|--------|
| 阶段1: 字段扩充 | ⚠️ 部分就绪 | merged_fields 需从 21 扩充至 ~50+，需添加多表输出逻辑 |
| 阶段2: 小样本验证 (5-10 URL) | ✅ 就绪 | 单表输出可用于验证，无需等待多表 |
| 阶段3: 小批量 (20-50 URL) | ⚠️ 条件就绪 | 需先完成多表输出才能验证所有表 |
| 阶段4: 中批量 (200-300) | ❌ 未就绪 | 需先完成字段扩充和多表输出 |
| 阶段5: 大批量 (1000+) | ❌ 未就绪 | 浏览器稳定性需进一步验证 |

---

## 7. 推荐执行路径

### 短期 (可并行):
1. 清理 `configs/fields.yaml` 中的旧字段名
2. 修正 `parser.py` 中的 `publish_time_std` 输出
3. 规范 `comment_count_raw` → `comment_count`, `share_count_raw` → `share_count`
4. 修正 `origin_cover_url` → `origin_cover_url_list`, `dynamic_cover_url` → `dynamic_cover_url_list`

### 中期:
5. 扩充 `merged_fields` 加入高优先级缺失字段 (sec_item_id, group_id, play_count, music_id 等)
6. 添加评论列表响应监听与解析 (网络请求拦截 `/aweme/v1/web/comment/list/`)
7. 添加相关推荐响应监听与解析 (网络请求拦截 `/aweme/v1/web/related/`)
8. 实现多表输出: 从单表 `real_web_video_meta` → 11 张 raw 表

### 长期:
9. 添加 video_tag 和 status_control 字段提取
10. 添加 chapter 字段提取
11. 切到 `data/raw/<run_id>/` 输出 + parquet 格式
12. 执行小批量 → 中批量 → 大批量采集

---

## 8. 关键风险

1. **fields.yaml 长期未维护** — 可能导致新开发者错误理解当前 schema
2. **评论/相关推荐无拦截** — 需要确认 Playwright 网络拦截是否能正确捕获分页响应
3. **单表 → 多表重构** — 涉及 scheduler.py 核心输出逻辑，需注意向后兼容
4. **sample0427 中 5 张完全补齐表** — 意味着当前没有任何真实提取经验可参考
