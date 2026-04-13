# 样本数据

本目录包含抖音数据项目的样本数据文件，用于开发和测试。

## 文件说明

### `sample_web_video_meta.csv`
- **描述**: 网页视频元数据样本
- **格式**: CSV
- **行数**: 3
- **字段**: 包含 `web_video_meta` 表的所有字段
- **用途**: 用于测试数据清洗和解析流程

### `sample_processed_video_data.csv`
- **描述**: 处理后视频数据样本
- **格式**: CSV
- **行数**: 3
- **字段**: 包含 `processed_video_data` 表的所有字段
- **用途**: 用于测试特征工程和EDA流程

## 数据说明

1. **数据来源**: 模拟数据，基于真实抖音数据模式生成
2. **数据量**: 小样本（3条记录），便于快速测试
3. **字段值**: 使用中文和典型抖音数据模式
4. **时间范围**: 2023年10月1日至3日
5. **互动指标**: 点赞、评论、分享等指标符合典型分布

## 使用方法

### 1. 测试数据清洗
```bash
python run_clean.py --input data/samples/sample_web_video_meta.csv
```

### 2. 测试特征工程
```bash
python run_features.py --input data/interim/cleaned_video_data.parquet
```

### 3. 测试EDA
```bash
python run_eda.py --input data/processed/featured_video_data.parquet
```

### 4. 使用Mock模式
```bash
# 各脚本都支持--mock参数，无需真实数据
python run_clean.py --mock
python run_features.py --mock
python run_eda.py --mock
```

## 字段映射

### 网页视频元数据字段
| 字段名 | 类型 | 描述 | 示例 |
|--------|------|------|------|
| video_id | STRING | 视频唯一标识 | video_001 |
| page_url | STRING | 视频页面URL | https://www.douyin.com/video/... |
| author_id | STRING | 作者ID | author_001 |
| author_name | STRING | 作者昵称 | 美食博主 |
| desc_text | STRING | 视频文案/标题 | 今天吃了超级好吃的火锅 #美食 #火锅 |
| publish_time_std | TIMESTAMP | 标准化发布时间 | 2023-10-01 12:00:00 |
| like_count | BIGINT | 点赞数 | 12000 |
| comment_count | BIGINT | 评论数 | 450 |
| share_count | BIGINT | 分享数 | 120 |
| hashtag_list | ARRAY<STRING> | 话题标签列表 | ["美食", "火锅"] |
| source_entry | STRING | 数据来源 | search/topic/manual_url |
| crawl_time | TIMESTAMP | 抓取时间 | 2024-04-13 10:00:00 |

### 处理后视频数据字段
| 字段名 | 类型 | 描述 | 示例 |
|--------|------|------|------|
| video_id | STRING | 视频唯一标识 | video_001 |
| desc_clean | STRING | 清洗后文案 | 今天吃了超级好吃的火锅 #美食 #火锅 |
| text_length | INT | 文案长度 | 12 |
| publish_date | DATE | 发布日期 | 2023-10-01 |
| publish_hour | INT | 发布时间小时 | 12 |
| publish_weekday | INT | 星期几（0=周一） | 6 |
| is_weekend | INT | 是否周末 | 1 |
| hashtag_count | INT | 话题数量 | 2 |
| like_count | BIGINT | 点赞数 | 12000 |
| comment_count | BIGINT | 评论数 | 450 |
| share_count | BIGINT | 分享数 | 120 |
| engagement_score | FLOAT | 互动分数 | 13440.0 |
| source_entry | STRING | 数据来源 | search |
| crawl_date | DATE | 抓取日期 | 2024-04-13 |
| data_version | STRING | 数据版本 | v0.1 |

## 注意事项

1. 样本数据仅供测试，不包含真实用户信息
2. 互动指标为模拟数据，仅用于演示计算逻辑
3. 时间戳使用ISO 8601格式
4. 数组字段使用JSON数组格式表示
5. 所有文件使用UTF-8编码

## 扩展建议

如需更多测试数据：
1. 使用 `run_crawl.py --mock` 生成更多mock数据
2. 修改样本数据中的数值范围
3. 添加新的字段或数据变体