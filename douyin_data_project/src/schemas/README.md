# src/schemas/ — Raw Schema Definitions

## 用途

本目录存放 11 张 raw 表的目标 schema 定义、字段来源映射和轻量 schema 加载工具。为后续 parser / scheduler 多表输出改造提供统一 schema 来源。

## 文件说明

| 文件 | 说明 |
|------|------|
| `raw_tables.yaml` | 11 张 raw 表的目标 schema 定义（粒度、主键、外键、必填字段、可选字段、分类） |
| `field_sources.yaml` | 逐字段字段来源与采集策略（响应路径、是否原生字段、优先级、备注） |
| `raw_schema.py` | 轻量 schema 加载与验证工具（YAML 读取 → Python dict） |
| `README.md` | 本文件 |

此外，`tables.py` 是历史上基于 Pydantic 的 schema 定义文件，将逐步过渡到本目录的 YAML + 轻量工具方案。

## raw_tables.yaml 结构

每张表包含以下属性：

| 属性 | 说明 |
|------|------|
| `table_name` | 表名 |
| `grain` | 粒度说明 |
| `description` | 表描述 |
| `primary_key_candidates` | 候选主键 |
| `foreign_key_candidates` | 候选外键 |
| `required_fields` | 必填字段（P0） |
| `optional_fields` | 可选字段 |
| `quality_fields` | 质量字段（匹配类型、置信度等） |
| `list_like_fields` | 列表类型字段（ARRAY） |
| `json_like_fields` | JSON 类型字段 |
| `timestamp_fields` | 时间戳字段 |
| `numeric_fields` | 数值字段 |
| `text_fields` | 文本字段 |
| `bool_fields` | 布尔字段 |
| `source_response_types` | 字段来源响应类型列表 |
| `notes` | 备注 |

## field_sources.yaml 结构

每个字段包含以下属性：

| 属性 | 说明 |
|------|------|
| `table_name` | 所属表 |
| `field_name` | 字段名 |
| `source_type` | 来源类型（video_detail_response / comment_response / related_video_response / constructed / constructed_quality_field / crawler_runtime / unknown） |
| `source_path_or_hint` | 响应路径或构造来源说明 |
| `is_native_response_field` | 是否为响应原生字段 |
| `is_constructed_field` | 是否为构造字段 |
| `allow_null` | 是否允许为空 |
| `priority` | 优先级 P0(必填) / P1(高) / P2(中) / P3(低) |
| `notes` | 备注 |

## raw_schema.py 使用方式

```python
from src.schemas.raw_schema import (
    load_raw_tables_schema,
    load_field_sources,
    get_table_columns,
    get_all_raw_table_names,
    validate_table_columns,
    build_empty_raw_record,
    find_field,
)

# 加载全部 schema
schema = load_raw_tables_schema()

# 获取所有表名
tables = get_all_raw_table_names()

# 获取某表的列顺序（required + optional + quality）
cols = get_table_columns("raw_video_detail")

# 验证实际数据列是否匹配 schema
result = validate_table_columns("raw_video_detail", df.columns.tolist())

# 生成空记录 dict
record = build_empty_raw_record("raw_comment")

# 查找某个字段的来源定义
field_info = find_field("raw_music", "music_id")

# 直接运行自检
python src/schemas/raw_schema.py
```

## 关于 sample0427

`data/sample0427/` 的 CSV 文件仅作为**结构参考**（表头、字段名示意），不作为真实字段来源依据。原因：

1. 5 张表（raw_video_tag、raw_video_status_control、raw_chapter、raw_comment、raw_related_video）的所有字段均为样本补齐值。
2. 部分 string 类型字段被 Pandas 推断为 int64（如 video_id、author_id、music_id）。
3. 部分 ARRAY 类型字段（如 avatar_thumb_url_list）在 CSV 中存储为字符串 `'[]'`。
4. 部分字段（如 sec_uid、unique_id、hashtag_id）为规则生成值，非真实 API 返回。

本目录的 schema 定义已经考虑了这些差异。

## 后续使用

1. **parser 层**：应使用 `build_empty_raw_record(table_name)` 初始化多表输出，然后填充字段。
2. **scheduler 层**：应使用 `get_table_columns(table_name)` 确定每表的输出列顺序。
3. **字段验证**：`validate_table_columns()` 可在采集后用于字段对齐检查。
4. **字段扩充**：新增字段时，应同时在 `raw_tables.yaml` 和 `field_sources.yaml` 中添加定义。
