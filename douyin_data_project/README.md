# 抖音公开数据采集与处理项目

本项目是一个用于采集、处理和分析抖音公开数据的可扩展工程骨架。

## 项目目标

构建一个**可落地、可复现、可用于后续推荐实验设计**的数据集，支持：

1. 网页端公开视频页面数据采集
2. 数据清洗与特征工程
3. 探索性数据分析（EDA）
4. 为推荐系统提供数据基础

## 项目状态（当前阶段：最小可运行骨架）

本项目当前处于**最小可运行骨架**阶段，边界如下：

| 模块                     | 当前能力                                                                                                                                                                                 | 边界说明                                                                       |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **crawler**        | 支持输入一个或多个测试 URL；抓取 HTML；预留 headers/cookies/delay/retry 接口；先用 mock/sample HTML 演示解析少量字段；输出 raw 层样例文件                                                | 真实抓取需配置反爬策略，当前以 mock 模式为主                                   |
| **processing**     | 提供 clean、transform、feature_engineering、quality_check 的最小实现；可基于 mock 数据运行；函数签名清晰，方便后续补充真实逻辑                                                           | 各阶段已实现骨架函数，具体清洗规则可根据业务需要扩展                           |
| **analysis**       | 提供最小 EDA 入口；能输出样本数据的基本统计；包含基础可视化与报告生成                                                                                                                    | 已实现快速分析模式与综合报告模式，支持 mock 数据                               |
| **api**            | 只做占位结构；明确未来用于 OAuth 与 video.data；不伪造真实密钥或真实返回                                                                                                                 | 提供完整的授权、用户信息、视频数据等演示流程，但实际调用需申请抖音开放平台权限 |
| **schemas**        | 根据 `docs/Data_description.md` 定义核心 schema/dataclass/pydantic 模型，包括：raw_web_video_data, web_video_meta, processed_video_data, author_dim, api_video_stats, api_user_profile | 模型字段与文档完全对齐，支持 Pydantic 验证与类型提示                           |
| **utils & config** | 提供配置加载、日志、文本处理、时间处理、I/O 等基础工具；配置文件采用 YAML 格式                                                                                                           | 工具函数已实现，可根据实际需求扩展                                             |

**注**：所有模块均支持 **mock 模式**，无需网络请求或真实数据即可运行完整流程，便于开发与测试。

## 项目结构

```
douyin_data_project/
├── docs/                           # 项目文档
│   └── Data_description.md        # 数据描述文档
├── src/                           # 源代码
│   ├── crawler/                   # 爬虫模块
│   │   ├── __init__.py
│   │   ├── client.py              # HTTP客户端
│   │   ├── parser.py              # HTML解析器
│   │   ├── scheduler.py           # 爬虫调度器
│   │   ├── anti_block.py          # 反反爬策略
│   │   └── extractors.py          # 数据提取器
│   ├── processing/                # 数据处理模块
│   │   ├── __init__.py
│   │   ├── clean.py               # 数据清洗
│   │   ├── transform.py           # 数据转换
│   │   ├── feature_engineering.py # 特征工程
│   │   └── quality_check.py       # 质量检查
│   ├── analysis/                  # 分析模块
│   │   ├── __init__.py
│   │   └── eda.py                 # 探索性数据分析
│   ├── api/                       # API模块（待实现）
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── client.py
│   │   └── video_data.py
│   ├── schemas/                   # 数据模式
│   │   ├── __init__.py
│   │   └── tables.py              # 数据表定义
│   └── utils/                     # 工具模块
│       ├── __init__.py
│       ├── config_loader.py       # 配置加载
│       ├── logger.py              # 日志工具
│       ├── text_utils.py          # 文本工具
│       ├── time_utils.py          # 时间工具
│       └── io_utils.py            # I/O工具
├── configs/                       # 配置文件
│   ├── settings.yaml             # 项目设置
│   ├── sources.yaml              # 数据源配置
│   ├── fields.yaml               # 字段定义
│   └── logging.yaml              # 日志配置
├── data/                          # 数据目录
│   ├── raw/                      # 原始数据
│   ├── interim/                  # 中间数据
│   ├── processed/                # 处理数据
│   ├── samples/                  # 样本数据
│   └── logs/                     # 日志目录
├── tests/                        # 测试目录
│   ├── __init__.py
│   └── test_smoke.py             # 冒烟测试
├── run_crawl.py                  # 爬虫运行脚本
├── run_clean.py                  # 清洗运行脚本
├── run_features.py               # 特征工程运行脚本
├── run_eda.py                    # EDA运行脚本
├── requirements.txt              # 依赖包
├── .env.example                  # 环境变量示例
└── README.md                     # 本文档
```

## 快速开始

### 1. 环境设置

```bash
# 克隆项目
git clone <repository-url>
cd douyin_data_project

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 复制环境变量文件
cp .env.example .env
# 编辑 .env 文件配置你的设置
```

### 2. 使用Mock模式测试（最小可运行骨架验证）

本项目所有模块均支持 **Mock 模式**，无需网络请求、无需真实数据即可运行完整流程。

```bash
# 测试爬虫（Mock模式）- 模拟抓取并解析HTML
python run_crawl.py --mock

# 测试数据清洗（Mock模式）- 使用mock数据演示清洗、转换、质量检查
python run_clean.py --mock

# 测试特征工程（Mock模式）- 基于mock数据计算基础特征、交互特征等
python run_features.py --mock

# 测试EDA分析（Mock模式）- 对mock数据进行探索性分析，生成统计报告
python run_eda.py --mock

# 测试API演示（Mock模式）- 演示OAuth流程、用户信息、视频数据等mock接口
python run_api_demo.py --mode=mock
```

**Mock 模式特点**：

- 无外部依赖：不访问抖音网站，不调用真实API
- 可重复运行：每次生成相同的模拟数据，便于调试
- 完整流程：覆盖从爬虫到分析的全链路，验证各模块衔接
- 快速反馈：秒级运行，立即查看结果与输出文件位置

### 3. 使用样本数据

```bash
# 使用提供的样本数据
python run_clean.py --input data/samples/sample_web_video_meta.csv
```

### 4. 运行测试

```bash
# 运行冒烟测试
python tests/test_smoke.py
```

## 核心功能

### 1. 爬虫模块（crawler）

- **URL输入**：支持命令行、配置文件、交互式输入一个或多个测试 URL
- **抓取能力**：抓取 HTML 页面，支持真实抓取与 mock 模式切换
- **接口预留**：预留 headers/cookies/delay/retry 配置接口，便于后续扩展
- **解析演示**：先用 mock/sample HTML 演示解析少量字段（视频ID、文案、点赞、评论等）
- **输出**：生成 raw 层样例文件（JSONL/Parquet），保留原始 HTML 路径与解析状态
- **调度与反爬**：内置任务调度器与基础反反爬策略（频率控制、随机延迟）

### 2. 处理模块（processing）

- **清洗（clean）**：提供文本清洗（URL、表情、特殊字符）、缺失值处理、异常值检测的最小实现
- **转换（transform）**：提供字段标准化（时间格式统一、计数归一化、标签提取）的骨架函数
- **特征工程（feature_engineering）**：提供基础特征（文本长度、时间特征）、交互特征（互动分数）、复合特征的示例计算
- **质量检查（quality_check）**：提供完整性、一致性、有效性的检查项，可输出质量报告
- **Mock支持**：所有处理阶段均可基于 mock 数据运行，函数签名清晰，便于后续补充业务逻辑

### 3. 分析模块（analysis）

- **EDA入口**：提供最小 EDA 入口，支持快速模式（基础统计）与综合模式（详细分析）
- **统计输出**：能输出样本数据的基本统计（均值、中位数、分布、相关性）
- **可视化**：内置基础图表（分布直方图、相关性热图、时间序列图）
- **报告生成**：支持 JSON/HTML 格式报告，包含关键发现与建议
- **Mock数据**：可使用 mock 数据运行完整分析流程，无需真实数据文件

### 4. API模块（api）

- **占位结构**：仅提供完整的模块结构与接口定义，不伪造真实密钥或真实返回
- **OAuth演示**：实现 OAuth 授权流程演示（生成授权URL、换取token、刷新token）
- **接口演示**：提供用户信息获取、视频数据查询、视频列表演示的 mock 实现
- **明确未来用途**：结构明确未来用于抖音开放平台的 OAuth 与 video.data 接口
- **安全提示**：所有演示均为 mock 数据，真实调用需申请抖音开放平台权限并配置有效凭证

### 5. 数据模式（schemas）

- **完整覆盖**：根据 `docs/Data_description.md` 定义全部六个核心模型：
  - `RawWebVideoData`：原始网页数据层（调试用）
  - `WebVideoMeta`：网页视频元数据（主表）
  - `ProcessedVideoData`：处理后视频数据（分析用）
  - `AuthorDim`：作者维表
  - `ApiVideoStats`：API视频统计表
  - `ApiUserProfile`：API用户公开信息表
- **Pydantic支持**：使用 Pydantic 实现数据验证与序列化，未安装时自动降级
- **字段对齐**：模型字段与文档完全对齐，包含字段说明、校验规则、默认值
- **类型提示**：提供完整的类型提示，便于 IDE 智能补全与静态检查

## 配置说明

### 主要配置文件

1. **settings.yaml** - 项目设置（路径、超时、版本等）
2. **sources.yaml** - 数据源配置（入口类型、样本URL等）
3. **fields.yaml** - 字段定义（数据类型、提取规则等）
4. **logging.yaml** - 日志配置（格式、级别、输出等）

### 环境变量

通过 `.env` 文件配置敏感信息和环境特定设置：

- API密钥和令牌
- 日志级别
- 功能开关
- 路径覆盖

## 数据流程

```
网页采集 → 原始数据 → 清洗转换 → 特征工程 → 分析报告
    ↓          ↓           ↓           ↓         ↓
raw_web   web_video   processed   featured   eda_report
```

### 数据表设计

1. **raw_web_video_data** - 原始网页数据（调试用）
2. **web_video_meta** - 网页视频元数据（主表）
3. **processed_video_data** - 处理后视频数据（分析用）
4. **author_dim** - 作者维表（待扩展）
5. **api_video_stats** - API视频统计（待实现）
6. **api_user_profile** - API用户资料（待实现）

*注：以上六个表的 Pydantic 模型已定义在 `src/schemas/tables.py` 中，字段与 `docs/Data_description.md` 完全对齐。*

## 扩展开发

### 添加新的数据源

1. 在 `configs/sources.yaml` 中添加新入口类型
2. 在 `src/crawler/extractors.py` 中添加解析逻辑
3. 更新 `src/schemas/tables.py` 中的字段定义

### 添加新的特征

1. 在 `src/processing/feature_engineering.py` 中添加特征计算方法
2. 更新 `configs/fields.yaml` 中的字段定义
3. 在 `run_features.py` 中添加对应的特征类型参数

### 集成API

1. 在 `.env` 中配置API密钥
2. 实现 `src/api/` 中的客户端和认证逻辑
3. 更新 `configs/settings.yaml` 中的API配置

## 注意事项

### 数据隐私

- 本项目仅处理公开可访问的数据
- 不采集用户隐私信息
- 遵守抖音平台的服务条款

### 使用限制

- 请合理控制请求频率，避免对目标服务器造成压力
- Mock模式适用于开发和测试
- 生产使用需要适当的错误处理和监控

### 兼容性

- 使用 pathlib 保证 Windows/Linux/macOS 路径兼容性
- 主要依赖包支持 Python 3.9+
- 配置使用 YAML 格式，便于阅读和修改

## 待实现功能

- [ ] API端数据采集（抖音开放平台）
- [ ] 实时数据更新管道
- [ ] 机器学习特征扩展
- [ ] 数据监控和告警
- [ ] 分布式爬虫支持
