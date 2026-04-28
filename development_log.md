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
