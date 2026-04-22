# LR_baseline

## 目标

本目录用于实现“高互动视频预测”的第一版逻辑回归（LR）baseline。

注意：

- 这里不是个性化推荐实验
- 不是 Item-CF
- 不是 CTR 预估
- 当前任务是基于视频级特征做二分类预测

## 输入数据

训练集：

- `data/train/train_v1_fix5.csv`

测试集：

- `data/test/test_v1_fix5.csv`

这些数据由上游 `douyin_data_project` 生成后导入本目录。

## 当前任务定义

高互动视频预测（二分类）

标签基于互动分数构造，上游已完成 train/test 切分。

## 默认入模字段

- `author_follower_count`
- `author_total_favorited`
- `hashtag_count`
- `duration_sec`
- `publish_hour`
- `publish_weekday`
- `is_weekend`
- `days_since_publish`
- `author_verification_type`
- `desc_text_length`
- `has_desc_text`
- `has_hashtag`

## 默认不入模字段

### 标签泄漏相关字段

- `collect_count`
- `like_count_num`
- `comment_count_num`
- `share_count_num`

### 质量控制/追溯字段

- `source_entry`
- `match_type`
- `confidence`
- `video_id`
- `page_url`
- `author_id`
- `publish_time_raw`
- `crawl_time`
- `desc_text`
- `author_name`
- `hashtag_list`

## 目录结构建议

```text
LR_baseline/
├── CLAUDE.md
├── README.md
├── data/
│   ├── train/
│   └── test/
├── outputs/
├── run_lr_baseline.py
└── ...
```

## 实验记录

### 2026-04-23 03:40:17 (run_id: 202604230339)

**1. 本次任务目标**
- 实现高互动视频预测（二分类）的第一版逻辑回归baseline

**2. 训练集/测试集路径**
- 训练集: `data/train/train_v1_fix5.csv` (129个样本)
- 测试集: `data/test/test_v1_fix5.csv` (31个样本)

**3. 默认入模字段**
- 使用前12个字段作为特征: `author_follower_count`, `author_total_favorited`, `hashtag_count`, `duration_sec`, `publish_hour`, `publish_weekday`, `is_weekend`, `days_since_publish`, `author_verification_type`, `desc_text_length`, `has_desc_text`, `has_hashtag`

**4. 预处理方式**
- 数值特征: 中位数填充缺失值 + 标准化(StandardScaler)
- 类别特征(`author_verification_type`): 最频繁值填充 + LabelEncoder编码
- 布尔特征: 最频繁值填充
- 所有预处理器只在训练集拟合，测试集只做transform

**5. 模型参数**
- 模型: `LogisticRegression`
- `class_weight`: 'balanced'
- 其他参数: 默认

**6. 运行命令**
```bash
cd "D:\CodeData\Program Coding\ByteDance\RA_PTA\LR_baseline" && "D:\CodeData\software\Anaconda\Anaconda3\envs\ra\python.exe" run_lr_baseline.py --train "D:\CodeData\Program Coding\ByteDance\RA_PTA\LR_baseline\data\train\train_v1_fix5.csv" --test "D:\CodeData\Program Coding\ByteDance\RA_PTA\LR_baseline\data\test\test_v1_fix5.csv" --output-dir "D:\CodeData\Program Coding\ByteDance\RA_PTA\LR_baseline\outputs\v1_fix5\202604230339"
```

**7. 输出结果目录**
- `outputs/v1_fix5/202604230339/`

**8. 评估指标**
- AUC: 0.5727
- Accuracy: 0.5161
- Precision: 0.2500
- Recall: 0.1818
- F1: 0.2105
- Recall@10: 0.2727
- Precision@10: 0.3000
- K值: 10

**9. 当前结论**
- 逻辑回归baseline在测试集上表现略高于随机猜测(AUC=0.5727)，但整体预测能力有限。
- 准确率51.61%接近随机猜测水平，正样本召回率较低(18.18%)。
- 模型对负样本的预测偏好较强（预测为0的样本占74%）。
- 特征工程和模型选择有较大优化空间。

**10. 下一步建议**
- 进一步分析特征重要性，筛选有效特征
- 尝试其他分类模型（如随机森林、XGBoost）
- 考虑特征交叉、多项式特征等非线性变换
- 增加更多视频内容、作者历史表现等特征
- 调整类别不平衡处理策略

### 2026-04-23 03:52:50 (run_id: 20260423_repair)

**1. 本次任务目标**
- 对第一版逻辑回归baseline进行小修，修复实现细节和可复现性问题

**2. 修复点**
- 修复 `--scale` 参数逻辑：原使用 `action='store_true', default=True` 导致无法关闭标准化；现改为支持 `--scale` / `--no-scale` 选项，默认启用标准化
- 清理类别特征编码逻辑：移除 `create_preprocessor()` 中未使用的 `LabelEncoder` pipeline，只保留主流程中的一套编码逻辑，保证 train/test 使用同一套编码器
- 检查预处理与输出的一致性：确认实际入模字段只有12个，标准化只作用于应处理的列，`feature_columns.json`、`run_config.json`、`summary.txt` 三者内容一致

**3. 训练集/测试集路径**
- 训练集: `data/train/train_v1_fix5.csv` (129个样本)
- 测试集: `data/test/test_v1_fix5.csv` (31个样本)

**4. 默认入模字段**
- 使用前12个字段作为特征: `author_follower_count`, `author_total_favorited`, `hashtag_count`, `duration_sec`, `publish_hour`, `publish_weekday`, `is_weekend`, `days_since_publish`, `author_verification_type`, `desc_text_length`, `has_desc_text`, `has_hashtag`

**5. 预处理方式**
- 数值特征: 中位数填充缺失值 + 标准化(StandardScaler)
- 类别特征(`author_verification_type`): 最频繁值填充 + LabelEncoder编码
- 布尔特征: 最频繁值填充
- 所有预处理器只在训练集拟合，测试集只做transform

**6. 模型参数**
- 模型: `LogisticRegression`
- `class_weight`: 'balanced'
- 其他参数: 默认

**7. 运行命令**
```bash
cd "D:\CodeData\Program Coding\ByteDance\RA_PTA\LR_baseline" && "D:\CodeData\software\Anaconda\Anaconda3\envs\ra\python.exe" run_lr_baseline.py --train "D:\CodeData\Program Coding\ByteDance\RA_PTA\LR_baseline\data\train\train_v1_fix5.csv" --test "D:\CodeData\Program Coding\ByteDance\RA_PTA\LR_baseline\data\test\test_v1_fix5.csv" --output-dir "D:\CodeData\Program Coding\ByteDance\RA_PTA\LR_baseline\outputs\v1_fix5\20260423_repair"
```

**8. 输出结果目录**
- `outputs/v1_fix5/20260423_repair/`

**9. 评估指标**
- AUC: 0.5727
- Accuracy: 0.5161
- Precision: 0.2500
- Recall: 0.1818
- F1: 0.2105
- Recall@10: 0.2727
- Precision@10: 0.3000
- K值: 10

**10. 当前结论**
- 修复后的逻辑回归baseline在测试集上表现与修复前一致（AUC=0.5727），验证了修复未改变模型行为。
- 参数控制更加清晰，标准化可通过 `--no-scale` 关闭。
- 类别编码逻辑更简洁，避免了重复定义。
- 输出文件内容一致，提高了可复现性。

**11. 补充测试（禁用标准化）**
- 使用 `--no-scale` 运行一次，输出目录: `outputs/v1_fix5/20260423_no_scale/`
- 评估指标: AUC=0.5955, Accuracy=0.5484, Precision=0.4118, Recall=0.6364, F1=0.5000
- 结果表明标准化对模型性能有一定影响，禁用标准化后AUC略有提升。

### 2026-04-23 04:04:56 (run_id: 20260423_repair2)

**1. 本次修复目标**
- 修复类别编码信息泄漏问题：原代码合并train和test数据拟合LabelEncoder，导致测试集信息泄漏
- 修复类别列缺失值处理与文档描述不一致问题：文档说明"最频繁值填充 + LabelEncoder编码"，但原代码未对类别列进行缺失值填充
- 保持其他逻辑完全不变（默认入模字段、标签定义、train/test划分、标准化逻辑、class_weight设置、评估指标、输出文件结构）

**2. 修复了什么泄漏问题**
- 原逻辑：合并train和test的所有可能值拟合LabelEncoder，测试集值参与了编码器拟合，造成信息泄漏
- 新逻辑：只在训练集上拟合LabelEncoder，测试集只能使用训练集得到的映射做transform
- 兜底处理：如果测试集出现训练集中未见过的新类别，映射到-1，并输出警告

**3. 类别缺失值如何处理**
- 对 `author_verification_type` 列，在训练集上使用最频繁值填充缺失值，保存填充器
- 测试集使用训练阶段得到的填充值（最频繁值）处理缺失值
- 填充后再进行LabelEncoder编码
- 实现与 `run_config.json`、`summary.txt`、`README.md` 中的说明保持一致

**4. 新的运行命令**
```bash
cd "D:\CodeData\Program Coding\ByteDance\RA_PTA\LR_baseline" && "D:\CodeData\software\Anaconda\Anaconda3\envs\ra\python.exe" run_lr_baseline.py --train "D:\CodeData\Program Coding\ByteDance\RA_PTA\LR_baseline\data\train\train_v1_fix5.csv" --test "D:\CodeData\Program Coding\ByteDance\RA_PTA\LR_baseline\data\test\test_v1_fix5.csv" --output-dir "D:\CodeData\Program Coding\ByteDance\RA_PTA\LR_baseline\outputs\v1_fix5\20260423_repair2"
```

**5. 新输出目录**
- `outputs/v1_fix5/20260423_repair2/`

**6. 修复后指标**
- AUC: 0.5727
- Accuracy: 0.5161
- Precision: 0.2500
- Recall: 0.1818
- F1: 0.2105
- Recall@10: 0.2727
- Precision@10: 0.3000
- K值: 10

**7. 当前结论**
- 修复后的逻辑回归baseline在测试集上表现与修复前一致（AUC=0.5727），验证了修复未改变模型行为（因为数据中没有缺失值，也没有未见过的类别）。
- 类别编码信息泄漏问题已解决：LabelEncoder只在训练集上拟合，测试集只做transform。
- 类别缺失值填充已实现：与文档描述一致，使用最频繁值填充。
- 代码实现与配置文件、总结文档的描述保持一致，提高了可复现性。
