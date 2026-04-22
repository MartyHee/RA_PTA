#!/usr/bin/env python
"""
逻辑回归baseline脚本 for 高互动视频预测（二分类）
使用指定的12个特征字段，训练逻辑回归模型，评估并输出结果。
"""

import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# 默认特征列（前12列）
DEFAULT_FEATURE_COLS = [
    'author_follower_count',
    'author_total_favorited',
    'hashtag_count',
    'duration_sec',
    'publish_hour',
    'publish_weekday',
    'is_weekend',
    'days_since_publish',
    'author_verification_type',
    'desc_text_length',
    'has_desc_text',
    'has_hashtag'
]

TARGET_COL = 'label'

def parse_args():
    parser = argparse.ArgumentParser(description='逻辑回归baseline训练与评估')
    parser.add_argument('--train', type=str, required=True,
                        help='训练集CSV文件路径')
    parser.add_argument('--test', type=str, required=True,
                        help='测试集CSV文件路径')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='输出目录路径')
    parser.add_argument('--feature-cols', type=str, nargs='+',
                        default=DEFAULT_FEATURE_COLS,
                        help='特征列列表，默认为12个默认字段')
    parser.add_argument('--scale', action='store_true', default=True,
                        help='启用标准化（默认）')
    parser.add_argument('--no-scale', action='store_false', dest='scale',
                        help='禁用标准化')
    parser.add_argument('--class-weight', type=str, default='balanced',
                        help="class_weight参数，可选'balanced'或None，默认'balanced'")
    parser.add_argument('--k', type=int, default=None,
                        help='Recall@K/Precision@K中的K值，默认根据测试集大小自动计算')
    return parser.parse_args()

def load_data(train_path, test_path, feature_cols, target_col):
    """加载训练集和测试集，返回特征和标签"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 检查必要的列是否存在
    missing_train = [col for col in feature_cols if col not in train_df.columns]
    missing_test = [col for col in feature_cols if col not in test_df.columns]
    if missing_train:
        raise ValueError(f"训练集中缺少特征列: {missing_train}")
    if missing_test:
        raise ValueError(f"测试集中缺少特征列: {missing_test}")
    if target_col not in train_df.columns:
        raise ValueError(f"训练集中缺少目标列: {target_col}")
    if target_col not in test_df.columns:
        raise ValueError(f"测试集中缺少目标列: {target_col}")

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    return X_train, y_train, X_test, y_test

def create_preprocessor(feature_cols, scale=True):
    """创建预处理管道"""
    # 识别列类型
    numeric_cols = []
    categorical_cols = []
    bool_cols = []

    # 根据列名和数据类型简单判断
    # 这里假设数值列是那些名称中包含'count'、'total'、'length'、'sec'、'hour'、'day'的列
    # 实际上更严谨的做法是根据实际数据类型判断，但为简化我们使用预定义列表
    numeric_candidates = ['author_follower_count', 'author_total_favorited',
                         'hashtag_count', 'duration_sec', 'publish_hour',
                         'publish_weekday', 'days_since_publish', 'desc_text_length']
    categorical_candidates = ['author_verification_type']
    bool_candidates = ['is_weekend', 'has_desc_text', 'has_hashtag']

    for col in feature_cols:
        if col in numeric_candidates:
            numeric_cols.append(col)
        elif col in categorical_candidates:
            categorical_cols.append(col)
        elif col in bool_candidates:
            bool_cols.append(col)
        else:
            # 默认视为数值列
            numeric_cols.append(col)

    # 数值列处理：缺失值填充为中位数，可选标准化
    numeric_transformer_steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale:
        numeric_transformer_steps.append(('scaler', StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_transformer_steps)

    # 类别列将在主流程中单独处理，不包含在ColumnTransformer中

    # 布尔列处理：缺失值填充为最频繁值（0或1）
    bool_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    # 由于LabelEncoder不能直接用在ColumnTransformer中，我们需要自定义处理
    # 这里简化：将所有列视为数值或布尔，对author_verification_type特殊处理
    # 重新定义预处理：数值列和布尔列使用相同的imputer，类别列单独处理
    transformers = []

    if numeric_cols:
        transformers.append(('numeric', numeric_transformer, numeric_cols))

    # 布尔列使用简单imputer
    if bool_cols:
        bool_transformer_simple = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])
        transformers.append(('bool', bool_transformer_simple, bool_cols))

    # 类别列特殊处理：在fit中单独处理
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

    return preprocessor, categorical_cols

def compute_recall_precision_at_k(y_true, y_score, k):
    """计算全局排序下的Recall@K和Precision@K"""
    # 按预测得分降序排序
    sorted_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true.iloc[sorted_indices] if hasattr(y_true, 'iloc') else y_true[sorted_indices]

    # 取前K个
    top_k_indices = sorted_indices[:k]
    top_k_true = y_true_sorted[:k]

    # 计算指标
    total_positives = np.sum(y_true)
    true_positives_k = np.sum(top_k_true)

    recall_at_k = true_positives_k / total_positives if total_positives > 0 else 0.0
    precision_at_k = true_positives_k / k if k > 0 else 0.0

    return recall_at_k, precision_at_k

def main():
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    print(f"加载训练数据: {args.train}")
    print(f"加载测试数据: {args.test}")
    X_train, y_train, X_test, y_test = load_data(
        args.train, args.test, args.feature_cols, TARGET_COL
    )

    print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
    print(f"训练集正样本比例: {y_train.mean():.4f}, 测试集正样本比例: {y_test.mean():.4f}")

    # 检查缺失值
    print("\n训练集缺失值统计:")
    print(X_train.isnull().sum())
    print("\n测试集缺失值统计:")
    print(X_test.isnull().sum())

    # 创建预处理管道
    preprocessor, categorical_cols = create_preprocessor(args.feature_cols, scale=args.scale)

    # 处理类别列：在预处理前单独编码，消除信息泄漏
    # 1. 缺失值填充（最频繁值）
    # 2. LabelEncoder编码（只在训练集上拟合）
    categorical_imputers = {}
    label_encoders = {}
    for col in categorical_cols:
        # 创建并拟合缺失值填充器（最频繁值）
        imp = SimpleImputer(strategy='most_frequent')
        # 训练集：拟合并转换
        X_train_col_filled = imp.fit_transform(X_train[[col]]).ravel()
        categorical_imputers[col] = imp
        # 测试集：使用训练集的填充器转换
        X_test_col_filled = imp.transform(X_test[[col]]).ravel()

        # 创建并拟合LabelEncoder（只在训练集上拟合）
        le = LabelEncoder()
        # 将填充后的值转换为字符串（LabelEncoder需要字符串输入）
        X_train_col_str = X_train_col_filled.astype(str)
        le.fit(X_train_col_str)
        label_encoders[col] = le

        # 转换训练集
        X_train[col] = le.transform(X_train_col_str)

        # 转换测试集：处理未见过的类别
        X_test_col_str = X_test_col_filled.astype(str)
        # 对于测试集中未在训练集中出现过的类别，映射到-1
        unseen_mask = ~np.isin(X_test_col_str, le.classes_)
        if unseen_mask.any():
            print(f"警告: 测试集列 '{col}' 中有 {unseen_mask.sum()} 个未见过的类别，将被映射为-1")
            # 先初始化为-1
            X_test_encoded = np.full_like(X_test_col_str, -1, dtype=int)
            # 对见过的类别进行编码
            seen_mask = ~unseen_mask
            if seen_mask.any():
                X_test_encoded[seen_mask] = le.transform(X_test_col_str[seen_mask])
            X_test[col] = X_test_encoded
        else:
            X_test[col] = le.transform(X_test_col_str)

    # 应用预处理
    print("\n拟合预处理管道...")
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)

    # 训练逻辑回归模型
    print("训练逻辑回归模型...")
    class_weight = args.class_weight if args.class_weight != 'None' else None
    model = LogisticRegression(
        class_weight=class_weight,
        random_state=42,
        max_iter=1000
    )
    model.fit(X_train_processed, y_train)

    # 预测
    y_pred = model.predict(X_test_processed)
    y_score = model.predict_proba(X_test_processed)[:, 1]

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_score)
    cm = confusion_matrix(y_test, y_pred)

    # 计算Recall@K和Precision@K
    if args.k is None:
        # 自动计算K
        k = max(10, int(len(y_test) * 0.2))
    else:
        k = args.k
    recall_at_k, precision_at_k = compute_recall_precision_at_k(y_test, y_score, k)

    # 输出指标
    print("\n" + "="*50)
    print("评估结果:")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Recall@{k}: {recall_at_k:.4f}")
    print(f"Precision@{k}: {precision_at_k:.4f}")
    print(f"K值: {k}")
    print("混淆矩阵:")
    print(cm)

    # 保存预测结果
    predictions_df = pd.DataFrame({
        'y_true': y_test.values,
        'y_pred': y_pred,
        'y_score': y_score
    })
    predictions_path = os.path.join(args.output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\n预测结果保存至: {predictions_path}")

    # 保存指标
    metrics = {
        'auc': float(auc),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        f'recall_at_{k}': float(recall_at_k),
        f'precision_at_{k}': float(precision_at_k),
        'k': k,
        'confusion_matrix': cm.tolist()
    }
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"指标保存至: {metrics_path}")

    # 保存特征列
    feature_cols_path = os.path.join(args.output_dir, 'feature_columns.json')
    with open(feature_cols_path, 'w', encoding='utf-8') as f:
        json.dump(args.feature_cols, f, indent=2, ensure_ascii=False)
    print(f"特征列保存至: {feature_cols_path}")

    # 保存运行配置
    run_config = {
        'train_path': args.train,
        'test_path': args.test,
        'model_type': 'LogisticRegression',
        'feature_columns': args.feature_cols,
        'preprocessing': {
            'numeric_imputation': 'median',
            'categorical_imputation': 'most_frequent',
            'scaling': args.scale,
            'categorical_encoding': 'LabelEncoder'
        },
        'class_weight': args.class_weight,
        'k': k,
        'run_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'python_version': sys.version
    }
    run_config_path = os.path.join(args.output_dir, 'run_config.json')
    with open(run_config_path, 'w', encoding='utf-8') as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)
    print(f"运行配置保存至: {run_config_path}")

    # 保存混淆矩阵为CSV
    cm_df = pd.DataFrame(cm, columns=['Predicted 0', 'Predicted 1'],
                         index=['Actual 0', 'Actual 1'])
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.csv')
    cm_df.to_csv(cm_path)
    print(f"混淆矩阵保存至: {cm_path}")

    # 生成总结文本
    summary = f"""逻辑回归baseline实验总结
运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

数据信息:
- 训练集: {args.train} (样本数: {X_train.shape[0]})
- 测试集: {args.test} (样本数: {X_test.shape[0]})
- 特征数: {X_train.shape[1]}
- 训练集正样本比例: {y_train.mean():.4f}
- 测试集正样本比例: {y_test.mean():.4f}

预处理:
- 数值特征: 中位数填充缺失值{' + 标准化' if args.scale else ''}
- 类别特征: 最频繁值填充 + LabelEncoder编码
- 布尔特征: 最频繁值填充

模型参数:
- 模型: LogisticRegression
- class_weight: {args.class_weight}
- 其他参数: 默认

评估结果:
- AUC: {auc:.4f}
- Accuracy: {accuracy:.4f}
- Precision: {precision:.4f}
- Recall: {recall:.4f}
- F1: {f1:.4f}
- Recall@{k}: {recall_at_k:.4f}
- Precision@{k}: {precision_at_k:.4f}

结论:
逻辑回归baseline在测试集上表现[请根据实际结果填写]。AUC为{auc:.4f}，准确率为{accuracy:.4f}。
"""
    summary_path = os.path.join(args.output_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"实验总结保存至: {summary_path}")

    print("\n" + "="*50)
    print(f"所有输出已保存至: {args.output_dir}")

    # 输出正负样本预测分布
    print("\n正负样本预测分布:")
    pred_dist = pd.Series(y_pred).value_counts().sort_index()
    for label, count in pred_dist.items():
        print(f"  预测为{label}: {count}个样本")

    # 输出部分预测样本
    print("\n部分预测样本 (前10个):")
    sample_df = pd.DataFrame({
        'y_true': y_test.values[:10],
        'y_pred': y_pred[:10],
        'y_score': y_score[:10]
    })
    print(sample_df.to_string(index=False))

if __name__ == '__main__':
    main()