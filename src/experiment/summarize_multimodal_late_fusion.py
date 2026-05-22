"""
Summarize Multimodal late_fusion multi-seed validation results.

Usage:
    python src/experiment/summarize_multimodal_late_fusion.py \
        --run-ids <run_id1>,<run_id2>,<run_id3> \
        --output-dir <output_dir> \
        --validation-run-id <validation_run_id>

Example:
    python src/experiment/summarize_multimodal_late_fusion.py \
        --run-ids 202605221529,202605221538,202605221539 \
        --output-dir outputs/validation/multimodal_late_fusion_multiseed/real_raw_5000/202605221540 \
        --validation-run-id 202605221540
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_stats(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = variance ** 0.5
    sorted_vals = sorted(values)
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]
    if n % 2 == 1:
        median = sorted_vals[n // 2]
    else:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    return mean, std, min_val, max_val, median


def main():
    parser = argparse.ArgumentParser(description='Summarize late_fusion multi-seed validation')
    parser.add_argument('--run-ids', type=str, required=True,
                        help='Comma-separated list of 3 run_ids')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for validation files')
    parser.add_argument('--validation-run-id', type=str, default=None,
                        help='Validation run ID (default: auto-generated)')
    parser.add_argument('--config-type', type=str, default='all_modalities_cat_late_fusion',
                        help='Config type name')
    args = parser.parse_args()

    run_ids = [r.strip() for r in args.run_ids.split(',')]
    assert len(run_ids) == 3, f"Expected 3 run_ids, got {len(run_ids)}: {run_ids}"

    validation_run_id = args.validation_run_id or datetime.now().strftime('%Y%m%d%H%M%S')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path('outputs/multimodal/real_raw_5000')

    # Load all run data
    runs = []
    for rid in run_ids:
        run_dir = base_dir / rid
        if not run_dir.exists():
            print(f"ERROR: run_dir {run_dir} not found", file=sys.stderr)
            sys.exit(1)

        meta = load_json(run_dir / 'run_meta.json')
        metrics = load_json(run_dir / 'metrics.json')
        runs.append({
            'run_id': rid,
            'meta': meta,
            'metrics': metrics,
        })

    # Extract data for each run
    seeds_info = []
    for r in runs:
        meta = r['meta']
        metrics = r['metrics']

        # Determine seed from ablation_name or meta
        seed = meta.get('random_seed', None)
        if seed is None:
            name = meta.get('ablation_name', '')
            # Try to extract seed from ablation name
            for part in name.split('_'):
                if part.startswith('seed'):
                    try:
                        seed = int(part.replace('seed', ''))
                    except ValueError:
                        pass

        val = metrics.get('val_metrics', {})
        test = metrics.get('test_metrics', {})

        wts = meta.get('late_fusion_weights_softmax', [None, None, None])
        seeds_info.append({
            'seed': seed,
            'run_id': r['run_id'],
            'ablation_name': meta.get('ablation_name', ''),
            'categorical_enabled': meta.get('categorical_enabled', False),
            'enabled_modalities': ', '.join(meta.get('enabled_modalities', [])),
            'status': 'completed',
            'val_auc': val.get('auc'),
            'test_auc': test.get('auc'),
            'test_f1': test.get('f1'),
            'best_epoch': meta.get('best_epoch'),
            'best_val_loss': meta.get('best_val_loss'),
            'num_params': meta.get('num_params'),
            'fusion_type': meta.get('fusion_type'),
            'late_fusion_weight_text': wts[0] if len(wts) > 0 else None,
            'late_fusion_weight_media': wts[1] if len(wts) > 1 else None,
            'late_fusion_weight_structured': wts[2] if len(wts) > 2 else None,
            'text_profile': meta.get('text_profile', {}).get('name', ''),
            'categorical_features': meta.get('categorical_features', []),
            'modality_order': meta.get('late_fusion_modality_order', []),
        })

    # Sort by seed
    seeds_info.sort(key=lambda x: (x['seed'] or 0))

    config_type = args.config_type

    # ========================================================================
    # 1. validation_config.json
    # ========================================================================
    first_meta = runs[0]['meta']
    first_info = seeds_info[0]
    vocab_sizes = first_meta.get('categorical_vocab_sizes', [])
    embed_dims = first_meta.get('categorical_embedding_dims', [])
    modality_order = first_meta.get('late_fusion_modality_order', [])

    validation_config = {
        'validation_run_id': validation_run_id,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'batch': 'Batch 14L-experiment',
        'dataset': 'real_raw_5000',
        'experiment_name': 'Multimodal Late Fusion 3 seed 验证',
        'experiment_design_ref': 'docs/multimodal_fusion_strategy_design.md',
        'prior_impl_ref': 'Batch 14L-impl',
        'categorical_features': first_meta.get('categorical_features', []),
        'categorical_vocab_sizes': vocab_sizes,
        'categorical_embedding_dims': embed_dims,
        'categorical_embedding_total_dim': first_meta.get('categorical_embedding_total_dim', 0),
        'fusion_type': 'late_fusion',
        'late_fusion_mode': 'weighted_sum',
        'late_fusion_modality_order': modality_order,
        'text_profile': first_info['text_profile'],
        'seeds': [info['seed'] for info in seeds_info],
        'config_types': {
            config_type: {
                'categorical_enabled': True,
                'enabled_modalities': ['structured', 'text', 'media'],
                'fusion_type': 'late_fusion',
            },
        },
        'concat_mlp_baseline': {
            'test_auc_mean': 0.8131,
            'test_auc_std': 0.0056,
            'source': 'outputs/validation/multimodal_categorical_multiseed/real_raw_5000/202605211854/',
        },
        'dnn_baseline': {
            'test_auc': 0.8414,
            'source': 'outputs/dnn/real_raw_5000/202605132017/metrics.json',
        },
        'notes': [
            '对 all_modalities_cat + late_fusion (weighted_sum) 做 3 seed (2025/2026/2027) 验证。',
            '所有 runs 使用相同的 categorical 配置 (region+hashtag_name_top, embed_dim=4+16=20)。',
            '所有 runs 使用相同的 text_profile (merged_text_v2_dim64)。',
            '不修改代码/config/data。不运行 gated_fusion/dropout/residual。',
            '离线代理标签结果不代表真实线上收益。',
        ],
    }

    with open(output_dir / 'validation_config.json', 'w', encoding='utf-8') as f:
        json.dump(validation_config, f, indent=2, ensure_ascii=False)
    print(f"[OK] validation_config.json -> {output_dir / 'validation_config.json'}")

    # ========================================================================
    # 2. runs_manifest.csv
    # ========================================================================
    manifest_fields = [
        'config_type', 'seed', 'run_id', 'ablation_name',
        'categorical_enabled', 'enabled_modalities', 'status',
        'val_auc', 'test_auc', 'test_f1', 'best_epoch', 'best_val_loss',
        'num_params', 'fusion_type',
        'late_fusion_weight_text', 'late_fusion_weight_media', 'late_fusion_weight_structured',
        'text_profile', 'categorical_features',
    ]

    with open(output_dir / 'runs_manifest.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(manifest_fields)
        for info in seeds_info:
            writer.writerow([
                config_type,
                info['seed'],
                info['run_id'],
                info['ablation_name'],
                info['categorical_enabled'],
                info['enabled_modalities'],
                info['status'],
                info['val_auc'],
                info['test_auc'],
                info['test_f1'],
                info['best_epoch'],
                info['best_val_loss'],
                info['num_params'],
                info['fusion_type'],
                info['late_fusion_weight_text'],
                info['late_fusion_weight_media'],
                info['late_fusion_weight_structured'],
                info['text_profile'],
                '; '.join(info['categorical_features']),
            ])
    print(f"[OK] runs_manifest.csv -> {output_dir / 'runs_manifest.csv'}")

    # ========================================================================
    # 3. multiseed_summary.csv
    # ========================================================================
    metrics_list = ['val_auc', 'test_auc', 'test_f1', 'best_epoch', 'num_params',
                    'late_fusion_weight_text', 'late_fusion_weight_media', 'late_fusion_weight_structured']

    with open(output_dir / 'multiseed_summary.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['config_type', 'metric', 'mean', 'std', 'min', 'max', 'median', 'count'])
        for metric in metrics_list:
            values = [info[metric] for info in seeds_info if info[metric] is not None]
            if len(values) == 0:
                continue
            mean, std, min_val, max_val, median = compute_stats(values)
            writer.writerow([config_type, metric,
                             f'{mean:.6f}', f'{std:.6f}',
                             f'{min_val:.6f}', f'{max_val:.6f}',
                             f'{median:.6f}', len(values)])
    print(f"[OK] multiseed_summary.csv -> {output_dir / 'multiseed_summary.csv'}")

    # ========================================================================
    # 4. late_fusion_multiseed_report.json
    # ========================================================================
    # Compute stats
    test_auc_vals = [info['test_auc'] for info in seeds_info if info['test_auc'] is not None]
    test_auc_mean, test_auc_std, test_auc_min, test_auc_max, test_auc_median = compute_stats(test_auc_vals)

    val_auc_vals = [info['val_auc'] for info in seeds_info if info['val_auc'] is not None]
    val_auc_mean, val_auc_std, _, _, _ = compute_stats(val_auc_vals)

    # Late fusion weights analysis
    w_text = [info['late_fusion_weight_text'] for info in seeds_info if info['late_fusion_weight_text'] is not None]
    w_media = [info['late_fusion_weight_media'] for info in seeds_info if info['late_fusion_weight_media'] is not None]
    w_struct = [info['late_fusion_weight_structured'] for info in seeds_info if info['late_fusion_weight_structured'] is not None]

    def w_stats(vals):
        m, s, lo, hi, med = compute_stats(vals)
        return {'mean': round(m, 4), 'std': round(s, 4), 'min': round(lo, 4), 'max': round(hi, 4), 'median': round(med, 4), 'values': [round(v, 4) for v in vals]}

    weight_analysis = {
        'text': w_stats(w_text),
        'media': w_stats(w_media),
        'structured': w_stats(w_struct),
        'modality_order': modality_order,
    }

    # Comparisons
    cat_mean = 0.8131
    cat_mean_plus_5 = 0.8181
    dnn_auc = 0.8414

    delta_vs_cat_mean = round(test_auc_mean - cat_mean, 4)
    delta_vs_cat_mean_plus_5 = round(test_auc_mean - cat_mean_plus_5, 4)
    delta_vs_dnn = round(test_auc_mean - dnn_auc, 4)

    # Diagnoses
    if test_auc_mean >= cat_mean_plus_5:
        stable_diagnosis = f"Late fusion 3 seed mean ({test_auc_mean:.4f}) >= cat mean + 0.005 ({cat_mean_plus_5}), 判定稳定优于 concat_mlp"
        stable_judgment = 'pass'
    elif test_auc_mean > cat_mean:
        stable_diagnosis = f"Late fusion 3 seed mean ({test_auc_mean:.4f}) > cat mean ({cat_mean}), 但未达到 +0.005 标准 ({cat_mean_plus_5}), 判定小幅提升但不充分"
        stable_judgment = 'borderline'
    else:
        stable_diagnosis = f"Late fusion 3 seed mean ({test_auc_mean:.4f}) <= cat mean ({cat_mean}), 判定未稳定提升"
        stable_judgment = 'fail'

    # Weight collapse check
    max_weight_spread = max(
        abs(wt - 1/3) for wts in [w_text, w_media, w_struct]
        for wt in wts
    )
    avg_text_w = sum(w_text) / len(w_text)
    avg_media_w = sum(w_media) / len(w_media)
    avg_struct_w = sum(w_struct) / len(w_struct)

    if max_weight_spread > 0.3:
        weight_diagnosis = f"部分模态权重偏离等权超过 0.3 (最大偏差={max_weight_spread:.4f})，但未确认极端单模态塌缩"
        weight_collapse = False
    elif max_weight_spread > 0.45:
        weight_diagnosis = f"系统存在极端权重 (最大偏差={max_weight_spread:.4f})，接近单模态塌缩"
        weight_collapse = True
    else:
        weight_diagnosis = f"权重分布稳定，最大偏差={max_weight_spread:.4f}，无单模态塌缩"
        weight_collapse = False

    # Conclusions
    conclusions = []
    if stable_judgment == 'pass':
        conclusions.append(f"Late fusion 3 seed mean Test AUC = {test_auc_mean:.4f} >= {cat_mean_plus_5}，判定 late_fusion 稳定优于 concat_mlp。")
        conclusions.append("建议考虑将 late_fusion 作为 Multimodal 内部默认 fusion_type。")
    elif stable_judgment == 'borderline':
        conclusions.append(f"Late fusion 3 seed mean Test AUC = {test_auc_mean:.4f} > {cat_mean}，但未达到 +0.005 标准。")
        conclusions.append("Late fusion 有小幅提升但不充分，建议继续尝试 gated_fusion。")
    else:
        conclusions.append(f"Late fusion 3 seed mean Test AUC = {test_auc_mean:.4f} <= {cat_mean}，未稳定提升。")
        conclusions.append("建议回归 concat_mlp 并考虑其他改进方向。")

    if weight_collapse:
        conclusions.append("Late fusion weights 出现极端分布，融合策略失效风险高。")
    else:
        conclusions.append(f"Late fusion weights 分布相对稳定 (text={avg_text_w:.3f}, media={avg_media_w:.3f}, structured={avg_struct_w:.3f})，未出现单模态塌缩。")

    if test_auc_mean >= 0.83:
        conclusions.append(f"Late fusion mean ({test_auc_mean:.4f}) 已接近 DNN ({dnn_auc})，差距仅 {delta_vs_dnn:.4f}。")
    else:
        conclusions.append(f"Late fusion mean ({test_auc_mean:.4f}) 与 DNN ({dnn_auc}) 仍有差距 (Δ={delta_vs_dnn:.4f})，不替换 DNN baseline。")

    if stable_judgment in ('pass', 'borderline') and not weight_collapse:
        conclusions.append("建议继续尝试 gated_fusion 作为下一个改进方向。")
    else:
        conclusions.append("Late fusion 不显著优于 concat_mlp，gated_fusion 实施需谨慎评估优先级。")

    next_steps = []
    if stable_judgment == 'pass':
        next_steps.append("考虑将 late_fusion 作为 Multimodal 内部新的默认 fusion_type。")
        next_steps.append("继续尝试 gated_fusion 看是否能进一步提升。")
    elif stable_judgment == 'borderline':
        next_steps.append("尝试 gated_fusion (Phase 3)，评估加权门控是否能进一步提升融合质量。")
    else:
        next_steps.append("回归 concat_mlp，考虑其他改进方向（如模型容量、新特征）。")

    next_steps.append("DNN 仍保留为推荐 baseline，直至 Multimodal 在 3 seed 上超过 0.8414。")
    next_steps.append("如后续验证充分，可写 Multimodal 融合策略阶段总结文档。")

    report_json = {
        'validation_run_id': validation_run_id,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'batch': 'Batch 14L-experiment',
        'dataset': 'real_raw_5000',
        'fusion_type': 'late_fusion',
        'late_fusion_mode': 'weighted_sum',
        'categorical_features': first_meta.get('categorical_features', []),
        'categorical_embedding_total_dim': first_meta.get('categorical_embedding_total_dim', 0),
        'seeds': [info['seed'] for info in seeds_info],
        'summary': [
            {'config_type': config_type, 'metric': 'test_auc',
             'mean': round(test_auc_mean, 6), 'std': round(test_auc_std, 6),
             'min': round(test_auc_min, 6), 'max': round(test_auc_max, 6),
             'values': [round(v, 6) for v in test_auc_vals]},
            {'config_type': config_type, 'metric': 'val_auc',
             'mean': round(val_auc_mean, 6), 'std': round(val_auc_std, 6),
             'min': round(min(val_auc_vals), 6), 'max': round(max(val_auc_vals), 6),
             'values': [round(v, 6) for v in val_auc_vals]},
        ],
        'late_fusion_weights': weight_analysis,
        'comparisons': {
            'late_fusion_vs_concat_mlp': {
                'late_fusion_mean_test_auc': round(test_auc_mean, 4),
                'late_fusion_std': round(test_auc_std, 4),
                'concat_mlp_mean': cat_mean,
                'concat_mlp_std': 0.0056,
                'delta_mean': delta_vs_cat_mean,
                'concat_mlp_source': 'outputs/validation/multimodal_categorical_multiseed/real_raw_5000/202605211854/',
                'diagnosis': stable_diagnosis,
            },
            'late_fusion_vs_dnn': {
                'dnn_test_auc': dnn_auc,
                'late_fusion_mean_test_auc': round(test_auc_mean, 4),
                'late_fusion_max_test_auc': round(test_auc_max, 4),
                'gap_vs_dnn_mean': delta_vs_dnn,
                'gap_vs_dnn_best': round(test_auc_max - dnn_auc, 4),
                'dnn_source': 'outputs/dnn/real_raw_5000/202605132017/metrics.json',
            },
        },
        'conclusions': conclusions,
        'next_steps': next_steps,
        'concat_mlp_baseline': {
            'test_auc_mean': cat_mean,
            'test_auc_std': 0.0056,
            'source': 'outputs/validation/multimodal_categorical_multiseed/real_raw_5000/202605211854/',
        },
        'dnn_baseline': {
            'test_auc': dnn_auc,
            'source': 'outputs/dnn/real_raw_5000/202605132017/metrics.json',
        },
        'judgment': {
            'stable_superior': stable_judgment == 'pass',
            'weight_collapse': weight_collapse,
            'recommend_gated_fusion': stable_judgment in ('pass', 'borderline') and not weight_collapse,
            'recommend_late_fusion_as_default': stable_judgment == 'pass',
            'keep_dnn_as_baseline': True,
        },
    }

    with open(output_dir / 'late_fusion_multiseed_report.json', 'w', encoding='utf-8') as f:
        json.dump(report_json, f, indent=2, ensure_ascii=False)
    print(f"[OK] late_fusion_multiseed_report.json -> {output_dir / 'late_fusion_multiseed_report.json'}")

    # ========================================================================
    # 5. late_fusion_multiseed_report.md
    # ========================================================================
    # Build per-seed table
    seed_rows = []
    for info in seeds_info:
        wts_str = ', '.join([
            f"text={info['late_fusion_weight_text']:.4f}",
            f"media={info['late_fusion_weight_media']:.4f}",
            f"structured={info['late_fusion_weight_structured']:.4f}",
        ])
        seed_rows.append(
            f"| {info['seed']} | {info['run_id']} | {info['val_auc']:.4f} | "
            f"**{info['test_auc']:.4f}** | {info['test_f1']:.4f} | "
            f"{info['best_epoch']} | {info['num_params']} | {wts_str} |"
        )
    seed_table = '\n'.join(seed_rows)

    # Weight stability table
    weight_rows = []
    for i, info in enumerate(seeds_info):
        weight_rows.append(
            f"| seed={info['seed']} | "
            f"{info['late_fusion_weight_text']:.4f} | "
            f"{info['late_fusion_weight_media']:.4f} | "
            f"{info['late_fusion_weight_structured']:.4f} |"
        )
    weight_table = '\n'.join(weight_rows)

    # Key comparisons
    if delta_vs_cat_mean >= 0:
        cat_comparison = f"Late fusion mean **{test_auc_mean:.4f}** > concat_mlp mean **{cat_mean}** (Δ=**+{delta_vs_cat_mean:.4f}**)"
    else:
        cat_comparison = f"Late fusion mean **{test_auc_mean:.4f}** < concat_mlp mean **{cat_mean}** (Δ=**{delta_vs_cat_mean:.4f}**)"

    if test_auc_mean >= dnn_auc:
        dnn_comparison = f"Late fusion mean **{test_auc_mean:.4f}** >= DNN **{dnn_auc}** 🎉"
    else:
        dnn_comparison = f"Late fusion mean **{test_auc_mean:.4f}** < DNN **{dnn_auc}** (Δ=**{delta_vs_dnn:.4f}**)"

    # Conclusion bullets
    conclusion_bullets = '\n'.join(f'{i+1}. {c}' for i, c in enumerate(conclusions))
    next_step_bullets = '\n'.join(f'{i+1}. {s}' for i, s in enumerate(next_steps))

    report_md = f"""# Multimodal Late Fusion 多 seed 验证报告

> 项目根目录：`D:/CodeData/Program Coding/ByteDance/RA_PTA/`
> 数据集：`real_raw_5000`
> 批次：Batch 14L-experiment
> 验证 Run ID：{validation_run_id}
> 执行日期：{datetime.now().strftime('%Y-%m-%d')}
> Fusion 类型：`late_fusion`（可学习加权 sum）
> 实验设计参考：`docs/multimodal_fusion_strategy_design.md`
> 前置实现参考：Batch 14L-impl

---

## 一、实验目的

1. 对 all_modalities_cat + late_fusion（weighted_sum）配置做 3 seed（2025/2026/2027）验证。
2. 与 concat_mlp all_modalities_cat 3 seed mean（0.8131）对比。
3. 判断 late_fusion 是否稳定优于 concat_mlp。
4. 分析 late_fusion 可学习权重的稳定性和分布。
5. 不修改代码/config/data。不运行 gated_fusion/dropout/residual。

---

## 二、实验配置

| 参数 | 值 |
|------|-----|
| dataset | real_raw_5000 |
| enabled_modalities | structured, text, media |
| categorical_enabled | true |
| categorical_features | region (vocab=13, dim=4) + hashtag_name_top (vocab=1185, dim=16) |
| fusion_type | late_fusion |
| late_fusion_mode | weighted_sum |
| late_fusion_modality_order | text, media, structured |
| text_profile | merged_text_v2_dim64 |
| seeds | 2025, 2026, 2027 |

---

## 三、逐 seed 结果

### 3.1 指标汇总

| seed | run_id | val_auc | test_auc | test_f1 | best_epoch | num_params | late_fusion_weights |
|------|--------|---------|----------|---------|------------|------------|---------------------|
{seed_table}

---

## 四、统计汇总

### 4.1 Test AUC

| 配置 | mean | std | min | max | median |
|------|------|-----|-----|-----|--------|
| all_modalities_cat_late_fusion | **{test_auc_mean:.4f}** | {test_auc_std:.4f} | {test_auc_min:.4f} | {test_auc_max:.4f} | {test_auc_median:.4f} |

### 4.2 Val AUC

| 配置 | mean | std | min | max |
|------|------|-----|-----|-----|
| all_modalities_cat_late_fusion | {val_auc_mean:.4f} | {val_auc_std:.4f} | {min(val_auc_vals):.4f} | {max(val_auc_vals):.4f} |

### 4.3 与 concat_mlp baseline 对比

| 指标 | Late fusion mean | Concat+MLP mean | Δ |
|------|-----------------|-----------------|----|
| Test AUC | {test_auc_mean:.4f} | {cat_mean} | **{delta_vs_cat_mean:+.4f}** |

### 4.4 与 DNN baseline 对比

| 指标 | Late fusion mean | Late fusion max | DNN | 差距(mean) | 差距(max) |
|------|-----------------|-----------------|-----|------------|-----------|
| Test AUC | {test_auc_mean:.4f} | {test_auc_max:.4f} | {dnn_auc} | **{delta_vs_dnn:.4f}** | **{round(test_auc_max - dnn_auc, 4):.4f}** |

---

## 五、Late Fusion 权重分析

| seed | w_text | w_media | w_structured |
|------|--------|---------|-------------|
{weight_table}

**权重统计：**

| 模态 | mean | std | min | max |
|------|------|-----|-----|-----|
| text | {avg_text_w:.4f} | {w_stats(w_text)['std']:.4f} | {min(w_text):.4f} | {max(w_text):.4f} |
| media | {avg_media_w:.4f} | {w_stats(w_media)['std']:.4f} | {min(w_media):.4f} | {max(w_media):.4f} |
| structured | {avg_struct_w:.4f} | {w_stats(w_struct)['std']:.4f} | {min(w_struct):.4f} | {max(w_struct):.4f} |

---

## 六、核心对比

### 6.1 Late fusion vs concat_mlp

{cat_comparison}

判断标准：
- >= 0.8181 (cat mean + 0.005)：**稳定优于**
- > 0.8131 但 < 0.8181：**小幅提升但不充分**
- <= 0.8131：**未稳定提升**

**判定：{stable_diagnosis}**

### 6.2 Late fusion vs DNN

{dnn_comparison}

### 6.3 Weight 稳定性

{weight_diagnosis}

---

## 七、结论

{conclusion_bullets}

---

## 八、下一步建议

{next_step_bullets}

---

## 九、离线实验声明

1. 所有结果基于**离线代理标签**（interaction_score P60 分位数二分类），不代表真实线上推荐收益。
2. 当前为本地模拟实验，不包含真实用户行为、曝光、点击、完播、转化或留存数据。
3. 不要将本文档中的数值直接引用为线上推荐系统效果。
4. DNN 基线（0.8414）仍为推荐 baseline，直至 Multimodal 以多 seed 验证稳定超过该数值。
"""

    with open(output_dir / 'late_fusion_multiseed_report.md', 'w', encoding='utf-8') as f:
        f.write(report_md)
    print(f"[OK] late_fusion_multiseed_report.md -> {output_dir / 'late_fusion_multiseed_report.md'}")

    print(f"\n=== Summary ===")
    print(f"Config Type: {config_type}")
    print(f"Test AUC: mean={test_auc_mean:.4f}, std={test_auc_std:.4f}, min={test_auc_min:.4f}, max={test_auc_max:.4f}")
    print(f"Weights (mean): text={avg_text_w:.4f}, media={avg_media_w:.4f}, structured={avg_struct_w:.4f}")
    print(f"Judgment: {stable_judgment}")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
