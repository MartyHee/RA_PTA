"""A/B 实验指标计算。

从 online_simulation JSONL 日志读取请求、分配、推荐和模拟事件记录，
按 experiment_id + group 聚合计算基础指标，输出报告。

用法：
    python src/serving/ab_metrics.py --log-dir outputs/online_simulation/<online_run_id>
    python src/serving/ab_metrics.py --log-dir outputs/online_simulation/<online_run_id> --experiment-id exp_dnn_baseline_vs_tuned
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 数据结构
# ------------------------------------------------------------------

@dataclass
class RequestRecord:
    timestamp: str
    request_id: Optional[str]
    user_id: Optional[str]
    endpoint: str
    num_items: int
    top_k: Optional[int]
    status: str
    latency_ms: float
    error_code: Optional[str] = None


@dataclass
class AssignmentRecord:
    timestamp: str
    experiment_id: str
    request_id: Optional[str]
    user_id: Optional[str]
    group: str
    model_key: str
    assignment_strategy: str
    hash_value: Optional[int] = None
    traffic_bucket: Optional[str] = None


@dataclass
class RecommendationRecord:
    timestamp: str
    request_id: Optional[str]
    user_id: Optional[str]
    experiment_id: Optional[str]
    group: Optional[str]
    model_key: str
    video_id: Optional[str]
    rank: int
    score: float


@dataclass
class EventRecord:
    timestamp: str
    experiment_id: Optional[str]
    group: Optional[str]
    model_key: Optional[str]
    user_id: Optional[str]
    request_id: Optional[str]
    video_id: Optional[str]
    event_type: str
    event_value: Optional[float] = None
    rank: Optional[int] = None
    score: Optional[float] = None


@dataclass
class GroupMetrics:
    """单组聚合指标。"""
    # 标识
    experiment_id: str = ""
    group: str = ""
    model_key: str = ""

    # request 指标
    request_count: int = 0
    unique_users: int = 0
    success_requests: int = 0
    failed_requests: int = 0
    error_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    # assignment 指标
    assignment_count: int = 0
    traffic_ratio: float = 0.0
    expected_traffic_ratio: Optional[float] = None
    traffic_diff_abs: Optional[float] = None
    traffic_balance_warning: bool = False

    # recommendation 指标
    impression_count: int = 0
    unique_recommended_items: int = 0
    avg_score: float = 0.0
    score_std: float = 0.0
    score_min: float = 0.0
    score_max: float = 0.0
    score_p25: float = 0.0
    score_p50: float = 0.0
    score_p75: float = 0.0
    coverage: float = 0.0

    # event 指标
    clicks: Optional[int] = None
    ctr: Optional[float] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    shares: Optional[int] = None
    collects: Optional[int] = None
    interaction_count: Optional[int] = None
    interaction_rate: Optional[float] = None
    event_metrics_available: bool = False

    # 候选池总量（用于 coverage 分母）
    total_candidate_items: int = 0
    coverage_note: str = ""


# ------------------------------------------------------------------
# JSONL 读取
# ------------------------------------------------------------------

def read_jsonl(path: Path, skip_empty: bool = True) -> list[dict[str, Any]]:
    """读取 JSONL 文件，返回解析后的 dict 列表。如果文件不存在，抛出 FileNotFoundError。"""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped and skip_empty:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as e:
                logger.warning("JSON 解析失败 (%s:%d): %s", path.name, line_no, e)
    return records


def read_optional_jsonl(path: Path) -> list[dict[str, Any]] | None:
    """读取可选的 JSONL 文件。文件不存在时返回 None。"""
    if not path.is_file():
        return None
    return read_jsonl(path)


def read_single_json(path: Path) -> dict[str, Any] | None:
    """读取单个 JSON 对象文件（非 JSONL）。文件不存在时返回 None。"""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("JSON 解析失败 (%s): %s", path.name, e)
        return None


def parse_request_records(raw: list[dict[str, Any]]) -> list[RequestRecord]:
    records = []
    for r in raw:
        records.append(RequestRecord(
            timestamp=r.get("timestamp", ""),
            request_id=r.get("request_id"),
            user_id=r.get("user_id"),
            endpoint=r.get("endpoint", ""),
            num_items=r.get("num_items", 0),
            top_k=r.get("top_k"),
            status=r.get("status", ""),
            latency_ms=float(r.get("latency_ms", 0)),
            error_code=r.get("error_code"),
        ))
    return records


def parse_assignment_records(raw: list[dict[str, Any]]) -> list[AssignmentRecord]:
    records = []
    for r in raw:
        records.append(AssignmentRecord(
            timestamp=r.get("timestamp", ""),
            experiment_id=r.get("experiment_id", ""),
            request_id=r.get("request_id"),
            user_id=r.get("user_id"),
            group=r.get("group", ""),
            model_key=r.get("model_key", ""),
            assignment_strategy=r.get("assignment_strategy", ""),
            hash_value=r.get("hash_value"),
            traffic_bucket=r.get("traffic_bucket"),
        ))
    return records


def parse_recommendation_records(raw: list[dict[str, Any]]) -> list[RecommendationRecord]:
    records = []
    for r in raw:
        records.append(RecommendationRecord(
            timestamp=r.get("timestamp", ""),
            request_id=r.get("request_id"),
            user_id=r.get("user_id"),
            experiment_id=r.get("experiment_id"),
            group=r.get("group"),
            model_key=r.get("model_key", ""),
            video_id=r.get("video_id"),
            rank=int(r.get("rank", 0)),
            score=float(r.get("score", 0)),
        ))
    return records


def parse_event_records(raw: list[dict[str, Any]]) -> list[EventRecord]:
    records = []
    for r in raw:
        records.append(EventRecord(
            timestamp=r.get("timestamp", ""),
            experiment_id=r.get("experiment_id"),
            group=r.get("group"),
            model_key=r.get("model_key"),
            user_id=r.get("user_id"),
            request_id=r.get("request_id"),
            video_id=r.get("video_id"),
            event_type=r.get("event_type", ""),
            event_value=float(r["event_value"]) if r.get("event_value") is not None else None,
            rank=int(r["rank"]) if r.get("rank") is not None else None,
            score=float(r["score"]) if r.get("score") is not None else None,
        ))
    return records


# ------------------------------------------------------------------
# 指标计算
# ------------------------------------------------------------------

def compute_request_metrics(requests: list[RequestRecord]) -> dict[str, Any]:
    """从 request 记录计算基础指标。"""
    total = len(requests)
    if total == 0:
        return {
            "request_count": 0,
            "unique_users": 0,
            "success_requests": 0,
            "failed_requests": 0,
            "error_rate": 0.0,
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }

    unique_users = set()
    success = 0
    failed = 0
    latencies = []
    for r in requests:
        if r.user_id:
            unique_users.add(r.user_id)
        if r.status == "ok":
            success += 1
        else:
            failed += 1
        latencies.append(r.latency_ms)

    latencies.sort()
    avg_lat = sum(latencies) / len(latencies)
    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)

    return {
        "request_count": total,
        "unique_users": len(unique_users),
        "success_requests": success,
        "failed_requests": failed,
        "error_rate": round(failed / total, 6) if total > 0 else 0.0,
        "avg_latency_ms": round(avg_lat, 2),
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
    }


def _percentile(sorted_values: list[float], p: int) -> float:
    """计算已排序列表的百分位数（线性插值）。"""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    k = (p / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def compute_assignment_metrics(
    assignments: list[AssignmentRecord],
    experiment_id: str,
    group: str,
    total_assignments: int,
    expected_traffic_ratio: Optional[float] = None,
) -> dict[str, Any]:
    """从 assignment 记录计算指标。"""
    count = len(assignments)
    traffic_ratio = round(count / total_assignments, 6) if total_assignments > 0 else 0.0

    result: dict[str, Any] = {
        "assignment_count": count,
        "traffic_ratio": traffic_ratio,
        "expected_traffic_ratio": expected_traffic_ratio,
        "traffic_diff_abs": None,
        "traffic_balance_warning": False,
    }

    if expected_traffic_ratio is not None:
        diff = abs(traffic_ratio - expected_traffic_ratio)
        result["traffic_diff_abs"] = round(diff, 6)
    return result


def compute_recommendation_metrics(
    recommendations: list[RecommendationRecord],
    total_candidate_items: int = 0,
) -> dict[str, Any]:
    """从 recommendation 记录计算指标。"""
    if not recommendations:
        return {
            "impression_count": 0,
            "unique_recommended_items": 0,
            "avg_score": 0.0,
            "score_std": 0.0,
            "score_min": 0.0,
            "score_max": 0.0,
            "score_p25": 0.0,
            "score_p50": 0.0,
            "score_p75": 0.0,
            "coverage": 0.0,
            "total_candidate_items": total_candidate_items,
            "coverage_note": "",
        }

    scores = [r.score for r in recommendations]
    scores.sort()
    unique_video_ids = set()
    for r in recommendations:
        if r.video_id:
            unique_video_ids.add(r.video_id)

    avg_score = sum(scores) / len(scores)
    variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
    score_std = math.sqrt(variance)

    # coverage
    coverage = 0.0
    coverage_note = ""
    if total_candidate_items > 0:
        coverage = round(len(unique_video_ids) / total_candidate_items, 6)
    else:
        # 如果没有给出候选池总量，用当前日志中的 unique video_id
        total_candidate_items = len(unique_video_ids)
        coverage = 1.0
        coverage_note = "分母为当前日志 unique video_id 总数（非完整候选池），coverage=1.0 是归一化结果"

    return {
        "impression_count": len(recommendations),
        "unique_recommended_items": len(unique_video_ids),
        "avg_score": round(avg_score, 6),
        "score_std": round(score_std, 6),
        "score_min": round(scores[0], 6),
        "score_max": round(scores[-1], 6),
        "score_p25": round(_percentile(scores, 25), 6),
        "score_p50": round(_percentile(scores, 50), 6),
        "score_p75": round(_percentile(scores, 75), 6),
        "coverage": coverage,
        "total_candidate_items": total_candidate_items,
        "coverage_note": coverage_note,
    }


def compute_event_metrics(events: list[EventRecord]) -> dict[str, Any]:
    """从 event 记录计算模拟事件指标。"""
    if not events:
        return {
            "clicks": None,
            "ctr": None,
            "likes": None,
            "comments": None,
            "shares": None,
            "collects": None,
            "interaction_count": None,
            "interaction_rate": None,
            "event_metrics_available": False,
        }

    impressions = sum(1 for e in events if e.event_type == "impression")
    clicks = sum(1 for e in events if e.event_type == "click")
    likes = sum(1 for e in events if e.event_type == "like")
    comments = sum(1 for e in events if e.event_type == "comment")
    shares = sum(1 for e in events if e.event_type == "share")
    collects = sum(1 for e in events if e.event_type == "collect")
    interaction_count = likes + comments + shares + collects

    ctr = round(clicks / impressions, 6) if impressions > 0 else None
    interaction_rate = round(interaction_count / impressions, 6) if impressions > 0 else None

    return {
        "clicks": clicks,
        "ctr": ctr,
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "collects": collects,
        "interaction_count": interaction_count,
        "interaction_rate": interaction_rate,
        "event_metrics_available": True,
    }


# ------------------------------------------------------------------
# 异常检测
# ------------------------------------------------------------------

@dataclass
class WarningItem:
    type: str
    severity: str  # "info" | "warning" | "error"
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


def detect_warnings(
    group_metrics: dict[str, GroupMetrics],
    all_requests: list[RequestRecord],
    all_assignments: list[AssignmentRecord],
    all_recommendations: list[RecommendationRecord],
    min_requests_per_group: int = 100,
    traffic_tolerance: float = 0.05,
    latency_threshold_ms: float = 1000.0,
    error_rate_threshold: float = 0.05,
) -> list[WarningItem]:
    """综合异常检测。"""
    warnings: list[WarningItem] = []

    req_ids_request = {r.request_id for r in all_requests if r.request_id}
    req_ids_assignment = {a.request_id for a in all_assignments if a.request_id}
    req_ids_recommendation = {r.request_id for r in all_recommendations if r.request_id}

    # ---- sample_size_warning ----
    for g_key, gm in group_metrics.items():
        if gm.request_count < min_requests_per_group:
            warnings.append(WarningItem(
                type="sample_size_warning",
                severity="warning",
                message=f"组 '{g_key}' 样本量不足: request_count={gm.request_count} < {min_requests_per_group}",
                detail={"group": g_key, "request_count": gm.request_count,
                        "min_requests_per_group": min_requests_per_group},
            ))

    # ---- traffic_balance_warning ----
    for g_key, gm in group_metrics.items():
        if gm.expected_traffic_ratio is not None and gm.traffic_diff_abs is not None:
            if gm.traffic_diff_abs > traffic_tolerance:
                warnings.append(WarningItem(
                    type="traffic_balance_warning",
                    severity="warning",
                    message=(
                        f"组 '{g_key}' 流量比例偏离: 实际={gm.traffic_ratio:.1%}, "
                        f"预期={gm.expected_traffic_ratio:.1%}, 差值={gm.traffic_diff_abs:.1%}"
                    ),
                    detail={
                        "group": g_key,
                        "actual_ratio": gm.traffic_ratio,
                        "expected_ratio": gm.expected_traffic_ratio,
                        "diff_abs": gm.traffic_diff_abs,
                        "tolerance": traffic_tolerance,
                    },
                ))

    # ---- log_consistency_warning ----
    # recommendation 中的 request_id 在 request_log 中找不到
    for rid in req_ids_recommendation:
        if rid and rid not in req_ids_request:
            warnings.append(WarningItem(
                type="log_consistency_warning",
                severity="warning",
                message=f"recommendation_log 中的 request_id='{rid}' 在 request_log 中找不到",
                detail={"request_id": rid, "source": "recommendation_log",
                        "missing_in": "request_log"},
            ))
            break  # 只报告第一个

    # assignment 中的 request_id 在 request_log 中找不到
    for rid in req_ids_assignment:
        if rid and rid not in req_ids_request:
            warnings.append(WarningItem(
                type="log_consistency_warning",
                severity="warning",
                message=f"ab_assignment_log 中的 request_id='{rid}' 在 request_log 中找不到",
                detail={"request_id": rid, "source": "ab_assignment_log",
                        "missing_in": "request_log"},
            ))
            break

    # ---- error_rate_warning ----
    for g_key, gm in group_metrics.items():
        if gm.error_rate > error_rate_threshold:
            warnings.append(WarningItem(
                type="error_rate_warning",
                severity="error",
                message=f"组 '{g_key}' 错误率过高: error_rate={gm.error_rate:.2%} > {error_rate_threshold:.0%}",
                detail={"group": g_key, "error_rate": gm.error_rate,
                        "threshold": error_rate_threshold},
            ))

    # ---- latency_warning ----
    for g_key, gm in group_metrics.items():
        if gm.p95_latency_ms > latency_threshold_ms:
            warnings.append(WarningItem(
                type="latency_warning",
                severity="warning",
                message=f"组 '{g_key}' p95 延迟过高: {gm.p95_latency_ms:.1f}ms > {latency_threshold_ms:.0f}ms",
                detail={"group": g_key, "p95_latency_ms": gm.p95_latency_ms,
                        "threshold_ms": latency_threshold_ms},
            ))

    return warnings


# ------------------------------------------------------------------
# 胜负判定（第一版：不做显著性检验）
# ------------------------------------------------------------------

def make_verdict(
    group_metrics: dict[str, GroupMetrics],
    warnings: list[WarningItem],
    event_metrics_available: bool,
) -> dict[str, Any]:
    """输出胜负判定。第一版不做显著性检验。"""
    groups = list(group_metrics.values())
    if len(groups) < 2:
        return {
            "primary_metric": "avg_score",
            "winner": None,
            "effect_size": None,
            "effect_percentage": None,
            "significance_test": "not_performed",
            "conclusion": "inconclusive",
            "note": "至少需要两个组才能做比较",
        }

    # 检查是否有严重异常
    severe_warnings = [w for w in warnings if w.severity in ("warning", "error")]
    has_severe_warnings = len(severe_warnings) > 0

    # 检查 sample_size
    has_sample_warning = any(w.type == "sample_size_warning" for w in warnings)

    # primary_metric
    if event_metrics_available:
        primary_metric = "interaction_rate"
    else:
        primary_metric = "avg_score"

    # 组间比较：avg_score
    g1, g2 = groups[0], groups[1]
    effect_size = g1.avg_score - g2.avg_score
    effect_pct = (
        (effect_size / max(abs(g2.avg_score), 1e-10)) * 100
        if g2.avg_score != 0
        else 0.0
    )

    # 判断 winner
    if has_severe_warnings or has_sample_warning:
        conclusion = "inconclusive"
        winner = None
        note = "存在 warning，不能判定胜负"
    elif event_metrics_available:
        # 如果有事件指标，用 interaction_rate 或 ctr
        winner = _compare_groups_event(g1, g2, primary_metric)
        if winner:
            conclusion = f"{winner} 优于对照组（模拟事件指标）"
            note = "基于模拟事件指标，不代表真实线上业务效果"
        else:
            conclusion = "无显著差异"
            note = "模拟事件指标差异不显著"
    else:
        # 没有事件指标，仅以 avg_score 参考
        if abs(effect_pct) >= 3.0:
            better = g1.group if effect_size > 0 else g2.group
            conclusion = f"simulation_only"
            winner = f"{better} ({g1.model_key if effect_size > 0 else g2.model_key})"
            note = (
                f"avg_score 差异 >= 3%，但 avg_score 是模型输出分数，"
                f"不是用户真实反馈，不能作为推荐效果判定依据"
            )
        else:
            conclusion = "inconclusive"
            winner = None
            note = (
                f"avg_score 差异 < 3%，没有真实用户行为日志，"
                f"无法判定推荐效果差异"
            )

    return {
        "primary_metric": primary_metric,
        "winner": winner,
        "effect_size": round(effect_size, 6),
        "effect_percentage": round(effect_pct, 4),
        "significance_test": "not_performed",
        "conclusion": conclusion,
        "note": note,
    }


def _compare_groups_event(g1: GroupMetrics, g2: GroupMetrics,
                           primary_metric: str) -> str | None:
    """基于事件指标比较两组（不做显著性检验，仅供第一版参考）。"""
    # 简单比较：如果某组指标明显高则标记
    v1 = getattr(g1, primary_metric, None)
    v2 = getattr(g2, primary_metric, None)
    if v1 is None or v2 is None or v1 == v2:
        return None
    if v1 > v2 * 1.03:
        return g1.group
    elif v2 > v1 * 1.03:
        return g2.group
    return None


# ------------------------------------------------------------------
# 聚合主逻辑
# ------------------------------------------------------------------

def compute_metrics(
    log_dir: Path,
    experiment_id_filter: Optional[str] = None,
    min_requests_per_group: int = 100,
    traffic_tolerance: float = 0.05,
    latency_threshold_ms: float = 1000.0,
    error_rate_threshold: float = 0.05,
) -> dict[str, Any]:
    """主聚合逻辑。

    读取 JSONL 日志，按 experiment_id + group + model_key 聚合，计算指标。
    返回可序列化 dict。
    """
    # ---- 读取必需日志 ----
    request_path = log_dir / "request_log.jsonl"
    rec_path = log_dir / "recommendation_log.jsonl"
    assignment_path = log_dir / "ab_assignment_log.jsonl"
    event_path = log_dir / "event_log.jsonl"
    meta_path = log_dir / "online_run_meta.json"

    if not request_path.is_file():
        raise FileNotFoundError(f"必需日志不存在: {request_path}")
    if not rec_path.is_file():
        raise FileNotFoundError(f"必需日志不存在: {rec_path}")

    # ---- 解析 ----
    raw_requests = read_jsonl(request_path)
    raw_recommendations = read_jsonl(rec_path)
    raw_assignments = read_optional_jsonl(assignment_path)
    raw_events = read_optional_jsonl(event_path)
    raw_meta = read_single_json(meta_path)
    meta_list = [raw_meta] if raw_meta else []

    requests = parse_request_records(raw_requests)
    recommendations = parse_recommendation_records(raw_recommendations)
    assignments = parse_assignment_records(raw_assignments) if raw_assignments else []
    events = parse_event_records(raw_events) if raw_events else []

    # ---- 按 experiment_id 分组 ----
    # 收集所有 experiment_id
    all_exp_ids: set[str] = set()
    for a in assignments:
        if a.experiment_id:
            all_exp_ids.add(a.experiment_id)
    # 也能从 recommendation 中收集
    for r in recommendations:
        if r.experiment_id:
            all_exp_ids.add(r.experiment_id)

    if not all_exp_ids:
        # 没有 experiment_id → 按 model_key 聚合
        all_exp_ids.add("(no_experiment)")

    # 如果指定了 experiment_id 且不在集合中
    if experiment_id_filter and experiment_id_filter not in all_exp_ids:
        logger.warning("experiment_id '%s' 在日志中不存在", experiment_id_filter)
        # 仍然允许，但返回空

    if experiment_id_filter:
        all_exp_ids = {eid for eid in all_exp_ids if eid == experiment_id_filter}

    # ---- 按 experiment_id 分别计算 ----
    experiment_results: dict[str, Any] = {}

    for exp_id in sorted(all_exp_ids):
        # 过滤 assignment
        exp_assignments = [a for a in assignments if a.experiment_id == exp_id]

        # 找出该实验涉及的所有 group
        groups_in_exp: set[str] = set()
        for a in exp_assignments:
            if a.group:
                groups_in_exp.add(a.group)
        for r in recommendations:
            if r.experiment_id == exp_id and r.group:
                groups_in_exp.add(r.group)

        if not groups_in_exp:
            groups_in_exp.add("(no_group)")

        # 统计总 assignment（用于 traffic ratio 分母）
        total_assignments = len(exp_assignments)

        # 获取实验配置（如果有 online_run_meta，尝试读取）
        expected_traffic: dict[str, float] = {}
        if meta_list:
            meta = meta_list[0] if isinstance(meta_list, list) and meta_list else {}
            # 目前 meta 中没有 experiment config，用均匀分配作为预期
            n_groups = len(groups_in_exp)
            for g in groups_in_exp:
                expected_traffic[g] = 1.0 / n_groups if n_groups > 0 else 0.0

        n_groups = len(groups_in_exp)
        for g in groups_in_exp:
            expected_traffic[g] = 1.0 / n_groups if n_groups > 0 else 0.0

        # 逐组计算
        group_metrics_map: dict[str, GroupMetrics] = {}

        for group in sorted(groups_in_exp):
            gm = GroupMetrics(
                experiment_id=exp_id,
                group=group,
            )

            # ---- request 过滤 ----
            # request_log 不直接包含 experiment_id/group，需要从 assignment 或 recommendation 反查
            exp_req_ids: set[str] = set()
            for a in exp_assignments:
                if a.group == group and a.request_id:
                    exp_req_ids.add(a.request_id)
            # 也看 recommendation
            for r in recommendations:
                if r.experiment_id == exp_id and r.group == group and r.request_id:
                    exp_req_ids.add(r.request_id)

            group_requests = [r for r in requests if r.request_id in exp_req_ids]
            req_metrics = compute_request_metrics(group_requests)
            gm.request_count = req_metrics["request_count"]
            gm.unique_users = req_metrics["unique_users"]
            gm.success_requests = req_metrics["success_requests"]
            gm.failed_requests = req_metrics["failed_requests"]
            gm.error_rate = req_metrics["error_rate"]
            gm.avg_latency_ms = req_metrics["avg_latency_ms"]
            gm.p50_latency_ms = req_metrics["p50_latency_ms"]
            gm.p95_latency_ms = req_metrics["p95_latency_ms"]

            # ---- assignment ----
            group_assignments = [a for a in exp_assignments if a.group == group]
            expected = expected_traffic.get(group)
            assign_metrics = compute_assignment_metrics(
                group_assignments, exp_id, group, total_assignments, expected,
            )
            gm.assignment_count = assign_metrics["assignment_count"]
            gm.traffic_ratio = assign_metrics["traffic_ratio"]
            gm.expected_traffic_ratio = assign_metrics["expected_traffic_ratio"]
            gm.traffic_diff_abs = assign_metrics["traffic_diff_abs"]
            gm.traffic_balance_warning = assign_metrics["traffic_balance_warning"]

            # 从 assignment 记录中获取 model_key
            if group_assignments:
                gm.model_key = group_assignments[0].model_key

            # ---- recommendation ----
            group_recs = [
                r for r in recommendations
                if r.experiment_id == exp_id and r.group == group
            ]
            total_candidate = len({r.video_id for r in recommendations
                                   if r.experiment_id == exp_id})
            rec_metrics = compute_recommendation_metrics(group_recs, total_candidate)
            gm.impression_count = rec_metrics["impression_count"]
            gm.unique_recommended_items = rec_metrics["unique_recommended_items"]
            gm.avg_score = rec_metrics["avg_score"]
            gm.score_std = rec_metrics["score_std"]
            gm.score_min = rec_metrics["score_min"]
            gm.score_max = rec_metrics["score_max"]
            gm.score_p25 = rec_metrics["score_p25"]
            gm.score_p50 = rec_metrics["score_p50"]
            gm.score_p75 = rec_metrics["score_p75"]
            gm.coverage = rec_metrics["coverage"]
            gm.total_candidate_items = rec_metrics["total_candidate_items"]
            gm.coverage_note = rec_metrics["coverage_note"]

            # ---- event ----
            group_events = [
                e for e in events
                if e.experiment_id == exp_id and e.group == group
            ]
            event_metrics = compute_event_metrics(group_events)
            gm.clicks = event_metrics["clicks"]
            gm.ctr = event_metrics["ctr"]
            gm.likes = event_metrics["likes"]
            gm.comments = event_metrics["comments"]
            gm.shares = event_metrics["shares"]
            gm.collects = event_metrics["collects"]
            gm.interaction_count = event_metrics["interaction_count"]
            gm.interaction_rate = event_metrics["interaction_rate"]
            gm.event_metrics_available = event_metrics["event_metrics_available"]

            group_key = f"{group} ({gm.model_key})" if gm.model_key else group
            group_metrics_map[group_key] = gm

        # ---- 异常检测 ----
        warnings_list = detect_warnings(
            group_metrics_map, requests, assignments, recommendations,
            min_requests_per_group=min_requests_per_group,
            traffic_tolerance=traffic_tolerance,
            latency_threshold_ms=latency_threshold_ms,
            error_rate_threshold=error_rate_threshold,
        )

        # ---- 事件可用性 ----
        event_available = any(gm.event_metrics_available for gm in group_metrics_map.values())

        # ---- 胜负判定 ----
        verdict = make_verdict(group_metrics_map, warnings_list, event_available)

        # ---- 构建 groups dict ----
        groups_dict: dict[str, dict[str, Any]] = {}
        for g_key, gm in group_metrics_map.items():
            groups_dict[g_key] = {
                "experiment_id": gm.experiment_id,
                "group": gm.group,
                "model_key": gm.model_key,
                # request
                "request_count": gm.request_count,
                "unique_users": gm.unique_users,
                "success_requests": gm.success_requests,
                "failed_requests": gm.failed_requests,
                "error_rate": gm.error_rate,
                "avg_latency_ms": gm.avg_latency_ms,
                "p50_latency_ms": gm.p50_latency_ms,
                "p95_latency_ms": gm.p95_latency_ms,
                # assignment
                "assignment_count": gm.assignment_count,
                "traffic_ratio": gm.traffic_ratio,
                "expected_traffic_ratio": gm.expected_traffic_ratio,
                "traffic_diff_abs": gm.traffic_diff_abs,
                "traffic_balance_warning": gm.traffic_balance_warning,
                # recommendation
                "impression_count": gm.impression_count,
                "unique_recommended_items": gm.unique_recommended_items,
                "avg_score": gm.avg_score,
                "score_std": gm.score_std,
                "score_min": gm.score_min,
                "score_max": gm.score_max,
                "score_p25": gm.score_p25,
                "score_p50": gm.score_p50,
                "score_p75": gm.score_p75,
                "coverage": gm.coverage,
                "total_candidate_items": gm.total_candidate_items,
                "coverage_note": gm.coverage_note,
                # event
                "clicks": gm.clicks,
                "ctr": gm.ctr,
                "likes": gm.likes,
                "comments": gm.comments,
                "shares": gm.shares,
                "collects": gm.collects,
                "interaction_count": gm.interaction_count,
                "interaction_rate": gm.interaction_rate,
            }

        # warnings 的 dict 副本
        warnings_dict = [
            {"type": w.type, "severity": w.severity, "message": w.message, "detail": w.detail}
            for w in warnings_list
        ]

        experiment_results[exp_id] = {
            "experiment_id": exp_id,
            "groups": groups_dict,
            "warnings": warnings_dict,
            "verdict": verdict,
        }

    return {
        "log_dir": str(log_dir),
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "experiments": experiment_results,
        "disclaimer": (
            "This is a local simulation A/B test. "
            "Results do not represent real online business metrics. "
            "All metrics are computed from offline simulation logs."
        ),
    }


# ------------------------------------------------------------------
# 输出
# ------------------------------------------------------------------

def write_report(result: dict[str, Any], output_dir: Path) -> Path:
    """输出 ab_metrics_report.json。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "ab_metrics_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return report_path


def write_summary(result: dict[str, Any], output_dir: Path) -> Path:
    """输出 ab_metrics_summary.csv，每行一个 group。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "ab_metrics_summary.csv"

    rows: list[dict[str, Any]] = []
    for exp_id, exp_data in result.get("experiments", {}).items():
        for g_key, g in exp_data.get("groups", {}).items():
            # 收集该组的 warning
            group_warnings = []
            for w in exp_data.get("warnings", []):
                if w.get("detail", {}).get("group") == g.get("group"):
                    group_warnings.append(w["type"])
                elif g_key in w.get("message", ""):
                    group_warnings.append(w["type"])

            rows.append({
                "experiment_id": exp_id,
                "group": g.get("group", ""),
                "model_key": g.get("model_key", ""),
                "request_count": g.get("request_count", 0),
                "unique_users": g.get("unique_users", 0),
                "impression_count": g.get("impression_count", 0),
                "avg_score": g.get("avg_score", 0),
                "p95_latency_ms": g.get("p95_latency_ms", 0),
                "error_rate": g.get("error_rate", 0),
                "traffic_ratio": g.get("traffic_ratio", 0),
                "ctr": g.get("ctr", ""),
                "interaction_rate": g.get("interaction_rate", ""),
                "coverage": g.get("coverage", 0),
                "warnings": "; ".join(group_warnings) if group_warnings else "",
            })

    if not rows:
        rows.append({"experiment_id": result.get("experiments", {}).get("", {}).get("experiment_id", ""),
                      "note": "no data"})

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return summary_path


def write_warnings(result: dict[str, Any], output_dir: Path) -> Path:
    """可选输出 warnings.json。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    warn_path = output_dir / "warnings.json"

    all_warnings: list[dict[str, Any]] = []
    for exp_data in result.get("experiments", {}).values():
        all_warnings.extend(exp_data.get("warnings", []))

    with warn_path.open("w", encoding="utf-8") as f:
        json.dump(all_warnings, f, ensure_ascii=False, indent=2)

    return warn_path


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A/B 实验指标计算：从 online_simulation JSONL 日志计算实验指标",
    )
    parser.add_argument(
        "--log-dir", required=True,
        help="online_simulation 日志目录，如 outputs/online_simulation/<online_run_id>/",
    )
    parser.add_argument(
        "--experiment-id", default=None,
        help="可选，指定 experiment_id；不指定则读取全部",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="可选输出目录，默认 <log_dir>/metrics/",
    )
    parser.add_argument(
        "--min-requests-per-group", type=int, default=100,
        help="每组最低请求数（用于 sample_size_warning），默认 100",
    )
    parser.add_argument(
        "--traffic-tolerance", type=float, default=0.05,
        help="流量比例偏离容忍度，默认 0.05（5%）",
    )
    parser.add_argument(
        "--latency-threshold-ms", type=float, default=1000.0,
        help="p95 latency 阈值（ms），默认 1000",
    )
    parser.add_argument(
        "--error-rate-threshold", type=float, default=0.05,
        help="错误率阈值，默认 0.05（5%）",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    log_dir = Path(args.log_dir)

    if not log_dir.is_dir():
        print(f"错误: 日志目录不存在: {log_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else log_dir / "metrics"

    print(f"日志目录: {log_dir}")
    print(f"输出目录: {output_dir}")
    if args.experiment_id:
        print(f"实验过滤: {args.experiment_id}")
    print(f"参数: min_requests={args.min_requests_per_group}, "
          f"traffic_tolerance={args.traffic_tolerance}, "
          f"latency_threshold={args.latency_threshold_ms}ms, "
          f"error_rate_threshold={args.error_rate_threshold}")
    print()

    # 检查必需日志
    request_path = log_dir / "request_log.jsonl"
    rec_path = log_dir / "recommendation_log.jsonl"
    if not request_path.is_file():
        print(f"错误: 必需日志不存在: {request_path}", file=sys.stderr)
        sys.exit(1)
    if not rec_path.is_file():
        print(f"错误: 必需日志不存在: {rec_path}", file=sys.stderr)
        sys.exit(1)

    # 检查 event_log
    event_path = log_dir / "event_log.jsonl"
    if event_path.is_file():
        print("模拟事件日志 (event_log.jsonl) 存在")
    else:
        print("模拟事件日志不存在 — CTR/interaction_rate 将标记为 unavailable")

    # 检查 assignment_log
    assign_path = log_dir / "ab_assignment_log.jsonl"
    if assign_path.is_file():
        print("A/B 分配日志 (ab_assignment_log.jsonl) 存在")
    else:
        print("A/B 分配日志不存在 — 无法计算流量比例")

    print()

    # 计算
    result = compute_metrics(
        log_dir=log_dir,
        experiment_id_filter=args.experiment_id,
        min_requests_per_group=args.min_requests_per_group,
        traffic_tolerance=args.traffic_tolerance,
        latency_threshold_ms=args.latency_threshold_ms,
        error_rate_threshold=args.error_rate_threshold,
    )

    # 输出
    report_file = write_report(result, output_dir)
    summary_file = write_summary(result, output_dir)
    warnings_file = write_warnings(result, output_dir)

    print(f"指标报告: {report_file}")
    print(f"指标摘要: {summary_file}")
    print(f"异常检测: {warnings_file}")
    print()

    # 打印摘要
    for exp_id, exp_data in result.get("experiments", {}).items():
        print(f"=== 实验: {exp_id} ===")
        for g_key, g in exp_data.get("groups", {}).items():
            print(f"  [{g_key}]")
            print(f"    请求数: {g['request_count']}, 用户数: {g['unique_users']}")
            print(f"    曝光数: {g['impression_count']}, 平均分: {g['avg_score']}")
            print(f"    p95 延迟: {g['p95_latency_ms']}ms, 错误率: {g['error_rate']:.2%}")
            print(f"    流量比例: {g['traffic_ratio']:.1%} (预期: {g.get('expected_traffic_ratio', 'N/A')})")
            print(f"    覆盖率: {g['coverage']:.2%}")
            if g.get("event_metrics_available"):
                print(f"    CTR: {g['ctr']}, Interaction Rate: {g['interaction_rate']}")
            else:
                print(f"    模拟事件: unavailable (event_log 不存在)")
            print()

        print("  Warnings:")
        if exp_data.get("warnings"):
            for w in exp_data["warnings"]:
                print(f"    [{w['severity']}] {w['type']}: {w['message']}")
        else:
            print("    (无)")
        print()

        v = exp_data.get("verdict", {})
        print(f"  判定:")
        print(f"    Primary metric: {v.get('primary_metric')}")
        print(f"    Winner: {v.get('winner', 'N/A')}")
        print(f"    Effect size: {v.get('effect_size', 'N/A')}")
        print(f"    Effect %%: {v.get('effect_percentage', 'N/A')}%%")
        print(f"    Conclusion: {v.get('conclusion', 'N/A')}")
        print(f"    Note: {v.get('note', '')}")
        print()

    print(f"Disclaimer: {result['disclaimer']}")


if __name__ == "__main__":
    main()