"""请求日志与实验日志。

输出目录: outputs/online_simulation/<online_run_id>/

文件列表:
- request_log.jsonl        — 推荐请求日志
- ab_assignment_log.jsonl  — A/B 分配日志
- recommendation_log.jsonl — 推荐结果日志（每个推荐 item 一条记录）
- online_run_meta.json     — 服务运行元信息

日志写入失败不影响推荐主流程（try/except + warning）。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RequestLogger:
    """推荐请求日志记录器。

    以 JSONL 格式记录推荐请求、A/B 分配和推荐结果。
    """

    def __init__(
        self,
        online_run_id: Optional[str] = None,
        output_dir: Optional[str | Path] = None,
    ):
        if output_dir is not None:
            self._output_dir = Path(output_dir)
        else:
            run_id = online_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
            self._output_dir = Path("outputs") / "online_simulation" / run_id

        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._request_log_path = self._output_dir / "request_log.jsonl"
        self._assignment_log_path = self._output_dir / "ab_assignment_log.jsonl"
        self._recommendation_log_path = self._output_dir / "recommendation_log.jsonl"

        self._write_meta()

    def _write_meta(self) -> None:
        meta = {
            "online_run_id": self._output_dir.name,
            "output_dir": str(self._output_dir),
            "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "generator": "RequestLogger",
        }
        meta_path = self._output_dir / "online_run_meta.json"
        try:
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("online_run_meta.json 写入失败: %s", e)

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("日志写入失败 (%s): %s", path.name, e)

    def log_request(
        self,
        request_id: Optional[str],
        user_id: Optional[str],
        endpoint: str,
        num_items: int,
        top_k: Optional[int],
        status: str,
        latency_ms: float,
        error_code: Optional[str] = None,
    ) -> None:
        """记录推荐请求日志到 request_log.jsonl。"""
        record: dict[str, Any] = {
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "request_id": request_id,
            "user_id": user_id,
            "endpoint": endpoint,
            "num_items": num_items,
            "top_k": top_k,
            "status": status,
            "latency_ms": round(latency_ms, 2),
        }
        if error_code:
            record["error_code"] = error_code
        self._append_jsonl(self._request_log_path, record)

    def log_assignment(
        self,
        experiment_id: str,
        request_id: Optional[str],
        user_id: Optional[str],
        group: str,
        model_key: str,
        assignment_strategy: str,
        hash_value: Optional[int] = None,
        traffic_bucket: Optional[str] = None,
    ) -> None:
        """记录 A/B 分配日志到 ab_assignment_log.jsonl。"""
        record: dict[str, Any] = {
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "experiment_id": experiment_id,
            "request_id": request_id,
            "user_id": user_id,
            "group": group,
            "model_key": model_key,
            "assignment_strategy": assignment_strategy,
        }
        if hash_value is not None:
            record["hash_value"] = hash_value
        if traffic_bucket is not None:
            record["traffic_bucket"] = traffic_bucket
        self._append_jsonl(self._assignment_log_path, record)

    def log_recommendation(
        self,
        request_id: Optional[str],
        user_id: Optional[str],
        model_key: str,
        video_id: Optional[str],
        rank: int,
        score: float,
        experiment_id: Optional[str] = None,
        group: Optional[str] = None,
    ) -> None:
        """记录推荐结果到 recommendation_log.jsonl（每个 item 一条）。"""
        record: dict[str, Any] = {
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "request_id": request_id,
            "user_id": user_id,
            "experiment_id": experiment_id,
            "group": group,
            "model_key": model_key,
            "video_id": video_id,
            "rank": rank,
            "score": round(score, 6),
        }
        self._append_jsonl(self._recommendation_log_path, record)

    @property
    def output_dir(self) -> Path:
        return self._output_dir