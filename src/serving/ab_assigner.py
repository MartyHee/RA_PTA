"""用户分组与流量分配。

第一版支持两种分组策略：
- hash_mod: md5(user_id + experiment_id + salt) % 100（默认，同一用户稳定分配同一组）
- random: 每次请求独立随机分配

默认实验: exp_dnn_baseline_vs_tuned
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Optional


class ExperimentConfig:
    """实验配置定义。"""

    def __init__(self, config: dict[str, Any]):
        self.experiment_id: str = config["experiment_id"]
        self.status: str = config.get("status", "active")
        self.unit: str = config.get("unit", "user_id")
        self.description: str = config.get("description", "")
        assignment = config.get("assignment", {})
        self.assignment_type: str = assignment.get("type", "hash_mod")
        self.salt: str = assignment.get("salt", "")
        self.groups: list[dict[str, Any]] = assignment.get("groups", [])
        self.metrics: list[str] = config.get("metrics", [])
        self.created_at: str = config.get("created_at", "")

        # 校验 traffic 总和
        total = sum(g.get("traffic", 0) for g in self.groups)
        if total != 100:
            raise ValueError(
                f"实验 '{self.experiment_id}' 各组 traffic 之和须为 100，当前为 {total}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "unit": self.unit,
            "description": self.description,
            "assignment": {
                "type": self.assignment_type,
                "salt": self.salt,
                "groups": self.groups,
            },
            "metrics": self.metrics,
            "created_at": self.created_at,
        }


# 默认实验配置（第一版）
_DEFAULT_EXPERIMENTS = [
    {
        "experiment_id": "exp_dnn_baseline_vs_tuned",
        "status": "active",
        "unit": "user_id",
        "description": "DNN baseline vs DNN tuned 对比实验",
        "assignment": {
            "type": "hash_mod",
            "salt": "202605_ab_v1",
            "groups": [
                {"name": "A", "traffic": 50, "model_key": "dnn_baseline"},
                {"name": "B", "traffic": 50, "model_key": "dnn_tuned"},
            ],
        },
        "metrics": ["avg_score", "top_k_diversity", "coverage"],
        "created_at": "2026-05-18T00:00:00",
    },
]


class ABAssigner:
    """用户分组分配器。

    职责：
    - 管理实验配置列表。
    - 根据 user_id / request_id 和实验配置计算分组（纯函数，不存储状态）。
    """

    def __init__(self, experiments: Optional[list[dict[str, Any]]] = None):
        self._experiments: dict[str, ExperimentConfig] = {}
        exp_list = experiments if experiments is not None else _DEFAULT_EXPERIMENTS
        for exp_config in exp_list:
            exp = ExperimentConfig(exp_config)
            self._experiments[exp.experiment_id] = exp

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentConfig]:
        return self._experiments.get(experiment_id)

    def list_experiments(self) -> list[ExperimentConfig]:
        return list(self._experiments.values())

    def assign(
        self,
        experiment_id: str,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """执行分组分配。

        Args:
            experiment_id: 实验标识。
            user_id: 用户标识（优先使用）。
            request_id: 请求标识（user_id 缺失时 fallback）。

        Returns:
            包含 group、model_key、assignment_strategy 等信息的 dict。

        Raises:
            ValueError: 实验不存在、非 active、或标识均缺失。
        """
        experiment = self._experiments.get(experiment_id)
        if experiment is None:
            raise ValueError(f"experiment_id '{experiment_id}' 不存在")

        if experiment.status != "active":
            raise ValueError(
                f"实验 '{experiment_id}' 当前状态为 '{experiment.status}'，非 active"
            )

        # 确定分组单元标识
        unit_id = user_id or request_id
        if not unit_id:
            raise ValueError("user_id 和 request_id 均缺失，无法分组")

        if experiment.assignment_type == "hash_mod":
            return self._hash_assign(unit_id, experiment)
        elif experiment.assignment_type == "random":
            return self._random_assign(experiment)
        else:
            raise ValueError(f"不支持的分组策略: {experiment.assignment_type}")

    def _hash_assign(self, unit_id: str, experiment: ExperimentConfig) -> dict[str, Any]:
        hash_input = unit_id + experiment.experiment_id + experiment.salt
        hash_hex = hashlib.md5(hash_input.encode()).hexdigest()
        hash_val = int(hash_hex, 16) % 100

        cumulative = 0
        for group in experiment.groups:
            cumulative += group["traffic"]
            if hash_val < cumulative:
                bucket_start = cumulative - group["traffic"]
                return {
                    "group": group["name"],
                    "model_key": group["model_key"],
                    "assignment_strategy": "hash_mod",
                    "hash_value": hash_val,
                    "traffic_bucket": f"[{bucket_start}, {cumulative - 1}]",
                }

        # fallback（不应到达此处）
        return {
            "group": experiment.groups[-1]["name"],
            "model_key": experiment.groups[-1]["model_key"],
            "assignment_strategy": "hash_mod",
            "hash_value": hash_val,
            "traffic_bucket": "fallback",
        }

    def _random_assign(self, experiment: ExperimentConfig) -> dict[str, Any]:
        r = random.randint(0, 99)
        cumulative = 0
        for group in experiment.groups:
            cumulative += group["traffic"]
            if r < cumulative:
                return {
                    "group": group["name"],
                    "model_key": group["model_key"],
                    "assignment_strategy": "random",
                    "random_value": r,
                }
        return {
            "group": experiment.groups[-1]["name"],
            "model_key": experiment.groups[-1]["model_key"],
            "assignment_strategy": "random",
        }