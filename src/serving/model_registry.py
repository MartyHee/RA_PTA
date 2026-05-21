"""多模型注册与加载管理。

第一版支持 5 个已训练模型的注册与状态管理：
- dnn_baseline: DNN baseline，auto_load=true
- dnn_tuned: DNN tuned，auto_load 按 model.pt 是否存在决定
- wide_deep_baseline / graphsage_baseline / multimodal_baseline: 注册但 inference_supported=false
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    """单个模型在 registry 中的注册信息。"""
    model_key: str
    model_name: str
    dataset_name: str
    run_id: str
    model_dir: str
    status: str = "unavailable"       # loaded | unavailable | error
    inference_supported: bool = False
    auto_load: bool = False
    loaded_at: Optional[str] = None
    error_message: Optional[str] = None
    metrics_summary: dict[str, Any] = field(default_factory=dict)
    predictor: Any = None              # Predictor 实例（loaded 状态时可用）


class ModelRegistry:
    """多模型注册中心。

    职责：
    - 维护模型注册列表（model_key -> ModelEntry）。
    - 按需加载 / 卸载模型实例。
    - 提供模型查询和状态反馈。
    - 单模型加载失败不影响其他模型。
    """

    def __init__(self):
        self._models: dict[str, ModelEntry] = {}

    def register(self, entry: ModelEntry) -> None:
        self._models[entry.model_key] = entry

    def get_model(self, model_key: str) -> Optional[ModelEntry]:
        return self._models.get(model_key)

    def list_models(self) -> list[ModelEntry]:
        return list(self._models.values())

    def load_all(self) -> None:
        """遍历所有 auto_load=true 且 inference_supported=true 的模型并加载。"""
        for entry in self._models.values():
            if entry.auto_load and entry.inference_supported:
                self._load_model(entry)

    def _load_model(self, entry: ModelEntry) -> None:
        """加载单个模型，失败时标记 error 并记录原因。"""
        from src.inference.predictor import Predictor

        model_dir = Path(entry.model_dir)
        if not model_dir.is_dir():
            entry.status = "error"
            entry.error_message = f"模型目录不存在: {model_dir}"
            logger.warning("模型加载失败 [%s]: 目录不存在: %s", entry.model_key, model_dir)
            return

        try:
            predictor = Predictor(model_dir=str(model_dir))
            predictor.load()
            entry.predictor = predictor
            entry.status = "loaded"
            entry.loaded_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            logger.info("模型加载成功 [%s]: %s", entry.model_key, model_dir)
        except Exception as e:
            entry.status = "error"
            entry.error_message = str(e)
            logger.warning("模型加载失败 [%s]: %s", entry.model_key, e)

    @staticmethod
    def register_defaults(
        project_root: str | Path,
        device: str | None = None,
    ) -> ModelRegistry:
        """注册所有预定义模型并返回 ModelRegistry 实例。

        Args:
            project_root: 项目根目录，用于解析模型路径。
            device: 推理设备（传递给 Predictor）。

        Returns:
            已注册所有预定义模型的 ModelRegistry 实例。
        """
        root = Path(project_root)
        registry = ModelRegistry()

        def _load_metrics(model_name: str, dataset_name: str, run_id: str) -> dict:
            path = root / "outputs" / model_name / dataset_name / run_id / "metrics.json"
            if path.is_file():
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return {
                    "val_auc": data.get("val_metrics", {}).get("auc"),
                    "test_auc": data.get("test_metrics", {}).get("auc"),
                }
            return {}

        def _model_pt_exists(model_name: str, dataset_name: str, run_id: str) -> bool:
            return (root / "outputs" / model_name / dataset_name / run_id / "model.pt").is_file()

        # ---- dnn_baseline ----
        registry.register(ModelEntry(
            model_key="dnn_baseline",
            model_name="dnn",
            dataset_name="real_raw_5000",
            run_id="202605132017",
            model_dir=str(root / "outputs" / "dnn" / "real_raw_5000" / "202605132017"),
            inference_supported=True,
            auto_load=True,
            metrics_summary=_load_metrics("dnn", "real_raw_5000", "202605132017"),
        ))

        # ---- dnn_tuned ----
        dnn_tuned_dir = root / "outputs" / "dnn" / "real_raw_5000" / "202605141914"
        registry.register(ModelEntry(
            model_key="dnn_tuned",
            model_name="dnn",
            dataset_name="real_raw_5000",
            run_id="202605141914",
            model_dir=str(dnn_tuned_dir),
            inference_supported=True,
            auto_load=_model_pt_exists("dnn", "real_raw_5000", "202605141914"),
            metrics_summary=_load_metrics("dnn", "real_raw_5000", "202605141914"),
        ))

        # ---- wide_deep_baseline ----
        registry.register(ModelEntry(
            model_key="wide_deep_baseline",
            model_name="wide_deep",
            dataset_name="real_raw_5000",
            run_id="202605132026",
            model_dir=str(root / "outputs" / "wide_deep" / "real_raw_5000" / "202605132026"),
            inference_supported=False,
            auto_load=False,
            metrics_summary=_load_metrics("wide_deep", "real_raw_5000", "202605132026"),
        ))

        # ---- graphsage_baseline ----
        registry.register(ModelEntry(
            model_key="graphsage_baseline",
            model_name="graphsage",
            dataset_name="real_raw_5000",
            run_id="202605132107",
            model_dir=str(root / "outputs" / "graphsage" / "real_raw_5000" / "202605132107"),
            inference_supported=False,
            auto_load=False,
            metrics_summary=_load_metrics("graphsage", "real_raw_5000", "202605132107"),
        ))

        # ---- multimodal_baseline ----
        registry.register(ModelEntry(
            model_key="multimodal_baseline",
            model_name="multimodal",
            dataset_name="real_raw_5000",
            run_id="202605132210",
            model_dir=str(root / "outputs" / "multimodal" / "real_raw_5000" / "202605132210"),
            inference_supported=False,
            auto_load=False,
            metrics_summary=_load_metrics("multimodal", "real_raw_5000", "202605132210"),
        ))

        return registry