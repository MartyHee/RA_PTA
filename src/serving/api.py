"""DNN REST API 推理服务。

第一版只支持 DNN + tabular JSON 输入。
复用 src/inference/predictor.py 的特征对齐和模型推理逻辑。

使用示例：
    D:/CodeData/software/Anaconda/Anaconda3/envs/ra/python.exe src/serving/api.py ^
        --model dnn ^
        --dataset real_raw_5000 ^
        --run-id 202605132017 ^
        --host 127.0.0.1 ^
        --port 8000
"""

from __future__ import annotations

import argparse
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import pandas as pd

# 将项目根目录加入 sys.path，使 src 可作为模块导入
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from starlette.exceptions import HTTPException as StarletteHTTPException

from src.inference.predictor import Predictor
from src.serving.ab_assigner import ABAssigner
from src.serving.model_registry import ModelRegistry
from src.serving.request_logger import RequestLogger
from src.serving.schemas import (
    ABRecommendRequest,
    ABRecommendResponse,
    ExperimentInfo,
    ExperimentsListResponse,
    HealthResponse,
    ModelInfo,
    ModelInfoResponse,
    ModelsListResponse,
    PredictRequest,
    PredictResponse,
    PredictResult,
    RankRequest,
    RankResponse,
    RankResult,
    RecommendRequest,
    RecommendResult,
    RecommendResponse,
)


def resolve_model_dir(model: str, dataset: str, run_id: str) -> Path:
    return Path("outputs") / model / dataset / run_id


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RA_PTA REST API 推理服务")
    parser.add_argument("--model", required=True, help="模型名称（第一版只支持 dnn）")
    parser.add_argument("--dataset", required=True, help="数据集名称")
    parser.add_argument("--run-id", required=True, help="模型训练 run_id")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--device", default=None, help="推理设备（cuda / cpu）")
    return parser.parse_args(argv)


# ------------------------------------------------------------------
# FastAPI app factory
# ------------------------------------------------------------------

def create_app(model: str, dataset: str, run_id: str,
               device: str | None = None) -> FastAPI:
    """创建 FastAPI app，在 lifespan 中加载 Predictor 并注入 app.state。"""

    if model != "dnn":
        raise ValueError(f"当前版本只支持 --model dnn，收到: {model}")

    model_dir = resolve_model_dir(model, dataset, run_id)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 初始化 ModelRegistry，注册并加载所有预定义模型
        registry = ModelRegistry.register_defaults(
            project_root=PROJECT_ROOT,
            device=device,
        )
        registry.load_all()
        app.state.registry = registry

        # 向后兼容：将 dnn_baseline 的 Predictor 设为 app.state.predictor
        dnn_entry = registry.get_model("dnn_baseline")
        if dnn_entry and dnn_entry.status == "loaded":
            app.state.predictor = dnn_entry.predictor
            app.state.loaded_at = dnn_entry.loaded_at
            print(f"dnn_baseline 加载成功: {dnn_entry.model_dir}")
        else:
            app.state.predictor = None
            app.state.loaded_at = None
            err = dnn_entry.error_message if dnn_entry else "dnn_baseline 未注册"
            print(f"dnn_baseline 加载失败: {err}")

        # 初始化请求日志记录器
        app.state.logger = RequestLogger()
        print(f"在线模拟日志目录: {app.state.logger.output_dir}")

        # 初始化 A/B 分配器
        app.state.ab_assigner = ABAssigner()

        yield

        app.state.registry = None
        app.state.predictor = None
        app.state.logger = None
        app.state.ab_assigner = None

    app = FastAPI(title="RA_PTA Inference API", lifespan=lifespan)

    # ------------------------------------------------------------------
    # 异常处理器
    # ------------------------------------------------------------------

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request,
                                       exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "INVALID_REQUEST",
                               "message": str(exc.errors())}},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request,
                                     exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": "HTTP_ERROR",
                               "message": exc.detail}},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "INTERNAL_ERROR",
                               "message": str(exc)}},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _items_to_df(items: list[dict[str, Any]]) -> tuple[pd.DataFrame, list]:
        """将 items 列表转为 DataFrame，同时提取 video_ids（统一转 str）。"""
        video_ids = []
        cleaned = []
        for item in items:
            vid = item.get("video_id")
            video_ids.append(str(vid) if vid is not None else None)
            cleaned.append(item)
        df = pd.DataFrame(cleaned)
        return df, video_ids

    def _get_model_meta(predictor: Predictor) -> tuple[str, str, str]:
        rm = predictor.run_meta or {}
        fc = predictor.feature_config or {}
        model_name = rm.get("model_name", "dnn")
        dataset_name = rm.get("dataset_name",
                              fc.get("dataset_name", "unknown"))
        run_id_val = rm.get("run_id", "unknown")
        return model_name, dataset_name, run_id_val  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # 路由
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health():
        predictor: Predictor | None = getattr(app.state, "predictor", None)
        return HealthResponse(
            status="ok",
            model_loaded=predictor is not None,
            timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        )

    @app.get("/model-info", response_model=ModelInfoResponse)
    async def model_info():
        predictor: Predictor | None = getattr(app.state, "predictor", None)
        if predictor is None:
            return JSONResponse(
                status_code=503,
                content={"error": {"code": "MODEL_NOT_LOADED",
                                   "message": "模型未加载"}},
            )

        fc = predictor.feature_config or {}
        numeric_count = len(fc.get("numeric_cols", []))
        categorical_count = len(fc.get("categorical_cols", []))
        model_name, dataset_name, run_id_val = _get_model_meta(predictor)

        return ModelInfoResponse(
            model_name=model_name,
            dataset_name=dataset_name,
            run_id=run_id_val,
            model_dir=str(predictor.model_dir),
            feature_count=numeric_count + categorical_count,
            numeric_cols=numeric_count,
            categorical_cols=categorical_count,
            device=predictor.device,
            loaded_at=getattr(app.state, "loaded_at", ""),
        )

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest):
        predictor: Predictor | None = getattr(app.state, "predictor", None)
        if predictor is None:
            return JSONResponse(
                status_code=503,
                content={"error": {"code": "MODEL_NOT_LOADED",
                                   "message": "模型未加载"}},
            )

        df, video_ids = _items_to_df(req.items)
        try:
            scores = predictor.predict(df)
        except ValueError as e:
            return JSONResponse(
                status_code=422,
                content={"error": {"code": "MISSING_FEATURE",
                                   "message": str(e)}},
            )

        results = [
            PredictResult(video_id=vid, score=float(s))
            for vid, s in zip(video_ids, scores)
        ]

        model_name, dataset_name, run_id_val = _get_model_meta(predictor)
        return PredictResponse(
            model_name=model_name,
            dataset_name=dataset_name,
            run_id=run_id_val,
            num_items=len(results),
            results=results,
        )

    @app.post("/rank", response_model=RankResponse)
    async def rank(req: RankRequest):
        predictor: Predictor | None = getattr(app.state, "predictor", None)
        if predictor is None:
            return JSONResponse(
                status_code=503,
                content={"error": {"code": "MODEL_NOT_LOADED",
                                   "message": "模型未加载"}},
            )

        df, video_ids = _items_to_df(req.items)
        try:
            scores = predictor.predict(df)
        except ValueError as e:
            return JSONResponse(
                status_code=422,
                content={"error": {"code": "MISSING_FEATURE",
                                   "message": str(e)}},
            )

        # 构建结果并按 score 降序排序
        raw = [
            {"video_id": vid, "score": float(s)}
            for vid, s in zip(video_ids, scores)
        ]
        raw.sort(key=lambda x: x["score"], reverse=True)

        # top_k 截断
        k = req.top_k
        if k is not None:
            raw = raw[:k]

        results = [
            RankResult(rank=i + 1, video_id=r["video_id"],
                       score=r["score"])
            for i, r in enumerate(raw)
        ]

        model_name, dataset_name, run_id_val = _get_model_meta(predictor)
        return RankResponse(
            model_name=model_name,
            dataset_name=dataset_name,
            run_id=run_id_val,
            num_items=len(results),
            results=results,
        )

    # ------------------------------------------------------------------
    # 新增：Online Simulation 路由
    # ------------------------------------------------------------------

    @app.get("/models", response_model=ModelsListResponse)
    async def list_models():
        registry: ModelRegistry | None = getattr(app.state, "registry", None)
        if registry is None:
            return JSONResponse(
                status_code=503,
                content={"error": {"code": "SERVICE_NOT_READY",
                                   "message": "服务未就绪"}},
            )
        entries = registry.list_models()
        models = []
        for entry in entries:
            models.append(ModelInfo(
                model_key=entry.model_key,
                model_name=entry.model_name,
                dataset_name=entry.dataset_name,
                run_id=entry.run_id,
                status=entry.status,
                inference_supported=entry.inference_supported,
                auto_load=entry.auto_load,
                metrics_summary=entry.metrics_summary,
                loaded_at=entry.loaded_at,
                error_message=entry.error_message,
            ))
        return ModelsListResponse(models=models)

    # ------------------------------------------------------------------

    class _ServiceError(Exception):
        def __init__(self, status_code: int, code: str, message: str):
            self.status_code = status_code
            self.code = code
            self.message = message

    @app.exception_handler(_ServiceError)
    async def service_error_handler(request: Request, exc: _ServiceError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    def _predict_items(
        registry: ModelRegistry,
        model_key: str,
        items: list[dict[str, Any]],
    ):
        """通用推荐逻辑：获取模型、预测、返回 (entry, scores, video_ids)。"""
        entry = registry.get_model(model_key)
        if entry is None:
            raise _ServiceError(404, "MODEL_KEY_NOT_FOUND",
                                f"model_key '{model_key}' 未在 registry 中注册")
        if entry.status != "loaded":
            raise _ServiceError(503, "MODEL_NOT_LOADED",
                                f"模型 '{model_key}' 状态为 '{entry.status}'，未加载")
        if not entry.inference_supported:
            raise _ServiceError(503, "INFERENCE_NOT_SUPPORTED",
                                f"模型 '{model_key}' 不支持推理")

        df, video_ids = _items_to_df(items)
        try:
            scores = entry.predictor.predict(df)
        except ValueError as e:
            raise _ServiceError(422, "MISSING_FEATURE", str(e))
        except Exception as e:
            raise _ServiceError(500, "PREDICTION_ERROR", str(e))

        return entry, scores, video_ids

    # ------------------------------------------------------------------

    @app.post("/recommend", response_model=RecommendResponse)
    async def recommend(req: RecommendRequest):
        registry: ModelRegistry | None = getattr(app.state, "registry", None)
        logger_inst: RequestLogger | None = getattr(app.state, "logger", None)

        if registry is None:
            return JSONResponse(
                status_code=503,
                content={"error": {"code": "SERVICE_NOT_READY",
                                   "message": "服务未就绪"}},
            )

        start_time = time.time()
        try:
            entry, scores, video_ids = _predict_items(registry, req.model_key, req.items)
        except _ServiceError as e:
            latency = (time.time() - start_time) * 1000
            if logger_inst:
                logger_inst.log_request(req.request_id, req.user_id, "/recommend",
                                         len(req.items), req.top_k, "error", latency, e.code)
            return JSONResponse(
                status_code=e.status_code,
                content={"error": {"code": e.code, "message": e.message}},
            )

        latency = (time.time() - start_time) * 1000

        # 排序 + top_k 截断
        raw = [
            {"video_id": vid, "score": float(s)}
            for vid, s in zip(video_ids, scores)
        ]
        raw.sort(key=lambda x: x["score"], reverse=True)
        k = req.top_k
        if k is not None:
            raw = raw[:k]

        results = [
            RecommendResult(rank=i + 1, video_id=r["video_id"], score=r["score"])
            for i, r in enumerate(raw)
        ]

        # 日志
        if logger_inst:
            logger_inst.log_request(req.request_id, req.user_id, "/recommend",
                                     len(req.items), req.top_k, "ok", latency)
            for r in results:
                logger_inst.log_recommendation(
                    request_id=req.request_id,
                    user_id=req.user_id,
                    model_key=req.model_key,
                    video_id=r.video_id,
                    rank=r.rank,
                    score=r.score,
                )

        return RecommendResponse(
            request_id=req.request_id,
            user_id=req.user_id,
            model_key=req.model_key,
            model_name=entry.model_name,
            dataset_name=entry.dataset_name,
            run_id=entry.run_id,
            num_items=len(results),
            top_k=req.top_k,
            results=results,
        )

    # ------------------------------------------------------------------

    @app.post("/ab/recommend", response_model=ABRecommendResponse)
    async def ab_recommend(req: ABRecommendRequest):
        registry: ModelRegistry | None = getattr(app.state, "registry", None)
        assigner: ABAssigner | None = getattr(app.state, "ab_assigner", None)
        logger_inst: RequestLogger | None = getattr(app.state, "logger", None)

        if registry is None or assigner is None:
            return JSONResponse(
                status_code=503,
                content={"error": {"code": "SERVICE_NOT_READY",
                                   "message": "服务未就绪"}},
            )

        # 1. 校验 experiment_id
        experiment = assigner.get_experiment(req.experiment_id)
        if experiment is None:
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "EXPERIMENT_NOT_FOUND",
                                   "message": f"experiment_id '{req.experiment_id}' 不存在"}},
            )

        # 2. 校验实验状态
        if experiment.status != "active":
            return JSONResponse(
                status_code=400,
                content={"error": {"code": "EXPERIMENT_NOT_ACTIVE",
                                   "message": f"实验 '{req.experiment_id}' 状态为 '{experiment.status}'"}},
            )

        # 3. 校验 user_id / request_id
        if not req.user_id and not req.request_id:
            return JSONResponse(
                status_code=400,
                content={"error": {"code": "MISSING_USER_ID",
                                   "message": "user_id 和 request_id 均缺失，至少需要提供一个"}},
            )

        start_time = time.time()

        # 4. 分组
        try:
            assignment = assigner.assign(req.experiment_id, req.user_id, req.request_id)
        except ValueError as e:
            return JSONResponse(
                status_code=400,
                content={"error": {"code": "ASSIGNMENT_ERROR",
                                   "message": str(e)}},
            )

        group = assignment["group"]
        model_key = assignment["model_key"]

        # 5. 获取模型并推理
        try:
            entry, scores, video_ids = _predict_items(registry, model_key, req.items)
        except _ServiceError as e:
            latency = (time.time() - start_time) * 1000
            if logger_inst:
                logger_inst.log_request(req.request_id, req.user_id, "/ab/recommend",
                                         len(req.items), req.top_k, "error", latency, e.code)
            return JSONResponse(
                status_code=e.status_code,
                content={"error": {"code": e.code, "message": e.message}},
            )

        latency = (time.time() - start_time) * 1000

        # 6. 排序 + top_k 截断
        raw = [
            {"video_id": vid, "score": float(s)}
            for vid, s in zip(video_ids, scores)
        ]
        raw.sort(key=lambda x: x["score"], reverse=True)
        k = req.top_k
        if k is not None:
            raw = raw[:k]

        results = [
            RecommendResult(rank=i + 1, video_id=r["video_id"], score=r["score"])
            for i, r in enumerate(raw)
        ]

        # 7. 日志
        if logger_inst:
            logger_inst.log_request(req.request_id, req.user_id, "/ab/recommend",
                                     len(req.items), req.top_k, "ok", latency)
            logger_inst.log_assignment(
                experiment_id=req.experiment_id,
                request_id=req.request_id,
                user_id=req.user_id,
                group=group,
                model_key=model_key,
                assignment_strategy=assignment.get("assignment_strategy", "hash_mod"),
                hash_value=assignment.get("hash_value"),
                traffic_bucket=assignment.get("traffic_bucket"),
            )
            for r in results:
                logger_inst.log_recommendation(
                    request_id=req.request_id,
                    user_id=req.user_id,
                    model_key=model_key,
                    video_id=r.video_id,
                    rank=r.rank,
                    score=r.score,
                    experiment_id=req.experiment_id,
                    group=group,
                )

        assignment_reason = assignment.get("assignment_strategy", "hash_mod")

        return ABRecommendResponse(
            experiment_id=req.experiment_id,
            user_id=req.user_id,
            request_id=req.request_id,
            group=group,
            model_key=model_key,
            model_name=entry.model_name,
            dataset_name=entry.dataset_name,
            run_id=entry.run_id,
            assignment_reason=assignment_reason,
            num_items=len(results),
            top_k=req.top_k,
            results=results,
        )

    return app


# ------------------------------------------------------------------
# CLI 入口
# ------------------------------------------------------------------

def main():
    args = parse_args()

    if args.model != "dnn":
        print(f"错误: 当前版本只支持 --model dnn，收到: {args.model}",
              file=sys.stderr)
        sys.exit(1)

    model_dir = resolve_model_dir(args.model, args.dataset, args.run_id)
    if not model_dir.is_dir():
        print(f"错误: 模型目录不存在: {model_dir}", file=sys.stderr)
        sys.exit(1)

    import uvicorn

    app = create_app(
        model=args.model,
        dataset=args.dataset,
        run_id=args.run_id,
        device=args.device,
    )

    print(f"RA_PTA 推理服务启动")
    print(f"  模型: {args.model}")
    print(f"  数据集: {args.dataset}")
    print(f"  Run ID: {args.run_id}")
    print(f"  地址: http://{args.host}:{args.port}")
    print(f"  路由:")
    print(f"    GET  /health")
    print(f"    GET  /model-info")
    print(f"    GET  /models")
    print(f"    POST /predict")
    print(f"    POST /rank")
    print(f"    POST /recommend")
    print(f"    POST /ab/recommend")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()