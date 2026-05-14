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
from src.serving.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    PredictResult,
    RankRequest,
    RankResponse,
    RankResult,
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
        predictor = Predictor(model_dir=str(model_dir), device=device)
        predictor.load()
        app.state.predictor = predictor
        app.state.loaded_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        print(f"模型加载成功: {model_dir}")
        yield
        app.state.predictor = None

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
    print(f"    POST /predict")
    print(f"    POST /rank")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()