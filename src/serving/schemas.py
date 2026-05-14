"""REST API 请求/响应 Pydantic schema。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# 请求
# ------------------------------------------------------------------

class PredictRequest(BaseModel):
    """POST /predict 请求体。"""
    items: list[dict[str, Any]] = Field(
        ..., min_length=1, description="候选样本列表，每个元素为特征字典（需包含模型所有必需特征）"
    )


class RankRequest(BaseModel):
    """POST /rank 请求体。"""
    items: list[dict[str, Any]] = Field(
        ..., min_length=1, description="候选样本列表，每个元素为特征字典"
    )
    top_k: Optional[int] = Field(
        None, ge=1, description="返回 top-K 条排序结果（可选，默认返回全部）"
    )


# ------------------------------------------------------------------
# 结果条目
# ------------------------------------------------------------------

class PredictResult(BaseModel):
    """单条 predict 结果（原顺序，不排序）。"""
    video_id: Optional[str] = None
    score: float


class RankResult(BaseModel):
    """单条 rank 结果（按 score 降序）。"""
    rank: int = Field(..., ge=1, description="排序位置，从 1 开始")
    video_id: Optional[str] = None
    score: float


# ------------------------------------------------------------------
# 响应
# ------------------------------------------------------------------

class PredictResponse(BaseModel):
    """POST /predict 响应。"""
    model_name: str
    dataset_name: str
    run_id: str
    num_items: int
    results: list[PredictResult]


class RankResponse(BaseModel):
    """POST /rank 响应。"""
    model_name: str
    dataset_name: str
    run_id: str
    num_items: int
    results: list[RankResult]


class ModelInfoResponse(BaseModel):
    """GET /model-info 响应。"""
    model_name: str
    dataset_name: str
    run_id: str
    model_dir: str
    feature_count: int
    numeric_cols: int
    categorical_cols: int
    device: str
    loaded_at: str


class HealthResponse(BaseModel):
    """GET /health 响应。"""
    status: str
    model_loaded: bool
    timestamp: str


# ------------------------------------------------------------------
# 错误
# ------------------------------------------------------------------

class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail