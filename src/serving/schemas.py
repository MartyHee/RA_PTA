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


# ------------------------------------------------------------------
# Online Simulation 相关 Schema
# ------------------------------------------------------------------

class ModelInfo(BaseModel):
    """单条模型注册信息。"""
    model_key: str
    model_name: str
    dataset_name: str
    run_id: str
    status: str
    inference_supported: bool
    auto_load: bool = False
    metrics_summary: dict[str, Any] = Field(default_factory=dict)
    loaded_at: Optional[str] = None
    error_message: Optional[str] = None


class ModelsListResponse(BaseModel):
    """GET /models 响应。"""
    models: list[ModelInfo]


class RecommendRequest(BaseModel):
    """POST /recommend 请求体。"""
    user_id: Optional[str] = Field(None, description="用户标识")
    request_id: Optional[str] = Field(None, description="请求标识")
    model_key: str = Field(..., description="模型标识（如 dnn_baseline）")
    items: list[dict[str, Any]] = Field(
        ..., min_length=1, description="候选样本列表"
    )
    top_k: Optional[int] = Field(None, ge=1, description="返回 top-K 条排序结果")


class RecommendResult(BaseModel):
    """单条推荐结果。"""
    rank: int = Field(..., ge=1, description="排序位置，从 1 开始")
    video_id: Optional[str] = None
    score: float


class RecommendResponse(BaseModel):
    """POST /recommend 响应。"""
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    model_key: str
    model_name: str
    dataset_name: str
    run_id: str
    num_items: int
    top_k: Optional[int] = None
    results: list[RecommendResult]


class ABRecommendRequest(BaseModel):
    """POST /ab/recommend 请求体。"""
    experiment_id: str = Field(..., description="实验标识")
    user_id: Optional[str] = Field(None, description="用户标识")
    request_id: Optional[str] = Field(None, description="请求标识")
    items: list[dict[str, Any]] = Field(
        ..., min_length=1, description="候选样本列表"
    )
    top_k: Optional[int] = Field(None, ge=1, description="返回 top-K 条排序结果")


class ABRecommendResponse(BaseModel):
    """POST /ab/recommend 响应。"""
    experiment_id: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    group: str
    model_key: str
    model_name: str
    dataset_name: str
    run_id: str
    assignment_reason: str
    num_items: int
    top_k: Optional[int] = None
    results: list[RecommendResult]


class ExperimentInfo(BaseModel):
    """实验配置信息。"""
    experiment_id: str
    status: str
    unit: str
    description: str = ""
    assignment: dict[str, Any]
    metrics: list[str] = Field(default_factory=list)
    created_at: str = ""


class ExperimentsListResponse(BaseModel):
    """GET /experiments 响应。"""
    experiments: list[ExperimentInfo]