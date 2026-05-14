"""DNN batch inference predictor.

第一版只支持 DNN 模型 + tabular CSV 输入。
核心逻辑可被 batch CLI 和 REST API 复用。
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.dnn.model import DNNModel

# 泄漏字段列表 — 推理时应排除（不参与特征构建）
LEAKAGE_COLS = {"digg_count", "comment_count", "share_count", "collect_count"}

# 审计/标识列 — 不参与模型推理，但可在输出中保留
ID_COLS = {"sample_id", "video_id", "author_id", "interaction_score", "label", "split"}


class Predictor:
    """模型批量推理预测器。

    职责：
    1. 加载已训练 DNN 模型（model.pt）和预处理状态（feature_config_used.json）
    2. 校验输入 CSV 的列完整性
    3. 特征对齐：数值 median 填充 + z-score 标准化，类别 vocab lookup
    4. 执行模型推理
    5. 按 score 降序排序并生成 rank

    第一版只支持 DNN 模型。
    """

    def __init__(self, model_dir: str, device: str | None = None):
        """
        Args:
            model_dir: 模型输出目录（含 model.pt, feature_config_used.json, run_meta.json）
            device: 推理设备，默认自动检测（优先 cuda）
        """
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: DNNModel | None = None
        self.feature_config: dict | None = None
        self.run_meta: dict | None = None

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def load(self) -> "Predictor":
        """加载模型权重和预处理配置。"""
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"模型目录不存在: {self.model_dir}")

        # 检查必需文件
        required = ["model.pt", "feature_config_used.json", "run_meta.json"]
        for fname in required:
            fpath = self.model_dir / fname
            if not fpath.is_file():
                raise FileNotFoundError(f"缺少必需文件: {fpath}")

        # 加载 feature_config
        with (self.model_dir / "feature_config_used.json").open("r", encoding="utf-8") as f:
            self.feature_config = json.load(f)

        # 加载 run_meta
        with (self.model_dir / "run_meta.json").open("r", encoding="utf-8") as f:
            self.run_meta = json.load(f)

        # 验证模型类型
        model_name = self.run_meta.get("model_name", "")
        if model_name != "dnn":
            raise ValueError(
                f"当前 Predictor 只支持 DNN 模型，但 model_name 为 '{model_name}'"
            )

        # 解析模型结构参数
        numeric_dim = len(self.feature_config.get("numeric_cols", []))
        cat_embed_dims_raw = self.feature_config.get("cat_embed_dims", [])
        cat_embed_dims = [tuple(d) for d in cat_embed_dims_raw]

        # 从 state_dict 检测 hidden_units
        model_path = self.model_dir / "model.pt"
        state_dict = torch.load(model_path, map_location=self.device)
        hidden_units = self._detect_hidden_units(state_dict)

        # 推理时 dropout = 0
        self.model = DNNModel(
            numeric_dim=numeric_dim,
            cat_embed_dims=cat_embed_dims,
            hidden_units=hidden_units,
            dropout=0.0,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        return self

    @staticmethod
    def _detect_hidden_units(state_dict: dict) -> list[int]:
        """从 state_dict 的 mlp 层权重中自动检测 hidden_units。"""
        keys = sorted(
            k for k in state_dict.keys()
            if k.startswith("mlp.") and k.endswith(".weight")
        )
        return [state_dict[k].shape[0] for k in keys]

    # ------------------------------------------------------------------
    # 输入校验
    # ------------------------------------------------------------------

    def validate_input(self, df: pd.DataFrame):
        """检查输入 DataFrame 是否包含模型所需的全部特征列。"""
        numeric_cols = self.feature_config.get("numeric_cols", [])
        categorical_cols = self.feature_config.get("categorical_cols", [])
        required = numeric_cols + categorical_cols

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"输入数据缺少以下必需特征列: {missing}")

        # 警告：泄漏字段
        found_leakage = [c for c in LEAKAGE_COLS if c in df.columns]
        if found_leakage:
            warnings.warn(
                f"输入包含泄漏字段（将自动排除不参与推理）: {found_leakage}"
            )

    # ------------------------------------------------------------------
    # 特征对齐
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        """特征对齐：列排序、缺失值填充、标准化、类别编码。

        Returns:
            (numeric_tensor, categorical_tensor)，形状分别为
            (N, numeric_dim) 和 (N, num_cat)
        """
        numeric_cols = self.feature_config.get("numeric_cols", [])
        categorical_cols = self.feature_config.get("categorical_cols", [])
        medians = self.feature_config.get("medians", {})
        scaler_mean = self.feature_config.get("scaler_mean", [])
        scaler_scale = self.feature_config.get("scaler_scale", [])
        cat_vocabs = self.feature_config.get("cat_vocabs", {})

        # ---- 数值特征 ----
        numeric_values = df[numeric_cols].values.copy()
        # 强制转为 float（防止 __MISSING__ 等字符串导致 dtype object）
        numeric_values = numeric_values.astype(np.float64)

        # 中位数填充 NaN
        for i, col in enumerate(numeric_cols):
            col_median = medians.get(col)
            if col_median is not None:
                col_data = numeric_values[:, i]
                nan_mask = np.isnan(col_data)
                if nan_mask.any():
                    numeric_values[nan_mask, i] = col_median

        # 检查残留 NaN
        if np.any(np.isnan(numeric_values)):
            nan_cols = [
                numeric_cols[i]
                for i in range(numeric_values.shape[1])
                if np.any(np.isnan(numeric_values[:, i]))
            ]
            raise ValueError(
                f"数值特征包含无法填充的 NaN（medians 缺失）: {nan_cols}"
            )

        # z-score 标准化
        if scaler_mean and scaler_scale:
            mean_arr = np.array(scaler_mean, dtype=np.float64)
            scale_arr = np.array(scaler_scale, dtype=np.float64)
            numeric_values = (numeric_values - mean_arr) / scale_arr

        # 检查 Inf
        if np.any(np.isinf(numeric_values)):
            raise ValueError("数值特征包含 Inf（可能由除零导致）")

        numeric_tensor = torch.tensor(numeric_values, dtype=torch.float32)

        # ---- 类别特征 ----
        if categorical_cols:
            cat_indices = []
            for col in categorical_cols:
                vocab = cat_vocabs.get(col, {})
                if not vocab:
                    raise ValueError(f"类别特征 '{col}' 的 vocab 为空或缺失")
                raw_values = df[col].fillna("__MISSING__").values
                unk_id = vocab.get("__UNK__", 0)
                indices = [vocab.get(str(v), unk_id) for v in raw_values]
                cat_indices.append(indices)
            cat_tensor = torch.tensor(cat_indices, dtype=torch.long).t()
        else:
            cat_tensor = torch.empty((len(df), 0), dtype=torch.long)

        return numeric_tensor, cat_tensor

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """对 DataFrame 执行模型推理，返回 score 数组 [0,1]。"""
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 .load()")

        self.validate_input(df)
        numeric_tensor, cat_tensor = self.transform(df)

        numeric_tensor = numeric_tensor.to(self.device)
        cat_tensor = cat_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(numeric_tensor, cat_tensor)
            scores = torch.sigmoid(logits).cpu().numpy().flatten()

        return scores

    # ------------------------------------------------------------------
    # 排序
    # ------------------------------------------------------------------

    @staticmethod
    def rank(pred_df: pd.DataFrame) -> pd.DataFrame:
        """按 score 降序排序并生成 rank（从 1 开始）。"""
        result = pred_df.sort_values("score", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)
        return result

    # ------------------------------------------------------------------
    # CSV 全流程
    # ------------------------------------------------------------------

    def predict_csv(
        self,
        input_path: str | os.PathLike,
        output_dir: str | os.PathLike | None = None,
        dry_run: bool = False,
    ) -> dict:
        """从 CSV 文件执行完整推理流程。

        Args:
            input_path: 输入 CSV 路径
            output_dir: 输出目录，默认自动生成
            dry_run: 仅验证输入，不执行推理

        Returns:
            meta_info: 推理元信息 dict
        """
        input_path = Path(input_path)
        if not input_path.is_file():
            raise FileNotFoundError(f"输入 CSV 不存在: {input_path}")

        # 读取 CSV
        df = pd.read_csv(input_path, encoding="utf-8-sig", skipinitialspace=True)
        if df.empty:
            raise ValueError("输入 CSV 为空")

        # 排除泄漏字段
        cols_to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        if dry_run:
            self.validate_input(df)
            num_rows = len(df)
            numeric_cols = self.feature_config.get("numeric_cols", [])
            categorical_cols = self.feature_config.get("categorical_cols", [])
            print(f"[Dry-run] 输入校验通过")
            print(f"  行数: {num_rows}")
            print(f"  数值特征: {len(numeric_cols)} 列")
            print(f"  类别特征: {len(categorical_cols)} 列")
            print(f"  总特征数: {len(numeric_cols) + len(categorical_cols)}")
            return {"dry_run": True, "num_rows": num_rows}

        # 推理
        scores = self.predict(df)

        # 构建输出 DataFrame
        result_df = pd.DataFrame({"score": scores})

        # 保留标识列
        for col in ["video_id", "sample_id", "author_id"]:
            if col in df.columns:
                result_df[col] = df[col].values

        # 保留 label（if present for comparison）
        label_col = self.feature_config.get("label_col", "label")
        if label_col in df.columns:
            result_df[label_col] = df[label_col].values

        # 排序
        result_df = self.rank(result_df)

        # 补充元信息列
        model_name = self.run_meta.get("model_name", "dnn")
        dataset_name = self.feature_config.get("dataset_name", "unknown")
        run_id = self.run_meta.get("run_id", "unknown")
        input_file = input_path.name
        inference_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        result_df["model_name"] = model_name
        result_df["dataset_name"] = dataset_name
        result_df["run_id"] = run_id
        result_df["input_file"] = input_file
        result_df["inference_time"] = inference_time

        # 列顺序
        output_cols = ["rank", "video_id", "score"]
        if label_col in result_df.columns:
            output_cols.append(label_col)
        for col in ["sample_id", "author_id"]:
            if col in result_df.columns:
                output_cols.append(col)
        output_cols.extend(["model_name", "dataset_name", "run_id", "input_file", "inference_time"])
        result_df = result_df[output_cols]

        # 确定输出目录
        if output_dir is None:
            inference_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = (
                Path("outputs") / "inference" / model_name / dataset_name / run_id / inference_run_id
            )
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存 predictions.csv
        csv_path = output_dir / "predictions.csv"
        result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"预测结果已保存: {csv_path}")

        # 保存 inference_meta.json
        meta = {
            "inference_run_id": output_dir.name,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "source_model_run_id": run_id,
            "model_dir": str(self.model_dir),
            "input_path": str(input_path),
            "output_path": str(csv_path),
            "num_input_rows": len(df),
            "num_scored_rows": len(result_df),
            "feature_count": len(self.feature_config.get("numeric_cols", []))
                            + len(self.feature_config.get("categorical_cols", [])),
            "device": self.device,
            "top_k": None,
            "created_at": inference_time,
        }
        meta_path = output_dir / "inference_meta.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"推理元信息已保存: {meta_path}")

        return meta