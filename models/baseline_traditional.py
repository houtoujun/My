"""
传统机器学习基线模型实现（ARIMA、SVR）。

这些模型主要用于与深度学习模型进行对比实验。接口设计为先调用
``fit`` 完成训练，再使用 ``predict`` 对批量历史序列输出未来多步预测结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings

import numpy as np

try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning  # type: ignore
except Exception:  # pragma: no cover - statsmodels 未安装时无需处理
    ConvergenceWarning = Warning  # type: ignore

class BaselineNotFittedError(RuntimeError):
    """在模型尚未训练时调用 predict 会抛出该异常。"""


@dataclass
class ARIMAConfig:
    order: Tuple[int, int, int] = (2, 1, 2)
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False
    maxiter: int = 200


class ARIMABaseline:
    """
    针对每个监测站点分别拟合独立 ARIMA 模型，并支持滑动窗口预测。
    """

    def __init__(self, config: Dict[str, Any], num_nodes: int, num_pred: int) -> None:
        self.num_nodes = int(num_nodes)
        self.num_pred = int(num_pred)
        self.history_len = int(config.get("history_len", 12))

        cfg = config.get("arima", {})
        self.arima_cfg = ARIMAConfig(
            order=tuple(cfg.get("order", (2, 1, 2))),
            seasonal_order=tuple(cfg.get("seasonal_order", (0, 0, 0, 0))),
            enforce_stationarity=bool(cfg.get("enforce_stationarity", False)),
            enforce_invertibility=bool(cfg.get("enforce_invertibility", False)),
            maxiter=int(cfg.get("maxiter", 200)),
        )
        self._results: List[Any] = []

    def fit(self, train_series: np.ndarray) -> None:
        """
        训练阶段仅依赖完整的训练集序列，形状为 [T, N]。
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA  # type: ignore
        except ImportError as exc:  # pragma: no cover - 依赖缺失时提示
            raise RuntimeError(
                "缺少 statsmodels 依赖，请先运行 `pip install statsmodels`。"
            ) from exc

        series = np.asarray(train_series, dtype=np.float32)
        if series.ndim != 2 or series.shape[1] != self.num_nodes:
            raise ValueError("train_series 需为 [T, N] 数组，并且 N 与 num_nodes 一致。")

        self._results = []
        for node_idx in range(self.num_nodes):
            endog = series[:, node_idx]
            model = ARIMA(
                endog=endog,
                order=self.arima_cfg.order,
                seasonal_order=self.arima_cfg.seasonal_order,
                enforce_stationarity=self.arima_cfg.enforce_stationarity,
                enforce_invertibility=self.arima_cfg.enforce_invertibility,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                fitted = model.fit(
                    method_kwargs={
                        "warn_convergence": False,
                        "maxiter": int(self.arima_cfg.maxiter),
                    }
                )
            if (
                hasattr(fitted, "mle_retvals")
                and isinstance(fitted.mle_retvals, dict)
                and not fitted.mle_retvals.get("converged", True)
            ):
                try:
                    fallback = model.fit(method="css")
                    fitted = fallback
                except Exception:
                    pass
            self._results.append(fitted)

    def predict(self, history: np.ndarray) -> np.ndarray:
        """
        根据最近 ``history_len`` 个时间步的序列预测未来 ``num_pred`` 步。

        参数:
            history: [B, history_len, N] 的浮点数组。
        返回:
            preds:  [B, num_pred, N] 的预测结果。
        """
        if not self._results:
            raise BaselineNotFittedError("ARIMABaseline 尚未训练。")

        history = np.asarray(history, dtype=np.float32)
        if history.ndim != 3 or history.shape[2] != self.num_nodes:
            raise ValueError("history 需为 [B, history_len, N] 数组。")

        batch, hist_len, _ = history.shape
        if hist_len != self.history_len:
            raise ValueError(
                f"history_len={hist_len} 与初始化时设置的 {self.history_len} 不一致。"
            )

        preds = np.zeros((batch, self.num_pred, self.num_nodes), dtype=np.float32)
        for node_idx, base_result in enumerate(self._results):
            for b in range(batch):
                window = history[b, :, node_idx]
                try:
                    applied = base_result.apply(window, refit=False)
                    forecast = applied.forecast(steps=self.num_pred)
                except Exception:  # pragma: no cover - 容错以免个别失败
                    # 若 apply 失败则回退到直接拟合一次以保证流程可继续
                    applied = base_result.model.fit(start_params=base_result.params)
                    forecast = applied.forecast(steps=self.num_pred)
                preds[b, :, node_idx] = np.asarray(forecast, dtype=np.float32)
        return preds

    def save(self, path: Path | str) -> None:
        from joblib import dump  # type: ignore

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_nodes": self.num_nodes,
            "num_pred": self.num_pred,
            "history_len": self.history_len,
            "arima_cfg": self.arima_cfg,
            "results": self._results,
        }
        dump(payload, path)

    def load(self, path: Path | str) -> None:
        from joblib import load  # type: ignore

        payload = load(Path(path))
        self.num_nodes = int(payload["num_nodes"])
        self.num_pred = int(payload["num_pred"])
        self.history_len = int(payload["history_len"])
        self.arima_cfg = payload["arima_cfg"]
        self._results = payload["results"]


@dataclass
class SVRConfig:
    kernel: str = "rbf"
    C: float = 1.0
    epsilon: float = 0.1
    gamma: str | float = "scale"


class SVRBaseline:
    """
    使用支持向量回归（Support Vector Regression）对每个节点、每个预测步建立模型。
    """

    def __init__(self, config: Dict[str, Any], num_nodes: int, num_pred: int) -> None:
        self.num_nodes = int(num_nodes)
        self.num_pred = int(num_pred)
        self.history_len = int(config.get("history_len", 12))

        cfg = config.get("svr", {})
        self.svr_cfg = SVRConfig(
            kernel=str(cfg.get("kernel", "rbf")),
            C=float(cfg.get("C", 1.0)),
            epsilon=float(cfg.get("epsilon", 0.1)),
            gamma=cfg.get("gamma", "scale"),
        )
        self._models: List[List[Any]] = []

    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> None:
        """
        参数:
            train_X: [M, history_len, N] 训练窗口。
            train_y: [M, num_pred, N]    对应未来序列。
        """
        try:
            from sklearn.svm import SVR  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "缺少 scikit-learn 依赖，请先运行 `pip install scikit-learn`。"
            ) from exc

        X = np.asarray(train_X, dtype=np.float32)
        y = np.asarray(train_y, dtype=np.float32)

        if X.ndim != 3 or X.shape[2] != self.num_nodes:
            raise ValueError("train_X 需为 [M, history_len, N] 数组。")
        if y.shape != (X.shape[0], self.num_pred, self.num_nodes):
            raise ValueError("train_y 需为 [M, num_pred, N]，且与 train_X 对应。")

        samples, hist_len, _ = X.shape
        if hist_len != self.history_len:
            raise ValueError(
                f"history_len={hist_len} 与初始化时设置的 {self.history_len} 不一致。"
            )

        self._models = []
        features = X.reshape(samples, hist_len, self.num_nodes)
        for node_idx in range(self.num_nodes):
            node_X = features[:, :, node_idx]
            node_models: List[Any] = []
            for step in range(self.num_pred):
                target = y[:, step, node_idx]
                reg = SVR(
                    kernel=self.svr_cfg.kernel,
                    C=self.svr_cfg.C,
                    epsilon=self.svr_cfg.epsilon,
                    gamma=self.svr_cfg.gamma,
                )
                reg.fit(node_X, target)
                node_models.append(reg)
            self._models.append(node_models)

    def predict(self, history: np.ndarray) -> np.ndarray:
        if not self._models:
            raise BaselineNotFittedError("SVRBaseline 尚未训练。")

        history = np.asarray(history, dtype=np.float32)
        if history.ndim != 3 or history.shape[2] != self.num_nodes:
            raise ValueError("history 需为 [B, history_len, N] 数组。")

        batch, hist_len, _ = history.shape
        if hist_len != self.history_len:
            raise ValueError(
                f"history_len={hist_len} 与初始化时设置的 {self.history_len} 不一致。"
            )

        preds = np.zeros((batch, self.num_pred, self.num_nodes), dtype=np.float32)
        for node_idx, node_models in enumerate(self._models):
            features = history[:, :, node_idx]
            for step, reg in enumerate(node_models):
                preds[:, step, node_idx] = reg.predict(features)
        return preds

    def save(self, path: Path | str) -> None:
        from joblib import dump  # type: ignore

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_nodes": self.num_nodes,
            "num_pred": self.num_pred,
            "history_len": self.history_len,
            "svr_cfg": self.svr_cfg,
            "models": self._models,
        }
        dump(payload, path)

    def load(self, path: Path | str) -> None:
        from joblib import load  # type: ignore

        payload = load(Path(path))
        self.num_nodes = int(payload["num_nodes"])
        self.num_pred = int(payload["num_pred"])
        self.history_len = int(payload["history_len"])
        self.svr_cfg = payload["svr_cfg"]
        self._models = payload["models"]


__all__ = ["ARIMABaseline", "SVRBaseline", "BaselineNotFittedError"]
