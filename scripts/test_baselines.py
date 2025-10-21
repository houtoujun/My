"""
基线模型测试脚本。

从训练阶段生成的 checkpoint / joblib 文件加载模型，在指定数据集上评估。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from My.models import (  # noqa: E402
    ARIMABaseline,
    SVRBaseline,
    GRUBaseline,
    LSTMBaseline,
    GCNBaseline,
)
from My.utils.data_pm25 import PM25DatasetConfig, load_pm25_dataloaders  # noqa: E402
from My.utils.metrics import mae, rmse, rse, corr  # noqa: E402


def load_config(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        print("未安装 PyYAML，使用默认配置。")
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def inverse_transform_sequence(scaler, array: np.ndarray) -> np.ndarray:
    if scaler is None:
        return array
    shape = array.shape
    restored = scaler.inverse_transform(array.reshape(-1, shape[-1]))
    return restored.reshape(shape)


def compute_metrics_np(preds: np.ndarray, target: np.ndarray, mask: np.ndarray, scaler) -> Dict[str, float]:
    preds_inv = inverse_transform_sequence(scaler, preds)
    target_inv = inverse_transform_sequence(scaler, target)
    valid = 1.0 - mask
    preds_inv = preds_inv * valid + target_inv * (1.0 - valid)

    preds_tensor = torch.from_numpy(preds_inv)
    target_tensor = torch.from_numpy(target_inv)
    return {
        "mae": float(mae(preds_tensor, target_tensor)),
        "rmse": float(rmse(preds_tensor, target_tensor)),
        "rse": float(rse(preds_tensor, target_tensor)),
        "corr": float(corr(preds_tensor, target_tensor)),
    }


def evaluate_torch_model(model: torch.nn.Module, loader, device: torch.device, scaler) -> Dict[str, float]:
    model.eval()
    preds_list = []
    targets = []
    masks = []
    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            mask_y = batch["mask_y"].to(device)
            preds = model(X)
            preds_list.append(preds.detach().cpu())
            targets.append(y.detach().cpu())
            masks.append(mask_y.detach().cpu())
    if not preds_list:
        return {"mae": float("nan"), "rmse": float("nan"), "rse": float("nan"), "corr": float("nan")}
    preds_np = torch.cat(preds_list, dim=0).numpy()
    target_np = torch.cat(targets, dim=0).numpy()
    mask_np = torch.cat(masks, dim=0).numpy()
    return compute_metrics_np(preds_np, target_np, mask_np, scaler)


def evaluate_traditional_model(model, loader, scaler) -> Dict[str, float]:
    preds_list = []
    targets = []
    masks = []
    for batch in loader:
        X = batch["X"].detach().cpu().numpy()
        y = batch["y"].detach().cpu().numpy()
        mask_y = batch["mask_y"].detach().cpu().numpy()
        preds = model.predict(X)
        preds_list.append(preds)
        targets.append(y)
        masks.append(mask_y)
    if not preds_list:
        return {"mae": float("nan"), "rmse": float("nan"), "rse": float("nan"), "corr": float("nan")}
    preds_np = np.concatenate(preds_list, axis=0)
    target_np = np.concatenate(targets, axis=0)
    mask_np = np.concatenate(masks, axis=0)
    return compute_metrics_np(preds_np, target_np, mask_np, scaler)


def main() -> None:
    parser = argparse.ArgumentParser(description="PM2.5 基线模型测试脚本")
    parser.add_argument("--model", type=str, required=True, choices=["ARIMA", "SVR", "GRU", "LSTM", "GCN"])
    parser.add_argument("--checkpoint", type=Path, required=True, help="模型权重或 joblib 路径")
    parser.add_argument("--config", type=Path, default=Path("My/config/pm25.yaml"), help="配置文件路径")
    parser.add_argument("--device", type=str, default=None, help="神经网络模型使用的设备")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test", help="评估数据集划分")
    parser.add_argument("--output", type=Path, default=None, help="将指标另存为 JSON（可选）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_cfg = PM25DatasetConfig(
        data_dir=Path("My/dataset"),
        history_len=int(cfg.get("history_len", 24)),
        pred_len=int(cfg.get("num_pred", 12)),
        batch_size=int(cfg.get("batch_size", 32)),
        num_workers=int(cfg.get("num_workers", 0)),
        feature_idx=int(cfg.get("feature_idx", 0)),
    )

    train_loader, val_loader, test_loader, metadata = load_pm25_dataloaders(dataset_cfg)
    cfg["num_nodes"] = int(metadata["nodes"].shape[0])
    loader = val_loader if args.split == "val" else test_loader
    scaler = metadata.get("scaler")

    model_name = args.model.upper()
    if model_name in {"GRU", "LSTM", "GCN"}:
        device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        if model_name == "GRU":
            model = GRUBaseline(cfg, num_nodes=cfg["num_nodes"])
        elif model_name == "LSTM":
            model = LSTMBaseline(cfg, num_nodes=cfg["num_nodes"])
        else:
            model = GCNBaseline(cfg, num_nodes=cfg["num_nodes"])
        state = torch.load(args.checkpoint, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)
        model = model.to(device)
        metrics = evaluate_torch_model(model, loader, device, scaler)
    else:
        if model_name == "ARIMA":
            model = ARIMABaseline(cfg, num_nodes=cfg["num_nodes"], num_pred=int(cfg.get("num_pred", 12)))
        else:
            model = SVRBaseline(cfg, num_nodes=cfg["num_nodes"], num_pred=int(cfg.get("num_pred", 12)))
        model.load(args.checkpoint)
        metrics = evaluate_traditional_model(model, loader, scaler)

    print(
        f"{model_name} ({args.split}) -> MAE={metrics['mae']:.4f}, "
        f"RMSE={metrics['rmse']:.4f}, RSE={metrics['rse']:.4f}, CORR={metrics['corr']:.4f}"
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

