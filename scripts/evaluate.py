"""
统一评估多个 PM2.5 预测模型的脚本。

使用方式:
    python scripts/evaluate.py --models GMAN_PDFusion,GRU,ARIMA \
        --checkpoints GMAN_PDFusion=path/to/best_model.pt,GRU=...,ARIMA=...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

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
    GMANPDFusion,
    MLPBaseline,
)
from My.utils.data_pm25 import PM25DatasetConfig, load_pm25_dataloaders  # noqa: E402
from My.utils.metrics import mae, rmse, rse, corr  # noqa: E402


def load_config(path: Path | None) -> Dict[str, Any]:
    default = {
        "num_pred": 12,
        "history_len": 24,
        "batch_size": 32,
        "feature_idx": 0,
    }
    if path is None or not path.exists():
        return default
    try:
        import yaml  # type: ignore
    except ImportError:
        print("未安装 PyYAML，使用默认配置。")
        return default
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return {**default, **loaded}


def parse_checkpoints(spec: str | None) -> Mapping[str, Path]:
    mapping: Dict[str, Path] = {}
    if not spec:
        return mapping
    for item in spec.split(","):
        if not item or "=" not in item:
            continue
        name, path = item.split("=", 1)
        mapping[name.strip().upper()] = Path(path.strip())
    return mapping


def inverse_transform_sequence(scaler, array: np.ndarray) -> np.ndarray:
    if scaler is None:
        return array
    shape = array.shape
    restored = scaler.inverse_transform(array.reshape(-1, shape[-1]))
    return restored.reshape(shape)


def compute_metrics(preds: np.ndarray, target: np.ndarray, mask: np.ndarray, scaler) -> Dict[str, float]:
    preds_unscaled = inverse_transform_sequence(scaler, preds)
    target_unscaled = inverse_transform_sequence(scaler, target)
    valid = 1.0 - mask
    preds_unscaled = preds_unscaled * valid + target_unscaled * (1.0 - valid)

    preds_tensor = torch.from_numpy(preds_unscaled)
    target_tensor = torch.from_numpy(target_unscaled)
    return {
        "MAE": float(mae(preds_tensor, target_tensor)),
        "RMSE": float(rmse(preds_tensor, target_tensor)),
        "RSE": float(rse(preds_tensor, target_tensor)),
        "CORR": float(corr(preds_tensor, target_tensor)),
    }


def evaluate_torch_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    scaler,
    max_batches: int,
) -> Dict[str, float]:
    model.eval()
    preds_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []
    masks_list: List[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            mask_y = batch["mask_y"].to(device)
            if hasattr(model, "__call__"):
                preds = model(X)
            else:
                raise RuntimeError("模型缺少前向接口。")

            preds_list.append(preds.detach().cpu().numpy())
            targets_list.append(y.detach().cpu().numpy())
            masks_list.append(mask_y.detach().cpu().numpy())

            if 0 < max_batches <= batch_idx + 1:
                break

    if not preds_list:
        return {"MAE": float("nan"), "RMSE": float("nan"), "RSE": float("nan"), "CORR": float("nan")}

    preds_np = np.concatenate(preds_list, axis=0)
    target_np = np.concatenate(targets_list, axis=0)
    mask_np = np.concatenate(masks_list, axis=0)
    return compute_metrics(preds_np, target_np, mask_np, scaler)


def evaluate_traditional_model(model, loader, scaler, max_batches: int) -> Dict[str, float]:
    preds_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []
    masks_list: List[np.ndarray] = []

    for batch_idx, batch in enumerate(loader):
        X = batch["X"].detach().cpu().numpy()
        y = batch["y"].detach().cpu().numpy()
        mask_y = batch["mask_y"].detach().cpu().numpy()
        preds = model.predict(X)
        preds_list.append(preds)
        targets_list.append(y)
        masks_list.append(mask_y)
        if 0 < max_batches <= batch_idx + 1:
            break

    if not preds_list:
        return {"MAE": float("nan"), "RMSE": float("nan"), "RSE": float("nan"), "CORR": float("nan")}

    preds_np = np.concatenate(preds_list, axis=0)
    target_np = np.concatenate(targets_list, axis=0)
    mask_np = np.concatenate(masks_list, axis=0)
    return compute_metrics(preds_np, target_np, mask_np, scaler)


def main() -> None:
    parser = argparse.ArgumentParser(description="PM2.5 模型统一评估脚本")
    parser.add_argument("--models", type=str, required=True, help="逗号分隔的模型名称列表")
    parser.add_argument("--checkpoints", type=str, default=None, help="模型与 checkpoint 的映射，格式 Model=Path,Model2=Path2")
    parser.add_argument("--config", type=Path, default=Path("My/config/pm25.yaml"), help="配置文件路径")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test", help="评估数据集划分")
    parser.add_argument("--device", type=str, default=None, help="神经网络模型使用的设备")
    parser.add_argument("--max-batches", type=int, default=0, help="限制评估批次数，0 表示使用全部数据")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_cfg = PM25DatasetConfig(
        data_dir=Path("My/dataset"),
        history_len=int(cfg["history_len"]),
        pred_len=int(cfg["num_pred"]),
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg.get("num_workers", 0)),
        feature_idx=int(cfg.get("feature_idx", 0)),
    )
    train_loader, val_loader, test_loader, metadata = load_pm25_dataloaders(dataset_cfg)
    loader = val_loader if args.split == "val" else test_loader
    scaler = metadata.get("scaler")
    cfg["num_nodes"] = int(metadata["nodes"].shape[0])
    checkpoints = parse_checkpoints(args.checkpoints)

    model_names = [name.strip().upper() for name in args.models.split(",") if name.strip()]
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    for name in model_names:
        ckpt = checkpoints.get(name)
        if name == "GMAN_PDFUSION":
            if ckpt is None:
                print(f"[{name}] 缺少 checkpoint，跳过。")
                continue
            model = GMANPDFusion(cfg, num_nodes=cfg["num_nodes"]).to(device)
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model_state"] if "model_state" in state else state)
            metrics = evaluate_torch_model(model, loader, device, scaler, args.max_batches)
        elif name == "MLP":
            if ckpt is None:
                print(f"[{name}] 缺少 checkpoint，跳过。")
                continue
            model = MLPBaseline(cfg, num_nodes=cfg["num_nodes"]).to(device)
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model_state"] if "model_state" in state else state)
            metrics = evaluate_torch_model(model, loader, device, scaler, args.max_batches)
        elif name in {"GRU", "LSTM", "GCN"}:
            if ckpt is None:
                print(f"[{name}] 缺少 checkpoint，跳过。")
                continue
            if name == "GRU":
                model = GRUBaseline(cfg, num_nodes=cfg["num_nodes"]).to(device)
            elif name == "LSTM":
                model = LSTMBaseline(cfg, num_nodes=cfg["num_nodes"]).to(device)
            else:
                model = GCNBaseline(cfg, num_nodes=cfg["num_nodes"]).to(device)
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model_state"] if "model_state" in state else state)
            metrics = evaluate_torch_model(model, loader, device, scaler, args.max_batches)
        elif name in {"ARIMA", "SVR"}:
            if ckpt is None:
                print(f"[{name}] 缺少 checkpoint，跳过。")
                continue
            if name == "ARIMA":
                model = ARIMABaseline(cfg, num_nodes=cfg["num_nodes"], num_pred=int(cfg["num_pred"]))
            else:
                model = SVRBaseline(cfg, num_nodes=cfg["num_nodes"], num_pred=int(cfg["num_pred"]))
            model.load(ckpt)
            metrics = evaluate_traditional_model(model, loader, scaler, args.max_batches)
        else:
            print(f"[{name}] 不支持的模型，跳过。")
            continue

        print(
            f"{name} ({args.split}) -> "
            f"MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, "
            f"RSE={metrics['RSE']:.4f}, CORR={metrics['CORR']:.4f}"
        )


if __name__ == "__main__":
    main()
