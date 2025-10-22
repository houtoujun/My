"""
基线模型训练脚本。

支持的模型：
- 传统方法：ARIMA、SVR
- 神经网络：GRU、LSTM、GCN
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

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
    default = {
        "num_pred": 12,
        "history_len": 24,
        "batch_size": 32,
        "epochs": 30,
        "learning_rate": 1e-3,
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


def prepare_run_dir(model_name: str) -> Tuple[Path, logging.Logger]:
    run_root = Path("My/results/baselines") / model_name.lower()
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"train-{model_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(fmt)

    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return run_dir, logger


def masked_mae_loss(preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = 1.0 - mask
    denom = valid.sum().clamp_min(1.0)
    loss = (preds - target).abs() * valid
    return loss.sum() / denom


def inverse_transform_sequence(scaler, array: np.ndarray) -> np.ndarray:
    if scaler is None:
        return array
    shape = array.shape
    restored = scaler.inverse_transform(array.reshape(-1, shape[-1]))
    return restored.reshape(shape)


def compute_metrics_np(
    preds: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    scaler,
) -> Dict[str, float]:
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


def evaluate_torch_model(
    model: torch.nn.Module,
    loader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    scaler,
) -> Dict[str, float]:
    model.eval()
    losses = []
    preds_list = []
    target_list = []
    mask_list = []

    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            mask_y = batch["mask_y"].to(device)

            preds = model(X)
            loss = masked_mae_loss(preds, y, mask_y)
            losses.append(float(loss.item()))

            preds_list.append(preds.detach().cpu())
            target_list.append(y.detach().cpu())
            mask_list.append(mask_y.detach().cpu())

    if not preds_list:
        return {"loss": float("nan"), "mae": float("nan"), "rmse": float("nan"), "rse": float("nan"), "corr": float("nan")}

    preds_np = torch.cat(preds_list, dim=0).numpy()
    target_np = torch.cat(target_list, dim=0).numpy()
    mask_np = torch.cat(mask_list, dim=0).numpy()

    metrics = compute_metrics_np(preds_np, target_np, mask_np, scaler)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def evaluate_traditional_model(
    model,
    loader,
    scaler,
) -> Dict[str, float]:
    preds_list = []
    target_list = []
    mask_list = []

    for batch in loader:
        X = batch["X"].detach().cpu().numpy()
        y = batch["y"].detach().cpu().numpy()
        mask_y = batch["mask_y"].detach().cpu().numpy()
        preds = model.predict(X)
        preds_list.append(preds)
        target_list.append(y)
        mask_list.append(mask_y)

    if not preds_list:
        return {"mae": float("nan"), "rmse": float("nan"), "rse": float("nan"), "corr": float("nan")}

    preds_np = np.concatenate(preds_list, axis=0)
    target_np = np.concatenate(target_list, axis=0)
    mask_np = np.concatenate(mask_list, axis=0)
    return compute_metrics_np(preds_np, target_np, mask_np, scaler)


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    return value


def build_run_summary(
    *,
    model_name: str,
    run_dir: Path,
    cfg: Dict[str, Any],
    dataset_cfg: PM25DatasetConfig,
    metadata: Dict[str, Any],
    best_epoch: int | None,
    best_val: float,
    best_metrics: Dict[str, float] | None,
    train_size: int,
    val_size: int,
    test_size: int,
    checkpoint: str | None,
) -> Dict[str, Any]:
    dataset_dict = {k: _sanitize_for_json(v) for k, v in asdict(dataset_cfg).items()}
    config_dict = {k: _sanitize_for_json(v) for k, v in cfg.items()}
    nodes = metadata.get("nodes")
    num_nodes = int(nodes.shape[0]) if nodes is not None else int(cfg.get("num_nodes", 0))
    summary = {
        "model": model_name,
        "timestamp": run_dir.name,
        "config": config_dict,
        "dataset": dataset_dict,
        "num_nodes": num_nodes,
        "train_samples": int(train_size),
        "val_samples": int(val_size),
        "test_samples": int(test_size),
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val) if np.isfinite(best_val) else None,
        "best_val_metrics": _sanitize_for_json(best_metrics) if best_metrics else None,
        "checkpoint": checkpoint,
    }
    return summary


def collect_windows(loader) -> Tuple[np.ndarray, np.ndarray]:
    X_list = []
    y_list = []
    for batch in loader:
        X_list.append(batch["X"].detach().cpu().numpy())
        y_list.append(batch["y"].detach().cpu().numpy())
    if not X_list:
        raise RuntimeError("训练集中没有任何样本。")
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


def train_torch_model(
    model_name: str,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    test_loader,
    cfg: Dict[str, Any],
    metadata: Dict[str, Any],
    dataset_cfg: PM25DatasetConfig,
    args,
) -> None:
    run_dir, logger = prepare_run_dir(model_name)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)))

    scaler = metadata.get("scaler")
    best_val = float("inf")
    best_epoch: int | None = None
    best_metrics: Dict[str, float] | None = None
    best_path = run_dir / "best_model.pt"

    logger.info("Using device: %s", device)
    logger.info("Starting training %s, epochs=%s", model_name, cfg.get("epochs", args.epochs))

    # Save initial configuration snapshot
    pre_summary = build_run_summary(
        model_name=model_name,
        run_dir=run_dir,
        cfg=cfg,
        dataset_cfg=dataset_cfg,
        metadata=metadata,
        best_epoch=None,
        best_val=float("inf"),
        best_metrics=None,
        train_size=len(train_loader.dataset),
        val_size=len(val_loader.dataset),
        test_size=len(test_loader.dataset),
        checkpoint=None,
    )
    pre_summary["device"] = str(device)
    pre_summary["status"] = "in_progress"
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(pre_summary, f, ensure_ascii=False, indent=2)

    epochs = int(cfg.get("epochs", args.epochs))
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            optimizer.zero_grad()
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            mask_y = batch["mask_y"].to(device)

            preds = model(X)
            loss = masked_mae_loss(preds, y, mask_y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        avg_train = total_loss / max(steps, 1)
        val_metrics = evaluate_torch_model(model, val_loader, device, scaler)
        logger.info(
            "Epoch %d | train_loss=%.6f | val_loss=%.6f | val_mae=%.6f | val_rmse=%.6f | val_rse=%.6f | val_corr=%.6f",
            epoch + 1,
            avg_train,
            val_metrics["loss"],
            val_metrics["mae"],
            val_metrics["rmse"],
            val_metrics["rse"],
            val_metrics["corr"],
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch + 1
            best_metrics = val_metrics
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch + 1,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
            logger.info("Saved new best model: %s", best_path)

    logger.info("Training finished. Best val_loss=%.6f", best_val)

    summary = build_run_summary(
        model_name=model_name,
        run_dir=run_dir,
        cfg=cfg,
        dataset_cfg=dataset_cfg,
        metadata=metadata,
        best_epoch=best_epoch,
        best_val=best_val,
        best_metrics=best_metrics,
        train_size=len(train_loader.dataset),
        val_size=len(val_loader.dataset),
        test_size=len(test_loader.dataset),
        checkpoint=str(best_path.relative_to(run_dir)) if best_path.exists() else None,
    )
    summary["device"] = str(device)
    summary["status"] = "completed"

    if args.run_test and best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model_state"])
        test_metrics = evaluate_torch_model(model, test_loader, device, scaler)
        logger.info(
            "Test -> mae=%.6f rmse=%.6f rse=%.6f corr=%.6f",
            test_metrics["mae"],
            test_metrics["rmse"],
            test_metrics["rse"],
            test_metrics["corr"],
        )
        with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)
        summary["test_metrics"] = _sanitize_for_json(test_metrics)

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def train_traditional_model(
    model_name: str,
    model,
    train_loader,
    val_loader,
    test_loader,
    metadata: Dict[str, Any],
    cfg: Dict[str, Any],
    dataset_cfg: PM25DatasetConfig,
    args,
) -> None:
    run_dir, logger = prepare_run_dir(model_name)
    scaler = metadata.get("scaler")

    pre_summary = build_run_summary(
        model_name=model_name,
        run_dir=run_dir,
        cfg=cfg,
        dataset_cfg=dataset_cfg,
        metadata=metadata,
        best_epoch=None,
        best_val=float("inf"),
        best_metrics=None,
        train_size=len(train_loader.dataset),
        val_size=len(val_loader.dataset),
        test_size=len(test_loader.dataset),
        checkpoint=None,
    )
    pre_summary["device"] = args.device or "cpu"
    pre_summary["status"] = "in_progress"
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(pre_summary, f, ensure_ascii=False, indent=2)

    if model_name == "ARIMA":
        train_series = train_loader.dataset.series[: metadata["train_end"]]
        logger.info("Training ARIMA with %d time steps", train_series.shape[0])
        model.fit(train_series)
    else:
        X_train, y_train = collect_windows(train_loader)
        logger.info("Training SVR samples=%d window=%d", X_train.shape[0], X_train.shape[1])
        model.fit(X_train, y_train)

    ckpt_path = run_dir / "model.joblib"
    model.save(ckpt_path)
    logger.info("Model saved to %s", ckpt_path)

    val_metrics = evaluate_traditional_model(model, val_loader, scaler)
    logger.info(
        "Validation -> mae=%.6f rmse=%.6f rse=%.6f corr=%.6f",
        val_metrics["mae"],
        val_metrics["rmse"],
        val_metrics["rse"],
        val_metrics["corr"],
    )
    with (run_dir / "val_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(val_metrics, f, ensure_ascii=False, indent=2)

    test_metrics = None
    if args.run_test:
        test_metrics = evaluate_traditional_model(model, test_loader, scaler)
        logger.info(
            "Test -> mae=%.6f rmse=%.6f rse=%.6f corr=%.6f",
            test_metrics["mae"],
            test_metrics["rmse"],
            test_metrics["rse"],
            test_metrics["corr"],
        )
        with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    summary = build_run_summary(
        model_name=model_name,
        run_dir=run_dir,
        cfg=cfg,
        dataset_cfg=dataset_cfg,
        metadata=metadata,
        best_epoch=None,
        best_val=val_metrics.get("mae", float("nan")),
        best_metrics=val_metrics,
        train_size=len(train_loader.dataset),
        val_size=len(val_loader.dataset),
        test_size=len(test_loader.dataset),
        checkpoint=str(ckpt_path.relative_to(run_dir)) if ckpt_path.exists() else None,
    )
    summary["device"] = args.device or "cpu"
    summary["status"] = "completed"
    if test_metrics is not None:
        summary["test_metrics"] = _sanitize_for_json(test_metrics)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="PM2.5 基线模型训练脚本")
    parser.add_argument("--model", type=str, required=True, choices=["ARIMA", "SVR", "GRU", "LSTM", "GCN"])
    parser.add_argument("--config", type=Path, default=Path("My/config/pm25.yaml"), help="配置文件路径")
    parser.add_argument("--device", type=str, default=None, help="指定训练设备")
    parser.add_argument("--epochs", type=int, default=30, help="默认训练轮数，用于配置文件未指定时")
    parser.add_argument("--run-test", action="store_true", help="训练后在测试集评估")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_cfg = PM25DatasetConfig(
        data_dir=Path("My/dataset"),
        history_len=int(cfg["history_len"]),
        pred_len=int(cfg["num_pred"]),
        batch_size=int(cfg.get("batch_size", 32)),
        num_workers=int(cfg.get("num_workers", 0)),
        feature_idx=int(cfg.get("feature_idx", 0)),
    )

    train_loader, val_loader, test_loader, metadata = load_pm25_dataloaders(dataset_cfg)
    cfg["num_nodes"] = int(metadata["nodes"].shape[0])

    model_name = args.model.upper()
    if model_name == "GRU":
        model = GRUBaseline(cfg, num_nodes=cfg["num_nodes"])
        train_torch_model(model_name, model, train_loader, val_loader, test_loader, cfg, metadata, dataset_cfg, args)
    elif model_name == "LSTM":
        model = LSTMBaseline(cfg, num_nodes=cfg["num_nodes"])
        train_torch_model(model_name, model, train_loader, val_loader, test_loader, cfg, metadata, dataset_cfg, args)
    elif model_name == "GCN":
        model = GCNBaseline(cfg, num_nodes=cfg["num_nodes"])
        train_torch_model(model_name, model, train_loader, val_loader, test_loader, cfg, metadata, dataset_cfg, args)
    elif model_name == "ARIMA":
        model = ARIMABaseline(cfg, num_nodes=cfg["num_nodes"], num_pred=int(cfg["num_pred"]))
        train_traditional_model(model_name, model, train_loader, val_loader, test_loader, metadata, cfg, dataset_cfg, args)
    elif model_name == "SVR":
        model = SVRBaseline(cfg, num_nodes=cfg["num_nodes"], num_pred=int(cfg["num_pred"]))
        train_traditional_model(model_name, model, train_loader, val_loader, test_loader, metadata, cfg, dataset_cfg, args)
    else:
        raise ValueError(f"未知模型 {model_name}")


if __name__ == "__main__":
    main()
