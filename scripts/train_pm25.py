"""
PM2.5 预测训练脚本。

- 自动加载配置与数据集，构建 GMAN_PDFusion 模型；
- 使用带缺失屏蔽的 MAE 作为训练损失，支持 GPU 训练；
- 输出训练/验证日志，并在验证集上保存最佳模型权重。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
from torch.cuda.amp import GradScaler, autocast
import inspect

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from My.models.hybrid_gman_pdformer_pd import GMANPDFusion
from My.utils.data_pm25 import PM25DatasetConfig, load_pm25_dataloaders
from My.utils.metrics import mae, rmse, rse, corr


def load_config(config_path: Path | None) -> Dict[str, Any]:
    """读取 YAML 配置，若缺失则返回默认参数。"""
    default = {
        "num_pred": 12,
        "history_len": 24,
        "batch_size": 16,
        "learning_rate": 5e-4,
        "epochs": 50,
        "feature_idx": 0,
        "K": 8,
        "d": 8,
        "L": 3,
        "bn_decay": 0.1,
        "geo_ratio": 0.6,
        "S": 6,
        "num_patterns": 16,
        "time_steps_per_day": 24,
        "spatial_chunk_size": 64,
        "pattern_path": "My/artifacts/pattern_keys.npy",
        "geo_mask_path": "My/artifacts/geo_mask.npy",
        "sem_mask_path": "My/artifacts/sem_mask.npy",
        "geo_neighbors_path": "My/artifacts/geo_neighbors.npz",
        "sem_neighbors_path": "My/artifacts/sem_neighbors.npz",
    }
    if config_path is None or not config_path.exists():
        return default
    try:
        import yaml  # type: ignore
    except ImportError:
        print("未安装 PyYAML，使用默认配置。")
        return default
    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return {**default, **loaded}


def prepare_run_dir() -> tuple[Path, logging.Logger]:
    run_root = Path("My/results/runs")
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return run_dir, logger


def masked_mae_loss(preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """计算带缺失掩码的 MAE，原始 mask==1 表示缺失。"""
    valid = 1.0 - mask
    denom = valid.sum().clamp_min(1.0)
    loss = (preds - target).abs() * valid
    return loss.sum() / denom


def evaluate(
    model: GMANPDFusion,
    loader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    losses, maes, rmses, rses, corrs = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            TE = batch["TE"].to(device)
            mask_y = batch["mask_y"].to(device)

            autocast_params = inspect.signature(autocast).parameters
            autocast_kwargs = {"enabled": use_amp}
            if "device_type" in autocast_params:
                autocast_kwargs["device_type"] = device.type if device.type in ("cuda", "cpu", "mps") else "cpu"
            with autocast(**autocast_kwargs):
                preds = model(X, TE)
            loss = masked_mae_loss(preds, y, mask_y)
            losses.append(float(loss.item()))

            preds_cpu = preds.detach().cpu()
            y_cpu = y.detach().cpu()
            maes.append(float(mae(preds_cpu, y_cpu)))
            rmses.append(float(rmse(preds_cpu, y_cpu)))
            rses.append(float(rse(preds_cpu, y_cpu)))
            corrs.append(float(corr(preds_cpu, y_cpu)))

    if not losses:
        return {"loss": float("nan"), "mae": float("nan"), "rmse": float("nan"), "rse": float("nan"), "corr": float("nan")}
    return {
        "loss": float(sum(losses) / len(losses)),
        "mae": float(sum(maes) / len(maes)),
        "rmse": float(sum(rmses) / len(rmses)),
        "rse": float(sum(rses) / len(rses)),
        "corr": float(sum(corrs) / len(corrs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PM2.5 预测训练脚手架")
    parser.add_argument("--config", type=Path, default=Path("My/config/pm25.yaml"))
    parser.add_argument("--device", type=str, default=None, help="指定训练设备（cuda/cpu）")
    parser.add_argument("--max-train-steps", type=int, default=0, help="每 epoch 训练步数上限（0 表示全量）")
    parser.add_argument("--max-val-steps", type=int, default=0, help="验证步数上限")
    parser.add_argument("--max-test-steps", type=int, default=0, help="测试步数上限")
    parser.add_argument("--run-test", action="store_true", help="训练结束后自动在测试集评估最佳模型")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_cfg = PM25DatasetConfig(
        data_dir=Path("My/dataset"),
        history_len=int(cfg["history_len"]),
        pred_len=int(cfg["num_pred"]),
        batch_size=int(cfg["batch_size"]),
        num_workers=0,
        feature_idx=int(cfg.get("feature_idx", 0)),
    )
    train_loader, val_loader, test_loader, metadata = load_pm25_dataloaders(dataset_cfg)

    cfg["num_nodes"] = int(metadata["nodes"].shape[0])
    cfg.setdefault("time_steps_per_day", int(metadata.get("steps_per_day", cfg["time_steps_per_day"])))

    run_dir, logger = prepare_run_dir()
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Using device: %s", device)

    model = GMANPDFusion(cfg, num_nodes=cfg["num_nodes"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    use_amp = device.type == "cuda"
    scaler_kwargs = {"enabled": use_amp}
    scaler_params = inspect.signature(GradScaler).parameters
    if "device_type" in scaler_params:
        scaler_kwargs["device_type"] = device.type if device.type in ("cuda", "cpu", "mps") else "cpu"
    scaler = GradScaler(**scaler_kwargs)
    if use_amp:
        logger.info("混合精度已启用 (AMP)")

    best_val = float("inf")
    ckpt_path = run_dir / "best_model.pt"

    for epoch in range(int(cfg["epochs"])):
        model.train()
        running_loss = 0.0
        steps = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            TE = batch["TE"].to(device)
            mask_y = batch["mask_y"].to(device)

            autocast_params = inspect.signature(autocast).parameters
            autocast_kwargs = {"enabled": use_amp}
            if "device_type" in autocast_params:
                autocast_kwargs["device_type"] = device.type if device.type in ("cuda", "cpu", "mps") else "cpu"
            with autocast(**autocast_kwargs):
                preds = model(X, TE)
                loss = masked_mae_loss(preds, y, mask_y)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            steps += 1
            if args.max_train_steps > 0 and steps >= args.max_train_steps:
                break

        avg_train_loss = running_loss / max(steps, 1)
        logger.info("Epoch %d train_loss=%.6f steps=%d", epoch + 1, avg_train_loss, steps)

        # 验证评估
        def limited_loader():
            for idx, batch in enumerate(val_loader):
                if 0 < args.max_val_steps <= idx:
                    break
                yield batch

        val_iterable = val_loader if args.max_val_steps == 0 else limited_loader()
        val_metrics = evaluate(model, val_iterable, device, use_amp)
        logger.info(
            "Epoch %d val_loss=%.6f mae=%.6f rmse=%.6f rse=%.6f corr=%.6f",
            epoch + 1,
            val_metrics["loss"],
            val_metrics["mae"],
            val_metrics["rmse"],
            val_metrics["rse"],
            val_metrics.get("corr", float("nan")),
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict() if use_amp else None,
                    "epoch": epoch + 1,
                    "val_metrics": val_metrics,
                    "config": cfg,
                },
                ckpt_path,
            )
            logger.info("Saved new best checkpoint to %s", ckpt_path)

    logger.info("Training finished. Best val_loss=%.6f", best_val)

    summary = metadata.get("summary", {})
    feature_idx = int(cfg.get("feature_idx", 0))
    feature_names = summary.get("feature_names", [])
    if feature_names and feature_idx < len(feature_names):
        logger.info("使用特征: %s", feature_names[feature_idx])

    if args.run_test:
        if not ckpt_path.exists():
            logger.warning("未找到最佳模型 checkpoint，跳过测试评估。")
            return
        logger.info("加载最佳模型进行测试评估: %s", ckpt_path)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])

        def limited_test_loader():
            for idx, batch in enumerate(test_loader):
                if 0 < args.max_test_steps <= idx:
                    break
                yield batch

        test_iterable = test_loader if args.max_test_steps == 0 else limited_test_loader()
        test_metrics = evaluate(model, test_iterable, device, use_amp)
        logger.info(
            "Test metrics -> loss=%.6f mae=%.6f rmse=%.6f rse=%.6f corr=%.6f",
            test_metrics["loss"],
            test_metrics["mae"],
            test_metrics["rmse"],
            test_metrics["rse"],
            test_metrics.get("corr", float("nan")),
        )


if __name__ == "__main__":
    main()
