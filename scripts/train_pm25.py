"""
PM2.5 预测训练脚本

- 自动加载 PM25 数据集，构建 GMAN_PDFusion 模型
- 支持 AMP 混合精度、断点续训、训练完成后自动评估
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import inspect
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from My.models.hybrid_gman_pdformer_pd import GMANPDFusion
from My.utils.data_pm25 import PM25DatasetConfig, load_pm25_dataloaders
from My.utils.metrics import mae, rmse, rse, corr


TOP_K_CHECKPOINTS = 10
EARLY_STOP_PATIENCE = 20


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


def prepare_run_dir(resume_dir: Optional[Path] = None) -> tuple[Path, logging.Logger]:
    run_root = Path("My/results/runs")
    run_root.mkdir(parents=True, exist_ok=True)
    if resume_dir is not None:
        run_dir = resume_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        log_mode = "a"
    else:
        run_dir = run_root / datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        log_mode = "w"

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8", mode=log_mode)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return run_dir, logger


def masked_mae_loss(preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """带缺失掩码的 MAE 损失。"""
    valid = 1.0 - mask
    denom = valid.sum().clamp_min(1.0)
    loss = (preds - target).abs() * valid
    return loss.sum() / denom


def _inverse_transform_array(scaler, array: np.ndarray) -> np.ndarray:
    if scaler is None:
        return array
    shape = array.shape
    restored = scaler.inverse_transform(array.reshape(-1, shape[-1]))
    return restored.reshape(shape)


def evaluate(
    model: GMANPDFusion,
    loader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    use_amp: bool,
    scaler,
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

            preds_np = preds.detach().cpu().numpy()
            target_np = y.detach().cpu().numpy()
            mask_np = mask_y.detach().cpu().numpy()

            preds_inv = _inverse_transform_array(scaler, preds_np)
            target_inv = _inverse_transform_array(scaler, target_np)
            valid = 1.0 - mask_np
            preds_inv = preds_inv * valid + target_inv * (1.0 - valid)

            preds_tensor = torch.from_numpy(preds_inv)
            target_tensor = torch.from_numpy(target_inv)

            maes.append(float(mae(preds_tensor, target_tensor)))
            rmses.append(float(rmse(preds_tensor, target_tensor)))
            rses.append(float(rse(preds_tensor, target_tensor)))
            corrs.append(float(corr(preds_tensor, target_tensor)))

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
    parser = argparse.ArgumentParser(description="PM2.5 预测训练脚本")
    parser.add_argument("--config", type=Path, default=Path("My/config/pm25.yaml"))
    parser.add_argument("--device", type=str, default=None, help="指定训练设备（cuda/cpu）")
    parser.add_argument("--max-train-steps", type=int, default=0, help="每个 epoch 最大训练步数，0 表示不限制")
    parser.add_argument("--max-val-steps", type=int, default=0, help="验证集最大步数，0 表示不限制")
    parser.add_argument("--max-test-steps", type=int, default=0, help="测试集最大步数，0 表示不限制")
    parser.add_argument("--run-test", action="store_true", help="训练结束后自动在测试集评估")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="断点续训所用的 checkpoint 路径（latest_checkpoint.pt 或 best_model.pt）",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    resume_state: Optional[Dict[str, Any]] = None
    resume_path: Optional[Path] = None
    if args.resume_from is not None:
        resume_path = args.resume_from.expanduser()
        if not resume_path.exists():
            raise FileNotFoundError(f"未找到 resume checkpoint: {resume_path}")
        resume_state = torch.load(resume_path, map_location="cpu")
        ckpt_cfg = resume_state.get("config")
        if isinstance(ckpt_cfg, dict):
            cfg.update(ckpt_cfg)

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
    cfg.setdefault("time_steps_per_day", int(metadata.get("steps_per_day", cfg.get("time_steps_per_day", 24))))

    resume_dir = resume_path.parent if resume_path is not None else None
    run_dir, logger = prepare_run_dir(resume_dir)
    if resume_path is not None:
        logger.info("Resume training from %s", resume_path)

    (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    topk_file = run_dir / "topk_checkpoints.json"
    topk_entries: list[Dict[str, Any]] = []
    if topk_file.exists():
        try:
            loaded_topk = json.loads(topk_file.read_text(encoding="utf-8"))
            if isinstance(loaded_topk, list):
                for item in loaded_topk:
                    path_str = item.get("path")
                    if not path_str:
                        continue
                    entry_path = run_dir / path_str
                    if not entry_path.exists():
                        continue
                    try:
                        loss_val = float(item.get("loss", float("inf")))
                        epoch_val = int(item.get("epoch", 0))
                    except (TypeError, ValueError):
                        continue
                    topk_entries.append({"path": path_str, "loss": loss_val, "epoch": epoch_val})
        except Exception:
            logger.warning("无法解析 %s，重新开始记录 top-k checkpoint。", topk_file)

    # 清理缺失文件并限制在 TOP_K_CHECKPOINTS 个
    topk_entries = [
        entry
        for entry in topk_entries
        if (run_dir / entry["path"]).exists()
    ]
    topk_entries.sort(key=lambda x: x["loss"])
    if len(topk_entries) > TOP_K_CHECKPOINTS:
        for removed in topk_entries[TOP_K_CHECKPOINTS:]:
            removed_path = run_dir / removed["path"]
            if removed_path.exists():
                try:
                    removed_path.unlink()
                except Exception:
                    logger.warning("删除旧 checkpoint %s 失败", removed_path)
        topk_entries = topk_entries[:TOP_K_CHECKPOINTS]
    if topk_entries:
        topk_file.write_text(json.dumps(topk_entries, ensure_ascii=False, indent=2), encoding="utf-8")

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
        logger.info("启用混合精度训练 (AMP)")

    start_epoch = 0
    best_val = float("inf")
    best_metrics: Optional[Dict[str, float]] = None
    epochs_since_improve = 0

    if resume_state is not None:
        model.load_state_dict(resume_state["model_state"])
        optimizer_state = resume_state.get("optimizer_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        scaler_state = resume_state.get("scaler_state")
        if use_amp and scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        start_epoch = int(resume_state.get("epoch", 0))
        best_val = float(
            resume_state.get(
                "best_val",
                resume_state.get("val_metrics", {}).get("loss", float("inf")),
            )
        )
        best_metrics = resume_state.get("best_metrics") or resume_state.get("val_metrics")
        epochs_since_improve = resume_state.get("epochs_since_improve", 0)
        logger.info(
            "Loaded checkpoint epoch=%d best_val=%.6f",
            start_epoch,
            best_val if best_val < float("inf") else float("nan"),
        )

    ckpt_path = run_dir / "best_model.pt"
    latest_path = run_dir / "latest_checkpoint.pt"

    total_epochs = int(cfg["epochs"])
    if start_epoch >= total_epochs:
        logger.info(
            "Checkpoint epoch (%d) 已达到或超过配置的 epochs=%d，跳过训练阶段。",
            start_epoch,
            total_epochs,
        )
    else:
        for epoch in range(start_epoch, total_epochs):
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

            def limited_loader():
                for idx, batch in enumerate(val_loader):
                    if 0 < args.max_val_steps <= idx:
                        break
                    yield batch

            val_iterable = val_loader if args.max_val_steps == 0 else limited_loader()
            val_metrics = evaluate(model, val_iterable, device, use_amp, metadata.get("scaler"))
            logger.info(
                "Epoch %d val_loss=%.6f mae=%.6f rmse=%.6f rse=%.6f corr=%.6f",
                epoch + 1,
                val_metrics["loss"],
                val_metrics["mae"],
                val_metrics["rmse"],
                val_metrics["rse"],
                val_metrics.get("corr", float("nan")),
            )

            is_best = val_metrics["loss"] < best_val
            if is_best:
                best_val = val_metrics["loss"]
                best_metrics = val_metrics
                epochs_since_improve = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scaler_state": scaler.state_dict() if use_amp else None,
                        "epoch": epoch + 1,
                        "val_metrics": val_metrics,
                        "best_metrics": best_metrics,
                        "best_val": best_val,
                        "config": cfg,
                    },
                    ckpt_path,
                )
                logger.info("Saved new best checkpoint to %s", ckpt_path)

            latest_state = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if use_amp else None,
                "epoch": epoch + 1,
                "val_metrics": val_metrics,
                "best_metrics": best_metrics or val_metrics,
                "best_val": best_val,
                "epochs_since_improve": epochs_since_improve if is_best else epochs_since_improve + 1,
                "config": cfg,
            }

            epoch_ckpt_name = f"checkpoint_epoch{epoch + 1:04d}.pt"
            epoch_ckpt_path = run_dir / epoch_ckpt_name
            if epoch_ckpt_path.exists():
                try:
                    epoch_ckpt_path.unlink()
                except Exception:
                    logger.warning("覆盖旧 checkpoint 失败: %s", epoch_ckpt_path)
            torch.save(latest_state, epoch_ckpt_path)

            # 更新 top-k checkpoint 列表
            topk_entries = [
                entry for entry in topk_entries if (run_dir / entry["path"]).exists()
            ]
            topk_entries = [entry for entry in topk_entries if entry["path"] != epoch_ckpt_name]
            candidate = {
                "path": epoch_ckpt_name,
                "loss": float(val_metrics["loss"]),
                "epoch": int(epoch + 1),
            }
            topk_entries.append(candidate)
            topk_entries.sort(key=lambda x: x["loss"])
            overflow = []
            if len(topk_entries) > TOP_K_CHECKPOINTS:
                overflow = topk_entries[TOP_K_CHECKPOINTS:]
                topk_entries = topk_entries[:TOP_K_CHECKPOINTS]
            for removed in overflow:
                removed_path = run_dir / removed["path"]
                if removed_path.exists():
                    try:
                        removed_path.unlink()
                    except Exception:
                        logger.warning("删除超出 top-%d 的 checkpoint 失败: %s", TOP_K_CHECKPOINTS, removed_path)
            if candidate in topk_entries:
                logger.info(
                    "Checkpoint %s 保留在前%d，val_loss=%.6f",
                    epoch_ckpt_name,
                    TOP_K_CHECKPOINTS,
                    candidate["loss"],
                )
            else:
                if epoch_ckpt_path.exists():
                    try:
                        epoch_ckpt_path.unlink()
                    except Exception:
                        logger.warning("删除非前%d checkpoint 失败: %s", TOP_K_CHECKPOINTS, epoch_ckpt_path)
            topk_file.write_text(json.dumps(topk_entries, ensure_ascii=False, indent=2), encoding="utf-8")

            torch.save(latest_state, latest_path)

            if is_best:
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= EARLY_STOP_PATIENCE:
                    logger.info(
                        "验证集连续 %d 轮无提升，触发早停。",
                        EARLY_STOP_PATIENCE,
                    )
                    break

    logger.info("Training finished. Best val_loss=%.6f", best_val)

    summary = metadata.get("summary", {})
    feature_idx = int(cfg.get("feature_idx", 0))
    feature_names = summary.get("feature_names", [])
    if feature_names and feature_idx < len(feature_names):
        logger.info("使用特征: %s", feature_names[feature_idx])

    if args.run_test:
        eval_checkpoint = ckpt_path
        if not eval_checkpoint.exists() and resume_path is not None and resume_path.exists():
            eval_checkpoint = resume_path
        if not eval_checkpoint.exists():
            logger.warning("未找到可用的 checkpoint，跳过测试阶段。")
            return

        logger.info("使用 checkpoint 进行测试: %s", eval_checkpoint)
        state = torch.load(eval_checkpoint, map_location=device)
        model.load_state_dict(state["model_state"])

        def limited_test_loader():
            for idx, batch in enumerate(test_loader):
                if 0 < args.max_test_steps <= idx:
                    break
                yield batch

        test_iterable = test_loader if args.max_test_steps == 0 else limited_test_loader()
        test_metrics = evaluate(model, test_iterable, device, use_amp, metadata.get("scaler"))
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
