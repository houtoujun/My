"""
模型测试脚本。

加载训练好的 GMAN_PDFusion 权重，对测试集计算 MAE / RMSE / RSE / CORR 等指标。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from My.models.hybrid_gman_pdformer_pd import GMANPDFusion
from My.utils.data_pm25 import PM25DatasetConfig, load_pm25_dataloaders
from My.utils.metrics import mae, rmse, rse, corr


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        import yaml  # type: ignore

        return yaml.safe_load(f) or {}


def masked_mae_loss(preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = 1.0 - mask
    denom = valid.sum().clamp_min(1.0)
    return ((preds - target).abs() * valid).sum() / denom


def evaluate(
    model: GMANPDFusion,
    loader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    losses, maes, rmses, rses_metrics, corrs = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            TE = batch["TE"].to(device)
            mask_y = batch["mask_y"].to(device)

            preds = model(X, TE)
            losses.append(float(masked_mae_loss(preds, y, mask_y).item()))
            preds_cpu = preds.detach().cpu()
            y_cpu = y.detach().cpu()
            maes.append(float(mae(preds_cpu, y_cpu)))
            rmses.append(float(rmse(preds_cpu, y_cpu)))
            rses_metrics.append(float(rse(preds_cpu, y_cpu)))
            corr_value = float(corr(preds_cpu, y_cpu))
            if math.isfinite(corr_value):
                corrs.append(corr_value)

    if not losses:
        return {"loss": float("nan"), "mae": float("nan"), "rmse": float("nan"), "rse": float("nan"), "corr": float("nan")}

    metrics = {
        "loss": sum(losses) / len(losses),
        "mae": sum(maes) / len(maes),
        "rmse": sum(rmses) / len(rmses),
        "rse": sum(rses_metrics) / len(rses_metrics),
    }
    metrics["corr"] = sum(corrs) / len(corrs) if corrs else float("nan")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="PM2.5 模型测试脚本")
    parser.add_argument("--config", type=Path, required=True, help="训练配置文件路径")
    parser.add_argument("--checkpoint", type=Path, required=True, help="best_model.pt 路径")
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu，默认自动检测")
    parser.add_argument("--max-test-steps", type=int, default=0, help="测试步数上限（0 表示全量）")
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
    _, _, test_loader, metadata = load_pm25_dataloaders(dataset_cfg)

    cfg["num_nodes"] = int(metadata["nodes"].shape[0])
    cfg.setdefault("time_steps_per_day", int(metadata.get("steps_per_day", cfg.get("time_steps_per_day", 24))))

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[test] device: {device}")

    model = GMANPDFusion(cfg, num_nodes=cfg["num_nodes"]).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    def limited_loader():
        for idx, batch in enumerate(test_loader):
            if 0 < args.max_test_steps <= idx:
                break
            yield batch

    loader_iter = test_loader if args.max_test_steps == 0 else limited_loader()
    metrics = evaluate(model, loader_iter, device)
    print(json.dumps({"metrics": metrics, "checkpoint_epoch": state.get("epoch")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
