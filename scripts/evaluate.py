"""
PM2.5 预测评估脚本。

基于真实数据加载器，遍历指定模型并计算 MAE/RMSE/RSE/CORR。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from My.models.gcn import GCNBaseline
from My.models.hybrid_gman_pdformer_pd import GMANPDFusion
from My.models.mlp import MLPBaseline
from My.utils.data_pm25 import PM25DatasetConfig, load_pm25_dataloaders
from My.utils.metrics import mae, rmse, rse, corr


def _load_config(config_path: Path | None) -> Dict[str, Any]:
    default = {
        "num_pred": 12,
        "history_len": 12,
        "batch_size": 16,
        "num_nodes": 341,
        "feature_idx": 0,
    }
    if config_path is None or not config_path.exists():
        return default
    try:
        import yaml  # type: ignore
    except ImportError:
        print("未安装 PyYAML，评估使用默认配置。")
        return default
    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return {**default, **loaded}


def inverse_transform(scaler, array: np.ndarray) -> np.ndarray:
    if scaler is None:
        return array
    return scaler.inverse_transform(array)


def evaluate_model(
    name: str,
    cfg: Dict[str, Any],
    test_loader,
    scaler,
    max_batches: int,
) -> Dict[str, float]:
    if name == "GMAN_PDFusion":
        model = GMANPDFusion(cfg, num_nodes=int(cfg["num_nodes"]))
        def forward(batch):
            return model(batch["X"], batch["TE"])
    elif name == "MLP":
        model = MLPBaseline(cfg, num_nodes=int(cfg["num_nodes"]))
        def forward(batch):
            return model(batch["X"])
    elif name == "GCN":
        model = GCNBaseline(cfg, num_nodes=int(cfg["num_nodes"]))
        def forward(batch):
            return model(batch["X"])
    else:
        raise ValueError(f"未知模型: {name}")

    mae_list: List[float] = []
    rmse_list: List[float] = []
    rse_list: List[float] = []
    corr_list: List[float] = []

    for batch_idx, batch in enumerate(test_loader):
        preds_scaled = forward(batch)
        target_scaled = batch["y"]

        preds_np = preds_scaled.detach().cpu().numpy()
        target_np = target_scaled.detach().cpu().numpy()

        preds = inverse_transform(scaler, preds_np)
        target = inverse_transform(scaler, target_np)

        preds_tensor = torch.from_numpy(preds)
        target_tensor = torch.from_numpy(target)

        mae_list.append(float(mae(preds_tensor, target_tensor)))
        rmse_list.append(float(rmse(preds_tensor, target_tensor)))
        rse_list.append(float(rse(preds_tensor, target_tensor)))
        corr_list.append(float(corr(preds_tensor, target_tensor)))

        if 0 < max_batches <= batch_idx + 1:
            break

    return {
        "MAE": float(np.mean(mae_list)) if mae_list else float("nan"),
        "RMSE": float(np.mean(rmse_list)) if rmse_list else float("nan"),
        "RSE": float(np.mean(rse_list)) if rse_list else float("nan"),
        "CORR": float(np.mean(corr_list)) if corr_list else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PM2.5 评估脚本")
    parser.add_argument(
        "--models",
        type=str,
        default="GMAN_PDFusion,MLP,GCN",
        help="逗号分隔的模型名称列表",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("My/config/pm25.yaml"),
        help="配置文件路径",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=5,
        help="每个模型评估的最大 batch 数（0 表示全量）",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    dataset_cfg = PM25DatasetConfig(
        data_dir=Path("My/dataset"),
        history_len=int(cfg["history_len"]),
        pred_len=int(cfg["num_pred"]),
        batch_size=int(cfg["batch_size"]),
        num_workers=0,
        feature_idx=int(cfg.get("feature_idx", 0)),
    )
    _, _, test_loader, metadata = load_pm25_dataloaders(dataset_cfg)
    scaler = metadata.get("scaler")

    cfg["num_nodes"] = int(metadata["nodes"].shape[0])
    if "time_steps_per_day" not in cfg:
        cfg["time_steps_per_day"] = int(metadata.get("steps_per_day", 288))

    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    for name in model_names:
        scores = evaluate_model(name, cfg, test_loader, scaler, args.max_batches)
        print(f"{name}: MAE={scores['MAE']:.4f}, RMSE={scores['RMSE']:.4f}, RSE={scores['RSE']:.4f}, CORR={scores['CORR']:.4f}")


if __name__ == "__main__":
    main()
