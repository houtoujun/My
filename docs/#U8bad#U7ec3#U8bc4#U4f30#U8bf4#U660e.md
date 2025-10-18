# PM2.5 训练与测试指南

本文档汇总数据准备、配置、训练、调参与测试流程，便于后续工作者快速上手。

---

## 1. 环境准备

- Python ≥ 3.8（建议使用 `conda`/`venv`）
- 安装依赖：
  ```bash
  pip install torch numpy pandas scikit-learn pyyaml
  ```
- 数据已整理在 `My/dataset/`（结构详见 `My/docs/数据交接说明.md`）。

---

## 2. 预处理产物

所有中间结果位于 `My/artifacts/`：

| 文件 | 说明 |
|------|------|
| `dtw_matrix.npy` | DTW 距离矩阵（按全量数据计算） |
| `geo_mask.npy` / `sem_mask.npy` | 地理/语义邻接掩码（bool，[N,N]） |
| `pattern_keys.npy` | DFT 延迟模式原型（[num_patterns, window, N]） |
| `geo_neighbors.npz` / `sem_neighbors.npz` | 稀疏邻居索引（`indices` + `valid`） |

如需重新生成可参考命令（原数据不变时，仅 `pattern_keys.npy` 会随训练划分调整）：
- `My/utils/dtw.py`：生成 DTW 矩阵
- `My/utils/masks.py`：生成地理/语义掩码与邻居索引
- `My/utils/patterns.py`：学习 DFT 模式

当前已按最新训练集（比例 7:1:1）重新学习了 `pattern_keys.npy`。

---

## 3. 配置文件 `My/config/pm25.yaml`

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `num_pred` | 12 | 预测步数 Q |
| `history_len` | 24 | 历史窗口 P |
| `batch_size` | 16 | 训练批大小 |
| `feature_idx` | 0 | 预测目标（0=PM2.5） |
| `learning_rate` | 5e-4 | Adam 学习率 |
| `epochs` | 50 | 训练轮数 |
| `K` / `d` / `L` | 8 / 8 / 3 | 空间头数/单头维度/块数 |
| `geo_ratio` | 0.6 | 地理分支比例 |
| `S` / `num_patterns` | 6 / 16 | DFT 窗口与模式数 |
| `time_steps_per_day` | 24 | 每天时间步（小时） |
| `spatial_chunk_size` | 64 | 空间注意力分块大小 |
| `pattern_path` | `My/artifacts/pattern_keys.npy` | DFT 模式路径 |
| `geo_mask_path` / `sem_mask_path` | ... | 地理/语义掩码路径 |
| `geo_neighbors_path` / `sem_neighbors_path` | ... | 稀疏邻接索引路径 |

数据划分：训练/验证/测试 = **6 : 2 : 2**，按时间顺序切分，避免信息泄露。

---

## 4. 脚本与模型

- `My/models/hybrid_gman_pdformer_pd.py`：GMAN_PDFusion（混合空间注意力 + DFT，支持稀疏邻域）
- `My/models/gman_components.py`：GMAN 基础模块
- `My/utils/data_pm25.py`：数据加载（`load_pm25_dataloaders`）
- `My/scripts/train_pm25.py`：训练（AMP + best checkpoint + 可选自动测试）
- `My/scripts/test_pm25.py`：加载 checkpoint 在测试集评估
- `My/scripts/evaluate.py`：快速对比 GMAN_PDFusion / MLP / GCN 的前向指标

---

## 5. 训练

```bash
python My/scripts/train_pm25.py \
  --config My/config/pm25.yaml \
  --device cuda \
  --run-test
```

- `--run-test`：训练后自动使用最佳 checkpoint 在测试集评估。
- 调试可增加 `--max-train-steps` / `--max-val-steps` / `--max-test-steps`（限制批次数）。
- 当 `--device cuda` 时默认启用 AMP（混合精度），可显著降低显存占用；若需关闭改用 `--device cpu` 或手动修改脚本。

输出内容：
- `My/results/runs/<timestamp>/train.log`
- `My/results/runs/<timestamp>/config.json`
- `My/results/runs/<timestamp>/best_model.pt`
- 日志记录指标：loss、MAE、RMSE、RSE、CORR。

---

## 6. 测试（单独运行）

```bash
python My/scripts/test_pm25.py \
  --config My/config/pm25.yaml \
  --checkpoint My/results/runs/<timestamp>/best_model.pt \
  --device cuda
```

输出 JSON：`loss`、`mae`、`rmse`、`rse`、`corr`，以及 checkpoint 训练轮数。

---

## 7. 显存与调参建议

- 稀疏注意力已接入，仅在邻居集合上做 softmax；但 Q/K/V/DFT 激活依旧是密集张量 `[B,K,T,N,d]`，是主要显存来源。
- 若显存紧张：
  1. 降低 `batch_size`（16 → 8/4）、缩短 `history_len`、减小 `K` 或 `geo_ratio`；
  2. 调整 `spatial_chunk_size` 或启用梯度检查点；
  3. 保持 AMP；必要时可进一步实现仅对邻居采样的 K/V（待扩展）。

---

## 8. 接入其它模型做对比

1. 数据加载：使用 `My/utils/data_pm25.py` 提供的 `load_pm25_dataloaders`，可直接获得 Train/Val/Test Loader。
2. 模型接口：建议继承 `nn.Module`，实现 `forward(X, TE) -> [B, Q, N]`，与现有指标对齐。
3. 训练：参考 `My/scripts/train_pm25.py`，替换模型构造和损失即可复用 AMP / 日志 / 自动测试。
4. 评估：在 `My/scripts/evaluate.py` 中注册模型分支，使用相同指标（`My/utils/metrics.py`）。
5. 稀疏邻域：可直接加载 `geo_neighbors.npz` 等文件重复利用现成邻居；若无需稀疏，可用 `geo_mask.npy` 作为稠密邻接。

---

## 9. 常见问题

| 问题 | 处理建议 |
|------|----------|
| `pattern_keys.npy` 缺失 | 重新执行预处理步骤 2.3 |
| CUDA OOM | 降低 `batch_size`、缩短 `history_len`、减小 `K`/`geo_ratio` 或 `spatial_chunk_size` |
| 验证指标停滞 | 调整 `learning_rate`，或减少 `K`/`L`、重新生成邻域掩码 |
| 测试出现 NaN | 检查数据是否仍含 NaN，确认掩码/模式文件是否对应当前划分 |
| 自定义划分 | 修改 `PM25DatasetConfig(train_ratio=7/9, ...)`，并按需重新学习模式/掩码 |

---

## 10. 后续建议

- 若需进一步压缩显存，可继续将 K/V 也实现为稀疏 gather/scatter。
- 可将 `evaluate.py` 抽象为模型注册表，便于批量对比新模型。
- 如需更多 baseline（例如 STGCN、Informer 等），推荐复用现有数据加载和指标接口。
