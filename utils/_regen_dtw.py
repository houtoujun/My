import numpy as np
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from My.utils.dtw import compute_dtw_from_dataset

def main():
    cache_path = Path('My/artifacts/dtw_matrix.npy')
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    matrix, _ = compute_dtw_from_dataset(
        data_dir=Path('My/dataset'),
        feature_idx=0,
        radius=None,
        radius_ratio=0.05,
        downsample=168,
        cache_path=cache_path,
        reuse_if_exists=False,
        verbose=True,
    )
    print('dtw_matrix saved:', matrix.shape)

if __name__ == '__main__':
    main()
