import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from My.utils.patterns import learn_delay_patterns_from_dataset

Path('My/artifacts').mkdir(parents=True, exist_ok=True)
patterns, _ = learn_delay_patterns_from_dataset(
    Path('My/dataset'),
    window=6,
    num_patterns=16,
    stride=6,
    normalize=True,
    cache_path=Path('My/artifacts/pattern_keys.npy'),
)
print('patterns shape', patterns.shape)
