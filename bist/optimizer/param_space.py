"""Parametre arama uzayı ve rastgele örnekleme."""
import random
from typing import Any

PARAM_SPACE: dict[str, list] = {
    "entry_score_ranging":          [0.48, 0.51, 0.54, 0.57, 0.60],
    "entry_score_trend":            [0.44, 0.47, 0.50, 0.53],
    "atr_stop_multiplier":          [1.5, 2.0, 2.5, 3.0],
    "trailing_stop_atr_multiplier": [2.5, 3.0, 4.0, 5.0],
    "risk_per_trade":               [0.006, 0.008, 0.010, 0.012],
    "volume_sma_multiplier":        [0.4, 0.6, 0.8],
    "adx_threshold":                [15, 18, 20, 22],
    "regime_ranging_threshold":     [0.30, 0.35, 0.40],
}

def random_params(seed: int | None = None) -> dict[str, Any]:
    if seed is not None:
        random.seed(seed)
    return {k: random.choice(v) for k, v in PARAM_SPACE.items()}

def grid_sample(n_samples: int, seed: int = 42) -> list[dict]:
    """n_samples rastgele parametre seti üret (Latin Hypercube benzeri)."""
    random.seed(seed)
    return [random_params() for _ in range(n_samples)]
