"""Optimizasyon sonucunu config_bist.yaml'a yaz."""
import yaml
from pathlib import Path


def update_config_with_params(
    config_path: str,
    params: dict,
    backup: bool = True,
) -> None:
    """
    En iyi parametreleri config_bist.yaml'daki strategy ve risk bölümlerine yazar.
    Orijinal config'i .bak olarak saklar.
    """
    path = Path(config_path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if backup:
        bak_path = path.with_suffix(".yaml.bak")
        with open(bak_path, "w") as f:
            yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
        print(f"  Yedek: {bak_path}")

    STRATEGY_KEYS = {
        "entry_score_ranging", "entry_score_trend",
        "adx_threshold", "volume_sma_multiplier",
        "regime_ranging_threshold", "regime_trending_threshold",
        "adx_boost", "slope_bars", "momentum_lookback",
    }
    RISK_KEYS = {
        "risk_per_trade", "atr_stop_multiplier",
        "trailing_stop_atr_multiplier", "daily_max_loss",
        "max_open_positions", "max_position_pct",
    }

    for k, v in params.items():
        if k in STRATEGY_KEYS:
            cfg.setdefault("strategy", {})[k] = v
        elif k in RISK_KEYS:
            cfg.setdefault("risk", {})[k] = v

    with open(path, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"  Config güncellendi: {path}")
