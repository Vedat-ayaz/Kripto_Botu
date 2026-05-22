"""Skor fonksiyonu."""
from statistics import mean

def compute_score(results: list[dict]) -> float:
    """
    Backtest sonuç listesinden skalar skor hesaplar.
    results: [{"metrics": {...}}, ...]
    """
    if not results:
        return -999.0

    metrics_list = [r["metrics"] for r in results if r.get("metrics")]
    if not metrics_list:
        return -999.0

    total_trades = sum(m.get("num_trades", 0) for m in metrics_list)
    if total_trades < 5:
        return -999.0

    sharpe = mean(m.get("sharpe_ratio", -1) for m in metrics_list)
    pf     = mean(m.get("profit_factor", 0) for m in metrics_list)
    dd     = mean(m.get("max_drawdown_pct", 100) for m in metrics_list)
    wr     = mean(m.get("win_rate_pct", 0) for m in metrics_list)

    # Sharpe negatifse cezalandır
    if sharpe < -2:
        return -999.0

    return sharpe * 0.40 + (pf - 1.0) * 0.30 + (wr / 100.0) * 0.20 - (dd / 100.0) * 0.10
