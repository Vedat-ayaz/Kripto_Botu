import math
from typing import Optional
import numpy as np
import pandas as pd


def calculate_metrics(
    trades: list[dict],
    equity_curve: list[float],
    initial_capital: float,
    n_hours: Optional[int] = None,
) -> dict:
    """
    Backtest sonuçlarından performans metriklerini hesaplar.

    trades: [{"pnl": float, "entry": float, "exit": float, ...}, ...]
    equity_curve: [float, ...] her bar sonrası bakiye
    """
    if not trades:
        return _empty_metrics(initial_capital)

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    final_capital = equity_curve[-1] if equity_curve else initial_capital
    total_return = (final_capital - initial_capital) / initial_capital * 100

    win_rate = len(wins) / len(pnls) * 100 if pnls else 0.0

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    best_trade = max(pnls) if pnls else 0.0
    worst_trade = min(pnls) if pnls else 0.0

    max_dd = _max_drawdown(equity_curve)
    sharpe = _sharpe_ratio(equity_curve)
    calmar = _calmar_ratio(equity_curve, initial_capital, max_dd, n_hours=n_hours)

    return {
        "total_return_pct": round(total_return, 2),
        "final_capital": round(final_capital, 2),
        "initial_capital": initial_capital,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 3) if math.isfinite(profit_factor) else 999.0,
        "sharpe_ratio": round(sharpe, 3),
        "calmar_ratio": round(calmar, 3),
        "num_trades": len(pnls),
        "num_wins": len(wins),
        "num_losses": len(losses),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "best_trade": round(best_trade, 4),
        "worst_trade": round(worst_trade, 4),
        "gross_profit": round(gross_profit, 4),
        "gross_loss": round(gross_loss, 4),
    }


def _max_drawdown(equity: list[float]) -> float:
    """Peak-to-trough maksimum drawdown oranı."""
    if len(equity) < 2:
        return 0.0
    arr = np.array(equity, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / peak
    return float(abs(dd.min()))


def _calmar_ratio(
    equity: list[float],
    initial_capital: float,
    max_dd: float,
    n_hours: Optional[int] = None,
) -> float:
    """
    Calmar Ratio = Yıllıklandırılmış Getiri / Max Drawdown
    Crypto için 365 günlük yıl (8760 saat) varsayımı.

    n_hours: Gerçek test süresinin saat cinsinden uzunluğu.
             Per-symbol backtestlerde equity curve saat-bazlı olduğu için
             len(equity) kullanılabilir.
             Aggregate (trade-bazlı) equity curve için gerçek saati geç,
             aksi hâlde 679 trade = 679 saat ≈ 28 gün zannetmiş olur.
    """
    if len(equity) < 2 or max_dd <= 0:
        return 0.0
    final = equity[-1]
    hours = n_hours if n_hours and n_hours > 0 else len(equity)
    exponent = 8760 / max(hours, 1)
    # Çok küçük bar count ile taşma önlemi: üs 4'ten büyük olmamalı
    # (örn. 2 barlık test curve'ü → 8760/2=4380 → 1.78^4380 overflow)
    exponent = min(exponent, 4.0)
    try:
        ann_return = (final / initial_capital) ** exponent - 1
    except OverflowError:
        return 0.0
    return float(ann_return / max_dd)


def _sharpe_ratio(equity: list[float], risk_free: float = 0.0, periods_per_year: int = 8760) -> float:
    """
    Sharpe oranı — 1 saatlik bar varsayımıyla yıllıklandırılır.
    periods_per_year = 365 × 24 = 8760 (crypto 7/24 işlem görür).
    Eski 252 değeri günlük getiri varsayımıydı → Sharpe ~5.9× düşük çıkıyordu.
    """
    if len(equity) < 2:
        return 0.0
    arr = np.array(equity, dtype=float)
    returns = np.diff(arr) / arr[:-1]
    std = returns.std()
    if std == 0:
        return 0.0
    mean_return = returns.mean() - risk_free
    return float(mean_return / std * math.sqrt(periods_per_year))


def _empty_metrics(initial_capital: float) -> dict:
    return {
        "total_return_pct": 0.0,
        "final_capital": initial_capital,
        "initial_capital": initial_capital,
        "max_drawdown_pct": 0.0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "sharpe_ratio": 0.0,
        "calmar_ratio": 0.0,
        "num_trades": 0,
        "num_wins": 0,
        "num_losses": 0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "best_trade": 0.0,
        "worst_trade": 0.0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
    }


def print_metrics(metrics: dict, symbol: str = "") -> None:
    header = f"===== BACKTEST SONUÇLARI {symbol} =====" if symbol else "===== BACKTEST SONUÇLARI ====="
    print(f"\n{header}")
    print(f"  Başlangıç Sermayesi : {metrics['initial_capital']:>12,.2f} USDT")
    print(f"  Final Sermaye       : {metrics['final_capital']:>12,.2f} USDT")
    print(f"  Toplam Getiri       : {metrics['total_return_pct']:>11.2f}%")
    print(f"  Max Drawdown        : {metrics['max_drawdown_pct']:>11.2f}%")
    print(f"  Sharpe Oranı        : {metrics['sharpe_ratio']:>12.3f}")
    print(f"  Calmar Oranı        : {metrics['calmar_ratio']:>12.3f}")
    print(f"  Profit Factor       : {metrics['profit_factor']:>12.3f}")
    print(f"  Win Rate            : {metrics['win_rate_pct']:>11.2f}%")
    print(f"  Toplam İşlem        : {metrics['num_trades']:>12}")
    print(f"  Kazanan / Kaybeden  : {metrics['num_wins']:>5} / {metrics['num_losses']}")
    print(f"  Ort. Kazanç         : {metrics['avg_win']:>12.4f} USDT")
    print(f"  Ort. Kayıp          : {metrics['avg_loss']:>12.4f} USDT")
    print(f"  En İyi İşlem        : {metrics['best_trade']:>12.4f} USDT")
    print(f"  En Kötü İşlem       : {metrics['worst_trade']:>12.4f} USDT")
    print("=" * len(header))
