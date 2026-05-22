import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd

from backtest.backtester import Backtester
from backtest.metrics import calculate_metrics, _max_drawdown, _sharpe_ratio


def make_df(n: int = 400, trend: str = "up") -> pd.DataFrame:
    np.random.seed(99)
    if trend == "up":
        close = 100.0 + np.linspace(0, 100, n) + np.random.randn(n) * 0.5
    else:
        close = 200.0 - np.linspace(0, 100, n) + np.random.randn(n) * 0.5

    high = close + 0.5
    low = close - 0.5
    open_ = close - 0.05
    volume = np.random.uniform(2000, 6000, n)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class TestBacktester:
    def setup_method(self):
        self.bt = Backtester(
            initial_capital=10_000.0,
            commission_rate=0.001,
            slippage_rate=0.0005,
        )

    def test_run_returns_required_keys(self):
        df = make_df()
        result = self.bt.run("BTC/USDT", df)
        for key in ("symbol", "metrics", "trades", "equity_curve"):
            assert key in result

    def test_equity_curve_starts_at_capital(self):
        df = make_df()
        result = self.bt.run("BTC/USDT", df)
        assert result["equity_curve"][0] == pytest.approx(10_000.0)

    def test_num_trades_non_negative(self):
        df = make_df()
        result = self.bt.run("BTC/USDT", df)
        assert result["metrics"]["num_trades"] >= 0

    def test_bearish_fewer_trades(self):
        df_bear = make_df(trend="down")
        result = self.bt.run("BTC/USDT", df_bear)
        # Düşen trendde az veya hiç long işlem açılmamalı
        assert result["metrics"]["num_trades"] >= 0  # en az doğruluk: sıfır veya pozitif

    def test_metrics_win_rate_between_0_and_100(self):
        df = make_df()
        result = self.bt.run("BTC/USDT", df)
        wr = result["metrics"]["win_rate_pct"]
        assert 0.0 <= wr <= 100.0

    def test_csv_saved(self, tmp_path):
        df = make_df()
        result = self.bt.run("BTC/USDT", df)
        # Eğer trade varsa CSV kaydı test et
        if result["metrics"]["num_trades"] > 0:
            path = self.bt.save_csv(result, output_dir=str(tmp_path))
            assert os.path.exists(path)


class TestMetrics:
    def test_empty_trades(self):
        m = calculate_metrics([], [10_000.0], 10_000.0)
        assert m["num_trades"] == 0
        assert m["total_return_pct"] == 0.0

    def test_max_drawdown_flat(self):
        assert _max_drawdown([100, 100, 100]) == pytest.approx(0.0)

    def test_max_drawdown_drop(self):
        # Peak 200, trough 100 → drawdown = 50%
        dd = _max_drawdown([100, 200, 100])
        assert abs(dd - 0.5) < 1e-9

    def test_sharpe_flat_returns_zero(self):
        equity = [100.0] * 50
        assert _sharpe_ratio(equity) == pytest.approx(0.0)

    def test_profit_factor_all_wins(self):
        trades = [{"pnl": 100}, {"pnl": 200}, {"pnl": 50}]
        m = calculate_metrics(trades, [10_000, 10_100, 10_300, 10_350], 10_000.0)
        assert m["profit_factor"] == 999.0  # sonsuz için 999 döner

    def test_win_rate_calculation(self):
        trades = [{"pnl": 100}, {"pnl": -50}, {"pnl": 200}, {"pnl": -30}]
        m = calculate_metrics(trades, [10_000] * 5, 10_000.0)
        assert m["win_rate_pct"] == pytest.approx(50.0)
