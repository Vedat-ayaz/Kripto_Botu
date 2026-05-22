import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd

from indicators.technical_indicators import TechnicalIndicators
from strategy.trend_following_strategy import TrendFollowingStrategy
from strategy.signal import Side


def make_bullish_df(n: int = 250) -> pd.DataFrame:
    """
    Tüm long koşullarını sağlayan yapay veri:
    EMA50 > EMA200, close yukarıda, yüksek hacim ve ADX.
    """
    np.random.seed(0)
    # Güçlü yükselen trend
    close = 100.0 + np.linspace(0, 80, n) + np.random.randn(n) * 0.3
    high = close + 0.5
    low = close - 0.5
    open_ = close - 0.1
    volume = np.random.uniform(2000, 5000, n)

    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def make_bearish_df(n: int = 250) -> pd.DataFrame:
    """EMA50 < EMA200 (düşen trend) verisi."""
    np.random.seed(1)
    close = 180.0 - np.linspace(0, 80, n) + np.random.randn(n) * 0.3
    high = close + 0.5
    low = close - 0.5
    open_ = close - 0.1
    volume = np.random.uniform(500, 1500, n)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class TestTrendFollowingStrategy:
    def setup_method(self):
        self.strategy = TrendFollowingStrategy()

    def test_buy_signal_on_bullish_data(self):
        df = make_bullish_df()
        signal = self.strategy.generate_signal("BTC/USDT", df)
        # Güçlü trend verisinde BUY veya HOLD beklenir
        assert signal.side in (Side.BUY, Side.HOLD)
        assert signal.symbol == "BTC/USDT"
        assert signal.price > 0

    def test_hold_on_bearish_data(self):
        df = make_bearish_df()
        signal = self.strategy.generate_signal("ETH/USDT", df)
        assert signal.side == Side.HOLD

    def test_insufficient_data_returns_hold(self):
        df = make_bullish_df(n=100)  # EMA200 için yetersiz
        signal = self.strategy.generate_signal("SOL/USDT", df)
        assert signal.side == Side.HOLD
        assert "Yetersiz" in signal.reason

    def test_signal_has_required_fields(self):
        df = make_bullish_df()
        signal = self.strategy.generate_signal("BTC/USDT", df)
        assert signal.timestamp is not None
        assert signal.price > 0
        assert 0.0 <= signal.confidence_score <= 1.0

    def test_should_exit_when_below_ema50(self):
        df = make_bullish_df()
        ind = TechnicalIndicators()
        df_ind = ind.calculate(df)
        # Son barda close'u EMA50'nin altına çek
        df_ind = df_ind.copy()
        ema50_last = df_ind.iloc[-1]["ema_50"]
        df_ind.iloc[-1, df_ind.columns.get_loc("close")] = ema50_last * 0.95
        should, reason = self.strategy.should_exit("BTC/USDT", df_ind, entry_price=150)
        assert should is True
        assert "EMA50" in reason

    def test_composite_score_between_0_and_1(self):
        """
        _composite_score yeni API'yi test eder.
        Tüm kombinasyonlar 0-1 aralığında skor üretmeli.
        """
        # (close, ema_fast, ema_slow, rsi, adx, macd, macd_hist,
        #  bb_pct_b, bb_width, stoch_k, volume, volume_sma,
        #  obv, obv_ema, tsmom, is_trending)
        cases = [
            (150, 140, 120, 55, 30, 0.5, 0.1, 0.7, 0.05, 55, 3000, 2000, 5000, 4000, 0.8, True),
            (110, 108, 100, 47, 21, 0.2, 0.05, 0.55, 0.03, 42, 1200, 1000, 2000, 1800, 0.3, False),
            (200, 195, 180, 68, 45, 1.2, 0.3, 0.8, 0.06, 72, 5000, 4000, 9000, 8000, 1.2, True),
        ]
        for args in cases:
            score, detail = self.strategy._composite_score(*args)
            assert 0.0 <= score <= 1.0, f"Geçersiz skor: {score}"
            assert isinstance(detail, str)

    def test_momentum_rank_score_returns_float(self):
        """get_momentum_rank_score float döndürmeli."""
        ind = TechnicalIndicators()
        df = make_bullish_df(n=250)
        df_ind = ind.calculate(df)
        score = self.strategy.get_momentum_rank_score(df_ind)
        assert isinstance(score, float)

    def test_momentum_rank_score_insufficient_data(self):
        """Yetersiz veri için 0.0 döndürmeli."""
        df = make_bullish_df(n=10)
        score = self.strategy.get_momentum_rank_score(df)
        assert score == 0.0
