import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd

from indicators.technical_indicators import TechnicalIndicators, _ema, _rsi, _atr, _adx


def make_df(n: int = 300) -> pd.DataFrame:
    """Test için sentetik OHLCV verisi üretir."""
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.uniform(1000, 5000, n)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


class TestManualIndicators:
    def test_ema_length(self):
        s = pd.Series(np.arange(1, 101, dtype=float))
        result = _ema(s, 20)
        assert len(result) == 100

    def test_ema_monotonic_with_rising_series(self):
        s = pd.Series(np.arange(1, 51, dtype=float))
        result = _ema(s, 5)
        # EMA of strictly rising series should be rising after warmup
        assert result.iloc[-1] > result.iloc[10]

    def test_rsi_range(self):
        df = make_df()
        rsi = _rsi(df["close"], 14)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_atr_positive(self):
        df = make_df()
        atr = _atr(df["high"], df["low"], df["close"], 14)
        assert (atr.dropna() > 0).all()

    def test_adx_range(self):
        df = make_df()
        adx = _adx(df["high"], df["low"], df["close"], 14)
        valid = adx.dropna()
        assert (valid >= 0).all()


class TestTechnicalIndicators:
    def setup_method(self):
        self.ind = TechnicalIndicators()

    def test_calculate_adds_columns(self):
        df = make_df()
        result = self.ind.calculate(df)
        for col in ["ema_50", "ema_200", "rsi", "atr", "adx", "volume_sma", "atr_ratio"]:
            assert col in result.columns, f"Eksik sütun: {col}"

    def test_does_not_modify_input(self):
        df = make_df()
        original_cols = set(df.columns)
        self.ind.calculate(df)
        assert set(df.columns) == original_cols

    def test_atr_ratio_positive(self):
        df = make_df()
        result = self.ind.calculate(df)
        valid = result["atr_ratio"].dropna()
        assert (valid > 0).all()

    def test_missing_column_raises(self):
        df = make_df().drop(columns=["volume"])
        with pytest.raises(ValueError):
            self.ind.calculate(df)
