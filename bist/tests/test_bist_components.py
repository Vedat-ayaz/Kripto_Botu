"""BIST modül bileşen testleri — yfinance bağlantısı gerekmez."""
import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timezone


# ── Universe ──────────────────────────────────────────────────────────────────
class TestUniverse:
    def test_bist_30_has_30_items(self):
        from bist.data.universe import BIST_30
        assert len(BIST_30) == 30

    def test_no_is_suffix(self):
        from bist.data.universe import BIST_30, BIST_100
        for sym in BIST_30 + BIST_100:
            assert not sym.endswith(".IS"), f"{sym} .IS eki içeriyor"

    def test_no_duplicates_bist_30(self):
        from bist.data.universe import BIST_30
        assert len(BIST_30) == len(set(BIST_30))

    def test_get_universe_returns_list(self):
        from bist.data.universe import get_universe
        assert isinstance(get_universe("30"), list)
        assert isinstance(get_universe("100"), list)

    def test_get_yf_symbols(self):
        from bist.data.universe import get_yf_symbols
        result = get_yf_symbols(["GARAN", "AKBNK"])
        assert result == ["GARAN.IS", "AKBNK.IS"]

    def test_get_sector(self):
        from bist.data.universe import get_sector
        assert get_sector("GARAN") == "banks"
        assert get_sector("THYAO") == "aviation"
        assert get_sector("XXXXXX") is None


# ── PriceConverter ────────────────────────────────────────────────────────────
class TestPriceConverter:
    def _make_df(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
        return pd.DataFrame({
            "open":  [100.0, 110.0, 120.0, 115.0, 125.0],
            "high":  [105.0, 115.0, 125.0, 120.0, 130.0],
            "low":   [ 95.0, 105.0, 115.0, 110.0, 120.0],
            "close": [102.0, 112.0, 122.0, 117.0, 127.0],
            "volume": [1000, 1100, 1200, 1150, 1250],
        }, index=idx)

    def _make_usdtry(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
        return pd.Series([30.0, 31.0, 32.0, 31.5, 33.0], index=idx, name="usdtry")

    def test_convert_series_divides_ohlc(self):
        from bist.adapters.price_converter import PriceConverter
        pc = PriceConverter(mode="convert_series")
        df = self._make_df()
        rate = self._make_usdtry()
        result = pc.convert_ohlcv(df, rate)
        assert result["close"].iloc[0] == pytest.approx(102.0 / 30.0)
        assert result["volume"].iloc[0] == 1000  # volume değişmemeli

    def test_convert_pnl_mode_unchanged(self):
        from bist.adapters.price_converter import PriceConverter
        pc = PriceConverter(mode="convert_pnl")
        df = self._make_df()
        rate = self._make_usdtry()
        result = pc.convert_ohlcv(df, rate)
        assert result["close"].iloc[0] == pytest.approx(102.0)

    def test_invalid_mode_raises(self):
        from bist.adapters.price_converter import PriceConverter
        with pytest.raises(ValueError):
            PriceConverter(mode="invalid")

    def test_convert_price(self):
        from bist.adapters.price_converter import PriceConverter
        pc = PriceConverter(mode="convert_series")
        assert pc.convert_price(310.0, 31.0) == pytest.approx(10.0)

    def test_convert_pnl_to_usd(self):
        from bist.adapters.price_converter import PriceConverter
        pc = PriceConverter()
        assert pc.convert_pnl_to_usd(310.0, 31.0) == pytest.approx(10.0)

    def test_missing_usdtry_dates_filled(self):
        """Haftasonu boşlukları forward-fill ile doldurulmalı."""
        from bist.adapters.price_converter import PriceConverter
        idx = pd.date_range("2024-01-01", periods=7, freq="D", tz="UTC")
        df = pd.DataFrame({
            "open": [100.0] * 7, "high": [105.0] * 7,
            "low":  [95.0] * 7,  "close": [102.0] * 7,
            "volume": [1000] * 7,
        }, index=idx)
        # USDTRY sadece iş günleri (5 gün)
        rate_idx = pd.date_range("2024-01-01", periods=5, freq="B", tz="UTC")
        rate = pd.Series([30.0, 31.0, 32.0, 31.5, 33.0], index=rate_idx, name="usdtry")
        pc = PriceConverter(mode="convert_series")
        result = pc.convert_ohlcv(df, rate)
        assert not result["close"].isna().any()


# ── TickLot ───────────────────────────────────────────────────────────────────
class TestTickLot:
    def test_tick_size(self):
        from bist.adapters.tick_lot import get_tick_size
        assert get_tick_size(5.0) == 0.01
        assert get_tick_size(100.0) == 0.01
        assert get_tick_size(1000.0) == 0.01

    def test_round_to_tick(self):
        from bist.adapters.tick_lot import round_to_tick
        assert round_to_tick(10.004) == pytest.approx(10.0)   # 10.004 → aşağı yuvarla
        assert round_to_tick(10.016) == pytest.approx(10.02)  # 10.016 → yukarı yuvarla

    def test_round_to_lot(self):
        from bist.adapters.tick_lot import round_to_lot
        assert round_to_lot(4.9) == 4
        assert round_to_lot(0.5) == 0
        assert round_to_lot(100.0) == 100

    def test_commission_includes_bsmv(self):
        from bist.adapters.tick_lot import calculate_commission
        base = 10_000.0 * 0.0005   # 5 TL komisyon
        bsmv = base * 0.05          # 0.25 TL BSMV
        expected = base + bsmv + 10_000.0 * 0.0000001
        assert calculate_commission(10_000.0) == pytest.approx(expected, rel=1e-5)


# ── MarketHours ───────────────────────────────────────────────────────────────
class TestMarketHours:
    def test_weekday_open(self):
        from bist.adapters.market_hours import is_market_open
        # Pazartesi 12:00 UTC = 15:00 Istanbul → açık
        dt = datetime(2024, 1, 8, 9, 0, tzinfo=timezone.utc)   # 12:00 Istanbul
        assert is_market_open(dt)

    def test_weekend_closed(self):
        from bist.adapters.market_hours import is_market_open
        # Cumartesi
        dt = datetime(2024, 1, 6, 10, 0, tzinfo=timezone.utc)
        assert not is_market_open(dt)

    def test_before_open_closed(self):
        from bist.adapters.market_hours import is_market_open
        # Pazartesi 06:00 UTC = 09:00 Istanbul → kapalı (10:00 açılıyor)
        dt = datetime(2024, 1, 8, 6, 0, tzinfo=timezone.utc)
        assert not is_market_open(dt)

    def test_after_close_closed(self):
        from bist.adapters.market_hours import is_market_open
        # Pazartesi 16:00 UTC = 19:00 Istanbul → kapalı
        dt = datetime(2024, 1, 8, 16, 0, tzinfo=timezone.utc)
        assert not is_market_open(dt)

    def test_trading_day_weekday(self):
        from bist.adapters.market_hours import is_trading_day
        assert is_trading_day(date(2024, 1, 8))   # Pazartesi

    def test_trading_day_weekend(self):
        from bist.adapters.market_hours import is_trading_day
        assert not is_trading_day(date(2024, 1, 6))  # Cumartesi

    def test_holiday_not_trading(self):
        from bist.adapters.market_hours import is_trading_day
        assert not is_trading_day(date(2024, 10, 29))  # Cumhuriyet Bayramı


# ── MacroFilter ───────────────────────────────────────────────────────────────
class TestMacroFilter:
    def _xu100_bull(self) -> pd.Series:
        idx = pd.date_range("2023-01-01", periods=300, freq="D", tz="UTC")
        # Sürekli yükselen → EMA altındaysa bear yok
        vals = [100.0 + i * 0.5 for i in range(300)]
        return pd.Series(vals, index=idx, name="xu100")

    def _xu100_bear(self) -> pd.Series:
        idx = pd.date_range("2023-01-01", periods=300, freq="D", tz="UTC")
        # Düşen → zamanla EMA altına geçer
        vals = [200.0 - i * 0.5 for i in range(300)]
        return pd.Series(vals, index=idx, name="xu100")

    def _usdtry_stable(self) -> pd.Series:
        idx = pd.date_range("2023-01-01", periods=300, freq="D", tz="UTC")
        return pd.Series([30.0] * 300, index=idx, name="usdtry")

    def _usdtry_crisis(self) -> pd.Series:
        idx = pd.date_range("2023-01-01", periods=300, freq="D", tz="UTC")
        # İlk 150 gün sabit, sonra günde %1 artış → 20 günde ~%22 (kriz)
        vals = [30.0] * 150 + [30.0 * (1.01 ** i) for i in range(150)]
        return pd.Series(vals, index=idx, name="usdtry")

    def test_bull_regime_allows_entry(self):
        from bist.filters.macro_filter import MacroFilter
        mf = MacroFilter(usdtry_guard_enabled=False)
        mf.fit(self._xu100_bull(), self._usdtry_stable())
        ts = pd.Timestamp("2023-11-01", tz="UTC")
        assert mf.is_entry_allowed(ts)

    def test_bear_regime_blocks_entry(self):
        from bist.filters.macro_filter import MacroFilter
        mf = MacroFilter(usdtry_guard_enabled=False)
        mf.fit(self._xu100_bear(), self._usdtry_stable())
        ts = pd.Timestamp("2023-11-01", tz="UTC")
        assert not mf.is_entry_allowed(ts)

    def test_usdtry_crisis_blocks_entry(self):
        from bist.filters.macro_filter import MacroFilter
        mf = MacroFilter(regime_filter_enabled=False)
        mf.fit(self._xu100_bull(), self._usdtry_crisis())
        # Kriz döneminin ortasında
        ts = pd.Timestamp("2023-10-01", tz="UTC")
        assert not mf.is_entry_allowed(ts)

    def test_disabled_filters_allow_all(self):
        from bist.filters.macro_filter import MacroFilter
        mf = MacroFilter(regime_filter_enabled=False, usdtry_guard_enabled=False)
        mf.fit(self._xu100_bear(), self._usdtry_crisis())
        ts = pd.Timestamp("2023-11-01", tz="UTC")
        assert mf.is_entry_allowed(ts)

    def test_get_block_reason_bear(self):
        from bist.filters.macro_filter import MacroFilter
        mf = MacroFilter(usdtry_guard_enabled=False)
        mf.fit(self._xu100_bear(), self._usdtry_stable())
        ts = pd.Timestamp("2023-11-01", tz="UTC")
        reason = mf.get_block_reason(ts)
        assert "bear" in reason.lower() or "EMA" in reason

    def test_summary_stats(self):
        from bist.filters.macro_filter import MacroFilter
        mf = MacroFilter()
        mf.fit(self._xu100_bull(), self._usdtry_stable())
        stats = mf.summary_stats()
        assert "regime_bull_pct" in stats
        assert 0 <= stats["regime_bull_pct"] <= 100

    def test_fail_open_no_data(self):
        from bist.filters.macro_filter import MacroFilter
        mf = MacroFilter()
        # fit çağrılmadan
        ts = pd.Timestamp("2023-11-01", tz="UTC")
        assert mf.is_entry_allowed(ts)
