"""
BIST makro rejim filtresi — kripto bottaki BTC regime filter'ın BIST karşılığı.

İki filtre:
  1. XU100 200-day EMA → bull/bear rejim
  2. USDTRY 20-day momentum → TRY kriz koruması
"""
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class MacroFilter:
    """
    BIST endeks rejim + USDTRY kriz filtresi.

    Kullanım:
        mf = MacroFilter()
        mf.fit(xu100_series, usdtry_series)
        allowed = mf.is_entry_allowed(current_date)
    """

    def __init__(
        self,
        regime_filter_enabled: bool = True,
        regime_ema_period: int = 200,
        regime_usd_adjusted: bool = False,  # True → XU100/USDTRY (USD-cinsinden)
        usdtry_guard_enabled: bool = True,
        usdtry_momentum_period: int = 20,
        usdtry_crisis_threshold: float = 0.15,
    ):
        self.regime_filter_enabled = regime_filter_enabled
        self.regime_ema_period = regime_ema_period
        self.regime_usd_adjusted = regime_usd_adjusted
        self.usdtry_guard_enabled = usdtry_guard_enabled
        self.usdtry_momentum_period = usdtry_momentum_period
        self.usdtry_crisis_threshold = usdtry_crisis_threshold

        self._regime_series: Optional[pd.Series] = None
        self._usdtry_crisis: Optional[pd.Series] = None
        self._xu100_usd: Optional[pd.Series] = None

    def fit(self, xu100: pd.Series, usdtry: pd.Series) -> "MacroFilter":
        """
        Tüm tarihler için rejim ve kriz sinyallerini önceden hesaplar.
        is_entry_allowed() çağrısından önce mutlaka çağrılmalı.
        """
        if self.regime_filter_enabled and xu100 is not None and len(xu100) > 0:
            if self.regime_usd_adjusted and usdtry is not None and len(usdtry) > 0:
                # XU100 USD = XU100 TRY / USDTRY → TRY değer kaybını filtreler
                usdtry_aligned = usdtry.reindex(xu100.index, method="ffill").bfill()
                xu100_usd = xu100 / usdtry_aligned
                base = xu100_usd
                label = "XU100-USD"
            else:
                base = xu100
                label = "XU100-TRY"
            ema = base.ewm(span=self.regime_ema_period, adjust=False).mean()
            above_ema = base > ema
            # İkinci onay: 20-günlük momentum da pozitif olmalı (sahte bull sinyallerini azaltır)
            momentum_pos = base.pct_change(20) > 0
            self._regime_series = above_ema & momentum_pos
            bull_pct = self._regime_series.mean() * 100
            logger.info(
                f"[MacroFilter] {label} rejim: {bull_pct:.1f}% bull "
                f"({self._regime_series.sum()}/{len(self._regime_series)} bar)"
            )
            # xu100_usd'yi sakla (relative strength için)
            if self.regime_usd_adjusted and usdtry is not None and len(usdtry) > 0:
                self._xu100_usd = base  # base zaten xu100/usdtry
            else:
                self._xu100_usd = None

        if self.usdtry_guard_enabled and usdtry is not None and len(usdtry) > 0:
            momentum = usdtry.pct_change(self.usdtry_momentum_period)
            self._usdtry_crisis = momentum > self.usdtry_crisis_threshold
            crisis_pct = self._usdtry_crisis.mean() * 100
            logger.info(
                f"[MacroFilter] USDTRY kriz guard: {crisis_pct:.1f}% kriz bar "
                f"(eşik: {self.usdtry_crisis_threshold:.0%})"
            )

        return self

    def is_entry_allowed(self, current_date: pd.Timestamp) -> bool:
        """
        current_date'de yeni long giriş yapılabilir mi?
        Filtre devre dışıysa veya veri yoksa True döner (fail-open).
        """
        if self.regime_filter_enabled and self._regime_series is not None:
            val = self._regime_series.asof(current_date)
            if not pd.isna(val) and not bool(val):
                return False

        if self.usdtry_guard_enabled and self._usdtry_crisis is not None:
            val = self._usdtry_crisis.asof(current_date)
            if not pd.isna(val) and bool(val):
                return False

        return True

    def get_block_reason(self, current_date: pd.Timestamp) -> str:
        """Giriş engelleniyorsa nedeni döndürür, aksi halde boş string."""
        if self.regime_filter_enabled and self._regime_series is not None:
            val = self._regime_series.asof(current_date)
            if not pd.isna(val) and not bool(val):
                return "XU100 bear market (200-day EMA altında)"

        if self.usdtry_guard_enabled and self._usdtry_crisis is not None:
            val = self._usdtry_crisis.asof(current_date)
            if not pd.isna(val) and bool(val):
                return (
                    f"USDTRY kriz aktif "
                    f"({self.usdtry_momentum_period}g momentum > "
                    f"{self.usdtry_crisis_threshold:.0%})"
                )

        return ""

    def get_relative_strength_series(
        self, stock_prices: pd.Series, lookback: int = 20
    ) -> pd.Series:
        """
        Hisse XU100'ü outperform ediyor mu?
        stock_prices: USD cinsinden hisse fiyatı
        Döndürür: True = hisse XU100'den güçlü (relative outperformance)
        """
        if self._regime_series is None:
            return pd.Series(True, index=stock_prices.index)

        # Hisse 20-günlük momentum
        stock_mom = stock_prices.pct_change(lookback)
        # XU100 USD 20-günlük momentum (eğer hesaplanmışsa)
        if self._xu100_usd is not None:
            xu100_mom = self._xu100_usd.pct_change(lookback)
            xu100_aligned = xu100_mom.reindex(stock_prices.index, method="ffill")
            return stock_mom > xu100_aligned  # hisse endeksten güçlü
        else:
            return stock_mom > -0.05  # en azından son 20 günde %5'ten fazla düşmemiş

    def get_regime_series(self) -> Optional[pd.Series]:
        return self._regime_series

    def get_crisis_series(self) -> Optional[pd.Series]:
        return self._usdtry_crisis

    def summary_stats(self) -> dict:
        stats: dict = {}
        if self._regime_series is not None:
            stats["regime_bull_pct"] = round(self._regime_series.mean() * 100, 1)
        if self._usdtry_crisis is not None:
            stats["usdtry_crisis_pct"] = round(self._usdtry_crisis.mean() * 100, 1)
        return stats


def fetch_xu100(start: str, end: str) -> pd.Series:
    """
    BIST 100 endeksini Yahoo Finance'ten çeker.
    Yahoo ticker: "XU100.IS"
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance gerekli. 'pip install yfinance' ile kurun."
        ) from exc

    raw = yf.download("XU100.IS", start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"XU100.IS için veri gelmedi: {start} → {end}")

    if hasattr(raw.columns, "droplevel"):
        raw.columns = raw.columns.droplevel(1) if isinstance(raw.columns, pd.MultiIndex) else raw.columns

    series = raw["Close"].squeeze()
    series.index = pd.to_datetime(series.index, utc=True)
    series.name = "xu100"
    return series
