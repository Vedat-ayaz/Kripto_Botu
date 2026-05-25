"""
Teknik göstergeler — saf pandas/numpy, harici bağımlılık yok.
Eklenenler: MACD, Bollinger Bands, Stochastic RSI, OBV, Hurst exponent,
            Williams %R, Donchian kanalları, piyasa rejimi skoru.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ─── Temel hesaplama fonksiyonları ───────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_high = high.shift(1)
    prev_low  = low.shift(1)
    up_move   = high - prev_high
    down_move = prev_low - low
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_vals  = _atr(high, low, close, period)
    plus_di   = 100 * pd.Series(plus_dm, index=high.index).ewm(
        com=period - 1, min_periods=period).mean() / atr_vals
    minus_di  = 100 * pd.Series(minus_dm, index=high.index).ewm(
        com=period - 1, min_periods=period).mean() / atr_vals
    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(com=period - 1, min_periods=period).mean()


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD, signal ve histogram döner."""
    ema_fast   = _ema(close, fast)
    ema_slow   = _ema(close, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Üst, orta, alt bant + %B + bant genişliği döner."""
    mid   = close.rolling(period).mean()
    std   = close.rolling(period).std(ddof=0)
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    width = (upper - lower) / mid.replace(0, np.nan)
    return upper, mid, lower, pct_b, width


def _stoch_rsi(close: pd.Series, rsi_period: int = 14,
               stoch_period: int = 14, smooth_k: int = 3, smooth_d: int = 3):
    """Stochastic RSI: %K ve %D döner."""
    rsi    = _rsi(close, rsi_period)
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    stoch  = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan) * 100
    k      = stoch.rolling(smooth_k).mean()
    d      = k.rolling(smooth_d).mean()
    return k, d


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R — aşırı alım/satım."""
    highest = high.rolling(period).max()
    lowest  = low.rolling(period).min()
    return -100 * (highest - close) / (highest - lowest).replace(0, np.nan)


def _donchian(high: pd.Series, low: pd.Series, period: int = 20):
    """Donchian kanalı üst, orta, alt döner."""
    upper = high.rolling(period).max()
    lower = low.rolling(period).min()
    mid   = (upper + lower) / 2
    return upper, mid, lower


def _hurst_exponent(series: pd.Series, min_lag: int = 2, max_lag: int = 20) -> pd.Series:
    """
    Hurst üstel katsayısı (R/S analizi).
    H > 0.55  → trend piyasası (momentum geçerli)
    H < 0.45  → mean-reversion piyasası
    0.45–0.55 → rastgele yürüyüş
    Rolling pencere üzerinden hesaplanır.
    """
    window = max_lag * 2

    def _hurst_single(arr: np.ndarray) -> float:
        lags = range(min_lag, min(max_lag, len(arr) // 2))
        tau  = []
        for lag in lags:
            diff = arr[lag:] - arr[:-lag]
            if len(diff) > 1:
                tau.append(np.sqrt(np.std(diff, ddof=1)))
        if len(tau) < 2:
            return 0.5
        lags_arr = np.log(np.arange(min_lag, min_lag + len(tau)))
        tau_arr  = np.log(np.array(tau) + 1e-10)
        if lags_arr.std() == 0:
            return 0.5
        return float(np.polyfit(lags_arr, tau_arr, 1)[0])

    # raw=True → x numpy array olarak gelir (.values gerekmez)
    return series.rolling(window).apply(
        lambda x: _hurst_single(x), raw=True
    )


def _choppiness_index(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Choppiness Index (Dreiss, 1991) — piyasanın trending mi yoksa choppy mi olduğunu ölçer.

    Formül: CI = 100 × log10(sum(ATR, n) / (HH(n) - LL(n))) / log10(n)

    Yorumlama:
      CI > 61.8  → Choppy/sideways piyasa (Fibonacci eşiği) — TREND-FOLLOWING ÖLDÜRÜR
      CI < 38.2  → Strong trending piyasa
      38.2-61.8  → Geçiş zonu

    Trend-following bot için: CI > 61.8 olduğunda yeni entry blokla.
    Akademik referans: Hurst, Ooi, Pedersen (AQR 2017) — trend-following P&L'in %80'i
    %20 zaman diliminde yapılır; geri kalan zaman "no-trade" rejimidir.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr_sum = tr.rolling(period).sum()
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    range_n = (hh - ll).replace(0, np.nan)
    ci = 100 * np.log10(atr_sum / range_n) / np.log10(period)
    return ci


def _momentum_score(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Zaman serisi momentum skoru (Moskowitz et al., 2012 TSMOM).
    Son dönem getirisi normalize edilmiş — cross-sectional sıralama için.
    """
    ret = close.pct_change(period)
    vol = close.pct_change().rolling(period).std()
    return ret / vol.replace(0, np.nan)


# ─── Ana sınıf ────────────────────────────────────────────────────────────────

class TechnicalIndicators:
    """
    Tüm teknik göstergeleri hesaplar.
    DataFrame'e yeni sütunlar ekler, orjinali değiştirmez.

    Yeni göstergeler:
      macd, macd_signal, macd_hist
      bb_upper, bb_mid, bb_lower, bb_pct_b, bb_width
      stoch_k, stoch_d
      obv, obv_ema
      williams_r
      donchian_upper, donchian_mid, donchian_lower
      hurst          (piyasa rejimi: >0.55 trend, <0.45 mean-rev)
      regime_score   (0=ranging, 1=trending — ADX + Hurst birleşimi)
      tsmom          (time-series momentum skoru)
    """

    def __init__(
        self,
        ema_fast: int = 50,
        ema_slow: int = 200,
        rsi_period: int = 14,
        atr_period: int = 14,
        adx_period: int = 14,
        volume_sma_period: int = 20,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        donchian_period: int = 20,
        tsmom_period: int = 20,
    ):
        self.ema_fast          = ema_fast
        self.ema_slow          = ema_slow
        self.rsi_period        = rsi_period
        self.atr_period        = atr_period
        self.adx_period        = adx_period
        self.volume_sma_period = volume_sma_period
        self.macd_fast         = macd_fast
        self.macd_slow         = macd_slow
        self.macd_signal_period = macd_signal
        self.bb_period         = bb_period
        self.bb_std            = bb_std
        self.donchian_period   = donchian_period
        self.tsmom_period      = tsmom_period

    def add_higher_timeframe(
        self,
        df_1h: pd.DataFrame,
        htf_rule: str = "4h",
        ema_period: int = 50,
        adx_period: int = 14,
    ) -> pd.DataFrame:
        """
        1h DataFrame'i alır, üst timeframe (varsayılan 4h) trend göstergelerini
        hesaplar ve ffill ile 1h indeksine geri yayar.

        Eklenir:
          htf_close, htf_ema_fast, htf_ema_slope, htf_adx, htf_trend_up (bool)

        Multi-timeframe confirmation (Elder Triple Screen, 1986):
        Üst TF trendinin yönünde olan alt TF sinyalleri 1.3-1.8× tahmin gücüne sahip.

        htf_trend_up koşulu:
          close > htf_ema_fast VE htf_ema_slope > 0 (yükselen) VE htf_adx > 18 (gerçek trend)
        """
        out = df_1h.copy()

        # 1h → 4h resample (OHLCV)
        htf = df_1h[["open", "high", "low", "close", "volume"]].resample(htf_rule).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()

        if len(htf) < ema_period + adx_period:
            # Yeterli üst TF verisi yok → trend kabul (filtre devre dışı)
            out["htf_close"]      = float("nan")
            out["htf_ema_fast"]   = float("nan")
            out["htf_ema_slope"]  = 0.0
            out["htf_adx"]        = 0.0
            out["htf_trend_up"]   = True
            return out

        htf_ema = _ema(htf["close"], ema_period)
        # Slope: EMA'nın 5-bar değişimi (yön)
        htf_ema_slope = htf_ema - htf_ema.shift(5)
        htf_adx = _adx(htf["high"], htf["low"], htf["close"], adx_period)

        # 1h indeksine ffill (üst TF bar kapanırken günceller, kapanmamış bar için son geçerli)
        out["htf_close"]      = htf["close"].reindex(out.index, method="ffill")
        out["htf_ema_fast"]   = htf_ema.reindex(out.index, method="ffill")
        out["htf_ema_slope"]  = htf_ema_slope.reindex(out.index, method="ffill")
        out["htf_adx"]        = htf_adx.reindex(out.index, method="ffill")

        # Trend yukarı mı? (3 koşul birden)
        out["htf_trend_up"] = (
            (out["close"] > out["htf_ema_fast"]) &
            (out["htf_ema_slope"] > 0) &
            (out["htf_adx"] > 18)
        ).fillna(True)  # NaN olduğunda izin ver (warmup koruması)

        return out

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame şu sütunları içermeli: {required}")

        out = df.copy()

        # ── Temel göstergeler ────────────────────────────────────────
        out[self.ema_fast_col] = _ema(out["close"], self.ema_fast)
        out[self.ema_slow_col] = _ema(out["close"], self.ema_slow)
        out["rsi"]             = _rsi(out["close"], self.rsi_period)
        out["atr"]             = _atr(out["high"], out["low"], out["close"], self.atr_period)
        out["atr_sma_20"]      = out["atr"].rolling(20).mean()   # ATR 20-bar ortalaması (volatilite rejimi)
        out["adx"]             = _adx(out["high"], out["low"], out["close"], self.adx_period)
        out["volume_sma"]      = out["volume"].rolling(self.volume_sma_period).mean()
        out["atr_ratio"]       = out["atr"] / out["close"]

        # ── MACD ─────────────────────────────────────────────────────
        out["macd"], out["macd_signal"], out["macd_hist"] = _macd(
            out["close"], self.macd_fast, self.macd_slow, self.macd_signal_period
        )

        # ── Bollinger Bands ───────────────────────────────────────────
        (out["bb_upper"], out["bb_mid"],
         out["bb_lower"], out["bb_pct_b"],
         out["bb_width"]) = _bollinger(out["close"], self.bb_period, self.bb_std)

        # ── Stochastic RSI ────────────────────────────────────────────
        out["stoch_k"], out["stoch_d"] = _stoch_rsi(out["close"])

        # ── OBV + OBV EMA (trend teyidi) ─────────────────────────────
        out["obv"]     = _obv(out["close"], out["volume"])
        out["obv_ema"] = _ema(out["obv"], 20)

        # ── Williams %R ───────────────────────────────────────────────
        out["williams_r"] = _williams_r(out["high"], out["low"], out["close"])

        # ── Donchian Kanalları ────────────────────────────────────────
        (out["donchian_upper"],
         out["donchian_mid"],
         out["donchian_lower"]) = _donchian(out["high"], out["low"], self.donchian_period)

        # ── Hurst Üstel Katsayısı (piyasa rejimi) ────────────────────
        if len(out) >= 40:
            out["hurst"] = _hurst_exponent(out["close"])
        else:
            out["hurst"] = 0.5

        # ── Piyasa Rejimi Skoru (0=ranging, 1=trending) ───────────────
        # ADX > 25 → trending, Hurst > 0.55 → trending
        adx_score   = (out["adx"] - 15).clip(0, 25) / 25        # 0–1
        hurst_score = (out["hurst"] - 0.45).clip(0, 0.20) / 0.20  # 0–1
        out["regime_score"] = (adx_score * 0.6 + hurst_score * 0.4).clip(0, 1)

        # ── Time-Series Momentum (Moskowitz 2012) ─────────────────────
        out["tsmom"] = _momentum_score(out["close"], self.tsmom_period)

        # ── Choppiness Index (Dreiss, anti-whipsaw filter) ───────────
        # CI > 61.8 → choppy/sideways → trend-following bu rejimde ölür
        out["choppiness"] = _choppiness_index(out["high"], out["low"], out["close"], period=14)

        return out

    @property
    def ema_fast_col(self) -> str:
        return f"ema_{self.ema_fast}"

    @property
    def ema_slow_col(self) -> str:
        return f"ema_{self.ema_slow}"
