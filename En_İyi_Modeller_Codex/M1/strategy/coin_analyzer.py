"""
Coin Analyzer — Coin Karakterizasyon ve Dinamik Parametre Önerisi
==================================================================
Son N günlük veriyi istatistiksel olarak analiz eder:
  - ATR volatilite rejimi
  - ADX trend kalitesi
  - Hurst trendi kalıcılığı
  - Fiyat momentum yönü
  - Hacim likiditesi

Bu analize göre:
  1. trade_score: 0-1 → o an işlem için ne kadar uygun
  2. suggest_params: başlangıç parametresi tahmini (WFO için warm-start)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CoinAnalyzer:
    """
    Analyzes a coin's recent market behavior to produce a tradability score
    and suggest initial strategy parameters for Walk-Forward Optimization.

    All methods accept a DataFrame with the standard indicator columns produced
    by TechnicalIndicators.calculate(): close, high, low, open, volume,
    ema_50, ema_200, rsi, atr, adx, macd, macd_hist, bb_pct_b, bb_width,
    stoch_k, obv, obv_ema, hurst, regime_score, tsmom, choppiness.
    """

    # Minimum bar sayısı — bu eşiğin altında analiz güvenilmez
    MIN_BARS = 300

    def __init__(self) -> None:
        pass

    # ──────────────────────────────────────────────────────────────────
    # Yardımcı metotlar
    # ──────────────────────────────────────────────────────────────────

    def _slice(self, df: pd.DataFrame, lookback_bars: int) -> pd.DataFrame:
        """Son lookback_bars barı döner; eğer df daha kısaysa tamamını döner."""
        return df.iloc[-lookback_bars:] if len(df) > lookback_bars else df

    def _atr_pct(self, df: pd.DataFrame) -> float:
        """
        Returns the median ATR as a percentage of close price.
        Median, ortalamadan daha sağlam (outlier'lara duyarsız).
        """
        if "atr" not in df.columns or "close" not in df.columns:
            return 0.0
        close = df["close"].replace(0, np.nan)
        atr_pct_series = df["atr"] / close * 100.0
        val = float(atr_pct_series.median())
        return val if np.isfinite(val) else 0.0

    def _adx_stats(self, df: pd.DataFrame) -> tuple[float, float]:
        """
        Returns (adx_median, adx_q75) for the slice.
        Q75 = üst çeyrek ADX → piyasanın en trend olduğu anların kalitesi.
        """
        if "adx" not in df.columns:
            return 0.0, 0.0
        adx = df["adx"].dropna()
        if adx.empty:
            return 0.0, 0.0
        return float(adx.median()), float(adx.quantile(0.75))

    def _hurst_median(self, df: pd.DataFrame) -> float:
        """
        Returns median Hurst exponent from the slice.
        Hurst > 0.5 → trendin devam etme eğilimi (persistence).
        Hurst < 0.5 → mean-reversion eğilimi.
        """
        if "hurst" not in df.columns:
            return 0.5
        hurst = df["hurst"].dropna()
        if hurst.empty:
            return 0.5
        val = float(hurst.median())
        return val if np.isfinite(val) else 0.5

    def _volume_stability(self, df: pd.DataFrame) -> float:
        """
        Returns a volume stability score in [0, 1].
        Stable volume (low CV = coefficient of variation) → higher score.
        Yüksek skor: hacim tutarlı → likidite güvenilir.
        """
        if "volume" not in df.columns:
            return 0.5
        vol = df["volume"].dropna().replace(0, np.nan).dropna()
        if vol.empty or vol.mean() == 0:
            return 0.0
        cv = vol.std() / vol.mean()  # coefficient of variation
        # CV < 0.5 çok stabil, CV > 2.0 çok değişken
        score = float(np.clip(1.0 - cv / 2.5, 0.0, 1.0))
        return score if np.isfinite(score) else 0.3

    def _directional_bias(self, df: pd.DataFrame) -> float:
        """
        Returns a directional bias score in [-1, 1].
        +1 → güçlü yükseliş trendi, -1 → güçlü düşüş trendi, 0 → yatay.
        Kullanım: mutlak değer büyük → güçlü yön (LONG veya SHORT için iyi).
        """
        if "close" not in df.columns or len(df) < 10:
            return 0.0
        close = df["close"].dropna()
        if close.empty:
            return 0.0
        # Toplam fiyat değişimi normalize edilmiş
        total_ret = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
        # [-1, +1] aralığına sıkıştır (±%30 tam doyuma yeterli)
        bias = float(np.clip(total_ret / 0.30, -1.0, 1.0))
        return bias if np.isfinite(bias) else 0.0

    # ──────────────────────────────────────────────────────────────────
    # Ana yöntemler
    # ──────────────────────────────────────────────────────────────────

    def score_for_trading(self, df: pd.DataFrame, lookback_bars: int = 200 * 24) -> float:
        """
        Analyzes the coin's recent price history and returns a tradability
        score between 0.0 and 1.0. Higher is better.

        Scoring factors:
          - ADX quality   : 20-40 sweet spot (hem trend var hem de exhaustion yok)
          - ATR%          : 1-5% hourly ideal (volatilite yeterli ama aşırı değil)
          - Hurst exponent: > 0.48 → trending tendency (persistence)
          - Volume stability: stable volume → reliable liquidity

        Args:
            df           : Full indicator DataFrame.
            lookback_bars: Kaç bar geriye bakılacak (default 200 gün × 24 saat).

        Returns:
            float: 0.0 – 1.0 arası işlem skoru.
        """
        # Yeterli veri yoksa sıfır döndür
        if len(df) < self.MIN_BARS:
            logger.debug(
                f"[CoinAnalyzer] Yetersiz veri: {len(df)} bar < {self.MIN_BARS} minimum"
            )
            return 0.0

        sliced = self._slice(df, lookback_bars)

        # ── ADX kalite bileşeni ──────────────────────────────────────
        # Hedef: 20-40 arası ADX → 1.0, 15-20 → 0.6, 40+ → 0.7, <15 → 0.2
        adx_med, adx_q75 = self._adx_stats(sliced)
        if 20 <= adx_med <= 40:
            adx_score = 1.0
        elif 15 <= adx_med < 20:
            adx_score = 0.6
        elif adx_med > 40:
            # Aşırı trended → yakında exhaustion riski, biraz düşür
            adx_score = 0.7
        else:
            # ADX < 15 → ranging piyasa, trend-following için zayıf
            adx_score = 0.2
        # Q75 >= 25 ekstra bonus: piyasa zaman zaman güçlü trendler üretiyor
        if adx_q75 >= 25:
            adx_score = min(adx_score + 0.1, 1.0)

        # ── ATR% bileşeni ────────────────────────────────────────────
        # Saatlik: 1-5% ideal, <1% çok düşük volatilite, >5% aşırı riskli
        atr_pct = self._atr_pct(sliced)
        if 1.0 <= atr_pct <= 5.0:
            # 2-3% tam orta → 1.0 puan; uçlara doğru lineer düşüş
            atr_score = 1.0 - abs(atr_pct - 3.0) / 4.0
            atr_score = float(np.clip(atr_score, 0.5, 1.0))
        elif atr_pct < 1.0:
            # Çok düşük volatilite → stop tetiklenmez, sinyal yoktur
            atr_score = max(0.1, atr_pct / 1.0 * 0.5)
        else:
            # ATR% > 5% → çok riskli; 10%'de tam sıfır
            atr_score = float(np.clip(1.0 - (atr_pct - 5.0) / 5.0, 0.0, 0.5))

        # ── Hurst bileşeni ───────────────────────────────────────────
        # > 0.55 → güçlü persistence → 1.0
        # 0.48-0.55 → hafif persistence → 0.6-0.9
        # < 0.48 → mean-reversion veya random walk → 0.2
        hurst = self._hurst_median(sliced)
        if hurst > 0.55:
            hurst_score = 1.0
        elif hurst >= 0.48:
            # 0.48-0.55 arası lineer ölçek
            hurst_score = 0.2 + (hurst - 0.48) / (0.55 - 0.48) * 0.8
        else:
            # Trending yok → trend-following için çok zayıf
            hurst_score = 0.2

        # ── Hacim likiditesi bileşeni ────────────────────────────────
        vol_score = self._volume_stability(sliced)

        # ── Bileşik ağırlıklı skor ───────────────────────────────────
        # ADX kalitesi en kritik (trend var mı?), ardından volatilite uygunluğu
        weights = {
            "adx":    0.35,
            "atr":    0.30,
            "hurst":  0.25,
            "volume": 0.10,
        }
        total = (
            adx_score    * weights["adx"]
            + atr_score  * weights["atr"]
            + hurst_score * weights["hurst"]
            + vol_score  * weights["volume"]
        )
        return round(float(np.clip(total, 0.0, 1.0)), 4)

    def suggest_params(self, df: pd.DataFrame, lookback_bars: int = 200 * 24) -> dict:
        """
        Analyzes the coin's recent behavior and suggests an initial parameter
        set to use as a warm-start for Walk-Forward Optimization.

        Returned keys match TrendFollowingStrategy constructor arguments:
          adx_threshold, entry_score_trend, entry_score_ranging,
          trailing_stop_atr_multiplier, atr_stop_multiplier, max_position_pct

        Logic summary:
          - ATR% drives stop width and position sizing
          - ADX Q75 drives entry strictness
          - Hurst < 0.45 → very conservative params
          - Strong directional trend → wider trailing stop

        Args:
            df           : Full indicator DataFrame.
            lookback_bars: Kaç bar geriye bakılacak.

        Returns:
            dict: Önerilen parametre sözlüğü.
        """
        # Yeterli veri yoksa güvenli varsayılan döndür
        if len(df) < self.MIN_BARS:
            return self._default_params()

        sliced = self._slice(df, lookback_bars)

        atr_pct           = self._atr_pct(sliced)
        adx_med, adx_q75  = self._adx_stats(sliced)
        hurst             = self._hurst_median(sliced)
        direction         = self._directional_bias(sliced)

        # ── ATR% → stop genişliği ve pozisyon boyutu ─────────────────
        # Düşük ATR: dar stop, orta pozisyon
        # Yüksek ATR: geniş stop (aşılmaması için), küçük pozisyon (risk kontrolü)
        if atr_pct < 1.0:
            atr_stop_mult  = 2.0
            trail_mult     = 4.0
            max_pos_pct    = 0.04
        elif atr_pct < 3.0:
            atr_stop_mult  = 2.5
            trail_mult     = 5.0
            max_pos_pct    = 0.06
        elif atr_pct < 5.0:
            atr_stop_mult  = 3.0
            trail_mult     = 6.0
            max_pos_pct    = 0.04
        else:
            # ATR% > 5%: aşırı volatil → geniş stop zorunlu, küçük pozisyon
            atr_stop_mult  = 3.5
            trail_mult     = 7.0
            max_pos_pct    = 0.03

        # ── ADX Q75 → giriş katılığı ─────────────────────────────────
        # Yüksek ADX Q75 → piyasa sık sık güçlü trend üretiyor → eşiği düşür
        if adx_q75 > 30:
            adx_threshold    = 22
            entry_score_trend = 0.60
        elif adx_q75 >= 20:
            adx_threshold    = 25
            entry_score_trend = 0.65
        else:
            # ADX seyrek veya zayıf → daha seçici ol
            adx_threshold    = 28
            entry_score_trend = 0.70

        # Ranging eşiği her zaman trend eşiğinden biraz daha yüksek
        entry_score_ranging = round(entry_score_trend + 0.05, 2)

        # ── Hurst < 0.45 → çok tutucu parametreler ───────────────────
        # Trend kalıcılığı neredeyse yok → küçük pozisyon, dar stop
        if hurst < 0.45:
            max_pos_pct       = min(max_pos_pct, 0.03)
            entry_score_trend = min(entry_score_trend + 0.05, 0.80)
            entry_score_ranging = min(entry_score_ranging + 0.05, 0.85)
            logger.debug(
                f"[CoinAnalyzer] Düşük Hurst ({hurst:.3f}) → tutucu parametreler"
            )

        # ── Güçlü yön (yüksek directional bias) → daha geniş trailing ─
        # Güçlü tek yönlü trendlerde trailing'i erken kapatmamak önemli
        if abs(direction) > 0.5:
            trail_mult = min(trail_mult + 0.5, 8.0)

        return {
            "adx_threshold":                int(adx_threshold),
            "entry_score_trend":            round(float(entry_score_trend), 2),
            "entry_score_ranging":          round(float(entry_score_ranging), 2),
            "trailing_stop_atr_multiplier": round(float(trail_mult), 1),
            "atr_stop_multiplier":          round(float(atr_stop_mult), 1),
            "max_position_pct":             round(float(max_pos_pct), 3),
        }

    def score_summary(
        self,
        df: pd.DataFrame,
        sym: str,
        lookback_bars: int = 200 * 24,
    ) -> None:
        """
        Logs a one-line summary of the coin's tradability analysis.

        Örnek çıktı:
            [CoinAnalyzer] BTC/USDT: score=0.82 | ATR%=2.14 | ADX_med=28.3 |
            ADX_q75=34.1 | Hurst=0.56 | vol_stab=0.71 | → TRADE

        Args:
            df           : Full indicator DataFrame.
            sym          : Coin sembolü (log için).
            lookback_bars: Kaç bar geriye bakılacak.
        """
        score = self.score_for_trading(df, lookback_bars)

        if len(df) < self.MIN_BARS:
            logger.info(
                f"[CoinAnalyzer] {sym}: Yetersiz veri ({len(df)} bar) — analiz yapılamadı"
            )
            return

        sliced          = self._slice(df, lookback_bars)
        atr_pct         = self._atr_pct(sliced)
        adx_med, adx_q75 = self._adx_stats(sliced)
        hurst           = self._hurst_median(sliced)
        vol_stab        = self._volume_stability(sliced)
        verdict         = "TRADE" if score >= 0.55 else ("DIKKATLI" if score >= 0.35 else "SKIP")

        logger.info(
            f"[CoinAnalyzer] {sym}: score={score:.3f} | "
            f"ATR%={atr_pct:.2f} | ADX_med={adx_med:.1f} | ADX_q75={adx_q75:.1f} | "
            f"Hurst={hurst:.3f} | vol_stab={vol_stab:.2f} | → {verdict}"
        )

    # ──────────────────────────────────────────────────────────────────
    # Dahili yardımcılar
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _default_params() -> dict:
        """
        Returns safe default parameters when there is not enough data to
        perform a proper analysis. These are intentionally conservative.
        """
        return {
            "adx_threshold":                25,
            "entry_score_trend":            0.65,
            "entry_score_ranging":          0.70,
            "trailing_stop_atr_multiplier": 5.0,
            "atr_stop_multiplier":          2.5,
            "max_position_pct":             0.04,
        }
