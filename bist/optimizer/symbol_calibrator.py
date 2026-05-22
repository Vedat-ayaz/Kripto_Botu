"""
Per-symbol adaptif parametre kalibrasyonu.

Her sembol için son N günlük USD fiyat verisini analiz eder:
  - ADX (trend gücü)
  - ATR/fiyat oranı (volatilite)
  - EMA50 üzerinde geçen gün yüzdesi (trend tutarlılığı)
  - Momentum (N günlük getiri)
  - Choppiness Index (60 günlük)

Bu metriklere göre 4 profil atar: STRONG_TREND, MILD_TREND, RANGING, CHOPPY
Her profilin farklı entry_threshold, stop, risk parametreleri var.
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

# ── Profil parametreleri ──────────────────────────────────────────────────────
PROFILES = {
    "STRONG_TREND": {
        "entry_score_ranging": 0.46,
        "entry_score_trend":   0.42,
        "atr_stop_multiplier": 2.5,
        "trailing_stop_atr_multiplier": 5.0,
        "risk_per_trade": 0.012,
        "adx_threshold": 15,
    },
    "MILD_TREND": {
        "entry_score_ranging": 0.52,
        "entry_score_trend":   0.47,
        "atr_stop_multiplier": 2.0,
        "trailing_stop_atr_multiplier": 4.0,
        "risk_per_trade": 0.009,
        "adx_threshold": 18,
    },
    "RANGING": {
        "entry_score_ranging": 0.57,
        "entry_score_trend":   0.52,
        "atr_stop_multiplier": 1.5,
        "trailing_stop_atr_multiplier": 3.0,
        "risk_per_trade": 0.006,
        "adx_threshold": 20,
    },
    "CHOPPY": {
        "entry_score_ranging": 0.64,
        "entry_score_trend":   0.58,
        "atr_stop_multiplier": 1.5,
        "trailing_stop_atr_multiplier": 2.5,
        "risk_per_trade": 0.004,
        "adx_threshold": 25,
    },
}


def _choppiness(high, low, close, period=14):
    """Choppiness Index hesapla."""
    atr_sum = pd.Series(
        np.maximum(high - low,
        np.maximum(abs(high - close.shift(1)),
                   abs(low  - close.shift(1))))
    ).rolling(period).sum()
    range_ = high.rolling(period).max() - low.rolling(period).min()
    ci = 100 * np.log10(atr_sum / range_.clip(lower=1e-9)) / np.log10(period)
    return ci


def calibrate(
    df_usd: pd.DataFrame,
    lookback: int = 60,
    macro_is_bear: bool = False,
    rs_positive: bool = False,   # YENİ: bear dönemde XU100'ü outperform ediyor mu?
) -> dict:
    """
    Son lookback günlük veriyi analiz et, uygun profili döndür.

    df_usd: open/high/low/close/volume içeren DataFrame (UTC index)
    macro_is_bear: True ise bear mod lojigi devreye girer
    rs_positive: True ise hisse XU100'ü outperform ediyor (bear'da bile trade et)
    Döndürür: PROFILES'dan birinin parametre dict'i + "profile" anahtarı
    """
    if len(df_usd) < lookback:
        logger.debug(f"[Calibrator] Yetersiz veri ({len(df_usd)} < {lookback}), RANGING profil")
        return {**PROFILES["RANGING"], "profile": "RANGING"}

    recent = df_usd.tail(lookback).copy()
    close = recent["close"]
    high  = recent["high"]
    low   = recent["low"]

    # ── Göstergeler ──────────────────────────────────────────────────────────
    # ADX (basit hesap)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().iloc[-1]

    # Directional movement
    dm_pos = (high.diff().clip(lower=0))
    dm_neg = (-low.diff().clip(upper=0))
    atr_s  = tr.rolling(14).mean()
    di_pos = 100 * dm_pos.rolling(14).mean() / atr_s.clip(lower=1e-9)
    di_neg = 100 * dm_neg.rolling(14).mean() / atr_s.clip(lower=1e-9)
    dx     = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg).clip(lower=1e-9)
    adx    = dx.rolling(14).mean().iloc[-1]

    # ATR oranı
    atr_ratio = atr14 / close.iloc[-1] if close.iloc[-1] > 0 else 0.02

    # EMA50 üstünde geçirilen gün yüzdesi
    ema50     = close.ewm(span=50, adjust=False).mean()
    trend_pct = float((close > ema50).mean())

    # Momentum (son 60 günlük getiri)
    momentum  = float(close.iloc[-1] / close.iloc[0] - 1) if close.iloc[0] > 0 else 0.0

    # Choppiness
    ci_val    = float(_choppiness(high, low, close).iloc[-1]) if len(recent) >= 14 else 50.0

    logger.debug(
        f"[Calibrator] ADX={adx:.1f} ATR%={atr_ratio*100:.2f} "
        f"Trend%={trend_pct*100:.0f} Mom={momentum*100:.1f}% CI={ci_val:.1f}"
    )

    # ── Profil belirleme ──────────────────────────────────────────────────────
    if ci_val > 61.8:
        profile = "CHOPPY"
    elif adx >= 22 and trend_pct >= 0.60 and momentum > 0.03:
        profile = "STRONG_TREND"
    elif adx >= 16 and trend_pct >= 0.45:
        profile = "MILD_TREND"
    else:
        profile = "RANGING"

    params = {**PROFILES[profile], "profile": profile}

    # ATR oranı çok yüksekse riski azalt
    if atr_ratio > 0.04:
        params["risk_per_trade"] = min(params["risk_per_trade"], 0.005)
        params["atr_stop_multiplier"] = min(params["atr_stop_multiplier"], 2.0)

    # ── Makro bear modifikasyonu ──────────────────────────────────────────────
    _tier_down = {"STRONG_TREND": "MILD_TREND", "MILD_TREND": "RANGING", "RANGING": "CHOPPY", "CHOPPY": "CHOPPY"}
    if macro_is_bear:
        if rs_positive:
            # Bear + RS pozitif: tüm profiller bir kademe düşer, risk yarıya iner
            new_profile = _tier_down[profile]
            logger.debug(f"[Calibrator] Bear+RS pozitif: {profile} → {new_profile}")
            profile = new_profile
            params = {**PROFILES[profile], "profile": profile}
            params["risk_per_trade"] = round(params["risk_per_trade"] * 0.6, 4)
            params["trailing_stop_atr_multiplier"] = min(params["trailing_stop_atr_multiplier"], 3.0)
            if atr_ratio > 0.04:
                params["risk_per_trade"] = min(params["risk_per_trade"], 0.004)
        else:
            # Bear + RS negatif → HİÇ İŞLEM YAPMA
            logger.debug(f"[Calibrator] Bear+RS negatif: {profile} → CHOPPY")
            profile = "CHOPPY"
            params = {**PROFILES[profile], "profile": profile}

    params["profile"] = profile
    return params
