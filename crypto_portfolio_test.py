"""
Kripto Portfolio Backtest — Paylaşımlı $10,000 Sermaye
=====================================================
config.yaml'daki 10 coin ile son 1 yıl (1h bar) portfolio simülasyonu.

Kullanım:
    python crypto_portfolio_test.py
    python crypto_portfolio_test.py --days 365 --capital 10000
    python crypto_portfolio_test.py --days 180

Çıktı:
    - Portfolio özeti (başlangıç/bitiş sermaye, toplam PnL, WR, max drawdown)
    - Coin bazlı tablo (işlem sayısı, WR%, PnL, ortalama işlem)
    - Buy & Hold karşılaştırması (bot getirisi vs coin'in 1 yıllık artışı)
    - Tüm işlemlerin kronolojik listesi
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import ccxt
import numpy as np
import pandas as pd

# Proje kökünü path'e ekle
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from indicators.technical_indicators import TechnicalIndicators
from strategy.trend_following_strategy import TrendFollowingStrategy
from strategy.signal import Side
from strategy.adaptive_regime import AdaptiveRegimeController, Regime, COIN_TIERS
from risk import correlation_registry
from strategy.coin_analyzer import CoinAnalyzer
from strategy.wfo_engine import WalkForwardOptimizer

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Sabitler ──────────────────────────────────────────────────────────────────

# ── Coin Evreni (25+ coin) — bot her dönem için en uygunları seçer ────────────
UNIVERSE = [
    # Büyük cap — yüksek likidite
    "BNB/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
    # Orta cap — trend-following uygun
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "TRX/USDT",
    "DOT/USDT", "LINK/USDT", "LTC/USDT", "ATOM/USDT",
    # Büyüyen ekosistem
    "NEAR/USDT", "UNI/USDT", "APT/USDT", "INJ/USDT",
    "FET/USDT", "ARB/USDT", "OP/USDT",
    # Özel / sabit
    "LEO/USDT", "ETC/USDT", "HBAR/USDT",
    "ALGO/USDT", "VET/USDT", "FIL/USDT",
    # M6 v7: yeni yüksek-momentumlu coinler (2024-2025 büyük hareketler)
    "SUI/USDT", "TIA/USDT", "TON/USDT", "JUP/USDT", "WIF/USDT",
]

# Varsayılan aktif coin listesi (--universe olmadan kullanılan sabit liste)
SYMBOLS = [
    "BNB/USDT", "ETH/USDT", "SOL/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT",
    "DOT/USDT", "TRX/USDT",
    "LEO/USDT",
]

COIN_SELECT_N      = 15     # --universe modunda seçilecek max coin sayısı
COIN_SELECT_MIN_SCORE = 0.40  # Bu skorun altındaki coinler hiç alınmaz (eski: 0.20)
WFO_ENABLED    = False  # --wfo flag'i ile True yapılır
WFO_LOOKBACK   = 200    # WFO için önceki N gün

# Bear/Strong_Bear dönemde dinamik coin sayısı kısıtlaması
# Rejim → Universe modunda kullanılabilecek max coin
REGIME_MAX_COINS: dict[str, int] = {
    "STRONG_BEAR": 5,
    "BEAR":        8,
    "NEUTRAL":     12,
    "BULL":        15,
    "STRONG_BULL": 18,
}

# ── M5 — Agresif Adaptif Model (M4 üstüne ek katmanlar) ──────
# Araştırma kaynakları: Kaufman (KAMA/ER), Lopez de Prado (Kelly/Kelly frac),
#   Freqtrade community (ATR percentile sizing), Elder Triple Screen (MTF)
# M5 REVİZE (v2): Koruma yerine KÂR AMPLİFİKASYONU odaklı
# Sorun: ATR percentile sizing bull trendinde tam tersi çalışıyordu
#   (trend güçlenince ATR artar → boyut kesilir → en iyi fırsatlar kaçar)
# Çözüm: 3 net iyileştirme, giriş sayısını AZALTMADAN kârı artır
#
# M5-1: PARTIAL EXIT at +1.5R → kâr kilidi (Elder'ın R-çıkış yöntemi)
# M5-2: Circuit Breaker sadece çok yıllık testlerde (duration > 400 gün)
# M5-3: ER Gate çok düşük eşikle (sadece tam gürültüye karşı) + ADX bypass
# M5-4: Momentum Decay exit (değişmedi — kâr geri verme önleme)

# v4: Partial Exit ve Momentum Decay KALDIRILDI
# Sorun: Kripto "fat-tail" dağılımı — büyük kazananlar (outlier trades) toplam kârın
# büyük bölümünü oluşturuyor. Her iki mekanizma da bu büyük kazananları erkenden kesiyor.
# Sonuç: v3'te Boğa PF 1.42→1.27, Karma PF 1.16→1.01. M4'ten sürekli daha kötü.
# v4 yaklaşımı: Kazananları kesmeden kaybedenleri azalt → Re-entry Cooldown.
M5_COOLDOWN_DAYS         = 3      # Stop hit'ten sonra X gün aynı coinden uzak dur
# ER Gate: v3'te KALDIRILDI (v2'de Ayı 304 → 25 işlem çöküşüne yol açtı)
M5_CB_DURATION_DAYS      = 400    # Circuit breaker sadece bu kadar günden uzun testlerde
M5_CB_THRESHOLDS = [              # Circuit Breaker: (DD eşiği, boyut çarpanı)
    (0.22, 0.0 ),   # %22+ DD → tüm yeni girişler durduruldu
    (0.15, 0.40),   # %15-22 DD → boyut %40'a indir
    (0.09, 0.65),   # %9-15 DD → boyut %65'e indir
    (0.00, 1.00),   # Normal → tam boyut
]

# ── M4 — Dynamic Adaptive Model ─────────────────────────────
M4_REGIME_CHECK_DAYS     = 30    # Her 30 günde rejim yeniden değerlendir
M4_WFO_ROLLING_DAYS      = 60    # Her 60 günde WFO yeniden çalıştır
M4_WFO_ROLLING_LOOKBACK  = 200   # Rolling WFO in-sample penceresi (gün)
M4_BULL_VS720_THRESHOLD  = 0.02  # BULL tespiti eşiği (eski: 0.04)
M4_BULL_ABOVE_FRAC       = 0.50  # BULL above_frac eşiği (eski: 0.55)
M4_POSITION_MULT: dict[str, float] = {
    "STRONG_BEAR": 0.6,   # Ayıda küçült — sermaye koru
    "BEAR":        0.8,   # Ayıda küçült — sermaye koru
    "NEUTRAL":     1.0,   # Değişiklik yok
    "BULL":        1.0,   # Boğada büyütme — Max DD artırıyor, kaldırıldı
    "STRONG_BULL": 1.0,   # Boğada büyütme — Max DD artırıyor, kaldırıldı
}

SYMBOL_ALIASES = {}

# Hangi sembol hangi exchange'den çekilir (Binance default)
SYMBOL_EXCHANGE: dict[str, str] = {
    "LEO/USDT": "okx",  # LEO Binance'te yok, OKX'te var
}

# Baseline strateji parametreleri
BASELINE = dict(
    adx_threshold=15,
    rsi_lower=45,
    rsi_upper=70,
    min_atr_ratio=0.002,
    volume_sma_multiplier=0.4,
    entry_score_trend=0.55,
    entry_score_ranging=0.60,
    choppiness_threshold=55.0,  # v9: 61.8 → 55.0 (sahte breakout sinyallerini eler, WR yükseltir)
    choppiness_enabled=True,
    mtf_filter_enabled=False,
    slope_bars=20,
    momentum_lookback=720,
    adx_boost=0.06,
    regime_trending_threshold=0.60,
    regime_ranging_threshold=0.35,
)

# Per-coin overrides
PROFILES: dict[str, dict] = {
    # BNB: choppy dönemlerde yanlış giriş çok → ADX 27 ile sadece güçlü trendlerde işlem
    "BNB/USDT":  dict(adx_threshold=27, rsi_lower=50, atr_stop_multiplier=2.2,
                      trailing_stop_atr_multiplier=5.0, entry_score_trend=0.63, entry_score_ranging=0.67),
    # ETH: ayıda güçlü SHORT + boğada uzun LONG → 4.5 trailing koruyucu ama geniş
    "ETH/USDT":  dict(adx_threshold=20, rsi_lower=50, atr_stop_multiplier=2.5,
                      trailing_stop_atr_multiplier=4.5, entry_score_trend=0.60, entry_score_ranging=0.65),
    # SOL: yüksek volatilite → geniş stop, yüksek ADX şartı, küçük pozisyon
    "SOL/USDT":  dict(adx_threshold=26, rsi_lower=50, atr_stop_multiplier=2.5,
                      trailing_stop_atr_multiplier=5.5, entry_score_trend=0.65, entry_score_ranging=0.70,
                      max_position_pct=0.06, sl_cooldown_hours=36),
    # XRP: güçlü trendlerde iyi, choppy dönemlerde kötü → seçici giriş + cooldown
    "XRP/USDT":  dict(adx_threshold=26, rsi_lower=50, atr_stop_multiplier=2.5,
                      trailing_stop_atr_multiplier=5.5, entry_score_trend=0.66, entry_score_ranging=0.70,
                      max_position_pct=0.08, sl_cooldown_hours=48),
    # ADA: tüm dönemlerde düşük WR, D3'te 2 trade 0%WR → küçük poz, yüksek eşik
    "ADA/USDT":  dict(adx_threshold=29, rsi_lower=52, atr_stop_multiplier=2.5,
                      trailing_stop_atr_multiplier=5.5, entry_score_trend=0.72, entry_score_ranging=0.76,
                      max_position_pct=0.02, breakout_bars=24, sl_cooldown_hours=72),
    # DOGE: meme coin → çok seçici, çok küçük pozisyon; ayı dönemde 33%WR → eşik yükselt
    "DOGE/USDT": dict(adx_threshold=30, rsi_lower=50, atr_stop_multiplier=3.0,
                      trailing_stop_atr_multiplier=6.0, entry_score_trend=0.71, entry_score_ranging=0.75,
                      max_position_pct=0.04, sl_cooldown_hours=48),
    # AVAX: D3'te aşırı trade → seçici giriş (adx=32, eşik=0.72/0.76), küçük poz
    "AVAX/USDT": dict(adx_threshold=32, rsi_lower=50, atr_stop_multiplier=3.0,
                      trailing_stop_atr_multiplier=6.0, entry_score_trend=0.72, entry_score_ranging=0.76,
                      max_position_pct=0.03, sl_cooldown_hours=72),
    # DOT: yavaş trend → breakout filtresi, küçük pozisyon
    "DOT/USDT":  dict(adx_threshold=26, rsi_lower=50, atr_stop_multiplier=2.5,
                      trailing_stop_atr_multiplier=5.5, entry_score_trend=0.66, entry_score_ranging=0.70,
                      max_position_pct=0.05, breakout_bars=24, sl_cooldown_hours=48),
    # LEO: düşük kaliteli sinyal, tüm dönemlerde kayıp → pozisyonu çok küçük tut (max %2), kısıtlı hasar
    "LEO/USDT":  dict(adx_threshold=26, rsi_lower=50, atr_stop_multiplier=2.5,
                      trailing_stop_atr_multiplier=6.0, entry_score_trend=0.70, entry_score_ranging=0.75,
                      risk_per_trade=0.008, max_position_pct=0.02, sl_cooldown_hours=48),
    # TRX: M4v11 — kısıtlamalar gevşetildi (breakout_bars=40→20, sl_cooldown=72→36h,
    # risk_per_trade=0.006→0.010, max_position_pct=0.05→0.12)
    # Önceki 7 ay / 7 işlem → +0.63% (coin +34%) → filtreler çok sıkıydı.
    "TRX/USDT":  dict(adx_threshold=25, rsi_lower=47, atr_stop_multiplier=2.5,
                      trailing_stop_atr_multiplier=6.0, entry_score_trend=0.67, entry_score_ranging=0.71,
                      risk_per_trade=0.010, max_position_pct=0.12, breakout_bars=20, sl_cooldown_hours=36),
    # ── Yeni coinler (UNIVERSE genişlemesi) ──────────────────────────────────
    "LINK/USDT": dict(adx_threshold=25, rsi_lower=48, atr_stop_multiplier=2.5,
                      trailing_stop_atr_multiplier=5.0, entry_score_trend=0.65, entry_score_ranging=0.70,
                      max_position_pct=0.08, sl_cooldown_hours=24),
    "LTC/USDT":  dict(adx_threshold=23, rsi_lower=48, atr_stop_multiplier=2.0,
                      trailing_stop_atr_multiplier=4.5, entry_score_trend=0.62, entry_score_ranging=0.67,
                      max_position_pct=0.08, sl_cooldown_hours=24),
    "ATOM/USDT": dict(adx_threshold=26, rsi_lower=50, atr_stop_multiplier=2.5,
                      trailing_stop_atr_multiplier=5.5, entry_score_trend=0.65, entry_score_ranging=0.70,
                      max_position_pct=0.06, sl_cooldown_hours=36),
    "NEAR/USDT": dict(adx_threshold=27, rsi_lower=50, atr_stop_multiplier=2.8,
                      trailing_stop_atr_multiplier=5.5, entry_score_trend=0.67, entry_score_ranging=0.72,
                      max_position_pct=0.05, sl_cooldown_hours=36),
    "UNI/USDT":  dict(adx_threshold=25, rsi_lower=50, atr_stop_multiplier=2.5,
                      trailing_stop_atr_multiplier=5.0, entry_score_trend=0.65, entry_score_ranging=0.70,
                      max_position_pct=0.06, sl_cooldown_hours=36),
    "APT/USDT":  dict(adx_threshold=28, rsi_lower=50, atr_stop_multiplier=3.0,
                      trailing_stop_atr_multiplier=6.0, entry_score_trend=0.68, entry_score_ranging=0.73,
                      max_position_pct=0.04, sl_cooldown_hours=48),
    "INJ/USDT":  dict(adx_threshold=28, rsi_lower=50, atr_stop_multiplier=3.0,
                      trailing_stop_atr_multiplier=6.0, entry_score_trend=0.68, entry_score_ranging=0.73,
                      max_position_pct=0.04, sl_cooldown_hours=48),
    "FET/USDT":  dict(adx_threshold=28, rsi_lower=50, atr_stop_multiplier=3.0,
                      trailing_stop_atr_multiplier=6.0, entry_score_trend=0.68, entry_score_ranging=0.73,
                      max_position_pct=0.04, sl_cooldown_hours=48),
    "ARB/USDT":  dict(adx_threshold=26, rsi_lower=50, atr_stop_multiplier=2.5,
                      trailing_stop_atr_multiplier=5.5, entry_score_trend=0.66, entry_score_ranging=0.71,
                      max_position_pct=0.05, sl_cooldown_hours=36),
    "OP/USDT":   dict(adx_threshold=26, rsi_lower=50, atr_stop_multiplier=2.5,
                      trailing_stop_atr_multiplier=5.5, entry_score_trend=0.66, entry_score_ranging=0.71,
                      max_position_pct=0.05, sl_cooldown_hours=36),
    "ETC/USDT":  dict(adx_threshold=24, rsi_lower=48, atr_stop_multiplier=2.5,
                      trailing_stop_atr_multiplier=5.0, entry_score_trend=0.64, entry_score_ranging=0.69,
                      max_position_pct=0.07, sl_cooldown_hours=36),
    "HBAR/USDT": dict(adx_threshold=26, rsi_lower=50, atr_stop_multiplier=2.5,
                      trailing_stop_atr_multiplier=5.5, entry_score_trend=0.67, entry_score_ranging=0.72,
                      max_position_pct=0.05, sl_cooldown_hours=48),
    "ALGO/USDT": dict(adx_threshold=25, rsi_lower=50, atr_stop_multiplier=2.5,
                      trailing_stop_atr_multiplier=5.0, entry_score_trend=0.66, entry_score_ranging=0.71,
                      max_position_pct=0.05, sl_cooldown_hours=48),
    "VET/USDT":  dict(adx_threshold=25, rsi_lower=50, atr_stop_multiplier=2.5,
                      trailing_stop_atr_multiplier=5.0, entry_score_trend=0.66, entry_score_ranging=0.71,
                      max_position_pct=0.05, sl_cooldown_hours=48),
    "FIL/USDT":  dict(adx_threshold=27, rsi_lower=50, atr_stop_multiplier=3.0,
                      trailing_stop_atr_multiplier=6.0, entry_score_trend=0.68, entry_score_ranging=0.73,
                      max_position_pct=0.04, sl_cooldown_hours=48),
    # ── M6 v7: yüksek momentumlu yeni coinler (volatil → temkinli profil) ────
    "SUI/USDT":  dict(adx_threshold=27, rsi_lower=50, atr_stop_multiplier=2.8,
                      trailing_stop_atr_multiplier=6.0, entry_score_trend=0.67, entry_score_ranging=0.72,
                      max_position_pct=0.06, sl_cooldown_hours=36),
    "TIA/USDT":  dict(adx_threshold=28, rsi_lower=50, atr_stop_multiplier=3.0,
                      trailing_stop_atr_multiplier=6.0, entry_score_trend=0.68, entry_score_ranging=0.73,
                      max_position_pct=0.04, sl_cooldown_hours=48),
    "TON/USDT":  dict(adx_threshold=25, rsi_lower=50, atr_stop_multiplier=2.5,
                      trailing_stop_atr_multiplier=5.5, entry_score_trend=0.65, entry_score_ranging=0.70,
                      max_position_pct=0.06, sl_cooldown_hours=36),
    "JUP/USDT":  dict(adx_threshold=28, rsi_lower=50, atr_stop_multiplier=3.0,
                      trailing_stop_atr_multiplier=6.0, entry_score_trend=0.68, entry_score_ranging=0.73,
                      max_position_pct=0.04, sl_cooldown_hours=48),
    "WIF/USDT":  dict(adx_threshold=30, rsi_lower=50, atr_stop_multiplier=3.5,
                      trailing_stop_atr_multiplier=7.0, entry_score_trend=0.70, entry_score_ranging=0.75,
                      max_position_pct=0.03, sl_cooldown_hours=48),
}

# Risk parametreleri (global)
INITIAL_CAPITAL   = 10_000.0
COMMISSION        = 0.001
SLIPPAGE          = 0.0005
RISK_PER_TRADE    = 0.015
DAILY_MAX_LOSS    = 0.04
ATR_STOP_MULT     = 2.0
TRAILING_MULT     = 3.8
MAX_POSITIONS     = 8
MAX_POSITION_PCT  = 0.20
MIN_ORDER_SIZE    = 10.0

WARMUP_BARS = 210   # EMA200 + buffer


# ── Multi-Timeframe Helpers (v13) ─────────────────────────────────────────────
# Per-model timeframe: M4/M5 → 15m, M6 → 1m. Helper'lar bar-bazlı varsayımları
# (örn "14 gün × 24 saat = bar") TF-aware hale getirir.

def _bars_per_day(timeframe: str) -> int:
    """Verilen timeframe'de bir gündeki bar sayısı."""
    return {"1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}.get(timeframe, 24)

def _tf_to_minutes(timeframe: str) -> int:
    """Timeframe'in dakika cinsinden uzunluğu."""
    return {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}.get(timeframe, 60)

def _htf_rule_for(timeframe: str) -> str:
    """Higher-timeframe trend filtresi için uygun pandas resample rule."""
    # pandas 2.2+: "H" → "h", "D" → "D" (büyük harf korunur)
    return {"1m": "15min", "5m": "1h", "15m": "1h", "1h": "4h", "4h": "1D"}.get(timeframe, "4h")


# ── Veri çekimi ───────────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, days: int = 365, timeframe: str = "1h") -> pd.DataFrame:
    """ccxt ile OHLCV çeker. LEO gibi Binance'te olmayan coinler için OKX kullanır."""
    ex_name = SYMBOL_EXCHANGE.get(symbol, "binance")
    exchange = getattr(ccxt, ex_name)({
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
            "fetchMarkets": ["spot"],   # ccxt 4.x: sadece spot marketleri yükle
        },
    })
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days + 10)).timestamp() * 1000)
    # OKX max 300 bar, Binance 1000 bar döndürür
    chunk_size = 300 if ex_name == "okx" else 1000
    all_data: list = []
    while True:
        chunk = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=chunk_size)
        if not chunk:
            break
        all_data.extend(chunk)
        if len(chunk) < chunk_size:
            break
        since_ms = chunk[-1][0] + 1
        time.sleep(0.2)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    df.sort_index(inplace=True)
    df = df.iloc[:-1]  # son kapanmamış mumu çıkar

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_ts = pd.Timestamp(cutoff)
    # v13: WARMUP_BARS bar cinsinden — TF'ye göre dakika hesapla (1h: 210h, 15m: 52.5h, 1m: 3.5h)
    df = df[df.index >= cutoff_ts - pd.Timedelta(minutes=WARMUP_BARS * _tf_to_minutes(timeframe))]
    return df


# ── İndikatör hazırlama ───────────────────────────────────────────────────────

def prepare_indicators(df: pd.DataFrame, indicators: TechnicalIndicators, timeframe: str = "1h") -> pd.DataFrame:
    df_ind = indicators.calculate(df)
    # v13: HTF kuralı timeframe'e göre seçilir (1h→4H, 15m→1H, 1m→15min)
    df_ind = indicators.add_higher_timeframe(df_ind, htf_rule=_htf_rule_for(timeframe))
    # v19 FIX-A: "ema_slow" alias ekle — AdaptiveRegimeController ve set_btc_regime() bu ismi arıyor.
    # TechnicalIndicators kolonu "ema_200" üretiyor ama regime kodu "ema_slow" arıyor → her zaman 0 dönüyor.
    # Bu yüzden BTC EMA200 rejim faktörü (ağırlık 1.5) HİÇBİR ZAMAN çalışmıyor.
    ema_slow_col = indicators.ema_slow_col  # "ema_200" (varsayılan)
    if ema_slow_col in df_ind.columns and "ema_slow" not in df_ind.columns:
        df_ind["ema_slow"] = df_ind[ema_slow_col]
    # v19 FIX: 15m için Choppiness Index'i TF-aware periyotla yeniden hesapla.
    # Varsayılan period=14 → 15m'de 14×15min = 3.5 saat penceresi (çok kısa → CI hep yüksek).
    # Hedef: 1h'daki 14-bar CI ile eşdeğer "14 saatlik" pencere = 56 bar.
    # Sonuç: 15m CI değerleri artık 1h CI ile karşılaştırılabilir → threshold anlamlı çalışır.
    if timeframe == "15m":
        from indicators.technical_indicators import _choppiness_index as _ci_fn
        df_ind["choppiness"] = _ci_fn(df_ind["high"], df_ind["low"], df_ind["close"], period=56)
    elif timeframe == "1m":
        # 1m: 56 bar = 56 dakika (1 saatlik pencere → makul scalping CI)
        from indicators.technical_indicators import _choppiness_index as _ci_fn
        df_ind["choppiness"] = _ci_fn(df_ind["high"], df_ind["low"], df_ind["close"], period=60)
    return df_ind


# ── Strateji factory ──────────────────────────────────────────────────────────

def _adaptive_choppiness_threshold(df: pd.DataFrame, timeframe: str = "1h") -> float:
    """Coinin son 60 günlük CI ortalamasına göre adaptif choppiness threshold döner.
    Choppy coin (yüksek CI) → düşük threshold (daha katı giriş filtresi).
    Trendy coin (düşük CI) → yüksek threshold (daha gevşek, daha fazla işlem).
    """
    lookback = 60 * _bars_per_day(timeframe)  # v13: 60 gün × bar/gün (1h→1440, 15m→5760, 1m→86400)
    recent = df.tail(lookback)
    if 'choppiness' not in recent.columns or len(recent) < 200:
        return 61.8  # fallback default
    ci_mean = float(recent['choppiness'].dropna().mean())
    if ci_mean > 60:    # Çok choppy coin → sert filtre
        result = 52.0
    elif ci_mean > 55:  # Choppy
        result = 56.0
    elif ci_mean > 50:  # Orta
        result = 61.8
    else:               # Trending coin → gevşek filtre
        result = 65.0
    # v19 FIX: Kısa TF barlarda CI doğal olarak yüksek çıkar (14-bar penceresi = 3.5h for 15m, 14min for 1m).
    # Adaptive threshold 52.0'a düşünce 15m'de neredeyse HİÇBİR bar geçemiyor → 0 trade.
    # 15m için minimum 65.0, 1m için 70.0 — 1h için mevcut mantık korunur.
    _TF_FLOOR = {"1m": 70.0, "5m": 67.0, "15m": 65.0}
    return max(result, _TF_FLOOR.get(timeframe, result))


# ── M5 Yardımcı Fonksiyonlar ────────────────────────────────────────────────

def _calc_efficiency_ratio(prices: pd.Series, n: int = 10) -> float:
    """
    Kaufman Efficiency Ratio (ER): yönsel etkinliği ölçer.
    ER = net_hareket / toplam_path
    ER → 1.0  : güçlü düz trend (trend-following için ideal)
    ER → 0.0  : rastlantısal/choppy (trend-following kayıplandırır)
    """
    if len(prices) < n + 2:
        return 0.5
    net_change   = abs(float(prices.iloc[-1]) - float(prices.iloc[-(n + 1)]))
    total_path   = float(prices.diff().abs().tail(n).sum())
    return float(net_change / total_path) if total_path > 0 else 0.5


def _calc_atr_percentile(atr_series: pd.Series, lookback: int = 100) -> float:
    """
    Mevcut ATR'ın son `lookback` bar içindeki yüzdelik sırası.
    0.0 = en düşük volatilite (kırılım fırsatı)
    1.0 = en yüksek volatilite (yüksek risk, pozisyon küçült)
    """
    if len(atr_series) < 10:
        return 0.5
    recent = atr_series.tail(lookback + 1).dropna()
    if len(recent) < 5:
        return 0.5
    current = float(recent.iloc[-1])
    return float((recent.iloc[:-1] < current).mean())


def _calc_momentum_decay(df_slice: pd.DataFrame) -> int:
    """
    Momentum tükenme skoru (0-3).
    Her bileşen 1 puan:
      1) RSI negatif diverjans (fiyat yüksek, RSI düşüyor)
      2) Hacim azalması (son 3 bar ortalama < önceki 6 bar × 0.80)
      3) ATR daralması (son bar ATR < son 14 bar ort × 0.75)
    Skor ≥ 2 → trailing stop sıkıştır
    Skor ≥ 3 → erken çıkış tetikle
    """
    score = 0
    # 1) RSI divergence
    if "rsi" in df_slice.columns and "close" in df_slice.columns:
        tail5 = df_slice.tail(5)
        if len(tail5) >= 5:
            price_at_high = float(tail5["close"].iloc[-1]) >= float(tail5["close"].max()) * 0.995
            rsi_falling   = float(tail5["rsi"].iloc[-1]) < float(tail5["rsi"].iloc[-3])
            if price_at_high and rsi_falling:
                score += 1
    # 2) Volume decay
    if "volume" in df_slice.columns and len(df_slice) >= 9:
        vol = df_slice["volume"].tail(9)
        if float(vol.tail(3).mean()) < float(vol.head(6).mean()) * 0.80:
            score += 1
    # 3) ATR contraction
    if "atr" in df_slice.columns and len(df_slice) >= 14:
        atr_vals = df_slice["atr"].tail(14).dropna()
        if len(atr_vals) >= 7:
            curr_atr = float(atr_vals.iloc[-1])
            avg_atr  = float(atr_vals.mean())
            if avg_atr > 0 and curr_atr < avg_atr * 0.75:
                score += 1
    return score


def make_strategy(symbol: str, wfo_params: Optional[dict] = None, coin_df: Optional[pd.DataFrame] = None,
                  timeframe: str = "1h") -> tuple[TrendFollowingStrategy, dict]:
    """
    Coin'e özgü strateji oluşturur.
    Öncelik: WFO params > PROFILES > BASELINE
    wfo_params: WFO motorunun bulduğu parametreler (None ise PROFILES kullanılır)
    timeframe: v13 — adaptive choppiness ve TF-bağımlı parametreler için
    """
    # WFO metadata anahtarlarını (_wfo_score vb.) temizle
    clean_wfo = {k: v for k, v in (wfo_params or {}).items() if not k.startswith('_')} if wfo_params else {}
    # Adaptif choppiness: WFO'da yoksa coinin CI ortalamasından hesapla
    adaptive_chop = _adaptive_choppiness_threshold(coin_df, timeframe=timeframe) if coin_df is not None else 61.8
    if 'choppiness_threshold' not in clean_wfo:
        p = {**BASELINE, 'choppiness_threshold': adaptive_chop, **PROFILES.get(symbol, {}), **clean_wfo}
    else:
        p = {**BASELINE, **PROFILES.get(symbol, {}), **clean_wfo}
    strat_keys = {
        "adx_threshold", "rsi_lower", "rsi_upper", "min_atr_ratio",
        "volume_sma_multiplier", "entry_score_trend", "entry_score_ranging",
        "choppiness_threshold", "choppiness_enabled", "mtf_filter_enabled",
        "slope_bars", "momentum_lookback", "adx_boost",
        "regime_trending_threshold", "regime_ranging_threshold",
    }
    strat_params = {k: v for k, v in p.items() if k in strat_keys}
    risk_params = {k: v for k, v in p.items() if k not in strat_keys}

    # v19 FIX: 1m (M6 scalping) için strateji parametrelerini TF-aware yap.
    # 1m barlarında varsayılan EMA50 = 50dk, EMA200 = 3.3h → çok kısa → filtreler hep tetikleniyor.
    # Çözüm: M6'da slope_bars ve momentum_lookback'i 1h eşdeğerine ölçekle.
    # slope_bars: 1h=20bar(20saat) → 1m=1200bar(20saat). Ama çok uzun → scalping için 60bar(1saat).
    # momentum_lookback: 1h=720bar(30gün) → 1m=43200bar (çok büyük) → 1440bar(1gün) kullan.
    # mtf_filter_enabled: 1m'de HTF=15min resample. Bu makul ama strict; M6'da devre dışı bırak.
    if timeframe == "1m":
        strat_params.setdefault("slope_bars", 60)          # 60×1min = 1 saatlik EMA eğimi
        strat_params["momentum_lookback"] = 1440           # 1440×1min = 1 günlük momentum (eskisi 720→12h, çok kısa)
        strat_params["mtf_filter_enabled"] = False         # 1m'de 15min HTF filtresi M6 girişlerini çok blokluyor

    inds = TechnicalIndicators()
    strategy = TrendFollowingStrategy(**strat_params, indicators=inds)
    return strategy, risk_params


# ── Portfolio Pozisyon ────────────────────────────────────────────────────────

@dataclass
class PPos:
    symbol: str
    entry_price: float
    stop_price: float     # LONG: giriş - N*ATR | SHORT: giriş + N*ATR
    trail_price: float    # LONG: yukarı gider (aşağı limit) | SHORT: aşağı gider (yukarı limit)
    size: float           # coin miktarı
    cost: float           # giriş maliyeti rezervi (nakit bloke)
    entry_time: pd.Timestamp
    entry_atr: float
    trailing_mult: float

    min_hold_bars: int = 6
    is_coin_bull: bool = False   # coin kendi EMA200 üzerindeydi
    is_short: bool = False       # True → açığa satış (SHORT) pozisyonu

    exit_price: float = 0.0
    exit_time: Optional[pd.Timestamp] = None
    exit_reason: str = ""
    pnl: float = 0.0     # net (komisyon dahil) — SHORT'ta giriş-çıkış farkı
    bars_held: int = 0

    # M4v11 — Pyramiding (Turtle Trading "add-to-winners")
    size_at_entry: float = 0.0   # ilk girişteki orijinal lot (pyramid hesabı için)
    pyramid_count: int = 0       # kaç kez pyramid yapıldı (maks 2)
    pyramid_cost: float = 0.0    # pyramid birimlerinin toplam maliyeti (PnL muhasebesi için)

    # M5 — Partial Exit (R-multiple kâr kilidi)
    r_value: float = 0.0         # 1R = giriş ile stop arası mesafe (partial exit hesabı için)
    partial_exit_done: bool = False  # +1.5R kısmi çıkış yapıldı mı?


# ── Dinamik Coin Seçimi ───────────────────────────────────────────────────────

def select_active_coins(
    sym_ind: dict[str, pd.DataFrame],
    n: int = COIN_SELECT_N,
    min_score: float = COIN_SELECT_MIN_SCORE,
) -> list[str]:
    """
    Coin Analyzer puanlarına göre evrenden en uygun N coini seçer.

    Puanlama kriterleri:
      - ADX kalitesi (35%): 20-40 arası ideal
      - ATR% volatilite (30%): 1-5% hourly ideal
      - Hurst kalıcılığı (25%): >0.48 trending
      - Hacim stabilitesi (10%): tutarlı işlem hacmi
    """
    analyzer = CoinAnalyzer()
    scores: list[tuple[str, float]] = []

    for sym, df in sym_ind.items():
        if sym in ("BTC/USDT", "_BTC_REGIME_"):
            continue
        if len(df) < 500:
            continue
        try:
            base_score = analyzer.score_for_trading(df)
            scores.append((sym, base_score))
        except Exception:
            scores.append((sym, 0.0))

    score_map = dict(scores)
    scores.sort(key=lambda x: x[1], reverse=True)

    # ── Hibrit Seçim Stratejisi ──────────────────────────────────────────
    # 1) SYMBOLS (kanıtlanmış M1 coinleri) her zaman dahil edilir — bunların
    #    PROFILES parametreleri var, WFO başarısız olsa da güvenli fallback.
    # 2) Kalan slotları Universe'den en yüksek scorlu coinlerle doldur.
    # Bu yaklaşım "proven core + opportunistic extras" dengesi kurar.
    base_syms = [s for s in SYMBOLS if s in score_map]  # M1 coinleri (max 9)
    extra_slots = max(0, n - len(base_syms))
    extra_syms = [
        s for s, sc in scores
        if s not in base_syms and sc >= min_score
    ][:extra_slots]
    selected_list = base_syms + extra_syms
    selected = [(sym, score_map.get(sym, 0.0)) for sym in selected_list]

    print(f"\n{'─'*60}")
    print(f"  COİN SEÇİMİ ({len(scores)} aday → {len(base_syms)} sabit + {len(extra_syms)} universe = {len(selected_list)} coin)")
    print(f"  (● = M1 sabit coin | ○ = Universe seçimi)")
    print(f"{'─'*60}")
    # Tüm adayları göster
    shown = set()
    for sym in selected_list:
        tag = "●" if sym in base_syms else "○"
        s = score_map.get(sym, 0.0)
        bar = "█" * int(s * 20)
        print(f"  ✓ {tag} {sym:<12} {s:.3f}  {bar}")
        shown.add(sym)
    for sym, s in scores:
        if sym not in shown:
            bar = "█" * int(s * 20)
            print(f"  ✗   {sym:<12} {s:.3f}  {bar}")

    return [sym for sym, _ in selected]


# ── M4 State ─────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class _M4State:
    """M4 intra-simulation state — simülasyon döngüsü boyunca taşınan durum."""
    last_regime_check:  pd.Timestamp
    last_wfo_check:     pd.Timestamp
    active_syms:        list
    wfo_results:        dict
    regime_switches:    list = dataclasses.field(default_factory=list)
    wfo_updates:        list = dataclasses.field(default_factory=list)


# ── Başlangıç Rejim Tespiti ───────────────────────────────────────────────────

def _assess_btc_regime(btc_df: pd.DataFrame, trade_start: pd.Timestamp) -> str:
    """
    Simülasyon başlamadan önceki BTC verisine bakarak rejimi tahmin eder.

    Kullanım: --auto modunda, WFO'nun başlatılıp başlatılmamasına karar vermek için.

    Returns:
        'BULL'    → Son 30 günde BTC yukarı trendde  → M1 modu (PROFILES, WFO yok)
        'NEUTRAL' → Belirsiz/geçiş dönemi             → M3_v4 (WFO + hibrit)
        'BEAR'    → Son 30 günde BTC aşağı trendde   → M3_v4 (WFO + hibrit, daha defansif)
    """
    pre = btc_df[btc_df.index < trade_start] if not btc_df.empty else btc_df
    if len(pre) < 200:
        return "NEUTRAL"

    close = pre["close"].dropna()
    if len(close) < 200:
        return "NEUTRAL"

    # BTC'nin son konumu: 200-bar ve 720-bar (≈30 gün saatlik) EMA'larına göre
    ema200  = close.ewm(span=200,  adjust=False).mean()
    ema720  = close.ewm(span=720,  adjust=False).mean()
    ema168  = close.ewm(span=168,  adjust=False).mean()  # 7-gün

    last    = float(close.iloc[-1])
    e200    = float(ema200.iloc[-1])
    e720    = float(ema720.iloc[-1])
    e168    = float(ema168.iloc[-1])

    # Son 30 gün × 24h = 720 bar içinde EMA200'ün üzerinde geçen süre
    window  = min(720, len(close))
    above_frac = float((close.iloc[-window:] > ema200.iloc[-window:]).mean())

    vs_200 = (last - e200) / e200   # + → yukarı
    vs_720 = (last - e720) / e720
    vs_168 = (last - e168) / e168

    # BULL: 30g EMA %4 üstü VE 200-bar EMA üstü VE zamanın %55+'ı üstte
    if vs_720 > M4_BULL_VS720_THRESHOLD and vs_200 > 0.01 and above_frac > M4_BULL_ABOVE_FRAC:
        return "BULL"
    # BEAR: 30g EMA %3 altı VEYA (200-bar EMA %2 altı VE zamanın %40-'ı üstte)
    elif vs_720 < -0.03 or (vs_200 < -0.02 and above_frac < 0.40):
        return "BEAR"
    else:
        return "NEUTRAL"


# ── M4: İntra-Simülasyon Rejim Checkpoint ────────────────────────────────────

def _run_regime_checkpoint(
    ts: pd.Timestamp,
    btc_ind: pd.DataFrame,
    sym_ind: dict,
    m4_state: "_M4State",
    use_universe: bool,
    n_coins: int,
    open_positions: dict,
) -> tuple:
    """
    Simülasyon ortasında BTC rejimini yeniden değerlendirir.
    Returns: (detected_regime_str, new_active_syms_list)
    Açık pozisyonlara dokunmaz.
    """
    btc_slice = btc_ind[btc_ind.index <= ts].tail(720 + 200)
    if len(btc_slice) < 200:
        return "NEUTRAL", m4_state.active_syms

    close  = btc_slice["close"].dropna()
    if len(close) < 200:
        return "NEUTRAL", m4_state.active_syms

    ema200 = close.ewm(span=200, adjust=False).mean()
    ema720 = close.ewm(span=720, adjust=False).mean()

    last       = float(close.iloc[-1])
    e200       = float(ema200.iloc[-1])
    e720       = float(ema720.iloc[-1])
    window     = min(720, len(close))
    above_frac = float((close.iloc[-window:] > ema200.iloc[-window:]).mean())

    vs_720 = (last - e720) / e720
    vs_200 = (last - e200) / e200

    if vs_720 > M4_BULL_VS720_THRESHOLD and vs_200 > 0.01 and above_frac > M4_BULL_ABOVE_FRAC:
        new_regime = "BULL"
    elif vs_720 < -0.03 or (vs_200 < -0.02 and above_frac < 0.40):
        new_regime = "BEAR"
    else:
        new_regime = "NEUTRAL"

    if use_universe:
        if new_regime in ("BEAR", "STRONG_BEAR"):
            # Ayıda universe altcoinleri dışla — sadece kanıtlanmış 9 sabit coin
            # Bu Max DD'yi düşürür: altcoinler ayıda %50-70 düşebiliyor
            fixed_only = [s for s in SYMBOLS if s in sym_ind]
            locked = [s for s in open_positions if s in sym_ind]  # açık pozisyonları koru
            new_active = list(dict.fromkeys(locked + fixed_only))  # önce locked, sonra fixed
        else:
            sym_ind_now = {s: df[df.index <= ts] for s, df in sym_ind.items() if s != "BTC/USDT"}
            raw_selected = select_active_coins(sym_ind_now, n=n_coins)
            max_coins = REGIME_MAX_COINS.get(new_regime, COIN_SELECT_N)
            locked = [s for s in open_positions if s in sym_ind and s in raw_selected]
            free_slots = max(0, max_coins - len(locked))
            candidates = [s for s in raw_selected if s not in locked][:free_slots]
            new_active = locked + candidates
    else:
        new_active = m4_state.active_syms

    return new_regime, new_active


# ── M4: Rolling Walk-Forward Optimizasyon ────────────────────────────────────

def _run_rolling_wfo(
    ts: pd.Timestamp,
    sym_ind: dict,
    active_syms: list,
    current_wfo_results: dict,
    open_positions: dict,
) -> dict:
    """
    Rolling WFO: ts anına kadar son M4_WFO_ROLLING_LOOKBACK günlük veriyle
    aktif coinleri (açık pozisyonu olmayanları) yeniden optimize eder.
    """
    optimizer = WalkForwardOptimizer(lookback_days=M4_WFO_ROLLING_LOOKBACK)
    new_results = dict(current_wfo_results)

    for sym in active_syms:
        if sym == "BTC/USDT":
            continue
        if sym in open_positions:
            continue
        if sym not in sym_ind:
            continue

        df_upto = sym_ind[sym][sym_ind[sym].index < ts]
        # v13: bars_per_day TF'ye göre — 1h:24, 15m:96, 1m:1440
        min_bars = M4_WFO_ROLLING_LOOKBACK * _bpd + 250
        if len(df_upto) < min_bars:
            continue

        try:
            params = optimizer.optimize(sym, df_upto)
            if params is not None:
                new_results[sym] = params
                logger.info(f"[RollingWFO] {ts.strftime('%Y-%m-%d')} {sym}: güncellendi skor={params.get('_wfo_score', 0):.3f}")
        except Exception as ex:
            logger.warning(f"[RollingWFO] {sym}: hata — {ex}")

    return new_results


# ── Ana Backtest ──────────────────────────────────────────────────────────────

def run_portfolio_backtest(
    days: int = 365,
    initial_capital: float = INITIAL_CAPITAL,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    label: Optional[str] = None,
    use_universe: bool = False,     # True → UNIVERSE'den dinamik seçim
    use_wfo: bool = False,          # True → WFO ile parametre optimizasyonu
    n_coins: int = COIN_SELECT_N,   # aktif coin sayısı
    auto_mode: bool = False,        # True → Rejime göre otomatik mod seçimi
    m4_mode: bool = False,          # True → M4: intra-simulation rejim checkpoint + rolling WFO
    m5_mode: bool = False,          # True → M5: ATR-percentile sizing + circuit breaker + ER gate + momentum decay
    m6_mode: bool = False,          # True → M6: agresif pyramiding + erken trailing zoom + büyük pozisyon
    timeframe: Optional[str] = None,# v13: Bar timeframe. None → M4/M5=15m, M6=1m, default=1h
    json_out: Optional[str] = None, # Opsiyonel JSON state dosyası yolu (live dashboard için)
) -> None:
    # v13: Per-model timeframe — M4/M5 → 15m (orta-vade swing), M6 → 1m (scalping)
    if timeframe is None:
        if m6_mode:
            timeframe = "1m"
        elif m5_mode or m4_mode:
            timeframe = "15m"
        else:
            timeframe = "1h"
    _bpd = _bars_per_day(timeframe)
    _tf_mins = _tf_to_minutes(timeframe)
    _label = label or f"Son {days} Gün"
    print(f"\n{'='*72}")
    print(f"  KRIPTO PORTFOLIO BACKTEST — {_label} | Sermaye: ${initial_capital:,.0f}")
    print(f"{'='*72}\n")

    # 1) Veri çek
    print("Veri çekiliyor...")
    raw_data: dict[str, pd.DataFrame] = {}
    # auto_mode veya use_universe=True ise tüm evreni çek (rejim tespiti sonrası mod belirlenir)
    fetch_list = UNIVERSE if (use_universe or auto_mode) else SYMBOLS
    print(f"  Timeframe: {timeframe} ({_bpd} bar/gün)")
    for sym in fetch_list:
        try:
            df = fetch_ohlcv(sym, days=days + 5, timeframe=timeframe)
            raw_data[sym] = df
            print(f"  {sym:<12} {len(df):>5} bar  "
                  f"({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})")
        except Exception as e:
            print(f"  {sym:<12} HATA: {e}")

    # Rejim için BTC verisi (listede yoksa ayrıca çek)
    # v13: BTC rejim tespiti her zaman GÜNLÜK (1d) bar kullanır — daha stabil ve gürültüsüz
    if "BTC/USDT" not in raw_data:
        try:
            print("  BTC/USDT  (rejim için ayrıca çekiliyor...)")
            raw_data["_BTC_REGIME_"] = fetch_ohlcv("BTC/USDT", days=days + 5, timeframe=timeframe)
        except Exception as e:
            print(f"  BTC rejim verisi alınamadı: {e}")

    # 2) İndikatörleri hesapla
    print("\nİndikatörler hesaplanıyor...")
    indicators_obj = TechnicalIndicators()
    sym_ind: dict[str, pd.DataFrame] = {}
    for sym, df in raw_data.items():
        try:
            sym_ind[sym] = prepare_indicators(df, indicators_obj, timeframe=timeframe)
        except Exception as e:
            print(f"  {sym}: indikatör hatası — {e}")

    # BTC rejim DataFrame'ini ayır (işlem yapılacak listede değil)
    btc_regime_df = sym_ind.pop("_BTC_REGIME_", sym_ind.get("BTC/USDT"))

    # 3) Trade start / end — ÖNCE hesapla (auto_mode ve WFO için gerekli)
    if start_date:
        trade_start = pd.Timestamp(start_date, tz="UTC")
    else:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        trade_start = pd.Timestamp(cutoff)
    if end_date:
        trade_end = pd.Timestamp(end_date, tz="UTC")
    else:
        trade_end = pd.Timestamp(datetime.now(timezone.utc))

    # ── AUTO MODE: trade_start belli olduktan HEMEN SONRA mod seç ───────────
    # Coin seçimi ve WFO'dan ÖNCE çalışmalı — sıra kritik!
    if auto_mode:
        _btc_for_regime = btc_regime_df
        if _btc_for_regime is not None:
            detected = _assess_btc_regime(_btc_for_regime, trade_start)
        else:
            detected = "NEUTRAL"

        print(f"\n{'─'*60}")
        print(f"  🤖 AUTO MODE — BTC Rejim Tespiti: {detected}")
        if detected == "BULL":
            use_universe = False
            use_wfo      = False
            print(f"  → Boğa piyasası tespit edildi")
            print(f"  → MOD: M1 (9 sabit coin, PROFILES parametreleri, WFO yok)")
        else:
            use_universe = True
            use_wfo      = True
            regime_emoji = "🔴" if detected == "BEAR" else "🟡"
            print(f"  {regime_emoji} → {detected} piyasası tespit edildi")
            if m4_mode:
                # M4v8: Universe yerine SYMBOLS (9 coin) + WFO — gerçek Hibrit modu.
                # AUTO modun 15 coin (9+6 universe) seçimi kötü sonuç veriyordu.
                # Hibrit testi SYMBOLS+WFO ile +8.16% (Boğa) / +4.94% (Ayı) yaptı.
                use_universe = False
                print(f"  → MOD: M4-Hibrit (9 SYMBOLS coin, WFO aktif)")
            else:
                print(f"  → MOD: M3_v4 Hibrit (9 sabit + {n_coins-9} universe, WFO aktif)")
        print(f"{'─'*60}")

    # ── Dinamik Coin Seçimi — auto_mode kararından SONRA çalışır ────────────
    if use_universe:
        active_syms = select_active_coins(sym_ind, n=n_coins)
        # sym_ind'i SKOR SIRASIYLA yeniden oluştur — dinamik coin kısıtlaması için şart
        ordered_ind: dict[str, pd.DataFrame] = {}
        for s in active_syms:
            if s in sym_ind:
                ordered_ind[s] = sym_ind[s]
        sym_ind = ordered_ind
        print(f"\n  Aktif coinler ({len(active_syms)}): {', '.join(active_syms)}")
    else:
        # auto_mode=True + BULL tespiti → UNIVERSE verisi çekildi ama sadece SYMBOLS kullan
        # use_universe=False + auto_mode → sym_ind'i SYMBOLS ile kısıtla
        if auto_mode:
            for s in list(sym_ind.keys()):
                if s not in SYMBOLS:
                    sym_ind.pop(s, None)
        active_syms = [s for s in sym_ind.keys() if s != "BTC/USDT"]

    # ── Walk-Forward Optimizasyon ────────────────────────────────────────────
    # CRITICAL: WFO'ya yalnızca trade_start ÖNCESİ veri ver (look-ahead bias önlemi)
    wfo_results: dict[str, dict] = {}
    if use_wfo and sym_ind:
        print(f"\n{'─'*60}")
        print(f"  WALK-FORWARD OPTİMİZASYON ({WFO_LOOKBACK} günlük in-sample)")
        print(f"  In-sample pencere sonu: {trade_start.strftime('%Y-%m-%d')} (simülasyon başlangıcı)")
        print(f"{'─'*60}")
        # Sadece trade_start öncesi veri — simülasyon dönemi görünmez
        pre_start_ind: dict[str, pd.DataFrame] = {}
        for sym, df in sym_ind.items():
            pre_df = df[df.index < trade_start]
            if len(pre_df) > WARMUP_BARS + 50:
                pre_start_ind[sym] = pre_df
            else:
                print(f"  ⚠ {sym}: WFO için yeterli pre-start veri yok ({len(pre_df)} bar), PROFILES kullanılacak")
        optimizer = WalkForwardOptimizer(lookback_days=WFO_LOOKBACK)
        wfo_results = optimizer.optimize_all(pre_start_ind)
        found = len(wfo_results)
        total = len(active_syms)
        print(f"\n  WFO tamamlandı: {found}/{total} coin için optimum parametreler bulundu.")

        # WFO başarısız olan coinleri ele: PROFILE varsa kalsın, yoksa listeden çıkar
        # (BASELINE ile çalışmak en kötü senaryo — gereksiz noise işlem üretir)
        wfo_failed = [s for s in active_syms if s not in wfo_results]
        dropped = []
        for s in wfo_failed:
            if s in PROFILES:
                print(f"  ↩ {s}: WFO yok → PROFILE kullanılıyor")
            else:
                print(f"  ✗ {s}: WFO yok + PROFILE yok → listeden çıkarıldı (BASELINE riskli)")
                dropped.append(s)
        if dropped:
            active_syms = [s for s in active_syms if s not in dropped]
            for s in dropped:
                sym_ind.pop(s, None)
            print(f"  → Kalan aktif coinler ({len(active_syms)}): {', '.join(active_syms)}")

    # ── M4 State başlatma ────────────────────────────────────────
    _m4_state = None
    if m4_mode:
        _m4_state = _M4State(
            last_regime_check=trade_start,
            last_wfo_check=trade_start,
            active_syms=list(sym_ind.keys()),
            wfo_results=dict(wfo_results) if wfo_results else {},
        )

    # 4) Per-coin strateji + BTC rejim
    strategies: dict[str, TrendFollowingStrategy] = {}
    coin_risk: dict[str, dict] = {}
    for sym in sym_ind:
        if sym == "BTC/USDT":
            continue
        # WFO sonuçları varsa kullan, yoksa PROFILES fallback
        wfo_p = wfo_results.get(sym) if wfo_results else None
        if wfo_p:
            logger.info(f"[WFO] {sym} için optimize parametreler kullanılıyor (skor={wfo_p.get('_wfo_score', 0):.3f})")
        strat, rp = make_strategy(sym, wfo_params=wfo_p, coin_df=sym_ind.get(sym), timeframe=timeframe)
        strategies[sym] = strat
        coin_risk[sym] = rp

    btc_df = btc_regime_df
    btc_regime: Optional[pd.Series] = None
    if btc_df is not None and "ema_slow" in btc_df.columns:
        btc_regime = (btc_df["close"] > btc_df["ema_slow"]).rename("btc_bull")

    # 5) Tüm timestamp'leri birleştir (belirtilen pencere içindeki)
    all_ts = sorted(set(
        ts for sym, df in sym_ind.items()
        for ts in df.index if trade_start <= ts <= trade_end
    ))

    # 6) Portfolio loop
    balance = initial_capital
    open_positions: dict[str, PPos] = {}
    closed_trades: list[PPos] = []
    equity_curve: list[tuple] = []

    # Adaptif rejim kontrolörü
    regime_ctrl = AdaptiveRegimeController(smooth_window=12, wr_window=20)
    _current_regime_params = regime_ctrl.current_params()
    _effective_max_positions = MAX_POSITIONS
    _effective_allowed: Optional[list[str]] = None  # None = hepsi

    # BTC indikatör DataFrame'i (rejim hesabı için)
    btc_ind = btc_regime_df

    # ── BTC uzun vadeli EMA sütunları (AdaptiveRegimeController için) ────────
    # Kısa vadeli ema_slow (200h ≈ 8 gün) BOĞA/AYI YÖN tespiti için yetersiz.
    # 168h (7 gün) ve 720h (30 gün) EMA ekleyerek gerçek döngü yönü tespit edilir.
    if btc_ind is not None and "close" in btc_ind.columns:
        btc_ind = btc_ind.copy()  # orijinali değiştirme
        btc_ind["ema_168h"] = btc_ind["close"].ewm(span=168, adjust=False).mean()   # 7-gün EMA
        btc_ind["ema_720h"] = btc_ind["close"].ewm(span=720, adjust=False).mean()   # 30-gün EMA

    daily_pnl_today: float = 0.0
    _last_day: Optional[str] = None

    # Coin bazlı kümülatif PnL takibi kaldırıldı — rolling window kullanılıyor
    # Her coinin kaybedebileceği max (son 30 günde): başlangıç sermayesinin %1.2'si
    COIN_MAX_LOSS = initial_capital * 0.012

    # Son çıkış zamanı (re-entry cooldown için — coin_own_bull modda her türlü çıkış)
    # Dinamik: aktif semboller listesinden oluştur (yeni coin eklenince otomatik dahil olur)
    _active_syms = list(strategies.keys())
    coin_last_exit: dict[str, Optional[pd.Timestamp]] = {sym: None for sym in _active_syms}
    coin_last_stoploss: dict[str, Optional[pd.Timestamp]] = {sym: None for sym in _active_syms}

    # Test süresi (M4 dinamik gate + M5 CB gate için)
    _test_duration_days = int((trade_end - trade_start).days)

    # M5: Portfolio Drawdown Circuit Breaker
    # Portföy peak'ten uzaklaştıkça yeni giriş boyutları otomatik küçülür.
    # DD > %22 → tüm yeni girişler durdurulur (sermaye koruması).
    # YALNIZCA çok yıllık testlerde (> M5_CB_DURATION_DAYS gün) aktif.
    # Kısa testlerde (Boğa/Ayı/Karma) devre dışı → işlem sayısını kesmez.
    _equity_peak: float = initial_capital
    _cb_mult: float = 1.0   # 1.0 = normal, 0.0 = tam durduruldu
    _m5_cb_active: bool = m5_mode and (_test_duration_days > M5_CB_DURATION_DAYS)

    # M4v14: Dinamik BTC BULL amplifikatör bayrağı
    # Problem: use_universe başlangıçta sabitlenir. 3 yıllık testte Jan 2023 BEAR → use_universe=True.
    # Bu durumda pyramiding, subrejim boost, performans boost HIÇ devreye girmez — tüm bull 2024-2025'te bile.
    # Çözüm: _btc_m1_active bayrağı her 30 günde _assess_btc_regime() ile güncellenir.
    #   BULL → amplifikatörler açık, BEAR/NEUTRAL → sadece AdaptiveRegimeCtrl (amplifikatör yok)
    # use_universe coin seçim mantığına dokunulmaz — sadece boost/pyramid kararları dinamikleşir.
    #
    # ÖNEMLİ: Dinamik güncelleme yalnızca ÇOK YILLIK testlerde (>400 gün) aktif.
    # Kısa testlerde (≤400 gün) _btc_m1_active sabit kalır → M4v13 davranışı korunur.
    # Boğa (7ay), Ayı (8ay), Karma (12ay) testleri etkilenmez.
    # 2023-2026 gibi çok yıllık testlerde dinamik devreye girer.
    # v15 FIX: Başlangıç değeri gerçek BTC verisinden belirle.
    # Önceki hata: use_universe=True → _btc_m1_active=False başlıyordu.
    # Kısa testlerde (<400 gün) dinamik güncelleme yoktu → stay-flat daima True → 0 trade.
    # Çözüm: trade_start ÖNCESI BTC datasına bakarak anlık rejimi tespit et.
    # BUG FIX A: btc_ind'i trade_start'a kadar kes — ileriye sızan EMA kirlenmesin.
    # BUG FIX B: NEUTRAL rejim de trade izni vermeli — sadece BEAR stay-flat yapar.
    if btc_ind is not None and len(btc_ind) > 0:
        _btc_pre_start = btc_ind[btc_ind.index < trade_start]  # Sadece geçmiş veri
        if len(_btc_pre_start) < 200:
            _btc_pre_start = btc_ind  # Yeterli geçmiş veri yoksa tamamını kullan
        _init_btc_regime = _assess_btc_regime(_btc_pre_start, trade_start)
        # NEUTRAL de trade izni verir — sadece BEAR stay-flat yapar (v15b fix)
        _btc_m1_active: bool = (_init_btc_regime != "BEAR")
        print(f"  [v15] BTC başlangıç rejimi: {_init_btc_regime} → "
              f"{'⬆ Trade izni açık' if _btc_m1_active else '⬇ BEAR — stay-flat aktif'}")
    else:
        _btc_m1_active: bool = True  # Veri yoksa iyimser başla
    _btc_m1_dynamic = m4_mode and (_test_duration_days > 400)  # Sadece çok yıllık testlerde
    _btc_m1_last_check: Optional[pd.Timestamp] = None   # İlk günde hemen kontrol edilsin

    print(f"\nSimülasyon çalışıyor... ({len(all_ts):,} bar adımı)")

    for ts in all_ts:
        day_str = ts.strftime("%Y-%m-%d")

        # Gün sıfırlama
        if day_str != _last_day:
            daily_pnl_today = 0.0
            _last_day = day_str

            # ── M4v7: Checkpoint kaldırıldı — intra-simülasyon coin değişimi zararlı.
            # Yeni coinler için WFO parametresi olmadığından default parametrelerle çalışıp
            # performansı düşürüyordu. Başlangıçta belirlenen coin seti ve WFO yeterli.
            # (Rolling WFO da M4v6'da kaldırılmıştı — aynı sebep: 8 günlük overfit.)

            # ── M5: Portfolio Drawdown Circuit Breaker (günlük güncelleme) ─────
            if _m5_cb_active:
                # Açık pozisyonların mevcut değerini ekle
                _open_val = 0.0
                for _ps, _pp in open_positions.items():
                    _prc = float(sym_ind[_ps].loc[ts, "close"]) if (
                        _ps in sym_ind and ts in sym_ind[_ps].index) else _pp.entry_price
                    _open_val += _prc * _pp.size - _pp.cost - _pp.pyramid_cost  # unrealized PnL
                _current_equity = balance + _open_val
                _equity_peak = max(_equity_peak, _current_equity)
                _current_dd   = 1.0 - _current_equity / _equity_peak if _equity_peak > 0 else 0.0
                # Tiered circuit breaker
                _cb_mult = 1.0
                for _dd_thresh, _mult in M5_CB_THRESHOLDS:
                    if _current_dd >= _dd_thresh:
                        _cb_mult = _mult
                        break

            # ── M4v14: Dinamik BTC BULL amplifikatör kontrolü (her 30 günde) ──────
            # use_universe coin seçimini değiştirmez — sadece boost/pyramid kararı
            # Yalnızca _btc_m1_dynamic=True (çok yıllık test, >400 gün) ise aktif.
            if _btc_m1_dynamic and btc_ind is not None and (
                _btc_m1_last_check is None
                or ts - _btc_m1_last_check >= pd.Timedelta(days=30)
            ):
                _btc_slice_now = btc_ind[btc_ind.index <= ts]
                _new_btc_regime = _assess_btc_regime(_btc_slice_now, ts)
                _new_m1 = (_new_btc_regime == "BULL")
                if _new_m1 != _btc_m1_active:
                    _mode_label = "AÇIK  ▶ pyramid+boost devreye girdi" if _new_m1 else "KAPALI ▶ defansif mod"
                    print(f"  [M4v14] {ts.strftime('%Y-%m-%d')}: BTC={_new_btc_regime} → amplifikatörler {_mode_label}")
                _btc_m1_active = _new_m1
                _btc_m1_last_check = ts

        # ── Adaptif Rejim Güncellemesi (her barda) ───────────────────────
        if btc_ind is not None and ts in btc_ind.index:
            regime, _current_regime_params = regime_ctrl.update(ts, btc_ind, sym_ind)
            _effective_max_positions = _current_regime_params.max_positions
            tier_coins = COIN_TIERS.get(_current_regime_params.coin_tier)
            _effective_allowed = tier_coins  # None = tüm coinler

            # Stratejilere giriş eşiği boost'unu uygula
            for sym, strat in strategies.items():
                strat.apply_regime_params(_current_regime_params.entry_score_boost)

            # BTC EMA200 pozisyonunu stratejilere bildir (eski mekanizma korunur)
            if "ema_slow" in btc_ind.columns and ts in btc_ind.index:
                btc_row = btc_ind.loc[ts]
                _btc_bull = float(btc_row["close"]) > float(btc_row["ema_slow"])
                for strat in strategies.values():
                    strat.set_btc_regime(_btc_bull)

        # Pause sayaçlarını ilerlet
        for sym, strat in strategies.items():
            strat.tick_pause(sym)

        # ── Stop kontrolü ────────────────────────────────────────────────
        to_close: list[str] = []
        for sym, pos in list(open_positions.items()):
            if sym not in sym_ind:
                continue
            df = sym_ind[sym]
            if ts not in df.index:
                continue
            row = df.loc[ts]
            price = float(row["close"])
            atr   = float(row.get("atr", 0.0))

            if pos.is_short:
                # ── SHORT: trailing stop AŞAĞI iner (fiyat düştükçe)
                new_trail = price + pos.trailing_mult * atr  # başlangıç: entry + trail*ATR
                if new_trail < pos.trail_price:              # trail fiyatı aşağı kayar
                    pos.trail_price = new_trail
                hit_stop  = price >= pos.stop_price          # fiyat yukarı stop'u kırdı
                hit_trail = price >= pos.trail_price         # fiyat trailing'i kırdı
            else:
                # ── LONG: trailing stop YUKARI çıkar + ZOOM-OUT (Kaufman, Turtle Trading)
                # Kâr arttıkça trailing stop genişler → büyük trendi kaçırmama
                _pnl_pct = (price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0.0
                _zoom_trail = pos.trailing_mult
                # Zoom-out sadece coin kendi boğa trendindeyse aktif (Ayı/Karma'da normal davranış)
                if pos.is_coin_bull:
                    if m6_mode:
                        # M6: kazananı erken bırak — düşük kârda shake-out azalt
                        if   _pnl_pct >= 0.25: _zoom_trail = pos.trailing_mult * 2.2
                        elif _pnl_pct >= 0.15: _zoom_trail = pos.trailing_mult * 1.80
                        elif _pnl_pct >= 0.08: _zoom_trail = pos.trailing_mult * 1.45
                        elif _pnl_pct >= 0.03: _zoom_trail = pos.trailing_mult * 1.20
                    elif _pnl_pct >= 0.20:          # %20+ kârda — büyük trend yakala
                        _zoom_trail = pos.trailing_mult * 2.0
                    elif _pnl_pct >= 0.12:         # %12+ kârda — rahat tut
                        _zoom_trail = pos.trailing_mult * 1.50
                    elif _pnl_pct >= 0.06:         # %6+ kârda — hafif genişlet
                        _zoom_trail = pos.trailing_mult * 1.25
                new_trail = price - _zoom_trail * atr
                if new_trail > pos.trail_price:
                    pos.trail_price = new_trail
                hit_stop  = price <= pos.stop_price
                hit_trail = price <= pos.trail_price

                # ── M5-1: Partial Exit ── v4'te KALDIRILDI ──────────────────────────
                # v2/v3'te Boğa PF 1.42→1.27, Karma PF 1.16→1.01 regresyonuna yol açtı.
                # Kripto fat-tail doğası: outlier kazananlar (top-5% işlemler) toplam kârın
                # büyük bölümünü oluşturuyor. Partial exit bu kazananları erken kesiyor.
                # pass — partial exit devre dışı

                # ── M5-4: Momentum Decay Exit ── v4'te KALDIRILDI ───────────────────
                # v3'te PF düşüşünün ikinci nedeni: büyük trendleri "tükenme" olarak yorumluyor.
                # Kripto güçlü trend dönemlerinde RSI yüksek + hacim azalabilir + ATR daralabilir
                # ama trend devam ediyor → erken çıkış.
                # pass — momentum decay devre dışı

                # ── M4v11: PYRAMIDING — Turtle Trading "add-to-winners" ────────────
                # Kârlı pozisyona ek birim ekle (Turtle: her 0.5N harekette 1 unit daha)
                # Koşullar: M4 BTC-BULL modu + coin kendi bull trendinde + maks 2 pyramid
                # + ADX ≥ 28 filtresi: choppy coinde (BNB gibi) pyramid ateşlenmesin
                # Threshold: %5 kârda 1. ekleme (%50 lot), %12'de 2. ekleme (%25 lot)
                _adx_for_pyramid = float(row.get("adx", 0.0))
                # M6: pyramiding HER ZAMAN açık (coin kendi bull'undayken) — BTC bull gate yok
                #     3 birim, daha büyük lot, %18 nakit cap → büyük trende para döker
                # M4v14: not use_universe → _btc_m1_active (dinamik BTC bull tespiti)
                _pyr_base = pos.is_coin_bull and not pos.is_short and atr > 0
                # M6 v9: Pyramid SADECE BTC-bull rejiminde (önceden BEAR'de de pyramid yapıyordu,
                # bear bounce'larda compound kayıp veriyordu)
                _pyr_m6 = (_pyr_base and m6_mode and pos.pyramid_count < 4
                           and _adx_for_pyramid >= 24 and _btc_m1_active)
                _pyr_m4 = (_pyr_base and not m6_mode and pos.pyramid_count < 2
                           and m4_mode and _btc_m1_active and _adx_for_pyramid >= 28)
                if _pyr_m6 or _pyr_m4:
                    if m6_mode:
                        # M6: v3 pyramid (sweet spot) + 4. moonshot birim korunuyor
                        _pyramid_thresholds = [0.05, 0.12, 0.22, 0.35]
                        _pyramid_sizes      = [0.60, 0.40, 0.25, 0.15]
                        _pyr_cash_cap       = 0.15
                    else:
                        _pyramid_thresholds = [0.05, 0.12]
                        _pyramid_sizes      = [0.50, 0.25]
                        _pyr_cash_cap       = 0.10
                    for _pi, _pthresh in enumerate(_pyramid_thresholds):
                        if _pnl_pct >= _pthresh and pos.pyramid_count == _pi:
                            # M6: 2. ve 3. birim için BTC bull onayı gerekir
                            # → bear bölgesinde compounding'i kapat (Karma DD'sini azaltır)
                            if m6_mode and _pi >= 1 and not _btc_m1_active:
                                break
                            _add_size = pos.size_at_entry * _pyramid_sizes[_pi]
                            _add_fill = price * (1 + SLIPPAGE)
                            _add_comm = _add_fill * _add_size * COMMISSION
                            _add_cost = _add_fill * _add_size + _add_comm
                            # Güvenlik: yeterli nakit var mı?
                            if _add_cost <= balance * _pyr_cash_cap and _add_cost >= MIN_ORDER_SIZE:
                                pos.size += _add_size
                                pos.pyramid_cost += _add_cost
                                balance -= _add_cost
                                pos.pyramid_count = _pi + 1
                                # Stop'u yeni giriş fiyatının 2 ATR altına çek (mevcut stop'u koru)
                                _new_stop = price - 2.0 * atr
                                if _new_stop > pos.stop_price:
                                    pos.stop_price = _new_stop
                            break  # Sadece bir eşik aynı anda kontrol edilsin

            if hit_stop or hit_trail:
                reason = "stop_loss" if hit_stop else "trailing_stop"
                if pos.is_short:
                    # SHORT kapama: daha pahalıya geri al → ters PnL
                    exit_px      = price * (1 + SLIPPAGE)
                    exit_comm    = exit_px * pos.size * COMMISSION
                    gross_pnl    = (pos.entry_price - exit_px) * pos.size
                    net_pnl      = gross_pnl - exit_comm
                    pos.exit_price  = exit_px
                    pos.exit_time   = ts
                    pos.exit_reason = reason
                    pos.pnl         = net_pnl
                    balance        += pos.cost + net_pnl  # marjini geri al + kar/zarar
                else:
                    # LONG kapama (M4v11: pyramid_cost PnL muhasebesine dahil)
                    exit_px = price * (1 - SLIPPAGE)
                    net_proceeds = exit_px * pos.size * (1 - COMMISSION)
                    pos.exit_price = exit_px
                    pos.exit_time  = ts
                    pos.exit_reason = reason
                    # PnL = çıkış geliri - ilk giriş maliyeti - pyramid maliyetleri
                    pos.pnl = net_proceeds - pos.cost - pos.pyramid_cost
                    balance += net_proceeds
                to_close.append(sym)
                closed_trades.append(pos)
                # LONG SL cooldown kaydı: SHORT SL'de cooldown yok (D2 SHORT zinciri korunur)
                if reason == "stop_loss" and not pos.is_short:
                    coin_last_stoploss[sym] = ts
                coin_last_exit[sym] = ts
                correlation_registry.register_close(sym)
                daily_pnl_today += pos.pnl
                strategies[sym].record_outcome(sym, pos.pnl > 0, pnl=pos.pnl)
                regime_ctrl.record_trade(pos.pnl > 0)

        for sym in to_close:
            del open_positions[sym]

        # ── Strategy exit sinyali ────────────────────────────────────────
        for sym in list(open_positions.keys()):
            if sym not in sym_ind:
                continue
            df = sym_ind[sym]
            if ts not in df.index:
                continue
            open_positions[sym].bars_held += 1
            # Minimum hold time: erken çıkışları engelle
            if open_positions[sym].bars_held < open_positions[sym].min_hold_bars:
                continue
            slice_df = df.loc[:ts]
            try:
                should_exit, _ = strategies[sym].should_exit(
                    sym, slice_df, entry_price=0,
                    is_short=open_positions[sym].is_short,
                )
            except Exception:
                should_exit = False
            if should_exit:
                pos = open_positions[sym]
                price = float(df.loc[ts, "close"])
                if pos.is_short:
                    exit_px   = price * (1 + SLIPPAGE)
                    exit_comm = exit_px * pos.size * COMMISSION
                    gross_pnl = (pos.entry_price - exit_px) * pos.size
                    net_pnl   = gross_pnl - exit_comm
                    pos.exit_price  = exit_px
                    pos.exit_time   = ts
                    pos.exit_reason = "strategy_exit"
                    pos.pnl = net_pnl
                    balance += pos.cost + net_pnl
                else:
                    # M4v11: pyramid_cost PnL muhasebesine dahil
                    exit_px = price * (1 - SLIPPAGE)
                    net_proceeds = exit_px * pos.size * (1 - COMMISSION)
                    pos.exit_price = exit_px
                    pos.exit_time  = ts
                    pos.exit_reason = "strategy_exit"
                    pos.pnl = net_proceeds - pos.cost - pos.pyramid_cost
                    balance += net_proceeds
                del open_positions[sym]
                closed_trades.append(pos)
                coin_last_exit[sym] = ts
                correlation_registry.register_close(sym)
                daily_pnl_today += pos.pnl
                strategies[sym].record_outcome(sym, pos.pnl > 0, pnl=pos.pnl)
                regime_ctrl.record_trade(pos.pnl > 0)

        # ── Günlük max kayıp kontrolü ────────────────────────────────────
        daily_loss_ok = (daily_pnl_today / max(balance, 1.0)) > -DAILY_MAX_LOSS

        # ── Giriş sinyalleri ─────────────────────────────────────────────
        # Rejim kontrolöründen gelen dinamik limitler kullanılır
        eff_max = _current_regime_params.max_positions
        pos_size_mult = _current_regime_params.position_size_mult

        if daily_loss_ok and len(open_positions) < eff_max:
            # Universe modunda rejime göre coin sayısını dinamik kısıtla
            # (CoinAnalyzer skoru yüksek olanlar önce sıralanmış → en iyiler listede önce)
            all_candidates = list(sym_ind.keys())
            if use_universe:
                regime_name = regime_ctrl.current_regime().name  # "BEAR", "BULL" vb.
                max_coins_for_regime = REGIME_MAX_COINS.get(regime_name, COIN_SELECT_N)
                candidate_syms = all_candidates[:max_coins_for_regime]
            else:
                candidate_syms = all_candidates
            for sym in candidate_syms:
                if sym in open_positions:
                    continue
                if len(open_positions) >= eff_max:
                    break

                df = sym_ind[sym]
                if ts not in df.index:
                    continue
                row  = df.loc[ts]
                price = float(row["close"])
                atr   = float(row.get("atr", 0.0))
                if atr <= 0 or price <= 0:
                    continue

                slice_df = df.loc[:ts]
                if len(slice_df) < WARMUP_BARS:
                    continue

                # ── Per-coin trend durumu ────────────────────────────────
                # Coin kendi EMA200'ünün üzerindeyse → kendi boğa rejiminde
                coin_above_ema200 = False
                _ema200_col = next((c for c in ("ema_200", "ema_slow", "ema200") if c in slice_df.columns), None)
                if _ema200_col:
                    coin_ema200 = float(slice_df[_ema200_col].iloc[-1])
                    # v19 FIX: 15m'de EMA200 = 50 saatlik EMA (200×15min).
                    # Önceki %3 eşiği: 1h'da ~200h EMA için mantıklıydı, 15m'de 50h EMA için çok sert.
                    # Normal çekimlerde coin %1-2 üstünde kalır → %3 eşiği BEAR stay-flat'i tetikler.
                    # Çözüm: coin EMA200 ÜZERİNDEYSE coin_own_bull=True (0% tolerans).
                    if coin_ema200 > 0:
                        coin_above_ema200 = price >= coin_ema200

                # Coinin kendi trend gücü (ADX)
                coin_adx_strong = False
                if "adx" in slice_df.columns:
                    coin_adx_strong = float(slice_df["adx"].iloc[-1]) > 22

                # Global bear rejimde coin kendi boğa trendindeyse kısıtlamaları gevşet
                in_global_bear   = _current_regime_params.entry_score_boost >= 0.07
                in_strong_bear   = _current_regime_params.entry_score_boost >= 0.20  # STRONG_BEAR rejimi
                # BULL/STRONG_BULL tespiti (entry_score_boost negatif = daha kolay giriş)
                in_global_bull   = _current_regime_params.entry_score_boost <= -0.03
                coin_own_bull    = coin_above_ema200  # coin kendi EMA200'ü üzerinde

                # Kısıtlama seviyeleri:
                #   coin_own_bull=True  → global bear'a rağmen neredeyse normal giriş
                #   coin_own_bull=False → global bear kısıtlamaları tam devreye girer
                effective_entry_boost = _current_regime_params.entry_score_boost
                effective_pos_mult    = pos_size_mult

                if in_global_bear and coin_own_bull:
                    # Coin kendi yukarı trendinde → entry eşiği hafifçe gevşet
                    effective_entry_boost = effective_entry_boost * 0.5
                    effective_pos_mult    = max(pos_size_mult, 0.75)
                    strategies[sym].apply_regime_params(effective_entry_boost)

                # M4: Hybrid modda (universe=True) AdaptiveRegimeCtrl çok kısıtlayıcı oluyor
                # M4v14 FIX: Hybrid sizing yalnızca use_universe=True'da uygulanır.
                # M1/M4-Hibrit (use_universe=False) modunda AdaptiveRegimeCtrl yeterlidir;
                # _btc_m1_active=False sadece amplifikatörleri (pyramid/boost) kapatır —
                # bazal pozisyon boyutunu düşürmez → Karma testinde regresyon önlendi.
                if m4_mode and use_universe and not _btc_m1_active:
                    _rn = regime_ctrl.current_regime().name
                    # AdaptiveCtrl (0.25/0.45/0.80/1.10/1.20) ile M4v2 (0.60/0.80/1.0/1.0/1.0)
                    # arasında denge: hybrid coin seti daha fazla çeşitlendirme → orta seviye risk
                    _M4_HYBRID_MULT = {
                        "STRONG_BEAR": 0.35,
                        "BEAR":        0.55,
                        "NEUTRAL":     0.85,
                        "BULL":        1.05,
                        "STRONG_BULL": 1.10,
                    }
                    effective_pos_mult = _M4_HYBRID_MULT.get(_rn, effective_pos_mult)

                # M4v11/v12: BTC BULL döneminde subrejime-göre pozisyon boost
                # BEAR/NEUTRAL'da boost yok (choppy BNB gibi coinleri korur)
                # Sadece BULL/STRONG_BULL mikro-rejimde büyük pozisyon → gerçek trend yakalanır
                # M4v14: not use_universe → _btc_m1_active (dinamik)
                if m4_mode and _btc_m1_active:
                    _rn_bull = regime_ctrl.current_regime().name
                    _M1_BULL_BOOST = {
                        "STRONG_BEAR": 1.00,  # Boost yok — micro-bear koru
                        "BEAR":        1.00,  # Boost yok — micro-bear koru
                        "NEUTRAL":     1.10,  # Küçük boost — nötr dönem
                        "BULL":        1.40,  # Büyük boost — güçlü trend
                        "STRONG_BULL": 1.55,  # En büyük boost — rallide kal
                    }
                    effective_pos_mult = float(np.clip(
                        effective_pos_mult * _M1_BULL_BOOST.get(_rn_bull, 1.0), 0.1, 1.80
                    ))

                # ── Kalite filtreleri ─────────────────────────────────────────

                # 1) Son 30 günlük kayan kayıp limiti (geçmiş kötü dönem gelecek girişi engellesin)
                sym_trades_30d = [t for t in closed_trades
                                  if t.symbol == sym and t.exit_time is not None
                                  and ts - t.exit_time <= pd.Timedelta(days=30)]
                rolling_pnl_30d = sum(t.pnl for t in sym_trades_30d)
                if rolling_pnl_30d < -COIN_MAX_LOSS:
                    continue

                # M4v13: Performans-bazlı pozisyon boost (BTC BULL modunda)
                # Coinin son 30 günlük performansına göre boost ver:
                #   Kârlı coin (+$30 üstü) → %30 ek pozisyon (gerçek trend yakalansın)
                #   Zararlı coin → boost yok (BNB, DOGE gibi choppy coinleri korur)
                # Bu yaklaşım: ETH gibi iyi coinlerde büyük pozisyon,
                #              BNB gibi kötü coinlerde normal boyut
                # M4v14: not use_universe → _btc_m1_active (dinamik)
                if m4_mode and _btc_m1_active:
                    if len(sym_trades_30d) >= 3 and rolling_pnl_30d >= 30.0:
                        # Son 30 günde kârlı → ek boost
                        effective_pos_mult = float(np.clip(effective_pos_mult * 1.30, 0.1, 1.80))
                    elif len(sym_trades_30d) >= 3 and rolling_pnl_30d < -20.0:
                        # Son 30 günde zararlı → boost iptal (subrejim boostunu da geri al)
                        effective_pos_mult = float(np.clip(effective_pos_mult * 0.85, 0.1, 1.80))

                # 2) Çıkış sonrası bekleme
                _sl_hours = coin_risk[sym].get("sl_cooldown_hours", 24)
                if coin_own_bull:
                    # Boğa trendinde: LONG SL sonrası sl_cooldown_hours bekle (yanlış re-entry engeli)
                    # NOT: coin_last_stoploss yalnızca LONG SL'de set edilir → SHORT SL buraya düşmez
                    sym_last_sl_b = coin_last_stoploss.get(sym)
                    if sym_last_sl_b is not None and ts - sym_last_sl_b < pd.Timedelta(hours=_sl_hours):
                        continue
                    # M4v11: Normal çıkış sonrası 24h→8h (boğa trendinde daha hızlı re-entry)
                    # Trend devam ederken 24h beklemek fırsatı kaçırıyordu.
                    last_exit_ts = coin_last_exit.get(sym)
                    if last_exit_ts is not None and ts - last_exit_ts < pd.Timedelta(hours=8):
                        continue
                else:
                    # Ayı trendinde LONG SL sonrası bekle (SHORT SL cooldown triggerlamaz → zincir korunur)
                    sym_last_sl = coin_last_stoploss.get(sym)
                    if sym_last_sl is not None and ts - sym_last_sl < pd.Timedelta(hours=_sl_hours):
                        continue

                # 3) ADX: güçlü trend şart
                if "adx" in slice_df.columns:
                    adx_val = float(slice_df["adx"].iloc[-1])
                    # Coin EMA200'e yakınsa (±5%) → sideways piyasa → daha yüksek ADX şart
                    if _ema200_col and coin_ema200 > 0:
                        ema_gap_pct = abs(price / coin_ema200 - 1)
                        min_adx = 27 if ema_gap_pct < 0.05 else (22 if coin_own_bull else 18)
                    else:
                        min_adx = 22 if coin_own_bull else 18
                    if adx_val < min_adx:
                        continue

                # 4) ATR kalite: %60 minimum
                if "atr" in slice_df.columns and len(slice_df) >= 50:
                    atr_mean = float(slice_df["atr"].tail(50).mean())
                    if atr_mean > 0 and atr < atr_mean * 0.60:
                        continue

                # 5) 7-günlük WR filtresi: tüm coinlere uygula (kayıp serisi sonrası dur)
                sym_trades_wr = [t for t in closed_trades if t.symbol == sym]
                sym_recent_7d = [t for t in sym_trades_wr
                                 if t.exit_time is not None and ts - t.exit_time < pd.Timedelta(days=7)]
                if len(sym_recent_7d) >= 3:
                    recent_wr = sum(1 for t in sym_recent_7d if t.pnl > 0) / len(sym_recent_7d)
                    if recent_wr < 0.40:
                        continue

                # 6) Breakout filtresi: yavaş trendli coinlerde (yalnızca breakout_bars > 0 ise)
                _bo_bars = coin_risk[sym].get("breakout_bars", 0)
                if _bo_bars > 0 and len(slice_df) >= _bo_bars:
                    _recent_high = float(slice_df["high"].tail(_bo_bars).max())
                    if price < _recent_high * 0.995:
                        continue
                # ────────────────────────────────────────────────────────

                try:
                    signal = strategies[sym].generate_signal(
                        sym, slice_df, allow_short=True,
                    )
                except Exception:
                    continue

                is_short_signal = signal.side == Side.SHORT
                if signal.side not in (Side.BUY, Side.SHORT):
                    continue

                # ── Yön bazlı post-signal filtreler ──────────────────────
                ema_col = next((c for c in ("ema_50", "ema50", "ema_fast") if c in slice_df.columns), None)

                if not is_short_signal:
                    # LONG-specific: EMA50 onayı (son 3 bar close > EMA50 olmalı)
                    if ema_col and len(slice_df) >= 3:
                        last3 = slice_df.tail(3)
                        if not (last3["close"] > last3[ema_col]).all():
                            continue

                    # LONG-specific (v16): BEAR rejimde akıllı stay-flat.
                    # v11b sadece BEAR=stay-flat yapıyordu ama coin_own_bull'u görmezden geliyordu.
                    # Bu tutarsızlık: satır 1334'te coin_own_bull → kısıtla gevşet deniyor,
                    # ama burada coin_own_bull olsa bile bloklanıyordu → 0 trade.
                    #
                    # v16 kuralı:
                    #   STRONG_BEAR → tüm coinleri blokla (piyasa çöküyor)
                    #   BEAR + coin kendi EMA200 ALTINDA → blokla (hem global hem coin bearish)
                    #   BEAR + coin kendi EMA200 ÜSTÜNDE → izin ver (coin güçlü, global bear geçici)
                    #   NEUTRAL/BULL → serbest
                    if in_strong_bear or (in_global_bear and not coin_own_bull):
                        continue

                else:
                    # SHORT-specific: EMA50 aşağıda olmalı (son 2 bar close < EMA50)
                    if ema_col and len(slice_df) >= 2:
                        last2 = slice_df.tail(2)
                        if not (last2["close"] < last2[ema_col]).all():
                            continue  # EMA50 üzerindeyken short açma
                    # SHORT-specific: 14-günlük negatif momentum teyidi (downtrend onayı)
                    bars_14d = 14 * _bpd  # v13: TF'ye göre (1h:336, 15m:1344, 1m:20160)
                    if len(slice_df) > bars_14d:
                        price_14d_ago = float(slice_df.iloc[-bars_14d]["close"])
                        change_14d = (price / price_14d_ago) - 1
                        if change_14d > 0.05:
                            continue  # 14 günde %5+ yükseldiyse short açma
                        if change_14d < -0.25:
                            continue  # 14 günde %25+ düştüyse short açma: aşırı satım → bounce riski

                # ── M5-3: Re-entry Cooldown (v4) ─────────────────────────────────────
                # Stop hit'ten sonra M5_COOLDOWN_DAYS gün aynı coinden uzak dur.
                # Neden: Stop bölgesinde whipsaw yaygın — hemen re-entry genellikle tekrar stop.
                # Kanıt: Trend-following literatüründe "dead zone" sonrası entry kalitesi düşük.
                # Kaynak: Covel "Trend Following", Schwager "Market Wizards" exit/re-entry disiplin.
                # Kazananlar etkilenmez: 3 gün sonra trend devam ediyorsa sinyal yeniden üretilir.
                if m5_mode and not is_short_signal:
                    _last_sl = coin_last_stoploss.get(sym)
                    if _last_sl is not None and (ts - _last_sl).days < M5_COOLDOWN_DAYS:
                        continue  # cooldown — whipsaw bölgesi, bekle

                # SHORT sinyal — sadece coin ayı trendindeyse (coin_own_bull değilse)
                if is_short_signal and coin_own_bull:
                    continue  # coin boğa trendinde → short açma

                # BULL/STRONG_BULL global rejimde SHORT yasak (ana trende karşı gidilmez)
                if is_short_signal and in_global_bull:
                    continue  # global boğa piyasasında short → kayıp

                # ── M5-2: Circuit Breaker — DD > %22 ise yeni giriş yok ────────────
                # Sadece çok yıllık testlerde (>400 gün) aktif
                if _m5_cb_active and _cb_mult == 0.0:
                    continue  # portföy DD > %22 → tüm yeni girişler durduruldu

                # Pozisyon boyutu: per-coin efektif büyüklük çarpanı
                risk_pct = coin_risk[sym].get("risk_per_trade", RISK_PER_TRADE)
                # ADX scale: güçlü trendlerde daha büyük pozisyon (Turtle Trading prensibi)
                _adx_now = float(slice_df["adx"].iloc[-1]) if "adx" in slice_df.columns else 20
                adx_scale = strategies[sym].get_adx_scale(_adx_now) if hasattr(strategies[sym], 'get_adx_scale') else 1.0
                # Kelly fraction: geçmiş işlemlerden dinamik boyut
                _sym_past = [t for t in closed_trades if t.symbol == sym and t.exit_time is not None]
                kelly_scale = 1.0
                if len(_sym_past) >= 10:
                    _wins = [t for t in _sym_past if t.pnl > 0]
                    _loss = [t for t in _sym_past if t.pnl <= 0]
                    if _wins and _loss:
                        _wr = len(_wins) / len(_sym_past)
                        _avg_win  = sum(t.pnl for t in _wins)  / len(_wins)
                        _avg_loss = abs(sum(t.pnl for t in _loss) / len(_loss))
                        _rr = _avg_win / _avg_loss if _avg_loss > 0 else 1.0
                        _k  = _wr - (1 - _wr) / _rr
                        kelly_scale = float(np.clip(_k / 2.0, 0.5, 2.0))  # half-Kelly
                # Kombine büyüklük çarpanı (rejim × ADX × Kelly)
                # M6: coin kendi bull'undayken üst sınır 3.0 (agresif sizing)
                _mult_cap = 3.0 if (m6_mode and coin_own_bull) else 2.0
                combined_mult = float(np.clip(effective_pos_mult * adx_scale * kelly_scale, 0.1, _mult_cap))

                # M5-NOT: ATR Percentile Sizing kaldırıldı (v2 revize).
                # Sorun: Bull trendde ATR yükseliyor → percentile artar → boyut kesilir
                # = tam tersi etki (en iyi fırsatlar kaçıyor).
                # ATR percentile ancak ADX ile birleştirilince (vol × trend quality) anlamlı.
                # Şimdilik ER gate + momentum decay + partial exit daha temiz iyileştirme sağlıyor.

                # ── M5-2: Circuit Breaker boyut çarpanı (sadece çok yıllık testlerde) ──
                if _m5_cb_active and _cb_mult < 1.0:
                    combined_mult = float(np.clip(combined_mult * _cb_mult, 0.1, _mult_cap))

                _conf = float(getattr(signal, 'confidence_score', 0.0) or 0.0)
                # M4v11: Yüksek güven + coin boğa trendi → daha büyük pozisyon
                # risk_pct_adj direkt artırılır (max_cost boost sadece capleme durumunu kapsar)
                # Koşullar: conf≥0.78 + EMA200 üstünde + ADX≥28 (gerçek trend teyidi)
                # M4v11: "İyi giden coin" tespiti — 3 koşul
                # 1) coin EMA200'ün %3+ üstünde (gerçek uptrend)
                # 2) ADX ≥ 28 (güçlü trend)
                # 3) Choppiness Index < 56 (trending, 61.8'in altı = normal, <38.2 = güçlü trend)
                #    BNB gibi choppy coinlerde CI 55-65 → filtre engeller; ETH rally'de CI < 50
                _chop_now = float(slice_df["choppiness"].iloc[-1]) if "choppiness" in slice_df.columns else 61.8
                # M6: high-conf bull eşiklerini gevşet (ADX 22, chop 62) → daha çok entry size boost alır
                _chop_thr = 62.0 if m6_mode else 56.0
                _chop_trending = not pd.isna(_chop_now) and _chop_now < _chop_thr
                _adx_thr = 22 if m6_mode else 28
                _is_high_conf_bull = (
                    not is_short_signal
                    and coin_own_bull
                    and _adx_now >= _adx_thr
                    and _chop_trending
                )
                # SHORT sinyalde global ayı teyidi yoksa (%70 boyut — geçiş döneminde yanlış SHORT riski)
                if is_short_signal and not in_global_bear:
                    combined_mult *= 0.65
                risk_pct_adj = risk_pct * combined_mult
                if _is_high_conf_bull:
                    # M6 v9: BTC bull onayında SÜPER (×2.6/×3.0), bear/nötr → DEFANSİF (×1.0)
                    # Önceki ×2.0 BEAR'de tek SL'de %1.5+ kayıp veriyordu (DD %36).
                    if m6_mode:
                        if _btc_m1_active:
                            _m6_boost = 3.0 if _adx_now >= 35 else 2.6
                        else:
                            _m6_boost = 1.0   # bear/nötr → normal risk (defansif)
                        risk_pct_adj = min(risk_pct_adj * _m6_boost, risk_pct * 5.0)
                    else:
                        risk_pct_adj = min(risk_pct_adj * 1.5, risk_pct * 3.0)
                risk_amt = balance * risk_pct_adj
                atr_stop = coin_risk[sym].get("atr_stop_multiplier", ATR_STOP_MULT)

                if is_short_signal:
                    stop_px   = price + atr_stop * atr   # SHORT stop: yukarıda
                    stop_dist = max(stop_px - price, price * 0.001)
                else:
                    stop_px   = price - atr_stop * atr
                    stop_dist = max(price - stop_px, price * 0.001)
                size = risk_amt / stop_dist

                # Max position cap (per-coin override mümkün)
                # M4v11: Yüksek güven → max_pos_pct de artır (capleme durumunu da kapsar)
                _max_pos_pct = coin_risk[sym].get("max_position_pct", MAX_POSITION_PCT)
                # v9: BEAR rejimde TÜM modellerde pos boyutu yarıya
                # (M4/M5 default %6-8 pos × 60-80 trade × WR %25 = %8-10 zarar veriyordu)
                if in_global_bear:
                    _max_pos_pct *= 0.5
                    risk_pct_adj *= 0.5
                if _is_high_conf_bull:
                    if m6_mode:
                        # M6 v9: BTC bull onayında SÜPER (×3.0, cap %60), bear/nötr → DEFANSİF (×1.0, cap %20)
                        # Önceki %45 cap BEAR'de tek trade'de sermayenin yarısını riske atıyordu.
                        if _btc_m1_active:
                            _m6_mult = 3.0
                            _m6_max_cap = 0.60 if _adx_now >= 35 else 0.50
                        else:
                            _m6_mult = 1.0       # normal mod (boost yok)
                            _m6_max_cap = 0.20   # max %20 (önceden %45 → DD %36)
                        _max_pos_pct = min(_max_pos_pct * _m6_mult, _m6_max_cap)
                    else:
                        _max_pos_pct = min(_max_pos_pct * 1.5, 0.35)   # 20% → 30% (max 35%)
                _pos_cap = (3.0 if m6_mode else 2.0) if _is_high_conf_bull else 1.5
                max_cost = balance * _max_pos_pct * min(combined_mult, _pos_cap)
                if size * price > max_cost:
                    size = max_cost / price

                cost = size * price
                if cost < MIN_ORDER_SIZE:
                    continue
                if cost > balance * 0.95:
                    continue

                if is_short_signal:
                    # SHORT giriş: daha düşük fiyattan sat (slippage ters)
                    fill_price = price * (1 - SLIPPAGE)
                    entry_comm = fill_price * size * COMMISSION
                    # Marjin rezerv = potansiyel maks kayıp
                    total_cost = atr_stop * atr * size + entry_comm
                else:
                    fill_price = price * (1 + SLIPPAGE)
                    entry_comm = fill_price * size * COMMISSION
                    total_cost = fill_price * size + entry_comm

                if total_cost > balance:
                    continue

                # Per-coin trailing mult
                base_trail = coin_risk[sym].get("trailing_stop_atr_multiplier", TRAILING_MULT)
                regime_trail_adj = _current_regime_params.trailing_mult_boost
                if coin_own_bull and not is_short_signal:
                    # Boğa trendli coin: 1.4× geniş trail + BULL rejiminde ek bonus
                    # Boğa döneminde daha uzun tut → daha fazla kar yakala
                    bull_bonus = max(0.0, regime_trail_adj)  # sadece pozitif adj ekle
                    # ADX boost: güçlü trend → daha geniş trail (erken çıkma)
                    # NOT: _conf_trail kaldırıldı — geniş trail Karma/Ayı tersine dönüşlerinde zarar büyütüyor
                    _adx_trail = 1.0 if _adx_now >= 35 else (0.5 if _adx_now >= 30 else 0.0)
                    trail_mult = max(2.0, base_trail * 1.4 + bull_bonus + _adx_trail)
                else:
                    trail_mult = max(1.5, base_trail + regime_trail_adj)

                if is_short_signal:
                    trail_px = fill_price + trail_mult * atr  # SHORT: trail YUKARDA başlar
                else:
                    trail_px = fill_price - trail_mult * atr

                # M4v11: hold_bars değiştirilmedi — 24h mecburi tutma Karma/Ayı döneminde zararlı
                # Trend tersine döndüğünde SE erken çıkışı koruyucu, kaldırılmaz
                # v13: bar-cinsinden hold süresi (1h: 12/6 saat, 15m: 3/1.5 saat, 1m: 12/6 dk)
                # M6 1m için min hold çok kısa olur — bu yüzden bar oranıyla scale ediyoruz
                _hold_scale = _bpd / 24  # 1h:1, 15m:4, 1m:60
                hold_bars = int((12 if coin_own_bull else 6) * _hold_scale)

                pos = PPos(
                    symbol=sym,
                    entry_price=fill_price,
                    stop_price=(fill_price + atr_stop * atr) if is_short_signal else (fill_price - atr_stop * atr),
                    trail_price=trail_px,
                    size=size,
                    cost=total_cost,
                    entry_time=ts,
                    entry_atr=atr,
                    trailing_mult=trail_mult,
                    min_hold_bars=hold_bars,
                    is_coin_bull=coin_own_bull,
                    is_short=is_short_signal,
                    size_at_entry=size,   # M4v11: pyramid hesabı için orijinal lot
                    r_value=stop_dist,    # M5: 1R mesafesi (partial exit için)
                )
                open_positions[sym] = pos
                balance -= total_cost
                correlation_registry.register_open(sym)

        # ── Equity curve — sadece tüm açık pozisyonların fiyatı varsa ekle ──
        # balance = serbest nakit (giriş maliyetleri zaten düşüldü)
        # equity = balance + anlık pozisyon değerleri (px * size)
        all_priced = True
        open_val = 0.0
        for sym, p in open_positions.items():
            if sym in sym_ind and ts in sym_ind[sym].index:
                px = float(sym_ind[sym].loc[ts, "close"])
                if px > 0:
                    open_val += px * p.size
                else:
                    all_priced = False
            else:
                all_priced = False
        if all_priced or not open_positions:
            equity_curve.append((ts, balance + open_val))

    # ── Açık pozisyonları kapat (backtest sonu) ───────────────────────────────
    for sym, pos in list(open_positions.items()):
        df = sym_ind[sym]
        # trade_end sınırındaki son fiyatı kullan (tüm veri değil!)
        df_within = df[df.index <= trade_end]
        if df_within.empty:
            df_within = df
        last_row   = df_within.iloc[-1]
        last_price = float(last_row["close"])
        be_ts      = df_within.index[-1]

        if pos.is_short:
            exit_px   = last_price * (1 + SLIPPAGE)
            exit_comm = exit_px * pos.size * COMMISSION
            gross_pnl = (pos.entry_price - exit_px) * pos.size
            net_pnl   = gross_pnl - exit_comm
            pos.exit_price  = exit_px
            pos.exit_time   = be_ts
            pos.exit_reason = "backtest_end"
            pos.pnl         = net_pnl
            balance        += pos.cost + net_pnl
        else:
            exit_px      = last_price * (1 - SLIPPAGE)
            net_proceeds = exit_px * pos.size * (1 - COMMISSION)
            pos.exit_price  = exit_px
            pos.exit_time   = be_ts
            pos.exit_reason = "backtest_end"
            # M4v11: pyramid_cost PnL muhasebesine dahil
            pos.pnl         = net_proceeds - pos.cost - pos.pyramid_cost
            balance        += net_proceeds
        closed_trades.append(pos)
        correlation_registry.register_close(sym)

    # ── Raporla ──────────────────────────────────────────────────────────────
    _print_report(
        closed_trades, equity_curve, initial_capital, balance, raw_data, days,
        regime_summary=regime_ctrl.regime_summary(),
        trade_start=trade_start,
        trade_end=trade_end,
        active_syms=active_syms,
    )

    # ── JSON State Export (live dashboard için) ───────────────────────────
    if json_out:
        import json as _json
        _mode = "M6" if m6_mode else ("M5" if m5_mode else ("M4" if m4_mode else "M1"))
        # PnL hesapla
        _total_pnl_pct = (balance - initial_capital) / initial_capital * 100
        # Max drawdown
        _eq = [e[1] for e in equity_curve] if equity_curve and isinstance(equity_curve[0], (list, tuple)) else equity_curve
        _peak = initial_capital
        _max_dd = 0.0
        for _eq_val in _eq:
            _peak = max(_peak, _eq_val)
            _dd = (_peak - _eq_val) / _peak if _peak > 0 else 0
            _max_dd = max(_max_dd, _dd)
        # Win rate
        _finished = [t for t in closed_trades if t.exit_time is not None]
        _wins = sum(1 for t in _finished if t.pnl > 0)
        _wr = _wins / len(_finished) * 100 if _finished else 0.0
        # Açık pozisyonlar
        _open = []
        for _sym, _pos in open_positions.items():
            _last_price = float(sym_ind[_sym].iloc[-1]["close"]) if _sym in sym_ind else _pos.entry_price
            _unreal_pnl = (_last_price - _pos.entry_price) * _pos.size if not _pos.is_short else (_pos.entry_price - _last_price) * _pos.size
            _unreal_pct = (_last_price - _pos.entry_price) / _pos.entry_price * 100 if not _pos.is_short else (_pos.entry_price - _last_price) / _pos.entry_price * 100
            _open.append({
                "symbol": _sym,
                "side": "SHORT" if _pos.is_short else "LONG",
                # v14: saatli tarih (dakika cinsi) — kullanıcı isteği
                "entry_date": _pos.entry_time.strftime("%Y-%m-%d %H:%M") if _pos.entry_time else "",
                "entry_price": round(_pos.entry_price, 6),
                "last_price": round(_last_price, 6),
                "size": round(_pos.size, 6),
                "stop_price": round(_pos.stop_price, 6),
                "trail_price": round(_pos.trail_price, 6),
                "unrealized_pnl": round(_unreal_pnl, 2),
                "unrealized_pct": round(_unreal_pct, 2),
                "cost": round(_pos.cost, 2),
                "pyramid_count": _pos.pyramid_count,
            })
        # Kapalı işlemler (son 100)
        _closed = []
        for _t in reversed(_finished[-100:]):
            # v14: closed trade'lere size + cost eklendi, tarihler dakika cinsi
            _t_cost = round(_t.entry_price * _t.size, 2) if _t.size else 0.0
            _closed.append({
                "symbol": _t.symbol,
                "side": "SHORT" if _t.is_short else "LONG",
                "entry_date": _t.entry_time.strftime("%Y-%m-%d %H:%M") if _t.entry_time else "",
                "exit_date":  _t.exit_time.strftime("%Y-%m-%d %H:%M")  if _t.exit_time  else "",
                "entry_price": round(_t.entry_price, 6),
                "exit_price": round(_t.exit_price, 6),
                "size": round(_t.size, 6) if _t.size else 0.0,
                "cost": _t_cost,
                "pnl": round(_t.pnl, 2),
                "pnl_pct": round((_t.exit_price - _t.entry_price) / _t.entry_price * 100 if not _t.is_short else (_t.entry_price - _t.exit_price) / _t.entry_price * 100, 2),
                "exit_reason": _t.exit_reason,
                "bars_held": _t.bars_held,
            })
        # Coin benchmark — bot başlangıcından (trade_start) şimdiye fiyat değişimi
        # Kullanıcı isteği v15: restart anından itibaren her coinin performansı izlensin
        _coin_benchmarks = []
        for _sym, _df_sym in sym_ind.items():
            _df_range = _df_sym[_df_sym.index >= trade_start]
            if _df_range.empty:
                continue  # BUG FIX: trade_start sonrası veri yok → atla, warmup veriyle yanlış benchmark oluşmasın
            _start_px = float(_df_range.iloc[0]["close"])
            _end_px   = float(_df_range.iloc[-1]["close"])
            _pct_chg  = (_end_px - _start_px) / _start_px * 100 if _start_px > 0 else 0.0
            _coin_benchmarks.append({
                "symbol": _sym,
                "start_price": round(_start_px, 6),
                "current_price": round(_end_px, 6),
                "pct_change": round(_pct_chg, 2),
                "start_date": trade_start.strftime("%Y-%m-%d %H:%M"),
            })
        _coin_benchmarks.sort(key=lambda x: x["pct_change"], reverse=True)

        _state = {
            "mode": _mode,
            "run_time": datetime.now(timezone.utc).isoformat(),
            "start_date": str(trade_start.date()),
            "end_date": str(trade_end.date()),
            "initial_capital": initial_capital,
            "final_balance": round(balance, 2),
            "total_pnl": round(balance - initial_capital, 2),
            "total_pnl_pct": round(_total_pnl_pct, 2),
            "max_drawdown_pct": round(_max_dd * 100, 2),
            "win_rate": round(_wr, 1),
            "total_trades": len(_finished),
            "open_positions": _open,
            "closed_trades": _closed,
            "coin_benchmarks": _coin_benchmarks,
        }
        import pathlib as _pathlib
        _pathlib.Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as _jf:
            _json.dump(_state, _jf, ensure_ascii=False, indent=2)
        print(f"\n  ✓ State kaydedildi: {json_out}")

    if m4_mode and _m4_state is not None and _m4_state.regime_switches:
        print(f"\n{'─'*60}")
        print(f"  M4 Rejim Kontrolleri ({len(_m4_state.regime_switches)} adet):")
        for sw_ts, sw_old, sw_new in _m4_state.regime_switches:
            marker = "→" if sw_old != sw_new else "="
            print(f"    {sw_ts.strftime('%Y-%m-%d')}  {sw_old:12} {marker}  {sw_new}")
        if _m4_state.wfo_updates:
            print(f"\n  Rolling WFO Güncellemeleri: {len(_m4_state.wfo_updates)}")
            for wts in _m4_state.wfo_updates:
                print(f"    {wts.strftime('%Y-%m-%d')}")


# ── Rapor ─────────────────────────────────────────────────────────────────────

def _print_report(
    trades: list[PPos],
    equity_curve: list[tuple],
    initial_capital: float,
    final_balance: float,
    raw_data: dict[str, pd.DataFrame],
    days: int,
    regime_summary: Optional[dict] = None,
    trade_start: Optional[pd.Timestamp] = None,
    trade_end: Optional[pd.Timestamp] = None,
    active_syms: Optional[list[str]] = None,  # aktif coin listesi (None=SYMBOLS)
) -> None:
    total_pnl = final_balance - initial_capital
    total_pct = total_pnl / initial_capital * 100

    wins   = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    wr     = len(wins) / len(trades) * 100 if trades else 0
    gross_win  = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    max_dd = 0.0
    if equity_curve:
        peak = equity_curve[0][1]
        for _, v in equity_curve:
            peak = max(peak, v)
            max_dd = max(max_dd, (peak - v) / peak)

    # ── coin bazlı istatistikler + buy&hold ──────────────────────────────────
    by_coin: dict[str, list[PPos]] = {}
    for t in trades:
        by_coin.setdefault(t.symbol, []).append(t)

    coin_stats: list[dict] = []
    _bnh_start = trade_start if trade_start is not None else pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=days))
    _bnh_end   = trade_end   if trade_end   is not None else pd.Timestamp(datetime.now(timezone.utc))
    report_syms = active_syms if active_syms is not None else SYMBOLS
    for sym in report_syms:
        df_raw = raw_data.get(sym, pd.DataFrame())
        # Buy&Hold karşılaştırması: trade_start..trade_end penceresini kullan
        if not df_raw.empty:
            df_p = df_raw[(df_raw.index >= _bnh_start) & (df_raw.index <= _bnh_end)]
        else:
            df_p = df_raw
        if len(df_p) >= 2:
            bnh_start = float(df_p.iloc[0]["close"])
            bnh_end   = float(df_p.iloc[-1]["close"])
            bnh_pct   = (bnh_end / bnh_start - 1) * 100
        else:
            bnh_start = bnh_end = bnh_pct = 0.0

        ts_list  = by_coin.get(sym, [])
        bot_pnl  = sum(t.pnl for t in ts_list)
        n_trades = len(ts_list)
        n_wins   = sum(1 for t in ts_list if t.pnl > 0)
        coin_wr  = n_wins / n_trades * 100 if n_trades else 0.0

        coin_stats.append({
            "sym": sym,
            "bnh_start": bnh_start, "bnh_end": bnh_end, "bnh_pct": bnh_pct,
            "bot_pnl": bot_pnl, "n_trades": n_trades, "n_wins": n_wins, "wr": coin_wr,
        })

    # ════════════════════════════════════════════════════════════════════════
    # 1. PORTFOLIO ÖZET
    # ════════════════════════════════════════════════════════════════════════
    pnl_icon = "▲" if total_pnl >= 0 else "▼"
    print(f"\n{'═'*60}")
    print(f"  SONUÇLAR  —  Son {days} Gün  |  Başlangıç: ${initial_capital:,.0f}")
    print(f"{'═'*60}")
    print(f"  Bitiş Sermaye   : ${final_balance:>9,.2f}   {pnl_icon} {total_pct:+.2f}%  (${total_pnl:+,.2f})")
    print(f"  Toplam İşlem    : {len(trades)}  |  Kazanılan: {len(wins)}  Kaybedilen: {len(losses)}")
    print(f"  Kazanma Oranı   : %{wr:.1f}  |  Profit Factor: {pf:.2f}")
    print(f"  Max Düşüş       : %{max_dd*100:.2f}")

    if regime_summary:
        order = ["STRONG_BEAR","BEAR","NEUTRAL","BULL","STRONG_BULL"]
        icons = {"STRONG_BEAR":"🔴","BEAR":"🟠","NEUTRAL":"🟡","BULL":"🟢","STRONG_BULL":"💚"}
        parts = [f"{icons[r]} {r.replace('_',' ')}: {regime_summary[r]}"
                 for r in order if r in regime_summary]
        print(f"\n  Piyasa Rejimi   : {' | '.join(parts)}")

    # ════════════════════════════════════════════════════════════════════════
    # 2. COİN BAZLI — bu sürede coin ne oldu, bot ne yaptı
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  COİN BAZLI SONUÇLAR")
    print(f"{'─'*60}")

    bnh_list: list[float] = []
    bot_list: list[float] = []

    for s in coin_stats:
        sym        = s["sym"]
        bnh_pct    = s["bnh_pct"]
        bot_pnl    = s["bot_pnl"]
        n_trades   = s["n_trades"]
        n_wins     = s["n_wins"]
        coin_wr    = s["wr"]
        bnh_start  = s["bnh_start"]
        bnh_end    = s["bnh_end"]
        bot_pct_of_cap = bot_pnl / initial_capital * 100

        # Coinin durumu
        bnh_icon = "▲" if bnh_pct >= 0 else "▼"
        # Bot durumu
        bot_icon = "▲" if bot_pnl > 0 else ("▼" if bot_pnl < 0 else "·")

        print(f"\n  {sym}")
        if bnh_start > 0:
            print(f"    Coin bu sürede  : {bnh_icon} {bnh_pct:+.1f}%  "
                  f"(${bnh_start:,.4f} → ${bnh_end:,.4f})")
        if n_trades == 0:
            print(f"    Bot             : İşlem yapmadı  (rejim uygun değildi)")
        else:
            print(f"    Bot             : {bot_icon} {bot_pct_of_cap:+.2f}%  (${bot_pnl:+,.2f})  "
                  f"—  {n_trades} işlem, %{coin_wr:.0f} kazanma")

        if n_trades > 0:
            bnh_list.append(bnh_pct)
            bot_list.append(bot_pct_of_cap)

    # Özet satırı
    if bnh_list:
        avg_bnh = np.mean(bnh_list)
        avg_bot = np.mean(bot_list)
        print(f"\n{'─'*60}")
        print(f"  İşlem yapılan {len(bnh_list)} coinde ortalama:")
        print(f"    Coinler bu sürede  : {avg_bnh:+.1f}%")
        print(f"    Bot kazancı        : {avg_bot:+.2f}%  (sermayeye göre)")
        diff = avg_bot - avg_bnh
        diff_icon = "▲ Bot daha iyi" if diff > 0 else "▼ Coin daha iyi"
        print(f"    Fark               : {diff:+.1f} puan  →  {diff_icon}")

    # ════════════════════════════════════════════════════════════════════════
    # 3. İŞLEM LİSTESİ
    # ════════════════════════════════════════════════════════════════════════
    if not trades:
        print(f"\n  İşlem yapılmadı.\n")
        return

    print(f"\n{'─'*60}")
    print(f"  TÜM İŞLEMLER")
    print(f"{'─'*60}")
    print(f"  {'#':>3}  {'Coin':<11} {'Tarih':<12} {'Süre':>6}  "
          f"{'Giriş':>10} {'Çıkış':>10}  {'K/Z':>8}  Sebep")
    print(f"  {'─'*57}")

    sorted_trades = sorted(trades, key=lambda t: t.entry_time)
    for i, t in enumerate(sorted_trades, 1):
        e_str = t.entry_time.strftime("%m/%d %H:%M") if t.entry_time else "—"
        dur   = ""
        if t.entry_time and t.exit_time:
            mins = int((t.exit_time - t.entry_time).total_seconds() / 60)
            dur  = f"{mins//60}sa" if mins >= 60 else f"{mins}dk"
        icon  = "+" if t.pnl > 0 else "-"
        reason_short = {"stop_loss":"SL","trailing_stop":"TS","strategy_exit":"SE","backtest_end":"BE"}.get(
            t.exit_reason, t.exit_reason[:2])
        print(f"  {i:>3}  {t.symbol:<11} {e_str:<12} {dur:>6}  "
              f"{t.entry_price:>10.4f} {t.exit_price:>10.4f}  "
              f"{icon}${abs(t.pnl):>6.2f}  {reason_short}")

    print(f"\n  Kısaltmalar: SL=Stop Loss  TS=Trailing Stop  SE=Strateji Çıkışı  BE=Test Sonu")
    print(f"{'═'*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Kripto Portfolio Backtest")
    parser.add_argument("--days",    type=int,   default=365,    help="Kaç günlük veri çekilecek (varsayılan: 365)")
    parser.add_argument("--capital", type=float, default=10_000, help="Başlangıç sermaye $ (varsayılan: 10000)")
    # --start / --start_date her ikisi de çalışır
    parser.add_argument("--start", "--start_date", dest="start_date", type=str, default=None,
                        help="Backtest başlangıç tarihi (YYYY-MM-DD)")
    # --end / --end_date her ikisi de çalışır
    parser.add_argument("--end",   "--end_date",   dest="end_date",   type=str, default=None,
                        help="Backtest bitiş tarihi (YYYY-MM-DD)")
    parser.add_argument("--label", type=str,   default=None,   help="Dönem etiketi (örn: 'Boğa 2025')")
    parser.add_argument("--universe", action="store_true",
                        help="25+ coinlik evrenden dinamik coin seçimi yap")
    parser.add_argument("--wfo",      action="store_true",
                        help="Walk-forward parametre optimizasyonu çalıştır (yavaş ~2-5 dk)")
    parser.add_argument("--coins",    type=int, default=COIN_SELECT_N,
                        help=f"--universe modunda aktif coin sayısı (varsayılan: {COIN_SELECT_N})")
    parser.add_argument("--auto",     action="store_true",
                        help="Otomatik mod: BTC rejimine göre M1 (boğa) veya M3_v4 hibrit (ayı/nötr) seç")
    parser.add_argument("--m4", action="store_true",
                        help="M4 mod: intra-simulation rejim checkpoint (30g) + rolling WFO (60g) + dinamik pozisyon büyüklüğü")
    parser.add_argument("--m5", action="store_true",
                        help="M5 mod: M4 + ATR-percentile sizing + portfolio circuit breaker + ER gate + momentum decay exit")
    parser.add_argument("--m6", action="store_true",
                        help="M6 mod: M5 + agresif pyramiding + erken trailing zoom + büyük pozisyon (upside capture)")
    args = parser.parse_args()

    m6_mode   = args.m6
    m5_mode   = args.m5 or m6_mode   # M6, M5'i içerir
    m4_mode   = args.m4 or m5_mode   # M5, M4'ü içerir
    if m4_mode:
        auto_mode = True
        use_wfo   = True

    # --start verilmişse, o tarihe kadar geriye gidecek kadar veri çek
    # (warmup dahil, ama gereksiz eski veriyi çekme)
    days = args.days
    if args.start_date:
        try:
            start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
            # start_date'den bugüne + warmup (EMA200 için ~9 gün) + 10 gün buffer
            # WFO veya auto modunda ek olarak WFO_LOOKBACK günü de geri git
            wfo_extra = WFO_LOOKBACK if (args.wfo or args.auto or args.m4) else 0
            days_needed = (datetime.now() - start_dt).days + WARMUP_BARS // 24 + wfo_extra + 10
            # --start açıkça belirtilmişse days_needed'i kullan;
            # sadece kullanıcı --days'i de açıkça verdiyse max al
            if args.days != 365:  # kullanıcı --days'i değiştirmiş
                days = max(args.days, days_needed)
            else:
                days = days_needed  # default 365'i override et
        except ValueError:
            pass

    run_portfolio_backtest(
        days=days,
        initial_capital=args.capital,
        start_date=args.start_date,
        end_date=args.end_date,
        label=args.label or (
            f"{args.start_date} → {args.end_date}" if args.start_date else None
        ),
        use_universe=args.universe,
        use_wfo=args.wfo or m4_mode,
        n_coins=args.coins,
        auto_mode=args.auto or m4_mode,
        m4_mode=m4_mode,
        m5_mode=m5_mode,
        m6_mode=m6_mode,
    )


if __name__ == "__main__":
    main()
