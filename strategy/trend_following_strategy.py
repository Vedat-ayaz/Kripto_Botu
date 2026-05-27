"""
Gelişmiş Trend-Following Stratejisi
====================================
Akademik referanslar:
  - Moskowitz, Ooi & Pedersen (2012): Time-series momentum (TSMOM)
  - Jegadeesh & Titman (1993): Cross-sectional momentum
  - Grobys & Sapkota (2019): RSI-based crypto strategies
  - Liu & Tsyvinski (2021): Crypto momentum

Bileşenler:
  1. Piyasa Rejimi Tespiti: ADX + Hurst üstel katsayısı
  2. Trend Filtresi       : EMA50/200 hizalama
  3. Momentum Teyidi      : MACD histogram + TSMOM skoru
  4. Aşırı Alım/Satım     : RSI + Stochastic RSI + Williams %R
  5. Hacim Teyidi         : OBV trendi + volume/SMA
  6. Bollinger Squeeze    : Düşük volatilite → kırılım potansiyeli
  7. Bileşik Skor         : Ağırlıklı sinyal füzyonu (0.0 – 1.0)
"""

import logging
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from .signal import Signal, Side
from indicators.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class TrendFollowingStrategy:
    """
    Çok-sinyal birleşik trend-following stratejisi.

    Piyasa rejimine göre otomatik parametre ayarı:
      - Trending  (ADX>25 ve Hurst>0.55): Trend-following ağırlıkları
      - Ranging   (ADX<20 ve Hurst<0.45): Bollinger mean-reversion ağırlıkları
      - Geçiş     (arada): Azaltılmış pozisyon boyutu
    """

    # Bileşik skor eşiği — class default'ları (instance seviyesinde override edilebilir)
    # Kalite filtresi: gevşetildiğinde XRP -5.6%, BNB -10.4% oldu.
    # Per-coin profil ile ETH/DOT için 0.60-0.68'e çıkarıldı.
    ENTRY_SCORE_TREND    = 0.55   # Trending rejimde gerekli minimum skor (baseline)
    ENTRY_SCORE_RANGING  = 0.60   # Ranging rejimde daha seçici ol (baseline)

    def __init__(
        self,
        rsi_lower: float = 45.0,
        rsi_upper: float = 75.0,
        adx_threshold: float = 18.0,
        min_atr_ratio: float = 0.002,
        volume_sma_multiplier: float = 0.4,
        entry_score_trend: float = 0.55,    # Per-coin override: ETH→0.60, DOT→0.62
        entry_score_ranging: float = 0.60,  # Per-coin override: ETH→0.65, DOT→0.68
        # Anti-whipsaw filter (Choppiness Index — Dreiss 1991)
        choppiness_threshold: float = 61.8,  # CI > 61.8 → choppy → entry blok
        choppiness_enabled: bool = True,
        # Multi-timeframe trend filter (Elder Triple Screen)
        mtf_filter_enabled: bool = True,
        # Timeframe-bağımlı lookback parametreleri
        # 1h barlarda: slope_bars=20 (20 saat), momentum_lookback=720 (30 gün)
        # Daily barlarda: slope_bars=5 (1 hafta), momentum_lookback=30 (1 ay)
        slope_bars: int = 20,
        momentum_lookback: int = 720,
        # ADX dinamik eşik ayarı (ADX<20 → +boost, ADX>30 → -0.03)
        # 1h crypto: 0.06 iyi çalışır; daily BIST: 0.0 (ADX 15-25 normaldir)
        adx_boost: float = 0.06,
        # Rejim sınırları (regime_score bazlı)
        # 1h: trending>0.6, ranging<0.35; daily: 0.40/0.20 daha uygun
        regime_trending_threshold: float = 0.60,
        regime_ranging_threshold: float = 0.35,
        # SHORT tetik eşiği: close < EMA200 * short_ema_pct
        # 1h/15m: 0.985 (EMA200'ün %1.5 altı)
        # 1m scalping: 0.9995 (hemen altı yeterli)
        short_ema_pct: float = 0.985,
        # SHORT momentum lookback: kaç bar öncesiyle kıyasla
        # 1h: 336 bar = 14 gün | 1m: 1440 bar = 1 gün (kısa TF'te daha reaktif)
        short_momentum_lookback: int = 336,
        # SHORT momentum: fiyat N bar önceye göre en az ne kadar düşmüş olmalı
        # 1h: 0.97 (%3 düşüş zorunlu) | 1m: 1.001 (sadece fiyat yükseliyorsa engelle)
        short_mom_pct: float = 0.97,
        # SHORT score eşiği (priority block için)
        # 1h: 0.38/0.34 | 1m: 0.28/0.26 (daha gevşek)
        short_score_trend_thr: float = 0.38,
        short_score_range_thr: float = 0.34,
        # SHORT için EMA slope gereksin mi?
        # 1h: True (EMA50 düşüyor olmalı) | 1m: False (ranging'de slope flat olur)
        short_require_ema_slope: bool = True,
        indicators: Optional[TechnicalIndicators] = None,
    ):
        self.rsi_lower             = rsi_lower
        self.rsi_upper             = rsi_upper
        self.adx_threshold         = adx_threshold
        self.min_atr_ratio         = min_atr_ratio
        self.volume_sma_multiplier = volume_sma_multiplier
        # Instance-level override (per-coin profiling)
        self.ENTRY_SCORE_TREND   = entry_score_trend
        self.ENTRY_SCORE_RANGING = entry_score_ranging
        # Rejim adaptasyonu için başlangıç değerlerini sakla
        self._base_entry_trend   = entry_score_trend
        self._base_entry_ranging = entry_score_ranging
        # Filtre konfigürasyonu
        self.choppiness_threshold         = choppiness_threshold
        self._slope_bars                  = slope_bars
        self._momentum_lookback           = momentum_lookback
        self._adx_boost                   = adx_boost
        self._regime_trending_threshold   = regime_trending_threshold
        self._regime_ranging_threshold    = regime_ranging_threshold
        self.choppiness_enabled        = choppiness_enabled
        self.mtf_filter_enabled        = mtf_filter_enabled
        self.short_ema_pct             = short_ema_pct
        self.short_momentum_lookback   = short_momentum_lookback
        self.short_mom_pct             = short_mom_pct
        self.short_score_trend_thr     = short_score_trend_thr
        self.short_score_range_thr     = short_score_range_thr
        self.short_require_ema_slope   = short_require_ema_slope
        self.indicators            = indicators or TechnicalIndicators()
        self._last_logged_bar: dict[str, str] = {}

        # ── Rolling Performans Takibi (Stage 3 — Adaptif Coin Pause) ───
        # Son N işlemin win rate'i belirli eşiğin altına düşerse coin
        # otomatik olarak PAUSE_BARS bar boyunca durdurulur.
        # ÖNCEKİ BUG: datetime.now() kullanılıyordu → backtest'te yanlış çalışıyordu
        # (geçmiş veride bar timestamp'i, datetime.now() ise gerçek zaman).
        # YENİ: bar-sayaç tabanlı pause → backtest ve live tutarlı.
        self._ROLLING_WINDOW   = 15       # son 15 işlem (orta pencere)
        # M4v14: 0.28 → 0.20 — 3 yıllık testlerde WR=27% vs 28% sürekli tetikliyordu
        # Eski eşik: 15 işlemde 4 kazanç (27%) → pause. Yeni: 3 kazanç (20%) → pause.
        # Kısa testlerde (7 ay) fark yok; uzun testlerde (3 yıl) gereksiz blok kaldırılır.
        self._PAUSE_THRESHOLD  = 0.20     # WR çok düşükse tetikle (her küçük dalgalanmaya değil)
        self._RESUME_THRESHOLD = 0.45     # mola sonrası dönüşte daha yüksek bar (raporlama)
        # M4v11: Pause süresi 3 hafta → 10 gün, eşik -100 → -200 USDT
        # Önceki 3 haftalık blok boğa döneminde kazanan coinleri durduruyordu.
        # -200 USDT = $10.000 sermayenin %-2'si → daha gerçekçi "kötü coin" tanımı.
        self._PAUSE_BARS       = 10 * 24  # 240 bar = 10 gün
        # Net PnL eşiği: son 15 işlemde NET KAYIP > 200 USDT ise pause
        self._NET_PNL_THRESHOLD = -200.0

        self._symbol_outcomes: dict[str, deque] = {}
        # Net PnL tabanlı pause (Stage 3.5): WR yüksek olsa bile küçük kazanç+büyük kayıp
        # toplam negatife götürürse coin'i pause et. Yalnız WR'ye bakmak yanıltıcı.
        self._symbol_pnl_history: dict[str, deque] = {}
        # bar-sayaç tabanlı pause (backtest-uyumlu)
        self._symbol_pause_remaining: dict[str, int] = {}
        # ESKİ datetime-based — geriye uyumluluk için tutulur (live/paper'da kullanılabilir)
        self._symbol_paused_until: dict[str, Optional[datetime]] = {}

        # ── BTC Piyasa Rejim Filtresi ──────────────────────────────────
        # BTC 200-günlük EMA üstündeyse bull, altındaysa bear.
        # Bear rejimde yalnızca BTC/USDT ve ETH/USDT işlem görür.
        self._btc_is_bull: bool = True   # varsayılan: bull (güvenli taraf)
        self._BTC_SAFE_SYMS = {"BTC/USDT", "ETH/USDT"}  # bear'da işlem yapılacaklar

    # ──────────────────────────────────────────────────────────────────
    # Ana sinyal üretimi
    # ──────────────────────────────────────────────────────────────────

    def generate_signal(self, symbol: str, df: pd.DataFrame,
                        allow_short: bool = False) -> Signal:
        """Son kapanmış mumdan bileşik sinyal üretir. allow_short=True ise SHORT sinyali de değerlendirilir."""
        # ── BTC Rejim Ön Filtresi ─────────────────────────────────────
        # Bear rejimde LONG engellenir — ama SHORT için bu filtre atlanır
        if not self._btc_is_bull and symbol not in self._BTC_SAFE_SYMS and not allow_short:
            return self._hold(symbol, df, "BTC bear rejimi — altcoin girişi engellendi")

        if "atr" not in df.columns:
            df = self.indicators.calculate(df)

        min_rows = max(self.indicators.ema_slow, 50) + 10
        if len(df) < min_rows:
            return self._hold(symbol, df, f"Yetersiz veri ({len(df)} < {min_rows})")

        row = df.iloc[-1]
        needed = [
            self.indicators.ema_fast_col, self.indicators.ema_slow_col,
            "rsi", "atr", "adx", "volume_sma", "atr_ratio",
            "macd", "macd_hist", "bb_pct_b", "bb_width",
            "stoch_k", "obv", "obv_ema", "regime_score",
        ]
        if any(pd.isna(row.get(c)) for c in needed):
            return self._hold(symbol, df, "Göstergeler hazır değil (NaN)")

        # ── Temel değerler ──────────────────────────────────────
        close      = row["close"]
        ema_fast   = row[self.indicators.ema_fast_col]
        ema_slow   = row[self.indicators.ema_slow_col]
        rsi        = row["rsi"]
        adx        = row["adx"]
        atr        = row["atr"]
        atr_ratio  = row["atr_ratio"]
        volume     = row["volume"]
        volume_sma = row["volume_sma"]
        macd_hist  = row["macd_hist"]
        macd       = row["macd"]
        bb_pct_b   = row["bb_pct_b"]
        bb_width   = row["bb_width"]
        stoch_k    = row["stoch_k"]
        obv        = row["obv"]
        obv_ema    = row["obv_ema"]
        regime     = row["regime_score"]  # 0=ranging, 1=trending
        tsmom      = row.get("tsmom", 0.0) or 0.0

        # ── Piyasa Rejimi ────────────────────────────────────────
        is_trending = regime > self._regime_trending_threshold
        is_ranging  = regime < self._regime_ranging_threshold
        regime_label = "TREND" if is_trending else ("RANGE" if is_ranging else "TRANS")

        # ── SHORT Sinyal Öncelikli Değerlendirmesi ────────────────
        # LONG-spesifik filtreler (rolling WR, momentum, EMA slope) SHORT'u engellemesin.
        # Bear piyasada bu filtreler ZATEN tetiklenir ve SHORT fırsatı kaçırılır.
        # Bu blok LONG filtrelerinden ÖNCE çalışır → bağımsız SHORT değerlendirmesi.
        if allow_short and close < ema_slow * self.short_ema_pct:   # SHORT tetik eşiği (1m: 0.9995, 1h: 0.985)
            # RSI aşırı satım kontrolü: fiyat zaten dipteyse SHORT kapama yakın (sekme riski)
            # EMA50 eğimi aşağı mı?
            _ema_slope_down = False
            if len(df) >= self._slope_bars + 1:
                _ema_fc = self.indicators.ema_fast_col
                _ema_prev = df[_ema_fc].iloc[-(self._slope_bars + 1)] if _ema_fc in df.columns else float('nan')
                _ema_slope_down = (not pd.isna(_ema_prev)) and ema_fast < float(_ema_prev)

            # Momentum kontrolü: coin gerçekten düşüyor mu?
            # 1h: 336 bar = 14 gün | 1m: 1440 bar = 1 gün
            _short_mom_ok = True
            _short_lb = self.short_momentum_lookback
            if len(df) >= _short_lb:
                _p14 = df["close"].iloc[-_short_lb]
                if not pd.isna(_p14) and close >= float(_p14) * self.short_mom_pct:
                    _short_mom_ok = False  # momentum flat/pozitif → düşüş trendi yok → SHORT engel

            # rsi >= 32: aşırı satımda değil (sekme riski düşük)
            # short_require_ema_slope=False ise 1m scalping'de slope şartı aranmaz
            _slope_ok = (not self.short_require_ema_slope) or _ema_slope_down
            if rsi >= 32 and _slope_ok and _short_mom_ok:
                _short_score, _short_detail = self._composite_score_short(
                    close, ema_fast, ema_slow, rsi, adx, macd, macd_hist,
                    bb_pct_b, bb_width, stoch_k, volume, volume_sma,
                    obv, obv_ema, tsmom, is_trending,
                )
                # BTC boğa piyasasında SHORT çok nadir tetiklenir (0.62 eşik = neredeyse imkânsız)
                # Boğa döneminde coin kısa düşüşlerde SHORT yapma → false positive engeli
                if self._btc_is_bull:
                    _short_thresh = 0.62   # boğa rejimde yüksek bar
                else:
                    _short_thresh = self.short_score_trend_thr if is_trending else self.short_score_range_thr
                    if adx >= 30:
                        _short_thresh = max(_short_thresh - 0.04, 0.20)
                    _short_thresh = max(_short_thresh - 0.02, 0.20)  # bear global → daha kolay SHORT

                if _short_score >= _short_thresh:
                    logger.info(
                        f"[Strateji] {symbol} [{regime_label}] 🔻 SHORT (öncelikli) "
                        f"skor={_short_score:.3f} eşik={_short_thresh:.2f}"
                    )
                    return Signal(
                        symbol=symbol,
                        side=Side.SHORT,
                        reason=f"[{regime_label}] SHORT skor={_short_score:.3f} eşik={_short_thresh:.2f} | {_short_detail}",
                        timestamp=datetime.now(timezone.utc),
                        price=close,
                        confidence_score=round(_short_score, 3),
                        atr=atr,
                        adx=adx,
                        rsi=rsi,
                    )

        # ── Rolling performans kontrolü ──────────────────────────
        if not self.is_tradeable(symbol):
            paused_until = self._symbol_paused_until.get(symbol)
            until_str = paused_until.strftime("%Y-%m-%d") if paused_until else "?"
            return self._hold(symbol, df, f"Rolling WR düşük → {until_str}'e kadar durduruldu")

        # ── Hard Filtreler (rejimden bağımsız) ───────────────────
        hard_failed = []
        if atr_ratio < self.min_atr_ratio:
            hard_failed.append(f"atr_ratio<{self.min_atr_ratio}")
        if close <= 0:
            hard_failed.append("close<=0")

        if hard_failed:
            return self._hold(symbol, df, "Hard filtre: " + ", ".join(hard_failed))

        # ── Coin-specific Trend Filtresi (EMA200) ────────────────
        # Coin kendi EMA200'ünün ALTINDAYSA LONG girişi engellenir.
        # Bu filtre, ranging modda BB/RSI sinyaliyle downtrend'deki
        # coinde "dip alımı" yapılmasını önler — 2024 H1'deki MATIC,
        # ADA, UNI kayıplarının temel sebebi buydu.
        if close < ema_slow:
            return self._hold(
                symbol, df,
                f"Coin downtrend: close={close:.4f} < EMA200={ema_slow:.4f} — LONG engellendi"
            )

        # ── Anti-Whipsaw Filtresi (Choppiness Index — Dreiss) ─────────
        # CI > 61.8 → piyasa choppy/sideways → trend-following ölüm bölgesi.
        # FET/NEAR gibi coinlerin küçük kazanç + büyük kayıp pattern'i tam buradan.
        # AQR (Hurst, Ooi, Pedersen 2017): trend P&L %80'i %20 zamanda yapılır.
        if self.choppiness_enabled:
            ci = row.get("choppiness", 50.0)
            if pd.notna(ci) and ci > self.choppiness_threshold:
                return self._hold(
                    symbol, df,
                    f"Choppy market (CI={ci:.1f} > {self.choppiness_threshold}) — no-trade rejimi"
                )

        # ── Multi-Timeframe Trend Filtresi (Elder Triple Screen) ──────
        # 4h timeframe'de trend AŞAĞIYA dönmüşse 1h LONG sinyallerini reddet.
        # htf_trend_up sütunu indicators.add_higher_timeframe() tarafından eklenir.
        if self.mtf_filter_enabled and "htf_trend_up" in df.columns:
            htf_up = row.get("htf_trend_up", True)
            if not bool(htf_up):
                htf_ema = row.get("htf_ema_fast", 0.0)
                htf_slope = row.get("htf_ema_slope", 0.0)
                return self._hold(
                    symbol, df,
                    f"4h trend AŞAĞI: htf_ema={htf_ema:.4f}, slope={htf_slope:+.4f} — LONG engellendi"
                )

        # ── EMA50 Eğim Filtresi ───────────────────────────────────
        # slope_bars bar önceki EMA50 ile karşılaştır.
        # 1h: 20 bar = 20 saat; daily: 5 bar = 1 hafta (config ile ayarlanır).
        if len(df) >= self._slope_bars + 1:
            ema_fast_col = self.indicators.ema_fast_col
            ema_nb_ago = df[ema_fast_col].iloc[-(self._slope_bars + 1)]
            if not pd.isna(ema_nb_ago) and ema_fast <= ema_nb_ago:
                return self._hold(
                    symbol, df,
                    f"EMA50 eğimi düz/negatif: {ema_fast:.4f} ≤ {ema_nb_ago:.4f} ({self._slope_bars} bar önce)"
                )

        # ── Momentum Filtresi ─────────────────────────────────────
        # momentum_lookback bar öncesine göre negatif momentum varsa LONG engellenir.
        # 1h: 720 bar = 30 gün; daily: 30 bar = 30 işlem günü (config ile ayarlanır).
        LOOKBACK_30D = self._momentum_lookback
        if len(df) >= LOOKBACK_30D:
            price_30d_ago = df["close"].iloc[-LOOKBACK_30D]
            if not pd.isna(price_30d_ago) and close < price_30d_ago * 0.97:
                return self._hold(
                    symbol, df,
                    f"30g momentum negatif: {close:.4f} < {price_30d_ago:.4f}×0.97 "
                    f"= {price_30d_ago * 0.97:.4f}"
                )

        # ── Bileşik Skor Hesaplama ───────────────────────────────
        score, detail = self._composite_score(
            close, ema_fast, ema_slow, rsi, adx, macd, macd_hist,
            bb_pct_b, bb_width, stoch_k, volume, volume_sma,
            obv, obv_ema, tsmom, is_trending,
        )

        entry_threshold = self.ENTRY_SCORE_TREND if is_trending else (
            self.ENTRY_SCORE_RANGING if is_ranging else 0.58
        )

        # ── ADX'e göre dinamik eşik ayarı ───────────────────────
        # adx_boost=0.0 ise devre dışı (daily BIST'te ADX 15-25 normal)
        if adx >= 30:
            entry_threshold = max(entry_threshold - 0.03, 0.40)
        elif adx < self.adx_threshold + 2:
            entry_threshold = min(entry_threshold + self._adx_boost, 0.72)

        # ── Bear rejimde BTC/ETH için ek güçlük ─────────────────
        # Altcoinler zaten engellendi; BTC/ETH için daha yüksek bar
        if not self._btc_is_bull:
            entry_threshold = min(entry_threshold + 0.05, 0.75)

        # ── Log (yeni mum geldiğinde) ────────────────────────────
        current_bar_ts = str(df.index[-1])
        if self._last_logged_bar.get(symbol) != current_bar_ts:
            self._last_logged_bar[symbol] = current_bar_ts
            logger.info(
                f"[Strateji] {symbol} [{regime_label}|R={regime:.2f}] | "
                f"close={close:.4f} EMA50={ema_fast:.2f} EMA200={ema_slow:.2f} | "
                f"RSI={rsi:.1f} ADX={adx:.1f} MACD_H={macd_hist:.4f} | "
                f"BB%B={bb_pct_b:.2f} StochK={stoch_k:.1f} | "
                f"Vol/SMA={volume/volume_sma:.2f}x | "
                f"Skor={score:.3f}/eşik={entry_threshold:.2f} | "
                f"{'✅ BUY' if score >= entry_threshold else '❌ HOLD'}"
            )

        if score >= entry_threshold:
            return Signal(
                symbol=symbol,
                side=Side.BUY,
                reason=f"[{regime_label}] Bileşik skor={score:.3f} | {detail}",
                timestamp=datetime.now(timezone.utc),
                price=close,
                confidence_score=round(score, 3),
                atr=atr,
                adx=adx,
                rsi=rsi,
            )

        # ── SHORT Sinyal Değerlendirmesi ──────────────────────────────
        if allow_short:
            short_score, short_detail = self._composite_score_short(
                close, ema_fast, ema_slow, rsi, adx, macd, macd_hist,
                bb_pct_b, bb_width, stoch_k, volume, volume_sma,
                obv, obv_ema, tsmom, is_trending,
            )
            short_threshold = 0.40 if is_trending else 0.35
            if adx >= 30:
                short_threshold = max(short_threshold - 0.03, 0.28)
            # Bear global rejimde SHORT daha kolay tetiklenir
            if not self._btc_is_bull:
                short_threshold = min(short_threshold + 0.02, 0.48)

            coin_in_downtrend = close < ema_slow
            ema_slope_down = False
            if len(df) >= self._slope_bars + 1:
                _ema_col = self.indicators.ema_fast_col
                _ema_nb_ago = df[_ema_col].iloc[-(self._slope_bars + 1)] if _ema_col in df.columns else float('nan')
                ema_slope_down = (not pd.isna(_ema_nb_ago)) and ema_fast < float(_ema_nb_ago)

            if coin_in_downtrend and ema_slope_down and short_score >= short_threshold:
                logger.info(
                    f"[Strateji] {symbol} [{regime_label}] 🔻 SHORT skor={short_score:.3f} eşik={short_threshold:.2f}"
                )
                return Signal(
                    symbol=symbol,
                    side=Side.SHORT,
                    reason=f"[{regime_label}] SHORT skor={short_score:.3f} eşik={short_threshold:.2f} | {short_detail}",
                    timestamp=datetime.now(timezone.utc),
                    price=close,
                    confidence_score=round(short_score, 3),
                    atr=atr,
                    adx=adx,
                    rsi=rsi,
                )

        return self._hold(symbol, df, f"[{regime_label}] Skor={score:.3f} < {entry_threshold:.2f}")

    # ──────────────────────────────────────────────────────────────────
    # Bileşik Skor
    # ──────────────────────────────────────────────────────────────────

    def _composite_score(
        self,
        close: float, ema_fast: float, ema_slow: float,
        rsi: float, adx: float, macd: float, macd_hist: float,
        bb_pct_b: float, bb_width: float,
        stoch_k: float, volume: float, volume_sma: float,
        obv: float, obv_ema: float, tsmom: float,
        is_trending: bool,
    ) -> tuple[float, str]:
        """
        Ağırlıklı çok-sinyal skoru (0.0 – 1.0).
        Trending rejim: trend + momentum ağırlıklı
        Ranging rejim : bollinger + RSI mean-reversion ağırlıklı
        """
        components = {}

        # ── 1. Trend bileşeni — ADX kapılı EMA hizalama ───────────────
        # ADX trendın ne kadar güçlü olduğunu ölçer.
        # Düşük ADX: EMA'lar hizalı görünse bile gerçek trend yok → skoru bastır
        if close > ema_slow and ema_fast > ema_slow:
            gap_ratio  = (close - ema_slow) / ema_slow
            base_trend = min(0.5 + gap_ratio * 5, 1.0)
            if adx >= 25:
                adx_mult = 1.0      # güçlü trend — tam skor
            elif adx >= 18:
                adx_mult = 0.65     # zayıf trend — bastırılmış skor
            else:
                adx_mult = 0.30     # çok zayıf / ranging — neredeyse sıfır
            components["trend"] = base_trend * adx_mult
        elif close > ema_fast > ema_slow * 0.99:   # henüz başlayan trend
            components["trend"] = 0.25 if adx >= 18 else 0.08
        else:
            components["trend"] = 0.0

        # ── 2. MACD momentum ───────────────────────────────────────────
        if macd > 0 and macd_hist > 0:
            components["macd"] = min(0.5 + abs(macd_hist) * 50, 1.0)
        elif macd > 0:
            components["macd"] = 0.35
        else:
            components["macd"] = 0.0

        # ── 3. RSI (asimetrik — crypto bullish bias) ───────────────────
        if 52 <= rsi <= 70:
            components["rsi"] = 0.8 + (rsi - 52) / 18 * 0.2
        elif 45 <= rsi < 52:
            components["rsi"] = 0.5
        elif rsi > 70:  # aşırı alım → zayıf
            components["rsi"] = max(0.0, 1.0 - (rsi - 70) / 15)
        else:
            components["rsi"] = 0.0

        # ── 4. Bollinger %B (trend rejimde kırılım, range'de mean-rev) ─
        if is_trending:
            # Trendle birlikte üst banda yakın = güçlü trend
            if 0.5 <= bb_pct_b <= 1.0:
                components["bb"] = bb_pct_b
            elif bb_pct_b > 1.0:  # çok aşırı uzatılmış
                components["bb"] = 0.3
            else:
                components["bb"] = 0.0
        else:
            # Ranging: alt banda yakın = al fırsatı
            if bb_pct_b < 0.2:
                components["bb"] = 1.0 - bb_pct_b * 3
            elif 0.2 <= bb_pct_b <= 0.5:
                components["bb"] = 0.4
            else:
                components["bb"] = 0.0

        # ── 5. Stochastic RSI ─────────────────────────────────────────
        if 40 <= stoch_k <= 80:
            components["stoch"] = 0.7
        elif stoch_k > 80:  # aşırı alım
            components["stoch"] = 0.3
        elif stoch_k < 20:  # oversold → potansiyel dönüş
            components["stoch"] = 0.6
        else:
            components["stoch"] = 0.4

        # ── 6. Hacim teyidi ───────────────────────────────────────────
        vol_ratio = volume / volume_sma if volume_sma > 0 else 1.0
        if vol_ratio >= self.volume_sma_multiplier:
            components["volume"] = min(vol_ratio / 2.0, 1.0)
        else:
            components["volume"] = vol_ratio / self.volume_sma_multiplier * 0.5

        # ── 7. OBV trendi teyidi ──────────────────────────────────────
        if obv > obv_ema:
            components["obv"] = 0.8
        else:
            components["obv"] = 0.2

        # ── 8. TSMOM (time-series momentum) — güçlendirildi ─────────────
        # Pozitif TSMOM: fiyat tarihsel ortalamasının üzerinde ivmeleniyor
        if not pd.isna(tsmom):
            if tsmom > 1.0:
                components["tsmom"] = 1.0
            elif tsmom > 0.5:
                components["tsmom"] = 0.85
            elif tsmom > 0:
                components["tsmom"] = 0.55
            elif tsmom > -0.5:
                components["tsmom"] = 0.20
            else:
                components["tsmom"] = 0.0
        else:
            components["tsmom"] = 0.40  # nötr

        # ── Ağırlıklar (trending vs ranging) ──────────────────────────
        # Trending: trend + momentum ağırlıklı (trend-following odağı)
        # Ranging : RSI + BB mean-reversion ağırlıklı
        # NOT — Denenen ek bileşenler:
        #   Williams %R: OBV/stoch/TSMOM'dan alındığında net negatif sonuç.
        #   Chandelier Exit: ATR azalınca erken çıkış yaptı (XRP +30→+1%).
        #   Vol-targeting: yüksek-getirili altcoinleri kalıcı kısıtladı.
        # Sonuç: Orijinal ağırlıklar bu portföy için optimal.
        if is_trending:
            weights = {
                "trend": 0.30, "macd": 0.20, "rsi": 0.13,
                "bb": 0.08, "stoch": 0.07, "volume": 0.10,
                "obv": 0.05, "tsmom": 0.07,
            }
        else:
            weights = {
                "trend": 0.08, "macd": 0.10, "rsi": 0.22,
                "bb": 0.28, "stoch": 0.15, "volume": 0.08,
                "obv": 0.05, "tsmom": 0.04,
            }

        score = sum(components.get(k, 0) * w for k, w in weights.items())
        detail = " | ".join(f"{k}={v:.2f}" for k, v in components.items())
        return round(float(score), 4), detail

    def _composite_score_short(
        self,
        close: float, ema_fast: float, ema_slow: float,
        rsi: float, adx: float, macd: float, macd_hist: float,
        bb_pct_b: float, bb_width: float,
        stoch_k: float, volume: float, volume_sma: float,
        obv: float, obv_ema: float, tsmom: float,
        is_trending: bool,
    ) -> tuple[float, str]:
        """
        SHORT (açığa satış) için ayı bileşik skoru — 0.0–1.0 arası.
        Yüksek skor = daha güçlü SHORT sinyali.
        LONG _composite_score() metodunun tersine çalışır.
        """
        components: dict[str, float] = {}

        # 1. Trend bileşeni (ağırlık 0.30) — EMA altında ve aşağı eğimli
        if close < ema_slow and ema_fast < ema_slow:
            gap_ratio = (ema_slow - close) / ema_slow
            base_trend = min(0.5 + gap_ratio * 5, 1.0)
            adx_mult = 1.0 if adx >= 25 else (0.65 if adx >= 18 else 0.30)
            components["trend"] = base_trend * adx_mult
        elif close < ema_fast < ema_slow * 1.01:
            components["trend"] = 0.25 if adx >= 18 else 0.08
        else:
            components["trend"] = 0.0

        # 2. MACD (ağırlık 0.20) — negatif ve histogram negatif
        if macd < 0 and macd_hist < 0:
            components["macd"] = min(0.5 + abs(macd_hist) * 50, 1.0)
        elif macd < 0:
            components["macd"] = 0.35
        else:
            components["macd"] = 0.0

        # 3. RSI (ağırlık 0.13) — zayıflık bölgesi 30–48 (aşırı satım değil)
        if 30 <= rsi <= 48:
            components["rsi"] = 0.8 + (48 - rsi) / 18 * 0.2
        elif rsi < 30:      # Aşırı satım → SHORT kapama yakın, zayıf sinyal
            components["rsi"] = 0.3
        elif 48 < rsi <= 55:
            components["rsi"] = 0.4
        else:               # rsi > 55 → boğa bölgesi
            components["rsi"] = 0.0

        # 4. Bollinger %B (ağırlık 0.08) — alt banda yakın = düşüş devam
        if bb_pct_b < 0.3:
            components["bb"] = 1.0 - bb_pct_b * 2
        elif bb_pct_b < 0.5:
            components["bb"] = 0.4
        else:
            components["bb"] = 0.0

        # 5. Stochastic (ağırlık 0.07)
        if stoch_k < 30:
            components["stoch"] = 0.8
        elif stoch_k < 50:
            components["stoch"] = 0.5
        else:
            components["stoch"] = 0.1

        # 6. Hacim (ağırlık 0.10) — yüksek hacim yönü onaylar
        vol_ratio = volume / volume_sma if volume_sma > 0 else 1.0
        if vol_ratio >= self.volume_sma_multiplier:
            components["volume"] = min(vol_ratio / 2.0, 1.0)
        else:
            components["volume"] = vol_ratio / self.volume_sma_multiplier * 0.5

        # 7. OBV (ağırlık 0.05) — OBV < OBV_EMA → satış baskısı
        components["obv"] = 0.8 if obv < obv_ema else 0.2

        # 8. TSMOM (ağırlık 0.07) — negatif momentum
        if not pd.isna(tsmom):
            if tsmom < -1.0:
                components["tsmom"] = 1.0
            elif tsmom < -0.5:
                components["tsmom"] = 0.85
            elif tsmom < 0:
                components["tsmom"] = 0.55
            elif tsmom < 0.5:
                components["tsmom"] = 0.20
            else:
                components["tsmom"] = 0.0
        else:
            components["tsmom"] = 0.40

        weights = {
            "trend": 0.30, "macd": 0.20, "rsi": 0.13,
            "bb": 0.08, "stoch": 0.07, "volume": 0.10,
            "obv": 0.05, "tsmom": 0.07,
        }
        score = sum(components.get(k, 0.0) * w for k, w in weights.items())
        detail = " | ".join(f"{k}={v:.2f}" for k, v in components.items())
        return round(float(score), 4), detail

    # ──────────────────────────────────────────────────────────────────
    # Çıkış koşulları
    # ──────────────────────────────────────────────────────────────────

    def should_exit(self, symbol: str, df: pd.DataFrame, entry_price: float,
                    is_short: bool = False) -> tuple[bool, str]:
        """
        Strateji tabanlı çıkış koşulları.
        Stop-loss ve trailing stop PositionManager tarafından yönetilir.
        """
        if "atr" not in df.columns:
            df = self.indicators.calculate(df)

        row = df.iloc[-1]
        close     = row["close"]
        ema_fast  = row.get(self.indicators.ema_fast_col)
        rsi       = row.get("rsi", 50)
        macd_hist = row.get("macd_hist", 0)
        regime    = row.get("regime_score", 0.5)
        bb_pct_b  = row.get("bb_pct_b", 0.5)

        if pd.isna(ema_fast):
            return False, ""

        atr    = row.get("atr", 0.0) or 0.0
        # Coin kendi EMA200 üzerindeyse (uptrend) → geniş tampon, aşağıda → dar tampon
        ema_slow_val = row.get("ema_slow", float("nan"))
        _coin_in_uptrend = (not pd.isna(ema_slow_val)) and (close > float(ema_slow_val) * 1.02)
        if _coin_in_uptrend:
            # Boğa trendinde: geniş tampon → trendin nefes almasına izin ver, kazananı erken kesme
            long_buffer = 2.5 * atr if atr > 0 else ema_fast * 0.025
        else:
            # Ayı/nötr: dar tampon → bear dönemde LONG'ları hızlı kapat
            long_buffer = 1.5 * atr if atr > 0 else ema_fast * 0.015
        # SHORT için her zaman dar tampon
        short_buffer = 1.5 * atr if atr > 0 else ema_fast * 0.015

        # ── SHORT çıkış koşulları ─────────────────────────────────────────
        if is_short:
            # 1. EMA50 yukarı kırılımı → trend döndü, SHORT kapat
            if close > ema_fast + short_buffer:
                return True, f"SHORT kapama: EMA50 üstü kırıldı close={close:.4f} > EMA50+buffer={ema_fast+short_buffer:.4f}"
            # 2. RSI aşırı satım + MACD pozitife döndü → dip, SHORT kapat
            if rsi < 22 and macd_hist > 0:
                return True, f"SHORT kapama: RSI aşırı satım ({rsi:.1f}) + MACD pozitif"
            # 3. Ranging piyasada BB alt bandına ulaşıldı (SHORT hedef tutuldu)
            if regime < 0.4 and bb_pct_b < 0.05:
                return True, f"SHORT kapama: Ranging BB alt bandı (%B={bb_pct_b:.2f})"
            # 4. Piyasa güçlü trending'e döndü (SHORT tehlikeli)
            if regime > 0.75:
                return True, f"SHORT kapama: Trending piyasa (regime={regime:.2f})"
            return False, ""

        # ── LONG çıkış koşulları ──────────────────────────────────────────
        # EMA50 kırılımı — tampon boyutu trend durumuna göre adaptif
        if close < ema_fast - long_buffer:
            _buf_x = "2.5" if _coin_in_uptrend else "1.5"
            return True, f"EMA50 belirgin kırıldı: close={close:.4f} < EMA50-{_buf_x}ATR={ema_fast - long_buffer:.4f}"

        # RSI aşırı alım + MACD negatife döndü (trend zayıflıyor) — orijinal eşikler korunur
        if rsi > 78 and macd_hist < 0:
            return True, f"RSI aşırı alım ({rsi:.1f}) + MACD negatif"

        # Ranging piyasada BB üst bandına ulaşıldı
        if regime < 0.4 and bb_pct_b > 0.95:
            return True, f"Ranging piyasa BB üst bandı aşıldı (%B={bb_pct_b:.2f})"

        # Rejim tamamen ranging'e döndü (trend bitti)
        if regime < 0.25:
            return True, f"Piyasa rejimi ranging'e döndü (regime={regime:.2f})"

        return False, ""

    # ──────────────────────────────────────────────────────────────────
    # Rolling Performans Yönetimi
    # ──────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────
    # BTC Rejim Yönetimi
    # ──────────────────────────────────────────────────────────────────

    def set_btc_regime(self, is_bull: bool) -> None:
        """
        BTC 200-günlük EMA konumuna göre piyasa rejimini günceller.
        Backtester ve PaperTrader her barda çağırır.

        Bull (is_bull=True) : BTC close > EMA200 → tüm semboller aktif
        Bear (is_bull=False): BTC close < EMA200 → sadece BTC/ETH aktif,
                               pozisyon boyutu etkili biçimde küçülür
        """
        if is_bull != self._btc_is_bull:
            logger.info(
                f"[Rejim] BTC piyasa rejimi değişti: "
                f"{'🐻 BEAR' if not is_bull else '🐂 BULL'}"
            )
        self._btc_is_bull = is_bull

    @property
    def btc_is_bull(self) -> bool:
        return self._btc_is_bull

    def apply_regime_params(
        self,
        entry_score_boost: float,
        base_entry_trend: Optional[float] = None,
        base_entry_ranging: Optional[float] = None,
    ) -> None:
        """
        AdaptiveRegimeController tarafından her barda çağrılır.
        Giriş eşiklerini piyasa rejimine göre dinamik olarak ayarlar.

        Args:
            entry_score_boost : rejim katkısı (+= ayı'da daha sıkı, -= boğa'da gevşek)
            base_entry_trend  : temel trending eşiği (None → per-coin başlangıç değeri)
            base_entry_ranging: temel ranging eşiği  (None → per-coin başlangıç değeri)
        """
        base_t = base_entry_trend   if base_entry_trend   is not None else self._base_entry_trend
        base_r = base_entry_ranging if base_entry_ranging is not None else self._base_entry_ranging
        self.ENTRY_SCORE_TREND   = float(np.clip(base_t + entry_score_boost, 0.35, 0.88))
        self.ENTRY_SCORE_RANGING = float(np.clip(base_r + entry_score_boost, 0.40, 0.92))

    def record_outcome(self, symbol: str, won: bool, pnl: float = 0.0) -> None:
        """
        Trade kapandığında çağrılır.
        İki kriterden HERHANGİ BİRİ tetiklenirse coin pause edilir:

        1. Win Rate kriteri: son ROLLING_WINDOW işlemde WR < PAUSE_THRESHOLD
        2. Net PnL kriteri: son ROLLING_WINDOW işlemde kümülatif PnL < 0
           (WR yüksek olsa bile küçük kazanç + büyük kayıp net kayıp yapabilir)

        İkinci kriter daha sağlam — "kötü streak yerine gerçekten kötü coin" yakalar.
        """
        # WR takibi
        if symbol not in self._symbol_outcomes:
            self._symbol_outcomes[symbol] = deque(maxlen=self._ROLLING_WINDOW)
        self._symbol_outcomes[symbol].append(1 if won else 0)

        # Net PnL takibi (Stage 3.5)
        if symbol not in self._symbol_pnl_history:
            self._symbol_pnl_history[symbol] = deque(maxlen=self._ROLLING_WINDOW)
        self._symbol_pnl_history[symbol].append(pnl)

        outcomes = self._symbol_outcomes[symbol]
        pnls = self._symbol_pnl_history[symbol]

        if len(outcomes) >= self._ROLLING_WINDOW:
            wr = sum(outcomes) / len(outcomes)
            net_pnl = sum(pnls)

            # İki kriter, ikisinden biri trigger eder
            wr_bad = wr < self._PAUSE_THRESHOLD
            pnl_bad = net_pnl < self._NET_PNL_THRESHOLD

            if wr_bad or pnl_bad:
                self._symbol_pause_remaining[symbol] = self._PAUSE_BARS
                trigger = []
                if wr_bad:
                    trigger.append(f"WR={wr:.0%}<{self._PAUSE_THRESHOLD:.0%}")
                if pnl_bad:
                    trigger.append(f"net_PnL={net_pnl:+.2f}<0")
                logger.warning(
                    f"[TrendStrategy] {symbol}: Son {self._ROLLING_WINDOW} işlem KÖTÜ "
                    f"({' | '.join(trigger)}) → "
                    f"{self._PAUSE_BARS} bar (~{self._PAUSE_BARS//24} gün) durduruldu"
                )

    def tick_pause(self, symbol: Optional[str] = None) -> None:
        """
        Her bar'da çağrılır — pause sayacını azaltır.
        symbol verilirse sadece o sembolün sayacı azaltılır (backtest single-symbol),
        yoksa tüm semboller (paper-live multi-symbol).
        """
        if symbol is not None:
            if symbol in self._symbol_pause_remaining:
                self._symbol_pause_remaining[symbol] -= 1
                if self._symbol_pause_remaining[symbol] <= 0:
                    del self._symbol_pause_remaining[symbol]
                    logger.info(f"[TrendStrategy] {symbol}: Mola sona erdi, tekrar izleniyor")
        else:
            for sym in list(self._symbol_pause_remaining):
                self._symbol_pause_remaining[sym] -= 1
                if self._symbol_pause_remaining[sym] <= 0:
                    del self._symbol_pause_remaining[sym]
                    logger.info(f"[TrendStrategy] {sym}: Mola sona erdi, tekrar izleniyor")

    def is_tradeable(self, symbol: str) -> bool:
        """
        Sembol şu an işlem yapılabilir mi?
        Bar-sayaç tabanlı pause'a bakar (backtest-uyumlu).
        Geriye uyumluluk için datetime-based pause'u da kontrol eder.
        """
        # Yeni mekanizma: bar-sayaç
        if self._symbol_pause_remaining.get(symbol, 0) > 0:
            return False
        # Eski mekanizma (live/paper için): datetime-based
        paused_until = self._symbol_paused_until.get(symbol)
        if paused_until is not None:
            if datetime.now(timezone.utc) >= paused_until:
                self._symbol_paused_until[symbol] = None
                return True
            return False
        return True

    def should_allow_pyramid(
        self,
        symbol: str,
        row,
        min_regime_score: float = 0.50,
        max_vol_spike: float = 1.50,
        max_atr_ratio: float = 0.040,
        min_adx: float = 22.0,
    ) -> tuple[bool, str]:
        """
        Adaptif pyramid gate — anlık piyasa durumuna göre pyramid'in
        bu coin ve bu an için uygun olup olmadığını dinamik karar verir.

        Backtest verisinden öğrenilen örüntü:
        ─ Stabil trender (düşük ATR/price, yüksek ADX, regime_score yüksek)
          pyramid'den FAYDA görür (BTC, SOL, MATIC tarzı)
        ─ Volatil patlamacılar (yüksek vol-spike, geniş ATR ranges)
          pyramid'den ZARAR görür (XRP, INJ, FET tarzı)

        Hardcoded per-symbol kararı yerine, indikatörlerden anlık olarak
        coin'in karakterini "okur". Bu sayede aynı coin zamanla karakterini
        değiştirirse (BTC volatil olabilir, XRP stabilleşebilir) sistem uyum sağlar.

        Eşikler tüm değerler [None] verilirse o kontrol atlanır → testte gevşetilebilir.
        Döner: (allowed, reason)
        """
        try:
            atr = float(row.get("atr", 0.0) or 0.0)
            close = float(row.get("close", 0.0) or 0.0)
            atr_sma_20 = float(row.get("atr_sma_20", 0.0) or 0.0)
            adx = float(row.get("adx", 0.0) or 0.0)
            regime = float(row.get("regime_score", 0.0) or 0.0)
        except Exception:
            return False, "indikatör değerleri okunamadı"

        # 1) Trend rejimi yeterince güçlü mü?
        if min_regime_score is not None and regime < min_regime_score:
            return False, f"regime_score düşük ({regime:.2f} < {min_regime_score})"

        # 2) Volatilite spike var mı? (mevcut ATR, 20-bar ortalamanın üstünde mi)
        if max_vol_spike is not None and atr_sma_20 > 0:
            vol_spike = atr / atr_sma_20
            if vol_spike > max_vol_spike:
                return False, f"volatilite spike ({vol_spike:.2f} > {max_vol_spike})"

        # 3) Coin yapısal olarak çok mu volatil? (ATR/price oranı yüksek mi)
        if max_atr_ratio is not None and close > 0:
            atr_ratio = atr / close
            if atr_ratio > max_atr_ratio:
                return False, f"yapısal volatilite yüksek (ATR/price {atr_ratio:.3f} > {max_atr_ratio})"

        # 4) ADX trend gücü yeterli mi?
        if min_adx is not None and adx < min_adx:
            return False, f"ADX zayıf ({adx:.1f} < {min_adx})"

        return True, (
            f"ok (regime={regime:.2f}, ADX={adx:.1f}, "
            f"vol_spike={atr / atr_sma_20 if atr_sma_20 > 0 else 0:.2f}, "
            f"atr_ratio={atr / close if close > 0 else 0:.3f})"
        )

    def get_corr_scale(self, symbol: str, open_symbols: list[str]) -> float:
        """
        Cross-symbol correlation scale (Carver-light MVP).

        Crypto'da BTC-alt korelasyon 0.7-0.9; 8 paralel pozisyon ≠ 8 bağımsız risk.
        Bear günü gelince hepsi birlikte stop yiyor → MaxDD patlar.

        Bu heuristic: açık alt sayısı arttıkça yeni alt entry'leri küçültür.
        BTC tek başına kabul edilir (genelde liderlik eder), alt'lar diğer alt sayısına
        göre ölçeklenir.

        Akademik kaynak: Carver, Systematic Trading (2015), Ch. 4 — IDM.
        MVP — sonra Carver IDM formülüne (√(N_eff/N)) geçilebilir.
        """
        if symbol == "BTC/USDT":
            return 1.0
        n_alts = sum(1 for s in open_symbols if s != "BTC/USDT")
        if n_alts >= 4:
            return 0.6   # 5+ alts → tüm yeni alt 0.6× sizing
        if n_alts >= 2:
            return 0.8   # 3-4 alts → 0.8×
        return 1.0       # 0-2 alts → full size

    def get_adx_scale(self, adx: float) -> float:
        """
        ADX gücüne göre pozisyon boyutu çarpanı [0.4× – 2.5×].
        Turtle Trader / Donchian prensibi: trendin gücüyle bahsi büyüt.

        ADX < 15  → Ranging: çok küçük pozisyon
        ADX 15-20 → Zayıf trend: küçük pozisyon
        ADX 20-28 → Normal trend: standart pozisyon
        ADX 28-36 → Güçlü trend: büyük pozisyon
        ADX 36-44 → Çok güçlü: maksimum normal
        ADX > 44  → İstisnaî (Bitcoin rally, SOL breakout): 2.5×
        """
        if adx >= 44:
            return 2.5   # istisnaî güçlü trend — maksimum bahis
        elif adx >= 36:
            return 2.0
        elif adx >= 28:
            return 1.5
        elif adx >= 20:
            return 1.0
        elif adx >= 15:
            return 0.7
        else:
            return 0.4

    def performance_summary(self) -> dict:
        """Per-symbol rolling win rate özetini döner (dashboard için)."""
        result = {}
        for sym, outcomes in self._symbol_outcomes.items():
            if not outcomes:
                continue
            wr = sum(outcomes) / len(outcomes)
            paused = self._symbol_paused_until.get(sym)
            result[sym] = {
                "rolling_wr":    round(wr, 3),
                "sample_size":   len(outcomes),
                "is_paused":     paused is not None and datetime.now(timezone.utc) < paused,
                "paused_until":  paused.strftime("%Y-%m-%d") if paused else None,
            }
        return result

    # ──────────────────────────────────────────────────────────────────
    # Cross-sectional momentum skoru (PaperTrader tarafından kullanılır)
    # ──────────────────────────────────────────────────────────────────

    def get_momentum_rank_score(self, df: pd.DataFrame) -> float:
        """
        Sembolün cross-sectional momentum skoru.
        PaperTrader tüm sembolleri bu skora göre sıralar,
        en yüksek N sembol öncelik alır.
        Returns: float (yüksek = daha güçlü momentum)
        """
        if len(df) < 24:
            return 0.0
        try:
            close = df["close"]
            ret_24h = close.pct_change(24).iloc[-1]   # 24 saatlik getiri
            ret_1h  = close.pct_change(1).iloc[-1]    # 1 saatlik getiri
            vol_24h = close.pct_change().rolling(24).std().iloc[-1]
            tsmom   = df.get("tsmom", pd.Series([0])).iloc[-1] if "tsmom" in df.columns else 0.0
            if vol_24h and vol_24h > 0:
                sharpe = ret_24h / vol_24h
            else:
                sharpe = 0.0
            # Bileşik momentum skoru
            score = sharpe * 0.5 + (ret_24h * 100) * 0.3 + (ret_1h * 100) * 0.2
            return float(score) if not pd.isna(score) else 0.0
        except Exception:
            return 0.0

    # ──────────────────────────────────────────────────────────────────
    # Yardımcılar
    # ──────────────────────────────────────────────────────────────────

    def _hold(self, symbol: str, df: pd.DataFrame, reason: str) -> Signal:
        close = df.iloc[-1]["close"] if not df.empty else 0.0
        return Signal(
            symbol=symbol, side=Side.HOLD, reason=reason,
            timestamp=datetime.now(timezone.utc), price=close,
            confidence_score=0.0,
        )
