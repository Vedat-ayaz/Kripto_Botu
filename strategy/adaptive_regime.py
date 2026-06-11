"""
Adaptif Piyasa Rejim Kontrolörü
================================
Bot, mevcut piyasa koşulunu her barda otomatik olarak tespit eder ve
parametrelerini buna göre ayarlar — manuel re-tune gerekmez.

Rejim Seviyeleri (0-4):
    STRONG_BEAR (0): BTC < EMA200 + zayıf trend + düşük WR
    BEAR        (1): BTC < EMA200 veya trend zayıf
    NEUTRAL     (2): Belirsiz / geçiş dönemi
    BULL        (3): BTC > EMA200 + orta trend
    STRONG_BULL (4): BTC > EMA200 + güçlü trend + yüksek WR

Her Rejim İçin Otomatik Ayarlar:
    - Giriş eşiği (entry_score): bear'da yükselir → daha seçici giriş
    - Pozisyon boyutu: bear'da küçülür → riski azalt
    - Trailing stop genişliği: bear'da daralır → kazanıları hızlı kilitle
    - Açık pozisyon limiti: bear'da azalır
    - Coin erişim katmanı: bear'da yalnızca BTC/ETH
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Regime(IntEnum):
    STRONG_BEAR = 0
    BEAR        = 1
    NEUTRAL     = 2
    BULL        = 3
    STRONG_BULL = 4


@dataclass
class RegimeParams:
    """Her rejime karşılık gelen dinamik parametreler."""
    position_size_mult: float    # risk_per_trade çarpanı (1.0 = değişmez)
    entry_score_boost: float     # entry_score'a eklenir (+= daha katı, -= daha gevşek)
    trailing_mult_boost: float   # trailing_stop_atr_multiplier'a eklenir
    max_positions: int           # aynı anda maksimum açık pozisyon
    coin_tier: int               # 1..5 — hangi coinler işlem görebilir


# ── Sabit: Rejim → Parametre tablosu ─────────────────────────────────────────

_REGIME_TABLE: dict[Regime, RegimeParams] = {
    # v18 (2 Haziran) değerleri — canlı AWS ile eşleşiyor, 13 günlük live test
    # daha iyi sonuç verdi (+2.94% live vs +0.79% backtest v17 ile).
    Regime.STRONG_BEAR: RegimeParams(
        position_size_mult=0.25,
        entry_score_boost=+0.20,
        trailing_mult_boost=-0.5,
        max_positions=5,
        coin_tier=5,
    ),
    Regime.BEAR: RegimeParams(
        position_size_mult=0.45,
        entry_score_boost=+0.12,
        trailing_mult_boost=-0.2,
        max_positions=6,
        coin_tier=5,
    ),
    Regime.NEUTRAL: RegimeParams(
        position_size_mult=0.38,   # v18 (2 Haziran): canlı ile eşleşiyor
        entry_score_boost=+0.10,   # v18 (2 Haziran): canlı ile eşleşiyor
        trailing_mult_boost=+0.0,
        max_positions=4,           # v18 (2 Haziran): canlı ile eşleşiyor
        coin_tier=5,
    ),
    Regime.BULL: RegimeParams(
        position_size_mult=1.10,
        entry_score_boost=-0.03,
        trailing_mult_boost=+0.5,
        max_positions=9,
        coin_tier=5,
    ),
    Regime.STRONG_BULL: RegimeParams(
        position_size_mult=1.20,
        entry_score_boost=-0.06,
        trailing_mult_boost=+1.0,
        max_positions=10,
        coin_tier=5,
    ),
}

# ── Coin erişim katmanları ────────────────────────────────────────────────────
# Tier 1 → sadece BTC (en güvenli, her rejimde işlem görür)
# Tier 5 → tüm coinler (sadece bull/strong_bull'da)

COIN_TIERS: dict[int, Optional[list[str]]] = {
    1: ["BTC/USDT"],
    2: ["BTC/USDT", "ETH/USDT"],
    3: ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOT/USDT"],
    4: ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOT/USDT",
        "INJ/USDT", "FET/USDT"],
    5: None,  # None = tüm coinler serbest
    # v21: M5 özel tier'ları — AVAX/DOGE BEAR'da tutarlı kaybettiriyor (29%/34% WR)
    # BEAR'da sadece güvenli coinlere izin ver, volatil küçük coinleri dışarıda bırak
    6: ["BNB/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "TRX/USDT",
        "DOT/USDT", "LEO/USDT"],  # AVAX + DOGE hariç
    7: ["ETH/USDT", "SOL/USDT", "ADA/USDT", "TRX/USDT", "DOT/USDT"],  # STRONG_BEAR: sadece en güçlüler
}


# ── Kontrolör ─────────────────────────────────────────────────────────────────

class AdaptiveRegimeController:
    """
    Her barda piyasa rejimini hesaplar ve uygun RegimeParams döndürür.

    Girdi sinyalleri (ağırlıklı bileşik puan):
        1. BTC EMA200 pozisyonu   → ağırlık 1.5  (birincil)
        2. BTC regime_score (indikatörden)   → ağırlık 1.0
        3. Cross-asset ortalama regime_score → ağırlık 1.0
        4. Son işlemlerin WR performansı     → ağırlık 0.5 (bonus/malus)

    Ham puan 0–4 arası, ardından flip-flop önlemek için
    EMA yumuşatması uygulanır.
    """

    def __init__(
        self,
        smooth_window: int = 12,       # puan EMA penceresi (12 bar ≈ 12 saat)
        wr_window: int = 20,           # rolling WR için son N işlem
        wr_bonus_threshold: float = 0.50,   # WR bu değeri aşarsa +0.5 bonus
        wr_penalty_threshold: float = 0.30, # WR bu değerin altında -0.5 malus
    ):
        self._smooth_window = smooth_window
        self._wr_window     = wr_window
        self._wr_bonus_thr  = wr_bonus_threshold
        self._wr_pen_thr    = wr_penalty_threshold

        self._score_ema: Optional[float] = None   # yumuşatılmış ham puan
        self._current_regime: Regime = Regime.NEUTRAL
        self._recent_wins: deque[bool] = deque(maxlen=wr_window)

        # Performans log'u (raporlama için)
        self._regime_log: list[tuple[pd.Timestamp, Regime, float]] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        ts: pd.Timestamp,
        btc_df: pd.DataFrame,
        all_sym_ind: dict[str, pd.DataFrame],
    ) -> tuple[Regime, RegimeParams]:
        """
        Mevcut timestamp için rejimi hesapla ve parametrelerini döndür.

        Args:
            ts           : Mevcut bar timestamp'i
            btc_df       : BTC indikatör DataFrame'i (ema_slow, regime_score içermeli)
            all_sym_ind  : Tüm sembollerin indikatör DataFrame'leri

        Returns:
            (Regime, RegimeParams) ikilisi
        """
        raw_score = self._compute_raw_score(ts, btc_df, all_sym_ind)
        smoothed  = self._smooth(raw_score)
        regime    = self._classify(smoothed)

        if regime != self._current_regime:
            logger.info(
                f"[AdaptiveRegime] {ts.strftime('%Y-%m-%d %H:%M')} "
                f"REJİM DEĞİŞTİ: {self._current_regime.name} → {regime.name} "
                f"(ham={raw_score:.2f} yumuşatılmış={smoothed:.2f})"
            )
            self._current_regime = regime

        params = _REGIME_TABLE[regime]
        self._regime_log.append((ts, regime, smoothed))
        return regime, params

    def record_trade(self, won: bool) -> None:
        """İşlem sonucunu kaydet (WR hesabı için)."""
        self._recent_wins.append(won)

    def current_regime(self) -> Regime:
        return self._current_regime

    def current_params(self) -> RegimeParams:
        return _REGIME_TABLE[self._current_regime]

    def allowed_coins(self, all_coins: list[str]) -> list[str]:
        """Mevcut rejimde işlem yapılabilecek coinleri döndürür."""
        tier = _REGIME_TABLE[self._current_regime].coin_tier
        allowed = COIN_TIERS.get(tier)
        return all_coins if allowed is None else [c for c in all_coins if c in allowed]

    def regime_summary(self) -> dict:
        """Toplam simülasyon boyunca rejim dağılımı (raporlama)."""
        if not self._regime_log:
            return {}
        counts: dict[str, int] = {}
        for _, r, _ in self._regime_log:
            counts[r.name] = counts.get(r.name, 0) + 1
        total = len(self._regime_log)
        return {k: f"{v/total*100:.1f}%" for k, v in counts.items()}

    # ── İç hesaplamalar ───────────────────────────────────────────────────────

    def _compute_raw_score(
        self,
        ts: pd.Timestamp,
        btc_df: pd.DataFrame,
        all_sym_ind: dict[str, pd.DataFrame],
    ) -> float:
        """0–4 arası ham rejim puanı hesaplar."""
        score = 0.0

        # ── Faktör 1: BTC EMA200 pozisyonu (ağırlık 1.5) ─────────────────
        if ts in btc_df.index:
            row = btc_df.loc[ts]
            btc_close     = float(row.get("close", 0))
            btc_ema_slow  = float(row.get("ema_slow", row.get("ema200", row.get("ema_200", 0))))
            if btc_ema_slow > 0:
                if btc_close > btc_ema_slow:
                    # Kaç % üstünde? Güçlü boğada daha fazla puan
                    gap = (btc_close - btc_ema_slow) / btc_ema_slow
                    score += min(1.5, 1.0 + gap * 10)
                else:
                    gap = (btc_ema_slow - btc_close) / btc_ema_slow
                    score += max(0.0, 0.5 - gap * 10)

        # ── Faktör 2: BTC regime_score (ADX+Hurst) (ağırlık 1.0) ─────────
        if ts in btc_df.index and "regime_score" in btc_df.columns:
            btc_rs = float(btc_df.loc[ts, "regime_score"])
            if not np.isnan(btc_rs):
                score += btc_rs * 1.0

        # ── Faktör 3: Cross-asset ortalama regime_score (ağırlık 1.0) ────
        cross_scores = []
        for sym, df in all_sym_ind.items():
            if sym == "BTC/USDT":
                continue
            if ts in df.index and "regime_score" in df.columns:
                rs = float(df.loc[ts, "regime_score"])
                if not np.isnan(rs):
                    cross_scores.append(rs)
        if cross_scores:
            score += float(np.mean(cross_scores)) * 1.0

        # ── Faktör 4: BTC uzun vadeli EMA karşılaştırması (YÖN tespiti) ─────
        # regime_score (ADX+Hurst) yalnızca TREND GÜCÜ gösterir, yönü değil.
        # Kısa vadeli ema_slow (200h≈8 gün) fiyat-EMA farkı %1-3 → yetersiz.
        # Önceden hesaplanmış ema_168h (7 gün) ve ema_720h (30 gün) ile YÖN belirlenir:
        #   - Her ikisi de fiyat < EMA → AYI
        #   - İkisi de fiyat > EMA → BOĞA
        #   - Çelişki → belirsiz/geçiş
        if ts in btc_df.index and btc_close > 0:
            row_btc = btc_df.loc[ts]
            ema_30d = float(row_btc.get("ema_720h", 0) or 0)  # 720h = 30 gün EMA
            ema_7d  = float(row_btc.get("ema_168h", 0) or 0)  # 168h = 7 gün EMA

            if ema_30d > 0:
                vs_30d = (btc_close - ema_30d) / ema_30d   # + = fiyat EMA üstünde
                vs_7d  = (btc_close - ema_7d)  / ema_7d  if ema_7d > 0 else vs_30d

                # Her iki EMA onayladığında güçlü sinyal
                if vs_30d > 0.08 and vs_7d > 0.02:       score += 1.5  # güçlü boğa
                elif vs_30d > 0.03 and vs_7d >= 0:        score += 1.0  # boğa
                elif vs_30d > 0.00:                        score += 0.5  # zayıf boğa
                elif vs_30d < -0.05 and vs_7d < -0.01:   score -= 0.4  # güçlü ayı
                elif vs_30d < -0.02:                       score -= 0.2  # ayı
                # -0.02 ile 0 arası → yatay/geçiş → nötr (0 ekle)

        # ── Faktör 5: Rolling portfolio WR bonus/malus (ağırlık 0.5) ─────
        if len(self._recent_wins) >= 5:
            wr = sum(self._recent_wins) / len(self._recent_wins)
            if wr >= self._wr_bonus_thr:
                score += 0.5
            elif wr <= self._wr_pen_thr:
                score -= 0.5

        return float(np.clip(score, 0.0, 4.0))

    def _smooth(self, raw: float) -> float:
        """EMA yumuşatması — ani rejim geçişlerini önler."""
        alpha = 2.0 / (self._smooth_window + 1)
        if self._score_ema is None:
            self._score_ema = raw
        else:
            self._score_ema = alpha * raw + (1 - alpha) * self._score_ema
        return self._score_ema

    @staticmethod
    def _classify(score: float) -> Regime:
        """Yumuşatılmış puana göre rejim sınıflandırması."""
        if score >= 3.2:
            return Regime.STRONG_BULL
        elif score >= 2.2:
            return Regime.BULL
        elif score >= 1.2:
            return Regime.NEUTRAL
        elif score >= 0.4:
            return Regime.BEAR
        else:
            return Regime.STRONG_BEAR


def get_regime_params(regime: Regime) -> RegimeParams:
    """Dış kullanım için yardımcı fonksiyon."""
    return _REGIME_TABLE[regime]
