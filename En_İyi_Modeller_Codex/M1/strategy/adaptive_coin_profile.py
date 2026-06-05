"""
Adaptif Coin Profili
====================
Her coin için son işlem performansını VE anlık piyasa mikroyapısını birleştirerek
dinamik giriş eşiği (entry_score threshold) hesaplar.

Tasarım Felsefesi
-----------------
Sabit threshold (örn. ALGO için adx_threshold=32) yerine:
  - Bot kendi geçmişini öğrenir → iyi dönemde daha agresif, kötü dönemde daha seçici
  - Piyasa bağlamı (EMA200, ADX, ATR) anlık duruma göre eşiği ayarlar
  - Elle kural yazılmaz; her coin kendi verisinden adapte olur

İki Faktör
----------
1. Performans Faktörü  : Son N işlemde WR, ortalama kâr, ardışık kayıp, SL oranı
2. Piyasa Mikroyapısı  : EMA200 uzaklığı, ADX güç, ATR durum, 24h momentum

Sınır: ±max_adj (varsayılan ±0.12) — ani büyük sapmaları önler.

Kullanım
--------
    # Başlangıçta:
    profile = AdaptiveCoinProfile("SOL/USDT", base_entry_trend=0.72, base_entry_ranging=0.76)

    # Her işlem kapandığında:
    profile.record_trade(pnl=45.2, pnl_pct=0.018, exit_reason="strategy_exit", duration_h=36)

    # Giriş kararı öncesinde:
    thr = profile.adaptive_threshold(slice_df, is_trending=True)
    # → örn. 0.76 (kötü geçmiş varsa base 0.72'den yükseldi)
    # → ya da 0.69 (iyi geçmiş + güçlü trend varsa düştü)
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import pandas as pd


class AdaptiveCoinProfile:
    """
    Coin başına adaptif giriş eşiği yönetimi.

    Parametreler
    ------------
    sym               : Coin sembolü (log/debug için)
    base_entry_trend  : Trending rejimde temel eşik (örn. 0.72)
    base_entry_ranging: Ranging rejimde temel eşik (örn. 0.76)
    window            : Son kaç işlem dikkate alınsın (varsayılan 8)
    max_adj           : Maksimum ± eşik değişimi (varsayılan 0.12 = ±12 puan)
    """

    def __init__(
        self,
        sym: str,
        base_entry_trend: float,
        base_entry_ranging: float,
        window: int = 8,
        max_adj: float = 0.12,
    ) -> None:
        self.sym              = sym
        self.base_entry_trend   = base_entry_trend
        self.base_entry_ranging = base_entry_ranging
        self.window           = window
        self.max_adj          = max_adj

        # Yuvarlanır pencere — maksimum 30 işlem tutulur, eski olanlar düşer
        self._trades: deque = deque(maxlen=30)

    # ──────────────────────────────────────────────────────────────
    def record_trade(
        self,
        pnl: float,
        pnl_pct: float,
        exit_reason: str,
        duration_h: float,
    ) -> None:
        """
        Bir işlem kapandığında çağrılır.

        Parametreler
        ------------
        pnl         : Net kâr/zarar (USD)
        pnl_pct     : İşlem getirisi (pnl / maliyet), örn. 0.018 = %1.8
        exit_reason : 'stop_loss' | 'trailing_stop' | 'strategy_exit' | 'backtest_end'
        duration_h  : Pozisyon tutulma süresi (saat)
        """
        self._trades.append({
            "pnl":         pnl,
            "pnl_pct":     pnl_pct,
            "exit_reason": exit_reason,
            "duration_h":  duration_h,
        })

    # ──────────────────────────────────────────────────────────────
    def adaptive_threshold(
        self,
        slice_df: pd.DataFrame,
        is_trending: bool,
    ) -> float:
        """
        Dinamik giriş eşiği hesapla.

        Yeterli işlem geçmişi varsa (≥3) iki faktörü birleştirir:
          - Performans faktörü   : son window işlemden öğrenilir
          - Mikroyapı faktörü    : anlık EMA200/ADX/ATR/momentum'dan

        Geçmiş yetersizse yalnızca mikroyapı faktörü uygulanır.

        Returns
        -------
        float : adjusted threshold, [0.50, 0.95] arasında sınırlı
        """
        base   = self.base_entry_trend if is_trending else self.base_entry_ranging
        recent = list(self._trades)[-self.window:]

        perf_adj  = self._performance_adj(recent) if len(recent) >= 3 else 0.0
        micro_adj = self._market_micro_adj(slice_df)

        total = float(np.clip(perf_adj + micro_adj, -self.max_adj, +self.max_adj))
        return float(np.clip(base + total, 0.50, 0.95))

    # ──────────────────────────────────────────────────────────────
    def _performance_adj(self, recent: list) -> float:
        """
        Son işlem geçmişinden eşik farkı hesapla.

        Mantık:
          - WR < %35 + ortalama kayıp  → daha seçici (+)
          - WR > %65 + ortalama kâr    → daha agresif (−)
          - Ardışık kayıp sayısı        → ceza (+)
          - SL oranı yüksekse           → ekstra ceza (+)
        """
        n   = len(recent)
        wins = sum(1 for t in recent if t["pnl"] > 0)
        wr   = wins / n

        avg_pnl_pct = sum(t["pnl_pct"] for t in recent) / n
        sl_rate     = sum(1 for t in recent if t["exit_reason"] == "stop_loss") / n

        # Ardışık kayıp sayısı (en yeniden geriye say)
        consec_loss = 0
        for t in reversed(recent):
            if t["pnl"] <= 0:
                consec_loss += 1
            else:
                break

        # ── Bileşik performans skoru: −1 (kötü) → +1 (iyi) ──────
        # WR=0.50 → 0.0 | WR=0.30 → −0.40 | WR=0.70 → +0.40
        score = (wr - 0.50) * 2.0

        if avg_pnl_pct < -0.015:   # Ortalama kayıp %1.5+ → kötü dönem
            score -= 0.30
        elif avg_pnl_pct > 0.020:  # Ortalama kâr %2+ → iyi dönem
            score += 0.20

        if sl_rate > 0.50:         # İşlemlerin yarısından fazlası SL
            score -= 0.20

        # ── Skoru threshold farkına dönüştür ─────────────────────────
        # Kötü performans koruma sağlar; iyi performans ise yalnızca küçük
        # bir gevşeme verir. Böylece bot, çalışan rejimde maruziyeti artırır
        # ama tek iyi işlemden sonra kapıları sonuna kadar açmaz.
        if   score < -0.40:   adj = +0.08   # çok kötü → çok daha seçici
        elif score < -0.15:   adj = +0.04   # kötü     → daha seçici
        elif score > 0.45 and sl_rate < 0.25 and avg_pnl_pct > 0:
            adj = -0.03                   # temiz iyi dönem → biraz daha erken gir
        elif score > 0.20 and avg_pnl_pct > 0.01:
            adj = -0.015
        else:
            adj = 0.00

        # Ardışık kayıp ekstra cezası
        if   consec_loss >= 3:  adj += 0.06
        elif consec_loss == 2:  adj += 0.03

        return float(adj)

    # ──────────────────────────────────────────────────────────────
    def _market_micro_adj(self, df: pd.DataFrame) -> float:
        """
        Anlık piyasa durumundan eşik farkı hesapla.

        Değişkenler:
          - EMA200 uzaklığı : trend bağlamı (fiyat nerede?)
          - ADX              : trend gücü (ne kadar güvenilir?)
          - ATR durum        : volatilite kalitesi (gürültü mu trend mi?)
          - 24h momentum     : anlık ivme

        Tasarım notu: Bu faktör işlem geçmişi olmasa bile çalışır.
        Yeni coinlerde veya uzun süre işlem yapılmayan coinlerde devreye girer.
        """
        adj = 0.0
        if len(df) < 10:
            return adj

        close = float(df["close"].iloc[-1])

        # ── EMA200 uzaklığı ───────────────────────────────────────
        ema200 = self._get_ema200(df)
        if ema200 and ema200 > 0:
            dist = close / ema200 - 1.0   # pozitif = EMA üstünde, negatif = altında

            if   dist < -0.05:   adj += 0.04   # EMA200 %5 altında → ayı bağlamı → seçici
            elif dist < -0.02:   adj += 0.02   # Hafif altında → dikkatli
            elif dist > 0.08:    adj -= 0.015  # Temiz yukarı bağlam → küçük gevşeme

        # ── ADX: trend gücü ───────────────────────────────────────
        if "adx" in df.columns:
            adx = float(df["adx"].iloc[-1])
            if not pd.isna(adx):
                if adx < 18:  adj += 0.03   # Zayıf trend → choppy piyasa → seçici
                elif adx >= 35: adj -= 0.02 # Çok güçlü trend → daha erken katıl

        # ── ATR durumu: normalleştirilmiş volatilite ──────────────
        # SADECE KORUMA: Aşırı volatilite = gürültü, eşiği yükselt.
        if "atr" in df.columns and len(df) >= 50:
            atr_now  = float(df["atr"].iloc[-1])
            atr_mean = float(df["atr"].tail(50).mean())
            if atr_mean > 0 and not pd.isna(atr_now):
                ratio = atr_now / atr_mean
                if ratio > 2.5:  adj += 0.03   # Çok aşırı volatil → gürültü artmış

        # ── 24h fiyat momentumu ───────────────────────────────────
        # SADECE KORUMA: Güçlü düşüş momentumu → bekle.
        if len(df) >= 25:
            c_now = float(df["close"].iloc[-1])
            c_24h = float(df["close"].iloc[-25])
            if c_24h > 0:
                mom_24h = c_now / c_24h - 1.0
                if mom_24h < -0.07:  adj += 0.02   # Son 24h %7+ düşüş → zayıf ivme
                elif mom_24h > 0.08: adj -= 0.01   # Güçlü ivme → biraz daha proaktif

        # ── Yeni trend_quality skoru varsa onu ana bağlam sinyali olarak kullan
        tq_cols = [c for c in ("trend_quality_up", "trend_quality_down") if c in df.columns]
        if tq_cols:
            tq_vals = [float(df[c].iloc[-1]) for c in tq_cols if not pd.isna(df[c].iloc[-1])]
            if tq_vals:
                tq = max(tq_vals)
                if tq >= 0.70:
                    adj -= 0.025
                elif tq < 0.30:
                    adj += 0.025

        return float(adj)

    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _get_ema200(df: pd.DataFrame) -> Optional[float]:
        """DataFrame'den EMA200 değerini çek (farklı sütun isimlerini dene)."""
        for col in ("ema_200", "ema200", "ema_slow"):
            if col in df.columns:
                v = df[col].iloc[-1]
                if not pd.isna(v) and float(v) > 0:
                    return float(v)
        return None

    # ──────────────────────────────────────────────────────────────
    def status(self) -> dict:
        """
        Debug ve loglama için anlık durumu döndür.

        Returns
        -------
        dict: sym, son işlem sayısı, WR, ortalama kâr, performans farkı
        """
        recent = list(self._trades)[-self.window:]
        if not recent:
            return {
                "sym":      self.sym,
                "trades":   0,
                "wr":       None,
                "avg_pnl":  None,
                "perf_adj": 0.0,
            }
        wins = sum(1 for t in recent if t["pnl"] > 0)
        return {
            "sym":      self.sym,
            "trades":   len(recent),
            "wr":       round(wins / len(recent), 3),
            "avg_pnl":  round(sum(t["pnl_pct"] for t in recent) / len(recent), 4),
            "perf_adj": round(self._performance_adj(recent), 3),
        }
