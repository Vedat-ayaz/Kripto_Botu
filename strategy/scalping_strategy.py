"""
Yüksek Frekanslı Scalping Stratejisi v2 (5m grafik)
====================================================
Radikal yeniden yazım — 7 kritik bug fix + yeni indikatörler

Değişiklikler v1 → v2:
  TP/SL     : %0.5/%0.25 → %0.8/%0.3  (fee dahil net R:R 1:0.67 → 1:1.33)
  Supertrend: bearish iken giriş HARD GATE (yeni)
  VWAP      : kurumsal seviye filtresi (yeni)
  Vol Delta : alıcı/satıcı baskısı EMA (yeni)
  MACD      : 5/13/3 → 3/10/16  (signal=3 çok gürültülüydü)
  RSI       : period 7 → 14  (kısa vadeli gürültü azaltıldı)
  BB        : period 10 → 20
  α         : 0.25 → 0.10  (etkin pencere ~4 → ~10 trade)
  Threshold : cap 0.65 → 0.58  (sinyal starvation döngüsü önlendi)
  Decay     : 24h idle → BASE'e %15 geri çekiş  (kilitlenme kırıcı)
  Exit      : EMA flip kaldırıldı → Supertrend flip  (daha güvenilir)
  Bar count : global _bar_counter yerine timestamp bazlı (bug fix)
  MACD edge : prev≈0 bölme overflow koruması eklendi  (bug fix)
  BB %B     : 0.0 or 0.5 yanlış fallback düzeltildi  (bug fix)
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from .signal import Signal, Side

logger = logging.getLogger(__name__)


# ─── Gösterge Fonksiyonları ───────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    ag = gain.ewm(com=period - 1, min_periods=period).mean()
    al = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 3, slow: int = 10, sig: int = 16):
    """MACD 3/10/16 — 5/13/3'ten daha az whipsaw (signal=16 daha stabil)."""
    ema_f  = _ema(close, fast)
    ema_s  = _ema(close, slow)
    line   = ema_f - ema_s
    signal = _ema(line, sig)
    hist   = line - signal
    return line, signal, hist


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10) -> pd.Series:
    prev_c = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """
    ATR tabanlı Supertrend.
    Returns: (supertrend_values, direction)  direction: +1 bullish, -1 bearish
    """
    atr = _atr(high, low, close, period)
    hl2 = (high + low) / 2.0

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    n        = len(close)
    st_vals  = np.full(n, np.nan)
    dir_vals = np.zeros(n, dtype=int)

    for i in range(n):
        if i < period:
            continue
        if i == period:
            st_vals[i]  = lower_band.iloc[i]
            dir_vals[i] = 1
            continue

        prev_dir = dir_vals[i - 1]
        prev_st  = st_vals[i - 1]
        lb_i     = lower_band.iloc[i]
        ub_i     = upper_band.iloc[i]

        if prev_dir == 1:                    # bullish
            final_lb = max(lb_i, prev_st)
            if close.iloc[i] < final_lb:
                st_vals[i]  = ub_i
                dir_vals[i] = -1
            else:
                st_vals[i]  = final_lb
                dir_vals[i] = 1
        else:                                # bearish
            final_ub = min(ub_i, prev_st)
            if close.iloc[i] > final_ub:
                st_vals[i]  = lb_i
                dir_vals[i] = 1
            else:
                st_vals[i]  = final_ub
                dir_vals[i] = -1

    return (
        pd.Series(st_vals,  index=close.index, dtype=float),
        pd.Series(dir_vals, index=close.index, dtype=int),
    )


def _vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """Session VWAP — DataFrame başından kümülatif."""
    typical = (high + low + close) / 3.0
    cum_tpv = (typical * volume).cumsum()
    cum_vol = volume.cumsum().replace(0, np.nan)
    return cum_tpv / cum_vol


def _volume_delta_ema(
    open_: pd.Series, close: pd.Series, volume: pd.Series, period: int = 10
) -> pd.Series:
    """
    Volume delta yaklaşımı — mum yönüne göre signed hacim EMA'sı.
    Pozitif → alıcı baskısı, Negatif → satıcı baskısı.
    """
    body   = close - open_
    range_ = (close - open_).abs().replace(0, np.nan)
    signed = volume * (body / range_).clip(-1.0, 1.0).fillna(0.0)
    return signed.ewm(span=period, adjust=False).mean()


# Geriye dönük uyumluluk için eski _atr adını dışa aç
_fast_atr = _atr   # scalper_trader.py hâlâ bu isimle import ediyor


# ─── Scalping Stratejisi v2 ───────────────────────────────────────────────────

class ScalpingStrategy:
    """
    5 dakikalık scalping — Supertrend gate, VWAP + Volume Delta destekli.

    Kritik düzeltmeler (v1 → v2):
      1. Fee-adjusted R:R düzeltildi  (TP↑, SL↑ — artık pozitif beklenti)
      2. α = 0.10  →  daha kararlı win rate EMA
      3. Threshold cap = 0.58  →  sinyal starvation döngüsü kırıldı
      4. Decay mekanizması  →  24h idle sonrası BASE'e %15 geri çek
      5. MACD prev≈0 edge case koruması
      6. BB %B sıfır fallback bug düzeltildi
      7. EMA flip exit kaldırıldı, Supertrend flip eklendi
    """

    BASE_THRESHOLD  = 0.44
    ALPHA           = 0.10   # eski 0.25
    THRESHOLD_CAP   = 0.58   # eski 0.65
    THRESHOLD_FLOOR = 0.38

    def __init__(
        self,
        profit_target_pct: float = 0.008,   # %0.8  (config'den gelir)
        stop_loss_pct:     float = 0.003,   # %0.3  (config'den gelir)
        max_hold_bars:     int   = 8,       # 8×5m = 40 dk
        volume_mult:       float = 0.4,
    ):
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct     = stop_loss_pct
        self.max_hold_bars     = max_hold_bars
        self.volume_mult       = volume_mult

        # Per-symbol adaptive state
        self._thresholds:      dict[str, float] = {}
        self._win_rates:       dict[str, float] = {}
        self._trade_counts:    dict[str, int]   = {}
        self._last_trade_time: dict[str, str]   = {}   # ISO — decay için
        self._last_bar:        dict[str, str]   = {}

    # ── Adaptif Öğrenme ───────────────────────────────────────────────────────

    def record_trade_result(self, symbol: str, pnl: float) -> None:
        """
        Trade kapandığında çağrılır.
        Win rate EMA günceller, threshold ayarlar.
        """
        win    = float(pnl > 0)
        old_wr = self._win_rates.get(symbol, 0.50)
        new_wr = self.ALPHA * win + (1 - self.ALPHA) * old_wr
        self._win_rates[symbol]       = new_wr
        self._trade_counts[symbol]    = self._trade_counts.get(symbol, 0) + 1
        self._last_trade_time[symbol] = datetime.now(timezone.utc).isoformat()

        old_t = self._thresholds.get(symbol, self.BASE_THRESHOLD)
        if   new_wr > 0.65: delta = -0.005
        elif new_wr > 0.58: delta = -0.002
        elif new_wr < 0.35: delta = +0.010
        elif new_wr < 0.45: delta = +0.004
        else:               delta =  0.0

        new_t = float(np.clip(old_t + delta, self.THRESHOLD_FLOOR, self.THRESHOLD_CAP))
        self._thresholds[symbol] = new_t

        logger.info(
            f"[ScalpLearn] {symbol} | {'✅WIN' if win else '❌LOSS'} PnL={pnl:+.4f} | "
            f"WR(EMA)={new_wr:.2%} | Eşik={old_t:.3f}→{new_t:.3f} | "
            f"Trade#{self._trade_counts[symbol]}"
        )

    def maybe_decay_thresholds(self) -> None:
        """
        24+ saat işlem olmayan sembollerde threshold'u BASE'e doğru %15 çek.
        Sinyal starvation kilitlenme döngüsünü kırar.

        Çağrı yeri: ScalperTrader._log_status() — her ~40 tick (~20 dk).
        """
        now = datetime.now(timezone.utc)
        for sym in list(self._thresholds):
            last_str = self._last_trade_time.get(sym)
            if not last_str:
                continue
            try:
                last_dt = datetime.fromisoformat(last_str)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                hours_since = (now - last_dt).total_seconds() / 3600
                if hours_since >= 24:
                    old = self._thresholds[sym]
                    self._thresholds[sym] = round(
                        old + 0.15 * (self.BASE_THRESHOLD - old), 4
                    )
                    logger.info(
                        f"[ScalpLearn] {sym}: Threshold decay "
                        f"{old:.3f}→{self._thresholds[sym]:.3f} "
                        f"({hours_since:.0f}h idle)"
                    )
            except Exception:
                pass

    def get_threshold(self, symbol: str) -> float:
        return self._thresholds.get(symbol, self.BASE_THRESHOLD)

    def get_position_scale(self, symbol: str) -> float:
        """Kelly-inspired pozisyon skalası [0.6× – 1.5×]."""
        wr    = self._win_rates.get(symbol, 0.50)
        kelly = 2 * wr - 1
        return float(np.clip(0.6 + kelly * 0.9, 0.6, 1.5))

    # ── Gösterge Hesaplama ────────────────────────────────────────────────────

    # Backtest hız optimizasyonu için önceden hesaplanmış kolonlar
    _PRECOMPUTED_COLS = [
        "ema9", "ema21", "rsi14", "macd", "macd_sig", "macd_hist",
        "vol_sma10", "atr10", "supertrend", "st_dir", "vwap",
        "vol_delta", "bb_upper", "bb_lower", "bb_pct_b",
    ]

    def _calc(self, df: pd.DataFrame) -> pd.DataFrame:
        # Hızlı yol: indikatörler zaten hesaplanmışsa (backtest ön-hesaplama)
        # _calc'ı tekrar çalıştırma — slice üzerinde doğrudan dön.
        if all(c in df.columns for c in self._PRECOMPUTED_COLS):
            return df

        out = df.copy()

        out["ema9"]  = _ema(out["close"], 9)
        out["ema21"] = _ema(out["close"], 21)
        out["rsi14"] = _rsi(out["close"], 14)
        out["macd"], out["macd_sig"], out["macd_hist"] = _macd(out["close"])
        out["vol_sma10"] = out["volume"].rolling(10).mean()
        out["atr10"]     = _atr(out["high"], out["low"], out["close"], 10)

        out["supertrend"], out["st_dir"] = _supertrend(
            out["high"], out["low"], out["close"], period=10, multiplier=3.0
        )
        out["vwap"] = _vwap(out["high"], out["low"], out["close"], out["volume"])

        if "open" in out.columns:
            out["vol_delta"] = _volume_delta_ema(
                out["open"], out["close"], out["volume"], period=10
            )
        else:
            out["vol_delta"] = 0.0

        # BB 20 (eski BB 10)
        mid = out["close"].rolling(20).mean()
        std = out["close"].rolling(20).std(ddof=0)
        out["bb_upper"] = mid + 2.0 * std
        out["bb_lower"] = mid - 2.0 * std
        bb_range        = (out["bb_upper"] - out["bb_lower"]).replace(0, np.nan)
        out["bb_pct_b"] = (out["close"] - out["bb_lower"]) / bb_range

        return out

    # ── Sinyal Üretimi ────────────────────────────────────────────────────────

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Signal:
        """5m veriden scalp sinyali üretir (v2)."""
        if len(df) < 40:
            return self._hold(symbol, df, "Yetersiz veri (<40 bar)")

        df_i = self._calc(df)
        row  = df_i.iloc[-1]
        prev = df_i.iloc[-2]

        for col in ["ema9", "ema21", "rsi14", "macd_hist", "vol_sma10", "atr10", "st_dir"]:
            if pd.isna(row.get(col)):
                return self._hold(symbol, df_i, f"NaN: {col}")

        close     = float(row["close"])
        ema9      = float(row["ema9"])
        ema21     = float(row["ema21"])
        rsi14     = float(row["rsi14"])
        macd_h    = float(row["macd_hist"])
        prev_macd = float(prev["macd_hist"]) if not pd.isna(prev.get("macd_hist")) else 0.0
        vol       = float(row["volume"])
        vol_sma   = float(row["vol_sma10"])
        atr10     = float(row["atr10"])
        st_dir    = int(row["st_dir"])
        vwap      = float(row["vwap"])  if pd.notna(row.get("vwap"))      else float("nan")
        vol_delta = float(row["vol_delta"]) if pd.notna(row.get("vol_delta")) else 0.0

        # FIX: bb_pct_b=0.0 geçerli bir değer — `0.0 or 0.5` bug'ı kaldırıldı
        bb_raw   = row.get("bb_pct_b")
        bb_pct_b = float(bb_raw) if pd.notna(bb_raw) else 0.5

        # ── HARD GATE: Supertrend bullish olmalı ──────────────────────────
        if st_dir != 1:
            return self._hold(symbol, df_i, "Supertrend bearish → HOLD")

        score, detail = self._score(
            close, ema9, ema21, rsi14,
            macd_h, prev_macd,
            vol, vol_sma, bb_pct_b,
            vwap, vol_delta,
        )

        threshold = self.get_threshold(symbol)

        bar_ts = str(df_i.index[-1])
        if self._last_bar.get(symbol) != bar_ts:
            self._last_bar[symbol] = bar_ts
            logger.info(
                f"[Scalp] {symbol} | {close:.4f} "
                f"EMA9/21={ema9:.2f}/{ema21:.2f} | "
                f"RSI14={rsi14:.1f} MACD_H={macd_h:+.5f} "
                f"ST={'🟢' if st_dir == 1 else '🔴'} | "
                f"Skor={score:.3f}/eşik={threshold:.3f} | "
                f"{'✅ BUY' if score >= threshold else '❌ HOLD'}"
            )

        if score >= threshold:
            return Signal(
                symbol=symbol,
                side=Side.BUY,
                reason=f"Scalp v2 skor={score:.3f} (eşik={threshold:.3f}) | {detail}",
                timestamp=datetime.now(timezone.utc),
                price=close,
                confidence_score=round(score, 3),
                atr=atr10,
                adx=0.0,
                rsi=rsi14,
            )

        return self._hold(symbol, df_i, f"Skor={score:.3f} < {threshold:.3f}")

    # ── Bileşik Skor ─────────────────────────────────────────────────────────

    def _score(
        self,
        close: float, ema9: float, ema21: float,
        rsi14: float, macd_h: float, prev_macd_h: float,
        vol: float, vol_sma: float, bb_pct_b: float,
        vwap: float, vol_delta: float,
    ) -> tuple[float, str]:
        """
        3-bileşen ağırlıklı skor (Supertrend gate geçildikten sonra):
          Trend    (0.40) — EMA9/21 gap
          Momentum (0.35) — RSI14 + MACD 3/10/16
          Yapı     (0.25) — VWAP + Volume Delta + Hacim + BB
        """

        # 1. TREND (0.40)
        ema_gap = (ema9 - ema21) / max(abs(ema21), 1e-8)
        if ema9 > ema21:
            trend = min(0.55 + ema_gap * 25.0, 1.0)
        elif ema_gap > -0.001:   # neredeyse kesişim
            trend = 0.35
        else:
            trend = 0.15

        # 2. MOMENTUM (0.35)
        # RSI14
        if 48 <= rsi14 <= 68:
            rsi_score = 0.65 + (rsi14 - 48) / 20.0 * 0.35
        elif 40 <= rsi14 < 48:
            rsi_score = 0.40
        elif rsi14 > 68:
            rsi_score = max(0.0, 1.0 - (rsi14 - 68) / 12.0)
        else:
            rsi_score = 0.0

        # MACD — FIX: prev≈0 bölme overflow koruması
        price_epsilon = 0.0001 * close          # fiyatın %0.01'i
        if macd_h > 0:
            if abs(prev_macd_h) > price_epsilon:
                ratio      = min(macd_h / abs(prev_macd_h), 4.0)   # cap at 4×
                macd_score = (
                    min(0.50 + ratio * 0.12, 1.0)
                    if macd_h >= prev_macd_h * 0.8
                    else 0.45
                )
            else:
                # prev histogram ≈ 0 → nötr/hafif pozitif (eski kod burada 1.0 veriyordu)
                macd_score = 0.55
        elif macd_h > -close * 0.0002:
            macd_score = 0.20
        else:
            macd_score = 0.0

        momentum = rsi_score * 0.55 + macd_score * 0.45

        # 3. PİYASA YAPISI (0.25)
        vol_sma_safe = max(vol_sma, 1e-8)

        # VWAP pozisyonu
        if not (np.isnan(vwap) or vwap <= 0):
            vwap_pct = (close - vwap) / vwap
            if 0.0 <= vwap_pct <= 0.008:
                vwap_score = 0.80
            elif -0.004 <= vwap_pct < 0.0:
                vwap_score = 0.55
            elif vwap_pct > 0.008:
                vwap_score = max(0.0, 0.80 - (vwap_pct - 0.008) * 60.0)
            else:
                vwap_score = 0.10
        else:
            vwap_score = 0.50   # VWAP verisi yok → nötr

        # Volume Delta
        if not np.isnan(vol_delta) and vol_delta > 0:
            vd_score = min(0.60 + abs(vol_delta) / (vol_sma_safe * 0.05) * 0.10, 1.0)
        elif not np.isnan(vol_delta) and vol_delta < 0:
            vd_score = 0.0      # satıcı baskısı → giriş yapma
        else:
            vd_score = 0.40

        # Ham hacim
        vol_ratio = vol / vol_sma_safe
        raw_vol   = (
            min(vol_ratio / 1.5, 1.0)
            if vol_ratio >= self.volume_mult
            else vol_ratio * 0.3
        )

        # BB pozisyon bonusu
        bb_bonus = 0.15 if 0.40 <= bb_pct_b <= 0.85 else 0.0

        structure = min(
            vwap_score * 0.40 + vd_score * 0.30 + raw_vol * 0.20 + bb_bonus,
            1.0,
        )

        score  = trend * 0.40 + momentum * 0.35 + structure * 0.25
        detail = f"trend={trend:.2f} mom={momentum:.2f} struct={structure:.2f}"
        return round(float(score), 4), detail

    # ── Çıkış Kararı ─────────────────────────────────────────────────────────

    def should_exit(
        self,
        symbol: str,
        df: pd.DataFrame,
        entry_price: float,
        open_ts,      # datetime veya pd.Timestamp
        current_ts,   # datetime veya pd.Timestamp
    ) -> tuple[bool, str]:
        """
        TP / SL / Zaman / RSI / Supertrend flip.

        FIX: Global _bar_counter yerine timestamp bazlı bar sayımı.
        5 sembol varken global sayaç 5× hızlı artıyor ve pozisyonlar
        60 dk yerine ~12 dk'da kapanıyordu.
        """
        df_i   = self._calc(df)
        row    = df_i.iloc[-1]
        close  = float(row["close"])
        rsi14  = float(row["rsi14"])  if not pd.isna(row.get("rsi14"))  else 50.0
        st_dir = int(row["st_dir"])   if not pd.isna(row.get("st_dir")) else 1

        # Take Profit
        tp_price = entry_price * (1 + self.profit_target_pct)
        if close >= tp_price:
            return True, f"take_profit @ {close:.4f} (hedef {tp_price:.4f})"

        # Stop Loss
        sl_price = entry_price * (1 - self.stop_loss_pct)
        if close <= sl_price:
            return True, f"stop_loss @ {close:.4f} (stop {sl_price:.4f})"

        # Zaman limiti — FIX: timestamp farkı / 300s
        try:
            open_dt = pd.Timestamp(open_ts)
            curr_dt = pd.Timestamp(current_ts)
            if open_dt.tzinfo is None:
                open_dt = open_dt.tz_localize("UTC")
            if curr_dt.tzinfo is None:
                curr_dt = curr_dt.tz_localize("UTC")
            bars_held = int((curr_dt - open_dt).total_seconds() / 300)
        except Exception:
            bars_held = 0

        if bars_held >= self.max_hold_bars:
            pnl_pct = (close - entry_price) / entry_price * 100
            return True, f"max_hold {bars_held} bar ({pnl_pct:+.2f}%)"

        # RSI aşırı alım (80 — eski 78'den biraz daha esnek)
        if rsi14 > 80:
            return True, f"rsi_overbought RSI14={rsi14:.1f}"

        # Supertrend döndü (EMA flip yerine daha güvenilir çıkış)
        if st_dir == -1:
            return True, f"supertrend_flip close={close:.4f}"

        return False, ""

    # ── Yardımcılar ──────────────────────────────────────────────────────────

    def _hold(self, symbol: str, df: pd.DataFrame, reason: str) -> Signal:
        close = float(df.iloc[-1]["close"]) if not df.empty else 0.0
        return Signal(
            symbol=symbol, side=Side.HOLD, reason=reason,
            timestamp=datetime.now(timezone.utc), price=close,
            confidence_score=0.0,
        )

    # ── Durum Kaydet / Yükle ─────────────────────────────────────────────────

    def save_state(self) -> dict:
        return {
            "thresholds":      dict(self._thresholds),
            "win_rates":       dict(self._win_rates),
            "trade_counts":    dict(self._trade_counts),
            "last_trade_time": dict(self._last_trade_time),
            "saved_version":   "v2",
        }

    def load_state(self, state: dict) -> None:
        if not state:
            return
        version = state.get("saved_version", "v1")
        loaded  = 0

        for sym, wr in state.get("win_rates", {}).items():
            self._win_rates[sym] = float(np.clip(wr, 0.0, 1.0))
        for sym, cnt in state.get("trade_counts", {}).items():
            self._trade_counts[sym] = int(cnt)
            loaded += 1
        for sym, ts in state.get("last_trade_time", {}).items():
            self._last_trade_time[sym] = ts

        if version == "v1":
            # v1 threshold'lar eski 0.65 cap'le hesaplandı — sıfırla (win_rates korunuyor)
            logger.info(
                "[ScalpLearn] v1 state algılandı → "
                "threshold'lar sıfırlanıyor (win_rates & trade_counts korunuyor)"
            )
            self._thresholds.clear()
        else:
            for sym, thr in state.get("thresholds", {}).items():
                self._thresholds[sym] = float(
                    np.clip(thr, self.THRESHOLD_FLOOR, self.THRESHOLD_CAP)
                )

        logger.info(
            f"[ScalpLearn] Durum yüklendi: {loaded} sembol (version={version}) | "
            + " | ".join(
                f"{s}: WR={self._win_rates.get(s, 0.5):.1%} "
                f"eşik={self._thresholds.get(s, self.BASE_THRESHOLD):.3f}"
                for s in list(self._trade_counts)[:5]
            )
        )

    @property
    def learning_stats(self) -> dict:
        all_syms = set(
            list(self._thresholds)
            + list(self._win_rates)
            + list(self._trade_counts)
        )
        return {
            sym: {
                "threshold":  round(self._thresholds.get(sym, self.BASE_THRESHOLD), 3),
                "win_rate":   round(self._win_rates.get(sym, 0.5), 3),
                "trades":     self._trade_counts.get(sym, 0),
                "pos_scale":  round(self.get_position_scale(sym), 2),
                "last_trade": self._last_trade_time.get(sym, "—"),
            }
            for sym in all_syms
        }
