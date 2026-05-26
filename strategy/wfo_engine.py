"""
Walk-Forward Optimization (WFO) Engine
=======================================
Her coin için önceki lookback_days günlük veriyle mini-simülasyon çalıştırır.
20 parametre kombinasyonunu test edip en yüksek skoru döndürür.

Skor fonksiyonu: profit_factor × win_rate × min(1, n_trades/8)
Minimum 4 işlem şartı.

Kullanım:
    optimizer = WalkForwardOptimizer(lookback_days=200)
    params = optimizer.optimize("BTC/USDT", df_with_indicators)
    # → {'adx_threshold': 25, 'entry_score_trend': 0.63, ...}

    all_params = optimizer.optimize_all(indicators_dict)
    # → {'BTC/USDT': {...}, 'ETH/USDT': {...}, ...}
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from indicators.technical_indicators import TechnicalIndicators
from strategy.trend_following_strategy import TrendFollowingStrategy
from strategy.signal import Side

logger = logging.getLogger(__name__)

# ── Parametre Izgarası ─────────────────────────────────────────────────────
# Her tuple: (adx_threshold, entry_score_trend, entry_score_ranging,
#              trailing_stop_atr_multiplier, atr_stop_multiplier)
# Aralıklar: ADX 20-32, entry_trend 0.58-0.72, entry_ranging 0.63-0.77,
#            trailing 3.0-7.5, stop 1.5-3.5
PARAM_GRID: list[tuple[int, float, float, float, float, float]] = [
    (20, 0.58, 0.63, 3.5, 2.0, 61.8),
    (20, 0.62, 0.67, 5.0, 2.5, 61.8),
    (22, 0.60, 0.65, 4.0, 2.0, 61.8),
    (22, 0.63, 0.68, 5.5, 2.5, 61.8),
    (25, 0.60, 0.65, 4.5, 2.0, 61.8),
    (25, 0.63, 0.68, 5.5, 2.5, 61.8),
    (25, 0.67, 0.72, 5.0, 2.5, 56.0),
    (25, 0.70, 0.75, 4.5, 2.5, 56.0),
    (27, 0.63, 0.68, 5.0, 2.5, 61.8),
    (27, 0.67, 0.72, 5.5, 2.5, 56.0),
    (27, 0.70, 0.75, 6.0, 3.0, 56.0),
    (30, 0.65, 0.70, 5.5, 2.5, 56.0),
    (30, 0.68, 0.73, 6.0, 3.0, 56.0),
    (30, 0.72, 0.77, 5.5, 3.0, 52.0),
    (32, 0.68, 0.73, 5.5, 3.0, 56.0),
    (32, 0.72, 0.77, 6.0, 3.5, 52.0),
    (25, 0.60, 0.65, 7.5, 2.5, 61.8),
    (27, 0.65, 0.70, 7.5, 3.0, 56.0),
    (25, 0.60, 0.65, 3.0, 1.5, 61.8),
    (28, 0.66, 0.71, 5.5, 2.8, 61.8),
]

# Göstergelerin stabilize olması için beklenecek bar sayısı
# EMA-200 için en az 200 bar + biraz tampon = 250 bar
WARMUP_BARS: int = 250


def _calc_wfo_score(pnls: list[float]) -> tuple[float, float, float, int]:
    """
    Calculates the WFO composite score from a list of trade PnL percentages.

    Score formula: profit_factor × win_rate × min(1.0, n_trades / 8)

    The trade-count multiplier ensures that results from only 1-2 lucky trades
    don't dominate; scores become fully reliable after 8+ trades.

    Args:
        pnls: List of per-trade PnL percentages (e.g. 0.03 = +3%).

    Returns:
        Tuple of (score, profit_factor, win_rate, n_trades).
        Returns (0.0, 0.0, 0.0, len(pnls)) if there are no winning trades.
    """
    n = len(pnls)
    if n == 0:
        return 0.0, 0.0, 0.0, 0

    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / n

    gross_profit = sum(wins)   if wins   else 0.0
    gross_loss   = abs(sum(losses)) if losses else 0.0

    # Kâr faktörü: toplam kazanç / toplam kayıp
    # Yalnızca kazananlar varsa sonsuz yerine 3.0 kapağı uygula
    if gross_loss > 0:
        profit_factor = min(gross_profit / gross_loss, 5.0)
    elif gross_profit > 0:
        profit_factor = 3.0   # 100% win rate üst kapak
    else:
        return 0.0, 0.0, 0.0, n

    # Trade sayısı ceza çarpanı: 4 trade minimum etkin, 8'de tam puan
    trade_weight = min(1.0, n / 8.0)

    score = profit_factor * win_rate * trade_weight
    return (
        round(float(score), 4),
        round(float(profit_factor), 4),
        round(float(win_rate), 4),
        n,
    )


class WalkForwardOptimizer:
    """
    Walk-Forward Optimizer for TrendFollowingStrategy parameters.

    For each coin it:
      1. Takes the last ``lookback_days`` days of indicator data.
      2. Runs all 20 PARAM_GRID combinations through a lightweight
         bar-by-bar simulation (no should_exit — trailing + hard stop only).
      3. Picks the combination with the highest WFO score.
      4. Returns the best parameter dict enriched with diagnostics keys.

    The optimizer is intentionally stateless between coins so it can be used
    in a fresh instance or called multiple times safely.
    """

    def __init__(self, lookback_days: int = 200, min_trades: int = 4) -> None:
        """
        Args:
            lookback_days: In-sample window length in days.
                           Her coin için kaç günlük in-sample verisi kullanılır.
            min_trades   : Bir kombinasyonun geçerli sayılması için gereken
                           minimum işlem sayısı.
        """
        self.lookback_days = lookback_days
        self.min_trades    = min_trades
        # Paylaşılan TechnicalIndicators örneği (strateji oluşturulurken kullanılır)
        self._indicators   = TechnicalIndicators()

    # ──────────────────────────────────────────────────────────────────
    # Mini simülasyon
    # ──────────────────────────────────────────────────────────────────

    def _run_mini_sim(
        self,
        sym: str,
        df: pd.DataFrame,
        adx_thresh: int,
        entry_trend: float,
        entry_ranging: float,
        trail_mult: float,
        stop_mult: float,
        chop_thresh: float = 61.8,
    ) -> list[float]:
        """
        Runs a lightweight bar-by-bar backtest on ``df`` with the given params.

        Entry: signal from TrendFollowingStrategy.generate_signal()
        Exit : trailing ATR stop OR hard ATR stop (no should_exit for speed)

        Args:
            sym          : Coin symbol (for logging only).
            df           : Indicator DataFrame (in-sample slice, WARMUP_BARS already included).
            adx_thresh   : ADX threshold for strategy.
            entry_trend  : entry_score_trend for strategy.
            entry_ranging: entry_score_ranging for strategy.
            trail_mult   : ATR multiplier for trailing stop.
            stop_mult    : ATR multiplier for hard stop.
            chop_thresh  : Choppiness index threshold for strategy.

        Returns:
            List of per-trade PnL percentages. Empty list on failure or < WARMUP_BARS.
        """
        try:
            # Her kombinasyon için yeni, temiz bir strateji örneği
            strategy = TrendFollowingStrategy(
                adx_threshold=adx_thresh,
                entry_score_trend=entry_trend,
                entry_score_ranging=entry_ranging,
                choppiness_threshold=chop_thresh,
                indicators=self._indicators,
            )
        except Exception as exc:
            logger.debug(f"[WFO] {sym}: Strateji oluşturma hatası — {exc}")
            return []

        pnls: list[float] = []

        # Pozisyon durumu
        in_position   = False
        is_long       = False          # True → LONG, False → SHORT
        entry_price   = 0.0
        stop_price    = 0.0
        trail_ref     = 0.0            # Trailing stop referans seviyesi

        n_bars = len(df)

        # WARMUP_BARS'tan itibaren tarama başlar
        for i in range(WARMUP_BARS, n_bars):
            # Strateji her bar için mevcut slice'ı görür (gerçekçi ilerleyen pencere)
            slice_df = df.iloc[: i + 1]
            row      = df.iloc[i]

            # Gerekli değerleri güvenli şekilde çek
            try:
                close = float(row["close"])
                atr   = float(row["atr"])
            except (KeyError, TypeError, ValueError):
                # Kritik sütun eksik veya NaN → bu barı atla
                continue

            if not np.isfinite(close) or not np.isfinite(atr) or close <= 0 or atr <= 0:
                continue

            # ── Pozisyondayken çıkış kontrolü ─────────────────────
            if in_position:
                exit_price: Optional[float] = None
                exit_reason = ""

                if is_long:
                    # Trailing güncelle: en yüksek kapanışa göre stop'u yükselt
                    new_trail = close - atr * trail_mult
                    trail_ref  = max(trail_ref, new_trail)

                    if close <= trail_ref:
                        exit_price  = close
                        exit_reason = "trailing_stop"
                    elif close <= stop_price:
                        exit_price  = stop_price   # hard stop → daha kötü fiyat
                        exit_reason = "hard_stop"
                else:
                    # SHORT: trailing'i en düşük kapanışa göre aşağıya çek
                    new_trail = close + atr * trail_mult
                    trail_ref  = min(trail_ref, new_trail)

                    if close >= trail_ref:
                        exit_price  = close
                        exit_reason = "trailing_stop"
                    elif close >= stop_price:
                        exit_price  = stop_price
                        exit_reason = "hard_stop"

                if exit_price is not None:
                    # PnL hesapla (yüzde bazlı)
                    if is_long:
                        pnl = (exit_price - entry_price) / entry_price
                    else:
                        pnl = (entry_price - exit_price) / entry_price

                    pnls.append(pnl)
                    in_position = False

                # Hâlâ pozisyondaysa sinyal üretmeye gerek yok
                continue

            # ── Pozisyon yokken giriş sinyali ara ─────────────────
            try:
                signal = strategy.generate_signal(sym, slice_df, allow_short=True)
            except Exception as exc:
                logger.debug(f"[WFO] {sym} bar={i}: generate_signal hatası — {exc}")
                continue

            if signal.side in (Side.BUY, Side.SHORT):
                in_position = True
                is_long     = (signal.side == Side.BUY)
                entry_price = close

                if is_long:
                    stop_price = entry_price - atr * stop_mult
                    trail_ref  = entry_price - atr * trail_mult
                else:
                    stop_price = entry_price + atr * stop_mult
                    trail_ref  = entry_price + atr * trail_mult

        # Simülasyon sonunda açık pozisyonu son kapanışla kapat
        if in_position and n_bars > WARMUP_BARS:
            last_close = float(df.iloc[-1]["close"])
            if np.isfinite(last_close) and last_close > 0:
                if is_long:
                    pnl = (last_close - entry_price) / entry_price
                else:
                    pnl = (entry_price - last_close) / entry_price
                pnls.append(pnl)

        return pnls

    # ──────────────────────────────────────────────────────────────────
    # Tekli coin optimizasyonu
    # ──────────────────────────────────────────────────────────────────

    def optimize(self, sym: str, df: pd.DataFrame) -> Optional[dict]:
        """
        Runs WFO for a single coin and returns the best parameter dict.

        Uses only the last ``lookback_days * 24`` bars from ``df`` as the
        in-sample window.  The full ``df`` is accepted so the caller doesn't
        need to slice in advance.

        Args:
            sym: Coin symbol (e.g. "BTC/USDT").
            df : Full indicator DataFrame (all available history).

        Returns:
            Best parameter dict with extra diagnostic keys:
              _wfo_score, _wfo_pf, _wfo_wr, _wfo_n_trades
            Returns None if there is not enough data or every param combination
            produces fewer than ``min_trades`` trades.
        """
        # v17 FIX: TF-aware bar hesabı (önceden hardcoded * 24 → 1h varsayıyordu)
        # DataFrame index'inden timeframe'i otomatik tespit et
        _tf_bars_per_day = 24  # default 1h
        if len(df) >= 2:
            _delta_min = (df.index[1] - df.index[0]).total_seconds() / 60
            if _delta_min <= 1:      _tf_bars_per_day = 1440  # 1m
            elif _delta_min <= 5:    _tf_bars_per_day = 288   # 5m
            elif _delta_min <= 15:   _tf_bars_per_day = 96    # 15m
            elif _delta_min <= 60:   _tf_bars_per_day = 24    # 1h
        in_sample_bars = self.lookback_days * _tf_bars_per_day
        min_required   = in_sample_bars + WARMUP_BARS

        if len(df) < min_required:
            logger.debug(
                f"[WFO] {sym}: Yetersiz veri ({len(df)} bar < {min_required} gerekli)"
            )
            return None

        # In-sample penceresi: son lookback_days günlük veri
        # WARMUP_BARS dahil → slice'ın başından itibaren strateji ısınabilir
        in_sample_df = df.iloc[-(in_sample_bars + WARMUP_BARS):]

        best_score:  float                = -1.0
        best_params: Optional[dict]       = None
        MIN_PF = 1.05   # In-sample PF bu eşiğin altındaysa kabul etme

        for (adx_t, e_trend, e_ranging, trail, stop, chop_thresh) in PARAM_GRID:
            pnls = self._run_mini_sim(
                sym, in_sample_df, adx_t, e_trend, e_ranging, trail, stop, chop_thresh
            )

            if len(pnls) < self.min_trades:
                # Minimum işlem şartı sağlanmadı → bu kombinasyonu atla
                continue

            score, pf, wr, n = _calc_wfo_score(pnls)

            # PF < MIN_PF olan kombinasyonları reddet — in-sample'da bile kâr
            # edemiyorsa out-of-sample'da kesinlikle işe yaramaz
            if pf < MIN_PF:
                continue

            if score > best_score:
                best_score = score
                best_params = {
                    "adx_threshold":                adx_t,
                    "entry_score_trend":            e_trend,
                    "entry_score_ranging":          e_ranging,
                    "trailing_stop_atr_multiplier": trail,
                    "atr_stop_multiplier":          stop,
                    "choppiness_threshold":         chop_thresh,
                    "_wfo_score":                   score,
                    "_wfo_pf":                      pf,
                    "_wfo_wr":                      wr,
                    "_wfo_n_trades":                n,
                }

        if best_params is None:
            logger.debug(
                f"[WFO] {sym}: Hiçbir kombinasyon {self.min_trades}+ işlem üretemedi"
            )

        return best_params

    # ──────────────────────────────────────────────────────────────────
    # Tüm coinler için optimizasyon
    # ──────────────────────────────────────────────────────────────────

    def optimize_all(
        self, indicators: dict[str, pd.DataFrame]
    ) -> dict[str, dict]:
        """
        Runs WFO for every coin in ``indicators`` and returns a mapping of
        symbol → best parameter dict.

        Coins that fail (not enough data, no valid combinations) are omitted
        from the result with a warning log.

        Args:
            indicators: Dict mapping symbol strings to full indicator DataFrames.
                        E.g. {"BTC/USDT": df_btc, "ETH/USDT": df_eth, ...}

        Returns:
            Dict of {symbol: params_dict}.  Only successful coins are included.
        """
        results: dict[str, dict] = {}
        total  = len(indicators)

        logger.info(
            f"[WFO] Optimizasyon başladı — {total} coin, "
            f"lookback={self.lookback_days} gün, grid={len(PARAM_GRID)} kombinasyon"
        )

        for idx, (sym, df) in enumerate(indicators.items(), start=1):
            logger.debug(f"[WFO] ({idx}/{total}) {sym} işleniyor...")

            params = self.optimize(sym, df)

            if params is None:
                logger.warning(
                    f"[WFO] {sym}: Optimizasyon başarısız — sonuç dışında bırakıldı "
                    f"(veri yetersizliği veya hiçbir kombinasyon {self.min_trades}+ işlem üretemedi)"
                )
                continue

            results[sym] = params

            logger.info(
                f"[WFO] {sym}: best ADX={params['adx_threshold']}, "
                f"entry={params['entry_score_trend']:.2f}, "
                f"score={params['_wfo_score']:.3f} "
                f"({params['_wfo_n_trades']} işlem, "
                f"WR={params['_wfo_wr']:.0%}, "
                f"PF={params['_wfo_pf']:.2f})"
            )

        # ── Özet istatistik ──────────────────────────────────────────
        n_ok   = len(results)
        n_fail = total - n_ok
        if results:
            scores = [v["_wfo_score"] for v in results.values()]
            logger.info(
                f"[WFO] Tamamlandı — {n_ok}/{total} coin başarılı "
                f"(başarısız: {n_fail}), "
                f"ortalama WFO skoru={np.mean(scores):.3f}, "
                f"max={max(scores):.3f}, "
                f"min={min(scores):.3f}"
            )
        else:
            logger.warning(
                f"[WFO] Tamamlandı ama hiçbir coin optimize edilemedi "
                f"({total} coinın tamamı başarısız)"
            )

        return results
