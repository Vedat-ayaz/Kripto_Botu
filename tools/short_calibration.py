"""
SHORT Kalibrasyon Analizi
==========================
Mevcut SHORT parametrelerinin performansını ölçer,
sonra farklı parametre setleri dener ve en iyisini bulur.

Çalıştır:
    python tools/short_calibration.py
"""
from __future__ import annotations

import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from crypto_portfolio_test import (
    fetch_ohlcv, prepare_indicators, make_strategy,
    SYMBOLS, UNIVERSE,
    COMMISSION, SLIPPAGE, RISK_PER_TRADE, ATR_STOP_MULT, TRAILING_MULT,
    TechnicalIndicators,
)
from strategy.trend_following_strategy import Side, TrendFollowingStrategy

# ── Konfigürasyon ──────────────────────────────────────────────────────────────
TIMEFRAME    = "15m"     # M5 ile uyumlu
DAYS         = 180       # 6 aylık geçmiş veri
TEST_SYMBOLS = SYMBOLS   # 9 coin
CAPITAL      = 1000.0
BARS_PER_DAY = 96        # 15m → 96 bar/gün

# ── Backtest çekirdeği (mini, sadece SHORT için) ───────────────────────────────

@dataclass
class Trade:
    symbol:     str
    side:       str       # "SHORT" veya "LONG"
    entry:      float
    exit:       float
    size:       float
    pnl:        float
    bars_held:  int
    exit_reason: str


def _simulate_short_trades(
    dfs: dict[str, pd.DataFrame],
    short_ema_pct: float         = 0.985,
    short_mom_pct: float         = 0.97,
    short_momentum_lookback: int = 288,    # 3 gün @ 15m
    short_score_trend_thr: float = 0.38,
    short_score_range_thr: float = 0.34,
    short_require_ema_slope: bool = False,
    atr_stop: float              = ATR_STOP_MULT,
    trail_mult: float            = TRAILING_MULT,
    max_hold_bars: int           = 96 * 5,  # max 5 gün tutma
    allow_long: bool             = True,
) -> list[Trade]:
    """Tek geçiş simülasyonu — parametrelerle SHORT (ve opsiyonel LONG) trade sim."""
    trades: list[Trade] = []
    balance   = CAPITAL
    positions: dict[str, dict] = {}   # sym → {entry, stop, trail, size, bars, side}

    all_syms = list(dfs.keys())
    if not all_syms:
        return []

    # Tüm bar indekslerini hizala (en uzun df kullan)
    max_len = max(len(df) for df in dfs.values())

    for bar_i in range(200, max_len):
        for sym, df in dfs.items():
            if bar_i >= len(df):
                continue
            row   = df.iloc[bar_i]
            price = float(row["close"])
            atr   = float(row.get("atr", price * 0.01))
            if atr == 0 or price == 0:
                continue

            # ── Açık pozisyon çıkış kontrolü ──────────────────────────────
            if sym in positions:
                pos   = positions[sym]
                is_sh = pos["side"] == "SHORT"

                # Trailing stop güncelle
                if is_sh:
                    new_trail = price + trail_mult * atr
                    if new_trail < pos["trail"]:
                        pos["trail"] = new_trail
                else:
                    new_trail = price - trail_mult * atr
                    if new_trail > pos["trail"]:
                        pos["trail"] = new_trail

                pos["bars"] += 1

                # Çıkış kontrolleri
                exit_reason = None
                if is_sh:
                    if price >= pos["stop"] or price >= pos["trail"]:
                        exit_reason = "stop"
                    elif pos["bars"] >= max_hold_bars:
                        exit_reason = "timeout"
                else:
                    if price <= pos["stop"] or price <= pos["trail"]:
                        exit_reason = "stop"
                    elif pos["bars"] >= max_hold_bars:
                        exit_reason = "timeout"

                if exit_reason:
                    fill = price * (1 + SLIPPAGE if is_sh else 1 - SLIPPAGE)
                    comm = fill * pos["size"] * COMMISSION
                    if is_sh:
                        pnl = (pos["entry"] - fill) * pos["size"] - comm
                        balance += pos["margin"] + pnl
                    else:
                        pnl = (fill - pos["entry"]) * pos["size"] - comm
                        balance += fill * pos["size"] - comm

                    trades.append(Trade(
                        symbol=sym, side=pos["side"],
                        entry=pos["entry"], exit=fill,
                        size=pos["size"], pnl=pnl,
                        bars_held=pos["bars"], exit_reason=exit_reason,
                    ))
                    del positions[sym]
                continue  # Bir barda hem çıkış hem giriş yok

            # ── Giriş sinyali ──────────────────────────────────────────────
            if len(positions) >= 6:
                continue  # Max 6 pozisyon
            if balance < CAPITAL * 0.05:
                continue

            slice_df = df.iloc[:bar_i]
            if len(slice_df) < 100:
                continue

            # Strateji oluştur (kalibre edilecek parametrelerle)
            try:
                strategy, risk_p = make_strategy(sym, coin_df=slice_df, timeframe=TIMEFRAME)
                # Parametreleri override et
                strategy.short_ema_pct           = short_ema_pct
                strategy.short_mom_pct           = short_mom_pct
                strategy.short_momentum_lookback = short_momentum_lookback
                strategy.short_score_trend_thr   = short_score_trend_thr
                strategy.short_score_range_thr   = short_score_range_thr
                strategy.short_require_ema_slope = short_require_ema_slope

                signal = strategy.generate_signal(sym, slice_df, allow_short=True)
            except Exception:
                continue

            is_sh_sig = signal.side == Side.SHORT
            is_lg_sig = signal.side == Side.BUY

            if not (is_sh_sig or (allow_long and is_lg_sig)):
                continue

            # EMA50 filtresi
            ema_col = next((c for c in ("ema_50", "ema50") if c in slice_df.columns), None)
            if is_sh_sig and ema_col and len(slice_df) >= 2:
                if not (slice_df["close"].tail(2) < slice_df[ema_col].tail(2)).all():
                    continue
            if is_lg_sig and ema_col and len(slice_df) >= 3:
                if not (slice_df["close"].tail(3) > slice_df[ema_col].tail(3)).all():
                    continue

            # 14-gün filtresi (sadece SHORT)
            if is_sh_sig:
                bars_14d = 14 * BARS_PER_DAY
                if len(slice_df) > bars_14d:
                    p14 = float(slice_df.iloc[-bars_14d]["close"])
                    chg = (price / p14) - 1
                    if chg > 0.05 or chg < -0.25:
                        continue

            # Boyut hesabı
            stop_dist = atr_stop * atr
            size = (balance * RISK_PER_TRADE) / stop_dist
            cost = size * price
            if cost > balance * 0.95 or cost < 5.0:
                continue

            if is_sh_sig:
                fill   = price * (1 - SLIPPAGE)
                comm   = fill * size * COMMISSION
                margin = atr_stop * atr * size + comm
                stop   = fill + atr_stop * atr
                trail  = fill + trail_mult * atr
                balance -= margin
                positions[sym] = {
                    "side": "SHORT", "entry": fill, "stop": stop,
                    "trail": trail, "size": size, "bars": 0, "margin": margin,
                }
            else:
                fill   = price * (1 + SLIPPAGE)
                comm   = fill * size * COMMISSION
                total  = fill * size + comm
                if total > balance:
                    continue
                stop   = fill - atr_stop * atr
                trail  = fill - trail_mult * atr
                balance -= total
                positions[sym] = {
                    "side": "LONG", "entry": fill, "stop": stop,
                    "trail": trail, "size": size, "bars": 0, "margin": total,
                }

    return trades


def _report(label: str, trades: list[Trade]) -> dict:
    shorts = [t for t in trades if t.side == "SHORT"]
    longs  = [t for t in trades if t.side == "LONG"]

    def stats(ts: list[Trade]) -> dict:
        if not ts:
            return {"n": 0, "wr": 0, "avg_pnl": 0, "total_pnl": 0, "avg_bars": 0}
        wins = [t for t in ts if t.pnl > 0]
        return {
            "n":         len(ts),
            "wr":        len(wins) / len(ts) * 100,
            "avg_pnl":   np.mean([t.pnl for t in ts]),
            "total_pnl": sum(t.pnl for t in ts),
            "avg_bars":  np.mean([t.bars_held for t in ts]),
        }

    ss = stats(shorts)
    ls = stats(longs)

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  SHORT → {ss['n']:3d} işlem | WR {ss['wr']:5.1f}% | "
          f"Ort PnL ${ss['avg_pnl']:+.2f} | Toplam ${ss['total_pnl']:+.2f} | "
          f"Ort bekleme {ss['avg_bars']:.0f} bar")
    print(f"  LONG  → {ls['n']:3d} işlem | WR {ls['wr']:5.1f}% | "
          f"Ort PnL ${ls['avg_pnl']:+.2f} | Toplam ${ls['total_pnl']:+.2f} | "
          f"Ort bekleme {ls['avg_bars']:.0f} bar")
    return {"short": ss, "long": ls}


# ── Ana kalibrasyon ───────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("  SHORT KALİBRASYON ANALİZİ")
    print(f"  {TIMEFRAME} | {DAYS} gün | {len(TEST_SYMBOLS)} coin")
    print("="*60)

    # 1. Veri çek
    print("\nVeri çekiliyor...")
    indicators = TechnicalIndicators()
    dfs: dict[str, pd.DataFrame] = {}
    for sym in TEST_SYMBOLS:
        try:
            raw = fetch_ohlcv(sym, days=DAYS + 5, timeframe=TIMEFRAME)
            processed = prepare_indicators(raw, indicators, timeframe=TIMEFRAME)
            dfs[sym] = processed
            print(f"  {sym:<14} {len(processed)} bar")
        except Exception as e:
            print(f"  {sym:<14} HATA: {e}")

    if not dfs:
        print("Veri yok, çıkılıyor.")
        return

    # 2. Mevcut parametreler (baseline)
    print("\n\n[1] MEVCUT PARAMETRELER (Baseline)")
    baseline_trades = _simulate_short_trades(
        dfs,
        short_ema_pct=0.985,
        short_mom_pct=0.97,
        short_momentum_lookback=288,
        short_score_trend_thr=0.38,
        short_score_range_thr=0.34,
        short_require_ema_slope=False,
    )
    r_baseline = _report("Baseline", baseline_trades)

    # 3. Farklı parametre setlerini dene
    param_sets = [
        {
            "label": "Gevşek eşik (score 0.28/0.24)",
            "short_score_trend_thr": 0.28,
            "short_score_range_thr": 0.24,
            "short_ema_pct": 0.985,
            "short_mom_pct": 0.97,
            "short_momentum_lookback": 288,
            "short_require_ema_slope": False,
        },
        {
            "label": "Sıkı eşik (score 0.45/0.40)",
            "short_score_trend_thr": 0.45,
            "short_score_range_thr": 0.40,
            "short_ema_pct": 0.985,
            "short_mom_pct": 0.97,
            "short_momentum_lookback": 288,
            "short_require_ema_slope": False,
        },
        {
            "label": "EMA daha uzakta (1.5% → 2.5% altı)",
            "short_score_trend_thr": 0.38,
            "short_score_range_thr": 0.34,
            "short_ema_pct": 0.975,
            "short_mom_pct": 0.97,
            "short_momentum_lookback": 288,
            "short_require_ema_slope": False,
        },
        {
            "label": "Uzun momentum (5 gün)",
            "short_score_trend_thr": 0.38,
            "short_score_range_thr": 0.34,
            "short_ema_pct": 0.985,
            "short_mom_pct": 0.95,
            "short_momentum_lookback": 480,   # 5 gün @ 15m
            "short_require_ema_slope": False,
        },
        {
            "label": "EMA slope gerekli",
            "short_score_trend_thr": 0.38,
            "short_score_range_thr": 0.34,
            "short_ema_pct": 0.985,
            "short_mom_pct": 0.97,
            "short_momentum_lookback": 288,
            "short_require_ema_slope": True,
        },
        {
            "label": "Kombinasyon: sıkı eşik + uzun momentum + slope",
            "short_score_trend_thr": 0.42,
            "short_score_range_thr": 0.37,
            "short_ema_pct": 0.980,
            "short_mom_pct": 0.95,
            "short_momentum_lookback": 480,
            "short_require_ema_slope": True,
        },
        {
            "label": "Agresif SHORT (çok gevşek)",
            "short_score_trend_thr": 0.22,
            "short_score_range_thr": 0.18,
            "short_ema_pct": 0.995,
            "short_mom_pct": 0.99,
            "short_momentum_lookback": 192,
            "short_require_ema_slope": False,
        },
    ]

    results = []
    for i, ps in enumerate(param_sets, 2):
        label = ps.pop("label")
        trades = _simulate_short_trades(dfs, **ps)
        r = _report(f"[{i}] {label}", trades)
        results.append({"label": label, **r, "total": len(trades)})

    # 4. Özet karşılaştırma
    print("\n\n" + "="*60)
    print("  ÖZET KARŞILAŞTIRMA (SHORT performansı)")
    print("="*60)
    print(f"  {'Parametre Seti':<42} {'N':>4} {'WR%':>6} {'Toplam PnL':>11}")
    print(f"  {'─'*42} {'─'*4} {'─'*6} {'─'*11}")

    b_sh = r_baseline["short"]
    print(f"  {'Baseline (mevcut)':<42} {b_sh['n']:>4} {b_sh['wr']:>6.1f}% ${b_sh['total_pnl']:>+9.2f}")
    for r in results:
        sh = r["short"]
        print(f"  {r['label']:<42} {sh['n']:>4} {sh['wr']:>6.1f}% ${sh['total_pnl']:>+9.2f}")

    # En iyi parametre seti (WR × total_pnl)
    best_score = b_sh["wr"] * max(b_sh["total_pnl"], 0.01)
    best_label = "Baseline"
    for r in results:
        sh = r["short"]
        if sh["n"] >= 5:  # En az 5 işlem olsun
            sc = sh["wr"] * max(sh["total_pnl"], 0.01)
            if sc > best_score:
                best_score = sc
                best_label = r["label"]

    print(f"\n  ✅ En iyi parametre seti: {best_label}")
    print()


if __name__ == "__main__":
    main()
