#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARRY SLEEVE — delta-nötr funding hasadı (spot long + perp short)
==================================================================
Piyasa yönünden bağımsız gelir: funding pozitifken perp-short taraf 8 saatte bir
funding tahsil eder. Chop dönemlerinin doğal kazananı — portföyün "taban geliri".

Dürüstlük kuralları:
- Lookahead YOK: t dönemindeki pozisyon, t-1'e kadarki funding'le seçilir
  (öngörücü: son 3 funding ortalaması = son 24 saat)
- Maliyet: pozisyona giriş 0.3% (spot+perp, taker+slip), çıkış 0.3%
- Histerezis: giriş eşiği 0.005%/8h; çıkış ancak 24h-ortalama funding < 0 olunca
  (churn'ü sınırlar)
- Eşit ağırlık top-K (K=5); yetersiz aday varsa kalan sermaye nakit (0 getiri)
- Basis kayması ihmal edilir (vadesiz perp'te funding mekanizması basis'i bağlar;
  standart carry-backtest varsayımı)

Çıktı: sonuclar/_alloc/carry_eq_full.csv (günlük equity) + istatistik raporu
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd

import m9_backtest as m9

K_MAX = 5            # aynı anda en çok kaç coin
ENTRY_THR = 0.00005  # giriş: 24h ort funding ≥ %0.005/8h (~%5.5 APY)
COST_SIDE = 0.003    # giriş VEYA çıkış maliyeti (spot+perp iki bacak toplam)
START = "2025-08-01"
END = "2026-06-09"


def load_funding():
    lo = pd.Timestamp("2025-06-27", tz="UTC")
    hi = pd.Timestamp("2026-06-10", tz="UTC")
    cols = {}
    for sym in m9.UNIVERSE:
        try:
            df = m9.fetch_funding(sym, lo, hi)
            if not df.empty:
                cols[sym] = df["funding"]
        except Exception:
            continue
    F = pd.DataFrame(cols).sort_index()
    # 8h ızgaraya hizala (bazı kayıtlar saniye kaymalı olabilir)
    F.index = F.index.round("1h")
    return F


def simulate(F: pd.DataFrame):
    pred = F.rolling(3, min_periods=2).mean().shift(1)   # t için öngörü: t-1'e kadar son 24h
    periods = F.index[(F.index >= pd.Timestamp(START, tz="UTC"))
                      & (F.index <= pd.Timestamp(END, tz="UTC") + pd.Timedelta(days=1))]
    held: set = set()
    equity = 1.0
    rows = []
    n_entries = n_exits = 0
    for ts in periods:
        p = pred.loc[ts] if ts in pred.index else None
        if p is None:
            continue
        # çıkışlar: 24h-ort funding negatife döndü
        for sym in list(held):
            if not np.isnan(p.get(sym, np.nan)) and p[sym] < 0:
                held.discard(sym)
                equity *= (1 - COST_SIDE / K_MAX)   # sleeve'in 1/K payı çıkış maliyeti öder
                n_exits += 1
        # girişler: eşik üstü en iyi adaylar, boş slot kadar
        cands = p[p >= ENTRY_THR].dropna().sort_values(ascending=False)
        for sym in cands.index:
            if len(held) >= K_MAX:
                break
            if sym not in held:
                held.add(sym)
                equity *= (1 - COST_SIDE / K_MAX)
                n_entries += 1
        # bu dönem gerçekleşen funding tahsilatı (perp-short alır; negatifse öder)
        if held:
            f_real = F.loc[ts, list(held)].dropna()
            period_ret = f_real.sum() / K_MAX     # sermayenin 1/K'sı her coinde
            equity *= (1 + period_ret)
        rows.append((ts, equity, len(held)))
    eq = pd.DataFrame(rows, columns=["ts", "equity", "n_pos"]).set_index("ts")
    return eq, n_entries, n_exits


def main():
    print("Funding verisi yükleniyor...")
    F = load_funding()
    print(f"  {F.shape[1]} coin, {len(F)} funding dönemi")
    eq, ne, nx = simulate(F)
    d = eq["equity"].resample("1D").last().dropna()
    anchor = pd.Series([1.0], index=[d.index[0] - pd.Timedelta(days=1)])
    rets = pd.concat([anchor, d]).pct_change().dropna()

    out = os.path.join(_HERE, "sonuclar", "_alloc", "carry_eq_full.csv")
    (d * 10_000).rename("equity").rename_axis("ts").to_csv(out)

    total = d.iloc[-1] - 1
    days = len(rets)
    ann = (1 + total) ** (365 / days) - 1
    sd = rets.std()
    sharpe = rets.mean() / sd * np.sqrt(365) if sd > 0 else float("nan")
    peak, mdd = 1.0, 0.0
    for v in d:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak)
    pozg = 100 * (rets >= 0).mean()
    print(f"\nCARRY SLEEVE  {START} → {END}  ({days} gün)")
    print(f"  Getiri: {100*total:+.2f}%   Yıllık: {100*ann:+.2f}%   MaxDD: %{100*mdd:.2f}")
    print(f"  Sharpe: {sharpe:.2f}   Pozitif gün: %{pozg:.0f}   giriş/çıkış: {ne}/{nx}")
    print(f"  Ortalama pozisyon sayısı: {eq['n_pos'].mean():.1f}/{K_MAX}")
    print(f"  📁 {out}")


if __name__ == "__main__":
    main()
