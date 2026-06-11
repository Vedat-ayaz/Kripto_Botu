#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EPİSOD SİMÜLATÖRÜ — üretim-sadık tahsisçi değerlendirmesi
==========================================================
Günlük eğri-karıştırma yaklaşımının iki gerçekdışılığını düzeltir:
1. TAZE SERMAYE: her aktivasyon episodunda M9 sıfır durumla koşar (gerçek tahsisçi
   sleeve'e NAV yüzdesi verir; önceki ayların DD-valisi izi taşınmaz)
2. POZİSYON SÜREKLİLİĞİ: gün-bazlı aç/kapa yerine histerezisli episodlar —
   AÇILIŞ: 2 ardışık sinyal günü (teyit kuralı: tek-gün episodlar zehirli, ajan kanıtı)
   KAPANIŞ: 3 ardışık sinyalsiz gün (kısa boşluklarda pozisyonlar taşınır)
   Episod sonunda M9 pozisyonları kapatılır (EOT), sermaye tabana döner.

Kullanım: python3 episod_sim.py [--base m7] [--start 2025-08-01] [--end 2026-06-09]
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd

import m9_backtest as m9
import allocator as al

ON_CONFIRM = 2    # bu kadar ardışık sinyal günü → episod açılır
OFF_CONFIRM = 3   # bu kadar ardışık sinyalsiz gün → episod kapanır


def find_episodes(sig: pd.Series, start: str, end: str):
    """Histerezisli episodlar: [(ilk_gün, son_gün)] — karar günleri sinyalin kendisi
    zaten shift'li olduğu için ek kaydırma gerekmez."""
    days = sig[(sig.index >= pd.Timestamp(start, tz="UTC"))
               & (sig.index <= pd.Timestamp(end, tz="UTC"))]
    episodes = []
    state = False
    on_run = off_run = 0
    ep_start = None
    for ts, on in days.items():
        if not state:
            on_run = on_run + 1 if on else 0
            if on_run >= ON_CONFIRM:
                state = True
                ep_start = ts - pd.Timedelta(days=0)  # teyit günü episodun ilk tahsis günü
                off_run = 0
        else:
            off_run = off_run + 1 if not on else 0
            if off_run >= OFF_CONFIRM:
                episodes.append((ep_start, ts - pd.Timedelta(days=OFF_CONFIRM)))
                state = False
                on_run = 0
    if state:
        episodes.append((ep_start, days.index[-1]))
    return episodes


def run_episode_m9(ep_start: pd.Timestamp, ep_end: pd.Timestamp) -> pd.Series:
    """Episodu taze sermayeyle koşar, günlük getiri serisi döner (gün-0 çapalı)."""
    s = ep_start.strftime("%Y-%m-%d")
    e = ep_end.strftime("%Y-%m-%d")
    res = m9.run_backtest(s, e, verbose=False)
    eq = pd.Series({ts: v for ts, v in res["equity_curve"]}).sort_index()
    d = eq.resample("1D").last().dropna()
    anchor = pd.Series([res["capital"]], index=[d.index[0] - pd.Timedelta(days=1)])
    d = pd.concat([anchor, d])
    return d.pct_change().dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="m7", choices=["m5", "m7"])
    ap.add_argument("--start", default="2025-08-01")
    ap.add_argument("--end", default="2026-06-09")
    args = ap.parse_args()

    print("Sinyal hesaplanıyor...")
    sig = al.build_signal()
    episodes = find_episodes(sig, args.start, args.end)
    print(f"  {len(episodes)} episod bulundu (AÇ={ON_CONFIRM} ardışık gün, KAPAT={OFF_CONFIRM} sinyalsiz gün):")
    for a, b in episodes:
        print(f"    {a.date()} → {b.date()}  ({(b-a).days + 1} gün)")

    base_path = os.path.join(al.M5_DIR, f"{args.base}_eq_full.csv")
    rb = al.daily_rets(base_path)
    rb = rb[(rb.index >= pd.Timestamp(args.start, tz='UTC'))
            & (rb.index <= pd.Timestamp(args.end, tz='UTC') + pd.Timedelta(days=1))]

    # portföy: tabandan başla, episod günlerinde M9-episod getirisi
    r_port = rb.copy()
    n_ep_days = 0
    ep_results = []
    for a, b in episodes:
        r_ep = run_episode_m9(a, b)
        tot = (1 + r_ep).prod() - 1
        ep_results.append((a, b, 100 * tot))
        print(f"    episod {a.date()}→{b.date()}: M9 taze {100*tot:+.2f}%")
        common = r_ep.index.intersection(r_port.index)
        r_port.loc[common] = r_ep.loc[common]
        n_ep_days += len(common)

    mP = al.metrics(r_port)
    mB = al.metrics(rb)
    print(f"\n{'═'*60}")
    print(f"  EPİSOD SİMÜLASYONU  |  {args.start} → {args.end}  |  taban={args.base.upper()}")
    print(f"{'═'*60}")
    print(f"  {args.base.upper():12s}: {mB['total']:+8.2f}%   DD %{mB['mdd']:5.1f}   Sharpe {mB['sharpe']:6.2f}")
    print(f"  ORTAKLIK-EP : {mP['total']:+8.2f}%   DD %{mP['mdd']:5.1f}   Sharpe {mP['sharpe']:6.2f}")
    print(f"  M9-episod günü: {n_ep_days} ({100*n_ep_days/len(r_port):.0f}%)  |  episod sayısı: {len(episodes)}")
    pos_eps = sum(1 for _, _, t in ep_results if t > 0)
    print(f"  kârlı episod: {pos_eps}/{len(ep_results)}")


if __name__ == "__main__":
    main()
