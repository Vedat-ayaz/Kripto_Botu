#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M9 / M5 / M7 / ORTAKLIK — istediğin tarih aralığında test aracı
================================================================
Kullanım:
    python3 dene.py --start 2026-03-01 --end 2026-04-15                 # M9 (varsayılan)
    python3 dene.py --start 2026-03-01 --end 2026-04-15 --mode m5      # M5 legacy
    python3 dene.py --start 2026-03-01 --end 2026-04-15 --mode m7      # M7 legacy
    python3 dene.py --start 2026-03-01 --end 2026-04-15 --mode ortaklik [--base m7]
    python3 dene.py --start ... --end ... --trades                     # işlem listesiyle

Not: yeni tarih aralığında ilk koşu veriyi Binance'ten indirir (birkaç dk);
sonraki koşular cache'ten okur (M9/.cache ve ana repo .ohlcv_cache).
"""
import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd

import m9_backtest as m9
import allocator as al


def _outdir(start: str, end: str) -> str:
    d = os.path.join(_HERE, "sonuclar", f"{start}_{end}")
    os.makedirs(d, exist_ok=True)
    return d


def run_m9(start: str, end: str, show_trades: bool = False, dump: str = ""):
    from dataclasses import asdict
    res = m9.run_backtest(start, end, verbose=True)
    m9.report(res, start, end, show_trades=show_trades)
    out = _outdir(start, end)
    tpath = os.path.join(out, "m9_islemler.csv")
    epath = os.path.join(out, "m9_equity.csv")
    pd.DataFrame([asdict(t) for t in res["trades"]]).to_csv(tpath, index=False)
    pd.DataFrame(res["equity_curve"], columns=["ts", "equity"]).to_csv(epath, index=False)
    print(f"\n  📁 İşlem listesi : {tpath}")
    print(f"  📁 Equity eğrisi : {epath}")
    if dump:
        pd.DataFrame(res["equity_curve"], columns=["ts", "equity"]).to_csv(dump, index=False)
    return res


def run_legacy(model: str, start: str, end: str, dump: str = "", quiet: bool = False):
    """M5/M7 legacy backtest'i alt süreçte koşar (--universe, ana repo varsayılanları)."""
    env = dict(os.environ)
    if dump:
        env["EQUITY_DUMP"] = dump
    cmd = [sys.executable, os.path.join(_ROOT, "crypto_portfolio_test.py"),
           f"--{model}", "--universe", "--start", start, "--end", end]
    if quiet:
        out = subprocess.run(cmd, env=env, cwd=_ROOT, capture_output=True, text=True)
        for line in out.stdout.splitlines():
            if any(k in line for k in ("Bitiş Sermaye", "Kazanma Oranı", "Max Düşüş", "equity eğrisi")):
                print(" ", line.strip())
    else:
        subprocess.run(cmd, env=env, cwd=_ROOT)


def run_ortaklik(start: str, end: str, base: str):
    out = _outdir(start, end)
    m9_csv = os.path.join(out, "m9_equity.csv")
    base_csv = os.path.join(out, f"{base}_equity.csv")

    print(f"━━ 1/3: M9 koşuluyor — sleeve modu: DD valisi KAPALI ({start} → {end}) ━━")
    # v24 FİNAL: sleeve mimarisinde risk yönetimi tahsisçide (T2 + sağlık kapısı);
    # model-içi vali Haziran-tipi rallileri küçük boyla geçiriyordu (+10.2 → -5.3 fark)
    m9.GOV_STICKY = False
    m9.DD_STAGES = []
    run_m9(start, end, dump=m9_csv)

    print(f"\n━━ 2/3: {base.upper()} koşuluyor (legacy, özet) ━━")
    run_legacy(base, start, end, dump=base_csv, quiet=True)

    print("\n━━ 3/3: Rejim sinyali + tahsis ━━")
    sig_start = (pd.Timestamp(start) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    sig = al.build_signal(start=sig_start, end=end)

    r9 = al.daily_rets(m9_csv)
    rb = al.daily_rets(base_csv)
    idx = rb.index.union(r9.index)
    rbx = rb.reindex(idx).fillna(0.0)
    r9x = r9.reindex(idx).fillna(0.0)
    w_sig = sig.reindex(idx).fillna(False)
    eq9 = (1 + r9x).cumprod()
    roll14 = (eq9 / eq9.shift(14) - 1).fillna(0.0).shift(1).fillna(0.0)
    # v23 FİNAL — T2 (esnek teyit): bugün açık VE son 2 günün en az birinde açıktı.
    # İzole tek-gün sinyalleri bloklar (35 episod -25.2p zehir, ajan kanıtı),
    # episod içi tek-gün titremeyi tolere eder. Split-sample doğrulamalı.
    w_conf = w_sig & (w_sig.shift(1).fillna(False) | w_sig.shift(2).fillna(False))
    wE = (w_conf & (roll14 > -0.02)).astype(float)
    rE = wE * r9x + (1 - wE) * rbx

    # v24: NAKİT-taban varyantı (FİNAL öneri — sessiz günlerde sermaye nakitte/stablecoin'de)
    r_cash = wE * r9x

    print(f"\n{'═'*56}")
    print(f"  ORTAKLIK SONUCU  |  {start} → {end}")
    print(f"{'═'*56}")
    for label, r in ((base.upper(), rbx), ("M9", r9x),
                     (f"ORT({base.upper()})", rE), ("ORT(NAKİT)★", r_cash)):
        m = al.metrics(r)
        print(f"  {label:12s}: {m['total']:+8.2f}%   MaxDD %{m['mdd']:5.1f}   Sharpe {m['sharpe']:6.2f}")
    print(f"  M9-aktif gün: %{100*wE.mean():.0f}  (T2 teyit + 14g sağlık; ★=önerilen final)")
    if len(idx) < 21:
        print("  ⚠ Kısa aralık: 14 günlük sağlık kapısı yeterli geçmiş bulamaz — kapı nötr başlar.")
    # ortaklık günlük eğrisini de kaydet
    epath = os.path.join(out, "ortaklik_equity.csv")
    ((1 + rE).cumprod() * 10_000).rename("equity").rename_axis("ts").to_csv(epath)
    print(f"\n  📁 Çıktı klasörü : {out}")
    print(f"     m9_islemler.csv | m9_equity.csv | {base}_equity.csv | ortaklik_equity.csv")


def main():
    ap = argparse.ArgumentParser(description="M9/M5/M7/Ortaklık tarih aralığı testi")
    ap.add_argument("--start", required=True, help="YYYY-AA-GG")
    ap.add_argument("--end", required=True, help="YYYY-AA-GG")
    ap.add_argument("--mode", default="m9", choices=["m9", "m5", "m7", "ortaklik"])
    ap.add_argument("--base", default="m7", choices=["m5", "m7"], help="ortaklık taban modeli")
    ap.add_argument("--trades", action="store_true", help="M9 işlem listesini yaz")
    args = ap.parse_args()

    if args.mode == "m9":
        run_m9(args.start, args.end, show_trades=args.trades)
    elif args.mode in ("m5", "m7"):
        run_legacy(args.mode, args.start, args.end)
    else:
        run_ortaklik(args.start, args.end, args.base)


if __name__ == "__main__":
    main()
