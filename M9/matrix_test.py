#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Konfigürasyon matrisi: her iyileştirmeyi izole test et.
Taban = v8 (5m, BO+PB, tam boy, v6 piramidi, 1h/3h cooldown, fren/thrust yok).
"""
import sys
import m9_backtest as m9

WINDOWS = [
    ("SON13G", "2026-05-27", "2026-06-09"),
    ("AYI",    "2026-01-29", "2026-02-05"),
    ("BOGA",   "2025-09-27", "2025-10-04"),
    ("6AY",    "2025-08-01", "2026-01-31"),
]

PYR_V6 = [
    dict(at_r=1.0, size_mult=0.6, stop_to_r=0.1),
    dict(at_r=2.2, size_mult=0.4, stop_to_r=1.0),
]

def baseline_v8(tf="5m"):
    """v8 konfigürasyonunu kur."""
    m9.set_tf(tf)
    m9.ENABLE_CS = False
    m9.ENABLE_BO = True
    m9.ENABLE_PB = True
    m9.ENABLE_SQ = False
    m9.ENABLE_MR = False
    m9.INITIAL_SIZE_FRAC = 1.0
    m9.PYR_LEVELS = list(PYR_V6)
    m9.COOLDOWN_EXIT = 1 * m9.K
    m9.COOLDOWN_LOSS = 3 * m9.K
    m9.THRUST_ON = False
    m9.THRUST_DIRECTIONAL = False
    m9.BRAKES_ON = False
    m9.MACRO_ON = False

CONFIGS = {
    "A_v8_taban":      lambda: baseline_v8("5m"),
    "B_v8+thrust":     lambda: (baseline_v8("5m"), setattr(m9, "THRUST_ON", True)),
    "C_v8+frenler":    lambda: (baseline_v8("5m"), setattr(m9, "BRAKES_ON", True),
                                setattr(m9, "COOLDOWN_LOSS", 36 * m9.K)),
    "D_v8+thr+fren":   lambda: (baseline_v8("5m"), setattr(m9, "THRUST_ON", True),
                                setattr(m9, "BRAKES_ON", True),
                                setattr(m9, "COOLDOWN_LOSS", 36 * m9.K)),
    "E_D+starter":     lambda: (baseline_v8("5m"), setattr(m9, "THRUST_ON", True),
                                setattr(m9, "BRAKES_ON", True),
                                setattr(m9, "COOLDOWN_LOSS", 36 * m9.K),
                                setattr(m9, "INITIAL_SIZE_FRAC", 0.5)),
    "F_D+yonluThrust": lambda: (baseline_v8("5m"), setattr(m9, "THRUST_ON", True),
                                setattr(m9, "THRUST_DIRECTIONAL", True),
                                setattr(m9, "BRAKES_ON", True),
                                setattr(m9, "COOLDOWN_LOSS", 36 * m9.K)),
    "G_F+sikiXsec":    lambda: (baseline_v8("5m"), setattr(m9, "THRUST_ON", True),
                                setattr(m9, "THRUST_DIRECTIONAL", True),
                                setattr(m9, "BRAKES_ON", True),
                                setattr(m9, "COOLDOWN_LOSS", 36 * m9.K),
                                setattr(m9, "XSEC_LONG_TOP", 6),
                                setattr(m9, "XSEC_SHORT_BOT", 6)),
    "H_D+makro":       lambda: (baseline_v8("5m"), setattr(m9, "THRUST_ON", True),
                                setattr(m9, "BRAKES_ON", True),
                                setattr(m9, "COOLDOWN_LOSS", 36 * m9.K),
                                setattr(m9, "MACRO_ON", True)),
}

def run_one(name, cfg_fn, windows):
    cfg_fn()
    row = {"config": name}
    for wname, s, e in windows:
        res = m9.run_backtest(s, e, verbose=False)
        cap = res["capital"]
        fin = res["final_balance"]
        trades = res["trades"]
        wins = sum(1 for t in trades if t.pnl > 0)
        gw = sum(t.pnl for t in trades if t.pnl > 0)
        gl = abs(sum(t.pnl for t in trades if t.pnl <= 0))
        pf = gw / gl if gl > 0 else float("inf")
        peak, mdd = -1e18, 0.0
        for _, eq in res["equity_curve"]:
            peak = max(peak, eq)
            mdd = max(mdd, (peak - eq) / peak if peak > 0 else 0)
        row[wname] = f"{100*(fin-cap)/cap:+.2f}% n={len(trades)} pf={pf:.2f} dd={100*mdd:.1f}"
    return row

if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, fn in CONFIGS.items():
        if only and only not in name:
            continue
        row = run_one(name, fn, WINDOWS)
        print(f"\n### {row['config']}")
        for wname, _, _ in WINDOWS:
            print(f"  {wname:8s}: {row[wname]}")
        sys.stdout.flush()
