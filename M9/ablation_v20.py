#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v20 düzeltmelerinin kümülatif ablasyonu — 5 pencere.
Hedef: SON13G +20.81 korunmalı, 6AY/-16.66 ve OOS/-9.85 iyileşmeli."""
import sys
import m9_backtest as m9

WINDOWS = [
    ("SON13G", "2026-05-27", "2026-06-09"),
    ("AYI",    "2026-01-29", "2026-02-05"),
    ("BOGA",   "2025-09-27", "2025-10-04"),
    ("6AY",    "2025-08-01", "2026-01-31"),
    ("OOS",    "2026-02-05", "2026-05-26"),
]

def base():
    """v19 final (config I) — v20 bayrakları kapalı."""
    m9.DISP_MIN = 0.0
    m9.TURN_GATE = False
    m9.BE_STOP_AT_R = 0.0
    m9.IDEAL_UNLOCK = False

CONFIGS = [
    ("0_v19_taban",   lambda: base()),
    ("1_+K2_disp",    lambda: (base(), setattr(m9, "DISP_MIN", 0.02))),
    ("2_+Kturn",      lambda: (base(), setattr(m9, "DISP_MIN", 0.02),
                               setattr(m9, "TURN_GATE", True))),
    ("3_+K4_BEstop",  lambda: (base(), setattr(m9, "DISP_MIN", 0.02),
                               setattr(m9, "TURN_GATE", True),
                               setattr(m9, "BE_STOP_AT_R", 0.5))),
    ("4_+K5_unlock",  lambda: (base(), setattr(m9, "DISP_MIN", 0.02),
                               setattr(m9, "TURN_GATE", True),
                               setattr(m9, "BE_STOP_AT_R", 0.5),
                               setattr(m9, "IDEAL_UNLOCK", True))),
]

if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, fn in CONFIGS:
        if only and only not in name:
            continue
        fn()
        print(f"\n### {name}")
        for wname, s, e in WINDOWS:
            res = m9.run_backtest(s, e, verbose=False)
            cap, fin = res["capital"], res["final_balance"]
            tr = res["trades"]
            gw = sum(t.pnl for t in tr if t.pnl > 0)
            gl = abs(sum(t.pnl for t in tr if t.pnl <= 0))
            pf = gw / gl if gl > 0 else float("inf")
            peak, mdd = -1e18, 0.0
            for _, eq in res["equity_curve"]:
                peak = max(peak, eq)
                mdd = max(mdd, (peak - eq) / peak if peak > 0 else 0)
            print(f"  {wname:8s}: {100*(fin-cap)/cap:+7.2f}%  n={len(tr):4d}  pf={pf:5.2f}  dd={100*mdd:4.1f}")
            sys.stdout.flush()
