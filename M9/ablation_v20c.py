#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v20c — son tur: BTC-itki niteleyicileri (G: disp şartı, M: makro çapa) + E kombosu."""
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
    m9.DISP_MIN = 0.0
    m9.DISP_SCALE = False
    m9.TURN_GATE = False
    m9.BE_STOP_AT_R = 0.0
    m9.IDEAL_UNLOCK = False
    m9.THRUST_DISP_QUAL = 0.0
    m9.THRUST_MACRO_ANCHOR = False

CONFIGS = [
    ("G_dispQual022",  lambda: (base(), setattr(m9, "THRUST_DISP_QUAL", 0.022))),
    ("M_macroAnchor",  lambda: (base(), setattr(m9, "THRUST_MACRO_ANCHOR", True))),
    ("GM_birlikte",    lambda: (base(), setattr(m9, "THRUST_DISP_QUAL", 0.022),
                                setattr(m9, "THRUST_MACRO_ANCHOR", True))),
    ("ME_kombo",       lambda: (base(), setattr(m9, "THRUST_MACRO_ANCHOR", True),
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
