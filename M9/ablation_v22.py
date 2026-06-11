#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v22 ablasyonu: maker maliyet modeli + funding kapısı.
Taban v21 (+22.38 / +5.62 / +0.17 / -16.19 / -10.65). Hedef: SON13G düşmesin, 6AY/OOS iyileşsin."""
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
    """v21 final varsayılanları (v22 bayrakları kapalı)."""
    m9.MAKER_PB = False
    m9.FUNDING_GATE = False

CONFIGS = [
    ("0_v21_taban",     lambda: base()),
    ("1_maker_opt",     lambda: (base(), setattr(m9, "MAKER_PB", True),
                                 setattr(m9, "MAKER_FILL", "optimistic"))),
    ("2_maker_strict",  lambda: (base(), setattr(m9, "MAKER_PB", True),
                                 setattr(m9, "MAKER_FILL", "strict"))),
    ("3_fund_03",       lambda: (base(), setattr(m9, "FUNDING_GATE", True),
                                 setattr(m9, "FUND_EXT", 0.0003))),
    ("4_fund_05",       lambda: (base(), setattr(m9, "FUNDING_GATE", True),
                                 setattr(m9, "FUND_EXT", 0.0005))),
    ("5_fund_10",       lambda: (base(), setattr(m9, "FUNDING_GATE", True),
                                 setattr(m9, "FUND_EXT", 0.001))),
    ("6_strict+fund05", lambda: (base(), setattr(m9, "MAKER_PB", True),
                                 setattr(m9, "MAKER_FILL", "strict"),
                                 setattr(m9, "FUNDING_GATE", True),
                                 setattr(m9, "FUND_EXT", 0.0005))),
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
