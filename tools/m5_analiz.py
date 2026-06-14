#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M5 İŞLEM TEŞHİSİ — neden kaybediyoruz?
=======================================
Canlı m5_state.json'ı (veya verilen başka state'i) okur, kapanan işlemleri
çok açıdan kırıp kaybın nereden geldiğini gösterir. SADECE OKUR, hiçbir şey değiştirmez.

Kullanım:
    python3 tools/m5_analiz.py                         # live/state/m5_state.json
    python3 tools/m5_analiz.py live/state/m7_state.json
    python3 tools/m5_analiz.py --csv /tmp/m5_islemler.csv   # işlemleri CSV'ye de yaz
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime


def _f(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def _parse_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def _bar(label, val, total, width=28):
    n = int(round(width * abs(val) / total)) if total else 0
    return f"{label:>14s} {'█'*n:<{width}} {val:+10.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state", nargs="?", default="live/state/m5_state.json")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    if not os.path.exists(args.state):
        # ev dizininden çalıştırıldıysa repo köküne bak
        alt = os.path.expanduser(f"~/Kripto_Botu/{args.state}")
        if os.path.exists(alt):
            args.state = alt
        else:
            print(f"❌ state bulunamadı: {args.state}")
            sys.exit(1)

    st = json.load(open(args.state))
    trades = st.get("closed_trades", []) or []
    name = st.get("mode", "M5")
    cap = _f(st.get("initial_capital"), 1000.0)
    bal = _f(st.get("final_balance"), _f(st.get("balance"), cap))

    print("═" * 64)
    print(f"  {name} İŞLEM TEŞHİSİ  ·  {args.state}")
    print("═" * 64)
    print(f"  Başlangıç: ${cap:,.2f}   Güncel nakit: ${bal:,.2f}")
    print(f"  Oluşturma: {st.get('created_at','?')}   Son güncelleme: {st.get('updated_at','?')}")
    print(f"  Kapanan işlem: {len(trades)}   Açık pozisyon: {len(st.get('open_positions',[]))}")
    if not trades:
        print("\n  Kapanan işlem yok — analiz edilecek veri yok.")
        return

    pnls = [_f(t.get("pnl")) for t in trades]
    wins = [t for t in trades if _f(t.get("pnl")) > 0]
    losses = [t for t in trades if _f(t.get("pnl")) <= 0]
    gross_w = sum(_f(t.get("pnl")) for t in wins)
    gross_l = abs(sum(_f(t.get("pnl")) for t in losses))
    pf = gross_w / gross_l if gross_l else float("inf")
    wr = 100 * len(wins) / len(trades)
    net = sum(pnls)
    avg_w = gross_w / len(wins) if wins else 0
    avg_l = -gross_l / len(losses) if losses else 0
    expectancy = net / len(trades)

    print("─" * 64)
    print(f"  Net realize PnL : ${net:+,.2f}   ({100*net/cap:+.2f}% sermayeye göre)")
    print(f"  Kazanma oranı   : %{wr:.1f}  ({len(wins)}K / {len(losses)}Z)")
    print(f"  Profit Factor   : {pf:.2f}")
    print(f"  Ort. kazanç     : ${avg_w:+.2f}   Ort. kayıp: ${avg_l:+.2f}   "
          f"(K/Z oranı: {abs(avg_w/avg_l):.2f})" if avg_l else "")
    print(f"  Beklenti/işlem  : ${expectancy:+.2f}")
    print(f"  En büyük kazanç : ${max(pnls):+.2f}   En büyük kayıp: ${min(pnls):+.2f}")

    # ── Çıkış sebebi ──
    by_reason = defaultdict(lambda: [0, 0.0, 0])
    for t in trades:
        r = t.get("exit_reason", "?")
        by_reason[r][0] += 1
        by_reason[r][1] += _f(t.get("pnl"))
        if _f(t.get("pnl")) > 0:
            by_reason[r][2] += 1
    print("─" * 64)
    print("  ÇIKIŞ SEBEBİNE GÖRE (kaybın kaynağı burada görünür)")
    for r, (c, pnl, w) in sorted(by_reason.items(), key=lambda x: x[1][1]):
        print(f"   {r:18s}: {c:4d} işlem  WR %{100*w/c:4.0f}  PnL ${pnl:+9.2f}")

    # ── LONG / SHORT ──
    print("─" * 64)
    print("  YÖN")
    for short in (False, True):
        sel = [t for t in trades if bool(t.get("is_short")) == short]
        if not sel:
            continue
        w = sum(1 for t in sel if _f(t.get("pnl")) > 0)
        pnl = sum(_f(t.get("pnl")) for t in sel)
        print(f"   {'SHORT' if short else 'LONG ':5s}: {len(sel):4d} işlem  "
              f"WR %{100*w/len(sel):4.0f}  PnL ${pnl:+9.2f}")

    # ── Coin bazlı ──
    by_coin = defaultdict(lambda: [0, 0.0, 0])
    for t in trades:
        c = t.get("symbol", "?")
        by_coin[c][0] += 1
        by_coin[c][1] += _f(t.get("pnl"))
        if _f(t.get("pnl")) > 0:
            by_coin[c][2] += 1
    ranked = sorted(by_coin.items(), key=lambda x: x[1][1])
    print("─" * 64)
    print("  EN ÇOK KAYBETTİREN 8 COİN")
    for c, (n, pnl, w) in ranked[:8]:
        print(f"   {c.replace('/USDT',''):8s}: {n:3d} işlem  WR %{100*w/n:4.0f}  PnL ${pnl:+9.2f}")
    print("  EN ÇOK KAZANDIRAN 5 COİN")
    for c, (n, pnl, w) in ranked[-5:][::-1]:
        print(f"   {c.replace('/USDT',''):8s}: {n:3d} işlem  WR %{100*w/n:4.0f}  PnL ${pnl:+9.2f}")

    # ── Tutma süresi ──
    holds = [(_f(t.get("bars_held")), _f(t.get("pnl"))) for t in trades if t.get("bars_held") is not None]
    if holds:
        buckets = {"<2sa (≤8 bar)": (0, 8), "2-8sa (9-32)": (9, 32),
                   "8-24sa (33-96)": (33, 96), ">24sa (>96)": (97, 10**9)}
        print("─" * 64)
        print("  TUTMA SÜRESİNE GÖRE (15m bar)")
        for lab, (lo, hi) in buckets.items():
            sel = [p for b, p in holds if lo <= b <= hi]
            if sel:
                w = sum(1 for p in sel if p > 0)
                print(f"   {lab:16s}: {len(sel):4d} işlem  WR %{100*w/len(sel):4.0f}  "
                      f"PnL ${sum(sel):+9.2f}")

    # ── Günlük kümelenme + drawdown ──
    daily = defaultdict(float)
    for t in trades:
        dt = _parse_dt(t.get("exit_date"))
        if dt:
            daily[dt.strftime("%Y-%m-%d")] += _f(t.get("pnl"))
    if daily:
        worst = sorted(daily.items(), key=lambda x: x[1])[:6]
        print("─" * 64)
        print("  EN KÖTÜ 6 GÜN (kayıp kümelenmesi)")
        for d, pnl in worst:
            print(f"   {d}: ${pnl:+9.2f}")
        neg_days = sum(1 for v in daily.values() if v < 0)
        print(f"  Toplam {len(daily)} işlem günü · {neg_days} negatif (%{100*neg_days/len(daily):.0f})")

    # cumulative drawdown (işlem sırasına göre)
    eq = cap
    peak = cap
    maxdd = 0.0
    ordered = sorted(trades, key=lambda t: _parse_dt(t.get("exit_date")) or datetime.min)
    for t in ordered:
        eq += _f(t.get("pnl"))
        peak = max(peak, eq)
        maxdd = max(maxdd, (peak - eq) / peak if peak else 0)
    print(f"  İşlem-bazlı Max Drawdown: %{100*maxdd:.1f}")

    # ── Son 12 işlem ──
    print("─" * 64)
    print("  SON 12 İŞLEM")
    for t in ordered[-12:]:
        side = "S" if t.get("is_short") else "L"
        sym = t.get("symbol", "?").replace("/USDT", "")
        print(f"   {sym:7s} {side} {str(t.get('exit_date',''))[:16]}  "
              f"${_f(t.get('pnl')):+8.2f}  {t.get('exit_reason','?')}")
    print("═" * 64)

    if args.csv:
        import csv
        keys = ["symbol", "is_short", "entry_price", "exit_price", "size",
                "pnl", "pnl_pct", "entry_date", "exit_date", "exit_reason", "bars_held"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for t in ordered:
                w.writerow(t)
        print(f"  📁 CSV yazıldı: {args.csv}")


if __name__ == "__main__":
    main()
