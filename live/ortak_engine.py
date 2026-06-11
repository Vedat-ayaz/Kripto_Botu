#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORTAK ENGINE — M9-sleeve shadow + T2 rejim tahsisçisi + defter aynalama (paper)
================================================================================
Mimari (backtest'le doğrulanan v24 final tasarım):
- SHADOW: M9 (valisiz sleeve modu) son SHADOW_DAYS günü her gün yeniden koşulur —
  test edilmiş backtest kodu birebir kullanılır, porte riski sıfır.
- TAHSİS: T2 teyit (sinyal bugün açık VE son 2 günün ≥1'inde açıktı) + 14g sağlık
  kapısı (shadow'un son 14 gün getirisi > -%2).
- AYNALAMA: tahsis açıkken gerçek (paper) hesap shadow'un defterini sermaye oranıyla
  kopyalar; kapalıyken nakitte bekler. Günlük senkron; gün içi tick'lerde shadow'un
  stop seviyeleri izlenir (stop kırılırsa pozisyon kapanır).
- OI GÖZLEMCİ: BTC open interest her tick loglanır — KARAR VERMEZ (30 gün veri
  birikince analiz edilip kapıya çevrilebilir).
Tüm dolumlar paper: son fiyat + taker maliyeti (%0.1 + %0.05).
"""

from __future__ import annotations

import json
import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "M9"))

import m9_backtest as m9          # noqa: E402
import allocator as al            # noqa: E402

logger = logging.getLogger(__name__)

SHADOW_DAYS   = 75       # shadow penceresi (warmup dahil yeterli bağlam)
HEALTH_ROLL   = 14       # sağlık kapısı: shadow son N gün getirisi
HEALTH_THR    = -0.02
TAKER_COST    = 0.0015   # paper dolum maliyeti (komisyon + slip, tek taraf)
RESIZE_TOL    = 0.25     # mevcut/hedef boyut farkı bunu aşarsa pozisyon yenilenir
INFLATION_ANNUAL = 0.03  # dolar enflasyonu varsayımı (reel getiri için; dashboard kullanır)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _last_price(symbol: str) -> float | None:
    """Binance spot anlık fiyat (paper dolum/mark-to-market için)."""
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/price",
                         params={"symbol": symbol}, timeout=10)
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception:
        return None


def _btc_open_interest() -> float | None:
    """BTC perp açık pozisyon (OI) — gözlemci, karar vermez."""
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/openInterest",
                         params={"symbol": "BTCUSDT"}, timeout=10)
        r.raise_for_status()
        return float(r.json()["openInterest"])
    except Exception:
        return None


class OrtakEngine:
    def __init__(self, capital: float, state_file: str):
        self.capital = capital
        self.state_file = state_file

    # ── state ──────────────────────────────────────────────────────────────
    def _load_state(self) -> dict:
        if os.path.exists(self.state_file):
            with open(self.state_file) as f:
                return json.load(f)
        now = _now_utc().isoformat()
        return {
            "mode": "ORTAK",
            "initial_capital": self.capital,
            "balance": self.capital,
            "final_balance": self.capital,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "peak_balance": self.capital,
            "open_positions": [],
            "closed_trades": [],
            "created_at": now,
            "updated_at": now,
            # ortak-özel
            "allocated": False,
            "signal_today": False,
            "shadow_equity": None,
            "shadow_return_pct": None,
            "last_sync_date": None,
            "equity_history": [],     # [[YYYY-MM-DD, equity], ...]
            "oi_log": [],             # [[iso_ts, btc_oi], ...]
        }

    def _save_state(self, st: dict):
        st["updated_at"] = _now_utc().isoformat()
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(st, f, indent=2, default=str)
        os.replace(tmp, self.state_file)

    # ── paper işlemler (LiveEngine/dashboard spot-marj muhasebesiyle uyumlu:
    #    LONG: açılışta cost nakitten düşer; SHORT: cost kadar marj kilitlenir) ──
    def _close_position(self, st: dict, pos: dict, px: float, reason: str):
        qty = pos["size"]
        fee = qty * px * TAKER_COST
        cost = pos.get("cost", pos["entry_price"] * qty)
        if pos["is_short"]:
            pnl = (pos["entry_price"] - px) * qty - fee
            st["balance"] += pos.get("margin_locked", cost) + pnl
        else:
            proceeds = qty * px - fee
            pnl = proceeds - cost
            st["balance"] += proceeds
        st["closed_trades"].append({
            "symbol": pos["symbol"], "entry_price": pos["entry_price"],
            "exit_price": px, "size": qty, "cost": pos.get("cost", 0.0),
            "pnl": round(pnl, 4), "pnl_pct": round(100 * pnl / max(pos.get("cost", 1), 1), 3),
            "entry_date": pos.get("entry_date"), "exit_date": _now_utc().isoformat(),
            "exit_reason": reason, "is_short": pos["is_short"],
        })
        st["open_positions"] = [p for p in st["open_positions"]
                                if p["symbol"] != pos["symbol"]]
        print(f"  ORTAK kapat: {pos['symbol']} {'S' if pos['is_short'] else 'L'} "
              f"@{px:.6g} pnl={pnl:+.2f} ({reason})")

    def _open_position(self, st: dict, symbol: str, is_short: bool, qty: float,
                       px: float, stop: float, strategy: str):
        cost = qty * px
        fee = cost * TAKER_COST
        if st["balance"] < cost + fee:          # nakit yetmiyorsa boyu kırp
            scale = max(st["balance"] - fee, 0) / cost if cost > 0 else 0
            if scale < 0.05:
                return
            qty *= scale
            cost = qty * px
            fee = cost * TAKER_COST
        st["balance"] -= cost + fee             # LONG: cost nakitten; SHORT: marj kilidi
        st["open_positions"].append({
            "symbol": symbol, "entry_price": px,
            "entry_date": _now_utc().isoformat(), "size": qty,
            "cost": cost, "margin_locked": cost if is_short else 0.0,
            "stop_price": stop, "trail_price": stop,
            "unrealized_pnl": 0.0, "is_short": is_short,
            "strategy": strategy, "source": "shadow_mirror",
        })
        print(f"  ORTAK aç: {symbol} {'S' if is_short else 'L'} qty={qty:.6g} @{px:.6g}")

    @staticmethod
    def _open_value(st: dict) -> float:
        """Dashboard compute_open_position_value ile aynı: LONG cost+upnl, SHORT marj+upnl."""
        total = 0.0
        for p in st.get("open_positions", []):
            upnl = float(p.get("unrealized_pnl", 0.0) or 0.0)
            if p.get("is_short"):
                total += float(p.get("margin_locked", p.get("cost", 0.0)) or 0.0) + upnl
            else:
                total += float(p.get("cost", 0.0) or 0.0) + upnl
        return total

    # ── günlük senkron: shadow + sinyal + defter aynalama ─────────────────
    def _daily_sync(self, st: dict, today: str):
        end = (pd.Timestamp(today) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        start = (pd.Timestamp(today) - pd.Timedelta(days=SHADOW_DAYS)).strftime("%Y-%m-%d")

        # shadow: M9 sleeve modu (vali yok — risk tahsisçide)
        m9.GOV_STICKY = False
        m9.DD_STAGES = []
        res = m9.run_backtest(start, end, verbose=False, keep_open=True)
        eq = pd.Series({ts: v for ts, v in res["equity_curve"]}).sort_index()
        d = eq.resample("1D").last().dropna()
        shadow_eq = float(d.iloc[-1])
        roll14 = (d.iloc[-1] / d.iloc[-HEALTH_ROLL] - 1) if len(d) > HEALTH_ROLL else 0.0
        healthy = roll14 > HEALTH_THR

        # T2 sinyali (lookahead'siz: dünün kapanışına kadarki veriyle)
        sig = al.build_signal(start=(pd.Timestamp(today) - pd.Timedelta(days=40)).strftime("%Y-%m-%d"),
                              end=today)
        s = sig.reindex(sig.index.sort_values())
        sig_today = bool(s.iloc[-1]) if len(s) else False
        sig_recent = bool(s.iloc[-2] or s.iloc[-3]) if len(s) >= 3 else False
        confirmed = sig_today and sig_recent
        allocated = confirmed and healthy

        st["signal_today"] = sig_today
        st["allocated"] = allocated
        st["shadow_equity"] = round(shadow_eq, 2)
        st["shadow_return_pct"] = round(100 * (shadow_eq / res["capital"] - 1), 2)
        st["last_sync_date"] = today

        # hedef defter
        equity_now = st["balance"] + self._open_value(st)
        scale = equity_now / shadow_eq if shadow_eq > 0 else 0.0
        targets = {}
        if allocated:
            for op in res["open_positions"]:
                if op["last_px"] is None:
                    continue
                targets[op["symbol"]] = dict(
                    is_short=op["is_short"], qty=op["qty"] * scale,
                    px=op["last_px"], stop=op["stop"], strategy=op["strategy"])

        # mevcutları hedefle uyumla
        for pos in list(st["open_positions"]):
            sym = pos["symbol"]
            t = targets.get(sym)
            px = _last_price(sym.replace("/", "")) or pos["entry_price"]
            if t is None or t["is_short"] != pos["is_short"]:
                self._close_position(st, pos, px, "tahsis_kapandi" if t is None else "yon_degisti")
            elif abs(t["qty"] - pos["size"]) / max(t["qty"], 1e-12) > RESIZE_TOL:
                self._close_position(st, pos, px, "boyut_yenileme")
            else:
                pos["stop_price"] = t["stop"]   # shadow'un güncel stopunu taşı
                targets.pop(sym, None)
                continue
            # kapatılanlar yeniden açılabilir (aşağıda)
        held = {p["symbol"] for p in st["open_positions"]}
        for sym, t in targets.items():
            if sym in held or t["qty"] <= 0:
                continue
            px = _last_price(sym.replace("/", "")) or t["px"]
            self._open_position(st, sym, t["is_short"], t["qty"], px, t["stop"], t["strategy"])

        durum = "AÇIK ✅" if allocated else ("sinyal var, teyit/sağlık yok" if sig_today else "nakitte 💤")
        print(f"  ORTAK senkron: shadow={st['shadow_return_pct']:+.2f}% "
              f"sağlık={'OK' if healthy else 'ZAYIF'} tahsis={durum} "
              f"pozisyon={len(st['open_positions'])}")

    # ── tick ───────────────────────────────────────────────────────────────
    def tick(self):
        st = self._load_state()
        today = _now_utc().strftime("%Y-%m-%d")

        # 1) günlük senkron (günde bir, 00:35 UTC sonrası — dünün barları kesin kapalı)
        if st.get("last_sync_date") != today and _now_utc().hour * 60 + _now_utc().minute >= 35:
            try:
                self._daily_sync(st, today)
            except Exception as e:
                logger.exception("ORTAK günlük senkron hatası")
                print(f"  ❌ ORTAK senkron hata: {e}")

        # 2) gün içi: stop takibi + mark-to-market
        for pos in list(st["open_positions"]):
            px = _last_price(pos["symbol"].replace("/", ""))
            if px is None:
                continue
            stop = pos.get("stop_price")
            hit = stop and ((pos["is_short"] and px >= stop) or
                            (not pos["is_short"] and px <= stop))
            if hit:
                self._close_position(st, pos, float(stop), "stop")
            else:
                gross = (pos["entry_price"] - px if pos["is_short"]
                         else px - pos["entry_price"]) * pos["size"]
                pos["unrealized_pnl"] = round(gross, 4)

        # 3) metrikler
        equity = st["balance"] + self._open_value(st)
        st["final_balance"] = round(st["balance"], 2)
        st["total_pnl"] = round(equity - st["initial_capital"], 2)
        st["total_pnl_pct"] = round(100 * (equity / st["initial_capital"] - 1), 2)
        st["peak_balance"] = max(st.get("peak_balance", equity), equity)
        if st["peak_balance"] > 0:
            dd = 100 * (1 - equity / st["peak_balance"])
            st["max_drawdown_pct"] = round(max(st.get("max_drawdown_pct", 0.0), dd), 2)
        wins = sum(1 for t in st["closed_trades"] if t["pnl"] > 0)
        st["total_trades"] = len(st["closed_trades"])
        st["win_rate"] = round(wins / st["total_trades"], 3) if st["total_trades"] else 0.0

        # 4) equity history (günde bir nokta; gün içi son değer üzerine yazılır)
        hist = [e for e in st.get("equity_history", []) if e[0] != today]
        hist.append([today, round(equity, 2)])
        st["equity_history"] = hist[-400:]

        # 5) OI gözlemci (BTC)
        oi = _btc_open_interest()
        if oi is not None:
            st.setdefault("oi_log", []).append([_now_utc().isoformat(), oi])
            st["oi_log"] = st["oi_log"][-2000:]

        self._save_state(st)
        print(f"  ORTAK: equity=${equity:,.2f} ({st['total_pnl_pct']:+.2f}%) "
              f"tahsis={'AÇIK' if st.get('allocated') else 'nakit'} "
              f"poz={len(st['open_positions'])} OI={'%.0f' % oi if oi else 'n/a'}")
