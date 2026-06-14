"""
Kripto Bot — Live Runner (Gerçek Zamanlı Paper Trading)
========================================================
CCXT public endpoint'leriyle çalışır — API anahtarı gerekmez.
Her çalıştırmada son kapanmış barı işler (duplicate guard ile çift işlem önlenir).

Kullanım:
    python live/live_runner.py                          # Tek tick
    python live/live_runner.py --loop 900               # Her 15 dakikada tick (15m)
    python live/live_runner.py --loop 60                # Her dakika tick (M6 1m)
    python live/live_runner.py --fresh --capital 1000   # Testi SIFIRLA

Cron (server):
    */15 * * * * cd /srv/kripto && venv/bin/python live/live_runner.py >> logs/live.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Proje kökünü path'e ekle
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from live.live_engine import LiveEngine
from dashboard.state import BotStateDB
from crypto_portfolio_test import SYMBOLS, UNIVERSE

# ── Loglama ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Konfigürasyon ─────────────────────────────────────────────────────────────

STATE_DIR    = _ROOT / "live" / "state"
M4_STATE     = str(STATE_DIR / "m4_state.json")
M5_STATE     = str(STATE_DIR / "m5_state.json")
M6_STATE     = str(STATE_DIR / "m6_state.json")
M7_STATE     = str(STATE_DIR / "m7_state.json")
M8_STATE     = str(STATE_DIR / "m8_state.json")
ORTAK_STATE  = str(STATE_DIR / "ortak_state.json")
CONFIG_FILE  = STATE_DIR / "config.json"
DB_PATH      = str(_ROOT / "dashboard" / "bot_state.db")
SNAPSHOT_DIR = _ROOT / "logs" / "snapshots"
TRADES_LOG   = _ROOT / "logs" / "trades"

DEFAULT_CAPITAL = 1000.0
DEFAULT_COINS   = 9


# ── Config yönetimi ───────────────────────────────────────────────────────────

def _load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    cfg = {
        "capital": DEFAULT_CAPITAL,
        "coins":   DEFAULT_COINS,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_config(cfg)
    return cfg


def _save_config(cfg: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def _fresh_start(capital: float, coins: int) -> None:
    """Tüm state'i sil ve sıfırdan başla."""
    print("\n🔄 SIFIRDAN BAŞLATILIYOR...")
    for f in [M4_STATE, M5_STATE, M6_STATE, M7_STATE, M8_STATE, ORTAK_STATE]:
        if Path(f).exists():
            Path(f).unlink()
            print(f"  🗑  Silindi: {f}")
    if Path(DB_PATH).exists():
        Path(DB_PATH).unlink()
        print(f"  🗑  DB temizlendi: {DB_PATH}")
    cfg = {
        "capital": capital,
        "coins":   coins,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_config(cfg)
    print(f"\n  ✅ Sıfırlama tamamlandı")
    print(f"  💰 Sermaye: ${capital:,.0f}")
    print(f"  🪙 Coin sayısı: {coins}\n")


# ── DB Senkronizasyon ─────────────────────────────────────────────────────────

def _sync_to_db(state_file: str, db: BotStateDB) -> None:
    """JSON state → SQLite DB senkronizasyonu."""
    if not Path(state_file).exists():
        return
    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)

    mode = state.get("mode", "M4")
    db.update_bot_status(
        mode=mode,
        running=True,
        account_balance=state.get("final_balance", DEFAULT_CAPITAL),
        initial_balance=state.get("initial_capital", DEFAULT_CAPITAL),
        daily_pnl=0.0,
        total_pnl=state.get("total_pnl", 0.0),
        trading_allowed=True,
    )

    for pos in state.get("open_positions", []):
        db.upsert_open_position(
            symbol=pos["symbol"],
            entry_price=pos["entry_price"],
            position_size=pos["size"],
            stop_price=pos["stop_price"],
            trailing_stop_price=pos["trail_price"],
            unrealized_pnl=pos.get("unrealized_pnl", 0.0),
            cost_basis=pos["cost"],
            opened_at=pos["entry_date"],
        )

    existing = {
        (t["symbol"], t.get("opened_at", ""), t.get("closed_at", ""))
        for t in db.get_closed_trades(limit=500)
    }
    for trade in state.get("closed_trades", []):
        key = (trade["symbol"], trade.get("entry_date", ""), trade.get("exit_date", ""))
        if key not in existing:
            db.insert_closed_trade(
                symbol=trade["symbol"],
                entry_price=trade["entry_price"],
                exit_price=trade["exit_price"],
                position_size=trade.get("size", 0.0),
                realized_pnl=trade["pnl"],
                close_reason=trade.get("exit_reason", ""),
                opened_at=trade.get("entry_date", ""),
                closed_at=trade.get("exit_date", ""),
            )

    db.insert_equity_point(state.get("final_balance", DEFAULT_CAPITAL))
    print(
        f"  ✓ DB: {mode} | {len(state.get('open_positions',[]))} açık | "
        f"{len(state.get('closed_trades',[]))} trade | "
        f"bakiye=${state.get('final_balance', 0):.2f}"
    )


# ── Snapshot ──────────────────────────────────────────────────────────────────

def _save_snapshots() -> None:
    """Her run sonrası state dosyalarını tarihli kopya olarak kaydeder."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    TRADES_LOG.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for label, path in [("m4", M4_STATE), ("m5", M5_STATE), ("m6", M6_STATE), ("m7", M7_STATE), ("m8", M8_STATE)]:
        if not Path(path).exists():
            continue
        try:
            shutil.copy2(path, SNAPSHOT_DIR / f"{label}_{ts}.json")
            with open(path) as f:
                st = json.load(f)
            trades = st.get("closed_trades", [])
            if trades:
                with open(TRADES_LOG / f"{label}_trades.jsonl", "a") as f:
                    f.write(json.dumps({
                        "snapshot_ts":    ts,
                        "total_trades":   len(trades),
                        "final_balance":  st.get("final_balance", 0),
                        "total_pnl_pct":  st.get("total_pnl_pct", 0),
                        "win_rate":       st.get("win_rate", 0),
                        "max_drawdown_pct": st.get("max_drawdown_pct", 0),
                    }) + "\n")
        except Exception as e:
            print(f"  ⚠️  Snapshot kaydedilemedi ({label}): {e}")
    print(f"  💾 Snapshot: {ts}")
    # 14 günden eski snapshot'ları sil
    try:
        cutoff = datetime.now() - timedelta(days=14)
        for snap in SNAPSHOT_DIR.glob("*.json"):
            if datetime.fromtimestamp(snap.stat().st_mtime) < cutoff:
                snap.unlink()
    except Exception:
        pass


# ── Ana Tick ─────────────────────────────────────────────────────────────────

def _append_equity_history(state_files: list[str]) -> None:
    """Her modelin state'ine günlük equity noktası ekler (Sharpe/Omega hesabı için).
    Equity = serbest nakit + açık pozisyon unrealized PnL toplamı."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for sf in state_files:
        p = Path(sf)
        if not p.exists():
            continue
        try:
            with open(p) as f:
                st = json.load(f)
            # dashboard compute_open_position_value ile aynı formül:
            # equity = serbest nakit + Σ(LONG: cost+upnl | SHORT: marj+upnl)
            equity = float(st.get("balance", st.get("final_balance", 0.0)))
            for pos in st.get("open_positions", []):
                upnl = float(pos.get("unrealized_pnl", 0.0) or 0.0)
                if pos.get("is_short"):
                    locked = float(pos.get("margin_locked", 0.0) or 0.0)
                    if locked < 0.01:
                        locked = float(pos.get("cost", 0.0) or 0.0)
                    equity += locked + upnl
                else:
                    equity += float(pos.get("cost", 0.0) or 0.0) + upnl
            hist = [e for e in st.get("equity_history", []) if e[0] != today]
            hist.append([today, round(equity, 2)])
            st["equity_history"] = hist[-400:]
            with open(p, "w") as f:
                json.dump(st, f, indent=2, default=str)
        except Exception:
            logger.exception(f"equity history yazılamadı: {sf}")


def run_once(cfg: dict) -> None:
    """Tüm modeller için bir tick çalıştırır."""
    capital = cfg.get("capital", DEFAULT_CAPITAL)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  LIVE ENGINE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Sermaye: ${capital:,.0f}")
    print(f"{'='*60}")

    # ── M4 (15m, tüm UNIVERSE) ───────────────────────────────────────────────
    print(f"\n🔵 M4 tick (15m, UNIVERSE — {len(UNIVERSE)} coin)...")
    try:
        engine_m4 = LiveEngine(
            mode="M4",
            symbols=UNIVERSE,
            timeframe="15m",
            capital=capital,
            state_file=M4_STATE,
            m4_mode=True,
        )
        engine_m4.tick()
    except Exception as e:
        print(f"  ❌ M4 hata: {e}")
        logger.exception("M4 tick hata")

    # ── M5 (15m, tüm UNIVERSE) ───────────────────────────────────────────────
    print(f"\n🟣 M5 tick (15m, UNIVERSE — {len(UNIVERSE)} coin)...")
    try:
        engine_m5 = LiveEngine(
            mode="M5",
            symbols=UNIVERSE,
            timeframe="15m",
            capital=capital,
            state_file=M5_STATE,
            use_universe=True,
            m5_mode=True,
        )
        engine_m5.tick()
    except Exception as e:
        print(f"  ❌ M5 hata: {e}")
        logger.exception("M5 tick hata")

    # ── M6 (YENİ) = M5 KLONU — değişiklik sandbox'ı ──────────────────────────
    #    Eski M6 (hızlı-swing scalper) ve M7 EMEKLİYE AYRILDI (14 Haz 2026, kullanıcı
    #    kararı: canlıda istenen gibi çalışmadılar). Eski state'ler _archive/'a taşındı.
    #    YENİ M6 = M5 ile BİREBİR (m5_mode), ayrı state → M5 dokunulmadan M6 üzerinde
    #    iyileştirme denenir. Backtest ablasyonu (14 Haz): re-entry/scale-out backtest'te
    #    ölü, swing-stop marjinal/trend-eğimli → şimdilik SAF KLON, yamalar sonra.
    print(f"\n🟢 M6 tick (M5 KLONU — 15m, UNIVERSE)...")
    try:
        engine_m6 = LiveEngine(
            mode="M6",
            symbols=UNIVERSE,
            timeframe="15m",
            capital=capital,
            state_file=M6_STATE,
            use_universe=True,
            m5_mode=True,      # M5 KLONU — birebir M5 davranışı, ayrı state
        )
        engine_m6.tick()
    except Exception as e:
        print(f"  ❌ M6 hata: {e}")
        logger.exception("M6 tick hata")

    # ── M8 (15m, tüm UNIVERSE) — M7-klon + hacim iyileştirmeleri ─────────────
    # NOT: M7 DONUKTUR — M8 ayrı motor/state, M7 dokunulmaz.
    print(f"\n🟤 M8 tick (15m, UNIVERSE — {len(UNIVERSE)} coin)...")
    try:
        engine_m8 = LiveEngine(
            mode="M8",
            symbols=UNIVERSE,
            timeframe="15m",
            capital=capital,
            state_file=M8_STATE,
            use_universe=True,
            m8_mode=True,
        )
        engine_m8.tick()
    except Exception as e:
        print(f"  ❌ M8 hata: {e}")
        logger.exception("M8 tick hata")

    # ── ORTAK (M9-sleeve shadow + T2 tahsisçi + defter aynalama + OI gözlemci) ─
    print(f"\n🟡 ORTAK tick (tahsisçi — shadow M9 + T2)...")
    try:
        from ortak_engine import OrtakEngine
        engine_ortak = OrtakEngine(capital=capital, state_file=ORTAK_STATE)
        engine_ortak.tick()
    except Exception as e:
        print(f"  ❌ ORTAK hata: {e}")
        logger.exception("ORTAK tick hata")

    # ── Günlük equity geçmişi (Sharpe/Omega için — aktif modeller) ────────────
    try:
        _append_equity_history([M4_STATE, M5_STATE, M6_STATE, M8_STATE])
    except Exception as e:
        logger.exception("equity history hata")

    # ── Dashboard güncelle ────────────────────────────────────────────────────
    print("\n📊 Dashboard güncelleniyor...")
    try:
        db = BotStateDB(DB_PATH)
        _sync_to_db(M5_STATE, db)    # M5 = ana model
    except Exception as e:
        print(f"  ❌ DB sync hata: {e}")

    # ── Snapshot ─────────────────────────────────────────────────────────────
    _save_snapshots()

    print(f"\n✅ Tick tamamlandı — {datetime.now().strftime('%H:%M:%S')}")


def run_loop(cfg: dict, interval_seconds: int = 900) -> None:
    """Döngüsel çalıştırma — her N saniyede bir tick."""
    print(f"🔄 Loop modu: her {interval_seconds}s ({interval_seconds//60} dk) tick")
    while True:
        run_once(cfg)
        nxt = datetime.now() + timedelta(seconds=interval_seconds)
        print(f"\n⏰ Sonraki tick: {nxt.strftime('%H:%M:%S')}")
        time.sleep(interval_seconds)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kripto Bot — Gerçek Zamanlı Live Paper Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python live/live_runner.py                        # Tek tick (M4+M5+M6)
  python live/live_runner.py --loop 900             # Her 15 dk (15m botlar)
  python live/live_runner.py --loop 60              # Her dakika (M6 1m için)
  python live/live_runner.py --fresh --capital 1000 # Testi sıfırla
        """
    )
    parser.add_argument("--loop",    type=int,   default=0,
                        help="Loop modu: her N saniyede tick çalıştır (0=tek sefer)")
    parser.add_argument("--fresh",   action="store_true",
                        help="Tüm state'i sil, sıfırdan başlat")
    parser.add_argument("--capital", type=float, default=None,
                        help=f"Test sermayesi $ (varsayılan: {DEFAULT_CAPITAL})")
    parser.add_argument("--coins",   type=int,   default=None,
                        help=f"Coin sayısı (varsayılan: {DEFAULT_COINS})")
    args = parser.parse_args()

    cfg = _load_config()

    if args.fresh:
        cap   = args.capital or DEFAULT_CAPITAL
        coins = args.coins   or DEFAULT_COINS
        _fresh_start(cap, coins)
        cfg = _load_config()

    if args.capital and not args.fresh:
        cfg["capital"] = args.capital
        _save_config(cfg)

    if args.coins and not args.fresh:
        cfg["coins"] = args.coins
        _save_config(cfg)

    if args.loop > 0:
        run_loop(cfg, args.loop)
    else:
        run_once(cfg)
