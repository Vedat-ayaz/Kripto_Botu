"""
Kripto Bot — Live Runner (Paper Mode)
======================================
API anahtarı olmadan çalışır: sadece CCXT public endpoints kullanır.
Her çalıştırmada M4 ve M5 backtesti bugüne kadar çalıştırır,
sonuçları state/*.json dosyalarına ve dashboard SQLite DB'sine yazar.

Kullanım:
    python live/live_runner.py                        # Tek sefer güncelle
    python live/live_runner.py --loop 3600            # Her saat güncelle
    python live/live_runner.py --fresh --capital 1000 # Testi SIFIRLA ve başlat

Cron (server):
    0 * * * * cd /srv/kripto && venv/bin/python live/live_runner.py >> logs/live.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Proje kökünü path'e ekle
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from crypto_portfolio_test import run_portfolio_backtest, INITIAL_CAPITAL
from dashboard.state import BotStateDB

# ── Konfigürasyon ─────────────────────────────────────────────────────────────

STATE_DIR       = _ROOT / "live" / "state"
M4_STATE        = str(STATE_DIR / "m4_state.json")
M5_STATE        = str(STATE_DIR / "m5_state.json")
M6_STATE        = str(STATE_DIR / "m6_state.json")
CONFIG_FILE     = STATE_DIR / "config.json"       # sermaye, başlangıç tarihi
DB_PATH         = str(_ROOT / "dashboard" / "bot_state.db")
# v13: Her loop iterasyonunda state kopyalanır — geçmiş test sonuçları kayıt altında
SNAPSHOT_DIR    = _ROOT / "logs" / "snapshots"
TRADES_LOG_DIR  = _ROOT / "logs" / "trades"

DEFAULT_CAPITAL      = 1000.0
DEFAULT_COINS        = 15


# ── Config yönetimi ───────────────────────────────────────────────────────────

def _load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    # Varsayılan: bugünden başla
    cfg = {
        "start_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "capital": DEFAULT_CAPITAL,
        "coins": DEFAULT_COINS,
        "test_started_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_config(cfg)
    return cfg


def _save_config(cfg: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def _fresh_start(capital: float, coins: int) -> None:
    """
    Tüm state'i sil ve sıfırdan başla.
    - Bugünü başlangıç tarihi yap
    - Sermayeyi sıfırla ($capital)
    - DB'yi temizle
    """
    print("\n🔄 SIFIRDAN BAŞLATILIYOR...")

    # State dosyalarını sil
    for f in [M4_STATE, M5_STATE, M6_STATE]:
        if Path(f).exists():
            Path(f).unlink()
            print(f"  🗑  Silindi: {f}")

    # DB'yi temizle
    if Path(DB_PATH).exists():
        Path(DB_PATH).unlink()
        print(f"  🗑  DB temizlendi: {DB_PATH}")

    # Yeni config kaydet
    # v12: start_date = BUGÜN (forward paper trading).
    # Önceden 90 gün geri ayarlanıyordu → trade'ler geçmişte oluşuyordu.
    # Şimdi: bot başlatıldığı andan itibaren paper-trade yapar, trade tarihleri ileri.
    # Veri yine 365 gün geri çekilir (indikatör warmup için), ama trade simülasyonu
    # sadece start_date'ten itibaren çalışır. İlk gün 0 trade, her saat 1 bar eklenir.
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cfg = {
        "start_date": today,
        "capital": capital,
        "coins": coins,
        "test_started_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_config(cfg)

    print(f"\n  ✅ Sıfırlama tamamlandı")
    print(f"  📅 Başlangıç: {today}")
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
            unrealized_pnl=pos["unrealized_pnl"],
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
                position_size=0.0,
                realized_pnl=trade["pnl"],
                close_reason=trade.get("exit_reason", ""),
                opened_at=trade.get("entry_date", ""),
                closed_at=trade.get("exit_date", ""),
            )

    db.insert_equity_point(state.get("final_balance", DEFAULT_CAPITAL))
    print(f"  ✓ DB: {mode} | {len(state.get('open_positions',[]))} açık | {len(state.get('closed_trades',[]))} trade")


# ── Ana Fonksiyon ─────────────────────────────────────────────────────────────

def run_once(cfg: dict) -> None:
    start_date = cfg["start_date"]
    capital    = cfg.get("capital", DEFAULT_CAPITAL)
    coins      = cfg.get("coins", DEFAULT_COINS)
    # v18 BUG FIX: end_date sadece tarih olunca trade_end = gece yarısı oluyor.
    # start_date ile aynı gün ise all_ts = 1 bar → sıfır trade.
    # Çözüm: end_date=None → run_portfolio_backtest datetime.now() kullanır (saatli).
    end_date   = None  # Her run güncel datetime.now(tz=UTC) kullanılsın

    print(f"\n{'='*60}")
    print(f"  LIVE RUNNER — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Test başlangıcı : {cfg.get('test_started_at','?')[:10]}")
    print(f"  Analiz dönemi   : {start_date} → {end_date}")
    print(f"  Sermaye         : ${capital:,.0f}")
    print(f"{'='*60}")

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    # ── M4 ────────────────────────────────────────────────────────────────────
    # v17: Live modda WFO kapalı — her run 2-5 dk (önceden 20+ dk).
    # WFO yerine PROFILES parametreleri, adaptive regime controller çalışmaya devam eder.
    # auto_mode=False: BTC rejim tespiti kaldırıldı (auto_mode NEUTRAL/BEAR'de use_wfo=True
    # zorluyor — caller'ın use_wfo=False ayarını override ediyordu → asıl hang sebebi).
    # days=90: 365 gün × 96 bar/gün × 30 coin = 1M bar. 90 gün ile 4× hızlı.
    print("\n🔵 M4 çalışıyor (15m)...")
    try:
        run_portfolio_backtest(
            start_date=start_date,
            end_date=end_date,
            n_coins=coins,
            initial_capital=capital,
            m4_mode=True,
            auto_mode=False,
            use_universe=False,   # 9 sabit coin (SYMBOLS) — PROFILES ile hızlı
            use_wfo=False,        # Live'da WFO yok — PROFILES parametreleri kullan
            timeframe="15m",
            days=90,              # 90 gün × 96 bar = ~8640 bar/coin (hızlı)
            json_out=M4_STATE,
        )
    except Exception as e:
        print(f"  ❌ M4 hata: {e}")

    # ── M5 ────────────────────────────────────────────────────────────────────
    print("\n🟣 M5 çalışıyor (15m)...")
    try:
        run_portfolio_backtest(
            start_date=start_date,
            end_date=end_date,
            n_coins=coins,
            initial_capital=capital,
            m5_mode=True,
            auto_mode=False,
            use_universe=True,    # UNIVERSE (15 coin) — daha geniş seçim
            use_wfo=False,
            timeframe="15m",
            days=90,
            json_out=M5_STATE,
        )
    except Exception as e:
        print(f"  ❌ M5 hata: {e}")

    # ── M6 ────────────────────────────────────────────────────────────────────
    # 1m bar: 365 gün × 1440 bar/gün × 9 coin = 4.7M bar → AWS'de hang.
    # days=7 ile: 7 × 1440 × 9 = 90,720 bar → hızlı (<5 dk).
    # v19: Scalping için 7 gün yeterli — daha uzun geçmiş anlamsız (1m volatilite çok farklı).
    print("\n🟢 M6 çalışıyor (1m scalping)...")
    try:
        run_portfolio_backtest(
            start_date=start_date,
            end_date=end_date,
            n_coins=coins,
            initial_capital=capital,
            m6_mode=True,
            auto_mode=False,
            use_universe=False,   # 9 coin — 1m'de geniş universe gerekmiyor
            use_wfo=False,
            timeframe="1m",
            days=7,               # v19: 14→7 gün (90K bar, ~3 dk) — scalping için yeterli
            json_out=M6_STATE,
        )
    except Exception as e:
        print(f"  ❌ M6 hata: {e}")

    # ── DB sync ───────────────────────────────────────────────────────────────
    print("\n📊 Dashboard güncelleniyor...")
    db = BotStateDB(DB_PATH)
    _sync_to_db(M5_STATE, db)   # M5 ana bot (dashboard'da gösterilecek)

    # ── State Snapshot (v13) ──────────────────────────────────────────────────
    # Her iterasyonda state kopyalanır → ileride model performansını analiz edebilmek için
    # Format: logs/snapshots/m{N}_YYYYMMDD_HHMMSS.json
    _save_snapshots()

    print(f"\n✅ Tamamlandı — {datetime.now().strftime('%H:%M:%S')}")


def _save_snapshots() -> None:
    """Her run sonrası 3 model state'ini tarihli kopya olarak kaydeder."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    TRADES_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for model_label, path in [("m4", M4_STATE), ("m5", M5_STATE), ("m6", M6_STATE)]:
        if not Path(path).exists():
            continue
        try:
            # Tam state snapshot
            dest = SNAPSHOT_DIR / f"{model_label}_{ts}.json"
            shutil.copy2(path, dest)
            # Trades-only append log (analiz için kompakt)
            with open(path) as f:
                state = json.load(f)
            trades = state.get("closed_trades", [])
            if trades:
                trades_log = TRADES_LOG_DIR / f"{model_label}_trades.jsonl"
                with open(trades_log, "a") as f:
                    f.write(json.dumps({
                        "snapshot_ts": ts,
                        "total_trades": len(trades),
                        "final_balance": state.get("final_balance", 0),
                        "total_pnl_pct": state.get("total_pnl_pct", 0),
                        "win_rate": state.get("win_rate", 0),
                        "max_drawdown_pct": state.get("max_drawdown_pct", 0),
                    }) + "\n")
        except Exception as e:
            print(f"  ⚠️  Snapshot kaydedilemedi ({model_label}): {e}")
    print(f"  💾 Snapshot kaydedildi: {ts}")
    # Eski snapshot temizliği — son 14 günden eski olanları sil
    try:
        cutoff = datetime.now() - timedelta(days=14)
        for snap in SNAPSHOT_DIR.glob("*.json"):
            if datetime.fromtimestamp(snap.stat().st_mtime) < cutoff:
                snap.unlink()
    except Exception:
        pass


def run_loop(cfg: dict, interval_seconds: int = 3600) -> None:
    print(f"🔄 Loop: her {interval_seconds//60} dakika güncelleme")
    while True:
        run_once(cfg)
        nxt = datetime.now() + timedelta(seconds=interval_seconds)
        print(f"\n⏰ Sonraki: {nxt.strftime('%H:%M:%S')} ({interval_seconds//60} dk)")
        time.sleep(interval_seconds)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kripto Bot Live Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python live/live_runner.py                        # Tek güncelleme
  python live/live_runner.py --loop 3600            # Saatlik döngü
  python live/live_runner.py --fresh --capital 1000 # Testi sıfırla ve başlat
        """
    )
    parser.add_argument("--loop",    type=int,   default=0,
                        help="Loop modu: her N saniyede çalıştır")
    parser.add_argument("--fresh",   action="store_true",
                        help="Tüm state'i sil, sıfırdan başlat")
    parser.add_argument("--capital", type=float, default=None,
                        help=f"Test sermayesi $ (varsayılan: {DEFAULT_CAPITAL})")
    parser.add_argument("--coins",   type=int,   default=None,
                        help=f"Coin sayısı (varsayılan: {DEFAULT_COINS})")
    parser.add_argument("--start",   type=str,   default=None,
                        help="Başlangıç tarihi override (YYYY-MM-DD)")
    args = parser.parse_args()

    # Mevcut config'i yükle
    cfg = _load_config()

    # --fresh: her şeyi sıfırla
    if args.fresh:
        cap   = args.capital or DEFAULT_CAPITAL
        coins = args.coins   or DEFAULT_COINS
        _fresh_start(cap, coins)
        cfg = _load_config()

    # Parametre override'ları
    if args.capital and not args.fresh:
        cfg["capital"] = args.capital
        _save_config(cfg)

    if args.coins and not args.fresh:
        cfg["coins"] = args.coins
        _save_config(cfg)

    if args.start:
        cfg["start_date"] = args.start
        _save_config(cfg)
        print(f"  📅 Başlangıç tarihi: {args.start}")

    if args.loop > 0:
        run_loop(cfg, args.loop)
    else:
        run_once(cfg)
