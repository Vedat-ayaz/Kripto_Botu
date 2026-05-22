"""
SQLite tabanlı paylaşımlı durum yöneticisi.
Bot yazar, dashboard okur. Thread-safe.
"""

import sqlite3
import json
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

DB_PATH = str(Path(__file__).resolve().parent / "bot_state.db")


class BotStateDB:
    """
    Bot ve dashboard arasındaki tek iletişim noktası.
    Bot her tick'te state'i günceller, Streamlit dashboard bunu okur.
    """

    _lock = threading.Lock()

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = str(Path(db_path).expanduser())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS bot_status (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    mode TEXT,
                    running INTEGER,
                    account_balance REAL,
                    initial_balance REAL,
                    daily_pnl REAL,
                    total_pnl REAL,
                    trading_allowed INTEGER,
                    last_update TEXT
                );

                CREATE TABLE IF NOT EXISTS open_positions (
                    symbol TEXT PRIMARY KEY,
                    entry_price REAL,
                    position_size REAL,
                    stop_price REAL,
                    trailing_stop_price REAL,
                    unrealized_pnl REAL,
                    cost_basis REAL,
                    opened_at TEXT
                );

                CREATE TABLE IF NOT EXISTS closed_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    position_size REAL,
                    realized_pnl REAL,
                    close_reason TEXT,
                    opened_at TEXT,
                    closed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS equity_curve (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    balance REAL
                );

                CREATE TABLE IF NOT EXISTS signal_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    side TEXT,
                    price REAL,
                    confidence REAL,
                    rsi REAL,
                    adx REAL,
                    reason TEXT
                );

                CREATE TABLE IF NOT EXISTS adaptive_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    old_params TEXT,
                    new_params TEXT,
                    old_sharpe REAL,
                    new_sharpe REAL,
                    trade_count INTEGER
                );

                CREATE TABLE IF NOT EXISTS scalp_learn (
                    symbol TEXT PRIMARY KEY,
                    threshold REAL,
                    win_rate REAL,
                    pos_scale REAL,
                    trade_count INTEGER,
                    updated_at TEXT
                );
            """)

    # ------------------------------------------------------------------ #
    #  BOT YAZAR
    # ------------------------------------------------------------------ #

    def update_bot_status(
        self,
        mode: str,
        running: bool,
        account_balance: float,
        initial_balance: float,
        daily_pnl: float,
        total_pnl: float,
        trading_allowed: bool,
    ) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO bot_status (id, mode, running, account_balance, initial_balance,
                        daily_pnl, total_pnl, trading_allowed, last_update)
                    VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        mode=excluded.mode, running=excluded.running,
                        account_balance=excluded.account_balance,
                        initial_balance=excluded.initial_balance,
                        daily_pnl=excluded.daily_pnl, total_pnl=excluded.total_pnl,
                        trading_allowed=excluded.trading_allowed,
                        last_update=excluded.last_update
                """, (mode, int(running), account_balance, initial_balance,
                      daily_pnl, total_pnl, int(trading_allowed),
                      datetime.now(timezone.utc).isoformat()))

    def upsert_open_position(
        self, symbol: str, entry_price: float, position_size: float,
        stop_price: float, trailing_stop_price: float, unrealized_pnl: float,
        cost_basis: float, opened_at: str,
    ) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO open_positions VALUES (?,?,?,?,?,?,?,?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        entry_price=excluded.entry_price,
                        position_size=excluded.position_size,
                        stop_price=excluded.stop_price,
                        trailing_stop_price=excluded.trailing_stop_price,
                        unrealized_pnl=excluded.unrealized_pnl,
                        cost_basis=excluded.cost_basis,
                        opened_at=excluded.opened_at
                """, (symbol, entry_price, position_size, stop_price,
                      trailing_stop_price, unrealized_pnl, cost_basis, opened_at))

    def remove_open_position(self, symbol: str) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute("DELETE FROM open_positions WHERE symbol = ?", (symbol,))

    def insert_closed_trade(
        self, symbol: str, entry_price: float, exit_price: float,
        position_size: float, realized_pnl: float, close_reason: str,
        opened_at: str, closed_at: str,
    ) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO closed_trades
                    (symbol, entry_price, exit_price, position_size, realized_pnl,
                     close_reason, opened_at, closed_at)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (symbol, entry_price, exit_price, position_size, realized_pnl,
                      close_reason, opened_at, closed_at))

    def insert_equity_point(self, balance: float) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO equity_curve (timestamp, balance) VALUES (?, ?)",
                    (datetime.now(timezone.utc).isoformat(), balance)
                )

    def insert_signal(
        self, symbol: str, side: str, price: float, confidence: float,
        rsi: Optional[float], adx: Optional[float], reason: str,
    ) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO signal_log (timestamp, symbol, side, price, confidence, rsi, adx, reason)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (datetime.now(timezone.utc).isoformat(), symbol, side, price,
                      confidence, rsi, adx, reason))

    def save_scalp_state(self, state: dict) -> None:
        """
        ScalpingStrategy'nin tam öğrenme durumunu JSON olarak saklar.
        Bot yeniden başlatıldığında load_scalp_state() ile geri yüklenebilir.
        """
        with self._lock:
            with self._conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS scalp_state (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        state_json TEXT,
                        saved_at TEXT
                    )
                """)
                conn.execute("""
                    INSERT INTO scalp_state (id, state_json, saved_at)
                    VALUES (1, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        state_json=excluded.state_json,
                        saved_at=excluded.saved_at
                """, (json.dumps(state), datetime.now(timezone.utc).isoformat()))

    def load_scalp_state(self) -> Optional[dict]:
        """
        Kaydedilmiş öğrenme durumunu döner. Kayıt yoksa None.
        """
        try:
            with self._conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS scalp_state (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        state_json TEXT,
                        saved_at TEXT
                    )
                """)
                row = conn.execute(
                    "SELECT state_json, saved_at FROM scalp_state WHERE id = 1"
                ).fetchone()
                if row:
                    return {"state": json.loads(row["state_json"]), "saved_at": row["saved_at"]}
        except Exception:
            pass
        return None

    def upsert_scalp_learn(
        self,
        symbol: str,
        threshold: float,
        win_rate: float,
        pos_scale: float,
        trade_count: int,
    ) -> None:
        """Scalp modunun per-symbol adaptif öğrenme durumunu günceller."""
        with self._lock:
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO scalp_learn (symbol, threshold, win_rate, pos_scale, trade_count, updated_at)
                    VALUES (?,?,?,?,?,?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        threshold=excluded.threshold,
                        win_rate=excluded.win_rate,
                        pos_scale=excluded.pos_scale,
                        trade_count=excluded.trade_count,
                        updated_at=excluded.updated_at
                """, (symbol, threshold, win_rate, pos_scale, trade_count,
                      datetime.now(timezone.utc).isoformat()))

    def insert_adaptive_log(
        self, symbol: str, old_params: dict, new_params: dict,
        old_sharpe: float, new_sharpe: float, trade_count: int,
    ) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO adaptive_log
                    (timestamp, symbol, old_params, new_params, old_sharpe, new_sharpe, trade_count)
                    VALUES (?,?,?,?,?,?,?)
                """, (datetime.now(timezone.utc).isoformat(), symbol,
                      json.dumps(old_params), json.dumps(new_params),
                      old_sharpe, new_sharpe, trade_count))

    # ------------------------------------------------------------------ #
    #  DASHBOARD OKUR
    # ------------------------------------------------------------------ #

    def get_bot_status(self) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM bot_status WHERE id = 1").fetchone()
            return dict(row) if row else None

    def get_open_positions(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM open_positions ORDER BY opened_at DESC").fetchall()
            return [dict(r) for r in rows]

    def get_closed_trades(self, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM closed_trades ORDER BY closed_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_equity_curve(self, limit: int = 1000) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM equity_curve ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in reversed(rows)]

    def get_signal_log(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM signal_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_scalp_learn(self) -> list[dict]:
        """Scalp modunun per-symbol öğrenme durumunu döner, en çok işlem yapılana göre sıralı."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM scalp_learn ORDER BY trade_count DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_adaptive_log(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM adaptive_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_trade_stats(self) -> dict:
        with self._conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN realized_pnl <= 0 THEN 1 ELSE 0 END) as losses,
                    SUM(realized_pnl) as total_pnl,
                    AVG(CASE WHEN realized_pnl > 0 THEN realized_pnl END) as avg_win,
                    AVG(CASE WHEN realized_pnl <= 0 THEN realized_pnl END) as avg_loss,
                    MAX(realized_pnl) as best_trade,
                    MIN(realized_pnl) as worst_trade
                FROM closed_trades
            """).fetchone()
            d = dict(row)
            d["win_rate"] = (d["wins"] / d["total"] * 100) if d["total"] > 0 else 0.0
            return d
