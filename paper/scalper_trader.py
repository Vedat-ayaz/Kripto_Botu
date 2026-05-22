"""
Yüksek Frekanslı Scalper Trader (Paper Mode)
=============================================
- Her 30 saniyede bir döngü çalışır (PaperTrader'ın 60s'sine karşı)
- 5m mumlarla çalışır (1h yerine)
- ScalpingStrategy kullanır — adaptif online öğrenme dahil
- Her kapatılan trade'den sonra strategy.record_trade_result() çağrılır
- Kelly-inspired pozisyon skalası ile daha güçlü coinde büyük lot açılır
- Dashboard'a mode="SCALP" olarak yazar
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from data.exchange_client import ExchangeClient
from data.candle_repository import CandleRepository
from data.market_data_service import MarketDataService
from strategy.scalping_strategy import ScalpingStrategy
from strategy.signal import Side
from risk.risk_manager import RiskManager
from execution.position_manager import PositionManager
from execution.order_manager import OrderManager
from monitoring.telegram_notifier import TelegramNotifier
from dashboard.state import BotStateDB

logger = logging.getLogger(__name__)


class ScalperTrader:
    """
    Gerçek zamanlı 5m scalping trader (paper mod).

    Fark:
      - LOOP_INTERVAL = 30s (iki kat daha hızlı kontrol)
      - 5m timeframe
      - ScalpingStrategy — adaptif eşik + Kelly pozisyon skalası
      - Her trade'de strategy.record_trade_result() çağrılır
    """

    LOOP_INTERVAL = 30   # saniye
    TIMEFRAME     = "5m"

    def __init__(
        self,
        client: ExchangeClient,
        symbols: list[str],
        timeframe: str = "5m",
        initial_capital: float = 10_000.0,
        risk_per_trade: float = 0.01,
        daily_max_loss: float = 0.05,
        max_open_positions: int = 5,
        min_order_size: float = 10.0,
        max_position_pct: float = 0.15,
        profit_target_pct: float = 0.005,
        stop_loss_pct: float = 0.0025,
        max_hold_bars: int = 12,
        notifier: Optional[TelegramNotifier] = None,
    ):
        self.client           = client
        self.symbols          = symbols
        self.timeframe        = timeframe
        self.initial_capital  = initial_capital
        self.notifier         = notifier or TelegramNotifier(enabled=False)

        # Servis katmanları
        self.repo        = CandleRepository()
        self.market_data = MarketDataService(client, self.repo)

        # Scalping stratejisi (adaptif öğrenme dahil)
        self.strategy = ScalpingStrategy(
            profit_target_pct=profit_target_pct,
            stop_loss_pct=stop_loss_pct,
            max_hold_bars=max_hold_bars,
        )

        # Risk / pozisyon yönetimi
        # Scalping için ATR stop'u kullanmıyoruz (TP/SL sabit yüzde bazlı)
        # ama RiskManager.risk_per_trade × Kelly skalası ile lot belirliyoruz
        self.risk_manager = RiskManager(
            account_balance=initial_capital,
            risk_per_trade=risk_per_trade,
            daily_max_loss=daily_max_loss,
            atr_stop_multiplier=1.0,       # scalp'ta kullanılmaz
            max_open_positions=max_open_positions,
            min_order_size=min_order_size,
            max_position_pct=max_position_pct,
        )
        self.position_manager = PositionManager(trailing_stop_atr_multiplier=0.0)
        self.order_manager    = OrderManager(
            position_manager=self.position_manager,
            risk_manager=self.risk_manager,
            live_mode=False,
            trailing_stop_multiplier=0.0,
        )

        # Dashboard
        self.db = BotStateDB()

        # Durum
        self._running   = False
        self._last_day: Optional[str] = None
        # { symbol: bar_timestamp }  — aynı mumda yeniden giriş engeli
        self._exit_cooldown: dict[str, str] = {}
        # Açılışta her pozisyon için giriş yapılan bar timestamp'i
        self._open_bar_ts: dict[str, pd.Timestamp] = {}
        # Son bar timestamp — bar değişim takibi için
        self._last_bar_ts: dict[str, str] = {}
        # Periyodik kaydetme için sayaçlar
        self._last_save_tick: int = 0
        self._closed_since_save: int = 0
        self._SAVE_EVERY_N_TICKS  = 20   # ~10 dakika (20×30s)
        self._SAVE_EVERY_N_TRADES = 5    # her 5 işlemde bir

    # ─────────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        logger.info("[ScalperTrader] SCALP TRADING başladı (5m / 30s döngü).")
        self._load_learning_state()
        self.notifier.bot_started("SCALP")
        self._running = True
        self._write_bot_status()

        try:
            while self._running:
                self._tick()
                time.sleep(self.LOOP_INTERVAL)
        except KeyboardInterrupt:
            logger.info("[ScalperTrader] Kullanıcı tarafından durduruldu.")
        finally:
            self._shutdown()

    # ─────────────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        now     = datetime.now(timezone.utc)
        day_str = now.strftime("%Y-%m-%d")

        # Gece yarısı günlük PnL sıfırla
        if self._last_day and day_str != self._last_day:
            logger.info("[ScalperTrader] Yeni gün, günlük PnL sıfırlandı.")
            self.risk_manager.reset_daily_pnl()
        self._last_day = day_str

        current_prices: dict[str, float] = {}
        atrs: dict[str, float] = {}

        # ── Veri çek ──────────────────────────────────────────────────
        for symbol in self.symbols:
            try:
                df = self.market_data.fetch_and_store(symbol, self.timeframe, limit=200)
                last = df.iloc[-1]
                current_prices[symbol] = float(last["close"])

                # Hızlı ATR hesabı (7 bar) — TP/SL için değil, dashboard için
                if len(df) >= 10:
                    df_c = df.copy()
                    from strategy.scalping_strategy import _atr as _fast_atr
                    df_c["atr7"] = _fast_atr(df_c["high"], df_c["low"], df_c["close"], 7)
                    atrs[symbol] = float(df_c["atr7"].iloc[-1]) if not df_c["atr7"].isna().iloc[-1] else 0.0

                # Bar değişimini takip et
                bar_ts = str(df.index[-1])
                if self._last_bar_ts.get(symbol) != bar_ts:
                    self._last_bar_ts[symbol] = bar_ts

            except Exception as e:
                logger.error(f"[ScalperTrader] {symbol} veri hatası: {e}")
                self.notifier.api_error(str(e))

        # ── Açık pozisyonları güncelle ─────────────────────────────────
        # Scalping'de TP/SL strateji katmanında; PositionManager sadece
        # trailing stop yapar (devre dışı). Biz burada manuel kontrol ediyoruz.
        for pos in list(self.position_manager.open_positions):
            symbol = pos.symbol
            price = current_prices.get(symbol, pos.entry_price)
            pos.update_unrealized_pnl(price)

            df = self.repo.get(symbol, self.timeframe)
            if df is None:
                continue

            open_ts    = self._open_bar_ts.get(symbol, df.index[-1])
            current_ts = df.index[-1]
            should_exit, exit_reason = self.strategy.should_exit(
                symbol, df, pos.entry_price,
                open_ts=open_ts,
                current_ts=current_ts,
            )

            if should_exit:
                closed = self.order_manager.close_position(symbol, price, exit_reason)
                if closed:
                    close_price = closed.close_price or price
                    pnl = closed.realized_pnl
                    logger.info(
                        f"[ScalperTrader] {symbol} KAPATILDI: {exit_reason} | "
                        f"giriş={closed.entry_price:.4f} çıkış={close_price:.4f} "
                        f"pnl={pnl:+.4f}"
                    )
                    self.risk_manager.record_trade_pnl(pnl)

                    # ── Online öğrenme — trade sonucunu stratejiye bildir ──
                    self.strategy.record_trade_result(symbol, pnl)
                    self._closed_since_save += 1

                    # Dashboard
                    self.db.insert_closed_trade(
                        symbol, closed.entry_price, close_price,
                        closed.position_size, pnl, exit_reason,
                        str(closed.opened_at), str(closed.closed_at),
                    )
                    self.db.remove_open_position(symbol)
                    self.db.insert_signal(
                        symbol, "SELL", close_price,
                        0.0, None, None,
                        f"Scalp çıkış: {exit_reason} | PnL: {pnl:+.4f}",
                    )

                    # Cooldown
                    bar_ts = str(df.index[-1])
                    self._exit_cooldown[symbol] = bar_ts
                    self._open_bar_ts.pop(symbol, None)

                    self.notifier.position_closed(symbol, pnl, exit_reason)
                    if "stop_loss" in exit_reason:
                        self.notifier.stop_loss_triggered(symbol, close_price, pnl)

        # ── Günlük limit ────────────────────────────────────────────────
        if not self.risk_manager.trading_allowed:
            summary = self.risk_manager.summary()
            logger.warning(
                f"[ScalperTrader] Günlük zarar limiti! "
                f"PnL={summary['daily_pnl']:+.4f}"
            )
            self.notifier.daily_loss_limit_hit(
                summary["daily_pnl"], summary["daily_loss_limit"]
            )
            self._write_bot_status()
            self._update_open_positions_in_db(current_prices)
            return

        # ── Sinyal üretimi ──────────────────────────────────────────────
        for symbol in self.symbols:
            df = self.repo.get(symbol, self.timeframe)
            if df is None or len(df) < 30:
                continue

            # Zaten bu sembolde açık pozisyon varsa atla
            if self.position_manager.has_open_position(symbol):
                continue

            try:
                signal = self.strategy.generate_signal(symbol, df)

                if signal.side == Side.BUY:
                    bar_ts = str(df.index[-1])

                    # Cooldown: aynı mumda çıkış varsa yeni giriş yapma
                    if self._exit_cooldown.get(symbol) == bar_ts:
                        logger.debug(
                            f"[ScalperTrader] {symbol}: Cooldown aktif, bu mumda giriş yok."
                        )
                        continue

                    # Max pozisyon ve günlük limit kontrolü
                    if self.position_manager.open_count >= self.risk_manager.max_open_positions:
                        logger.debug(f"[ScalperTrader] Max açık pozisyon ({self.risk_manager.max_open_positions}), atlanıyor.")
                        continue

                    position = self._open_scalp_position(signal)
                    if position:
                        self._open_bar_ts[symbol] = df.index[-1]
                        kelly_scale = self.strategy.get_position_scale(symbol)
                        logger.info(
                            f"[ScalperTrader] {symbol} GİRİŞ | "
                            f"fiyat={position.entry_price:.4f} "
                            f"lot={position.position_size:.6f} "
                            f"KellySkala={kelly_scale:.2f} "
                            f"skor={signal.confidence_score:.3f}"
                        )
                        self.db.insert_signal(
                            symbol, "BUY", signal.price,
                            signal.confidence_score, signal.rsi, signal.adx,
                            signal.reason,
                        )
                        self.notifier.new_signal(
                            symbol, "BUY", signal.price, signal.reason
                        )
                        self.notifier.position_opened(
                            symbol, position.entry_price,
                            position.stop_price, position.position_size,
                        )

            except Exception as e:
                logger.error(f"[ScalperTrader] {symbol} sinyal hatası: {e}")

        # ── Dashboard güncellemeleri ─────────────────────────────────────
        self._write_bot_status()
        self._update_open_positions_in_db(current_prices)
        self.db.insert_equity_point(self.risk_manager.account_balance)
        self._log_status()

    # ─────────────────────────────────────────────────────────────────────────

    def _open_scalp_position(self, signal) -> "Optional[Position]":
        """
        Scalp'a özel pozisyon açma: ATR yerine sabit stop_loss_pct kullanır.
        Kelly skalası ile lot büyüklüğü ayarlanır.
        """
        from execution.position_manager import Position

        symbol      = signal.symbol
        entry_price = signal.price
        kelly_scale = self.strategy.get_position_scale(symbol)

        # Sabit % stop → risk_per_unit hesabı
        stop_loss_pct = self.strategy.stop_loss_pct        # örn. 0.0025
        risk_per_unit = entry_price * stop_loss_pct
        if risk_per_unit <= 0:
            return None

        # Kelly-adjusted risk amount
        risk_amount   = self.risk_manager.account_balance * self.risk_manager.risk_per_trade * kelly_scale
        position_size = risk_amount / risk_per_unit

        # Sermaye limiti
        max_size      = (self.risk_manager.account_balance * self.risk_manager.max_position_pct) / entry_price
        position_size = min(position_size, max_size)

        order_value = position_size * entry_price
        if order_value < self.risk_manager.min_order_size:
            logger.debug(
                f"[ScalperTrader] {symbol}: Pozisyon çok küçük "
                f"({order_value:.2f} < {self.risk_manager.min_order_size}), atlanıyor."
            )
            return None

        stop_price = entry_price * (1 - stop_loss_pct)

        position = Position(
            symbol=symbol,
            entry_price=entry_price,
            position_size=position_size,
            stop_price=stop_price,
            trailing_stop_price=stop_price,
            opened_at=signal.timestamp,
        )
        self.position_manager.open_position(position)
        return position

    def _write_bot_status(self) -> None:
        summary   = self.risk_manager.summary()
        total_pnl = summary["account_balance"] - self.initial_capital
        self.db.update_bot_status(
            mode="SCALP",
            running=self._running,
            account_balance=summary["account_balance"],
            initial_balance=self.initial_capital,
            daily_pnl=summary["daily_pnl"],
            total_pnl=total_pnl,
            trading_allowed=summary["trading_allowed"],
        )

    def _update_open_positions_in_db(self, current_prices: dict) -> None:
        for pos in self.position_manager.open_positions:
            price = current_prices.get(pos.symbol, pos.entry_price)
            pos.update_unrealized_pnl(price)
            self.db.upsert_open_position(
                symbol=pos.symbol,
                entry_price=pos.entry_price,
                position_size=pos.position_size,
                stop_price=pos.stop_price,
                trailing_stop_price=pos.trailing_stop_price,
                unrealized_pnl=pos.unrealized_pnl,
                cost_basis=pos.cost_basis,
                opened_at=str(pos.opened_at),
            )

    def _log_status(self) -> None:
        summary = self.risk_manager.summary()
        stats   = self.strategy.learning_stats
        total_trades = sum(v["trades"] for v in stats.values()) if stats else 0
        logger.info(
            f"[ScalperTrader] Bakiye={summary['account_balance']:.2f} | "
            f"GünlükPnL={summary['daily_pnl']:+.4f} | "
            f"AçıkPoz={self.position_manager.open_count} | "
            f"ToplTradeAdaptif={total_trades}"
        )
        # Per-symbol öğrenme durumunu dashboard DB'ye yaz
        for sym, s in stats.items():
            try:
                self.db.upsert_scalp_learn(
                    symbol=sym,
                    threshold=s["threshold"],
                    win_rate=s["win_rate"],
                    pos_scale=s["pos_scale"],
                    trade_count=s["trades"],
                )
            except Exception as e:
                logger.debug(f"[ScalperTrader] scalp_learn yazma hatası {sym}: {e}")

        # Periyodik kaydetme — tick bazlı VEYA trade bazlı
        self._last_save_tick += 1
        # Her 40 tick'te (~20 dakika) threshold decay çalıştır
        if self._last_save_tick % 40 == 0:
            self.strategy.maybe_decay_thresholds()
        should_save = (
            self._last_save_tick >= self._SAVE_EVERY_N_TICKS or
            self._closed_since_save >= self._SAVE_EVERY_N_TRADES
        )
        if should_save and stats:
            self._save_learning_state()
            self._last_save_tick  = 0
            self._closed_since_save = 0

    def _load_learning_state(self) -> None:
        """Bot başlarken kaydedilmiş öğrenme durumunu yükler."""
        saved = self.db.load_scalp_state()
        if saved:
            age_info = f" (kaydedilme: {saved['saved_at'][:16]})" if saved.get("saved_at") else ""
            logger.info(f"[ScalperTrader] Öğrenme durumu geri yükleniyor{age_info}...")
            self.strategy.load_state(saved["state"])
        else:
            logger.info("[ScalperTrader] Kayıtlı öğrenme durumu bulunamadı, sıfırdan başlanıyor.")

    def _save_learning_state(self) -> None:
        """Öğrenme durumunu DB'ye kaydeder."""
        try:
            state = self.strategy.save_state()
            self.db.save_scalp_state(state)
        except Exception as e:
            logger.warning(f"[ScalperTrader] Öğrenme durumu kaydedilemedi: {e}")

    def _shutdown(self) -> None:
        logger.info("[ScalperTrader] Kapatılıyor...")
        # Öğrenme durumunu kaydet (bir sonraki açılışta devam etsin)
        self._save_learning_state()
        # Öğrenme istatistiklerini logla
        stats = self.strategy.learning_stats
        if stats:
            logger.info("[ScalperTrader] Adaptif Öğrenme Özeti:")
            for sym, s in sorted(stats.items()):
                logger.info(
                    f"  {sym}: WR={s['win_rate']:.1%} | "
                    f"Eşik={s['threshold']:.3f} | "
                    f"Skala={s['pos_scale']:.2f}× | "
                    f"Trade={s['trades']}"
                )
        self.position_manager.report_open_positions()
        self._running = False
        self._write_bot_status()
