"""
BIST Canlı/Paper Signal Runner.

Her yeni mum kapandığında:
  1. Tarihsel veri + yeni mum birleştirilir
  2. İndikatörler hesaplanır
  3. Makro filtre kontrol edilir
  4. Sinyal üretilir
  5. Emir simüle edilir (paper) veya broker'a gönderilir (live)
  6. Telegram bildirimi gönderilir

Kullanım:
    # Paper (yfinance polling):
    from bist.live.data_source import YFinancePollingSource
    from bist.live.live_runner import BistLiveRunner
    import yaml

    cfg = yaml.safe_load(open("bist/config_bist.yaml"))
    runner = BistLiveRunner(cfg, data_source=YFinancePollingSource(interval="1d", poll_seconds=900))
    runner.start()

    # Canlı (broker API — StreamingDataSource sub-class gerekir):
    runner = BistLiveRunner(cfg, data_source=MatriksSource(), live_mode=True)
    runner.start()
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bist.live.data_source import BistDataSource, BarEvent

logger = logging.getLogger(__name__)


class BistLiveRunner:
    """
    Canlı veya paper BIST sinyal motoru.

    data_source değiştirilerek:
      - yfinance polling (paper / dev)
      - broker WebSocket streaming (production)
    arasında geçiş yapılır. Strateji/risk kodu değişmez.
    """

    HISTORY_BARS = 350   # İndikatör ısınması için gereken minimum bar sayısı

    def __init__(
        self,
        cfg: dict,
        data_source: BistDataSource,
        live_mode: bool = False,
    ) -> None:
        self.cfg         = cfg
        self.data_source = data_source
        self.live_mode   = live_mode

        trading_cfg = cfg.get("trading", {})
        risk_cfg    = cfg.get("risk", {})
        strat_cfg   = cfg.get("strategy", {})
        fil_cfg     = cfg.get("filters", {}) or {}
        bt_cfg      = cfg.get("backtest", {})
        pyr_cfg     = cfg.get("pyramiding", {}) or {}
        macro_cfg   = cfg.get("macro", {}) or {}
        tg_cfg      = cfg.get("telegram", {}) or {}

        self.symbols          = trading_cfg.get("symbols", [])
        self.commission_rate  = bt_cfg.get("commission_rate", 0.0005)

        # ── Teknik indikatörler ──────────────────────────────────────────────
        from indicators.technical_indicators import TechnicalIndicators
        self.indicators = TechnicalIndicators()

        # ── Strateji ────────────────────────────────────────────────────────
        from strategy.trend_following_strategy import TrendFollowingStrategy
        self.strategy = TrendFollowingStrategy(
            rsi_lower             = strat_cfg.get("rsi_lower", 45),
            rsi_upper             = strat_cfg.get("rsi_upper", 70),
            adx_threshold         = strat_cfg.get("adx_threshold", 22),
            min_atr_ratio         = strat_cfg.get("min_atr_ratio", 0.005),
            volume_sma_multiplier = strat_cfg.get("volume_sma_multiplier", 0.8),
            choppiness_threshold  = fil_cfg.get("choppiness_threshold", 61.8),
            choppiness_enabled    = fil_cfg.get("choppiness_enabled", True),
            mtf_filter_enabled    = False,
            indicators            = self.indicators,
        )

        # ── Risk ─────────────────────────────────────────────────────────────
        from risk.risk_manager import RiskManager
        self.risk_manager = RiskManager(
            account_balance    = bt_cfg.get("initial_capital", 10_000.0),
            risk_per_trade     = risk_cfg.get("risk_per_trade", 0.01),
            daily_max_loss     = risk_cfg.get("daily_max_loss", 0.03),
            atr_stop_multiplier = risk_cfg.get("atr_stop_multiplier", 3.0),
            max_open_positions = risk_cfg.get("max_open_positions", 4),
            min_order_size     = risk_cfg.get("min_order_size", 10.0),
            max_position_pct   = risk_cfg.get("max_position_pct", 0.15),
        )

        # ── Pozisyon yönetimi ────────────────────────────────────────────────
        from execution.position_manager import PositionManager
        self.position_manager = PositionManager(
            trailing_stop_atr_multiplier=risk_cfg.get("trailing_stop_atr_multiplier", 3.0)
        )

        # ── Makro filtre ─────────────────────────────────────────────────────
        from bist.filters.macro_filter import MacroFilter
        self.macro_filter = MacroFilter(
            regime_filter_enabled   = macro_cfg.get("regime_filter_enabled", True),
            regime_ema_period       = macro_cfg.get("regime_ema_period", 100),
            regime_usd_adjusted     = macro_cfg.get("regime_usd_adjusted", True),
            usdtry_guard_enabled    = macro_cfg.get("usdtry_guard_enabled", True),
            usdtry_momentum_period  = macro_cfg.get("usdtry_momentum_period", 20),
            usdtry_crisis_threshold = macro_cfg.get("usdtry_crisis_threshold", 0.15),
        )
        self._macro_fitted = False

        # ── Telegram ─────────────────────────────────────────────────────────
        try:
            from monitoring.telegram_notifier import TelegramNotifier
            self.notifier = TelegramNotifier(
                enabled = tg_cfg.get("enabled", False),
                token   = tg_cfg.get("token", ""),
                chat_id = tg_cfg.get("chat_id", ""),
            )
        except Exception:
            self.notifier = None

        # ── Per-symbol state ─────────────────────────────────────────────────
        # history cache: son get_history() sonucu
        self._history: dict[str, pd.DataFrame] = {}
        # son kapanan bar timestamp'i (cooldown)
        self._exit_cooldown: dict[str, pd.Timestamp | None] = {s: None for s in self.symbols}
        self._last_day: str | None = None

    # ── Başlat / Durdur ───────────────────────────────────────────────────────

    def start(self) -> None:
        logger.info(f"[BistLiveRunner] {'CANLI' if self.live_mode else 'PAPER'} mod başlatılıyor.")
        logger.info(f"[BistLiveRunner] Sembol sayısı: {len(self.symbols)}")

        self._fit_macro()
        self._preload_history()

        self.data_source.subscribe(self.symbols, self.on_bar)
        self._notify(f"{'🔴 CANLI' if self.live_mode else '📄 PAPER'} BIST runner başladı.\n"
                     f"Semboller: {', '.join(self.symbols[:10])}{'...' if len(self.symbols) > 10 else ''}")
        self.data_source.start()

    def stop(self) -> None:
        self.data_source.stop()
        logger.info("[BistLiveRunner] Durduruldu.")

    # ── Ana callback: her mum kapanışında çağrılır ────────────────────────────

    def on_bar(self, event: BarEvent) -> None:
        """
        Veri kaynağı (polling veya streaming) tarafından çağrılır.
        Tüm strateji/risk/order mantığı burada çalışır.
        """
        sym = event["symbol"]
        ts  = event["timestamp"]
        now = datetime.now(timezone.utc)

        # ── Günlük sıfırlama ─────────────────────────────────────────────────
        day_str = now.strftime("%Y-%m-%d")
        if self._last_day and day_str != self._last_day:
            self.risk_manager.reset_daily_pnl()
            logger.info(f"[BistLiveRunner] Yeni gün: {day_str}")
        self._last_day = day_str

        # ── Makro filtre ─────────────────────────────────────────────────────
        if not self.macro_filter.is_entry_allowed(pd.Timestamp(now)):
            reason = self.macro_filter.get_block_reason(pd.Timestamp(now))
            logger.debug(f"[BistLiveRunner] {sym}: Makro engel — {reason}")
            # Makro engel olsa bile stop kontrolü yapalım
            self._check_stops(sym, event)
            return

        # ── Günlük kayıp limiti ───────────────────────────────────────────────
        if not self.risk_manager.trading_allowed:
            logger.debug(f"[BistLiveRunner] Günlük kayıp limiti — yeni giriş yok.")
            self._check_stops(sym, event)
            return

        # ── Tarihsel veri + yeni mum birleştir ───────────────────────────────
        df = self._get_updated_history(sym, event)
        if df is None or len(df) < 50:
            logger.warning(f"[BistLiveRunner] {sym}: Yetersiz tarihsel veri.")
            return

        # ── İndikatörler ─────────────────────────────────────────────────────
        try:
            df_ind = self.indicators.calculate(df)
        except Exception as e:
            logger.error(f"[BistLiveRunner] {sym}: İndikatör hatası: {e}")
            return

        last_row = df_ind.iloc[-1]
        price    = float(last_row["close"])
        atr      = float(last_row.get("atr", 0.0))

        # ── Stop / Trailing stop güncelle ────────────────────────────────────
        current_prices = {sym: price}
        atrs           = {sym: atr}
        closed_list    = self.position_manager.update_positions(current_prices, atrs)
        for pos, reason in closed_list:
            self._on_position_closed(pos, reason, price)

        # ── Cooldown (aynı mumda tekrar giriş yok) ───────────────────────────
        if self._exit_cooldown.get(sym) == ts:
            return

        # ── Sinyal üret ───────────────────────────────────────────────────────
        from strategy.signal import Side
        try:
            signal = self.strategy.generate_signal(sym, df_ind)
        except Exception as e:
            logger.error(f"[BistLiveRunner] {sym}: Sinyal hatası: {e}")
            return

        if signal.side == Side.BUY:
            if self.position_manager.has_open_position(sym):
                logger.debug(f"[BistLiveRunner] {sym}: Zaten açık pozisyon var.")
                return

            size = self.risk_manager.calculate_position_size(price, atr)
            if size <= 0 or price * size < self.risk_manager.min_order_size:
                logger.debug(f"[BistLiveRunner] {sym}: Çok küçük pozisyon ({size}).")
                return

            from execution.position_manager import Position
            stop = price - self.risk_manager.atr_stop_multiplier * atr
            pos_obj = Position(
                symbol=sym,
                entry_price=price,
                stop_price=stop,
                trailing_stop_price=stop,
                position_size=size,
            )
            self.position_manager.open_position(pos_obj)
            commission = price * size * self.commission_rate
            self.risk_manager.account_balance -= commission

            msg = (
                f"✅ GİRİŞ: {sym}\n"
                f"Fiyat: {price:.4f} USD\n"
                f"Stop: {stop:.4f} USD\n"
                f"Boyut: {size:.4f}\n"
                f"Skor: {signal.confidence_score:.2f}"
            )
            logger.info(f"[BistLiveRunner] {msg.replace(chr(10), ' | ')}")
            self._notify(msg)

            if self.live_mode:
                self._send_live_order(sym, "BUY", size, price)

        elif signal.side == Side.HOLD:
            should_exit, exit_reason = self.strategy.should_exit(sym, df_ind, entry_price=0)
            if should_exit and self.position_manager.has_open_position(sym):
                closed = self.position_manager.close_position(sym, price, exit_reason)
                if closed:
                    self._on_position_closed(closed, exit_reason, price)
                    self._exit_cooldown[sym] = ts

        # ── Bakiye özeti ─────────────────────────────────────────────────────
        bal = self.risk_manager.account_balance
        logger.info(
            f"[BistLiveRunner] {sym} | close={price:.4f} | "
            f"bakiye={bal:.2f} | açık={self.position_manager.open_count}"
        )

    # ── Yardımcı metodlar ────────────────────────────────────────────────────

    def _fit_macro(self) -> None:
        from bist.data.usdtry_provider import USDTRYProvider
        from bist.filters.macro_filter import fetch_xu100
        macro_cfg = self.cfg.get("macro", {})
        try:
            usdtry = USDTRYProvider().fetch(start="2020-01-01", end="2030-01-01")
            xu100  = fetch_xu100(start="2020-01-01", end="2030-01-01") \
                     if macro_cfg.get("regime_filter_enabled", True) else None
            self.macro_filter.fit(
                xu100 if xu100 is not None else pd.Series(dtype=float),
                usdtry,
            )
            self._macro_fitted = True
            logger.info(f"[BistLiveRunner] Makro filtre: {self.macro_filter.summary_stats()}")
        except Exception as e:
            logger.warning(f"[BistLiveRunner] Makro filtre fit edilemedi: {e} — devre dışı.")

    def _preload_history(self) -> None:
        """Başlangıçta tüm sembollerin geçmiş verisini cache'e yükler."""
        logger.info(f"[BistLiveRunner] {len(self.symbols)} sembol geçmişi yükleniyor...")
        for sym in self.symbols:
            try:
                df = self.data_source.get_history(sym, bars=self.HISTORY_BARS)
                self._history[sym] = df
            except Exception as e:
                logger.warning(f"[BistLiveRunner] {sym} geçmiş yüklenemedi: {e}")
        logger.info("[BistLiveRunner] Geçmiş yükleme tamamlandı.")

    def _get_updated_history(self, sym: str, event: BarEvent) -> Optional[pd.DataFrame]:
        """
        Cached geçmişe yeni mum satırını ekler.
        Yeni mum zaten varsa (polling'de aynı timestamp tekrar gelebilir) günceller.
        """
        hist = self._history.get(sym)
        if hist is None or len(hist) < 10:
            try:
                hist = self.data_source.get_history(sym, bars=self.HISTORY_BARS)
                self._history[sym] = hist
            except Exception:
                return None

        # Yeni mum satırı
        new_row = pd.DataFrame(
            [{
                "open":   event["open"],
                "high":   event["high"],
                "low":    event["low"],
                "close":  event["close"],
                "volume": event["volume"],
            }],
            index=pd.DatetimeIndex([event["timestamp"]]),
        )
        new_row.index = new_row.index.tz_localize("UTC") if new_row.index.tz is None else new_row.index

        # Zaten varsa son satırı güncelle, yoksa ekle
        if event["timestamp"] in hist.index:
            hist.loc[event["timestamp"]] = new_row.iloc[0]
        else:
            hist = pd.concat([hist, new_row])

        self._history[sym] = hist
        return hist

    def _check_stops(self, sym: str, event: BarEvent) -> None:
        """Makro engel / günlük limit durumunda bile stop kontrolü çalışır."""
        price = event["close"]
        atr   = 0.0
        try:
            hist = self._history.get(sym)
            if hist is not None and len(hist) > 14:
                df_ind = self.indicators.calculate(hist)
                atr    = float(df_ind.iloc[-1].get("atr", 0.0))
        except Exception:
            pass
        closed_list = self.position_manager.update_positions({sym: price}, {sym: atr})
        for pos, reason in closed_list:
            self._on_position_closed(pos, reason, price)

    def _on_position_closed(self, pos, reason: str, close_price: float) -> None:
        commission  = close_price * pos.position_size * self.commission_rate
        net_pnl     = pos.realized_pnl - commission
        self.risk_manager.record_trade_pnl(net_pnl)
        self.strategy.record_outcome(pos.symbol, net_pnl > 0, pnl=net_pnl)

        pnl_str = f"{'+' if net_pnl >= 0 else ''}{net_pnl:.2f}"
        msg = (
            f"{'🟢' if net_pnl >= 0 else '🔴'} ÇIKIŞ: {pos.symbol}\n"
            f"Neden: {reason}\n"
            f"Fiyat: {close_price:.4f} USD\n"
            f"Net PnL: {pnl_str} USD"
        )
        logger.info(f"[BistLiveRunner] {msg.replace(chr(10), ' | ')}")
        self._notify(msg)

        if self.live_mode:
            self._send_live_order(pos.symbol, "SELL", pos.position_size, close_price)

    def _send_live_order(self, symbol: str, side: str, size: float, price: float) -> None:
        """
        Gerçek broker'a emir gönderir.
        Şu an stub — broker API entegrasyonu buraya eklenir.
        """
        logger.warning(
            f"[BistLiveRunner] CANLI EMİR (STUB): {side} {size:.4f} {symbol} @ ~{price:.4f} USD\n"
            f"  → Broker API entegrasyonu henüz yapılmadı."
        )

    def _notify(self, msg: str) -> None:
        if self.notifier:
            try:
                self.notifier.send(msg)
            except Exception as e:
                logger.debug(f"[BistLiveRunner] Telegram hatası: {e}")

    # ── Durum özeti ──────────────────────────────────────────────────────────

    def status(self) -> dict:
        summary = self.risk_manager.summary()
        positions = []
        for pos in self.position_manager.open_positions:
            positions.append({
                "symbol":     pos.symbol,
                "entry":      pos.entry_price,
                "stop":       pos.stop_price,
                "size":       pos.position_size,
                "unrealized": pos.unrealized_pnl,
            })
        return {
            "mode":            "LIVE" if self.live_mode else "PAPER",
            "account_balance": summary["account_balance"],
            "daily_pnl":       summary["daily_pnl"],
            "open_positions":  positions,
            "trading_allowed": summary["trading_allowed"],
            "macro_fitted":    self._macro_fitted,
        }
