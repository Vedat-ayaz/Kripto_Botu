import logging
import time
from typing import Optional

from data.exchange_client import ExchangeClient
from data.candle_repository import CandleRepository
from data.market_data_service import MarketDataService
from indicators.technical_indicators import TechnicalIndicators
from strategy.trend_following_strategy import TrendFollowingStrategy
from strategy.signal import Side
from risk.risk_manager import RiskManager
from execution.position_manager import PositionManager
from execution.order_manager import OrderManager
from monitoring.telegram_notifier import TelegramNotifier
from dashboard.state import BotStateDB
from learning.parameter_optimizer import ParameterOptimizer

logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Gerçek zamanlı fiyatları takip eden paper trading modu.
    - Dashboard'a SQLite üzerinden state yazar
    - Walk-forward adaptasyon ile parametrelerini iyileştirir
    """

    LOOP_INTERVAL = 60

    def __init__(
        self,
        client: ExchangeClient,
        symbols: list[str],
        timeframe: str = "1h",
        initial_capital: float = 10_000.0,
        risk_per_trade: float = 0.01,
        daily_max_loss: float = 0.03,
        atr_stop_multiplier: float = 2.0,
        trailing_stop_atr_multiplier: float = 2.5,
        adx_threshold: float = 20.0,
        rsi_lower: float = 45.0,
        rsi_upper: float = 70.0,
        min_atr_ratio: float = 0.002,
        max_open_positions: int = 3,
        min_order_size: float = 10.0,
        max_position_pct: float = 0.20,
        volume_sma_multiplier: float = 0.8,
        notifier: Optional[TelegramNotifier] = None,
        adaptation_window: int = 20,
        # Pyramiding (Stage 1) — Turtle/Clenow
        pyramid_enabled: bool = False,
        pyramid_thresholds_atr: Optional[list[float]] = None,
        pyramid_size_pcts: Optional[list[float]] = None,
        pyramid_max_adds: int = 2,
        pyramid_stop_atr_multiplier: float = 2.0,
    ):
        self.client = client
        self.symbols = symbols
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.notifier = notifier or TelegramNotifier(enabled=False)
        self.min_atr_ratio = min_atr_ratio
        self.trailing_stop_atr_multiplier = trailing_stop_atr_multiplier

        # Adaptive learning — strateji parametrelerini dinamik tutar
        self._optimizer = ParameterOptimizer(
            initial_params={
                "adx_threshold": adx_threshold,
                "rsi_lower": rsi_lower,
                "rsi_upper": rsi_upper,
                "atr_stop_multiplier": atr_stop_multiplier,
                "min_atr_ratio": min_atr_ratio,
                "trailing_stop_atr_multiplier": trailing_stop_atr_multiplier,
                "volume_sma_multiplier": volume_sma_multiplier,
            },
            adaptation_window=adaptation_window,
            on_params_updated=self._on_params_updated,
        )

        # Servis katmanları
        self.repo = CandleRepository()
        self.market_data = MarketDataService(client, self.repo)
        self.indicators = TechnicalIndicators()

        # Strateji ve risk — optimizer'dan güncel parametreler alır
        self._volume_sma_multiplier = volume_sma_multiplier
        self.strategy = self._build_strategy()
        self.risk_manager = RiskManager(
            account_balance=initial_capital,
            risk_per_trade=risk_per_trade,
            daily_max_loss=daily_max_loss,
            atr_stop_multiplier=atr_stop_multiplier,
            max_open_positions=max_open_positions,
            min_order_size=min_order_size,
            max_position_pct=max_position_pct,
        )
        self.position_manager = PositionManager(
            trailing_stop_atr_multiplier=trailing_stop_atr_multiplier
        )
        self.order_manager = OrderManager(
            position_manager=self.position_manager,
            risk_manager=self.risk_manager,
            live_mode=False,
            trailing_stop_multiplier=trailing_stop_atr_multiplier,
        )

        # Pyramiding parametreleri
        self._pyramid_enabled = pyramid_enabled
        self._pyramid_thresholds_atr = pyramid_thresholds_atr or [1.5, 3.0]
        self._pyramid_size_pcts = pyramid_size_pcts or [0.5, 0.25]
        self._pyramid_max_adds = pyramid_max_adds
        self._pyramid_stop_atr_multiplier = pyramid_stop_atr_multiplier

        # Dashboard state DB
        self.db = BotStateDB()

        self._running = False
        self._last_day: Optional[str] = None
        # Pozisyon kapandıktan sonra aynı mumda yeniden giriş engeli
        # { symbol: son_kapanan_mum_timestamp }
        self._exit_cooldown: dict[str, str] = {}

    def _build_strategy(self) -> TrendFollowingStrategy:
        """Optimizer'daki güncel parametrelerle yeni strateji nesnesi oluşturur."""
        p = self._optimizer.current_params
        return TrendFollowingStrategy(
            rsi_lower=p["rsi_lower"],
            rsi_upper=p["rsi_upper"],
            adx_threshold=p["adx_threshold"],
            min_atr_ratio=p.get("min_atr_ratio", self.min_atr_ratio),
            volume_sma_multiplier=p.get("volume_sma_multiplier", self._volume_sma_multiplier),
            indicators=self.indicators,
        )

    def _on_params_updated(
        self, old_params: dict, new_params: dict,
        old_sharpe: float, new_sharpe: float,
    ) -> None:
        """Parametreler güncellendiğinde stratejiyi yeniden inşa eder ve loglar."""
        self.strategy = self._build_strategy()
        # ATR stop'u da güncelle
        self.risk_manager.atr_stop_multiplier = new_params["atr_stop_multiplier"]

        # Dashboard'a yaz
        self.db.insert_adaptive_log(
            symbol="ALL",
            old_params=old_params,
            new_params=new_params,
            old_sharpe=old_sharpe,
            new_sharpe=new_sharpe,
            trade_count=self._optimizer.trade_count,
        )
        self.notifier.send(
            f"🧠 <b>Parametre Adaptasyonu</b>\n"
            f"Sharpe: {old_sharpe:.3f} → {new_sharpe:.3f}\n"
            f"ADX: {old_params['adx_threshold']} → {new_params['adx_threshold']}\n"
            f"RSI: [{old_params['rsi_lower']}-{old_params['rsi_upper']}] → "
            f"[{new_params['rsi_lower']}-{new_params['rsi_upper']}]\n"
            f"ATR mult: {old_params['atr_stop_multiplier']} → {new_params['atr_stop_multiplier']}"
        )
        logger.info(f"[PaperTrader] Strateji parametreleri güncellendi ve yeniden yüklendi.")

    def start(self) -> None:
        logger.info("[PaperTrader] PAPER TRADING başladı.")
        self.notifier.bot_started("PAPER")
        self._running = True

        # Dashboard'a başlangıç state'i yaz
        self._write_bot_status()

        try:
            while self._running:
                self._tick()
                time.sleep(self.LOOP_INTERVAL)
        except KeyboardInterrupt:
            logger.info("[PaperTrader] Kullanıcı tarafından durduruldu.")
        finally:
            self._shutdown()

    def _tick(self) -> None:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        day_str = now.strftime("%Y-%m-%d")
        if self._last_day and day_str != self._last_day:
            logger.info("[PaperTrader] Yeni gün, günlük PnL sıfırlandı.")
            self.risk_manager.reset_daily_pnl()
        self._last_day = day_str

        current_prices: dict[str, float] = {}
        atrs: dict[str, float] = {}

        # ── BTC Rejim Güncellemesi (her tick) ────────────────────────
        try:
            btc_df = self.market_data.fetch_and_store("BTC/USDT", self.timeframe, limit=500)
            btc_close   = btc_df["close"]
            btc_daily   = btc_close.resample("1D").last().dropna()
            btc_sma200  = btc_daily.rolling(200, min_periods=50).mean()
            if len(btc_sma200.dropna()) > 0:
                last_close = float(btc_daily.iloc[-1])
                last_sma   = float(btc_sma200.dropna().iloc[-1])
                self.strategy.set_btc_regime(last_close > last_sma)
        except Exception as _e:
            logger.debug(f"[PaperTrader] BTC rejim hesaplanamadı: {_e}")

        # Stage 3: Bar başı adaptif pause sayaç ilerletme (tüm semboller)
        self.strategy.tick_pause()

        for symbol in self.symbols:
            try:
                df = self.market_data.fetch_and_store(symbol, self.timeframe, limit=500)
                df_ind = self.indicators.calculate(df)
                last = df_ind.iloc[-1]
                current_prices[symbol] = last["close"]
                atrs[symbol] = last.get("atr", 0.0)
                # Optimizer'a en son veriyi ver
                self._optimizer.update_candle_data(symbol, df_ind)
            except Exception as e:
                logger.error(f"[PaperTrader] {symbol} veri çekme hatası: {e}")
                self.notifier.api_error(str(e))
                continue

        # Açık pozisyonları güncelle (stop / trailing stop kontrolü)
        closed_list = self.position_manager.update_positions(current_prices, atrs)
        for pos, reason in closed_list:
            close_price = pos.close_price or current_prices.get(pos.symbol, pos.entry_price)
            logger.info(f"[PaperTrader] {pos.symbol} KAPATILDI: {reason} | pnl={pos.realized_pnl:+.4f}")
            self.risk_manager.record_trade_pnl(pos.realized_pnl)
            # Rolling performans güncelle
            total_pnl = pos.realized_pnl + getattr(pos, "realized_pnl_partial", 0.0)
            self.strategy.record_outcome(pos.symbol, total_pnl > 0, pnl=total_pnl)
            self.db.insert_closed_trade(
                pos.symbol, pos.entry_price, close_price,
                pos.position_size, pos.realized_pnl, reason,
                str(pos.opened_at), str(pos.closed_at),
            )
            self.db.remove_open_position(pos.symbol)
            # Sinyal loguna SELL olarak yaz — dashboard'da görünsün
            self.db.insert_signal(
                pos.symbol, "SELL", close_price,
                0.0, None, None, f"Çıkış: {reason} | PnL: {pos.realized_pnl:+.4f}",
            )
            # Aynı mumda yeniden giriş yapma (cooldown)
            last_bar_ts = str(self.repo.get(pos.symbol, self.timeframe).index[-1]) \
                if self.repo.has_data(pos.symbol, self.timeframe) else ""
            self._exit_cooldown[pos.symbol] = last_bar_ts

            self._optimizer.record_trade(pos.symbol)
            self.notifier.position_closed(pos.symbol, pos.realized_pnl, reason)
            if "stop" in reason.lower():
                self.notifier.stop_loss_triggered(pos.symbol, close_price, pos.realized_pnl)

        # Günlük limit
        if not self.risk_manager.trading_allowed:
            summary = self.risk_manager.summary()
            logger.warning(f"[PaperTrader] Günlük zarar limiti. PnL: {summary['daily_pnl']:+.4f}")
            self.notifier.daily_loss_limit_hit(summary["daily_pnl"], summary["daily_loss_limit"])
            self._write_bot_status()
            self._update_open_positions_in_db(current_prices)
            return

        # ── Cross-sectional momentum sıralaması ──────────────────────────
        # Tüm sembollerin momentum skorunu hesapla, en güçlü TOP_N'i önceliklendir
        TOP_N_MOMENTUM = 8   # Evrenden en iyi 8 sembol sinyal üretebilir
        momentum_scores: dict[str, float] = {}
        for sym in self.symbols:
            df_m = self.repo.get(sym, self.timeframe)
            if df_m is not None and len(df_m) >= 24:
                try:
                    df_m_ind = self.indicators.calculate(df_m)
                    momentum_scores[sym] = self.strategy.get_momentum_rank_score(df_m_ind)
                except Exception:
                    momentum_scores[sym] = 0.0
        # Momentum sıralaması — yüksek skor = güçlü momentum
        ranked_symbols = sorted(momentum_scores, key=lambda s: momentum_scores[s], reverse=True)
        top_symbols = set(ranked_symbols[:TOP_N_MOMENTUM])

        # Her sembol için sinyal
        for symbol in self.symbols:
            df = self.repo.get(symbol, self.timeframe)
            if df is None:
                continue
            try:
                df_ind = self.indicators.calculate(df)
                # Multi-timeframe (4h) trend sütunlarını ekle
                try:
                    df_ind = self.indicators.add_higher_timeframe(df_ind, htf_rule="4h")
                except Exception as _e:
                    logger.debug(f"[PaperTrader] {symbol} MTF eklenemedi: {_e}")
                current_bar_ts = str(df_ind.index[-1])

                # ── Pyramid (Turtle/Clenow) — açık pozisyon kâra geçtikçe ekle ──
                # Sinyal üretiminden ÖNCE, mevcut pozisyon kâr eşiklerini kontrol et.
                # Adaptif gate: bu coin'in mevcut karakteri pyramid'e uygun mu?
                added = None
                if self._pyramid_enabled and self.position_manager.has_open_position(symbol):
                    cur_price = current_prices.get(symbol, 0.0)
                    cur_atr = atrs.get(symbol, 0.0)
                    last_row = df_ind.iloc[-1]
                    gate_ok, gate_reason = self.strategy.should_allow_pyramid(symbol, last_row)
                    if cur_price > 0 and cur_atr > 0 and gate_ok:
                        added = self.order_manager.process_pyramid_add(
                            symbol, cur_price, cur_atr,
                            self._pyramid_thresholds_atr,
                            self._pyramid_size_pcts,
                            self._pyramid_max_adds,
                            stop_atr_multiplier=self._pyramid_stop_atr_multiplier,
                        )
                    elif not gate_ok:
                        logger.debug(f"[PaperTrader] {symbol} pyramid gate kapalı: {gate_reason}")
                        if added:
                            self.notifier.new_signal(
                                symbol, "PYRAMID", cur_price,
                                f"Pyramid #{added.pyramid_adds_count} eklendi "
                                f"(avg={added.avg_entry_price:.4f}, size={added.position_size:.6f})",
                            )

                signal = self.strategy.generate_signal(symbol, df_ind)

                if signal.side == Side.BUY:
                    # Cooldown kontrolü: bu mumda daha önce çıkış yapıldıysa giriş yapma
                    if self._exit_cooldown.get(symbol) == current_bar_ts:
                        logger.debug(f"[PaperTrader] {symbol}: Aynı mumda çıkış yapıldı, yeniden giriş bekleniyor.")
                        continue

                    # Cross-sectional filtre: düşük momentumlu semboller atlanır
                    if symbol not in top_symbols:
                        mom_rank = ranked_symbols.index(symbol) + 1
                        logger.debug(f"[PaperTrader] {symbol}: Cross-sectional filtre — momentum sırası {mom_rank}/{len(ranked_symbols)}")
                        continue

                    logger.info(f"[PaperTrader] {signal} | MomSkor={momentum_scores.get(symbol,0):.3f}")
                    self.db.insert_signal(
                        symbol, signal.side.value, signal.price,
                        signal.confidence_score, signal.rsi, signal.adx, signal.reason,
                    )
                    self.notifier.new_signal(symbol, signal.side.value, signal.price, signal.reason)
                    atr = atrs.get(symbol, 0.0)
                    position = self.order_manager.process_signal(signal, atr)
                    if position:
                        self.notifier.position_opened(
                            symbol, position.entry_price, position.stop_price, position.position_size
                        )

                elif signal.side == Side.HOLD:
                    should_exit, exit_reason = self.strategy.should_exit(symbol, df_ind, entry_price=0)
                    if should_exit and self.position_manager.has_open_position(symbol):
                        price = current_prices.get(symbol, 0.0)
                        closed = self.order_manager.close_position(symbol, price, exit_reason)
                        if closed:
                            close_price = closed.close_price or price
                            self.db.insert_closed_trade(
                                closed.symbol, closed.entry_price, close_price,
                                closed.position_size, closed.realized_pnl, exit_reason,
                                str(closed.opened_at), str(closed.closed_at),
                            )
                            self.db.remove_open_position(symbol)
                            # Sinyal loguna SELL yaz
                            self.db.insert_signal(
                                symbol, "SELL", close_price,
                                0.0, None, None, f"Çıkış: {exit_reason} | PnL: {closed.realized_pnl:+.4f}",
                            )
                            self._exit_cooldown[symbol] = current_bar_ts
                            self._optimizer.record_trade(symbol)
                            # Rolling performans güncelle
                            total_pnl_c = closed.realized_pnl + getattr(closed, "realized_pnl_partial", 0.0)
                            self.strategy.record_outcome(symbol, total_pnl_c > 0, pnl=total_pnl_c)
                            self.notifier.position_closed(symbol, closed.realized_pnl, exit_reason)

            except Exception as e:
                logger.error(f"[PaperTrader] {symbol} sinyal hatası: {e}")

        # Dashboard güncellemeleri
        self._write_bot_status()
        self._update_open_positions_in_db(current_prices)
        self.db.insert_equity_point(self.risk_manager.account_balance)
        self._log_status()

    def _write_bot_status(self) -> None:
        summary = self.risk_manager.summary()
        total_pnl = summary["account_balance"] - self.initial_capital
        self.db.update_bot_status(
            mode="PAPER",
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
        adapting = " [Adapting...]" if self._optimizer.is_adapting else ""
        logger.info(
            f"[PaperTrader] Bakiye={summary['account_balance']:.2f} | "
            f"GünlükPnL={summary['daily_pnl']:+.4f} | "
            f"AçıkPoz={self.position_manager.open_count} | "
            f"İşlem#{self._optimizer.trade_count}{adapting}"
        )

    def _shutdown(self) -> None:
        logger.info("[PaperTrader] Kapatılıyor...")
        self.position_manager.report_open_positions()
        self._running = False
        self._write_bot_status()
