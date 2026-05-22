"""
BIST 100 USD Paper Trader.

15 dakikada bir yfinance'ten gerçek zamanlı (15dk gecikmeli) fiyat çeker.
BIST piyasa saatlerini denetler — piyasa kapalıysa döngü atlanır.
Gerçek emir gönderilmez; simüle edilmiş pozisyon defteri tutar.
"""
import logging
import time as _time
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

POLL_INTERVAL_SEC = 60 * 15   # 15 dakika


class BistPaperTrader:
    """
    BIST sembollerini polling ile izler ve trend-following stratejisi uygular.
    Tüm pozisyon/risk mantığı mevcut bot modüllerinden reuse edilir.
    """

    def __init__(self, cfg: dict):
        import os, sys
        _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _ROOT not in sys.path:
            sys.path.insert(0, _ROOT)

        from bist.data.yfinance_provider import YFinanceProvider
        from bist.data.usdtry_provider import USDTRYProvider
        from bist.adapters.price_converter import PriceConverter
        from bist.adapters.market_hours import is_market_open
        from bist.filters.macro_filter import MacroFilter, fetch_xu100
        from indicators.technical_indicators import TechnicalIndicators
        from strategy.trend_following_strategy import TrendFollowingStrategy
        from risk.risk_manager import RiskManager
        from execution.position_manager import PositionManager
        from strategy.signal import Side

        self.cfg         = cfg
        self.symbols     = cfg["trading"]["symbols"]
        self.Side        = Side
        self.is_market_open = is_market_open

        risk_cfg  = cfg.get("risk", {})
        strat_cfg = cfg.get("strategy", {})
        bt_cfg    = cfg.get("backtest", {})
        pyr_cfg   = cfg.get("pyramiding", {}) or {}
        fil_cfg   = cfg.get("filters", {}) or {}

        self.provider  = YFinanceProvider(cache_enabled=True)
        self.usdtry_p  = USDTRYProvider(cache_enabled=True)
        self.converter = PriceConverter(mode=cfg.get("data", {}).get("usd_mode", "convert_series"))

        self.indicators = TechnicalIndicators()
        self.strategy   = TrendFollowingStrategy(
            rsi_lower             = strat_cfg.get("rsi_lower", 45),
            rsi_upper             = strat_cfg.get("rsi_upper", 70),
            adx_threshold         = strat_cfg.get("adx_threshold", 18),
            min_atr_ratio         = strat_cfg.get("min_atr_ratio", 0.005),
            volume_sma_multiplier = strat_cfg.get("volume_sma_multiplier", 0.4),
            choppiness_threshold  = fil_cfg.get("choppiness_threshold", 61.8),
            choppiness_enabled    = fil_cfg.get("choppiness_enabled", True),
            mtf_filter_enabled    = False,   # daily bars'ta MTF devre dışı
            indicators            = self.indicators,
        )

        initial_capital = bt_cfg.get("initial_capital", 10_000.0)
        self.risk_manager = RiskManager(
            account_balance   = initial_capital,
            risk_per_trade    = risk_cfg.get("risk_per_trade", 0.015),
            daily_max_loss    = risk_cfg.get("daily_max_loss", 0.04),
            atr_stop_multiplier = risk_cfg.get("atr_stop_multiplier", 2.0),
            max_open_positions  = risk_cfg.get("max_open_positions", 6),
            min_order_size      = risk_cfg.get("min_order_size", 10.0),
            max_position_pct    = risk_cfg.get("max_position_pct", 0.20),
        )
        self.position_manager = PositionManager(
            trailing_stop_atr_multiplier=risk_cfg.get("trailing_stop_atr_multiplier", 4.0)
        )

        macro_cfg = cfg.get("macro", {})
        self.macro_filter = MacroFilter(
            regime_filter_enabled   = macro_cfg.get("regime_filter_enabled", True),
            usdtry_guard_enabled    = macro_cfg.get("usdtry_guard_enabled", True),
            usdtry_momentum_period  = macro_cfg.get("usdtry_momentum_period", 20),
            usdtry_crisis_threshold = macro_cfg.get("usdtry_crisis_threshold", 0.15),
        )
        self._macro_fitted = False

    def _fit_macro(self) -> None:
        """İlk çalışmada makro filtreyi fit et."""
        from bist.filters.macro_filter import fetch_xu100
        macro_cfg = self.cfg.get("macro", {})
        try:
            usdtry = self.usdtry_p.fetch(start="2020-01-01", end="2030-01-01")
            xu100  = fetch_xu100(start="2020-01-01", end="2030-01-01") if macro_cfg.get("regime_filter_enabled", True) else None
            self.macro_filter.fit(xu100 if xu100 is not None else pd.Series(dtype=float), usdtry)
            self._macro_fitted = True
            logger.info(f"[BistPaper] Makro filtre fit edildi: {self.macro_filter.summary_stats()}")
        except Exception as e:
            logger.warning(f"[BistPaper] Makro filtre fit edilemedi: {e} — devre dışı.")

    def _fetch_latest(self, symbol: str) -> pd.DataFrame | None:
        """Son 300 günlük OHLCV çekip USD'ye çevirir."""
        try:
            from datetime import timedelta
            end   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            start = (datetime.now(timezone.utc) - timedelta(days=350)).strftime("%Y-%m-%d")
            df_try = self.provider.fetch(symbol, start=start, end=end, interval="1d", force_refresh=True)
            usdtry = self.usdtry_p.fetch(start=start, end=end)
            return self.converter.convert_ohlcv(df_try, usdtry)
        except Exception as e:
            logger.error(f"[BistPaper] {symbol} veri hatası: {e}")
            return None

    def start(self) -> None:
        """Paper trading döngüsü. Ctrl+C ile durdurulur."""
        logger.info(f"[BistPaper] Başlatılıyor. Semboller: {self.symbols}")
        print(f"\n{'='*60}")
        print(f"  BIST 100 PAPER TRADER")
        print(f"  Semboller: {', '.join(self.symbols)}")
        print(f"  Ctrl+C ile durdurulur.")
        print(f"{'='*60}\n")

        self._fit_macro()
        last_day: str | None = None

        try:
            while True:
                now = datetime.now(timezone.utc)

                if not self.is_market_open(now):
                    logger.debug(f"[BistPaper] Piyasa kapalı ({now.strftime('%H:%M UTC')}). Bekleniyor.")
                    _time.sleep(60)
                    continue

                day_str = now.strftime("%Y-%m-%d")
                if last_day and day_str != last_day:
                    self.risk_manager.reset_daily_pnl()
                    logger.info(f"[BistPaper] Yeni gün: {day_str} — günlük PnL sıfırlandı.")
                last_day = day_str

                current_prices: dict[str, float] = {}
                atrs: dict[str, float] = {}

                # Veri çek + indikatör hesapla
                sym_data: dict[str, pd.DataFrame] = {}
                for sym in self.symbols:
                    df = self._fetch_latest(sym)
                    if df is None or len(df) < 50:
                        continue
                    df_ind = self.indicators.calculate(df)
                    last   = df_ind.iloc[-1]
                    current_prices[sym] = last["close"]
                    atrs[sym]           = last.get("atr", 0.0)
                    sym_data[sym]       = df_ind

                # Stop / trailing güncelle
                closed_list = self.position_manager.update_positions(current_prices, atrs)
                for pos, reason in closed_list:
                    commission = pos.close_price * pos.position_size * self.cfg.get("backtest", {}).get("commission_rate", 0.0005)
                    net_pnl = pos.realized_pnl - commission
                    self.risk_manager.record_trade_pnl(net_pnl)
                    self.strategy.record_outcome(pos.symbol, net_pnl > 0, pnl=net_pnl)
                    print(f"  [KAPANDI] {pos.symbol} | {reason} | PnL: {'+' if net_pnl >= 0 else ''}{net_pnl:.2f} USD")

                if not self.risk_manager.trading_allowed:
                    logger.warning("[BistPaper] Günlük kayıp limitine ulaşıldı.")
                    _time.sleep(POLL_INTERVAL_SEC)
                    continue

                ts_now = pd.Timestamp(now)
                if not self.macro_filter.is_entry_allowed(ts_now):
                    reason = self.macro_filter.get_block_reason(ts_now)
                    logger.info(f"[BistPaper] Makro filtre giriş engelledi: {reason}")
                    _time.sleep(POLL_INTERVAL_SEC)
                    continue

                # Sinyal üret
                for sym, df_ind in sym_data.items():
                    self.strategy.tick_pause(sym)
                    try:
                        signal = self.strategy.generate_signal(sym, df_ind)
                        if signal.side == self.Side.BUY and not self.position_manager.has_open_position(sym):
                            atr = atrs.get(sym, 0.0)
                            price = current_prices[sym]
                            size = self.risk_manager.calculate_position_size(price, atr)
                            if size > 0 and price * size >= self.risk_manager.min_order_size:
                                stop = price - self.risk_manager.atr_stop_multiplier * atr
                                self.position_manager.open_position(sym, price, stop, size, atr)
                                commission = price * size * self.cfg.get("backtest", {}).get("commission_rate", 0.0005)
                                self.risk_manager.account_balance -= commission
                                print(f"  [AÇILDI] {sym} @ {price:.4f} USD | size={size:.4f} | stop={stop:.4f}")
                    except Exception as e:
                        logger.error(f"[BistPaper] {sym} sinyal hatası: {e}")

                bal = self.risk_manager.account_balance
                print(f"  [{now.strftime('%H:%M UTC')}] Bakiye: ${bal:,.2f} USD | "
                      f"Açık: {len(self.position_manager.get_all_positions())}")
                _time.sleep(POLL_INTERVAL_SEC)

        except KeyboardInterrupt:
            logger.info("[BistPaper] Durduruldu.")
            print("\n  Açık pozisyonlar:")
            for sym, pos in self.position_manager.get_all_positions().items():
                price = current_prices.get(sym, pos.entry_price)
                unrealized = (price - pos.entry_price) * pos.position_size
                print(f"    {sym}: giriş={pos.entry_price:.4f}, şimdi={price:.4f}, "
                      f"gerçekleşmemiş PnL={'+' if unrealized >= 0 else ''}{unrealized:.2f} USD")
            print(f"  Bakiye: ${self.risk_manager.account_balance:,.2f} USD")
