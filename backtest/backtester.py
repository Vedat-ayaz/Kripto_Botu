import logging
import csv
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from indicators.technical_indicators import TechnicalIndicators
from strategy.trend_following_strategy import TrendFollowingStrategy
from strategy.signal import Side
from risk.risk_manager import RiskManager
from execution.position_manager import PositionManager, Position
from backtest.metrics import calculate_metrics, print_metrics

from risk import correlation_registry

logger = logging.getLogger(__name__)


class Backtester:
    """
    Geçmiş OHLCV verisi üzerinde stratejiyi simüle eder.
    Komisyon, slippage ve spread dahil edilir.
    Bar-by-bar simülasyon yapılır (look-ahead bias yok).
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        risk_per_trade: float = 0.01,
        daily_max_loss: float = 0.03,
        atr_stop_multiplier: float = 2.0,
        trailing_stop_atr_multiplier: float = 3.0,
        adx_threshold: float = 20.0,
        rsi_lower: float = 45.0,
        rsi_upper: float = 70.0,
        min_atr_ratio: float = 0.002,
        volume_sma_multiplier: float = 0.3,
        max_open_positions: int = 3,
        min_order_size: float = 10.0,
        max_position_pct: float = 0.20,
        entry_score_trend: float = 0.55,    # Per-coin override desteği
        entry_score_ranging: float = 0.60,  # Per-coin override desteği
        # ── Pyramiding (Turtle/Clenow) — Stage 1 ────────────────────────────
        pyramid_enabled: bool = False,
        pyramid_thresholds_atr: Optional[list[float]] = None,  # ör. [1.5, 3.0]
        pyramid_size_pcts: Optional[list[float]] = None,       # ör. [0.5, 0.25]
        pyramid_max_adds: int = 2,
        pyramid_stop_atr_multiplier: float = 2.0,              # Turtle: add_price - 2×ATR
        pyramid_gate_min_regime: float = 0.50,                 # adaptif gate eşikleri
        pyramid_gate_max_vol_spike: float = 1.50,
        pyramid_gate_max_atr_ratio: float = 0.040,
        pyramid_gate_min_adx: float = 22.0,
        # ── Partial Exits (R-multiple kademeli realize) — Stage 2 ──────────
        partial_exit_enabled: bool = False,
        partial_exit_r_levels: Optional[list[float]] = None,   # ör. [1.5, 3.0]
        partial_exit_pcts: Optional[list[float]] = None,       # ör. [0.30, 0.30]
        partial_exit_max: int = 2,
        # ── Anti-whipsaw + MTF (Stage 4) ───────────────────────────────────
        choppiness_threshold: float = 61.8,
        choppiness_enabled: bool = True,
        mtf_filter_enabled: bool = True,
        # Timeframe-bağımlı lookback (1h default; daily: slope_bars=5, momentum_lookback=30)
        slope_bars: int = 20,
        momentum_lookback: int = 720,
        # ADX boost + regime thresholds (daily BIST'te adx_boost=0.0, threshold'lar düşürülmeli)
        adx_boost: float = 0.06,
        regime_trending_threshold: float = 0.60,
        regime_ranging_threshold: float = 0.35,
        monthly_loss_limit: float = 0.05,
    ):
        self.initial_capital = initial_capital
        self._monthly_loss_limit = monthly_loss_limit
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        self.indicators = TechnicalIndicators()
        self.strategy = TrendFollowingStrategy(
            rsi_lower=rsi_lower,
            rsi_upper=rsi_upper,
            adx_threshold=adx_threshold,
            min_atr_ratio=min_atr_ratio,
            volume_sma_multiplier=volume_sma_multiplier,
            entry_score_trend=entry_score_trend,
            entry_score_ranging=entry_score_ranging,
            choppiness_threshold=choppiness_threshold,
            choppiness_enabled=choppiness_enabled,
            mtf_filter_enabled=mtf_filter_enabled,
            slope_bars=slope_bars,
            momentum_lookback=momentum_lookback,
            adx_boost=adx_boost,
            regime_trending_threshold=regime_trending_threshold,
            regime_ranging_threshold=regime_ranging_threshold,
            indicators=self.indicators,
        )

        self._risk_manager: Optional[RiskManager] = None
        self._position_manager: Optional[PositionManager] = None

        # Konfig parametreleri
        self._risk_per_trade = risk_per_trade
        self._daily_max_loss = daily_max_loss
        self._atr_stop_multiplier = atr_stop_multiplier
        self._trailing_multiplier = trailing_stop_atr_multiplier
        self._max_open_positions = max_open_positions
        self._min_order_size = min_order_size
        self._max_position_pct = max_position_pct

        # Pyramiding parametreleri
        self._pyramid_enabled = pyramid_enabled
        self._pyramid_thresholds_atr = pyramid_thresholds_atr or [1.5, 3.0]
        self._pyramid_size_pcts = pyramid_size_pcts or [0.5, 0.25]
        self._pyramid_max_adds = pyramid_max_adds
        self._pyramid_stop_atr_multiplier = pyramid_stop_atr_multiplier
        self._pyramid_gate_min_regime = pyramid_gate_min_regime
        self._pyramid_gate_max_vol_spike = pyramid_gate_max_vol_spike
        self._pyramid_gate_max_atr_ratio = pyramid_gate_max_atr_ratio
        self._pyramid_gate_min_adx = pyramid_gate_min_adx

        # Partial Exit parametreleri (Stage 2)
        self._partial_exit_enabled = partial_exit_enabled
        self._partial_exit_r_levels = partial_exit_r_levels or [1.5, 3.0]
        self._partial_exit_pcts = partial_exit_pcts or [0.30, 0.30]
        self._partial_exit_max = partial_exit_max

    def run(
        self,
        symbol: str,
        df: pd.DataFrame,
        btc_regime: Optional[pd.Series] = None,
        trade_start: Optional[pd.Timestamp] = None,
        trade_end: Optional[pd.Timestamp] = None,
    ) -> dict:
        """
        Tek sembol üzerinde backtest yapar.
        df          : timestamp indeksli ham OHLCV DataFrame
        btc_regime  : pd.Series[bool] — BTC'nin her bar için bull/bear durumu.
                      None ise rejim filtresi devre dışı (varsayılan: bull).
        trade_start : pd.Timestamp — bu tarihten önce sinyal üretilmez (warmup koruması).
                      None ise tüm veriye sinyal üretilir.
        trade_end   : pd.Timestamp — bu tarihten sonra yeni işlem açılmaz.
                      Mevcut açık pozisyonlar kapanmaya devam eder.
                      None ise sınır yok.
        """
        logger.info(f"[Backtester] {symbol} backtest başladı. {len(df)} mum.")

        # Göstergeleri hesapla (tüm veri üzerinden)
        df_ind = self.indicators.calculate(df)

        # Multi-timeframe (4h) trend filter sütunlarını ekle.
        # 1h verisinden 4h resample → EMA50, ADX hesapla → ffill ile 1h indekse yay.
        # Strategy.generate_signal bu sütunları kontrol eder (mtf_filter_enabled).
        # NOT: pandas 2.x'te "4h" (küçük) zorunlu, "4H" deprecate edildi.
        df_ind = self.indicators.add_higher_timeframe(df_ind, htf_rule="4h")
        if "htf_trend_up" not in df_ind.columns:
            logger.error(f"[Backtester] {symbol}: MTF sütunları eklenemedi — filtre devre dışı!")

        # Risk ve pozisyon yöneticilerini sıfırla
        self._risk_manager = RiskManager(
            account_balance=self.initial_capital,
            risk_per_trade=self._risk_per_trade,
            daily_max_loss=self._daily_max_loss,
            atr_stop_multiplier=self._atr_stop_multiplier,
            max_open_positions=self._max_open_positions,
            min_order_size=self._min_order_size,
            max_position_pct=self._max_position_pct,
        )
        self._position_manager = PositionManager(self._trailing_multiplier)

        capital = self.initial_capital
        equity_curve: list[float] = [capital]
        trades: list[dict] = []
        last_day: Optional[str] = None

        # Circuit breaker durumu — aylık kayıp takibi
        _circuit_month: object = None
        _circuit_month_start_capital: float = capital
        _circuit_breaker_active: bool = False

        # Minimum gösterge warmup (EMA200 için en az 200 mum)
        warmup = self.indicators.ema_slow + 10

        # trade_start varsa, warmup'ı trade_start'a kadar ilerlet
        # (extra ısınma verisi ile göstergeler gerçek başlangıca hazır olur)
        if trade_start is not None:
            trade_start_idx = df_ind.index.searchsorted(trade_start)
            warmup = max(warmup, trade_start_idx)

        for i in range(warmup, len(df_ind)):
            row = df_ind.iloc[i]
            current_price = row["close"]
            atr = row.get("atr", 0.0)
            ts = df_ind.index[i]

            # Stage 3: Adaptif pause sayacını ilerlet (bar-tabanlı)
            self.strategy.tick_pause(symbol)

            # BTC rejim güncellemesi — her barda
            if btc_regime is not None:
                is_bull = bool(btc_regime.asof(ts)) if ts >= btc_regime.index[0] else True
                self.strategy.set_btc_regime(is_bull)

            # Gün değişimi kontrolü — günlük PnL sıfırlama
            day_str = str(ts.date()) if hasattr(ts, "date") else str(ts)[:10]
            if last_day and day_str != last_day:
                self._risk_manager.reset_daily_pnl()
            last_day = day_str

            # 1) Mevcut pozisyonları güncelle (stop, trailing kontrolü)
            slice_df = df_ind.iloc[: i + 1]
            current_prices = {symbol: current_price}
            atrs = {symbol: atr}
            closed_list = self._position_manager.update_positions(current_prices, atrs)

            for pos, reason in closed_list:
                fill_price = self._apply_costs(pos.close_price, "sell")
                commission = fill_price * pos.position_size * self.commission_rate
                net_pnl = pos.realized_pnl - commission
                self._risk_manager.record_trade_pnl(net_pnl)
                capital = self._risk_manager.account_balance
                equity_curve.append(capital)
                trades.append(self._trade_record(pos, fill_price, net_pnl, reason))
                correlation_registry.register_close(symbol)
                # Rolling performans kaydı — strateji adaptasyonu için
                # Stage 3: WR + net PnL ikili kriteri için pnl da geç
                # Toplam trade PnL = final close PnL + partial exits PnL (varsa)
                total_trade_pnl = net_pnl + getattr(pos, "realized_pnl_partial", 0.0)
                self.strategy.record_outcome(symbol, total_trade_pnl > 0, pnl=total_trade_pnl)
                logger.debug(f"[BT] {symbol} CLOSED via {reason} | pnl={net_pnl:+.4f}")

            # 1b) Pyramid (Turtle/Clenow) — açık pozisyon kâra geçtikçe lot ekle
            # Stop kontrolünden SONRA, yeni sinyalden ÖNCE çalışır.
            # Bu sırayla: önce kaybedenler kapansın, sonra kazanan pozisyonlara ekle.
            if self._pyramid_enabled and self._risk_manager.trading_allowed:
                open_pos = self._position_manager.get_position(symbol)
                if open_pos is not None:
                    # Adaptif gate: anlık piyasa durumu pyramid'e uygun mu?
                    # Hardcoded per-coin kararı yerine indikatörlerden okur.
                    gate_ok, gate_reason = self.strategy.should_allow_pyramid(
                        symbol, row,
                        min_regime_score=self._pyramid_gate_min_regime,
                        max_vol_spike=self._pyramid_gate_max_vol_spike,
                        max_atr_ratio=self._pyramid_gate_max_atr_ratio,
                        min_adx=self._pyramid_gate_min_adx,
                    )
                    if gate_ok:
                        allowed_p, add_level, _ = self._risk_manager.can_pyramid_add(
                            symbol, current_price, atr, self._position_manager,
                            self._pyramid_thresholds_atr, self._pyramid_max_adds,
                        )
                        if allowed_p:
                            add_size = self._risk_manager.calculate_pyramid_size(
                                open_pos.initial_size, add_level, self._pyramid_size_pcts,
                            )
                            if add_size > 0:
                                # Min order size kontrolü
                                add_value = add_size * current_price
                                if add_value >= self._min_order_size:
                                    fill_price = self._apply_costs(current_price, "buy")
                                    commission = fill_price * add_size * self.commission_rate
                                    capital -= commission  # pyramid açılış komisyonu
                                    # Pyramid trailing'i mevcut fiyata göre yenile
                                    # (asla aşağı çekilmez — add_to_position bunu garanti eder)
                                    new_trailing = self._risk_manager.calculate_trailing_stop(
                                        current_price, atr, self._trailing_multiplier,
                                    )
                                    self._position_manager.add_to_position(
                                        symbol, add_size, fill_price,
                                        new_trailing_stop=new_trailing,
                                        atr=atr,
                                        pyramid_stop_atr_multiplier=self._pyramid_stop_atr_multiplier,
                                    )
                                    logger.debug(
                                        f"[BT] {symbol} PYRAMID #{add_level} @ {fill_price:.4f} "
                                        f"add_size={add_size:.6f} comm={commission:.4f}"
                                    )
                    else:
                        logger.debug(f"[BT] {symbol} pyramid gate rejected: {gate_reason}")

            # 1c) Partial Exit (Stage 2 — kazananı kademeli realize et)
            # Pyramid'ten sonra çalışır: önce trend güçlüyse büyüt, sonra
            # kâr eşikleri aşıldıkça kademeli kilitle.
            # "Winner turn into loser" matematiksel olarak elimine edilir:
            # her R-eşiğinde kâr realize edilir, kalan parça trailing'le devam.
            if self._partial_exit_enabled and self._risk_manager.trading_allowed:
                open_pos = self._position_manager.get_position(symbol)
                if open_pos is not None and open_pos.partial_exits_done < self._partial_exit_max:
                    pe_allowed, pe_level, current_r, _ = self._risk_manager.should_partial_exit(
                        symbol, current_price, self._position_manager,
                        self._partial_exit_r_levels, self._partial_exit_max,
                    )
                    if pe_allowed:
                        exit_pct = self._partial_exit_pcts[pe_level - 1]
                        # Min order size check on remaining piece
                        remaining_after = open_pos.position_size * (1 - exit_pct)
                        if remaining_after * current_price >= self._min_order_size:
                            fill_price = self._apply_costs(current_price, "sell")
                            result = self._position_manager.partial_close(
                                symbol, exit_pct, fill_price,
                                reason=f"partial_exit_{self._partial_exit_r_levels[pe_level - 1]:.1f}R",
                            )
                            if result is not None:
                                _, exit_size, partial_pnl_gross = result
                                partial_commission = fill_price * exit_size * self.commission_rate
                                net_partial = partial_pnl_gross - partial_commission
                                # Pozisyon üzerinde net partial'ı biriktir
                                open_pos.realized_pnl_partial += net_partial
                                # risk_manager bakiye + günlük PnL'i günceller
                                self._risk_manager.record_trade_pnl(net_partial)
                                capital = self._risk_manager.account_balance
                                equity_curve.append(capital)
                                logger.debug(
                                    f"[BT] {symbol} PARTIAL EXIT #{pe_level} @ {fill_price:.4f} "
                                    f"({exit_pct:.0%}, R={current_r:.2f}) "
                                    f"net_pnl={net_partial:+.4f}"
                                )

            # 2) Strateji sinyali üret
            if not self._risk_manager.trading_allowed:
                equity_curve.append(capital)
                continue

            # ── Aylık circuit breaker kontrolü ──────────────────────────────
            # Her ay başında sayacı sıfırla; aylık kayıp limit'i aşarsa
            # yeni entry'leri engelle (mevcut pozisyon takibi devam eder).
            # Unrealized PnL dahil anlık equity kullanılır (gerçekçi kayıp tespiti).
            current_month = ts.year * 100 + ts.month
            if _circuit_month != current_month:
                _circuit_month = current_month
                # Ay başında unrealized equity'yi de dahil et
                open_pos_now = self._position_manager.get_position(symbol)
                unrealized = 0.0
                if open_pos_now is not None:
                    unrealized = (current_price - open_pos_now.entry_price) * open_pos_now.position_size
                _circuit_month_start_capital = capital + unrealized
                _circuit_breaker_active = False

            # Anlık equity = realized capital + unrealized PnL açık pozisyonda
            open_pos_now = self._position_manager.get_position(symbol)
            unrealized_now = 0.0
            if open_pos_now is not None:
                unrealized_now = (current_price - open_pos_now.entry_price) * open_pos_now.position_size
            current_equity = capital + unrealized_now

            monthly_loss = (current_equity - _circuit_month_start_capital) / max(_circuit_month_start_capital, 1.0)
            if monthly_loss < -self._monthly_loss_limit:
                _circuit_breaker_active = True

            signal = self.strategy.generate_signal(symbol, slice_df)

            # trade_end kontrolü: bu tarihten sonra yeni pozisyon açma
            entry_allowed = (trade_end is None or ts <= trade_end) and not _circuit_breaker_active

            if signal.side == Side.BUY and entry_allowed:
                allowed, reject_reason = self._risk_manager.can_open_trade(
                    symbol, current_price, atr, self._position_manager
                )
                if allowed:
                    stop_price = self._risk_manager.calculate_stop_price(current_price, atr)
                    trailing_stop = self._risk_manager.calculate_trailing_stop(
                        current_price, atr, self._trailing_multiplier
                    )
                    # ADX gücüne göre pozisyon boyutunu ölçekle
                    adx_scale = self.strategy.get_adx_scale(signal.adx or 0.0)
                    # Cross-symbol correlation scale (MVP): açık alt sayısına göre küçült
                    # (Carver IDM yaklaşımı, MaxDD'yi düşürür)
                    corr_scale = self.strategy.get_corr_scale(
                        symbol, correlation_registry.get_open_symbols()
                    )
                    size = (
                        self._risk_manager.calculate_position_size(current_price, atr)
                        * adx_scale
                        * corr_scale
                    )

                    # ATR Volatilite Rejim Filtresi:
                    # Anlık ATR, 20-bar ATR ortalamasının 2× üzerindeyse piyasa
                    # anormal derecede oynak → pozisyonu yarıya indir.
                    atr_sma_20 = row.get("atr_sma_20", 0.0) or 0.0
                    if atr_sma_20 > 0 and atr > atr_sma_20 * 2.0:
                        size *= 0.5
                        logger.debug(
                            f"[BT] {symbol} volatilite spike: ATR={atr:.4f} > 2×ATR_SMA={atr_sma_20*2:.4f} "
                            f"→ pozisyon yarıya indirildi"
                        )

                    fill_price = self._apply_costs(current_price, "buy")
                    commission = fill_price * size * self.commission_rate
                    capital -= commission  # açılış komisyonu

                    pos = Position(
                        symbol=symbol,
                        entry_price=fill_price,
                        position_size=size,
                        stop_price=stop_price,
                        trailing_stop_price=trailing_stop,
                        opened_at=datetime.now(timezone.utc),
                    )
                    self._position_manager.open_position(pos)
                    correlation_registry.register_open(symbol)
                    logger.debug(f"[BT] {symbol} OPEN @ {fill_price:.4f} | size={size:.6f} (corr_scale={corr_scale:.2f})")

            # 3) Sinyal bazlı çıkış
            elif signal.side == Side.SELL or (
                signal.side == Side.HOLD and self._should_exit_strategy(slice_df, symbol)
            ):
                pos = self._position_manager.get_position(symbol)
                if pos:
                    fill_price = self._apply_costs(current_price, "sell")
                    commission = fill_price * pos.position_size * self.commission_rate
                    closed_pos = self._position_manager.close_position(symbol, fill_price, "strategy_exit")
                    if closed_pos:
                        net_pnl = closed_pos.realized_pnl - commission
                        self._risk_manager.record_trade_pnl(net_pnl)
                        capital = self._risk_manager.account_balance
                        trades.append(self._trade_record(closed_pos, fill_price, net_pnl, "strategy_exit"))
                        correlation_registry.register_close(symbol)

            equity_curve.append(capital)

        # Backtest sona erdi, açık pozisyonları piyasa fiyatından kapat
        last_price = df_ind.iloc[-1]["close"]
        for pos in self._position_manager.open_positions:
            fill_price = self._apply_costs(last_price, "sell")
            commission = fill_price * pos.position_size * self.commission_rate
            pos.close(fill_price, "backtest_end")
            net_pnl = pos.realized_pnl - commission
            self._risk_manager.record_trade_pnl(net_pnl)
            capital = self._risk_manager.account_balance
            equity_curve.append(capital)
            trades.append(self._trade_record(pos, fill_price, net_pnl, "backtest_end"))
            correlation_registry.register_close(pos.symbol)

        metrics = calculate_metrics(trades, equity_curve, self.initial_capital)
        print_metrics(metrics, symbol)

        return {
            "symbol": symbol,
            "metrics": metrics,
            "trades": trades,
            "equity_curve": equity_curve,
        }

    def save_csv(self, result: dict, output_dir: str = "backtest_results") -> str:
        """Backtest trade listesini CSV'ye yazar."""
        os.makedirs(output_dir, exist_ok=True)
        symbol_clean = result["symbol"].replace("/", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"backtest_{symbol_clean}_{ts}.csv")

        trades = result.get("trades", [])
        if not trades:
            logger.info("[Backtester] CSV yazılmadı: trade yok.")
            return ""

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=trades[0].keys())
            writer.writeheader()
            writer.writerows(trades)

        logger.info(f"[Backtester] CSV kaydedildi: {filepath}")
        return filepath

    # ------------------------------------------------------------------ #

    def _apply_costs(self, price: float, side: str) -> float:
        """Slippage uygular: alışta yukarı, satışta aşağı."""
        if side == "buy":
            return price * (1 + self.slippage_rate)
        return price * (1 - self.slippage_rate)

    def _should_exit_strategy(self, df: pd.DataFrame, symbol: str) -> bool:
        should_exit, _ = self.strategy.should_exit(symbol, df, entry_price=0)
        return should_exit

    def _trade_record(self, pos: Position, exit_price: float, net_pnl: float, reason: str) -> dict:
        # TOPLAM trade PnL = nihai çıkış PnL'i + kümülatif partial exit PnL'leri
        # (her partial exit anında risk_manager'a kayıt edildi, ama trade-bazlı
        # metrikler (WR, PF, avg_win) için tek bir trade-toplam değer gerekli)
        total_trade_pnl = net_pnl + pos.realized_pnl_partial
        return {
            "symbol": pos.symbol,
            "entry_price": round(pos.entry_price, 6),
            "avg_entry_price": round(pos.avg_entry_price, 6),
            "exit_price": round(exit_price, 6),
            "initial_size": round(pos.initial_size, 6),
            "position_size": round(pos.position_size, 6),
            "pyramid_adds": pos.pyramid_adds_count,
            "partial_exits": pos.partial_exits_done,
            "pnl_final": round(net_pnl, 6),                # son kapanış PnL'i
            "pnl_partial": round(pos.realized_pnl_partial, 6),  # kademeli realize toplamı
            "pnl": round(total_trade_pnl, 6),              # TRADE TOPLAM (metrikler için)
            "reason": reason,
            "opened_at": str(pos.opened_at),
            "closed_at": str(pos.closed_at),
        }
