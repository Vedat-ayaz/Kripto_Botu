"""
Kripto Trend-Following Bot
Kullanım: python main.py --mode [backtest|paper|scalp|live]
"""

import argparse
import sys
import os
import logging

import yaml
from dotenv import load_dotenv

from monitoring.logger import setup_logger

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve_project_path(path: str) -> str:
    """
    Varsayılan proje dosyalarını çalışma dizininden bağımsız çöz.
    Kullanıcı mutlak yol veya mevcut dizinde özel bir dosya verirse ona dokunma.
    """
    if os.path.isabs(path) or os.path.exists(path):
        return path
    return os.path.join(PROJECT_ROOT, path)

# Config yükleme
def load_config(path: str = "config.yaml") -> dict:
    with open(resolve_project_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_symbol_config(symbol: str, cfg: dict) -> dict:
    """
    Per-coin profil sistemi:
    Base risk + strategy config üzerine symbol_profiles override'ları uygular.
    Belirtilmeyen parametreler baseline değerini kullanır.
    """
    risk_cfg  = dict(cfg.get("risk", {}))
    strat_cfg = dict(cfg.get("strategy", {}))
    merged    = {**strat_cfg, **risk_cfg}   # risk > strategy (çakışırsa risk kazanır)

    profile = cfg.get("symbol_profiles", {}).get(symbol, {})
    if profile:
        merged.update(profile)
        logging.getLogger("backtest").info(
            f"[PerCoin] {symbol} özel profil uygulandı: {profile}"
        )
    return merged


def make_backtester_for_symbol(symbol: str, cfg: dict) -> "object":
    """Her sembol için özelleştirilmiş Backtester oluştur."""
    from backtest.backtester import Backtester
    bt_cfg  = cfg.get("backtest", {})
    sym_cfg = get_symbol_config(symbol, cfg)
    pyr_cfg = cfg.get("pyramiding", {}) or {}
    pe_cfg  = cfg.get("partial_exits", {}) or {}
    fil_cfg = cfg.get("filters", {}) or {}

    # Per-symbol pyramid override:
    # symbol_profiles içinde "pyramid_enabled: false" varsa o coin için pyramid kapanır.
    # Bu, BTC/SOL/MATIC gibi pyramid'in faydalı olduğu coinlerde aktif tutmamızı,
    # XRP/INJ/FET gibi volatil patlamacılarda kapatmamızı sağlar.
    sym_pyramid_enabled = sym_cfg.get("pyramid_enabled", pyr_cfg.get("enabled", False))

    return Backtester(
        initial_capital              = bt_cfg.get("initial_capital", 10_000.0),
        commission_rate              = bt_cfg.get("commission_rate", 0.001),
        slippage_rate                = bt_cfg.get("slippage_rate", 0.0005),
        risk_per_trade               = sym_cfg.get("risk_per_trade", 0.015),
        daily_max_loss               = sym_cfg.get("daily_max_loss", 0.04),
        atr_stop_multiplier          = sym_cfg.get("atr_stop_multiplier", 2.0),
        trailing_stop_atr_multiplier = sym_cfg.get("trailing_stop_atr_multiplier", 3.8),
        adx_threshold                = sym_cfg.get("adx_threshold", 15.0),
        rsi_lower                    = sym_cfg.get("rsi_lower", 45.0),
        rsi_upper                    = sym_cfg.get("rsi_upper", 70.0),
        min_atr_ratio                = sym_cfg.get("min_atr_ratio", 0.002),
        volume_sma_multiplier        = sym_cfg.get("volume_sma_multiplier", 0.3),
        max_open_positions           = sym_cfg.get("max_open_positions", 8),
        min_order_size               = sym_cfg.get("min_order_size", 10.0),
        max_position_pct             = sym_cfg.get("max_position_pct", 0.20),
        entry_score_trend            = sym_cfg.get("entry_score_trend", 0.55),
        entry_score_ranging          = sym_cfg.get("entry_score_ranging", 0.60),
        # ── Pyramiding (Stage 1) — global; adaptif gate coin'leri filtreler ──
        pyramid_enabled              = bool(sym_pyramid_enabled),
        pyramid_thresholds_atr       = list(sym_cfg.get("pyramid_thresholds_atr", pyr_cfg.get("thresholds_atr", [1.5, 3.0]))),
        pyramid_size_pcts            = list(sym_cfg.get("pyramid_size_pcts", pyr_cfg.get("size_pcts", [0.5, 0.25]))),
        pyramid_max_adds             = int(sym_cfg.get("pyramid_max_adds", pyr_cfg.get("max_adds", 2))),
        pyramid_stop_atr_multiplier  = float(sym_cfg.get("pyramid_stop_atr_multiplier", pyr_cfg.get("stop_atr_multiplier", 2.0))),
        # Adaptif gate eşikleri (sym_cfg ile per-coin override edilebilir)
        # Örn: FET için ATR/price ~3-5% olduğundan global 0.020 reddediyor;
        # symbol_profiles altına "pyramid_gate_max_atr_ratio: 0.045" ekle.
        pyramid_gate_min_regime      = float(sym_cfg.get("pyramid_gate_min_regime", pyr_cfg.get("gate", {}).get("min_regime_score", 0.50))),
        pyramid_gate_max_vol_spike   = float(sym_cfg.get("pyramid_gate_max_vol_spike", pyr_cfg.get("gate", {}).get("max_vol_spike", 1.50))),
        pyramid_gate_max_atr_ratio   = float(sym_cfg.get("pyramid_gate_max_atr_ratio", pyr_cfg.get("gate", {}).get("max_atr_ratio", 0.040))),
        pyramid_gate_min_adx         = float(sym_cfg.get("pyramid_gate_min_adx", pyr_cfg.get("gate", {}).get("min_adx", 22.0))),
        # ── Partial Exits (Stage 2) ───────────────────────────────────────
        partial_exit_enabled         = bool(sym_cfg.get("partial_exit_enabled", pe_cfg.get("enabled", False))),
        partial_exit_r_levels        = list(sym_cfg.get("partial_exit_r_levels", pe_cfg.get("r_multiple_levels", [1.5, 3.0]))),
        partial_exit_pcts            = list(sym_cfg.get("partial_exit_pcts", pe_cfg.get("exit_pcts", [0.30, 0.30]))),
        partial_exit_max             = int(sym_cfg.get("partial_exit_max", pe_cfg.get("max_exits", 2))),
        # ── Anti-whipsaw + MTF (Stage 4) ─────────────────────────────────
        choppiness_threshold         = float(sym_cfg.get("choppiness_threshold", fil_cfg.get("choppiness_threshold", 61.8))),
        choppiness_enabled           = bool(sym_cfg.get("choppiness_enabled", fil_cfg.get("choppiness_enabled", True))),
        mtf_filter_enabled           = bool(sym_cfg.get("mtf_filter_enabled", fil_cfg.get("mtf_filter_enabled", True))),
    )


def run_backtest(cfg: dict) -> None:
    import pandas as pd
    from data.exchange_client import ExchangeClient
    from data.candle_repository import CandleRepository
    from data.market_data_service import MarketDataService
    from backtest.backtester import Backtester

    logger = logging.getLogger("backtest")

    client = ExchangeClient(cfg["exchange"]["name"], testnet=cfg["exchange"]["testnet"])
    client.connect()

    repo = CandleRepository()
    market_data = MarketDataService(client, repo)

    bt_cfg = cfg.get("backtest", {})
    risk_cfg = cfg.get("risk", {})
    strat_cfg = cfg.get("strategy", {})

    backtester = Backtester(
        initial_capital=bt_cfg.get("initial_capital", 10_000.0),
        commission_rate=bt_cfg.get("commission_rate", 0.001),
        slippage_rate=bt_cfg.get("slippage_rate", 0.0005),
        risk_per_trade=risk_cfg.get("risk_per_trade", 0.01),
        daily_max_loss=risk_cfg.get("daily_max_loss", 0.03),
        atr_stop_multiplier=risk_cfg.get("atr_stop_multiplier", 2.0),
        trailing_stop_atr_multiplier=risk_cfg.get("trailing_stop_atr_multiplier", 2.5),
        adx_threshold=strat_cfg.get("adx_threshold", 20.0),
        rsi_lower=strat_cfg.get("rsi_lower", 45.0),
        rsi_upper=strat_cfg.get("rsi_upper", 70.0),
        min_atr_ratio=strat_cfg.get("min_atr_ratio", 0.002),
        max_open_positions=risk_cfg.get("max_open_positions", 3),
        min_order_size=risk_cfg.get("min_order_size", 10.0),
        max_position_pct=risk_cfg.get("max_position_pct", 0.20),
    )

    symbols = cfg["trading"]["symbols"]
    timeframe = cfg["trading"]["timeframe"]

    for symbol in symbols:
        logger.info(f"[Backtest] {symbol} için veri çekiliyor...")
        try:
            df = market_data.fetch_and_store(symbol, timeframe, limit=1000)
            result = backtester.run(symbol, df)
            backtester.save_csv(result)
        except Exception as e:
            logger.error(f"[Backtest] {symbol} başarısız: {e}")


def run_paper(cfg: dict) -> None:
    from data.exchange_client import ExchangeClient
    from paper.paper_trader import PaperTrader
    from monitoring.telegram_notifier import TelegramNotifier

    client = ExchangeClient(cfg["exchange"]["name"], testnet=cfg["exchange"]["testnet"])
    client.connect()

    tg_cfg = cfg.get("telegram", {})
    notifier = TelegramNotifier(enabled=tg_cfg.get("enabled", False))

    risk_cfg = cfg.get("risk", {})
    strat_cfg = cfg.get("strategy", {})
    paper_cfg = cfg.get("paper", {})
    pyr_cfg = cfg.get("pyramiding", {}) or {}

    trader = PaperTrader(
        client=client,
        symbols=cfg["trading"]["symbols"],
        timeframe=cfg["trading"]["timeframe"],
        initial_capital=paper_cfg.get("initial_capital", 10_000.0),
        risk_per_trade=risk_cfg.get("risk_per_trade", 0.01),
        daily_max_loss=risk_cfg.get("daily_max_loss", 0.03),
        atr_stop_multiplier=risk_cfg.get("atr_stop_multiplier", 2.0),
        trailing_stop_atr_multiplier=risk_cfg.get("trailing_stop_atr_multiplier", 2.5),
        adx_threshold=strat_cfg.get("adx_threshold", 20.0),
        rsi_lower=strat_cfg.get("rsi_lower", 45.0),
        rsi_upper=strat_cfg.get("rsi_upper", 70.0),
        min_atr_ratio=strat_cfg.get("min_atr_ratio", 0.002),
        max_open_positions=risk_cfg.get("max_open_positions", 3),
        min_order_size=risk_cfg.get("min_order_size", 10.0),
        max_position_pct=risk_cfg.get("max_position_pct", 0.20),
        volume_sma_multiplier=strat_cfg.get("volume_sma_multiplier", 0.8),
        notifier=notifier,
        adaptation_window=cfg.get("learning", {}).get("adaptation_window", 20),
        # ── Pyramiding (Stage 1) ───────────────────────────────────────────
        pyramid_enabled=bool(pyr_cfg.get("enabled", False)),
        pyramid_thresholds_atr=list(pyr_cfg.get("thresholds_atr", [1.5, 3.0])),
        pyramid_size_pcts=list(pyr_cfg.get("size_pcts", [0.5, 0.25])),
        pyramid_max_adds=int(pyr_cfg.get("max_adds", 2)),
        pyramid_stop_atr_multiplier=float(pyr_cfg.get("stop_atr_multiplier", 2.0)),
    )
    trader.start()


def run_scalp(cfg: dict) -> None:
    """
    SCALP MOD — 5m grafik üzerinde yüksek frekanslı scalping.
    - Her 30 saniyede bir döngü
    - Adaptif online öğrenme (per-symbol Kelly eşik + pozisyon skalası)
    - Paper mod — gerçek emir gönderilmez
    """
    from data.exchange_client import ExchangeClient
    from paper.scalper_trader import ScalperTrader
    from monitoring.telegram_notifier import TelegramNotifier

    client = ExchangeClient(cfg["exchange"]["name"], testnet=cfg["exchange"]["testnet"])
    client.connect()

    tg_cfg    = cfg.get("telegram", {})
    notifier  = TelegramNotifier(enabled=tg_cfg.get("enabled", False))

    risk_cfg  = cfg.get("risk", {})
    scalp_cfg = cfg.get("scalping", {})

    trader = ScalperTrader(
        client=client,
        symbols=cfg["trading"]["symbols"],
        timeframe=scalp_cfg.get("timeframe", "5m"),
        initial_capital=scalp_cfg.get("initial_capital", 10_000.0),
        risk_per_trade=scalp_cfg.get("risk_per_trade", risk_cfg.get("risk_per_trade", 0.01)),
        daily_max_loss=scalp_cfg.get("daily_max_loss", 0.05),
        max_open_positions=scalp_cfg.get("max_open_positions", 5),
        min_order_size=risk_cfg.get("min_order_size", 10.0),
        max_position_pct=scalp_cfg.get("max_position_pct", 0.15),
        profit_target_pct=scalp_cfg.get("profit_target_pct", 0.005),
        stop_loss_pct=scalp_cfg.get("stop_loss_pct", 0.0025),
        max_hold_bars=scalp_cfg.get("max_hold_bars", 12),
        notifier=notifier,
    )
    trader.start()


def run_live(cfg: dict) -> None:
    """
    LIVE MOD — Gerçek para ile gerçek emir gönderir.
    config.yaml içinde live.enabled: true olmadan bu fonksiyon çalışmaz.
    """
    live_cfg = cfg.get("live", {})

    if not live_cfg.get("enabled", False):
        print("\n" + "=" * 60)
        print("  CANLI MOD KAPALI")
        print("  config.yaml dosyasında live.enabled: true yapılmadan")
        print("  gerçek işlem başlatılmaz.")
        print("  ÖNEMLİ: Canlı mod gerçek para kaybettirebilir.")
        print("=" * 60 + "\n")
        sys.exit(0)

    print("\n" + "!" * 60)
    print("  UYARI: CANLI MOD AKTİF — GERÇEK PARA RİSKİ VAR!")
    print("  Devam etmek için 'EVET' yazın:")
    confirm = input("  > ").strip()
    if confirm.upper() != "EVET":
        print("  İptal edildi.")
        sys.exit(0)
    print("!" * 60 + "\n")

    # Live trading skeleti — Paper trader ile aynı ama live_mode=True
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
    import time

    logger = logging.getLogger("live")

    client = ExchangeClient(cfg["exchange"]["name"], testnet=False)
    client.connect()

    risk_cfg = cfg.get("risk", {})
    strat_cfg = cfg.get("strategy", {})
    tg_cfg = cfg.get("telegram", {})
    notifier = TelegramNotifier(enabled=tg_cfg.get("enabled", False))

    # Gerçek bakiye çek
    balance_data = client.fetch_balance()
    usdt_balance = balance_data.get("USDT", {}).get("free", 0.0)
    logger.info(f"[Live] Gerçek USDT bakiyesi: {usdt_balance:.2f}")

    repo = CandleRepository()
    market_data = MarketDataService(client, repo)
    indicators = TechnicalIndicators()
    strategy = TrendFollowingStrategy(
        rsi_lower=strat_cfg.get("rsi_lower", 45),
        rsi_upper=strat_cfg.get("rsi_upper", 70),
        adx_threshold=strat_cfg.get("adx_threshold", 20),
        min_atr_ratio=strat_cfg.get("min_atr_ratio", 0.002),
        indicators=indicators,
    )
    risk_manager = RiskManager(
        account_balance=usdt_balance,
        risk_per_trade=risk_cfg.get("risk_per_trade", 0.01),
        daily_max_loss=risk_cfg.get("daily_max_loss", 0.03),
        atr_stop_multiplier=risk_cfg.get("atr_stop_multiplier", 2.0),
        max_open_positions=risk_cfg.get("max_open_positions", 3),
        min_order_size=risk_cfg.get("min_order_size", 10.0),
        max_position_pct=risk_cfg.get("max_position_pct", 0.20),
    )
    position_manager = PositionManager(
        trailing_stop_atr_multiplier=risk_cfg.get("trailing_stop_atr_multiplier", 2.5)
    )
    order_manager = OrderManager(
        position_manager=position_manager,
        risk_manager=risk_manager,
        live_mode=True,
        exchange_client=client,
        trailing_stop_multiplier=risk_cfg.get("trailing_stop_atr_multiplier", 2.5),
    )

    notifier.bot_started("LIVE")
    symbols = cfg["trading"]["symbols"]
    timeframe = cfg["trading"]["timeframe"]
    last_day = None

    logger.info("[Live] Bot çalışıyor. Ctrl+C ile dur.")
    try:
        while True:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            day_str = now.strftime("%Y-%m-%d")
            if last_day and day_str != last_day:
                risk_manager.reset_daily_pnl()
            last_day = day_str

            current_prices = {}
            atrs = {}

            for symbol in symbols:
                try:
                    df = market_data.fetch_and_store(symbol, timeframe, limit=250)
                    df_ind = indicators.calculate(df)
                    last = df_ind.iloc[-1]
                    current_prices[symbol] = last["close"]
                    atrs[symbol] = last.get("atr", 0.0)
                except Exception as e:
                    logger.error(f"[Live] {symbol} veri hatası: {e}")
                    notifier.api_error(str(e))

            closed_list = position_manager.update_positions(current_prices, atrs)
            for pos, reason in closed_list:
                risk_manager.record_trade_pnl(pos.realized_pnl)
                notifier.position_closed(pos.symbol, pos.realized_pnl, reason)
                if "stop" in reason:
                    notifier.stop_loss_triggered(pos.symbol, pos.close_price or 0, pos.realized_pnl)

            if not risk_manager.trading_allowed:
                summary = risk_manager.summary()
                notifier.daily_loss_limit_hit(summary["daily_pnl"], summary["daily_loss_limit"])
                time.sleep(60)
                continue

            for symbol in symbols:
                df = repo.get(symbol, timeframe)
                if df is None:
                    continue
                try:
                    df_ind = indicators.calculate(df)
                    signal = strategy.generate_signal(symbol, df_ind)
                    if signal.side == Side.BUY:
                        notifier.new_signal(symbol, signal.side.value, signal.price, signal.reason)
                        position = order_manager.process_signal(signal, atrs.get(symbol, 0.0))
                        if position:
                            notifier.position_opened(
                                symbol, position.entry_price, position.stop_price, position.position_size
                            )
                    elif signal.side == Side.HOLD:
                        should_exit, exit_reason = strategy.should_exit(symbol, df_ind, 0)
                        if should_exit and position_manager.has_open_position(symbol):
                            price = current_prices.get(symbol, 0.0)
                            closed = order_manager.close_position(symbol, price, exit_reason)
                            if closed:
                                notifier.position_closed(symbol, closed.realized_pnl, exit_reason)
                except Exception as e:
                    logger.error(f"[Live] {symbol} sinyal hatası: {e}")

            time.sleep(60)

    except KeyboardInterrupt:
        logger.info("[Live] Kapatılıyor...")
        position_manager.report_open_positions()


def run_trend_backtest(cfg: dict, start_date: str, end_date: str) -> None:
    """
    TREND BACKTEST MOD — TrendFollowingStrategy (1h) için sayfalı veri + bar-by-bar sim.
    Mevcut Backtester sınıfını kullanır (35/35 test geçmiş).
    """
    import time as _time
    import numpy as np
    from data.exchange_client import ExchangeClient
    from backtest.metrics import calculate_metrics, print_metrics

    client = ExchangeClient(cfg["exchange"]["name"], testnet=False)
    client.connect()

    bt_cfg    = cfg.get("backtest", {})
    risk_cfg  = cfg.get("risk", {})
    strat_cfg = cfg.get("strategy", {})

    symbols   = cfg["trading"]["symbols"]
    timeframe = cfg["trading"]["timeframe"]   # "1h"
    BARS_PER_REQ = 1000

    import datetime as _dt
    _start_dt = _dt.datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=_dt.timezone.utc)
    _end_dt   = _dt.datetime.strptime(end_date,   "%Y-%m-%d").replace(tzinfo=_dt.timezone.utc)

    # BTC 200-günlük SMA ve coin EMA göstergelerinin doğru ısınması için
    # gerçek başlangıçtan 210 gün önce veri çekiyoruz.
    # Backtest yalnızca user'ın istediği start_date'ten itibaren trade üretir.
    WARMUP_DAYS = 210
    _fetch_start_dt = _start_dt - _dt.timedelta(days=WARMUP_DAYS)

    start_ts      = int(_fetch_start_dt.timestamp() * 1000)   # veri çekme başlangıcı
    trade_start_ts = int(_start_dt.timestamp() * 1000)         # trade kaydı başlangıcı
    end_ts        = int(_end_dt.timestamp() * 1000)

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  TREND BACKTEST — TrendFollowingStrategy (1h) + Per-Coin Profil")
    print(f"  Dönem  : {start_date} → {end_date}")
    print(f"  Sermaye: ${bt_cfg.get('initial_capital', 10000):>8,.0f}  |  "
          f"Baseline ADX≥{strat_cfg.get('adx_threshold',15)}  |  "
          f"RSI {strat_cfg.get('rsi_lower',45)}-{strat_cfg.get('rsi_upper',70)}")
    print(f"  Semboller ({len(symbols)}): {', '.join(symbols)}")
    profiles = cfg.get("symbol_profiles", {})
    if profiles:
        print(f"  Per-Coin Profil: {', '.join(profiles.keys())}")
    print(f"{sep}\n")

    all_results  = []
    all_sym_data = {}   # ham veri — BTC rejimi için saklıyoruz

    for sym in symbols:
        print(f"  ↓ {sym} veri çekiliyor ({timeframe})...")
        all_rows, since = [], start_ts
        while since < end_ts:
            rows = client.fetch_ohlcv(sym, timeframe, since=since, limit=BARS_PER_REQ)
            if not rows:
                break
            all_rows.extend(rows)
            last_ts = rows[-1][0]
            if last_ts >= end_ts or len(rows) < BARS_PER_REQ:
                break
            since = last_ts + 1
            _time.sleep(0.15)

        if not all_rows:
            print(f"    ⚠️  {sym}: veri bulunamadı, atlanıyor.")
            continue

        import pandas as pd
        df = pd.DataFrame(all_rows, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df[df.index < pd.Timestamp(end_date, tz="UTC")]
        df.drop_duplicates(inplace=True)
        df.sort_index(inplace=True)

        if df.empty:
            print(f"    ⚠️  {sym}: tarih filtresi sonrası veri kalmadı, atlanıyor.")
            continue

        print(f"    ✅ {sym}: {len(df)} bar ({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})")
        all_sym_data[sym] = df

    if not all_sym_data:
        print("Hiçbir sembol için veri indirilemedi.")
        return

    # ── BTC Rejim Serisi Oluştur ──────────────────────────────────────
    # BTC 1h verisi → günlük kapanış → 200-günlük SMA → bull/bear flag
    btc_regime = None
    if "BTC/USDT" in all_sym_data:
        btc_1h       = all_sym_data["BTC/USDT"]["close"]
        btc_daily    = btc_1h.resample("1D").last().dropna()
        btc_sma200   = btc_daily.rolling(200, min_periods=50).mean()
        btc_is_bull  = (btc_daily > btc_sma200).astype(bool)
        # 1h indeksine geri genişlet (ileriye doldur — o günün durumu geçerli)
        btc_regime   = btc_is_bull.reindex(btc_1h.index, method="ffill").fillna(True)
        bull_pct = btc_regime.mean() * 100
        print(f"\n📊 BTC Rejim: Bull={bull_pct:.0f}%  Bear={100-bull_pct:.0f}%  "
              f"(200-günlük SMA bazlı)\n")
    else:
        print("⚠️  BTC/USDT verisi yok — rejim filtresi devre dışı\n")

    # ── Her sembol için özelleştirilmiş Backtester ile çalıştır ─────────
    import pandas as _pd
    _trade_start = _pd.Timestamp(_start_dt)   # trade kaydı başlangıcı (warmup hariç)

    # Cross-symbol correlation registry sıfırla — process-level shared dict
    # NOT: Per-symbol Backtester'lar SIRAYLA çalışıyor (paralel değil), bu nedenle
    # registry mevcut mimaride beklenen "eş-zamanlı" davranışı sergilemez.
    # MVP olarak bırakıldı; canlı (PaperTrader) eş-zamanlıdır, orada doğru çalışır.
    # Backtest'te düzgün çalışması için time-synchronized multi-symbol refactor gerekir.
    from risk import correlation_registry as _corr_reg
    _corr_reg.reset()

    for sym, df in all_sym_data.items():
        try:
            # Per-coin profil: ETH/DOT için sıkı filtre, FET/INJ için geniş trailing
            bt = make_backtester_for_symbol(sym, cfg)
            result = bt.run(sym, df, btc_regime=btc_regime,
                            trade_start=_trade_start)
            bt.save_csv(result)
            all_results.append(result)
        except Exception as e:
            print(f"    ❌ {sym} backtest hatası: {e}")

    if not all_results:
        print("Hiçbir sembol için sonuç üretilemedi.")
        return

    # Aggregate özet tablo
    print(f"\n{'─'*80}")
    print(f"  {'SEMBOL':<12} {'İŞLEM':>6} {'WR%':>7} {'Getiri%':>9} {'PF':>7} {'Sharpe':>8} {'MaxDD%':>8}")
    print(f"  {'─'*72}")
    for r in sorted(all_results, key=lambda x: -x["metrics"]["total_return_pct"]):
        m   = r["metrics"]
        sym = r["symbol"]
        print(
            f"  {sym:<12} {m['num_trades']:>6} {m['win_rate_pct']:>6.1f}% "
            f"{m['total_return_pct']:>8.2f}% {m['profit_factor']:>7.3f} "
            f"{m['sharpe_ratio']:>8.3f} {m['max_drawdown_pct']:>7.2f}%"
        )
    print(f"{'─'*80}\n")

    # Aggregate metrik — tüm işlemleri kapanış zamanına göre sıralayıp
    # tek bir $10k hesabında kronolojik olarak uygula.
    # (Önceki yöntem: sembol equity curve'lerini yan yana yapıştırıyordu
    #  → son sembolün değeri aggregate sonu sayılıyordu, BUG.)
    initial_cap = bt_cfg.get("initial_capital", 10_000.0)
    all_trades_flat = [t for r in all_results for t in r["trades"]]

    if all_trades_flat:
        # Kapanış zamanına göre sırala (string ISO format sıralanabilir)
        all_trades_sorted = sorted(
            all_trades_flat,
            key=lambda t: t.get("closed_at") or t.get("opened_at") or "",
        )
        # Tek hesap üzerinde çalıştır
        agg_equity: list[float] = [initial_cap]
        for t in all_trades_sorted:
            agg_equity.append(agg_equity[-1] + t["pnl"])

        # Gerçek test süresi (saat cinsinden) → Calmar yıllıklandırması için
        _period_hours = int((_end_dt - _start_dt).total_seconds() / 3600)
        agg = calculate_metrics(all_trades_sorted, agg_equity, initial_cap,
                                n_hours=_period_hours)
        print_metrics(agg, "TÜM SEMBOLLER (AGGREGATE)")


def run_scalp_backtest(cfg: dict, start_date: str, end_date: str) -> None:
    """
    SCALP BACKTEST MOD — ScalpingStrategy v2 için bar-by-bar simülasyon.
    5m OHLCV verisi paginated olarak çekilir; TP/SL bar high/low ile simüle edilir.
    """
    from data.exchange_client import ExchangeClient
    from backtest.scalp_backtester import ScalpBacktester

    client = ExchangeClient(cfg["exchange"]["name"], testnet=False)
    client.connect()

    scalp_cfg = cfg.get("scalping", {})
    risk_cfg  = cfg.get("risk", {})

    bt = ScalpBacktester(
        client            = client,
        symbols           = cfg["trading"]["symbols"],
        start_date        = start_date,
        end_date          = end_date,
        initial_capital   = scalp_cfg.get("initial_capital", 10_000.0),
        risk_per_trade    = scalp_cfg.get("risk_per_trade",  risk_cfg.get("risk_per_trade", 0.01)),
        max_open_positions= scalp_cfg.get("max_open_positions", 5),
        max_position_pct  = scalp_cfg.get("max_position_pct", 0.15),
        profit_target_pct = scalp_cfg.get("profit_target_pct", 0.008),
        stop_loss_pct     = scalp_cfg.get("stop_loss_pct", 0.003),
        max_hold_bars     = scalp_cfg.get("max_hold_bars", 8),
        daily_max_loss    = scalp_cfg.get("daily_max_loss", 0.05),
        volume_mult       = cfg.get("strategy", {}).get("volume_sma_multiplier", 0.4),
        output_dir        = "backtest_results",
    )
    bt.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Kripto Trend-Following Bot")
    parser.add_argument(
        "--mode",
        choices=["backtest", "paper", "scalp", "live", "scalp-backtest", "trend-backtest"],
        required=True,
        help="Çalışma modu",
    )
    parser.add_argument("--config", default="config.yaml", help="Config dosyası yolu")
    parser.add_argument(
        "--start",
        default=None,
        help="Scalp backtest başlangıç tarihi (YYYY-MM-DD). Varsayılan: 3 ay önce.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Scalp backtest bitiş tarihi (YYYY-MM-DD). Varsayılan: bugün.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    log_cfg = cfg.get("logging", {})
    setup_logger(
        "root",
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file", "logs/bot.log"),
        max_bytes=log_cfg.get("max_bytes", 10_485_760),
        backup_count=log_cfg.get("backup_count", 5),
    )

    if args.mode == "backtest":
        run_backtest(cfg)
    elif args.mode == "paper":
        run_paper(cfg)
    elif args.mode == "scalp":
        run_scalp(cfg)
    elif args.mode == "live":
        run_live(cfg)
    elif args.mode == "scalp-backtest":
        from datetime import date, timedelta
        end_date   = args.end   or date.today().isoformat()
        start_date = args.start or (date.today() - timedelta(days=90)).isoformat()
        run_scalp_backtest(cfg, start_date, end_date)
    elif args.mode == "trend-backtest":
        from datetime import date, timedelta
        end_date   = args.end   or date.today().isoformat()
        start_date = args.start or (date.today() - timedelta(days=180)).isoformat()
        run_trend_backtest(cfg, start_date, end_date)


if __name__ == "__main__":
    main()
