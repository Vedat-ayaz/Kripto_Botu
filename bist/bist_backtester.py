"""
BIST 100 USD Backtester.

Mevcut Backtester sınıfını (backtest/backtester.py) yeniden kullanır.
BIST'e özgü katkılar:
  - yfinance ile OHLCV veri çekimi
  - USD/TRY dönüşümü (convert_series modu)
  - XU100 rejim filtresi + USDTRY kriz koruması (btc_regime parametresiyle entegre)
  - Lot rounding ve BIST komisyon modeli

Kullanım:
    from bist.bist_backtester import run_bist_backtest
    run_bist_backtest(cfg, start_date="2020-01-01", end_date="2026-01-01")
"""
import logging
import os
import sys

import pandas as pd

logger = logging.getLogger(__name__)

# Project root'u sys.path'e ekle (bist/ altından çalıştırılırken)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def make_bist_backtester_for_symbol(symbol: str, cfg: dict):
    """
    Sembol bazlı özelleştirilmiş Backtester oluşturur.
    main.py'deki make_backtester_for_symbol'ün BIST eşdeğeri.
    """
    from backtest.backtester import Backtester

    risk_cfg  = dict(cfg.get("risk", {}))
    strat_cfg = dict(cfg.get("strategy", {}))
    bt_cfg    = cfg.get("backtest", {})
    pyr_cfg   = cfg.get("pyramiding", {}) or {}
    pe_cfg    = cfg.get("partial_exits", {}) or {}
    fil_cfg   = cfg.get("filters", {}) or {}

    merged = {**strat_cfg, **risk_cfg}
    profile = cfg.get("symbol_profiles", {}).get(symbol, {})
    if profile:
        merged.update(profile)
        logger.info(f"[BistBT] {symbol} özel profil uygulandı: {profile}")

    sym_pyramid_enabled = merged.get("pyramid_enabled", pyr_cfg.get("enabled", False))

    return Backtester(
        initial_capital              = bt_cfg.get("initial_capital", 10_000.0),
        commission_rate              = bt_cfg.get("commission_rate", 0.0005),
        slippage_rate                = bt_cfg.get("slippage_rate", 0.001),
        risk_per_trade               = merged.get("risk_per_trade", 0.015),
        daily_max_loss               = merged.get("daily_max_loss", 0.04),
        atr_stop_multiplier          = merged.get("atr_stop_multiplier", 2.0),
        trailing_stop_atr_multiplier = merged.get("trailing_stop_atr_multiplier", 4.0),
        adx_threshold                = merged.get("adx_threshold", 18.0),
        rsi_lower                    = merged.get("rsi_lower", 45.0),
        rsi_upper                    = merged.get("rsi_upper", 70.0),
        min_atr_ratio                = merged.get("min_atr_ratio", 0.005),
        volume_sma_multiplier        = merged.get("volume_sma_multiplier", 0.4),
        max_open_positions           = merged.get("max_open_positions", 6),
        min_order_size               = merged.get("min_order_size", 10.0),
        max_position_pct             = merged.get("max_position_pct", 0.20),
        entry_score_trend            = merged.get("entry_score_trend", 0.55),
        entry_score_ranging          = merged.get("entry_score_ranging", 0.60),
        pyramid_enabled              = bool(sym_pyramid_enabled),
        pyramid_thresholds_atr       = list(merged.get("pyramid_thresholds_atr", pyr_cfg.get("thresholds_atr", [1.5, 3.0]))),
        pyramid_size_pcts            = list(merged.get("pyramid_size_pcts", pyr_cfg.get("size_pcts", [0.5, 0.25]))),
        pyramid_max_adds             = int(merged.get("pyramid_max_adds", pyr_cfg.get("max_adds", 2))),
        pyramid_stop_atr_multiplier  = float(merged.get("pyramid_stop_atr_multiplier", pyr_cfg.get("stop_atr_multiplier", 2.0))),
        pyramid_gate_min_regime      = float(merged.get("pyramid_gate_min_regime", pyr_cfg.get("gate", {}).get("min_regime_score", 0.50))),
        pyramid_gate_max_vol_spike   = float(merged.get("pyramid_gate_max_vol_spike", pyr_cfg.get("gate", {}).get("max_vol_spike", 1.50))),
        pyramid_gate_max_atr_ratio   = float(merged.get("pyramid_gate_max_atr_ratio", pyr_cfg.get("gate", {}).get("max_atr_ratio", 0.060))),
        pyramid_gate_min_adx         = float(merged.get("pyramid_gate_min_adx", pyr_cfg.get("gate", {}).get("min_adx", 20.0))),
        partial_exit_enabled         = bool(merged.get("partial_exit_enabled", pe_cfg.get("enabled", False))),
        partial_exit_r_levels        = list(merged.get("partial_exit_r_levels", pe_cfg.get("r_multiple_levels", [1.5, 3.0]))),
        partial_exit_pcts            = list(merged.get("partial_exit_pcts", pe_cfg.get("exit_pcts", [0.30, 0.30]))),
        partial_exit_max             = int(merged.get("partial_exit_max", pe_cfg.get("max_exits", 2))),
        choppiness_threshold         = float(merged.get("choppiness_threshold", fil_cfg.get("choppiness_threshold", 61.8))),
        choppiness_enabled           = bool(merged.get("choppiness_enabled", fil_cfg.get("choppiness_enabled", True))),
        mtf_filter_enabled           = bool(merged.get("mtf_filter_enabled", fil_cfg.get("mtf_filter_enabled", False))),
        # Daily bar'a göre ayarlanmış lookback'ler
        # 1h'de slope_bars=20 (20 saat), daily'de 5 (1 hafta)
        # 1h'de momentum_lookback=720 (30 gün), daily'de 30 (1 ay)
        slope_bars                   = int(merged.get("slope_bars", 5)),
        momentum_lookback            = int(merged.get("momentum_lookback", 30)),
        adx_boost                    = float(merged.get("adx_boost", 0.0)),
        regime_trending_threshold    = float(merged.get("regime_trending_threshold", 0.40)),
        regime_ranging_threshold     = float(merged.get("regime_ranging_threshold", 0.20)),
        monthly_loss_limit           = float(merged.get("monthly_loss_limit", 0.04)),
    )


def run_bist_backtest(cfg: dict, start_date: str, end_date: str) -> None:
    """
    BIST 100 trend-following backtest.
    yfinance'ten günlük veri çeker, USD'ye çevirir, mevcut Backtester ile simüle eder.
    """
    import datetime as _dt
    from backtest.metrics import calculate_metrics, print_metrics
    from bist.data.yfinance_provider import YFinanceProvider
    from bist.data.usdtry_provider import USDTRYProvider
    from bist.adapters.price_converter import PriceConverter
    from bist.filters.macro_filter import MacroFilter, fetch_xu100
    from risk import correlation_registry as _corr_reg

    bt_cfg    = cfg.get("backtest", {})
    strat_cfg = cfg.get("strategy", {})
    macro_cfg = cfg.get("macro", {})
    data_cfg  = cfg.get("data", {})

    symbols   = cfg["trading"]["symbols"]
    usd_mode  = data_cfg.get("usd_mode", "convert_series")

    # Warmup için gerçek başlangıçtan 250 gün önce veri çek
    WARMUP_DAYS = 250
    _start_dt = _dt.datetime.strptime(start_date, "%Y-%m-%d")
    _fetch_start = (_start_dt - _dt.timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
    _fetch_end   = end_date

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  BIST 100 USD TREND BACKTEST")
    print(f"  Dönem   : {start_date} → {end_date}")
    print(f"  Sermaye : ${bt_cfg.get('initial_capital', 10_000):>8,.0f}")
    print(f"  USD Mod : {usd_mode}")
    print(f"  Semboller ({len(symbols)}): {', '.join(symbols)}")
    print(f"{sep}\n")

    # ── USD/TRY kur serisi ────────────────────────────────────────────────────
    print("  ↓ USD/TRY kuru çekiliyor...")
    try:
        usdtry = USDTRYProvider(cache_enabled=data_cfg.get("cache_enabled", True)).fetch(
            start=_fetch_start, end=_fetch_end
        )
        print(f"    ✅ USDTRY: {len(usdtry)} gün, "
              f"min={usdtry.min():.2f}, max={usdtry.max():.2f}")
    except Exception as e:
        print(f"    ❌ USDTRY çekilemedi: {e}")
        return

    # ── XU100 rejim + USDTRY kriz filtresi ───────────────────────────────────
    regime_enabled  = macro_cfg.get("regime_filter_enabled", True)
    usdtry_guard    = macro_cfg.get("usdtry_guard_enabled", True)
    macro_filter    = MacroFilter(
        regime_filter_enabled  = regime_enabled,
        regime_ema_period      = macro_cfg.get("regime_ema_period", 200),
        regime_usd_adjusted    = macro_cfg.get("regime_usd_adjusted", False),
        usdtry_guard_enabled   = usdtry_guard,
        usdtry_momentum_period = macro_cfg.get("usdtry_momentum_period", 20),
        usdtry_crisis_threshold= macro_cfg.get("usdtry_crisis_threshold", 0.15),
    )

    xu100 = None
    if regime_enabled:
        print("  ↓ XU100 endeks çekiliyor...")
        try:
            xu100 = fetch_xu100(start=_fetch_start, end=_fetch_end)
            print(f"    ✅ XU100: {len(xu100)} gün")
        except Exception as e:
            print(f"    ⚠️  XU100 çekilemedi ({e}) — rejim filtresi devre dışı.")
            macro_filter.regime_filter_enabled = False

    macro_filter.fit(xu100 if xu100 is not None else pd.Series(dtype=float), usdtry)
    stats = macro_filter.summary_stats()
    if stats:
        parts = []
        if "regime_bull_pct" in stats:
            parts.append(f"Bull={stats['regime_bull_pct']:.0f}%")
        if "usdtry_crisis_pct" in stats:
            parts.append(f"TRY-kriz={stats['usdtry_crisis_pct']:.0f}%")
        print(f"\n  Makro filtre: {' | '.join(parts)}\n")

    # ── OHLCV verisi + backtest ──────────────────────────────────────────────
    provider  = YFinanceProvider(cache_enabled=data_cfg.get("cache_enabled", True))
    converter = PriceConverter(mode=usd_mode)

    all_results: list[dict] = []
    all_sym_data: dict[str, pd.DataFrame] = {}

    for sym in symbols:
        print(f"  ↓ {sym} ({sym}.IS) veri çekiliyor...")
        try:
            df_try = provider.fetch(
                symbol=sym,
                start=_fetch_start,
                end=_fetch_end,
                interval=data_cfg.get("interval", "1d"),
            )
            df_usd = converter.convert_ohlcv(df_try, usdtry)
            print(
                f"    ✅ {sym}: {len(df_usd)} bar "
                f"({df_usd.index[0].strftime('%Y-%m-%d')} → "
                f"{df_usd.index[-1].strftime('%Y-%m-%d')})"
            )
            all_sym_data[sym] = df_usd
        except Exception as e:
            print(f"    ❌ {sym} atlandı: {e}")

    if not all_sym_data:
        print("Hiçbir sembol için veri indirilemedi.")
        return

    # Macro regime series → btc_regime parametresiyle mevcut Backtester'a geç.
    # XU100 bull (True) AND USDTRY kriz yok (False) → giriş izni True
    # Bu boolean seri, backtester'daki strategy.set_btc_regime(is_bull) ile beslenir.
    def _build_macro_regime(
        index: pd.DatetimeIndex,
        df_usd: pd.DataFrame,
        sym_rs_positive: bool = False,
        sym_macro_bear: bool = False,
    ) -> pd.Series:
        """Sembol indeksine hizalanmış birleşik makro rejim serisi.

        Bear dönemde:
          - RS pozitif hisseler: entry izni devam eder (True)
          - RS negatif hisseler: entry kapatılır (False) → hiç trade olmaz

        Dinamik RS: Her gün hisse vs XU100 USD 20-günlük momentum karşılaştırması.
        """
        series = pd.Series(True, index=index)  # default: izin ver
        regime_s = macro_filter.get_regime_series()
        crisis_s = macro_filter.get_crisis_series()

        if regime_s is not None:
            aligned = regime_s.reindex(index, method="ffill").bfill().fillna(True)
            series &= aligned.astype(bool)

        if crisis_s is not None:
            aligned = crisis_s.reindex(index, method="ffill").bfill().fillna(False)
            series &= ~aligned.astype(bool)

        # Hangi günlerin bear/bull olduğunu belirle
        if regime_s is not None:
            bull_mask = regime_s.reindex(index, method="ffill").bfill().fillna(True).astype(bool)
            bear_mask = ~bull_mask
        else:
            bull_mask = pd.Series(True, index=index)
            bear_mask = pd.Series(False, index=index)

        if xu100 is not None:
            xu100_usd = xu100 / usdtry.reindex(xu100.index, method="ffill").bfill()
            stock_mom_20 = df_usd["close"].pct_change(20)
            xu100_mom = xu100_usd.pct_change(20).reindex(index, method="ffill").bfill()
            outperform = (stock_mom_20 > xu100_mom).astype(float)
            rs_rolling = outperform.rolling(15, min_periods=8).mean().reindex(index, method="ffill").bfill().fillna(0.0)

            # Bull dönem: RS eşiği orta (0.50) — XU100'ü en az yarı günde geçmeli
            # Bear dönem: RS eşiği sert (0.65) — sadece gerçek outperformer'lar
            rs_threshold = pd.Series(
                bull_mask.map({True: 0.50, False: 0.65}), index=index
            )
            rs_pass = rs_rolling >= rs_threshold
            series = series & rs_pass

        # Mutlak momentum filtresi:
        # Bull dönem: sadece pozitif yönde (>0) yeterli
        # Bear dönem: en az %0.5 yükselmiş olmalı
        abs_mom_20 = df_usd["close"].pct_change(20).reindex(index).fillna(0.0)
        mom_threshold = pd.Series(
            bull_mask.map({True: 0.0, False: 0.005}), index=index
        )
        mom_pass = abs_mom_20 > mom_threshold
        series = series & mom_pass

        # EMA50 filtresi: hisse kendi EMA50'sinin üzerinde olmalı (tüm dönemlerde)
        ema50 = df_usd["close"].ewm(span=50, adjust=False).mean()
        above_ema50 = (df_usd["close"] > ema50).reindex(index).fillna(False)
        series = series & above_ema50

        return series

    # Correlation registry sıfırla
    _corr_reg.reset()

    trade_start = pd.Timestamp(start_date, tz="UTC")

    import copy
    from bist.optimizer.symbol_calibrator import calibrate, PROFILES

    for sym, df_usd in all_sym_data.items():
        try:
            # Sembol için adaptif parametre kalibrasyonu
            # trade_start'tan önceki veriyi kullan (look-ahead bias yok)
            warmup_df = df_usd[df_usd.index < trade_start]
            # Kalibrasyon tarihinde makro rejim durumunu al
            cal_date = trade_start - pd.Timedelta(days=1)
            macro_bear = not macro_filter.is_entry_allowed(cal_date)

            # RS (relative strength) hesapla — gerçek XU100 USD karşılaştırması
            rs_positive = False
            if len(warmup_df) >= 20 and xu100 is not None:
                # XU100 USD = XU100 TRY / USDTRY
                xu100_usd_warm = xu100 / usdtry.reindex(xu100.index, method="ffill").bfill()
                stock_mom_warm = warmup_df["close"].pct_change(20)
                xu100_mom_warm = xu100_usd_warm.pct_change(20).reindex(
                    warmup_df.index, method="ffill"
                ).bfill()
                outperform_warm = (stock_mom_warm > xu100_mom_warm).astype(float)
                recent_rs_score = float(outperform_warm.iloc[-10:].mean())  # Son 10 günün ortalaması
                rs_positive = recent_rs_score >= 0.55  # %55+ günde XU100'ü outperform

            cal_params = calibrate(
                warmup_df,
                lookback=60,
                macro_is_bear=macro_bear,
                rs_positive=rs_positive,
            )

            logger.info(
                f"[BistBT] {sym} profil: {cal_params['profile']} | "
                f"entry={cal_params['entry_score_ranging']:.2f} "
                f"stop={cal_params['atr_stop_multiplier']:.1f}x "
                f"risk={cal_params['risk_per_trade']:.3f}"
            )

            # cal_params'ı cfg'e geçici olarak uygula (symbol_profiles override'ı)
            sym_cfg = copy.deepcopy(cfg)
            # cal_params → varsayılan, symbol_profiles → öncelikli override
            # (elle ayarlanan sembol ayarları otomatik kalibrasyonu ezer)
            sym_cfg.setdefault("symbol_profiles", {})[sym] = {
                **{k: v for k, v in cal_params.items() if k != "profile"},
                **sym_cfg.get("symbol_profiles", {}).get(sym, {}),
            }

            macro_regime = _build_macro_regime(
                df_usd.index, df_usd,
                sym_rs_positive=rs_positive,
                sym_macro_bear=macro_bear,
            )
            bt = make_bist_backtester_for_symbol(sym, sym_cfg)
            result = bt.run(sym, df_usd, btc_regime=macro_regime, trade_start=trade_start)
            bt.save_csv(result, output_dir="backtest_results/bist")
            all_results.append(result)
        except Exception as e:
            logger.exception(f"[BistBT] {sym} backtest hatası")
            print(f"    ❌ {sym} backtest hatası: {e}")

    if not all_results:
        print("Hiçbir sembol için sonuç üretilemedi.")
        return

    # ── Sonuç tablosu ─────────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"  {'SEMBOL':<10} {'İŞLEM':>6} {'WR%':>7} {'Getiri%':>9} {'PF':>7} {'Sharpe':>8} {'MaxDD%':>8}")
    print(f"  {'─'*72}")
    for r in sorted(all_results, key=lambda x: -x["metrics"]["total_return_pct"]):
        m   = r["metrics"]
        sym = r["symbol"]
        ret = m["total_return_pct"]
        sign = "+" if ret >= 0 else ""
        print(
            f"  {sym:<10} {m['num_trades']:>6} {m['win_rate_pct']:>6.1f}% "
            f"{sign}{ret:>7.2f}% {m['profit_factor']:>7.3f} "
            f"{m['sharpe_ratio']:>8.3f} {m['max_drawdown_pct']:>7.2f}%"
        )
    print(f"{'─'*80}")

    # Toplam sermaye büyümesi (tüm trade'leri kronolojik olarak uygula)
    all_trades = []
    for r in all_results:
        all_trades.extend(r.get("trades", []))
    if all_trades:
        all_trades.sort(key=lambda t: t.get("exit_time", ""))
        initial = bt_cfg.get("initial_capital", 10_000.0)
        final = initial
        for t in all_trades:
            final += t.get("pnl", 0.0)
        total_ret = (final / initial - 1) * 100
        print(f"\n  Toplam Getiri (kronolojik): {'+' if total_ret >= 0 else ''}{total_ret:.2f}%")
        print(f"  Başlangıç: ${initial:,.0f}  →  Son: ${final:,.0f}")
    print()
