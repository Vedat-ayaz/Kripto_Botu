"""Walk-forward optimizasyon çerçevesi."""
import logging
import datetime as dt
from typing import Any

import pandas as pd

from bist.optimizer.param_space import grid_sample
from bist.optimizer.objective import compute_score

logger = logging.getLogger(__name__)


def _make_windows(
    start: str, end: str,
    train_months: int = 15,
    test_months: int = 6,
    step_months: int = 6,
) -> list[tuple[str, str, str, str]]:
    """(train_start, train_end, test_start, test_end) listesi döndürür."""
    s = dt.datetime.strptime(start, "%Y-%m-%d")
    e = dt.datetime.strptime(end, "%Y-%m-%d")
    windows = []
    cur = s
    while True:
        train_end  = cur + dt.timedelta(days=train_months * 30)
        test_start = train_end
        test_end   = test_start + dt.timedelta(days=test_months * 30)
        if test_end > e:
            break
        windows.append((
            cur.strftime("%Y-%m-%d"),
            train_end.strftime("%Y-%m-%d"),
            test_start.strftime("%Y-%m-%d"),
            test_end.strftime("%Y-%m-%d"),
        ))
        cur += dt.timedelta(days=step_months * 30)
    return windows


def _run_window(
    cfg: dict,
    symbols: list[str],
    sym_data: dict[str, pd.DataFrame],
    usdtry: pd.Series,
    macro_regime_fn,
    params: dict[str, Any],
    period_start: str,
    period_end: str,
) -> list[dict]:
    """Verilen params ile tek bir pencerede tüm sembolleri backtest et."""
    import copy
    from bist.bist_backtester import make_bist_backtester_for_symbol

    trade_start = pd.Timestamp(period_start, tz="UTC")
    trade_end   = pd.Timestamp(period_end,   tz="UTC")

    # Pencereye göre veri dilimi al (bellek tasarrufu için)
    ts_start = pd.Timestamp(period_start, tz="UTC") - pd.Timedelta(days=300)  # warmup
    ts_end   = pd.Timestamp(period_end,   tz="UTC")

    # Config'e params'ı geçici olarak yaz
    patched_cfg = copy.deepcopy(cfg)
    patched_cfg.setdefault("strategy", {}).update(params)
    patched_cfg.setdefault("risk", {}).update(params)

    results = []
    for sym in symbols:
        if sym not in sym_data:
            continue
        df = sym_data[sym]
        df = df[df.index >= ts_start]  # sadece ilgili pencereyi geç
        macro_regime = macro_regime_fn(df.index)
        try:
            bt = make_bist_backtester_for_symbol(sym, patched_cfg)
            result = bt.run(
                sym, df,
                btc_regime=macro_regime,
                trade_start=trade_start,
                trade_end=trade_end,
            )
            results.append(result)
        except Exception as ex:
            logger.debug(f"[WF] {sym} hata: {ex}")
    return results


def run_walk_forward(
    cfg: dict,
    start: str = "2020-01-01",
    end: str   = "2026-01-01",
    n_samples: int = 80,
    train_months: int = 18,
    test_months: int  = 6,
    step_months: int  = 6,
    top_k: int = 3,
) -> dict[str, Any]:
    """
    Walk-forward optimizasyon çalıştırır.
    Döndürür: {"best_params": {...}, "windows": [...], "all_scores": [...]}
    """
    from bist.data.yfinance_provider import YFinanceProvider
    from bist.data.usdtry_provider import USDTRYProvider
    from bist.adapters.price_converter import PriceConverter
    from bist.filters.macro_filter import MacroFilter, fetch_xu100

    windows = _make_windows(start, end, train_months, test_months, step_months)
    logger.info(f"[WF] {len(windows)} pencere, {n_samples} örnek/pencere")
    print(f"\n  Walk-Forward: {len(windows)} pencere x {n_samples} parametre seti")

    # En erken tarihten veriyi bir kere çek
    WARMUP = 250
    fetch_start = (
        dt.datetime.strptime(windows[0][0], "%Y-%m-%d")
        - dt.timedelta(days=WARMUP)
    ).strftime("%Y-%m-%d")

    data_cfg  = cfg.get("data", {})
    macro_cfg = cfg.get("macro", {})
    symbols   = cfg["trading"]["symbols"]

    print("  Veri cekiliyor (tek seferlik)...")
    usdtry_prov = USDTRYProvider(cache_enabled=data_cfg.get("cache_enabled", True))
    usdtry = usdtry_prov.fetch(start=fetch_start, end=end)

    macro_filter = MacroFilter(
        regime_filter_enabled = macro_cfg.get("regime_filter_enabled", True),
        regime_ema_period     = macro_cfg.get("regime_ema_period", 200),
        regime_usd_adjusted   = macro_cfg.get("regime_usd_adjusted", False),
        usdtry_guard_enabled  = macro_cfg.get("usdtry_guard_enabled", True),
        usdtry_momentum_period    = macro_cfg.get("usdtry_momentum_period", 20),
        usdtry_crisis_threshold   = macro_cfg.get("usdtry_crisis_threshold", 0.15),
    )
    try:
        xu100 = fetch_xu100(start=fetch_start, end=end)
    except Exception:
        xu100 = pd.Series(dtype=float)
    macro_filter.fit(xu100 if len(xu100) else pd.Series(dtype=float), usdtry)

    def _macro_regime_fn(index: pd.DatetimeIndex) -> pd.Series:
        """Sembol indeksine hizalanmis birlesik makro rejim serisi."""
        series = pd.Series(True, index=index)
        regime_s = macro_filter.get_regime_series()
        crisis_s = macro_filter.get_crisis_series()
        if regime_s is not None:
            aligned = regime_s.reindex(index, method="ffill").bfill().fillna(True)
            series &= aligned.astype(bool)
        if crisis_s is not None:
            aligned = crisis_s.reindex(index, method="ffill").bfill().fillna(False)
            series &= ~aligned.astype(bool)
        return series

    provider  = YFinanceProvider(cache_enabled=data_cfg.get("cache_enabled", True))
    converter = PriceConverter(mode=data_cfg.get("usd_mode", "convert_series"))
    sym_data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df_try = provider.fetch(sym, start=fetch_start, end=end,
                                     interval=data_cfg.get("interval", "1d"))
            sym_data[sym] = converter.convert_ohlcv(df_try, usdtry)
        except Exception as ex:
            logger.debug(f"[WF] {sym} veri hatasi: {ex}")
    print(f"  {len(sym_data)}/{len(symbols)} sembol yuklendi")

    # Rastgele parametre setleri üret (tüm pencerelerde aynı setler kullanılır)
    param_sets = grid_sample(n_samples)

    all_window_results = []
    param_test_scores: dict[int, list[float]] = {i: [] for i in range(n_samples)}

    for wi, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        print(f"\n  Pencere {wi+1}/{len(windows)}: train={tr_s}->{tr_e}, test={te_s}->{te_e}")

        # Train
        train_scores = []
        for i, params in enumerate(param_sets):
            print(f"  [{wi+1}/{len(windows)}] Pencere | Parametre {i+1}/{n_samples}", end="\r", flush=True)
            res = _run_window(cfg, symbols, sym_data, usdtry,
                              _macro_regime_fn, params, tr_s, tr_e)
            sc = compute_score(res)
            train_scores.append((sc, i))

        train_scores.sort(reverse=True)
        top_indices = [idx for _, idx in train_scores[:top_k]]
        best_train = train_scores[0][0]
        print(f"    Train en iyi skor: {best_train:.4f} "
              f"(params #{top_indices[0]})")

        # Test (sadece top_k üzerinde)
        for idx in top_indices:
            params = param_sets[idx]
            res = _run_window(cfg, symbols, sym_data, usdtry,
                              _macro_regime_fn, params, te_s, te_e)
            sc = compute_score(res)
            param_test_scores[idx].append(sc)
            print(f"    Test #{idx}: skor={sc:.4f}")

        all_window_results.append({
            "window": (tr_s, tr_e, te_s, te_e),
            "top_train": [(s, param_sets[i]) for s, i in train_scores[:top_k]],
        })

    # Global en iyi: tüm pencere test skorlarının ortalaması
    avg_scores = []
    for idx, scores in param_test_scores.items():
        if scores:
            avg_scores.append((sum(scores) / len(scores), idx, scores))
    avg_scores.sort(reverse=True)

    best_idx = avg_scores[0][1]
    best_params = param_sets[best_idx]
    best_avg    = avg_scores[0][0]

    print(f"\n  {'='*60}")
    print(f"  EN IYI PARAMETRELER (avg test skoru: {best_avg:.4f})")
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    print(f"  {'='*60}")

    return {
        "best_params": best_params,
        "best_avg_score": best_avg,
        "windows": all_window_results,
        "all_avg_scores": avg_scores[:10],
    }
