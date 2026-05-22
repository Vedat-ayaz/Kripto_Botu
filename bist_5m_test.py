"""
BIST 5 dakikalık portföy backtest — son 60 gün.

TEK HAVUZ: $10.000 tüm hisselerde paylaşılır.
Robot BIST 100 içinden uygun hisse seçer, aynı anda max 4 pozisyon açar.
Sermaye otomatik bölünür, al-sat sonrası portföy bakiyesi güncellenir.

Kullanım:
    python bist_5m_test.py                          # varsayılan 20 hisse
    python bist_5m_test.py --all                    # tüm BIST 100
    python bist_5m_test.py --symbols GARAN,AKBNK    # belirli hisseler
    python bist_5m_test.py --days 40 --capital 5000
"""
import argparse
import datetime as dt
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.basicConfig(level=logging.WARNING)

# BIST seansı 09:30-18:00 İstanbul = 06:30-15:00 UTC → 510 dk / 5 = 102 bar/gün
BARS_PER_DAY = 102

# ── Portföy parametreleri ─────────────────────────────────────────────────────
MAX_POSITIONS        = 4      # Eş zamanlı maksimum açık pozisyon
RISK_PER_TRADE       = 0.012  # Portföy bakiyesinin %1.2'si her işlemde riske girer
MAX_POSITION_PCT     = 0.25   # Tek pozisyon portföyün max %25'i
ATR_STOP_MULT        = 2.0    # Stop = giriş - 2×ATR
TRAILING_ATR_MULT    = 4.0    # Trailing stop = fiyat - 4×ATR
COMMISSION           = 0.0005 # %0.05 alış + %0.05 satış
SLIPPAGE             = 0.001  # %0.1 slippage (gerçekçi BIST tahmini)
DAILY_MAX_LOSS_PCT   = 0.025  # Günlük max %2.5 kayıp → o gün yeni giriş yok
MIN_ATR_RATIO        = 0.003  # Bar başı minimum %0.3 hareket (komisyon güvencesi)

# Seans filtreleri (UTC)
MARKET_OPEN_UTC      = 6   # 06:30 UTC = 09:30 İstanbul
MORNING_ENTRY_END    = 10  # 10:00 UTC = 13:00 İstanbul (sabah seansı girişi sonu)
MARKET_CLOSE_UTC     = 15  # 15:00 UTC = 18:00 İstanbul (seans kapanışı)
EOD_FORCE_CLOSE_UTC  = 14  # 14:30 UTC = 17:30 İstanbul (gün sonu zorla kapat)

# Hisse seçim parametreleri
TOP_N_CANDIDATES     = 8    # CI filtresi geçen en iyi 8 hisse aday havuzu
MIN_CI_SCORE         = 0.0  # Skor > 0 olan tüm adaylar
CANDIDATE_CI_MAX     = 57   # Choppiness Index üst sınırı

# Varsayılan semboller (kısa test için)
DEFAULT_SYMBOLS = [
    "GARAN", "AKBNK", "THYAO", "EREGL", "TUPRS",
    "FROTO", "BIMAS", "TCELL", "SISE",  "KCHOL",
    "ASELS", "YKBNK", "SAHOL", "ARCLK", "TOASO",
    "EKGYO", "LOGO",  "KOZAL", "ENKAI", "PETKM",
]


# ── Pozisyon veri yapısı ──────────────────────────────────────────────────────
@dataclass
class PortfolioPosition:
    symbol:      str
    entry_price: float
    stop_price:  float
    trail_price: float
    size:        float        # Hisse adedi (USD cinsinden)
    cost:        float        # Giriş maliyeti (USD, komisyon dahil)
    entry_time:  pd.Timestamp
    entry_atr:   float
    exit_price:  float = 0.0
    exit_time:   Optional[pd.Timestamp] = None
    exit_reason: str = ""
    pnl:         float = 0.0


# ── Veri çekme ────────────────────────────────────────────────────────────────
def fetch_5m(symbol: str, start: str, end: str) -> pd.DataFrame:
    """5 dakikalık OHLCV çeker. yfinance 5m verisi son 60 günle sınırlıdır."""
    import yfinance as yf

    ticker  = f"{symbol}.IS"
    cutoff  = (dt.date.today() - dt.timedelta(days=58)).strftime("%Y-%m-%d")
    if start < cutoff:
        raw = yf.download(ticker, period="59d", interval="5m",
                          auto_adjust=True, progress=False)
    else:
        raw = yf.download(ticker, start=start, end=end, interval="5m",
                          auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"{ticker}: 5m veri boş")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    raw.columns = [c.lower() for c in raw.columns]
    if not raw.index.tz:
        raw.index = raw.index.tz_localize("UTC")
    else:
        raw.index = raw.index.tz_convert("UTC")
    raw.index.name = "datetime"
    return raw[["open", "high", "low", "close", "volume"]].copy()


# ── Teknik indikatör hesaplama ────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """EMA, RSI, ATR, ADX, hacim ortalaması + 1 saatlik trend onayı hesaplar."""
    d = df.copy()
    c, h, l, v = d["close"], d["high"], d["low"], d["volume"]

    # ── 5m indikatörler ─────────────────────────────────────────────────────
    d["ema50"]  = c.ewm(span=50,  adjust=False).mean()
    d["ema200"] = c.ewm(span=200, adjust=False).mean()

    delta  = c.diff()
    gain   = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss   = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + gain / loss.clip(lower=1e-9))

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()

    dm_pos = h.diff().clip(lower=0)
    dm_neg = (-l.diff()).clip(lower=0)
    atr14  = d["atr"]
    di_p   = 100 * dm_pos.rolling(14).mean() / atr14.clip(lower=1e-9)
    di_n   = 100 * dm_neg.rolling(14).mean() / atr14.clip(lower=1e-9)
    dx     = 100 * (di_p - di_n).abs() / (di_p + di_n).clip(lower=1e-9)
    d["adx"] = dx.rolling(14).mean()

    d["vol_sma"]    = v.rolling(20).mean()
    d["ema50_slope"] = d["ema50"].diff(8)

    # ── Önceki gün kapanışı (dünün son barı) ────────────────────────────────
    # "prev_day_close" sütunu: o barın ait olduğu güne önceki iş günü kapanışı.
    # Giriş filtresi: bugünün açılış fiyatı önceki gün kapanışının üzerinde olmalı
    # (gap-up) — bu negatif günlerden kaçınmayı sağlar.
    daily_close = d["close"].resample("1D").last().dropna().shift(1)
    d["prev_day_close"] = daily_close.reindex(d.index, method="ffill")

    return d


# ── 5m kalite skoru ───────────────────────────────────────────────────────────
def _5m_quality_score(df_full: pd.DataFrame) -> tuple[float, float]:
    """
    5m veriyle Choppiness Index + trend skoru hesaplar.
    Döndürür: (skor, avg_ci)
    """
    if len(df_full) < 200:
        return 0.0, 99.0

    c, h, l = df_full["close"], df_full["high"], df_full["low"]

    # Sıfır-aralık barları filtrele
    valid = (h - l) > 0
    if valid.sum() < 100:
        return 0.0, 99.0

    tr_v   = pd.concat([h[valid]-l[valid],
                        (h[valid]-c[valid].shift()).abs(),
                        (l[valid]-c[valid].shift()).abs()], axis=1).max(axis=1)
    tr_sum = tr_v.rolling(14).sum()
    rng    = (h[valid].rolling(14).max() - l[valid].rolling(14).min()).clip(lower=1e-6)
    ci_s   = 100 * np.log10((tr_sum / rng).clip(lower=1e-6)) / np.log10(14)
    ci_s   = ci_s.replace([np.inf, -np.inf], np.nan)
    if ci_s.dropna().empty:
        return 0.0, 99.0
    avg_ci = float(ci_s.dropna().mean())

    if avg_ci > CANDIDATE_CI_MAX:
        return 0.0, avg_ci

    # ATR/fiyat kontrolü
    atr14     = tr_v.rolling(14).mean()
    atr_ratio = float(atr14.dropna().iloc[-1] / c.iloc[-1]) if c.iloc[-1] > 0 else 0.0
    if atr_ratio < 0.002:
        return 0.0, avg_ci

    # EMA50 üzeri bar yüzdesi
    ema50     = c.ewm(span=50, adjust=False).mean()
    above_pct = float((c > ema50).mean())

    # ADX
    dm_pos = h.diff().clip(lower=0)
    dm_neg = (-l.diff()).clip(lower=0)
    di_p   = 100 * dm_pos.rolling(14).mean() / atr14.clip(lower=1e-9)
    di_n   = 100 * dm_neg.rolling(14).mean() / atr14.clip(lower=1e-9)
    dx     = 100 * (di_p - di_n).abs() / (di_p + di_n).clip(lower=1e-9)
    adx_v  = dx.rolling(14).mean()
    adx    = float(adx_v.dropna().iloc[-1]) if len(adx_v.dropna()) > 0 else 0.0

    if above_pct < 0.40 or adx < 13:
        return 0.0, avg_ci

    # Momentum (son 5 gün)
    lb       = min(BARS_PER_DAY * 5, len(c) - 1)
    momentum = float(c.iloc[-1] / c.iloc[-lb] - 1) if lb > 0 else 0.0

    ci_score   = max(0.0, (60 - avg_ci) / 22)
    mom_factor = 1.3 if momentum > 0.01 else (1.0 if momentum >= -0.01 else 0.6)
    score      = above_pct * min(adx / 30, 1.0) * ci_score * mom_factor
    return score, avg_ci


# ── Giriş sinyali kontrolü ────────────────────────────────────────────────────
def _check_entry_signal(row: pd.Series, prev_row: pd.Series) -> bool:
    """
    5m giriş koşulları (hepsi aynı anda sağlanmalı):
      1. EMA50 > EMA200 (trend yönü)
      2. EMA50 bu barda EMA200'ü yukarı kesti VEYA yakın zamanda kesti
      3. RSI 52-78 arası (aşırı alım değil, momentum var)
      4. ADX > 22 (trend güçlü)
      5. Hacim ortalamanın 1.8x üzerinde (gerçek ilgi)
      6. EMA50 eğimi pozitif (trend devam ediyor)
      7. ATR/fiyat > min_atr_ratio (yeterli hareket alanı)
    """
    if pd.isna(row.get("ema50")) or pd.isna(row.get("ema200")):
        return False
    if pd.isna(row.get("atr")) or row["atr"] <= 0:
        return False

    # 1-2: EMA crossover veya trend devam
    ema_bull = row["ema50"] > row["ema200"]
    if not ema_bull:
        return False

    # EMA yeni kesişim (son barda veya önceki barda)
    prev_cross = (prev_row["ema50"] <= prev_row["ema200"]) if not pd.isna(prev_row.get("ema50")) else False
    fresh_cross = prev_cross  # Yeni kesişim güçlü sinyal

    # 3. RSI
    rsi = row.get("rsi", 50)
    if pd.isna(rsi) or not (52 <= rsi <= 78):
        return False

    # 4. ADX
    adx = row.get("adx", 0)
    if pd.isna(adx) or adx < 22:
        return False

    # 5. Hacim
    vol     = row.get("volume", 0)
    vol_sma = row.get("vol_sma", 1)
    if pd.isna(vol_sma) or vol_sma <= 0 or vol < vol_sma * 1.8:
        return False

    # 6. EMA eğimi
    slope = row.get("ema50_slope", 0)
    if pd.isna(slope) or slope <= 0:
        return False

    # 7. ATR/fiyat
    atr_ratio = row["atr"] / row["close"] if row["close"] > 0 else 0
    if atr_ratio < MIN_ATR_RATIO:
        return False

    # 8. Önceki gün kapanışının üzerinde (günlük trend doğrulama)
    # Dünün kapanışından bugün en az -0.5% düşmemeli
    prev_close = row.get("prev_day_close")
    if prev_close and not pd.isna(prev_close) and prev_close > 0:
        if row["close"] < prev_close * 0.995:
            return False

    return True


# ── Ana portföy backtest ───────────────────────────────────────────────────────
def run_portfolio_backtest(
    symbols: list,
    days: int = 55,
    initial_capital: float = 10_000.0,
) -> None:
    from bist.adapters.price_converter import PriceConverter
    from bist.data.usdtry_provider import USDTRYProvider
    from bist.filters.macro_filter import MacroFilter, fetch_xu100

    end_dt      = dt.date.today()
    start_dt    = end_dt - dt.timedelta(days=days)
    fetch_start = (start_dt - dt.timedelta(days=10)).strftime("%Y-%m-%d")
    start_str   = start_dt.strftime("%Y-%m-%d")
    end_str     = end_dt.strftime("%Y-%m-%d")
    trade_start = pd.Timestamp(start_str, tz="UTC")

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  BIST 5 DAKİKALIK PORTFÖy BACKTEST")
    print(f"  Dönem     : {start_str} → {end_str}  ({days} gün)")
    print(f"  Sermaye   : ${initial_capital:,.0f}")
    print(f"  Max Poz   : {MAX_POSITIONS} | Risk/işlem: %{RISK_PER_TRADE*100:.1f}")
    print(f"  Sembol havuzu: {len(symbols)}")
    print(f"{sep}\n")

    # ── USD/TRY ve XU100 ─────────────────────────────────────────────────────
    print("  ↓ Makro veriler çekiliyor...")
    usdtry = USDTRYProvider(cache_enabled=False).fetch(start=fetch_start, end=end_str)
    xu100  = fetch_xu100(start=fetch_start, end=end_str)
    macro  = MacroFilter(
        regime_filter_enabled=True, regime_ema_period=100, regime_usd_adjusted=True,
        usdtry_guard_enabled=True, usdtry_momentum_period=20, usdtry_crisis_threshold=0.15,
    )
    macro.fit(xu100, usdtry)
    stats = macro.summary_stats()
    print(f"    Makro: Bull={stats.get('regime_bull_pct',0):.0f}%  "
          f"TRY-kriz={stats.get('usdtry_crisis_pct',0):.0f}%\n")

    converter = PriceConverter(mode="convert_series")

    # ── 5m veri çekme ────────────────────────────────────────────────────────
    print(f"  ↓ {len(symbols)} sembol için 5m veri çekiliyor...")
    all_data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        print(f"    {sym}...", end=" ", flush=True)
        try:
            df_try = fetch_5m(sym, fetch_start, end_str)
            df_usd = converter.convert_ohlcv(df_try, usdtry)
            if len(df_usd) < BARS_PER_DAY * 3:
                print(f"az veri ({len(df_usd)}), atlandı")
                continue
            print(f"✅ {len(df_usd)} bar")
            all_data[sym] = df_usd
        except Exception:
            print("❌")

    # ── 5m kalite filtresi (SADECE test öncesi veri) ─────────────────────────
    # Look-ahead'i önlemek için: yalnızca trade_start'tan ÖNCE gelen barlar
    # kullanılarak skor hesaplanır. Bu şekilde "gelecekteki" performans skoru
    # etkileyemez. Pre-trade verisi az olsa da CI/ADX/EMA için yeterli.
    print(f"\n  ↓ 5m kalite taraması — test öncesi veri ({len(all_data)} sembol)...")
    scored: list[tuple[str, float, float]] = []
    for sym, df_full in all_data.items():
        pre_trade = df_full[df_full.index < trade_start]
        # Yeterli veri yoksa tam veriyi kullan (yfinance 60-gün sınırı nedeniyle)
        score_df = pre_trade if len(pre_trade) >= 100 else df_full
        score, ci = _5m_quality_score(score_df)
        print(f"    {sym}: skor={score:.3f}  CI={ci:.1f}", end="\r", flush=True)
        if score > MIN_CI_SCORE:
            scored.append((sym, score, ci))
    print(" " * 60)

    scored.sort(key=lambda x: -x[1])
    candidates = scored[:TOP_N_CANDIDATES]
    candidate_syms = [s for s, _, _ in candidates]
    print(f"  Kalite filtresi: {len(scored)}/{len(all_data)} geçti"
          f" → En iyi {len(candidates)} aday seçildi")
    print(f"  {'Sembol':<10} {'Skor':>7} {'CI':>6}")
    for sym, sc, ci in candidates:
        print(f"  {sym:<10} {sc:>7.3f} {ci:>6.1f}")
    print()

    # ── İndikatör ön-hesaplama ────────────────────────────────────────────────
    print("  ↓ İndikatörler hesaplanıyor...")
    sym_ind: dict[str, pd.DataFrame] = {}
    for sym in candidate_syms:
        sym_ind[sym] = compute_indicators(all_data[sym])
    print(f"    {len(sym_ind)} sembol hazır\n")

    # ── Makro rejim serisi (günlük → 5m indeksine hizala) ────────────────────
    regime_s = macro.get_regime_series()
    crisis_s = macro.get_crisis_series()

    def macro_allowed(ts: pd.Timestamp) -> bool:
        """Verilen timestamp'te yeni giriş yapılabilir mi?"""
        if regime_s is not None:
            v = regime_s.asof(ts)
            if not pd.isna(v) and not bool(v):
                return False
        if crisis_s is not None:
            v = crisis_s.asof(ts)
            if not pd.isna(v) and bool(v):
                return False
        return True

    # ── Portföy durumu ────────────────────────────────────────────────────────
    portfolio_balance = initial_capital
    daily_pnl        = 0.0
    current_day      = None
    open_positions:  dict[str, PortfolioPosition] = {}
    closed_trades:   list[PortfolioPosition]      = []
    equity_curve:    list[tuple[pd.Timestamp, float]] = []

    # ── Per-symbol koruma kuralları ───────────────────────────────────────────
    # 1. Günde max 1 giriş: aynı sembol aynı günde tekrar açılamaz
    last_entry_day:  dict[str, dt.date]      = {}
    # 2. Stop sonrası cooldown: stop yedikten sonra 1 tam gün (102 bar) beklenir
    stop_cooldown_until: dict[str, pd.Timestamp] = {}
    # 3. Art arda kayıp: 2 üst üste stop → 5 gün (510 bar) duraklama
    consecutive_stops: dict[str, int]        = {}
    pause_until:       dict[str, pd.Timestamp] = {}

    # ── Tüm zaman dilimlerini kronolojik olarak al ───────────────────────────
    all_timestamps = sorted(set().union(*[
        set(df.loc[trade_start:].index) for df in sym_ind.values()
    ]))
    print(f"  ↓ Portföy simülasyonu başlıyor...")
    print(f"    {len(all_timestamps):,} bar × {len(candidate_syms)} sembol\n")

    prev_rows: dict[str, pd.Series] = {}

    for ts in all_timestamps:
        hour = ts.hour

        # Seans dışı barları atla
        if hour < MARKET_OPEN_UTC or hour >= MARKET_CLOSE_UTC:
            continue

        # ── Yeni gün kontrolü ─────────────────────────────────────────────────
        day = ts.date()
        if current_day != day:
            if current_day is not None and daily_pnl < -(initial_capital * DAILY_MAX_LOSS_PCT):
                pass  # Sıfırlama aşağıda
            daily_pnl  = 0.0
            current_day = day

        # ── Gün sonu zorla kapatma (14:30 UTC+) ──────────────────────────────
        eod_close = (hour == EOD_FORCE_CLOSE_UTC and ts.minute >= 30) or hour > EOD_FORCE_CLOSE_UTC
        if eod_close and open_positions:
            to_close = list(open_positions.keys())
            for sym in to_close:
                pos = open_positions[sym]
                df_s = sym_ind[sym]
                if ts not in df_s.index:
                    continue
                exit_price = float(df_s.loc[ts, "close"]) * (1 - SLIPPAGE)
                gross_pnl  = (exit_price - pos.entry_price) * pos.size
                commission = exit_price * pos.size * COMMISSION
                net_pnl    = gross_pnl - commission
                pos.exit_price  = exit_price
                pos.exit_time   = ts
                pos.exit_reason = "EOD"
                pos.pnl         = net_pnl
                portfolio_balance += pos.cost + net_pnl
                daily_pnl        += net_pnl
                closed_trades.append(pos)
                del open_positions[sym]
            equity_curve.append((ts, portfolio_balance))
            continue

        # ── Açık pozisyonları güncelle ────────────────────────────────────────
        to_close_stops = []
        for sym, pos in open_positions.items():
            df_s = sym_ind[sym]
            if ts not in df_s.index:
                continue
            row   = df_s.loc[ts]
            price = float(row["close"])
            atr   = float(row["atr"]) if not pd.isna(row.get("atr")) else pos.entry_atr

            # Trailing stop güncelle
            new_trail = price - TRAILING_ATR_MULT * atr
            if new_trail > pos.trail_price:
                pos.trail_price = new_trail

            # Stop tetiklendi mi?
            low_price = float(row.get("low", price))
            hit_price = None
            if low_price <= pos.stop_price:
                hit_price  = pos.stop_price * (1 - SLIPPAGE)
                exit_reason = "ATR stop"
            elif low_price <= pos.trail_price:
                hit_price   = pos.trail_price * (1 - SLIPPAGE)
                exit_reason = "Trailing stop"
            # EMA bear cross çıkış
            elif row.get("ema50", 1) < row.get("ema200", 0):
                hit_price   = price * (1 - SLIPPAGE)
                exit_reason = "EMA bear cross"

            if hit_price:
                gross_pnl  = (hit_price - pos.entry_price) * pos.size
                commission = hit_price * pos.size * COMMISSION
                net_pnl    = gross_pnl - commission
                pos.exit_price  = hit_price
                pos.exit_time   = ts
                pos.exit_reason = exit_reason
                pos.pnl         = net_pnl
                portfolio_balance += pos.cost + net_pnl
                daily_pnl        += net_pnl
                to_close_stops.append(sym)
                closed_trades.append(pos)

                # Stop kuralları
                is_stop = "stop" in exit_reason.lower()
                if is_stop:
                    # 1 gün cooldown (stop yedikten sonra yarın girilir)
                    stop_cooldown_until[sym] = ts + pd.Timedelta(hours=24)
                    # Art arda sayacı artır
                    consecutive_stops[sym] = consecutive_stops.get(sym, 0) + 1
                    if consecutive_stops[sym] >= 2:
                        # 5 gün duraklama
                        pause_until[sym] = ts + pd.Timedelta(days=5)
                        consecutive_stops[sym] = 0
                else:
                    # Kârlı çıkış → sayacı sıfırla
                    consecutive_stops[sym] = 0

        for sym in to_close_stops:
            del open_positions[sym]

        # ── Yeni giriş sinyali ────────────────────────────────────────────────
        # Sabah seansı: 06:30-10:00 UTC
        can_enter = (hour < MORNING_ENTRY_END and macro_allowed(ts)
                     and daily_pnl > -(portfolio_balance * DAILY_MAX_LOSS_PCT)
                     and len(open_positions) < MAX_POSITIONS)

        if can_enter:
            for sym in candidate_syms:
                if sym in open_positions:
                    continue
                if len(open_positions) >= MAX_POSITIONS:
                    break

                df_s = sym_ind[sym]
                if ts not in df_s.index:
                    continue

                row  = df_s.loc[ts]
                prev = prev_rows.get(sym, row)

                if not _check_entry_signal(row, prev):
                    continue

                # Kural 1: Günde max 1 giriş
                if last_entry_day.get(sym) == ts.date():
                    continue

                # Kural 2: Stop sonrası cooldown (1 gün)
                if stop_cooldown_until.get(sym) and ts < stop_cooldown_until[sym]:
                    continue

                # Kural 3: Art arda 2 stop → 5 gün duraklama
                if pause_until.get(sym) and ts < pause_until[sym]:
                    continue

                price = float(row["close"]) * (1 + SLIPPAGE)
                atr   = float(row["atr"]) if not pd.isna(row.get("atr")) else 0.0
                if atr <= 0:
                    continue

                # Pozisyon boyutu: portföyün risk_per_trade'i kadar zarar göze alınır
                risk_amount = portfolio_balance * RISK_PER_TRADE
                size        = risk_amount / (ATR_STOP_MULT * atr)
                cost        = price * size * (1 + COMMISSION)

                # Portföy limitleri
                if cost > portfolio_balance * MAX_POSITION_PCT:
                    size = (portfolio_balance * MAX_POSITION_PCT) / (price * (1 + COMMISSION))
                    cost = price * size * (1 + COMMISSION)

                if cost > portfolio_balance * 0.95 or size <= 0:
                    continue

                stop  = price - ATR_STOP_MULT * atr
                trail = price - TRAILING_ATR_MULT * atr

                pos = PortfolioPosition(
                    symbol=sym, entry_price=price, stop_price=stop,
                    trail_price=trail, size=size, cost=cost,
                    entry_time=ts, entry_atr=atr,
                )
                open_positions[sym]       = pos
                portfolio_balance        -= cost
                last_entry_day[sym]       = ts.date()
                consecutive_stops[sym]    = consecutive_stops.get(sym, 0)

        # Önceki satır güncelle
        for sym in candidate_syms:
            df_s = sym_ind[sym]
            if ts in df_s.index:
                prev_rows[sym] = df_s.loc[ts]

        # Equity: sadece TÜM açık pozisyonların fiyatı mevcut olan barları ekle.
        # Eksik timestamp'lerde pozisyon değeri sıfır görünür → sahte drawdown.
        unrealized = 0.0
        all_priced = True
        for s, p in open_positions.items():
            if ts in sym_ind[s].index:
                px = float(sym_ind[s].loc[ts, "close"])
                if px > 0:
                    unrealized += px * p.size
                else:
                    all_priced = False
            else:
                all_priced = False
        if all_priced or not open_positions:
            total_equity = portfolio_balance + unrealized
            if total_equity > 0:
                equity_curve.append((ts, total_equity))

    # ── Gün sonu açık pozisyonları son fiyatla kapat ─────────────────────────
    for sym, pos in list(open_positions.items()):
        df_s       = sym_ind[sym]
        last_price = float(df_s["close"].iloc[-1]) * (1 - SLIPPAGE)
        gross_pnl  = (last_price - pos.entry_price) * pos.size
        commission = last_price * pos.size * COMMISSION
        net_pnl    = gross_pnl - commission
        pos.exit_price  = last_price
        pos.exit_time   = df_s.index[-1]
        pos.exit_reason = "Backtest sonu"
        pos.pnl         = net_pnl
        portfolio_balance += pos.cost + net_pnl
        closed_trades.append(pos)

    # ── Sonuç raporu ──────────────────────────────────────────────────────────
    _print_report(closed_trades, initial_capital, portfolio_balance, equity_curve)


def _print_report(
    trades: list[PortfolioPosition],
    initial_capital: float,
    final_balance: float,
    equity_curve: list,
) -> None:
    total_return = (final_balance / initial_capital - 1) * 100
    total_pnl    = final_balance - initial_capital

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  PORTFÖy BACKTEST SONUÇLARI")
    print(f"{sep}")
    print(f"  Başlangıç  : ${initial_capital:>10,.2f}")
    print(f"  Final      : ${final_balance:>10,.2f}")
    print(f"  Kar/Zarar  : ${total_pnl:>+10,.2f}  ({total_return:+.2f}%)")
    print(f"  İşlem sayısı: {len(trades)}")

    if not trades:
        print("  Hiç işlem yapılmadı.")
        return

    # Win rate
    wins     = [t for t in trades if t.pnl > 0]
    losses   = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / len(trades) * 100
    avg_win  = np.mean([t.pnl for t in wins])  if wins   else 0.0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0.0
    gross_p  = sum(t.pnl for t in wins)
    gross_l  = abs(sum(t.pnl for t in losses))
    pf       = gross_p / gross_l if gross_l > 0 else float("inf")

    print(f"  Win Rate   : {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Profit Factor: {pf:.3f}")
    print(f"  Ort. Kazanç: ${avg_win:>+.2f}  |  Ort. Kayıp: ${avg_loss:>+.2f}")

    # Max drawdown (equity curve)
    if equity_curve:
        eq_vals = [e for _, e in equity_curve]
        peak    = eq_vals[0]
        max_dd  = 0.0
        for v in eq_vals:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100
            if dd > max_dd:
                max_dd = dd
        print(f"  Max Drawdown: {max_dd:.2f}%")

    # ── Hisse bazlı özet ──────────────────────────────────────────────────────
    sym_summary: dict[str, dict] = {}
    for t in trades:
        s = t.symbol
        if s not in sym_summary:
            sym_summary[s] = {"pnl": 0.0, "n": 0, "wins": 0}
        sym_summary[s]["pnl"]  += t.pnl
        sym_summary[s]["n"]    += 1
        sym_summary[s]["wins"] += 1 if t.pnl > 0 else 0

    print(f"\n  {'SEMBOL':<10} {'İŞLEM':>6} {'WR%':>7} {'PnL (USD)':>12} {'Ort PnL':>10}")
    print(f"  {'─'*52}")
    for sym, d in sorted(sym_summary.items(), key=lambda x: -x[1]["pnl"]):
        wr      = d["wins"] / d["n"] * 100
        avg_pnl = d["pnl"] / d["n"]
        sign    = "+" if d["pnl"] >= 0 else ""
        print(f"  {sym:<10} {d['n']:>6} {wr:>6.0f}% {sign}{d['pnl']:>10.2f} {sign}{avg_pnl:>8.2f}")

    # ── Tüm işlemler detay ────────────────────────────────────────────────────
    print(f"\n  {'#':>4} {'SEMBOL':<8} {'GİRİŞ':>12} {'ÇIKIŞ':>12} {'BOYUT':>10} {'PnL':>10} {'NEDEN'}")
    print(f"  {'─'*72}")
    for i, t in enumerate(sorted(trades, key=lambda x: x.entry_time), 1):
        entry_dt = t.entry_time.strftime("%m-%d %H:%M") if t.entry_time else "-"
        exit_dt  = t.exit_time.strftime("%m-%d %H:%M")  if t.exit_time  else "-"
        sign     = "+" if t.pnl >= 0 else ""
        print(f"  {i:>4} {t.symbol:<8} {entry_dt:>12} {exit_dt:>12} "
              f"{t.size:>10.3f} {sign}{t.pnl:>8.2f}  {t.exit_reason}")
    print(f"  {'─'*72}")
    print(f"  Toplam PnL: ${total_pnl:+.2f}  |  Final Bakiye: ${final_balance:,.2f}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIST 5m Portföy Backtest")
    parser.add_argument("--symbols", type=str, default="",
                        help="Virgülle ayrılmış semboller.")
    parser.add_argument("--all", action="store_true",
                        help="config_bist.yaml'daki tüm 100 sembolü kullan.")
    parser.add_argument("--days", type=int, default=55,
                        help="Kaç günlük veri (max 59, yfinance sınırı)")
    parser.add_argument("--capital", type=float, default=10_000.0,
                        help="Başlangıç sermayesi (USD)")
    args = parser.parse_args()

    if args.all:
        import yaml
        _cfg_path = os.path.join(_ROOT, "bist", "config_bist.yaml")
        with open(_cfg_path) as _f:
            _cfg = yaml.safe_load(_f)
        symbols = _cfg["trading"]["symbols"]
        print(f"  Config'den {len(symbols)} sembol yüklendi.")
    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = DEFAULT_SYMBOLS

    days = min(args.days, 59)
    run_portfolio_backtest(symbols, days=days, initial_capital=args.capital)
