#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M5 + M9 REJİM TAHSİSÇİSİ ("ortaklık" sistemi)
==============================================
Sermayeyi günlük olarak M5 (her-mevsim taban) ile M9 (itki uzmanı) arasında kaydırır.

Sinyal (lookahead'siz): her gün 00:00 UTC'de, son KAPANMIŞ 1h barıyla:
  m9_aktif = BTC 1h rejimi TREND (UP/DOWN) VE thrust (|BTC24h|>=%2 veya disp>=%3)
Karar bir sonraki günün ağırlığını belirler (shift ile uygulanır).

Varyantlar:
  A_binary : aktif → %100 M9, değil → %100 M5
  C_statik : her gün 50/50
  D_uydu   : aktif → %50 M9 + %50 M5, değil → %100 M5 (muhafazakâr)

Varsayım: sermaye gün sınırında maliyetsiz kayar (tahsisçi-çalışması standardı);
gerçekte pozisyon devri birkaç saat sürer — sonuçlar yaklaşıktır.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/vedatayaz/Desktop/Projeler/Kripto_Robotu/M9")
import m9_backtest as m9

_ALC_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else \
    "/Users/vedatayaz/Desktop/Projeler/Kripto_Robotu/M9"
M5_DIR = os.path.join(_ALC_HERE, "sonuclar", "_alloc")   # kalıcı (/tmp temizliğine dayanıklı)
M9_DIR = M5_DIR

WINDOWS = [
    ("son13g", "2026-05-27", "2026-06-09"),
    ("ayi",    "2026-01-29", "2026-02-05"),
    ("boga",   "2025-09-27", "2025-10-04"),
    ("6ay",    "2025-08-01", "2026-01-31"),
    ("oos",    "2026-02-05", "2026-05-26"),
    ("full",   "2025-08-01", "2026-06-09"),
]


def build_signal(start="2025-06-20", end="2026-06-10", anchor_min: float = 0.0) -> pd.Series:
    """Günlük m9_aktif sinyali (bool, gün 00:00 UTC kararı — ertesi gün uygulanır).
    anchor_min > 0: BTC 30g EMA'sından en az bu kadar uzaklaşmış olmalı
    (range içinde sıkışıkken M9 kapalı — M5 yedekte olduğu için maliyetsiz)."""
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    closes = {}
    for sym in m9.UNIVERSE:
        df = m9.fetch_klines(sym, "1h", lo, hi)
        if not df.empty:
            closes[sym] = df["close"]
    px = pd.DataFrame(closes).sort_index()
    btc = px["BTCUSDT"]

    # BTC 1h rejimi (m9.prepare_symbol'daki 1h mantığının birebir kopyası)
    h = pd.DataFrame({"close": btc})
    # ADX/chop için high/low gerek — BTC 1h OHLC'yi tam çek
    btc_df = m9.fetch_klines("BTCUSDT", "1h", lo, hi)
    adx = m9._adx(btc_df)
    er = m9._efficiency_ratio(btc_df["close"], 48)
    ema50 = m9._ema(btc_df["close"], 50)
    ema200 = m9._ema(btc_df["close"], 200)
    trendy = ((er >= 0.30) & (adx >= 20)) | (adx >= 27)
    up = (ema50 > ema200) & (btc_df["close"] > ema50)
    dn = (ema50 < ema200) & (btc_df["close"] < ema50)
    is_trend = trendy & (up | dn)

    # thrust: BTC 24h getiri + kesitsel dispersiyon
    ret24 = px.pct_change(24)
    btc_ret24 = ret24["BTCUSDT"]
    disp24 = ret24.std(axis=1)
    thrust = (btc_ret24.abs() >= m9.THRUST_BTC_24H) | (disp24 >= m9.THRUST_DISP_24H)

    active_1h = (is_trend.reindex(px.index).fillna(False) & thrust.fillna(False))
    if anchor_min > 0:
        ema720 = m9._ema(btc_df["close"], 720)
        dist = (btc_df["close"] / ema720 - 1).abs()
        far = (dist >= anchor_min).reindex(px.index).fillna(False)
        active_1h = active_1h & far
    # Gün kararı: günün SON kapanmış barı (00:00 kararı için önceki günün son barı)
    daily = active_1h.resample("1D").last().fillna(False)
    # shift(1): karar ertesi gün uygulanır (lookahead yok)
    return daily.shift(1).fillna(False)


def daily_rets(path: str, capital: float = 10_000.0) -> pd.Series:
    eq = pd.read_csv(path, parse_dates=["ts"])
    s = eq.set_index("ts")["equity"].sort_index()
    d = s.resample("1D").last().dropna()
    # gün-0 çapası: ilk günün getirisi kaybolmasın (pct_change ilk değeri NaN yapar —
    # OOS'ta ilk gün +%14.7'ydi, çapasız seri bunu yutuyordu)
    anchor = pd.Series([capital], index=[d.index[0] - pd.Timedelta(days=1)])
    d = pd.concat([anchor, d])
    return d.pct_change().dropna()


def metrics(rets: pd.Series) -> dict:
    eq = (1 + rets).cumprod()
    total = eq.iloc[-1] - 1
    peak, mdd = 1.0, 0.0
    for v in eq:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak)
    mu, sd = rets.mean(), rets.std()
    sharpe = mu / sd * np.sqrt(365) if sd > 0 else float("nan")
    return dict(total=100 * total, mdd=100 * mdd, sharpe=sharpe, days=len(rets))


def main():
    print("Rejim sinyali hesaplanıyor (30 coin 1h)...")
    sig = build_signal()
    print(f"  sinyal: {len(sig)} gün, M9-aktif oranı: %{100*sig.mean():.1f}")

    import os
    rows = []
    for name, s, e in WINDOWS:
        m5p = f"{M5_DIR}/m5_eq_{name}.csv"
        m9p = f"{M9_DIR}/m9_eq_{name}.csv"
        if not (os.path.exists(m5p) and os.path.exists(m9p)):
            print(f"  {name}: eksik eğri, atlandı ({m5p if not os.path.exists(m5p) else m9p})")
            continue
        r5 = daily_rets(m5p)
        r9 = daily_rets(m9p)
        idx = r5.index.union(r9.index)
        r5 = r5.reindex(idx).fillna(0.0)
        r9 = r9.reindex(idx).fillna(0.0)
        w = sig.reindex(idx).fillna(False).astype(float)

        # E: rejim sinyali + M9'un kendi son-14g performans kapısı (fund-of-funds
        # tahsisi: kanayan yöneticiye sermaye verilmez; M9 shadow'da koştuğu için
        # bu bilgi canlıda gözlemlenebilir — shift(1) ile lookahead'siz)
        eq9 = (1 + r9).cumprod()
        roll14 = (eq9 / eq9.shift(14) - 1).fillna(0.0).shift(1).fillna(0.0)
        w_sig_bool = sig.reindex(idx).fillna(False)
        wE = (w_sig_bool & (roll14 > -0.02)).astype(float)

        # v23 TEYİT KURALI (T2, FİNAL): izole tek-gün sinyalleri blokla — bugün açık
        # VE son 2 günün en az birinde açıktı. Ajan analizi: tüm uzun-dönem zararı
        # tek-günlük episodlardan (35 episod, -25.2p, WR %11); çok-günlü episodlar net
        # pozitif; split-sample tutarlı. Episod-içi tek-gün titreme tolere edilir.
        w_confirmed = w_sig_bool & (w_sig_bool.shift(1).fillna(False)
                                    | w_sig_bool.shift(2).fillna(False))
        wT = (w_confirmed & (roll14 > -0.02)).astype(float)

        combos = {
            "M5": r5,
            "M9": r9,
            "A_binary": w * r9 + (1 - w) * r5,
            "C_50/50": 0.5 * r9 + 0.5 * r5,
            "D_uydu": (0.5 * w) * r9 + (1 - 0.5 * w) * r5,
            "E_FINAL": wE * r9 + (1 - wE) * r5,
            "T_TEYIT": wT * r9 + (1 - wT) * r5,
        }
        act = 100 * w.mean()
        print(f"\n### {name}  ({s} → {e})  M9-aktif gün: %{act:.0f}")
        for cname, r in combos.items():
            m = metrics(r)
            print(f"  {cname:9s}: {m['total']:+8.2f}%   DD %{m['mdd']:5.1f}   Sharpe {m['sharpe']:6.2f}")
        rows.append(name)

    print("\nNot: tahsis gün sınırında maliyetsiz varsayıldı; M9-aktif sinyali yalnızca")
    print("kapanmış 1h barları kullanır (lookahead yok), karar ertesi gün uygulanır.")


if __name__ == "__main__":
    main()
