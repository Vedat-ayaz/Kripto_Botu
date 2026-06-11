#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M9 "APEX" — Rejim-anahtarlı çok stratejili portföy backtest motoru
===================================================================
5m işlem + 1h rejim bağlamı | LONG + SHORT | Order-flow (taker buy hacmi)
Kesitsel momentum rotasyonu | Piramitleme | Vol-target boyutlandırma

Bağımsız model — M1/M5/M7/M8 koduna dokunmaz.

Kullanım:
    python3 m9_backtest.py --start 2026-05-27 --end 2026-06-09
    python3 m9_backtest.py --start 2025-08-01 --end 2026-01-31 --capital 10000
"""

import argparse
import os
import sys
import time
import math
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

# ── Evren ─────────────────────────────────────────────────────────────────────
# Ana repo UNIVERSE'i baz alındı; LEO çıkarıldı (Binance'te yok), BTC eklendi.
UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "TRXUSDT",
    "DOTUSDT", "LINKUSDT", "LTCUSDT", "ATOMUSDT",
    "NEARUSDT", "UNIUSDT", "APTUSDT", "INJUSDT",
    "FETUSDT", "ARBUSDT", "OPUSDT",
    "ETCUSDT", "HBARUSDT", "ALGOUSDT", "VETUSDT", "FILUSDT",
    "SUIUSDT", "TIAUSDT", "TONUSDT", "JUPUSDT", "WIFUSDT",
]

# ── Maliyetler (ana repo ile aynı — adil kıyas) ───────────────────────────────
COMMISSION = 0.001     # taker %0.1 / taraf
SLIPPAGE   = 0.0005    # %0.05 / taraf

# ── Uygulama dilimi ───────────────────────────────────────────────────────────
# FİNAL (v19): 5m — matris deneyi (A-H) 5m + thrust + frenlerin (config D) en dengeli
# sonucu verdiğini gösterdi. 15m denendi (v10), haftalık impuls kârlarını bozdu.
EXEC_TF = "5m"
K = 12                 # saat başına bar (5m: 12, 15m: 4) — set_tf() ile değişir

def set_tf(tf: str):
    """Uygulama dilimini değiştirir; bar-bazlı tüm pencereleri yeniden hesaplar."""
    global EXEC_TF, K, DONCHIAN_BARS, STAGNATION_BARS, MR_TIME_BARS
    global COOLDOWN_EXIT, COOLDOWN_LOSS
    EXEC_TF = tf
    K = {"5m": 12, "15m": 4}[tf]
    DONCHIAN_BARS   = 8 * K    # 8 saatlik kanal
    STAGNATION_BARS = 8 * K    # 8 saat
    MR_TIME_BARS    = 3 * K    # 3 saat
    COOLDOWN_EXIT   = 1 * K    # 1 saat (v8 değeri — matris D)
    COOLDOWN_LOSS   = 36 * K   # 36 saat (v12 — legacy M5 sl_cooldown_hours=48'e yakın)

# ── Risk parametreleri ────────────────────────────────────────────────────────
INITIAL_CAPITAL   = 10_000.0
RISK_TREND        = 0.0175   # trend/squeeze işlemi risk payı
RISK_MR           = 0.010    # mean-reversion risk payı
MAX_POSITIONS     = 6
MAX_SAME_DIR      = 5
MAX_POSITION_PCT  = 0.25     # tek pozisyon notional tavanı (equity payı)
MIN_ORDER_USD     = 10.0
DAILY_KILL_PCT    = -0.03    # gün içi gerçekleşen PnL bu eşiğin altına inerse → gün bitene dek yeni giriş yok
DD_GOVERNOR       = 0.12     # tepe-equity'den bu kadar düşüşte risk yarıya iner
VOL_TARGET_DAILY  = 0.04     # coin günlük vol hedefi (boyut ölçekleme), clip [0.5, 1.5]

# ── Stop / çıkış parametreleri ───────────────────────────────────────────────
# v2: stop mesafeleri 1h ATR bazlı — 5m ATR komisyondan bile küçük kalıyordu (v1: PF 0.33)
ATR_STOP_TREND    = 1.8      # ilk stop (× ATR_1h)
CHANDELIER_MULT   = 2.5      # iz süren stop (× ATR_1h)
CHANDELIER_TIGHT  = 2.2      # piramit sonrası sıkılaşır
ATR_STOP_MR       = 1.0      # × ATR_1h
MIN_STOP_PCT      = 0.012    # v8: 0.008 → 0.012 (fee/R oranını düşür: %25 notional tavanında
                             # %0.8 stop → risk $25, maliyet $7.5 = R'nin %30'u — yenilmez)
MR_TIME_BARS      = 3 * K    # 3 saat
PARTIAL_AT_R      = 1.5      # +1.5R'de kısmi kâr al
PARTIAL_PCT       = 0.25
TRAIL_START_R     = 0.8      # v6: chandelier ancak +0.8R sonrası devreye girer
                             # (boğa grind'inde erken trail 35 işlemi kârsız silkeliyordu)
STAGNATION_BARS   = 8 * K    # 8 saat: trend işlemi hâlâ negatifse çık
STAGNATION_MIN_R  = 0.0      # v6: 0.3 → 0.0 (yavaş boğa grind'inde +0.2R işlemler kesiliyordu)
# v12: legacy M5 dersleri — kısa cooldown chop'ta intihar (M5: sl_cooldown_hours=48)
COOLDOWN_EXIT     = 1 * K    # çıkıştan sonra aynı coine 1 saat girilmez
COOLDOWN_LOSS     = 36 * K   # zararla çıkıştan sonra 36 saat
SYM_PAUSE_TRADES  = 5        # sembolün son N işlemi izlenir
SYM_PAUSE_LOSSES  = 4        # N işlemin ≥bu kadarı zararsa ve toplam negatifse →
SYM_PAUSE_BARS    = 96 * K   # 4 gün sembol duraklatılır (legacy rolling-WR pause karşılığı)
WEEK_BRAKE_PCT    = 0.06     # 7 günlük equity düşüşü bu eşiği aşarsa →
WEEK_BRAKE_BARS   = 24 * K   # 24 saat yeni giriş yok (portföy freni)

# ── Piramitleme ───────────────────────────────────────────────────────────────
# v9 starter sizing DENENDİ, GERİ ALINDI (matris E): ayı kaskadında ilk hamle
# kazancını yarılıyor (+9.6% → −6.2%). Tam boy giriş + v6 piramidi kanıtlı yapı.
INITIAL_SIZE_FRAC = 1.0
PYR_LEVELS = [
    dict(at_r=1.0, size_mult=0.6, stop_to_r=0.1),   # +1.0R → %60 ekle, stop BE+0.1R
    dict(at_r=2.2, size_mult=0.4, stop_to_r=1.0),   # +2.2R → %40 ekle, stop +1R
]

# ── Strateji anahtarları ─────────────────────────────────────────────────────
# v11: CS (kompozit skor) ana motor — M5'in kanıtlanmış çok-faktörlü giriş yaklaşımı
# (6ay WR %45) + M9'un order-flow faktörleri. BO/PB ham halleriyle 6ayda kanadı
# (BO LONG WR %15.6, PB-only -24%) → kapalı.
# v13: CS kapatıldı (eşik-kesme tetikleri chop'ta patladı: 1061 işlem, WR %18.5).
# BO+PB (haftalık pencerelerin kanıtlı motoru) + THRUST kapısı ana yapı.
ENABLE_CS = False
ENABLE_BO = True
ENABLE_PB = True

# v13: PİYASA İTKİSİ (THRUST) KAPISI — M9 bir saldırı sistemi: piyasa gerçekten
# hareket etmiyorsa hiç işlem açılmaz. 6ay kanıtı: chop aylarında her konfigürasyon
# komisyona yenildi; impuls haftalarında hepsi kazandı.
THRUST_ON         = True
THRUST_DIRECTIONAL = False   # v14 DENENDİ: ayıyı bozdu (24h gecikme dönüm noktasında ters tepiyor)

# ── v20: TEŞHİS-TABANLI DÜZELTMELER (ajan analizi, 6AY/OOS/SON13G CSV kanıtı) ──
# K2: dispersiyon tabanı — coinler arası 24h getiri std < eşik ise piyasa "dağınık değil
#     ölü"; SON13G kârının %98'i yüksek-disp girişlerden, düşük-disp girişler her kaybeden
#     pencerede kanama (6AY +$1333 kurtarma / SON13G maliyeti -$181)
DISP_MIN          = 0.0      # REDDEDİLDİ (ablasyon: kaskad başını blokluyor, AYI kırıldı) — 0 = kapalı
# K2b: disp boyut ölçekleyici (sert kapı yerine) — düşük dispersiyonda küçük boy.
# Ablasyon dersi: sert taban kaskad BAŞLANGICINI blokluyor (AYI +6.5→-6.2 kırıldı);
# ölçekleyici girişe izin verir, piramit kazananı tam boya çıkarır.
DISP_SCALE        = False    # risk × clamp(disp/0.03, alt, 1.0)
DISP_SCALE_FLOOR  = 0.4
# K3g: BTC-itkisi dispersiyon niteleyicisi — BTC oynuyor ama coinler ayrışmıyorsa
# (dağınık piyasa) BTC-itkisi geçersiz. 6AY kaybının %117'si bu gruptan (-$1955).
THRUST_DISP_QUAL  = 0.0      # 0 = kapalı; örn. 0.022
# K-M: makro-çapalı itki — BTC-itkisi ancak fiyat 30g EMA'sından AYNI YÖNDE kopmuşsa
# geçerli (range içindeki %2 = tükenmiş hareket → reversiyon; çapadan kopuş = devam)
THRUST_MACRO_ANCHOR = False
# K-M şiddet istisnası: |BTC 24h| bu eşiğin üstündeyse çapa onayı beklenmez —
# gerçek crash çapanın üstünden de başlayabilir (AYI dersi: çapa onayı geç kalır)
THRUST_VIOLENT      = 0.045  # 0 = istisna yok
# K-turn: V-dönüş kapısı — BTC 24h ile 4h momentum işaretleri çelişiyorsa dönüm noktası
#     (OOS kaybının %61'i tek crash-dibi gününden; akademik kanıt: changepoint Sharpe +%33)
TURN_GATE         = False    # REDDEDİLDİ (ablasyon: kaskad içi re-entry'leri kesiyor, AYI -7.8)
TURN_FAST_H       = 4        # hızlı momentum penceresi (saat)
TURN_MIN_24H      = 0.01     # |BTC 24h| bu eşiğin üstündeyken uyum aranır
TURN_MIN_4H       = 0.003    # hızlı bacak bu eşiğin üstünde ters işaretliyse blok
# K4: break-even stop — +0.5R görmüş işlem zarara dönmesin (6AY kayıplarının %48'i
#     0.3-1R görüp ölüyor; üst sınır etkisi her pencerede pozitif)
BE_STOP_AT_R      = 0.0      # REDDEDİLDİ (ablasyon: nadir büyük kazananları öldürüyor, SON13G -5.5) — 0 = kapalı
# K5: ideal-koşul kilidi açma — yön BTC trendiyle uyumlu + |BTC24h|≥%2 + disp≥%3 iken
#     xsec rank kapısı atlanır, zarar cooldown'u yarıya iner (SON13G'de bu bloklar
#     +%2.7/+%3.3 ileri getirili fırsat kaçırmıştı; 6AY'da aynı bloklar negatif fwd → risksiz)
IDEAL_UNLOCK      = True
IDEAL_BTC24       = 0.02
IDEAL_DISP        = 0.03

# ── v22: YENİ BİLGİ KAYNAĞI + MALİYET MODELİ ─────────────────────────────────
# FUNDING KAPISI — Binance perp funding oranı = pozisyonlanma göstergesi.
# Aşırı negatif funding = kalabalık short (squeeze-up riski) → SHORT girme;
# aşırı pozitif = kalabalık long → LONG girme. "Kalabalığa uçta katılma."
# (OI geçmişi Binance'te 30 günle sınırlı → backtest'te kullanılamaz, canlı-katman işi)
FUNDING_GATE      = True     # v22 FINAL: kabul (SON13G +0.8, OOS +0.3, AYI/-6AY ≈ nötr)
FUND_EXT          = 0.0005   # 8 saatlik funding eşiği (%0.05)
# MAKER MODU — PB pullback girişleri LIMIT emirle (M7 emsali): giriş maliyeti
# %0.15 → %0.05. BO kırılım girişleri taker kalır (momentum kovalama), çıkışlar taker.
MAKER_PB          = True     # v22 FINAL: kabul — strict dolum modeliyle bile 4/5 pencere iyileşti
MAKER_FILL        = "strict" # "optimistic": sinyal barında dolum | "strict": 3 bar içinde
                             # fiyat limitten geçmezse işlem İPTAL (dürüst dolum modeli)
MAKER_COMMISSION  = 0.0002   # Binance maker ~%0.02 (M7 ile aynı)
MAKER_SLIPPAGE    = 0.0003   # kuyruk riski payı (M7 ile aynı)
MAKER_TTL_BARS    = 3        # strict: limit emrin ömrü (bar)

# v15: MAKRO YÖN KATMANI — BTC 30g EMA pozisyonu (M5'in 1.5× ağırlıklı ana faktörü).
# Sürünen ayıda 1h rejimi RANGE okur (6ay: %54 RANGE vs M5 %68 BEAR) → ralliye short
# satma fırsatı kaçar, dipte thrust-short yenir. Makro katman bunu düzeltir:
#   macro=-1 (ayı): short serbest (1h rejim şartı yok, exec-trend yeter), long sıkı kapı
#   macro=+1 (boğa): tam tersi | macro=0: mevcut davranış
MACRO_ON          = False    # v15/v16 DENENDİ, GERİ ALINDI: ayıyı +24'e çıkardı ama
                             # boğa ve 6ay'ı bozdu (favori yön gevşeyince komisyon seli)
MACRO_EMA_H       = 720      # 30 gün (1h bar)
MACRO_HYST        = 0.01     # ±%1 histerezis
# v17: KADEMELİ DD VALİSİ — impuls haftaları DD<%11'de gezer, 6ay chop %36'ya sürükler.
# Vali derinleştikçe boyu agresif kısar: chop kaybı küçülür, impuls neredeyse dokunulmaz.
DD_STAGES = [(0.20, 0.20), (0.15, 0.35), (0.08, 0.60)]   # (eşik, çarpan)
# v21: YAPIŞKAN VALİ — tetiklenen kademe, DD < GOV_RELEASE olana dek gevşemez.
# Gerekçe: chop'ta DD %8 altına iner inmez tam boya dönüş → tekrar kanama döngüsü.
# Strateji kanarken küçük kalmalı, ancak equity kendini kanıtlayınca büyümeli.
GOV_STICKY  = True
GOV_RELEASE = 0.04
THRUST_BTC_24H    = 0.02     # BTC 24h |getiri| eşiği
THRUST_DISP_24H   = 0.03     # coinler arası 24h getiri std eşiği
BRAKES_ON         = True     # v12 frenleri (sembol duraklatma + haftalık fren)
ENABLE_SQ = False    # v4: KAPALI — v2/v3'te her iki yönde zarar (78 işlem -$956; 5m'de squeeze = gürültü)
ENABLE_MR = False    # v5: KAPALI — v4'te her iki yönde zarar (59 işlem -$475; aynı patern)

# CS skor ağırlıkları (toplam 1.0) ve eşik
CS_THRESHOLD = 0.65

# ── Sinyal eşikleri ──────────────────────────────────────────────────────────
DONCHIAN_BARS     = 8 * K    # 8 saatlik kanal
RVOL_BREAKOUT     = 1.3
DELTA_BREAKOUT    = 0.04     # taker-buy delta oranı eşiği
RVOL_SQUEEZE      = 2.0      # v2: 1.8 → 2.0 (v1'de SQ 123 işlem, WR %12)
DELTA_SQUEEZE     = 0.10
SQUEEZE_PCTRANK   = 0.12
MR_Z_ENTRY        = 2.2
MAX_EXT_ATR       = 99.0     # v7 DENENDİ, GERİ ALINDI: 4.0 her iki pencereyi de kötüleştirdi
                             # (kripto impulsunda uzama = güç; filtre kazanan girişleri kesti)
XSEC_LONG_TOP     = 10       # long yalnızca en güçlü N coin
XSEC_SHORT_BOT    = 10       # short yalnızca en zayıf N coin

# v8: AKTİVİTE KAPISI — 6ay testi dersi: BTC RANGE %54'tü ve chop'taki işlemler
# komisyona yenildi (-76%). 5m motoru yalnızca impuls koşullarında çalışır:
#   LONG : BTC TREND_UP + thrust | SHORT: BTC TREND_DOWN + thrust
#   BTC RANGE/SQUEEZE/NA → işlem yok
# v19 FİNAL: RANGE-modu girişleri TAMAMEN KAPALI (XSEC_RANGE_TOP=0).
# Kanıt: RANGE-modu (top-5 + rvol yolu) 6ay'da −23.6→−16.7 ve OOS'ta −11.7→−9.9
# fark yaratan kanama kaynağıydı; haftalık impuls kârlarına dokunmuyor.
XSEC_RANGE_TOP    = 0        # 0 = BTC RANGE'de hiç giriş yok
RVOL_RANGE_MIN    = 1.8      # (RANGE-modu kapalıyken kullanılmaz)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")

REGIME_RANGE, REGIME_UP, REGIME_DOWN, REGIME_SQUEEZE, REGIME_NA = 0, 1, 2, 3, 4
REGIME_NAMES = {0: "RANGE", 1: "TREND_UP", 2: "TREND_DOWN", 3: "SQUEEZE", 4: "NA"}


# ══════════════════════════════════════════════════════════════════════════════
# VERİ — Binance klines (taker buy hacmi dahil)
# ══════════════════════════════════════════════════════════════════════════════

_SESSION = requests.Session()

def fetch_klines(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Binance REST klines. Kolonlar: open high low close volume quote_vol trades taker_buy"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    ck = f"{symbol}_{interval}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.pkl"
    cpath = os.path.join(CACHE_DIR, ck)
    if os.path.exists(cpath):
        try:
            return pd.read_pickle(cpath)
        except Exception:
            pass

    # Kapsayan cache'ten dilimle (episod koşuları her aralığı yeniden indirmesin).
    # Güvenlik: istenen bitiş bugüne yakınsa dilimleme yapma (eski dosya güncel olmayabilir).
    if end < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1):
        import glob as _glob
        import re as _re
        for f in _glob.glob(os.path.join(CACHE_DIR, f"{symbol}_{interval}_*.pkl")):
            m = _re.search(rf"{_re.escape(symbol)}_{interval}_(\d{{8}})_(\d{{8}})\.pkl",
                           os.path.basename(f))
            if not m:
                continue
            s0 = pd.Timestamp(m.group(1), tz="UTC")
            e0 = pd.Timestamp(m.group(2), tz="UTC")
            if s0 <= start and e0 >= end:
                try:
                    big = pd.read_pickle(f)
                except Exception:
                    continue
                sl = big[(big.index >= start) & (big.index < end)]
                if not sl.empty:
                    try:
                        sl.to_pickle(cpath)
                    except Exception:
                        pass
                    return sl

    ms = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000}[interval]
    since = int(start.timestamp() * 1000)
    until = int(end.timestamp() * 1000)
    rows = []
    while since < until:
        for attempt in range(5):
            try:
                r = _SESSION.get(
                    "https://api.binance.com/api/v3/klines",
                    params=dict(symbol=symbol, interval=interval, startTime=since,
                                endTime=until, limit=1000),
                    timeout=20,
                )
                if r.status_code == 429:
                    time.sleep(10 * (attempt + 1))
                    continue
                r.raise_for_status()
                chunk = r.json()
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
        if not chunk:
            break
        rows.extend(chunk)
        since = chunk[-1][0] + ms
        if len(chunk) < 1000:
            break
        time.sleep(0.12)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "ts", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy", "taker_buy_quote", "_ig",
    ])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts")[["open", "high", "low", "close", "volume", "quote_vol", "trades", "taker_buy"]]
    df = df.astype(float).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.iloc[:-1]  # kapanmamış son mum
    try:
        df.to_pickle(cpath)
    except Exception:
        pass
    return df


def fetch_funding(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Binance USDT-M perp funding oranı geçmişi (8 saatlik). Kolon: funding."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    ck = f"{symbol}_fund_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.pkl"
    cpath = os.path.join(CACHE_DIR, ck)
    if os.path.exists(cpath):
        try:
            return pd.read_pickle(cpath)
        except Exception:
            pass
    since = int(start.timestamp() * 1000)
    until = int(end.timestamp() * 1000)
    rows = []
    while since < until:
        for attempt in range(4):
            try:
                r = _SESSION.get(
                    "https://fapi.binance.com/fapi/v1/fundingRate",
                    params=dict(symbol=symbol, startTime=since, endTime=until, limit=1000),
                    timeout=20,
                )
                if r.status_code == 429:
                    time.sleep(10 * (attempt + 1))
                    continue
                if r.status_code == 400:   # perp yok (ör. listelenmemiş coin)
                    chunk = []
                    break
                r.raise_for_status()
                chunk = r.json()
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        if not chunk:
            break
        rows.extend(chunk)
        since = int(chunk[-1]["fundingTime"]) + 1
        if len(chunk) < 1000:
            break
        time.sleep(0.1)
    if not rows:
        df = pd.DataFrame(columns=["funding"])
    else:
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["fundingTime"].astype(np.int64), unit="ms", utc=True)
        df["funding"] = df["fundingRate"].astype(float)
        df = df.set_index("ts")[["funding"]].sort_index()
        df = df[~df.index.duplicated(keep="first")]
    try:
        df.to_pickle(cpath)
    except Exception:
        pass
    return df


def resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """5m veriyi daha büyük dilime örnekler (taker_buy/volume toplanır)."""
    if tf == "5m":
        return df
    rule = {"15m": "15min"}[tf]
    agg = df.resample(rule, label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "quote_vol": "sum", "trades": "sum", "taker_buy": "sum",
    })
    return agg.dropna(subset=["close"])


# ══════════════════════════════════════════════════════════════════════════════
# İNDİKATÖRLER
# ══════════════════════════════════════════════════════════════════════════════

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()

def _choppiness(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    tr_sum = tr.rolling(n).sum()
    hh = df["high"].rolling(n).max()
    ll = df["low"].rolling(n).min()
    rng = (hh - ll).replace(0, np.nan)
    return 100 * np.log10(tr_sum / rng) / np.log10(n)

def _efficiency_ratio(close: pd.Series, n: int = 48) -> pd.Series:
    change = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n).sum()
    return change / path.replace(0, np.nan)

def _pct_rank(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).rank(pct=True)

def _anchored_vwap_day(df: pd.DataFrame) -> pd.Series:
    day = df.index.floor("1D")
    pv = (df["quote_vol"]).groupby(day).cumsum()
    v = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    return pv / v


# ══════════════════════════════════════════════════════════════════════════════
# ÖZELLİK HAZIRLAMA — 5m sinyaller + 1h rejim
# ══════════════════════════════════════════════════════════════════════════════

def prepare_symbol(df5: pd.DataFrame, df1h: pd.DataFrame) -> pd.DataFrame:
    """Tüm sinyal/rejim kolonlarını üretir. Lookahead yok: 1h kolonları shift(1) ile 5m'e taşınır."""
    out = df5.copy()

    # ── Order-flow ──
    vol = out["volume"].replace(0, np.nan)
    out["delta_pct"] = (2 * out["taker_buy"] - out["volume"]) / vol
    out["delta_pct"] = out["delta_pct"].fillna(0.0)
    out["cvd"] = (2 * out["taker_buy"] - out["volume"]).cumsum()
    out["rvol"] = out["volume"] / out["volume"].rolling(24 * K).median().replace(0, np.nan)

    # ── Temel 5m ──
    out["atr"] = _atr(out)
    out["ema20"] = _ema(out["close"], 20)
    out["ema50"] = _ema(out["close"], 50)
    out["ema200"] = _ema(out["close"], 200)
    out["rsi"] = _rsi(out["close"])
    out["vwap_d"] = _anchored_vwap_day(out)
    out["don_hi"] = out["high"].rolling(DONCHIAN_BARS).max().shift(1)
    out["don_lo"] = out["low"].rolling(DONCHIAN_BARS).min().shift(1)
    sma48 = out["close"].rolling(4 * K).mean()     # 4 saatlik ortalama
    std48 = out["close"].rolling(4 * K).std().replace(0, np.nan)
    out["sma48"] = sma48
    out["z48"] = (out["close"] - sma48) / std48
    bb_std = out["close"].rolling(20).std()
    bb_mid = out["close"].rolling(20).mean()
    out["bb_up"] = bb_mid + 2 * bb_std
    out["bb_dn"] = bb_mid - 2 * bb_std
    out["bbw_rank"] = _pct_rank((4 * bb_std / bb_mid), 48 * K)
    # günlük gerçekleşen vol (boyutlama): 24 saatlik getiri std × sqrt(gün barı)
    ret = out["close"].pct_change()
    out["dvol"] = ret.rolling(24 * K).std() * math.sqrt(24 * K)

    # ── 1h rejim ──
    h = df1h.copy()
    h["adx"] = _adx(h)
    h["chop"] = _choppiness(h)
    h["er"] = _efficiency_ratio(h["close"], 48)
    h["ema50"] = _ema(h["close"], 50)
    h["ema200"] = _ema(h["close"], 200)
    h["atr_rank"] = _pct_rank(_atr(h) / h["close"], 720)
    h_bbstd = h["close"].rolling(20).std()
    h_bbmid = h["close"].rolling(20).mean()
    h["bbw_rank"] = _pct_rank(4 * h_bbstd / h_bbmid, 720)
    h["atr1h"] = _atr(h)

    trendy = ((h["er"] >= 0.30) & (h["adx"] >= 20)) | (h["adx"] >= 27)
    up = (h["ema50"] > h["ema200"]) & (h["close"] > h["ema50"])
    dn = (h["ema50"] < h["ema200"]) & (h["close"] < h["ema50"])
    regime = pd.Series(REGIME_RANGE, index=h.index, dtype="int8")
    regime[trendy & up] = REGIME_UP
    regime[trendy & dn] = REGIME_DOWN
    regime[(~trendy) & (h["bbw_rank"] < 0.15)] = REGIME_SQUEEZE
    regime[h["ema200"].isna()] = REGIME_NA
    h["regime"] = regime

    # 1h → 5m (yalnızca kapanmış 1h barı: shift(1) sonra ffill)
    h_shift = h[["regime", "chop", "atr_rank", "atr1h", "ema50"]].shift(1)
    aligned = h_shift.reindex(out.index, method="ffill")
    out["regime"] = aligned["regime"].fillna(REGIME_NA).astype("int8")
    out["chop_1h"] = aligned["chop"]
    out["atr_rank_1h"] = aligned["atr_rank"]
    out["atr1h"] = aligned["atr1h"]
    # v7: uzama — fiyatın 1h EMA50'den ATR cinsinden mesafesi (geç kovalama filtresi)
    out["ext1h"] = (out["close"] - aligned["ema50"]) / aligned["atr1h"].replace(0, np.nan)

    # ── Sinyaller (vektörel, lookahead'siz) ──
    r_up = out["regime"] == REGIME_UP
    r_dn = out["regime"] == REGIME_DOWN
    r_rng = out["regime"] == REGIME_RANGE
    r_sq = out["regime"] == REGIME_SQUEEZE

    # v7: uzama filtresi — geç kovalama yok (boğa haftası: tepe fazı pullback alımları
    # impuls kârının tamamını geri veriyordu)
    not_ext_up = out["ext1h"] <= MAX_EXT_ATR
    not_ext_dn = out["ext1h"] >= -MAX_EXT_ATR

    # Trend breakout — KENAR TETİK: önceki bar kanal içindeyken bu bar kırılım
    bo_l_raw = (out["close"] > out["don_hi"])
    bo_s_raw = (out["close"] < out["don_lo"])
    out["sig_bo_L"] = (r_up & bo_l_raw & ~bo_l_raw.shift(1).fillna(False)
                       & (out["rvol"] >= RVOL_BREAKOUT) & not_ext_up
                       & (out["delta_pct"] >= DELTA_BREAKOUT) & (out["close"] > out["vwap_d"]))
    out["sig_bo_S"] = (r_dn & bo_s_raw & ~bo_s_raw.shift(1).fillna(False)
                       & (out["rvol"] >= RVOL_BREAKOUT) & not_ext_dn
                       & (out["delta_pct"] <= -DELTA_BREAKOUT) & (out["close"] < out["vwap_d"]))

    # Trend pullback — TAZE DOKUNUŞ: fiyat son 12 barda EMA20'den ≥1 ATR uzaklaşmış olmalı,
    # önceki bar dokunmamışken bu bar dokunuyor (v1: seviye-tetik → 222 işlem, hepsi gürültü)
    near20 = (out["close"] - out["ema20"]).abs() <= 0.35 * out["atr"]
    was_away = ((out["close"] - out["ema20"]).abs().rolling(1 * K).max() >= 1.0 * out["atr"]).shift(1).fillna(False)
    fresh_touch = near20 & ~near20.shift(1).fillna(False) & was_away
    out["sig_pb_L"] = (r_up & fresh_touch & (out["close"] > out["ema50"]) & not_ext_up
                       & out["rsi"].between(35, 58) & (out["delta_pct"] > 0.02))
    out["sig_pb_S"] = (r_dn & fresh_touch & (out["close"] < out["ema50"]) & not_ext_dn
                       & out["rsi"].between(42, 65) & (out["delta_pct"] < -0.02))

    # v15: MAKRO-GEVŞETİLMİŞ pullback — 1h rejim şartı yok, exec-TF trend yeter.
    # Sürünen ayıda (1h RANGE etiketli) ralliye short satmayı mümkün kılar.
    # Yalnızca makro yön doğruysa motor tarafından kullanılır.
    out["sig_pb_L2"] = (~r_dn & fresh_touch
                        & (out["close"] > out["ema50"]) & (out["close"] > out["ema200"])
                        & out["rsi"].between(35, 58) & (out["delta_pct"] > 0.02))
    out["sig_pb_S2"] = (~r_up & fresh_touch
                        & (out["close"] < out["ema50"]) & (out["close"] < out["ema200"])
                        & out["rsi"].between(42, 65) & (out["delta_pct"] < -0.02))

    # Squeeze patlaması — kenar tetik
    sq_ctx = r_sq | (out["bbw_rank"].shift(1) < SQUEEZE_PCTRANK)
    sq_l_raw = out["close"] > out["bb_up"]
    sq_s_raw = out["close"] < out["bb_dn"]
    out["sig_sq_L"] = (sq_ctx & sq_l_raw & ~sq_l_raw.shift(1).fillna(False)
                       & (out["rvol"] >= RVOL_SQUEEZE) & (out["delta_pct"] >= DELTA_SQUEEZE))
    out["sig_sq_S"] = (sq_ctx & sq_s_raw & ~sq_s_raw.shift(1).fillna(False)
                       & (out["rvol"] >= RVOL_SQUEEZE) & (out["delta_pct"] <= -DELTA_SQUEEZE))

    # Range mean-reversion — kenar tetik (z eşiği geçiş anı)
    mr_ok = r_rng & (out["chop_1h"] >= 55) & (out["atr_rank_1h"] <= 0.85)
    mr_l_raw = out["z48"] < -MR_Z_ENTRY
    mr_s_raw = out["z48"] > MR_Z_ENTRY
    out["sig_mr_L"] = (mr_ok & mr_l_raw & ~mr_l_raw.shift(1).fillna(False)
                       & (out["rsi"] < 30) & (out["delta_pct"] > 0))
    out["sig_mr_S"] = (mr_ok & mr_s_raw & ~mr_s_raw.shift(1).fillna(False)
                       & (out["rsi"] > 70) & (out["delta_pct"] < 0))

    # CVD diverjans çıkışı için: son 24 saatin fiyat & cvd uçları
    out["price_hh"] = out["high"].rolling(24 * K).max()
    out["cvd_max"] = out["cvd"].rolling(24 * K).max()
    out["price_ll"] = out["low"].rolling(24 * K).min()
    out["cvd_min"] = out["cvd"].rolling(24 * K).min()

    # ── v11: KOMPOZİT SKOR (CS) — M5 tarzı çok-faktör + M9 order-flow ──
    ema12 = _ema(out["close"], 12)
    ema26 = _ema(out["close"], 26)
    macd = ema12 - ema26
    macd_sig = _ema(macd, 9)
    macd_hist = macd - macd_sig
    out["ema200"] = _ema(out["close"], 200)
    adx15 = _adx(out)
    cvd_ema = _ema(out["cvd"], 20)
    tsmom = out["close"].pct_change(24 * K)
    bb_rng = (out["bb_up"] - out["bb_dn"]).replace(0, np.nan)
    bb_pct_b = (out["close"] - out["bb_dn"]) / bb_rng

    def _w(cond, w):
        return cond.fillna(False).astype(float) * w

    long_score = (
        _w((out["close"] > out["ema50"]) & (out["ema50"] > out["ema200"]), 0.25)   # trend hizası
        + _w((macd_hist > 0) & (macd_hist > macd_hist.shift(1)), 0.15)             # momentum taze
        + _w(out["rsi"].between(50, 75), 0.10)                                      # sağlıklı bölge
        + _w(adx15 >= 22, 0.10)                                                     # trend gücü
        + _w(out["cvd"] > cvd_ema, 0.15)                                            # order-flow yönü
        + _w(out["delta_pct"] > 0.05, 0.10)                                         # bar içi alıcı baskısı
        + _w(out["rvol"] >= 1.2, 0.05)                                              # hacim katılımı
        + _w(tsmom > 0, 0.10)                                                       # 24h momentum
    )
    short_score = (
        _w((out["close"] < out["ema50"]) & (out["ema50"] < out["ema200"]), 0.25)
        + _w((macd_hist < 0) & (macd_hist < macd_hist.shift(1)), 0.15)
        + _w(out["rsi"].between(25, 50), 0.10)
        + _w(adx15 >= 22, 0.10)
        + _w(out["cvd"] < cvd_ema, 0.15)
        + _w(out["delta_pct"] < -0.05, 0.10)
        + _w(out["rvol"] >= 1.2, 0.05)
        + _w(tsmom < 0, 0.10)
    )
    # Vetolar: aşırı uç bölgede giriş yok (sekme/parabolik risk)
    long_veto = (out["rsi"] > 85) | (out["z48"] > 3)
    short_veto = (out["rsi"] < 20) | (out["z48"] < -3)

    cs_l_raw = (long_score >= CS_THRESHOLD) & ~long_veto
    cs_s_raw = (short_score >= CS_THRESHOLD) & ~short_veto
    # Kenar tetik: skor eşiği yeni aşıldıysa (her barda yeniden giriş yok)
    out["sig_cs_L"] = r_up & cs_l_raw & ~cs_l_raw.shift(1).fillna(False)
    out["sig_cs_S"] = r_dn & cs_s_raw & ~cs_s_raw.shift(1).fillna(False)

    return out


def xsec_momentum_ranks(closes_1h: pd.DataFrame) -> pd.DataFrame:
    """Kesitsel momentum: 4h/24h/72h vol-ayarlı getiri z-skoru karışımı → saatlik rank (1=en güçlü).
    shift(1): yalnızca kapanmış saat kullanılır."""
    r4 = closes_1h.pct_change(4)
    r24 = closes_1h.pct_change(24)
    r72 = closes_1h.pct_change(72)
    vol = closes_1h.pct_change().rolling(72).std().replace(0, np.nan)
    score = (r4 / vol + r24 / (vol * 2) + r72 / (vol * 3)) / 3
    z = score.sub(score.mean(axis=1), axis=0).div(score.std(axis=1).replace(0, np.nan), axis=0)
    ranks = z.rank(axis=1, ascending=False)  # 1 = en güçlü momentum
    return ranks.shift(1)


# ══════════════════════════════════════════════════════════════════════════════
# PORTFÖY MOTORU
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Pos:
    sym: str
    is_short: bool
    qty: float
    avg_entry: float
    stop: float
    risk_usd: float          # 1R (ilk giriş riski $)
    entry_price0: float      # ilk giriş fiyatı (R hesabı)
    stop_dist0: float        # ilk stop mesafesi (R hesabı)
    hw: float                # high-water (long) / low-water (short)
    strategy: str            # BO/PB/SQ/MR
    opened_i: int
    adds_done: int = 0
    partial_done: bool = False
    div_exit_done: bool = False
    fees_paid: float = 0.0
    atr_at_entry: float = 0.0
    realized: float = 0.0    # tüm parçaların net PnL'i (giriş ücretleri dahil)
    be_done: bool = False    # v20 K4: break-even stop uygulandı mı
    # teşhis alanları (giriş anı bağlamı + yolculuk)
    mfe_r: float = 0.0       # max lehte gidiş (R)
    mae_r: float = 0.0       # max aleyhte gidiş (R)
    btc_reg_e: int = -1      # giriş anında BTC rejimi
    rank_e: float = float("nan")
    rvol_e: float = float("nan")
    delta_e: float = float("nan")
    thrust_btc_e: float = float("nan")
    disp_e: float = float("nan")


@dataclass
class Trade:
    sym: str
    is_short: bool
    strategy: str
    entry_ts: object
    exit_ts: object
    entry_px: float
    exit_px: float
    pnl: float
    reason: str
    adds: int
    # teşhis alanları
    r_final: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    bars: int = 0
    btc_reg: int = -1
    rank: float = float("nan")
    rvol: float = float("nan")
    delta: float = float("nan")
    thrust_btc: float = float("nan")
    disp: float = float("nan")


def run_backtest(start: str, end: str, capital: float = INITIAL_CAPITAL,
                 symbols=None, risk_mult: float = 1.0, verbose: bool = True,
                 keep_open: bool = False):
    """keep_open=True (canlı shadow modu): test sonunda açık pozisyonlar EOT ile
    KAPATILMAZ; sonuçta 'open_positions' anlık görüntüsü döner (defter aynalama için)."""
    symbols = symbols or UNIVERSE
    t_start = pd.Timestamp(start, tz="UTC")
    t_end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    fetch_lo = t_start - pd.Timedelta(days=35)   # 1h ema200 + 720 bar pct_rank ısınması

    # ── Veri ──
    data5, data1h = {}, {}
    for i, sym in enumerate(symbols):
        df5 = fetch_klines(sym, "5m", fetch_lo, t_end)
        # v15: 1h verisi 75 gün geriden — BTC makro EMA720 (30g) yakınsaması için
        df1h = fetch_klines(sym, "1h", fetch_lo - pd.Timedelta(days=40), t_end)
        if df5.empty or len(df5) < 600 or df1h.empty:
            if verbose:
                print(f"  ⚠ {sym}: veri yok/yetersiz — atlandı")
            continue
        data5[sym], data1h[sym] = resample_ohlcv(df5, EXEC_TF), df1h
        if verbose and (i + 1) % 10 == 0:
            print(f"  veri {i+1}/{len(symbols)}")
    syms = list(data5.keys())
    if "BTCUSDT" not in syms:
        raise RuntimeError("BTC verisi şart (global rejim)")

    # ── Özellikler ──
    feats = {s: prepare_symbol(data5[s], data1h[s]) for s in syms}

    # Kesitsel momentum (1h kapanışlardan)
    closes_1h = pd.DataFrame({s: data1h[s]["close"] for s in syms}).sort_index()
    ranks_1h = xsec_momentum_ranks(closes_1h)

    # ── Global zaman çizgisi + hizalı matrisler ──
    # Global saat = BTC'nin bar çizgisi: BTC-okumaları (rejim, thrust) asla NaN olamaz.
    # (Eski union yaklaşımı, bitiş=bugün iken BTC'den sonra çekilen bir coine 1 bar
    #  daha yeni veri gelirse BTC satırını NaN bırakıp int(NaN) çökmesine yol açıyordu.)
    gidx = feats["BTCUSDT"].index
    gidx = gidx[gidx <= t_end]
    cols = ["open", "high", "low", "close", "atr", "atr1h", "rvol", "delta_pct", "cvd", "regime",
            "sig_cs_L", "sig_cs_S",
            "sig_bo_L", "sig_bo_S", "sig_pb_L", "sig_pb_S",
            "sig_pb_L2", "sig_pb_S2",
            "sig_sq_L", "sig_sq_S", "sig_mr_L", "sig_mr_S",
            "sma48", "dvol", "price_hh", "cvd_max", "price_ll", "cvd_min"]
    M = {}
    for c in cols:
        M[c] = pd.DataFrame({s: feats[s][c] for s in syms}).reindex(gidx).to_numpy()
    R = ranks_1h.reindex(columns=syms).reindex(gidx, method="ffill").to_numpy()
    sym_i = {s: j for j, s in enumerate(syms)}
    btc_j = sym_i["BTCUSDT"]

    trade_mask = (gidx >= t_start) & (gidx < t_end)
    n = len(gidx)

    # v22: funding oranı matrisi (kalabalık-pozisyon kapısı için)
    F = np.full((n, len(syms)), np.nan)
    if FUNDING_GATE:
        for s in syms:
            try:
                fdf = fetch_funding(s, fetch_lo, t_end)
            except Exception:
                continue
            if fdf.empty:
                continue
            F[:, sym_i[s]] = fdf["funding"].reindex(gidx, method="ffill").to_numpy()

    # v13: piyasa itkisi (thrust) — BTC 24h hareketi + kesitsel dağılım
    _lag = 24 * K
    _cl = M["close"]
    btc_ret24 = np.full(n, np.nan)
    btc_ret24[_lag:] = _cl[_lag:, btc_j] / _cl[:-_lag, btc_j] - 1
    all_ret24 = np.full_like(_cl, np.nan)
    all_ret24[_lag:, :] = _cl[_lag:, :] / _cl[:-_lag, :] - 1
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        disp24 = np.nanstd(all_ret24, axis=1)
    # v15: BTC makro yön (30g EMA + histerezis), kapanmış 1h barından
    if MACRO_ON:
        _btc1h = data1h["BTCUSDT"]
        _m_ema = _ema(_btc1h["close"], MACRO_EMA_H)
        _m_raw = pd.Series(np.where(_btc1h["close"] > _m_ema * (1 + MACRO_HYST), 1.0,
                           np.where(_btc1h["close"] < _m_ema * (1 - MACRO_HYST), -1.0, np.nan)),
                           index=_btc1h.index)
        macro = (_m_raw.ffill().fillna(0).shift(1)
                 .reindex(gidx, method="ffill").fillna(0).to_numpy())
    else:
        macro = np.zeros(n)

    # v20 K-turn: BTC hızlı momentum (4h) — 24h ile işaret çelişkisi = dönüm noktası
    _lag4 = TURN_FAST_H * K
    btc_ret4 = np.full(n, np.nan)
    btc_ret4[_lag4:] = _cl[_lag4:, btc_j] / _cl[:-_lag4, btc_j] - 1
    turn_block = np.zeros(n, dtype=bool)
    if TURN_GATE:
        with np.errstate(invalid="ignore"):
            turn_block = ((btc_ret24 * btc_ret4 < 0)
                          & (np.abs(btc_ret24) >= TURN_MIN_24H)
                          & (np.abs(btc_ret4) >= TURN_MIN_4H))

    _btc_up = btc_ret24 >= THRUST_BTC_24H
    _btc_dn = btc_ret24 <= -THRUST_BTC_24H
    _disp_hi = disp24 >= THRUST_DISP_24H
    # v20 K-M: makro-çapalı itki — BTC 30g EMA'sının üstünde/altında kopuş şartı
    if THRUST_MACRO_ANCHOR:
        _b1h = data1h["BTCUSDT"]
        _anchor = _ema(_b1h["close"], MACRO_EMA_H)
        _above = ((_b1h["close"] > _anchor * (1 + MACRO_HYST)).shift(1)
                  .reindex(gidx, method="ffill").fillna(False).to_numpy(dtype=bool))
        _below = ((_b1h["close"] < _anchor * (1 - MACRO_HYST)).shift(1)
                  .reindex(gidx, method="ffill").fillna(False).to_numpy(dtype=bool))
        with np.errstate(invalid="ignore"):
            _viol_up = (btc_ret24 >= THRUST_VIOLENT) if THRUST_VIOLENT > 0 else np.zeros(n, dtype=bool)
            _viol_dn = (btc_ret24 <= -THRUST_VIOLENT) if THRUST_VIOLENT > 0 else np.zeros(n, dtype=bool)
        _btc_up = (_btc_up & _above) | _viol_up
        _btc_dn = (_btc_dn & _below) | _viol_dn
    # v20 K3g: BTC-itkisi yalnızca dispersiyon eşliğindeyse geçerli
    if THRUST_DISP_QUAL > 0:
        _disp_qual = ~(np.isnan(disp24)) & (disp24 >= THRUST_DISP_QUAL)
        _btc_up = _btc_up & _disp_qual
        _btc_dn = _btc_dn & _disp_qual
    thrust_ok = _btc_up | _btc_dn | _disp_hi
    if THRUST_DIRECTIONAL:
        # v14: yönlü itki — BTC yönü baskınsa sadece o yön; BTC sakin + dispersiyon
        # yüksekse (alt-spesifik hareket) iki yön de serbest
        thrust_long = _btc_up | (~_btc_dn & _disp_hi)
        thrust_short = _btc_dn | (~_btc_up & _disp_hi)
    else:
        thrust_long = thrust_ok.copy()
        thrust_short = thrust_ok.copy()
    if not THRUST_ON:
        thrust_ok[:] = True
        thrust_long[:] = True
        thrust_short[:] = True

    # ── Durum ──
    balance = capital
    peak_equity = capital
    positions: dict[str, Pos] = {}
    trades: list[Trade] = []
    equity_curve = []
    day_realized = 0.0
    cur_day = None
    kill_until_day = None
    regime_counter = {k: 0 for k in REGIME_NAMES.values()}
    last_exit_i: dict[str, int] = {}    # cooldown: son çıkış barı
    last_loss_i: dict[str, int] = {}    # cooldown: son zararlı çıkış barı
    sym_history: dict[str, list] = {}   # v12: sembol başına son işlem PnL'leri
    sym_paused_until: dict[str, int] = {}
    eq_hist: list = []                  # v12: bar bazlı equity (haftalık fren için)
    brake_until = -1
    blocked_log: list = []              # teşhis: kapılarca bloklanan sinyaller
    gov_stage = 1.0                     # v21: yapışkan vali durumu
    pending_limits: dict[str, dict] = {}  # v22: bekleyen maker limit emirleri (strict)

    def _fees(notional):
        return notional * (COMMISSION + SLIPPAGE)

    def _close_pos(p: Pos, px: float, ts, i: int, reason: str, frac: float = 1.0):
        nonlocal balance, day_realized
        qty = p.qty * frac
        notional = qty * px
        fee = _fees(notional)
        if p.is_short:
            pnl = (p.avg_entry - px) * qty - fee
        else:
            pnl = (px - p.avg_entry) * qty - fee
        balance += pnl
        day_realized += pnl
        p.realized += pnl
        p.qty -= qty
        p.fees_paid += fee
        if frac >= 0.999 or p.qty * px < MIN_ORDER_USD:
            if p.qty > 0 and p.qty * px >= 0.01:  # kalan kırıntıyı da kapat
                rq = p.qty
                rfee = _fees(rq * px)
                rpnl = ((p.avg_entry - px) if p.is_short else (px - p.avg_entry)) * rq - rfee
                balance += rpnl
                day_realized += rpnl
                p.realized += rpnl
                p.qty = 0.0
            trades.append(Trade(p.sym, p.is_short, p.strategy,
                                gidx[p.opened_i], ts, p.entry_price0, px, p.realized, reason, p.adds_done,
                                r_final=(p.realized / p.risk_usd if p.risk_usd > 0 else 0.0),
                                mfe_r=p.mfe_r, mae_r=p.mae_r, bars=i - p.opened_i,
                                btc_reg=p.btc_reg_e, rank=p.rank_e, rvol=p.rvol_e,
                                delta=p.delta_e, thrust_btc=p.thrust_btc_e, disp=p.disp_e))
            del positions[p.sym]
            last_exit_i[p.sym] = i
            if p.realized < 0:
                last_loss_i[p.sym] = i
            # v12: sembol rolling performans duraklatması (legacy rolling-WR pause)
            if BRAKES_ON:
                h = sym_history.setdefault(p.sym, [])
                h.append(p.realized)
                recent = h[-SYM_PAUSE_TRADES:]
                if (len(recent) >= SYM_PAUSE_TRADES
                        and sum(1 for x in recent if x < 0) >= SYM_PAUSE_LOSSES
                        and sum(recent) < 0):
                    sym_paused_until[p.sym] = i + SYM_PAUSE_BARS
                    h.clear()  # duraklama sonrası temiz sayfa
        return pnl

    def _unreal(p: Pos, px: float) -> float:
        return ((p.avg_entry - px) if p.is_short else (px - p.avg_entry)) * p.qty

    def _r_now(p: Pos, px: float) -> float:
        d = (p.entry_price0 - px) if p.is_short else (px - p.entry_price0)
        return d / p.stop_dist0 if p.stop_dist0 > 0 else 0.0

    # ── Ana döngü ──
    for i in range(n):
        ts = gidx[i]
        in_window = trade_mask[i]

        d = ts.floor("1D")
        if cur_day is None or d != cur_day:
            cur_day = d
            day_realized = 0.0

        # ── Çıkış yönetimi ──
        for sym in list(positions.keys()):
            p = positions[sym]
            j = sym_i[sym]
            hi, lo, cl = M["high"][i, j], M["low"][i, j], M["close"][i, j]
            atr = M["atr1h"][i, j]   # v2: iz süren stop 1h ATR ile nefes alır
            if np.isnan(cl):
                continue

            r = _r_now(p, cl)
            # teşhis: MFE/MAE (R cinsinden, bar uçlarıyla)
            if p.stop_dist0 > 0:
                if p.is_short:
                    p.mfe_r = max(p.mfe_r, (p.entry_price0 - lo) / p.stop_dist0)
                    p.mae_r = min(p.mae_r, (p.entry_price0 - hi) / p.stop_dist0)
                else:
                    p.mfe_r = max(p.mfe_r, (hi - p.entry_price0) / p.stop_dist0)
                    p.mae_r = min(p.mae_r, (lo - p.entry_price0) / p.stop_dist0)
            # v20 K4: +BE_STOP_AT_R gören işlem zarara dönmesin — stop girişe çekilir
            # (6AY kayıplarının %48'i 0.3-1R görüp ölüyordu)
            if BE_STOP_AT_R > 0 and not p.be_done and r >= BE_STOP_AT_R:
                if p.is_short:
                    p.stop = min(p.stop, p.entry_price0)
                else:
                    p.stop = max(p.stop, p.entry_price0)
                p.be_done = True

            # v6: trail ancak işlem +TRAIL_START_R'ye ulaştıktan (veya piramit eklendikten)
            # sonra başlar — öncesinde ilk stop sabit kalır, trend nefes alır
            trail_on = (r >= TRAIL_START_R) or (p.adds_done > 0) or p.partial_done

            # high/low-water + chandelier
            if p.is_short:
                p.hw = min(p.hw, lo)
                ch_mult = CHANDELIER_TIGHT if p.adds_done > 0 else CHANDELIER_MULT
                if p.strategy != "MR" and trail_on and not np.isnan(atr):
                    new_stop = p.hw + ch_mult * atr
                    if new_stop < p.stop:
                        p.stop = new_stop
                if hi >= p.stop:
                    _close_pos(p, p.stop, ts, i, "SL/TS")
                    continue
            else:
                p.hw = max(p.hw, hi)
                ch_mult = CHANDELIER_TIGHT if p.adds_done > 0 else CHANDELIER_MULT
                if p.strategy != "MR" and trail_on and not np.isnan(atr):
                    new_stop = p.hw - ch_mult * atr
                    if new_stop > p.stop:
                        p.stop = new_stop
                if lo <= p.stop:
                    _close_pos(p, p.stop, ts, i, "SL/TS")
                    continue

            if p.strategy == "MR":
                tgt = M["sma48"][i, j]
                hit = (cl <= tgt) if p.is_short else (cl >= tgt)
                if hit and not np.isnan(tgt):
                    _close_pos(p, cl, ts, i, "TP")
                    continue
                if i - p.opened_i >= MR_TIME_BARS:
                    _close_pos(p, cl, ts, i, "TIME")
                    continue
                continue

            # Kısmi kâr
            if not p.partial_done and r >= PARTIAL_AT_R:
                _close_pos(p, cl, ts, i, "PARTIAL", frac=PARTIAL_PCT)
                if sym not in positions:
                    continue
                p.partial_done = True

            # CVD diverjans: fiyat yeni uç yapıyor ama CVD doğrulamıyor → yarıyı kapat
            if not p.div_exit_done and r >= 1.0:
                if p.is_short:
                    px_new_low = lo <= M["price_ll"][i, j]
                    cvd_no = M["cvd"][i, j] > M["cvd_min"][i, j]
                    if px_new_low and cvd_no:
                        _close_pos(p, cl, ts, i, "CVD-DIV", frac=0.5)
                        if sym not in positions:
                            continue
                        p.div_exit_done = True
                else:
                    px_new_high = hi >= M["price_hh"][i, j]
                    cvd_no = M["cvd"][i, j] < M["cvd_max"][i, j]
                    if px_new_high and cvd_no:
                        _close_pos(p, cl, ts, i, "CVD-DIV", frac=0.5)
                        if sym not in positions:
                            continue
                        p.div_exit_done = True

            # Stagnasyon
            if i - p.opened_i >= STAGNATION_BARS and r < STAGNATION_MIN_R:
                _close_pos(p, cl, ts, i, "STAG")
                continue

            # Rejim tersine döndü → trend işleminden çık (kârda ise)
            reg = M["regime"][i, j]
            if p.strategy in ("CS", "BO", "PB") and r > 0.5:
                if (not p.is_short and reg == REGIME_DOWN) or (p.is_short and reg == REGIME_UP):
                    _close_pos(p, cl, ts, i, "REGIME-FLIP")
                    continue

            # ── Piramitleme ──
            if p.adds_done < len(PYR_LEVELS) and p.strategy in ("CS", "BO", "PB", "SQ"):
                lvl = PYR_LEVELS[p.adds_done]
                flow_ok = (M["delta_pct"][i, j] < 0) if p.is_short else (M["delta_pct"][i, j] > 0)
                reg_ok = (reg == REGIME_DOWN) if p.is_short else (reg == REGIME_UP)
                if r >= lvl["at_r"] and flow_ok and (reg_ok or p.strategy == "SQ"):
                    add_qty = (p.risk_usd * lvl["size_mult"]) / p.stop_dist0
                    notional = add_qty * cl
                    equity_now = balance + sum(_unreal(q, M["close"][i, sym_i[q.sym]]) for q in positions.values())
                    if notional >= MIN_ORDER_USD and (p.qty * cl + notional) <= MAX_POSITION_PCT * equity_now * 1.6:
                        fee = _fees(notional)
                        balance -= fee
                        p.fees_paid += fee
                        p.realized -= fee
                        p.avg_entry = (p.avg_entry * p.qty + cl * add_qty) / (p.qty + add_qty)
                        p.qty += add_qty
                        p.adds_done += 1
                        # stop'u kilitle: BE+ / +1R
                        lock = lvl["stop_to_r"] * p.stop_dist0
                        if p.is_short:
                            p.stop = min(p.stop, p.entry_price0 - lock)
                        else:
                            p.stop = max(p.stop, p.entry_price0 + lock)

        # ── Girişler ──
        if in_window:
            _breg = M["regime"][i, btc_j]
            regime_counter[REGIME_NAMES[REGIME_NA if np.isnan(_breg) else int(_breg)]] += 1

        equity = balance + sum(_unreal(p, M["close"][i, sym_i[p.sym]]) for p in positions.values()
                               if not np.isnan(M["close"][i, sym_i[p.sym]]))
        peak_equity = max(peak_equity, equity)
        # v17: kademeli DD valisi (+v21: yapışkan mod)
        _dd = 1 - equity / peak_equity if peak_equity > 0 else 0.0
        dd_factor = 1.0
        for _thr, _mult in DD_STAGES:
            if _dd > _thr:
                dd_factor = _mult
                break
        if GOV_STICKY:
            if _dd < GOV_RELEASE:
                gov_stage = 1.0
            else:
                gov_stage = min(gov_stage, dd_factor)
            dd_factor = gov_stage
        eq_hist.append(equity)

        kill = kill_until_day is not None and cur_day < kill_until_day
        if day_realized <= DAILY_KILL_PCT * equity:
            kill_until_day = cur_day + pd.Timedelta(days=1)
            kill = True

        # v12: haftalık portföy freni — 7 günde > %6 erime → 24 saat dur
        if BRAKES_ON:
            wb_idx = i - 7 * 24 * K
            if wb_idx >= 0 and eq_hist[wb_idx] > 0:
                if (eq_hist[wb_idx] - equity) / eq_hist[wb_idx] > WEEK_BRAKE_PCT:
                    brake_until = max(brake_until, i + WEEK_BRAKE_BARS)
            if i < brake_until:
                kill = True

        if in_window:
            _breg2 = M["regime"][i, btc_j]
            btc_reg = REGIME_NA if np.isnan(_breg2) else int(_breg2)
            mc = int(macro[i]) if MACRO_ON else 0
            n_long = sum(1 for p in positions.values() if not p.is_short)
            n_short = sum(1 for p in positions.values() if p.is_short)

            # v22: bekleyen maker limit emirlerini doldur (strict dolum modeli)
            # Not: resting emir kill-switch'ten etkilenmez (borsada zaten beklemekte)
            for s in list(pending_limits.keys()):
                pl = pending_limits[s]
                if i - pl["created_i"] > MAKER_TTL_BARS:
                    del pending_limits[s]
                    continue
                j = sym_i[s]
                lo, hi = M["low"][i, j], M["high"][i, j]
                if np.isnan(lo):
                    continue
                hit = (hi >= pl["limit_px"]) if pl["is_short"] else (lo <= pl["limit_px"])
                if not hit:
                    continue
                del pending_limits[s]
                if s in positions or len(positions) >= MAX_POSITIONS:
                    continue
                if (pl["is_short"] and n_short >= MAX_SAME_DIR) or \
                        (not pl["is_short"] and n_long >= MAX_SAME_DIR):
                    continue
                _fill = pl["limit_px"] * (1 - MAKER_SLIPPAGE) if pl["is_short"] \
                    else pl["limit_px"] * (1 + MAKER_SLIPPAGE)
                _notional = pl["qty"] * _fill
                _fee = _notional * MAKER_COMMISSION
                balance -= _fee
                _stop = _fill + pl["stop_dist"] if pl["is_short"] else _fill - pl["stop_dist"]
                positions[s] = Pos(
                    sym=s, is_short=pl["is_short"], qty=pl["qty"], avg_entry=_fill,
                    stop=_stop, risk_usd=pl["risk_usd"], entry_price0=_fill,
                    stop_dist0=pl["stop_dist"], hw=_fill, strategy="PB",
                    opened_i=i, atr_at_entry=pl["atr"], realized=-_fee,
                    btc_reg_e=pl["btc_reg"], rank_e=pl["rank"], rvol_e=pl["rvol"],
                    delta_e=pl["delta"], thrust_btc_e=pl["thrust_btc"], disp_e=pl["disp"],
                )
                if pl["is_short"]:
                    n_short += 1
                else:
                    n_long += 1

            # TEŞHİS YENİDEN YAPISI: sinyaller kapılardan bağımsız tespit edilir,
            # bloklanan her sinyal sebebiyle loglanır. AÇILAN işlemler birebir aynı
            # (eski akışla eşdeğerlik son test +20.81 ile doğrulanır).
            cands = []  # (öncelik, rank, sym, yön, strateji)
            for s in syms:
                j = sym_i[s]
                if s in positions or np.isnan(M["close"][i, j]):
                    continue
                rk = R[i, j]
                rvol_ij = M["rvol"][i, j]

                # — sinyal tespiti (öncelik: CS > BO > PB > PB2 > SQ > MR) —
                sig_long = sig_short = None
                if ENABLE_CS and M["sig_cs_L"][i, j]:
                    sig_long = (0, "CS")
                elif ENABLE_BO and M["sig_bo_L"][i, j]:
                    sig_long = (0, "BO")
                elif ENABLE_PB and M["sig_pb_L"][i, j]:
                    sig_long = (1, "PB")
                elif ENABLE_PB and mc == 1 and M["sig_pb_L2"][i, j]:
                    sig_long = (1, "PB")
                elif ENABLE_SQ and M["sig_sq_L"][i, j]:
                    sig_long = (2, "SQ")
                elif ENABLE_MR and M["sig_mr_L"][i, j]:
                    sig_long = (3, "MR")
                if ENABLE_CS and M["sig_cs_S"][i, j]:
                    sig_short = (0, "CS")
                elif ENABLE_BO and M["sig_bo_S"][i, j]:
                    sig_short = (0, "BO")
                elif ENABLE_PB and M["sig_pb_S"][i, j]:
                    sig_short = (1, "PB")
                elif ENABLE_PB and mc == -1 and M["sig_pb_S2"][i, j]:
                    sig_short = (1, "PB")
                elif ENABLE_SQ and M["sig_sq_S"][i, j]:
                    sig_short = (2, "SQ")
                elif ENABLE_MR and M["sig_mr_S"][i, j]:
                    sig_short = (3, "MR")
                if sig_long is None and sig_short is None:
                    continue

                # — yön kapıları (thrust + BTC rejim, v8/v15/v16/v18 mantığı) —
                if mc == -1:
                    _thr_l = bool(thrust_long[i])
                    _dir_l = (btc_reg == REGIME_UP and not np.isnan(rk) and rk <= XSEC_RANGE_TOP)
                    _thr_s = bool(thrust_ok[i])
                    _dir_s = btc_reg != REGIME_UP
                else:
                    _thr_l = bool(thrust_long[i])
                    _dir_l = (btc_reg == REGIME_UP) or (
                        btc_reg == REGIME_RANGE
                        and not np.isnan(rk) and rk <= XSEC_RANGE_TOP
                        and not np.isnan(rvol_ij) and rvol_ij >= RVOL_RANGE_MIN)
                    _thr_s = bool(thrust_short[i])
                    _dir_s = (btc_reg == REGIME_DOWN) or (
                        btc_reg == REGIME_RANGE
                        and not np.isnan(rk) and rk >= len(syms) - XSEC_RANGE_TOP + 1
                        and not np.isnan(rvol_ij) and rvol_ij >= RVOL_RANGE_MIN)

                for sig, is_short_c in ((sig_long, False), (sig_short, True)):
                    if sig is None:
                        continue
                    prio, strat = sig
                    if is_short_c:
                        rank_ok = (strat in ("SQ", "MR")) or np.isnan(rk) or rk >= len(syms) - XSEC_SHORT_BOT + 1
                        key = -(rk if not np.isnan(rk) else 0)
                        thr_ok_c, dir_ok_c = _thr_s, _dir_s
                        _ideal_reg = btc_reg == REGIME_DOWN
                    else:
                        rank_ok = (strat in ("SQ", "MR")) or np.isnan(rk) or rk <= XSEC_LONG_TOP
                        key = rk if not np.isnan(rk) else 99
                        thr_ok_c, dir_ok_c = _thr_l, _dir_l
                        _ideal_reg = btc_reg == REGIME_UP
                    # v20 K5: ideal koşullar — yön BTC trendiyle uyumlu + güçlü itki +
                    # yüksek dispersiyon → rank kapısı atlanır, zarar cooldown'u yarıya
                    _ideal = (IDEAL_UNLOCK and _ideal_reg
                              and not np.isnan(btc_ret24[i]) and abs(btc_ret24[i]) >= IDEAL_BTC24
                              and not np.isnan(disp24[i]) and disp24[i] >= IDEAL_DISP)
                    _cd_loss = COOLDOWN_LOSS // 2 if _ideal else COOLDOWN_LOSS
                    block = None
                    if kill:
                        block = "kill"
                    elif DISP_MIN > 0 and (np.isnan(disp24[i]) or disp24[i] < DISP_MIN):
                        block = "disp_dusuk"     # v20 K2: piyasa ölü — coinler ayrışmıyor
                    elif bool(turn_block[i]):
                        block = "v_donus"        # v20 K-turn: 24h/4h momentum çelişkisi
                    elif not thr_ok_c:
                        block = "thrust"
                    elif not dir_ok_c:
                        block = "btc_rejim"
                    elif FUNDING_GATE and not np.isnan(F[i, j]) and (
                            (is_short_c and F[i, j] < -FUND_EXT)
                            or (not is_short_c and F[i, j] > FUND_EXT)):
                        block = "funding_kalabalik"   # v22: kalabalığa uçta katılma
                    elif not rank_ok and not _ideal:
                        block = "xsec"
                    elif i - last_loss_i.get(s, -10**9) < _cd_loss:
                        block = "cooldown_loss"
                    elif i - last_exit_i.get(s, -10**9) < COOLDOWN_EXIT:
                        block = "cooldown"
                    elif i < sym_paused_until.get(s, -1):
                        block = "sym_pause"
                    if block is not None:
                        blocked_log.append(dict(i=i, ts=ts, sym=s, is_short=is_short_c,
                                                strat=strat, block=block))
                    else:
                        cands.append((prio, key, s, is_short_c, strat))

            cands.sort(key=lambda x: (x[0], x[1]))
            for _, _, s, is_short, strat in cands:
                if len(positions) >= MAX_POSITIONS:
                    blocked_log.append(dict(i=i, ts=ts, sym=s, is_short=is_short,
                                            strat=strat, block="max_pos"))
                    continue
                if (is_short and n_short >= MAX_SAME_DIR) or (not is_short and n_long >= MAX_SAME_DIR):
                    blocked_log.append(dict(i=i, ts=ts, sym=s, is_short=is_short,
                                            strat=strat, block="ayni_yon"))
                    continue
                j = sym_i[s]
                cl = M["close"][i, j]
                atr = M["atr1h"][i, j]   # v2: stop mesafesi 1h ATR bazlı
                if np.isnan(atr) or atr <= 0:
                    continue
                rk = R[i, j]

                # (v3 BTC-karşı yasağı v8 aktivite kapısına taşındı — aday listesinde uygulanıyor)
                risk_pct = (RISK_MR if strat == "MR" else RISK_TREND) * risk_mult * dd_factor
                # v20 K2b: dispersiyon boyut ölçekleyici — ölü piyasada küçük keşif boyu
                if DISP_SCALE and not np.isnan(disp24[i]):
                    risk_pct *= float(np.clip(disp24[i] / IDEAL_DISP, DISP_SCALE_FLOOR, 1.0))
                # v15: makroya karşı işlem küçük boy (keşif pozisyonu)
                if (mc == -1 and not is_short) or (mc == 1 and is_short):
                    risk_pct *= 0.6
                stop_mult = ATR_STOP_MR if strat == "MR" else ATR_STOP_TREND
                stop_dist = max(stop_mult * atr, MIN_STOP_PCT * cl)

                # vol-target ölçekleme
                dv = M["dvol"][i, j]
                vt = 1.0
                if not np.isnan(dv) and dv > 0:
                    vt = float(np.clip(VOL_TARGET_DAILY / dv, 0.5, 1.5))

                risk_usd = equity * risk_pct * vt
                qty = risk_usd / stop_dist
                notional = qty * cl
                # v8: tavan da dd_factor'a uyar — yoksa DD valisi tavana çarpan
                # pozisyonlarda etkisiz kalıyordu
                max_notional = MAX_POSITION_PCT * equity * dd_factor
                if notional > max_notional:
                    qty = max_notional / cl
                    notional = qty * cl
                    risk_usd = qty * stop_dist
                if notional < MIN_ORDER_USD:
                    continue

                # v9: starter — ilk giriş yarım boy (R tanımı tam kalır, piramit tam boya çıkarır)
                if strat != "MR":
                    qty *= INITIAL_SIZE_FRAC
                    notional = qty * cl

                # v22 MAKER: PB girişleri limit emirle
                if MAKER_PB and strat == "PB":
                    if MAKER_FILL == "strict":
                        if s not in pending_limits:
                            pending_limits[s] = dict(
                                created_i=i, is_short=is_short, limit_px=cl, qty=qty,
                                stop_dist=stop_dist, risk_usd=risk_usd, atr=atr,
                                btc_reg=btc_reg,
                                rank=float(rk) if not np.isnan(rk) else float("nan"),
                                rvol=float(M["rvol"][i, j]), delta=float(M["delta_pct"][i, j]),
                                thrust_btc=float(btc_ret24[i]) if not np.isnan(btc_ret24[i]) else float("nan"),
                                disp=float(disp24[i]) if not np.isnan(disp24[i]) else float("nan"),
                            )
                        continue
                    # optimistic: sinyal barında maker dolum
                    fill = cl * (1 - MAKER_SLIPPAGE) if is_short else cl * (1 + MAKER_SLIPPAGE)
                    fee = notional * MAKER_COMMISSION
                else:
                    fill = cl * (1 + SLIPPAGE) if not is_short else cl * (1 - SLIPPAGE)
                    fee = notional * COMMISSION
                balance -= fee
                stop = fill + stop_dist if is_short else fill - stop_dist
                positions[s] = Pos(
                    sym=s, is_short=is_short, qty=qty, avg_entry=fill, stop=stop,
                    risk_usd=risk_usd, entry_price0=fill, stop_dist0=stop_dist,
                    hw=fill, strategy=strat, opened_i=i, atr_at_entry=atr,
                    realized=-fee,
                    btc_reg_e=btc_reg, rank_e=float(rk) if not np.isnan(rk) else float("nan"),
                    rvol_e=float(M["rvol"][i, j]), delta_e=float(M["delta_pct"][i, j]),
                    thrust_btc_e=float(btc_ret24[i]) if not np.isnan(btc_ret24[i]) else float("nan"),
                    disp_e=float(disp24[i]) if not np.isnan(disp24[i]) else float("nan"),
                )
                if is_short:
                    n_short += 1
                else:
                    n_long += 1

        if in_window:
            equity_curve.append((ts, equity))

    # ── Açık pozisyonları kapat (keep_open=True ise snapshot al, kapatma) ──
    last_i = n - 1
    open_snapshot = []
    if keep_open:
        for sym, p in positions.items():
            j = sym_i[sym]
            cl = M["close"][last_i, j]
            open_snapshot.append(dict(
                symbol=sym, is_short=p.is_short, qty=p.qty,
                avg_entry=p.avg_entry, stop=p.stop, strategy=p.strategy,
                last_px=float(cl) if not np.isnan(cl) else None,
                unreal=_unreal(p, cl) if not np.isnan(cl) else 0.0,
                opened_ts=str(gidx[p.opened_i]),
            ))
    else:
        for sym in list(positions.keys()):
            p = positions[sym]
            j = sym_i[sym]
            cl = M["close"][last_i, j]
            if not np.isnan(cl):
                _close_pos(p, cl, gidx[last_i], last_i, "EOT")

    # teşhis: bloklanan sinyallerin ileri getirileri (4h / 24h, işlem yönünde)
    blocked_df = None
    if blocked_log:
        _rows = []
        for b in blocked_log:
            j = sym_i[b["sym"]]
            i0 = b["i"]
            c0 = M["close"][i0, j]
            sgn = -1.0 if b["is_short"] else 1.0
            f4 = f24 = float("nan")
            if c0 > 0 and not np.isnan(c0):
                c4 = M["close"][min(i0 + 4 * K, n - 1), j]
                c24 = M["close"][min(i0 + 24 * K, n - 1), j]
                if not np.isnan(c4):
                    f4 = (c4 / c0 - 1) * sgn
                if not np.isnan(c24):
                    f24 = (c24 / c0 - 1) * sgn
            _rows.append({**b, "fwd4h": f4, "fwd24h": f24})
        blocked_df = pd.DataFrame(_rows)

    return dict(
        trades=trades, equity_curve=equity_curve, final_balance=balance,
        capital=capital, regime_counter=regime_counter, n_syms=len(syms),
        blocked=blocked_df, open_positions=open_snapshot,
    )


# ══════════════════════════════════════════════════════════════════════════════
# RAPOR
# ══════════════════════════════════════════════════════════════════════════════

def report(res: dict, start: str, end: str, show_trades: bool = False):
    trades: list[Trade] = res["trades"]
    cap = res["capital"]
    fin = res["final_balance"]
    eq = res["equity_curve"]

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gross_w = sum(t.pnl for t in wins)
    gross_l = abs(sum(t.pnl for t in losses))
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    wr = 100 * len(wins) / len(trades) if trades else 0

    max_dd = 0.0
    peak = -1e18
    for _, e in eq:
        peak = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak if peak > 0 else 0)

    print("═" * 64)
    print(f"  M9 APEX  |  {start} → {end}  |  {res['n_syms']} coin")
    print("═" * 64)
    ret = 100 * (fin - cap) / cap
    arrow = "▲" if ret >= 0 else "▼"
    print(f"  Bitiş Sermaye   : ${fin:,.2f}   {arrow} {ret:+.2f}%  (${fin-cap:+,.2f})")
    print(f"  Toplam İşlem    : {len(trades)}  |  Kazanılan: {len(wins)}  Kaybedilen: {len(losses)}")
    print(f"  Kazanma Oranı   : %{wr:.1f}  |  Profit Factor: {pf:.2f}")
    print(f"  Max Düşüş       : %{100*max_dd:.2f}")
    rc = res["regime_counter"]
    tot = sum(rc.values()) or 1
    print(f"  BTC Rejimi      : " + " | ".join(f"{k}: {100*v/tot:.1f}%" for k, v in rc.items() if v > 0))

    # Strateji kırılımı
    print("─" * 64)
    print("  STRATEJİ KIRILIMI")
    for strat in ("CS", "BO", "PB", "SQ", "MR"):
        for short in (False, True):
            sel = [t for t in trades if t.strategy == strat and t.is_short == short]
            if not sel:
                continue
            sw = [t for t in sel if t.pnl > 0]
            spnl = sum(t.pnl for t in sel)
            side = "SHORT" if short else "LONG "
            print(f"   {strat} {side}: {len(sel):4d} işlem  WR %{100*len(sw)/len(sel):4.1f}  PnL ${spnl:+9.2f}")

    # Ay bazlı PnL (hangi dönem kanıyor?)
    by_month = {}
    for t in trades:
        mk = t.exit_ts.strftime("%Y-%m")
        m0 = by_month.setdefault(mk, [0, 0.0])
        m0[0] += 1
        m0[1] += t.pnl
    if len(by_month) > 1:
        print("─" * 64)
        print("  AY BAZLI")
        for mk in sorted(by_month):
            cnt, pnl = by_month[mk]
            print(f"   {mk}: {cnt:4d} işlem  PnL ${pnl:+9.2f}")

    # Çıkış sebebi kırılımı
    by_reason = {}
    for t in trades:
        r0 = by_reason.setdefault(t.reason, [0, 0.0])
        r0[0] += 1
        r0[1] += t.pnl
    print("─" * 64)
    print("  ÇIKIŞ SEBEPLERİ")
    for rsn, (cnt, pnl) in sorted(by_reason.items(), key=lambda x: x[1][1]):
        print(f"   {rsn:12s}: {cnt:4d} işlem  PnL ${pnl:+9.2f}")

    # Coin kırılımı (en iyi/kötü 5)
    by_coin = {}
    for t in trades:
        by_coin.setdefault(t.sym, 0.0)
        by_coin[t.sym] += t.pnl
    top = sorted(by_coin.items(), key=lambda x: -x[1])
    if top:
        print("─" * 64)
        best = "  ".join(f"{s.replace('USDT','')}: ${p:+.0f}" for s, p in top[:5])
        worst = "  ".join(f"{s.replace('USDT','')}: ${p:+.0f}" for s, p in top[-5:])
        print(f"  En iyi : {best}")
        print(f"  En kötü: {worst}")

    pyr = [t for t in trades if t.adds > 0]
    if pyr:
        print(f"  Piramitli işlem : {len(pyr)}  (PnL ${sum(t.pnl for t in pyr):+,.2f})")

    if show_trades and trades:
        print("─" * 64)
        for k, t in enumerate(trades, 1):
            side = "S" if t.is_short else "L"
            print(f"  {k:4d} {t.sym.replace('USDT',''):6s} {side} {t.strategy} "
                  f"{t.entry_ts.strftime('%m/%d %H:%M')} → {t.exit_ts.strftime('%m/%d %H:%M')} "
                  f"{t.entry_px:10.4f} → {t.exit_px:10.4f}  ${t.pnl:+8.2f}  {t.reason}"
                  + (f" (+{t.adds} ekle)" if t.adds else ""))
    print("═" * 64)


def main():
    ap = argparse.ArgumentParser(description="M9 APEX backtest")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--capital", type=float, default=INITIAL_CAPITAL)
    ap.add_argument("--risk-mult", type=float, default=1.0)
    ap.add_argument("--coins", type=int, default=0, help="ilk N coin (0=hepsi)")
    ap.add_argument("--trades", action="store_true", help="işlem listesini yaz")
    ap.add_argument("--tf", default="5m", choices=["5m", "15m"], help="uygulama dilimi")
    ap.add_argument("--dump", default="", help="teşhis CSV öneki (X → X_trades.csv + X_blocked.csv)")
    args = ap.parse_args()
    set_tf(args.tf)

    syms = UNIVERSE[:args.coins] if args.coins > 0 else UNIVERSE
    if "BTCUSDT" not in syms:
        syms = ["BTCUSDT"] + syms

    t0 = time.time()
    res = run_backtest(args.start, args.end, capital=args.capital,
                       symbols=syms, risk_mult=args.risk_mult)
    report(res, args.start, args.end, show_trades=args.trades)
    if args.dump:
        from dataclasses import asdict
        pd.DataFrame([asdict(t) for t in res["trades"]]).to_csv(f"{args.dump}_trades.csv", index=False)
        if res["blocked"] is not None:
            res["blocked"].to_csv(f"{args.dump}_blocked.csv", index=False)
        print(f"  teşhis: {args.dump}_trades.csv + {args.dump}_blocked.csv yazıldı")
    print(f"  süre: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
