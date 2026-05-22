# BIST 100 USD Trend-Following Bot

Mevcut kripto trend-following botunun Borsa İstanbul (BIST 100) adaptasyonu.
**Sermaye USD bazlı, strateji TRY fiyatlarını USD'ye çevirerek çalışır.**

## Hızlı Başlangıç

```bash
# Bağımlılıkları kur
pip install yfinance

# Backtest çalıştır (2020–2026)
python bist/bist_main.py --mode backtest

# Özel tarih aralığı
python bist/bist_main.py --mode backtest --start 2022-01-01 --end 2026-01-01

# Paper trading (piyasa saatlerinde polling)
python bist/bist_main.py --mode paper
```

## Klasör Yapısı

```
bist/
├── config_bist.yaml          # Tüm parametreler
├── bist_main.py              # CLI giriş noktası
├── bist_backtester.py        # Backtest orkestrasyon
├── bist_paper_trader.py      # Paper trading döngüsü
├── data/
│   ├── universe.py           # BIST 30 / BIST 100 listesi
│   ├── yfinance_provider.py  # OHLCV veri çekimi (parquet cache)
│   └── usdtry_provider.py    # USD/TRY kur serisi
├── adapters/
│   ├── price_converter.py    # TRY → USD dönüşüm
│   ├── market_hours.py       # BIST piyasa saatleri
│   └── tick_lot.py           # Lot/tick boyutu + komisyon
├── filters/
│   └── macro_filter.py       # XU100 rejim + USDTRY kriz filtresi
└── tests/
    └── test_bist_components.py
```

## Strateji (Kripto Botla Aynı)

Mevcut `strategy/`, `risk/`, `indicators/`, `execution/` modülleri **doğrudan reuse edilir**.
BIST'e özgü eklemeler:
- **USD dönüşüm**: `price_converter.py` — her bar'da `price_usd = price_try / USDTRY`
- **Makro filtre**: XU100 200-day EMA (bear'da yeni long yok) + USDTRY 20-day momentum (TRY krizde yeni long yok)
- **Sektör korelasyon**: `config_bist.yaml` → `sector_correlation.clusters` (bankalar grubu vb.)

## Kripto Bot'tan Farklar

| Boyut | Kripto | BIST |
|---|---|---|
| Veri kaynağı | Binance/CCXT | yfinance (.IS) |
| Bar timeframe | 1h | 1d (önerilen) |
| Piyasa | 24/7 | Mon–Fri 10:00–18:00 TR |
| Para birimi | USDT (direkt) | TRY → USD dönüşüm |
| Lot | Fractional | Tam adet |
| Komisyon | ~%0.04 round-trip | ~%0.10–0.15 (BSMV dahil) |
| Rejim filtresi | BTC 200-day SMA | XU100 200-day EMA |
| Kriz koruması | Yok | USDTRY 20-day momentum > %15 |

## Konfigürasyon (`config_bist.yaml`)

Tüm parametreler `bist/config_bist.yaml` dosyasında. Önemli bölümler:

```yaml
data:
  usd_mode: "convert_series"  # OHLCV USD'ye çevrilir (önerilen)

macro:
  regime_filter_enabled: true      # XU100 bear'da giriş engeli
  usdtry_guard_enabled: true       # TRY krizde giriş engeli
  usdtry_crisis_threshold: 0.15    # 20g momentum > %15 = kriz

sector_correlation:
  max_positions_per_cluster: 2     # Aynı sektörden max 2 pozisyon
```

## Testleri Çalıştır

```bash
# Proje kökünden
pytest bist/tests/ -v
```

Testler internet bağlantısı gerektirmez (yfinance mock'lanmaz, unit test).

## Canlı / Real-Time Trading

`bist/live/` paketi gerçek zamanlı veri altyapısını hazırlar:

```bash
# Daily bar ile live runner (yfinance polling, 15 dakikada bir)
python bist/bist_main.py --mode live

# 5 dakikalık bar ile (her 5 dakikada kontrol)
python bist/bist_main.py --mode live --interval 5m --poll 300
```

### Broker API Entegrasyonu

Gerçek canlı emir göndermek için `StreamingDataSource` sub-class'ı yeterli:

```python
from bist.live.data_source import StreamingDataSource
from bist.live.live_runner import BistLiveRunner

class MatriksSource(StreamingDataSource):
    def get_history(self, symbol, bars=300):
        return matriks_api.get_ohlcv(symbol, bars)

    def _connect_to_broker(self):
        matriks_api.on_bar = lambda raw: self._emit_bar({
            "symbol": raw["ticker"], "timestamp": pd.Timestamp(raw["ts"], tz="UTC"),
            "open": raw["o"], "high": raw["h"], "low": raw["l"],
            "close": raw["c"], "volume": raw["v"],
        })
        matriks_api.subscribe(self._symbols)

cfg = yaml.safe_load(open("bist/config_bist.yaml"))
runner = BistLiveRunner(cfg, data_source=MatriksSource(), live_mode=True)
runner.start()
```

### Mimari

```
Veri Kaynağı (pull/push)          BistLiveRunner             Çıktı
─────────────────────────          ────────────────           ──────
YFinancePollingSource ──on_bar()──▶ Makro filtre    ──────▶  Pozisyon
StreamingDataSource   ──on_bar()──▶ İndikatörler   ──────▶  Risk
MatriksSource         ──on_bar()──▶ Sinyal üretimi ──────▶  Telegram
                                  ▶ Stop kontrolü  ──────▶  (Broker emri)
```

## Notlar

- **yfinance intraday sınırı**: 1h veri için max ~730 gün. Daily için 10+ yıl.
- **T+2 settlement**: Backtest'te göz ardı edilir; canlıda broker halleder.
- **Açığa satış**: Strateji long-only (BIST kısıtı uyumlu).
- **Live emir (stub)**: `_send_live_order()` şu an log yazdırır. Broker API gelince doldurulur.
