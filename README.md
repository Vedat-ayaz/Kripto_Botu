# Kripto Trend-Following Bot

Risk kontrollü, modüler bir Python kripto ticaret botu. Spot piyasa için tasarlanmıştır.

> ⚠️ **YASAL UYARI**: Bu yazılım yatırım tavsiyesi değildir. Kripto para ticareti ciddi finansal kayıplara yol açabilir. Gerçek para kullanmadan önce tam olarak anlayın ve kabul edin.

---

## Strateji Mantığı

EMA crossover tabanlı trend-following stratejisi.

**Long Giriş — Tüm koşullar sağlanmalı:**
1. `close > EMA200` — Fiyat uzun vadeli trendin üzerinde
2. `EMA50 > EMA200` — Kısa vade uzun vadenin üzerinde (golden cross bölgesi)
3. `ADX > 20` — Trend gücü yeterli
4. `RSI [45-70]` — Aşırı alım/satım yok, momentum pozitif
5. `volume > volume_sma20` — Hacim ortalamanın üzerinde
6. `ATR/close > 0.002` — Yeterli volatilite var

**Çıkış Koşulları (herhangi biri):**
- `close < EMA50` — Trend zayıfladı
- ATR tabanlı stop-loss tetiklendi
- Trailing stop tetiklendi
- Günlük %3 zarar limitine ulaşıldı

---

## Risk Yönetimi

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|---------|
| `risk_per_trade` | %1 | İşlem başı maksimum risk |
| `daily_max_loss` | %3 | Günlük maksimum zarar |
| `max_open_positions` | 3 | Eş zamanlı maksimum pozisyon |
| `atr_stop_multiplier` | 2.0 | Stop = entry - ATR × 2.0 |
| `trailing_stop_atr_multiplier` | 2.5 | Trailing = current - ATR × 2.5 |

**Pozisyon Büyüklüğü Formülü:**
```
risk_amount   = bakiye × risk_per_trade
stop_distance = ATR × atr_stop_multiplier
position_size = risk_amount / stop_distance
```

---

## Kurulum

### 1. Python ortamı

```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# veya
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 2. Ortam değişkenleri

```bash
cp .env.example .env
# .env dosyasını düzenle
```

### 3. Config

`config.yaml` dosyasını ihtiyacınıza göre düzenleyin.

---

## .env Örneği

```env
EXCHANGE_API_KEY=your_api_key_here
EXCHANGE_API_SECRET=your_api_secret_here
TESTNET_API_KEY=your_testnet_key
TESTNET_API_SECRET=your_testnet_secret
TELEGRAM_BOT_TOKEN=optional
TELEGRAM_CHAT_ID=optional
```

**ÖNEMLİ**: API key oluştururken **sadece Spot Trade izni** verin. Withdrawal izni **KESİNLİKLE** vermeyin.

---

## config.yaml Açıklaması

```yaml
exchange:
  name: "binance"       # Desteklenen: binance, bybit, okx, vb.
  testnet: true         # false = gerçek API

trading:
  symbols: ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
  timeframe: "1h"       # 1h veya 4h

risk:
  risk_per_trade: 0.01  # İşlem başı sermayenin %1'i
  daily_max_loss: 0.03  # Günlük sermayenin %3'ü

live:
  enabled: false        # TRUE YAPMADAN GERÇEK EMIR GİTMEZ
```

---

## Backtest Nasıl Çalıştırılır?

```bash
python main.py --mode backtest
```

- Borsadan son 1000 mum çeker
- Stratejiyi bar-by-bar çalıştırır
- Komisyon (%0.1) ve slippage (%0.05) dahil edilir
- Terminal çıktısı + `backtest_results/` klasörüne CSV kaydeder

---

## Paper Trading Nasıl Çalıştırılır?

```bash
python main.py --mode paper
```

- Gerçek zamanlı fiyat çeker
- Sanal bakiye ile simülasyon yapar
- **Gerçek emir göndermez**
- Ctrl+C ile güvenli kapatma

---

## Live Mod Nasıl Aktif Edilir?

1. `config.yaml` içinde `live.enabled: true` yapın
2. `.env` içine gerçek API bilgilerinizi girin
3. API key'de **sadece Spot Trade** izni olsun
4. `python main.py --mode live` çalıştırın
5. Terminal'de "EVET" yazarak onaylayın

```bash
python main.py --mode live
```

---

## Testleri Çalıştır

```bash
pytest tests/ -v
```

---

## Güvenlik Uyarıları

- API key'e Withdrawal izni verme
- `.env` dosyasını git'e commit etme (`.gitignore`'a ekle)
- `live.enabled: false` varsayılan olarak kalmalı
- Bot bir güvence veya kâr garantisi vermez
- Küçük sermaye ile başla ve sık izle

---

## Proje Yapısı

```
crypto-trend-bot/
├── main.py                         # Giriş noktası
├── config.yaml                     # Tüm parametreler
├── .env.example                    # API key şablonu
├── data/                           # Borsa bağlantısı ve veri yönetimi
├── indicators/                     # EMA, RSI, ATR, ADX, Volume SMA
├── strategy/                       # Sinyal üretimi
├── risk/                           # Risk kontrolleri, pozisyon boyutu
├── execution/                      # Emir ve pozisyon yönetimi
├── backtest/                       # Geçmiş test + metrikler
├── paper/                          # Paper trading döngüsü
├── monitoring/                     # Loglama + Telegram
└── tests/                          # Unit testler
```

---

*Bu bot yatırım tavsiyesi içermez. Tüm sorumluluk kullanıcıya aittir.*
