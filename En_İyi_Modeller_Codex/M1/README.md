# M1 Model — Kripto Trend-Following Bot

## Özet
Temel trend-following modeli. Saf, sade, en iyi test sonuçlarını bu dönemde veren model.

## Performans (Test Sonuçları)
| Dönem | Sermaye | Kazanç | Win Rate | Profit Factor |
|-------|---------|--------|----------|---------------|
| 51 gün (May-Jun 2026) | $10,000 | **+%22.03** | %48.5 | 4.86 |
| 6 ay (Ara 2025 – Haz 2026) | $10,000 | **+%16.12** | %39.3 | 1.78 |

## Özellikler
- **Timeframe:** 1h (saatlik bar)
- **Coin sayısı:** 21 (9 sabit + 12 universe)
- **Pyramid:** YOK
- **WFO:** YOK
- **Scalping:** YOK
- **Strateji:** EMA200 + ADX + RSI + MACD + Bollinger Bands
- **Çıkış:** Trailing Stop + Strategy Exit (SE) + Stop Loss

## Çalıştırma
```bash
# Son 30 gün
./backtest.sh 30

# Belirli tarih aralığı
./backtest.sh 2026-01-01 2026-06-01

# Bugünden itibaren
./backtest.sh 2025-12-01

# Sonucu kaydet
./backtest.sh 180 kaydet
```

## Yapılan Geliştirmeler (v39/v40)
- **v39:** ADX-aware re-entry cooldown (güçlü trendde 96h→6h)
- **v40:** RSI threshold stratifikasyonu (boğa: 90, normal: 85, nötr: 82)
- **Adaptif Profil:** Her coin için pasif öğrenme altyapısı (AdaptiveCoinProfile)
- **SHORT Pyramid:** Altyapı hazır, varsayılan kapalı (`SHORT_PYR=1` ile aç)

## Dosya Yapısı
```
M1/
├── crypto_portfolio_test.py   ← Ana backtest motoru
├── strategy/                  ← Sinyal üretimi, adaptif profil
│   ├── trend_following_strategy.py
│   ├── adaptive_coin_profile.py
│   ├── adaptive_regime.py
│   └── coin_analyzer.py
├── indicators/                ← Teknik indikatörler
├── risk/                      ← Korelasyon ve risk yönetimi
├── backtest.sh               ← Test scripti
└── requirements.txt
```
