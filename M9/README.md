# M9 "APEX" — İtki-Kapılı Çift Yönlü Trend Modeli

## Özet
5 dakikalık işlem + 1 saatlik rejim bağlamı. LONG + SHORT tam simetrik.
Order-flow analizi (taker buy hacmi → delta/CVD), kesitsel momentum rotasyonu,
piramitleme ve **piyasa itkisi (thrust) kapısı**: piyasa gerçekten hareket etmiyorsa
işlem açılmaz. Bağımsız model — M1/M5/M7/M8 koduna dokunmaz. GitHub'a pushlanmadı.

## Final Mimari (config D = v8 + thrust + frenler + kademeli DD valisi)

```
Binance klines API (5m + 1h, taker_buy dahil) → M9/.cache/
        │
        ├── Order-flow: delta_pct, CVD, RVOL (24h medyan)
        ├── 1h Rejim: ER + Choppiness + ADX → TREND_UP / TREND_DOWN / RANGE / SQUEEZE
        ├── Kesitsel momentum: 4h/24h/72h vol-ayarlı getiri → saatlik rank
        ├── THRUST: |BTC 24h getiri| ≥ %2 VEYA coinler arası 24h getiri std ≥ %3
        │
        ▼
   Sinyaller (kenar-tetikli, lookahead'siz)
        ├── BO: Donchian(8h) kırılımı + RVOL≥1.3 + delta + VWAP onayı
        └── PB: EMA20 taze dokunuş + flow onayı (ana motor)
        │
        ▼
   Giriş kapıları (hepsi geçilmeli)
        ├── THRUST aktif (yoksa hiç işlem yok — chop = komisyon kanaması)
        ├── BTC 1h rejimi: long için TREND_UP, short için TREND_DOWN
        │   (RANGE/SQUEEZE → işlem YOK; v19'da RANGE-modu kanıtla kapatıldı)
        ├── xsec rank: long top-10, short bottom-10
        ├── cooldown: çıkış 1h, zarar 36h | sembol duraklatma: 5 işlemin 4'ü zarar → 4 gün
        ├── günlük kill-switch −%3 | haftalık fren: 7g'de −%6 → 24h dur
        │
        ▼
   Pozisyon yönetimi
        ├── Boyut: %1.75 risk × vol-target [0.5-1.5] × DD valisi
        │   DD valisi kademeli: >%8 → ×0.6 | >%15 → ×0.35 | >%20 → ×0.2
        ├── Stop: max(1.8×ATR_1h, %1.2) | chandelier 2.5×ATR_1h, +0.8R'den sonra
        ├── Piramit: +1R → %60 ekle (stop BE+0.1R), +2.2R → %40 ekle (stop +1R)
        ├── Kısmi kâr: +1.5R'de %25 | CVD diverjansında %50
        └── Stagnasyon: 8h sonunda negatifse çık | rejim dönüşünde kârlıysa çık
```

## Test Sonuçları ($10,000, komisyon %0.1 + slippage %0.05/taraf — ana repo ile aynı)

| Pencere | M9 APEX (v22 final) | M1 | M5 | M7 | M8 |
|---------|---------|-----|-----|-----|-----|
| Son test (27May→9Haz, 13g) | **+25.30%** PF 5.02 DD %5.7 | -0.12% | +0.46% | — | — |
| Ayı haftası (29Oca→5Şub) | **+6.71%** PF 1.36 DD %10.4 | 0.00% | +0.25% | +0.16% | +0.16% |
| Boğa haftası (27Eyl→4Eki) | **+0.41%** PF 1.11 DD %4.0 | +2.24% | +1.49% | +0.09% | +0.09% |
| 6 ay (Ağu→Oca) | **-17.68%** DD %22.4 | -0.56% | -3.16% | +0.29% | -0.34% |
| OOS (5Şub→26May, optimize edilmedi) | **-9.20%** DD %25.6 | — | — | — | — |

### v22: yeni bilgi kaynağı + maliyet modeli (KABUL — varsayılan)
- **Maker modu (PB girişleri limit emir, M7 emsali):** giriş maliyeti %0.15 → %0.05.
  DÜRÜST dolum modeliyle test edildi ("strict": limit 3 bar içinde dolmazsa işlem iptal;
  dolum oranı ~%97). Tek başına 4/5 pencereyi iyileştirdi. Canlı port limit emir desteği ister.
- **Funding kapısı (Binance perp funding):** aşırı negatif funding'de short / aşırı pozitifte
  long açılmaz (kalabalığa uçta katılma; eşik ±%0.05/8h, duyarlılık 0.03-0.10 tarandı).
- Kombo: SON13G +22.38→+25.30, AYI +5.62→+6.71, BOGA +0.17→+0.41, OOS -10.65→-9.20;
  bedel 6AY -16.19→-17.68 (strict modelde dolmayan limitler = anında koşan girişler,
  kazananın kıt olduğu pencerede acıtıyor). Kaybeden pencereler net ≈ başa baş, kazananlar +4.2.
- **OI (open interest):** Binance geçmişi 30 günle sınırlı → backtest'te KULLANILAMAZ;
  canlı motora port edilirken canlı-katman filtresi olarak eklenebilir.

### v20/v21 teşhis kampanyası (ajan-destekli, 35 konfigürasyon)
Trade-düzeyi MFE/MAE + bloklanan-sinyal enstrümantasyonu eklendi (`--dump` ile CSV).
İki ajan (veri analizi + literatür) kayıp/kaçan-kâr paternlerini çıkardı; aday düzeltmeler
izole+kümülatif ablasyondan geçirildi (`ablation_v20*.py`):
- **KABUL — K5 ideal-kilit** (yön BTC trendiyle uyumlu + |BTC24h|≥%2 + disp≥%3 → rank
  kapısı atlanır, zarar-cooldown yarıya): SON13G +20.81→+22.38, kayıp pencereler ≈ nötr.
- **KABUL — yapışkan DD valisi** (tetiklenen kademe DD<%4'e dek kalır): 6AY -16.66→-16.19,
  impuls pencerelerinde matematiksel olarak etkisiz (DD eşiğe ulaşmıyor).
- **RED — disp tabanı/ölçekleyici/niteleyici, makro çapa**: dispersiyon GECİKMELİ ölçü;
  kaskad başlangıcını blokluyor (AYI +6.5→-6.2/-8.6 kırıldı).
- **RED — V-dönüş kapısı**: kaskad içi re-entry'leri kesiyor (AYI -7.8).
- **RED — BE-stop 0.5R**: küçük kaybedenleri kurtarırken nadir büyük kazananları
  öldürüyor (SON13G -5.5) — MFE üst-sınır analizi yol-bağımlılığı yakalayamaz.

**6AY kanamasının kök teşhisi (kanıtlı):** kapılar DOĞRU çalışıyor (bloklanan sinyallerin
ileri getirisi negatif); kanama kapıları GEÇEN işlemlerden geliyor ve bunlar giriş-anı
özellikleriyle (disp/rvol/delta/rank/thrust) kazananlardan ayırt EDİLEMİYOR. Kayıp kronik
(79 günün 60'ı negatif), tek şok değil. Eldeki özellik uzayı tükendi — sonraki adım yeni
bilgi kaynağı (funding rate, open interest, derinlik) veya maker-emir maliyet modeli.

**Dürüst değerlendirme:** M9 yönlü/itkili piyasalarda eski modelleri ezici farkla geçiyor
(13 günde +21% vs M5'in +0.5%), ama uzun yatay/chop dönemlerde negatif. Bu yapısal:
5m işlem + taker komisyonu, kenarda durmayı bilmek zorunda; thrust kapısı bunu kısmen
çözüyor ama 6 aylık karma dönemde tam koruma sağlamıyor. Kullanım önerisi: M5 (her
mevsim, düşük frekans) yanında uydu model olarak, küçük sermaye payıyla.

## Denenen ve VERİYLE REDDEDİLEN fikirler (geliştirme günlüğü)
| Fikir | Sonuç | Versiyon |
|-------|-------|----------|
| 5m ATR stoplar | komisyondan küçük stop → PF 0.33 | v1→v2 |
| Seviye-tetik sinyaller | 424 işlem/13g churn → kenar-tetik şart | v1→v2 |
| Squeeze (SQ) motoru | iki yönde de zarar (78 işlem −$956) | v4 |
| Mean-reversion (MR) motoru | iki yönde de zarar (59 işlem −$475) | v5 |
| Uzama filtresi (4×ATR) | impulsta uzama = güç; kazananları kesti | v7 |
| Starter sizing (yarım giriş) | ayı kaskad kârını yarıladı | v9/E |
| 15m'e geçiş | haftalık impuls kârlarını bozdu | v10 |
| Kompozit skor (CS) girişi | eşik-kesme tetikleri churn'e döndü (WR %18.5) | v11 |
| Yönlü thrust | 24h gecikme dönüm noktasında ters | v14/F |
| Makro yön katmanı (30g EMA) | ayıyı +24'e çıkardı ama boğa+6ay'ı bozdu | v15-18/H |

## Çalıştırma
```bash
./backtest.sh 2026-05-27 2026-06-09          # tarih aralığı
./backtest.sh 2026-05-27 2026-06-09 trades   # işlem listesiyle
python3 m9_backtest.py --start ... --end ... --tf 15m   # alternatif dilim
python3 matrix_test.py                        # konfigürasyon matrisi (A-H)
```

## Bilinen Sınırlar
- Uzun chop dönemlerinde negatif (yukarıda) — canlıya alınacaksa M5 ile portföy olarak
- Sinyal bar kapanışında, dolum aynı bar kapanış + slippage (canlı: sonraki bar açılışı)
- LEO evrende yok (Binance'te yok); BTC eklendi → 30 coin
- Canlı motor portu YOK (yalnızca backtest) — istenirse ayrı iş
