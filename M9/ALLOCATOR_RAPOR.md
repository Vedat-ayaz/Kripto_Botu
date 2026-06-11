# M5/M7 + M9 Rejim Tahsisçisi ("Ortaklık") — Sonuç Raporu
*11 Haziran 2026 — allocator.py ile üretildi, GitHub'a pushlanmadı*

## Mimari
- **Taban sleeve:** M5 veya M7 (her-mevsim, düşük frekans) — sermaye varsayılan olarak burada
- **Uydu sleeve:** M9 v22 (itki uzmanı)
- **Sinyal (lookahead'siz, günlük):** `m9_aktif = BTC 1h rejimi TREND (UP/DOWN) VE thrust
  (|BTC24h|≥%2 veya kesitsel disp≥%3)` — gün kapanışında karar, ertesi gün uygulanır
- **E-kapısı (fund-of-funds):** M9 ancak kendi son-14g getirisi > -%2 iken sermaye alır
  (M9 shadow'da sürekli koşar; kanayan yöneticiye tahsis yok)

## Sonuçlar (günlük eğrilerden, $10k, tahsis gün sınırında maliyetsiz varsayımı)

### Pencere bazlı (pencere-içi taze sermaye)
| Pencere | M5 | M7 | M9 | **E(M5)** | E(M7) |
|---|---|---|---|---|---|
| Son test 13g | +0.49 | -0.13 | +25.30 | **+26.96** | +20.45 |
| Ayı haftası | +3.61 | +1.98 | +6.80 | **+11.07** | +8.63 |
| Boğa haftası | +2.32 | +0.96 | +0.41 | +1.85 | +1.14 |
| 6AY | -7.84 | **+1.89** | -17.66 | -14.90 | -10.00 |
| OOS | -2.76 | -5.10 | -9.20 | **+4.20** | -0.62 |

### Tam dönem kesintisiz (1 Ağu 2025 → 9 Haz 2026, 313 gün)
| | M5 | M7 | M9 | E(M5) | E(M7) |
|---|---|---|---|---|---|
| Getiri | -14.79% | **-5.67%** | -23.93% | -11.01% | -12.26% |
| Max DD | %18.7 | — | %29.2 | %20.3 | %19.7 |
| Sharpe | -0.89 | — | -1.66 | **-0.58** | -1.28 |

## Bulgular
1. **Tahsis alfası gerçek:** E(M5) tam dönemde iki bileşenden de iyi (+3.8/+12.9 puan);
   OOS'ta iki kaybeden modelden +4.2% çıkardı; itki pencerelerinde bileşenleri katladı.
2. **M7 en iyi taban:** tam dönem -5.67 ile tüm tekil modellerin en iyisi; 6AY'da tek pozitif.
3. **Hiçbir kombinasyon tam dönemi pozitife çeviremedi** — Ağu→Haz dönemi eldeki TÜM
   motorlara düşmandı; tahsisçi kaybı %25-60 azaltır ama negatif bileşenlerden pozitif üretemez.
4. **Değerlendirme belirsizliği:** pencere-bazlı (her aktivasyonda taze sermaye → +12.6%
   zincirleme tahmin) ile kesintisiz-shadow (-11%) arasındaki fark, M9 sleeve'inin DD valisi
   durumunun aktivasyonlar arasında taşınıp taşınmamasına bağlı. Gerçek tahsisçi her
   aktivasyonda NAV yüzdesi olarak taze sermaye verir → gerçek değer ikisinin arasında.
5. **Denendi-reddedildi:** nakit üçüncü sleeve (her eşikte daha kötü — equity-curve-trading
   tuzağı: M5'in yavaş kanaması ortalamaya döner, kapı dibi satar); çapa-uzaklık niteleyicisi
   (6AY'ı ayıramadı); pozitif-kanıt kapısı ≥0/≥+1% (M9'u fiilen kapatıyor, M7-yalnızdan kötü).

## v23 GÜNCELLEME (11 Haz 2026) — Teyit kuralı kampanyası
Ajan-destekli gün-düzeyi analiz kök nedeni buldu: **uzun dönem zararının tamamı tek-günlük
sinyal episodlarından** (47 episodun 35'i tek gün; toplam -25.2p, WR %11; çok-günlüler net
pozitif; split-sample TUTARLI). ER/korelasyon/breadth gün-filtreleri (literatür önerileri)
bizim veride OOS'ta kâr blokladığı için REDDEDİLDİ.

**T2 kuralı (FİNAL):** M9'a tahsis ancak `bugün sinyal açık VE son 2 günün en az birinde
açıktı` (+ 14g sağlık kapısı). İzole tek-günleri bloklar, episod-içi titremeyi tolere eder.

| (taban=M7) | M7 | eski E | **T2** |
|---|---|---|---|
| Son test 13g | -0.13 | +20.45 | **+20.06** |
| Ayı | +1.98 | +8.63 | **+8.72** |
| 6AY | +1.89 | -10.00 | **-2.52** |
| OOS | -5.10 | -0.62 | -6.93 |
| **Tam dönem** | -5.67 | -12.26 | **-5.31** — ilk kez tahsisçi en iyi tekil modeli geçti |

**Episod simülasyonu dersi (kritik):** taze-sermaye stop/start episodları KAYBEDER (-15.3%;
teyit gecikmesi + taze giriş = hareketin ilk bacağı kaçar, dönüş yenir; Şubat episodu kanıtı:
günlük-karışımdaki +14.7'lik gün Ocak'ta açılmış pozisyonların devamıydı). Doğru üretim
tasarımı **defter aynalama**: M9 paper'da kesintisiz koşar, tahsisçi aktive olunca gerçek
hesap M9'un MEVCUT defterini kopyalar — günlük-karışım matematiğine denk, gerçekleştirilebilir.

## v24 FİNAL (11 Haz 2026) — Tutarlılık kampanyası: "her pencere makul olsun"
Kayan-pencere teşhisi kullanıcı şikâyetini doğruladı (T2'de 13g pencerelerin sadece %33'ü
pozitif, medyan -0.32). İki yapısal değişiklikle çözüldü:
1. **Shadow M9 valisiz koşar** (GOV_STICKY=False, DD_STAGES=[]): sleeve mimarisinde risk
   tahsisçide; vali Haziran-tipi rallileri küçük boyla geçiriyordu. Tek başına -55.8 ama
   T2 sadece teyitli itki günlerinde (%9) tahsis ediyor → o günler tam boy.
2. **Taban model YOK — nakit taban**: M7 tabanı -4 puan sürüklüyordu. Sermaye normalde
   nakit/stablecoin (gerçekte +%4-8 APY bonus), sadece T2 günlerinde M9 defterini aynalar.

| Tam dönem (Ağu→Haz) | T2(M7, valili) | T2(M7, valisiz) | **T2(NAKİT, valisiz) FİNAL** |
|---|---|---|---|
| Getiri | -5.31% | +10.20% | **+14.32%** |
| MaxDD | %12.0 | %9.6 | **%4.4** |
| Sharpe | -0.59 | 0.84 | **1.30** |
| 13g pencere dağılımı | %33 poz, med -0.32 | %29 poz, med -0.28 | **%88 sıfır, %7 poz, med 0.00, enKötü -4.4, enİyi +17.7** |

Kullanıcı aralığı (11May→11Haz): **+19.86%, MaxDD %0.0, Sharpe 6.09.**
Denendi-RED: carry sleeve (delta-nötr funding hasadı) — dönemin ortalama funding'i negatif
(-1.5%/yıl, %34 pozitif dönem), mükemmel öngörüyle bile üst sınır +10.5 ve churn yer.
Geçiş maliyeti notu: episod başına defter-aynalama ~%0.3 × ~10 episod ≈ -3 puan → net ~+11.
Şerh: vali/taban seçimleri ilkesel ama aynı örneklemde doğrulandı — paper-trade şart.

## Üretim önerisi
- **Mimari: M7 taban + E-kapılı M9 uydu** (pencere kanıtı; tam-dönem belirsizliği
  paper-trade ile çözülmeli)
- Canlıya giden yol: (1) M9'un live-engine portu (limit emir + funding verisi gerekir),
  (2) dashboard'a tahsisçi katmanı, (3) ≥1 ay paper-trade — pencere-bazlı vs
  kesintisiz-shadow tartışmasını gerçek davranış kapatır.
- Her aktivasyon episodunda M9'a taze NAV yüzdesi verilmeli (model-içi DD valisi
  episod başında sıfırlanır) — backtest kanıtı bunun kritik olduğunu gösteriyor.
