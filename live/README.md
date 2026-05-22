# Kripto Bot — Canlı Panel

## Hızlı Başlangıç (Lokal)

```bash
cd ~/Desktop/Projeler/Kripto_Robotu
bash live/start.sh
```

Dashboard açılır: **http://localhost:8501**  
**"⚡ M4 / M5 CANLI"** sekmesine tıkla.

---

## Komutlar

| Komut | Açıklama |
|-------|----------|
| `bash live/start.sh` | State güncelle + Dashboard aç |
| `bash live/start.sh --update` | Sadece M4/M5 state güncelle |
| `bash live/start.sh --loop` | Saatlik döngü + dashboard |
| `python live/live_runner.py --start 2025-01-01` | Özel başlangıç tarihi |

---

## Sunucu Kurulumu

### 1. Bağımlılıkları kur
```bash
pip install -r requirements.txt
```

### 2. İlk çalıştırma
```bash
python live/live_runner.py --start 2025-01-01
```

### 3. Dashboard servisi (systemd)
```ini
# /etc/systemd/system/kripto-dashboard.service
[Unit]
Description=Kripto Bot Dashboard
After=network.target

[Service]
WorkingDirectory=/srv/kripto
ExecStart=/srv/kripto/venv/bin/streamlit run dashboard/app.py --server.port 8501 --server.headless true
Restart=always
User=ubuntu

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable kripto-dashboard
sudo systemctl start kripto-dashboard
```

### 4. Saatlik güncelleme (cron)
```bash
crontab -e
# Şunu ekle:
0 * * * * cd /srv/kripto && venv/bin/python live/live_runner.py >> logs/live.log 2>&1
```

---

## Dashboard Sekmeleri

| Sekme | İçerik |
|-------|--------|
| ⚡ M4/M5 CANLI | **Ana izleme ekranı** — M4 ve M5 yan yana |
| 📊 Scanner | Açık pozisyonlar, market tarama |
| 📈 Equity | Sermaye eğrisi, performans grafikleri |
| 🔔 Sinyal | Son sinyaller |
| 🧮 Analitik | Detaylı istatistikler |

---

## M4/M5 Canlı Sekmesi — Ne Gösterir?

- **Özet metrikler**: Sermaye, PnL%, Max Drawdown, Kazanma Oranı
- **Açık Pozisyonlar**: Coin, giriş tarihi/fiyatı, son fiyat, değişim%, unrealized PnL, stop seviyesi
- **Kapalı İşlemler**: Son 50 trade, PnL$, PnL%, çıkış nedeni
- **Karşılaştırma tablosu**: M4 vs M5 yan yana

---

## API Alınca Ne Değişir?

`paper/paper_trader.py` → gerçek emirler için hazır.  
`live/live_runner.py` → şimdilik backtest tabanlı (paper mode).  
API gelince sadece `ExchangeClient` bağlanacak, strateji kodu değişmeyecek.
