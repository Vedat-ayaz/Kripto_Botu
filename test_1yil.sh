#!/bin/bash
# =====================================================================
#  3 Model — 1 Yıllık Backtest
#  Çalıştır: bash test_1yil.sh
# =====================================================================

PYTHON="./venv/bin/python"
LOG_DIR="logs/backtest_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOG_DIR"

echo ""
echo "======================================================================"
echo "  3 MODEL — 1 YILLIK BACKTEST"
echo "  Sermaye: \$1000 | Süre: son 365 gün | TF: 15m (M4/M5) + 5m (M6)"
echo "======================================================================"
echo ""
echo "Loglar kaydediliyor: $LOG_DIR/"
echo ""

# ── M4: Stabil Referans ──────────────────────────────────────────────
echo "🔵 [1/3] M4 başlıyor... (3-5 dk sürebilir)"
$PYTHON crypto_portfolio_test.py \
    --m4 \
    --days 365 \
    --capital 1000 \
    --universe \
    --label "M4 Stabil - Son 1 Yil" \
    2>/dev/null | tee "$LOG_DIR/m4.txt"

echo ""
echo "──────────────────────────────────────────────────────────────────────"
echo "🟣 [2/3] M5 başlıyor... (3-5 dk sürebilir)"
$PYTHON crypto_portfolio_test.py \
    --m5 \
    --days 365 \
    --capital 1000 \
    --universe \
    --label "M5 Risk Dengeli - Son 1 Yil" \
    2>/dev/null | tee "$LOG_DIR/m5.txt"

echo ""
echo "──────────────────────────────────────────────────────────────────────"
echo "🟢 [3/3] M6 başlıyor... (3-5 dk sürebilir)"
$PYTHON crypto_portfolio_test.py \
    --m6 \
    --days 365 \
    --capital 1000 \
    --label "M6 Agresif - Son 1 Yil" \
    2>/dev/null | tee "$LOG_DIR/m6.txt"

echo ""
echo "======================================================================"
echo "  ÖZET"
echo "======================================================================"
echo ""

for model in M4 M5 M6; do
    file="$LOG_DIR/$(echo $model | tr '[:upper:]' '[:lower:]').txt"
    if [ -f "$file" ]; then
        echo "--- $model ---"
        grep -E "Son Bakiye|Toplam Getiri|Win Rate|Profit Factor|Max Drawdown|Toplam İşlem|SHORT|LONG" "$file" | head -10
        echo ""
    fi
done

echo "Tüm loglar: $LOG_DIR/"
echo "======================================================================"
