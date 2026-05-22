#!/bin/bash
# ============================================================
#  Kripto Bot — Canlı Panel Başlatıcı
#  Kullanım:
#    bash live/start.sh             → Dashboard aç + ilk güncelleme
#    bash live/start.sh --update    → Sadece state güncelle
#    bash live/start.sh --loop      → Saatlik döngü + dashboard
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"
source venv/bin/activate

BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
RESET='\033[0m'

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║           KRİPTO BOT — CANLI PANEL                      ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""

case "${1:-}" in
    --update)
        echo -e "${CYAN}▶ State güncelleniyor (M4 + M5)...${RESET}"
        python live/live_runner.py
        echo ""
        echo -e "${GREEN}✓ Tamamlandı.${RESET}"
        ;;

    --loop)
        echo -e "${CYAN}▶ Saatlik döngü başlatılıyor...${RESET}"
        # Dashboard arka planda
        streamlit run dashboard/app.py --server.port 8501 --server.headless true &
        DASH_PID=$!
        echo -e "${GREEN}  Dashboard: http://localhost:8501  (PID: $DASH_PID)${RESET}"
        echo ""
        # Runner ön planda (her saat)
        python live/live_runner.py --loop 3600
        ;;

    *)
        # Varsayılan: state güncelle + dashboard aç
        echo -e "${CYAN}▶ İlk state güncellemesi çalışıyor...${RESET}"
        echo -e "  (M4 + M5, 90 gün geriye — ~3-4 dakika sürer)"
        echo ""
        python live/live_runner.py
        echo ""
        echo -e "${CYAN}▶ Dashboard başlatılıyor...${RESET}"
        echo -e "${GREEN}  → Tarayıcıda aç: http://localhost:8501${RESET}"
        echo -e "${YELLOW}  (Durdurmak için: Ctrl+C)${RESET}"
        echo ""
        streamlit run dashboard/app.py --server.port 8501
        ;;
esac
