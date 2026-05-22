#!/bin/bash
# ============================================================
#  1 HAFTALIK TEST BAŞLATICI
#  Bu scripti sadece TEK KEZ çalıştır — testi sıfırlar ve başlatır.
#  Kullanım: bash live/start_test.sh
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"
source venv/bin/activate 2>/dev/null || true

BOLD='\033[1m'; GREEN='\033[0;32m'; RED='\033[0;31m'; CYAN='\033[0;36m'; RESET='\033[0m'

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║           1 HAFTALIK BOT TESTİ — BAŞLATILIYOR           ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  ${RED}⚠️  Bu komut tüm önceki test verilerini silecek!${RESET}"
echo ""
printf "  Sermaye ($, varsayılan=1000): "
read -r capital
capital=${capital:-1000}

printf "  Emin misin? (evet/hayır): "
read -r confirm
if [[ "$confirm" != "evet" ]]; then
    echo -e "  ${RED}İptal edildi.${RESET}"
    exit 1
fi

echo ""
echo -e "${CYAN}▶ Sıfırlanıyor ve başlatılıyor...${RESET}"
python live/live_runner.py --fresh --capital "$capital"

echo ""
echo -e "${GREEN}✅ Test başlatıldı!${RESET}"
echo -e "  Sermaye   : \$$capital"
echo -e "  Başlangıç : $(date +%Y-%m-%d)"
echo ""
echo -e "  Dashboard : streamlit run dashboard/app.py"
echo -e "  Güncelle  : python live/live_runner.py"
echo -e "  Döngü     : python live/live_runner.py --loop 3600"
