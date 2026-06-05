#!/bin/bash
# ════════════════════════════════════════════════════════════════════
#  M1 Model — Backtest Script
#  ──────────────────────────────────────────────────────────────────
#  M1: Temel trend-following model
#      1h timeframe, 21 coin, sabit profil parametreleri
#      Pyramid YOK, WFO YOK — saf strateji
#
#  KULLANIM:
#    ./backtest.sh 2025-01-06 2025-06-30    → tarih aralığı
#    ./backtest.sh 2025-01-06               → başlangıç → bugün
#    ./backtest.sh 30                       → son 30 gün
#    ./backtest.sh 180                      → son 180 gün
#    ./backtest.sh 30 kaydet                → dosyaya kaydet
# ════════════════════════════════════════════════════════════════════

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── venv ─────────────────────────────────────────────────────────
VENV_DIRS=(".venv" "../../venv" "../../../venv" "venv")
ACTIVATED=0
for V in "${VENV_DIRS[@]}"; do
  if [[ -f "$V/bin/activate" ]]; then
    source "$V/bin/activate"
    ACTIVATED=1
    break
  fi
done
if [[ $ACTIVATED -eq 0 ]]; then
  echo "❌  venv bulunamadı."
  exit 1
fi

# ── Renkler ──────────────────────────────────────────────────────
BOLD='\033[1m'; CYAN='\033[0;36m'; BLUE='\033[0;34m'
DIM='\033[2m';  GREEN='\033[0;32m'; RED='\033[0;31m'
YELLOW='\033[1;33m'; RESET='\033[0m'

# ── Argüman ayrıştır ─────────────────────────────────────────────
SAVE_FLAG=""
CMD_FLAGS=""
DONEM_LABEL=""
DATE_ARGS=()

for arg in "$@"; do
  if [[ "$arg" == "kaydet" ]] || [[ "$arg" == "-k" ]]; then
    SAVE_FLAG="evet"
  else
    DATE_ARGS+=("$arg")
  fi
done

ARG1="${DATE_ARGS[0]:-}"
ARG2="${DATE_ARGS[1]:-}"

if [[ "$ARG1" =~ ^[0-9]+$ ]]; then
  CMD_FLAGS="--days $ARG1"
  DONEM_LABEL="Son $ARG1 Gün"
else
  if ! [[ "$ARG1" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo -e "${RED}❌  Geçersiz tarih: '$ARG1'  (YYYY-MM-DD gerekli)${RESET}"
    exit 1
  fi
  CMD_FLAGS="--start $ARG1"
  DONEM_LABEL="$ARG1 → bugün"
  if [[ -n "$ARG2" ]] && [[ "$ARG2" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    CMD_FLAGS="$CMD_FLAGS --end $ARG2"
    DONEM_LABEL="$ARG1 → $ARG2"
  fi
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ── Banner ───────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${BLUE}╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${BLUE}║         KRİPTO BOT — M1 MODEL BACKTEST                      ║${RESET}"
echo -e "${BOLD}${BLUE}║  Trend-Following | 1h | 21 Coin | Saf Strateji              ║${RESET}"
echo -e "${BOLD}${BLUE}╚══════════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  ${BOLD}Dönem:${RESET}  ${CYAN}$DONEM_LABEL${RESET}"
echo -e "  ${BOLD}Model:${RESET}  ${YELLOW}M1 (varsayılan — flag yok)${RESET}"
echo -e "  ${DIM}Veri çekiliyor...${RESET}"
echo ""

# ── Çalıştır ─────────────────────────────────────────────────────
START_TS=$(date +%s)

if [[ -n "$SAVE_FLAG" ]]; then
  mkdir -p backtest_results
  OUT="backtest_results/M1_${TIMESTAMP}.txt"
  python3 crypto_portfolio_test.py $CMD_FLAGS 2>&1 | tee "$OUT"
  EXIT_CODE=${PIPESTATUS[0]}
  echo -e "\n  ${GREEN}✅  Kaydedildi: $OUT${RESET}"
else
  python3 crypto_portfolio_test.py $CMD_FLAGS 2>&1
  EXIT_CODE=$?
fi

ELAPSED=$(( $(date +%s) - START_TS ))
echo -e "\n  ${DIM}⏱  Süre: ${ELAPSED}sn${RESET}\n"
exit $EXIT_CODE
