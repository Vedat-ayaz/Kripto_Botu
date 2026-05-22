#!/bin/bash
# ============================================================
#  Kripto Bot — Test Paneli  (M5 + M4v14c)
#  Kullanım:
#    bash test_bot.sh              → interaktif menü
#    bash test_bot.sh m5-bull      → Boğa dönemi M5 testi  ← YENİ
#    bash test_bot.sh m5-bear      → Ayı dönemi M5 testi   ← YENİ
#    bash test_bot.sh m5-karma     → Karma M5 testi        ← YENİ
#    bash test_bot.sh m5-all       → 3 dönem M5 paralel    ← YENİ
#    bash test_bot.sh m5-now       → Son 90 gün M5         ← YENİ
#    bash test_bot.sh m4-bull      → Boğa dönemi M4 testi
#    bash test_bot.sh m4-bear      → Ayı dönemi M4 testi
#    bash test_bot.sh m4-karma     → Karma (tam yıl) M4 testi
#    bash test_bot.sh m4-all       → 3 dönem birden çalıştır
#    bash test_bot.sh m4-now       → Son 90 gün M4 (güncel)
#    bash test_bot.sh quick        → Son 30 gün M1 (hızlı kontrol)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source venv/bin/activate

# ── Renkler ─────────────────────────────────────────────────
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
RESET='\033[0m'

# ── Yardımcılar ─────────────────────────────────────────────
run_test() {
    local cmd="$1"
    local desc="$2"
    local outfile="$3"   # opsiyonel: çıktı dosyası
    echo ""
    echo -e "${BOLD}▶ $desc${RESET}"
    echo -e "${CYAN}  $ $cmd${RESET}"
    echo ""
    if [[ -n "$outfile" ]]; then
        eval "$cmd" | tee "uzun_donem_testler/$outfile"
        echo ""
        echo -e "${GREEN}  ✓ Kaydedildi: uzun_donem_testler/$outfile${RESET}"
    else
        eval "$cmd"
    fi
    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
}

show_summary() {
    # Son test sonucundan kısa özet çek
    local file="$1"
    if [[ -f "$file" ]]; then
        echo -e "${BOLD}  Özet:${RESET}"
        grep -E "Bitiş Sermaye|Max Düşüş|Kazanma Oranı|Toplam İşlem" "$file" | \
            sed 's/^/    /'
    fi
}

# ── Direkt argüman modu ─────────────────────────────────────
if [[ $# -gt 0 ]]; then
    case "$1" in
        quick|q)
            run_test "python3 crypto_portfolio_test.py --days 30" "Son 30 gün — M1 (hızlı kontrol)"
            exit 0 ;;
        m5-bull)
            run_test \
                "python3 crypto_portfolio_test.py --start 2025-02-01 --end 2025-08-31 --m5 --coins 15" \
                "M5 — Boğa Dönemi (2025-02-01 → 2025-08-31)"
            exit 0 ;;
        m5-bear)
            run_test \
                "python3 crypto_portfolio_test.py --start 2025-06-01 --end 2026-01-31 --m5 --coins 15" \
                "M5 — Ayı Dönemi (2025-06-01 → 2026-01-31)"
            exit 0 ;;
        m5-karma)
            run_test \
                "python3 crypto_portfolio_test.py --start 2025-05-15 --end 2026-05-15 --m5 --coins 15" \
                "M5 — Karma / Tam Yıl (2025-05-15 → 2026-05-15)"
            exit 0 ;;
        m5-now|m5-current)
            run_test \
                "python3 crypto_portfolio_test.py --m5 --coins 15 --days 90" \
                "M5 — Son 90 Gün (güncel durum)"
            exit 0 ;;
        m5-all)
            echo ""
            echo -e "${BOLD}  ⚡ M5: 3 dönem testi başlatılıyor (paralel)...${RESET}"
            echo ""
            python3 crypto_portfolio_test.py --start 2025-02-01 --end 2025-08-31 --m5 --coins 15 \
                > uzun_donem_testler/boga_M5.txt 2>&1 &
            PID1=$!
            python3 crypto_portfolio_test.py --start 2025-06-01 --end 2026-01-31 --m5 --coins 15 \
                > uzun_donem_testler/ayi_M5.txt 2>&1 &
            PID2=$!
            python3 crypto_portfolio_test.py --start 2025-05-15 --end 2026-05-15 --m5 --coins 15 \
                > uzun_donem_testler/karma_M5.txt 2>&1 &
            PID3=$!
            echo -e "  Boğa  PID: $PID1 → uzun_donem_testler/boga_M5.txt"
            echo -e "  Ayı   PID: $PID2 → uzun_donem_testler/ayi_M5.txt"
            echo -e "  Karma PID: $PID3 → uzun_donem_testler/karma_M5.txt"
            echo ""
            echo -e "  ${YELLOW}Beklemek için: wait $PID1 $PID2 $PID3 && bash test_bot.sh m5-karsilastir${RESET}"
            exit 0 ;;
        m5-karsilastir)
            echo ""
            echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${RESET}"
            echo -e "${BOLD}║           M5 vs M4 KARŞILAŞTIRMA                        ║${RESET}"
            echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${RESET}"
            for label in "boga:🐂 BOĞA " "ayi:🐻 AYI  " "karma:🔀 KARMA"; do
                fname="${label%%:*}"
                lname="${label##*:}"
                echo ""
                echo -e "  ${BOLD}$lname${RESET}"
                if [[ -f "uzun_donem_testler/${fname}_M5.txt" ]]; then
                    echo -e "    ${GREEN}M5:${RESET}"
                    grep -E "Bitiş Sermaye|Max Düşüş|Kazanma Oranı" \
                        "uzun_donem_testler/${fname}_M5.txt" | sed 's/^/      /'
                fi
                if [[ -f "uzun_donem_testler/${fname}_test.txt" ]]; then
                    echo -e "    ${YELLOW}M4:${RESET}"
                    grep -E "Bitiş Sermaye|Max Düşüş|Kazanma Oranı" \
                        "uzun_donem_testler/${fname}_test.txt" | sed 's/^/      /'
                fi
            done
            echo ""
            exit 0 ;;
        m4-bull)
            run_test \
                "python3 crypto_portfolio_test.py --start 2025-02-01 --end 2025-08-31 --m4 --coins 15" \
                "M4 — Boğa Dönemi (2025-02-01 → 2025-08-31)"
            exit 0 ;;
        m4-bear)
            run_test \
                "python3 crypto_portfolio_test.py --start 2025-06-01 --end 2026-01-31 --m4 --coins 15" \
                "M4 — Ayı Dönemi (2025-06-01 → 2026-01-31)"
            exit 0 ;;
        m4-karma)
            run_test \
                "python3 crypto_portfolio_test.py --start 2025-05-15 --end 2026-05-15 --m4 --coins 15" \
                "M4 — Karma / Tam Yıl (2025-05-15 → 2026-05-15)"
            exit 0 ;;
        m4-now|m4-current)
            run_test \
                "python3 crypto_portfolio_test.py --m4 --coins 15 --days 90" \
                "M4 — Son 90 Gün (güncel durum)"
            exit 0 ;;
        m4-all)
            echo ""
            echo -e "${BOLD}  ⚡ 3 dönem testi başlatılıyor (paralel)...${RESET}"
            echo ""
            python3 crypto_portfolio_test.py --start 2025-02-01 --end 2025-08-31 --m4 --coins 15 \
                > uzun_donem_testler/boga_test.txt 2>&1 &
            PID1=$!
            python3 crypto_portfolio_test.py --start 2025-06-01 --end 2026-01-31 --m4 --coins 15 \
                > uzun_donem_testler/ayi_test.txt 2>&1 &
            PID2=$!
            python3 crypto_portfolio_test.py --start 2025-05-15 --end 2026-05-15 --m4 --coins 15 \
                > uzun_donem_testler/karma_test.txt 2>&1 &
            PID3=$!
            echo -e "  Boğa  PID: $PID1 → uzun_donem_testler/boga_test.txt"
            echo -e "  Ayı   PID: $PID2 → uzun_donem_testler/ayi_test.txt"
            echo -e "  Karma PID: $PID3 → uzun_donem_testler/karma_test.txt"
            echo ""
            echo -e "  ${YELLOW}Testler arka planda çalışıyor. Tamamlanınca sonuçlar görünür.${RESET}"
            echo -e "  Beklemek için: wait $PID1 $PID2 $PID3 && bash test_bot.sh karsilastir"
            exit 0 ;;
        karsilastir|compare)
            echo ""
            echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${RESET}"
            echo -e "${BOLD}║           TEST SONUÇLARI KARŞILAŞTIRMASI                ║${RESET}"
            echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${RESET}"
            for label in "boga_test:🐂 BOĞA " "ayi_test:🐻 AYI  " "karma_test:🔀 KARMA"; do
                fname="${label%%:*}.txt"
                lname="${label##*:}"
                if [[ -f "uzun_donem_testler/$fname" ]]; then
                    echo ""
                    echo -e "  ${BOLD}$lname${RESET}"
                    grep -E "Bitiş Sermaye|Max Düşüş|Kazanma Oranı" \
                        "uzun_donem_testler/$fname" | sed 's/^/    /'
                fi
            done
            echo ""
            exit 0 ;;
        help|-h|--help)
            echo ""
            echo "Kullanım: bash test_bot.sh [komut]"
            echo ""
            echo "  quick          Son 30 gün M1 (hızlı kontrol)"
            echo "  m4-bull        Boğa dönemi (2025-02-01 → 2025-08-31)"
            echo "  m4-bear        Ayı dönemi  (2025-06-01 → 2026-01-31)"
            echo "  m4-karma       Karma/Tam yıl (2025-05-15 → 2026-05-15)"
            echo "  m4-now         Son 90 gün (güncel durum)"
            echo "  m4-all         3 dönem paralel (arka planda)"
            echo "  karsilastir    m4-all sonuçlarını göster"
            echo "  (yok)          İnteraktif menü"
            echo ""
            exit 0 ;;
        *)
            echo -e "${RED}  Bilinmeyen komut: $1${RESET}"
            echo "  bash test_bot.sh help  → yardım"
            exit 1 ;;
    esac
fi

# ── İnteraktif menü ─────────────────────────────────────────
show_menu() {
    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${BOLD}║          KRİPTO BOT — TEST PANELİ (M5 + M4v14c)         ║${RESET}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${RESET}"
    echo ""
    echo -e "  ${MAGENTA}── M5 (Son Model — Önerilen) ───────────────────────────${RESET}"
    echo -e "  ${GREEN}[1]${RESET}  M5 — Boğa dönemi    2025-02-01 → 2025-08-31  (+8.45%)"
    echo -e "  ${GREEN}[2]${RESET}  M5 — Ayı dönemi     2025-06-01 → 2026-01-31  (+13.54%)"
    echo -e "  ${GREEN}[3]${RESET}  M5 — Karma/Tam yıl  2025-05-15 → 2026-05-15  (+5.32%)"
    echo -e "  ${GREEN}[4]${RESET}  M5 — Son 90 gün     (güncel piyasa)"
    echo ""
    echo -e "  ${MAGENTA}── M4 (Stabil Referans) ────────────────────────────────${RESET}"
    echo -e "  ${YELLOW}[5]${RESET}  M4 — Boğa dönemi    2025-02-01 → 2025-08-31  (+9.25%)"
    echo -e "  ${YELLOW}[6]${RESET}  M4 — Ayı dönemi     2025-06-01 → 2026-01-31  (+11.12%)"
    echo -e "  ${YELLOW}[7]${RESET}  M4 — Karma/Tam yıl  2025-05-15 → 2026-05-15  (+5.26%)"
    echo -e "  ${YELLOW}[8]${RESET}  M4 — Son 90 gün     (güncel piyasa)"
    echo ""
    echo -e "  ${MAGENTA}── Özel Tarih ──────────────────────────────────────────${RESET}"
    echo -e "  ${CYAN}[9]${RESET}  İstediğim tarihi test et  (tarih + mod seçimi)"
    echo ""
    echo -e "  ${MAGENTA}── Diğer ───────────────────────────────────────────────${RESET}"
    echo -e "  ${YELLOW}[10]${RESET} M1 — Son 30 gün (hızlı kontrol)"
    echo -e "  ${CYAN}[11]${RESET} M5 vs M4 — Karşılaştırma tablosu"
    echo ""
    echo -e "  ${RED}[0]${RESET}  Çıkış"
    echo ""
    printf "  Seçim: "
}

while true; do
    show_menu
    read -r secim

    case "$secim" in
        # ── M5 hazır dönemler ──────────────────────────────────
        1)
            run_test \
                "python3 crypto_portfolio_test.py --start 2025-02-01 --end 2025-08-31 --m5 --coins 15" \
                "M5 — Boğa Dönemi (2025-02-01 → 2025-08-31)"
            ;;
        2)
            run_test \
                "python3 crypto_portfolio_test.py --start 2025-06-01 --end 2026-01-31 --m5 --coins 15" \
                "M5 — Ayı Dönemi (2025-06-01 → 2026-01-31)"
            ;;
        3)
            run_test \
                "python3 crypto_portfolio_test.py --start 2025-05-15 --end 2026-05-15 --m5 --coins 15" \
                "M5 — Karma / Tam Yıl (2025-05-15 → 2026-05-15)"
            ;;
        4)
            run_test \
                "python3 crypto_portfolio_test.py --m5 --coins 15 --days 90" \
                "M5 — Son 90 Gün (güncel)"
            ;;
        # ── M4 hazır dönemler ──────────────────────────────────
        5)
            run_test \
                "python3 crypto_portfolio_test.py --start 2025-02-01 --end 2025-08-31 --m4 --coins 15" \
                "M4 — Boğa Dönemi (2025-02-01 → 2025-08-31)"
            ;;
        6)
            run_test \
                "python3 crypto_portfolio_test.py --start 2025-06-01 --end 2026-01-31 --m4 --coins 15" \
                "M4 — Ayı Dönemi (2025-06-01 → 2026-01-31)"
            ;;
        7)
            run_test \
                "python3 crypto_portfolio_test.py --start 2025-05-15 --end 2026-05-15 --m4 --coins 15" \
                "M4 — Karma / Tam Yıl (2025-05-15 → 2026-05-15)"
            ;;
        8)
            run_test \
                "python3 crypto_portfolio_test.py --m4 --coins 15 --days 90" \
                "M4 — Son 90 Gün (güncel)"
            ;;
        # ── Özel tarih seçimi ──────────────────────────────────
        9)
            echo ""
            printf "  Başlangıç tarihi (YYYY-MM-DD, örn: 2024-01-01): "
            read -r t_start
            printf "  Bitiş tarihi    (YYYY-MM-DD, örn: 2024-06-30): "
            read -r t_end
            echo ""
            echo -e "  Mod seç:"
            echo -e "  ${GREEN}[1]${RESET} M5  — Son model (önerilen)"
            echo -e "  ${YELLOW}[2]${RESET} M4  — Stabil referans"
            echo -e "  ${CYAN}[3]${RESET} M1  — Sade trend-following"
            printf "  [1/2/3, varsayılan=1]: "
            read -r t_mod
            printf "  Dosyaya kaydet? (örn: test_ocak2025  — boş=sadece ekran): "
            read -r t_file

            S_FLAG=""; E_FLAG=""
            [[ -n "$t_start" ]] && S_FLAG="--start $t_start"
            [[ -n "$t_end"   ]] && E_FLAG="--end $t_end"
            case "$t_mod" in
                2) M_FLAG="--m4 --coins 15" ;;
                3) M_FLAG="--coins 15" ;;
                *) M_FLAG="--m5 --coins 15" ;;
            esac

            if [[ -n "$t_file" ]]; then
                [[ "$t_file" != *.txt ]] && t_file="${t_file}.txt"
                run_test "python3 crypto_portfolio_test.py $S_FLAG $E_FLAG $M_FLAG" \
                    "Özel Test: $t_start → $t_end" "$t_file"
            else
                run_test "python3 crypto_portfolio_test.py $S_FLAG $E_FLAG $M_FLAG" \
                    "Özel Test: $t_start → $t_end"
            fi
            ;;
        # ── Diğer ──────────────────────────────────────────────
        10)
            run_test \
                "python3 crypto_portfolio_test.py --days 30" \
                "M1 — Son 30 Gün (hızlı kontrol)"
            ;;
        11)
            echo ""
            echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${RESET}"
            echo -e "${BOLD}║              M5 vs M4 — KARŞILAŞTIRMA                   ║${RESET}"
            echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${RESET}"
            for label in "boga:🐂 BOĞA " "ayi:🐻 AYI  " "karma:🔀 KARMA"; do
                fname="${label%%:*}"
                lname="${label##*:}"
                echo ""
                echo -e "  ${BOLD}$lname${RESET}"
                if [[ -f "uzun_donem_testler/${fname}_M5.txt" ]]; then
                    echo -e "    ${GREEN}M5:${RESET}"
                    grep -E "Bitiş Sermaye|Max Düşüş|Kazanma Oranı" \
                        "uzun_donem_testler/${fname}_M5.txt" | sed 's/^/      /'
                fi
                if [[ -f "uzun_donem_testler/${fname}_test.txt" ]]; then
                    echo -e "    ${YELLOW}M4:${RESET}"
                    grep -E "Bitiş Sermaye|Max Düşüş|Kazanma Oranı" \
                        "uzun_donem_testler/${fname}_test.txt" | sed 's/^/      /'
                fi
            done
            echo ""
            ;;
        0)
            echo ""
            echo -e "${GREEN}  Çıkış.${RESET}"
            echo ""
            break
            ;;
        *)
            echo -e "${RED}  Geçersiz seçim: '$secim'${RESET}"
            ;;
    esac

    echo ""
    printf "  Enter'a bas devam et..."
    read -r
done
