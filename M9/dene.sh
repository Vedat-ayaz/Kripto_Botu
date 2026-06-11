#!/bin/bash
# ════════════════════════════════════════════════════════════
#  M9 / M5 / M7 / ORTAKLIK — kendi tarih aralığında test
# ════════════════════════════════════════════════════════════
#  Kullanım:
#    ./dene.sh 2026-03-01 2026-04-15              → M9 (varsayılan)
#    ./dene.sh 2026-03-01 2026-04-15 m9 trades    → M9 + işlem listesi
#    ./dene.sh 2026-03-01 2026-04-15 m5           → M5 legacy
#    ./dene.sh 2026-03-01 2026-04-15 m7           → M7 legacy
#    ./dene.sh 2026-03-01 2026-04-15 ortaklik     → M7 taban + M9 uydu tahsisçi
#
#  Not: yeni aralıkta ilk koşu veri indirir (birkaç dk), sonrası cache'ten.
cd "$(dirname "$0")"
PY=../venv/bin/python
[ -x "$PY" ] || PY=python3

START="$1"; END="$2"; MODE="${3:-m9}"; EXTRA=""
[ "$4" = "trades" ] && EXTRA="--trades"

if [ -z "$START" ] || [ -z "$END" ]; then
    grep '^#' "$0" | head -12
    exit 1
fi

exec "$PY" dene.py --start "$START" --end "$END" --mode "$MODE" $EXTRA
