#!/bin/bash
# M9 APEX backtest çalıştırıcı
# Kullanım: ./backtest.sh 2026-05-27 2026-06-09 [trades]
cd "$(dirname "$0")"
PY=../venv/bin/python
[ -x "$PY" ] || PY=python3

START="$1"
END="$2"
EXTRA=""
[ "$3" = "trades" ] && EXTRA="--trades"

if [ -z "$START" ] || [ -z "$END" ]; then
    echo "Kullanım: ./backtest.sh BAŞLANGIÇ BİTİŞ [trades]"
    echo "Örnek:    ./backtest.sh 2026-05-27 2026-06-09"
    exit 1
fi

exec "$PY" m9_backtest.py --start "$START" --end "$END" $EXTRA
