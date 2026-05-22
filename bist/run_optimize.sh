#!/bin/bash
# BIST Walk-Forward Optimizasyon — arka planda çalıştır
cd "$(dirname "$0")/.."

LOG_FILE="bist/optimizer_log_$(date +%Y%m%d_%H%M).txt"

echo "Walk-Forward Optimizasyon başlıyor..."
echo "Log: $LOG_FILE"
echo "Durdurmak için: kill \$(cat bist/optimizer.pid)"

nohup ./venv/bin/python bist/bist_main.py \
    --mode optimize \
    --start 2020-01-01 \
    --end 2025-01-01 \
    > "$LOG_FILE" 2>&1 &

echo $! > bist/optimizer.pid
echo "PID: $(cat bist/optimizer.pid)"
echo ""
echo "Logu takip etmek için:"
echo "  tail -f $LOG_FILE"
