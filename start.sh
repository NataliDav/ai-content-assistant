#!/usr/bin/env bash
# Запуск ИИ-ассистента и открытие в браузере.
# Можно запускать двойным кликом или из терминала: ./start.sh
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PID=$(pgrep -f "python3 server\.py$" | head -1)
[ -n "$PID" ] && kill "$PID"
sleep 1

setsid nohup python3 server.py >"$DIR/server.log" 2>&1 &
sleep 2

xdg-open http://127.0.0.1:8000