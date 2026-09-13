#!/usr/bin/env bash
# Start (or restart) the live dashboard server on the workstation: bash start_server.sh [port]
# No pgrep/pkill (a pattern kill matches the calling shell): the pid is kept in server.pid.
set -u
d=$(cd "$(dirname "$0")" && pwd); port="${1:-8020}"; pidfile="$d/server.pid"
if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then kill "$(cat "$pidfile")"; sleep 1; fi
mkdir -p /iris/u/kewalk/memory_project_v7/v7/logs
cd "$d" && setsid nohup python3 serve_dashboard.py "$port" > /iris/u/kewalk/memory_project_v7/v7/logs/serve_dashboard.out 2>&1 < /dev/null &
echo $! > "$pidfile"; sleep 5
echo "pid $(cat "$pidfile") on $(hostname):$port"; curl -s -o /dev/null -w "page http %{http_code}, %{size_download} bytes\n" "http://127.0.0.1:$port/"
