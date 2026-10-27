#!/usr/bin/env bash
# Stop the render page: recording loop (and any recording in progress), page server, tunnel.
HERE="$(cd "$(dirname "$0")" && pwd)"
for name in render_cloudflared render_http render_loop; do
  pidfile="$HERE/run/$name.pid"
  if [[ -f "$pidfile" ]]; then
    pid="$(cat "$pidfile")"
    pkill -TERM -P "$pid" 2>/dev/null || true   # a recording the loop started
    kill "$pid" 2>/dev/null && echo "stopped $name"
    rm -f "$pidfile"
  fi
done
rm -f "$HERE/RENDER_URL.txt"
