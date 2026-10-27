#!/usr/bin/env bash
# Stop the dashboard's TensorBoard and Cloudflare tunnel.
HERE="$(cd "$(dirname "$0")" && pwd)"
for name in cloudflared tensorboard; do
  pidfile="$HERE/run/$name.pid"
  if [[ -f "$pidfile" ]]; then
    kill "$(cat "$pidfile")" 2>/dev/null && echo "stopped $name"
    rm -f "$pidfile"
  fi
done
rm -f "$HERE/URL.txt"
