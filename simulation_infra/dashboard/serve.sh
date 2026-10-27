#!/usr/bin/env bash
# Serve this desktop's training TensorBoard on the web through a Cloudflare tunnel.
#
#   ./serve.sh                     # TensorBoard over logs/sb3 (every task), public tunnel
#   ./serve.sh logs/sb3            # any log folder (relative to simulation_infra/)
#   ./stop.sh                      # stop both
#
# TensorBoard listens on 127.0.0.1 only; cloudflared makes an outbound connection to
# Cloudflare, so no router ports are opened. The public URL is written to URL.txt.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SIM="$(dirname "$HERE")"
LOGDIR="$SIM/${1:-logs/sb3}"
PORT="${PORT:-6006}"
RUN="$HERE/run"
TB="${TB:-$HOME/isaac/env_isaaclab/bin/tensorboard}"
CF="${CF:-$(command -v cloudflared || echo "$HOME/.local/bin/cloudflared")}"
mkdir -p "$RUN" "$LOGDIR"

"$HERE/stop.sh" >/dev/null 2>&1 || true

setsid nohup "$TB" --logdir "$LOGDIR" --host 127.0.0.1 --port "$PORT" --reload_interval 30 \
  > "$RUN/tensorboard.log" 2>&1 &
echo $! > "$RUN/tensorboard.pid"

for _ in $(seq 60); do curl -fs "http://127.0.0.1:$PORT" >/dev/null && break; sleep 1; done
curl -fs "http://127.0.0.1:$PORT" >/dev/null || { echo "TensorBoard did not start; see $RUN/tensorboard.log"; exit 1; }
echo "TensorBoard: http://127.0.0.1:$PORT  (logdir $LOGDIR)"

if [[ -n "${CF_TUNNEL_TOKEN:-}" ]]; then
  # named tunnel from your Cloudflare account (fixed hostname, can sit behind Cloudflare Access)
  setsid nohup "$CF" tunnel --no-autoupdate run --token "$CF_TUNNEL_TOKEN" > "$RUN/cloudflared.log" 2>&1 &
  echo $! > "$RUN/cloudflared.pid"
  echo "named tunnel started; its hostname is the one configured in your Cloudflare dashboard"
  exit 0
fi

# quick tunnel: no account, random https://*.trycloudflare.com URL, new each start
setsid nohup "$CF" tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" > "$RUN/cloudflared.log" 2>&1 &
echo $! > "$RUN/cloudflared.pid"
for _ in $(seq 60); do
  URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$RUN/cloudflared.log" | head -1 || true)"
  [[ -n "$URL" ]] && break
  sleep 1
done
if [[ -z "${URL:-}" ]]; then echo "no tunnel URL yet; see $RUN/cloudflared.log"; exit 1; fi
echo "$URL" > "$HERE/URL.txt"
echo "Public dashboard: $URL"
