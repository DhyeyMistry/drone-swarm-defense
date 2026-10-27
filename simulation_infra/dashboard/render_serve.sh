#!/usr/bin/env bash
# Serve a web page of simulation videos, recorded from each new training checkpoint.
#
#   ./render_serve.sh              # record loop + page on 127.0.0.1:8790 + public Cloudflare tunnel
#   ./render_stop.sh               # stop all three
#
# render_loop.py watches logs/sb3/<task>/ for new checkpoints and records one episode of each
# with scripts/record_room.py (headless viewport capture), at most once per 30 min (--min_gap
# seconds): one full episode takes ~3 min alone, ~10 min while training shares the GPU. Any
# render_loop.py options pass through, e.g. ./render_serve.sh --min_gap 900 --max_seconds 30.
# The public URL is written to RENDER_URL.txt. Same tunnel setup as serve.sh, separate URL.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SIM="$(dirname "$HERE")"
PORT="${RENDER_PORT:-8790}"
RUN="$HERE/run"
PY="${PY:-$HOME/isaac/env_isaaclab/bin/python}"
CF="${CF:-$(command -v cloudflared || echo "$HOME/.local/bin/cloudflared")}"
mkdir -p "$RUN" "$HERE/render_site/videos"

"$HERE/render_stop.sh" >/dev/null 2>&1 || true

# the Isaac environment record_room.py needs (KIT_SETUP_GUIDE.md, Step 2)
unset CONDA_PREFIX CONDA_DEFAULT_ENV
export OMNI_KIT_ACCEPT_EULA=YES TELLO_USD_PATH="${TELLO_USD_PATH:-$HOME/isaac/assets/tello.usd}"
export PATH="$(dirname "$PY"):$PATH" VIRTUAL_ENV="$(dirname "$(dirname "$PY")")"

cd "$SIM"
setsid nohup "$PY" -u "$HERE/render_loop.py" "$@" > "$RUN/render_loop.log" 2>&1 &
echo $! > "$RUN/render_loop.pid"

setsid nohup "$PY" -m http.server "$PORT" --bind 127.0.0.1 --directory "$HERE/render_site" \
  > "$RUN/render_http.log" 2>&1 &
echo $! > "$RUN/render_http.pid"
for _ in $(seq 30); do curl -fs "http://127.0.0.1:$PORT" >/dev/null && break; sleep 1; done
curl -fs "http://127.0.0.1:$PORT" >/dev/null || { echo "page server did not start; see $RUN/render_http.log"; exit 1; }
echo "Render page: http://127.0.0.1:$PORT"

setsid nohup "$CF" tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" > "$RUN/render_cloudflared.log" 2>&1 &
echo $! > "$RUN/render_cloudflared.pid"
for _ in $(seq 60); do
  URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$RUN/render_cloudflared.log" | head -1 || true)"
  [[ -n "$URL" ]] && break
  sleep 1
done
if [[ -z "${URL:-}" ]]; then echo "no tunnel URL yet; see $RUN/render_cloudflared.log"; exit 1; fi
echo "$URL" > "$HERE/RENDER_URL.txt"
echo "Public render page: $URL"
echo "Recording log: $RUN/render_loop.log"
