#!/usr/bin/env python3
"""Hardware/software-in-the-loop GUI: YOU are the attackers. Drag them with the mouse.

Opens a top-down map of the room (from config/map.yaml) in the browser. Each attacker
is a red dot you drag; this GUI publishes them as the repo's fleet telemetry (`v: 1`,
pose in tag 10's frame, exactly the format defend.py reads) to --feed (default
127.0.0.1:9100). defend.py sends its status back on --status_port, so you see every
defender (blue), its target and its mode, live -- simulated (--sim) or a real drone.

  Drag a red dot             move that attacker (its height: the slider on its row)
  "Attack!"                  attackers fly at the table on their own (speed, weave)
  "Down" on a row            attacker dropped (fly=false) -- defenders should let go
  Contact (defender within 0.35 m) knocks an attacker down automatically, and an
  attacker reaching the table counts as a breach. Counters at the top.

  python hil_gui.py --attackers TELLO-8,TELLO-9           # then open http://127.0.0.1:8790/
  python defend.py --sim --defenders D1,D2,D3 --attackers TELLO-8,TELLO-9
"""

from __future__ import annotations

import argparse
import json
import math
import random
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from defense_core import default_fence  # noqa: E402

TAG10_HEIGHT_M = 1.22
CAPTURE_M = 0.35
SITE_RADIUS_M = 0.5


class World:
    def __init__(self, names, tag_poses, site, fence, feed, hz, tag_h=TAG10_HEIGHT_M) -> None:
        self.tag_h = tag_h
        self.names, self.tag_poses, self.site, self.fence = names, tag_poses, site, fence
        self.feed, self.hz = feed, hz
        self.lock = threading.Lock()
        self.status = None
        self.status_t = 0.0
        self.attack = False
        self.speed = 0.4
        self.weave = 0.0
        self.counts = {"contacts": 0, "breaches": 0}
        self.seq = 0
        self.reset()

    def reset(self) -> None:
        lo, hi = self.fence.lo, self.fence.hi
        k = len(self.names)
        self.att = {}
        for i, n in enumerate(self.names):
            x = lo[0] + (i + 1) * (hi[0] - lo[0]) / (k + 1)
            # attackers come from tag 10's side; our defenders and the table are further out
            self.att[n] = {"pos": np.array([x, 1.2 - self.tag_h, lo[2] + 0.3]), "alive": True,
                           "phase": random.random() * 6.28, "breach": False}
        self.attack = False
        self.counts = {"contacts": 0, "breaches": 0}

    def step(self, dt: float) -> None:
        with self.lock:
            defenders = []
            if self.status and time.time() - self.status_t < 1.0:
                defenders = [np.asarray(d["pos"]) for d in self.status["defenders"] if d.get("pos")]
            for n, a in self.att.items():
                if not a["alive"]:
                    continue
                if self.attack and not a["breach"]:
                    to = self.site - a["pos"]
                    to[1] = 0.0
                    d = float(np.linalg.norm(to))
                    if d > 1e-6:
                        u = to / d
                        side = np.array([-u[2], 0.0, u[0]])
                        a["phase"] += dt * 2.0
                        v = u * self.speed + side * self.weave * self.speed * math.sin(a["phase"])
                        a["pos"] = a["pos"] + v * dt
                flat = a["pos"] - self.site
                flat[1] = 0.0
                if float(np.linalg.norm(flat)) < SITE_RADIUS_M and not a["breach"]:
                    a["breach"] = True
                    self.counts["breaches"] += 1
                for p in defenders:
                    if float(np.linalg.norm(p - a["pos"])) < CAPTURE_M:
                        a["alive"] = False
                        self.counts["contacts"] += 1
                        break

    def telemetry(self) -> bytes:
        """Repo telemetry v1, pose in tag 10's frame exactly as drone_gui publishes it."""
        self.seq += 1
        drones = []
        with self.lock:
            for i, (n, a) in enumerate(self.att.items(), 1):
                x, y, z = a["pos"]
                drones.append({"name": n, "slot": i, "team": "attack", "link": True, "fly": a["alive"],
                               "mode": "auto" if a["alive"] else "ground", "bat": 90,
                               "h": round(y + self.tag_h, 2) if a["alive"] else 0.0, "tof": None,
                               "vel": [0, 0, 0],
                               # drone_gui's convention with official tag prints (feeds "readme"): x right, y down, z into wall
                               "pose": {"tag": 10, "xyz": [round(x, 3), round(-y, 3), round(-z, 3)],
                                        "yaw": 0.0, "age": 0.0}})
        return json.dumps({"v": 1, "src": "hil", "seq": self.seq, "t": time.time(), "hz": self.hz,
                           "drones": drones}, separators=(",", ":")).encode()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "tags": {str(k): np.asarray(M)[:3, 3].round(3).tolist() for k, M in self.tag_poses.items()},
                "site": self.site.round(3).tolist(), "site_radius": SITE_RADIUS_M,
                "fence": [self.fence.lo.round(2).tolist(), self.fence.hi.round(2).tolist()],
                "attackers": [{"name": n, "pos": a["pos"].round(3).tolist(), "alive": a["alive"],
                               "h": round(a["pos"][1] + self.tag_h, 2)} for n, a in self.att.items()],
                "status": self.status if time.time() - self.status_t < 1.0 else None,
                "attack": self.attack, "speed": self.speed, "weave": self.weave, "counts": self.counts,
            }

    def command(self, c: dict) -> None:
        with self.lock:
            op = c.get("op")
            a = self.att.get(c.get("name"))
            if op == "move" and a:
                p = np.array([float(c["x"]), a["pos"][1], float(c["z"])])
                a["pos"] = np.minimum(np.maximum(p, self.fence.lo), self.fence.hi)
                a["breach"] = False
            elif op == "height" and a:
                a["pos"][1] = float(c["h"]) - self.tag_h
            elif op == "toggle" and a:
                a["alive"] = not a["alive"]
            elif op == "attack":
                self.attack = bool(c.get("on"))
            elif op == "speed":
                self.speed = float(c["v"])
            elif op == "weave":
                self.weave = float(c["v"])
        if op == "reset":
            with self.lock:
                self.reset()


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Defense HIL</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f6f7f9;--fg:#1c1f24;--mut:#69707a;--card:#fff;--line:#d8dce2;--att:#d6453d;--def:#2f6fdb;--site:#1d9a5b}
@media (prefers-color-scheme:dark){:root{--bg:#15171a;--fg:#e8eaed;--mut:#9aa1ab;--card:#1f2226;--line:#33373d}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line)}
header b{font-size:16px}.cnt{font-variant-numeric:tabular-nums}
main{display:flex;flex-wrap:wrap;gap:16px;padding:16px}
canvas{background:var(--card);border:1px solid var(--line);border-radius:8px;touch-action:none;max-width:100%}
.side{flex:1;min-width:280px}.row{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin-bottom:8px}
button{font:inherit;padding:4px 10px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.on{background:var(--att);color:#fff;border-color:var(--att)}
.mut{color:var(--mut);font-size:12px}
</style></head><body>
<header><b>Defense HIL</b>
<button id="atk">Attack!</button><button id="rst">Reset</button>
<label>speed <input id="spd" type="range" min="0.1" max="1.0" step="0.05" value="0.4"> <span id="spdv"></span> m/s</label>
<label>weave <input id="wv" type="range" min="0" max="1" step="0.1" value="0"></label>
<span class="cnt" id="cnt"></span><span class="mut" id="link"></span></header>
<main><canvas id="c" width="640" height="720"></canvas><div class="side" id="side"></div></main>
<script>
const cv=document.getElementById('c'),g=cv.getContext('2d');let S=null,drag=null;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function view(){const [lo,hi]=S.fence;const xs=[lo[0]-0.6,hi[0]+0.6],zs=[Math.min(-0.5,lo[2]-0.6),hi[2]+0.6];
 const s=Math.min(cv.width/(xs[1]-xs[0]),cv.height/(zs[1]-zs[0]));return {s,ox:xs[0],oz:zs[0]}}
const P=(v,x,z)=>[(x-v.ox)*v.s,(z-v.oz)*v.s];
const W=(v,px,py)=>[px/v.s+v.ox,py/v.s+v.oz];
function draw(){if(!S)return;const v=view();g.clearRect(0,0,cv.width,cv.height);
 const [lo,hi]=S.fence;let [a,b]=P(v,lo[0],lo[2]),[c,d]=P(v,hi[0],hi[2]);
 g.setLineDash([6,4]);g.strokeStyle=css('--mut');g.strokeRect(a,b,c-a,d-b);g.setLineDash([]);
 g.fillStyle=css('--mut');g.font='12px system-ui';g.fillText('fence',a+4,b+14);
 for(const [id,p] of Object.entries(S.tags)){const [x,y]=P(v,p[0],p[2]);g.fillStyle=css('--fg');g.fillRect(x-6,y-6,12,12);g.fillText('tag '+id,x+9,y+4)}
 let [sx,sy]=P(v,S.site[0],S.site[2]);g.strokeStyle=css('--site');g.lineWidth=2;g.beginPath();g.arc(sx,sy,S.site_radius*v.s,0,6.3);g.stroke();
 g.fillStyle=css('--site');g.fillText('TABLE',sx-18,sy+4);
 const st=S.status;if(st){for(const d of st.defenders){if(!d.pos)continue;const [x,y]=P(v,d.pos[0],d.pos[2]);
  if(d.target){const [tx,ty]=P(v,d.target[0],d.target[2]);g.strokeStyle=css('--def');g.setLineDash([3,3]);g.beginPath();g.moveTo(x,y);g.lineTo(tx,ty);g.stroke();g.setLineDash([]);g.strokeRect(tx-4,ty-4,8,8)}
  g.fillStyle=css('--def');g.beginPath();g.arc(x,y,9,0,6.3);g.fill();g.fillStyle=css('--fg');g.fillText(d.name+' '+d.mode+(d.attacker?'→'+d.attacker:''),x+12,y-8)}}
 for(const t of S.attackers){const [x,y]=P(v,t.pos[0],t.pos[2]);g.fillStyle=t.alive?css('--att'):css('--mut');
  g.beginPath();g.arc(x,y,10,0,6.3);g.fill();g.fillStyle=css('--fg');g.fillText(t.name+' h'+t.h.toFixed(1)+(t.alive?'':' DOWN'),x+12,y+16)}
 g.fillStyle=css('--mut');g.fillText('BASE wall (tag 10) at top · front at bottom · x → right',8,cv.height-8)}
function side(){const el=document.getElementById('side');if(!S)return;
 el.innerHTML=S.attackers.map(t=>`<div class="row"><b style="color:var(--att)">${t.name}</b> ${t.alive?'flying':'DOWN'}
  <button data-t="${t.name}">${t.alive?'Down':'Revive'}</button><br>height <input type="range" min="0.4" max="2.0" step="0.05" value="${t.h}" data-h="${t.name}"> ${t.h.toFixed(2)} m</div>`).join('')+
  (S.status?S.status.defenders.map(d=>`<div class="row"><b style="color:var(--def)">${d.name}</b> ${d.mode}${d.attacker?' → '+d.attacker:''}<br><span class="mut">rc ${d.rc.join(' ')} · ${d.note}</span></div>`).join(''):'<div class="row mut">no defend.py status yet: start defend.py</div>');
 el.querySelectorAll('[data-t]').forEach(b=>b.onclick=()=>cmd({op:'toggle',name:b.dataset.t}));
 el.querySelectorAll('[data-h]').forEach(r=>r.onchange=()=>cmd({op:'height',name:r.dataset.h,h:r.value}))}
function cmd(c){return fetch('/cmd',{method:'POST',body:JSON.stringify(c)})}
let lastSide=0;async function poll(){try{S=await (await fetch('/state')).json();draw();
 document.getElementById('cnt').textContent=`contacts ${S.counts.contacts} · breaches ${S.counts.breaches}`;
 document.getElementById('link').textContent=S.status?`defend.py live · feed ${S.status.feed.datagrams} pkts`:'defend.py not connected';
 const a=document.getElementById('atk');a.className=S.attack?'on':'';a.textContent=S.attack?'Stop':'Attack!';
 document.getElementById('spdv').textContent=S.speed.toFixed(2);for(const [id,v] of [['spd',S.speed],['wv',S.weave]]){const el=document.getElementById(id);if(document.activeElement!==el)el.value=v}
 if(!document.activeElement||document.activeElement.type!=='range'){if(Date.now()-lastSide>400){side();lastSide=Date.now()}}}catch(e){}
 setTimeout(poll,100)}
function pick(e){const r=cv.getBoundingClientRect(),v=view();const px=(e.clientX-r.left)*cv.width/r.width,py=(e.clientY-r.top)*cv.height/r.height;return {v,px,py}}
cv.onpointerdown=e=>{if(!S)return;const {v,px,py}=pick(e);let best=null,bd=30;for(const t of S.attackers){const [x,y]=P(v,t.pos[0],t.pos[2]);const d=Math.hypot(x-px,y-py);if(d<bd){bd=d;best=t.name}}drag=best;if(drag)cv.setPointerCapture(e.pointerId)};
cv.onpointermove=e=>{if(!drag)return;const {v,px,py}=pick(e);const [x,z]=W(v,px,py);cmd({op:'move',name:drag,x,z})};
cv.onpointerup=()=>drag=null;
document.getElementById('atk').onclick=()=>cmd({op:'attack',on:!S.attack});
document.getElementById('rst').onclick=()=>cmd({op:'reset'});
document.getElementById('spd').oninput=e=>cmd({op:'speed',v:e.target.value});
document.getElementById('wv').oninput=e=>cmd({op:'weave',v:e.target.value});
poll();
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attackers", required=True, help="comma list, e.g. TELLO-8,TELLO-9")
    ap.add_argument("--map", default=None, help="map.yaml (default config/map.yaml)")
    ap.add_argument("--tag10_height_m", type=float, default=TAG10_HEIGHT_M,
                    help="tag 10 centre above the floor (venue 1.22; measure it at home)")
    ap.add_argument("--room", default=None,
                    help="no-fly box override x_min,x_max,z_max in metres (x sideways from tag 10, "
                         "z out from its wall); write it with = because of the minus: --room=-1.5,1.5,3.5")
    ap.add_argument("--site", default="0,1.0", help="table centre x,z (same as defend.py)")
    ap.add_argument("--feed", default="127.0.0.1:9100", help="publish attacker telemetry here")
    ap.add_argument("--status_port", type=int, default=9101, help="receive defend.py status here")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--port", type=int, default=8790, help="web page port")
    args = ap.parse_args()

    from tello_calibration.tags.mapping import TagMap
    from tello_calibration.utils.config import CONFIG_DIR
    tag_map = TagMap.load(Path(args.map) if args.map else CONFIG_DIR / "map.yaml")
    tag_poses = {int(k): np.asarray(v) for k, v in tag_map.poses.items()}
    sx, sz = (float(v) for v in args.site.split(","))
    fence = default_fence(tag_poses, args.tag10_height_m)
    if args.room:
        x0, x1, z1 = (float(v) for v in args.room.split(","))
        fence.lo[0], fence.hi[0], fence.hi[2] = x0, x1, z1
    host, _, port = args.feed.rpartition(":")
    world = World([s.strip() for s in args.attackers.split(",") if s.strip()], tag_poses,
                  np.array([sx, 0.0, sz]), fence, (host, int(port)), args.hz, args.tag10_height_m)
    stop = threading.Event()

    def publish():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        prev = time.time()
        while not stop.is_set():
            now = time.time()
            world.step(now - prev)
            prev = now
            try:
                sock.sendto(world.telemetry(), world.feed)
            except OSError:
                pass
            stop.wait(1.0 / world.hz)

    def listen():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", args.status_port))
        sock.settimeout(0.5)
        while not stop.is_set():
            try:
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue
            try:
                world.status, world.status_t = json.loads(data), time.time()
            except ValueError:
                continue
            if world.status.get("site"):              # defend.py's table position (site auto)
                with world.lock:
                    world.site = np.asarray(world.status["site"], float)

    threading.Thread(target=publish, daemon=True).start()
    threading.Thread(target=listen, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/state"):
                self._send(json.dumps(world.snapshot()).encode(), "application/json")
            else:
                self._send(PAGE.encode(), "text/html; charset=utf-8")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                world.command(json.loads(self.rfile.read(n) or b"{}"))
            except (ValueError, KeyError) as exc:
                print(f"bad command: {exc}", flush=True)
            self._send(b"{}", "application/json")

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"HIL GUI http://127.0.0.1:{args.port}/  attackers {world.names} -> telemetry {args.feed}; "
          f"defend.py status on :{args.status_port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
