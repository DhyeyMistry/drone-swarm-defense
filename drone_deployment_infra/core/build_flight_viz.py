#!/usr/bin/env python3
"""Build a self-contained flight visualization from extracted tracks.

Reads the per-frame tracks written by extract_track.py and bakes them into one
standalone HTML file: a top-down map of where the drone thought it was in the
tag-10 frame, a timeline of when it saw the tag, and a WiFi lane. The data is
inlined so the page opens straight from disk with no server.

  .venv/bin/python scripts/build_flight_viz.py      # newest 3 tracks
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Everything a flight produces lives in recordings/: the .avi, the live track
# (.live.jsonl), the offline reconstruction (.track.jsonl) and the training data
# (.train.jsonl). This reads the track files from there and writes the page there too.
TRACK_DIR = ROOT / "recordings"
OUT = TRACK_DIR / "flight_viz.html"


def wifi_spans(rows):
    """Collapse per-row wifi flags into {start, end, up} spans on the time axis."""
    spans = []
    cur = None
    for row in rows:
        up = row.get("wifi")
        if up is None:
            continue
        if cur is None or cur["up"] != bool(up):
            if cur is not None:
                cur["end"] = row["t"]
                spans.append(cur)
            cur = {"start": row["t"], "end": row["t"], "up": bool(up)}
        else:
            cur["end"] = row["t"]
    if cur is not None:
        spans.append(cur)
    return spans


def session_of(name):
    return name.split("_", 1)[0]


def label_of(path):
    """Human label like '013 TELLO-1' (plus ' live' for a live track)."""
    stem = path.stem                      # '<sess>_<drone>.live' or '<sess>_<drone>_<stamp>.track'
    live = stem.endswith(".live")
    if live:
        stem = stem[:-len(".live")]
    elif stem.endswith(".track"):
        stem = stem[:-len(".track")]
    parts = stem.split("_")
    drone = parts[1] if len(parts) > 1 else stem
    return f"{session_of(stem)} {drone}" + (" live" if live else "")


def all_tracks():
    """One track per session in recordings/, preferring the live one over the offline
    reconstruction. Only *.live.jsonl / *.track.jsonl count -- never the .train.jsonl
    action logs, which share the folder but have a different schema."""
    by_session = {}
    for path in list(TRACK_DIR.glob("*.live.jsonl")) + list(TRACK_DIR.glob("*.track.jsonl")):
        session = session_of(path.stem)
        live = path.stem.endswith(".live")
        if session not in by_session or (live and not by_session[session][1]):
            by_session[session] = (path, live)
    return [by_session[s][0] for s in sorted(by_session)]


def load_tracks(paths):
    flights = []
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        # Live tracks (from tello_dual_video) carry a per-row wifi flag; offline
        # reconstructions from old recordings do not.
        has_wifi = any("wifi" in row for row in rows)
        wifi = {"spans": wifi_spans(rows)} if has_wifi else None
        flights.append({"name": label_of(path), "rows": rows, "wifi": wifi})
    return flights


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Flight tracks</title>
<style>
  :root{color-scheme:light dark}
  *{box-sizing:border-box}
  body{margin:0;font:14px/1.4 ui-sans-serif,system-ui,sans-serif;background:#0f1115;color:#e7e7ea}
  header{padding:12px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #262a33}
  h1{font-size:16px;margin:0;font-weight:600}
  select,button{font:inherit;background:#1b1f27;color:#e7e7ea;border:1px solid #333a47;border-radius:6px;padding:5px 9px}
  button{cursor:pointer}
  main{padding:16px;display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:16px}
  @media (max-width:760px){main{grid-template-columns:1fr}}
  .panel{background:#151821;border:1px solid #262a33;border-radius:10px;padding:12px}
  canvas{width:100%;height:auto;display:block;background:#0b0d11;border-radius:8px}
  .lanes{display:flex;flex-direction:column;gap:10px;margin-top:14px}
  .lane h3{margin:0 0 4px;font-size:12px;font-weight:600;color:#9aa0ac;text-transform:uppercase;letter-spacing:.04em}
  .track{position:relative;height:16px;border-radius:4px;overflow:hidden;background:#20242e}
  .cursor{position:absolute;top:-3px;bottom:-3px;width:2px;background:#ffd166;pointer-events:none}
  .controls{display:flex;gap:10px;align-items:center;margin-top:14px}
  input[type=range]{flex:1}
  .stat{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #20242e}
  .stat b{font-variant-numeric:tabular-nums}
  .key{color:#9aa0ac}
  .legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;font-size:12px;color:#9aa0ac}
  .dot{display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:-1px;margin-right:4px}
  small{color:#9aa0ac}
</style></head>
<body>
<header>
  <h1>Flight tracks &mdash; tag-10 frame</h1>
  <label>Recording <select id="pick"></select></label>
  <button id="play">Play</button>
  <small id="note"></small>
</header>
<main>
  <div class="panel">
    <canvas id="map" width="900" height="560"></canvas>
    <div class="legend">
      <span><span class="dot" style="background:#e03131"></span>tag 10 (origin)</span>
      <span><span class="dot" style="background:#4dabf7"></span>drone path (where it thought it was)</span>
      <span><span class="dot" style="background:#ffd166"></span>current position</span>
    </div>
    <div class="controls">
      <span id="tnow" class="key">0.0 s</span>
      <input id="scrub" type="range" min="0" max="100" value="0" step="1">
    </div>
    <div class="lanes">
      <div class="lane"><h3>Saw an AprilTag</h3><div class="track" id="laneTag"><div class="cursor" id="curTag"></div></div></div>
      <div class="lane"><h3>Tag-10 pose (localized)</h3><div class="track" id="lanePose"><div class="cursor" id="curPose"></div></div></div>
      <div class="lane"><h3>WiFi connected</h3><div class="track" id="laneWifi"><div class="cursor" id="curWifi"></div></div></div>
    </div>
  </div>
  <div class="panel">
    <div class="stat"><span class="key">time</span><b id="s_t">&mdash;</b></div>
    <div class="stat"><span class="key">range to tag 10</span><b id="s_r">&mdash;</b></div>
    <div class="stat"><span class="key">x (right)</span><b id="s_x">&mdash;</b></div>
    <div class="stat"><span class="key">in front (-z)</span><b id="s_z">&mdash;</b></div>
    <div class="stat"><span class="key">tags in view</span><b id="s_ids">&mdash;</b></div>
    <p><small id="summary"></small></p>
  </div>
</main>
<script>
const DATA = __DATA__;
const names = Object.keys(DATA);
const pick = document.getElementById('pick');
names.forEach(n => { const o=document.createElement('option'); o.value=n; o.textContent=n; pick.appendChild(o); });
const cv = document.getElementById('map'), ctx = cv.getContext('2d');
let cur = names[0], playing = false, raf = null;

function bounds(rows){
  let xs=[], ds=[];
  rows.forEach(r=>{ if(r.range_m!==undefined){ xs.push(r.x); ds.push(-r.z); } });
  if(!xs.length) return {xmin:-1,xmax:1,dmax:4};
  const xmin=Math.min(-0.5,...xs), xmax=Math.max(0.5,...xs), dmax=Math.max(1,...ds);
  return {xmin,xmax,dmax};
}
function project(x,d,b){
  const pad=40, w=cv.width-2*pad, h=cv.height-2*pad;
  const px = pad + (x-b.xmin)/(b.xmax-b.xmin||1)*w;
  const py = pad + (1 - d/(b.dmax*1.05))*h;   // tag (d=0) at the bottom, far away at top
  return [px,py];
}
function draw(rows,b,t){
  ctx.clearRect(0,0,cv.width,cv.height);
  // grid rings at 1,2,3.. m distance
  ctx.strokeStyle='#20242e'; ctx.fillStyle='#5a6270'; ctx.font='11px sans-serif';
  for(let d=0; d<=b.dmax; d++){ const [,py]=project(0,d,b); ctx.beginPath(); ctx.moveTo(30,py); ctx.lineTo(cv.width-20,py); ctx.stroke(); ctx.fillText(d+' m',8,py+3); }
  // path
  ctx.strokeStyle='#4dabf7'; ctx.lineWidth=2; ctx.beginPath(); let started=false;
  rows.forEach(r=>{ if(r.range_m===undefined){started=false;return;} const [px,py]=project(r.x,-r.z,b); if(!started){ctx.moveTo(px,py);started=true;}else ctx.lineTo(px,py); });
  ctx.stroke();
  // tag 10 at origin
  const [ox,oy]=project(0,0,b); ctx.fillStyle='#e03131'; ctx.fillRect(ox-7,oy-7,14,14);
  // current position: nearest sample with a pose at/<= t
  let c=null; for(const r of rows){ if(r.t<=t && r.range_m!==undefined) c=r; if(r.t>t) break; }
  if(c){ const [px,py]=project(c.x,-c.z,b); ctx.fillStyle='#ffd166'; ctx.beginPath(); ctx.arc(px,py,7,0,7); ctx.fill(); }
  return c;
}
function paintLane(id,rows,test,color){
  const el=document.getElementById(id); [...el.querySelectorAll('.seg')].forEach(s=>s.remove());
  if(!rows.length) return; const T=rows[rows.length-1].t||1;
  rows.forEach((r,i)=>{ if(!test(r)) return; const a=r.t/T*100, next=(rows[i+1]?rows[i+1].t:r.t+0.1)/T*100;
    const s=document.createElement('div'); s.className='seg'; s.style.cssText=`position:absolute;top:0;bottom:0;left:${a}%;width:${Math.max(0.4,next-a)}%;background:${color}`; el.appendChild(s); });
}
function paintWifi(flight){
  const el=document.getElementById('laneWifi'); [...el.querySelectorAll('.seg')].forEach(s=>s.remove());
  const rows=flight.rows, T=rows.length?rows[rows.length-1].t:1;
  if(flight.wifi && flight.wifi.spans){
    flight.wifi.spans.forEach(sp=>{ const a=sp.start/T*100,w=(sp.end-sp.start)/T*100;
      const s=document.createElement('div'); s.className='seg'; s.style.cssText=`position:absolute;top:0;bottom:0;left:${a}%;width:${Math.max(0.4,w)}%;background:${sp.up?'#2f9e44':'#e8590c'}`; el.appendChild(s); });
    document.getElementById('note').textContent='';
  } else {
    const s=document.createElement('div'); s.className='seg'; s.style.cssText='position:absolute;inset:0;background:repeating-linear-gradient(45deg,#2a2f3a,#2a2f3a 6px,#20242e 6px,#20242e 12px)';
    el.appendChild(s);
  }
}
function render(){
  const flight=DATA[cur], rows=flight.rows, b=bounds(rows);
  const scrub=document.getElementById('scrub'); const T=rows.length?rows[rows.length-1].t:1;
  const t=scrub.value/100*T;
  const c=draw(rows,b,t);
  ['curTag','curPose','curWifi'].forEach(id=>document.getElementById(id).style.left=(t/T*100)+'%');
  document.getElementById('tnow').textContent=t.toFixed(1)+' s';
  document.getElementById('s_t').textContent=t.toFixed(2)+' s';
  if(c){ document.getElementById('s_r').textContent=c.range_m.toFixed(2)+' m';
    document.getElementById('s_x').textContent=c.x.toFixed(2)+' m';
    document.getElementById('s_z').textContent=(-c.z).toFixed(2)+' m';
    document.getElementById('s_ids').textContent=(c.ids||[]).join(', ')||'none'; }
  else { ['s_r','s_x','s_z','s_ids'].forEach(id=>document.getElementById(id).textContent='—'); }
}
function loadFlight(){
  const flight=DATA[cur], rows=flight.rows;
  paintLane('laneTag',rows,r=>r.ids&&r.ids.length,'#868e96');
  paintLane('lanePose',rows,r=>r.range_m!==undefined,'#2f9e44');
  paintWifi(flight);
  const poses=rows.filter(r=>r.range_m!==undefined);
  const T=rows.length?rows[rows.length-1].t:0;
  const rmin=poses.length?Math.min(...poses.map(r=>r.range_m)):0;
  document.getElementById('summary').textContent=`${rows.length} frames over ${T.toFixed(1)} s · ${poses.length} with a tag-10 pose · closest approach ${rmin.toFixed(2)} m`;
  render();
}
pick.onchange=()=>{cur=pick.value;document.getElementById('scrub').value=0;loadFlight();};
document.getElementById('scrub').oninput=render;
document.getElementById('play').onclick=function(){
  playing=!playing; this.textContent=playing?'Pause':'Play';
  const step=()=>{ if(!playing)return; const s=document.getElementById('scrub'); s.value=(+s.value+0.5)%100; render(); raf=requestAnimationFrame(step); };
  if(playing) step(); else cancelAnimationFrame(raf);
};
loadFlight();
</script>
</body></html>
"""


def main():
    args = [Path(a) for a in sys.argv[1:] if not a.startswith("-")]
    paths = args if args else all_tracks()
    if not paths:
        raise SystemExit("no tracks in recordings/ -- run extract_track.py first")
    flights = load_tracks(paths)
    data = {f["name"]: {"rows": f["rows"], "wifi": f["wifi"]} for f in flights}
    OUT.write_text(PAGE.replace("__DATA__", json.dumps(data)))
    print(f"wrote {OUT} ({len(flights)} flights)")


if __name__ == "__main__":
    main()
