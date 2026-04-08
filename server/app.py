"""
server/app.py — OpenEnv HTTP API for Indian Power Grid Load Balancer
Exposes reset() / step() / state() / tasks() / health() as REST endpoints.
Runs on 0.0.0.0:7860 (Hugging Face Space compatible).
"""

import os
import sys
from typing import Any, Dict, Optional

# Ensure project root is in Python path so 'grid_env' is importable inside container
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from grid_env import GridAction, GridEnv, TASK_IDS, _TOPOLOGIES


app = FastAPI(
    title="OpenEnv: Indian Power Grid Load Balancer",
    version="1.0.0",
    description=(
        "An OpenEnv-compliant environment where an AI agent manages a "
        "simulated Maharashtra regional electricity grid, preventing blackouts "
        "via load shedding and rerouting. Three tasks: easy → medium → hard."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env: Optional[GridEnv] = None
_current_task: str = "single_substation"


class ResetRequest(BaseModel):
    task: Optional[str] = None
    seed: Optional[int] = None


@app.post("/reset", summary="Start a new episode")
async def reset(body: Optional[ResetRequest] = None):
    global _env, _current_task
    task = _current_task
    seed = None
    if body:
        if body.task:
            if body.task not in TASK_IDS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown task '{body.task}'. Valid: {TASK_IDS}",
                )
            task = body.task
            _current_task = task
        seed = body.seed
    _env = GridEnv(task=task, seed=seed)
    obs = _env.reset()
    return obs.model_dump()


@app.post("/step", summary="Execute one agent action")
async def step(action: GridAction):
    if _env is None:
        raise HTTPException(status_code=400, detail="No active episode. POST /reset first.")
    if _env._ep.get("done", False):
        raise HTTPException(status_code=400, detail="Episode finished. POST /reset to start a new one.")
    result = _env.step(action)
    return result.model_dump()


@app.get("/state", summary="Inspect current environment state")
async def state():
    if _env is None:
        raise HTTPException(status_code=400, detail="No active episode. POST /reset first.")
    return _env.state()


@app.get("/tasks", summary="List available tasks")
async def tasks():
    return {
        "tasks": [
            {
                "id":          tid,
                "difficulty":  ["easy", "medium", "hard"][i],
                "description": meta["description"],
                "stations":    len(meta["stations"]),
                "max_steps":   meta["max_steps"],
                "cascade":     meta["cascade"],
            }
            for i, (tid, meta) in enumerate(_TOPOLOGIES.items())
        ]
    }


@app.get("/health", summary="Liveness probe")
async def health():
    return {"status": "ok", "service": "openenv-power-grid"}


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/web")


@app.get("/web", include_in_schema=False)
async def web():
    PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Maharashtra Power Grid &mdash; OpenEnv</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin=""/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#050d1a;--surf:#0d1b2e;--bdr:#1a2d45;
  --txt:#e2edf8;--mut:#5a7a9a;
  --blue:#3b82f6;--bglow:rgba(59,130,246,.22);
  --green:#22d3a0;--gglow:rgba(34,211,160,.2);
  --yel:#fbbf24;--red:#f43f5e;
}
html,body{height:100%;font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--txt);overflow-x:hidden}

/* ── HEADER ── */
.header{text-align:center;padding:32px 20px 20px;position:relative;z-index:2}
.pill{display:inline-flex;align-items:center;gap:6px;background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.3);color:var(--blue);border-radius:999px;padding:4px 14px;font-size:.72rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;margin-bottom:14px}
.pulsedot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pd 2s ease-in-out infinite}
@keyframes pd{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.75)}}
h1{font-size:clamp(1.7rem,4vw,2.6rem);font-weight:800;letter-spacing:-.03em;background:linear-gradient(135deg,#e2edf8 0%,#7cb9ff 50%,var(--green) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:8px}
.sub{color:var(--mut);font-size:.9rem;line-height:1.7;max-width:600px;margin:0 auto}

/* ── MAIN LAYOUT ── */
.layout{display:grid;grid-template-columns:1fr 320px;gap:0;max-width:1200px;margin:0 auto;padding:0 16px 40px}
@media(max-width:860px){.layout{grid-template-columns:1fr}}

/* ── MAP ── */
#map{height:520px;border-radius:16px;border:1px solid var(--bdr);overflow:hidden;background:#0a1628}
.leaflet-container{background:#0a1628!important}
.leaflet-tile-pane{filter:brightness(.75) saturate(.6)}

/* ── SIDEBAR ── */
.sidebar{padding:0 0 0 16px;display:flex;flex-direction:column;gap:12px}
@media(max-width:860px){.sidebar{padding:16px 0 0 0}}

/* ── CARDS ── */
.card{background:var(--surf);border:1px solid var(--bdr);border-radius:14px;padding:18px}
.card-title{font-size:.7rem;font-weight:700;color:var(--mut);text-transform:uppercase;letter-spacing:.08em;margin-bottom:14px}

/* ── SERVER STATUS ── */
.sbar{display:flex;align-items:center;gap:8px;margin-bottom:4px}
.sdot{width:8px;height:8px;border-radius:50%;background:var(--mut);transition:background .4s;flex-shrink:0}
.sdot.ok{background:var(--green);box-shadow:0 0 8px var(--gglow)}.sdot.err{background:var(--red)}
#stxt{font-family:'JetBrains Mono',monospace;font-size:.73rem;color:var(--mut)}

/* ── GRID STATS ── */
.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.scard{background:rgba(255,255,255,.03);border:1px solid var(--bdr);border-radius:10px;padding:12px;text-align:center}
.sval{font-size:1.5rem;font-weight:800;color:var(--blue);letter-spacing:-.03em}
.slbl{font-size:.65rem;color:var(--mut);text-transform:uppercase;letter-spacing:.07em;margin-top:2px}

/* ── RISK BADGE ── */
#risk-badge{display:inline-block;padding:4px 12px;border-radius:999px;font-size:.72rem;font-weight:700;background:rgba(34,211,160,.1);color:var(--green);border:1px solid rgba(34,211,160,.3);transition:all .3s}
#risk-badge.medium{background:rgba(251,191,36,.1);color:var(--yel);border-color:rgba(251,191,36,.3)}
#risk-badge.high{background:rgba(244,63,94,.15);color:var(--red);border-color:rgba(244,63,94,.3)}
#risk-badge.critical{background:rgba(244,63,94,.25);color:#ff2d55;border-color:rgba(255,45,85,.5);animation:pulse-red 1s infinite}
@keyframes pulse-red{0%,100%{box-shadow:0 0 0 0 rgba(244,63,94,.4)}50%{box-shadow:0 0 0 8px rgba(244,63,94,0)}}

/* ── PROGRESS BAR ── */
.prog-wrap{background:rgba(255,255,255,.05);border-radius:4px;height:6px;margin-top:8px;overflow:hidden}
.prog-bar{height:100%;background:linear-gradient(90deg,var(--blue),var(--green));border-radius:4px;transition:width .4s}

/* ── STATION LIST ── */
.st-list{display:flex;flex-direction:column;gap:6px;max-height:200px;overflow-y:auto}
.st-list::-webkit-scrollbar{width:4px}.st-list::-webkit-scrollbar-track{background:transparent}.st-list::-webkit-scrollbar-thumb{background:var(--bdr);border-radius:2px}
.st-row{display:flex;align-items:center;gap:8px;padding:6px 8px;border-radius:8px;border:1px solid transparent;transition:border-color .2s}
.st-row:hover{border-color:var(--bdr)}
.st-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.st-name{font-family:'JetBrains Mono',monospace;font-size:.72rem;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.st-bar-wrap{width:60px;height:5px;background:rgba(255,255,255,.1);border-radius:3px;flex-shrink:0}
.st-bar{height:100%;border-radius:3px;transition:width .4s,background .4s}
.st-pct{font-family:'JetBrains Mono',monospace;font-size:.68rem;color:var(--mut);width:32px;text-align:right;flex-shrink:0}

/* ── CONTROLS ── */
.ctrl-row{display:flex;gap:8px;flex-wrap:wrap}
.btn{display:inline-flex;align-items:center;gap:6px;padding:10px 18px;border-radius:9px;font-size:.82rem;font-weight:600;border:none;font-family:inherit;cursor:pointer;transition:all .2s;text-decoration:none}
.bpri{background:var(--blue);color:#fff;box-shadow:0 0 16px var(--bglow)}.bpri:hover:not(:disabled){background:#2563eb;transform:translateY(-1px)}
.bsec{background:transparent;color:var(--mut);border:1px solid var(--bdr)}.bsec:hover{border-color:var(--mut);color:var(--txt)}
.bdanger{background:rgba(244,63,94,.1);color:var(--red);border:1px solid rgba(244,63,94,.3)}.bdanger:hover:not(:disabled){background:rgba(244,63,94,.2)}
.btn:disabled{opacity:.4;cursor:not-allowed}

/* ── LAST ACTION ── */
#last-action{font-family:'JetBrains Mono',monospace;font-size:.7rem;color:var(--mut);line-height:1.6;min-height:36px;padding:8px;background:rgba(0,0,0,.2);border-radius:8px;border:1px solid var(--bdr);word-break:break-word}

/* ── FOOTER LINKS ── */
.footer{text-align:center;padding:6px 20px 28px;display:flex;gap:12px;justify-content:center;flex-wrap:wrap}

/* ── LEAFLET CUSTOM ── */
.leaflet-popup-content-wrapper{background:#0d1b2e;color:#e2edf8;border:1px solid #1a2d45;border-radius:10px;box-shadow:0 8px 32px rgba(0,0,0,.5)}
.leaflet-popup-tip{background:#0d1b2e}
.leaflet-popup-content{margin:10px 14px;font-size:13px;font-family:'JetBrains Mono',monospace}
.popup-name{font-weight:700;font-size:.85rem;margin-bottom:6px;color:#e2edf8}
.popup-row{display:flex;justify-content:space-between;gap:16px;font-size:.72rem;margin-bottom:3px}
.popup-key{color:#5a7a9a}.popup-val{font-weight:600}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="pill"><div class="pulsedot"></div>OpenEnv &mdash; Maharashtra Power Grid</div>
  <h1>Maharashtra Regional Grid<br>Live Load Monitor</h1>
  <p class="sub">12-substation cascade outage environment mapped to real Maharashtra geography. Watch load propagate across the Western grid in real-time.</p>
</div>

<!-- MAIN CONTENT -->
<div class="layout">

  <!-- MAP PANEL -->
  <div>
    <div id="map"></div>
  </div>

  <!-- SIDEBAR -->
  <div class="sidebar">

    <!-- Server Health -->
    <div class="card">
      <div class="card-title">Server Status</div>
      <div class="sbar"><div class="sdot" id="sdot"></div><span id="stxt">Checking&hellip;</span></div>
    </div>

    <!-- Simulation Controls -->
    <div class="card">
      <div class="card-title">Simulation Controls</div>
      <div class="ctrl-row">
        <button class="btn bpri" id="btn-start" onclick="startSim()">&#9654; Start Cascade Sim</button>
        <button class="btn bdanger" id="btn-reset" onclick="resetSim()" disabled>&#8635; Reset</button>
      </div>
      <div style="margin-top:10px">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">
          <span style="font-size:.72rem;color:var(--mut)">Blackout Risk</span>
          <span id="risk-badge">N/A</span>
        </div>
        <div style="font-size:.72rem;color:var(--mut);margin-bottom:4px">Step Progress</div>
        <div class="prog-wrap"><div class="prog-bar" id="prog" style="width:0%"></div></div>
        <div style="font-size:.68rem;color:var(--mut);margin-top:4px;font-family:'JetBrains Mono',monospace" id="step-lbl">&mdash;</div>
      </div>
    </div>

    <!-- Grid Stats -->
    <div class="card">
      <div class="card-title">Grid Metrics</div>
      <div class="stat-grid">
        <div class="scard"><div class="sval" id="m-load">--</div><div class="slbl">Grid Load %</div></div>
        <div class="scard"><div class="sval" id="m-shed">--</div><div class="slbl">Total Shed MW</div></div>
        <div class="scard"><div class="sval" id="m-faults">--</div><div class="slbl">Active Faults</div></div>
        <div class="scard"><div class="sval" id="m-blackouts">--</div><div class="slbl">Blackouts</div></div>
      </div>
    </div>

    <!-- Station List -->
    <div class="card" style="flex:1">
      <div class="card-title">Substation Status</div>
      <div class="st-list" id="st-list">
        <div style="color:var(--mut);font-size:.78rem;text-align:center;padding:20px 0">Start simulation to see live data</div>
      </div>
    </div>

    <!-- Last Action -->
    <div class="card">
      <div class="card-title">Last Action Message</div>
      <div id="last-action">No active episode.</div>
    </div>

  </div>
</div>

<!-- FOOTER -->
<div class="footer">
  <a href="/docs" class="btn bpri">&#9889; API Docs</a>
  <a href="https://github.com/TheKshitij/power-grid-india" target="_blank" class="btn bsec">GitHub &rarr;</a>
</div>

<script>
// ============================================================
// GEOGRAPHIC COORDINATES — Real Maharashtra locations
// ============================================================
const STATIONS = {
  0:  {name:"Mumbai-Central", lat:18.9696, lng:72.8193, cap:1200, neighbors:[1,4]},
  1:  {name:"Thane",          lat:19.2183, lng:72.9781, cap:900,  neighbors:[0,2,5]},
  2:  {name:"Pune",           lat:18.5204, lng:73.8567, cap:800,  neighbors:[1,3,6]},
  3:  {name:"Nashik",         lat:19.9975, lng:73.7898, cap:600,  neighbors:[2,7]},
  4:  {name:"Navi-Mumbai",    lat:19.0330, lng:73.0297, cap:700,  neighbors:[0,5,8]},
  5:  {name:"Raigad",         lat:18.5179, lng:73.1184, cap:500,  neighbors:[1,4,9]},
  6:  {name:"Satara",         lat:17.6805, lng:74.0183, cap:450,  neighbors:[2,7,10]},
  7:  {name:"Solapur",        lat:17.6599, lng:75.9064, cap:550,  neighbors:[3,6,11]},
  8:  {name:"Panvel",         lat:18.9894, lng:73.1175, cap:400,  neighbors:[4,9]},
  9:  {name:"Alibag",         lat:18.6414, lng:72.8722, cap:350,  neighbors:[5,8]},
  10: {name:"Kolhapur",       lat:16.7050, lng:74.2433, cap:500,  neighbors:[6,11]},
  11: {name:"Sangli",         lat:16.8524, lng:74.5815, cap:450,  neighbors:[7,10]}
};

// ============================================================
// MAP INITIALISATION
// ============================================================
const map = L.map('map', {zoomControl:true, attributionControl:true}).setView([18.4, 73.9], 7);

// Dark CartoDB basemap
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
  subdomains: 'abcd',
  maxZoom: 19
}).addTo(map);

// Fit map to Maharashtra substation bounds with padding
const lats = Object.values(STATIONS).map(s=>s.lat);
const lngs = Object.values(STATIONS).map(s=>s.lng);
map.fitBounds([[Math.min(...lats)-0.3, Math.min(...lngs)-0.3],[Math.max(...lats)+0.3, Math.max(...lngs)+0.3]]);

// ── Draw static transmission lines (will be updated dynamically) ──
const lineLayer = L.layerGroup().addTo(map);
const drawnEdges = new Set();
const lines = {};

function drawLines(colorFn) {
  lineLayer.clearLayers();
  drawnEdges.clear();
  Object.entries(STATIONS).forEach(([id, st]) => {
    st.neighbors.forEach(nb => {
      const key = [Math.min(+id,nb), Math.max(+id,nb)].join('-');
      if (!drawnEdges.has(key)) {
        drawnEdges.add(key);
        const nb_st = STATIONS[nb];
        const color = colorFn ? colorFn(+id, nb) : '#1a2d45';
        const line = L.polyline([[st.lat,st.lng],[nb_st.lat,nb_st.lng]], {
          color, weight:2.5, opacity:.8, dashArray: null
        }).addTo(lineLayer);
        lines[key] = line;
      }
    });
  });
}
drawLines(() => '#1a3050');  // Initial static lines

// ── Station markers ──
const markers = {};

function loadColor(pct) {
  if (pct >= 100) return '#f43f5e';
  if (pct >= 90)  return '#ff6b35';
  if (pct >= 75)  return '#fbbf24';
  if (pct >= 60)  return '#60d394';
  return '#22d3a0';
}

function markerRadius(cap) {
  if (cap >= 1000) return 16;
  if (cap >= 700)  return 13;
  if (cap >= 500)  return 11;
  return 9;
}

function makePopup(st, data) {
  const pct = data ? data.load_pct.toFixed(1) : '--';
  const load = data ? data.load_mw.toFixed(0) : '--';
  const shed = data ? data.shed_mw.toFixed(0) : '0';
  const status = data ? data.status : 'unknown';
  return \`<div class="popup-name">&#9889; \${st.name}</div>
    <div class="popup-row"><span class="popup-key">Capacity</span><span class="popup-val">\${st.cap} MW</span></div>
    <div class="popup-row"><span class="popup-key">Load</span><span class="popup-val">\${load} MW (\${pct}%)</span></div>
    <div class="popup-row"><span class="popup-key">Shed</span><span class="popup-val">\${shed} MW</span></div>
    <div class="popup-row"><span class="popup-key">Status</span><span class="popup-val" style="color:\${status==='fault'?'#f43f5e':status==='stressed'?'#fbbf24':'#22d3a0'}">\${status.toUpperCase()}</span></div>\`;
}

Object.entries(STATIONS).forEach(([id, st]) => {
  const circle = L.circleMarker([st.lat, st.lng], {
    radius: markerRadius(st.cap),
    fillColor: '#22d3a0',
    color: '#0d1b2e',
    weight: 2,
    opacity: 1,
    fillOpacity: 0.85
  }).addTo(map).bindPopup(makePopup(st, null), {maxWidth:220});

  // Station label
  const icon = L.divIcon({
    className:'',
    html: \`<div style="color:#94a3b8;font-size:10px;font-family:'JetBrains Mono',monospace;white-space:nowrap;text-shadow:0 0 4px #000;margin-top:2px;pointer-events:none">\${st.name}</div>\`,
    iconAnchor:[-4, 14]
  });
  L.marker([st.lat, st.lng], {icon, interactive:false}).addTo(map);

  markers[+id] = circle;
});

// ============================================================
// HEALTH CHECK
// ============================================================
async function ping() {
  const d = document.getElementById('sdot'), t = document.getElementById('stxt');
  try {
    const r = await fetch('/health'), j = await r.json();
    d.className = 'sdot ok';
    t.innerHTML = '<span style="color:#22d3a0">&#9679; LIVE</span>&nbsp; ' + j.service;
  } catch(e) {
    d.className = 'sdot err';
    t.textContent = 'Server unreachable';
  }
}
ping(); setInterval(ping, 12000);

// ============================================================
// SIMULATION
// ============================================================
let simInterval = null;
let simRunning = false;

async function startSim() {
  document.getElementById('btn-start').disabled = true;
  document.getElementById('btn-reset').disabled = false;
  document.getElementById('last-action').textContent = 'Initialising cascade_outage episode...';

  try {
    const res = await fetch('/reset', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({task:'cascade_outage'})
    });
    if (!res.ok) throw new Error(await res.text());
    const obs = await res.json();
    updateUI(obs);
    simRunning = true;
    simInterval = setInterval(pollState, 2500);
  } catch(e) {
    document.getElementById('last-action').textContent = 'Error: ' + e.message;
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-reset').disabled = true;
  }
}

async function resetSim() {
  if (simInterval) { clearInterval(simInterval); simInterval = null; }
  simRunning = false;
  document.getElementById('btn-start').disabled = false;
  document.getElementById('btn-reset').disabled = true;
  document.getElementById('step-lbl').textContent = '\u2014';
  document.getElementById('prog').style.width = '0%';
  document.getElementById('m-load').textContent = '--';
  document.getElementById('m-shed').textContent = '--';
  document.getElementById('m-faults').textContent = '--';
  document.getElementById('m-blackouts').textContent = '--';
  document.getElementById('last-action').textContent = 'No active episode.';
  document.getElementById('risk-badge').textContent = 'N/A';
  document.getElementById('risk-badge').className = '';
  document.getElementById('st-list').innerHTML = '<div style="color:var(--mut);font-size:.78rem;text-align:center;padding:20px 0">Start simulation to see live data</div>';
  // Reset all markers to default
  Object.values(markers).forEach(m => m.setStyle({fillColor:'#22d3a0'}));
  drawLines(() => '#1a3050');
}

async function pollState() {
  try {
    const res = await fetch('/state');
    if (!res.ok) { if (simInterval) { clearInterval(simInterval); simInterval = null; } return; }
    const obs = await res.json();
    updateUI(obs);
    if (obs.step >= obs.max_steps || obs.active_faults.length >= 6) {
      clearInterval(simInterval); simInterval = null;
      document.getElementById('last-action').textContent = 'Episode ended. Click Reset to start again.';
      document.getElementById('btn-start').textContent = '\u23f9 Episode Over';
    }
  } catch(e) { /* ignore transient errors */ }
}

function updateUI(obs) {
  // --- Progress ---
  const pct = Math.min(100, (obs.step / obs.max_steps) * 100).toFixed(0);
  document.getElementById('prog').style.width = pct + '%';
  document.getElementById('step-lbl').textContent = 'Step ' + obs.step + ' / ' + obs.max_steps + '  \u2014  ' + obs.time_label;

  // --- Risk ---
  const risk = obs.blackout_risk;
  const rb = document.getElementById('risk-badge');
  rb.textContent = risk.toUpperCase();
  rb.className = risk === 'low' ? '' : risk;

  // --- Metrics ---
  document.getElementById('m-load').textContent = obs.grid_load_pct.toFixed(1) + '%';
  document.getElementById('m-shed').textContent = obs.total_shed_mw.toFixed(0);
  document.getElementById('m-faults').textContent = obs.active_faults.length;
  document.getElementById('m-blackouts').textContent = obs.episode_blackouts || 0;

  // --- Last action ---
  document.getElementById('last-action').textContent = obs.last_action_message || '\u2014';

  // --- Station list + Map markers ---
  const listEl = document.getElementById('st-list');
  listEl.innerHTML = '';
  const byId = {};
  obs.substations.forEach(st => { byId[st.id] = st; });

  obs.substations.forEach(st => {
    const color = loadColor(st.load_pct);
    // Update marker
    if (markers[st.id]) {
      markers[st.id].setStyle({fillColor: color, color: st.status === 'fault' ? '#f43f5e' : '#0d1b2e', weight: st.status === 'fault' ? 3 : 2});
      markers[st.id].setPopupContent(makePopup(STATIONS[st.id], st));
      // Pulse effect for stressed/fault
      if (st.status !== 'normal') {
        const orig = markerRadius(STATIONS[st.id].cap);
        markers[st.id].setRadius(orig + (st.status === 'fault' ? 4 : 2));
      } else {
        markers[st.id].setRadius(markerRadius(STATIONS[st.id].cap));
      }
    }

    // Station list row
    const row = document.createElement('div');
    row.className = 'st-row';
    const barW = Math.min(100, st.load_pct).toFixed(0);
    const barC = loadColor(st.load_pct);
    row.innerHTML = \`<div class="st-dot" style="background:\${color}"></div>
      <div class="st-name">\${st.name}</div>
      <div class="st-bar-wrap"><div class="st-bar" style="width:\${barW}%;background:\${barC}"></div></div>
      <div class="st-pct">\${st.load_pct.toFixed(0)}%</div>\`;
    listEl.appendChild(row);
  });

  // --- Update transmission lines by stress ---
  drawLines((a, b) => {
    const stA = byId[a], stB = byId[b];
    if (!stA || !stB) return '#1a3050';
    const maxPct = Math.max(stA.load_pct, stB.load_pct);
    return loadColor(maxPct);
  });
}
</script>
</body>
</html>"""
    return HTMLResponse(content=PAGE)


def main():
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()