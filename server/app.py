"""
server/app.py — OpenEnv HTTP API for Indian Power Grid Load Balancer
Exposes reset() / step() / state() as POST/GET endpoints.
Runs on 0.0.0.0:7860 (Hugging Face Space compatible).
"""

import os
import sys
from typing import Any, Dict, Optional

# Ensure project root is in path so 'grid_env' is importable
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
        "simulated Indian regional electricity grid, preventing blackouts "
        "via load shedding and rerouting. Three tasks: easy \u2192 medium \u2192 hard."
    ),
    docs_url=None,    # Disable default Swagger UI — we use Scalar instead
    redoc_url=None,   # Disable default ReDoc
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


@app.get("/docs", include_in_schema=False)
async def scalar_docs():
    """Premium Scalar API reference — replaces default Swagger UI."""
    HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>API Reference &mdash; Indian Power Grid</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  /* Match home-page typography globally */
  :root {
    --scalar-font: 'Inter', system-ui, sans-serif;
    --scalar-font-code: 'JetBrains Mono', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }

  /* Slim branded top-bar */
  #topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 24px;
    background: #050d1a;
    border-bottom: 1px solid #1a2d45;
    position: sticky;
    top: 0;
    z-index: 9999;
  }
  #topbar .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    text-decoration: none;
  }
  #topbar .pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(59,130,246,.12);
    border: 1px solid rgba(59,130,246,.3);
    color: #3b82f6;
    border-radius: 999px;
    padding: 3px 12px;
    font-size: .68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .06em;
    font-family: 'JetBrains Mono', monospace;
  }
  #topbar .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #22d3a0;
    animation: pd 2s ease-in-out infinite;
  }
  @keyframes pd {
    0%,100%{opacity:1;transform:scale(1)}
    50%{opacity:.4;transform:scale(.75)}
  }
  #topbar .title {
    font-family: 'Inter', sans-serif;
    font-size: .9rem;
    font-weight: 700;
    color: #e2edf8;
  }
  #topbar .back {
    font-family: 'Inter', sans-serif;
    font-size: .8rem;
    color: #5a7a9a;
    text-decoration: none;
    border: 1px solid #1a2d45;
    padding: 5px 14px;
    border-radius: 8px;
    transition: all .2s;
  }
  #topbar .back:hover { color: #e2edf8; border-color: #5a7a9a; }

  /* Let Scalar fill the rest of the viewport */
  #scalar-wrap { height: calc(100vh - 47px); overflow: hidden; background: #050d1a; }
  #scalar-wrap > * { height: 100%; }
</style>
</head>
<body style="background: #050d1a;">

<!-- Branded top-bar (only change visible to users) -->
<div id="topbar">
  <a href="/web" class="brand">
    <div class="pill"><div class="dot"></div>OpenEnv</div>
    <span class="title">Indian Power Grid &mdash; API Reference</span>
  </a>
  <a href="/web" class="back">&larr; Back to Dashboard</a>
</div>

<!-- Scalar renders here -->
<div id="scalar-wrap">
  <script id="api-reference" data-url="../openapi.json"></script>
  <script>
    document.getElementById('api-reference').dataset.configuration = JSON.stringify({
      theme: 'moon',
      darkMode: true,
      forceDarkModeState: 'dark',
      hideDarkModeToggle: true,
      defaultOpenAllTags: true,
      customCss: ':root { --scalar-font: Inter, system-ui, sans-serif; --scalar-font-code: JetBrains Mono, monospace; } .dark-mode { background: #050d1a !important; }'
    });
  </script>
</div>

<script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
</body>
</html>"""
    return HTMLResponse(content=HTML)



@app.get("/web", include_in_schema=False)
async def web():
    PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Indian Power Grid &#8212; OpenEnv</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#050d1a;--surf:#0d1b2e;--bdr:#1a2d45;--txt:#e2edf8;--mut:#5a7a9a;--blue:#3b82f6;--bglow:rgba(59,130,246,.22);--green:#22d3a0;--gglow:rgba(34,211,160,.2);--yel:#fbbf24;--red:#f43f5e}
html,body{height:100%}
body{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--txt);min-height:100vh;overflow-x:hidden}
canvas{position:fixed;inset:0;z-index:0;opacity:.35;pointer-events:none}
main{position:relative;z-index:1;max-width:860px;margin:0 auto;padding:44px 20px 64px}
.hd{text-align:center;margin-bottom:44px}
.pill{display:inline-flex;align-items:center;gap:6px;background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.3);color:var(--blue);border-radius:999px;padding:4px 14px;font-size:.72rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;margin-bottom:18px}
.pulsedot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pd 2s ease-in-out infinite}
@keyframes pd{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.75)}}
h1{font-size:clamp(1.9rem,4.5vw,3rem);font-weight:800;letter-spacing:-.03em;background:linear-gradient(135deg,#e2edf8 0%,#7cb9ff 55%,var(--green) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:10px}
.sub{color:var(--mut);font-size:.95rem;max-width:540px;margin:0 auto;line-height:1.75}
.sbar{display:flex;align-items:center;justify-content:center;gap:8px;background:var(--surf);border:1px solid var(--bdr);border-radius:10px;padding:11px 20px;margin-bottom:32px}
.sdot{width:9px;height:9px;border-radius:50%;background:var(--mut);transition:background .4s;flex-shrink:0}
.sdot.ok{background:var(--green);box-shadow:0 0 8px var(--gglow)}
.sdot.err{background:var(--red)}
#stxt{font-family:'JetBrains Mono',monospace;font-size:.78rem;color:var(--mut)}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:28px}
.scard{background:var(--surf);border:1px solid var(--bdr);border-radius:12px;padding:20px;text-align:center}
.sval{font-size:2rem;font-weight:800;color:var(--blue);letter-spacing:-.03em}
.slbl{font-size:.72rem;color:var(--mut);text-transform:uppercase;letter-spacing:.07em;margin-top:4px}
.sec{font-size:.72rem;font-weight:600;color:var(--mut);text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px}
.tcards{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin-bottom:28px}
.tc{background:var(--surf);border:1px solid var(--bdr);border-radius:14px;padding:22px;transition:border-color .2s,transform .2s;cursor:default}
.tc:hover{transform:translateY(-3px)}
.tc.e:hover{border-color:var(--green)}.tc.m:hover{border-color:var(--yel)}.tc.h:hover{border-color:var(--red)}
.th{display:flex;align-items:center;justify-content:space-between;margin-bottom:9px}
.tn{font-family:'JetBrains Mono',monospace;font-size:.78rem;font-weight:600}
.dif{font-size:.68rem;font-weight:700;padding:2px 9px;border-radius:999px}
.dif.e{background:rgba(34,211,160,.1);color:var(--green)}.dif.m{background:rgba(251,191,36,.1);color:var(--yel)}.dif.h{background:rgba(244,63,94,.1);color:var(--red)}
.td{font-size:.82rem;color:var(--mut);line-height:1.65}
.tm{margin-top:10px;font-size:.7rem;color:#355570;font-family:'JetBrains Mono',monospace}
.ep-box{background:var(--surf);border:1px solid var(--bdr);border-radius:14px;padding:18px 22px;margin-bottom:28px}
.ep{display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid var(--bdr)}
.ep:last-child{border-bottom:none}
.mth{font-family:'JetBrains Mono',monospace;font-size:.68rem;font-weight:700;padding:3px 7px;border-radius:5px;min-width:40px;text-align:center}
.mth.p{background:rgba(34,211,160,.1);color:var(--green)}.mth.g{background:rgba(59,130,246,.1);color:var(--blue)}
.epath{font-family:'JetBrains Mono',monospace;font-size:.82rem}
.edsc{font-size:.78rem;color:var(--mut);margin-left:auto}
.acts{display:flex;gap:12px;justify-content:center;flex-wrap:wrap}
.btn{display:inline-flex;align-items:center;gap:8px;padding:13px 28px;border-radius:10px;font-size:.9rem;font-weight:600;text-decoration:none;border:none;font-family:inherit;cursor:pointer;transition:all .2s}
.bpri{background:var(--blue);color:#fff;box-shadow:0 0 24px var(--bglow)}.bpri:hover{background:#2563eb;box-shadow:0 0 40px var(--bglow);transform:translateY(-1px)}
.bgho{background:transparent;color:var(--mut);border:1px solid var(--bdr)}.bgho:hover{border-color:var(--mut);color:var(--txt)}
</style>
</head>
<body>
<canvas id="c"></canvas>
<main>
  <div class="hd">
    <div class="pill"><div class="pulsedot"></div>OpenEnv Environment</div>
    <h1>Indian Power Grid<br>Load Balancer</h1>
    <p class="sub">An OpenEnv-compliant agent environment simulating a regional Indian electricity grid. AI agents prevent cascading blackouts across 12 substations using real POSOCO demand patterns.</p>
  </div>
  <div class="sbar"><div class="sdot" id="sdot"></div><span id="stxt">Checking server&hellip;</span></div>
  <div class="stats">
    <div class="scard"><div class="sval">12</div><div class="slbl">Substations</div></div>
    <div class="scard"><div class="sval">3</div><div class="slbl">Task Levels</div></div>
    <div class="scard"><div class="sval">240</div><div class="slbl">Max Steps</div></div>
  </div>
  <div class="sec">Tasks</div>
  <div class="tcards">
    <div class="tc e"><div class="th"><span class="tn">single_substation</span><span class="dif e">Easy</span></div><p class="td">Manage load across a single stressed substation. Prevent overload via shedding and rerouting.</p><div class="tm">1 station &middot; 50 steps &middot; no cascade</div></div>
    <div class="tc m"><div class="th"><span class="tn">zone_rebalance</span><span class="dif m">Medium</span></div><p class="td">Coordinate load redistribution across a 6-node zone during a demand spike event.</p><div class="tm">6 stations &middot; 120 steps &middot; no cascade</div></div>
    <div class="tc h"><div class="th"><span class="tn">cascade_outage</span><span class="dif h">Hard</span></div><p class="td">Prevent full grid collapse as faults propagate across all 12 substations simultaneously.</p><div class="tm">12 stations &middot; 240 steps &middot; cascade ON</div></div>
  </div>
  <div class="sec">API Endpoints</div>
  <div class="ep-box">
    <div class="ep"><span class="mth p">POST</span><span class="epath">/reset</span><span class="edsc">Start a new episode</span></div>
    <div class="ep"><span class="mth p">POST</span><span class="epath">/step</span><span class="edsc">Execute one action</span></div>
    <div class="ep"><span class="mth g">GET</span><span class="epath">/state</span><span class="edsc">Inspect current state</span></div>
    <div class="ep"><span class="mth g">GET</span><span class="epath">/tasks</span><span class="edsc">List all tasks</span></div>
    <div class="ep"><span class="mth g">GET</span><span class="epath">/health</span><span class="edsc">Server liveness probe</span></div>
  </div>
  <div class="acts">
    <a href="/docs" class="btn bpri">&#9889; Interactive API Docs</a>
    <a href="https://github.com/TheKshitij/power-grid-india" target="_blank" class="btn bgho">GitHub &rarr;</a>
  </div>
</main>
<script>
async function ping(){
  const d=document.getElementById('sdot'),t=document.getElementById('stxt');
  try{
    const r=await fetch('/health'),j=await r.json();
    d.className='sdot ok';
    t.innerHTML='<span style="color:#22d3a0">&#9679; LIVE</span>&nbsp;&nbsp;status: "'+j.status+'" &middot; service: "'+j.service+'"';
  }catch(e){
    d.className='sdot err';
    t.textContent='Server unreachable';
  }
}
ping();setInterval(ping,10000);

const cv=document.getElementById('c'),cx=cv.getContext('2d');
let W,H,ns=[],es=[],ps=[];
function init(){
  W=cv.width=innerWidth;H=cv.height=innerHeight;ns=[];
  for(let i=0;i<18;i++)ns.push({x:Math.random()*W,y:Math.random()*H,vx:(Math.random()-.5)*.3,vy:(Math.random()-.5)*.3});
  es=[];
  for(let i=0;i<ns.length;i++)for(let j=i+1;j<ns.length;j++){
    if(Math.hypot(ns[i].x-ns[j].x,ns[i].y-ns[j].y)<250)es.push([i,j]);
  }
}
function addP(){if(!es.length)return;ps.push({e:es[Math.floor(Math.random()*es.length)],t:0,s:.007+Math.random()*.006})}
function draw(){
  cx.clearRect(0,0,W,H);
  ns.forEach(n=>{n.x+=n.vx;n.y+=n.vy;if(n.x<0||n.x>W)n.vx*=-1;if(n.y<0||n.y>H)n.vy*=-1});
  es.forEach(([i,j])=>{cx.strokeStyle='rgba(26,45,69,.8)';cx.lineWidth=1;cx.beginPath();cx.moveTo(ns[i].x,ns[i].y);cx.lineTo(ns[j].x,ns[j].y);cx.stroke()});
  ns.forEach(n=>{cx.fillStyle='rgba(59,130,246,.55)';cx.beginPath();cx.arc(n.x,n.y,2.5,0,Math.PI*2);cx.fill()});
  ps=ps.filter(p=>p.t<=1);
  ps.forEach(p=>{
    const[i,j]=p.e,x=ns[i].x+(ns[j].x-ns[i].x)*p.t,y=ns[i].y+(ns[j].y-ns[i].y)*p.t;
    const g=cx.createRadialGradient(x,y,0,x,y,9);
    g.addColorStop(0,'rgba(34,211,160,.9)');g.addColorStop(1,'transparent');
    cx.fillStyle=g;cx.beginPath();cx.arc(x,y,9,0,Math.PI*2);cx.fill();p.t+=p.s;
  });
  requestAnimationFrame(draw);
}
window.addEventListener('resize',init);init();draw();setInterval(addP,550);
</script>
</body>
</html>"""
    return HTMLResponse(content=PAGE)


def main():
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()