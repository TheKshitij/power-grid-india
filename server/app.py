"""
server.py — OpenEnv HTTP API for Indian Power Grid Load Balancer
Exposes reset() / step() / state() as POST/GET endpoints.
Runs on 0.0.0.0:7860 (Hugging Face Space compatible).
"""

import os
import sys
from typing import Any, Dict, Optional

# Force parent directory into sys.path to resolve 'grid_env' on HF Spaces
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
    """
    Start or restart an episode.
    Optionally specify `task` (single_substation | zone_rebalance | cascade_outage)
    and a `seed` for reproducibility.
    Returns the initial GridObservation.
    """
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
    obs  = _env.reset()
    return obs.model_dump()


@app.post("/step", summary="Execute one agent action")
async def step(action: GridAction):
    """
    Apply `action` to the environment and advance one time-step.
    Returns observation, reward, done flag, and info dict.
    """
    if _env is None:
        raise HTTPException(status_code=400, detail="No active episode. POST /reset first.")
    if _env._ep.get("done", False):
        raise HTTPException(status_code=400, detail="Episode finished. POST /reset to start a new one.")

    result = _env.step(action)
    return result.model_dump()


@app.get("/state", summary="Inspect current environment state")
async def state():
    """
    Return current grid state without advancing the simulation.
    """
    if _env is None:
        raise HTTPException(status_code=400, detail="No active episode. POST /reset first.")
    return _env.state()


@app.get("/tasks", summary="List available tasks")
async def tasks():
    """Return task catalogue with difficulty ratings and descriptions."""
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
    return RedirectResponse(url="/docs")


@app.get("/web", include_in_schema=False)
async def web():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Indian Power Grid Load Balancer — OpenEnv</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
            .card { background: #1e293b; border: 1px solid #334155; border-radius: 16px; padding: 40px; max-width: 640px; width: 90%; text-align: center; }
            .emoji { font-size: 3rem; margin-bottom: 16px; }
            h1 { font-size: 1.6rem; font-weight: 700; color: #f1f5f9; margin-bottom: 8px; }
            .subtitle { color: #94a3b8; margin-bottom: 28px; font-size: 0.95rem; }
            .tasks { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; margin-bottom: 28px; }
            .badge { background: #0f172a; border: 1px solid #475569; border-radius: 999px; padding: 4px 14px; font-size: 0.8rem; color: #94a3b8; }
            .badge.easy { border-color: #22c55e; color: #22c55e; }
            .badge.medium { border-color: #f59e0b; color: #f59e0b; }
            .badge.hard { border-color: #ef4444; color: #ef4444; }
            .btn { display: inline-block; background: #3b82f6; color: white; text-decoration: none; padding: 12px 28px; border-radius: 8px; font-weight: 600; font-size: 0.95rem; transition: background 0.2s; }
            .btn:hover { background: #2563eb; }
            .btn-sec { display: inline-block; background: transparent; color: #94a3b8; text-decoration: none; padding: 10px 20px; border-radius: 8px; font-size: 0.9rem; margin-left: 8px; border: 1px solid #334155; transition: border-color 0.2s; }
            .btn-sec:hover { border-color: #94a3b8; }
            .endpoints { text-align: left; margin-top: 28px; border-top: 1px solid #334155; padding-top: 20px; }
            .endpoints h3 { font-size: 0.85rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px; }
            .ep { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
            .method { background: #166534; color: #86efac; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; font-family: monospace; }
            .method.get { background: #1e3a5f; color: #93c5fd; }
            .path { font-family: monospace; font-size: 0.85rem; color: #cbd5e1; }
        </style>
    </head>
    <body>
        <div class="card">
            <div class="emoji">⚡</div>
            <h1>Indian Power Grid Load Balancer</h1>
            <p class="subtitle">An OpenEnv-compliant agent environment simulating a regional Indian electricity grid.<br>Prevent cascading blackouts across 12 substations using real POSOCO demand patterns.</p>
            <div class="tasks">
                <span class="badge easy">easy: single_substation</span>
                <span class="badge medium">medium: zone_rebalance</span>
                <span class="badge hard">hard: cascade_outage</span>
            </div>
            <a href="/docs" class="btn">Open Interactive API Docs</a>
            <a href="/health" class="btn-sec">Health Check</a>
            <div class="endpoints">
                <h3>Available Endpoints</h3>
                <div class="ep"><span class="method">POST</span><span class="path">/reset</span></div>
                <div class="ep"><span class="method">POST</span><span class="path">/step</span></div>
                <div class="ep"><span class="method get">GET</span><span class="path">/state</span></div>
                <div class="ep"><span class="method get">GET</span><span class="path">/tasks</span></div>
                <div class="ep"><span class="method get">GET</span><span class="path">/health</span></div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)



def main():
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()