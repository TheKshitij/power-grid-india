"""
server.py — OpenEnv HTTP API for Indian Power Grid Load Balancer
Exposes reset() / step() / state() as POST/GET endpoints.
Runs on 0.0.0.0:7860 (Hugging Face Space compatible).
"""

import os
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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


@app.get("/", summary="API root")
async def root():
    return {
        "name":    "OpenEnv: Indian Power Grid Load Balancer",
        "version": "1.0.0",
        "docs":    "/docs",
        "tasks":   TASK_IDS,
    }



def main():
    import sys
    if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()