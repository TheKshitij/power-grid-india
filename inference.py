"""
inference.py — Baseline agent for OpenEnv: Indian Power Grid Load Balancer
Runs an LLM agent (via OpenAI-compatible API) across all 3 tasks and emits
the mandatory [START] / [STEP] / [END] stdout log format.

MANDATORY environment variables:
    API_BASE_URL       LLM endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME         Model id       (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN           Your Hugging Face / API key.
    LOCAL_IMAGE_NAME   Docker image name (used if running env from local container)
"""

import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

from grid_env import GridAction, GridEnv, GridObservation, TASK_IDS


API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY          = os.getenv("HF_TOKEN")     or os.getenv("API_KEY", "no-key")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", None) 

BENCHMARK    = "power-grid-india"
MAX_STEPS    = {
    "single_substation": 12,
    "zone_rebalance":    20,
    "cascade_outage":    30,
    "renewable_crisis": 25,
}
TEMPERATURE  = 0.2  
MAX_TOKENS   = 60    
SEED         = 42    

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err_val  = error if error else "null"
    done_val = str(done).lower()
    action_safe = action.replace("\n", " ").replace("\r", "").strip("'\"")
    print(
        f"[STEP]  step={step} action={action_safe} "
        f"reward={reward:.2f} done={done_val} error={err_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    score_clamped = min(max(score, 0.0), 1.0)
    print(
        f"[END]   success={str(success).lower()} steps={steps} "
        f"score={score_clamped:.3f} rewards={rewards_str}",
        flush=True,
    )


_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert power grid operator managing an Indian regional electricity grid.
    Your job is to prevent blackouts by issuing grid actions every turn.

    AVAILABLE ACTIONS — output one or more separated by semicolons (up to 4):
      shed <station_id> <amount_mw>           — reduce load at a station
      reroute <from_id> <to_id> <amount_mw>   — shift load to an adjacent station
      restore <station_id>                    — re-add previously shed load
      hold                                    — do nothing

    RULES:
    - Output ONLY the action string(s). No explanation. No markdown. No quotes.
    - Stations with load_pct >= 87% are STRESSED — shed them before they trip.
    - If MULTIPLE stations are stressed, shed all of them in one step using semicolons.
    - Stations at load_pct >= 100% are FAULT — focus on preventing spread to neighbors.
    - reroute only works between directly connected (neighbor) stations.
    - When load_pct drops below 70% and shed_mw > 0, restore to avoid over-shedding.
    - Use the demand_forecast to anticipate which stations will be stressed next step.

    Examples of valid output:
      shed 2 80
      shed 1 60; shed 3 90
      reroute 0 1 120; shed 2 50
      restore 3
      hold
""").strip()


def _format_obs(obs: GridObservation, step: int, history: List[str]) -> str:
    lines = [
        f"=== STEP {step} | Time: {obs.time_label} | Risk: {obs.blackout_risk.upper()} ===",
        "",
        "SUBSTATION STATUS:",
    ]
    for s in obs.substations:
        bar   = "█" * int(s.load_pct / 10) + "░" * (10 - int(s.load_pct / 10))
        alert = "  STRESSED" if s.status.value == "stressed" else (
                "  FAULT"    if s.status.value == "fault"    else "")
        shed  = f" [shed={s.shed_mw:.0f}MW]" if s.shed_mw > 0 else ""
        lines.append(
            f"  [{s.id}] {s.name:<16} {bar} {s.load_pct:5.1f}% "
            f"({s.load_mw:.0f}/{s.capacity_mw:.0f} MW){shed}{alert}"
        )
        if s.neighbors:
            lines.append(f"       neighbors: {s.neighbors}")

    lines += [
        "",
        f"Grid aggregate: {obs.grid_load_pct:.1f}% | Blackouts this episode: {obs.episode_blackouts}",
        f"Active faults: {obs.active_faults if obs.active_faults else 'none'}",
        f"Demand forecast next step (% capacity): {obs.demand_forecast}",
        f"Last action result: {obs.last_action_message}",
        "",
        "RECENT HISTORY:",
    ]
    for h in history[-5:]:
        lines.append(f"  {h}")

    lines.append("")
    lines.append("Your action:")
    return "\n".join(lines)


def get_action(
    client:  OpenAI,
    obs:     GridObservation,
    step:    int,
    history: List[str],
) -> str:
    user_prompt = _format_obs(obs, step, history)
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        action = (resp.choices[0].message.content or "hold").strip()
        
        action = action.splitlines()[0].strip()
        return action if action else "hold"
    except Exception as exc:
        return "hold"


def run_episode(client: OpenAI, task: str) -> None:
    env      = GridEnv(task=task, seed=SEED)
    obs      = env.reset()
    rewards: List[float] = []
    history: List[str]   = []
    steps_taken = 0
    score  = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS[task] + 1):
            if env._ep.get("done", False):
                break

            action_str = get_action(client, obs, step, history)
            result     = env.step(GridAction(action=action_str))

            reward      = result.reward
            done        = result.done
            error_msg   = None

            msg = result.info.get("action_message", "")
            if msg.startswith("Error:") or msg.startswith("Parse error"):
                error_msg = msg[:80]

            rewards.append(reward)
            steps_taken = step
            score       = result.score

            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            history.append(
                f"step={step} action={action_str!r} reward={reward:+.2f} | {msg[:60]}"
            )
            obs = result.observation

            if done:
                break

        success = score >= 0.25  

    except Exception as exc:
        pass

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task in TASK_IDS:
        run_episode(client, task)
        print("", flush=True)   


if __name__ == "__main__":
    main()