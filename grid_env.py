"""
grid_env.py — Indian Power Grid Load Balancer
Core environment logic: typed models, simulation, reward functions.
Imported by server.py (HTTP API) and inference.py (direct agent use).
"""

import math
import random
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class StationStatus(str, Enum):
    NORMAL   = "normal"
    STRESSED = "stressed"
    FAULT    = "fault"


class SubstationObs(BaseModel):
    id:           int
    name:         str
    load_mw:      float = Field(..., description="Current load in MW")
    capacity_mw:  float = Field(..., description="Maximum rated capacity in MW")
    load_pct:     float = Field(..., description="load_mw / capacity_mw × 100")
    status:       StationStatus
    neighbors:    List[int] = Field(default_factory=list, description="IDs of directly connected substations")
    shed_mw:      float = Field(0.0, description="MW currently shed (load reduction active)")


class GridObservation(BaseModel):
    step:                  int
    max_steps:             int
    hour:                  int    = Field(..., description="Simulation hour 0-23")
    time_label:            str    = Field(..., description="Human-readable time e.g. '19:00'")
    substations:           List[SubstationObs]
    demand_forecast:       List[float] = Field(..., description="Predicted load_pct per station next step")
    active_faults:         List[int]   = Field(default_factory=list, description="Station IDs currently in FAULT")
    total_shed_mw:         float       = 0.0
    grid_load_pct:         float       = Field(..., description="Aggregate MW / aggregate capacity × 100")
    blackout_risk:         str         = Field(..., description="low | medium | high | critical")
    episode_blackouts:     int         = 0
    last_action_message:   str         = ""


class GridAction(BaseModel):
    action: str = Field(
        ...,
        description=(
            "One or more grid actions separated by semicolons (up to 4 per step).\n"
            "Primitives:\n"
            "  shed <station_id> <amount_mw>           — reduce load at station\n"
            "  reroute <from_id> <to_id> <amount_mw>  — transfer load between adjacent stations\n"
            "  restore <station_id>                    — re-add previously shed load\n"
            "  hold                                    — take no action\n"
            "Examples:\n"
            "  'shed 2 80'\n"
            "  'shed 1 60; shed 3 90'\n"
            "  'reroute 0 1 120; shed 2 50'\n"
            "  'hold'"
        ),
    )


class StepResult(BaseModel):
    observation: GridObservation
    reward:      float
    done:        bool
    score:       float = Field(..., description="Normalised episode score so far [0, 1]")
    info:        Dict[str, Any] = Field(default_factory=dict)



_TOPOLOGIES: Dict[str, Dict[str, Any]] = {
    "single_substation": {
        "description": "Single substation peak management — easy",
        "stations": [
            {"id": 0, "name": "Patna-North",  "capacity": 500.0, "neighbors": []},
        ],
        "max_steps":   12,
        "fault_prob":  0.0,
        "cascade":     False,
    },
    "zone_rebalance": {
        "description": "Four-zone Delhi NCR rebalancing with rolling faults — medium",
        "stations": [
            {"id": 0, "name": "Delhi-West",  "capacity": 800.0, "neighbors": [1, 2]},
            {"id": 1, "name": "Delhi-East",  "capacity": 600.0, "neighbors": [0, 3]},
            {"id": 2, "name": "Gurgaon",     "capacity": 500.0, "neighbors": [0, 3]},
            {"id": 3, "name": "Noida",       "capacity": 700.0, "neighbors": [1, 2]},
        ],
        "max_steps":   20,
        "fault_prob":  0.07,
        "cascade":     False,
    },
    "cascade_outage": {
        "description": "12-node Maharashtra grid with cascading fault propagation — hard",
        "stations": [
            {"id":  0, "name": "Mumbai-Central", "capacity": 1200.0, "neighbors": [1, 4]},
            {"id":  1, "name": "Thane",          "capacity":  900.0, "neighbors": [0, 2, 5]},
            {"id":  2, "name": "Pune",           "capacity":  800.0, "neighbors": [1, 3, 6]},
            {"id":  3, "name": "Nashik",         "capacity":  600.0, "neighbors": [2, 7]},
            {"id":  4, "name": "Navi-Mumbai",    "capacity":  700.0, "neighbors": [0, 5, 8]},
            {"id":  5, "name": "Raigad",         "capacity":  500.0, "neighbors": [1, 4, 9]},
            {"id":  6, "name": "Satara",         "capacity":  450.0, "neighbors": [2, 7, 10]},
            {"id":  7, "name": "Solapur",        "capacity":  550.0, "neighbors": [3, 6, 11]},
            {"id":  8, "name": "Panvel",         "capacity":  400.0, "neighbors": [4, 9]},
            {"id":  9, "name": "Alibag",         "capacity":  350.0, "neighbors": [5, 8]},
            {"id": 10, "name": "Kolhapur",       "capacity":  500.0, "neighbors": [6, 11]},
            {"id": 11, "name": "Sangli",         "capacity":  450.0, "neighbors": [7, 10]},
        ],
        "max_steps":   30,
        "fault_prob":  0.10,
        "cascade":     True,
    },
    "renewable_crisis": {
        "description": "12-node Maharashtra grid with solar intermittency and cascading faults — expert",
        "stations": [
            {"id":  0, "name": "Mumbai-Central", "capacity": 1200.0, "neighbors": [1, 4]},
            {"id":  1, "name": "Thane",          "capacity":  900.0, "neighbors": [0, 2, 5]},
            {"id":  2, "name": "Pune",           "capacity":  800.0, "neighbors": [1, 3, 6]},
            {"id":  3, "name": "Nashik",         "capacity":  600.0, "neighbors": [2, 7]},
            {"id":  4, "name": "Navi-Mumbai",    "capacity":  700.0, "neighbors": [0, 5, 8]},
            {"id":  5, "name": "Raigad",         "capacity":  500.0, "neighbors": [1, 4, 9]},
            {"id":  6, "name": "Satara",         "capacity":  450.0, "neighbors": [2, 7, 10]},
            {"id":  7, "name": "Solapur",        "capacity":  550.0, "neighbors": [3, 6, 11]},
            {"id":  8, "name": "Panvel",         "capacity":  400.0, "neighbors": [4, 9]},
            {"id":  9, "name": "Alibag",         "capacity":  350.0, "neighbors": [5, 8]},
            {"id": 10, "name": "Kolhapur",       "capacity":  500.0, "neighbors": [6, 11]},
            {"id": 11, "name": "Sangli",         "capacity":  450.0, "neighbors": [7, 10]},
        ],
        "max_steps":   25,
        "fault_prob":  0.10,
        "cascade":     True,
        "renewable_intermittency": True,
    },
}

TASK_IDS = list(_TOPOLOGIES.keys())

# India load curve — hourly demand multipliers (POSOCO-inspired)
_DEMAND_CURVE = [
    0.55, 0.50, 0.47, 0.46, 0.47, 0.52,   # 00-05  night trough
    0.60, 0.70, 0.80, 0.84, 0.86, 0.87,   # 06-11  morning ramp
    0.88, 0.85, 0.83, 0.85, 0.88, 0.93,   # 12-17  afternoon plateau
    0.97, 1.00, 0.98, 0.91, 0.80, 0.68,   # 18-23  evening peak → decline
]



def _demand_mult(hour: int, station_id: int) -> float:
    """Return demand multiplier with per-station sinusoidal variation."""
    base  = _DEMAND_CURVE[hour % 24]
    noise = 0.025 * math.sin(station_id * 1.9 + hour * 0.6)
    return max(0.40, min(1.08, base + noise))


def _status(load: float, cap: float) -> StationStatus:
    pct = load / cap * 100
    if pct >= 100: return StationStatus.FAULT
    if pct >= 87:  return StationStatus.STRESSED
    return StationStatus.NORMAL


class GridEnv:
    """
    Python environment for the Indian Power Grid Load Balancer task.
    Implements reset() / step() / state() returning typed Pydantic models.
    Thread-unsafe — instantiate one per episode / request.
    """

    def __init__(self, task: str = "single_substation", seed: Optional[int] = None):
        if task not in _TOPOLOGIES:
            raise ValueError(f"Unknown task '{task}'. Choose from {TASK_IDS}")
        self.task = task
        self._topo = _TOPOLOGIES[task]
        self._rng  = random.Random(seed)
        self._ep: Dict[str, Any] = {}


    def reset(self) -> GridObservation:
        topo = self._topo
        # Per-task starting conditions: more headroom so agent has time to react
        if self.task == "single_substation":
            hour   = self._rng.randint(17, 20)
            lo, hi = 0.88, 1.00
        elif self.task == "zone_rebalance":
            hour   = self._rng.randint(16, 18)   # pre-peak, demand rising
            lo, hi = 0.82, 0.93
        elif self.task == "cascade_outage":
            hour   = self._rng.randint(15, 17)   # early afternoon, time to plan
            lo, hi = 0.80, 0.92
        else:   # renewable_crisis — starts mid-morning when solar is ramping
            hour   = self._rng.randint(9, 12)
            lo, hi = 0.78, 0.90
        stations = []
        for st_def in topo["stations"]:
            mult = _demand_mult(hour, st_def["id"])
            load = st_def["capacity"] * mult * self._rng.uniform(lo, hi)
            load = min(load, st_def["capacity"] * 0.95)   # never start above 95%
            stations.append({
                "id":             st_def["id"],
                "name":           st_def["name"],
                "capacity":       st_def["capacity"],
                "neighbors":      list(st_def["neighbors"]),
                "load":           round(load, 1),
                "shed_mw":        0.0,
                "stressed_steps": 0,
                "status":         StationStatus.NORMAL,
            })
        self._ep = {
            "task":                    self.task,
            "step":                    0,
            "max_steps":               topo["max_steps"],
            "hour":                    hour,
            "stations":                stations,
            "blackouts":               0,
            "total_reward":            0.0,
            "fault_prob":              topo["fault_prob"],
            "cascade":                 topo["cascade"],
            "renewable_intermittency": topo.get("renewable_intermittency", False),
            "done":                    False,
            "last_msg":                "Episode started. Grid initialised.",
        }
        return self._make_obs()

    def step(self, action: GridAction) -> StepResult:
        if not self._ep:
            raise RuntimeError("Call reset() before step()")
        if self._ep["done"]:
            raise RuntimeError("Episode complete. Call reset()")

        ep = self._ep
        ep["step"] += 1

        # 1. Apply agent action
        action_reward, msg = self._apply_action(action.action)
        ep["last_msg"] = msg

        # 2. Advance demand (time passes)
        self._advance_demand()

        # 3. Inject stochastic faults
        self._inject_faults()

        # 4. Compute step reward
        step_reward, new_blackouts = self._step_reward(action_reward)
        ep["blackouts"]    += new_blackouts
        ep["total_reward"] += step_reward

        # Episode terminates: max steps OR too many blackouts
        blackout_limit = 5 if ep["task"] in ("cascade_outage", "renewable_crisis") else 3
        done = (ep["step"] >= ep["max_steps"]) or (ep["blackouts"] >= blackout_limit)
        ep["done"] = done

        obs   = self._make_obs()
        score = self._normalise_score()

        return StepResult(
            observation=obs,
            reward=round(step_reward, 4),
            done=done,
            score=round(score, 4),
            info={
                "step_reward":    round(step_reward, 4),
                "total_reward":   round(ep["total_reward"], 4),
                "blackouts":      ep["blackouts"],
                "score":          round(score, 4),
                "action_message": msg,
            },
        )

    def state(self) -> Dict[str, Any]:
        if not self._ep:
            return {"error": "No active episode. Call reset() first."}
        obs = self._make_obs()
        ep  = self._ep
        return {
            "task":         ep["task"],
            "step":         ep["step"],
            "max_steps":    ep["max_steps"],
            "total_reward": round(ep["total_reward"], 4),
            "blackouts":    ep["blackouts"],
            "score":        round(self._normalise_score(), 4),
            "done":         ep["done"],
            "observation":  obs.model_dump(),
        }


    def _make_obs(self) -> GridObservation:
        ep   = self._ep
        hour = ep["hour"]
        next_hour = (hour + 1) % 24

        subs: List[SubstationObs] = []
        active_faults: List[int]  = []
        total_shed  = 0.0
        total_load  = 0.0
        total_cap   = 0.0

        for st in ep["stations"]:
            st["status"] = _status(st["load"], st["capacity"])
            pct = round(st["load"] / st["capacity"] * 100, 1)
            if st["status"] == StationStatus.FAULT:
                active_faults.append(st["id"])
            total_shed += st["shed_mw"]
            total_load += st["load"]
            total_cap  += st["capacity"]
            subs.append(SubstationObs(
                id=st["id"], name=st["name"],
                load_mw=round(st["load"], 1),
                capacity_mw=st["capacity"],
                load_pct=pct,
                status=st["status"],
                neighbors=st["neighbors"],
                shed_mw=round(st["shed_mw"], 1),
            ))

        forecast = [
            round(_demand_mult(next_hour, st["id"]) * 100, 1)
            for st in ep["stations"]
        ]

        grid_pct = round(total_load / total_cap * 100, 1) if total_cap > 0 else 0.0
        if grid_pct >= 97:   risk = "critical"
        elif grid_pct >= 88: risk = "high"
        elif grid_pct >= 78: risk = "medium"
        else:                risk = "low"

        return GridObservation(
            step=ep["step"],
            max_steps=ep["max_steps"],
            hour=hour,
            time_label=f"{hour:02d}:00",
            substations=subs,
            demand_forecast=forecast,
            active_faults=active_faults,
            total_shed_mw=round(total_shed, 1),
            grid_load_pct=grid_pct,
            blackout_risk=risk,
            episode_blackouts=ep["blackouts"],
            last_action_message=ep.get("last_msg", ""),
        )

    def _apply_action(self, action_str: str) -> Tuple[float, str]:
        """
        Parse and apply one or more actions separated by semicolons.
        Returns (total_action_reward_delta, combined_message).
        """
        sub_actions = [a.strip() for a in action_str.split(";") if a.strip()]
        if not sub_actions:
            sub_actions = ["hold"]
        total_reward = 0.0
        messages: List[str] = []
        for sub in sub_actions[:4]:   # cap at 4 sub-actions per step
            r, m = self._apply_single_action(sub)
            total_reward += r
            messages.append(m)
        return total_reward, " | ".join(messages)

    def _apply_single_action(self, action_str: str) -> Tuple[float, str]:
        """Parse and apply a single primitive action."""
        parts = action_str.strip().lower().split()
        if not parts or parts[0] == "hold":
            return 0.0, "hold — no action taken."

        stations_by_id = {s["id"]: s for s in self._ep["stations"]}
        n = len(self._ep["stations"])
        cmd = parts[0]

        try:
            if cmd == "shed":
                if len(parts) < 3:
                    return -0.05, "Error: shed requires <station_id> <amount_mw>"
                sid, amt = int(parts[1]), float(parts[2])
                if sid not in stations_by_id:
                    return -0.05, f"Error: station {sid} does not exist."
                st  = stations_by_id[sid]
                # Compute status before applying action to detect unnecessary shed
                current_status = _status(st["load"], st["capacity"])
                amt = max(0.0, min(amt, st["load"]))
                st["load"]    -= amt
                st["shed_mw"] += amt
                # Penalise shedding from a healthy (NORMAL) station — real operators don't do this
                if current_status == StationStatus.NORMAL:
                    return -0.20, f"Unnecessary shed: {amt:.0f} MW removed from healthy {st['name']} (load was NORMAL). Penalty applied."
                return 0.0, f"Shed {amt:.0f} MW from {st['name']} (now {st['load']:.0f}/{st['capacity']:.0f} MW)."

            if cmd == "reroute":
                if len(parts) < 4:
                    return -0.05, "Error: reroute requires <from_id> <to_id> <amount_mw>"
                fid, tid, amt = int(parts[1]), int(parts[2]), float(parts[3])
                if fid not in stations_by_id or tid not in stations_by_id:
                    return -0.05, f"Error: station id out of range."
                f_st = stations_by_id[fid]
                t_st = stations_by_id[tid]
                if tid not in f_st["neighbors"]:
                    return -0.08, f"Error: {f_st['name']} and {t_st['name']} are not directly connected."
                headroom = max(0.0, t_st["capacity"] - t_st["load"])
                actual   = min(amt, f_st["load"] * 0.35, headroom)
                f_st["load"] -= actual
                t_st["load"] += actual
                return 0.02, (f"Rerouted {actual:.0f} MW from {f_st['name']} → {t_st['name']}. "
                               f"Source: {f_st['load']:.0f} MW, Dest: {t_st['load']:.0f} MW.")

            if cmd == "restore":
                if len(parts) < 2:
                    return -0.05, "Error: restore requires <station_id>"
                sid = int(parts[1])
                if sid not in stations_by_id:
                    return -0.05, f"Error: station {sid} does not exist."
                st = stations_by_id[sid]
                if st["shed_mw"] <= 0:
                    return -0.02, f"No shed load to restore at {st['name']}."
                restored      = st["shed_mw"]
                st["load"]   += restored
                st["shed_mw"] = 0.0
                return 0.0, f"Restored {restored:.0f} MW at {st['name']} (now {st['load']:.0f}/{st['capacity']:.0f} MW)."

        except (ValueError, IndexError) as exc:
            return -0.05, f"Parse error: {exc}"

        return -0.05, f"Unknown command '{cmd}'. Valid: shed | reroute | restore | hold"

    def _advance_demand(self) -> None:
        """Move clock forward one step and drift loads toward demand curve."""
        ep = self._ep
        ep["hour"] = (ep["hour"] + 1) % 24
        hour = ep["hour"]
        for st in ep["stations"]:
            if st["status"] == StationStatus.FAULT:
                continue
            target = st["capacity"] * _demand_mult(hour, st["id"])

            # Renewable intermittency: solar generation can collapse suddenly.
            # Between 06-18h, solar offsets some load. A 15% chance per step
            # simulates cloud cover / ramp-down, causing load to spike back up.
            if ep.get("renewable_intermittency", False) and 6 <= hour <= 18:
                solar_offset = st["capacity"] * 0.13 * math.sin(
                    (hour - 6) * math.pi / 12
                )
                if self._rng.random() < 0.15:
                    # Sudden solar drop — load surges back
                    solar_offset *= 0.30
                target += solar_offset * 0.40

            # Exponential smoothing + small noise
            st["load"] = round(
                st["load"] * 0.65 + target * 0.35 + self._rng.uniform(-8, 8), 1
            )
            st["load"] = max(0.0, st["load"])

    def _inject_faults(self) -> None:
        """Stochastic fault injection; cascade propagation for hard task."""
        ep = self._ep
        for st in ep["stations"]:
            pct = st["load"] / st["capacity"]

            # Track consecutive stressed steps
            if pct >= 0.93:
                st["stressed_steps"] += 1
            else:
                st["stressed_steps"] = 0

            # Cascade trip: stressed for 3+ consecutive steps → fault
            if ep["cascade"] and st["stressed_steps"] >= 3 and st["status"] != StationStatus.FAULT:
                self._trip_station(st)
                continue

            # Random fault weighted by load level
            effective_prob = ep["fault_prob"] * max(0.0, (pct - 0.80) / 0.20)
            if self._rng.random() < effective_prob:
                self._trip_station(st)

    def _trip_station(self, st: Dict) -> None:
        """Trip a station to FAULT, redistribute load to neighbors."""
        st["status"]         = StationStatus.FAULT
        st["stressed_steps"] = 0
        stations_by_id = {s["id"]: s for s in self._ep["stations"]}
        live_neighbors = [
            stations_by_id[nid] for nid in st["neighbors"]
            if stations_by_id.get(nid, {}).get("status") != StationStatus.FAULT
        ]
        if live_neighbors and st["load"] > 0:
            share = st["load"] / len(live_neighbors)
            for nb in live_neighbors:
                nb["load"] += share * 0.75   # 25% loss in redistribution
        st["load"] = 0.0

    def _step_reward(self, action_delta: float) -> Tuple[float, int]:
        """
        Compute dense step reward.
        Returns (step_reward, n_new_blackouts_this_step).
        """
        reward    = 0.0
        blackouts = 0
        total_load, total_cap = 0.0, 0.0

        for st in self._ep["stations"]:
            pct = st["load"] / st["capacity"]
            total_load += st["load"]
            total_cap  += st["capacity"]

            if pct >= 1.0:
                reward    -= 0.30    # blackout penalty
                blackouts += 1
            elif pct >= 0.93:
                reward    -= 0.08    # stressed penalty
            elif pct <= 0.85:
                reward    += 0.07    # stable bonus per station

        # Whole-grid balance bonus (target: 70-80% utilisation)
        grid_pct = total_load / total_cap if total_cap > 0 else 0
        balance  = max(0.0, 1.0 - abs(grid_pct - 0.75) * 3.0)
        reward  += balance * 0.12

        # Guard against the "shed-all" exploit: penalise excessive grid underutilisation.
        # An agent that sheds >40% of total capacity to trivially avoid blackouts
        # scores worse than one that reroutes and manages load precisely.
        total_shed_mw = sum(st.get("shed_mw", 0.0) for st in self._ep["stations"])
        if total_cap > 0 and total_shed_mw / total_cap > 0.40:
            reward -= 0.15

        reward += action_delta
        return round(reward, 4), blackouts

    def _normalise_score(self) -> float:
        """
        Normalise episode total_reward to [0, 1] with deterministic
        min/max bounds derived from the task parameters.
        """
        ep        = self._ep
        n         = len(ep["stations"])
        max_steps = ep["max_steps"]
        # Upper bound: every station stable + full balance bonus every step
        max_r = max_steps * (n * 0.07 + 0.12)
        # Lower bound: every station blackout + over-shed penalty every step
        min_r = max_steps * (n * -0.30 + -0.15)
        raw   = ep["total_reward"]
        score = (raw - min_r) / (max_r - min_r)
        return round(max(0.0, min(1.0, score)), 4)