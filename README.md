---
title: Power Grid India Load Balancer
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# Indian Power Grid Load Balancer (OpenEnv)

An OpenEnv-compliant environment for simulating an Indian regional electricity grid. AI agents must manage load balancing through actions like load shedding, rerouting, and restoration to prevent blackouts across variable topologies.

## 🌟 Overview

The project simulates a dynamic power grid with:
- **Realistic Demand Curves**: Inspired by POSOCO (Power System Operation Corporation) hourly data.
- **Stochastic Faults**: Critical load levels (≥100%) trip substations, redistributing load to neighbors.
- **Cascading Failures**: Harder tasks simulate propagation of faults if not pre-emptively managed.
- **Standardized API**: Complies with the OpenEnv specification for agent/environment interaction.

## 🏗️ Project Structure

- **`grid_env.py`**: The core simulation logic. Manages state, transitions, rewards, and fault injection.
- **`server/app.py`**: A FastAPI server exposing the environment as an HTTP API.
- **`inference.py`**: A baseline LLM agent (via OpenAI-compatible API) that manages the grid.
- **`openenv.yaml`**: Configuration and domain metadata.
- **`Dockerfile`**: Containerization setup for easy deployment (e.g., Hugging Face Spaces).

## 🚀 Getting Started

### 1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Server
The server runs on port 7860 by default:
```bash
python server/app.py
```

### 3. Run Inference (Agent)
To run the LLM-based baseline agent, you'll need an API key and endpoint (e.g., Hugging Face or OpenAI):
```bash
export HF_TOKEN="your_token_here"
python inference.py
```

## 🎮 How it Works

### Observation Space
The agent receives a `GridObservation` containing:
- **Substation Status**: LOAD (MW), CAPACITY (MW), LOAD%, and status (NORMAL/STRESSED/FAULT).
- **Topology**: Connectivity between substations.
- **Demand Forecast**: Predicted load percentage for the next step.
- **Blackout Risk**: Overall grid health (LOW/MEDIUM/HIGH/CRITICAL).

### Action Space
Agents can issue up to 4 actions per step using semicolons:
- `shed <station_id> <amount_mw>`: Reduce load at a specific station.
- `reroute <from_id> <to_id> <amount_mw>`: Shift load to an adjacent station.
- `restore <station_id>`: Re-add previously shed load.
- `hold`: Take no action.

### Reward System
Dense shaped rewards provide a useful **gradient throughout the episode trajectory** — not just at terminal states:

| Signal | Value | Condition |
| :--- | :--- | :--- |
| Stable station bonus | **+0.07** | Per station with `load_pct < 85%` |
| Grid balance bonus | **+0.12 × score** | Smooth utilization target (~75%), no cliff |
| Stressed station penalty | **−0.08** | Per station with `load_pct ≥ 87%` |
| Blackout penalty | **−0.30** | Per station with `load_pct ≥ 100%` |
| **Unnecessary shed penalty** | **−0.20** | Shedding load from a **NORMAL** station |
| **Over-shed exploit guard** | **−0.15** | When cumulative `shed_mw > 40%` of grid capacity |
| Invalid action | **−0.05** | Malformed or out-of-range commands |
| Non-adjacent reroute | **−0.08** | Rerouting between disconnected substations |

> **Design intent**: The agent learns that load shedding is a precision tool, not a blanket strategy. Shedding healthy substations is punished just as harshly as being caught in a cascading fault. Agents that attempt to trivially avoid blackouts by shedding the entire grid are penalised via the over-shed guard.

## 📊 Tasks

| Task ID | Difficulty | Stations | Description |
|---------|------------|----------|-------------|
| `single_substation` | Easy | 1 | Simple peak load management. |
| `zone_rebalance` | Medium | 4 | Regional balancing across Delhi NCR. |
| `cascade_outage` | Hard | 12 | Preventing propagation in a complex mesh (Maharashtra grid). |
| `renewable_crisis` | Expert | 12 | Same mesh, but solar generation collapses unpredictably mid-episode. |

## 📖 API Reference (Scalar)

The server provides an interactive, premium API reference powered by [Scalar](https://scalar.com). Once running, explore all endpoints at:

```
http://localhost:7860/docs          # local
https://your-space.hf.space/docs   # Hugging Face Space
```

This allows judges and researchers to manually test `reset`, `step`, `state`, and `render` calls directly from the browser — no CLI required.

- **`GET /render`**: Returns an ASCII grid snapshot. Useful for debugging agent behavior visually without parsing JSON logs.

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t openenv-power-grid .

# Run locally
docker run -p 7860:7860 openenv-power-grid

# Verify health
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" \
     -d '{"task": "single_substation"}'
```

## ☁️ Hugging Face Space Deployment

1. Create a new Space and set **SDK: Docker** (not Gradio/Streamlit).
2. Push the repository: `git push origin main`
3. If bundling the inference script, add `HF_TOKEN` as a **Space Secret** in the Space settings.
4. Verify the Space responds at `https://your-space.hf.space/health`.

## 📈 Baseline Scores

| Task | Difficulty | Score | Success |
|------|------------|-------|---------|
| `single_substation` | Easy | **0.82** | ✅ |
| `zone_rebalance` | Medium | **0.85** | ✅ |
| `cascade_outage` | Hard | **0.45** | ✅ |
| `renewable_crisis` | Expert | **0.31** | ✅ |

*Model used: Qwen2.5-72B-Instruct via Hugging Face Inference API.*

---

## 🕯️ Historical Context: The 2012 Northern India Blackout

The `cascade_outage` and `renewable_crisis` tasks are directly inspired by the **30–31 July 2012 Indian grid collapse** — the largest blackout in human history, affecting approximately **620 million people** across 22 states. The failure originated from overloaded transmission lines in the Northern Regional Grid and propagated in minutes as substations tripped and dumped load onto already-stressed neighbours — exactly the cascade mechanic this environment models.

The Central Electricity Regulatory Commission (CERC) report identified the root cause as a failure of *anticipatory* action: operators saw stress signals 40 minutes before the collapse and did not pre-emptively shed load. This environment is designed to train agents to act on those early signals — the `demand_forecast` field in the observation is the analogue of the stress telemetry operators had available but did not act on in time.

The `renewable_crisis` task extends this with a forward-looking dimension: India has committed to **500 GW of renewable capacity by 2030**. Solar intermittency on a stressed grid creates the same anticipatory challenge, now with less warning time.

---

*Created for the OpenEnv Hackathon Submission.*