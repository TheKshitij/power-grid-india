from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from grid_env import GridAction, GridObservation

class PowerGridClient(EnvClient[GridAction, GridObservation, State]):
    def _step_payload(self, action: GridAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[GridObservation]:
        obs_data = payload.get("observation", {})
        obs = GridObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False)
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("task", "0"),
            step_count=payload.get("step", 0)
        )
