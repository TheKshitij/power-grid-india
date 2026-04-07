from .grid_env import GridEnv

def load_env(task: str = "single_substation", seed: int = 42):
    return GridEnv(task=task, seed=seed)
