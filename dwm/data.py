"""Dataset generation (npz shards) + torch loading."""
import numpy as np
import torch
from torch.utils.data import Dataset

from .sim import SCENARIOS, DrivingSim, expert_action


def generate_dataset(out="data/dataset.npz", episodes=24, steps=80, seed=0):
    rng = np.random.default_rng(seed)
    D = {k: [] for k in ["bev", "scan", "ego", "goal", "action", "reward", "done", "collision"]}
    for ep in range(episodes):
        sim = DrivingSim(seed=int(rng.integers(1e9)), scenario=SCENARIOS[ep % len(SCENARIOS)], max_steps=steps)
        obs = sim.reset()
        for _ in range(steps):
            act = expert_action(sim, obs, noise=0.05, rng=rng)
            D["action"].append(act)
            for k in ["bev", "scan", "ego", "goal"]:
                D[k].append(obs[k])
            obs, r, done, info = sim.step(act)
            D["reward"].append(r); D["done"].append(done); D["collision"].append(info["collision"])
            if done:
                break
    D = {k: np.array(v) for k, v in D.items()}
    import os; os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, **D, episodes=np.array([episodes]))
    return out, {k: v.shape for k, v in D.items()}


class DrivingDataset(Dataset):
    """Fixed-length sequences for world-model / BC training."""
    def __init__(self, path, seq=16):
        d = np.load(path); self.seq = seq
        self.d = {k: d[k] for k in ["bev", "scan", "ego", "goal", "action", "reward", "done", "collision"]}
        self.n = max(1, len(self.d["action"]) - seq)
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        s = slice(i, i + self.seq)
        return {k: torch.tensor(v[s], dtype=torch.float32) for k, v in self.d.items()}
