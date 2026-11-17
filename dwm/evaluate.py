"""Closed-loop benchmark + world-model horizon metrics."""
import numpy as np
import torch

from .planning import apply_constraints
from .sim import SCENARIOS, DrivingSim, expert_action


def run_episode(policy_fn, scenario, seed, max_steps=120):
    sim = DrivingSim(seed=seed, scenario=scenario, max_steps=max_steps)
    obs, prev, jerks, coll, off, done = sim.reset(seed), np.zeros(2), [], False, False, False
    while not done:
        act = apply_constraints(np.asarray(policy_fn(obs, sim), float), prev); prev = act
        obs, r, done, info = sim.step(act)
        jerks.append(abs(info["jerk"])); coll |= info["collision"]; off |= info["offroad"]
    return {"scenario": scenario, "route_completion": info["route_completion"], "collision": float(coll),
            "offroad": float(off), "jerk": float(np.mean(jerks) if jerks else 0), "progress": info["progress"]}


def benchmark(policies, seeds=(100, 101), scenarios=SCENARIOS, max_steps=120):
    """policies: {name: fn(obs, sim)->action}. Returns {name: metrics + per_scenario}"""
    out = {}
    for name, fn in policies.items():
        rows = [run_episode(fn, s, seed + i, max_steps) for i, s in enumerate(scenarios) for seed in seeds]
        agg = {k: float(np.mean([r[k] for r in rows])) for k in ["route_completion", "collision", "offroad", "jerk", "progress"]}
        agg["per_scenario"] = {s: {k: float(np.mean([r[k] for r in rows if r["scenario"] == s]))
                                   for k in ["route_completion", "collision", "offroad"]} for s in scenarios}
        agg["episodes"] = rows; out[name] = agg
    return out


def expert_policy(obs, sim):
    return expert_action(sim, obs)


@torch.no_grad()
def horizon_errors(wm, dataset_path, horizons=(1, 5, 15)):
    """Open-loop latent MSE + decoded IoU at horizons, on dataset sequences."""
    from .data import DrivingDataset
    wm.eval(); ds = DrivingDataset(dataset_path, seq=max(horizons) + 1)
    res = {}
    for h in horizons:
        mse, ious = [], []
        for i in range(min(len(ds), 8)):
            b = ds[i]
            z0 = wm.encode({k: v[0:1] for k, v in b.items() if k in ("bev", "scan", "ego", "goal")})
            zs, _, decoded = wm.rollout(z0, b["action"][:h].unsqueeze(0))
            zt = wm.encode({k: v[h:h + 1] for k, v in b.items() if k in ("bev", "scan", "ego", "goal")})
            mse.append(float(torch.mean((zs[0, -1] - zt[0]) ** 2)))
            pred, true = (decoded[0, 0] > 0.5), (b["bev"][h, 0] > 0.5)
            ious.append(float((pred & true).sum() / max(1, (pred | true).sum())))
        res[h] = {"latent_mse": float(np.mean(mse)), "bev_iou": float(np.mean(ious))}
    return res


def query_trajectory(wm, obs, action_sequence):
    """Trajectory querying: roll out candidate actions, return decoded BEV strip (T,32,32)."""
    with torch.no_grad():
        z0 = wm.encode({k: torch.tensor(np.array(v)[None], dtype=torch.float32) for k, v in obs.items()})
        acts = torch.tensor(np.array(action_sequence)[None], dtype=torch.float32)
        zs, rs, _ = wm.rollout(z0, acts)
        frames = [wm.decode(zs[0, t:t + 1])[0, 0].numpy() for t in range(zs.size(1))]
    return np.array(frames), rs[0].numpy()
