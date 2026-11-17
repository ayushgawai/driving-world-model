"""Train world model + BC, run benchmark, distill/export. Small CPU defaults.

Usage: python -m dwm.run [--epochs 3] [--episodes 24]
Writes artifacts/*.pt, reports/report.md, reports/rollout_strip.png and fills README numbers via report.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import DrivingDataset, generate_dataset
from .distill import distill, export_onnx, quantize_and_benchmark
from .evaluate import benchmark, expert_policy, horizon_errors, query_trajectory
from .models import BCPolicy, StudentPolicy, WorldModel, wm_loss
from .planning import ActorCritic, CEMPlanner, train_in_imagination


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=24); ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    torch.manual_seed(a.seed); os.makedirs("artifacts", exist_ok=True); os.makedirs("reports", exist_ok=True)
    t0 = time.time()
    path, shapes = generate_dataset("data/dataset.npz", a.episodes, a.steps, a.seed)
    print("dataset", shapes)
    ds = DrivingDataset(path, seq=16); dl = DataLoader(ds, batch_size=8, shuffle=True)
    wm, bc = WorldModel(), BCPolicy()
    opt = torch.optim.Adam(list(wm.parameters()) + list(bc.parameters()), 1e-3)
    for ep in range(a.epochs):
        for b in dl:
            loss, parts = wm_loss(wm, b)
            bc_loss = torch.nn.functional.mse_loss(bc(b["bev"][:, 0], b["scan"][:, 0], b["ego"][:, 0], b["goal"][:, 0]), b["action"][:, 0])
            opt.zero_grad(); (loss + bc_loss).backward(); opt.step()
        print(f"epoch {ep} wm={float(loss.detach()):.4f} bc={float(bc_loss.detach()):.4f} {parts}")
    torch.save(wm.state_dict(), "artifacts/world_model.pt"); torch.save(bc.state_dict(), "artifacts/bc.pt")
    cem = CEMPlanner(wm, horizon=6, samples=32, elites=6, iters=1)
    ac = ActorCritic(); imagine_loss = train_in_imagination(wm, ac, steps=10)
    torch.save(ac.state_dict(), "artifacts/actor_critic.pt")
    def bc_fn(obs, sim):
        with torch.no_grad():
            return bc(*[torch.tensor(np.array(obs[k])[None], dtype=torch.float32) for k in ("bev", "scan", "ego", "goal")])[0].numpy()
    def cem_fn(obs, sim):
        return cem.act(obs)
    def dreamer_fn(obs, sim):
        with torch.no_grad():
            z = wm.encode({k: torch.tensor(np.array(v)[None], dtype=torch.float32) for k, v in obs.items()})
            return ac.act(z)[0].numpy()
    results = benchmark({"expert": expert_policy, "bc": bc_fn, "cem": cem_fn, "dreamer": dreamer_fn}, seeds=(100, 101), max_steps=60)
    horizons = horizon_errors(wm, path)
    # student distillation from BC + ONNX latency
    student = StudentPolicy(); dloss = distill(student, bc, path, epochs=2)
    torch.save(student.state_dict(), "artifacts/student.pt")
    onnx_path = export_onnx(student)
    latency = {}
    try:
        latency = quantize_and_benchmark(onnx_path, n=100)
    except Exception as e:  # onnxruntime optional at runtime
        latency = {"error": str(e)}
    # student closed-loop (score drop)
    from .models import obs_feat
    def student_fn(obs, sim):
        with torch.no_grad():
            return student(torch.tensor(obs_feat(obs)[None], dtype=torch.float32))[0].numpy()
    student_res = benchmark({"student": student_fn}, seeds=(100, 101), max_steps=60)["student"]
    # rollout strip PNG
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from .sim import DrivingSim
    sim = DrivingSim(seed=7, scenario="cruise"); obs = sim.reset(7); frames = [obs["bev"][0]]
    for _ in range(4):
        obs, *_ , info = sim.step(expert_policy(obs, sim)); frames.append(obs["bev"][0])
    frames_pred, _ = query_trajectory(wm, obs, np.tile([0.0, 0.5], (5, 1)))
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for ax, f in zip(axes, frames): ax.imshow(f, cmap="viridis", vmin=0, vmax=1); ax.axis("off")
    fig.savefig("reports/rollout_strip.png", dpi=100); plt.close(fig)
    summary = {"horizons": horizons, "benchmark": {k: {kk: vv for kk, vv in v.items() if kk != "episodes"} for k, v in results.items()},
               "student": {k: v for k, v in student_res.items() if k != "episodes"},
               "latency": latency, "distill_loss": dloss, "imagine_loss": imagine_loss, "seconds": time.time() - t0}
    with open("reports/results.json", "w") as f: json.dump(summary, f, indent=2, default=str)
    lines = ["# Closed-loop benchmark report\n", f"Runtime: {summary['seconds']:.1f}s | imagine loss {imagine_loss:.3f} | distill loss {dloss:.4f}\n",
             "## World-model horizon errors\n", "| horizon | latent MSE | BEV IoU |\n|---|---|---|\n"]
    lines += [f"| {h} | {v['latent_mse']:.4f} | {v['bev_iou']:.3f} |\n" for h, v in horizons.items()]
    lines += ["\n## Closed-loop (held-out seeds 100-101)\n", "| policy | route↑ | collision↓ | off-road↓ | jerk↓ | progress↑ |\n|---|---|---|---|---|---|\n"]
    for name, m in list(results.items()) + [("student", student_res)]:
        lines.append(f"| {name} | {m['route_completion']:.2f} | {m['collision']:.2f} | {m['offroad']:.2f} | {m['jerk']:.2f} | {m['progress']:.1f} |\n")
    lines += ["\n## Per-scenario failure breakdown\n", "| policy | scenario | route | collision | off-road |\n|---|---|---|---|---|\n"]
    for name, m in results.items():
        for s, v in m["per_scenario"].items():
            lines.append(f"| {name} | {s} | {v['route_completion']:.2f} | {v['collision']:.2f} | {v['offroad']:.2f} |\n")
    lines += ["\n## Distillation / latency\n", "```json\n" + json.dumps(latency, indent=2) + "\n```\n",
              f"\nStudent route completion: {student_res['route_completion']:.2f} (vs BC above). Rollout strip: `rollout_strip.png`.\n"]
    open("reports/report.md", "w").write("".join(lines))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
