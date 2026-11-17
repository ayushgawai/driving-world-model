import numpy as np
import torch

from dwm.data import DrivingDataset, generate_dataset
from dwm.evaluate import benchmark, expert_policy, horizon_errors
from dwm.models import BCPolicy, StudentPolicy, WorldModel, obs_feat
from dwm.planning import ActorCritic, CEMPlanner, apply_constraints, train_in_imagination
from dwm.sim import DrivingSim


def test_sim_determinism():
    a, b = DrivingSim(seed=3), DrivingSim(seed=3)
    for _ in range(10):
        oa, *_ = a.step([0.1, 0.5]); ob, *_ = b.step([0.1, 0.5])
    assert np.allclose(oa["bev"], ob["bev"]) and np.allclose(a.ego, b.ego)

def test_dataset_shapes(tmp_path):
    p, _ = generate_dataset(str(tmp_path / "d.npz"), episodes=3, steps=20, seed=1)
    ds = DrivingDataset(p, seq=8); b = ds[0]
    assert b["bev"].shape == (8, 1, 32, 32) and b["scan"].shape == (8, 16) and b["action"].shape == (8, 2)

def test_world_model_rollout(tmp_path):
    wm = WorldModel(); obs = DrivingSim(seed=0).reset()
    z = wm.encode({k: torch.tensor(v[None]) for k, v in obs.items()})
    zs, rs, bev = wm.rollout(z, torch.zeros(1, 5, 2))
    assert zs.shape == (1, 5, 32) and rs.shape == (1, 5) and bev.shape == (1, 1, 32, 32)
    p, _ = generate_dataset(str(tmp_path / "d.npz"), episodes=2, steps=20, seed=0)
    assert set(horizon_errors(wm, p, horizons=(1,))) == {1}

def test_planner_constraints():
    cem_prev = np.array([0.0, 0.0]); out = apply_constraints(np.array([0.5, 3.0]), cem_prev)
    assert abs(out[0]) <= 0.5 and abs(out[1]) <= 3.0
    ac = ActorCritic(); train_in_imagination(WorldModel(), ac, steps=2)
    cem = CEMPlanner(WorldModel(), horizon=3, samples=16, elites=4, iters=1)
    act = cem.act(DrivingSim(seed=0).reset()); assert act.shape == (2,) and np.all(np.abs(act) <= [0.5, 3.0])

def test_metrics_and_bc_student():
    res = benchmark({"expert": expert_policy}, seeds=(0,), max_steps=20)
    assert 0 <= res["expert"]["route_completion"] <= 1 and "per_scenario" in res["expert"]
    bc = BCPolicy(); obs = DrivingSim(seed=0).reset()
    out = bc(*[torch.tensor(obs[k][None]) for k in ("bev", "scan", "ego", "goal")])
    assert out.shape == (1, 2)
    s = StudentPolicy(); assert s(torch.tensor(obs_feat(obs)[None])).shape == (1, 2)

def test_onnx_parity(tmp_path):
    import pytest; ort = pytest.importorskip("onnxruntime")
    from dwm.distill import export_onnx
    s = StudentPolicy().eval(); p = export_onnx(s, str(tmp_path / "s.onnx"))
    x = np.random.randn(1, s.net[0].in_features).astype(np.float32)
    sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    y = sess.run(None, {"obs": x})[0]
    with torch.no_grad(): yt = s(torch.tensor(x)).numpy()
    assert np.allclose(y, yt, atol=1e-4)
