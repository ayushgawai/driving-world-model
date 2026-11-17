"""CEM/MPPI planning over world model, Dreamer-style actor-critic, constraints."""
import numpy as np
import torch
from torch import nn

MAX_STEER_RATE, MAX_JERK = 0.5, 8.0  # per step / per s^2 bounds used in tests
DT = 0.2


def apply_constraints(action, prev_action):
    a = np.array(action, float, copy=True)
    a[0] = np.clip(a[0], prev_action[0] - MAX_STEER_RATE * DT * 5, prev_action[0] + MAX_STEER_RATE * DT * 5)
    max_da = MAX_JERK * DT * DT * 5
    a[1] = np.clip(a[1], prev_action[1] - max_da, prev_action[1] + max_da)
    return np.clip(a, [-0.5, -3.0], [0.5, 3.0])


class CEMPlanner:
    """MPPI/CEM: sample action sequences, roll out in world model, keep elites."""
    def __init__(self, wm, horizon=10, samples=64, elites=8, iters=2, seed=0):
        self.wm, self.H, self.N, self.E, self.iters = wm, horizon, samples, elites, iters
        self.rng = np.random.default_rng(seed); self.prev = np.zeros(2)
    @torch.no_grad()
    def act(self, obs):
        wm = self.wm; wm.eval()
        z0 = wm.encode({k: torch.tensor(np.array(v)[None], dtype=torch.float32) for k, v in obs.items()})
        mean = np.zeros((self.H, 2)); std = np.tile([0.2, 1.0], (self.H, 1))
        best = mean[0]
        for _ in range(self.iters):
            cand = np.clip(self.rng.normal(mean, std, (self.N, self.H, 2)), [-0.5, -3], [0.5, 3])
            zs, rs, _ = wm.rollout(z0.repeat(self.N, 1), torch.tensor(cand, dtype=torch.float32))
            # penalise predicted collision via latent norm proxy handled by reward head; use return
            ret = rs.sum(1).numpy()
            idx = np.argsort(ret)[-self.E:]
            mean, std = cand[idx].mean(0), cand[idx].std(0) + 1e-3
            best = cand[idx[-1], 0]
        out = apply_constraints(best, self.prev); self.prev = out
        return out


class ActorCritic(nn.Module):
    def __init__(self, latent=32):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(latent, 128), nn.ReLU(), nn.Linear(128, 2))
        self.critic = nn.Sequential(nn.Linear(latent, 128), nn.ReLU(), nn.Linear(128, 1))
    def act(self, z):
        return torch.tanh(self.actor(z)) * torch.tensor([0.5, 3.0])


def train_in_imagination(wm, ac, steps=30, horizon=8, batch=32, lr=3e-3, seed=0):
    """Dreamer-style: imagine rollouts from random latents, actor-critic update. Returns final loss."""
    torch.manual_seed(seed); opt = torch.optim.Adam(ac.parameters(), lr)
    loss_val = 0.0
    for _ in range(steps):
        z = torch.randn(batch, wm.enc.mu.out_features)
        zs, actions = [z], []
        for _ in range(horizon):
            a = ac.act(z); nz, r, c, d = wm.dyn(z.unsqueeze(1), a.unsqueeze(1))
            z = nz[:, -1]; zs.append(z); actions.append(a)
        zs = torch.stack(zs, 1); values = ac.critic(zs).squeeze(-1)
        # rewards re-evaluated via dynamics head on imagined pairs
        _, rs, _, _ = wm.dyn(zs[:, :-1].reshape(-1, zs.size(-1)).unsqueeze(1),
                             torch.stack(actions, 1).reshape(-1, 2).unsqueeze(1))
        rs = rs.view(batch, horizon)
        returns = rs + 0.95 * values[:, 1:]
        adv = (returns - values[:, :-1]).detach()
        loss = -(adv * values[:, :-1]).mean() - 0.01 * torch.stack(actions, 1).pow(2).mean() \
            + nn.functional.mse_loss(values[:, :-1], returns.detach())
        opt.zero_grad(); loss.backward(); opt.step(); loss_val = float(loss.detach())
    return loss_val
