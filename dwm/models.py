"""VAE encoder + causal Transformer dynamics, BC policy, student policy."""
import torch
import torch.nn.functional as F
from torch import nn

LATENT = 32


class Encoder(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 16, 4, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
                                  nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(), nn.Flatten())  # 32*4*4
        self.aux = nn.Sequential(nn.Linear(16 + 5 + 2, 64), nn.ReLU())
        self.mu = nn.Linear(32 * 16 + 64, latent); self.logvar = nn.Linear(32 * 16 + 64, latent)
        self.dec = nn.Sequential(nn.Linear(latent, 32 * 16), nn.ReLU())
        self.deconv = nn.Sequential(nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
                                    nn.ConvTranspose2d(16, 16, 4, 2, 1), nn.ReLU(),
                                    nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.Sigmoid())
    def encode(self, bev, scan, ego, goal):
        h = torch.cat([self.conv(bev), self.aux(torch.cat([scan, ego, goal], -1))], -1)
        return self.mu(h), self.logvar(h)
    def decode(self, z):
        return self.deconv(self.dec(z).view(-1, 32, 4, 4))
    def forward(self, bev, scan, ego, goal):
        mu, lv = self.encode(bev, scan, ego, goal)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp() if self.training else mu
        return z, mu, lv, self.decode(z)


class Dynamics(nn.Module):
    """Causal Transformer over (latent, action) tokens -> next latent/reward/collision/done."""
    def __init__(self, latent=LATENT, d=128, layers=2, heads=4):
        super().__init__()
        self.inp = nn.Linear(latent + 2, d)
        self.pos = nn.Parameter(torch.zeros(1, 64, d))
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 256, batch_first=True), layers)
        self.next_z = nn.Linear(d, latent); self.head = nn.Linear(d, 3)  # reward, collision, done logits-ish
    def forward(self, z, a):  # z (B,T,L), a (B,T,2)
        h = self.inp(torch.cat([z, a], -1)) + self.pos[:, : z.size(1)]
        mask = torch.triu(torch.ones(z.size(1), z.size(1), device=z.device, dtype=torch.bool), 1)
        h = self.tf(h, mask=mask)
        out = self.head(h)
        return self.next_z(h), out[..., 0], out[..., 1], out[..., 2]


class WorldModel(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__(); self.enc = Encoder(latent); self.dyn = Dynamics(latent)
    def encode(self, obs):
        return self.enc.encode(obs["bev"], obs["scan"], obs["ego"], obs["goal"])[0]
    def decode(self, z):
        return self.enc.decode(z)
    def rollout(self, z0, actions):  # (B,L),(B,T,2) -> predicted latents (B,T,L), rewards, decoded last BEV
        z, zs, rs = z0, [], []
        for t in range(actions.size(1)):
            nz, r, c, d = self.dyn(z.unsqueeze(1), actions[:, t:t + 1])
            z = nz[:, -1]; zs.append(z); rs.append(r[:, -1])
        zs = torch.stack(zs, 1)
        return zs, torch.stack(rs, 1), self.decode(zs[:, -1])


def wm_loss(wm, batch):
    z, mu, lv, rec = wm.enc(batch["bev"][:, 0], batch["scan"][:, 0], batch["ego"][:, 0], batch["goal"][:, 0])
    recon = F.mse_loss(rec, batch["bev"][:, 0])
    kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    # encode all steps (teacher forcing), predict next
    B, T = batch["bev"].shape[:2]
    flat = lambda k: batch[k].reshape(B * T, *batch[k].shape[2:])
    zs, _, _, _ = wm.enc(flat("bev"), flat("scan"), flat("ego"), flat("goal"))
    zs = zs.view(B, T, -1)
    nz, r, c, d = wm.dyn(zs[:, :-1], batch["action"][:, :-1])
    dyn = F.mse_loss(nz, zs[:, 1:].detach()) + F.mse_loss(r, batch["reward"][:, 1:]) \
        + F.binary_cross_entropy_with_logits(c, batch["collision"][:, 1:]) \
        + F.binary_cross_entropy_with_logits(d, batch["done"][:, 1:].float())
    return recon + 0.01 * kl + dyn, {"recon": float(recon.detach()), "dyn": float(dyn.detach())}


class BCPolicy(nn.Module):
    """Multimodal Transformer: BEV patch tokens + scan/ego/goal tokens -> action."""
    def __init__(self, d=64):
        super().__init__()
        self.patch = nn.Conv2d(1, d, 8, 8)  # 4x4 tokens
        self.aux = nn.Linear(16 + 5 + 2, d)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 128, batch_first=True), 2)
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, bev, scan, ego, goal):
        tok = self.patch(bev).flatten(2).transpose(1, 2)
        tok = torch.cat([tok, self.aux(torch.cat([scan, ego, goal], -1)).unsqueeze(1)], 1)
        return torch.tanh(self.head(self.tf(tok).mean(1))) * torch.tensor([0.5, 3.0])


class StudentPolicy(nn.Module):
    """Small distilled policy: flattened obs features -> action (ONNX-friendly)."""
    def __init__(self, in_dim=32 * 8 + 23):  # downsampled BEV handled by caller feat fn
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x):
        return torch.tanh(self.net(x)) * torch.tensor([0.5, 3.0], device=x.device)


def obs_feat(obs):
    """Shared student feature: avg-pooled BEV (8x32) + scan/ego/goal."""
    import numpy as np
    bev = obs["bev"]
    if isinstance(bev, torch.Tensor):
        single = bev.dim() == 3
        pooled = F.avg_pool2d(bev[None] if single else bev, (4, 1)).flatten(1)
        rest = [v.reshape(1, -1) if single else v for v in (obs["scan"], obs["ego"], obs["goal"])]
        out = torch.cat([pooled, *rest], -1)
        return out[0] if single else out
    pooled = bev.reshape(8, 4, 32).mean(1).reshape(-1).astype(np.float32)
    return np.concatenate([pooled, obs["scan"], obs["ego"], obs["goal"]]).astype(np.float32)
