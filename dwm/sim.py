"""Seeded 2D kinematic driving simulator.

Bicycle-model ego on a 3-lane straight road (plus scenario perturbations).
Other agents: IDM car-following + occasional lane changes. Pedestrians cross
in the pedestrian scenario; unstructured patches are off-road rectangles that
penalise but do not end the episode as a collision.
Obs: bev (1,32,32) BEV occupancy, scan (16,) LiDAR-like ranges, ego (5,), goal (2,).
Action: [steer rad (-0.5..0.5), accel m/s^2 (-3..3)], dt=0.2s.
"""
import numpy as np

DT, L = 0.2, 2.7
LANES = np.array([-3.5, 0.0, 3.5])
SCENARIOS = ["cruise", "merge", "cutin", "pedestrian", "unstructured"]


def idm_accel(v, v_lead, gap, desired=8.0):
    a, b, s0 = 1.5, 2.0, 4.0
    free = a * (1 - (v / max(desired, 1e-3)) ** 2)
    if gap is None or gap > 60:
        return float(np.clip(free, -3, 3))
    s_star = s0 + max(0.0, v * 1.0 + v * (v - v_lead) / (2 * np.sqrt(a * b)))
    return float(np.clip(a * (1 - (v / desired) ** 2 - (s_star / max(gap, 0.1)) ** 2), -3, 3))


class DrivingSim:
    def __init__(self, seed=0, scenario="cruise", max_steps=120):
        self.rng = np.random.default_rng(seed)
        self.scenario, self.max_steps = scenario, max_steps
        self.reset(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        rng = self.rng
        self.t = 0
        self.ego = np.array([0.0, LANES[rng.integers(3)], 0.0, 6.0, 0.0])  # x,y,heading,v,steer
        self.goal = np.array([180.0, self.ego[1] + (3.5 if self.scenario == "merge" else 0.0)])
        self.agents = []  # x,y,v,lane_idx
        n = {"cruise": 3, "merge": 4, "cutin": 3, "pedestrian": 2, "unstructured": 2}[self.scenario]
        for i in range(n):
            lane = int(rng.integers(3))
            self.agents.append([20 + 25 * i + rng.uniform(0, 8), LANES[lane], rng.uniform(5, 8), lane])
        self.agents = np.array(self.agents, dtype=float) if self.agents else np.zeros((0, 4))
        if self.scenario == "cutin" and len(self.agents):
            self.agents[0] = [18.0, LANES[min(2, int(np.where(LANES == self.ego[1])[0][0]) + 1)], 6.0,
                              float(min(2, int(np.where(LANES == self.ego[1])[0][0]) + 1))]
        self.ped = np.array([60.0, -10.0, 0.8]) if self.scenario == "pedestrian" else None  # x,y,vy
        self.patch = np.array([50.0, self.ego[1], 14.0, 3.0]) if self.scenario == "unstructured" else None
        self.prev_accel, self.prev_steer, self.jerk = 0.0, 0.0, 0.0
        return self._obs()

    def _obs(self):
        bev = np.zeros((1, 32, 32), np.float32)  # ego-centred, 32m x 32m, 1m/cell approx
        def put(x, y, val):
            cx, cy = int((x - self.ego[0]) + 16), int((y - self.ego[1]) + 16)
            if 0 <= cx < 32 and 0 <= cy < 32:
                bev[0, cy, cx] = val
        for lx in range(-5, 60, 5):
            for ly in LANES:
                put(self.ego[0] + lx, ly + 1.75, 0.25)
        for a in self.agents:
            put(a[0], a[1], 1.0)
        if self.ped is not None:
            put(self.ped[0], self.ped[1], 0.8)
        if self.patch is not None:
            put(self.patch[0], self.patch[1], 0.5)
        put(self.ego[0], self.ego[1], 0.6)
        # LiDAR-like scan: 16 rays, range to nearest agent/ped in cone else 30
        scan = np.full(16, 30.0, np.float32)
        objs = [a[:2] for a in self.agents] + ([self.ped[:2]] if self.ped is not None else [])
        for i, ang in enumerate(np.linspace(-1.2, 1.2, 16)):
            d = scan[i]
            for o in objs:
                rel = o - self.ego[:2]
                if abs(np.arctan2(rel[1], rel[0]) - self.ego[2] - ang) < 0.08:
                    d = min(d, float(np.linalg.norm(rel)))
            scan[i] = d
        ego = np.array([self.ego[3], self.ego[2], self.ego[4], self.ego[1], self.prev_accel], np.float32)
        goal = np.array([self.goal[0] - self.ego[0], self.goal[1] - self.ego[1]], np.float32)
        return {"bev": bev, "scan": scan, "ego": ego, "goal": goal}

    def step(self, action):
        a = np.clip(np.asarray(action, float), [-0.5, -3.0], [0.5, 3.0])
        steer, accel = float(a[0]), float(a[1])
        self.jerk = (accel - self.prev_accel) / DT
        x, y, th, v, _ = self.ego
        v = max(0.0, v + accel * DT)
        th = th + v / L * np.tan(steer) * DT
        x, y = x + v * np.cos(th) * DT, y + v * np.sin(th) * DT
        self.ego = np.array([x, y, th, v, steer])
        # agents: IDM w.r.t. ego/ahead agent, cut-in agent drifts to ego lane once
        for i, ag in enumerate(self.agents):
            gap, v_lead = None, ag[2]
            for other in list(self.agents) + [np.array([self.ego[0], self.ego[1], self.ego[3], 0])]:
                if other is not ag and abs(other[1] - ag[1]) < 1.8 and 0 < other[0] - ag[0] < (gap or 1e9):
                    gap, v_lead = other[0] - ag[0] - 4.0, other[2]
            ag[2] = max(0.0, ag[2] + idm_accel(ag[2], v_lead, gap) * DT)
            ag[0] += ag[2] * DT
            if self.scenario == "cutin" and i == 0 and self.t > 10:
                ag[1] += np.clip(self.ego[1] - ag[1], -0.4 * DT * 3, 0.4 * DT * 3)
            elif self.rng.random() < 0.002:  # rare lane change
                ag[3] = float(np.clip(ag[3] + self.rng.choice([-1, 1]), 0, 2))
            ag[1] += np.clip(LANES[int(ag[3])] - ag[1], -DT, DT) if not (self.scenario == "cutin" and i == 0) else 0
        if self.ped is not None:
            self.ped[1] += self.ped[2] * DT
        self.prev_accel, self.prev_steer, self.t = accel, steer, self.t + 1
        collision = any(np.linalg.norm(a[:2] - self.ego[:2]) < 2.2 for a in self.agents) or (
            self.ped is not None and np.linalg.norm(self.ped[:2] - self.ego[:2]) < 1.5)
        offroad = abs(y) > 5.6 or (self.patch is not None and abs(x - self.patch[0]) < 7 and abs(y - self.patch[1]) < 1.6)
        progress = v * DT
        reward = progress * 0.1 - 5.0 * collision - 1.0 * offroad - 0.01 * abs(self.jerk) - 0.05 * abs(steer)
        done = bool(collision or self.t >= self.max_steps or x >= self.goal[0])
        info = {"collision": bool(collision), "offroad": bool(offroad), "jerk": float(self.jerk),
                "progress": float(x), "route_completion": float(np.clip(x / self.goal[0], 0, 1)),
                "scenario": self.scenario}
        return self._obs(), float(reward), done, info


def expert_action(sim, obs, noise=0.0, rng=None):
    """Pure-pursuit to goal lane + IDM-style accel; optional Gaussian noise."""
    ego = sim.ego
    target_y = sim.goal[1]
    if sim.ped is not None and abs(sim.ped[0] - ego[0]) < 15 and abs(sim.ped[1] - ego[1]) < 4:
        return np.array([0.0, -3.0])  # brake for pedestrian
    steer = float(np.clip(0.25 * (target_y - ego[1]) - 0.3 * ego[2], -0.5, 0.5))
    gap, v_lead = None, 8.0
    for a in sim.agents:
        if abs(a[1] - ego[1]) < 1.8 and a[0] > ego[0]:
            g = a[0] - ego[0] - 4.0
            if gap is None or g < gap:
                gap, v_lead = g, a[2]
    accel = idm_accel(ego[3], v_lead, gap)
    act = np.array([steer, accel])
    if noise and rng is not None:
        act = act + rng.normal(0, noise, 2)
    return np.clip(act, [-0.5, -3.0], [0.5, 3.0])
