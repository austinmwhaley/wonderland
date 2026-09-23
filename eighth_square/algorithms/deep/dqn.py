from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from ..base import BaseAgent, EpsilonScheduler, evaluate
from .networks import QNetwork, polyak_copy
from .replay import PrioritizedReplayBuffer, ReplayBuffer


class DQN(BaseAgent):
    family = "value-based"
    policy = "off-policy"
    action_space = "discrete"
    state_space = "continuous"

    def __init__(self, env, config):
        super().__init__(env, config)
        self.device = config.get("device", "cpu")
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("lr", 1e-3)
        self.hidden = config.get("hidden", 128)
        self.batch_size = config.get("batch_size", 64)
        self.update_freq = config.get("update_freq", 4)
        self.target_freq = config.get("target_freq", 1000)
        self.tau = config.get("tau", 0.0)
        self.double = config.get("double", False)
        self.dueling = config.get("dueling", False)
        self.prioritized = config.get("prioritized", False)
        self.noisy = config.get("noisy", False)
        self.n_step = config.get("n_step", 1)
        self.n_atoms = config.get("n_atoms", 0)
        self.n_quantiles = config.get("n_quantiles", 0)
        self.v_min = config.get("v_min", -10.0)
        self.v_max = config.get("v_max", 10.0)
        self.warmup = config.get("warmup", 1000)
        self.eval_freq = config.get("eval_freq", 5_000)
        self.eval_episodes = config.get("eval_episodes", 5)
        in_dim = int(getattr(env.observation_space, "n", None) or np.prod(env.observation_space.shape))
        self.nA = int(env.action_space.n)
        out_heads = max(self.n_atoms, self.n_quantiles, 1)
        self.online = QNetwork(in_dim, self.hidden, self.nA, self.dueling, self.noisy, out_heads).to(self.device)
        self.target = QNetwork(in_dim, self.hidden, self.nA, self.dueling, self.noisy, out_heads).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.lr)
        capacity = config.get("buffer_size", 100_000)
        if self.prioritized:
            self.buffer = PrioritizedReplayBuffer(capacity, in_dim, self.device,
                                                  alpha=config.get("p_alpha", 0.6),
                                                  beta=config.get("p_beta", 0.4),
                                                  beta_steps=config.get("steps", 100_000))
        else:
            self.buffer = ReplayBuffer(capacity, in_dim, self.device)
        self.sched = EpsilonScheduler(config.get("eps_start", 1.0), config.get("eps_end", 0.01),
                                      config.get("eps_decay_steps", config.get("steps", 100_000)), self.rng)
        self.gamma_n = self.gamma ** self.n_step
        self.t = 0

    def _t(self, state):
        return torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)

    def _q_means(self, net, x):
        if self.n_atoms:
            z = self._atoms()
            logits = net(x).view(x.size(0), self.nA, self.n_atoms)
            probs = torch.softmax(logits, dim=-1)
            return (probs * z).sum(-1)
        if self.n_quantiles:
            out = net(x).view(x.size(0), self.nA, self.n_quantiles)
            return out.mean(-1)
        return net(x)

    def _atoms(self):
        return torch.linspace(self.v_min, self.v_max, self.n_atoms, device=self.device)

    def act(self, state, eval=False):
        self.online.eval() if eval else self.online.train()
        with torch.no_grad():
            q = self._q_means(self.online, self._t(state))
            if eval or self.noisy or self.rng.random() >= self.sched.epsilon(self.t):
                return int(q.argmax().item())
            return int(self.rng.integers(self.nA))

    def _n_step_return(self):
        ring = list(self.ring)
        n = len(ring)
        G = 0.0
        done_at = None
        for i in range(n):
            G += self.gamma ** i * ring[i][2]
            if ring[i][3]:
                done_at = i
                break
        if done_at is not None:
            s2 = ring[done_at][0]
            done = True
        else:
            s2 = self._s2
            done = False
        return ring[0][0], ring[0][1], G, s2, done

    def push(self, s, a, r, s2, done):
        self._s2 = s2
        self.ring.append((s, a, r, done))
        if len(self.ring) == self.n_step:
            s0, a0, G, s2n, done_n = self._n_step_return()
            if self.prioritized:
                self.buffer.push(s0, a0, G, s2n, done_n)
            else:
                self.buffer.push(s0, a0, G, s2n, done_n)

    def _target_values(self, obs2, done):
        with torch.no_grad():
            if self.double:
                a2 = self._q_means(self.online, obs2).argmax(-1, keepdim=True)
                q2 = self._q_means(self.target, obs2)
                q2 = q2.gather(1, a2)
            else:
                q2 = self._q_means(self.target, obs2).max(-1, keepdim=True).values
            return q2

    def _loss(self, obs, act, rew, obs2, done, weights):
        if self.n_atoms:
            return self._categorical_loss(obs, act, rew, obs2, done, weights)
        if self.n_quantiles:
            return self._quantile_loss(obs, act, rew, obs2, done, weights)
        q = self.online(obs).gather(1, act.unsqueeze(1))
        target = rew + self.gamma_n * (1 - done) * self._target_values(obs2, done)
        loss = F.smooth_l1_loss(q, target, reduction="none")
        return (loss * weights).mean(), loss.detach().abs().squeeze(1)

    def _categorical_loss(self, obs, act, rew, obs2, done, weights):
        z = self._atoms()
        logits = self.online(obs).view(-1, self.nA, self.n_atoms)
        probs = torch.softmax(logits, dim=-1)
        idx = act.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.n_atoms)
        p_online = probs.gather(1, idx).squeeze(1).clamp(min=1e-8)
        with torch.no_grad():
            if self.double:
                a2 = self._q_means(self.online, obs2).argmax(-1, keepdim=True)
                t_logits = self.target(obs2).view(-1, self.nA, self.n_atoms)
                t_probs = torch.softmax(t_logits, dim=-1)
                a2 = a2.unsqueeze(2).expand(-1, -1, self.n_atoms)
                p_next = t_probs.gather(1, a2).squeeze(1)
            else:
                t_logits = self.target(obs2).view(-1, self.nA, self.n_atoms)
                t_probs = torch.softmax(t_logits, dim=-1)
                q_next = (t_probs * z).sum(-1)
                a2 = q_next.argmax(-1, keepdim=True).unsqueeze(2).expand(-1, -1, self.n_atoms)
                p_next = t_probs.gather(1, a2).squeeze(1)
            m = self._project(p_next, rew, done, z)
        loss = -(m * p_online.log()).sum(-1, keepdim=True)
        td = torch.abs(self._project(p_next, rew, done, z) - p_online).sum(-1, keepdim=True)
        return (loss * weights).mean(), td.squeeze(1)

    def _project(self, p_next, rew, done, z):
        delta = (self.v_max - self.v_min) / (self.n_atoms - 1)
        tz = rew + self.gamma_n * (1 - done) * z.unsqueeze(0)
        tz = tz.clamp(self.v_min, self.v_max)
        b = (tz - self.v_min) / delta
        l = b.floor().long()
        u = b.ceil().long()
        m = torch.zeros_like(p_next)
        for i in range(self.n_atoms):
            m.scatter_add_(1, l[:, i:i + 1], p_next[:, i:i + 1] * (u[:, i:i + 1] - b[:, i:i + 1]).float())
            m.scatter_add_(1, u[:, i:i + 1], p_next[:, i:i + 1] * (b[:, i:i + 1] - l[:, i:i + 1]).float())
        return m

    def _quantile_loss(self, obs, act, rew, obs2, done, weights):
        N = self.n_quantiles
        tau = (2 * torch.arange(N, device=self.device) + 1) / (2 * N)
        online = self.online(obs).view(-1, self.nA, N)
        idx = act.unsqueeze(1).unsqueeze(2).expand(-1, -1, N)
        q_online = online.gather(1, idx).squeeze(1)
        with torch.no_grad():
            if self.double:
                a2 = self._q_means(self.online, obs2).argmax(-1, keepdim=True)
                q2 = self.target(obs2).view(-1, self.nA, N)
                a2 = a2.unsqueeze(2).expand(-1, -1, N)
                z_next = q2.gather(1, a2).squeeze(1)
            else:
                q2 = self.target(obs2).view(-1, self.nA, N)
                a2 = q2.mean(-1).argmax(-1, keepdim=True).unsqueeze(2).expand(-1, -1, N)
                z_next = q2.gather(1, a2).squeeze(1)
            target = rew + self.gamma_n * (1 - done) * z_next
        delta = target.unsqueeze(1) - q_online.unsqueeze(2)
        huber = torch.where(delta.abs() <= 1.0, 0.5 * delta ** 2, delta.abs() - 0.5)
        loss = (torch.abs(tau.unsqueeze(0).unsqueeze(-1) - (delta < 0).float()) * huber).mean(-1)
        td = delta.mean(-1).abs()
        return (loss * weights).mean(), td.squeeze(1)

    def _update(self):
        if self.prioritized:
            obs, act, rew, obs2, done, idxs, weights = self.buffer.sample(self.batch_size)
        else:
            obs, act, rew, obs2, done = self.buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size, 1, device=self.device)
        loss, td = self._loss(obs, act, rew, obs2, done, weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.prioritized:
            self.buffer.update_priorities(idxs, td.detach().cpu().numpy())
        return float(loss.item())

    def train(self, env, config, tracker):
        self.ring = deque(maxlen=self.n_step)
        self._s2 = None
        state, _ = env.reset()
        self.t = 0
        ep = 0
        ep_ret = 0.0
        losses = []
        while self.t < config["steps"]:
            a = self.act(state)
            ns, r, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            self.push(state, a, r, ns, done)
            ep_ret += r
            state = ns
            self.t += 1
            if done:
                ep += 1
                tracker.log(timestep=self.t, episode=ep, ret=float(ep_ret),
                            loss=float(np.mean(losses)) if losses else None)
                ep_ret = 0.0
                losses = []
                state, _ = env.reset()
            if self.t >= self.warmup and self.t % self.update_freq == 0:
                losses.append(self._update())
            if self.tau > 0:
                polyak_copy(self.online, self.target, self.tau)
            elif self.t % self.target_freq == 0 and self.t > 0:
                self.target.load_state_dict(self.online.state_dict())
            if self.t % self.eval_freq == 0 and self.t > 0:
                ev = evaluate(self, env, self.eval_episodes)
                tracker.log(timestep=self.t, eval_return=ev)
                state, _ = env.reset()
        self.episodes = ep

    def save(self, path):
        torch.save({"online": self.online.state_dict(), "target": self.target.state_dict()}, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.online.load_state_dict(data["online"])
        self.target.load_state_dict(data["target"])


class DoubleDQN(DQN):
    def __init__(self, env, config):
        config["double"] = True
        super().__init__(env, config)


class DuelingDQN(DQN):
    def __init__(self, env, config):
        config["dueling"] = True
        super().__init__(env, config)


class PrioritizedDQN(DQN):
    def __init__(self, env, config):
        config["double"] = True
        config["prioritized"] = True
        super().__init__(env, config)


class C51(DQN):
    def __init__(self, env, config):
        config["n_atoms"] = config.get("n_atoms", 51)
        config["v_min"] = config.get("v_min", -10.0)
        config["v_max"] = config.get("v_max", 10.0)
        super().__init__(env, config)


class QRDQN(DQN):
    def __init__(self, env, config):
        config["n_quantiles"] = config.get("n_quantiles", 32)
        super().__init__(env, config)


class Rainbow(DQN):
    def __init__(self, env, config):
        config["double"] = True
        config["dueling"] = True
        config["prioritized"] = True
        config["noisy"] = True
        config["n_atoms"] = config.get("n_atoms", 51)
        config["n_step"] = config.get("n_step", 3)
        super().__init__(env, config)


class HERDQN(DQN):
    """DQN + Hindsight Experience Replay for sparse-reward goal reaching.

    Goal appears only in the last 2 obs dims (delta=(goal-pos)/W). We track the
    drone's absolute position each step; after each episode a fraction `frac` of
    transitions are replayed with a HINDSIGHT goal = the position actually reached,
    recomputing the delta and treating the final step as a success (+1, done).

    Requires obs_mode='lidar8' (goal = last 2 channels). Disables n-step so rewards
    can be relabeled cleanly.
    """

    def __init__(self, env, config):
        config["n_step"] = 1
        super().__init__(env, config)
        self.env = env
        self.her_frac = config.get("her_frac", 1.0)
        self.her_k = config.get("her_k", 4)
        # must be able to track absolute pos: env exposes .pos
        assert hasattr(env, "pos"), "HERDQN requires env exposing .pos"
        self._W = float(env.world_size)

    def train(self, env, config, tracker):
        self.ring = deque(maxlen=1)
        self._s2 = None
        self.t = 0
        ep = 0
        ep_ret = 0.0
        losses = []
        ep_obs, ep_act, ep_pos = [], [], []
        state, _ = env.reset()
        while self.t < config["steps"]:
            a = self.act(state)
            ns, r, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            ep_obs.append(np.asarray(state, dtype=np.float32))
            ep_act.append(a)
            ep_pos.append(np.array(env.pos, dtype=np.float32))
            # only push original when buffer believes HER (store original too)
            self.push(state, a, r, ns, done)
            ep_ret += r
            state = ns
            self.t += 1
            if done:
                ep += 1
                # hindsight relabel
                if not bool(term):
                    self._her(ep_obs, ep_act, ep_pos)
                elif np.random.random() < self.her_frac:
                    self._her(ep_obs, ep_act, ep_pos)
                ep_obs, ep_act, ep_pos = [], [], []
                tracker.log(timestep=self.t, episode=ep, ret=float(ep_ret),
                            loss=float(np.mean(losses)) if losses else None)
                ep_ret = 0.0
                losses = []
                state, _ = env.reset()
            if self.t >= self.warmup and self.t % self.update_freq == 0:
                losses.append(self._update())
            if self.tau > 0:
                polyak_copy(self.online, self.target, self.tau)
            elif self.t % self.target_freq == 0 and self.t > 0:
                self.target.load_state_dict(self.online.state_dict())
            if self.t % self.eval_freq == 0 and self.t > 0:
                ev = evaluate(self, env, self.eval_episodes)
                tracker.log(timestep=self.t, eval_return=ev)
                state, _ = env.reset()
        self.episodes = ep

    def _her(self, ep_obs, ep_act, ep_pos):
        # ep_obs/ep_act/ep_pos each have n entries; ep_pos[i] = pos before transition i (world coords).
        n = len(ep_obs)
        if n < 2:
            return
        achieved = ep_pos[-1] / self._W
        pn = [p / self._W for p in ep_pos]
        # relabel EVERY transition to the hindsight goal (matches episode length -> balanced buffer)
        for i in range(n):
            s = ep_obs[i].copy()
            s[-2:] = achieved - pn[i]
            if i + 1 < n:
                s2 = ep_obs[i + 1].copy()
                s2[-2:] = achieved - pn[i + 1]
                r, d = 0.0, 0.0
            else:
                s2 = s.copy()
                r, d = 1.0, 1.0
            self.buffer.push(s, ep_act[i], r, s2, d)


class RainbowHER(HERDQN):
    """HER + Rainbow ingredients for sparse-reward goal reaching.

    Uses prioritized replay, double Q, and dueling (the highest-value Rainbow
    ingredients for sample efficiency). n_step stays 1 so HER goal-relabeling
    stays exact; noisy nets are omitted for the same reason.
    """

    def __init__(self, env, config):
        config["double"] = True
        config["dueling"] = True
        config["prioritized"] = True
        super().__init__(env, config)