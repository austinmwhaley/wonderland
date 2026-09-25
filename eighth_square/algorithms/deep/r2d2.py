import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseAgent, EpsilonScheduler
from .replay import SumTree


def rescale(x, eps=1e-3):
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + eps * x


def unrescale(y, eps=1e-3):
    return torch.sign(y) * (
        ((torch.sqrt(1.0 + 4.0 * eps * (torch.abs(y) + 1.0 + eps)) - 1.0) / (2.0 * eps)) ** 2 - 1.0
    )


class RecurrentQNetwork(nn.Module):
    def __init__(self, in_dim, hidden, n_actions, out_heads=1):
        super().__init__()
        self.n_actions = n_actions
        self.out_heads = out_heads
        self.features = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_heads * n_actions)
        )

    def forward(self, x, state=None):
        h = self.features(x)
        out, state = self.lstm(h, state)
        q = self.head(out)
        if self.out_heads == 1:
            return q, state
        return q.view(x.size(0), x.size(1), self.out_heads, self.n_actions), state


class SequenceReplayBuffer:
    """Stores episodes; every position yields a replay sequence of
    burn_in + seq_len + n_step steps (clipped at the episode boundary)."""

    def __init__(
        self,
        capacity,
        obs_dim,
        device,
        seq_len,
        burn_in,
        n_step,
        alpha=0.6,
        beta=0.4,
        beta_steps=100_000,
        eps=1e-6,
    ):
        self.device = device
        self.seq_len = seq_len
        self.burn_in = burn_in
        self.n_step = n_step
        self.alpha = alpha
        self.beta = beta
        self.beta_steps = beta_steps
        self.eps = eps
        self.obs_dim = obs_dim
        self.tree = SumTree(capacity)
        self.episodes = []
        self.replay_counts = {}
        self._step = 0

    def push_episode(self, obs, act, rew, done, param_idx=0):
        ep_idx = len(self.episodes)
        self.episodes.append(
            (
                np.asarray(obs, dtype=np.float32),
                np.asarray(act, dtype=np.int64),
                np.asarray(rew, dtype=np.float32),
                np.asarray(done, dtype=np.float32),
                int(param_idx),
            )
        )
        p = max(1.0, float(max(abs(r) for r in rew) if len(rew) else 1.0))
        for start in range(len(rew)):
            self.tree.add(p**self.alpha, (ep_idx, start))
            self.replay_counts[(ep_idx, start)] = 0

    def _window(self, ep_idx, start):
        obs, act, rew, done, param_idx = self.episodes[ep_idx]
        n = len(rew)
        burn = min(self.burn_in, start)
        lo = start - burn
        hi = min(n, start + self.seq_len + self.n_step)
        t = torch.from_numpy
        obs_w = t(obs[lo:hi]).to(self.device)
        act_w = t(act[lo:hi]).to(self.device)
        rew_w = t(rew[lo:hi]).to(self.device)
        done_w = t(done[lo:hi]).to(self.device)
        mask = torch.zeros(hi - lo, dtype=torch.bool, device=self.device)
        mask[burn : burn + self.seq_len] = True
        return obs_w, act_w, rew_w, done_w, mask, param_idx

    def sample(self, batch_size):
        self._step += 1
        beta = min(1.0, self.beta + (1 - self.beta) * self._step / self.beta_steps)
        idxs, weights, seqs = [], [], []
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            idx, p, d = self.tree.get(np.random.uniform(a, b))
            idxs.append(idx)
            weights.append((p * self.tree.size) ** (-beta))
            seqs.append(self._window(*d))
            self.replay_counts[d] = self.replay_counts.get(d, 0) + 1
        weights = torch.as_tensor(weights, device=self.device)
        weights = (weights / weights.max()).unsqueeze(1)
        tmax = max(s[0].size(0) for s in seqs)
        obs = torch.stack([F.pad(s[0], (0, 0, 0, tmax - s[0].size(0)), value=0.0) for s in seqs])
        act = torch.stack([F.pad(s[1], (0, tmax - s[1].size(0)), value=0) for s in seqs])
        rew = torch.stack([F.pad(s[2], (0, tmax - s[2].size(0)), value=0.0) for s in seqs])
        done = torch.stack([F.pad(s[3], (0, tmax - s[3].size(0)), value=1.0) for s in seqs])
        mask = torch.stack([F.pad(s[4], (0, tmax - s[4].size(0)), value=False) for s in seqs])
        param = torch.as_tensor([s[5] for s in seqs], dtype=torch.long, device=self.device)
        return obs, act, rew, done, mask, param, idxs, weights

    def update_priorities(self, idxs, priorities):
        for idx, p in zip(idxs, priorities):
            self.tree.update(idx, (p + self.eps) ** self.alpha)


class R2D2(BaseAgent):
    """Recurrent Replay Distributed DQN (Kapturowski et al. 2019), single-learner
    port: a Q-network with an LSTM layer trained on sequences drawn from
    prioritized replay with burn-in, n-step returns clipped at episode
    boundaries, value-function rescaling h(x) = sign(x)(sqrt(|x|+1) - 1) + eps*x,
    and each stored sequence replayed a bounded number of times."""

    family = "value-based"
    policy = "off-policy"
    action_space = "discrete"
    state_space = "continuous"

    def __init__(self, env, config):
        super().__init__(env, config)
        self.device = config.get("device", "cpu")
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("lr", 1e-4)
        self.hidden = config.get("hidden", 128)
        self.batch_size = config.get("batch_size", 32)
        self.update_freq = config.get("update_freq", 1)
        self.target_freq = config.get("target_freq", 1000)
        self.seq_len = config.get("seq_len", 40)
        self.burn_in = config.get("burn_in", 20)
        self.n_step = config.get("n_step", 5)
        self.max_replay = config.get("max_replay", 2)
        self.warmup = config.get("warmup", 1000)
        self.eval_freq = config.get("eval_freq", 5_000)
        self.eval_episodes = config.get("eval_episodes", 5)
        in_dim = int(np.prod(env.observation_space.shape))
        self.nA = int(env.action_space.n)
        self.online = RecurrentQNetwork(in_dim, self.hidden, self.nA).to(self.device)
        self.target = RecurrentQNetwork(in_dim, self.hidden, self.nA).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.lr)
        capacity = config.get("buffer_size", 100_000)
        self.buffer = SequenceReplayBuffer(
            capacity,
            in_dim,
            self.device,
            self.seq_len,
            self.burn_in,
            self.n_step,
            alpha=config.get("p_alpha", 0.6),
            beta=config.get("p_beta", 0.4),
            beta_steps=config.get("steps", 100_000),
        )
        self.sched = EpsilonScheduler(
            config.get("eps_start", 1.0),
            config.get("eps_end", 0.01),
            config.get("eps_decay_steps", config.get("steps", 100_000)),
            self.rng,
        )
        self.gamma_n = self.gamma**self.n_step
        self.t = 0
        self._state = None

    def _t(self, obs):
        return torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).view(1, 1, -1)

    def act(self, state, eval=False):
        self.online.eval() if eval else self.online.train()
        with torch.no_grad():
            q, self._state = self.online(self._t(state), self._state)
            q = q.view(1, -1)
            if eval or self.rng.random() >= self.sched.epsilon(self.t):
                return int(q.argmax().item())
            return int(self.rng.integers(self.nA))

    def train(self, env, config, tracker):
        state, _ = env.reset()
        self.t = 0
        ep = 0
        losses = []
        ep_obs, ep_act, ep_rew, ep_done = [], [], [], []
        while self.t < config["steps"]:
            a = self.act(state)
            ns, r, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            ep_obs.append(state)
            ep_act.append(a)
            ep_rew.append(r)
            ep_done.append(1.0 if done else 0.0)
            state = ns
            self.t += 1
            if done:
                ep += 1
                self.buffer.push_episode(ep_obs, ep_act, ep_rew, ep_done)
                tracker.log(
                    timestep=self.t,
                    episode=ep,
                    ret=float(sum(ep_rew)),
                    loss=float(np.mean(losses)) if losses else None,
                )
                ep_obs, ep_act, ep_rew, ep_done = [], [], [], []
                losses = []
                state, _ = env.reset()
                self._state = None
            if self.t >= self.warmup and self.t % self.update_freq == 0:
                for _ in range(config.get("updates_per_step", 1)):
                    losses.append(self._update())
            if self.t % self.target_freq == 0 and self.t > 0:
                self.target.load_state_dict(self.online.state_dict())
            if self.t % self.eval_freq == 0 and self.t > 0:
                returns = []
                for _ in range(self.eval_episodes):
                    s, _ = env.reset()
                    self._state = None
                    d, r_e, st = False, 0.0, 0
                    while not d:
                        a = self.act(s, eval=True)
                        s, rr, term, trunc, _ = env.step(a)
                        d = bool(term or trunc)
                        r_e += rr
                        st += 1
                        if st > 10_000:
                            break
                    returns.append(r_e)
                tracker.log(timestep=self.t, eval_return=float(np.mean(returns)))
                state, _ = env.reset()
                self._state = None
        self.episodes = ep

    def _update(self):
        obs, act, rew, done, mask, param, idxs, weights = self.buffer.sample(self.batch_size)
        B, T, D = obs.shape
        q_all, _ = self.online(obs)
        q = q_all.gather(2, act.unsqueeze(2)).squeeze(2)
        with torch.no_grad():
            q2_all, _ = self.target(obs)
            a2 = self.online(obs)[0].argmax(-1)
            q2 = q2_all.gather(2, a2.unsqueeze(2)).squeeze(2)
            gammas = [self.gamma**k for k in range(self.n_step)]
            target = torch.zeros_like(rew)
            for t in range(T):
                G = rew[:, t].clone()
                live = done[:, t] <= 0.5
                for k in range(1, self.n_step):
                    if t + k < T:
                        G = G + gammas[k] * rew[:, t + k] * live.float()
                        live = live & (done[:, t + k] <= 0.5)
                if t + self.n_step < T:
                    G = G + self.gamma_n * unrescale(q2[:, t + self.n_step]) * live.float()
                target[:, t] = G
        td = rescale(target) - q
        td = torch.where(mask, td, torch.zeros_like(td))
        loss = (td**2 * weights).sum() / mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 40.0)
        self.optimizer.step()
        prio = td.detach().abs().amax(-1)
        self.buffer.update_priorities(idxs, prio.cpu().numpy())
        return float(loss.item())

    def save(self, path):
        torch.save({"online": self.online.state_dict(), "target": self.target.state_dict()}, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.online.load_state_dict(data["online"])
        self.target.load_state_dict(data["target"])
