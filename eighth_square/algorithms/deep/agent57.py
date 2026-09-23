from collections import deque

import numpy as np
import torch
import torch.nn as nn

from .r2d2 import R2D2, RecurrentQNetwork, rescale, unrescale


class IntrinsicPredictor(nn.Module):
    """RND-style predictor: P(s, a) -> features(s'), trained on the squared
    error against a frozen random target network."""

    def __init__(self, in_dim, hidden, n_actions, feat_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + n_actions, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, feat_dim),
        )
        self.target = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, feat_dim),
        )
        for p in self.target.parameters():
            p.requires_grad_(False)

    def features(self, obs2):
        return self.target(obs2)

    def forward(self, obs, act):
        act_oh = torch.zeros(obs.size(0), self.net[0].in_features - obs.size(1),
                             device=obs.device)
        act_oh.scatter_(1, act.unsqueeze(1), 1.0)
        return self.net(torch.cat([obs, act_oh], dim=-1))

    def intrinsic(self, obs, act, obs2):
        return (self.forward(obs, act) - self.features(obs2)).pow(2).mean(-1)


class Agent57(R2D2):
    """R2D2 with intrinsic exploration and a meta-controller (Badia et al.
    2020, Nature), single-learner port: a UCB bandit over (gamma, beta)
    parameterizations distributes training among them; each parameterization
    has its own value head and discount; an RND-style predictor supplies the
    intrinsic reward (std-normalized) scaled by beta."""

    family = "value-based"
    policy = "off-policy"
    action_space = "discrete"
    state_space = "continuous"

    def __init__(self, env, config):
        config["lr"] = config.get("lr", 1e-4)
        super().__init__(env, config)
        self.gammas = config.get("gammas", [0.97, 0.99, 0.9975, 0.9999])
        self.betas = config.get("betas", [0.5, 0.5, 1.0, 2.0])
        self.n_param = len(self.gammas)
        self.ucb_c = config.get("ucb_c", 1.0)
        self.window = config.get("window", 100)
        self.switch_interval = config.get("switch_interval", 10)
        self.param_returns = [deque(maxlen=self.window) for _ in range(self.n_param)]
        self.param_counts = [0] * self.n_param
        self.active_param = 0
        self._in_phase = 0
        self._int_mean = 0.0
        self._int_std = 1.0
        self._int_count = 0
        self.online = RecurrentQNetwork(int(np.prod(env.observation_space.shape)),
                                        self.hidden, self.nA,
                                        out_heads=self.n_param).to(self.device)
        self.target = RecurrentQNetwork(int(np.prod(env.observation_space.shape)),
                                        self.hidden, self.nA,
                                        out_heads=self.n_param).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.lr)
        self.predictor = IntrinsicPredictor(int(np.prod(env.observation_space.shape)),
                                            self.hidden, self.nA).to(self.device)
        self.pred_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-3)

    def _ucb_scores(self):
        scores = []
        n_ep = sum(self.param_counts)
        for i in range(self.n_param):
            w = self.param_returns[i]
            mu = sum(w) / len(w) if w else 0.0
            count = max(self.param_counts[i], 1)
            scores.append(mu + self.ucb_c * np.sqrt(np.log(n_ep + 1) / count))
        return scores

    def act(self, state, eval=False):
        self.online.eval() if eval else self.online.train()
        with torch.no_grad():
            q, self._state = self.online(self._t(state), self._state)
            q = q.view(1, -1)[:, self.active_param * self.nA:(self.active_param + 1) * self.nA]
            if eval or self.rng.random() >= self.sched.epsilon(self.t):
                return int(q.argmax().item())
            return int(self.rng.integers(self.nA))

    def train(self, env, config, tracker):
        state, _ = env.reset()
        self.t = 0
        ep = 0
        losses = []
        ep_obs, ep_act, ep_rew, ep_int, ep_done = [], [], [], [], []
        while self.t < config["steps"]:
            a = self.act(state)
            ns, r, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            with torch.no_grad():
                ri_raw = float(self.predictor.intrinsic(
                    self._t(state).view(1, -1), torch.as_tensor([a], device=self.device),
                    self._t(ns).view(1, -1)).item())
            self._int_count += 1
            self._int_mean += 0.001 * (ri_raw - self._int_mean)
            self._int_std += 0.001 * (abs(ri_raw - self._int_mean) - self._int_std)
            ri = ri_raw / max(self._int_std, 1e-6)
            ep_obs.append(state)
            ep_act.append(a)
            ep_rew.append(r)
            ep_int.append(ri)
            ep_done.append(1.0 if done else 0.0)
            state = ns
            self.t += 1
            if done:
                ep += 1
                beta = self.betas[self.active_param]
                rew_tot = [r + beta * i for r, i in zip(ep_rew, ep_int)]
                self.buffer.push_episode(ep_obs, ep_act, rew_tot, ep_done,
                                         param_idx=self.active_param)
                self.param_returns[self.active_param].append(float(sum(ep_rew)))
                self.param_counts[self.active_param] += 1
                self._in_phase += 1
                if self._in_phase >= self.switch_interval:
                    self._in_phase = 0
                    if sum(self.param_counts) < 2 * self.n_param * self.switch_interval:
                        self.active_param = int(sum(self.param_counts) // self.switch_interval) % self.n_param
                    else:
                        scores = self._ucb_scores()
                        self.active_param = int(np.argmax(scores))
                tracker.log(timestep=self.t, episode=ep, ret=float(sum(ep_rew)),
                            loss=float(np.mean(losses)) if losses else None,
                            active_param=self.active_param)
                ep_obs, ep_act, ep_rew, ep_int, ep_done = [], [], [], [], []
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
        qp = q_all.permute(0, 1, 3, 2)
        q_h = qp.gather(3, param.view(B, 1, 1, 1).expand(B, T, self.nA, 1)).squeeze(3)
        q = q_h.gather(2, act.unsqueeze(2)).squeeze(2)
        with torch.no_grad():
            q2_all, _ = self.target(obs)
            q2p = q2_all.permute(0, 1, 3, 2)
            a2 = q_all.argmax(-1)
            a2_p = a2.gather(2, param.view(B, 1, 1).expand(B, T, 1)).squeeze(2)
            q2_h = q2p.gather(3, param.view(B, 1, 1, 1).expand(B, T, self.nA, 1)).squeeze(3)
            q2 = q2_h.gather(2, a2_p.unsqueeze(2)).squeeze(2)
            target = torch.zeros_like(rew)
            for t in range(T):
                G = rew[:, t].clone()
                live = (done[:, t] <= 0.5)
                gamma = torch.as_tensor([self.gammas[p] for p in param.tolist()],
                                        device=self.device)
                for k in range(1, self.n_step):
                    if t + k < T:
                        G = G + (gamma ** k) * rew[:, t + k] * live.float()
                        live = live & (done[:, t + k] <= 0.5)
                if t + self.n_step < T:
                    G = G + (gamma ** self.n_step) * unrescale(q2[:, t + self.n_step]) * live.float()
                target[:, t] = G
        td = rescale(target) - q
        td = torch.where(mask, td, torch.zeros_like(td))
        loss = (td ** 2 * weights).sum() / mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 40.0)
        self.optimizer.step()
        self.buffer.update_priorities(idxs, td.detach().abs().amax(-1).cpu().numpy())
        with torch.no_grad():
            obs_n = obs[:, 1:]
            act_n = act[:, :-1]
            feat = self.predictor.features(obs_n).detach()
        pred = self.predictor(obs[:, :-1].reshape(-1, D), act[:, :-1].reshape(-1))
        pred = pred.view(B, T - 1, -1)
        pred_loss = (pred - feat).pow(2).mean()
        self.pred_optimizer.zero_grad()
        pred_loss.backward()
        self.pred_optimizer.step()
        return float(loss.item())
