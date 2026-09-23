import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseAgent, evaluate
from .common import collect_episodes, train_behavior


class DecisionTransformer(BaseAgent, nn.Module):
    """Decision Transformer (Chen et al. 2021, NeurIPS): offline trajectory
    transformer that conditions on the desired return-to-go and predicts
    actions autoregressively, framing RL as sequence modeling. Trained with
    cross-entropy on the actions of behavior-policy episodes."""

    family = "other"
    policy = "off-policy"
    action_space = "discrete"
    state_space = "continuous"

    def __init__(self, env, config):
        BaseAgent.__init__(self, env, config)
        nn.Module.__init__(self)
        self.device = config.get("device", "cpu")
        self.lr = config.get("lr", 1e-4)
        self.context = config.get("context", 20)
        self.d_model = config.get("d_model", 64)
        self.n_head = config.get("n_head", 2)
        self.n_layer = config.get("n_layer", 2)
        self.batch_size = config.get("batch_size", 32)
        self.eval_freq = config.get("eval_freq", 5_000)
        self.eval_episodes = config.get("eval_episodes", 5)
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.nA = int(env.action_space.n)
        self.max_len = 3 * self.context
        self.obs_emb = nn.Linear(self.obs_dim, self.d_model)
        self.rtg_emb = nn.Linear(1, self.d_model)
        self.act_emb = nn.Linear(self.nA, self.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_len, self.d_model))
        layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_head,
                                          dim_feedforward=4 * self.d_model,
                                          batch_first=True, dropout=0.0)
        self.transformer = nn.TransformerEncoder(layer, num_layers=self.n_layer)
        self.head = nn.Linear(self.d_model, self.nA)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self._traj = deque(maxlen=self.context)
        self._last_act = None
        self._rtg = 0.0

    def reset_episode(self):
        self._traj.clear()
        self._last_act = None

    def _mask(self, L):
        m = torch.triu(torch.ones(L, L, dtype=torch.bool, device=self.device), diagonal=1)
        return m

    def _forward(self, obs, rtg, act):
        B, K, D = obs.shape
        B, K, A = act.shape
        L = 3 * K
        obs = self.obs_emb(obs).unsqueeze(2)
        rtg = self.rtg_emb(rtg).unsqueeze(2)
        act = self.act_emb(act).unsqueeze(2)
        tok = torch.cat([rtg, obs, act], dim=2).view(B, L, self.d_model)
        tok = tok + self.pos_emb[:, :L]
        h = self.transformer(tok, mask=self._mask(L))
        logits = self.head(h).view(B, K, 3, self.nA)
        return logits[:, :, 1]

    def _update(self):
        idx = self.rng.integers(0, len(self.episodes), size=self.batch_size)
        ob = np.zeros((self.batch_size, self.context, self.obs_dim), dtype=np.float32)
        rt = np.zeros((self.batch_size, self.context, 1), dtype=np.float32)
        ac = np.zeros((self.batch_size, self.context, self.nA), dtype=np.float32)
        for i, j in enumerate(idx):
            obs, act, rew = self.episodes[int(j)]
            n = min(len(obs), self.context)
            start = int(self.rng.integers(0, len(obs) - n + 1))
            ob[i, -n:] = obs[start:start + n]
            ac[i, -n:] = np.eye(self.nA)[act[start:start + n]]
            G = np.cumsum(rew[start:start + n][::-1])[::-1]
            rt[i, -n:, 0] = G
        obs_t = torch.as_tensor(ob, device=self.device)
        rtg_t = torch.as_tensor(rt, device=self.device)
        act_t = torch.as_tensor(ac, device=self.device)
        logits = self._forward(obs_t, rtg_t, act_t)
        with torch.no_grad():
            act_target = torch.as_tensor(ac, device=self.device).argmax(-1)
            mask = (act_t.sum(-1) > 0).float()
        loss = F.cross_entropy(logits.transpose(1, 2), act_target, reduction="none")
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def act(self, state, eval=True):
        self._traj.append((np.asarray(state, dtype=np.float32), self._last_act, self._rtg))
        with torch.no_grad():
            K = len(self._traj)
            ob = np.zeros((1, self.context, self.obs_dim), dtype=np.float32)
            rt = np.zeros((1, self.context, 1), dtype=np.float32)
            ac = np.zeros((1, self.context, self.nA), dtype=np.float32)
            off = self.context - K
            for i, (s, a, g) in enumerate(self._traj):
                j = off + i
                ob[0, j] = s
                rt[0, j, 0] = g
                if a is not None:
                    ac[0, j] = np.eye(self.nA)[a]
            logits = self._forward(torch.as_tensor(ob, device=self.device),
                                   torch.as_tensor(rt, device=self.device),
                                   torch.as_tensor(ac, device=self.device))
            a = int(logits[0, -1].argmax().item())
            self._last_act = a
            return a

    def train(self, env, config, tracker):
        behavior = train_behavior(env, config, self.rng)
        self.episodes = collect_episodes(env, behavior, config.get("dataset_size", 40_000),
                                         config.get("collect_eps", 0.1), self.rng)
        self._rtg = max(float(np.sum(ep[2])) for ep in self.episodes)
        self.t = 0
        losses = []
        while self.t < config["steps"]:
            losses.append(self._update())
            self.t += 1
            if self.t % config.get("log_freq", 500) == 0:
                tracker.log(timestep=self.t, loss=float(np.mean(losses)))
                losses = []
            if self.t % self.eval_freq == 0:
                tracker.log(timestep=self.t, eval_return=float(evaluate(self, env, self.eval_episodes)))
        self.episodes = self.t

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))