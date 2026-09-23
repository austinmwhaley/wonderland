import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from ..base import BaseAgent
from .networks import Critic, DiscretePolicy


class KFACOptimizer:
    """Kronecker-factored approximate curvature (K-FAC) optimizer.

    For each Linear layer the Fisher is approximated by the Kronecker product
    G x A of the output-gradient covariance and the input covariance, with the
    statistics accumulated as an EMA so small batches do not make the
    curvature estimate too noisy.  Natural gradients then cost no Kronecker
    solve: (G x A)^-1 vec(grad) = vec(G^-1 grad A^-1)."""

    def __init__(self, model, lr, damping=1e-2, ema=0.05):
        self.lr = lr
        self.damping = damping
        self.ema = ema
        self.layers = []
        for mod in model.modules():
            if isinstance(mod, nn.Linear):
                mod.register_full_backward_hook(self._capture_grad)
                self.layers.append(mod)
        self.acts = {id(m): None for m in self.layers}
        self.grads = {id(m): None for m in self.layers}
        self.a_cov = {}
        self.g_cov = {}
        for mod in self.layers:
            mod.register_forward_hook(self._capture_act)

    def _capture_act(self, mod, inp, out):
        self.acts[id(mod)] = inp[0].detach()

    def _capture_grad(self, mod, grad_in, grad_out):
        self.grads[id(mod)] = grad_out[0].detach()

    def _factor(self, key, x):
        n = x.size(0)
        cov = (x.t() @ x) / n
        scale = cov.diagonal().max() + 1e-8
        cov = cov + (self.damping * scale + 1e-6) * torch.eye(cov.size(0), device=cov.device)
        if key not in self.a_cov:
            self.a_cov[key] = cov
        else:
            self.a_cov[key] = (1 - self.ema) * self.a_cov[key] + self.ema * cov
        return torch.linalg.cholesky(self.a_cov[key])

    def _inv_solve(self, b, chol):
        return torch.cholesky_solve(b, chol)

    def step(self, losses, max_update_norm=1e-2):
        for loss in losses:
            loss.backward(retain_graph=True)
        for mod in self.layers:
            a = self.acts[id(mod)]
            g = self.grads[id(mod)]
            if a is None or g is None or mod.weight.grad is None:
                continue
            n = a.size(0)
            A = self._factor(("A", id(mod)), a.reshape(n, -1))
            G = self._factor(("G", id(mod)), g.reshape(n, -1))
            gw = mod.weight.grad
            nat = self._inv_solve(self._inv_solve(gw.t(), A).t(), G)
            update = -self.lr * nat
            update = update * (max_update_norm / (update.norm() + 1e-12)).clamp(max=1.0)
            mod.weight.data.add_(update)
            if mod.bias is not None and mod.bias.grad is not None:
                gb = mod.bias.grad.unsqueeze(1)
                nb = self._inv_solve(gb, G).squeeze(1)
                ub = -self.lr * nb
                ub = ub * (max_update_norm / (ub.norm() + 1e-12)).clamp(max=1.0)
                mod.bias.data.add_(ub)
            mod.weight.grad = None
            if mod.bias is not None:
                mod.bias.grad = None
        self.acts = {id(m): None for m in self.layers}
        self.grads = {id(m): None for m in self.layers}


class ACKTR(BaseAgent):
    """Actor-critic with Kronecker-factored trust region: A2C's loss but the
    gradient step is the natural gradient approximated by K-FAC, giving
    curvature-aware updates that make the trust region implicit
    (Wu et al. 2017)."""

    family = "actor-critic"
    policy = "on-policy"
    action_space = "discrete"
    state_space = "continuous"

    def __init__(self, env, config):
        super().__init__(env, config)
        self.device = config.get("device", "cpu")
        self.gamma = config.get("gamma", 0.99)
        self.n_envs = config.get("n_envs", 8)
        self.n_steps = config.get("n_steps", 5)
        in_dim = int(np.prod(env.observation_space.shape))
        hidden = config.get("hidden", 128)
        self.policy = DiscretePolicy(in_dim, hidden, int(env.action_space.n)).to(self.device)
        self.critic = Critic(in_dim, hidden).to(self.device)
        lr = config.get("lr", 1e-2)
        self.kfac = KFACOptimizer(
            nn.ModuleList([self.policy, self.critic]),
            lr=lr, damping=config.get("damping", 1e-2))
        self.entropy_coef = config.get("entropy_coef", 0.01)
        try:
            gid = env.unwrapped.spec.id
            self.envs = gym.vector.SyncVectorEnv(
                [lambda: gym.make(gid) for _ in range(self.n_envs)])
        except AttributeError:
            self.envs = gym.vector.SyncVectorEnv([lambda: env] * self.n_envs)

    def _batch(self, obs):
        return torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)

    def act(self, state, eval=False):
        with torch.no_grad():
            probs = self.policy(self._batch([state]))
            if eval:
                return int(probs.argmax().item())
            return int(torch.distributions.Categorical(probs).sample().item())

    def train(self, env, config, tracker):
        obs, _ = self.envs.reset()
        ep = 0
        t_global = 0
        ep_returns = np.zeros(self.n_envs)
        while t_global < config["steps"]:
            obs_buf, act_buf, lp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
            for _ in range(self.n_steps):
                x = self._batch(obs)
                with torch.no_grad():
                    v = self.critic(x)
                    probs = self.policy(x)
                    dist = torch.distributions.Categorical(probs)
                    a = dist.sample()
                    lp = dist.log_prob(a)
                obs_buf.append(x)
                act_buf.append(a)
                lp_buf.append(lp)
                val_buf.append(v)
                nobs, r, term, trunc, _ = self.envs.step(a.cpu().numpy())
                done = np.logical_or(term, trunc)
                rew_buf.append(r)
                done_buf.append(term)
                ep_returns += r
                if done.any():
                    for i in np.where(done)[0]:
                        ep += 1
                        tracker.log(timestep=t_global, episode=ep, ret=float(ep_returns[i]))
                    ep_returns[done] = 0.0
                obs = nobs
                t_global += self.n_envs
            with torch.no_grad():
                v_last = self.critic(self._batch(obs))
            returns = []
            adv = []
            G = v_last.cpu().numpy().squeeze(1)
            for t in reversed(range(self.n_steps)):
                G = rew_buf[t] + self.gamma * G * (1 - done_buf[t])
                returns.insert(0, G.copy())
                adv.insert(0, G - val_buf[t].cpu().numpy().squeeze())
            obs_t = torch.cat(obs_buf)
            act_t = torch.cat(act_buf)
            lp_t = torch.cat(lp_buf)
            ret_t = torch.as_tensor(np.concatenate(returns), dtype=torch.float32,
                                    device=self.device).unsqueeze(1)
            adv_t = torch.as_tensor(np.concatenate(adv), dtype=torch.float32,
                                    device=self.device).unsqueeze(1)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            probs_new = self.policy(obs_t)
            dist_new = torch.distributions.Categorical(probs_new)
            lp_new = dist_new.log_prob(act_t)
            policy_loss = -(lp_new.unsqueeze(-1) * adv_t).mean()
            policy_loss = policy_loss - self.entropy_coef * dist_new.entropy().mean()
            value_loss = F.mse_loss(self.critic(obs_t), ret_t)
            self.kfac.step([policy_loss, value_loss])
            total = float((policy_loss + value_loss).item())
            tracker.log(timestep=t_global, episode=ep, loss=total)
        self.episodes = ep

    def save(self, path):
        torch.save({"policy": self.policy.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(data["policy"])
        self.critic.load_state_dict(data["critic"])