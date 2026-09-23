import numpy as np
import torch
import torch.nn.functional as F

from ..base import BaseAgent
from .networks import (
    Critic, DiscretePolicy, GaussianPolicy, kl_discrete, kl_gaussian,
    flat_params, set_params,
)


class TRPO(BaseAgent):
    family = "policy-gradient"
    policy = "on-policy"
    action_space = "both"
    state_space = "both"

    def __init__(self, env, config):
        super().__init__(env, config)
        self.device = config.get("device", "cpu")
        self.gamma = config.get("gamma", 0.99)
        self.lam = config.get("lambda", 0.95)
        self.delta = config.get("delta", 0.01)
        self.rollout = config.get("rollout", 2048)
        self.cg_iters = config.get("cg_iters", 10)
        self.max_backtracks = config.get("max_backtracks", 10)
        self.discrete = hasattr(env.action_space, "n")
        in_dim = int(np.prod(env.observation_space.shape))
        hidden = config.get("hidden", 128)
        if self.discrete:
            self.policy = DiscretePolicy(in_dim, hidden, int(env.action_space.n)).to(self.device)
        else:
            self.policy = GaussianPolicy(in_dim, hidden, int(env.action_space.shape[0])).to(self.device)
        self.critic = Critic(in_dim, hidden).to(self.device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.get("critic_lr", 1e-3))

    def _batch(self, obs):
        return torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)

    def act(self, state, eval=False):
        with torch.no_grad():
            x = self._batch([state])
            if self.discrete:
                probs = self.policy(x)
                if eval:
                    return int(probs.argmax().item())
                return int(torch.distributions.Categorical(probs).sample().item())
            if eval:
                return self.policy.mean_action(x).cpu().numpy()[0]
            a, _ = self.policy.sample(x)
            return a.cpu().numpy()[0]

    def _rollout(self, env, tracker):
        state, _ = env.reset()
        obs_buf, act_buf, lp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        ep = self.episodes
        ep_ret = self.ep_ret
        t_global = self.t_global
        for _ in range(self.rollout):
            x = self._batch([state])
            with torch.no_grad():
                v = self.critic(x)
                if self.discrete:
                    probs = self.policy(x)
                    dist = torch.distributions.Categorical(probs)
                    a = dist.sample()
                    lp = dist.log_prob(a)
                    action = int(a.item())
                else:
                    a, lp = self.policy.sample(x)
                    action = a.cpu().numpy()[0]
            obs_buf.append(x)
            act_buf.append(a)
            lp_buf.append(lp)
            val_buf.append(v)
            ns, r, term, trunc, _ = env.step(action)
            done = bool(term or trunc)
            rew_buf.append(r)
            done_buf.append(bool(term))
            ep_ret += r
            state = ns
            t_global += 1
            if done:
                ep += 1
                tracker.log(timestep=t_global, episode=ep, ret=float(ep_ret))
                ep_ret = 0.0
                state, _ = env.reset()
        with torch.no_grad():
            v_last = self.critic(self._batch([state]))
        returns, advantages = [], []
        G = v_last.item()
        adv = 0.0
        for t in reversed(range(self.rollout)):
            done = done_buf[t]
            delta = rew_buf[t] + self.gamma * (0.0 if done else G) - val_buf[t].item()
            adv = delta + self.gamma * self.lam * (0.0 if done else adv)
            G = rew_buf[t] + self.gamma * (0.0 if done else G)
            advantages.insert(0, adv)
            returns.insert(0, G)
        self.episodes, self.ep_ret, self.t_global = ep, ep_ret, t_global
        return (torch.cat(obs_buf), torch.cat(act_buf), torch.cat(lp_buf).detach(),
                torch.as_tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(1),
                torch.as_tensor(advantages, dtype=torch.float32, device=self.device).unsqueeze(1))

    def _surrogate(self, obs_t, act_t, lp_old, adv_t):
        if self.discrete:
            probs = self.policy(obs_t)
            lp_new = torch.distributions.Categorical(probs).log_prob(act_t)
        else:
            lp_new = self.policy.log_prob(obs_t, act_t)
        ratio = (lp_new - lp_old).exp().unsqueeze(-1)
        return (ratio * adv_t).mean(), ratio

    def _kl(self, obs_t):
        if self.discrete:
            logits = self.policy.net(obs_t)
            return kl_discrete(self.old_probs, logits)
        mean_new, std_new = self.policy(obs_t)
        return kl_gaussian(self.old_mean, self.old_std, mean_new, std_new)

    def train(self, env, config, tracker):
        self.episodes = 0
        self.ep_ret = 0.0
        self.t_global = 0
        while self.t_global < config["steps"]:
            obs_t, act_t, lp_old, ret_t, adv_t = self._rollout(env, tracker)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            with torch.no_grad():
                if self.discrete:
                    self.old_probs = self.policy(obs_t)
                else:
                    self.old_mean, self.old_std = self.policy(obs_t)
            params = list(self.policy.parameters())
            loss, _ = self._surrogate(obs_t, act_t, lp_old, adv_t)
            g = torch.autograd.grad(loss, params)
            g = torch.cat([x.contiguous().view(-1) for x in g]).detach()
            kl = self._kl(obs_t)
            g_kl = torch.autograd.grad(kl, params, create_graph=True)
            g_kl = torch.cat([x.contiguous().view(-1) for x in g_kl])

            def hvp(v):
                gv = torch.autograd.grad(g_kl @ v, params, retain_graph=True)
                return torch.cat([x.contiguous().view(-1) for x in gv]) + 0.1 * v

            step_dir = self._conjugate_gradient(hvp, g, self.cg_iters)
            shs = 0.5 * (step_dir @ hvp(step_dir))
            beta = torch.sqrt(shs / (self.delta + 1e-8)).item() + 1e-8
            step_dir = step_dir / beta
            old_params = flat_params(self.policy)
            loss_old = loss.item()
            for i in range(self.max_backtracks):
                alpha = 0.5 ** i
                set_params(self.policy, old_params + alpha * step_dir)
                kl_new = self._kl(obs_t).item()
                loss_new, _ = self._surrogate(obs_t, act_t, lp_old, adv_t)
                loss_new = loss_new.item()
                if kl_new <= self.delta and loss_new > loss_old:
                    break
            if i == self.max_backtracks - 1:
                set_params(self.policy, old_params)
            for _ in range(config.get("vf_iters", 80)):
                v = self.critic(obs_t)
                self.critic_opt.zero_grad()
                F.mse_loss(v, ret_t).backward()
                self.critic_opt.step()
            tracker.log(timestep=self.t_global, episode=self.episodes,
                        loss=float(max(loss_new - loss_old, 0.0)))
        self.episodes = self.episodes

    @staticmethod
    def _conjugate_gradient(hvp, b, iters, tol=1e-10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        r_dot = r @ r
        for _ in range(iters):
            z = hvp(p)
            alpha = r_dot / (p @ z + 1e-8)
            x = x + alpha * p
            r = r - alpha * z
            new_dot = r @ r
            if new_dot < tol:
                break
            p = r + (new_dot / r_dot) * p
            r_dot = new_dot
        return x

    def save(self, path):
        torch.save({"policy": self.policy.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(data["policy"])
        self.critic.load_state_dict(data["critic"])