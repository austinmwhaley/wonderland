"""Standalone temporal encoding stack for event streams."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .interfaces import TemporalEncoderBase


def _as_time_column(delta_t: Tensor) -> Tensor:
    """Ensure delta_t shape is (batch, seq_len, 1)."""
    if delta_t.dim() == 2:
        return delta_t.unsqueeze(-1)
    if delta_t.dim() == 3 and delta_t.size(-1) == 1:
        return delta_t
    raise ValueError("delta_t must have shape (batch, seq_len) or (batch, seq_len, 1)")


def log_scaled_delta_t(delta_t: Tensor, eps: float = 1e-8) -> Tensor:
    """Functional log scaling for positive time deltas.

    Input:
            delta_t: Tensor (batch, seq_len, 1)

    Output:
            Tensor (batch, seq_len, 1)
    """
    safe_delta = torch.clamp(delta_t, min=0.0) + eps
    return torch.log1p(safe_delta)


def cumulative_time(log_delta_t: Tensor) -> Tensor:
    """Convert log-scaled deltas into monotonic event time.

    Input:
            log_delta_t: Tensor (batch, seq_len, 1)

    Output:
            Tensor (batch, seq_len, 1)
    """
    return torch.cumsum(log_delta_t, dim=1)


class Time2Vec(nn.Module):
    """Time2Vec embedding.

    Input:
            t: Tensor (batch, seq_len, 1)

    Output:
            Tensor (batch, seq_len, out_dim)
    """

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        if out_dim < 2:
            raise ValueError("Time2Vec out_dim must be >= 2")

        self.linear_weight = nn.Parameter(torch.randn(1))
        self.linear_bias = nn.Parameter(torch.zeros(1))
        self.periodic_weight = nn.Parameter(torch.randn(out_dim - 1))
        self.periodic_bias = nn.Parameter(torch.zeros(out_dim - 1))

    def forward(self, t: Tensor) -> Tensor:
        linear = self.linear_weight * t + self.linear_bias
        periodic = torch.sin(t * self.periodic_weight + self.periodic_bias)
        return torch.cat([linear, periodic], dim=-1)


class TAPE(nn.Module):
    """Time-Aware Positional Encoding with time-indexed sinusoidal bases.

    Input:
            t: Tensor (batch, seq_len, 1)

    Output:
            Tensor (batch, seq_len, hidden_dim)
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even for sinusoidal TAPE")

        inv_freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            / float(hidden_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, t: Tensor) -> Tensor:
        phase = t * self.inv_freq
        sin_part = torch.sin(phase)
        cos_part = torch.cos(phase)
        return torch.cat([sin_part, cos_part], dim=-1)


class ODEDriftField(nn.Module):
    """Neural ODE drift field f(h, t)."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: Tensor, t: Tensor) -> Tensor:
        return self.net(torch.cat([h, t], dim=-1))


class NeuralODELatentDrift(nn.Module):
    """Single-step Euler integrator for latent drift.

    Input:
            hidden_states: Tensor (batch, seq_len, hidden_dim)
            log_dt: Tensor (batch, seq_len, 1)

    Output:
            Tensor (batch, seq_len, hidden_dim)
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.field = ODEDriftField(hidden_dim=hidden_dim)

    def forward(self, hidden_states: Tensor, log_dt: Tensor) -> Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        outputs: list[Tensor] = []
        h_prev = torch.zeros(batch_size, hidden_dim, device=hidden_states.device)

        for idx in range(seq_len):
            h_t = hidden_states[:, idx, :]
            dt_t = log_dt[:, idx, :]
            drift = self.field(h_prev + h_t, dt_t)
            h_next = h_prev + dt_t * drift
            outputs.append(h_next.unsqueeze(1))
            h_prev = h_next

        return torch.cat(outputs, dim=1)


class TemporalStack(TemporalEncoderBase):
    """Standalone temporal stack for event-time enrichment.

    This module combines:
    - Log-scaled delta time
    - TAPE positional signal
    - Time2Vec projection
    - Neural ODE latent drift

    Input:
            hidden_states: Tensor (batch, seq_len, hidden_dim)
            delta_t: Tensor (batch, seq_len, 1) or (batch, seq_len)

    Output:
            Tensor (batch, seq_len, hidden_dim)
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time2vec = Time2Vec(out_dim=hidden_dim)
        self.tape = TAPE(hidden_dim=hidden_dim)
        self.ode_drift = NeuralODELatentDrift(hidden_dim=hidden_dim)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states: Tensor, delta_t: Tensor) -> Tensor:
        dt = _as_time_column(delta_t)
        log_dt = log_scaled_delta_t(dt)
        t = cumulative_time(log_dt)

        t2v = self.time2vec(t)
        tape = self.tape(t)
        ode = self.ode_drift(hidden_states, log_dt)

        fused = self.fuse(torch.cat([t2v, tape, ode], dim=-1))
        return self.norm(hidden_states + fused)
