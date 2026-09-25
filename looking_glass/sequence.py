"""Sequence modeling stack with explicit Mamba-2 and Samba backends."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .adaptation import QDoRAConfig, QDoRALinear
from .interfaces import SequenceEngineBase

try:
    from mamba_ssm import Mamba2  # type: ignore

    _HAS_MAMBA2 = True
except ImportError:
    Mamba2 = None  # type: ignore
    _HAS_MAMBA2 = False


_SUPPORTED_SEQUENCE_BACKENDS = ("mamba2", "samba")
_DEFAULT_SEQUENCE_BACKEND = "samba"
_MAMBA_IMPLEMENTATION_NAME = "mamba_ssm" if _HAS_MAMBA2 else "portable_ssm"


def get_supported_sequence_backends() -> tuple[str, ...]:
    """Return the supported sequence backend names."""

    return _SUPPORTED_SEQUENCE_BACKENDS


def get_sequence_implementation_name() -> str:
    """Return the active Mamba implementation provider."""

    return _MAMBA_IMPLEMENTATION_NAME


def get_sequence_backend_name(sequence_backend: str = _DEFAULT_SEQUENCE_BACKEND) -> str:
    """Normalize and validate an explicit sequence backend choice."""

    backend_name = sequence_backend.strip().lower()
    if backend_name not in _SUPPORTED_SEQUENCE_BACKENDS:
        supported = ", ".join(_SUPPORTED_SEQUENCE_BACKENDS)
        raise ValueError(f"sequence_backend must be one of: {supported}")
    return backend_name


class PortableSelectiveStateSpaceLayer(nn.Module):
    """Pure PyTorch selective state-space layer used when fused kernels are absent."""

    def __init__(self, hidden_dim: int, state_dim: int, conv_kernel: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.input_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=conv_kernel,
            groups=hidden_dim,
            padding=conv_kernel - 1,
        )
        self.dt_proj = nn.Linear(hidden_dim, hidden_dim)
        self.bc_proj = nn.Linear(hidden_dim, state_dim * 2)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.A_log = nn.Parameter(torch.zeros(hidden_dim, state_dim))
        self.D = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        conv_out = self.input_conv(x.transpose(1, 2)).transpose(1, 2)[:, :seq_len, :]
        hidden = self.activation(conv_out)
        delta = F.softplus(self.dt_proj(hidden)) + 1e-4
        b_t, c_t = self.bc_proj(hidden).chunk(2, dim=-1)
        b_t = torch.tanh(b_t)
        c_t = torch.tanh(c_t)

        transition_base = -torch.exp(self.A_log.float()).to(dtype=hidden.dtype).unsqueeze(0)
        skip = self.D.to(dtype=hidden.dtype)
        state = torch.zeros(
            batch_size,
            self.hidden_dim,
            self.state_dim,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        outputs: list[Tensor] = []
        for position in range(seq_len):
            delta_t = delta[:, position, :].unsqueeze(-1)
            transition = torch.exp(delta_t * transition_base)
            drive = delta_t * b_t[:, position, :].unsqueeze(1)
            state = (state * transition) + hidden[:, position, :].unsqueeze(-1) * drive
            readout = torch.sum(state * c_t[:, position, :].unsqueeze(1), dim=-1)
            gated = (readout + skip * hidden[:, position, :]) * torch.sigmoid(
                self.gate_proj(hidden[:, position, :])
            )
            outputs.append(gated)
        return self.output_proj(torch.stack(outputs, dim=1))


class Mamba2SequenceLayer(nn.Module):
    """Mamba-2 wrapper used by both pure and hybrid sequence backends.

    Input:
            x: Tensor (batch, seq_len, hidden_dim)

    Output:
            Tensor (batch, seq_len, hidden_dim)
    """

    def __init__(self, hidden_dim: int, state_dim: int, conv_kernel: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.conv_kernel = conv_kernel
        self.backend_name = "mamba2"
        self.implementation_name = get_sequence_implementation_name()
        if _HAS_MAMBA2:
            self.mamba = Mamba2(
                d_model=hidden_dim,
                d_state=state_dim,
                d_conv=conv_kernel,
                expand=2,
            )
        else:
            self.mamba = PortableSelectiveStateSpaceLayer(
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                conv_kernel=conv_kernel,
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.mamba(x)


class FlashSelfAttention(nn.Module):
    """Native Flash Attention via scaled_dot_product_attention.

    Causality contract: when no ``attention_mask`` is supplied the attention
    is **causal by default** (``is_causal=True``), matching the library's
    next-event prediction semantics.  When a mask is supplied it is used
    as-is (the temporal core passes an explicit causal + padding mask).
    Callers composing ``EntityCore`` directly therefore get causal attention
    without having to build a mask themselves.

    Input:
            x: Tensor (batch, seq_len, hidden_dim)
            attention_mask: Optional tensor broadcastable to (batch, heads, seq, seq)

    Output:
            Tensor (batch, seq_len, hidden_dim)
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout

        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            # Causal when unmasked: direct EntityCore compositions get
            # next-event semantics without building a mask by hand.
            is_causal=attention_mask is None,
        )
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(attn_out)


class FeedForward(nn.Module):
    """Position-wise feed-forward layer with optional QDoRA adaptation.

    When ``qdora_config`` is provided the two inner linear layers become
    :class:`QDoRALinear` with frozen (optionally quantized) base weights and
    trainable low-rank adapters.  The SSM and attention sublayers stay
    untouched, so per-task adaptation is confined to the FFN path.
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion: float = 4.0,
        dropout: float = 0.1,
        qdora_config: QDoRAConfig | None = None,
    ) -> None:
        super().__init__()
        inner_dim = int(math.ceil(hidden_dim * expansion))
        if qdora_config is not None:
            self.linear_in: nn.Module = QDoRALinear(hidden_dim, inner_dim, config=qdora_config)
            self.linear_out = QDoRALinear(inner_dim, hidden_dim, config=qdora_config)
        else:
            self.linear_in = nn.Linear(hidden_dim, inner_dim)
            self.linear_out = nn.Linear(inner_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_out(self.dropout(self.activation(self.linear_in(x))))

    def upgrade_to_qdora(self, config: QDoRAConfig) -> None:
        """Mutate inner linear layers into QDoRA adapters for downstream LoRA.

        Call this on a **pretrained** ``FeedForward`` once before sharing
        the core across downstream tasks.  The existing ``nn.Linear`` weights
        are copied into the frozen base of each ``QDoRALinear``, preserving
        the pretrained representations.  Fresh low-rank adapters (``lora_a``,
        ``lora_b``, ``magnitude``) are initialised from scratch (kaiming +
        zeros), so every downstream ``SupervisedModel`` that deep-copies
        this retrofitted core starts with the same frozen base and
        independent, randomly-initialised adapter weights.

        This is a one-way, in-place mutation.  It is idempotent (calling it
        again on an already-upgraded ``FeedForward`` is a no-op).
        """

        if isinstance(self.linear_in, QDoRALinear):
            return
        hidden_dim = self.linear_in.in_features
        inner_dim = self.linear_in.out_features
        adapted_in = QDoRALinear(hidden_dim, inner_dim, config=config)
        adapted_out = QDoRALinear(inner_dim, hidden_dim, config=config)
        with torch.no_grad():
            adapted_in.base.weight.copy_(self.linear_in.weight.data)
            if self.linear_in.bias is not None:
                adapted_in.base.bias.copy_(self.linear_in.bias.data)
            adapted_out.base.weight.copy_(self.linear_out.weight.data)
            if self.linear_out.bias is not None:
                adapted_out.base.bias.copy_(self.linear_out.bias.data)
        self.linear_in = adapted_in
        self.linear_out = adapted_out


class MambaBlock(nn.Module):
    """Pure Mamba-2 recurrent block with residual feed-forward mixing.

    Input:
            hidden_states: Tensor (batch, seq_len, hidden_dim)
            attention_mask: Unused, accepted for interface compatibility

    Output:
            Tensor (batch, seq_len, hidden_dim)
    """

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        conv_kernel: int,
        dropout: float = 0.1,
        qdora_config: QDoRAConfig | None = None,
    ) -> None:
        super().__init__()
        self.norm_ssm = nn.LayerNorm(hidden_dim)
        self.norm_ff = nn.LayerNorm(hidden_dim)

        self.ssm = Mamba2SequenceLayer(
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            conv_kernel=conv_kernel,
        )
        self.ff = FeedForward(hidden_dim=hidden_dim, dropout=dropout, qdora_config=qdora_config)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        del attention_mask
        ssm_out = self.ssm(self.norm_ssm(hidden_states))
        x = hidden_states + self.dropout(ssm_out)
        ff_out = self.ff(self.norm_ff(x))
        return x + self.dropout(ff_out)


class SambaBlock(nn.Module):
    """Hybrid Samba block mixing Mamba-2 dynamics with global attention.

    This block mixes Mamba-2 sequence dynamics and Flash Attention under
    a standard residual formulation.

    Input:
            hidden_states: Tensor (batch, seq_len, hidden_dim)
            attention_mask: Optional tensor for attention

    Output:
            Tensor (batch, seq_len, hidden_dim)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        state_dim: int,
        conv_kernel: int,
        dropout: float = 0.1,
        qdora_config: QDoRAConfig | None = None,
    ) -> None:
        super().__init__()
        self.norm_ssm = nn.LayerNorm(hidden_dim)
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.norm_ff = nn.LayerNorm(hidden_dim)

        self.ssm = Mamba2SequenceLayer(
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            conv_kernel=conv_kernel,
        )
        self.attn = FlashSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ff = FeedForward(hidden_dim=hidden_dim, dropout=dropout, qdora_config=qdora_config)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        ssm_out = self.ssm(self.norm_ssm(hidden_states))
        x = hidden_states + self.dropout(ssm_out)

        attn_out = self.attn(self.norm_attn(x), attention_mask=attention_mask)
        x = x + self.dropout(attn_out)

        ff_out = self.ff(self.norm_ff(x))
        return x + self.dropout(ff_out)


class UniversalTransformerBlock(SambaBlock):
    """Backward-compatible alias for the Samba hybrid sequence block."""


class SequenceEngine(SequenceEngineBase):
    """Sequence engine with explicit Mamba-2 or Samba backend selection.

    Input:
            hidden_states: Tensor (batch, seq_len, hidden_dim)
            attention_mask: Optional tensor for attention

    Output:
            Tensor (batch, seq_len, hidden_dim)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        state_dim: int = 64,
        conv_kernel: int = 4,
        recurrent_steps: int = 4,
        dropout: float = 0.1,
        backend: str = _DEFAULT_SEQUENCE_BACKEND,
        qdora_config: QDoRAConfig | None = None,
    ) -> None:
        super().__init__()
        if recurrent_steps < 1:
            raise ValueError("recurrent_steps must be >= 1")

        self.backend_name = get_sequence_backend_name(backend)
        self.recurrent_steps = recurrent_steps
        if self.backend_name == "mamba2":
            self.shared_block: nn.Module = MambaBlock(
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                conv_kernel=conv_kernel,
                dropout=dropout,
                qdora_config=qdora_config,
            )
        else:
            self.shared_block = SambaBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                state_dim=state_dim,
                conv_kernel=conv_kernel,
                dropout=dropout,
                qdora_config=qdora_config,
            )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        x = hidden_states
        for _ in range(self.recurrent_steps):
            x = self.shared_block(x, attention_mask=attention_mask)
        return self.final_norm(x)
