"""Runs CPU float32 KDA for Haas's official prediction network, matching flash-linear-attention 0.5.2.

The recurrence is fla's own ops/kda/naive.py, loaded without importing its Triton-dependent packages.
Only the inference configuration used by the GGUF converter is supported.

The parameter layout and inference operations are adapted from fla/layers/kda.py and its modules;
parameter values come from the checkpoint. The adapted fla portions carry this license:
Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import importlib.metadata
import importlib.util

import torch
from torch import nn
from torch.nn import functional as F

# fla/ops/kda/fused_recurrent.py normalizes by sqrt(sum(x*x) + 1e-6), not F.normalize's clamped norm.
QK_NORM_EPS = 1e-6


def _load_recurrence():
    """Load the pinned distribution's pure torch reference without executing fla/__init__.py."""
    distribution = importlib.metadata.distribution("flash-linear-attention")
    if distribution.version != "0.5.2":
        raise ImportError(f"flash-linear-attention==0.5.2 is required, found {distribution.version}")
    path = distribution.locate_file("fla/ops/kda/naive.py")
    spec = importlib.util.spec_from_file_location("_fla_kda_naive_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.naive_recurrent_kda


naive_recurrent_kda = _load_recurrence()


class ShortConvolution(nn.Conv1d):
    """Runs fla/modules/convolution.py's ShortConvolution (defined in conv/short_conv.py) in torch."""

    def __init__(self, hidden_size, kernel_size, bias=False):
        super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size,
                         bias=bias, padding=kernel_size - 1)
        self.hidden_size = hidden_size
        self.activation = "silu"

    def forward(self, x, cache=None, output_final_state=False):
        width = self.kernel_size[0]
        if cache is None:
            cache = x.new_zeros(x.shape[0], self.hidden_size, width)
        outputs = []
        # ShortConvolution.step() keeps W projected inputs, oldest first, and updates the supplied cache in place.
        for step in x.unbind(1):
            cache.copy_(cache.roll(shifts=-1, dims=-1))
            cache[:, :, -1] = step
            y = (cache * self.weight[:, 0]).sum(-1)
            if self.bias is not None:
                y = y + self.bias
            outputs.append(F.silu(y))
        return torch.stack(outputs, dim=1), cache if output_final_state else None


class FusedRMSNormGated(nn.Module):
    """Applies fla/modules/fused_norm_gate.py's RMS normalization, affine weight, then sigmoid gate."""

    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.activation = "sigmoid"

    def forward(self, x, gate):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float() * gate.float().sigmoid()).to(x.dtype)


class KimiDeltaAttention(nn.Module):
    """Preserves fla/layers/kda.py's parameter layout and forward/cache contract for Haas CPU inference."""

    def __init__(self, hidden_size=2048, expand_v=1, head_dim=128, num_heads=16, num_v_heads=None,
                 mode="chunk", use_short_conv=True, allow_neg_eigval=False, safe_gate=False,
                 lower_bound=None, conv_size=4, conv_bias=False, layer_idx=None, norm_eps=1e-5):
        super().__init__()
        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.head_dim = self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.num_heads = num_heads
        self.num_v_heads = num_heads if num_v_heads is None else num_v_heads
        self.key_dim = num_heads * head_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.gate_dim = self.num_v_heads * head_dim
        self.use_short_conv, self.conv_size, self.conv_bias = use_short_conv, conv_size, conv_bias
        self.allow_neg_eigval, self.safe_gate, self.lower_bound = allow_neg_eigval, safe_gate, lower_bound
        self.layer_idx = layer_idx
        assert mode in ("chunk", "fused_recurrent")
        assert expand_v == 1 and self.num_v_heads == num_heads
        assert use_short_conv and not conv_bias and not allow_neg_eigval and not safe_gate and lower_bound is None

        # Parameter names and dimensions match fla/layers/kda.py; the checkpoint supplies every value.
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.q_conv1d = ShortConvolution(self.key_dim, conv_size, conv_bias)
        self.k_conv1d = ShortConvolution(self.key_dim, conv_size, conv_bias)
        self.v_conv1d = ShortConvolution(self.value_dim, conv_size, conv_bias)
        self.f_proj = nn.Sequential(nn.Linear(hidden_size, self.head_v_dim, bias=False),
                                    nn.Linear(self.head_v_dim, self.gate_dim, bias=False))
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads))
        self.dt_bias = nn.Parameter(torch.empty(self.gate_dim))
        self.g_proj = nn.Sequential(nn.Linear(hidden_size, self.head_v_dim, bias=False),
                                    nn.Linear(self.head_v_dim, self.value_dim, bias=True))
        self.o_norm = FusedRMSNormGated(self.head_v_dim, norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, use_cache=False,
                output_attentions=False, **kwargs):
        if self.training or hidden_states.device.type != "cpu" or hidden_states.dtype != torch.float32:
            raise ValueError("the KDA reference requires CPU float32 evaluation")
        if attention_mask is not None or kwargs:
            raise ValueError("the KDA reference takes dense sequences without attention masks or extra options")
        batch, length, _ = hidden_states.shape
        # fla/layers/utils.py's get_layer_cache() and update_layer_cache() use the layer's index.
        last = past_key_values[self.layer_idx] if past_key_values is not None and len(past_key_values) > self.layer_idx else None
        conv_states = last["conv_state"] if last is not None else (None, None, None)
        projected, new_conv = [], []
        for projection, convolution, cache in zip((self.q_proj, self.k_proj, self.v_proj),
                                                 (self.q_conv1d, self.k_conv1d, self.v_conv1d), conv_states):
            value, cache = convolution(projection(hidden_states), cache, use_cache)
            projected.append(value.reshape(batch, length, self.num_heads, self.head_dim))
            new_conv.append(cache)
        q, k, v = projected
        # fla/ops/kda/fused_recurrent.py applies q/k normalization before scaling q inside the recurrence.
        q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + QK_NORM_EPS)
        k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + QK_NORM_EPS)
        # fla/ops/kda/gate.py's naive_kda_gate() and fused_recurrent.py apply this gate and sigmoid beta.
        g = self.f_proj(hidden_states).reshape(batch, length, self.num_heads, self.head_dim)
        g = -self.A_log.float().exp()[:, None] * F.softplus(g.float() + self.dt_bias.float().view(self.num_heads, self.head_dim))
        beta = self.b_proj(hidden_states).float().sigmoid()
        # fla/layers/kda.py requests state_v_first=True ([B,H,V,K]); naive.py takes [B,H,K,V].
        state = last["recurrent_state"] if last is not None else None
        o, state = naive_recurrent_kda(q, k, v, g, beta,
                                      initial_state=None if state is None else state.transpose(-1, -2),
                                      output_final_state=use_cache)
        if past_key_values is not None:
            past_key_values.update(layer_idx=self.layer_idx, offset=length,
                                   recurrent_state=None if state is None else state.transpose(-1, -2).contiguous(),
                                   conv_state=tuple(new_conv))
        gate = self.g_proj(hidden_states).reshape(batch, length, self.num_heads, self.head_dim)
        o = self.o_norm(o, gate).reshape(batch, length, self.value_dim)
        return self.o_proj(o), None, past_key_values
