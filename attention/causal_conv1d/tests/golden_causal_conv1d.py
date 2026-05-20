"""
PyTorch reference (golden) implementation of CausalConv1d for correctness testing.

Registered as builtin golden plugin via ttk/user_defined_modules/op/golden_funcs/registry.py.

Causal 1D depthwise convolution (per-channel independent):
    y[t, d] = Act( sum_{j=0}^{W-1} weight[j, d] * x[t - j, d] + bias[d] )

Act in {Identity, SiLU}, W = weight.shape[0] in {2, 3, 4}.
History x[-1], x[-2], ... come from conv_states[-(W-1):].

Operator inputs (in order):
  x, weight, bias, conv_states, query_start_loc, cache_indices,
  initial_state_mode, num_accepted_tokens
Attr:
  activation_mode, pad_slot_id, run_mode
Outputs:
  conv_states, y
"""

import numpy as np
import torch
import torch.nn.functional as F


def _to_torch(arr):
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor):
        return arr
    try:
        return torch.from_numpy(arr)
    except TypeError:
        return torch.from_numpy(arr.view(np.uint16)).view(torch.bfloat16)


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _split_sequences(x: torch.Tensor, query_start_loc: torch.Tensor | None
                     ) -> tuple[list[torch.Tensor], int]:
    """Split 2D varlen or 3D batch input into per-sequence tensors."""
    if x.ndim == 3:
        B = x.shape[0]
        return [x[i] for i in range(B)], B
    qsl = query_start_loc.to(torch.int64)
    B = qsl.shape[0] - 1
    return [x[qsl[i] : qsl[i + 1]] for i in range(B)], B


def _compute_conv(x_seq: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
                  conv_state: torch.Tensor, activation: bool, has_init: bool) -> torch.Tensor:
    """Causal conv1d for a single sequence using F.conv1d(groups=dim)."""
    W = weight.shape[0]
    D = x_seq.shape[-1]
    if has_init:
        history = conv_state[:W - 1]
    else:
        history = torch.zeros(W - 1, D, dtype=x_seq.dtype, device=x_seq.device)
    # padded_input: [state_len + seq_len, D] -> [1, D, state_len + seq_len]
    padded = torch.cat([history, x_seq], dim=0).T.unsqueeze(0)
    # weight: [W, D] -> [D, 1, W]
    w = weight.T.unsqueeze(1)

    y = F.conv1d(padded, w, bias=None, stride=1, padding=0, groups=D)  # [1, D, seq_len]
    y = y.squeeze(0).T  # [seq_len, D]
    if not has_init:
        y[:W - 1] = 0

    if bias is not None:
        y += bias
    if activation:
        y = _silu(y)
    return y


# ============================================================
# Per-mode golden functions
# ============================================================

def causal_conv1d_fwd(
    x: torch.Tensor,              # [T, D] varlen or [B, S, D] batch
    weight: torch.Tensor,         # [W, D]
    bias: torch.Tensor | None,    # [D] or None
    conv_states: torch.Tensor,    # [num_cache_lines, state_len, D]
    query_start_loc: torch.Tensor | None = None,  # [batch+1] int64
    cache_indices: torch.Tensor | None = None,    # [batch] int64
    initial_state_mode: torch.Tensor | None = None,  # [batch] int64, 1=use conv_states
    activation_mode: int = 0,      # 0=Identity, 1=SiLU
    pad_slot_id: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward mode: full sequence, conv_states read-only."""
    activation = (activation_mode == 1)
    device, dtype = x.device, x.dtype
    xs, B = _split_sequences(x, query_start_loc)
    cache_idx = cache_indices.tolist() if cache_indices is not None else list(range(B))
    ism = initial_state_mode.tolist() if initial_state_mode is not None else [0] * B
    is_3d = (x.ndim == 3)

    ys = []
    for i in range(B):
        ci = int(cache_idx[i])
        if ci == pad_slot_id:
            seq_len = x.shape[1] if is_3d else xs[i].shape[0]
            ys.append(torch.zeros(seq_len, x.shape[-1], dtype=dtype, device=device))
            continue
        has_init = bool(ism[i])
        ys.append(_compute_conv(xs[i], weight, bias, conv_states[ci], activation, has_init))

    y = torch.stack(ys, dim=0) if is_3d else torch.cat(ys, dim=0)
    return conv_states, y


def causal_conv1d_update(
    x: torch.Tensor,              # [T, D] varlen or [B, S, D] batch
    weight: torch.Tensor,         # [W, D]
    bias: torch.Tensor | None,    # [D] or None
    conv_states: torch.Tensor,    # [num_cache_lines, state_len, D]
    query_start_loc: torch.Tensor | None = None,  # [batch+1] int64
    cache_indices: torch.Tensor | None = None,    # [batch] int64
    num_accepted_tokens: torch.Tensor | None = None,  # [batch] int64, speculative decode
    activation_mode: int = 0,      # 0=Identity, 1=SiLU
    pad_slot_id: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update mode: incremental step, writes back to conv_states."""
    activation = (activation_mode == 1)
    W = weight.shape[0]
    device, dtype = x.device, x.dtype
    xs, B = _split_sequences(x, query_start_loc)
    cache_idx = cache_indices.tolist() if cache_indices is not None else list(range(B))
    nat = num_accepted_tokens.tolist() if num_accepted_tokens is not None else None
    is_3d = (x.ndim == 3)
    state_len = conv_states.shape[1]

    cs_out = conv_states.clone()

    ys = []
    for i in range(B):
        ci = int(cache_idx[i])
        if ci == pad_slot_id:
            seq_len = x.shape[1] if is_3d else xs[i].shape[0]
            ys.append(torch.zeros(seq_len, x.shape[-1], dtype=dtype, device=device))
            continue

        x_seq = xs[i]
        S = x_seq.shape[0]
        offset = int(nat[i]) - 1 if nat is not None else 0  # speculative offset
        history = conv_states[ci][offset : offset + W - 1]
        padded = torch.cat([history, x_seq], dim=0)  # [W-1 + S, D]

        D_dim = x_seq.shape[-1]
        padded_ncl = padded.T.unsqueeze(0)  # [1, D, W-1+S]
        w = weight.T.unsqueeze(1)  # [D, 1, W], flipped for causal
        y_i = F.conv1d(padded_ncl, w, stride=1, padding=0, groups=D_dim).squeeze(0).T

        if bias is not None:
            y_i += bias
        if activation:
            y_i = _silu(y_i)

        # Write back to cloned output
        cs_out[ci].copy_(padded[-state_len:])
        ys.append(y_i)

    y = torch.stack(ys, dim=0) if is_3d else torch.cat(ys, dim=0)
    return cs_out, y


# ============================================================
# Ttk registry entry - matches operator input order + attrs
# ============================================================

def CausalConv1d(
    x,                  # [T, D] or [B, S, D]
    weight,             # [W, D]
    bias,               # [D] or None
    conv_states,        # [num_cache_lines, state_len, D]
    query_start_loc=None,            # [batch+1] int64
    cache_indices=None,              # [batch] int64
    initial_state_mode=None,         # [batch] int64
    num_accepted_tokens=None,        # [batch] int64
    *,
    activation_mode=0,   # attr
    pad_slot_id=-1,      # attr
    run_mode=0,          # attr
    **extra,
):
    """
    Golden function for CausalConv1d operator.
    Called by ttk test framework with numpy arrays, returns tuple of numpy arrays.

    Matches operator input order:
      x, weight, bias, conv_states, query_start_loc, cache_indices,
      initial_state_mode, num_accepted_tokens
    Attrs:
      activation_mode, pad_slot_id, run_mode
    """
    x_t = _to_torch(x)
    w_t = _to_torch(weight)
    b_t = _to_torch(bias)
    cs_t = _to_torch(conv_states)
    qsl_t = _to_torch(query_start_loc)
    ci_t = _to_torch(cache_indices)
    ism_t = _to_torch(initial_state_mode)

    if run_mode == 0:
        cs_out, y_t = causal_conv1d_fwd(x_t, w_t, b_t, cs_t,
                                        query_start_loc=qsl_t, cache_indices=ci_t,
                                        initial_state_mode=ism_t,
                                        activation_mode=activation_mode, pad_slot_id=pad_slot_id)
    else:
        nat_t = _to_torch(num_accepted_tokens)
        cs_out, y_t = causal_conv1d_update(x_t, w_t, b_t, cs_t,
                                           query_start_loc=qsl_t, cache_indices=ci_t,
                                           num_accepted_tokens=nat_t,
                                           activation_mode=activation_mode, pad_slot_id=pad_slot_id)
    return cs_out.detach().cpu().numpy(), y_t.detach().cpu().numpy()


# -----------------------------------------------------------
# Quick smoke test
# -----------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    B, S, D, W = 2, 8, 128, 4
    state_len_fwd = W - 1  # fwd: just need enough history
    state_len_upd = W - 1 + (1 - 1)  # update with S=1 (non-speculative)
    x = torch.randn(B, S, D, dtype=torch.float16)
    weight = torch.randn(W, D, dtype=torch.float16)
    bias = torch.randn(D, dtype=torch.float16)
    cs_fwd = torch.zeros(B, state_len_fwd, D, dtype=torch.float16)
    cs_upd = torch.zeros(B, state_len_upd, D, dtype=torch.float16)

    cs_out, y = causal_conv1d_fwd(x, weight, bias, cs_fwd, activation_mode=1)
    print(f"fwd:    y shape={y.shape}  cs_out shape={cs_out.shape}")

    x_upd = x[:, -1:, :]
    cs_out, y = causal_conv1d_update(x_upd, weight, bias, cs_upd, activation_mode=1)
    print(f"update: y shape={y.shape}  cs_out shape={cs_out.shape}")

    qsl = torch.tensor([0, 3, 8], dtype=torch.int64)
    x_v = torch.cat([x[0, :3], x[1, :5]], dim=0)
    cs_out, y = causal_conv1d_fwd(x_v, weight, bias, cs_fwd, query_start_loc=qsl, activation_mode=1)
    print(f"varlen: y shape={y.shape}  cs_out shape={cs_out.shape}")

    # Speculative decode: m=2 (seqlen=3), state_len = W-1 + m = 5
    m = 2
    cs_spec = torch.zeros(B, W - 1 + m, D, dtype=torch.float16)
    x_spec = torch.randn(B, m + 1, D, dtype=torch.float16)
    nat = torch.tensor([1, 2], dtype=torch.int64)  # batch0 accepts 1, batch1 accepts 2
    cs_out, y_spec = causal_conv1d_update(x_spec, weight, bias, cs_spec, num_accepted_tokens=nat, activation_mode=1)
    print(f"spec:    y shape={y_spec.shape}  cs_out shape={cs_out.shape}")

    # Test ttk-style call
    cs_np, y_np = CausalConv1d(x.numpy(), weight.numpy(), bias.numpy(), cs_fwd.numpy(),
                               None, None, None, None, activation_mode=1, run_mode=0)
    print(f"ttk:     cs shape={cs_np.shape}  y shape={y_np.shape}  dtype={y_np.dtype}")

    print("All passed.")
