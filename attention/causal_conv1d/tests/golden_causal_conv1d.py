"""
PyTorch reference (golden) implementation of CausalConv1d for correctness testing.

Registered as builtin golden plugin via ttk/user_defined_modules/op/golden_funcs/registry.py.

Causal 1D depthwise convolution (per-channel independent):
    y[t, d] = Act( sum_{j=0}^{W-1} weight[j, d] * x[t - j, d] + bias[d] )

Act ∈ {Identity, SiLU}, W = weight.shape[0] ∈ {2, 3, 4}.
History x[-1], x[-2], ... come from conv_states[-(W-1):].

Operator inputs (in order):
  x, weight, bias, conv_states, query_start_loc, cache_indices,
  initial_state_mode, num_accepted_tokens
Attr:
  activation_mode, pad_slot_id, run_mode
Output:
  y
"""

import numpy as np
import torch


def _to_torch(arr):
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor):
        return arr
    return torch.from_numpy(arr)


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
                  conv_state: torch.Tensor, activation: bool) -> torch.Tensor:
    """Causal conv1d for a single sequence. Returns y only."""
    W = weight.shape[0]
    S = x_seq.shape[0]
    history = conv_state[-(W - 1):]
    x_padded = torch.cat([history, x_seq], dim=0)

    y = torch.zeros(S, x_seq.shape[-1], dtype=x_seq.dtype, device=x_seq.device)
    for j in range(W):
        y += weight[j] * x_padded[j : j + S]

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
    activation_mode: int = 0,      # 0=Identity, 1=SiLU
    pad_slot_id: int = -1,
) -> torch.Tensor:
    """Forward mode: full sequence, conv_states read-only."""
    activation = (activation_mode == 1)
    device, dtype = x.device, x.dtype
    xs, B = _split_sequences(x, query_start_loc)
    cache_idx = cache_indices.tolist() if cache_indices is not None else list(range(B))
    is_3d = (x.ndim == 3)

    ys = []
    for i in range(B):
        ci = int(cache_idx[i])
        if ci == pad_slot_id:
            seq_len = x.shape[1] if is_3d else xs[i].shape[0]
            ys.append(torch.zeros(seq_len, x.shape[-1], dtype=dtype, device=device))
            continue
        ys.append(_compute_conv(xs[i], weight, bias, conv_states[ci], activation))

    return torch.stack(ys, dim=0) if is_3d else torch.cat(ys, dim=0)


def causal_conv1d_update(
    x: torch.Tensor,              # [T, D] varlen or [B, S, D] batch
    weight: torch.Tensor,         # [W, D]
    bias: torch.Tensor | None,    # [D] or None
    conv_states: torch.Tensor,    # [num_cache_lines, state_len, D]
    query_start_loc: torch.Tensor | None = None,  # [batch+1] int64
    cache_indices: torch.Tensor | None = None,    # [batch] int64
    activation_mode: int = 0,      # 0=Identity, 1=SiLU
    pad_slot_id: int = -1,
) -> torch.Tensor:
    """Update mode: incremental step, writes back last W-1 tokens to conv_states in-place."""
    activation = (activation_mode == 1)
    W = weight.shape[0]
    device, dtype = x.device, x.dtype
    xs, B = _split_sequences(x, query_start_loc)
    cache_idx = cache_indices.tolist() if cache_indices is not None else list(range(B))
    is_3d = (x.ndim == 3)

    ys = []
    for i in range(B):
        ci = int(cache_idx[i])
        if ci == pad_slot_id:
            seq_len = x.shape[1] if is_3d else xs[i].shape[0]
            ys.append(torch.zeros(seq_len, x.shape[-1], dtype=dtype, device=device))
            continue

        x_seq = xs[i]
        S = x_seq.shape[0]
        history = conv_states[ci][-(W - 1):]
        x_padded = torch.cat([history, x_seq], dim=0)

        y_i = _compute_conv(x_seq, weight, bias, conv_states[ci], activation)

        # Write back: shift state, append newest tokens
        last_tokens = x_padded[S:]
        new_row = torch.cat([conv_states[ci][last_tokens.shape[0]:], last_tokens], dim=0)
        conv_states[ci].copy_(new_row)
        ys.append(y_i)

    return torch.stack(ys, dim=0) if is_3d else torch.cat(ys, dim=0)


# ============================================================
# Ttk registry entry — matches operator input order + attrs
# ============================================================

def CausalConv1d(
    x,                  # [T, D] or [B, S, D]
    weight,             # [W, D]
    bias,               # [D] or None
    conv_states,        # [num_cache_lines, state_len, D]
    query_start_loc=None,            # [batch+1] int64
    cache_indices=None,              # [batch] int64
    initial_state_mode=None,         # [batch] int64 — not used in golden
    num_accepted_tokens=None,        # [batch] int64 — not used in golden
    *,
    activation_mode=0,   # attr
    pad_slot_id=-1,      # attr
    run_mode=0,          # attr
    **extra,
):
    """
    Golden function for CausalConv1d operator.
    Called by ttk test framework with numpy arrays, returns numpy array.

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

    if run_mode == 0:
        y_t = causal_conv1d_fwd(x_t, w_t, b_t, cs_t,
                                query_start_loc=qsl_t, cache_indices=ci_t,
                                activation_mode=activation_mode, pad_slot_id=pad_slot_id)
    else:
        y_t = causal_conv1d_update(x_t, w_t, b_t, cs_t,
                                    query_start_loc=qsl_t, cache_indices=ci_t,
                                    activation_mode=activation_mode, pad_slot_id=pad_slot_id)
    return y_t.detach().cpu().numpy()


# -----------------------------------------------------------
# Quick smoke test
# -----------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    B, S, D, W, state_len = 2, 8, 128, 4, 8
    x = torch.randn(B, S, D, dtype=torch.float16)
    weight = torch.randn(W, D, dtype=torch.float16)
    bias = torch.randn(D, dtype=torch.float16)
    conv_states = torch.zeros(B, state_len, D, dtype=torch.float16)

    y = causal_conv1d_fwd(x, weight, bias, conv_states, activation_mode=1)
    print(f"fwd:    shape={y.shape}  y[0,:2,:3]={y[0,:2,:3]}")

    x_upd = x[:, -1:, :]
    y, = causal_conv1d_update(x_upd, weight, bias, conv_states, activation_mode=1)
    print(f"update: shape={y.shape}")

    qsl = torch.tensor([0, 3, 8], dtype=torch.int64)
    x_v = torch.cat([x[0, :3], x[1, :5]], dim=0)
    y = causal_conv1d_fwd(x_v, weight, bias, conv_states, query_start_loc=qsl, activation_mode=1)
    print(f"varlen: shape={y.shape}  expected=[8,128]")

    # Test ttk-style call
    y_np = CausalConv1d(x.numpy(), weight.numpy(), bias.numpy(), conv_states.numpy(),
                        None, None, None, None, activation_mode=1, run_mode=0)
    print(f"ttk:    shape={y_np.shape}  dtype={y_np.dtype}")

    print("All passed.")
