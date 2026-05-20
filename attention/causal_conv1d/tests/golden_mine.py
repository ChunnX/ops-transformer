"""My golden (3D batch + 2D varlen, from tests/golden_causal_conv1d.py)."""
import torch
import torch.nn.functional as F


def _silu(x):
    return x * torch.sigmoid(x)


def causal_conv1d_mine(x, weight, bias, conv_states, query_start_loc=None,
                        cache_indices=None, initial_state_mode=None,
                        activation_mode=0, pad_slot_id=-1):
    """3D batch or 2D varlen causal_conv1d."""
    activation = (activation_mode == 1)
    W = weight.shape[0]
    D = x.shape[-1]
    device, dtype = x.device, x.dtype

    is_3d = (x.ndim == 3)
    if is_3d:
        B = x.shape[0]
        xs = [x[i] for i in range(B)]
    else:
        qsl = query_start_loc.to(torch.int64)
        B = qsl.shape[0] - 1
        xs = [x[qsl[i]: qsl[i + 1]] for i in range(B)]
    cache_idx = cache_indices.tolist() if cache_indices is not None else list(range(B))

    ys = []
    for i in range(B):
        ci = int(cache_idx[i])
        if ci == pad_slot_id:
            sl = x.shape[1] if is_3d else xs[i].shape[0]
            ys.append(torch.zeros(sl, D, dtype=dtype, device=device))
            continue

        x_seq = xs[i]
        cs = conv_states[ci]
        has_init = initial_state_mode is not None and bool(initial_state_mode[i].item())
        hist = cs[-(W - 1):] if has_init else torch.zeros(W - 1, D, dtype=dtype, device=device)
        padded = torch.cat([hist, x_seq], dim=0).T.unsqueeze(0)
        w = weight.T.unsqueeze(1)
        y_i = F.conv1d(padded, w, stride=1, padding=0, groups=D).squeeze(0).T

        if bias is not None:
            y_i += bias
        if activation:
            y_i = _silu(y_i)
        ys.append(y_i)

    return torch.stack(ys, dim=0) if is_3d else torch.cat(ys, dim=0)
