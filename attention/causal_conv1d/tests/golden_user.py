"""User's golden (2D varlen, no residual)."""
import torch
import torch.nn.functional as F


def causal_conv1d_user(x, weight, bias, conv_states, query_start_loc,
                        cache_indices, initial_state_mode, activation_mode):
    """2D varlen causal_conv1d, no residual."""
    cu_seq_len, dim = x.shape
    batch = query_start_loc.shape[0] - 1
    kernel_width = weight.size(0)
    state_len = kernel_width - 1
    out = torch.zeros_like(x)

    for i in range(batch):
        start = query_start_loc[i].item()
        end = query_start_loc[i + 1].item()
        seq_x = x[start:end]

        has_init = initial_state_mode is not None and bool(initial_state_mode[i].item())
        if has_init:
            cached_state = conv_states[cache_indices[i]]
        else:
            cached_state = torch.zeros(state_len, dim, dtype=x.dtype, device=x.device)

        padded = torch.cat([cached_state, seq_x], dim=0)
        result = F.conv1d(
            padded.transpose(0, 1).unsqueeze(0),
            weight.transpose(0, 1).unsqueeze(1),
            bias=None, stride=1, padding=0, groups=dim)
        result = result.squeeze(0).transpose(0, 1)

        if bias is not None:
            result += bias
        if activation_mode:
            result = result * torch.sigmoid(result)

        out[start:end] = result
        conv_states[cache_indices[i]] = padded[-state_len:]

    return out
