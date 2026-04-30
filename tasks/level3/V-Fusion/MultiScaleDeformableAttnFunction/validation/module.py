import torch
import torch_npu
import torch.nn as nn
import kernel_gen_ops


def ms_deform_attn_forward(value, spatial_shapes, level_start_index, sampling_locations, attention_weights):
    """PyTorch reference for multi-scale deformable attention."""
    # value: [B, N, H, D], spatial_shapes: [L, 2], level_start_index: [L]
    # sampling_locations: [B, Q, H, L, P, 2], attention_weights: [B, Q, H, L, P]
    B, N, H, D = value.shape
    _, Q, _, L, P = attention_weights.shape

    # Reshape value to [B, H, N, D] for easier indexing
    value = value.permute(0, 2, 1, 3)  # [B, H, N, D]

    output = torch.zeros(B, Q, H, D, dtype=value.dtype, device=value.device)

    for b in range(B):
        for q in range(Q):
            for h in range(H):
                val = 0.0
                for l in range(L):
                    h_l = spatial_shapes[l, 0].item()
                    w_l = spatial_shapes[l, 1].item()
                    start = level_start_index[l].item()

                    for p in range(P):
                        u = sampling_locations[b, q, h, l, p, 0].item()
                        v = sampling_locations[b, q, h, l, p, 1].item()
                        w = attention_weights[b, q, h, l, p].item()

                        # Map normalized [0,1] to pixel coordinates
                        x = u * w_l - 0.5
                        y = v * h_l - 0.5

                        x0 = int(torch.floor(torch.tensor(x)).item())
                        y0 = int(torch.floor(torch.tensor(y)).item())
                        x1 = x0 + 1
                        y1 = y0 + 1

                        alpha_x = x - x0
                        alpha_y = y - y0

                        # Bilinear interpolation with clamping
                        def get_value(yy, xx):
                            if 0 <= yy < h_l and 0 <= xx < w_l:
                                idx = start + yy * w_l + xx
                                return value[b, h, idx]
                            return torch.zeros(D, dtype=value.dtype, device=value.device)

                        v00 = get_value(y0, x0)
                        v10 = get_value(y0, x1)
                        v01 = get_value(y1, x0)
                        v11 = get_value(y1, x1)

                        bilinear = ((1 - alpha_y) * (1 - alpha_x) * v00 +
                                    (1 - alpha_y) * alpha_x * v10 +
                                    alpha_y * (1 - alpha_x) * v01 +
                                    alpha_y * alpha_x * v11)

                        val = val + w * bilinear
                output[b, q, h] = val

    # output: [B, Q, H*D]
    return output.reshape(B, Q, H * D)


def ms_deform_attn_forward_fast(value, spatial_shapes, level_start_index, sampling_locations, attention_weights):
    """Vectorized PyTorch reference for multi-scale deformable attention."""
    B, N, H, D = value.shape
    _, Q, _, L, P = attention_weights.shape

    # value: [B, N, H, D] -> [B, H, N, D]
    value = value.permute(0, 2, 1, 3)

    # sampling_locations: [B, Q, H, L, P, 2]
    # attention_weights: [B, Q, H, L, P]
    attn_w = attention_weights.unsqueeze(-1)  # [B, Q, H, L, P, 1]

    result = torch.zeros(B, Q, H, D, dtype=value.dtype, device=value.device)

    for l in range(L):
        h_l = spatial_shapes[l, 0].item()
        w_l = spatial_shapes[l, 1].item()
        start = level_start_index[l].item()

        # Get sampling coords for this level
        locs = sampling_locations[:, :, :, l, :, :]  # [B, Q, H, P, 2]
        u = locs[..., 0]  # [B, Q, H, P]
        v = locs[..., 1]  # [B, Q, H, P]

        x = u * w_l - 0.5
        y = v * h_l - 0.5

        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()
        x1 = x0 + 1
        y1 = y0 + 1

        alpha_x = (x - x0.float()).unsqueeze(-1)  # [B, Q, H, P, 1]
        alpha_y = (y - y0.float()).unsqueeze(-1)  # [B, Q, H, P, 1]

        def safe_idx(yy, xx):
            """Get value at (yy, xx) with clamping, returns [B, Q, H, P, D]."""
            valid = (yy >= 0) & (yy < h_l) & (xx >= 0) & (xx < w_l)
            idx = (start + yy.clamp(0, h_l - 1) * w_l + xx.clamp(0, w_l - 1))
            # idx: [B, Q, H, P] -> permute to [B, H, Q, P] for gathering from value [B, H, N, D]
            idx_exp = idx.permute(0, 2, 1, 3)  # [B, H, Q, P]
            idx_flat = idx_exp.reshape(B, H, -1)  # [B, H, Q*P]
            gathered = torch.gather(value, 2, idx_flat.unsqueeze(-1).expand(-1, -1, -1, D))
            gathered = gathered.reshape(B, H, Q, P, D).permute(0, 2, 1, 3, 4)  # [B, Q, H, P, D]
            gathered = gathered * valid.unsqueeze(-1).float()
            return gathered

        v00 = safe_idx(y0, x0)  # [B, Q, H, P, D]
        v10 = safe_idx(y0, x1)
        v01 = safe_idx(y1, x0)
        v11 = safe_idx(y1, x1)

        bilinear = ((1 - alpha_y) * (1 - alpha_x) * v00 +
                    (1 - alpha_y) * alpha_x * v10 +
                    alpha_y * (1 - alpha_x) * v01 +
                    alpha_y * alpha_x * v11)

        w_l2 = attn_w[:, :, :, l:l+1, :, :]  # [B, Q, H, 1, P, 1]
        result = result + (w_l2.squeeze(3) * bilinear).sum(dim=3)  # sum over P

    return result.reshape(B, Q, H * D)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights):
        return ms_deform_attn_forward_fast(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights
        )


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights):
        return kernel_gen_ops.multi_scale_deformable_attn_function(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights
        )