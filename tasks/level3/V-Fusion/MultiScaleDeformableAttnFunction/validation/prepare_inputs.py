import torch
import numpy as np


def get_inputs(param, device=None):
    """
    Generate input tensors for multi-scale deformable attention.

    Required param fields:
        batch_size, num_levels, num_queries, num_heads, num_points, embed_dims
    """
    device = device or "cpu"
    B = int(param['batch_size'])
    L = int(param['num_levels'])
    Q = int(param['num_queries'])
    H = int(param['num_heads'])
    P = int(param['num_points'])
    D = int(param['embed_dims'])

    # Generate spatial shapes: random H, W for each level
    np.random.seed(int(param.get('seed', 42)) + int(param.get('case_id', 0)))
    spatial_shapes = []
    num_keys_total = 0
    for l in range(L):
        h_l = np.random.randint(2, 8)
        w_l = np.random.randint(2, 8)
        spatial_shapes.append([h_l, w_l])
        num_keys_total += h_l * w_l

    spatial_shapes_tensor = torch.tensor(spatial_shapes, dtype=torch.int32, device=device)

    # level_start_index: cumulative sum of spatial sizes
    level_start_index = [0]
    for l in range(L - 1):
        level_start_index.append(level_start_index[-1] + spatial_shapes[l][0] * spatial_shapes[l][1])
    level_start_index_tensor = torch.tensor(level_start_index, dtype=torch.int32, device=device)

    # value: [B, num_keys_total, H, D]
    value = torch.randn(B, num_keys_total, H, D, dtype=torch.float32, device=device)

    # sampling_locations: [B, Q, H, L, P, 2] - normalized coords in [0, 1]
    sampling_locations = torch.rand(B, Q, H, L, P, 2, dtype=torch.float32, device=device)

    # attention_weights: [B, Q, H, L, P] - will be normalized per (Q, H, L, P) group
    attention_weights = torch.rand(B, Q, H, L, P, dtype=torch.float32, device=device)
    # Normalize so weights sum to 1 over (L, P) for each (B, Q, H)
    attention_weights = torch.softmax(attention_weights.view(B, Q, H, -1), dim=-1).view(B, Q, H, L, P)

    return (value, spatial_shapes_tensor, level_start_index_tensor, sampling_locations, attention_weights)


def get_init_inputs(param, device=None):
    return []


def custom_check_precision(param, ref_output, test_output):
    abs_diff = torch.abs(ref_output - test_output)
    rel_diff = abs_diff / (torch.abs(ref_output) + 1e-7)

    dtype = getattr(torch, param.get('dtype', 'float32'))
    if dtype == torch.float16:
        rtol = 1.0 / 512
    elif dtype == torch.bfloat16:
        rtol = 1.0 / 256 + 1.0 / 16384
    else:
        rtol = 1.0 / 2048 + 1.0 / 16384
    max_abs_error = 1e-3
    is_pass = bool(
        torch.all(torch.isfinite(ref_output)).item() and
        torch.all(torch.isfinite(test_output)).item() and
        torch.all((abs_diff <= max_abs_error) | (rel_diff <= rtol)).item()
    )

    return is_pass, abs_diff.reshape(-1), rel_diff.reshape(-1)