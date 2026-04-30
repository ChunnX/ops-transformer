import torch


def get_init_inputs(param, device=None):
    return []


def get_inputs(param, device=None):
    device = device or "cpu"
    bs = int(param['bs'])
    nh = int(param['nh'])
    ed = int(param['ed'])
    nq = int(param['nq'])
    np_val = int(param['np'])

    # Parse spatial shapes: e.g., "(4,4),(5,5)" -> [(4,4), (5,5)]
    shapes_str = param['spatial_shapes']
    level_shapes = []
    for pair in shapes_str.split('),('):
        pair = pair.strip('()')
        h, w = pair.split(',')
        level_shapes.append((int(h), int(w)))

    nL = len(level_shapes)

    # Compute level_start_index and total spatial size
    spatial_sizes = [h * w for h, w in level_shapes]
    total_spatial = sum(spatial_sizes)
    level_start_index = [0]
    for i in range(len(spatial_sizes) - 1):
        level_start_index.append(level_start_index[-1] + spatial_sizes[i])

    # Create tensors in kernel format
    # value: [B, N, nH, D]
    value = torch.rand(bs, total_spatial, nh, ed, dtype=torch.float32, device=device) * 0.01

    # spatial_shapes: [L, 2]
    spatial_shapes = torch.tensor(level_shapes, dtype=torch.int32, device=device).reshape(nL, 2)

    # level_start_index: [L]
    level_start_idx = torch.tensor(level_start_index, dtype=torch.int32, device=device)

    # sampling_locations: [B, nH, L, nP, 2, nQ] - normalized [0,1]
    sampling_locations = torch.rand(bs, nh, nL, np_val, 2, nq, dtype=torch.float32, device=device)

    # attention_weights: [B, nH, L, nP, nQ] - positive values
    attention_weights = torch.rand(bs, nh, nL, np_val, nq, dtype=torch.float32, device=device) + 1e-5

    # grad_output: [B, nQ, nH, D]
    grad_output = torch.ones(bs, nq, nh, ed, dtype=torch.float32, device=device)

    return (value, spatial_shapes, level_start_idx,
            sampling_locations, attention_weights, grad_output)
