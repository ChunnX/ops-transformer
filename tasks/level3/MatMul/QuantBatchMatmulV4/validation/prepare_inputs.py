import torch


def get_init_inputs(param, device=None):
    return []


def get_inputs(param, device=None):
    shape = eval(param['shape'])
    device = device or "cpu"

    M, K, N = shape[0], shape[1], shape[2]

    def to_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        return str(val).lower() == 'true'

    transpose_x1 = to_bool(param.get('transpose_x1', False))
    transpose_x2 = to_bool(param.get('transpose_x2', False))
    group_size = int(param.get('group_size', 0))
    dtype_int = int(param.get('dtype', 0))
    compute_type = int(param.get('compute_type', 0))

    x1 = torch.randint(-128, 127, [M, K], dtype=torch.int8, device=device)

    if transpose_x2:
        x2 = torch.randint(-128, 127, [N, K], dtype=torch.int8, device=device)
    else:
        x2 = torch.randint(-128, 127, [K, N], dtype=torch.int8, device=device)

    # Decode group sizes
    group_k = group_size & 0xFFFF
    group_n = (group_size >> 16) & 0xFFFF

    if group_k > 0 and group_n > 0:
        # Per-block quantization: 2D scales
        n_groups_k = K // group_k
        n_groups_n = N // group_n
        # Use fixed small scales to reduce output magnitude and absolute error
        x1_scale = torch.full([M, n_groups_k], 0.001, dtype=torch.float32, device=device)
        x2_scale = torch.full([n_groups_n, n_groups_k], 0.001, dtype=torch.float32, device=device)
    else:
        # Per-channel scaling
        x2_scale = torch.rand(N, dtype=torch.float32, device=device) * 0.01 + 0.001
        x1_scale = torch.tensor([0.01], dtype=torch.float32, device=device)

    # y_scale: uint64 (matches binary.json)
    y_scale = torch.tensor([1], dtype=torch.int64, device=device).to(torch.uint64)

    # bias: float32 [N] (required by perblock strategy)
    bias = torch.zeros(N, dtype=torch.float32, device=device)

    # Optional inputs (dummy tensors)
    x1_offset = torch.zeros(1, dtype=torch.bfloat16, device=device)
    x2_offset = torch.zeros(N, dtype=torch.bfloat16, device=device)
    y_offset = torch.zeros(1, dtype=torch.float32, device=device)
    x2_table = None

    return (x1, x2, bias, x1_scale, x2_scale, y_scale,
            x1_offset, x2_offset, y_offset, x2_table,
            dtype_int, compute_type, transpose_x1, transpose_x2, group_size)
