import torch


def get_init_inputs(param, device=None):
    return []


def get_inputs(param, device=None):
    shape = eval(param['shape'])
    dtype = param['dtype']
    device = device or "cpu"

    M, K, N = shape[0], shape[1], shape[2]
    transpose_x1 = bool(param.get('transpose_x1', False)) if not isinstance(param.get('transpose_x1', ''), str) else (param['transpose_x1'].lower() == 'true')
    transpose_x2 = bool(param.get('transpose_x2', False)) if not isinstance(param.get('transpose_x2', ''), str) else (param['transpose_x2'].lower() == 'true')
    has_bias = bool(param.get('has_bias', False)) if not isinstance(param.get('has_bias', ''), str) else (param['has_bias'].lower() == 'true')
    group_size = int(param.get('group_size', 0))
    out_dtype_int = int(param.get('out_dtype', 0))

    x1_shape = [M, K]
    x2_shape = [K, N]

    x1 = torch.randint(-128, 127, x1_shape, dtype=torch.int8, device=device)
    x2 = torch.randint(-128, 127, x2_shape, dtype=torch.int8, device=device)

    scale = torch.rand(N, dtype=torch.float32, device=device) * 0.1 + 0.01
    offset = torch.rand(N, dtype=torch.float32, device=device) * 0.01

    bias = None
    if has_bias:
        bias = torch.rand(M, N, dtype=torch.float32, device=device) * 0.1

    pertoken_scale = None

    return (x1, x2, scale, offset, bias, pertoken_scale,
            out_dtype_int, transpose_x1, transpose_x2, group_size)
