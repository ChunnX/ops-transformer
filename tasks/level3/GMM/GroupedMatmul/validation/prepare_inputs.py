import torch


def get_init_inputs(param, device=None):
    return []


def get_inputs(param, device=None):
    M = int(param['M'])
    K = int(param['K'])
    N = int(param['N'])
    dtype_name = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_name)
    device = device or "cpu"

    x = torch.randn(M, K, dtype=dtype, device=device)
    weight = torch.randn(N, K, dtype=dtype, device=device)

    has_bias = param.get('has_bias', 'false').lower() == 'true'
    bias = torch.randn(N, dtype=dtype, device=device) if has_bias else torch.empty(0, dtype=dtype, device=device)

    scale = torch.empty(0, dtype=torch.float32, device=device)
    offset = torch.empty(0, dtype=torch.float32, device=device)
    antiquant_scale = torch.empty(0, dtype=dtype, device=device)
    antiquant_offset = torch.empty(0, dtype=dtype, device=device)
    group_list = torch.empty(0, dtype=torch.int64, device=device)
    per_token_scale = torch.empty(0, dtype=torch.float32, device=device)

    split_item = int(param.get('split_item', '0'))
    group_type = int(param.get('group_type', '-1'))
    group_list_type = int(param.get('group_list_type', '0'))
    act_type = int(param.get('act_type', '0'))
    tuning_config = [int(v) for v in param.get('tuning_config', '0').split(',')]

    return (x, weight, bias, scale, offset, antiquant_scale, antiquant_offset,
            group_list, per_token_scale,
            split_item, group_type, group_list_type, act_type, tuning_config)
