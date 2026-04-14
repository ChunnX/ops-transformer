import torch


def get_inputs(param, device=None):
    batch = int(param['batch'])
    num_heads = int(param['num_heads'])
    seq_len = int(param['seq_len'])
    head_dim = int(param['head_dim'])
    mask_batch = int(param['mask_batch'])
    mask_num_heads = int(param['mask_num_heads'])
    dtype_str = param['dtype']
    scale = float(param['scale'])
    dtype = getattr(torch, dtype_str)

    x = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    mask = torch.zeros(mask_batch, mask_num_heads, seq_len, head_dim, device=device, dtype=torch.bool)
    # Randomly set some mask positions to True
    mask = torch.rand(mask_batch, mask_num_heads, seq_len, head_dim, device=device) > 0.7

    return (x, mask, scale)


def get_init_inputs(param, device=None):
    return []


def custom_check_precision(param, outputs, outputs_new):
    dtype_str = param['dtype']
    if dtype_str == 'float16':
        rtol = 1.0 / 512
    elif dtype_str == 'bfloat16':
        rtol = 1.0 / 256 + 1.0 / 16384
    else:
        rtol = 1.0 / 2048 + 1.0 / 16384

    is_pass = True
    abs_diffs = []
    rel_diffs = []

    for out, out_new in zip(outputs, outputs_new):
        if out.numel() == 0:
            continue
        abs_diff = (out.float() - out_new.float()).abs()
        abs_diffs.append(abs_diff.flatten())
        rel_diff = abs_diff / (out.float().abs() + 1e-8)
        rel_diffs.append(rel_diff.flatten())
        if abs_diff.max().item() > rtol * out.float().abs().max().item() + 1e-6:
            is_pass = False

    all_abs_diff = torch.cat(abs_diffs) if abs_diffs else torch.tensor([0.0])
    all_rel_diff = torch.cat(rel_diffs) if rel_diffs else torch.tensor([0.0])
    return is_pass, all_abs_diff, all_rel_diff
