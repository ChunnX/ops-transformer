import torch
import math


def get_inputs(param, device=None):
    """Generate input tensors for RoPE with sin/cos cache."""
    num_tokens = int(param['num_tokens'])
    num_q_heads = int(param['num_q_heads'])
    num_kv_heads = int(param['num_kv_heads'])
    head_size = int(param['head_size'])
    rotary_dim = int(param['rotary_dim'])
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    max_pos = int(param.get('max_pos', 128))

    positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int64, device=device)

    query_in = torch.randn(num_tokens, num_q_heads * head_size, dtype=dtype, device=device)
    key_in = torch.randn(num_tokens, num_kv_heads * head_size, dtype=dtype, device=device)

    cos_sin_cache = torch.randn(max_pos, rotary_dim, dtype=dtype, device=device)

    return [positions, query_in, key_in, cos_sin_cache]


def get_init_inputs(param, device=None):
    """Extract initialization parameters."""
    num_q_heads = int(param['num_q_heads'])
    num_kv_heads = int(param['num_kv_heads'])
    head_size = int(param['head_size'])
    is_neox_style = bool(int(param.get('is_neox_style', 1)))
    mrope_section = [int(x) for x in param.get('mrope_section', '0,0,0').split(',')]
    cache_mode = int(param.get('cache_mode', 0))

    return [num_q_heads, num_kv_heads, head_size, is_neox_style, mrope_section, cache_mode]


def custom_check_precision(param, outputs, outputs_new):
    """Check precision for RoPE outputs (query_out, key_out)."""
    query_out, key_out = outputs
    query_out_new, key_out_new = outputs_new

    dtype_str = param.get('dtype', 'float32')
    if dtype_str == 'float16':
        rtol = 1.0 / 512
    elif dtype_str == 'bfloat16':
        rtol = 1.0 / 256 + 1.0 / 16384
    else:
        rtol = 1.0 / 2048 + 1.0 / 16384

    abs_diffs = []
    rel_diffs = []

    is_pass = True
    for out_ref, out_new in [(query_out, query_out_new), (key_out, key_out_new)]:
        out_ref_f = out_ref.float()
        out_new_f = out_new.float()
        abs_diff = (out_ref_f - out_new_f).abs()
        abs_diffs.append(abs_diff.flatten())

        ref_abs = out_ref_f.abs()
        rel_diff = torch.where(ref_abs > 0, abs_diff / ref_abs, abs_diff)
        rel_diffs.append(rel_diff.flatten())

        if not torch.allclose(out_ref_f, out_new_f, rtol=rtol):
            is_pass = False

    all_abs_diff = torch.cat(abs_diffs)
    all_rel_diff = torch.cat(rel_diffs)

    return is_pass, all_abs_diff, all_rel_diff
