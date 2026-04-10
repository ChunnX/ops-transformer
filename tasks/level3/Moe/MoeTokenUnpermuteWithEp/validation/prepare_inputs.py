import torch
import numpy as np


def get_inputs(param, device=None):
    """Generate input tensors based on test case parameters."""
    num_tokens = param.get('num_tokens')
    num_topk = param.get('num_topk')
    hidden_size = param.get('hidden_size')
    range_start = param.get('range_start')
    range_end = param.get('range_end')
    dtype_str = param.get('dtype', 'float32')
    has_probs = param.get('has_probs', True)

    if dtype_str == 'float32':
        dtype = torch.float32
    elif dtype_str == 'float16':
        dtype = torch.float16
    elif dtype_str == 'bfloat16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Total number of permuted tokens in the local EP partition
    num_local_tokens = (range_end - range_start) * num_topk

    # Generate permuted_tokens: [num_local_tokens, hidden_size]
    permuted_tokens = torch.randn(num_local_tokens, hidden_size, dtype=dtype, device=device) * 0.1

    # Generate sorted_indices: [num_tokens * num_topk]
    # Values must be valid global token indices, some in [range_start, range_end) and some outside
    total_permuted = num_tokens * num_topk
    # Generate indices: some in range, some outside
    indices = torch.randint(0, range_end + 10, (total_permuted,), dtype=torch.int32, device=device)
    # Ensure at least some indices are in [range_start, range_end)
    for i in range(0, total_permuted, num_topk):
        idx_in_range = torch.randint(range_start, range_end, (1,), dtype=torch.int32)
        indices[i] = idx_in_range.item()

    # Generate probs (optional)
    if has_probs:
        probs = torch.rand(num_tokens, num_topk, dtype=dtype, device=device)
    else:
        probs = None

    return (permuted_tokens, indices, probs, num_topk, range_start, range_end)


def get_init_inputs(param, device=None):
    return []


def custom_check_precision(param, outputs, outputs_new):
    """Check precision of outputs."""
    dtype_str = param.get('dtype', 'float32')
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

        # Compute relative diff (avoid div by zero)
        ref_abs = out.float().abs()
        rel_diff = torch.where(ref_abs > 1e-6, abs_diff / ref_abs, abs_diff)
        rel_diffs.append(rel_diff.flatten())

        if abs_diff.max().item() > rtol * 10:
            is_pass = False

    all_abs_diff = torch.cat(abs_diffs) if abs_diffs else torch.tensor([0.0])
    all_rel_diff = torch.cat(rel_diffs) if rel_diffs else torch.tensor([0.0])

    return is_pass, all_abs_diff, all_rel_diff
