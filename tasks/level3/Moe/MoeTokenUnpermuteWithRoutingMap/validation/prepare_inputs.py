import torch
import numpy as np


def get_inputs(param, device=None):
    """Generate input tensors based on test case parameters."""
    num_tokens = param.get('num_tokens')
    num_experts = param.get('num_experts')
    topK = param.get('top_k')
    hidden_size = param.get('hidden_size')
    dtype_str = param.get('dtype', 'float32')
    has_probs = param.get('has_probs', True)
    has_routing_map = param.get('has_routing_map', True)
    drop_and_pad = param.get('drop_and_pad', False)

    if dtype_str == 'float32':
        dtype = torch.float32
    elif dtype_str == 'float16':
        dtype = torch.float16
    elif dtype_str == 'bfloat16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    num_out_tokens = num_tokens * topK
    total_length = num_tokens * topK

    # Generate permuted_tokens: [num_out_tokens, hidden_size]
    permuted_tokens = torch.randn(num_out_tokens, hidden_size, dtype=dtype, device=device) * 0.1

    # Generate sorted_indices: [total_length]
    # Values are valid indices into permuted_tokens [0, num_out_tokens)
    indices = torch.randint(0, num_out_tokens, (total_length,), dtype=torch.int32, device=device)

    # Generate routing_map (optional): [num_tokens, num_experts]
    # When has_routing_map is True, all-ones to simplify golden model
    if has_routing_map and has_probs:
        routing_map = torch.ones(num_tokens, num_experts, dtype=torch.bool, device=device)
    else:
        routing_map = None

    # Generate probs (optional): [num_tokens, num_experts]
    if has_probs:
        probs = torch.rand(num_tokens, num_experts, dtype=dtype, device=device) * 0.5 + 0.5
    else:
        probs = None

    # restore_shape: used when probs is None to compute topK
    if not has_probs:
        restore_shape = [num_tokens]
    else:
        restore_shape = [num_tokens]

    return (permuted_tokens, indices, routing_map, probs, drop_and_pad, restore_shape)


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

        ref_abs = out.float().abs()
        rel_diff = torch.where(ref_abs > 1e-6, abs_diff / ref_abs, abs_diff)
        rel_diffs.append(rel_diff.flatten())

        if abs_diff.max().item() > rtol * 10:
            is_pass = False

    all_abs_diff = torch.cat(abs_diffs) if abs_diffs else torch.tensor([0.0])
    all_rel_diff = torch.cat(rel_diffs) if rel_diffs else torch.tensor([0.0])

    return is_pass, all_abs_diff, all_rel_diff
