import torch
import numpy as np


def get_inputs(param, device=None):
    """Generate input tensors based on test case parameters."""
    num_tokens = param.get('num_tokens')
    num_experts = param.get('num_experts')
    top_k = param.get('top_k')
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

    # Compute shapes based on padded mode
    if drop_and_pad:
        capacity = param.get('capacity', top_k)
        total_length = num_experts * capacity
    else:
        capacity = top_k
        total_length = num_tokens * top_k

    # unpermuted_tokens_grad: [tokens_num, hidden_size]
    unpermuted_tokens_grad = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device) * 0.1

    if drop_and_pad:
        # Padded mode: out_index and permute_token_id shape = [experts_num * capacity]
        # Build realistic indices with unique (token, expert) pairs to avoid
        # non-deterministic overwrite in kernel's probs_grad computation
        out_index_list = []
        permute_token_id_list = []
        for e in range(num_experts):
            # Assign unique tokens per expert to avoid collisions
            token_ids = torch.randperm(num_tokens)[:capacity].tolist()
            for c in range(capacity):
                out_index_list.append(e * capacity + c)
                permute_token_id_list.append(token_ids[c])
        out_index = torch.tensor(out_index_list, dtype=torch.int32, device=device)
        permute_token_id = torch.tensor(permute_token_id_list, dtype=torch.int32, device=device)
    else:
        # Non-padded mode: out_index and permute_token_id shape = [tokens_num * topK]
        # Build a realistic mapping: each token is routed to topK experts
        out_index_list = []
        permute_token_id_list = []
        permuted_offset = 0
        for t in range(num_tokens):
            for k in range(top_k):
                out_index_list.append(permuted_offset)
                permute_token_id_list.append(t)
                permuted_offset += 1
        out_index = torch.tensor(out_index_list, dtype=torch.int32, device=device)
        permute_token_id = torch.tensor(permute_token_id_list, dtype=torch.int32, device=device)

    # routing_map (optional): [tokens_num, experts_num]
    if has_routing_map and has_probs:
        routing_map = torch.ones(num_tokens, num_experts, dtype=torch.bool, device=device)
    else:
        routing_map = None

    # permuted_tokens (optional): [total_length, hidden_size]
    if has_probs:
        permuted_tokens = torch.randn(total_length, hidden_size, dtype=dtype, device=device) * 0.1
    else:
        permuted_tokens = None

    # probs (optional): [tokens_num, experts_num]
    if has_probs:
        probs_dtype = dtype
        # Mixed dtype case: bf16 tokens with float probs
        if dtype_str == 'bfloat16' and param.get('mixed_probs', False):
            probs_dtype = torch.float32
        probs = torch.rand(num_tokens, num_experts, dtype=probs_dtype, device=device) * 0.5 + 0.5
    else:
        probs = None

    # restore_shape: used in padded mode
    restore_shape = [num_tokens]

    return (unpermuted_tokens_grad, out_index, permute_token_id,
            routing_map, permuted_tokens, probs,
            drop_and_pad, restore_shape)


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
        if out is None or out_new is None:
            continue
        if out.numel() == 0 or out_new.numel() == 0:
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
