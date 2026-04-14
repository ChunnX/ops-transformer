import torch
import numpy as np


def get_inputs(param, device=None):
    """Generate input tensors based on test case parameters."""
    dtype = getattr(torch, param['dtype'])
    x_shape = eval(param['x_shape'])
    index_shape = eval(param['index_shape'])
    dim = int(param['dim'])

    x = torch.randn(x_shape, dtype=dtype, device=device)

    # Generate valid index values for the gather dimension
    # index values must be in range [0, x.shape[dim])
    index_list = []
    for i in range(len(index_shape)):
        if i == (dim % len(index_shape)):
            index_list.append(torch.randint(0, x_shape[dim % len(x_shape)], (index_shape[i],)))
        else:
            index_list.append(torch.arange(index_shape[i]))
    index = torch.stack(torch.meshgrid(*index_list, indexing='ij'), dim=-1).long()
    # Use a simpler approach: generate random valid indices
    idx_dim = dim % len(index_shape)
    index = torch.zeros(index_shape, dtype=torch.int32, device=device)
    for i in range(index_shape[idx_dim]):
        # Build indices for gather
        pass
    # Simpler: just use randint for the gather dim, ensure valid range
    max_idx = x_shape[dim % len(x_shape)]
    index = torch.randint(0, max_idx, index_shape, dtype=torch.int32, device=device)

    return (x, index, dim)


def get_init_inputs(param, device=None):
    return []


def custom_check_precision(param, outputs, outputs_new):
    """Check precision between golden and kernel outputs."""
    dtype = param['dtype']
    rtol = 0.0
    atol = 0.0
    if dtype == 'float32':
        rtol = 1.0 / 2048 + 1.0 / 16384
        atol = 1e-5
    elif dtype == 'float16':
        rtol = 1.0 / 512
        atol = 1e-3
    elif dtype == 'bfloat16':
        rtol = 1.0 / 256 + 1.0 / 16384
        atol = 1e-2
    elif dtype == 'int32':
        rtol = 0
        atol = 0

    is_pass = True
    abs_diffs = []
    rel_diffs = []

    for golden, new in zip(outputs, outputs_new):
        if golden.numel() == 0:
            continue
        golden_f = golden.float()
        new_f = new.float()
        abs_diff = (golden_f - new_f).abs()
        abs_diffs.append(abs_diff.flatten())

        if rtol > 0:
            rel_diff = abs_diff / (golden_f.abs() + 1e-8)
            rel_diffs.append(rel_diff.flatten())
            if not torch.allclose(golden_f, new_f, rtol=rtol, atol=atol):
                is_pass = False
        else:
            if not torch.equal(golden, new):
                is_pass = False

    all_abs_diff = torch.cat(abs_diffs) if abs_diffs else torch.tensor([0.0])
    all_rel_diff = torch.cat(rel_diffs) if rel_diffs else torch.tensor([0.0])
    return is_pass, all_abs_diff, all_rel_diff
