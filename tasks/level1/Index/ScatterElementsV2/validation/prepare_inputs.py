import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for scatter_elements_v2 based on test case parameters.
    """
    shape = eval(param['shape'])
    axis = int(param['axis'])
    dtype_str = param['dtype']
    reduction = param['reduction']
    idx_size = int(param['idx_size'])

    dtype = getattr(torch, dtype_str)

    # Normalize axis
    ndim = len(shape)
    norm_axis = axis if axis >= 0 else axis + ndim

    # Create var tensor
    if dtype == torch.bool:
        var = torch.randint(0, 2, shape, device=device).to(dtype)
    else:
        var = torch.randn(shape, device=device, dtype=dtype) * 2 - 1

    # Create indices shape: same as var except on the scatter axis
    indices_shape = list(shape)
    indices_shape[norm_axis] = idx_size

    # Create indices (must be valid indices into the scatter dimension)
    max_idx = shape[norm_axis]
    indices = torch.randint(0, max_idx, indices_shape, device=device, dtype=torch.int64)

    # Create updates tensor (same shape as indices)
    if dtype == torch.bool:
        updates = torch.randint(0, 2, indices_shape, device=device).to(dtype)
    else:
        updates = torch.randn(indices_shape, device=device, dtype=dtype) * 2 - 1

    return (var, indices, updates, axis, reduction)


def get_init_inputs(param, device=None):
    return []


def custom_check_precision(param, outputs, outputs_new):
    """
    Custom precision check for scatter_elements_v2.
    """
    dtype_str = param['dtype']
    dtype = getattr(torch, dtype_str)

    if dtype in (torch.int32, torch.uint8, torch.int8, torch.bool):
        # Integer/bool types: exact match
        is_pass = torch.equal(outputs, outputs_new)
        abs_diff = torch.tensor([0.0])
        rel_diff = torch.tensor([0.0])
    else:
        abs_diffs = []
        rel_diffs = []
        for o, n in zip(outputs, outputs_new):
            if o.numel() == 0:
                continue
            ad = (o.float() - n.float()).abs()
            rd = torch.where(o.float().abs() > 1e-6,
                             ad / o.float().abs(),
                             ad)
            abs_diffs.append(ad.flatten())
            rel_diffs.append(rd.flatten())

        all_abs_diff = torch.cat(abs_diffs) if abs_diffs else torch.tensor([0.0])
        all_rel_diff = torch.cat(rel_diffs) if rel_diffs else torch.tensor([0.0])

        if dtype == torch.float16:
            rtol = 1.0 / 512
        elif dtype == torch.bfloat16:
            rtol = 1.0 / 256 + 1.0 / 16384
        else:  # float32
            rtol = 1.0 / 2048 + 1.0 / 16384

        is_pass = (all_rel_diff.max().item() < rtol)

    if dtype in (torch.int32, torch.uint8, torch.int8, torch.bool):
        return is_pass, torch.tensor([0.0]), torch.tensor([0.0])
    return is_pass, all_abs_diff, all_rel_diff
