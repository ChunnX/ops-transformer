import torch


def _grad_out_shape(input_shape, dim, size, step):
    inp = list(input_shape)
    d = int(dim)
    if d < 0:
        d += len(inp)
    g = inp.copy()
    g[d] = (inp[d] - size) // step + 1
    g.append(int(size))
    return g


def get_inputs(param, device=None):
    input_shape = eval(param.get("input_shape", "[2, 3, 16]"))
    dtype_str = param.get("dtype", "float16")
    dim = int(param["dim"])
    size = int(param["size"])
    step = int(param["step"])
    dtype = getattr(torch, dtype_str)
    gshape = _grad_out_shape(input_shape, dim, size, step)
    grad_out = torch.randn(gshape, device=device, dtype=dtype)
    return (grad_out,)


def get_init_inputs(param, device=None):
    input_shape = eval(param.get("input_shape", "[2, 3, 16]"))
    return (
        list(input_shape),
        int(param["dim"]),
        int(param["size"]),
        int(param["step"]),
    )


def custom_check_precision(param, ref_output, test_output):
    dtype_str = param.get("dtype", "float16")
    max_abs = 0.02 if dtype_str in ("float16", "bfloat16") else 0.001
    max_rel = 0.02 if dtype_str in ("float16", "bfloat16") else 0.001
    diff = (ref_output.float() - test_output.float()).abs()
    rdiff = diff / (ref_output.float().abs() + 1e-5)
    ok = (diff.max() <= max_abs) and (
        (rdiff.max() <= max_rel) or bool((ref_output.abs() < 1e-4).all())
    )
    return (1 if bool(ok) else 0), diff.flatten(), rdiff.flatten()
