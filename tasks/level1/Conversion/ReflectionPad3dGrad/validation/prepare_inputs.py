import torch


def get_inputs(param, device=None):
    """
    Generate (grad_output, self_input) for ReflectionPad3d backward.
    Padding is passed via get_init_inputs for Model(*init_params).
    """
    input_shape = eval(param.get("input_shape", "[1, 1, 3, 3, 3]"))
    dtype_str = param.get("dtype", "float16")
    padding = eval(param.get("padding", "[1, 1, 1, 1, 1, 1]"))

    dtype = getattr(torch, dtype_str)

    padded_shape = list(input_shape)
    padded_shape[-1] += padding[0] + padding[1]
    padded_shape[-2] += padding[2] + padding[3]
    padded_shape[-3] += padding[4] + padding[5]

    grad_output = torch.randn(padded_shape, device=device, dtype=dtype)
    self_input = torch.randn(input_shape, device=device, dtype=dtype)

    return (grad_output, self_input)


def get_init_inputs(param, device=None):
    padding = eval(param.get("padding", "[1, 1, 1, 1, 1, 1]"))
    if isinstance(padding, tuple):
        padding = list(padding)
    return (padding,)


def custom_check_precision(param, ref_output, test_output):
    dtype_str = param.get("dtype", "float16")
    if dtype_str not in ("float16", "bfloat16", "float32"):
        return None
    df = (ref_output.float() - test_output.float()).abs().flatten()
    rff = ref_output.float().abs().flatten()
    rd = df / (rff + 1e-2)
    if dtype_str == "float32":
        ok = df.max() <= 1e-3
    else:
        ok = df.max() <= 0.2
    return (1 if bool(ok) else 0), df, rd
