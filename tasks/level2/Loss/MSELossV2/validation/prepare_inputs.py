import torch


def get_inputs(param, device=None):
    shape = eval(param.get("shape", "[1]"))
    dtype_str = param.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)

    predict = torch.rand(shape, device=device, dtype=dtype)
    label = torch.rand(shape, device=device, dtype=dtype)
    return [predict, label]


def get_init_inputs(param, device=None):
    reduction = str(param.get("reduction", "mean"))
    return [reduction]


def custom_check_precision(param, ref_output, test_output):
    dtype_str = param.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)
    if dtype == torch.float16:
        rtol = 1 / 512
    elif dtype == torch.bfloat16:
        rtol = 1 / 256 + 1 / 16384
    else:
        rtol = 1 / 2048 + 1 / 16384

    ref_out = ref_output if isinstance(ref_output, torch.Tensor) else ref_output[0]
    test_out = test_output if isinstance(test_output, torch.Tensor) else test_output[0]

    ref_out = ref_out.detach().float().cpu()
    test_out = test_out.detach().float().cpu()

    abs_diff = torch.abs(ref_out - test_out)
    rel_diff = abs_diff / (torch.abs(ref_out) + 1e-7)
    tolerance = rtol * torch.maximum(torch.tensor(1.0), torch.abs(ref_out))

    is_pass = 0 if torch.any(abs_diff > tolerance) else 1
    return is_pass, abs_diff.reshape(-1), rel_diff.reshape(-1)
