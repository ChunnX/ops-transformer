import torch


def get_inputs(param, device=None):
    shape = eval(param.get("shape", "[2,3,4]"))
    dtype_str = param.get("dtype", "float32")
    dtype = getattr(torch, dtype_str)
    x = torch.rand(shape, device=device, dtype=dtype)
    return [x]


def get_init_inputs(param, device=None):
    perm = eval(param.get("perm", "[0,2,1]"))
    perm_t = torch.tensor(perm, dtype=torch.int64, device="cpu")
    return [perm_t]


def custom_check_precision(param, ref_output, test_output):
    dtype_str = param.get("dtype", "float32")
    dtype = getattr(torch, dtype_str)
    if dtype == torch.float16:
        rtol = 1 / 512
    elif dtype == torch.bfloat16:
        rtol = 1 / 256 + 1 / 16384
    else:
        rtol = 1 / 2048 + 1 / 16384

    ref = ref_output if isinstance(ref_output, torch.Tensor) else ref_output[0]
    tst = test_output if isinstance(test_output, torch.Tensor) else test_output[0]
    ref_f = ref.detach().float().cpu()
    tst_f = tst.detach().float().cpu()

    abs_diff = torch.abs(ref_f - tst_f)
    rel_diff = abs_diff / (torch.abs(ref_f) + 1e-7)
    tol = rtol * torch.maximum(torch.tensor(1.0), torch.abs(ref_f))
    is_pass = 0 if torch.any(abs_diff > tol) else 1
    return is_pass, abs_diff.reshape(-1), rel_diff.reshape(-1)
