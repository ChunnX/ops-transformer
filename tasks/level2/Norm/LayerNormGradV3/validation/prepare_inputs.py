import torch

try:
    from framework.utils import check_precision
except ImportError:
    check_precision = None


def custom_check_precision(param, outputs, outputs_new):
    """与 kernel 对齐：float16/bf16 时在 float32 下比较，避免 ref=0 导致 rel 爆炸，且与 kernel float32 再 cast 一致。"""
    outputs = [outputs] if not isinstance(outputs, list) else outputs
    outputs_new = [outputs_new] if not isinstance(outputs_new, list) else outputs_new

    dtype_str = param.get("dtype", "float32")
    if dtype_str in ("float16", "bfloat16"):
        max_abs_error, max_rel_error = 0.02, 0.02
        compare_in_fp32 = True
    else:
        max_abs_error, max_rel_error = 0.001, 0.001
        compare_in_fp32 = False

    all_abs_diff, all_rel_diff = [], []
    is_accurate = True

    for out, out_new in zip(outputs, outputs_new):
        if out_new.dtype == torch.bool:
            out = out.to(torch.int)
            out_new = out_new.to(torch.int)
        if compare_in_fp32 and out.dtype in (torch.float16, torch.bfloat16):
            out = out.float()
            out_new = out_new.float()
        abs_diff = torch.abs(out - out_new)
        rel_diff = abs_diff / (torch.abs(out) + 1e-7)
        all_abs_diff.append(abs_diff.reshape(-1))
        all_rel_diff.append(rel_diff.reshape(-1))

        if torch.any(~torch.isfinite(out)) or torch.any(~torch.isfinite(out_new)):
            is_accurate = False
        if dtype_str in ("float16", "bfloat16"):
            rel_safe = torch.where(torch.isfinite(rel_diff), rel_diff, torch.zeros_like(rel_diff))
            # 标杆接近 0 时只判绝对误差，避免 rel 爆炸
            ref_near_zero = torch.abs(out) < 1e-5
            fail_abs = abs_diff > max_abs_error
            fail_rel = rel_safe > max_rel_error
            if (fail_abs & (ref_near_zero | fail_rel)).any():
                is_accurate = False
        else:
            if torch.any(~torch.isfinite(abs_diff)) or torch.any(~torch.isfinite(rel_diff)):
                is_accurate = False
            if ((abs_diff > max_abs_error) & (rel_diff > max_rel_error)).any():
                is_accurate = False

    all_abs_diff = torch.cat(all_abs_diff)
    all_rel_diff = torch.cat(all_rel_diff)
    return (1 if is_accurate else 0), all_abs_diff, all_rel_diff


def get_inputs(param, device=None):
    """
    Generate inputs for LayerNorm backward: dy, x, mean, rstd, gamma (weight), bias.
    mean/rstd 用 PyTorch native_layer_norm 前向一次得到，与标杆 backward 完全一致。
    """
    input_shape = eval(param.get("input_shape", "[1, 1]"))
    normalized_shape = eval(param.get("normalized_shape", "[1]"))
    dtype_str = param.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)
    eps = float(param.get("epsilon", 1e-5))

    x = torch.rand(input_shape, device=device, dtype=dtype)
    dy = torch.rand(input_shape, device=device, dtype=dtype)

    weight_type = param.get("weight_type", "present")
    bias_type = param.get("bias_type", "present")
    gamma = torch.rand(normalized_shape, device=device, dtype=dtype) if weight_type == "present" else None
    bias = torch.rand(normalized_shape, device=device, dtype=dtype) if bias_type == "present" else None

    # 用 PyTorch 官方 native_layer_norm 前向得到 mean/rstd，保证与 ref backward 同源
    try:
        out, mean, rstd = torch.ops.aten.native_layer_norm(
            x, normalized_shape, gamma, bias, eps
        )
    except Exception:
        # 无 aten 时回退：手算 mean/rstd（与 PyTorch 公式一致：在 reduction_dims 上 mean）
        input_dim = len(input_shape)
        normalized_dim = len(normalized_shape)
        reduction_dims = tuple(range(input_dim - normalized_dim, input_dim))
        mean = x.mean(dim=reduction_dims, keepdim=True)
        var = (x - mean).pow(2).mean(dim=reduction_dims, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        mean_shape = list(input_shape)
        for i in range(normalized_dim):
            mean_shape[input_dim - 1 - i] = 1
        mean = mean.view(mean_shape)
        rstd = rstd.view(mean_shape)

    return (dy, x, mean, rstd, gamma, bias)


def get_init_inputs(param, device=None):
    normalized_shape = eval(param.get("normalized_shape", "[1]"))
    return [normalized_shape]
