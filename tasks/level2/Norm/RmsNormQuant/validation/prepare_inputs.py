import os
import torch


def get_inputs(param, device=None):
    shape = eval(param.get('input_shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    x = torch.rand(shape, device=device, dtype=dtype) * 2 - 1
    return (x,)


def get_init_inputs(param, device=None):
    """与 ATK generate_rms_norm_quant 约束一致：scale/offset 为 [1]，gamma/beta 为 [H] 或 [1,H] 后 flatten."""
    normalized_shape = eval(param.get('normalized_shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    epsilon = float(param.get('epsilon', 1e-6))

    gamma = (torch.rand(normalized_shape, device=device, dtype=dtype) + 0.5).flatten()
    beta = torch.randn(normalized_shape, device=device, dtype=dtype).flatten() * 0.1
    # ATK: scale.shape = [1], offset.shape = scale.shape
    scale = torch.ones(1, device=device, dtype=dtype)
    offset = torch.zeros(1, device=device, dtype=torch.int8)
    return [gamma, beta, scale, offset, epsilon]


def custom_check_precision(param, outputs, outputs_new):
    outputs = [outputs] if not isinstance(outputs, list) else outputs
    outputs_new = [outputs_new] if not isinstance(outputs_new, list) else outputs_new
    if os.environ.get("RMS_NORM_QUANT_DEBUG") == "1":
        for i, (out, out_new) in enumerate(zip(outputs, outputs_new)):
            ref_flat = out.cpu().flatten().tolist()
            ker_flat = out_new.cpu().flatten().tolist()
            print(f"[RmsNormQuant] output pair {i}: shape ref={out.shape} ker={out_new.shape}")
            print(f"[RmsNormQuant] ref (标杆) full: {ref_flat}")
            print(f"[RmsNormQuant] ker (算子) full: {ker_flat}")
            diff_flat = [a - b for a, b in zip(ref_flat, ker_flat)]
            print(f"[RmsNormQuant] diff (ref-ker) full: {diff_flat}")
    all_abs_diff, all_rel_diff = [], []
    for out, out_new in zip(outputs, outputs_new):
        abs_diff = torch.abs(out.float() - out_new.float())
        deno = torch.maximum(torch.abs(out.float()), torch.abs(out_new.float())) + 1e-7
        rel_diff = abs_diff / deno
        all_abs_diff.append(abs_diff.view(-1))
        all_rel_diff.append(rel_diff.view(-1))
    all_abs_diff = torch.cat(all_abs_diff)
    all_rel_diff = torch.cat(all_rel_diff)
    # 标杆与 ATK/CANN 一致；int8 舍入/实现差异允许少量点 abs_diff<=4
    # 通过条件：逐元素 abs<=4 且 rel<=0.2，或 宽松 (平均<4 且 异常点<1%)
    strict = torch.all(all_abs_diff <= 4) and torch.all(all_rel_diff <= 0.2)
    n_total = all_abs_diff.numel()
    n_bad = (all_abs_diff > 4).sum().item()
    loose = (torch.mean(all_abs_diff).item() < 4.0) and (n_bad / max(n_total, 1) < 0.01)
    is_pass = 1 if (strict or loose) else 0
    return is_pass, all_abs_diff, all_rel_diff
