import os
import random

import torch

from framework.utils import check_precision

# 统一基准种子 + case_id：多次执行同一 case 生成相同张量；不同 case 仍互不相同。
# 可通过环境变量覆盖基准，例如：NKB_PREPARE_INPUTS_SEED=12345
_DEFAULT_PREPARE_RNG_BASE = 34


def _case_id_as_int(param):
    cid = param.get("case_id", 0)
    try:
        import pandas as pd

        if pd.isna(cid):
            return 0
    except Exception:
        pass
    return int(cid)


def _seed_prepare_rng(param):
    base = int(os.environ.get("NKB_PREPARE_INPUTS_SEED", str(_DEFAULT_PREPARE_RNG_BASE)))
    seed = (base + _case_id_as_int(param)) % (2**31)
    torch.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.npu.manual_seed(seed)
    except Exception:
        pass


def _parse_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def _normalize_reduce_str(v, default="update"):
    """CSV 空单元会变成 NaN，勿传给 aclnn 成字符串 'nan'（tiling 仅接受 update/none/空串）。"""
    if v is None:
        return default
    try:
        import pandas as pd

        if pd.isna(v):
            return default
    except Exception:
        pass
    if isinstance(v, float) and v != v:  # NaN
        return default
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return default
    return s


def get_init_inputs(param, device=None):
    reduce_str = _normalize_reduce_str(param.get("reduce", "update"))
    axis = int(param.get("axis", -2))
    indices_rank = int(param.get("indices_rank", 1))
    return (reduce_str, axis, indices_rank)


def get_inputs(param, device=None):
    _seed_prepare_rng(param)

    var_shape = eval(param.get("var_shape", "[2,3,1,32]"), {"__builtins__": {}})
    if len(var_shape) != 4 or var_shape[2] < 1:
        raise ValueError("var_shape must be 4D [D0,D1,D2,D3] with D2>=1")

    d0, d1, d2, d3 = var_shape
    indices_rank = int(param.get("indices_rank", 1))
    idtype_str = param.get("indices_dtype", "int32")
    indices_dtype = getattr(torch, idtype_str) if hasattr(torch, idtype_str) else torch.int32

    upd_dtype_str = param.get("updates_dtype", "float16")
    upd_dtype = getattr(torch, upd_dtype_str)

    var = torch.randint(-64, 64, var_shape, device=device, dtype=torch.int8)
    # 与 ops UT 一致：var_scale 与 var 同秩，最后一维为 1（如 [24,1,128] -> [24,1,1]）
    scale_shape = tuple(var_shape[:-1]) + (1,)
    var_scale = (torch.rand(scale_shape, device=device, dtype=torch.float32) * 0.5 + 0.1).to(torch.float32)

    if indices_rank == 1:
        idx_len = d0
        indices = torch.zeros(idx_len, device=device, dtype=indices_dtype)
        for i in range(idx_len):
            max_v = max(0, d2 - 1)
            indices[i] = torch.randint(0, max_v + 1, (1,), device=device, dtype=indices_dtype).item()
    else:
        idx_len = d0
        indices = torch.zeros((idx_len, 2), device=device, dtype=indices_dtype)
        for i in range(idx_len):
            bs = torch.randint(0, d0, (1,), device=device, dtype=indices_dtype).item()
            max_v = max(0, d2 - 1)
            vv = torch.randint(0, max_v + 1, (1,), device=device, dtype=indices_dtype).item()
            indices[i, 0] = bs
            indices[i, 1] = vv

    updates = (torch.rand(var_shape, device="cpu", dtype=torch.float32) * 2.0 - 1.0).to(
        device=device, dtype=upd_dtype
    )

    if _parse_bool(param.get("has_smooth", False)):
        smooth = (torch.rand((d3,), device="cpu", dtype=torch.float32) * 0.6 + 0.2).to(
            device=device, dtype=upd_dtype
        )
    else:
        smooth = None

    return (var, var_scale, indices, updates, smooth)


def custom_check_precision(param, outputs, outputs_new):
    """var：int8 允许 |Δ|≤1（CAST_RINT/fp16 链路与 half 输入时可放宽至 2）；var_scale：fp32 混合阈值。"""
    outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    outputs_new = outputs_new if isinstance(outputs_new, (list, tuple)) else [outputs_new]
    rv, rs = outputs[0].cpu(), outputs[1].cpu()
    nv, ns = outputs_new[0].cpu(), outputs_new[1].cpu()
    if nv.dtype != rv.dtype:
        nv = nv.to(rv.dtype)
    y_ad = (rv.to(torch.int16) - nv.to(torch.int16)).abs().to(torch.float32).reshape(-1)
    y_tol = 2 if str(param.get("updates_dtype", "")).lower() in ("float16", "bfloat16") else 1
    y_ok = bool((y_ad <= y_tol).all().item())
    y_rd = y_ad / 128.0

    # fp32 scale：绝对 0.01 或 相对 1% 取宽松一侧（与半精度输入链路的 scale 漂移一致）
    diff_s = torch.abs(rs.float() - ns.float()).reshape(-1)
    denom = torch.abs(rs.float().reshape(-1)) + 1e-7
    thr = torch.maximum(
        torch.tensor(0.01, device=diff_s.device, dtype=diff_s.dtype),
        0.01 * denom,
    )
    scale_ok = bool((diff_s <= thr).all().item())
    ad_s = diff_s
    rd_s = diff_s / denom

    if os.environ.get("NKB_DEBUG_PREC"):
        print(
            f"[DynamicQuantUpdateScatter] max|var|={y_ad.max().item():.4g} max|scale|={diff_s.max().item():.4g} "
            f"y_ok={y_ok} scale_ok={scale_ok}",
            flush=True,
        )

    is_pass = 1 if (y_ok and scale_ok) else 0
    all_ad = torch.cat([y_ad, ad_s])
    all_rd = torch.cat([y_rd, rd_s])
    return is_pass, all_ad, all_rd
