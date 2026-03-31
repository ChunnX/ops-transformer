import math

import numpy as np
import torch


def parse_bool_param(param):
    if isinstance(param, str) and param.lower() in ["true", "t", "1"]:
        return True
    if param == 1:
        return True
    return False


def custom_check_precision(row, ref_output, test_output):
    dtype_str = str(row.get("dtype", "float32")).lower()
    if dtype_str in ("float16", "half"):
        max_abs, max_rel = 2e-2, 2e-2
    elif dtype_str in ("bfloat16", "bf16", "bfloat"):
        max_abs, max_rel = 5e-2, 5e-2
    else:
        max_abs, max_rel = 5e-3, 5e-3

    from framework.utils import check_precision

    return check_precision(ref_output, test_output, max_abs, max_rel)


def get_inputs(param, device=None):
    rng = np.random.default_rng(13 + int(param.get("case_id", 0)))

    dtype_map = {
        "float32": torch.float32,
        "float": torch.float32,
    }
    dtype_str = str(param.get("dtype", "float32")).lower()
    dtype = dtype_map.get(dtype_str, torch.float32)

    b = int(param.get("batch", 2))
    h = int(param.get("input_h", 24))
    w = int(param.get("input_w", 32))
    c = int(param.get("channels", 8))
    n = int(param.get("num_rois", 8))
    pooled_h = int(param.get("pooled_h", 4))
    pooled_w = int(param.get("pooled_w", 4))
    spatial_scale = float(param.get("spatial_scale", 0.25))
    sampling_ratio = int(param.get("sampling_ratio", 0))
    aligned = parse_bool_param(param.get("aligned", True))
    clockwise = parse_bool_param(param.get("clockwise", False))

    x = torch.tensor(rng.standard_normal((b, h, w, c), dtype=np.float32), dtype=dtype)
    rois = torch.zeros((6, n), dtype=dtype)
    for j in range(n):
        bi = int(rng.integers(0, b))
        cx_feat = float(rng.uniform(w * 0.25, w * 0.75))
        cy_feat = float(rng.uniform(h * 0.25, h * 0.75))
        cx = cx_feat / max(spatial_scale, 1e-8)
        cy = cy_feat / max(spatial_scale, 1e-8)
        rw_feat = float(rng.uniform(2.0, min(10.0, 0.45 * w)))
        rh_feat = float(rng.uniform(2.0, min(10.0, 0.45 * h)))
        rw = rw_feat / max(spatial_scale, 1e-8)
        rh = rh_feat / max(spatial_scale, 1e-8)
        theta = float(rng.uniform(-math.pi / 3, math.pi / 3))
        rois[0, j] = float(bi)
        rois[1, j] = cx
        rois[2, j] = cy
        rois[3, j] = rw
        rois[4, j] = rh
        rois[5, j] = theta

    grad_out = torch.tensor(
        rng.standard_normal((n, pooled_h, pooled_w, c), dtype=np.float32),
        dtype=dtype,
    )

    if device:
        x = x.to(device)
        rois = rois.to(device)
        grad_out = grad_out.to(device)

    # 最后一项 x 仅用于 CPU 参考求导；ModelNew 不传给 aclnn
    return (
        grad_out,
        rois,
        pooled_h,
        pooled_w,
        spatial_scale,
        sampling_ratio,
        aligned,
        clockwise,
        x,
    )


def get_init_inputs(param, device=None):
    return []
