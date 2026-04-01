import numpy as np
import torch


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


def _split_int(total: int, parts: int, rng: np.random.Generator) -> list:
    """将 total 拆成 parts 个正整数之和。"""
    if parts <= 0 or total < parts:
        raise ValueError("invalid split")
    base = total // parts
    rem = total - base * parts
    out = [base + (1 if i < rem else 0) for i in range(parts)]
    for i in range(parts):
        if out[i] < 1:
            d = 1 - out[i]
            out[i] = 1
            j = (i + 1) % parts
            out[j] = max(1, out[j] - d)
    return out


def get_inputs(param, device=None):
    rng = np.random.default_rng(17 + int(param.get("case_id", 0)))

    dtype_map = {
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
    }
    dtype_str = str(param.get("dtype", "float32")).lower()
    dtype = dtype_map.get(dtype_str, torch.float32)

    b = int(param.get("batch", 2))
    c = int(param.get("channels", 8))
    m = int(param.get("m", 16))
    nsample = int(param.get("nsample", 4))

    ibc_list = _split_int(m, b, rng)
    pts_per_batch = [int(rng.integers(6, 18)) + int(param.get("case_id", 0)) % 3 for _ in range(b)]
    for i in range(b):
        pts_per_batch[i] = max(pts_per_batch[i], ibc_list[i])
    n = sum(pts_per_batch)

    features = torch.tensor(rng.standard_normal((n, c), dtype=np.float32), dtype=dtype)
    # 与 ops-cv golden 一致：features_batch_cnt 长度为 B+1，末元素占位/对齐
    fbc_core = torch.tensor(pts_per_batch, dtype=torch.int32)
    features_batch_cnt = torch.cat([fbc_core, torch.zeros(1, dtype=torch.int32)])
    indices_batch_cnt = torch.tensor(ibc_list, dtype=torch.int32)

    indices = torch.zeros((m, nsample), dtype=torch.int32)
    row = 0
    for bi in range(b):
        hi_cin = pts_per_batch[bi] - 1
        for _ in range(ibc_list[bi]):
            indices[row] = torch.tensor(
                rng.integers(0, max(hi_cin, 0) + 1, size=nsample, dtype=np.int32)
            )
            row += 1

    if device:
        features = features.to(device)
        features_batch_cnt = features_batch_cnt.to(device)
        indices = indices.to(device)
        indices_batch_cnt = indices_batch_cnt.to(device)

    return features, features_batch_cnt, indices, indices_batch_cnt


def get_init_inputs(param, device=None):
    return []
