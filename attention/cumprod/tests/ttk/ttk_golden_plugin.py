"""TTK TestSpec plugin for Cumprod (kernel/geir) and aclnnCumprod (aclnn).

Golden semantics (spec.yaml): y = np.cumprod(x, axis=dim); fp16/bf16 inputs
accumulate in fp32 (torch CPU acc_type semantics: cumprod in fp32, cast back
per element on store).
"""

import numpy as np
import torch

__spec__ = {
    "Cumprod": "CumprodTestSpec",
    "aclnnCumprod": "AclnnCumprodTestSpec",
}


def _cumprod_golden_fp32(x, dim):
    """Golden always in fp32 (TTK promote convention): fp16/bf16 kernel
    outputs are compared against the fp32 reference with dtype tolerance."""
    if torch.is_tensor(x):
        x_t = x.float()
    else:
        arr = np.ascontiguousarray(x)
        # ml_dtypes bfloat16 is not accepted by torch.from_numpy: go through fp32.
        if arr.dtype not in (np.float64, np.float32, np.float16, np.int64, np.int32,
                             np.int16, np.int8, np.uint64, np.uint32, np.uint16,
                             np.uint8, np.bool_):
            arr = arr.astype(np.float32)
        x_t = torch.from_numpy(arr).float()
    return torch.cumprod(x_t, dim=dim)


def _axis_to_dim(axis):
    """axis value: plain int (op attr), small tensor, or numpy array."""
    if axis is None:
        return 0
    if isinstance(axis, int):
        return axis
    if torch.is_tensor(axis):
        return int(axis.reshape(-1)[0].item())
    if isinstance(axis, (list, tuple)):
        return int(axis[0])
    arr = np.asarray(axis).reshape(-1)
    return int(arr[0])


class CumprodTestSpec:
    """Cumprod golden for the kernel / geir paths (numpy inputs).

    axis arrives as an op attribute (int) after the attr-interface refactor.
    """

    @staticmethod
    def golden(x, axis=None, exclusive=False, reverse=False, **kwargs):
        return [_cumprod_golden_fp32(x, _axis_to_dim(axis))]

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }


class AclnnCumprodTestSpec:
    """aclnnCumprod golden for the aclnn path (torch tensor inputs).

    Official signature: aclnnCumprod(input, dim: aclScalar, dtype, out) — the
    golden receives (x, dim, dtype, out) in API order; `axis` kept as an alias
    for older callers.
    """

    @staticmethod
    def golden(x, dim=None, axis=None, dtype=None, out=None, **kwargs):
        d = dim if dim is not None else axis
        return [_cumprod_golden_fp32(x, _axis_to_dim(d))]

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
