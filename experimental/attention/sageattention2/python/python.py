"""
Compatibility bridge for legacy/test scripts importing `python as py_pkg`.
Directs all exports to `sageattention2`.
"""
from sageattention2 import (
    sageattn,
    sageattn_qk_int8_pv_fp8_asc,
    replace_sdpa,
    restore_sdpa,
    sdpa_patch,
    is_sdpa_patched,
    enable_npu_op_timeout,
    _ops,
    _extension,
)

__all__ = [
    "sageattn",
    "sageattn_qk_int8_pv_fp8_asc",
    "replace_sdpa",
    "restore_sdpa",
    "sdpa_patch",
    "is_sdpa_patched",
    "enable_npu_op_timeout",
    "_ops",
    "_extension",
]
