"""
SageAttention2 Python package for Ascend NPU (Atlas 950PR).
"""
from . import _extension, _ops
from .dispatch import (
    dispatch_sageattn,
    dispatch_sageattn_qk_int8_pv_fp8_asc,
    run_sageattn_cpu_reference,
    sageattn,
    sageattn_qk_int8_pv_fp8_asc,
)
from .monkey_patch import (
    is_sdpa_patched,
    replace_sdpa,
    restore_sdpa,
    sdpa_patch,
)

__all__ = [
    "sageattn",
    "sageattn_qk_int8_pv_fp8_asc",
    "dispatch_sageattn",
    "dispatch_sageattn_qk_int8_pv_fp8_asc",
    "run_sageattn_cpu_reference",
    "replace_sdpa",
    "restore_sdpa",
    "sdpa_patch",
    "is_sdpa_patched",
    "_ops",
    "_extension",
]
