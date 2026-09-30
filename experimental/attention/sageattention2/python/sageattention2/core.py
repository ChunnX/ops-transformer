"""Core API entry for SageAttention2 on Ascend NPU (aligning with open-source sageattention.core)."""

from ._ops import (
    replace_sdpa,
    run,
    sageattn,
    sageattn_qk_int8_pv_fp8_asc,
)

__all__ = ["sageattn", "sageattn_qk_int8_pv_fp8_asc", "replace_sdpa", "run"]
