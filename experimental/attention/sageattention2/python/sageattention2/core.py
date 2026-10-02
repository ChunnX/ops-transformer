"""SageAttention2 Core Python Entrypoints (INT8 QK^T + FP8 PV).

Provides standard operator interfaces:
- sageattn_qk_int8_pv_fp8_asc: Explicit Ascend NPU entrypoint with strict parameter validation.
- sageattn: Dispatcher entrypoint with environment variable fallbacks.
- replace_sdpa: Context manager for drop-in replacement of torch SDPA.
"""

from ._ops import (
    DEFAULT_PV_ACCUM_DTYPE,
    DEFAULT_QK_QUANT_GRAN,
    VALID_PV_ACCUM_DTYPES,
    VALID_QK_QUANT_GRANS,
    replace_sdpa,
    sageattn,
    sageattn_qk_int8_pv_fp8_asc,
)

__all__ = [
    "DEFAULT_PV_ACCUM_DTYPE",
    "DEFAULT_QK_QUANT_GRAN",
    "VALID_PV_ACCUM_DTYPES",
    "VALID_QK_QUANT_GRANS",
    "replace_sdpa",
    "sageattn",
    "sageattn_qk_int8_pv_fp8_asc",
]
