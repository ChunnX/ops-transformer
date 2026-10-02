"""
SageAttention2 Python package for Ascend NPU (Atlas 950PR).
"""

from . import _extension, _ops
from .core import sageattn, sageattn_qk_int8_pv_fp8_asc

__all__ = ["sageattn", "sageattn_qk_int8_pv_fp8_asc", "_ops", "_extension"]
