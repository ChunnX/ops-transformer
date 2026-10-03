"""
SageAttention2 Python package for Ascend NPU (Atlas 950PR).
"""
import torch
from . import _extension, _ops

sageattn = torch.ops.sageattention2.forward
sageattn_qk_int8_pv_fp8_asc = torch.ops.sageattention2.forward

__all__ = ["sageattn", "sageattn_qk_int8_pv_fp8_asc", "_ops", "_extension"]
