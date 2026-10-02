# SageAttention2 PyTorch 适配层公开 API（sageattn 自动分发 / FP8 PV 显式入口 / SDPA 替换 / QKV 工厂）
from . import core
from ._ops import replace_sdpa, sageattn, sageattn_qk_int8_pv_fp8_asc
from ._utils import QKV, create_attention_inputs, create_qkv

__all__ = [
    "QKV",
    "core",
    "create_attention_inputs",
    "create_qkv",
    "replace_sdpa",
    "sageattn",
    "sageattn_qk_int8_pv_fp8_asc",
]

