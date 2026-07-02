"""FP4 E2M1 / FP8 E8M0 解码与反量化工具函数库（Golden 参考实现）。

核心算法与测试端标准实现完全一致。
仅保留解码/反量化函数，正向量化（bf16→fp4）移至 my_test/。

KV 格式：unpacked fp4，每 uint8 存 1 个 nibble（值 0-15），D 维不减半。
Scale 格式：unpacked e8m0，每 uint8 存 1 个 scale，D/32。
反量化时自动 clamp 到 target_dtype 范围防止 fp16 溢出。
"""

import torch


# ============================================================
# 常量与枚举
# ============================================================

# FP4 E2M1 值表 (0-15 nibble)
FP4_E2M1_TABLE = [
    -0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]

_FP4_TABLE_TENSOR = torch.tensor(FP4_E2M1_TABLE, dtype=torch.float32)


class QuantMode:
    PER_TOKEN_GROUP = "per_token_group"
    PER_TOKEN = "per_token"
    PER_CHANNEL = "per_channel"


class QuantDtype:
    FP4_E2M1 = "fp4_e2m1"
    FP4_E1M2 = "fp4_e1m2"
    FP8_E4M3 = "fp8_e4m3"
    INT8 = "int8"


class ScaleDtype:
    FP8_E8M0 = "fp8_e8m0"
    FP32 = "fp32"
    BF16 = "bf16"


# ============================================================
# 核心解码函数
# ============================================================

def cvt_fp4_e2m1_to_bfloat16(x: int) -> float:
    """fp4 nibble → float 值。"""
    Fp4e2m1ToBf16 = {
        '0': 0.0, '1': 0.5, '2': 1.0, '3': 1.5,
        '4': 2.0, '5': 3.0, '6': 4.0, '7': 6.0,
        '8': -0.0, '9': -0.5, '10': -1.0, '11': -1.5,
        '12': -2.0, '13': -3.0, '14': -4.0, '15': -6.0,
    }
    x = int(x)
    first_fp4val = x & 0x0f
    first_fp4str = str(first_fp4val)
    return Fp4e2m1ToBf16[first_fp4str]


def ieee_754_conversion(sign: int, exponent_raw: int, mantissa: int,
                        exp_len: int = 8, m_len: int = 23) -> float:
    """IEEE 754 格式解析。"""
    sign_value = -1 if sign == 1 else 1
    exponent = exponent_raw - (2 ** (exp_len - 1) - 1)
    mant_mult = 1
    for b in range(m_len - 1, -1, -1):
        if mantissa & (2 ** b):
            mant_mult += 1 / (2 ** (m_len - b))
    return sign_value * (2 ** exponent) * mant_mult


# ============================================================
# 向量化解码函数
# ============================================================

def fp4_to_float(tensor_uint8: torch.Tensor) -> torch.Tensor:
    """unpacked fp4 uint8 (..., D) → float32 (..., D)。

    每个 uint8 元素取低 4 位 (& 0x0F) 查表，D 维不变。
    与测试端的 cvt_fp4_e2m1_to_bfloat16 逐元素逻辑完全一致。
    """
    flat = tensor_uint8.reshape(-1).to(torch.uint8)
    low_nibbles = (flat & 0x0F).long()
    out = _FP4_TABLE_TENSOR[low_nibbles]
    return out.reshape(tensor_uint8.shape)


def e8m0_to_float(tensor_uint8: torch.Tensor) -> torch.Tensor:
    """unpacked e8m0 uint8 (..., D/32) → float32 (..., D/32)。

    值 = 2^(byte - 127)，使用 pow 向量化，D/32 维不变。
    0xFF 在数学上 = 2^128，与 fp4 最大值相乘后超出 fp16 表示范围，随机生成时已避开。
    """
    flat = tensor_uint8.reshape(-1).to(torch.int32)
    out = torch.pow(2.0, flat.to(torch.float32) - 127.0)
    return out.reshape(tensor_uint8.shape)


# ============================================================
# 反量化
# ============================================================

def dequant_kv(
    kv_fp4: torch.Tensor,
    scale_e8m0: torch.Tensor,
    target_dtype: torch.dtype,
    group_size: int = 32,
    kv_quant_dtype: str = QuantDtype.FP4_E2M1,
    scale_quant_dtype: str = ScaleDtype.FP8_E8M0,
    quant_mode: str = QuantMode.PER_TOKEN_GROUP,
) -> torch.Tensor:
    """fp4 uint8 + e8m0 uint8 → 与 Q 同精度的 float tensor。

    Args:
        kv_fp4: (..., D) uint8, unpacked fp4 nibble, D 不变
        scale_e8m0: (..., D/32) uint8, unpacked e8m0 scale
        target_dtype: 目标精度，必须与 Q 的 dtype 一致 (fp16 或 bf16)
        group_size: 每组元素数，默认 32
        kv_quant_dtype: KV 量化精度（预留扩展）
        scale_quant_dtype: scale 量化精度（预留扩展）
        quant_mode: 量化模式（预留扩展）

    Returns:
        (..., D) tensor, dtype = target_dtype

     流程（与测试端老接口 antiquant 完全一致）：
       1. kv_float = fp4_to_float(kv_fp4)                           # D → D
       2. scale_float = e8m0_to_float(scale_e8m0)                    # D/32 → D/32
       3. scale_broadcast = scale_float.repeat_interleave(group_size) # D/32 → D
       4. kv_raw = kv_float * scale_broadcast
       5. kv_deq = kv_raw.clamp(-finfo.max, finfo.max)               # 防 fp16 溢出
       6. return kv_deq.to(target_dtype)
    """
    assert kv_fp4.dtype == torch.uint8, f"kv_fp4 must be uint8, got {kv_fp4.dtype}"
    assert scale_e8m0.dtype == torch.uint8, f"scale_e8m0 must be uint8, got {scale_e8m0.dtype}"
    assert target_dtype in (torch.float16, torch.bfloat16), \
        f"target_dtype must be fp16 or bf16, got {target_dtype}"

    d = kv_fp4.shape[-1]
    d_scale = scale_e8m0.shape[-1]
    assert d == d_scale * group_size, \
        f"D mismatch: kv D={d}, scale D/32={d_scale}, group_size={group_size}"

    kv_float = fp4_to_float(kv_fp4)
    scale_float = e8m0_to_float(scale_e8m0)
    scale_broadcast = scale_float.repeat_interleave(group_size, dim=-1)
    kv_raw = kv_float * scale_broadcast
    finfo = torch.finfo(target_dtype)
    kv_deq = kv_raw.clamp(-finfo.max, finfo.max)
    return kv_deq.to(target_dtype)
