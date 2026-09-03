import logging
import os
import sys
import math
import ctypes
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import zlib
import numpy as np
from typing import Optional, Tuple, List
from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import (
    AclDataType, AclFormat, AclIntArray, AclTensorStruct, Int64, TensorPtr,
    TORCH_TO_ACLTYPE, ACLTYPE_TO_CTYPE, nnopbase, ascendcl,
)
from ml_dtypes import bfloat16
from dataclasses import dataclass

logging = Logger().get_logger()

_FLOAT4_TORCH_STORAGE_ATTRS = ("float4_e2m1fn_x2", "float4_e2m1fn", "float4_e2m1")
_FLOAT4_NPU_STORAGE_ATTRS = ("float4_e2m1fn_x2", "float4_e2m1fn", "float4_e2m1fnu", "float4_e2m1")
_NOT_LIMITED = 2147483647


def ensure_atk_float8_e8m0_support():
    if not hasattr(AclDataType, "ACL_FLOAT8_E8M0"):
        return
    key = "torch.float8_e8m0fnu"
    if key not in TORCH_TO_ACLTYPE:
        TORCH_TO_ACLTYPE[key] = AclDataType.ACL_FLOAT8_E8M0
        logging.info(f"ATK dtype map: register {key} -> ACL_FLOAT8_E8M0")
    if AclDataType.ACL_FLOAT8_E8M0 not in ACLTYPE_TO_CTYPE:
        ACLTYPE_TO_CTYPE[AclDataType.ACL_FLOAT8_E8M0] = ctypes.c_ubyte


def ensure_atk_float4_e2m1_support():
    if not hasattr(AclDataType, "ACL_FLOAT4_E2M1"):
        return
    acl_dt = AclDataType.ACL_FLOAT4_E2M1
    for attr in _FLOAT4_TORCH_STORAGE_ATTRS:
        dt = getattr(torch, attr, None)
        if dt is None:
            continue
        key = f"torch.{attr}"
        if key not in TORCH_TO_ACLTYPE:
            TORCH_TO_ACLTYPE[key] = acl_dt
            logging.info(f"ATK dtype map: register {key} -> ACL_FLOAT4_E2M1")
    if acl_dt not in ACLTYPE_TO_CTYPE:
        ACLTYPE_TO_CTYPE[acl_dt] = ctypes.c_ubyte


try:
    ensure_atk_float8_e8m0_support()
    ensure_atk_float4_e2m1_support()
except Exception as _dtype_reg_err:
    logging.warning(f"BSA MXFP4 aclnn dtype map register skipped: {_dtype_reg_err}")


def get_float4_storage_dtype():
    for attr in _FLOAT4_TORCH_STORAGE_ATTRS:
        dt = getattr(torch, attr, None)
        if dt is not None:
            return dt
    try:
        import torch_npu
        for attr in _FLOAT4_NPU_STORAGE_ATTRS:
            dt = getattr(torch_npu, attr, None)
            if dt is not None:
                return dt
    except ImportError:
        pass
    return None


def prepare_fp4_tensor_on_npu(t: torch.Tensor) -> torch.Tensor:
    """uint8 nibble-pack 上 NPU 后 view 为 float4 storage (对齐 GMM aclnn 路径)."""
    dt = get_float4_storage_dtype()
    if dt is None:
        return t.npu().contiguous() if not t.is_npu else t.contiguous()
    if t.dtype == dt:
        u8 = t.view(torch.uint8).contiguous()
    elif t.dtype == torch.uint8:
        u8 = t.contiguous()
    else:
        u8 = t.to(torch.uint8).contiguous()
    return u8.npu().view(dt)


def to_float8_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """uint8 存储的 e8m0 指数 -> float8_e8m0fnu view。

    没有该 dtype 的环境保持 uint8, 由 aclCreateTensor 侧按 FLOAT8_E8M0 解读。
    """
    e8 = getattr(torch, "float8_e8m0fnu", None)
    if e8 is None:
        return scale.contiguous()
    if scale.dtype == e8:
        return scale.contiguous()
    return scale.to(torch.uint8).contiguous().view(e8)


def _fp4_logical_shape(storage_shape: Tuple[int, ...], head_dim: int) -> Tuple[int, ...]:
    logical = list(storage_shape)
    logical[-1] = int(head_dim)
    return tuple(logical)


def _contiguous_strides(shape: Tuple[int, ...]) -> List[int]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def _create_acl_tensor(npu_tensor: torch.Tensor, acl_dtype: int,
                       logical_shape: Tuple[int, ...]) -> AclTensorStruct:
    """按给定 ACL dtype 与逻辑 shape 建 aclTensor。

    view dims 与 storage dims 都取 logical_shape: FP4 的缓冲区是 uint8 [..., D/2],
    而算子要看到 [..., D]; 若 storage 填物理 shape, ACL 会按 0.5B/元素再算一遍偏移,
    读到的地址就错位了。
    """
    if npu_tensor.device.type != "npu":
        npu_tensor = npu_tensor.npu().contiguous()
    ndim = len(logical_shape)
    dims = (Int64 * ndim)(*[int(x) for x in logical_shape])
    strides = (Int64 * ndim)(*_contiguous_strides(tuple(int(x) for x in logical_shape)))
    addr = ctypes.c_void_p(npu_tensor.data_ptr())
    acl_tensor = nnopbase.aclCreateTensor(
        dims, ndim, acl_dtype, strides, 0, AclFormat.ACL_FORMAT_ND, dims, ndim, addr)
    if not acl_tensor:
        raise RuntimeError(
            f"aclCreateTensor 失败: dtype={acl_dtype} shape={tuple(logical_shape)}")
    return AclTensorStruct(acl_tensor, addr.value, npu_tensor,
                           ascendcl.aclDataTypeSize(acl_dtype))


def create_fp4_acl_tensor(packed_uint8: torch.Tensor, head_dim: int) -> AclTensorStruct:
    """uint8 nibble-pack [..., D/2] -> FLOAT4_E2M1 aclTensor [..., D]。"""
    npu_t = prepare_fp4_tensor_on_npu(packed_uint8)
    logical = _fp4_logical_shape(tuple(int(x) for x in npu_t.shape), head_dim)
    return _create_acl_tensor(npu_t, AclDataType.ACL_FLOAT4_E2M1, logical)


def create_e8m0_acl_tensor(scale: torch.Tensor) -> AclTensorStruct:
    """e8m0 scale -> FLOAT8_E8M0 aclTensor。

    先按 uint8 上 NPU 再 view: float8_e8m0fnu 的 CPU->NPU 直拷在部分 torch_npu
    版本上不支持, 与 FP4 同样的处理。
    """
    u8 = scale.view(torch.uint8) if scale.dtype != torch.uint8 else scale
    npu_t = to_float8_e8m0(u8.contiguous().npu())
    return _create_acl_tensor(npu_t, AclDataType.ACL_FLOAT8_E8M0,
                              tuple(int(x) for x in npu_t.shape))


def create_int64_array(values):
    """int 序列 -> aclIntArray*; None 表示不配置(空指针)。

    空指针必须带正确的 ctypes 类型: ATK before_call 会拿 input_args 的类型逐个与
    C++ 原型比对, 用 aclTensor* 的空指针冒充 aclIntArray* 会当场判成类型不匹配。
    """
    if values is None:
        return ctypes.POINTER(AclIntArray)()
    return nnopbase.create_x_list([Int64(int(v)) for v in values])


random.seed(21)
np.random.seed(21)
torch.manual_seed(21)
torch.set_printoptions(precision=6, sci_mode=False)
torch.set_printoptions(threshold=float('inf'))

DEBUG = False
if not DEBUG:
    print = lambda *args, **kwargs: None


# =====================================================================
# MXFP4 (E2M1) 量化/反量化/打包工具
# =====================================================================
QUANT_MODE_OCP = 2
QUANT_MODE_CX = 3
CX_DEFAULT_DST_TYPE_MAX = 6.0
E8M0_BIAS = 127
FP4_E2M1_EMAX = 2

FP4_E2M1_VALUES = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

_FP4_POS_LEVELS_FP32 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
_FP4_POS_MIDPOINTS_FP32 = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)


def _resolve_cx_params(dst_type_max):
    cx_val = CX_DEFAULT_DST_TYPE_MAX if (dst_type_max is None or dst_type_max == 0) else float(dst_type_max)
    log2_cx = math.log2(cx_val)
    ceil_log2_cx = math.ceil(log2_cx)
    return log2_cx, ceil_log2_cx


def generate_fp4_e2m1_encoding_table():
    encodings, values = [], []
    for s in [0, 1]:
        for e in range(4):
            for m in range(2):
                enc = (s << 3) | (e << 1) | m
                val = (-1) ** s * (2.0 ** (e - 1)) * (1.0 + m / 2.0) if e != 0 \
                    else (-1) ** s * (2.0 ** 0) * (m / 2.0)
                encodings.append(enc)
                values.append(val)
    fp32_to_encoding = {}
    for enc, val in zip(encodings, values):
        if val not in fp32_to_encoding:
            fp32_to_encoding[val] = enc
    return fp32_to_encoding


FP32_TO_FP4 = generate_fp4_e2m1_encoding_table()


def quantize_fp4(tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        expanded = tensor.unsqueeze(-1)
        idx = torch.argmin(torch.abs(expanded - FP4_E2M1_VALUES.to(tensor.device)), dim=-1)
        return FP4_E2M1_VALUES.to(tensor.device)[idx]


# quantize_fp4 会把最后一维展开成 16 档做 argmin, 峰值内存是输入的 16 倍;
# 大序列用例下按块数分片, 把峰值压到百 MB 量级(数值结果与整体处理完全一致)。
_QUANT_FP4_CHUNK_BLOCKS = 1 << 16


def _quantize_fp4_chunked(blocks: torch.Tensor) -> torch.Tensor:
    if blocks.shape[0] <= _QUANT_FP4_CHUNK_BLOCKS:
        return quantize_fp4(blocks)
    out = torch.empty_like(blocks)
    for lo in range(0, blocks.shape[0], _QUANT_FP4_CHUNK_BLOCKS):
        hi = min(lo + _QUANT_FP4_CHUNK_BLOCKS, blocks.shape[0])
        out[lo:hi] = quantize_fp4(blocks[lo:hi])
    return out


def compute_e8m0_scale_blocks(blocks: torch.Tensor, mode: str = "baseline",
                              quant_mode: int = QUANT_MODE_OCP, log2_cx: float = 0.0) -> torch.Tensor:
    """blocks: [n_blk, block_size] -> scales: [n_blk]。逐块版的向量化形式，语义等价。"""
    max_abs = blocks.abs().amax(dim=-1)
    log_max = torch.log2(max_abs)
    if quant_mode == QUANT_MODE_CX:
        exp = torch.ceil(log_max - log2_cx)
    else:
        exp = (torch.floor(log_max) if mode == "baseline" else torch.ceil(log_max)) - FP4_E2M1_EMAX
    exp = torch.clamp(exp, -127.0, 127.0)
    scales = torch.pow(torch.tensor(2.0, dtype=exp.dtype, device=exp.device), exp)
    # 全零块: log2(0)=-inf 会被 clamp 成 -127, 这里按逐块版语义统一取 1.0
    return torch.where(max_abs < 1e-30, torch.ones_like(scales), scales)



def mx_quantize(tensor: torch.Tensor, block_size: int = 32, mode: str = "baseline",
                quant_mode: int = QUANT_MODE_OCP, log2_cx: float = 0.0):
    original_shape = tensor.shape
    if tensor.dim() == 0:
        tensor = tensor.unsqueeze(0)
    last = tensor.shape[-1]
    pad = 0
    if last % block_size != 0:
        pad = block_size - (last % block_size)
        tensor = torch.nn.functional.pad(tensor, (0, pad), value=0.0)
    reshaped = tensor.reshape(-1, block_size)
    scales = compute_e8m0_scale_blocks(reshaped, mode=mode,
                                       quant_mode=quant_mode, log2_cx=log2_cx)
    fp4 = _quantize_fp4_chunked(reshaped / scales.unsqueeze(-1))
    fp4 = fp4.view(tensor.shape)
    if pad > 0:
        fp4 = fp4[..., :-pad]
    return fp4, scales


def mx_dequantize(fp4_vals: torch.Tensor, scales: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    original_shape = fp4_vals.shape
    last = original_shape[-1] if fp4_vals.dim() > 0 else 1
    pad = 0
    if fp4_vals.dim() > 0 and last % block_size != 0:
        pad = block_size - (last % block_size)
        fp4_vals = torch.nn.functional.pad(fp4_vals, (0, pad), value=0.0)
    reshaped = fp4_vals.reshape(-1, block_size)
    dq = reshaped * scales.unsqueeze(1).to(reshaped.device)
    dq = dq.view(fp4_vals.shape)
    if pad > 0:
        dq = dq[..., :last]
    return dq


def mxfp4_qdq(t: torch.Tensor, axis: int, block_size: int = 32, mode: str = "baseline",
              quant_mode: int = QUANT_MODE_OCP, log2_cx: float = 0.0) -> torch.Tensor:
    t_last = t.movedim(axis, -1).contiguous()
    fp4_sim, scales = mx_quantize(t_last, block_size=block_size, mode=mode,
                                  quant_mode=quant_mode, log2_cx=log2_cx)
    dq = mx_dequantize(fp4_sim, scales, block_size=block_size)
    return dq.movedim(-1, axis).contiguous()


def convert_fp32_simulated_to_fp4_encoding(fp32_sim: torch.Tensor) -> torch.Tensor:
    flat = fp32_sim.flatten().cpu()
    enc = torch.zeros_like(flat, dtype=torch.uint8)
    for fp32_val, encoding in FP32_TO_FP4.items():
        enc[flat == fp32_val] = encoding
    return enc.to(fp32_sim.device).reshape(fp32_sim.shape)


def pack_fp4_to_uint8(fp4_enc: torch.Tensor) -> torch.Tensor:
    shape = fp4_enc.shape
    if shape[-1] % 2 != 0:
        raise ValueError(f"last dim must be even, got {shape[-1]}")
    reshaped = fp4_enc.reshape(-1, 2)
    packed = (reshaped[:, 1] << 4) | (reshaped[:, 0] & 0x0F)
    return packed.reshape(*shape[:-1], shape[-1] // 2)


def mxfp4_quantize_pack_last(tensor: torch.Tensor, quant_axis: int,
                             block_size: int = 32, mode: str = "baseline",
                             quant_mode: int = QUANT_MODE_OCP, log2_cx: float = 0.0):
    nd = tensor.dim()
    qax = quant_axis % nd
    assert tensor.shape[-1] % 2 == 0, f"last dim must be even, got {tensor.shape[-1]}"
    t_q_last = tensor.movedim(qax, -1).contiguous()
    fp4_sim, scales_flat = mx_quantize(t_q_last, block_size=block_size, mode=mode,
                                       quant_mode=quant_mode, log2_cx=log2_cx)
    codes = convert_fp32_simulated_to_fp4_encoding(fp4_sim).movedim(-1, qax).contiguous()
    packed = pack_fp4_to_uint8(codes)
    qdim = tensor.shape[qax]
    n_blk = (qdim + block_size - 1) // block_size
    rest = list(t_q_last.shape[:-1])
    scales_real = scales_flat.reshape(*rest, n_blk).movedim(-1, qax).contiguous()
    exp_int = torch.log2(scales_real).round().long()
    e8m0 = (exp_int + E8M0_BIAS).clamp(0, 254).to(torch.uint8)
    return packed, e8m0


def _e8m0_qk_to_v3(e8m0_uint8):
    *lead, nd = e8m0_uint8.shape
    paired = e8m0_uint8.reshape(*lead, nd // 2, 2)
    return paired.contiguous().view(torch.float8_e8m0fnu)


def _e8m0_v_to_v3(e8m0_uint8):
    B, N2, nb, D = e8m0_uint8.shape
    if nb % 2 != 0:
        pad_block = torch.full((B, N2, 1, D), E8M0_BIAS, dtype=torch.uint8)
        e8m0_uint8 = torch.cat([e8m0_uint8, pad_block], dim=2)
        nb += 1
    return e8m0_uint8.reshape(B, N2, nb // 2, 2, D).permute(0, 1, 2, 4, 3).contiguous().view(torch.float8_e8m0fnu)


# =====================================================================
# blockwise_snap_local P量化
# =====================================================================
def _pack_fp4_e2m1_along_last(idx_tensor):
    *batch, n = idx_tensor.shape
    if n % 2 == 1:
        pad = torch.zeros(*batch, 1, dtype=idx_tensor.dtype, device=idx_tensor.device)
        idx_tensor = torch.cat([idx_tensor, pad], dim=-1)
        n += 1
    pairs = idx_tensor.reshape(*batch, n // 2, 2)
    low = pairs[..., 0].to(torch.uint8) & 0xF
    high = pairs[..., 1].to(torch.uint8) & 0xF
    return (high << 4) | low


def _blockwise_snap_local_quantize_p_eff(S: torch.Tensor, m_running, seqk_dim: int,
                                          mx_block_size: int = 32, mx_mode: str = "baseline",
                                          is_cx: bool = False,
                                          neg_log2_cx: float = 0.0,
                                          p_quant_exp_shift: float = 2.0):
    NEG_INF = float('-inf')
    device = S.device
    src_dtype = S.dtype
    ndim = S.dim()
    skp = seqk_dim if seqk_dim >= 0 else ndim + seqk_dim
    seqk_len = S.shape[skp]
    LN2 = math.log(2.0)

    pad = (mx_block_size - seqk_len % mx_block_size) % mx_block_size
    if pad > 0:
        pshape = list(S.shape)
        pshape[skp] = pad
        S_padded = torch.cat([S, torch.full(pshape, NEG_INF, dtype=src_dtype, device=device)], dim=skp)
    else:
        S_padded = S
    S_moved = S_padded.movedim(skp, -1)
    other = S_moved.shape[:-1]
    n_blk = S_moved.shape[-1] // mx_block_size
    S_r = S_moved.reshape(*other, n_blk, mx_block_size)

    m_block_raw = S_r.max(dim=-1).values
    m_block_raw_safe = torch.where(torch.isinf(m_block_raw) & (m_block_raw < 0),
                                   torch.zeros_like(m_block_raw), m_block_raw)
    if is_cx:
        INV_LN2_fp16 = torch.tensor(1.0 / LN2, dtype=torch.float16, device=device)
        LN2_fp16 = torch.tensor(LN2, dtype=torch.float16, device=device)
        neg_log2_cx_fp16 = torch.tensor(neg_log2_cx, dtype=torch.float16, device=device)
        k_raw = m_block_raw_safe * INV_LN2_fp16 + neg_log2_cx_fp16
        K_block = torch.ceil(k_raw)
        m_block_snap = K_block * LN2_fp16
    else:
        INV_LN2_fp16 = torch.tensor(1.0 / LN2, dtype=torch.float16, device=device)
        LN2_fp16 = torch.tensor(LN2, dtype=torch.float16, device=device)
        K_block = torch.floor(m_block_raw_safe * INV_LN2_fp16)
        m_block_snap = (K_block - 2) * LN2_fp16

    P_local = torch.exp(S_r - m_block_snap.unsqueeze(-1))
    P_local_bf = P_local.to(torch.bfloat16).to(P_local.dtype)

    levels = _FP4_POS_LEVELS_FP32.to(device=device, dtype=src_dtype)
    midpoints = _FP4_POS_MIDPOINTS_FP32.to(device=device, dtype=src_dtype)
    idx = torch.bucketize(P_local_bf.contiguous(), midpoints, right=True)
    P_q = levels[idx]

    K_ij = K_block.max(dim=-1).values
    K_ij_fp32 = K_ij.float()
    m_new = K_ij_fp32 if m_running is None else torch.maximum(m_running, K_ij_fp32)
    K_new_s = m_new.to(src_dtype)

    if is_cx:
        K_diff = (K_block - K_new_s.unsqueeze(-1))
        corr = torch.exp2(K_diff - torch.tensor(p_quant_exp_shift, dtype=torch.float16, device=device))
    else:
        K_diff = K_block - K_new_s.unsqueeze(-1)
        corr = torch.exp2(K_diff - 2.0)

    corr_broad = corr.repeat_interleave(mx_block_size, dim=-1)[..., :seqk_len].movedim(-1, skp)
    P_q_moved = P_q.reshape(*other, n_blk * mx_block_size)[..., :seqk_len].movedim(-1, skp)
    P_eff = P_q_moved * corr_broad
    return P_eff, m_new


# =====================================================================
# s_dtype 解析
# =====================================================================
_S_DTYPE_MAP = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
_OUT_DTYPE_MAP = {'bfloat16': torch.bfloat16, 'float16': torch.float16}


def _resolve_s_dtype(s_dtype):
    if s_dtype not in _S_DTYPE_MAP:
        raise ValueError(f"s_dtype must be one of {list(_S_DTYPE_MAP)}, got '{s_dtype}'")
    return _S_DTYPE_MAP[s_dtype]


def _resolve_out_dtype(out_dtype):
    if out_dtype not in _OUT_DTYPE_MAP:
        raise ValueError(f"out_dtype must be one of {list(_OUT_DTYPE_MAP)}, got '{out_dtype}'")
    return _OUT_DTYPE_MAP[out_dtype]


# =====================================================================
# block sparse pattern 生成
# =====================================================================
# 稀疏率候选。必须含 1.0: 每行保留块数按 ceil(ck * ratio) 算, ratio=1.0 才能取到 dense(mask 全 1),
# 而 dense 是最主流形态。1.0 给双份权重, 其余取值覆盖不同密度。
SPARSITY_RATIO_CHOICES = (1.0, 0.9, 0.8, 0.75, 0.5, 0.4, 0.25, 0.1)


def build_block_sparse_mask(sparsity_ratio, q_seqlen_list, kv_seqlen_list, block_shape,
                            batch, num_heads):
    """按 per-batch 实际长度分块生成 blockSparseMask [B, N1, ceil_q, ceil_kv] int8。

    每个 q-block 随机 attend ceil(该 batch 的 kv 块数 * sparsity_ratio) 个 kv 块(至少 1 个)。
    维度用 max(实际长度) 算; 某 batch 实际块数之外的位置保持 0(该 batch 没有对应 token)。
    调用方负责再把维度 0-pad 到 CeilDiv(S_padded, block) 以满足 host 校验。
    """
    block_x, block_y = int(block_shape[0]), int(block_shape[1])
    ceil_q = (max(q_seqlen_list) + block_x - 1) // block_x
    ceil_kv = (max(kv_seqlen_list) + block_y - 1) // block_y
    mask = np.zeros((batch, num_heads, ceil_q, ceil_kv), dtype=np.int8)

    for b in range(batch):
        cq_b = (q_seqlen_list[b] + block_x - 1) // block_x
        ck_b = (kv_seqlen_list[b] + block_y - 1) // block_y
        n_attend = min(ck_b, max(1, math.ceil(ck_b * sparsity_ratio)))
        if n_attend == ck_b:
            mask[b, :, :cq_b, :ck_b] = 1
            continue
        for n1 in range(num_heads):
            for q_blk_id in range(cq_b):
                sel = np.random.choice(ck_b, n_attend, replace=False)
                mask[b, n1, q_blk_id, sel] = 1
    return mask


# =====================================================================
# 数据生成辅助
# =====================================================================
def as_dim(value, name="dim"):
    """把维度值转成 Python 原生 int, 并挡住负值。

    维度非法时不同环境的报错差异很大: 有的 torch 版本对负维度直接说 "negative dimension",
    有的却报 "SymIntArrayRef expected to contain only concrete integers", 完全看不出是负数。
    在造数入口自己校验一遍, 把上游给的坏值当场暴露出来。
    """
    if isinstance(value, torch.Tensor):
        value = value.item()
    try:
        dim = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} 无法转为 int: type={type(value)} value={value!r}") from exc
    if dim < 0:
        raise ValueError(f"{name} 为负数({dim}), 上游传入的维度非法")
    return dim


def as_shape(shape):
    return tuple(as_dim(s, f"shape[{i}]") for i, s in enumerate(shape))


def randn_x(shape, x=1.0, dtype=torch.float32, device=None):
    data_range = [-x, x]
    return data_range[0] + (data_range[1] - data_range[0]) * torch.rand(as_shape(shape), dtype=dtype,
                                                                        device=device)


# 造数幅度。上限取 10.0(与参考脚本默认一致): 再放大后 S=Q·Kᵀ·scale 的量级为 sqrt(D)·x²,
# 在 s_dtype=fp16 下会溢出, 那是造数问题而非算子问题。
DATA_X_CHOICES = (1.0, 10.0, 0.1, 4.0)


def _canon_seed_part(part):
    """把种子输入规范化成与具体数值类型无关的字符串。

    不能直接 repr(): np.int64(2) / torch 标量 / Python int 的 repr 各不相同, 而两个后端喂过来的
    kwargs 类型未必一致, 那样 cpu 与 npu 会算出不同的种子, 造出不同的数据。
    """
    if isinstance(part, (list, tuple)):
        return "[" + ",".join(_canon_seed_part(p) for p in part) + "]"
    if isinstance(part, str):
        return part
    if isinstance(part, float):
        return f"{part:.6g}"
    return str(int(part))


def case_derived_seed(*parts):
    """由用例参数派生随机种子。

    cpu / npu 两个节点各自独立执行 init_by_input_data, 必须拿到同一份数据, 所以种子只能来自
    用例参数; 但固定成常量会让所有用例共用同一份数据(稀疏度、数值幅度都退化成单点取值)。
    用 crc32 而非内置 hash(): 后者对 str 逐进程加盐, 跨进程不稳定。
    """
    key = "|".join(_canon_seed_part(p) for p in parts)
    return zlib.crc32(key.encode("utf-8")) & 0x7FFFFFFF


def _act_seq_given(value):
    """判断 actualSeqLengths 这类 attr_tuple 是否真的配置了。

    "不配置"的表达形式不统一: None、字符串 "None"、空序列, 以及把未配置的整型 attr 物化成
    负数哨兵。最后一种最危险 —— 当成真实长度用会一路传到张量维度上, 而某些 torch 版本对负
    维度报的是 "SymIntArrayRef expected to contain only concrete integers", 根本看不出根因。
    """
    if value is None:
        return False
    if isinstance(value, str):
        return value != "None"
    if len(value) == 0:
        return False
    if all(isinstance(v, str) for v in value):
        return not all(v == "None" for v in value)
    try:
        return min(int(v) for v in value) >= 1
    except (TypeError, ValueError):
        return False


def zero_degenerate_(kv_parts, n_tokens=32):
    """把每个 batch 的前 n_tokens 个 kv token 置 0(就地)。

    目的是造出"整个 32 元素量化块全零"的输入, 覆盖 e8m0 scale 的退化分支
    (max_abs≈0 时取 scale=1.0)。只置零一小段而非全部, 输出仍非平凡, 精度指标才有分辨力。
    """
    for t in kv_parts:
        t[:, :, :min(n_tokens, t.shape[2]), :] = 0.0


# =====================================================================
# CPU golden — BNSD block-sparse MXFP4 分块 flash attention
# =====================================================================
def cpu_block_sparse_attention_bnsd_mxfp4_with_lse(query, key, value, block_sparse_mask,
                                                    block_shape, scale_value, num_kv_heads,
                                                    p_quant='snap_local', s_dtype='fp16',
                                                    out_dtype='bfloat16',
                                                    quant_mode=QUANT_MODE_OCP, dst_type_max=0.0):
    q = query.to(torch.float32)
    k = key.to(torch.float32)
    v = value.to(torch.float32)
    B, N, S, D = q.shape
    N2 = k.shape[1]
    S2 = k.shape[2]
    group_size = N // N2
    block_x, block_y = int(block_shape[0]), int(block_shape[1])
    ceil_q = (S + block_x - 1) // block_x
    ceil_kv = (S2 + block_y - 1) // block_y
    s_torch_dtype = _resolve_s_dtype(s_dtype)

    is_cx = (quant_mode == QUANT_MODE_CX)
    neg_log2_cx = 0.0
    log2_cx = 0.0
    if is_cx:
        log2_cx, ceil_log2_cx = _resolve_cx_params(dst_type_max)
        neg_log2_cx = -log2_cx
        p_quant_exp_shift = float(ceil_log2_cx)
    else:
        p_quant_exp_shift = 2.0

    q_q = mxfp4_qdq(q, axis=-1, quant_mode=quant_mode, log2_cx=log2_cx)
    k_q = mxfp4_qdq(k, axis=-1, quant_mode=quant_mode, log2_cx=log2_cx)
    v_q = mxfp4_qdq(v, axis=2, quant_mode=quant_mode, log2_cx=log2_cx)

    out = torch.zeros((B, N, S, D), dtype=torch.float32)
    lse = torch.full((B, N, S, 1), float('-inf'), dtype=torch.float32)
    scale_s = torch.as_tensor(scale_value, dtype=s_torch_dtype)

    for b in range(B):
        for n in range(N):
            n2 = n // group_size
            for qb in range(ceil_q):
                q_lo = qb * block_x
                q_hi = min(q_lo + block_x, S)
                if q_lo >= S:
                    break
                sel_j = [j for j in range(ceil_kv) if block_sparse_mask[b, n, qb, j] != 0]
                if not sel_j:
                    continue
                chunks = [sel_j[i:i + 2] for i in range(0, len(sel_j), 2)]
                Q_blk = q_q[b, n, q_lo:q_hi]
                nq = q_hi - q_lo
                m_run = None
                l_run = torch.zeros(nq, dtype=torch.float32)
                acc = torch.zeros((nq, D), dtype=torch.float32)

                for chunk in chunks:
                    positions = []
                    for j in chunk:
                        st = j * block_y
                        ed = min(st + block_y, S2)
                        positions.extend(range(st, ed))
                    if not positions:
                        continue
                    pos = torch.as_tensor(positions, dtype=torch.long)
                    K_blk = k_q[b, n2][pos]
                    V_blk = v_q[b, n2][pos]

                    S_fp32 = (K_blk @ Q_blk.t()).to(torch.float32)
                    S_s = (S_fp32 * scale_s).to(s_torch_dtype)

                    if p_quant == 'snap_local':
                        P_eff_s, m_new = _blockwise_snap_local_quantize_p_eff(
                            S_s, m_running=m_run, seqk_dim=0,
                            is_cx=is_cx, neg_log2_cx=neg_log2_cx, p_quant_exp_shift=p_quant_exp_shift)
                        P_eff_s = torch.nan_to_num(P_eff_s, nan=0.0)
                        P_fp32 = P_eff_s.float()
                        alpha = (torch.zeros(nq, dtype=torch.float32) if m_run is None
                                 else torch.exp2(m_run.float() - m_new.float()))
                    else:
                        m_s = S_s.max(dim=0).values
                        m_safe_s = torch.where(torch.isinf(m_s) & (m_s < 0), torch.zeros_like(m_s), m_s)
                        P_ij = torch.exp(S_s - m_safe_s.unsqueeze(0))
                        P_fp32 = torch.nan_to_num(P_ij, nan=0.0).float()
                        m_new = m_safe_s.float()
                        alpha = (torch.zeros(nq, dtype=torch.float32) if m_run is None
                                 else torch.exp(m_run - m_new))

                    reduce_sum = P_fp32.sum(dim=0)
                    l_run = alpha * l_run + reduce_sum
                    acc = alpha.unsqueeze(-1) * acc + (P_fp32.t() @ V_blk)
                    m_run = m_new

                l_safe = torch.where(l_run > 0, l_run, torch.ones_like(l_run))
                acc = acc / l_safe.unsqueeze(-1)
                out[b, n, q_lo:q_hi] = acc
    return out.to(_resolve_out_dtype(out_dtype)), lse.to(torch.float32)


# =====================================================================
# 精度比较
# =====================================================================
def check_result(expect, result, data_type, pct_thd=0.005):
    real_data = result.cpu().numpy().flatten()
    data_compe = expect.cpu().numpy().flatten()
    if real_data.size == 0 and real_data.size == data_compe.size:
        return 100.0, "Pass"
    if real_data.size != data_compe.size:
        return 0.0, "Failed"
    if data_type in ('bfloat16', 'float16'):
        rtol, atol, max_diff_hd = 0.0078125, 0.0001, 10.0
    else:
        rtol, atol, max_diff_hd = 0.005, 0.000025, 10.0
    diff_result = np.isclose(real_data.astype(np.float32), data_compe.astype(np.float32),
                             rtol=rtol, atol=atol, equal_nan=True)
    err_idx = np.where(diff_result != np.array((True,)))[0]
    split_count = real_data.size
    fulfill_percent = float(split_count - err_idx.size) / float(split_count) * 100.0
    diff_abs = np.abs(data_compe - real_data)
    b = np.add(np.maximum(np.abs(real_data), np.abs(data_compe)), 10e-10)
    err_diff = (diff_abs / b)[err_idx]
    pct_thd_val = (1 - pct_thd) * 100.0
    result_str = "Pass" if fulfill_percent >= pct_thd_val else "Failed"
    if len(err_diff) > 0 and np.max(err_diff) >= max_diff_hd:
        result_str = "Failed"
    return fulfill_percent, result_str


# =====================================================================
# ATK golden 计算类
# =====================================================================
class TestBlockSparseAttentionMxfp4GenBmGt():
    @dataclass
    class AuxAttrs:
        num_heads: int
        kv_heads: int
        embedding_size: int
        inner_prec: int
        scale: float
        q_layout: str
        kv_layout: str
        attention_out_dtype: str
        quant_mode: int
        dst_type_max: float

    @dataclass
    class AttentionInputs:
        # q/k/v_parts: per-batch 实际长度(未 padding)的 BNSD 规范张量列表
        #   q_parts[bi]: [1, N1, actualSeqLen[bi], D]; k/v_parts[bi]: [1, N2, actualSeqLenKv[bi], D]
        q_parts: any
        k_parts: any
        v_parts: any
        block_sparse_mask: any
        block_sparse_shape: any
        q_seqlen_list: any
        kv_seqlen_list: any
        padded_s1: any          # BNSD/BSND 输出张量 S 维(padded 大小); TND 为 None
        aux_attrs: any

    def compute_output(self, attention_inputs):
        """Benchmark 分支（is_benchmark=False）：MXFP4 qdq + snap_local P量化 分块flash"""
        block_sparse_mask_np = attention_inputs.block_sparse_mask
        block_sparse_shape = attention_inputs.block_sparse_shape
        q_seqlen_list = attention_inputs.q_seqlen_list
        kv_seqlen_list = attention_inputs.kv_seqlen_list

        batch = len(q_seqlen_list)
        num_heads = attention_inputs.aux_attrs.num_heads
        kv_heads = attention_inputs.aux_attrs.kv_heads
        embedding_size = attention_inputs.aux_attrs.embedding_size
        q_layout = attention_inputs.aux_attrs.q_layout
        kv_layout = attention_inputs.aux_attrs.kv_layout
        scale = attention_inputs.aux_attrs.scale
        inner_prec = attention_inputs.aux_attrs.inner_prec
        out_dtype_str = attention_inputs.aux_attrs.attention_out_dtype
        quant_mode = attention_inputs.aux_attrs.quant_mode
        dst_type_max = attention_inputs.aux_attrs.dst_type_max
        group_size = int(num_heads / kv_heads)

        block_shape_x = block_sparse_shape[0]
        block_shape_y = block_sparse_shape[1]

        q_parts = attention_inputs.q_parts
        k_parts = attention_inputs.k_parts
        v_parts = attention_inputs.v_parts

        mask_tensor = torch.from_numpy(block_sparse_mask_np)

        cpu_chunks = []
        for bi in range(batch):
            cq_b = (q_seqlen_list[bi] + block_shape_x - 1) // block_shape_x
            ck_b = (kv_seqlen_list[bi] + block_shape_y - 1) // block_shape_y
            mask_b = mask_tensor[bi:bi + 1, :, :cq_b, :ck_b]
            out_b, _ = cpu_block_sparse_attention_bnsd_mxfp4_with_lse(
                q_parts[bi], k_parts[bi], v_parts[bi], mask_b,
                [block_shape_x, block_shape_y], scale, kv_heads,
                p_quant='snap_local', s_dtype='fp16', out_dtype=out_dtype_str,
                quant_mode=quant_mode, dst_type_max=dst_type_max)
            cpu_chunks.append(out_b)

        return self._assemble_result(cpu_chunks, q_layout, batch, num_heads, embedding_size,
                                    q_seqlen_list, attention_inputs.padded_s1)

    def compute_output_benchmark(self, attention_inputs):
        """Benchmark 真值分支（is_benchmark=True）：用量化前的原始 fp32 QKV 做标准 BSA"""
        block_sparse_mask_np = attention_inputs.block_sparse_mask
        block_sparse_shape = attention_inputs.block_sparse_shape
        q_seqlen_list = attention_inputs.q_seqlen_list
        kv_seqlen_list = attention_inputs.kv_seqlen_list

        batch = len(q_seqlen_list)
        num_heads = attention_inputs.aux_attrs.num_heads
        kv_heads = attention_inputs.aux_attrs.kv_heads
        embedding_size = attention_inputs.aux_attrs.embedding_size
        q_layout = attention_inputs.aux_attrs.q_layout
        kv_layout = attention_inputs.aux_attrs.kv_layout
        scale = attention_inputs.aux_attrs.scale
        group_size = int(num_heads / kv_heads)

        block_shape_x = block_sparse_shape[0]
        block_shape_y = block_sparse_shape[1]

        # 直接用原始 fp32 数据（不做 mxfp4 qdq）
        q_parts = attention_inputs.q_parts
        k_parts = attention_inputs.k_parts
        v_parts = attention_inputs.v_parts

        mask_tensor = torch.from_numpy(block_sparse_mask_np)

        cpu_chunks = []
        for bi in range(batch):
            cq_b = (q_seqlen_list[bi] + block_shape_x - 1) // block_shape_x
            ck_b = (kv_seqlen_list[bi] + block_shape_y - 1) // block_shape_y
            mask_b = mask_tensor[bi:bi + 1, :, :cq_b, :ck_b]

            q_b = q_parts[bi]   # [1, N, S1, D] fp32
            k_b = k_parts[bi]   # [1, N2, S2, D] fp32
            v_b = v_parts[bi]   # [1, N2, S2, D] fp32
            B_b, N_b, S1_b, D_b = q_b.shape
            N2_b = k_b.shape[1]
            S2_b = k_b.shape[2]

            out_b = torch.zeros((1, N_b, S1_b, D_b), dtype=torch.float32)

            for n in range(N_b):
                n2 = n // group_size
                for qb_idx in range(cq_b):
                    q_lo = qb_idx * block_shape_x
                    q_hi = min(q_lo + block_shape_x, S1_b)
                    if q_lo >= S1_b:
                        break
                    sel_j = [j for j in range(ck_b) if mask_b[0, n, qb_idx, j] != 0]
                    if not sel_j:
                        continue
                    positions = []
                    for j in sel_j:
                        st = j * block_shape_y
                        ed = min(st + block_shape_y, S2_b)
                        positions.extend(range(st, ed))
                    if not positions:
                        continue
                    pos = torch.as_tensor(positions, dtype=torch.long)
                    Q_blk = q_b[0, n, q_lo:q_hi]       # [nq, D]
                    K_blk = k_b[0, n2][pos]             # [npos, D]
                    V_blk = v_b[0, n2][pos]             # [npos, D]

                    # fp32 标准 attention
                    s = (Q_blk.float() @ K_blk.float().t()) * scale
                    row_max = s.max(dim=-1, keepdim=True).values
                    p = torch.exp(s - row_max)
                    p = p / p.sum(dim=-1, keepdim=True)
                    o = p.float() @ V_blk.float()
                    out_b[0, n, q_lo:q_hi] = o

            cpu_chunks.append(out_b)

        return self._assemble_result(cpu_chunks, q_layout, batch, num_heads, embedding_size,
                                    q_seqlen_list, attention_inputs.padded_s1)

    def _assemble_result(self, cpu_chunks, q_layout, batch, num_heads, embedding_size,
                         q_seqlen_list, padded_s1=None):
        """将 per-batch BNSD 结果拼回原始 layout。

        BNSD/BSND 变长(padded)场景: 输出张量 S 维为 padded 大小, [actualSeqLen, S_padded) 区间
        算子不写(kernel 不清零 padding 区), golden 与 NPU 侧都显式置 0 后再比对。
        """
        if q_layout == "TND":
            result_parts = []
            for bi in range(batch):
                result_parts.append(cpu_chunks[bi].permute(0, 2, 1, 3).flatten(0, 1))
            result = torch.cat(result_parts, dim=0)
        elif q_layout == "BNSD":
            out_s = padded_s1 if padded_s1 is not None else max(q_seqlen_list)
            result = torch.zeros(as_shape((batch, num_heads, out_s, embedding_size)),
                                 dtype=cpu_chunks[0].dtype)
            for bi in range(batch):
                result[bi, :, :q_seqlen_list[bi]] = cpu_chunks[bi][0]
        elif q_layout == "BSND":
            out_s = padded_s1 if padded_s1 is not None else max(q_seqlen_list)
            result = torch.zeros(as_shape((batch, out_s, num_heads, embedding_size)),
                                 dtype=cpu_chunks[0].dtype)
            for bi in range(batch):
                result[bi, :q_seqlen_list[bi]] = cpu_chunks[bi][0].permute(1, 0, 2)

        return result.to(torch.float32)


def tensor_to_numpy_array(t):
    if t is None:
        return None
    if t.dtype is torch.bfloat16:
        return t.to(torch.float32).numpy().astype(bfloat16)
    else:
        return t.numpy()


def numpy_array_to_tensor(arr):
    if arr is None:
        return None
    if isinstance(arr, list):
        arr = np.array(arr)
    if arr.dtype.name == 'bfloat16':
        return torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(arr)


# =====================================================================
# ATK 接口注册: 本类只承担 CPU 标杆 + 造数(prep)。被测节点走
# PyAclnnBlockSparseAttentionV3Api(pyaclnn 直调两段式 aclnn)。
# 不 import cann_ops_transformer 的 torch 封装——import 会触发
# OpBuilder/torch.utils.cpp_extension.load 的 JIT 编译。
# =====================================================================
@register("aclnn_blocksparseattention_v3")
class BlockSparseAttentionV3Api(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(BlockSparseAttentionV3Api, self).__init__(task_result)

    def init_by_input_data(self, input_data: InputDataset):
        # logging.info(f"=======================================init_by_input_data=======================================")
        self.q_input_layout = input_data.kwargs["qInputLayout"]
        self.kv_input_layout = input_data.kwargs["kvInputLayout"]
        # 标量属性一律归一成 Python 原生类型: 后端给过来的可能是 np.int64 / 张量标量 / SymInt,
        # 直接拿去构造 torch shape 会报 "expected to contain only concrete integers",
        # 且 cpu/npu 两个后端的类型未必一致, 会让 case_derived_seed 算出不同的种子。
        self.kv_heads = int(input_data.kwargs["numKeyValueHeads"])
        self.scale_value = float(input_data.kwargs["scaleValue"])
        self.inner_precise = int(input_data.kwargs["innerPrecise"])
        self.softmax_lse_flag = int(input_data.kwargs["softmaxLseFlag"])
        self.quant_mode = int(input_data.kwargs.get("quantMode", QUANT_MODE_OCP))
        self.dst_type_max = float(input_data.kwargs.get("dstTypeMax", 0.0))
        self.block_shape = [int(v) for v in input_data.kwargs["blockShape"]]
        raw_act_q = input_data.kwargs["actualSeqLengths"]
        raw_act_kv = input_data.kwargs["actualSeqLengthsKv"]
        # actualSeqLengths/Kv 允许同时不配置(doc: 二者必须同时配置或同时不配置), 此时算子按张量
        # S 维全长计算。TND 下二者必选, 只有 BNSD/BSND 会走这条分支。
        self.pass_act_seq = _act_seq_given(raw_act_q) and _act_seq_given(raw_act_kv)

        attention_out = input_data.kwargs["attentionOut"]
        if attention_out.dtype == torch.bfloat16:
            self.attention_out_dtype_str = "bfloat16"
            self.attention_out_torch_dtype = torch.bfloat16
        else:
            self.attention_out_dtype_str = "float16"
            self.attention_out_torch_dtype = torch.float16

        query_packed = input_data.kwargs["query"]
        if self.q_input_layout == "TND":
            assert self.pass_act_seq, "TND 下 actualSeqLengths/Kv 为必选"
            self.batch = len(raw_act_q)
            self.num_heads = int(query_packed.shape[1])
            self.headDim = int(query_packed.shape[2]) * 2
        elif self.q_input_layout == "BNSD":
            self.batch = int(query_packed.shape[0])
            self.num_heads = int(query_packed.shape[1])
            self.headDim = int(query_packed.shape[3]) * 2
        elif self.q_input_layout == "BSND":
            self.batch = int(query_packed.shape[0])
            self.num_heads = int(query_packed.shape[2])
            self.headDim = int(query_packed.shape[3]) * 2

        if self.pass_act_seq:
            self.q_seqlen_list = tuple(int(x) for x in raw_act_q)
            self.kv_seqlen_list = tuple(int(x) for x in raw_act_kv)
        else:
            # 不配置时等价于每 batch 都取满张量 S 维, golden 侧仍需要具体长度
            key_shape = input_data.kwargs["key"].shape
            s_dim = 2 if self.q_input_layout == "BNSD" else 1
            self.q_seqlen_list = tuple([int(query_packed.shape[s_dim])] * self.batch)
            self.kv_seqlen_list = tuple([int(key_shape[s_dim])] * self.batch)

        assert min(self.q_seqlen_list) >= 1 and min(self.kv_seqlen_list) >= 1, \
            (f"actualSeqLengths 必须为正: q={self.q_seqlen_list} kv={self.kv_seqlen_list} "
             f"(上游原值 q={raw_act_q!r} kv={raw_act_kv!r})")
        assert len(self.q_seqlen_list) == self.batch and len(self.kv_seqlen_list) == self.batch, \
            (f"actualSeqLengths 长度须等于 batch({self.batch}): "
             f"q={self.q_seqlen_list} kv={self.kv_seqlen_list}")

        self.log2_cx = 0.0
        if self.quant_mode == QUANT_MODE_CX:
            self.log2_cx, _ = _resolve_cx_params(self.dst_type_max)

        q_s_list = self.q_seqlen_list
        kv_s_list = self.kv_seqlen_list

        # 种子必须只由用例参数决定(cpu/npu 两节点各自造数, 结果要逐位一致), 但不能是常量,
        # 否则所有用例共用同一份数据分布。稀疏率与造数幅度同样由它派生。
        case_seed = case_derived_seed(
            self.q_input_layout, self.batch, self.num_heads, self.kv_heads, self.headDim,
            q_s_list, kv_s_list, self.block_shape,
            self.quant_mode, self.dst_type_max, self.attention_out_dtype_str)
        random.seed(case_seed)
        np.random.seed(case_seed)
        torch.manual_seed(case_seed)

        sparsity_ratio = SPARSITY_RATIO_CHOICES[case_seed % len(SPARSITY_RATIO_CHOICES)]
        data_x = DATA_X_CHOICES[(case_seed // len(SPARSITY_RATIO_CHOICES)) % len(DATA_X_CHOICES)]
        zero_degenerate = (case_seed // (len(SPARSITY_RATIO_CHOICES) * len(DATA_X_CHOICES))) % 4 == 0
        logging.info(f"=======case_seed:{case_seed};sparsity_ratio:{sparsity_ratio};data_x:{data_x};"
                     f"zero_degenerate:{zero_degenerate};pass_act_seq:{self.pass_act_seq};========")

        # BNSD/BSND 变长场景: 张量 S 维为 padded 大小, actualSeqLengths 指示每 batch 实际有效长度
        # (V3 doc: BNSD/BSND 配置 actualSeqLengths 后算子按实际长度处理)。
        # 造数策略对齐 test_bsa_mxfp4_golden.py 的 padded 分支:
        #   per-batch 造"未 padding"的 fp32 数据 -> golden 直接用; NPU 侧对同一份数据 0-pad 到张量 S 维再打包。
        #   量化一致性靠 0-padding: Q/K 量化轴=D 不跨 seq; V 量化轴=seq, 末尾不满 32 的块在
        #   unpadded(mx_quantize 内部 0-pad) 与 0-padded(外部 0-pad) 两种情形下填充值均为 0 -> scale 相同。
        if self.q_input_layout == "TND":
            self.padded_s1 = None
            self.padded_s2 = None
            T1 = sum(q_s_list)
            T2 = sum(kv_s_list)
            q_tnd = randn_x((T1, self.num_heads, self.headDim), data_x)
            k_tnd = randn_x((T2, self.kv_heads, self.headDim), data_x)
            v_tnd = randn_x((T2, self.kv_heads, self.headDim), data_x)
            self.q_parts, self.k_parts, self.v_parts = [], [], []
            q_offset, kv_offset = 0, 0
            for bi in range(self.batch):
                sq, skv = q_s_list[bi], kv_s_list[bi]
                self.q_parts.append(q_tnd[q_offset:q_offset + sq].unsqueeze(0).permute(0, 2, 1, 3).contiguous())
                self.k_parts.append(k_tnd[kv_offset:kv_offset + skv].unsqueeze(0).permute(0, 2, 1, 3).contiguous())
                self.v_parts.append(v_tnd[kv_offset:kv_offset + skv].unsqueeze(0).permute(0, 2, 1, 3).contiguous())
                q_offset += sq
                kv_offset += skv
        else:
            key_packed = input_data.kwargs["key"]
            if self.q_input_layout == "BNSD":
                self.padded_s1 = int(query_packed.shape[2])
                self.padded_s2 = int(key_packed.shape[2])
            else:
                self.padded_s1 = int(query_packed.shape[1])
                self.padded_s2 = int(key_packed.shape[1])
            assert max(q_s_list) <= self.padded_s1 and max(kv_s_list) <= self.padded_s2, \
                (f"actualSeqLengths 须 <= 张量 S 维: q={q_s_list} vs {self.padded_s1}, "
                 f"kv={kv_s_list} vs {self.padded_s2}")
            self.q_parts = [randn_x((1, self.num_heads, q_s_list[bi], self.headDim), data_x)
                            for bi in range(self.batch)]
            self.k_parts = [randn_x((1, self.kv_heads, kv_s_list[bi], self.headDim), data_x)
                            for bi in range(self.batch)]
            self.v_parts = [randn_x((1, self.kv_heads, kv_s_list[bi], self.headDim), data_x)
                            for bi in range(self.batch)]

        if zero_degenerate:
            zero_degenerate_(self.k_parts)
            zero_degenerate_(self.v_parts)

        self.block_sparse_mask_np = build_block_sparse_mask(
            sparsity_ratio, q_s_list, kv_s_list, self.block_shape,
            self.batch, self.num_heads)

        # mask 按 per-batch 实际长度分块生成, 再 0-pad 到 host 校验要求的 CeilDiv(S_padded, block)
        # (host CheckSparsePattern 只校验维度, padding 块内容置 0 表示"无 token 可 attend")
        if self.padded_s1 is not None:
            ceil_q_pad = (self.padded_s1 + self.block_shape[0] - 1) // self.block_shape[0]
            ceil_kv_pad = (self.padded_s2 + self.block_shape[1] - 1) // self.block_shape[1]
            pad_q = max(0, ceil_q_pad - self.block_sparse_mask_np.shape[2])
            pad_kv = max(0, ceil_kv_pad - self.block_sparse_mask_np.shape[3])
            if pad_q > 0 or pad_kv > 0:
                self.block_sparse_mask_np = np.pad(
                    self.block_sparse_mask_np,
                    ((0, 0), (0, 0), (0, pad_q), (0, pad_kv)), constant_values=0)

        self.block_sparse_mask_tensor = numpy_array_to_tensor(self.block_sparse_mask_np)

        # 打包 QKV 为 MXFP4 格式（NPU 侧输入）
        if self.q_input_layout == "TND":
            q_p_ch, k_p_ch, v_p_ch = [], [], []
            q_s_ch, k_s_ch, v_s_ch = [], [], []
            for bi in range(self.batch):
                q_b = self.q_parts[bi]
                k_b = self.k_parts[bi]
                v_b = self.v_parts[bi]
                q_p, q_e = mxfp4_quantize_pack_last(q_b, quant_axis=-1, quant_mode=self.quant_mode, log2_cx=self.log2_cx)
                k_p, k_e = mxfp4_quantize_pack_last(k_b, quant_axis=-1, quant_mode=self.quant_mode, log2_cx=self.log2_cx)
                v_p, v_e = mxfp4_quantize_pack_last(v_b, quant_axis=-2, quant_mode=self.quant_mode, log2_cx=self.log2_cx)
                q_p_ch.append(q_p.permute(0, 2, 1, 3).flatten(0, 1))
                k_p_ch.append(k_p.permute(0, 2, 1, 3).flatten(0, 1))
                v_p_ch.append(v_p.permute(0, 2, 1, 3).flatten(0, 1))
                q_s_ch.append(_e8m0_qk_to_v3(q_e).permute(0, 2, 1, 3, 4).flatten(0, 1))
                k_s_ch.append(_e8m0_qk_to_v3(k_e).permute(0, 2, 1, 3, 4).flatten(0, 1))
                v_s_ch.append(_e8m0_v_to_v3(v_e).permute(0, 2, 1, 3, 4).flatten(0, 1))
            self.q_packed = torch.cat(q_p_ch, dim=0).contiguous()
            self.k_packed = torch.cat(k_p_ch, dim=0).contiguous()
            self.v_packed = torch.cat(v_p_ch, dim=0).contiguous()
            self.q_scale = torch.cat(q_s_ch, dim=0).contiguous()
            self.k_scale = torch.cat(k_s_ch, dim=0).contiguous()
            self.v_scale = torch.cat(v_s_ch, dim=0).contiguous()
        else:
            # per-batch 实际长度数据 0-pad 到张量 S 维后 cat 成 BNSD [B, N, S_padded, D]
            q_bnsd = torch.cat([
                torch.nn.functional.pad(self.q_parts[bi], (0, 0, 0, self.padded_s1 - q_s_list[bi]))
                for bi in range(self.batch)], dim=0).contiguous()
            k_bnsd = torch.cat([
                torch.nn.functional.pad(self.k_parts[bi], (0, 0, 0, self.padded_s2 - kv_s_list[bi]))
                for bi in range(self.batch)], dim=0).contiguous()
            v_bnsd = torch.cat([
                torch.nn.functional.pad(self.v_parts[bi], (0, 0, 0, self.padded_s2 - kv_s_list[bi]))
                for bi in range(self.batch)], dim=0).contiguous()

            q_packed_t, q_e8m0 = mxfp4_quantize_pack_last(q_bnsd, quant_axis=-1, quant_mode=self.quant_mode, log2_cx=self.log2_cx)
            k_packed_t, k_e8m0 = mxfp4_quantize_pack_last(k_bnsd, quant_axis=-1, quant_mode=self.quant_mode, log2_cx=self.log2_cx)
            v_packed_t, v_e8m0 = mxfp4_quantize_pack_last(v_bnsd, quant_axis=-2, quant_mode=self.quant_mode, log2_cx=self.log2_cx)
            self.q_scale = _e8m0_qk_to_v3(q_e8m0)
            self.k_scale = _e8m0_qk_to_v3(k_e8m0)
            self.v_scale = _e8m0_v_to_v3(v_e8m0)

            if self.q_input_layout == "BSND":
                self.q_packed = q_packed_t.permute(0, 2, 1, 3).contiguous()
                self.k_packed = k_packed_t.permute(0, 2, 1, 3).contiguous()
                self.v_packed = v_packed_t.permute(0, 2, 1, 3).contiguous()
                self.q_scale = self.q_scale.permute(0, 2, 1, 3, 4).contiguous()
                self.k_scale = self.k_scale.permute(0, 2, 1, 3, 4).contiguous()
                self.v_scale = self.v_scale.permute(0, 2, 1, 3, 4).contiguous()
            else:
                self.q_packed = q_packed_t
                self.k_packed = k_packed_t
                self.v_packed = v_packed_t

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        logging.info(f"=======q_input_layout:{self.q_input_layout};self.q_packed:{self.q_packed.shape};q_seqlen_list:{self.q_seqlen_list};;self.k_packed:{self.k_packed.shape};kv_seqlen_list:{self.kv_seqlen_list};padded_s1:{self.padded_s1};padded_s2:{self.padded_s2};========")
        if self.device == "cpu":
            testObj = TestBlockSparseAttentionMxfp4GenBmGt()
            aux_attrs = testObj.AuxAttrs(
                self.num_heads, self.kv_heads, self.headDim,
                self.inner_precise, self.scale_value,
                self.q_input_layout, self.kv_input_layout,
                self.attention_out_dtype_str,
                self.quant_mode, self.dst_type_max)

            attention_inputs = testObj.AttentionInputs(
                self.q_parts, self.k_parts, self.v_parts,
                self.block_sparse_mask_np,
                self.block_shape, self.q_seqlen_list, self.kv_seqlen_list,
                self.padded_s1, aux_attrs)

            if self.task_result.is_benchmark_task:
                result = testObj.compute_output_benchmark(attention_inputs)
            else:
                result = testObj.compute_output(attention_inputs)
            return result.to(torch.float32)

        raise ValueError(
            f"不支持的 device: {self.device}; pyaclnn 节点请使用 "
            f"aclnn_api_type=pyaclnn_aclnn_blocksparseattention_v3")


# =====================================================================
# pyaclnn 节点: 直调两段式 aclnnBlockSparseAttentionV3
# =====================================================================
def _bsa_out_shape(prep):
    """attentionOut 的 shape, 与 query 的 layout 对齐(doc L148-L152)。"""
    if prep.q_input_layout == "TND":
        return (sum(prep.q_seqlen_list), prep.num_heads, prep.headDim)
    if prep.q_input_layout == "BNSD":
        return (prep.batch, prep.num_heads, prep.padded_s1, prep.headDim)
    return (prep.batch, prep.padded_s1, prep.num_heads, prep.headDim)


@register("pyaclnn_aclnn_blocksparseattention_v3")
class PyAclnnBlockSparseAttentionV3Api(AclnnBaseApi):
    """按 C++ 原型逐个位置自己组装入参, 不走 ATK 的 kwargs->convert 通道。

    这么做的原因: MXFP4 的 query/key/value 缓冲区是 uint8 nibble-pack, 而算子要按
    FLOAT4_E2M1 看到两倍长的最后一维; dequantScale 又必须是 FLOAT8_E8M0。这两类
    "storage dtype 与逻辑 dtype 不一致"的入参没法靠 kwargs 的 dtype 表达, 只能自己
    建 aclTensor(与 Gmm_mxa8w4 的 aclnn 工程同一思路)。同时原型里还有 kwargs 侧不存在
    的 pQuantScale, 位置也只能显式补 nullptr。
    """

    def init_by_input_data(self, input_data: InputDataset):
        prep = self._get_prepared(input_data)
        self.prep = prep

        out_shape = _bsa_out_shape(prep)
        self.out_npu = torch.zeros(out_shape, dtype=prep.attention_out_torch_dtype,
                                   device="npu")
        out_pack = _create_acl_tensor(
            self.out_npu, TORCH_TO_ACLTYPE[str(prep.attention_out_torch_dtype)], out_shape)

        act_q = list(prep.q_seqlen_list) if prep.pass_act_seq else None
        act_kv = list(prep.kv_seqlen_list) if prep.pass_act_seq else None

        # 顺序严格对齐 docs/aclnnBlockSparseAttentionV3.md 的一段式原型
        input_args = [
            create_fp4_acl_tensor(prep.q_packed, prep.headDim),
            create_fp4_acl_tensor(prep.k_packed, prep.headDim),
            create_fp4_acl_tensor(prep.v_packed, prep.headDim),
            nnopbase.create_acl_tensor(
                prep.block_sparse_mask_tensor.contiguous().npu(), AclFormat.ACL_FORMAT_ND),
            TensorPtr(),                     # attenMaskOptional: 仅支持 nullptr
            create_int64_array(list(prep.block_shape)),
            create_int64_array(act_q),
            create_int64_array(act_kv),
            TensorPtr(),                     # blockTableOptional: 仅支持 nullptr
            create_e8m0_acl_tensor(prep.q_scale),
            create_e8m0_acl_tensor(prep.k_scale),
            create_e8m0_acl_tensor(prep.v_scale),
            TensorPtr(),                     # pQuantScaleOptional
            ctypes.c_char_p(prep.q_input_layout.encode("utf-8")),
            ctypes.c_char_p(prep.kv_input_layout.encode("utf-8")),
            Int64(prep.kv_heads),
            Int64(0),                        # maskType: 仅支持 0
            ctypes.c_double(prep.scale_value),
            Int64(prep.inner_precise),
            Int64(0),                        # blockSize: 仅支持 0
            Int64(_NOT_LIMITED),             # preTokens
            Int64(_NOT_LIMITED),             # nextTokens
            Int64(0),                        # softmaxLseFlag: MXFP4 不支持 lse
            Int64(prep.quant_mode),
            ctypes.c_double(prep.dst_type_max),
            out_pack,
            TensorPtr(),                     # softmaxLseOptional
        ]

        logging.info(
            f"pyaclnn BSA V3 MXFP4 | layout={prep.q_input_layout} "
            f"q={tuple(prep.q_packed.shape)} k={tuple(prep.k_packed.shape)} "
            f"out={out_shape} quantMode={prep.quant_mode} dstTypeMax={prep.dst_type_max} "
            f"actSeq={'on' if prep.pass_act_seq else 'off'}")

        torch.npu.synchronize()
        return input_args, [out_pack]

    def _get_prepared(self, input_data: InputDataset):
        """拿到造好 MXFP4 数据的 golden 侧对象。

        造数只由用例参数派生(case_derived_seed), 所以 cpu 与 pyaclnn 两个节点各自造出的
        数据逐位一致。后端已经先对 api_executor 调过一次 init_by_input_data, 能复用就复用,
        省掉一遍量化打包。
        """
        cached = getattr(self.backend, "api_executor", None)
        if isinstance(cached, BlockSparseAttentionV3Api) and hasattr(cached, "q_packed"):
            return cached
        prep = BlockSparseAttentionV3Api(self.task_result)
        prep.init_by_input_data(input_data)
        return prep

    def after_call(self, output_packages):
        prep = self.prep
        out = self.out_npu.cpu().to(torch.float32)
        # BNSD/BSND 变长: kernel 不写 [actualSeqLen, S_padded), golden 侧该区间为 0
        if prep.padded_s1 is not None:
            for bi in range(prep.batch):
                act_q = prep.q_seqlen_list[bi]
                if act_q >= prep.padded_s1:
                    continue
                if prep.q_input_layout == "BNSD":
                    out[bi, :, act_q:] = 0
                else:
                    out[bi, act_q:] = 0
        return [out]

    def get_cpp_func_signature_type(self):
        return ("aclnnStatus aclnnBlockSparseAttentionV3GetWorkspaceSize("
                "const aclTensor *query, "
                "const aclTensor *key, "
                "const aclTensor *value, "
                "const aclTensor *blockSparseMaskOptional, "
                "const aclTensor *attenMaskOptional, "
                "const aclIntArray *blockShapeOptional, "
                "const aclIntArray *actualSeqLengthsOptional, "
                "const aclIntArray *actualSeqLengthsKvOptional, "
                "const aclTensor *blockTableOptional, "
                "const aclTensor *qDequantScaleOptional, "
                "const aclTensor *kDequantScaleOptional, "
                "const aclTensor *vDequantScaleOptional, "
                "const aclTensor *pQuantScaleOptional, "
                "char *qInputLayout, "
                "char *kvInputLayout, "
                "int64_t numKeyValueHeads, "
                "int64_t maskType, "
                "double scaleValue, "
                "int64_t innerPrecise, "
                "int64_t blockSize, "
                "int64_t preTokens, "
                "int64_t nextTokens, "
                "int64_t softmaxLseFlag, "
                "int64_t quantMode, "
                "double dstTypeMax, "
                "aclTensor *attentionOut, "
                "aclTensor *softmaxLseOptional, "
                "uint64_t *workspaceSize, "
                "aclOpExecutor **executor)")
