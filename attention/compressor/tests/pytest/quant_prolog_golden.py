import os
os.environ['TNG_LOG_LEVEL'] = '0'

import functools
from typing import Callable
import torch_npu
import sys
import re
import contextlib
import torchair as tng
import torch.nn as nn
import torchair
import torch, os
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger
from torch_npu.profiler import profile, schedule
from torch_npu.profiler import ProfilerActivity, tensorboard_trace_handler
import math, random, copy
import numpy as np
import torch._dynamo
import en_dtypes
from en_dtypes import *
import struct
aten = torch.ops.aten
torch._dynamo.config.suppress_errors = True
# torch.npu.config.allow_internal_format = True

import logging
from torchair import logger
os.environ["ENABLE_ACLNN"] = "false"

logger.setLevel(logging.DEBUG)

# 设置 Python 内置随机数种子
# torch.manual_seed(42)
#torch_npu.npu.set_compile_mode(jit_compile=True) (kv_cache)[0][0][0][:100]
# torch.npu.config.allow_internal_format = True

class NetworkV3(nn.Module):
    def __init__(self):
        super(NetworkV3, self).__init__()

    def forward(self, x, w_dq, w_uq_qr, w_uk, w_dkv_kr, gamma_cq, gamma_ckv,
            sin, cos, cache_index, kv_cache, kr_cache, dequant_scale_x=None, dequant_scale_w_dq=None,
            dequant_scale_w_uq_qr=None, dequant_scale_w_dkv_kr=None, quant_scale_ckv=None, 
            epsilon_cq=1e-5, epsilon_ckv=1e-5, cache_mode="PA_BSND"):

        out0, out1, out2, out3, out4 = torch_npu.npu_mla_prolog_v2(   # torch_npu
            x,w_dq,w_uq_qr,w_uk,w_dkv_kr,gamma_cq,gamma_ckv,sin,cos,cache_index,kv_cache,kr_cache,
            dequant_scale_x=dequant_scale_x, dequant_scale_w_dq=dequant_scale_w_dq,
            dequant_scale_w_uq_qr=dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr=dequant_scale_w_dkv_kr,
            quant_scale_ckv=quant_scale_ckv, rmsnorm_epsilon_cq=epsilon_cq, rmsnorm_epsilon_ckv=epsilon_ckv, 
            cache_mode=cache_mode, quant_scale_ckr=None)
        return out0, out1, out2, out3, out4

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.concatenate((-x2, x1), dim=-1)

def nz_to_nd(nz_tensor, block_size=16):
    assert nz_tensor.dim() == 2
    M, K = nz_tensor.shape

    pad_M = (block_size - M % block_size) % block_size
    pad_K = (block_size - K % block_size) % block_size
    padded = torch.nn.functional.pad(nz_tensor, (0, pad_K, 0, pad_M))

    M16, K16 = padded.shape
    out = padded.view(M16 // block_size, K16 // block_size, block_size, block_size)
    out = out.permute(0, 2, 1, 3).contiguous().view(M16, K16)

    return out[:M, :K]

def nd_to_nz(nd_tensor, block_size=16):
    assert nd_tensor.dim() == 2
    M, K = nd_tensor.shape

    # 填充到16的倍数
    pad_M = (block_size - M % block_size) % block_size
    pad_K = (block_size - K % block_size) % block_size
    padded = torch.nn.functional.pad(nd_tensor, (0, pad_K, 0, pad_M))

    M16, K16 = padded.shape
    out = padded.view(M16 // block_size, block_size, K16 // block_size, block_size)
    out = out.permute(0, 2, 1, 3).contiguous().view(M16, K16)

    # 裁掉填充，保持二维 shape
    return out[:M, :K]

# def dynamic_quant(inputdata, smooth_scale=None):
#     if len(inputdata.shape) == 3:
#         inputdata = inputdata.reshape(-1,inputdata.size(-1))
#     T = inputdata.size(0)
#     H = inputdata.size(1)
#     y = torch.zeros(T, H).to(torch.int32)
#     scale = torch.zeros(T).to(torch.float32)

#     inputdata = inputdata.reshape(T, H).to(torch.float32)
#     if smooth_scale!=None:
#         if len(smooth_scale.shape) != 2 or smooth_scale.shape[1] != H:
#             return None, None
#         smooth_scale = smooth_scale.to(torch.float32)
#         for bs_index in range(T):
#             abs_bs_tensor = torch.abs(inputdata[bs_index, :] * smooth_scale[0, :])
#             scale_bs = abs_bs_tensor.max() / 127
#             scale[bs_index] = scale_bs
#             y[bs_index:] = torch.round(inputdata[bs_index:] * smooth_scale[0, :] / scale_bs)
#     else:
#         for bs_index in range(T):
#             abs_bs_tensor = torch.abs(inputdata[bs_index, :])
#             scale_bs = abs_bs_tensor.max() / 127
#             scale[bs_index] = scale_bs
#             y[bs_index:] = torch.round(inputdata[bs_index:]/ scale_bs)
#     return y, scale

def get_dtype_range(dt):
    if "bfloat16" in str(dt):
        return -float.fromhex("0x1.FEp127"), float.fromhex("0x1.FEp127")
    if "uint4" in str(dt):
        return 0, 15
    if "int4" in str(dt):
        return -8, 7
    if "bool" in str(dt):
        return 0, 1
    if "float4_e2m1" in str(dt):
        return -float.fromhex("0x1.8p2"), float.fromhex("0x1.8p2")
    if "float4_e1m2" in str(dt):
        return -float.fromhex("0x1.Cp0"), float.fromhex("0x1.Cp0")
    if "float8_e8m0" in str(dt):
        return float.fromhex("0x1.p-127"), float.fromhex("0x1.p127")
    if "float8_e5m2" in str(dt):
        return -float.fromhex("0x1.Cp15"), float.fromhex("0x1.Cp15")
    if "float8_e4m3fn" in str(dt):
        return -float.fromhex("0x1.Cp8"), float.fromhex("0x1.Cp8")
    if "hifloat8" in str(dt):
        return -float.fromhex("0x1.p15"), float.fromhex("0x1.p15")
    if "complex32" in str(dt):
        dt = "float16"
    numpy_dtype = np.dtype(dt)
    if numpy_dtype.kind in "iu":
        numpy_info = np.iinfo(numpy_dtype)
    else:
        numpy_info = np.finfo(numpy_dtype)
    return numpy_info.min, numpy_info.max

def _mx_reshape_to_blocks(fp_array: np.ndarray, axis: int, block_size: int):
    fp_array = np.expand_dims(fp_array, axis=axis + 1)
    orig_shape = fp_array.shape
    pad = [[0, 0] for _ in range(len(orig_shape))]
    pad_size = orig_shape[axis] % block_size
    pad[axis][1] = block_size - pad_size
    if pad_size > 0:
        fp_array = np.pad(fp_array, pad, 'constant')
    padded_shape = fp_array.shape
    reshape = list(padded_shape)
    reshape[axis + 1] = block_size
    reshape[axis] = reshape[axis] // block_size
    fp_array = fp_array.reshape(reshape)
    print(f"_mx_reshape_to_blocks: {fp_array.shape}")
    return fp_array, orig_shape, padded_shape

def _mx_calculate_share_exp(fp_array: np.ndarray, scale_axis: int, mx_ele_dtype: str):
    FP32_EXPONENT_BIAS = 127
    FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
    max_norm = get_dtype_range(mx_ele_dtype)[1]
    print(f"max_norm: {max_norm}")
    ele_emax = int(np.log2(max_norm))
    print(f"ele_emax: {ele_emax}")
    fp_abs_max = np.max(np.abs(fp_array), axis=scale_axis, keepdims=True)
    res = np.floor(
        np.log2(fp_abs_max.astype(np.float32) + FP32_MIN_NORMAL * (fp_abs_max == 0))
    ) - ele_emax
    print(f"res: {res}")
    #### 
    res = res + FP32_EXPONENT_BIAS
    res[fp_abs_max == 0] = -float("inf")
    return res

def _mx_round_mantissa(fp_array: np.ndarray, round_mode: str):
    """
    For example:
    fp_array  = [-4.5, -3.5, -2.5, -2.0, -1.7, -1.5, -1.4, -0.5, -0.2, -0.0, 0.0, 0.2, 0.5, 1.4, 1.5, 1.7, 2.0, 2.5, 3.4, 4.5]
    - rint    = [-4.,  -4.,  -2.,  -2.,  -2.,  -2.,  -1.,  -0.,  -0.,  -0.,  0.,  0.,  0.,  1.,  2.,  2.,  2.,  2.,  3.,  4.]
    - nearest = [-5.,  -4.,  -3.,  -2.,  -2.,  -2.,  -1.,  -1.,  -0.,  -0.,  0.,  0.,  1.,  1.,  2.,  2.,  2.,  3.,  3.,  5.]
    - floor   = [-5.,  -4.,  -3.,  -2.,  -2.,  -2.,  -2.,  -1.,  -1.,  -0.,  0,   0.,  0.,  1.,  1.,  1.,  2.,  2.,  3.,  4.]
    - ceil    = [-4.,  -3.,  -2.,  -2.,  -1.,  -1.,  -1.,  -0.,  -0.,  -0.,  0.,  1.,  1.,  2.,  2.,  2.,  2.,  3.,  4.,  5.]
    - trunc   = [-4.,  -3.,  -2.,  -2.,  -1.,  -1.,  -1.,  -0.,  -0.,  -0.,  0.,  0.,  0.,  1.,  1.,  1.,  2.,  2.,  3.,  4.]
    """
    if round_mode in ("rint", "even"):  # tie to even(c language rint)
        fp_array = np.rint(fp_array)
    elif round_mode in ("round", "nearest"):  # tie away from zero(c language round).
        sign = np.signbit(fp_array)
        rounded_abs = np.floor(np.abs(fp_array) + np.array([0.5], dtype=fp_array.dtype))
        fp_array = np.where(sign, -rounded_abs, rounded_abs)
    elif round_mode == "floor":  # round to minus infinity(c language floor)
        fp_array = np.floor(fp_array)
    elif round_mode == "ceil":   # round to positive infinity(c language ceil)
        fp_array = np.ceil(fp_array)
    elif round_mode == "trunc":  # round to zero(c language truncation)
        fp_array = np.trunc(fp_array)
    else:
        raise Exception(f"Unrecognized round method {round_mode}")
    return fp_array

def _mx_quantize_to_element_format(fp_array: np.ndarray, share_exp: np.ndarray,
                                   mx_ele_dtype: str, round_mode: str):
    mx_dtype = str(mx_ele_dtype)
    match = re.search(r'e(\d+)m(\d+)', mx_dtype)
    if match:
        exp_bits = int(match.group(1))
        mantissa_bits = int(match.group(2))
    else:
        raise ValueError(f"mx element dtype [{mx_ele_dtype}] is not recognized.")

    #### ret = fp_array / (2 ** share_exp)
    ret = fp_array / (2 ** (share_exp - 127))
    private_exp = np.floor(np.log2(np.abs(ret.astype(np.float32)) + (ret == 0))
                              ).astype(fp_array.dtype, copy=False)
    # The minimum representable exponent
    min_exp = 0 if "float4_e1m2" in mx_dtype else -(2 ** (exp_bits - 1)) + 2
    private_exp = private_exp.clip(min=min_exp)
    # Scale up so appropriate number of bits are in the integer portion of the number
    ret = ret / (2 ** private_exp) * (2 ** mantissa_bits)
    ret = _mx_round_mantissa(ret, round_mode)
    # Undo scaling
    ret = ret / (2 ** mantissa_bits) * (2 ** private_exp)
    # Set values > max_norm to Inf if desired, else clamp them
    max_norm = get_dtype_range(mx_dtype)[1]
    np.clip(ret, a_min=-max_norm, a_max=max_norm, out=ret)
    return ret

def pad_to_even(tensor: np.ndarray, axis: int) -> np.ndarray:
    """
    在指定的 axis 上将 tensor 对齐到偶数长度（即按 2 对齐），不足补 0。

    参数:
        tensor (np.ndarray): 输入数组
        axis (int): 要对齐的轴（从 0 开始）

    返回:
        np.ndarray: 对齐后的数组
    """
    if not isinstance(tensor, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions.")

    shape = tensor.shape
    length = shape[axis]

    # 如果已经是偶数，直接返回原数组
    if length % 2 == 0:
        return tensor

    # 构造 pad_width：仅对目标 axis 补一个 0
    pad_width = [(0, 0)] * tensor.ndim
    pad_width[axis] = (0, 1)  # 在 axis 维度末尾补一个 0

    padded_tensor = np.pad(tensor, pad_width, mode='constant', constant_values=2 ** -127)
    return padded_tensor

def _mx_undo_reshape_to_blocks(fp_array: np.ndarray, axis: int,
                               orig_shape: tuple, padded_shape: tuple):
    # Undo tile reshaping
    fp_array = fp_array.reshape(padded_shape)
    # Undo padding
    if tuple(padded_shape) != tuple(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        fp_array = fp_array[tuple(slices)]
    # Remove extra dimension
    fp_array = np.squeeze(fp_array, axis=axis + 1)
    return fp_array

def interleave(tensor: np.ndarray, axis: int, n_group: int = 2) -> np.ndarray:
    if not isinstance(tensor, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions.")
    # 获取目标轴的长度
    length = tensor.shape[axis]
    # 检查是否可整除
    if length % n_group != 0:
        raise ValueError(f"Axis length ({length}) must be divisible by n_group ({n_group})")

    group_length = length // n_group  # 每组长度
    shape = list(tensor.shape)

    # 重塑形状：在目标轴后插入组维度
    new_shape = (
        shape[:axis] +
        [group_length,2] +
        shape[axis+1:])
    reshaped = tensor.reshape(new_shape)

    # 构建转置顺序：交换组维度和组内维度
    transpose_order = (
        list(range(0, axis+1)) +  # 目标轴之前的维度
        list(range(axis + 2, len(new_shape))) +
        [axis+1,])  # 后续维度

    # 执行转置
    transposed = reshaped.transpose(transpose_order)

    return transposed

from ml_dtypes import float8_e4m3fn
def dynamic_mx_quant_cq(fp_array: np.ndarray, mx_ele_dtype: str = "float4_e2m1",
                axis: int = -1, block_size: int = 32, round_mode: str = "rint") -> tuple:
    if not isinstance(fp_array, np.ndarray):
        raise RuntimeError(f"Input tensor to be quantized should be numpy array. But got {type(fp_array)}")
    if fp_array.dtype.name not in ("bfloat16", "float16", "float32"):
        raise RuntimeError(f"Dtype of input tensor to be quantized is not supported: {fp_array.dtype.name}")
    if mx_ele_dtype not in ("float4_e2m1", "float4_e1m2", "float8_e4m3fn", "float8_e5m2"):
        raise NotImplementedError(f"Not support {mx_ele_dtype} yet!")
    print(f"fp_array: {fp_array}")
    axis = len(fp_array.shape) + axis if axis < 0 else axis
    fp_array, orig_shape, padded_shape = _mx_reshape_to_blocks(fp_array, axis, block_size)
   
    share_exp = _mx_calculate_share_exp(fp_array, scale_axis=axis + 1, mx_ele_dtype=mx_ele_dtype)
    scale_emax = 2 ** (8 - 1) - 1  # 8 for E8M0
    #### 
    share_exp[(share_exp - 127) > scale_emax] = float("NaN")
    share_exp[(share_exp - 127) < -scale_emax] = -scale_emax
    ele_array = _mx_quantize_to_element_format(fp_array, share_exp, mx_ele_dtype, round_mode)
    # undo reshape
    ele_array = _mx_undo_reshape_to_blocks(ele_array, axis, orig_shape, padded_shape)
    share_exp = np.squeeze(share_exp, axis=axis + 1)
    # convert to fp8_e8m0 & fp4/fp8 dtype
    ele_dtype_np = eval(f"{mx_ele_dtype}")
    # share_exp is always float32
    ####scale_array = 2 ** share_exp
    scale_array = share_exp
    # print(f"share_exp: {share_exp}")
    # print(f"scale_array: {scale_array}")
    if ele_array.dtype.name == "bfloat16":
        ele_array = ele_array.astype("float32", copy=False)

    # NPU will cast NaN (with or without sign) to positive ZERO (sign is dropped)
    ele_array = np.nan_to_num(ele_array, nan=0.0, copy=False)
    ele_array = ele_array.astype(ele_dtype_np, copy=False)
    # Cube only supports even scales. need to pad zero.
    scale_array_pad = pad_to_even(scale_array, axis=axis)
    # print(f"scale_array_pad: {scale_array_pad.shape}")
    # print(f"scale_array_pad: {scale_array_pad}")
    result_shape = copy.deepcopy(list(scale_array_pad.shape))
    result_shape.append(2)

    result_shape[axis] = scale_array_pad.shape[axis] // 2
    # when axis is -1, do not need interleave
    if axis != (len(fp_array.shape) - 1):
        scale_array_pad = interleave(scale_array_pad, axis=axis)
    scale_array_pad = scale_array_pad.reshape(result_shape)
    scale_array = scale_array_pad.astype("int8", copy=False)
    np.set_printoptions(threshold=np.inf)
    # print(f"scale_array: {scale_array}")
    # print(f"ele_array: {ele_array}")
    return scale_array, ele_array


def dynamic_mx_quant_qn(x):
    scale_max = np.float32(448.0)
    x = x.astype('float32')
    print(f"x: {x}")
    input_mul = x
    input_abs = np.abs(input_mul)
    input_max = np.max(input_abs, axis=-1, keepdims=True)
    scale = input_max * (np.float32(1.0) / scale_max)
    print(f"input_mul: {input_mul}")
    print(f"input_max: {input_max}")
    input_scaled = input_mul / scale
    output_data = input_scaled.astype("float8_e4m3fn", copy=False)
    print(f"scale shape: {scale.shape}")
    print(f"output shape: {output_data.shape}")
    print(f"scale: {scale}")
    print(f"output: {output_data}")
    return scale, output_data

def dynamic_quant_ckv_with_amax(inputs: torch.Tensor, amax: torch.Tensor, smooth_scale: torch.Tensor = None):
    """
        三维动态量化 FP32 -> INT8（每行 scale 由外部提供 amax）
        inputs: [T, N, H] 张量
        amax: [T, N, 1] 每行 H 的最大绝对值
        smooth_scale: None 或 [1, N, H] 张量
        返回:
            y: [T, N, H] 量化结果
            scale: [T, N] 每行 scale
    """
    T, N, H = inputs.shape
    inputs = inputs.to(torch.float32)
    amax = amax.to(torch.float32).clamp(min=1e-8)  # 避免除零

    if smooth_scale is not None:
        if smooth_scale.ndim != 3 or smooth_scale.shape[1:] != (N, H):
            print(f"[ERROR] dynamic_quant got wrong input smooth_scale:{tuple(smooth_scale.shape)}, expected:(1, {N}, {H})")
            return None, None
        smooth_scale = smooth_scale.to(torch.float32)
        scaled_inputs = inputs * smooth_scale  # 广播 [T,N,H] * [1,N,H]
    else:
        scaled_inputs = inputs

    scale = amax / 127.0

    # 量化
    y = torch.round(scaled_inputs / scale)

    return y, scale


def s8_saturation(inputdata):
    inputdata = torch.where(inputdata > 127, 127, inputdata)
    inputdata = torch.where(inputdata < -128, -128, inputdata)
    return inputdata.to(torch.int8)

def dynamic_quant_without_smooth_scale(input):
    T = input.size(0)
    N = input.size(1)
    H = input.size(2)
    quant_loops = input.size(0)
    eles_with_one_scale = input.size(1) * input.size(2)
    quant_loops = input.size(0) * input.size(1)
    eles_with_one_scale = input.size(2)

    y = torch.zeros(quant_loops, eles_with_one_scale).to(torch.int32)
    scale = torch.zeros(quant_loops).to(torch.float32)
    input = input.reshape(quant_loops, eles_with_one_scale).to(torch.float32)
    max_values, _ = torch.max(torch.abs(input), dim=-1, keepdim=True)
    scale = max_values / 127
    y = torch.round(input/scale)
    y = s8_saturation(y)
    print(f"[INFO]dynamic_quant_without_smooth_scale in per_head mode")
    return y.reshape(T, N, H), scale.reshape(T, N, 1).to(torch.float64)

def dequant(input, deq_scale_q_nope, quant_scale_ckv):
    org_shape = input.size()
    quant_loops = input.size(0)
    eles_with_one_scale = input.size(1) * input.size(2)
    if len(deq_scale_q_nope.size()) == 3:  # per_token_head
        quant_loops = input.size(0) * input.size(1)
        eles_with_one_scale = input.size(2)
    input = input.reshape(quant_loops, eles_with_one_scale)
    deq_scale_q_nope = deq_scale_q_nope.reshape(quant_loops, 1)
    for quant_idx in range(quant_loops):
        input[quant_idx, :] = input[quant_idx, :] * quant_scale_ckv / (deq_scale_q_nope[quant_idx, 0])
    return input.reshape(org_shape)

def mul(input, deq_scale_q_nope, quant_scale_ckv):
    print("=============in mul===========")
    print(input.shape)
    print(deq_scale_q_nope.shape)
    print(quant_scale_ckv.shape)
    org_shape = input.size()
    quant_loops = input.size(0)
    eles_with_one_scale = input.size(1) * input.size(2)
    if len(deq_scale_q_nope.size()) == 3:  # per_token_head
        quant_loops = input.size(0) * input.size(1)
        eles_with_one_scale = input.size(2)
    input = input.reshape(quant_loops, eles_with_one_scale)
    deq_scale_q_nope = deq_scale_q_nope.reshape(quant_loops, 1)
    for quant_idx in range(quant_loops):
        input[quant_idx, :] = input[quant_idx, :] * quant_scale_ckv / (deq_scale_q_nope[quant_idx, 0])
    return input.reshape(org_shape)

def s9_saturation(inputdata):
    inputdata = torch.where(inputdata > 255, 255, inputdata)
    inputdata = torch.where(inputdata < -256, -256, inputdata)
    return inputdata

def quant(x, qscale):
    # 使用广播机制来避免显式的循环
    scaled_values = (x * qscale).round().to(torch.float32)
    s9_res = s9_saturation(scaled_values)
    s8_res_cal = s8_saturation(s9_res)
    return s8_res_cal

def numpy_float8_e4m3fn():
    try:
        # noinspection PyUnresolvedReferences
        from ml_dtypes import float8_e4m3fn
        return float8_e4m3fn
    except ModuleNotFoundError:
        raise RuntimeError("ml_dtypes is needed to support float8_e4m3fn dtype!!! "
                           "Please install with `pip3 install ml-dtypes`")

def quant_ckv_per_tensor(input, quant_scale_ckv):
    # cast_dtype = "float"
    # input = input.to(cast_dtype)
    # quant_scale_ckv = quant_scale_ckv.to(cast_dtype)
    scaled_value = input * quant_scale_ckv
    scaled_value = np.round(scaled_value, 8)
    scaled_value = scaled_value.astype(numpy_float8_e4m3fn(), copy=False)
    return scaled_value

# def compare_result(name, a, b, rtol=0.005, atol=2.5e-05):
#     print(f"{name}:", torch.allclose(a, b, rtol=0.05, atol=1), a.sum(), b.sum(), (a - b).abs().sum(), flush=True)
#     print(f"==========={name} shape is {a.shape}, {a.dtype}===========", flush=True)
#     print(a.reshape(-1)[:20], flush=True)
#     print(b.reshape(-1)[:20], flush=True)
#     close_mask = torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
#     match_ratio = close_mask.float().mean().item() * 100
#     print(f"{name} precision is {match_ratio}\n")

def compare_result(name, a, b, rtol=0.005, atol=2.5e-05, max_print=20):
    print(f"{name}:", torch.allclose(a, b, rtol=0.05, atol=1), a.sum(), b.sum(), (a - b).abs().sum(), flush=True)
    print(f"==========={name} shape is {a.shape}, {a.dtype}===========", flush=True)
    print("a:", a.reshape(-1)[:20], flush=True)
    print("b:", b.reshape(-1)[:20], flush=True)

    # 逐元素比较
    close_mask = torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    match_ratio = close_mask.float().mean().item() * 100
    print(f"{name} precision is {match_ratio:.3f}%\n", flush=True)

    # 展开为1维
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    diff_mask = ~close_mask.reshape(-1)
    diff_indices = torch.nonzero(diff_mask, as_tuple=False).squeeze(-1)

    if diff_indices.numel() > 0:
        print(f"❌ {name} found {diff_indices.numel()} mismatched elements (showing up to {max_print}):", flush=True)
        for i in diff_indices[:max_print]:
            diff_val = (a_flat[i] - b_flat[i]).abs().item()
            print(f"  idx {i.item():>6d}: a={a_flat[i].item():>12.4f}, b={b_flat[i].item():>12.4f}, |diff|={diff_val:.4f}", flush=True)
    else:
        print(f"✅ {name} all elements match within tolerance.", flush=True)


def trans_numpy_fp8_e8m0_to_bf16(array):
    from ml_dtypes import bfloat16
    new_array = torch.zeros_like(array).to(torch.bfloat16)
    for i in range(array.size(0)):  # 遍历行
        for j in range(array.size(1)):  # 遍历列
            new_value = ieee_754_conversion(0, int(array[i][j]), 0)
            new_array[i][j] = new_value
    return new_array


def ieee_754_conversion(sign, exponent_raw, mantissa, exp_len=8, mant_len=7):
    """ Convert binary data into the floating point value """
    sign_mult = -1 if sign == 1 else 1
    exponent = exponent_raw - (2 ** (exp_len - 1) - 1)
    mant_mult = 1
    for b in range(mant_len - 1, -1, -1):
        if mantissa & (2 ** b):
            mant_mult += 1 / (2 ** (mant_len - b))
    return sign_mult * (2 ** exponent) * mant_mult

def BF16_to_E6M2(x):
    # input Non-negative BF16
    # output E6M2
    # E6M2: 2^[-48, 15] * 1.M2, E6_offset = 48, 2^15 * 1.75 is NaN
    # E6M2_Max = 2^15 * 1.50 = 49152
    # E6M2_Min = 2^(-48)
    x = 2 ** 15 * 1.5 if (x > 2**15 * 1.5) else x       # overflow handling
    x = 2 ** (-48) if (x < 2 ** (-48)) else x           # underflow handling
    E = np.floor(np.log2(x))                            # Exponent
    E6M2 = np.round(x * 2 ** (-E + 2)) * 2 ** (-2 + E)  # Round Half tie to even or away
    return E6M2

def To_BF16(x):
    x = np.float64(x)
    tmp = np.abs(x)
    E = np.floor(np.log2(tmp + 2**(-1000)))
    # E[E < -126] = -126
    # E = -126 if E < -126 else E
    np.where(E < -126, -126, E)
    res = np.round(x*2**(-E + 7))*2**(E - 7)
    return res

def combine_to_32bit(data_a, data_b, data_c):
    """
    将 8bit, 8bit, 16bit 数据打包成 32bit 整数
    :param data_a: int (0-255)
    :param data_b: int (0-255)
    :param data_c: int (0-65535)
    :return: int (32-bit signed or unsigned)
    """
    # 1. 验证范围 (可选，但推荐)
    assert 0 <= data_a < 256, "data_a 超出 8-bit 范围"
    assert 0 <= data_b < 256, "data_b 超出 8-bit 范围"
    assert 0 <= data_c < 65536, "data_c 超出 16-bit 范围"

    # 2. 位移并组合
    # data_a 移到最高 8 位 (31-24)
    # data_b 移到中间 8 位 (23-16)
    # data_c 保持在最低 16 位 (15-0)
    packed = (data_a << 24) | (data_b << 16) | data_c
    
    return packed

def float_to_E6M2_bits(x: float) -> int:
    # 常量定义
    MAX_VAL = 2.0 ** 15 * 1.5          # 49152.0
    MIN_VAL = 2.0 ** (-48)              # 约 3.55e-15
    NAN_VAL = 2.0 ** 15 * 1.75          # 57344.0 (保留为 NaN)
    # 处理特殊输入
    if math.isnan(x):
        return 0xFF
    if math.isinf(x):
        x = MAX_VAL if x > 0 else MIN_VAL   # 正无穷钳位到最大值，负无穷钳位到最小值
    if x < 0:
        x = MIN_VAL                          # 负数钳位到最小值（E6M2 无符号）
    # 钳位到可表示范围
    if x > MAX_VAL:
        x = MAX_VAL
    if x < MIN_VAL:
        x = MIN_VAL
    # 如果恰好等于 NaN 保留值（考虑浮点误差）
    if math.isclose(x, NAN_VAL, rel_tol=1e-12, abs_tol=1e-12):
        return 0xFF
    # 使用 frexp 分解：x = m * 2**e, 0.5 ≤ m < 1
    m, e = math.frexp(x)
    # 调整到 [1,2) 尾数，实际指数 exp = e-1
    f = m * 2.0          # 尾数部分（1 ≤ f < 2）
    exp = e - 1
    # 计算缩放因子 Q = f * 4，得到 [4,8) 的值
    Q = f * 4.0
    # 舍入到最近整数（平局时取偶数）
    R = int(round(Q))
    # 处理因浮点误差导致的边界溢出
    if R < 4:
        R = 4
    if R > 8:
        R = 8
    # 根据 R 确定实际指数和尾数编码
    if R == 8:
        # 进位：指数加 1，尾数为 0
        exp += 1
        m_val = 0
    else:
        m_val = R - 4    # 尾数编码：0,1,2,3
    # 计算存储指数（偏移 48）
    exp_storage = exp + 48
    # 组合编码：6 位指数 | 2 位尾数
    code = (exp_storage << 2) | m_val
    return code

def To_HiF4_reverse_order(x, is_low, is_low_low, N = 4, G = 64):
    '''
    一阶量化: is_low=1, is_low_low=1
    二阶量化: is_low=0, is_low_low=1
    三阶量化: is_low=0, is_low_low=0
    '''
    x = np.array(x)
    Mi, Ni = x.shape[0], x.shape[1]
    Ncnt = np.ceil(Ni/G).astype(int)
    x_hif4   = np.zeros((Mi, Ni))
    dequant_scale_arr = np.zeros((Mi, Ncnt), dtype=int)
    Ng = N - 2
    for i in range(Mi): # 本示例中Mi=1
        for j in range(Ncnt): # 本示例中Ni=1
            ori = x[i,j*G:j*G+G] # 一个Group，N方向64个数
            S = np.ones(G)  # Group的sign
            S[ori < 0] = -1

            tmpG = np.abs(ori)
            V16 = np.zeros(16)
            for k in range(16): # 对Group每4个数求最大值
                V16[k] = np.max(tmpG[k*4:k*4+4])
            # 本示例V16 [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]
            V8 = np.zeros(8)
            for k in range(8): # 对V16每2个数求最大值
                V8[k] = np.max(V16[k*2:k*2+2])
            # 本示例V8  [7, 15, 23, 31, 39, 47, 55, 63]
            Vmax = np.max(V8) # Group的最大值。本示例Vmax=63

            Const_rec = To_BF16(1 / 7.0) # 1.75是S1P2的最大值；7=4*1.75，这里的4就是Eb=1、Ec=1时的2^(Eb+Ec)
            if is_low : Const_rec = To_BF16(1 / 3.5)
            if is_low_low : Const_rec = To_BF16(1 / 1.75)
            VMax4Ea = To_BF16(Vmax * Const_rec) # 刨除HiF4*2^(Eb+Ec)后Ea要承担的数值。本示例VMax4Ea=9.0
            E6M2 = BF16_to_E6M2(VMax4Ea) # BF16_to_E6M2输出的仍然是numpy.float64，作为Ea在数学域上的表达。本示例E6M2=8.0
            E6M2_bits = float_to_E6M2_bits(VMax4Ea) # float_to_E6M2_bits输出的是int，Ea存储时真正的bit值。本示例E6M2_bits=0b11001100->2^(51-48)*(1+0+0)=8
            '''本示例E6M2_bits=204->(11001100)2'''
            REC_E6M2 = To_BF16(1 / E6M2) # 本示例REC_E6M2=0.125
            E1_8 = (V8 * REC_E6M2) >= 4 # 如果V8的值较大，则不仅需要Ea还需要叠加Eb=1、Ec=1
            if is_low : E1_8 = (V8 * REC_E6M2) >= 2
            if is_low_low : E1_8 = np.zeros(8)
            E1_8 = E1_8.astype(int).astype(float)
            '''本示例E1_8 [0, 0, 0, 0, 1, 1, 1, 1]'''

            E1_8x2 = np.repeat(E1_8, 2)
            # 本示例E1_8x2[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
            E1_16 = (V16 * REC_E6M2 * 2 ** (-E1_8x2)) >= 2 # 判断Ea是否需要叠加Eb=1。PS：EbEc只有三种组合{00, 10, 11}
            if is_low : E1_16 = np.zeros(16)
            E1_16 = E1_16.astype(int).astype(float)
            '''本示例E1_16 [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]'''

            DE16 = E1_16 + E1_8x2 # 本示例[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
            DE64 = np.repeat(DE16, 4)
            # group_hif4 = tmpG * REC_E6M2 * (2 ** (-DE64))，下面是该式子的变种
            group_hif4 = np.floor(tmpG.to(torch.bfloat16) * REC_E6M2.to(torch.bfloat16) * 2 ** (-DE64 + Ng) + 0.5) * 2.0 ** (-Ng)
            '''
            [0.   0.25 0.25 0.5  0.5  0.75 0.75 1.   1.   1.25 1.25 1.5  1.5  1.75
             1.75 2.   1.   1.   1.25 1.25 1.25 1.25 1.5  1.5  1.5  1.5  1.75 1.75
             1.75 1.75 2.   2.   1.   1.   1.   1.   1.25 1.25 1.25 1.25 1.25 1.25
             1.25 1.25 1.5  1.5  1.5  1.5  1.5  1.5  1.5  1.5  1.75 1.75 1.75 1.75
             1.75 1.75 1.75 1.75 2.   2.   2.   2.  ]
            '''
            # 得到2的原因来自于E6M2的精度损失和np.floor向上取整
            group_hif4[group_hif4 >= 2] = 2 - 2**(-Ng)       # Overflow Handling
            '''
            [0.   0.25 0.25 0.5  0.5  0.75 0.75 1.   1.   1.25 1.25 1.5  1.5  1.75
             1.75 1.75 1.   1.   1.25 1.25 1.25 1.25 1.5  1.5  1.5  1.5  1.75 1.75
             1.75 1.75 1.75 1.75 1.   1.   1.   1.   1.25 1.25 1.25 1.25 1.25 1.25
             1.25 1.25 1.5  1.5  1.5  1.5  1.5  1.5  1.5  1.5  1.75 1.75 1.75 1.75
             1.75 1.75 1.75 1.75 1.75 1.75 1.75 1.75]
            '''
            dequant_scale = combine_to_32bit(E6M2_bits, E1_8[::-1], E1_16[::-1])  # 得到真正的32bit的反量化参数，本示例dequant_scale=(int)4293980364
            '''1111111111110000 11110000 11001100'''
            x_hif4[i,j*G:j*G+G] = group_hif4 * S
            dequant_scale_arr[i, j] = dequant_scale # 
    return x_hif4, dequant_scale_arr

def split_32bit(combined: int):
    bits16_val = combined >> 16  & 0xFFFF
    bits8_val =  combined >> 8   & 0xFF
    e6m2_code =  combined        & 0xFF
    # 将 8 位整数转换为高位在前的 0/1 列表
    bits8_list = [(bits8_val >> (7 - i)) & 1 for i in range(8)]
    # 将 16 位整数转换为高位在前的 0/1 列表
    bits16_list = [(bits16_val >> (15 - i)) & 1 for i in range(16)]
    # 将 8 位整数转换为高位在前的 0/1 列表
    e6m2_bit = [(e6m2_code >> (7 - i)) & 1 for i in range(8)]
    return e6m2_bit, bits8_list, bits16_list

def E6M2_bits_to_float(code: int) -> float:
    if code == 0xFF:
        return float('nan')
    exp_storage = int(code >> 2)          # 6 位指数存储值 (0~63)
    m_val = code & 0x3                # 2 位尾数编码 (0~3)
    # 实际指数 = 存储指数 - 48
    exp = exp_storage - 48
    # 尾数值 = 1 + m_val / 4
    mantissa = 1.0 + m_val / 4.0
    return mantissa * (2.0 ** exp)

def HiFP4_To_BF16_P_reverse_order(in_tensor_hifp4, GHifp4Scale, G = 64):
    Mi = in_tensor_hifp4.shape[0]
    Ni = in_tensor_hifp4.shape[1]
    Ncnt = np.ceil(Ni/G).astype(int)
    x_bf16 = np.zeros((Mi, Ni))
    for i in range(Mi):
        for j in range(Ncnt):
            group_hif4 = in_tensor_hifp4[i,j*G : (j+1)*G]
            e6m2_bit, bits8, bits16 = split_32bit(GHifp4Scale[i][j])
            '''e6m2_bit=204->(11001100)2'''
            e6m2_float = E6M2_bits_to_float(e6m2_bit) # 本示例e6m2_float=8.0
            E1_8 = bits8[::-1]
            '''[0, 0, 0, 0, 1, 1, 1, 1]'''
            E1_8x2 = np.repeat(E1_8, 2)
            E1_16 = bits16[::-1]
            '''[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]'''
            DE16 = E1_16 + E1_8x2 # 本示例DE16=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
            DE64 = np.repeat(DE16, 4)
            dequant_scale64 = e6m2_float * (2.0 ** DE64)
            '''
            [ 8.  8.  8.  8.  8.  8.  8.  8.  8.  8.  8.  8.  8.  8.  8.  8. 16. 16.
             16. 16. 16. 16. 16. 16. 16. 16. 16. 16. 16. 16. 16. 16. 32. 32. 32. 32.
             32. 32. 32. 32. 32. 32. 32. 32. 32. 32. 32. 32. 32. 32. 32. 32. 32. 32.
             32. 32. 32. 32. 32. 32. 32. 32. 32. 32.]
            '''
            group_bf16 = dequant_scale64 * group_hif4
            '''
            [  0.  -2.   2.  -4.   4.  -6.   6.  -8.   8. -10.  10. -12.  12. -14.
            14. -14.  16. -16.  20. -20.  20. -20.  24. -24.  24. -24.  28. -28.
            28. -28.  28. -28.  32. -32.  32. -32.  40. -40.  40. -40.  40. -40.
            40. -40.  48. -48.  48. -48.  48. -48.  48. -48.  56. -56.  56. -56.
            56. -56.  56. -56.  56. -56.  56. -56.]
            '''
            x_bf16[i,j*G : (j+1)*G] = group_bf16
    x_bf16 = x_bf16.to(torch.bfloat16)
    return x_bf16

def compute_cpu(vars_list):
    query_quant_mode = 3
    # vars = []
    # for t in vars_list:
    #     vars.append(t.cpu())
    #     print(t.shape)
    vars_list = [t.cpu() for t in vars_list]
    # torch.save(vars_list, "vars_list.pt")
    # vars_list = torch.load("/home/a00887832/vars_list.pt", map_location="cpu")
    (token_x, w_dq, w_uq_qr, w_uk, w_dkv_kr, gamma_cq, gamma_ckv, sin, cos, cache_index, nope_cache, rope_cache, 
        deq_scale_x, deq_scale_w_dq, deq_scale_uqqr, deq_scale_dkvkr, quant_scale_ckv) = vars_list
    
    # mm在cpu上不支持bf16计算
    t_flag = True
    T = B * S
    if len(token_x.shape) == 3:
        t_flag = False
        token_x = token_x.reshape(T, He)
        cos = cos.reshape(T, Dr)
        sin = sin.reshape(T, Dr)
        deq_scale_x = deq_scale_x.reshape(B*S, He//64)
        seq_len = S
        cache_index = cache_index.reshape(T)

    print("[INFO]========================================")
    print("[INFO]>>>>>>>>  Start to calculate  >>>>>>>>>>")
    print("[INFO]========================================")
    # 将结果反量化为fp32
    # token_x类型转换
    # token_x = token_x.to(torch.bfloat16)
    # deq_scale_x类型转换
    # print(f"deq_scale_x0: {deq_scale_x}")
    # deq_scale_x = trans_numpy_fp8_e8m0_to_bf16(deq_scale_x)
    # print(f"deq_scale_x1: {deq_scale_x}")
    x_fp32 = np.zero(token_x.shape[0],token_x.shape[1])
    xs0 = deq_scale_x.shape[0]
    xs1 = deq_scale_x.shape[1]
    grp_size = 64
    #x 反量化
    for xs0_idx in range(xs0):
        for xs1_idx in range(xs1):
            token_cur = token_x[xs0_idx:xs0_idx + 1, xs1_idx * grp_size:(xs1_idx + 1) * grp_size]
            scale_x = deq_scale_x[xs0_idx:xs0_idx + 1, xs1_idx:(xs1_idx + 1)]
            x_fp32[xs0_idx:xs0_idx + 1, xs1_idx * grp_size:(xs1_idx + 1) * grp_size] = HiFP4_To_BF16_P_reverse_order(token_cur, scale_x)
    # w_dq反量化
    w_dq = w_dq
    deq_scale_w_dq = deq_scale_w_dq
    wdq_fp32 = np.zero(w_dq.shape[0],w_dq.shape[1])
    dqs0 = deq_scale_w_dq.shape[0]
    dqs1 = deq_scale_w_dq.shape[1]
    for dqs0_idx in range(dqs0):
        for dqs1_idx in range(dqs1):
            wdq_cur = token_x[dqs0_idx:dqs0_idx + 1, dqs1 * grp_size:(dqs1 + 1) * grp_size]
            scale_w_dq = deq_scale_w_dq[dqs0_idx:dqs0_idx + 1, dqs1:(dqs1 + 1)]
            wdq_fp32[dqs0_idx:dqs0_idx + 1, dqs1 * grp_size:(dqs1 + 1) * grp_size] = HiFP4_To_BF16_P_reverse_order(wdq_cur, scale_w_dq)
    # matmul1 : token_x(B*S1,He) * w_dq (He,Hcq) -> matmul1_res(B*S1,Hcq)
    matmul1_dtype = torch.float32
    matmul1_res = torch.matmul(x_fp32.to(torch.float32), wdq_fp32.to(torch.float32)).to(matmul1_dtype)
    print(f"=========matmul1_res: {torch.matmul(token_x, w_dq).to(matmul1_dtype).sum()}==================")
    print(f"=========matmul1_res: {matmul1_res.sum()}==================")
    # matmul1后处理
    # if query_quant_mode != 3:
    #     for t_index in range(T):
    #         matmul1_res[t_index, :] = matmul1_res[t_index, :] * deq_scale_x[t_index, 0]
    #     for h_index in range(Hcq):
    #         matmul1_res[:, h_index] = matmul1_res[:, h_index] * deq_scale_w_dq[0, h_index]
    # print(f"[INFO]deq1 end. matmul1_res dtype trans to {matmul1_res.dtype}")

    ep1 = float(1e-05)
    norm1_res = matmul1_res / torch.sqrt(torch.mean(matmul1_res ** 2, dim=-1, keepdim=True) + ep1)
    norm1_res *= gamma_cq
    np.set_printoptions(precision=6, threshold=np.inf)
    print(np.get_printoptions())
    print(f"=========norm1_res: {norm1_res.sum()}==================")
    print(f"====rmsnorm ouput: {np.array(norm1_res)}, {norm1_res.shape}")

    # matmul2预处理
    # w_uq_qr类型转换
    # w_uq_qr = w_uq_qr.to(torch.bfloat16)
    # deq_scale_uqqr类型转换
    # 2^(x-127)
    # deq_scale_uqqr = trans_numpy_fp8_e8m0_to_bf16(deq_scale_uqqr)
    uqqrs0 = deq_scale_uqqr.shape[0]
    uqqrs1 = deq_scale_uqqr.shape[1]
    wuqqr_fp32 = np.zero(w_uq_qr.shape[0],w_uq_qr.shape[1])
    # grp_size = 32
    for uqqrs0_idx in range(uqqrs0):
        for uqqrs1_idx in range(uqqrs1):
            wuqqr_cur = w_uq_qr[uqqrs1_idx * grp_size:(uqqrs1_idx + 1) * grp_size, uqqrs0_idx:uqqrs0_idx + 1]
            scale_uqqr = deq_scale_uqqr[uqqrs0_idx:uqqrs0_idx + 1, uqqrs1_idx:(uqqrs1_idx + 1)]
            wuqqr_fp32[uqqrs1_idx * grp_size:(uqqrs1_idx + 1) * grp_size, uqqrs0_idx:uqqrs0_idx + 1] = HiFP4_To_BF16_P_reverse_order(wuqqr_cur,scale_uqqr)

    # deq_scale_qcqr 为 fp32 类型，norm1_res 为 hif4 类型,先进行动态量化
    norm1_res = norm1_res.to(torch.bfloat16)
    deq_scale_qcqr, norm1_res = To_HiF4_reverse_order(norm1_res,True,True)
    # norm1_res = torch.tensor(norm1_res_np.astype(np.float32)).to(torch.float8_e4m3fn)
    # deq_scale_qcqr = torch.from_numpy(deq_scale_qcqr_np)
    # deq_scale_qcqr = deq_scale_qcqr.reshape(deq_scale_qcqr.shape[0], deq_scale_qcqr.shape[1] * deq_scale_qcqr.shape[2])
    # norm1_res类型转换
    # norm1_res = norm1_res.to(torch.bfloat16)
    # deq_scale_qcqr类型转换
    # deq_scale_qcqr = trans_numpy_fp8_e8m0_to_bf16(deq_scale_qcqr)
    print(f"deq_scale_qcqr: {deq_scale_qcqr}")
    qcqrs0 = deq_scale_qcqr.shape[0]
    qcqrs1 = deq_scale_qcqr.shape[1]
    norm1_res_fp32 = np.zero(norm1_res.shape[0],norm1_res.shape[1])
    # grp_size = 32
    for qcqrs0_idx in range(qcqrs0):
        for qcqrs1_idx in range(qcqrs1):
            # # copy from
            normal_cur = norm1_res[qcqrs0_idx:qcqrs0_idx + 1, qcqrs1_idx * grp_size:(qcqrs1_idx + 1) * grp_size]
            # # broadcast
            scale_qcqr = deq_scale_qcqr[qcqrs0_idx:qcqrs0_idx + 1, qcqrs1_idx:(qcqrs1_idx + 1)]
            norm1_res_fp32[qcqrs0_idx:qcqrs0_idx + 1, qcqrs1_idx * grp_size:(qcqrs1_idx + 1) * grp_size] = HiFP4_To_BF16_P_reverse_order(normal_cur, scale_qcqr)
            # scale_qcqr = torch.full((1, grp_size), scale_qcqr.item())
            # # mul
            # normal_cur = normal_cur * scale_qcqr
            # # copy to
            # norm1_res[qcqrs0_idx:qcqrs0_idx + 1, qcqrs1_idx * grp_size:(qcqrs1_idx + 1) * grp_size] = normal_cur
    print(f"norm1_res: {norm1_res_fp32}")
    print(f"w_uq_qr: {w_uq_qr}")

    # matmul2计算
    print(f"norm1_res: {norm1_res.dtype}, {w_uq_qr.dtype}") 
    # matmul2_res = torch.matmul(norm1_res.to(torch.float32), w_uq_qr.to(torch.float32)).to(matmul2_dtype)
    matmul2_res = norm1_res_fp32
    print(f"=========matmul2_res: {matmul2_res}==================")
    # if deq_scale_uqqr.dtype == 'int8':
    #     matmul2_res = matmul2_res.to(torch.float32)
    #     for t_index in range(T):
    #         matmul2_res[t_index, :] = matmul2_res[t_index, :] * deq_scale_qcqr[t_index]
    #     for nddr_index in range(matmul2_res.shape[1]):
    #         matmul2_res[:, nddr_index] = matmul2_res[:, nddr_index] * deq_scale_uqqr[0, nddr_index]
    #     print(f"[INFO]deq2 end. matmul2_res dtype trans to {matmul2_res.dtype}")
    matmul2_res = matmul2_res.reshape(T, N, D + Dr)

    splitd1_res1 = matmul2_res[:, :, :D]  # 取前 D 维度
    splitd1_res2 = matmul2_res[:, :, D:]  # 取剩余的 Dr 维度
    # print(f"=========splitd1_res1: {splitd1_res1.sum()}==================")
    # print(f"=========splitd1_res2: {splitd1_res2.sum()}==================")

    # matmul3预处理
    splitd1_res1 = splitd1_res1.transpose(0, 1)
    out1 = torch.zeros((N, T, Hckv))
    matmul3_dtype = torch.float32
    matmul3_dtype = torch.bfloat16
    splitd1_res1 = splitd1_res1.to(torch.bfloat16)
    print(f"splitd1_res1: {splitd1_res1}")
    for n1_index in range(N):
        out1[n1_index, :, :] = torch.matmul(splitd1_res1[n1_index, :, :].to(torch.float32), w_uk[n1_index, :, :].to(torch.float32))
    out1 = out1.transpose(0, 1)
    is_post_quant = True
    deq_scale_q_nope = None
    query_nope = out1.to(torch.bfloat16)
    # out1 = out1.to(torch.bfloat16).to(torch.float32)
    # deq_scale_q_nope_np, out1_np = dynamic_mx_quant_qn(out1.numpy())
    # out1 = torch.tensor(out1_np.astype(np.float32)).to(torch.float8_e4m3fn)
    # deq_scale_q_nope = torch.from_numpy(deq_scale_q_nope_np)
    if is_post_quant:
        deq_scale_q_nope, query_nope = To_HiF4_reverse_order(out1,True,True)
    # if query_quant_mode == 3:
    #     matmul3_dtype = torch.bfloat16
    #     splitd1_res1 = splitd1_res1.to(torch.bfloat16)
    #     print(f"splitd1_res1: {splitd1_res1}")
    # else:
    #     splitd1_res1 = splitd1_res1.to(torch.bfloat16).to(torch.float32)
    #     w_uk = w_uk.to(torch.float32)
    # for n1_index in range(N):
    #     out1[n1_index, :, :] = torch.matmul(splitd1_res1[n1_index, :, :].to(torch.float32), w_uk[n1_index, :, :].to(torch.float32))
    # out1 = out1.transpose(0, 1)
    # is_post_quant = True
    # deq_scale_q_nope = None
    # if quant_scale_ckv is not None and is_post_quant and query_quant_mode != 3:
    #     out1 = out1.to(torch.bfloat16).to(torch.float32)
    #     out1, deq_scale_q_nope = dynamic_quant_without_smooth_scale(out1)
    # if quant_scale_ckv is not None and is_post_quant and query_quant_mode == 3:
    #     out1 = out1.to(torch.bfloat16).to(torch.float32)
    #     deq_scale_q_nope_np, out1_np = dynamic_mx_quant_qn(out1.numpy())
    #     out1 = torch.tensor(out1_np.astype(np.float32)).to(torch.float8_e4m3fn)
    #     deq_scale_q_nope = torch.from_numpy(deq_scale_q_nope_np)
    # out1 = out1 if t_flag else out1.reshape(B, S, N, Hckv)
    # print(f"=========query: {out1.sum()}==================")

    # rotary1 : -> splitd1_res2(B*S1,N,Dr) * cos(B*S1,Dr) * sin(B*S1,Dr) -> query_rope_mla(B,S1,N,Dr)
    expanded_cos = cos.unsqueeze(1).repeat(1, N, 1)
    expanded_sin = sin.unsqueeze(1).repeat(1, N, 1)
    q = splitd1_res2.reshape(T, N, int(Dr / 2), 2).transpose(3, 2).reshape(T, N, Dr)
    query_rope_mla = (q * expanded_cos) + (rotate_half(q) * expanded_sin)
    # if quant_scale_ckv is not None and query_quant_mode != 3:
    #     query_rope_mla = query_rope_mla.to(torch.bfloat16).to(torch.float32)
    #     query_rope_mla = dequant(query_rope_mla, deq_scale_q_nope, quant_scale_ckv)
    # elif quant_scale_ckv is not None and query_quant_mode == 3:
    #     # ....
    query_rope_mla = query_rope_mla.to(torch.bfloat16).to(torch.float32)
    query_rope_mla = mul(query_rope_mla, deq_scale_q_nope, quant_scale_ckv)
    query_rope_mla = query_rope_mla if t_flag else query_rope_mla.reshape(B, S, N, Dr)
    print(f"=========query_rope: {query_rope_mla.sum()}==================")

    ### matmul_qr ####
    matmul4_dtype = torch.float32
    # if token_x.dtype == 'int8':
    #     w_dkv_kr = w_dkv_kr.to(torch.int32)
    #     matmul4_dtype = torch.int32
    # elif query_quant_mode == 3:
        # w_dkv_kr反量化
    # w_dkv_kr = w_dkv_kr.to(torch.bfloat16)
    # deq_scale_dkvkr 类型转换
    deq_scale_dkvkr = deq_scale_dkvkr
    # deq_scale_dkvkr = trans_numpy_fp8_e8m0_to_bf16(deq_scale_dkvkr)
    dkvkrs0 = deq_scale_dkvkr.shape[0]
    dkvkrs1 = deq_scale_dkvkr.shape[1]
    w_dkv_kr_fp32 = np.zero(norm1_res.shape[0],norm1_res.shape[1])
    for dkvkrs0_idx in range(dkvkrs0):
        for dkvkrs1_idx in range(dkvkrs1):
            scale_dkvkr_cur = deq_scale_dkvkr[dkvkrs0_idx:dkvkrs0_idx + 1, dkvkrs1_idx:(dkvkrs1_idx + 1)]
            w_dkv_kr_cir = w_dkv_kr[dkvkrs1_idx * grp_size:(dkvkrs1_idx + 1) * grp_size, dkvkrs0_idx:dkvkrs0_idx + 1]
            # w_dkv_kr[dkvkrs1_idx * grp_size:(dkvkrs1_idx + 1) * grp_size, dkvkrs0_idx:dkvkrs0_idx + 1] *= scale_dkvkr
            w_dkv_kr_fp32[dkvkrs1_idx * grp_size:(dkvkrs1_idx + 1) * grp_size, dkvkrs0_idx:dkvkrs0_idx + 1] = HiFP4_To_BF16_P_reverse_order(w_dkv_kr_cir,scale_dkvkr_cur)
            

    matmul4_res = torch.matmul(x_fp32.to(torch.float32), w_dkv_kr.to(torch.float32)).to(matmul4_dtype)
    print(f"token_x: {token_x.dtype}, {w_dkv_kr.dtype}, {matmul4_dtype}")
    print(f"=========matmul4_res: {matmul4_res.sum()}==================")
    print(f"===matmul4_res: {matmul4_res}")
    # # matmul4后处理
    # if token_x.dtype == 'int8':
    #     matmul4_res = matmul4_res.to(torch.float32)
    #     for t_index in range(T):
    #         matmul4_res[t_index, :] = matmul4_res[t_index, :] * deq_scale_x[t_index, 0]
    #     for h_index in range(Hckv + Dr):
    #         matmul4_res[:, h_index] = matmul4_res[:, h_index] * deq_scale_dkvkr[0, h_index]
    #     print(f"[INFO]deq3 end. matmul4_res dtype trans to {matmul4_res.dtype}")

    # splitD2 : matmul4_res(B*S1,Hckv+Dr) -> splitd2_res1(B*S1,Hckv) & splitd2_res2(B*S1,Dr)
    splitd2_res1 = matmul4_res[:, :Hckv]  # 取前 Hckv 维度
    splitd2_res2 = matmul4_res[:, Hckv:]  # 取剩余的 Dr 维度

    ep2 = 1e-05 # float(rmsnorm_epsilon_ckv)
    norm2_res = splitd2_res1 / torch.sqrt(torch.mean(splitd2_res1 ** 2, dim=-1, keepdim=True) + ep2)
    norm2_res *= gamma_ckv
    # if nope_cache.dtype == "int8":
    #     norm2_res = quant(norm2_res, quant_scale_ckv)
    #     print(f"[INFO]quant1 end. norm2_res dtype trans to {norm2_res.dtype}")
    # elif nope_cache.dtype is torch.float8_e4m3fn:
    if is_post_quant:
        norm2_res_np = quant_ckv_per_tensor(norm2_res.numpy(), quant_scale_ckv.numpy())
        norm2_res = torch.tensor(norm2_res_np.astype(np.float32)).to(torch.float8_e4m3fn)
        print(f"norm2_res: {norm2_res}")
        print(f"[INFO]quant1 end. norm2_res dtype trans to {norm2_res.dtype}")
    
    kv_cache = copy.deepcopy(nope_cache)
    out3_shape = kv_cache.shape
    if kv_cache.dtype == "int8":
        scatter_size = 32
    elif kv_cache.dtype == "bf16":
        scatter_size = 16
        kv_cache = kv_cache.to(torch.bfloat16)
    
    B_num = out3_shape[0]
    kv_cache = kv_cache.reshape(B_num * BlockSize, Nkv, Hckv)
    for i in range(T):
        for j in range(Nkv):
            kv_cache[cache_index[i], j, :] = norm2_res[i, :]
    out3 = kv_cache.reshape(out3_shape)


    k = splitd2_res2.reshape(T, 1, int(Dr / 2), 2).transpose(3, 2).reshape(T, Dr)
    rotary2_res = (k * cos) + (rotate_half(k) * sin)

    kr_cache = copy.deepcopy(rope_cache)
    out4_shape = kr_cache.shape
    scatter_size = 16
    kr_cache = kr_cache.to(torch.bfloat16)
    B_num = out4_shape[0]
    kr_cache = kr_cache.reshape(B_num * BlockSize, Nkv, Dr)
    for i in range(T):
        for j in range(Nkv):
            kr_cache[cache_index[i], j, :] = rotary2_res[i, :]
    out4 = kr_cache.reshape(out4_shape)
    return out1, query_rope_mla, out3, out4, deq_scale_q_nope

def compute_npu(vars_list):
    token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, gamma_cq, gamma_ckv, sin, cos, cache_index, nope_cache, dequant_scale_w_uq_qr = vars_list
    # import pdb; pdb.set_trace()
    qr = torch.matmul(token_x, w_dq_cast) # C1:Batch_matmul ;self.q_a_proj(token_x)
    print(f"=========matmul1_res: {qr.sum()}==================")
    q = torch_npu.npu_rms_norm(qr, gamma_cq, 1e-5)[0]        # V1:rmsNormCq
    print(f"=========norm1_res: {q.sum()}==================")

    # q_norm = self.q_b_proj(q)            # C3:QuantBatchMatmul
    q_norm, q_scale = torch_npu.npu_dynamic_quant(q)
    q_norm = q_norm.view(B, S, -1) if len(token_x.shape) == 3 else q_norm
    print(f"=========norm1_quant_res: {q_norm.sum()}, deq_scale_qcqr: {q_scale.sum()}==================")
    q_scale = q_scale.view(-1)
    # import pdb; pdb.set_trace()
    mm3_res = torch_npu.npu_quant_matmul(q_norm.view(-1, q_norm.size(-1)), w_uq_qr_cast,
            dequant_scale_w_uq_qr.view(-1),
            pertoken_scale=q_scale,
            bias=None,
            output_dtype=torch.bfloat16)
    # print("q_norm: ", q_norm.shape, q_norm.dtype, "q_scale: ", q_scale.shape, "mm3_res: ", mm3_res.shape, flush=True)

    print(f"=========matmul2_res: {mm3_res.sum()}==================")
    # q_quant, q_scale = torch_npu.npu_dynamic_quant(q)
    qr = {'x_int8':q_norm, 'pertoken_scale':q_scale}

    mm3_res = mm3_res.view(B, S, N, D + Dr)
    q_nope, q_pe = torch.split(mm3_res, [D, Dr], dim=-1)

    # C4:BatchMatmul
    q_nope = q_nope.view(-1, N, D)
    if w_uk.shape[0] * w_uk.shape[1] <= 65535:  # 65535: max value of uint16
        q_nope = torch_npu.npu_transpose_batchmatmul(q_nope, w_uk, bias=None, scale=None,
                perm_x1=(1, 0, 2), perm_x2=(0, 1, 2), perm_y=(1, 0, 2)
                )  # (b*s, n, d)
        q_nope = q_nope.view(B, S, N, Hckv) if len(token_x.shape) == 3 else q_nope.view(B*S, N, Hckv)
    else:
        q_nope = (
            torch.matmul(q_nope.transpose(0, 1), w_uk)
            .transpose(0, 1)
            .view(B, S, N, Hckv)
        )
    
    print(f"=========query: {q_nope.sum()}==================")

    # C2:Batch_matmul
    latent_cache = torch.matmul(token_x, w_dkv_kr_cast)  # self.kv_a_proj_with_mqa(hidden_states)
    print(f"=========matmul4_res: {latent_cache.sum()}==================")
    k_nope, k_pe = torch.split(latent_cache, [Hckv, Dr], dim=-1)

    # V2
    k_nope = (torch_npu.npu_rms_norm(k_nope, gamma_ckv, 1e-5)[0].view(B, S, 1, Hckv))
    print(f"=========norm2_res: {k_nope.sum()}==================")
    cos_q, sin_q = cos.reshape(-1, 1, 1, Dr), sin.reshape(-1, 1, 1, Dr)

    # q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos_q, sin_q, position_ids)
    q_pe = q_pe.view(-1, N, 1, Dr)
    k_pe = k_pe.view(-1, 1, 1, Dr)
    print("q_pe: ", q_pe.shape, "k_pe: ", k_pe.shape, "cos_q: ", cos_q.shape, "sin_q: ", sin_q.shape, flush=True)
    # import pdb; pdb.set_trace()
    q_pe = torch_npu.npu_interleave_rope(q_pe, cos_q, sin_q)
    q_pe = q_pe.view(B, S, N, Dr) if len(token_x.shape) == 3 else q_pe.view(B*S, N, Dr)  # (B,S,N,D)
    print(f"=========query_rope: {q_pe.sum()}==================")
    k_pe = torch_npu.npu_interleave_rope(k_pe, cos_q, sin_q).view(B, -1, 1, Dr)  # (B,S,N,D)

    # if self.kv_cache_c8:
    k_nope_int8, k_nope_scale = torch_npu.npu_dynamic_quant(k_nope.view(B, S, -1, 128))
    k_nope_int8 = k_nope_int8.view(B, S, 1, -1)
    k_nope_scale = k_nope_scale.view(B, S, 1, 4)
    print(f"=========norm2_res: {k_nope_int8.sum()}, deq_scale_ckv: {k_nope_scale.sum()}==================")
    k_nope = torch.cat((k_nope_int8, k_pe.view(torch.int8), k_nope_scale.view(torch.int8)), axis=3)
    
    # print(f"nope_cache={nope_cache.shape}", flush=True)
    print(f"=========kv_cache_input: {nope_cache.sum(), nope_cache.dtype}==================")
    torch_npu.npu_scatter_nd_update_(nope_cache.view(-1, 656), cache_index.view(-1, 1), k_nope.view(-1, k_nope.shape[-1]))
    print(f"=========kv_cache: {nope_cache.sum(), nope_cache.dtype}==================")
    
    vars_list = [q_nope, q_pe, q_norm, q_scale, nope_cache]
    return [t.cpu() for t in vars_list]

def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def IsRoundOne(sign, man, truncLen):
    roundingTruncLen = 64
    # mask0 = (truncLen >= roundingTruncLen) ? 0 : 0x1ul << truncLen;
    if truncLen >= roundingTruncLen:
        mask0 = 0
    else:
        mask0 = 0x1 << truncLen
    # mask1 = (truncLen > roundingTruncLen) ? 0 : 0x1ul << (truncLen - 1);
    if (truncLen > roundingTruncLen):
        mask1 = 0
    else:
        mask1 = 0x1 << (truncLen - 1)

    mask2 = mask1 - 1

    # ROUND_TO_NEAREST
    lastBit = (man & mask0) > 0  # Last bit after conversion
    truncHighBit = (man & mask1) > 0  # Highest bit in the truncated part
    truncLeft = (man & mask2) > 0  # Truncated left part (except for the highest bit)
    return truncHighBit and (truncLeft or lastBit)

def cvt_bfloat16_to_fp4_e1m2(x):
    sRet = 0
    if x < 0.0:
        sRet = 1

    x_abs = math.fabs(x)
    # print("===format(x_abs,'x')=====",float_to_hex(x_abs))
    x = eval(float_to_hex(x_abs))
    x = x >> 16
    # print("x=========",x)

    ef = x >> 7 & 0xff
    mf = x & 0x7f
    mLenDelta = 7 - 2  #
    maxExp = 1  # max E encoding value of e1m2 is 3
    expBias = 1  # Exponent Bias value of e2m1/e1m2 is 1

    eRet = 0
    mRet = 0
    eNorm = 0
    if (ef == 0 and mf != 0):
        eNorm = ef - 127 + 1  # the exp bias of subnormal bf16 is 126
    else:
        eNorm = ef - 127  # the exp bias of bf16 is 127

    if (eNorm > (maxExp - expBias)) or ((eNorm == (maxExp - expBias)) and ((mf >> mLenDelta) == 0b11)):
        return ((sRet << 3) | 0b111)
    elif eNorm <= -(expBias):
        eRet = 0
        mf = (mf | 0x80)
        mLenDelta -= eNorm + expBias - 1
        needRound = IsRoundOne(sRet, mf, mLenDelta)  # determine if need to carry
        mRet = (mf >> mLenDelta)
        if (needRound):
            mRet += 1
    else:
        eRet = (eNorm + expBias)
        needRound = IsRoundOne(sRet, mf, mLenDelta)
        mRet = (mf >> mLenDelta)
        if (needRound):
            mRet += 1
        if (((mRet & 0b100) != 0) and (needRound)):
            eRet += 1
            mRet = 0

    if (eRet >= 1):
        eRet = 1
    elif (eRet == 0 and mRet == 0b100):
        eRet += 1
        mRet = 0

    # print("eRet=========",eRet)
    # print("mRet=========",mRet)
    return (((sRet) << 3) | ((eRet) << 2) | ((mRet) & 3))

def new_trans_np_bfloat16_tensor_to_fp4_e1m2(in_tensor):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    new_in_tensor = in_tensor.reshape(multi_shape)
    out_tensor = np.zeros(multi_shape)
    for i in range(multi_shape):
        out_tensor[i] = cvt_bfloat16_to_fp4_e1m2(new_in_tensor[i])
    out_tensor = out_tensor.astype(np.uint8)

    # 每两个fp4拼成一个uint8保存
    fp4_shape = list(shape_tensor)
    fp4_shape[-1] = fp4_shape[-1] // 2
    fp4_tensor = np.zeros(multi_shape//2).astype(np.uint8)
    for i in range(multi_shape // 2):
        # fp4_tensor[i] = (out_tensor[i*2] << 4) | out_tensor[i*2+1] # 按常规顺序保存b4
        fp4_tensor[i] = (out_tensor[i*2+1] << 4) | out_tensor[i*2] # 按两两交叉顺序保存b4，比如b4两个数：0100 0010 存为b8后为0010 0100
    fp4_tensor = fp4_tensor.reshape(fp4_shape)
    return fp4_tensor

if __name__=="__main__":
    
    # kv_cache = torch.load("kv_cache.pt", map_location="cpu")
    # kv_cache_cpu = torch.load("kv_cache_cpu.pt", map_location="cpu")
    # kr_cache = torch.load("kr_cache.pt", map_location="cpu")
    # kr_cache_cpu = torch.load("kr_cache_cpu.pt", map_location="cpu")
    # dequantScaleCkv = torch.load("dequantScaleCkv.pt", map_location="cpu")
    # dequantScaleCkv_cpu = torch.load("dequantScaleCkv_cpu.pt", map_location="cpu")
    # compare_result("kv_cache", kv_cache, kv_cache_cpu, rtol=0, atol=1)
    # compare_result("kr_cache", kr_cache, kr_cache, rtol=0.005, atol=1e-04)
    # compare_result("dequantScaleCkv", dequantScaleCkv, dequantScaleCkv_cpu, rtol=0.0078125, atol=2.5e-05)

    # import pdb;pdb.set_trace()

    B = 1
    He = 7168
    Hcq = 1536
    Hckv = 512
    N = 64
    D = 128
    Dr = 64
    Skv = 2432
    S = 1
    Nkv = 1
    BlockSize = 128
    # BlockNum = 64
    BlockNum = 76
    # T = B * S

    # torch_npu.npu.set_device(0)
    torch.manual_seed(64)
    ################ bf16 ###############
    from en_dtypes import float4_e1m2
    from ml_dtypes import bfloat16 as np_bfloat16
    choices = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 0, -0.25, -0.5, -0.75, -1, -1.25, -1.5, -1.75]
    # token_x = (torch.rand(B, S, He).to(dst_type)).cpu()
    token_x = new_trans_np_bfloat16_tensor_to_fp4_e1m2(np.random.choice(choices, size=(B, S, He)).astype(np_bfloat16))
    print(token_x)
    w_dq = (torch.rand(He, Hcq).to(torch.float4_e1m2)).cpu()    # torch.randint(0, 100, (He, Hcq)).to(torch.int8).cpu()
    w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29) #.view(96, 448, 16, 16)
    w_uq_qr = (torch.rand(Hcq, N*(D+Dr))).to(torch.float4_e1m2).cpu()
    w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
    w_uk = (torch.rand(N, D, Hckv, dtype=torch.bfloat16)).cpu()
    w_dkv_kr = (torch.rand(He,Hckv+Dr)).to(torch.float4_e1m2).cpu() # torch.rand(He,Hckv+Dr).to(torch.int8).cpu()
    w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
    gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).cpu()
    gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).cpu()
    sin = torch.rand(B, S,Dr, dtype=torch.bfloat16).cpu()
    cos = torch.rand(B, S,Dr, dtype=torch.bfloat16).cpu()
    # cache_index = torch.rand(B, S).to(torch.int64).cpu()
    cache_index = torch.randint(0, BlockNum//B, (B, S,)).to(torch.int64).cpu()
    print(f"cache_index {cache_index}")
    kv_cache = (torch.rand((1, BlockNum*BlockSize*Nkv*Hckv))).to(torch.float4_e1m2).cpu()   # , dtype=torch.bfloat16
    kv_cache = kv_cache.view(BlockNum, BlockSize, Nkv, (Hckv))
    
    nope_cache_eager = kv_cache.clone().detach()
    nope_cache = kv_cache.clone().detach()

    kr_cache = (torch.rand((1, BlockNum*BlockSize*Nkv*Dr), dtype=torch.bfloat16)).cpu() # torch.randint(0, 100, (1, BlockNum*BlockSize*Nkv*Dr)).to(torch.int8).cpu() # 
    kr_cache = kr_cache.view(BlockNum, BlockSize, Nkv, Dr)
    rope_cache_eager = kr_cache.clone().detach()

    # dequant_scale_x = torch.randint(0, 100, (1, B*S, He//32)).to(torch.int8).cpu() # torch.rand(B*S, He//32).to(torch.int8).cpu()
    dequant_scale_x = torch.randint(120, 128, (B*S, He//64)).to(torch.float).cpu()
    dequant_scale_w_dq = torch.randint(120, 128, (Hcq, He//64)).to(torch.float).cpu() # torch.rand(Hcq, He//32).to(torch.int8).cpu()
    dequant_scale_w_uq_qr = torch.randint(120, 128, (N*(D+Dr), Hcq//64)).to(torch.float).cpu() # torch.rand(N*(D+Dr), Hcq//32).to(torch.int8).cpu()
    dequant_scale_w_dkv_kr = torch.randint(120, 128, (Hckv+Dr, He//64)).to(torch.float).cpu() # torch.rand(Hckv+Dr, He//32).to(torch.int8).cpu()
    # quant_scale_ckv = torch.rand(1, Hckv, dtype=torch.float).cpu()
    quant_scale_ckv = torch.rand((1), dtype=torch.float).cpu()
    quant_scale_ckr = None # torch.rand(1, Dr, dtype=torch.float).cpu()
    cache_mode = "PA_BSND" # BNSD  PA_BSND
    query_quant_mode = 1

    # # 路径0
    # query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm = torch_npu.npu_mla_prolog_v3( # torch_npu
    #     token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, gamma_cq, gamma_ckv, sin, cos, 
    #     nope_cache_eager, rope_cache_eager, 
    #     cache_index=cache_index, dequant_scale_x=dequant_scale_x, dequant_scale_w_dq=dequant_scale_w_dq, dequant_scale_w_uq_qr=dequant_scale_w_uq_qr, 
    #     dequant_scale_w_dkv_kr=dequant_scale_w_dkv_kr, quant_scale_ckv=quant_scale_ckv, cache_mode=cache_mode,
    #     weight_quant_mode=6, kv_cache_quant_mode=0, query_quant_mode=query_quant_mode)
 
    # print(f"query: {query}")
    # print(f"dequant_scale_q_nope: {dequant_scale_q_nope}")

    # prolog result
    # prolog_result_list_path0 = [query, query_rope, kv_cache, kr_cache, dequant_scale_q_nope]
    # prolog_result_list_path0 = [t.cpu() for t in prolog_result_list_path0]
    # (query, query_rope, kv_cache, kr_cache, dequant_scale_q_nope) = prolog_result_list_path0

    # # #####################################
    vars_list = [token_x, w_dq, w_uq_qr, w_uk, w_dkv_kr, gamma_cq, gamma_ckv, sin, cos, cache_index, nope_cache_eager, rope_cache_eager, 
         dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr, quant_scale_ckv]
    q_nope_cpu, q_pe_cpu, nope_cache_output_cpu, rope_cache_output_cpu, dequant_scale_q_nope_cpu = compute_cpu(vars_list)
    # print("!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"q_nope_cpu: {q_nope_cpu}")
    print(f"dequant_scale_q_nope_cpu: {dequant_scale_q_nope_cpu}")

    #############dynamic_mx_quant_cq#######################
#     dynamic_mx_quant_input_data_np = [4.68316949e-35,1.65509605e-35,-4.92767232e-35,4.75840113e-35
# ,1.45761300e-35,-2.95284181e-35,5.79283616e-35,6.62978814e-36
# ,-7.03415820e-35,1.07205085e-35,2.19676384e-34,-8.32250000e-36
# ,-4.87830156e-37,-3.29138418e-35,-5.21919492e-36,-8.51057910e-36
# ,5.75522034e-35,-1.00810396e-34,1.86903602e-36,4.34932910e-37
# ,3.40423164e-35,-6.11257062e-36,4.70197740e-35,1.04383898e-35
# ,-2.85880226e-35,1.23379887e-34,6.34766949e-36,4.58912995e-35
# ,1.01092514e-35,-6.95892656e-36,6.11257062e-36,-5.71290254e-36
# ,-3.36485258e-37,-2.46383616e-35,-5.15336723e-35,2.55787571e-35
# ,-1.00152119e-35,5.15336723e-35,2.36274365e-36,-1.58738757e-34
# ,2.89641808e-35,3.78038983e-35,1.41435480e-34,-6.39468927e-36
# ,-5.05462571e-36,-1.24884520e-34,-1.36921582e-34,1.08145480e-35
# ,1.24132203e-35,1.15198446e-35,1.22251412e-35,-7.86170622e-35
# ,2.01009534e-36,-3.81800565e-35,-2.89465484e-37,-7.09998588e-36
# ,5.15336723e-35,-2.58608757e-36,6.09376271e-35,4.38224294e-35
# ,-6.62038418e-35,-4.81952684e-36,-2.87996116e-36,3.14092091e-35
# ,-2.09708192e-35,-5.34144633e-35,8.55906824e-38,-4.81482486e-35
# ,-8.35071187e-35,1.53284463e-35,3.64873446e-35,-6.50753673e-35
# ,5.15336723e-35,1.07205085e-35,4.04370057e-36,-2.64486229e-36
# ,-1.21311017e-35,-5.04051978e-35,2.33218079e-35,-2.26635311e-35
# ,3.79919774e-35,-8.04978531e-35,-4.20826978e-36,3.17853672e-35
# ,-8.65163842e-36,-5.68939266e-36,9.63905368e-36,3.15032486e-36
# ,-7.74885876e-35,-3.46065537e-35,7.00594633e-36,-1.15856723e-34
# ,-9.62964972e-35,2.70363701e-36,-4.38224294e-35,-1.16609040e-34
# ,9.59203390e-35,9.70488136e-35,-9.25349153e-35,-1.67390396e-35
# ,1.86574463e-34,-7.52316385e-36,-6.86488701e-36,3.02807345e-35
# ,-4.77250706e-36,-3.87442938e-35,-3.44184746e-35,-1.06828927e-34
# ,1.40471575e-36,-2.98575565e-36,-3.76158192e-36,-2.36274365e-36
# ,-1.16138842e-35,1.54224859e-35,1.89019492e-35,-6.77084746e-35
# ,-1.01092514e-35,-2.46383616e-35,-2.46383616e-35,6.70031780e-37
# ,-6.16899435e-35,-6.35707345e-35,-1.40118927e-35,7.89932204e-35
# ,-9.68607345e-36,4.23177966e-36,2.37920057e-35,8.65163842e-36
# ,-6.80846328e-35,7.52316385e-35,1.26953390e-35,-1.06264689e-35
# ,4.77250706e-36,-7.09998588e-36,-2.15115466e-36,-2.99045763e-35
# ,-3.76158192e-36,7.38210452e-36,1.02503107e-35,-4.02019068e-36
# ,-1.21311017e-35,-1.54224859e-35,6.43230509e-35,-2.46853814e-36
# ,2.36274365e-36,-8.65163842e-36,-6.72382769e-36,-3.59231074e-35
# ,-7.05296610e-37,-1.57986441e-35,1.30903051e-34,-6.20661017e-35
# ,-3.62992656e-35,9.35693503e-36,2.10648588e-35,1.96542655e-35
# ,5.50131356e-36,-2.68953107e-35,-1.87138701e-35,-3.10330509e-35
# ,3.00926554e-36,-1.30714972e-35,1.22251412e-36,5.41667797e-35
# ,-7.08970030e-38,-2.11588983e-35,3.38542373e-36,-1.71622175e-36
# ,2.23814124e-35,8.32250000e-36,-4.75840113e-35,1.21311017e-35
# ,-3.72396610e-35,-1.20370622e-34,3.10330509e-35,5.49190961e-35
# ,-3.31489407e-36,-8.55759887e-36,-5.11575141e-35,1.11907062e-35
# ,-1.29398418e-34,3.29138418e-35,-2.83999435e-35,-1.97483051e-35
# ,-2.13469774e-35,-2.33218079e-35,-2.65191526e-35,1.41999718e-35
# ,8.83971752e-35,1.84317514e-35,-4.34462712e-35,5.87747175e-36
# ,-5.00290396e-35,-2.19112147e-35,-3.47946328e-35,-7.74885876e-35
# ,-6.95892656e-35,-4.96528814e-35,4.93707627e-36,-2.49204802e-36
# ,3.49827119e-35,2.51555791e-36,5.71760452e-35,9.47918645e-35
# ,-6.80846328e-35,-1.52814266e-36,5.92449153e-36,3.93085311e-35
# ,-1.80555932e-35,9.78011300e-36,2.50145198e-35,-2.61429944e-35
# ,6.64154308e-37,-2.44502825e-36,-3.06568927e-35,9.40395481e-36
# ,7.29746893e-35,-3.31019209e-35,-4.53740819e-36,1.42822564e-36
# ,4.89005650e-35,-5.41667797e-35,8.04038136e-36,-3.99668079e-36
# ,-1.26953390e-35,1.42822564e-36,-7.33508475e-35,-8.74567797e-36
# ,-6.69561582e-35,1.80555932e-35,-8.83971752e-35,-9.82713277e-36
# ,-2.59549153e-35,8.42594351e-35,-8.53879096e-35,3.02807345e-35
# ,9.82713277e-36,-3.19734463e-35,7.10938983e-35,-3.79919774e-35
# ,-4.92767232e-35,-3.19734463e-35,-2.71539195e-36,3.96846893e-35
# ,-3.72396610e-35,-1.52344068e-35,-2.65191526e-35,1.15104407e-34
# ,-6.20661017e-36,3.27257627e-35,-1.29398418e-34,5.41667797e-35
# ,1.33536158e-35,4.34462712e-35,-1.33160000e-34,-5.41667797e-35
# ,2.30396893e-36,1.78160863e-38,-4.32581921e-35,-9.16885594e-36
# ,-9.51680226e-35,-3.93085311e-35,9.55441808e-35,-2.76476271e-35
# ,-1.27893785e-35,2.83294139e-36,-4.06250848e-35,-1.68330791e-35
# ,-1.14728249e-35,-3.83211158e-36,-2.83999435e-35,1.53284463e-35
# ,-9.45097458e-36,-1.71151977e-35,-5.52952543e-35,-1.40118927e-35
# ,-6.24422599e-35,4.55151413e-35,-1.06076610e-34,-1.09838192e-34
# ,5.52952543e-35,2.43915078e-37,-2.56257768e-36,2.42445710e-37
# ,2.24754520e-35,4.47628249e-35,-6.43230509e-35,1.66450000e-35
# ,-1.55165254e-35,6.25362995e-36,3.23496045e-35,-4.49038842e-36
# ,2.89641808e-35,3.21615254e-35,4.81952684e-36,4.60793786e-35
# ,-1.88079096e-35,3.79096928e-37,1.39178531e-35,3.61111865e-35
# ,3.89323729e-35,3.85562147e-36,-1.16609040e-35,3.34780791e-35
# ,1.39296081e-36,7.93693786e-35,-5.34144633e-35,1.79615537e-35
# ,-2.33923376e-36,1.17549435e-35,3.25376836e-35,-2.24754520e-35
# ,-1.14610699e-37,8.01216950e-35,1.91840678e-35,3.66754237e-36
# ,7.09998588e-36,-8.72687006e-35,9.82713277e-36,-4.04370057e-35
# ,-4.30230932e-36,3.62052260e-36,1.27893785e-35,1.24132203e-34
# ,3.19734463e-36,-1.10590509e-34,2.01244633e-35,-3.91204520e-35
# ,-7.29746893e-35,1.66450000e-35,1.51403672e-35,1.49522881e-35
# ,-1.02973305e-35,-3.79919774e-35,2.98575565e-36,-1.88079096e-35
# ,2.15350565e-35,-6.01853108e-35,-8.74567797e-36,-5.52482345e-37
# ,1.70211582e-35,6.84607910e-35,-6.77084746e-35,3.15972882e-35
# ,1.76324153e-36,8.99018080e-35,7.89932204e-36,-4.23177966e-37
# ,-9.85534464e-35,-1.09838192e-34,9.47918645e-35,2.59549153e-35
# ,-2.22873729e-35,-3.36661582e-35,2.31572387e-36,2.63310735e-35
# ,1.22251412e-35,4.58912995e-35,2.05946610e-35,7.34683969e-37
# ,3.57350283e-35,-6.58276836e-36,2.28516102e-35,-4.02489266e-35
# ,1.68330791e-35,-1.07581243e-34,7.42912430e-36,-8.91494916e-35
# ,5.37906215e-35,-4.21297175e-35,-8.95256498e-35,-1.79615537e-35
# ,2.73890184e-36,-8.31309605e-35,6.34766949e-36,1.19753487e-37
# ,-4.38224294e-35,4.49509040e-35,-1.17079237e-35,1.46701695e-35
# ,-1.23191808e-35,-1.69271187e-36,9.36633899e-35,-2.51555791e-36
# ,6.35707345e-35,-5.86806780e-35,-3.62933881e-37,-1.15198446e-35
# ,-4.41985876e-35,3.02807345e-35,-1.24132203e-35,-2.67424965e-37
# ,-6.39468927e-36,6.62038418e-35,2.35098870e-35,1.50463277e-36
# ,-9.87415255e-37,1.56105650e-35,1.72092373e-35,-8.53879096e-35
# ,-7.47614407e-36,2.89641808e-35,-7.00594633e-36,-1.18724929e-36
# ,3.42303955e-35,6.92131074e-35,1.79615537e-35,5.64237288e-35
# ,-9.75660311e-37,2.10648588e-34,8.80210170e-35,-1.65509605e-35
# ,1.66450000e-35,-2.68953107e-35,-8.87733334e-35,1.11436864e-35
# ,3.15972882e-35,-1.66450000e-35,1.36357345e-36,-4.92767232e-35
# ,-1.27893785e-35,2.42622034e-35,5.41667797e-35,-7.22223729e-35
# ,3.85562147e-35,1.91840678e-35,4.20826978e-36,-6.80846328e-35
# ,-6.44170904e-36,-1.43880509e-35,2.38860452e-35,1.08615678e-35
# ,1.85728107e-36,-4.66436158e-35,5.49190961e-35,2.93403390e-35
# ,-1.24132203e-35,4.25058757e-35,5.40727401e-36,1.51967910e-34
# ,-4.23177966e-35,5.87747175e-36,1.76794350e-35,-3.46065537e-35
# ,-2.99045763e-35,1.01562712e-34,-7.78647458e-35,-3.59231074e-35
# ,4.68316949e-35,-1.37673898e-34,-3.36485258e-37,-1.97106893e-34
# ,1.51215593e-34,2.63310735e-35,7.33508475e-36,3.12211300e-35
# ,-1.78675141e-35,3.64403249e-36,1.00651704e-37,-5.61886300e-36
# ,4.91356639e-36,1.14352090e-34,-2.60959746e-36,-2.06887006e-35
# ,7.89932204e-36,-1.36169266e-34,-9.36633899e-35,1.79803616e-34
# ,-8.42594351e-35,1.48582486e-35,1.06828927e-34,-6.77084746e-35
# ,-1.92781074e-35,-7.14112818e-37,5.79283616e-35,-5.14866526e-36
# ,-2.40741243e-35,-9.40395481e-36,-4.81952684e-36,-1.14352090e-34
# ,5.19098305e-35,1.69271187e-35,8.31309605e-35,2.74595480e-35
# ,1.50463277e-35,1.03913701e-35,-5.45429379e-35,4.26116702e-38
# ,5.11575141e-35,-4.81952684e-36,-8.27548023e-35,-8.12501695e-35
# ,-4.96528814e-35,1.33712482e-37,3.68635028e-35,3.78038983e-35
# ,-1.10590509e-34,1.53284463e-35,-1.48582486e-35,1.46701695e-34
# ,8.18144068e-36,8.99018080e-35,-1.13317655e-35,-1.11907062e-35
# ,-1.05324294e-35,-7.09998588e-36,-8.20024859e-35,-4.68316949e-35
# ,1.43692429e-34,2.76476271e-35,-9.34518009e-37,-3.15032486e-36
# ,-3.57350283e-36,1.35416949e-34,5.43666137e-37,-1.89959887e-35
# ,4.96528814e-35,1.85257910e-35,-2.48264407e-35,2.38860452e-35
# ,-6.15959040e-36,-1.11436864e-35,2.24519421e-36,1.31655367e-35
# ,2.42445710e-37,-5.90568362e-35,3.36661582e-35,-1.14728249e-35
# ,-1.43880509e-35,-9.55441808e-35,4.73136476e-37,7.57018362e-36
# ,1.43692429e-34,-2.42622034e-35,2.39800848e-36,-2.08767797e-35
# ,-1.38238136e-35,4.40105085e-35,-6.44170904e-36,2.93873588e-36
# ,4.11893221e-35,1.17079237e-35,-6.92131074e-35,8.65163842e-35
# ,-4.65495763e-36,1.08615678e-35,1.64569209e-35,-1.95602260e-35
# ,5.07813560e-35,2.42151836e-36,-2.33923376e-36,-9.54501413e-36
# ,-5.71760452e-35,4.53270622e-35,2.50380297e-36,-7.89932204e-36
# ,-4.70197740e-36,-2.00304237e-35,2.76476271e-35,6.43230509e-35
# ,2.83999435e-35,-1.88079096e-35,2.48264407e-35,5.14866526e-36
# ,-1.18137182e-36,8.80210170e-35,2.79767655e-36,1.15104407e-34
# ,-1.88079096e-35,1.21122938e-34,-8.12501695e-35,9.32872317e-35
# ,2.76476271e-35,-2.72714689e-35,-8.12501695e-35,2.68953107e-35
# ,-1.19900424e-36,-2.83999435e-35,1.12377260e-35,5.50131356e-36
# ,-1.58738757e-34,-1.48582486e-35,-2.65191526e-35,-2.25694915e-36
# ,-5.71760452e-35,-1.83377119e-36,4.41985876e-36,-3.78038983e-35
# ,-2.78357062e-35,-4.55151413e-35,-2.59549153e-35,5.67998870e-35
# ,5.22859887e-35,3.76158192e-35,1.77499647e-36,-5.43078390e-36
# ,-6.34766949e-36,1.68518870e-34,6.39468927e-35,3.87913136e-36
# ,7.00594633e-36,-1.26389153e-34,3.04688136e-35,-4.21297175e-35
# ,5.64237288e-37,5.36025424e-36,5.36025424e-36,-2.93403390e-35
# ,6.58276836e-35,1.74913559e-35,5.75992232e-36,2.70833898e-35
# ,9.46272952e-37,-4.70197740e-35,-1.73973164e-35,3.73807204e-36
# ,7.52316385e-36,3.76158192e-35,5.28972458e-36,-1.80555932e-35
# ,7.25985311e-35,1.10966667e-35,9.17825989e-35,-1.38238136e-35
# ,-7.86170622e-35,-3.36661582e-35,-1.22251412e-35,1.52344068e-35
# ,6.77084746e-36,2.59549153e-35,4.96528814e-35,-3.73219456e-37
# ,3.70515819e-35,-2.63310735e-35,-1.14610699e-36,2.61429944e-35
# ,-1.21311017e-35,-1.53472542e-34,-5.90568362e-35,-5.90568362e-35
# ,-2.28516102e-35,1.58926836e-35,-5.94329944e-35,-4.60793786e-35
# ,-5.45429379e-35,-1.46701695e-35,4.26939548e-35,-5.22859887e-35
# ,3.15972882e-35,-1.01092514e-35,4.41985876e-35,-2.05711511e-36
# ,-6.05614690e-35,3.12211300e-35,7.44793221e-35,3.15972882e-35
# ,-5.56714125e-35,4.91356639e-36,6.39468927e-36,-3.27257627e-35
# ,-1.42940113e-35,-1.84317514e-35,4.16125000e-36,-2.05006215e-35
# ,1.05324294e-34,6.80846328e-35,-5.33674435e-36,8.53879096e-35
# ,1.82436723e-35,1.21122938e-34,2.59549153e-35,-1.02315028e-34
# ,-8.72687006e-35,-1.07675283e-35,-2.19112147e-35,-1.42940113e-35
# ,5.49190961e-35,-1.14728249e-35,1.94661864e-35,5.91420595e-38
# ,-6.73323164e-35,-4.39634887e-36,2.44502825e-35,3.14092091e-35
# ,-9.06541243e-35,-4.11423023e-36,3.43832098e-37,5.37906215e-35
# ,2.37449859e-36,6.77084746e-35,2.27575706e-35,-7.74885876e-35
# ,4.60793786e-36,-3.36191384e-36,6.73323164e-35,1.45173552e-36
# ,-1.73032768e-35,-3.46065537e-35,3.22085452e-36,-6.01853108e-35
# ,-2.34722712e-34,1.20370622e-34,-8.23786441e-35,8.08740113e-36
# ,-4.04370057e-35,-2.95049082e-36,3.78038983e-35,1.28834181e-35
# ,8.60461865e-36,-8.65163842e-36,1.82436723e-35,7.31745233e-37
# ,-5.59535311e-36,-6.20661017e-36,-4.46687853e-36,-6.02440855e-37
# ,7.82409040e-35,2.39800848e-35,-8.08740113e-35,-4.96528814e-35
# ,-3.99668079e-36,-3.14092091e-35,4.16125000e-36,5.78343221e-36
# ,3.59701271e-36,-4.92767232e-35,3.44184746e-35,-4.58442797e-36
# ,1.90900283e-35,-5.38376413e-36,-6.20661017e-35,2.04536017e-36
# ,-8.57640678e-35,-1.41059322e-35,6.62038418e-35,-1.70211582e-35
# ,4.73959322e-35,-3.66754237e-35,3.68635028e-35,-1.84317514e-35
# ,2.22873729e-35,4.04370057e-35,2.29456497e-35,-3.40893362e-36
# ,3.53588701e-35,-4.65495763e-36,4.40105085e-35,-5.86806780e-35
# ,9.06541243e-35,-8.18144068e-36,-1.15856723e-34,-4.75840113e-35
# ,2.91522599e-35,1.54283634e-38,1.64569209e-35,-1.37297740e-35
# ,-7.82409040e-35,-7.75826272e-36,1.45197062e-34,2.06887006e-35
# ,-3.72396610e-35,9.82713277e-36,3.14092091e-35,-3.99668079e-36
# ,9.55441808e-35,3.38542373e-36,-5.52482345e-36,9.25349153e-35
# ,-5.83045198e-35,3.04688136e-35,-3.55469492e-35,-1.29398418e-34
# ,2.27575706e-35,-2.28516102e-35,3.98727684e-35,1.42822564e-36
# ,6.50753673e-35,6.31945763e-35,-3.66754237e-35,-2.17466455e-36
# ,1.17079237e-35,-7.14700565e-36,7.82409040e-35,-1.31655367e-36
# ,-1.97483051e-35,2.52025989e-35,-2.07827401e-35,1.19430226e-35
# ,1.49522881e-35,-1.73032768e-35,3.46065537e-35,1.68330791e-35
# ,-4.92767232e-35,-1.19900424e-35,-9.12183616e-36,-1.00058079e-34
# ,-6.37705685e-37,-2.98575565e-36,2.08062500e-36,5.52952543e-35
# ,-1.40471575e-36,-1.91840678e-35,2.89641808e-35,2.53906780e-35
# ,-2.15350565e-35,1.53989760e-36,-7.10938983e-35,1.24132203e-35
# ,2.85880226e-35,-1.35416949e-35,3.49827119e-35,5.00290396e-35
# ,6.65800000e-35,-2.70833898e-35,1.31655367e-35,4.11423023e-38
# ,1.01092514e-36,6.91190678e-36,-2.74595480e-35,1.75853955e-35
# ,7.63601130e-35,3.47946328e-35,2.14410170e-35,6.35707345e-35
# ,1.75853955e-35,7.56077966e-35,-5.79283616e-35,-2.05006215e-35
# ,-1.07205085e-35,-1.21875254e-34,3.10330509e-36,4.13774011e-35
# ,2.06887006e-35,-5.15336723e-35,-3.66754237e-35,9.12183616e-36
# ,-1.99363842e-35,4.17535593e-35,-8.52233404e-37,1.69271187e-35
# ,-8.35071187e-35,-9.26289548e-36,-1.50463277e-35,-1.28569695e-37
# ,1.39178531e-35,-9.02779661e-35,-9.21587571e-36,1.55165254e-36
# ,-1.83377119e-35,-1.14728249e-35,-1.93721469e-35,1.95602260e-35
# ,2.38860452e-35,3.51707910e-35,-4.20826978e-36,-3.64403249e-36
# ,2.50145198e-35,7.81703743e-37,-1.34476554e-35,5.90568362e-35
# ,-8.88673729e-36,4.75840113e-35,9.85534464e-35,8.83971752e-36
# ,-1.19618305e-34,2.11588983e-35,-1.07581243e-34,3.66754237e-36
# ,-8.32250000e-36,1.20488171e-36,1.51403672e-35,1.03443503e-35
# ,-2.29456497e-35,5.37906215e-35,-1.88079096e-35,2.27575706e-35
# ,2.21933333e-35,8.87498235e-37,3.17853672e-35,6.62038418e-35
# ,1.02032910e-35,-2.65191526e-35,-6.34766949e-36,-9.92117232e-36
# ,-1.58926836e-35,-8.65163842e-35,-7.00594633e-36,-3.79919774e-35
# ,-7.63601130e-35,-1.53284463e-35,-7.52316385e-35,1.81026130e-36
# ,-2.01244633e-35,6.43230509e-35,9.14064407e-35,-6.62978814e-36
# ,-1.29774576e-35,4.41985876e-35,3.49827119e-35,2.31337288e-35
# ,-3.25376836e-35,-1.43880509e-35,1.07581243e-34,-1.50463277e-35
# ,-1.10590509e-34,1.24132203e-34,2.68953107e-35,1.04571977e-34
# ,2.87761017e-35,1.99363842e-35,-5.49190961e-35,-4.15654802e-35
# ,-1.20488171e-36,7.03415820e-35,3.46065537e-35,-2.40741243e-35
# ,2.68894333e-37,-1.10496469e-35,1.17079237e-35,6.88369492e-35
# ,-1.50463277e-34,-8.27548023e-35,-2.22873729e-35,2.06887006e-35
# ,-4.31994174e-37,1.46701695e-34,-2.50145198e-35,-2.91522599e-35
# ,2.83999435e-35,-1.00152119e-35,1.11436864e-35,-4.04370057e-35
# ,-2.93873588e-36,5.90568362e-35,-2.40741243e-35,1.18284119e-37
# ,1.34594103e-36,-2.61429944e-35,1.46701695e-35,-1.43410311e-36
# ,-3.17853672e-35,5.36025424e-36,1.12095141e-34,3.12681497e-36
# ,-1.27893785e-35,2.31337288e-35,-3.83681356e-35,-1.00622316e-35
# ,2.24754520e-35,-9.78011300e-35,3.10330509e-35,4.18475989e-36
# ,1.94661864e-35,-1.10496469e-35,1.08333559e-34,2.37920057e-35
# ,5.83045198e-35,4.26939548e-35,2.07827401e-35,1.18489831e-35
# ,-9.32872317e-35,2.35098870e-36,-5.52952543e-35,-5.43078390e-36
# ,6.20661017e-35,-4.72078531e-35,1.35416949e-35,-6.50753673e-35
# ,-6.73323164e-35,-8.68925424e-35,2.15115466e-36,-6.15959040e-36
# ,1.82060565e-34,-8.99987862e-38,-8.61402260e-35,-1.03067345e-34
# ,3.93085311e-35,-1.59491074e-34,1.30714972e-35,-5.94329944e-35
# ,-3.91204520e-35,-1.20370622e-35,1.06529176e-37,1.91840678e-34
# ,1.70211582e-35,-6.86488701e-36,-7.22223729e-35,-4.86654661e-36
# ,6.28184181e-35,-2.46383616e-35,-2.30396893e-35,-1.71151977e-35
# ,-3.85562147e-35,4.91356639e-36,-3.00926554e-35,-7.48554803e-35
# ,1.40118927e-35,3.92615113e-36,-4.36343503e-35,-1.91017832e-37
# ,-5.60475706e-35,2.05946610e-35,-3.08449718e-35,-1.69271187e-35
# ,-1.00152119e-35,-6.24422599e-35,-2.57668362e-35,-5.30383051e-35
# ,-3.31019209e-35,-5.10164548e-36,-5.75522034e-35,-1.51403672e-35
# ,2.35098870e-35,3.36661582e-35,-3.47946328e-35,1.44820904e-35
# ,2.17231356e-35,4.46687853e-36,2.53906780e-35,-1.88079096e-35
# ,-2.82118644e-35,5.30383051e-35,3.12681497e-36,6.80846328e-35
# ,-5.57184322e-36,-3.55469492e-35,-5.98091526e-35,9.73309322e-36
# ,1.06076610e-34,-5.56714125e-35,2.22873729e-35,-9.16885594e-36
# ,3.31489407e-36,-4.27953412e-38,-5.11575141e-35,-1.06828927e-34
# ,-7.63601130e-35,3.98727684e-35,9.93057628e-35,2.26635311e-35
# ,1.63628814e-35,-5.60475706e-35,4.06721045e-36,-7.25985311e-35
# ,4.75840113e-35,-3.31489407e-36,4.41985876e-36,-2.37920057e-35
# ,-7.03415820e-35,-1.06734887e-35,2.93403390e-35,2.06887006e-36
# ,-1.04854096e-35,-4.51389831e-35,-2.82118644e-36,-2.64486229e-36
# ,2.21933333e-35,-3.00926554e-35,-3.97317091e-36,5.64237288e-35
# ,9.55441808e-35,2.46383616e-35,-5.07813560e-35,-1.11342825e-34
# ,5.30383051e-35,3.72396610e-35,-2.48264407e-35,2.15350565e-35
# ,5.41667797e-35,-1.91840678e-35,1.54224859e-35,6.67680791e-36
# ,-6.84607910e-35,-5.71290254e-36,4.81482486e-35,8.91494916e-35
# ,1.09556073e-35,-1.39178531e-35,-1.07581243e-34,-3.23496045e-35
# ,-5.41667797e-35,1.16138842e-35,2.52025989e-35,3.57350283e-35
# ,-5.37906215e-35,8.95256498e-35,1.11342825e-34,-8.46355933e-36
# ,2.59549153e-35,4.96528814e-35,-3.21615254e-35,1.57986441e-35
# ,-8.55759887e-36,-5.94329944e-35,-1.65509605e-35,-7.86170622e-35
# ,-1.74913559e-35,-8.13442091e-36,4.13774011e-35,1.62688418e-35
# ,1.60995706e-34,-7.74885876e-35,9.54501413e-36,-2.99045763e-35
# ,7.25985311e-35,-4.17535593e-35,6.62038418e-35,5.50131356e-36
# ,-5.26621469e-36,2.39800848e-35,4.87830156e-37,8.42594351e-35
# ,-7.61720339e-36,4.00608475e-35,1.46936794e-36,4.15654802e-35
# ,-2.52731285e-36,-1.88079096e-35,3.85562147e-35,2.19817444e-36
# ,-2.85645127e-36,3.96846893e-35,9.06541243e-35,-6.48872882e-36
# ,-1.41059322e-35,-5.28972458e-36,5.05462571e-36,-8.38832769e-35
# ,-2.51273672e-34,9.96819209e-36,-6.31945763e-35,9.02779661e-35
# ,-1.34664633e-34,2.80237853e-35,-3.53588701e-35,2.09708192e-35
# ,-5.75992232e-36,2.26870410e-36,-4.81482486e-35,-1.14610699e-36
# ,3.08449718e-35,-1.23191808e-35,-1.09085876e-34,1.86609728e-37
# ,-2.42151836e-36,-1.36357345e-35,-2.03125424e-35,-6.05614690e-35
# ,8.32250000e-36,-3.08449718e-35,-9.92117232e-36,3.38542373e-36
# ,3.00926554e-35,2.24078611e-38,4.17535593e-35,3.14092091e-35
# ,1.95602260e-35,-9.26289548e-36,-4.40105085e-35,-5.84808440e-37
# ,-2.72714689e-35,3.72396610e-35,-5.99502119e-36,-1.27541137e-36
# ,-5.47780367e-36,5.22859887e-35,2.60301469e-34,2.70833898e-35
# ,-2.99045763e-35,5.94800142e-36,-3.72396610e-35,-6.62978814e-36
# ,-2.74595480e-35,-1.10966667e-35,-8.27548023e-35,-4.19416384e-35
# ,-4.85244068e-35,4.23177966e-35,1.56105650e-35,1.38238136e-35
# ,1.98658545e-36,-1.15668644e-35,1.74537401e-34,9.21587571e-35
# ,-1.54224859e-35,-2.46383616e-35,-1.04618997e-36,-4.02489266e-35
# ,4.79601695e-35,-7.86170622e-35,-5.43078390e-36,1.68518870e-34
# ,1.99363842e-35,-4.25528955e-36,7.38210452e-36,8.08740113e-35
# ,6.84607910e-35,-4.23177966e-35,3.38542373e-35,-3.39423994e-37
# ,8.55759887e-36,1.73973164e-35,-5.11575141e-35,-7.46438913e-37
# ,4.53270622e-35,2.15350565e-35,1.37673898e-34,-7.24104520e-36
# ,7.22223729e-35,6.17134534e-37,-4.08131639e-35,-4.49509040e-35
# ,-4.73959322e-35,7.22223729e-35,1.02315028e-34,-2.12529379e-35
# ,-4.60793786e-35,-1.16138842e-35,7.74885876e-35,1.39178531e-34
# ,-5.60475706e-35,1.18489831e-35,-4.32581921e-35,2.18171752e-35
# ,2.57668362e-35,3.40423164e-35,7.71124294e-35,-1.02315028e-34
# ,2.16290961e-35,-5.52952543e-35,9.81537783e-37,-3.32900000e-35
# ,2.36979661e-35,-1.69271187e-34,1.41435480e-34,7.85230226e-36
# ,-4.60793786e-35,-1.39930848e-34,4.86654661e-36,3.36661582e-35
# ,-6.54515255e-35,-2.91522599e-35,-1.41059322e-35,-8.46355933e-36
# ,2.06887006e-35,-4.02606815e-37,-4.00608475e-35,6.24422599e-35
# ,-2.00304237e-35,-2.91522599e-35,-1.22251412e-35,-3.85562147e-35
# ,-5.64237288e-36,5.03111582e-36,-1.63393715e-36,-8.50117515e-35
# ,8.04978531e-35,8.61402260e-35,-1.26953390e-35,9.25349153e-35
# ,-1.62500339e-34,-8.41653955e-36,6.09376271e-35,-3.80860170e-36
# ,-4.55151413e-35,5.56714125e-35,-6.28184181e-35,6.69561582e-35
# ,-1.04571977e-34,-2.48264407e-35,1.26953390e-35,-1.82436723e-35
# ,2.59549153e-35,7.07177401e-35,2.05946610e-35,-6.77084746e-35
# ,4.37283899e-36,-1.16138842e-35,6.77084746e-35,4.81482486e-35
# ,-2.97164972e-35,3.45595339e-36,3.14092091e-35,4.58912995e-35
# ,2.40741243e-35,-1.74913559e-35,-1.80555932e-35,-5.60475706e-35
# ,-1.44444746e-34,-4.41985876e-35,3.23496045e-35,-1.38238136e-35
# ,9.02779661e-36,4.15654802e-35,-1.19618305e-34,-8.95256498e-35
# ,4.55504061e-37,-1.25072599e-35,-1.04383898e-35,-9.70488136e-35
# ,4.91356639e-36,-4.81952684e-37,-4.43866667e-35,7.52316385e-35
# ,2.70363701e-37,-6.62038418e-35,2.91522599e-35,2.19112147e-35
# ,-8.93375707e-36,-8.46355933e-36,3.05628531e-36,-6.77084746e-36
# ,-7.66422317e-36,-2.99045763e-35,-5.98091526e-35,-3.02807345e-35
# ,2.22873729e-35,2.05946610e-35,1.28834181e-35,-4.06250848e-35
# ,-9.85534464e-35,1.24132203e-34,-4.00608475e-35,1.43998058e-36
# ,-1.06264689e-35,-1.09838192e-34,-4.51389831e-36,2.70833898e-35
# ,1.57986441e-35,1.71151977e-35,-1.18489831e-35,-2.55787571e-35
# ,6.88369492e-35,8.87733334e-35,1.53284463e-35,-7.67362712e-35
# ,-3.62052260e-36,-2.30396893e-35,8.50117515e-35,1.75853955e-35
# ,4.47628249e-35,-1.05794492e-35,6.99654238e-35,3.70515819e-35
# ,3.66754237e-35,-3.36661582e-35,1.38120586e-36,-5.15336723e-35
# ,-4.49509040e-35,-3.47946328e-35,-5.19098305e-35,-3.59231074e-35
# ,-1.18019633e-35,-5.00290396e-35,-2.36039266e-35,-4.13774011e-36
# ,-1.76324153e-36,-2.68012712e-36,1.92781074e-35,-4.49038842e-36
# ,1.04383898e-35,5.14278778e-37,-3.29138418e-35,7.22929026e-37
# ,-2.53906780e-35,-2.87761017e-35,-7.52316385e-35,1.98423446e-35
# ,-9.70488136e-35,1.45949379e-34,4.72078531e-35,1.00152119e-35
# ,7.05296610e-36,5.37906215e-35,-6.39468927e-35,1.70211582e-35
# ,-4.13774011e-35,3.73807204e-36,-1.89959887e-35,-1.33536158e-35
# ,4.11423023e-36,-1.80555932e-35,-1.64004972e-34,-8.27548023e-36
# ,-2.84469633e-36,-1.15856723e-34,1.40471575e-36,2.11588983e-36
# ,3.87442938e-35,-4.86654661e-36,4.40105085e-35,-7.14700565e-35
# ,4.23177966e-35,-4.25058757e-35,-4.32581921e-35,2.08062500e-36
# ,1.87138701e-35,3.73807204e-36,3.25376836e-35,-4.19416384e-35
# ,9.82713277e-36,2.18171752e-34,-3.71456215e-36,-2.78357062e-35
# ,1.27835011e-37,5.30383051e-35,-4.38224294e-35,-5.45429379e-36
# ,-1.18489831e-35,-1.27893785e-35,-1.19900424e-35,1.86198305e-35
# ,1.97483051e-36,-3.79919774e-35,7.67362712e-35,-1.22251412e-35
# ,-9.40395481e-35,1.69271187e-35,3.93085311e-35,8.99018080e-35
# ,4.72078531e-35,6.39468927e-35,2.87761017e-35,6.24422599e-35
# ,1.52720226e-34,-2.48264407e-35,-2.69188206e-36,2.20992938e-35
# ,1.55165254e-35,1.06734887e-35,5.26033722e-37,4.11893221e-35
# ,-1.83377119e-36,3.76158192e-35,3.06945085e-34,7.52316385e-35
# ,5.22859887e-35,6.28184181e-35,-4.66436158e-35,-9.51680226e-35
# ,-2.50145198e-35,4.15654802e-35,8.50117515e-35,-3.31019209e-35
# ,-1.64757288e-34,-6.20661017e-35,-4.08131639e-35,2.55082274e-36
# ,-9.21587571e-35,3.68635028e-35,-1.86198305e-35,-3.38542373e-35
# ,-9.70488136e-35,-5.52952543e-35,-1.88079096e-36,-8.74567797e-36
# ,-7.85230226e-36,-4.55151413e-35,6.06555085e-36,7.38210452e-36
# ,-1.31655367e-35,-4.73959322e-35,4.13774011e-35,7.58193856e-37
# ,3.19734463e-36,-1.56105650e-35,-3.34780791e-35,-9.87415255e-36
# ,4.49038842e-36,1.62500339e-34,-5.45429379e-35,2.42622034e-35
# ,-9.59203390e-36,1.34006356e-36,-4.26939548e-35,5.94329944e-35
# ,-3.44184746e-35,5.34144633e-35,5.45429379e-35,-5.52952543e-35
# ,-6.11257062e-36,-8.01216950e-35,3.08449718e-35,7.52316385e-35
# ,6.24422599e-35,-6.84607910e-35,-1.15198446e-35,6.35707345e-35
# ,-7.07177401e-35,7.75826272e-36,1.83377119e-36,-2.80943150e-36
# ,-2.64486229e-36,-1.83377119e-35,-1.26953390e-36,-1.81308249e-34
# ,-8.34600989e-37,-6.77084746e-35,1.01562712e-34,1.35181850e-36
# ,-5.37906215e-35,6.05614690e-35,1.47642090e-35,5.79283616e-35
# ,-5.41667797e-35,6.20661017e-35,-6.20661017e-36,2.82118644e-35
# ,-6.58276836e-36,1.29774576e-35,-6.77084746e-36,-1.57234124e-34
# ,4.86654661e-36,7.28806498e-37,1.89254590e-36,2.57668362e-35
# ,7.25985311e-35,-1.57986441e-35,-5.30383051e-35,-2.80237853e-35
# ,-2.10413489e-36,-1.84317514e-35,6.48872882e-36,-1.20370622e-34
# ,-4.28820339e-35,-2.40741243e-35,6.20661017e-35,5.71290254e-36
# ,-9.25349153e-35,-7.09998588e-36,1.36357345e-35,-4.25058757e-35
# ,1.22251412e-35,5.07813560e-36,1.46701695e-35,-6.91190678e-36]
#     dynamic_mx_quant_input_data_np = np.array(dynamic_mx_quant_input_data_np).reshape(1, 1536)
#     dynamic_mx_quant_input_data_np = dynamic_mx_quant_input_data_np.astype("float32")
#     np.set_printoptions(threshold=np.inf)
#     scale_out, output = dynamic_mx_quant_cq(dynamic_mx_quant_input_data_np, "float8_e4m3fn")
#     print(f"input shape: {dynamic_mx_quant_input_data_np.shape}")
#     print(f"scale_out: {scale_out.shape}")
#     print(f"scale_out: {scale_out}")
#     print(f"output: {output.shape}")
#     print(f"outdata: {output}")


#     dynamic_mx_quant_input_data_np = [6080.000000, 6144.000000, 6688.000000, 6176.000000, 5952.000000, 6144.000000, 5248.000000, 5920.000000, 5600.000000, 6080.000000, 6112.000000, 5632.000000, 5824.000000, 6496.000000, 5600.000000, 6016.000000, 6080.000000, 6368.000000, 5888.000000, 5984.000000, 6048.000000, 6208.000000, 5536.000000, 5920.000000, 5888.000000, 6464.000000, 6144.000000, 5920.000000, 5888.000000, 6624.000000, 5696.000000, 
# 6176.000000, 6176.000000, 6144.000000, 5696.000000, 5984.000000, 6016.000000, 6112.000000, 6208.000000, 6720.000000, 6112.000000, 5920.000000, 6560.000000, 6016.000000, 5760.000000, 6368.000000, 5920.000000, 6208.000000, 6048.000000, 5984.000000, 6176.000000, 5888.000000, 6016.000000, 5760.000000, 6272.000000, 6560.000000, 6176.000000, 5984.000000, 6368.000000, 6432.000000, 6112.000000, 
# 5920.000000, 5568.000000, 6272.000000, 5760.000000, 6208.000000, 6176.000000, 5856.000000, 5888.000000, 5888.000000, 6656.000000, 5472.000000, 6048.000000, 6496.000000, 5920.000000, 5792.000000, 5920.000000, 5856.000000, 6272.000000, 5856.000000, 6080.000000, 5536.000000, 5984.000000, 6272.000000, 6176.000000, 5728.000000, 6400.000000, 5792.000000, 5568.000000, 6176.000000, 6400.000000, 
# 6080.000000, 6112.000000, 5888.000000, 6048.000000, 6240.000000, 5472.000000, 6080.000000, 5952.000000, 5664.000000, 6176.000000, 6144.000000, 5376.000000, 6336.000000, 5632.000000, 5824.000000, 6176.000000, 5856.000000, 5376.000000, 6368.000000, 6368.000000, 5376.000000, 6240.000000, 6848.000000, 6208.000000, 6016.000000, 6048.000000, 5792.000000, 6080.000000, 6528.000000, 6304.000000, 
# 6592.000000, 6400.000000, 5888.000000, 5920.000000, 5632.000000, 5536.000000, 5728.000000, 5824.000000, 6400.000000, 5856.000000, 5920.000000, 6368.000000, 6400.000000, 5728.000000, 6272.000000, 6208.000000, 5632.000000, 6304.000000, 5888.000000, 5536.000000, 6112.000000, 5664.000000, 6112.000000, 5984.000000, 6496.000000, 6304.000000, 5792.000000, 6176.000000, 6144.000000, 5536.000000, 
# 6240.000000, 6720.000000, 6400.000000, 6976.000000, 5824.000000, 6240.000000, 5920.000000, 5728.000000, 6400.000000, 6304.000000, 6176.000000, 5984.000000, 6208.000000, 6080.000000, 6368.000000, 5216.000000, 5888.000000, 6176.000000, 5728.000000, 6144.000000, 6048.000000, 6112.000000, 5856.000000, 6624.000000, 5696.000000, 5984.000000, 5728.000000, 5312.000000, 6240.000000, 6560.000000, 
# 6080.000000, 6112.000000, 6432.000000, 6336.000000, 5728.000000, 6048.000000, 5952.000000, 6240.000000, 5440.000000, 5376.000000, 5344.000000, 6016.000000, 5920.000000, 6272.000000, 6080.000000, 6336.000000, 6208.000000, 6016.000000, 5984.000000, 6176.000000, 5856.000000, 6112.000000, 6560.000000, 5952.000000, 5568.000000, 6176.000000, 6112.000000, 6656.000000, 6624.000000, 6400.000000, 
# 5120.000000, 5792.000000, 5824.000000, 6112.000000, 6400.000000, 6048.000000, 6144.000000, 6528.000000, 6016.000000, 5824.000000, 6272.000000, 5984.000000, 5952.000000, 6016.000000, 6368.000000, 6112.000000, 6496.000000, 6432.000000, 5856.000000, 5888.000000, 5408.000000, 5664.000000, 6048.000000, 5952.000000, 6688.000000, 5760.000000, 5920.000000, 6304.000000, 5760.000000, 6368.000000, 
# 6112.000000, 6400.000000, 5952.000000, 5952.000000, 5760.000000, 6176.000000, 6432.000000, 5824.000000, 6240.000000, 6048.000000, 6336.000000, 6240.000000, 5888.000000, 6336.000000, 5888.000000, 6176.000000, 5856.000000, 6080.000000, 6592.000000, 6304.000000, 5824.000000, 5760.000000, 6208.000000, 5568.000000, 6336.000000, 5728.000000, 5600.000000, 6208.000000, 5696.000000, 5920.000000, 
# 5888.000000, 6176.000000, 5856.000000, 5984.000000, 6432.000000, 5664.000000, 6240.000000, 6464.000000, 5888.000000, 5440.000000, 6016.000000, 5952.000000, 6112.000000, 6336.000000, 6176.000000, 5664.000000, 5792.000000, 6208.000000, 5952.000000, 5888.000000, 6368.000000, 6080.000000, 6336.000000, 6304.000000, 5536.000000, 6208.000000, 5952.000000, 6048.000000, 6720.000000, 5632.000000, 
# 6336.000000, 6048.000000, 6016.000000, 6304.000000, 5856.000000, 5952.000000, 5760.000000, 5600.000000, 6112.000000, 5696.000000, 6176.000000, 6240.000000, 6176.000000, 6048.000000, 5728.000000, 6208.000000, 5728.000000, 6176.000000, 5920.000000, 6304.000000, 6240.000000, 6176.000000, 6112.000000, 5856.000000, 5888.000000, 5760.000000, 6528.000000, 6240.000000, 5536.000000, 6144.000000, 
# 5664.000000, 6016.000000, 6432.000000, 6112.000000, 5792.000000, 5920.000000, 6560.000000, 5920.000000, 6016.000000, 6400.000000, 6048.000000, 6048.000000, 5824.000000, 6336.000000, 6496.000000, 5888.000000, 5728.000000, 6112.000000, 6464.000000, 5472.000000, 5728.000000, 6208.000000, 6368.000000, 5792.000000, 6048.000000, 5536.000000, 6144.000000, 5856.000000, 6112.000000, 5344.000000, 
# 6080.000000, 6208.000000, 5792.000000, 5984.000000, 6432.000000, 6112.000000, 6400.000000, 6368.000000, 5984.000000, 5824.000000, 6016.000000, 5952.000000, 4960.000000, 5376.000000, 5760.000000, 5824.000000, 6080.000000, 6176.000000, 5568.000000, 5856.000000, 5664.000000, 6208.000000, 6368.000000, 5376.000000, 5568.000000, 5504.000000, 5856.000000, 6336.000000, 6112.000000, 5920.000000, 
# 5696.000000, 6240.000000, 6208.000000, 5984.000000, 6176.000000, 5888.000000, 6560.000000, 6336.000000, 5472.000000, 5792.000000, 5600.000000, 5792.000000, 5696.000000, 6112.000000, 6080.000000, 5760.000000, 6592.000000, 6240.000000, 6336.000000, 6016.000000, 6656.000000, 6112.000000, 5536.000000, 5504.000000, 6400.000000, 5888.000000, 6048.000000, 5760.000000, 6272.000000, 5984.000000, 
# 6176.000000, 6208.000000, 6528.000000, 6080.000000, 5504.000000, 6272.000000, 5504.000000, 6144.000000, 6016.000000, 6016.000000, 6176.000000, 5504.000000, 6048.000000, 6272.000000, 6080.000000, 5728.000000, 6048.000000, 5760.000000, 6176.000000, 6272.000000, 6048.000000, 6080.000000, 6272.000000, 6208.000000, 6048.000000, 6240.000000, 5760.000000, 5888.000000, 6400.000000, 6208.000000, 
# 6080.000000, 5216.000000, 5728.000000, 6464.000000, 5728.000000, 5856.000000, 5440.000000, 5472.000000, 6240.000000, 6400.000000, 5504.000000, 6080.000000, 6176.000000, 6176.000000, 6368.000000, 5728.000000, 5600.000000, 5856.000000, 6272.000000, 6016.000000, 6272.000000, 6336.000000, 5824.000000, 5888.000000, 6112.000000, 6176.000000, 5984.000000, 5856.000000, 5600.000000, 6048.000000, 
# 5664.000000, 6400.000000, 5760.000000, 6272.000000, 6112.000000, 5824.000000, 5504.000000, 5888.000000, 5984.000000, 5888.000000, 5600.000000, 5824.000000, 6016.000000, 6560.000000, 5504.000000, 6272.000000, 6080.000000, 5984.000000, 6208.000000, 6208.000000, 6176.000000, 6272.000000, 5888.000000, 5472.000000, 6112.000000, 5664.000000, 5696.000000, 6080.000000, 5472.000000, 5728.000000, 
# 6048.000000, 3120.000000, 2832.000000, 2992.000000, 3136.000000, 3152.000000, 3072.000000, 2544.000000, 3056.000000, 2768.000000, 3120.000000, 2816.000000, 2912.000000, 2688.000000, 3456.000000, 2848.000000, 3152.000000, 2896.000000, 3056.000000, 3104.000000, 3024.000000, 2896.000000, 2992.000000, 2832.000000, 2912.000000, 2880.000000, 3296.000000, 2928.000000, 2544.000000, 3040.000000, 
# 3360.000000, 2848.000000, 3136.000000, 3104.000000, 2976.000000, 2672.000000, 3168.000000, 2800.000000, 3024.000000, 2960.000000, 3488.000000, 3056.000000, 2768.000000, 3520.000000, 3152.000000, 3024.000000, 3184.000000, 3008.000000, 2944.000000, 2992.000000, 2928.000000, 2880.000000, 2832.000000, 3072.000000, 2864.000000, 2800.000000, 3280.000000, 2944.000000, 2752.000000, 2960.000000, 
# 3232.000000, 2976.000000, 3152.000000, 2720.000000, 3168.000000, 3008.000000, 2784.000000, 3088.000000, 2960.000000, 3072.000000, 2976.000000, 3056.000000, 2640.000000, 3056.000000, 3184.000000, 2912.000000, 2976.000000, 2896.000000, 3232.000000, 3040.000000, 2832.000000, 3024.000000, 2656.000000, 2912.000000, 3424.000000, 2832.000000, 3072.000000, 3088.000000, 2816.000000, 2848.000000, 
# 3360.000000, 3296.000000, 2880.000000, 3216.000000, 2832.000000, 2944.000000, 3248.000000, 2480.000000, 3184.000000, 2992.000000, 2688.000000, 3120.000000, 2928.000000, 2464.000000, 2992.000000, 2720.000000, 2864.000000, 3152.000000, 2896.000000, 2784.000000, 3168.000000, 3152.000000, 2880.000000, 3296.000000, 3360.000000, 3072.000000, 2912.000000, 3248.000000, 2896.000000, 2912.000000, 
# 3280.000000, 2896.000000, 3232.000000, 3280.000000, 2960.000000, 2960.000000, 2976.000000, 2800.000000, 2672.000000, 2640.000000, 3088.000000, 3056.000000, 3184.000000, 3440.000000, 3056.000000, 2784.000000, 2720.000000, 3040.000000, 2720.000000, 3376.000000, 3120.000000, 3008.000000, 2800.000000, 2944.000000, 2864.000000, 2656.000000, 3424.000000, 3152.000000, 2736.000000, 2976.000000, 
# 3200.000000, 2736.000000, 2944.000000, 3200.000000, 2976.000000, 3632.000000, 2992.000000, 3168.000000, 3104.000000, 2880.000000, 3056.000000, 3120.000000, 3136.000000, 3120.000000, 2992.000000, 3024.000000, 3328.000000, 2512.000000, 2992.000000, 3248.000000, 3152.000000, 2944.000000, 3184.000000, 3040.000000, 2880.000000, 3008.000000, 2832.000000, 3088.000000, 2976.000000, 2528.000000, 
# 3072.000000, 3232.000000, 2816.000000, 2976.000000, 3328.000000, 3008.000000, 2944.000000, 2880.000000, 2976.000000, 2848.000000, 2960.000000, 2688.000000, 2368.000000, 3056.000000, 2896.000000, 3152.000000, 3264.000000, 2992.000000, 3008.000000, 3104.000000, 2800.000000, 3040.000000, 2864.000000, 2656.000000, 3040.000000, 2800.000000, 2768.000000, 3040.000000, 3104.000000, 3536.000000, 
# 3280.000000, 3184.000000, 2528.000000, 2544.000000, 3008.000000, 2816.000000, 3104.000000, 2848.000000, 2944.000000, 3392.000000, 2704.000000, 2928.000000, 3056.000000, 2960.000000, 3200.000000, 2848.000000, 3104.000000, 2720.000000, 3040.000000, 3040.000000, 2720.000000, 3088.000000, 2688.000000, 2752.000000, 2976.000000, 2848.000000, 3280.000000, 2832.000000, 2960.000000, 2928.000000, 
# 2832.000000, 2992.000000, 3104.000000, 3168.000000, 2944.000000, 2912.000000, 3072.000000, 3056.000000, 3440.000000, 2560.000000, 3216.000000, 2848.000000, 3216.000000, 2960.000000, 2784.000000, 3312.000000, 2976.000000, 2944.000000, 2992.000000, 3024.000000, 3152.000000, 3008.000000, 2944.000000, 2816.000000, 2864.000000, 2768.000000, 3296.000000, 2576.000000, 2576.000000, 3312.000000, 
# 2736.000000, 2768.000000, 2784.000000, 2784.000000, 2704.000000, 2800.000000, 3392.000000, 2848.000000, 2928.000000, 3264.000000, 3136.000000, 2432.000000, 2672.000000, 2816.000000, 2976.000000, 3120.000000, 2880.000000, 2432.000000, 2752.000000, 3136.000000, 2864.000000, 3104.000000, 2752.000000, 3104.000000, 3168.000000, 3328.000000, 2768.000000, 3040.000000, 2736.000000, 2944.000000, 
# 3248.000000, 2912.000000, 3168.000000, 2624.000000, 2672.000000, 3264.000000, 2624.000000, 3104.000000, 2848.000000, 2720.000000, 2976.000000, 2800.000000, 2880.000000, 3136.000000, 2960.000000, 2704.000000, 2624.000000, 3168.000000, 2928.000000, 2896.000000, 2960.000000, 2848.000000, 2848.000000, 3056.000000, 2800.000000, 3056.000000, 2688.000000, 2768.000000, 3344.000000, 2624.000000, 
# 2624.000000, 3072.000000, 2624.000000, 3312.000000, 3216.000000, 2816.000000, 3008.000000, 2928.000000, 3184.000000, 2816.000000, 2880.000000, 3232.000000, 3104.000000, 2784.000000, 2848.000000, 3360.000000, 3168.000000, 2928.000000, 2848.000000, 2992.000000, 3392.000000, 2736.000000, 2640.000000, 3088.000000, 3008.000000, 2656.000000, 2896.000000, 2848.000000, 3120.000000, 2784.000000, 
# 3008.000000, 2576.000000, 2992.000000, 2992.000000, 2640.000000, 2912.000000, 3216.000000, 2992.000000, 3248.000000, 3152.000000, 2848.000000, 2832.000000, 2784.000000, 2720.000000, 2336.000000, 2512.000000, 2608.000000, 2992.000000, 2848.000000, 3152.000000, 2928.000000, 3024.000000, 2944.000000, 3280.000000, 3056.000000, 2656.000000, 2384.000000, 3072.000000, 2944.000000, 3120.000000, 
# 2720.000000, 2688.000000, 2640.000000, 3104.000000, 3008.000000, 3168.000000, 2864.000000, 2704.000000, 3152.000000, 3120.000000, 2816.000000, 2816.000000, 2976.000000, 2704.000000, 2672.000000, 2960.000000, 2880.000000, 2656.000000, 3440.000000, 3296.000000, 2944.000000, 2848.000000, 3360.000000, 3088.000000, 2752.000000, 3056.000000, 2880.000000, 2944.000000, 2944.000000, 2656.000000, 
# 2992.000000, 2880.000000, 2976.000000, 2944.000000, 3136.000000, 2864.000000, 2720.000000, 3072.000000, 2752.000000, 3280.000000, 2704.000000, 2832.000000, 3008.000000, 2496.000000, 2752.000000, 3040.000000, 3088.000000, 2704.000000, 2928.000000, 2752.000000, 2864.000000, 2912.000000, 3200.000000, 3168.000000, 3344.000000, 3264.000000, 3056.000000, 2944.000000, 2944.000000, 3120.000000, 
# 3200.000000, 3136.000000, 2784.000000, 2832.000000, 2800.000000, 3152.000000, 2832.000000, 2704.000000, 2544.000000, 2880.000000, 3216.000000, 3456.000000, 2880.000000, 3136.000000, 3296.000000, 2960.000000, 3168.000000, 2736.000000, 2624.000000, 2960.000000, 2928.000000, 3008.000000, 3072.000000, 3008.000000, 2608.000000, 2896.000000, 2848.000000, 3200.000000, 2912.000000, 2912.000000, 
# 2896.000000, 2800.000000, 2784.000000, 3040.000000, 2736.000000, 3168.000000, 3200.000000, 3120.000000, 2400.000000, 2912.000000, 2880.000000, 2848.000000, 2656.000000, 3200.000000, 2976.000000, 3328.000000, 2880.000000, 3168.000000, 3104.000000, 2928.000000, 3216.000000, 2976.000000, 3184.000000, 2976.000000, 2880.000000, 2624.000000, 2928.000000, 2800.000000, 2544.000000, 2944.000000, 
# 2544.000000, 2688.000000, 2880.000000]
#     dynamic_mx_quant_input_data_np = np.array(dynamic_mx_quant_input_data_np).reshape(2, 512)
#     dynamic_mx_quant_input_data_np = dynamic_mx_quant_input_data_np.astype("float32")
#     np.set_printoptions(threshold=np.inf)
#     scale_out, output = dynamic_mx_quant_qn(dynamic_mx_quant_input_data_np)
#     print(f"input shape: {dynamic_mx_quant_input_data_np.shape}")
#     print(f"scale_out: {scale_out.shape}")
#     print(f"scale_out: {scale_out}")
#     print(f"output: {output.shape}")
#     print(f"outdata: {output}")
    # # ################ CPU ###########################
    # import pdb; pdb.set_trace()
    # print("\n       prolog<===>cpu        \n")
    # compare_result("query", query, q_nope_cpu, rtol=0.005, atol=2.5e-05)
    # compare_result("query_rope", query_rope, q_pe_cpu)
    # compare_result("kv_cache", kv_cache, nope_cache_output_cpu, rtol=0, atol=1)
    # compare_result("kr_cache", kr_cache, rope_cache_output_cpu, rtol=0.005, atol=1e-04)
    # compare_result("dequant_scale_q_nope", dequant_scale_q_nope, dequant_scale_q_nope_cpu, rtol=0.0078125, atol=2.5e-05)

    # ## 路径3
    # with torch.no_grad():
    #     config = CompilerConfig()
    #     config.debug.graph_dump.type = "pbtxt"
    #     config.debug.graph_dump.path = "./test"
    #     npu_backend = tng.get_npu_backend(compiler_config=config)
    
    #     # import torch_npu, torchair
    #     # config = torchair.CompilerConfig()
    #     # # data dump开关：[必选]
    #     # config.dump_config.enable_dump = True
    #     # # dump类型：[可选]，all代表dump所有数据
    #     # config.dump_config.dump_mode = "all"
    #     # # dump路径：[可选]，缺省为当前目录
    #     # config.dump_config.dump_path = '/home/a00887832/dump'
    #     # # 量化data dump开关：[可选]，是否采集量化前的dump数据
    #     # config.dump_config.quant_dumpable = True
    #     # # 保存dump的步数，否则每一步都会保存
    #     # config.dump_config.dump_step = "0"
    #     # 取值为算子名
    #     # config.dump_config.dump_layer = "MlaPrologV3"
    #     # npu_backend= torchair.get_npu_backend(compiler_config=config)
    #     # opt_model = torch.compile(model, backend=npu_backend)

    #     print("path:3")
    #     npu_mode_v3 = NetworkV3().npu()

    #     npu_mode_v3 = torch.compile(npu_mode_v3, fullgraph=True, backend=npu_backend, dynamic=False)

    #     query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm = npu_mode_v3(   # torch_npu
    #         token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, gamma_cq, gamma_ckv, sin, cos, kv_cache=kv_cache, kr_cache=kv_cache, # torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16).npu(), # 
    #         dequant_scale_w_uq_qr=dequant_scale_w_uq_qr, cache_index=cache_index, 
    #         cache_mode=cache_mode, query_norm_flag=True, weight_quant_mode=1, kv_cache_quant_mode=3, query_quant_mode=0, ckvkr_repo_mode=1,
    #         quant_scale_repo_mode=1, tile_size=128, k_nope_clip_alpha=torch.tensor([1], dtype=torch.float).npu(), qc_qr_scale=1, kc_scale=1)

    # # prolog result
    # prolog_result_list = [query, query_rope, query_norm, dequant_scale_q_norm.squeeze(1), kv_cache] # , kr_cache
    # prolog_result_list = [t.cpu() for t in prolog_result_list]
    # (query, query_rope, query_norm, dequant_scale_q_norm, kv_cache) = prolog_result_list

    # print("\n       prolog eager<===>prolog graph       \n")
    # kv_cache = kv_cache.reshape(-1, 656)
    # prolog_result_list_path0[4] = prolog_result_list_path0[4].reshape(-1, 656) # 
    # nope_cache = nope_cache.reshape(-1, 656).cpu()   

    # compare_result("query", query, prolog_result_list_path0[0], rtol=0.005, atol=2.5e-05)
    # compare_result("query_rope", query_rope, prolog_result_list_path0[1])
    # compare_result("query_norm", query_norm, prolog_result_list_path0[2], rtol=0, atol=1)
    # compare_result("dequant_scale_q_norm", dequant_scale_q_norm, prolog_result_list_path0[3])
    
    # compare_result("kv_cache", kv_cache[:, :512], prolog_result_list_path0[4][:, :512], rtol=0, atol=1)
    # compare_result("kr_cache", kv_cache[:, 512:640].view(torch.bfloat16), prolog_result_list_path0[4][:, 512:640].view(torch.bfloat16), rtol=0.005, atol=1e-04)
    # compare_result("dequantScaleCkv", kv_cache[:, 640:].view(torch.float32), prolog_result_list_path0[4][:, 640:].view(torch.float32), rtol=0.0078125, atol=2.5e-05)
    
    # compare_result("kv_cache_graph_input", kv_cache[:, :512], nope_cache[:, :512], rtol=0, atol=1)
    # compare_result("kr_cache_graph_input", kv_cache[:, 512:640].view(torch.bfloat16), nope_cache[:, 512:640].view(torch.bfloat16), rtol=0.005, atol=1e-04)
    # compare_result("dequantScaleCkv_graph_input", kv_cache[:, 640:].view(torch.float32), nope_cache[:, 640:].view(torch.float32), rtol=0.0078125, atol=2.5e-05)
    
    # compare_result("kv_cache_eager_input", prolog_result_list_path0[4][:, :512], nope_cache[:, :512], rtol=0, atol=1)
    # compare_result("kr_cache_eager_input", prolog_result_list_path0[4][:, 512:640].view(torch.bfloat16), nope_cache[:, 512:640].view(torch.bfloat16), rtol=0.005, atol=1e-04)
    # compare_result("dequantScaleCkv_eager_input", prolog_result_list_path0[4][:, 640:].view(torch.float32), nope_cache[:, 640:].view(torch.float32), rtol=0.0078125, atol=2.5e-05)
    # # compare_result("k_rope", kr_cache, prolog_result_list_path0[5], rtol=0, atol=1)


    # ###########################################
    # print("\n       cpu<===>npu小算子        \n")
    # compare_result("query", q_nope_cpu, q_nope, rtol=0.005, atol=2.5e-05)
    # compare_result("query_rope", query_rope, q_pe)
    # compare_result("query_norm", q_norm_cpu, q_norm, rtol=0, atol=1)
    # compare_result("dequant_scale_q_norm", q_scale_cpu, q_scale)
    # compare_result("kv_cache", nope_cache_output_cpu[:, :512], nope_cache_output[:, :512], rtol=0, atol=1)
    # compare_result("kr_cache", nope_cache_output_cpu[:, 512:640].view(torch.bfloat16), nope_cache_output[:, 512:640].view(torch.bfloat16), rtol=0.005, atol=1e-04)
    # compare_result("dequantScaleCkv", nope_cache_output_cpu[:, 640:].view(torch.float32), nope_cache_output[:, 640:].view(torch.float32), rtol=0.0078125, atol=2.5e-05)

    # #####################################
    # # npu
    # vars_list = [token_x, w_dq, w_uq_qr, w_uk, w_dkv_kr, gamma_cq, gamma_ckv, sin, cos, cache_index, nope_cache, dequant_scale_w_uq_qr]
    # q_nope, q_pe, q_norm, q_scale, nope_cache_output = compute_npu(vars_list)
    # # import pdb; pdb.set_trace()

    # # print("\n       prolog<===>dump        \n")
    # # compare_result("query", q_nope, qnope, rtol=0.005, atol=2.5e-05)
    # # # compare_result("query_rope", q_pe, )
    # # compare_result("query_norm", q_norm, qr, rtol=0, atol=1)
    # # # compare_result("kv_cache", nope_cache_output, nope_cache_after_scatter, rtol=0, atol=1)

    # # ##########################################
    # kv_cache = kv_cache.reshape(-1, 656)
    # nope_cache_output = nope_cache_output.reshape(-1, 656)
    # print("\n       prolog<===>npu小算子        \n")
    # compare_result("query", query, q_nope, rtol=0.005, atol=2.5e-05)
    # compare_result("query_rope", query_rope, q_pe)
    # compare_result("query_norm", query_norm, q_norm, rtol=0, atol=1)
    # compare_result("dequant_scale_q_norm", dequant_scale_q_norm, q_scale)
    # compare_result("kv_cache", kv_cache[:, :512], nope_cache_output[:, :512], rtol=0, atol=1)
    # compare_result("kr_cache", kv_cache[:, 512:640].view(torch.bfloat16), nope_cache_output[:, 512:640].view(torch.bfloat16), rtol=0.005, atol=1e-04)
    # compare_result("dequantScaleCkv", kv_cache[:, 640:].view(torch.float32), nope_cache_output[:, 640:].view(torch.float32), rtol=0.0078125, atol=2.5e-05)
