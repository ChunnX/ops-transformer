import os
import numpy as np
import copy
import torch
import numpy
from en_dtypes import *

DATA_TYPE_INT_TO_STR = {
    0: 'float32',
    1: 'float16',
    2: 'int8',
    3: 'int32',
    4: 'uint8',
    6: 'int16',
    7: 'uint16',
    8: 'uint32',
    9: 'int64',
    10: 'uint64',
    11: 'double',
    12: 'bool',
    16: 'complex64',
    17: 'complex128',
    18: 'qint8',
    19: 'qint16',
    20: 'qint32',
    21: 'quint8',
    22: 'quint16',
    23: 'resource',
    25: 'dual',
    26: 'variant',
    27: 'bfloat16',
    29: 'int4',
    30: 'uint1',
    31: 'int2',
    32: 'uint2',
    33: 'complex32',
    35: 'float8_e5m2',
    36: 'float8_e4m3fn',
    37: 'float8_e8m0',
    40: 'float4_e2m1',
    41: 'float4_e1m2',
}


def mx_quantize(fp_array: numpy.ndarray, mx_ele_dtype: str = "float4_e2m1", axis: int = -1, block_size: int = 32,
                round_mode: str = "rint") -> tuple:
    """
    quantize BFP16/FP16/FP32 to MX dtypes
    :parameter fp_array: input numpy array with dtype BFP16/FP16/FP32
    :parameter mx_ele_dtype: dtype of element in MX dtype. support float4_e2m1/float4_e1m2/float8_e4m3fn/float8_e5m2
    :parameter axis: specify the axis across which shared scales/exponents are calculated.
    :parameter block_size: each block_size shares the same mx scale along the axis
    :parameter round_mode: round mode. support rint/floor/round/nearest
    :return: mx-scale-exponents & mx-elements

    NOTE: Scenarios below should be considered and tested when TRYING to modify this code:
    1. block with only subnormal floats
    2. block with only one nan
    3. block with only one inf or -inf
    """

    def numpy_float8_e5m2():
        try:
            # noinspection PyUnresolvedReferences
            from ml_dtypes import float8_e5m2
            return float8_e5m2
        except ModuleNotFoundError:
            raise RuntimeError("ml_dtypes is needed to support float8_e5m2 dtype!!! "
                               "Please install with `pip3 install ml-dtypes`")

    def numpy_float8_e4m3fn():
        try:
            # noinspection PyUnresolvedReferences
            from ml_dtypes import float8_e4m3fn
            return float8_e4m3fn
        except ModuleNotFoundError:
            raise RuntimeError("ml_dtypes is needed to support float8_e4m3fn dtype!!! "
                               "Please install with `pip3 install ml-dtypes`")

    def numpy_float8_e8m0():
        try:
            # noinspection PyUnresolvedReferences
            from en_dtypes import float8_e8m0
            return float8_e8m0
        except ModuleNotFoundError:
            raise RuntimeError("en_dtypes is needed to support float8_e8m0 dtype!!! "
                               "Please install with `pip3 install en-dtypes`")

    def numpy_float4_e2m1():
        try:
            # noinspection PyUnresolvedReferences
            from en_dtypes import float4_e2m1
            return float4_e2m1
        except ModuleNotFoundError:
            raise RuntimeError("en_dtypes is needed to support float4_e2m1 dtype!!! "
                               "Please install with `pip3 install en-dtypes`")

    def numpy_float4_e1m2():
        try:
            # noinspection PyUnresolvedReferences
            from en_dtypes import float4_e1m2
            return float4_e1m2
        except ModuleNotFoundError:
            raise RuntimeError("en_dtypes is needed to support float4_e1m2 dtype!!! "
                               "Please install with `pip3 install en-dtypes`")

    def _mx_reshape_to_blocks(fp_array: numpy.ndarray, axis: int, block_size: int):
        fp_array = numpy.expand_dims(fp_array, axis=axis + 1)
        orig_shape = fp_array.shape
        pad = [[0, 0] for _ in range(len(orig_shape))]
        pad_size = orig_shape[axis] % block_size
        pad[axis][1] = block_size - pad_size
        if pad_size > 0:
            fp_array = numpy.pad(fp_array, pad, 'constant')
        padded_shape = fp_array.shape
        reshape = list(padded_shape)
        reshape[axis + 1] = block_size
        reshape[axis] = reshape[axis] // block_size
        fp_array = fp_array.reshape(reshape)
        return fp_array, orig_shape, padded_shape

    def _mx_round_mantissa(fp_array: numpy.ndarray, round_mode: str):
        """
        For example:
        fp_array  = [-4.5, -3.5, -2.5, -2.0, -1.7, -1.5, -1.4, -0.5, -0.2, 0.2, 0.5, 1.4, 1.5, 1.7, 2.0, 2.5, 3.4, 4.5]
        - rint    = [-4.,  -4.,  -2.,  -2.,  -2.,  -2.,  -1.,  -0.,  -0.,  0.,  0.,  1.,  2.,  2.,  2.,  2.,  3.,  4.]
        - nearest = [-5.,  -4.,  -3.,  -2.,  -2.,  -2.,  -1.,  -1.,  -0.,  0.,  1.,  1.,  2.,  2.,  2.,  3.,  3.,  5.]
        - floor   = [-5.,  -4.,  -3.,  -2.,  -2.,  -2.,  -2.,  -1.,  -1.,  0.,  0.,  1.,  1.,  1.,  2.,  2.,  3.,  4.]
        - ceil    = [-4.,  -3.,  -2.,  -2.,  -1.,  -1.,  -1.,  -0.,  -0.,  1.,  1.,  2.,  2.,  2.,  2.,  3.,  4.,  5.]
        - trunc   = [-4.,  -3.,  -2.,  -2.,  -1.,  -1.,  -1.,  -0.,  -0.,  0.,  0.,  1.,  1.,  1.,  2.,  2.,  3.,  4.]
        """
        if round_mode in ("rint", "even"):  # tie to even(c language rint)
            fp_array = numpy.rint(fp_array)
        elif round_mode in ("round", "nearest"):  # tie away from zero(c language round).
            # fp_array = numpy.sign(fp_array) * numpy.floor(
            #    numpy.abs(fp_array) + numpy.array([0.5], dtype=fp_array.dtype))
            sign = numpy.signbit(fp_array)
            rounded_abs = numpy.floor(numpy.abs(fp_array) + numpy.array([0.5], dtype=fp_array.dtype))
            fp_array = numpy.where(sign, -rounded_abs, rounded_abs)
        elif round_mode == "floor":  # round to minus infinity(c language floor)
            fp_array = numpy.floor(fp_array)
        elif round_mode == "ceil":  # round to positive infinity(c language ceil)
            fp_array = numpy.ceil(fp_array)
        elif round_mode == "trunc":  # round to zero(c language truncation)
            fp_array = numpy.trunc(fp_array)
        else:
            raise Exception(f"Unrecognized round method {round_mode}")
        return fp_array

    def _mx_calculate_share_exp(fp_array: numpy.ndarray, scale_axis: int, mx_ele_dtype: str):
        FP32_EXPONENT_BIAS = 127
        FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
        # ele_emax = 2 if "float4_e2m1" in str(mx_ele_dtype) else 0  # 0 for fp4_e1m2
        ele_emax = 0
        mx_dtype = str(mx_ele_dtype)
        if "float4_e2m1" in mx_dtype:
            ele_emax = 2
        elif "float8_e4m3fn" in mx_dtype:
            ele_emax = 8
        elif "float8_e5m2" in mx_dtype:
            ele_emax = 15
        fp_abs_max = numpy.max(numpy.abs(fp_array), axis=scale_axis, keepdims=True)
        res = numpy.floor(
            numpy.log2(fp_abs_max.astype(numpy.float32) + FP32_MIN_NORMAL * (fp_abs_max == 0))
        ) - ele_emax
        res[fp_abs_max == 0] = -float("inf")
        return res

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
        numpy_dtype = numpy.dtype(dt)
        if numpy_dtype.kind in "iu":
            numpy_info = numpy.iinfo(numpy_dtype)
        else:
            numpy_info = numpy.finfo(numpy_dtype)
        return numpy_info.min, numpy_info.max

    def _mx_quantize_to_element_format(fp_array: numpy.ndarray, share_exp: numpy.ndarray,
                                       mx_ele_dtype: str, round_mode: str):
        mx_dtype = str(mx_ele_dtype)
        exp_bits = 0
        mantissa_bits = 0
        if "float4_e2m1" in mx_dtype:
            exp_bits = 2
            mantissa_bits = 1
        elif "float4_e1m2" in mx_dtype:
            exp_bits = 1
            mantissa_bits = 2
        elif "float8_e4m3fn" in mx_dtype:
            exp_bits = 4
            mantissa_bits = 3
        elif "float8_e5m2" in mx_dtype:
            exp_bits = 5
            mantissa_bits = 2

        max_norm = get_dtype_range(mx_dtype)[1]

        ret = fp_array / (2 ** share_exp)
        private_exp = numpy.floor(numpy.log2(numpy.abs(ret.astype(numpy.float32)) + (ret == 0))
                                  ).astype(fp_array.dtype, copy=False)
        # The minimum representable exponent for 8 exp bits is -126
        # 5bit exp  2^4-1 = 15  or 2 ** (exp_bits - 1) -1
        if "float8_e4m3fn" in mx_dtype or "float8_e5m2" in mx_dtype:
            min_exp = -(2 ** (exp_bits - 1)) + 2  # 指数位 -2^3+4 = -4
        else:
            min_exp = -(2 ** (exp_bits - 1)) + exp_bits
        private_exp = private_exp.clip(min=min_exp)
        # Scale up so appropriate number of bits are in the integer portion of the number
        ret = ret / (2 ** private_exp) * (2 ** mantissa_bits)
        ret = _mx_round_mantissa(ret, round_mode)
        # Undo scaling
        ret = ret / (2 ** mantissa_bits) * (2 ** private_exp)
        # Set values > max_norm to Inf if desired, else clamp them
        numpy.clip(ret, a_min=-max_norm, a_max=max_norm, out=ret)
        return ret

    def pad_to_even(tensor: numpy.ndarray, axis: int):
        if not isinstance(tensor, numpy.ndarray):
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

        padded_tensor = numpy.pad(tensor, pad_width, mode='constant', constant_values=2 ** -127)
        return padded_tensor

    def _mx_undo_reshape_to_blocks(fp_array: numpy.ndarray, axis: int, orig_shape: tuple, padded_shape: tuple):
        # Undo tile reshaping
        fp_array = fp_array.reshape(padded_shape)
        # Undo padding
        if tuple(padded_shape) != tuple(orig_shape):
            slices = [slice(0, x) for x in orig_shape]
            fp_array = fp_array[tuple(slices)]
        # Remove extra dimension
        fp_array = numpy.squeeze(fp_array, axis=axis + 1)
        return fp_array

    def inf_clip(tensor, dtype):
        if dtype == 'float4_e2m1':
            max_val = 6
            min_val = -6
        elif dtype == 'float4_e1m2':
            max_val = 1.75
            min_val = -1.75
        elif dtype == 'float8_e4m3fn':
            max_val = 448
            min_val = -448
        elif dtype == 'float8_e5m2':
            max_val = 57344
            min_val = -57344
        elif dtype == 'float8_e8m0':
            max_val = 127
            min_val = -127
        else:
            print(f"ERROR: dtype is Not supported, dtype_input is {dtype}")
            return
        tensor[np.isposinf(tensor)] = max_val
        tensor[np.isneginf(tensor)] = min_val
        tensor[np.isnan(tensor)] = max_val
        return tensor

    def interleave(tensor: numpy.ndarray, axis: int, n_group: int = 2) -> numpy.ndarray:
        if not isinstance(tensor, numpy.ndarray):
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
                [group_length, 2] +
                shape[axis + 1:])
        reshaped = tensor.reshape(new_shape)

        # 构建转置顺序：交换组维度和组内维度
        transpose_order = (
                list(range(0, axis + 1)) +  # 目标轴之前的维度
                list(range(axis + 2, len(new_shape))) +
                [axis + 1, ])  # 后续维度

        # 执行转置
        transposed = reshaped.transpose(transpose_order)

        return transposed

    if not isinstance(fp_array, numpy.ndarray):
        raise RuntimeError(f"Input tensor to be quantized should be numpy array. But got {type(fp_array)}")
    if fp_array.dtype.name not in ("bfloat16", "float16", "float32"):
        raise RuntimeError(f"Dtype of input tensor to be quantized is not supported: {fp_array.dtype.name}")
    if mx_ele_dtype not in ("float4_e2m1", "float4_e1m2", "float8_e4m3fn", "float8_e5m2"):
        raise NotImplementedError(f"Not support {mx_ele_dtype} yet!")

    axis = len(fp_array.shape) + axis if axis < 0 else axis
    # padding & reshape to block_size
    fp_array, orig_shape, padded_shape = _mx_reshape_to_blocks(fp_array, axis, block_size)
    # get mx scale exponents
    share_exp = _mx_calculate_share_exp(fp_array,
                                        scale_axis=axis + 1,
                                        mx_ele_dtype=mx_ele_dtype)
    scale_emax = 2 ** (8 - 1) - 1  # 8 for E8M0
    share_exp[share_exp > scale_emax] = float("NaN")
    share_exp[share_exp < -scale_emax] = -scale_emax

    # quantize mx element
    ele_array = _mx_quantize_to_element_format(fp_array, share_exp, mx_ele_dtype, round_mode)
    # undo reshape
    ele_array = _mx_undo_reshape_to_blocks(ele_array, axis, orig_shape, padded_shape)
    share_exp = numpy.squeeze(share_exp, axis=axis + 1)
    # convert to fp8_e8m0 & fp4/fp8 dtype
    ele_dtype_np = eval(f"numpy_{mx_ele_dtype}()")
    # share_exp is always float32
    scale_array = 2 ** share_exp
    if ele_array.dtype.name == "bfloat16":
        ele_array = ele_array.astype("float32", copy=False)
    ele_array = numpy.nan_to_num(ele_array, nan=0.0, copy=False)
    if mx_ele_dtype not in ["float4_e2m1", "float4_e1m2"]:  # fp4类型astype有问题，单独处理
        ele_array = ele_array.astype(ele_dtype_np, copy=False)
    else:
        ele_array = trans_fp4_to_uint8(ele_array, ele_array.shape, mx_ele_dtype)
    scale_array_pad = pad_to_even(scale_array, axis=axis)

    result_shape = copy.deepcopy(list(scale_array_pad.shape))
    result_shape.append(2)

    result_shape[axis] = scale_array_pad.shape[axis] // 2

    # when axis is -1, do not need interleave
    if axis != (len(fp_array.shape) - 1):
        scale_array_pad = interleave(scale_array_pad, axis=axis)
    scale_array_pad = scale_array_pad.reshape(result_shape)
    scale_array = scale_array_pad.astype(numpy_float8_e8m0(), copy=False)

    # 将inf赋值为最大值，-inf赋值为最小值
    ele_array = inf_clip(ele_array, mx_ele_dtype)
    scale_array = inf_clip(scale_array, 'float8_e8m0')

    return ele_array, scale_array


def trans_fp4_to_uint8(sampled_values, shape, dtype):
    fp4_e1m2_values = [
        0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75,
        -0.25, -0.5, -0.75, -1, -1.25, -1.5, -1.75
    ]
    fp4_e2m1_values = [
        0, 0.5, 1, 1.5, 2, 3, 4, 6,
        0, -0.5, -1, -1.5, -2, -3, -4, -6
    ]
    value_to_code_e1m2 = {val: idx for idx, val in enumerate(fp4_e1m2_values)}
    value_to_code_e2m1 = {val: idx for idx, val in enumerate(fp4_e2m1_values)}

    sampled_values = sampled_values.flatten()
    print("shape:", shape)
    if dtype == "float4_e1m2":
        encoded_codes = [value_to_code_e1m2[val] for val in sampled_values]
    elif dtype == "float4_e2m1":
        encoded_codes = [value_to_code_e2m1[val] for val in sampled_values]
    else:
        print("error fp4 dtype")
    # 转换为nparray 并reshape为目标形状
    fp4_tensor_encoded = np.array(encoded_codes).astype(np.uint8).reshape(shape)
    return fp4_tensor_encoded
