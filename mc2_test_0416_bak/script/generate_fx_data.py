
import os
import numpy as np
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
            fp_array = numpy.sign(fp_array) * numpy.floor(
                numpy.abs(fp_array) + numpy.array([0.5], dtype=fp_array.dtype))
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

    def _mx_quantize_to_element_format(fp_array: numpy.ndarray, share_exp: numpy.ndarray,
                                       mx_ele_dtype: str, round_mode: str):
        mx_dtype = str(mx_ele_dtype)
        exp_bits = 0
        mantissa_bits = 0
        emax = 0
        max_norm = 0
        if "float4_e2m1" in mx_dtype:
            exp_bits = 2
            mantissa_bits = 1
            emax = 2
            max_norm = 6
        elif "float4_e1m2" in mx_dtype:
            exp_bits = 1
            mantissa_bits = 2
            max_norm = 1.75
        elif "float8_e4m3fn" in mx_dtype:
            exp_bits = 4
            mantissa_bits = 3
            emax = 8
            max_norm = 448
        elif "float8_e5m2" in mx_dtype:
            exp_bits = 5
            mantissa_bits = 2
            emax = 15
            max_norm = 57344

        ret = fp_array / (2 ** share_exp + (share_exp == -float("inf"))) - (share_exp == -float("inf")) * fp_array
        private_exp = numpy.floor(numpy.log2(numpy.abs(ret.astype(numpy.float32)) + (ret == 0))
                                  ).astype(fp_array.dtype, copy=False)
        # The minimum representable exponent for 8 exp bits is -126
        min_exp = -(2 ** (exp_bits - 1)) + exp_bits
        private_exp = private_exp.clip(min=min_exp)
        # Scale up so appropriate number of bits are in the integer portion of the number
        ret = ret / (2 ** private_exp) * (2 ** mantissa_bits)
        ret = _mx_round_mantissa(ret, round_mode)
        # Undo scaling
        ret = ret / (2 ** mantissa_bits) * (2 ** private_exp)
        # Set values > max_norm to Inf if desired, else clamp them
        # numpy.clip(ret, a_min=-max_norm, a_max=max_norm, out=ret)
        return ret

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

    if not isinstance(fp_array, numpy.ndarray):
        raise RuntimeError(f"Input tensor to be quantized should be numpy array. But got {type(fp_array)}")
    if fp_array.dtype.name not in ("bfloat16", "float16", "float32"):
        raise RuntimeError(f"Dtype of input tensor to be quantized only does not support: {fp_array.dtype.name}")
    if mx_ele_dtype not in ("float4_e2m1", "float4_e1m2", "float8_e4m3fn", "float8_e5m2"):
        raise NotImplementedError(f"Not support {mx_ele_dtype} yet!")

    axis = len(fp_array.shape) + axis if axis < 0 else axis
    # padding & reshape to block_size
    fp_array, orig_shape, padded_shape = _mx_reshape_to_blocks(fp_array, axis, block_size)
    # get mx scale exponents
    share_exp = _mx_calculate_share_exp(fp_array, scale_axis=axis + 1, mx_ele_dtype=mx_ele_dtype)
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
    scale_array = 2 ** share_exp
    if ele_array.dtype.name == "bfloat16":
        ele_array = ele_array.astype("float32", copy=False)
    ele_array = ele_array.astype(ele_dtype_np, copy=False)
    scale_array = scale_array.astype(numpy_float8_e8m0(), copy=False)
    return ele_array, scale_array


tmp = torch.tensor([1,2,3,10000000]).to(torch.float).numpy()


result = mx_quantize(tmp, mx_ele_dtype="float4_e2m1", axis=-1, block_size=32, round_mode="rint")
print(result)