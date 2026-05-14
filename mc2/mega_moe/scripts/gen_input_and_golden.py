import torch
import math
import numpy
import numpy as np
import copy
import re
import importlib.metadata as importlib_metadata
from packaging import version
import subprocess
import sys

def swiglu(gmm_out):
    # 将结果分成两部分以应用 SwiGLU 激活函数
    c_temp4, gate = gmm_out.chunk(2, dim=-1)
    c_temp5 = c_temp4 * torch.sigmoid(c_temp4)  # SwiGLU 激活
    c_temp6 = c_temp5 * gate  # 与门控值进行逐元素相乘
    return c_temp6

def check_installed_version(lib_name: str, min_version: str) -> bool:
    """检查已安装库的版本（无需提前导入）"""
    try:
        # 获取已安装库的版本号
        current_version = importlib_metadata.version(lib_name)
        current = version.parse(current_version)
        min_required = version.parse(min_version)
        
        if current >= min_required:
            print(f"{lib_name}版本符合要求：{current_version}")
            return True
        else:
            print(f"{lib_name}版本过低！要求≥{min_version}，当前{current_version}")
            return False
    except importlib_metadata.PackageNotFoundError:
        print(f"{lib_name}未安装！")
        return False
    except version.InvalidVersion as e:
        print(f"版本解析错误：{e}")
        return False

def upgrade_lib(lib_name: str, min_version: str):
    """版本过低时自动升级"""
    if not check_installed_version(lib_name, min_version):
        print(f"正在升级{lib_name}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", f"{lib_name}=={min_version}", "--upgrade"],
            check=True,
            capture_output=True
        )
        print(f"{lib_name}升级完成！")

def golden_output_process(output, param_split_item, groupType, activation_feature_out=None, dyn_scale_out=None):
    golden = []

    if param_split_item in [2, 3]:
        if groupType == 0:
            ret = torch.cat(output, dim=-2)
            golden.append(ret)
        elif groupType == 2:
            ret = torch.stack(output, dim=0)
            golden.append(ret)
        else:
            print("unsupported group type, please check")
        # 检查inf、nan的数量
        num_inf = torch.isinf(golden[0]).sum().item()
        num_nan = torch.isnan(golden[0]).sum().item()
        ratio = 100 * ((num_inf + num_nan) / (ret.shape[0] * ret.shape[1])) if ret.shape[0] * ret.shape[1] != 0 else 0
        print(f"---INF_num = {num_inf}, NAN_num = {num_nan}, Ratio = {ratio} %")
    else:
        golden = output
    return golden

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

def numpy_float8_e4m3fn():
    try:
        # noinspection PyUnresolvedReferences
        from ml_dtypes import float8_e4m3fn
        return float8_e4m3fn
    except ModuleNotFoundError:
        raise RuntimeError("ml_dtypes is needed to support float8_e4m3fn dtype!!! "
                            "Please install with `pip3 install ml-dtypes`")

def numpy_float8_e5m2():
    try:
        # noinspection PyUnresolvedReferences
        from ml_dtypes import float8_e5m2
        return float8_e5m2
    except ModuleNotFoundError:
        raise RuntimeError("ml_dtypes is needed to support float8_e5m2 dtype!!! "
                            "Please install with `pip3 install ml-dtypes`")

def numpy_float8_e8m0():
    try:
        # noinspection PyUnresolvedReferences
        from en_dtypes import float8_e8m0
        return float8_e8m0
    except ModuleNotFoundError:
        raise RuntimeError("en_dtypes is needed to support float8_e8m0 dtype!!! "
                            "Please install with `pip3 install en-dtypes`")

def qbmm_mx_golden_calculate(x, weight, scale, pertoken_scale, has_bias=0, bias=None, bias_dtype=None, offset=None, act_type=0):
    output = [None] * len(x)
    for i in range(len(x)):
        if isinstance(x[i], torch.Tensor):
            x1 = x[i].numpy().astype(np.float32)
        else:
            x1 = x[i].astype(np.float32)
        
        if pertoken_scale is not None:
            if isinstance(pertoken_scale[i], torch.Tensor):
                pertoken_scale_i = pertoken_scale[i].numpy()
            else:
                pertoken_scale_i = pertoken_scale[i]
            pertoken_scale_mx_broadcast = np.repeat(pertoken_scale_i, 32, axis=-1)
            x1_dims = len(x1.shape)
            x1_pad_len = pertoken_scale_mx_broadcast.shape[-1] - x1.shape[-1]
            x1 = np.pad(x1, [(0, 0)] * (x1_dims - 1) + [(0, x1_pad_len)], mode='constant', constant_values=0)
            x1 = x1 * pertoken_scale_mx_broadcast
        
        if isinstance(weight[i], torch.Tensor):
            x2 = weight[i].numpy().astype(np.float32)
        else:
            x2 = weight[i].astype(np.float32)
        
        if isinstance(scale[i], torch.Tensor):
            scale_i = scale[i].numpy()
        else:
            scale_i = scale[i]
        deq_scale_mx_broadcast = np.repeat(scale_i, 32, axis=-2)
        x2_dims = len(x2.shape)
        x2_pad_len = deq_scale_mx_broadcast.shape[-2] - x2.shape[-2]
        x2 = np.pad(x2, [(0, 0)] * (x2_dims - 2) + [(0, x2_pad_len)] + [(0, 0)], mode='constant', constant_values=0)
        
        x2 = x2 * deq_scale_mx_broadcast
        
        x1 = torch.from_numpy(x1).to(torch.float)
        x2 = torch.from_numpy(x2).to(torch.float)
        
        output[i] = torch.matmul(x1, x2)
        if has_bias == 1:
            output[i] = torch.add(output[i], bias[i])
    return output

def qbmm_mx_split(x1, x2, scale, pertoken_scale, groupType, param_group_list, bias=None, has_bias=0, offset=None):
    x_new, weight_new, pertoken_scale_new, scale_new = [], [], [], []
    scale_k_offset = 0  # scale,pertoken当前偏移
    bias_new = [] if has_bias == 1 else None
    # TypeK场景 x1和x2都需要按照K分组进行切分，TypeM场景只有x1需要进行切分, GOLD计算时，x1和x2的转置均为False、False
    
    for i in range(len(param_group_list)):
        x2_g = x2[i]
        scale_g = scale[i]

        pre = 0 if i == 0 else param_group_list[i - 1]
        cur = param_group_list[i]
        x1_g = x1[pre: cur, :]
        pertoken_scale_g = pertoken_scale[pre: cur, :]

        x_new.append(x1_g)
        weight_new.append(x2_g)
        pertoken_scale_new.append(pertoken_scale_g)
        scale_new.append(scale_g)
        # 当前只有mxfp4 TypeM场景有bias， mxfp8 无bias
        if has_bias == 1:
            bias_g = bias[i]
            bias_new.append(bias_g)

    return x_new, weight_new, bias_new, pertoken_scale_new, scale_new, offset

def gmm_quant_data_split_per_expect(x1, x2, scale, pertoken_scale, groupType, param_group_list, bias=None, has_bias=0, offset = None, output_dtype = 'int32', scale_dtype = "float32", bias_dtype = "int32"):
    return qbmm_mx_split(x1, x2, scale, pertoken_scale, groupType, param_group_list, bias, has_bias, offset=None)

def transpose_last_two_dim(input, transpose_flag):
    def np_transpose_last_two_dim(input, transpose_flag):
        if transpose_flag == 1:
            # 转置场景
            print("The transpose flag is true, need to trans last 2 dim")
            # 获取当前维度数
            ndim = input.ndim
            # 构造新的轴顺序：保持前面的所有维度不变，最后两个交换
            axes = list(range(ndim - 2)) + [ndim - 1, ndim - 2]
            # 进行转置
            return np.transpose(input, axes=axes)
        else:
            print("The transpose flag is false, no need to trans")
            return input

    def torch_transpose_last_two_dim(input, transpose_flag):
        if transpose_flag == 1:
            ndim = input.dim()
            # 如果维度小于2，无法转置，抛出异常
            if ndim < 2:
                raise ValueError("Tensor must have at least 2 dimensions.")
            # 构建新的维度顺序：保留前面所有维度（从 0 到 ndim - 3）, 将最后两个维度交换（-1 和 -2）
            new_dims = list(range(ndim - 2)) + [ndim - 1, ndim - 2]
            # 使用 permute 转置
            return input.permute(*new_dims)
        else:
            print("The transpose flag is false, no need to trans")
            return input

    if isinstance(input, torch.Tensor):
        return torch_transpose_last_two_dim(input, transpose_flag)
    elif isinstance(input, np.ndarray):
        return np_transpose_last_two_dim(input, transpose_flag)
    else:
        raise TypeError("不支持的数据类型")

def interleave_gate_up_along_axis(arr, n_axis, interleave_tile=128):
    """把 arr 沿 n_axis 做 [gate||up] → [128 gate | 128 up] 每 256 列交织。
    arr 沿 n_axis 的前一半是 gate，后一半是 up；新 layout 每 interleave_tile*2 列 = gate 128 + up 128。

    参数：
        arr: numpy 数组，沿 n_axis 长度必须是 256 对齐 (= 2 × interleave_tile × num_tiles)
        n_axis: 要做交织的轴
        interleave_tile: 默认 128（每 256 列分 128 gate + 128 up）
    返回：
        交织后的同 dtype 同 ndim 数组，shape 不变
    """
    n_total = arr.shape[n_axis]
    assert n_total % (interleave_tile * 2) == 0, (
        f"n_axis 长度 {n_total} 必须 {interleave_tile*2} 对齐才能交织")
    output_n = n_total // 2
    num_tiles = output_n // interleave_tile

    # 把 n_axis 挪到位置 0 方便切片，处理完再挪回去
    arr_moved = np.moveaxis(arr, n_axis, 0)  # (n_total, ...)

    gate = arr_moved[:output_n]            # (output_n, ...) = gate 全部
    up = arr_moved[output_n:]              # (output_n, ...) = up 全部

    other = arr_moved.shape[1:]
    gate = gate.reshape((num_tiles, interleave_tile) + other)  # (num_tiles, 128, ...)
    up = up.reshape((num_tiles, interleave_tile) + other)      # (num_tiles, 128, ...)

    # 沿新轴叠成 (num_tiles, 2, 128, ...) → reshape 回 (num_tiles*2*128, ...) = (n_total, ...)
    stacked = np.stack([gate, up], axis=1)  # (num_tiles, 2, 128, ...)
    interleaved = stacked.reshape((n_total,) + other)

    # 移回原始 axis 位置
    interleaved = np.moveaxis(interleaved, 0, n_axis)
    assert interleaved.shape == arr.shape, f"shape mismatch: {interleaved.shape} vs {arr.shape}"
    return interleaved


def qbmm_mxfp8_quant_data_prepare(W_tmp, w_dtype, weight_transpose_flag):
    w_float_list, w_mx_list, scale_mx_list = [], [], []
    
    # mx typeM, x不支持转置(M, K), 支持weight转置(N, K)， B580 只维持weight不转置场景
    w_float_list.append(W_tmp)

    w_split, scale_split = mx_quantize(w_float_list[0], mx_ele_dtype=w_dtype,
                                        axis=-2 if weight_transpose_flag == 0 else -1,
                                        block_size=32, round_mode="rint")
    w_mx_list.append(w_split)
    scale_mx_list.append(scale_split)

    W_input = np.concatenate(w_mx_list, axis=-2 if weight_transpose_flag == 0 else -1)
    Scale_input = np.concatenate(scale_mx_list, axis=-2 if weight_transpose_flag == 0 else -1)
    Scale_npu = copy.deepcopy(Scale_input)
    w_npu = copy.deepcopy(W_input)
    # 转置（x weight 重写后需重新进行转置处理）
    W_input = transpose_last_two_dim(W_input, weight_transpose_flag)

    if weight_transpose_flag:
        Scale_input = Scale_input.reshape(Scale_input.shape[0], Scale_input.shape[1], Scale_input.shape[2] * 2)
    else:
        Scale_input_transposed = np.transpose(Scale_input, axes=(0, 2, 1, 3))
        Scale_input_transposed_reshape = Scale_input_transposed.reshape(Scale_input_transposed.shape[0], Scale_input_transposed.shape[1], Scale_input_transposed.shape[2] * 2)
        Scale_input = np.transpose(Scale_input_transposed_reshape, axes=(0, 2, 1))
    Scale_input = transpose_last_two_dim(Scale_input, weight_transpose_flag)
    

    return W_input, Scale_input, Scale_npu, w_npu

def regenerate_quant_data(w_shape, params_out_dtype):
    mu_W = -0.05
    Sigma_W = 4
    # 高斯分布 + 白噪声
    W_tmp = (np.random.normal(mu_W, Sigma_W, w_shape) - np.random.uniform(0.1 * mu_W, -0.1 * mu_W, w_shape)).astype(np.float32)
    W_tmp = np.clip(W_tmp, -1.0, 1.0)
    if params_out_dtype == 'fp32':
        W_tmp = np.abs(W_tmp)
    return W_tmp

def dispatch_mx_quantize(fp_array: numpy.ndarray, mx_ele_dtype: str = "float4_e2m1",
                axis: int = -1, block_size: int = 32, round_mode: str = "rint") -> tuple:
    """
    quantize BFP16/FP16/FP32 to MX dtypes
    :parameter fp_array: input numpy array with dtype BFP16/FP16/FP32
    :parameter mx_ele_dtype: dtype of element in MX dtype.
               support float4_e2m1/float4_e1m2/float8_e4m3fn/float8_e5m2
    :parameter axis: specify the axis across which shared scales/exponents are calculated.
    :parameter block_size: each block_size shares the same mx scale along the axis
    :parameter round_mode: round mode. support rint/floor/round/nearest
    :return: mx-scale-exponents & mx-elements

    NOTE: Scenarios below should be considered and tested when TRYING to modify this code:
    1. block with only subnormal floats
    2. block with only one nan
    3. block with only one inf or -inf
    """
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


    def _mx_undo_reshape_to_blocks(fp_array: numpy.ndarray, axis: int,
                                orig_shape: tuple, padded_shape: tuple):
        # Undo tile reshaping
        fp_array = fp_array.reshape(padded_shape)
        # Undo padding
        if tuple(padded_shape) != tuple(orig_shape):
            slices = [slice(0, x) for x in orig_shape]
            fp_array = fp_array[tuple(slices)]
        # Remove extra dimension
        fp_array = numpy.squeeze(fp_array, axis=axis + 1)
        return fp_array


    def _mx_calculate_share_exp(fp_array: numpy.ndarray, scale_axis: int, mx_ele_dtype: str):
        FP32_EXPONENT_BIAS = 127
        FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
        #ele_emax = 2 if "float4_e2m1" in str(mx_ele_dtype) else 0  # 0 for fp4_e1m2
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
            fp_array = numpy.sign(fp_array) * numpy.floor(numpy.abs(fp_array) +
                                                        numpy.array([0.5], dtype=fp_array.dtype))
        elif round_mode == "floor":  # round to minus infinity(c language floor)
            fp_array = numpy.floor(fp_array)
        elif round_mode == "ceil":   # round to positive infinity(c language ceil)
            fp_array = numpy.ceil(fp_array)
        elif round_mode == "trunc":  # round to zero(c language truncation)
            fp_array = numpy.trunc(fp_array)
        else:
            raise Exception(f"Unrecognized round method {round_mode}")
        return fp_array

    def _mx_quantize_to_element_format(fp_array: numpy.ndarray, share_exp: numpy.ndarray,
                                    mx_ele_dtype: str, round_mode: str):
        mx_dtype = str(mx_ele_dtype)
        match = re.search(r'e(\d+)m(\d+)', mx_dtype)
        if match:
            exp_bits = int(match.group(1))
            mantissa_bits = int(match.group(2))
        else:
            raise ValueError(f"mx element dtype [{mx_ele_dtype}] is not recognized.")

        ret = fp_array / (2 ** share_exp)
        private_exp = numpy.floor(numpy.log2(numpy.abs(ret.astype(numpy.float32)) + (ret == 0))
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
        numpy.clip(ret, a_min=-max_norm, a_max=max_norm, out=ret)
        return ret

    def pad_to_even(tensor: numpy.ndarray, axis: int) -> numpy.ndarray:
        """
        在指定的 axis 上将 tensor 对齐到偶数长度（即按 2 对齐），不足补 0。

        参数:
            tensor (numpy.ndarray): 输入数组
            axis (int): 要对齐的轴（从 0 开始）

        返回:
            numpy.ndarray: 对齐后的数组
        """
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
    # NPU will cast NaN (with or without sign) to positive ZERO (sign is dropped)
    ele_array = numpy.nan_to_num(ele_array, nan=0.0, copy=False)
    ele_array = ele_array.astype(ele_dtype_np, copy=False)
    # Cube only supports even scales. need to pad zero.
    scale_array_pad = pad_to_even(scale_array, axis=axis)
    scale_array = scale_array_pad.astype(numpy_float8_e8m0(), copy=False)

    return ele_array, scale_array

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
            #fp_array = numpy.sign(fp_array) * numpy.floor(
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


    if not isinstance(fp_array, numpy.ndarray):
        raise RuntimeError(f"Input tensor to be quantized should be numpy array. But got {type(fp_array)}")
    if fp_array.dtype.name not in ("bfloat16", "float16", "float32"):
        raise RuntimeError(f"Dtype of input tensor to be quantized is not supported: {fp_array.dtype.name}")
    if mx_ele_dtype not in ("float4_e2m1", "float4_e1m2", "float8_e4m3fn", "float8_e5m2"):
        raise NotImplementedError(f"Not support {mx_ele_dtype} yet!")
    if mx_ele_dtype in ["float4_e2m1", "float4_e1m2"]: #fp4类型需要升级到en_dtypes==0.0.4
        upgrade_lib("en_dtypes","0.0.4")
        
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
    ele_dtype_np = eval(f"numpy_{mx_ele_dtype}()")
    # share_exp is always float32
    scale_array = 2 ** share_exp
    if ele_array.dtype.name == "bfloat16":
        ele_array = ele_array.astype("float32", copy=False)
    ele_array = numpy.nan_to_num(ele_array, nan=0.0, copy=False)
    ele_array = ele_array.astype(ele_dtype_np, copy=False)
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

class MXFP8MatrixQuantizer:
    def __init__(self, data_format='E4M3', axis=1, block_size=32, epsilon=1e-12):
        self.data_format = data_format
        # self.axis = 0
        self.axis = axis
        self.block_size = block_size
        self.epsilon = epsilon
        
        # MXFP8 数据格式定义
        self.DATA_FORMATS = {
            'E4M3': {
                'exp_bits': 4,
                'mantissa_bits': 3,
                'bias': 7,
                'emax': 8,
                'max_value': 448.0,
                'min_value': -448.0
            },
            'E5M2': {
                'exp_bits': 5,
                'mantissa_bits': 2,
                'bias': 15,
                'emax': 15,
                'max_value': 57344.0,
                'min_value': -57344.0
            }
        }
        
        # FP8_E8M0FNU 缩放格式定义
        self.SCALE_FORMAT = {
            'name': 'FP8_E8M0FNU',
            'exp_bits': 8,
            'mantissa_bits': 0,
            'bias': 128,  # 偏置为128，0表示指数-128
            'max_exp': 127,
            'min_exp': -128,
            'max_value': 2**127,   # 约 1.7e38
            'min_value': 2**-128,  # 约 2.9e-39
            'signed': False,       # 无符号，仅正数
            'allow_zero': True     # 允许零值（指数-128表示零）
        }
        
        self.config = self.DATA_FORMATS[data_format]
        
        # 生成FP8查找表
        self.fp8_lut = self._generate_fp8_lut()
        # 预排序查找表，避免每次量化都排序
        self.sorted_fp8_lut, _ = torch.sort(self.fp8_lut)
        self.fp8_min = self.config['min_value']
        self.fp8_max = self.config['max_value']
    
    def _build_e4m3_lookup_table(self):
        """构建 E4M3 值查找表"""
        values = []

        # E4M3: 总共 256 个可能值 (0-255)
        for i in range(256):
            # 解码
            if i < 128:  # 正数
                sign = 1
                val = i
            else:  # 负数
                sign = -1
                val = i - 128

            if val == 0:
                value = 0.0
            elif val == 127:  # NaN，替换为最大值
                value = sign * self.config['max_value']
            else:
                exp = (val >> 3) & 0x0F
                mantissa = val & 0x07

                if exp == 0:
                    # 次正规数
                    value = (mantissa / 8.0) * (2.0 ** (1 - self.config['bias']))
                else:
                    # 正规数
                    value = (1.0 + mantissa / 8.0) * (2.0 ** (exp - self.config['bias']))

                value = sign * value

            # 钳位到有效范围
            if value > self.config['max_value']:
                value = self.config['max_value']
            elif value < self.config['min_value']:
                value = self.config['min_value']

            values.append(value)

        self.fp8_lut = torch.tensor(values, dtype=torch.float32)
        self.fp8_min = self.config['min_value']
        self.fp8_max = self.config['max_value']

    def _build_e5m2_lookup_table(self):
        """构建 E5M2 值查找表"""
        values = []

        for i in range(256):
            # 解码
            if i < 128:  # 正数
                sign = 1
                val = i
            else:  # 负数
                sign = -1
                val = i - 128

            if val == 0:
                value = 0.0
            elif val >= 124 and val <= 127:  # NaN/Inf，替换为最大值
                value = sign * self.config['max_value']
            else:
                exp = (val >> 2) & 0x1F
                mantissa = val & 0x03

                if exp == 0:
                    # 次正规数
                    value = (mantissa / 4.0) * (2.0 ** (1 - self.config['bias']))
                else:
                    # 正规数
                    value = (1.0 + mantissa / 4.0) * (2.0 ** (exp - self.config['bias']))

                value = sign * value

            # 钳位到有效范围
            if value > self.config['max_value']:
                value = self.config['max_value']
            elif value < self.config['min_value']:
                value = self.config['min_value']

            values.append(value)

        self.fp8_lut = torch.tensor(values, dtype=torch.float32)
        self.fp8_min = self.config['min_value']
        self.fp8_max = self.config['max_value']

    def _generate_fp8_lut(self):
        """生成FP8查找表"""
        if self.data_format == 'E4M3':
            self._build_e4m3_lookup_table()
        else:
            self._build_e5m2_lookup_table()
        return self.fp8_lut
    
    def _quantize_to_fp8(self, data: torch.Tensor) -> torch.Tensor:
        """
        量化数据到 FP8 格式

        使用查找表进行最近邻量化（性能优化版本）
        """
        original_shape = data.shape
        data_flat = data.flatten()
        data_clamped = torch.clamp(data_flat, self.fp8_min, self.fp8_max)
        
        # 向量化实现：一次性处理所有数据点
        # 使用 searchsorted 找到每个值的插入位置
        idx = torch.searchsorted(self.sorted_fp8_lut, data_clamped)
        
        # 处理边界情况
        idx = torch.clamp(idx, 0, len(self.sorted_fp8_lut))
        
        # 创建索引张量
        idx_prev = torch.clamp(idx - 1, 0, len(self.sorted_fp8_lut) - 1)
        idx_curr = torch.clamp(idx, 0, len(self.sorted_fp8_lut) - 1)
        
        # 计算距离
        dist_prev = torch.abs(self.sorted_fp8_lut[idx_prev] - data_clamped)
        dist_curr = torch.abs(self.sorted_fp8_lut[idx_curr] - data_clamped)
        
        # 选择距离较小的索引
        closest_idx = torch.where(dist_prev <= dist_curr, idx_prev, idx_curr)
        
        # 获取量化后的值
        quantized_flat = self.sorted_fp8_lut[closest_idx]
        
        return quantized_flat.view(original_shape)
    
    def _compute_scale_fp8_e8m0fnu(self, block_data: torch.Tensor) -> tuple:
        """
        计算 FP8_E8M0FNU 格式的缩放因子

        返回:
            scale: 缩放因子 (浮点数)
            exp: 指数 (E8M0FNU 格式)
        """
        # 计算块的最大绝对值
        max_abs = torch.max(torch.abs(block_data)).item()

        if max_abs < self.epsilon:
            # 全零或接近零的块
            return 1.0, 0  # 指数0表示缩放因子2^0=1

        # 计算对数 (log2)
        if max_abs > 0:
            log2_scale = math.log2(max_abs)
        else:
            log2_scale = -128  # 最小值

        # 四舍五入到最近的整数 (指数)
        exp = int(math.floor(log2_scale)) - self.config['emax']

        # 钳位到 E8M0FNU 范围 [-128, 127]
        exp = max(self.SCALE_FORMAT['min_exp'],
                    min(exp, self.SCALE_FORMAT['max_exp']))

        # 计算缩放因子
        scale = 2.0 ** exp

        return scale, exp

    def _process_block(self, block_data: torch.Tensor) -> tuple:
        """
        处理单个分块

        返回:
            quantized_block: 量化后的块
            scale: 缩放因子
            exp: 指数
        """
        # 计算缩放因子
        scale, exp = self._compute_scale_fp8_e8m0fnu(block_data)

        # 应用缩放
        if abs(scale) > 1e-12:
            scaled_data = block_data / scale
        else:
            scaled_data = block_data  # 缩放因子为0，不缩放

        # 量化到 FP8
        quantized_scaled = self._quantize_to_fp8(scaled_data)

        return quantized_scaled, scale, exp

    def _quantize_by_cols(self, matrix: torch.Tensor, M: int, N: int) -> tuple:
        """
        按列方向量化（部分向量化优化）

        分块: 每 block_size 列一个块
        """
        # 计算分块数量
        num_blocks = (N + self.block_size - 1) // self.block_size

        # 初始化结果
        quantized_matrix = torch.zeros_like(matrix)
        scale_matrix = torch.ones((M, (num_blocks + 1) // 2 * 2), dtype=torch.float32)

        # 处理每个列块
        for block_idx in range(num_blocks):
            # 计算当前块的起始和结束列
            start_col = block_idx * self.block_size
            end_col = min(start_col + self.block_size, N)
            block_width = end_col - start_col

            # 提取当前块 (M x block_width)
            block_data = matrix[:, start_col:end_col]

            # 逐行计算 scale
            scales = torch.zeros(M, dtype=torch.float32)
            for row in range(M):
                row_data = block_data[row, :]
                scale, exp = self._compute_scale_fp8_e8m0fnu(row_data)
                scales[row] = scale

            # 向量化应用 scale 和量化
            # 将 scale 扩展为 (M, block_width) 用于广播
            scales_expanded = scales.unsqueeze(1).expand(M, block_width)

            # 避免除零
            scales_expanded = torch.where(scales_expanded > 1e-12, scales_expanded, torch.ones_like(scales_expanded))

            # 应用缩放
            scaled_block = block_data / scales_expanded

            # 向量化量化整个块
            quantized_block = self._quantize_to_fp8(scaled_block)

            # 存储结果
            quantized_matrix[:, start_col:end_col] = quantized_block
            scale_matrix[:, block_idx] = scales

        return quantized_matrix, scale_matrix

    def _quantize_by_rows(self, matrix: torch.Tensor, M: int, N: int) -> tuple:
        """
        按行方向量化

        分块: 每 block_size 行一个块
        """
        # 计算分块数量
        num_blocks = (M + self.block_size - 1) // self.block_size

        # 初始化结果
        quantized_matrix = torch.zeros_like(matrix)
        scale_matrix = torch.ones(((num_blocks + 1) // 2 * 2, N), dtype=torch.float32)

        # 处理每个行块
        for block_idx in range(num_blocks):
            # 计算当前块的起始和结束行
            start_row = block_idx * self.block_size
            end_row = min(start_row + self.block_size, M)

            # 提取当前块
            block_data = matrix[start_row:end_row, :]

            # 对当前块的每一列单独处理（列方向）
            for col in range(N):
                # 提取列数据
                col_data = block_data[:, col]

                # 处理该列
                quantized_col, scale, exp = self._process_block(col_data)

                # 存储结果
                quantized_matrix[start_row:end_row, col] = quantized_col
                scale_matrix[block_idx, col] = scale

        return quantized_matrix, scale_matrix

    def quantize_matrix(self, matrix: torch.Tensor) -> tuple:
        """
        量化二维矩阵

        参数:
            matrix: 输入二维矩阵 [M, N]

        返回:
            quantized_matrix: 量化后的矩阵 [M, N]
            scale_matrix: 缩放因子矩阵
        """
        if matrix.dim() != 2:
            raise ValueError(f"输入必须是二维矩阵，当前维度: {matrix.dim()}")

        M, N = matrix.shape

        # 根据量化轴确定分块方式
        if self.axis == 0:  # 行方向量化
            return self._quantize_by_rows(matrix, M, N)
        else:  # 列方向量化
            return self._quantize_by_cols(matrix, M, N)

    def dequantize_matrix(self, quantized_matrix: torch.Tensor, scale_matrix: torch.Tensor) -> torch.Tensor:
        """
        反量化二维矩阵

        参数:
            quantized_matrix: 量化后的矩阵 [M, N]
            scale_matrix: 缩放因子矩阵

        返回:
            dequantized_matrix: 反量化后的矩阵 [M, N]
        """
        if quantized_matrix.dim() != 2:
            raise ValueError(f"输入必须是二维矩阵，当前维度: {quantized_matrix.dim()}")

        M, N = quantized_matrix.shape

        # 根据量化轴确定分块方式
        if self.axis == 0:  # 行方向量化
            return self._dequantize_by_rows(quantized_matrix, scale_matrix, M, N)
        else:  # 列方向量化
            return self._dequantize_by_cols(quantized_matrix, scale_matrix, M, N)

    def _dequantize_by_cols(self, quantized_matrix: torch.Tensor, scale_matrix: torch.Tensor, M: int, N: int) -> torch.Tensor:
        """
        按列方向反量化矩阵
        """
        num_blocks = (N + self.block_size - 1) // self.block_size
        dequantized_matrix = torch.zeros_like(quantized_matrix, dtype=torch.float32)

        for i in range(num_blocks):
            start = i * self.block_size
            end = min((i + 1) * self.block_size, N)
            block = quantized_matrix[:, start:end]
            scale = scale_matrix[:, i:i+1]
            dequantized_matrix[:, start:end] = block * scale

        return dequantized_matrix

    def _dequantize_by_rows(self, quantized_matrix: torch.Tensor, scale_matrix: torch.Tensor, M: int, N: int) -> torch.Tensor:
        """
        按行方向反量化矩阵
        """
        num_blocks = (M + self.block_size - 1) // self.block_size
        dequantized_matrix = torch.zeros_like(quantized_matrix, dtype=torch.float32)

        for i in range(num_blocks):
            start = i * self.block_size
            end = min((i + 1) * self.block_size, M)
            block = quantized_matrix[start:end, :]
            scale = scale_matrix[i:i+1, :]
            dequantized_matrix[start:end, :] = block * scale

        return dequantized_matrix

class GenInputAndGolden():
    def __init__(self, **kwargs):
        self.bs = kwargs['bs']
        self.h = kwargs['h']
        self.n = kwargs['n']
        self.quant_dtype = kwargs['quant_dtype']
        self.is_x_random = kwargs['is_x_random']
        self.global_bs = kwargs['global_bs']
        self.out_dtype = kwargs['out_dtype']
        self.moe_expert_num = kwargs['moe_expert_num']
        self.is_topk_balance = kwargs['is_topk_balance']
        self.k = kwargs['k']
        self.is_mask = kwargs['is_mask']
        self.is_unfixed_real_token = kwargs['is_unfixed_real_token']
        self.ep_world_size = kwargs['ep_world_size']
        self.is_variable_bs = kwargs['is_variable_bs']
        self.expert_scales_dtype = kwargs['expert_scales_dtype']
        self.local_moe_expert_num = kwargs['local_moe_expert_num']
        self.quant_mode = kwargs['quant_mode']
        self.transpose_weight = kwargs['transpose_weight']
        self.comm_quant_mode = kwargs['comm_quant_mode']
        self.weight_dtype = kwargs['weight_dtype']
        self.swiglu_quant_dtype = kwargs['swiglu_quant_dtype']
        self.is_interleaved = kwargs.get('is_interleaved', False)
        if self.is_interleaved:
            # gmm1 右矩阵将按 [128 gate | 128 up] 每 256 列交织排布，对 h 和 n 都要求 256 对齐
            assert self.h % 256 == 0, f"is_interleaved=True 要求 h ({self.h}) 256 对齐"
            assert self.n % 256 == 0, f"is_interleaved=True 要求 n ({self.n}) 256 对齐"
            print(f'[INTERLEAVE CHECK in GenInputAndGolden] h={self.h} 256对齐 ✓, n={self.n} 256对齐 ✓; '
                  f'tile 数={self.n // 256}, 每 tile = 128 gate + 128 up')
        self.x = self.gen_x()
        self.expert_ids = self.gen_expert_ids()
        self.x_active_mask = self.gen_x_active_mask()
        self.expert_scales = self.gen_expert_scales()
        self.a_reordered_list, self.token_indices_list, self.prob_list, self.a_scale_list, self.expert_token_nums = self.gen_dispatch_golden(torch.cat(self.expert_ids, dim=0), torch.cat(self.x, dim=0))
        self.c2_fp32_list, self.weight1, self.weight2, self.weight_scales1, self.weight_scales2, self.weight1_scale_npu, self.weight2_scale_npu, self.w1_npu, self.w2_npu = self.gen_ffn_golden()
        self.x_final = self.gen_combine_golden(torch.cat(self.expert_scales, dim=0), torch.cat(self.expert_ids, dim=0))
    
    def chunk_tensor(self, tensor, chunks):
        return list(tensor.chunk(chunks))

    def gen_x(self):
        if self.is_x_random:
            cur_x = torch.empty(size=[self.global_bs, self.h], dtype=self.out_dtype).uniform_(-0.1, 0.1)
        else:
            cur_x = torch.arange(1, self.global_bs + 1, dtype=self.out_dtype).view(-1, 1).repeat(1, self.h)
        return self.chunk_tensor(cur_x, self.ep_world_size)

    def gen_expert_ids(self):
        def gen_random_token_expert_ids(length, from_, to_):
            return (torch.randperm(to_ - from_, dtype=torch.int32) + from_)[:length]

        def gen_balanced_expert_ids_element(round_index):
            return torch.arange(round_index, round_index + self.moe_expert_num, dtype=torch.int32) % self.moe_expert_num

        if self.is_topk_balance:
            ep_expert_ids = [gen_balanced_expert_ids_element(round_index) for round_index in range((self.global_bs * self.k) // self.moe_expert_num + 1)]
        else:
            ep_expert_ids = [gen_random_token_expert_ids(self.k, 0, self.moe_expert_num) for _ in range(self.global_bs)]

        return self.chunk_tensor(torch.cat(ep_expert_ids)[:self.global_bs * self.k].view([self.global_bs, self.k]), self.ep_world_size)

    def gen_x_active_mask(self):
        def gen_card_active_mask():
            min_real_token_num = 0 if self.is_mask else 1
            real_token_num = torch.randint(min_real_token_num, self.bs + 1, (1,)).item() if self.is_unfixed_real_token else self.bs
            return torch.cat(
                (torch.ones(real_token_num), torch.zeros(self.bs - real_token_num))
            ).to(torch.bool)

        x_active_mask = [gen_card_active_mask() for _ in range(self.ep_world_size)]
        if self.is_variable_bs:
            upper_bs_card_ep_id = torch.randint(0, self.ep_world_size, (1,)).item()
            x_active_mask[upper_bs_card_ep_id] = torch.full((self.bs,), 1).to(torch.bool)

        return self.chunk_tensor(torch.cat(x_active_mask), self.ep_world_size)

    def gen_expert_scales(self):
        return self.chunk_tensor(torch.empty(size=[self.global_bs, self.k], dtype=self.expert_scales_dtype).uniform_(-0.1, 0.1), self.ep_world_size)

    def gen_weight_data(self, row, col, data_type, a, b):
        # 生成 [a, b] 范围内的均匀分布
        if data_type == torch.int8:
            matrix = torch.randint(a, b, (self.ep_world_size * self.local_moe_expert_num , row, col), dtype=data_type)
        else:
            matrix = a + (b - a) * torch.rand(self.ep_world_size * self.local_moe_expert_num , row, col, dtype=data_type)

        return self.chunk_tensor(matrix, self.ep_world_size)

    def gen_weight_scale_data(self, col, data_type, a, b):
        # 生成 [a, b] 范围内的均匀分布
        matrix = a + (b - a) * torch.rand(self.ep_world_size * self.local_moe_expert_num , col, dtype=data_type)

        return self.chunk_tensor(matrix, self.ep_world_size)
    
    def swish(self, x):
        return x * torch.sigmoid(x)
    
    def gen_data_fp8(self, m, k, trans, data_format='E4M3'):
        has_fp8 = True
        data_format = data_format
        a, b = 0, 0.1 # 目标范围
        # 生成 [a, b] 范围内的均匀分布
        matrix = a + (b - a) * torch.rand(m, k, dtype=torch.float32)
        # matrix = torch.randn((m, k), dtype=torch.float32)

        quantizer = MXFP8MatrixQuantizer(
            data_format=data_format,
            axis=0,
            block_size=32
        )

        # 量化
        quantized_matrix, scale_matrix = quantizer.quantize_matrix(matrix)

        # 转换为FP8类型
        if has_fp8:
            if data_format == 'E4M3':
                quantized_matrix = quantized_matrix.to(torch.float8_e4m3fn)
            else:  # E5M2
                quantized_matrix = quantized_matrix.to(torch.float8_e5m2)
            scale_matrix = scale_matrix.to(torch.float8_e8m0fnu)

        # 反量化
        dequantized_matrix = quantizer.dequantize_matrix(
            quantized_matrix.to(torch.float32),
            scale_matrix.to(torch.float32)
        )

        if trans:
            quantized_matrix = quantized_matrix.t()
            dequantized_matrix = dequantized_matrix.t()
        return quantized_matrix, scale_matrix, dequantized_matrix if self.quant_mode > 0 else matrix

    def gen_dispatch_golden(self, expert_ids, x):
        w_dtype = self.quant_dtype
        a_reordered_list = []
        token_indices_list = []
        prob_list = []
        
        input_tokens_fp32 = x.to(torch.float32)
        row_idx = torch.zeros(self.global_bs, self.k, dtype=torch.long)

        expert_token_nums = []
        current_row = 0
        
        for g in range(self.moe_expert_num):
            token_indices = []
            group_probs = []
            
            for idx in range(self.global_bs):
                g_positions = torch.where(expert_ids[idx] == g)[0]
                for pos in g_positions:
                    token_indices.append(idx)
                    group_probs.append(expert_ids[idx, pos])

            for i, idx in enumerate(token_indices):
                token_positions = torch.where(expert_ids[idx] == g)[0]
                for pos in token_positions:
                    row_idx[idx, pos] = current_row + i

            current_row += len(token_indices)

            token_indices = torch.tensor(token_indices, dtype=torch.long)
            
            if len(token_indices) > 0:
                a_reordered_list.append(input_tokens_fp32[token_indices])
                token_indices_list.append(token_indices)
                prob_list.append(torch.tensor(group_probs, dtype=torch.float32))
            else:
                a_reordered_list.append(torch.empty(0, self.h, dtype=torch.float32))
                token_indices_list.append(torch.empty(0, dtype=torch.long))
                prob_list.append(torch.empty(0, dtype=torch.float32))
            expert_token_nums.append(len(a_reordered_list))
        expert_token_nums = [(expert_ids == i).sum().item() for i in range(self.moe_expert_num)]
        if self.quant_mode == 4:
            a_quant_list = []
            a_scale_list = []
            for g in range(self.moe_expert_num):
                if a_reordered_list[g].shape[0] > 0:
                    # a_quant_g, a_scale_g = quantizer.quantize_matrix(a_reordered_list[g])
                    a_quant_g, a_scale_g = dispatch_mx_quantize(a_reordered_list[g].numpy(), mx_ele_dtype=w_dtype,
                                        axis=-1, block_size=32, round_mode="rint")
                    a_quant_list.append(a_quant_g)
                    a_scale_list.append(a_scale_g)
                else:
                    a_quant_list.append(np.empty((0, self.h), dtype=np.float32))
                    a_scale_list.append(np.empty((0, (self.h + 31) // 32), dtype=np.float32))

        return a_reordered_list if self.quant_mode == 0  else a_quant_list, token_indices_list, prob_list, None if self.quant_mode == 0 else a_scale_list, self.chunk_tensor(torch.tensor(expert_token_nums), self.ep_world_size)
    
    def gen_ffn_golden(self):
        params_out_dtype = torch.bfloat16
        if self.transpose_weight == 1:
            weight1 = regenerate_quant_data((self.moe_expert_num, self.n, self.h), params_out_dtype)
            weight2 = regenerate_quant_data((self.moe_expert_num, self.h, self.n // 2), params_out_dtype)
        else:
            weight1 = regenerate_quant_data((self.moe_expert_num, self.h, self.n), params_out_dtype)
            weight2 = regenerate_quant_data((self.moe_expert_num, self.n // 2, self.h), params_out_dtype)
        w_dtype = self.weight_dtype
        w_scale_dtype = "float8_e8m0"
        np_quant_mode = eval(f"numpy_{w_dtype}()")
        np_e8m0 = eval(f"numpy_{w_scale_dtype}()")

        weight1, weight_scale1, weight_scale1_npu_cpu, w1_npu = qbmm_mxfp8_quant_data_prepare(weight1, w_dtype, self.transpose_weight)
        weight2, weight_scale2, weight_scale2_npu_cpu, w2_npu = qbmm_mxfp8_quant_data_prepare(weight2, w_dtype, self.transpose_weight)

        # === 交织 gmm1 右矩阵给 NPU（不影响 golden）===
        # 只动 NPU 消费的两个数组：w1_npu 和 weight_scale1_npu_cpu
        # 不动 weight1（post-transpose） 和 weight_scale1（post-reshape）—— 这两个给 golden 用，保持非交织
        # 这样 golden 在原始 [gate||up] 布局上算 swiglu，NPU 在交织布局上做 1 次 mmad+swiglu，
        # 最终输出语义一致，精度可比对。
        # 注意只处理 weight1，weight2 不需要交织（GMM2 没 swiglu）。
        if self.is_interleaved:
            # w1_npu 是 pre-transpose 的 (E, n, h) for transpose_weight=1；(E, h, n) for transpose_weight=0
            # 我们沿 weight 的 "N" 轴（对应 gmm1 输出 N 维）做 [gate||up] → 每 256 列 [128 gate | 128 up]
            # transpose_weight=1：N 在 axis=1；transpose_weight=0：N 在 axis=2
            w_n_axis = 1 if self.transpose_weight == 1 else 2
            print(f'[INTERLEAVE] 交织前 w1_npu shape={w1_npu.shape}, weight_scale1_npu_cpu shape={weight_scale1_npu_cpu.shape}')
            w1_npu = interleave_gate_up_along_axis(w1_npu, n_axis=w_n_axis, interleave_tile=128)
            # scale 的 N 轴和 weight 的 N 轴位置一致（mx_quantize 不改 weight axis 顺序，只在 K 轴分块）
            weight_scale1_npu_cpu = interleave_gate_up_along_axis(weight_scale1_npu_cpu, n_axis=w_n_axis, interleave_tile=128)
            print(f'[INTERLEAVE] 交织后 w1_npu shape={w1_npu.shape}, weight_scale1_npu_cpu shape={weight_scale1_npu_cpu.shape}')
        w1_npu = np.array_split(np.stack(w1_npu, axis=0), self.ep_world_size)
        w2_npu = np.array_split(np.stack(w2_npu, axis=0), self.ep_world_size)
        weight1_list = list(weight1)
        weight2_list = list(weight2)
        weight1_scale_list = list(weight_scale1)
        weight2_scale_list = list(weight_scale2)
        x_list = self.a_reordered_list
        x_scale_list = self.a_scale_list

        print("---------------start to caculate Matmul_1 ------------- ")
        output = qbmm_mx_golden_calculate(x_list, weight1_list, weight1_scale_list, x_scale_list)
        output = golden_output_process(output, 2, 0)

        print("---------------start to process swiglu ------------- ")
        swiglu_out = swiglu(output[0])
        swiglu_out = swiglu_out.to(torch.bfloat16)
        swiglu_out = swiglu_out.to(torch.float32)
        print("---------------start to process swiglu_quant ------------- ")
        quant_output, quant_scale_output = mx_quantize(swiglu_out.numpy(), mx_ele_dtype=self.swiglu_quant_dtype, axis=-1)
        x_transpose_flag = 0
        perTokenScale_input = np.concatenate([quant_scale_output], axis=-2 if x_transpose_flag == 1 else -1)
        perTokenScale_input = perTokenScale_input.reshape(perTokenScale_input.shape[0], perTokenScale_input.shape[1] * 2)
        perTokenScale_input = transpose_last_two_dim(perTokenScale_input, x_transpose_flag)
        split_sizes = [t.shape[0] for t in self.a_reordered_list]
        split_indices = np.cumsum([0] + split_sizes)
        quant_output = [quant_output[split_indices[i] : split_indices[i+1]] for i in range(len(split_indices)-1)]
        quant_scale_output = [perTokenScale_input[split_indices[i] : split_indices[i+1]] for i in range(len(split_indices)-1)]
        print("---------------start to caculate Matmul_2 ------------- ")
        output = qbmm_mx_golden_calculate(quant_output, weight2_list, weight2_scale_list, quant_scale_output)
        weight1_list = np.array_split(np.stack(weight1_list, axis=0), self.ep_world_size)
        weight2_list = np.array_split(np.stack(weight2_list, axis=0), self.ep_world_size)

        weight1_scale_list = np.array_split(weight_scale1_npu_cpu, self.ep_world_size)
        weight2_scale_list = np.array_split(weight_scale2_npu_cpu, self.ep_world_size)
        weight1_scale_list_npu = np.array_split(weight_scale1, self.ep_world_size)
        weight2_scale_list_npu = np.array_split(weight_scale2, self.ep_world_size)
        return output, weight1_list, weight2_list, weight1_scale_list, weight2_scale_list, weight1_scale_list_npu, weight2_scale_list_npu, w1_npu, w2_npu
    
    def gen_combine_golden(self, expert_scales, expert_ids):
        c2_final = torch.zeros(self.global_bs, self.h, dtype=torch.float32)
        w_dtype = "float8_e4m3fn" if self.comm_quant_mode == 4 else "float8_e5m2"

        for g in range(self.moe_expert_num):
            if len(self.token_indices_list[g]) > 0:
                indices = self.token_indices_list[g]
                
                for i, idx in enumerate(indices):
                    expert_scale_cur = expert_scales[idx][torch.where(expert_ids[idx] == g)[0].item()]
                    
                    if self.comm_quant_mode == 3 or self.comm_quant_mode == 4:
                        c2_data = self.c2_fp32_list[g][i]
                        c2_quant, c2_scale = dispatch_mx_quantize(
                            c2_data.unsqueeze(0).numpy(),
                            mx_ele_dtype=w_dtype,
                            axis=-1,
                            block_size=32,
                            round_mode="rint"
                        )
                        c2_scale_broadcast = np.repeat(c2_scale, 32, axis=-1)
                        c2_dequant = torch.from_numpy(
                            (c2_quant.astype(np.float32) * c2_scale_broadcast.astype(np.float32)).squeeze(0)
                        )
                        c2_final[idx] += c2_dequant * expert_scale_cur
                    else:
                        c2_final[idx] += self.c2_fp32_list[g][i] * expert_scale_cur
        return self.chunk_tensor(c2_final, self.ep_world_size)