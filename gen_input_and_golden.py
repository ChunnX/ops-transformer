import torch
import math
import numpy
import numpy as np
import copy
import gc
import re
import importlib.metadata as importlib_metadata
from packaging import version
import subprocess
import sys
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor


def get_process_cpu_capacity():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def calculate_process_cpu_profile(cpu_seconds, wall_seconds, cpu_capacity):
    if wall_seconds <= 0:
        return 0.0, 0.0
    average_cores = cpu_seconds / wall_seconds
    utilization = average_cores / max(1, cpu_capacity) * 100.0
    return average_cores, utilization


def calculate_parallel_chunk_size(chunk_size, expert_num, worker_count):
    chunk_size = max(1, min(int(chunk_size), int(expert_num)))
    worker_count = max(1, min(int(worker_count), int(expert_num)))
    if chunk_size < worker_count:
        return chunk_size
    return max(worker_count, chunk_size // worker_count * worker_count)


def swiglu(gmm_out, clamp_limit=None):
    # 将结果分成两部分以应用 SwiGLU 激活函数
    c_temp4, gate = gmm_out.chunk(2, dim=-1)
    if clamp_limit is not None:
        c_temp4 = torch.clamp(c_temp4, max=clamp_limit)
        gate = torch.clamp(gate, min=-clamp_limit, max=clamp_limit)
    c_temp5 = c_temp4 * torch.sigmoid(c_temp4)  # SwiGLU 激活
    c_temp6 = c_temp5 * gate  # 与门控值进行逐元素相乘
    return c_temp6


def swiglu_step(gmm_out, clamp_limit=None):
    x0, x1 = gmm_out.chunk(2, dim=-1)
    x0 = x0 * torch.sigmoid(x0)
    if clamp_limit is not None:
        x0 = torch.clamp(x0, max=clamp_limit)
        x1 = torch.clamp(x1, min=-clamp_limit, max=clamp_limit)
    return x0 * x1


def swiglu_oai(gmm_out, clamp_limit=None, alpha=1.702, beta=1.0):
    gate, up = gmm_out.chunk(2, dim=-1)
    if clamp_limit is not None:
        gate = torch.clamp(gate, max=clamp_limit)
        up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + beta) * glu


def situ(gmm_out, beta=1.0, linear_beta=None):
    gate, up = gmm_out.chunk(2, dim=-1)
    gate = gate.to(torch.float32)
    up = up.to(torch.float32)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return gate * up


def apply_golden_activation(
    gmm_out,
    activation_type,
    clamp_limit=None,
    swiglu_alpha=1.702,
    swiglu_beta=1.0,
    situ_beta=1.0,
    situ_linear_beta=None,
):
    if activation_type == 'swiglu':
        return swiglu(gmm_out, clamp_limit)
    if activation_type == 'swiglustep':
        return swiglu_step(gmm_out, clamp_limit)
    if activation_type == 'swigluoai':
        return swiglu_oai(
            gmm_out, clamp_limit, swiglu_alpha, swiglu_beta
        )
    if activation_type == 'situ':
        return situ(gmm_out, situ_beta, situ_linear_beta)
    raise ValueError(f"unsupported golden activation_type: {activation_type}")


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

def golden_output_process(output, param_split_item, groupType, activation_feature_out=None, dyn_scale_out=None, log_stats=True):
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
        if log_stats:
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

def _to_numpy_float32(value):
    if isinstance(value, torch.Tensor):
        return value.numpy().astype(np.float32)
    return value.astype(np.float32)


def mx_arrays_to_raw_tensors(arrays):
    raw_tensors = []
    for value in arrays:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            raw_tensors.append(tensor if tensor.dtype == torch.uint8 else tensor.view(torch.uint8))
            continue
        array = np.ascontiguousarray(value)
        raw_tensors.append(torch.from_numpy(array.view(np.uint8)))
    return raw_tensors


def mx_array_to_torch(value, dtype):
    if isinstance(value, torch.Tensor):
        tensor = value.contiguous()
        if tensor.dtype == dtype:
            return tensor
    else:
        array = np.ascontiguousarray(value)
        tensor = torch.from_numpy(array.view(np.uint8))

    target_item_size = torch.empty((), dtype=dtype).element_size()
    if tensor.element_size() != target_item_size:
        raise ValueError(
            f"Cannot restore MX storage with item size {tensor.element_size()} "
            f"as {dtype} with item size {target_item_size}"
        )
    return tensor.view(dtype)


def _dequant_qbmm_input(input_value, scale_value):
    x1 = _to_numpy_float32(input_value)
    if scale_value is None:
        return x1

    if isinstance(scale_value, torch.Tensor):
        pertoken_scale_i = scale_value.numpy()
    else:
        pertoken_scale_i = scale_value
    pertoken_scale_mx_broadcast = np.repeat(pertoken_scale_i, 32, axis=-1)
    if pertoken_scale_mx_broadcast.shape[-1] < x1.shape[-1]:
        raise ValueError(
            f"input K dim ({x1.shape[-1]}) is larger than scale broadcast K dim "
            f"({pertoken_scale_mx_broadcast.shape[-1]})"
        )
    pertoken_scale_mx_broadcast = pertoken_scale_mx_broadcast[..., :x1.shape[-1]]
    return x1 * pertoken_scale_mx_broadcast


def _dequant_qbmm_weight(weight_value, scale_value, unit_weight_scale=False):
    x2 = _to_numpy_float32(weight_value)
    if scale_value is None:
        return x2

    if isinstance(scale_value, torch.Tensor):
        scale_i = scale_value.numpy()
    else:
        scale_i = scale_value
    if unit_weight_scale:
        return x2

    is_unit_scale = (
        scale_i.size > 0
        and float(scale_i.reshape(-1)[0]) == 1.0
        and np.all(scale_i == scale_i.reshape(-1)[0])
    )
    if is_unit_scale:
        return x2

    deq_scale_mx_broadcast = np.repeat(scale_i, 32, axis=-2)
    if deq_scale_mx_broadcast.shape[-2] < x2.shape[-2]:
        raise ValueError(
            f"weight K dim ({x2.shape[-2]}) is larger than scale broadcast K dim "
            f"({deq_scale_mx_broadcast.shape[-2]})"
        )
    scale_slices = [slice(None)] * deq_scale_mx_broadcast.ndim
    scale_slices[-2] = slice(0, x2.shape[-2])
    deq_scale_mx_broadcast = deq_scale_mx_broadcast[tuple(scale_slices)]
    return x2 * deq_scale_mx_broadcast


def qbmm_mx_golden_calculate(
    x, weight, scale, pertoken_scale, has_bias=0, bias=None, bias_dtype=None, offset=None, act_type=0,
    batch_matmul=False, batch_matmul_min_m=128, unit_weight_scale=False, expert_workers=1,
    expert_executor=None,
):
    output = [None] * len(x)
    if batch_matmul and has_bias == 0 and len(x) > 1:
        row_groups = {}
        for i, x_i in enumerate(x):
            row_groups.setdefault(x_i.shape[0], []).append(i)

        for row_count, group_indices in row_groups.items():
            if row_count >= batch_matmul_min_m and len(group_indices) > 1:
                x_batch = np.stack([
                    _dequant_qbmm_input(x[i], None if pertoken_scale is None else pertoken_scale[i])
                    for i in group_indices
                ], axis=0)
                weight_batch = np.stack([
                    _dequant_qbmm_weight(weight[i], scale[i], unit_weight_scale)
                    for i in group_indices
                ], axis=0)
                batch_output = torch.bmm(
                    torch.from_numpy(x_batch).to(torch.float),
                    torch.from_numpy(weight_batch).to(torch.float)
                )
                for batch_idx, output_idx in enumerate(group_indices):
                    output[output_idx] = batch_output[batch_idx]

        if all(item is not None for item in output):
            return output

    pending_indices = [i for i in range(len(x)) if output[i] is None]

    def calculate_one(i):
        if x[i].shape[0] == 0:
            output_shape = tuple(x[i].shape[:-1]) + (weight[i].shape[-1],)
            return i, torch.empty(output_shape, dtype=torch.float32)
        x1 = _dequant_qbmm_input(x[i], None if pertoken_scale is None else pertoken_scale[i])
        x2 = _dequant_qbmm_weight(weight[i], scale[i], unit_weight_scale)
        
        x1 = torch.from_numpy(x1).to(torch.float)
        x2 = torch.from_numpy(x2).to(torch.float)
        
        out_i = torch.matmul(x1, x2)
        if has_bias == 1:
            out_i = torch.add(out_i, bias[i])
        return i, out_i

    expert_workers = max(1, int(expert_workers or 1))
    if expert_workers > 1 and len(pending_indices) > 1:
        if expert_executor is not None:
            for i, out_i in expert_executor.map(calculate_one, pending_indices):
                output[i] = out_i
        else:
            worker_count = min(expert_workers, len(pending_indices))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                for i, out_i in executor.map(calculate_one, pending_indices):
                    output[i] = out_i
        return output

    for i in pending_indices:
        _, output[i] = calculate_one(i)
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

def transpose_last_two_dim(input, transpose_flag, log=False):
    def np_transpose_last_two_dim(input, transpose_flag):
        if transpose_flag == 1:
            # 转置场景
            if log:
                print("The transpose flag is true, need to trans last 2 dim")
            # 获取当前维度数
            ndim = input.ndim
            # 构造新的轴顺序：保持前面的所有维度不变，最后两个交换
            axes = list(range(ndim - 2)) + [ndim - 1, ndim - 2]
            # 进行转置
            return np.transpose(input, axes=axes)
        else:
            if log:
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
            if log:
                print("The transpose flag is false, no need to trans")
            return input

    if isinstance(input, torch.Tensor):
        return torch_transpose_last_two_dim(input, transpose_flag)
    elif isinstance(input, np.ndarray):
        return np_transpose_last_two_dim(input, transpose_flag)
    else:
        raise TypeError("不支持的数据类型")


def interleave_gate_up_along_axis(arr, n_axis, interleave_tile=128):
    """Convert ``[gate | up]`` into tiled ``[gate | up]`` pairs along N.

    The first half of the selected axis contains all gate columns and the
    second half contains all up columns.  For the default tile size, the NPU
    layout becomes ``[128 gate, 128 up, 128 gate, 128 up, ...]``.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"arr must be a numpy.ndarray, got {type(arr).__name__}")
    if interleave_tile <= 0:
        raise ValueError(f"interleave_tile must be positive, got {interleave_tile}")

    n_axis = n_axis % arr.ndim
    n_total = arr.shape[n_axis]
    tile_pair = interleave_tile * 2
    if n_total % tile_pair != 0:
        raise ValueError(
            f"N-axis length {n_total} must be divisible by {tile_pair} "
            "for gate/up interleaving"
        )

    output_n = n_total // 2
    num_tiles = output_n // interleave_tile
    arr_moved = np.moveaxis(arr, n_axis, 0)
    other_shape = arr_moved.shape[1:]
    gate = arr_moved[:output_n].reshape((num_tiles, interleave_tile) + other_shape)
    up = arr_moved[output_n:].reshape((num_tiles, interleave_tile) + other_shape)
    interleaved = np.stack((gate, up), axis=1).reshape((n_total,) + other_shape)
    return np.moveaxis(interleaved, 0, n_axis)


def validate_interleaved_weight_dimensions(h, n, is_interleaved):
    if not is_interleaved:
        return
    if h % 256 != 0 or n % 256 != 0:
        raise ValueError(
            f"is_interleaved=True requires h and n to be 256-aligned, got h={h}, n={n}"
        )


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

def _prepare_fast_weight_from_npu(w_npu, weight_transpose_flag):
    scale_dtype_np = numpy_float8_e8m0()
    w_shape = w_npu.shape
    axis = -2 if weight_transpose_flag == 0 else -1
    axis = axis % len(w_shape)
    scale_shape = list(w_shape)
    scale_block_num = (scale_shape[axis] + 31) // 32
    if scale_block_num % 2 != 0:
        scale_block_num += 1
    scale_shape[axis] = scale_block_num // 2
    scale_shape.append(2)
    Scale_npu = np.ones(scale_shape, dtype=np.float32).astype(scale_dtype_np, copy=False)

    W_input = transpose_last_two_dim(w_npu, weight_transpose_flag)
    if weight_transpose_flag:
        Scale_input = Scale_npu.reshape(Scale_npu.shape[0], Scale_npu.shape[1], Scale_npu.shape[2] * 2)
    else:
        Scale_input_transposed = np.transpose(Scale_npu, axes=(0, 2, 1, 3))
        Scale_input_transposed_reshape = Scale_input_transposed.reshape(
            Scale_input_transposed.shape[0],
            Scale_input_transposed.shape[1],
            Scale_input_transposed.shape[2] * 2
        )
        Scale_input = np.transpose(Scale_input_transposed_reshape, axes=(0, 2, 1))
    Scale_input = transpose_last_two_dim(Scale_input, weight_transpose_flag)

    return W_input, Scale_input, Scale_npu, w_npu


def _make_expert_weight_rng(random_seed, global_expert_id, weight_index):
    random_seed = int(random_seed)
    global_expert_id = int(global_expert_id)
    return np.random.default_rng(np.random.SeedSequence([
        random_seed & 0xFFFFFFFF,
        (random_seed >> 32) & 0xFFFFFFFF,
        global_expert_id & 0xFFFFFFFF,
        (global_expert_id >> 32) & 0xFFFFFFFF,
        int(weight_index),
    ]))


def fast_mxfp8_random_data_prepare(w_shape, w_dtype, weight_transpose_flag, rng=None):
    ele_dtype_np = eval(f"numpy_{w_dtype}()")
    storage_masks = {
        "float8_e4m3fn": 0xBF,
        "float8_e5m2": 0xBF,
        "float4_e2m1": 0x0B,
        "float4_e1m2": 0x0F,
    }
    storage_mask = storage_masks[w_dtype]
    if rng is None:
        random_codes = np.random.randint(0, 256, size=w_shape, dtype=np.uint8)
    else:
        random_codes = rng.integers(0, 256, size=w_shape, dtype=np.uint8)
    np.bitwise_and(random_codes, storage_mask, out=random_codes)
    w_npu = random_codes.view(ele_dtype_np)
    return _prepare_fast_weight_from_npu(w_npu, weight_transpose_flag)


def regenerate_quant_data(w_shape, params_out_dtype, rng=None):
    mu_W = -0.05
    Sigma_W = 4
    # 高斯分布 + 白噪声
    random_source = np.random if rng is None else rng
    W_tmp = (
        random_source.normal(mu_W, Sigma_W, w_shape)
        - random_source.uniform(0.1 * mu_W, -0.1 * mu_W, w_shape)
    ).astype(np.float32)
    W_tmp = np.clip(W_tmp, -1.0, 1.0)
    if params_out_dtype == 'fp32':
        W_tmp = np.abs(W_tmp)
    return W_tmp


def generate_ffn_expert_weight(
    global_expert_id,
    random_seed,
    *,
    h,
    n,
    weight_dtype,
    transpose_weight,
    fast_random_weight,
    **_,
):
    if transpose_weight == 1:
        weight1_shape = (1, n, h)
        weight2_shape = (1, h, n // 2)
    else:
        weight1_shape = (1, h, n)
        weight2_shape = (1, n // 2, h)

    use_fast_random_weight = fast_random_weight and weight_dtype in (
        "float8_e4m3fn", "float8_e5m2", "float4_e2m1", "float4_e1m2"
    )
    weight1_rng = _make_expert_weight_rng(random_seed, global_expert_id, 1)
    weight2_rng = _make_expert_weight_rng(random_seed, global_expert_id, 2)
    if use_fast_random_weight:
        weight1, weight_scale1, weight_scale1_npu_cpu, w1_npu = fast_mxfp8_random_data_prepare(
            weight1_shape, weight_dtype, transpose_weight, rng=weight1_rng
        )
        weight2, weight_scale2, weight_scale2_npu_cpu, w2_npu = fast_mxfp8_random_data_prepare(
            weight2_shape, weight_dtype, transpose_weight, rng=weight2_rng
        )
    else:
        weight1 = regenerate_quant_data(weight1_shape, torch.bfloat16, rng=weight1_rng)
        weight2 = regenerate_quant_data(weight2_shape, torch.bfloat16, rng=weight2_rng)
        weight1, weight_scale1, weight_scale1_npu_cpu, w1_npu = qbmm_mxfp8_quant_data_prepare(
            weight1, weight_dtype, transpose_weight
        )
        weight2, weight_scale2, weight_scale2_npu_cpu, w2_npu = qbmm_mxfp8_quant_data_prepare(
            weight2, weight_dtype, transpose_weight
        )
    return (
        weight1,
        weight2,
        weight_scale1,
        weight_scale2,
        weight_scale1_npu_cpu,
        weight_scale2_npu_cpu,
        w1_npu,
        w2_npu,
        use_fast_random_weight,
    )


def generate_ffn_weight_chunk(
    expert_start,
    cur_expert_num,
    random_seed,
    h,
    n,
    w_dtype,
    transpose_weight,
    fast_random_weight,
):
    expert_weights = [
        generate_ffn_expert_weight(
            expert_start + local_expert_id,
            random_seed,
            h=h,
            n=n,
            weight_dtype=w_dtype,
            transpose_weight=transpose_weight,
            fast_random_weight=fast_random_weight,
        )
        for local_expert_id in range(cur_expert_num)
    ]
    combined = [
        np.concatenate([expert_weight[index] for expert_weight in expert_weights], axis=0)
        for index in range(8)
    ]
    return (*combined, expert_weights[0][8])


def iter_rank_npu_weight_chunks(rank, random_seed, **kwargs):
    h = kwargs["h"]
    n = kwargs["n"]
    ep_world_size = kwargs["ep_world_size"]
    local_moe_expert_num = kwargs["local_moe_expert_num"]
    shared_expert_num = max(0, int(kwargs.get("shared_expert_num", 0)))
    expert_per_rank = local_moe_expert_num + shared_expert_num
    transpose_weight = kwargs["transpose_weight"]
    is_interleaved = bool(kwargs.get("is_interleaved", False))
    validate_interleaved_weight_dimensions(h, n, is_interleaved)
    weight_dtype = kwargs["weight_dtype"]
    fast_random_weight = kwargs.get("fast_random_weight", False)
    chunk_size = max(1, min(kwargs.get("golden_ffn_chunk_size", 16), expert_per_rank))
    if rank < 0 or rank >= ep_world_size:
        raise ValueError(f"rank must be in [0, {ep_world_size}), got {rank}")

    rank_start = rank * expert_per_rank
    rank_end = rank_start + expert_per_rank
    shared_start = rank_start
    moe_start = rank_start + shared_expert_num
    def get_overlap(chunk, chunk_start, destination_start, destination_count):
        chunk_end = chunk_start + chunk.shape[0]
        destination_end = destination_start + destination_count
        overlap_start = max(chunk_start, destination_start)
        overlap_end = min(chunk_end, destination_end)
        if overlap_start >= overlap_end:
            return None
        return (
            overlap_start - destination_start,
            chunk[overlap_start - chunk_start:overlap_end - chunk_start],
        )

    for start in range(rank_start, rank_end, chunk_size):
        end = min(start + chunk_size, rank_end)
        (
            weight1,
            weight2,
            weight_scale1,
            weight_scale2,
            weight_scale1_npu,
            weight_scale2_npu,
            w1_npu,
            w2_npu,
            _,
        ) = generate_ffn_weight_chunk(
            start,
            end - start,
            random_seed,
            h,
            n,
            weight_dtype,
            transpose_weight,
            fast_random_weight,
        )
        w1_npu_standalone = w1_npu
        weight_scale1_npu_standalone = weight_scale1_npu
        if is_interleaved:
            w_n_axis = 1 if transpose_weight == 1 else 2
            w1_npu = interleave_gate_up_along_axis(w1_npu, n_axis=w_n_axis)
            weight_scale1_npu = interleave_gate_up_along_axis(
                weight_scale1_npu, n_axis=w_n_axis
            )

        for partition, destination_start, destination_count in (
            ("moe", moe_start, local_moe_expert_num),
            ("shared", shared_start, shared_expert_num),
        ):
            if destination_count == 0:
                continue
            overlap = get_overlap(w1_npu, start, destination_start, destination_count)
            if overlap is None:
                continue
            destination_offset, w1_part = overlap
            source_offset = destination_start + destination_offset - start
            source_end = source_offset + w1_part.shape[0]
            yield {
                "partition": partition,
                "offset": destination_offset,
                "count": destination_count,
                "w1_npu": w1_part,
                "w2_npu": w2_npu[source_offset:source_end],
                "weight_scales1": weight_scale1_npu[source_offset:source_end],
                "weight_scales2": weight_scale2_npu[source_offset:source_end],
                "w1_npu_standalone": w1_npu_standalone[source_offset:source_end],
                "weight_scales1_standalone": weight_scale1_npu_standalone[
                    source_offset:source_end
                ],
            }

        del (
            weight1,
            weight2,
            weight_scale1,
            weight_scale2,
            weight_scale1_npu,
            weight_scale2_npu,
            w1_npu,
            w2_npu,
            w1_npu_standalone,
            weight_scale1_npu_standalone,
        )


def generate_rank_npu_weights(rank, random_seed, **kwargs):
    outputs = {
        "w1_npu": None,
        "w2_npu": None,
        "weight_scales1": None,
        "weight_scales2": None,
        "shared_l1_weights": None,
        "shared_l2_weights": None,
        "shared_l1_weights_sf": None,
        "shared_l2_weights_sf": None,
        "w1_npu_standalone": None,
        "weight_scales1_standalone": None,
        "shared_l1_weights_standalone": None,
        "shared_l1_weights_sf_standalone": None,
    }
    output_names = {
        "moe": (
            "w1_npu",
            "w2_npu",
            "weight_scales1",
            "weight_scales2",
            "w1_npu_standalone",
            "weight_scales1_standalone",
        ),
        "shared": (
            "shared_l1_weights",
            "shared_l2_weights",
            "shared_l1_weights_sf",
            "shared_l2_weights_sf",
            "shared_l1_weights_standalone",
            "shared_l1_weights_sf_standalone",
        ),
    }
    chunk_names = (
        "w1_npu",
        "w2_npu",
        "weight_scales1",
        "weight_scales2",
        "w1_npu_standalone",
        "weight_scales1_standalone",
    )
    for chunk in iter_rank_npu_weight_chunks(rank, random_seed, **kwargs):
        for output_name, chunk_name in zip(output_names[chunk["partition"]], chunk_names):
            value = chunk[chunk_name]
            if outputs[output_name] is None:
                outputs[output_name] = np.empty(
                    (chunk["count"],) + value.shape[1:], dtype=value.dtype
                )
            start = chunk["offset"]
            outputs[output_name][start:start + value.shape[0]] = value
    return outputs

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
    @staticmethod
    def print_golden_debug_values(marker, name, values, count):
        if isinstance(values, torch.Tensor):
            sample = values.detach().cpu().reshape(-1)[:count].to(torch.float32).tolist()
        else:
            sample = np.asarray(values).reshape(-1)[:count].astype(np.float32).tolist()
        print(f"[golden {marker}] {name}: {sample}")

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
        rank_bs = kwargs.get('rank_bs')
        self.has_explicit_rank_bs = rank_bs is not None
        if self.has_explicit_rank_bs:
            self.rank_bs = [int(value) for value in rank_bs]
            if len(self.rank_bs) != self.ep_world_size:
                raise ValueError(
                    f"rank_bs length {len(self.rank_bs)} does not match "
                    f"ep_world_size {self.ep_world_size}"
                )
            if any(value < 1 or value > self.bs for value in self.rank_bs):
                raise ValueError(
                    f"rank_bs values must be in [1, bs={self.bs}], got {self.rank_bs}"
                )
            self.global_bs = sum(self.rank_bs)
        else:
            self.rank_bs = [self.bs] * self.ep_world_size
        self.rank_bs_offset = [0]
        for rank_token_num in self.rank_bs:
            self.rank_bs_offset.append(
                self.rank_bs_offset[-1] + rank_token_num
            )
        self.expert_scales_dtype = kwargs['expert_scales_dtype']
        self.local_moe_expert_num = kwargs['local_moe_expert_num']
        self.expert_per_rank = kwargs.get('expert_per_rank', self.local_moe_expert_num)
        self.shared_expert_num = kwargs.get(
            'shared_expert_num',
            max(0, self.expert_per_rank - self.local_moe_expert_num)
        )
        self.shared_expert_num = max(0, int(self.shared_expert_num))
        self.expert_per_rank = self.local_moe_expert_num + self.shared_expert_num
        self.total_expert_num = self.ep_world_size * self.expert_per_rank
        self.quant_mode = kwargs['quant_mode']
        self.transpose_weight = kwargs['transpose_weight']
        self.is_interleaved = bool(kwargs.get('is_interleaved', False))
        self.a8w8_is_nz = bool(kwargs.get('a8w8_is_nz', False))
        validate_interleaved_weight_dimensions(self.h, self.n, self.is_interleaved)
        self.comm_quant_mode = kwargs['comm_quant_mode']
        self.weight_dtype = kwargs['weight_dtype']
        self.swiglu_quant_dtype = kwargs['swiglu_quant_dtype']
        clamp_limit = kwargs.get('clamp_limit', None)
        self.clamp_limit = (
            None if clamp_limit is None or clamp_limit == -1 else clamp_limit
        )
        self.activation_type = str(kwargs.get('activation_type', 'swiglu')).lower()
        if self.activation_type in ('swiglu/oai', 'swiglu_oai'):
            self.activation_type = 'swigluoai'
        if self.activation_type == 'situglu':
            self.activation_type = 'situ'
        self.swiglu_alpha = float(kwargs.get('swiglu_alpha', 1.702))
        self.swiglu_beta = float(kwargs.get('swiglu_beta', 1.0))
        self.situ_beta = float(kwargs.get('situ_beta', 1.0))
        situ_linear_beta = kwargs.get('situ_linear_beta', None)
        self.situ_linear_beta = None if situ_linear_beta is None else float(situ_linear_beta)
        if self.activation_type not in (
            'swiglu', 'swiglustep', 'swigluoai', 'situ'
        ):
            raise ValueError(
                "activation_type only supports swiglu, swiglustep, "
                f"swigluoai or situ, got {self.activation_type}"
            )
        self.golden_ffn_chunk_size = kwargs.get('golden_ffn_chunk_size', 16)
        self.fast_random_weight = kwargs.get('fast_random_weight', False)
        self.golden_log_interval = kwargs.get('golden_log_interval', 256)
        self.golden_log_stats = kwargs.get('golden_log_stats', False)
        self.golden_batch_matmul = kwargs.get('golden_batch_matmul', True)
        self.golden_batch_matmul_min_m = kwargs.get('golden_batch_matmul_min_m', 128)
        self.golden_expert_workers = kwargs.get('golden_expert_workers', 1)
        self.topk_weights_type = bool(kwargs.get('topk_weights_type', False))
        self.retain_npu_weights = bool(kwargs.get('retain_npu_weights', True))
        self.random_seed = int(kwargs.get('random_seed', 0))
        golden_total_start = time.perf_counter()
        golden_total_cpu_start = time.process_time()
        input_start = time.perf_counter()
        self.x = self.gen_x()
        self.expert_ids = self.gen_expert_ids()
        self.x_active_mask = self.gen_x_active_mask()
        self.expert_scales = self.gen_expert_scales()
        expert_ids_all = torch.cat(self.expert_ids, dim=0)
        expert_scales_all = torch.cat(self.expert_scales, dim=0)
        input_elapsed = time.perf_counter() - input_start

        dispatch_start = time.perf_counter()
        self.a_reordered_list, self.token_indices_list, self.expert_pos_list, self.prob_list, self.a_scale_list, self.expert_token_nums = self.gen_dispatch_golden(expert_ids_all, torch.cat(self.x, dim=0))
        dispatch_elapsed = time.perf_counter() - dispatch_start
        self.combine_golden_accumulator = torch.zeros(self.global_bs, self.h, dtype=torch.float32)
        self._golden_executor = None
        if self.golden_expert_workers > 1:
            self._golden_executor = ThreadPoolExecutor(max_workers=self.golden_expert_workers)
        print(
            "[golden-profile] 配置: "
            f"experts={self.total_expert_num}, "
            f"chunk={self.golden_ffn_chunk_size}, "
            f"expert_workers={self.golden_expert_workers}, "
            f"torch_threads={torch.get_num_threads()}, "
            f"torch_interop_threads={torch.get_num_interop_threads()}, "
            f"可用CPU={get_process_cpu_capacity()}, "
            f"batch_matmul={self.golden_batch_matmul}, "
            f"retain_npu_weights={self.retain_npu_weights}"
        )
        ffn_start = time.perf_counter()
        try:
            ffn_result = self.gen_ffn_golden(
                self.golden_ffn_chunk_size, expert_scales_all, expert_ids_all
            )
        finally:
            if self._golden_executor is not None:
                self._golden_executor.shutdown(wait=True)
                self._golden_executor = None
        ffn_elapsed = time.perf_counter() - ffn_start

        finalize_start = time.perf_counter()
        (
            self.c2_fp32_list,
            self.weight1,
            self.weight2,
            self.weight_scales1,
            self.weight_scales2,
            self.shared_l1_weights_sf,
            self.shared_l2_weights_sf,
            self.weight1_scale_npu,
            self.weight2_scale_npu,
            self.w1_npu,
            self.w2_npu,
            self.shared_l1_weights,
            self.shared_l2_weights,
            self.w1_npu_standalone,
            self.weight_scales1_standalone,
            self.shared_l1_weights_standalone,
            self.shared_l1_weights_sf_standalone,
        ) = ffn_result
        self.x_final = self.split_token_tensor(self.combine_golden_accumulator)
        self.apply_variable_bs_to_inputs()
        self.combine_golden_accumulator = None
        self.release_golden_intermediates()
        finalize_elapsed = time.perf_counter() - finalize_start
        golden_total_elapsed = time.perf_counter() - golden_total_start
        golden_total_cpu = time.process_time() - golden_total_cpu_start
        cpu_capacity = get_process_cpu_capacity()
        average_cores, cpu_utilization = calculate_process_cpu_profile(
            golden_total_cpu,
            golden_total_elapsed,
            cpu_capacity,
        )
        print(
            "[golden-profile] 总体阶段(s): "
            f"输入={input_elapsed:.3f}, "
            f"Dispatch={dispatch_elapsed:.3f}, "
            f"FFN={ffn_elapsed:.3f}, "
            f"收尾={finalize_elapsed:.3f}, "
            f"总计={golden_total_elapsed:.3f}; "
            f"CPU平均={average_cores:.2f}/{cpu_capacity}核 "
            f"({cpu_utilization:.2f}%)"
        )
    
    def chunk_tensor(self, tensor, chunks):
        return list(tensor.chunk(chunks))

    def split_token_tensor(self, tensor):
        return list(torch.split(tensor, self.rank_bs, dim=0))

    def gen_x(self):
        if self.is_x_random:
            cur_x = torch.empty(size=[self.global_bs, self.h], dtype=self.out_dtype).uniform_(-0.1, 0.1)
        else:
            cur_x = torch.arange(1, self.global_bs + 1, dtype=self.out_dtype).view(-1, 1).repeat(1, self.h)
        return self.split_token_tensor(cur_x)

    def gen_expert_ids(self):
        def gen_balanced_expert_ids_element(round_index):
            return torch.arange(round_index, round_index + self.moe_expert_num, dtype=torch.int32) % self.moe_expert_num

        def gen_unbalanced_expert_ids():
            hot_expert_count = min(8, self.moe_expert_num)
            warm_expert_count = min(
                16, max(self.moe_expert_num - hot_expert_count, 0)
            )
            global_cold_expert_ratio = 0.005
            rank_cold_expert_ratio = 0.015
            expert_weights = torch.ones(
                self.moe_expert_num, dtype=torch.float32
            )
            routing_scale = self.moe_expert_num / max(self.k, 1)
            expert_weights[0] = 20.0 * routing_scale
            if hot_expert_count > 1:
                expert_weights[1:hot_expert_count] = torch.logspace(
                    math.log10(max(1.25 * routing_scale, 1.0)),
                    math.log10(max(0.42 * routing_scale, 1.0)),
                    hot_expert_count - 1,
                )
            warm_expert_start = hot_expert_count
            warm_expert_end = warm_expert_start + warm_expert_count
            expert_weights[warm_expert_start:warm_expert_end] = max(
                0.21 * routing_scale, 1.0
            )
            global_cold_expert_count = min(
                round(self.moe_expert_num * global_cold_expert_ratio),
                max(self.moe_expert_num - warm_expert_end, 0),
                max(self.moe_expert_num - self.k, 0),
            )
            if global_cold_expert_count > 0:
                expert_weights[-global_cold_expert_count:] = 0

            ordinary_expert_end = (
                self.moe_expert_num - global_cold_expert_count
            )
            ordinary_expert_indices = torch.arange(
                warm_expert_end, ordinary_expert_end
            )
            rank_cold_expert_count = min(
                round(self.moe_expert_num * rank_cold_expert_ratio),
                ordinary_expert_indices.numel(),
                max(
                    self.moe_expert_num
                    - self.k
                    - global_cold_expert_count,
                    0,
                ),
            )
            expert_permutation = torch.randperm(self.moe_expert_num)
            rank_expert_ids = []
            sample_chunk_size = 4096
            for rank_token_num in self.rank_bs:
                rank_expert_weights = expert_weights.clone()
                if rank_cold_expert_count > 0:
                    rank_cold_indices = ordinary_expert_indices[
                        torch.randperm(ordinary_expert_indices.numel())[
                            :rank_cold_expert_count
                        ]
                    ]
                    rank_expert_weights[rank_cold_indices] = 0
                rank_expert_weights = rank_expert_weights[
                    expert_permutation
                ]
                sampled_chunks = []
                for start in range(0, rank_token_num, sample_chunk_size):
                    current_bs = min(sample_chunk_size, rank_token_num - start)
                    sampled_chunks.append(torch.multinomial(
                        rank_expert_weights.expand(current_bs, -1),
                        self.k,
                        replacement=False,
                    ).to(torch.int32))
                rank_expert_ids.append(torch.cat(sampled_chunks, dim=0))
            return torch.cat(rank_expert_ids, dim=0)

        if self.is_topk_balance:
            ep_expert_ids = [gen_balanced_expert_ids_element(round_index) for round_index in range((self.global_bs * self.k) // self.moe_expert_num + 1)]
            expert_ids = torch.cat(ep_expert_ids)[:self.global_bs * self.k].view([self.global_bs, self.k])
        else:
            expert_ids = gen_unbalanced_expert_ids()
            expert_counts = torch.bincount(
                expert_ids.reshape(-1).to(torch.long),
                minlength=self.moe_expert_num,
            )
            max_expert_hits = int(expert_counts.max().item())
            average_expert_hits = float(
                expert_counts.to(torch.float64).mean().item()
            )
            max_to_average_ratio = max_expert_hits / average_expert_hits
            print(
                f"[维测] 热点专家命中统计: 最大值={max_expert_hits}, "
                f"平均值={average_expert_hits:.2f}, "
                f"最大值/平均值={max_to_average_ratio:.2f}"
            )
            top_count = min(20, self.moe_expert_num)
            top_values, top_indices = torch.topk(expert_counts, top_count)
            top_expert_stats = ", ".join(
                f"expert_id={expert_id}, count={count}"
                for expert_id, count in zip(
                    top_indices.tolist(), top_values.tolist()
                )
            )
            print(f"[维测] 前{top_count}热点专家命中数量: {top_expert_stats}")

        return self.split_token_tensor(expert_ids)

    def gen_x_active_mask(self):
        def gen_card_active_mask(card_bs):
            if self.has_explicit_rank_bs and not self.is_mask:
                return torch.ones(card_bs, dtype=torch.bool)
            min_real_token_num = 0 if self.is_mask else 1
            real_token_num = torch.randint(min_real_token_num, card_bs + 1, (1,)).item() if self.is_unfixed_real_token else card_bs
            return torch.cat(
                (torch.ones(real_token_num), torch.zeros(card_bs - real_token_num))
            ).to(torch.bool)

        x_active_mask = [
            gen_card_active_mask(card_bs) for card_bs in self.rank_bs
        ]
        if self.is_variable_bs and not self.has_explicit_rank_bs:
            upper_bs_card_ep_id = torch.randint(0, self.ep_world_size, (1,)).item()
            x_active_mask[upper_bs_card_ep_id] = torch.full((self.bs,), 1).to(torch.bool)

        return self.split_token_tensor(torch.cat(x_active_mask))

    def gen_expert_scales(self):
        return self.split_token_tensor(torch.empty(size=[self.global_bs, self.k], dtype=self.expert_scales_dtype).uniform_(-0.1, 0.1))

    def gen_weight_data(self, row, col, data_type, a, b):
        # 生成 [a, b] 范围内的均匀分布
        if data_type == torch.int8:
            matrix = torch.randint(a, b, (self.total_expert_num, row, col), dtype=data_type)
        else:
            matrix = a + (b - a) * torch.rand(self.total_expert_num, row, col, dtype=data_type)

        return self.chunk_tensor(matrix, self.ep_world_size)

    def gen_weight_scale_data(self, col, data_type, a, b):
        # 生成 [a, b] 范围内的均匀分布
        matrix = a + (b - a) * torch.rand(self.total_expert_num, col, dtype=data_type)

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

    def is_shared_expert_slot(self, slot_id):
        return self.shared_expert_num > 0 and (slot_id % self.expert_per_rank) < self.shared_expert_num

    def get_rank_active_token_indices(self, rank):
        mask = self.x_active_mask[rank]
        if mask.dim() == 2:
            mask = mask.any(dim=1)
        mask = mask.to(torch.bool)
        local_indices = torch.arange(self.rank_bs[rank], dtype=torch.long)[mask]
        return local_indices + self.rank_bs_offset[rank]

    def apply_variable_bs_to_inputs(self):
        if not self.is_variable_bs or self.has_explicit_rank_bs:
            return
        for rank in range(self.ep_world_size):
            mask = self.x_active_mask[rank]
            if mask.dim() == 2:
                mask = mask.any(dim=1)
            mask = mask.to(torch.bool)
            self.x[rank] = self.x[rank][mask]
            self.expert_ids[rank] = self.expert_ids[rank][mask]
            self.expert_scales[rank] = self.expert_scales[rank][mask]
            self.x_final[rank] = self.x_final[rank][mask]

    def get_rank_token_indices(self, rank):
        return (
            torch.arange(self.rank_bs[rank], dtype=torch.long)
            + self.rank_bs_offset[rank]
        )

    def gen_dispatch_golden(self, expert_ids, x):
        w_dtype = self.quant_dtype
        token_indices_list = []
        expert_pos_list = []
        prob_list = []
        expert_token_nums_per_rank = []

        input_tokens_fp32 = x.to(torch.float32)
        if self.is_variable_bs:
            active_token_indices = torch.cat([
                self.get_rank_active_token_indices(rank)
                for rank in range(self.ep_world_size)
            ])
            routed_expert_ids = expert_ids[active_token_indices]
        else:
            active_token_indices = torch.arange(self.global_bs, dtype=torch.long)
            routed_expert_ids = expert_ids
        flat_expert_ids = routed_expert_ids.reshape(-1).to(torch.long)
        flat_token_indices = active_token_indices.repeat_interleave(self.k)
        flat_expert_pos = torch.arange(self.k, dtype=torch.long).repeat(
            active_token_indices.numel()
        )
        expert_token_nums_tensor = torch.bincount(flat_expert_ids, minlength=self.moe_expert_num)
        expert_offsets = torch.cat((
            torch.zeros(1, dtype=torch.long),
            torch.cumsum(expert_token_nums_tensor, dim=0)
        ))
        sort_order = torch.from_numpy(np.argsort(flat_expert_ids.numpy(), kind="stable")).to(torch.long)
        sorted_token_indices = flat_token_indices[sort_order]
        sorted_expert_pos = flat_expert_pos[sort_order]
        sorted_expert_ids = flat_expert_ids[sort_order]

        if self.quant_mode == 4:
            a_quant_list = []
            a_scale_list = []
            x_quant_all, x_scale_all = dispatch_mx_quantize(
                input_tokens_fp32.numpy(), mx_ele_dtype=w_dtype, axis=-1, block_size=32, round_mode="rint"
            )
            for rank in range(self.ep_world_size):
                rank_token_nums = []
                rank_token_indices = (
                    self.get_rank_active_token_indices(rank)
                    if self.is_variable_bs
                    else self.get_rank_token_indices(rank)
                )
                rank_token_indices_np = rank_token_indices.numpy()
                for shared_id in range(self.shared_expert_num):
                    token_indices = rank_token_indices.clone()
                    token_indices_list.append(token_indices)
                    expert_pos_list.append(torch.empty(0, dtype=torch.long))
                    prob_list.append(torch.full((len(token_indices),), shared_id, dtype=torch.float32))
                    if len(token_indices) > 0:
                        a_quant_list.append(x_quant_all[rank_token_indices_np])
                        a_scale_list.append(x_scale_all[rank_token_indices_np])
                    else:
                        a_quant_list.append(np.empty((0,) + x_quant_all.shape[1:], dtype=x_quant_all.dtype))
                        a_scale_list.append(np.empty((0,) + x_scale_all.shape[1:], dtype=x_scale_all.dtype))
                for local_g in range(self.local_moe_expert_num):
                    g = rank * self.local_moe_expert_num + local_g
                    start = expert_offsets[g].item()
                    end = expert_offsets[g + 1].item()
                    token_indices = sorted_token_indices[start:end].clone()
                    expert_pos = sorted_expert_pos[start:end].clone()
                    token_indices_list.append(token_indices)
                    expert_pos_list.append(expert_pos)
                    prob_list.append(sorted_expert_ids[start:end].to(torch.float32).clone())
                    rank_token_nums.append(end - start)
                    if end > start:
                        token_indices_np = token_indices.numpy()
                        a_quant_list.append(x_quant_all[token_indices_np])
                        a_scale_list.append(x_scale_all[token_indices_np])
                    else:
                        a_quant_list.append(np.empty((0,) + x_quant_all.shape[1:], dtype=x_quant_all.dtype))
                        a_scale_list.append(np.empty((0,) + x_scale_all.shape[1:], dtype=x_scale_all.dtype))
                expert_token_nums_per_rank.append(torch.tensor(rank_token_nums, dtype=torch.int64))
            return a_quant_list, token_indices_list, expert_pos_list, prob_list, a_scale_list, expert_token_nums_per_rank

        a_reordered_list = []
        for rank in range(self.ep_world_size):
            rank_token_nums = []
            rank_token_indices = (
                self.get_rank_active_token_indices(rank)
                if self.is_variable_bs
                else self.get_rank_token_indices(rank)
            )
            for shared_id in range(self.shared_expert_num):
                token_indices = rank_token_indices.clone()
                token_indices_list.append(token_indices)
                expert_pos_list.append(torch.empty(0, dtype=torch.long))
                prob_list.append(torch.full((len(token_indices),), shared_id, dtype=torch.float32))
                if len(token_indices) > 0:
                    a_reordered_list.append(input_tokens_fp32[token_indices])
                else:
                    a_reordered_list.append(torch.empty(0, self.h, dtype=torch.float32))
            for local_g in range(self.local_moe_expert_num):
                g = rank * self.local_moe_expert_num + local_g
                start = expert_offsets[g].item()
                end = expert_offsets[g + 1].item()
                token_indices = sorted_token_indices[start:end].clone()
                expert_pos = sorted_expert_pos[start:end].clone()
                token_indices_list.append(token_indices)
                expert_pos_list.append(expert_pos)
                prob_list.append(sorted_expert_ids[start:end].to(torch.float32).clone())
                rank_token_nums.append(end - start)
                if end > start:
                    a_reordered_list.append(input_tokens_fp32[token_indices])
                else:
                    a_reordered_list.append(torch.empty(0, self.h, dtype=torch.float32))
            expert_token_nums_per_rank.append(torch.tensor(rank_token_nums, dtype=torch.int64))

        return a_reordered_list, token_indices_list, expert_pos_list, prob_list, None, expert_token_nums_per_rank
    
    def _calculate_ffn_expert(
        self, global_expert_idx, expert_scales, weight_storage=None
    ):
        worker_start = time.perf_counter()
        stage_times = {
            "weight_generation": 0.0,
            "gmm1": 0.0,
            "swiglu_quant": 0.0,
            "gmm2": 0.0,
            "retain_weights": 0.0,
            "worker_total": 0.0,
            "active_expert": 0,
        }
        stage_start = time.perf_counter()
        (
            weight1,
            weight2,
            weight_scale1,
            weight_scale2,
            weight_scale1_npu,
            weight_scale2_npu,
            w1_npu,
            w2_npu,
            use_fast_random_weight,
        ) = generate_ffn_expert_weight(
            global_expert_idx,
            self.random_seed,
            h=self.h,
            n=self.n,
            weight_dtype=self.weight_dtype,
            transpose_weight=self.transpose_weight,
            fast_random_weight=self.fast_random_weight,
        )
        stage_times["weight_generation"] = time.perf_counter() - stage_start

        x_value = self.a_reordered_list[global_expert_idx]
        if x_value.shape[0] == 0:
            output2 = torch.empty(0, self.h, dtype=torch.float32)
        else:
            stage_times["active_expert"] = 1
            stage_start = time.perf_counter()
            output1 = qbmm_mx_golden_calculate(
                [x_value],
                [weight1[0]],
                [weight_scale1[0]],
                [self.a_scale_list[global_expert_idx]],
                unit_weight_scale=use_fast_random_weight,
                expert_workers=1,
            )[0]
            output1 = output1.to(torch.bfloat16).to(torch.float32)
            stage_times["gmm1"] = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            swiglu_out = apply_golden_activation(
                output1,
                self.activation_type,
                self.clamp_limit,
                self.swiglu_alpha,
                self.swiglu_beta,
                self.situ_beta,
                self.situ_linear_beta,
            )
            if self.topk_weights_type:
                row_count = len(self.token_indices_list[global_expert_idx])
                if self.is_shared_expert_slot(global_expert_idx):
                    topk_weights = torch.ones(row_count, dtype=swiglu_out.dtype)
                else:
                    token_indices = self.token_indices_list[global_expert_idx]
                    expert_positions = self.expert_pos_list[global_expert_idx]
                    topk_weights = expert_scales[
                        token_indices, expert_positions
                    ].to(swiglu_out.dtype)
                swiglu_out = swiglu_out * topk_weights.unsqueeze(1)
            swiglu_out = swiglu_out.to(torch.bfloat16)

            quant_output, quant_scale_output = mx_quantize(
                swiglu_out.to(torch.float32).numpy(),
                mx_ele_dtype=self.swiglu_quant_dtype,
                axis=-1,
            )
            per_token_scale = quant_scale_output.reshape(
                quant_scale_output.shape[0], quant_scale_output.shape[1] * 2
            )
            stage_times["swiglu_quant"] = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            output2 = qbmm_mx_golden_calculate(
                [quant_output],
                [weight2[0]],
                [weight_scale2[0]],
                [per_token_scale],
                unit_weight_scale=use_fast_random_weight,
                expert_workers=1,
            )[0]
            if global_expert_idx == 0:
                self.print_golden_debug_values(
                    910004, "gmm2_output", output2[0].to(torch.bfloat16), 20
                )
            stage_times["gmm2"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        retained_weights = None
        if self.retain_npu_weights:
            weight_scale1_npu_standalone = weight_scale1_npu
            w1_npu_standalone = w1_npu
            if self.is_interleaved:
                w_n_axis = 1 if self.transpose_weight == 1 else 2
                w1_npu = interleave_gate_up_along_axis(w1_npu, n_axis=w_n_axis)
                weight_scale1_npu = interleave_gate_up_along_axis(
                    weight_scale1_npu, n_axis=w_n_axis
                )
            retained_weights = (
                weight_scale1_npu,
                weight_scale2_npu,
                w1_npu,
                w2_npu,
                weight_scale1_npu_standalone,
                w1_npu_standalone,
            )
            if weight_storage is not None:
                self._store_retained_weights(
                    weight_storage,
                    global_expert_idx,
                    retained_weights,
                )
                retained_weights = None
        stage_times["retain_weights"] = time.perf_counter() - stage_start
        stage_times["worker_total"] = time.perf_counter() - worker_start
        return global_expert_idx, output2, retained_weights, stage_times

    def _initialize_parallel_weight_storage(
        self, weight_storage, retained_weights
    ):
        if weight_storage["initialized"]:
            return
        with weight_storage["lock"]:
            if weight_storage["initialized"]:
                return
            (
                weight_scale1_npu,
                weight_scale2_npu,
                w1_npu,
                w2_npu,
                weight_scale1_npu_standalone,
                w1_npu_standalone,
            ) = retained_weights
            weight_storage.update({
                "weight1_scale_ep_arrays": self.allocate_ep_arrays(
                    weight_scale1_npu, self.local_moe_expert_num
                ),
                "weight2_scale_ep_arrays": self.allocate_ep_arrays(
                    weight_scale2_npu, self.local_moe_expert_num
                ),
                "w1_npu_ep_arrays": self.allocate_ep_arrays(
                    w1_npu, self.local_moe_expert_num
                ),
                "w2_npu_ep_arrays": self.allocate_ep_arrays(
                    w2_npu, self.local_moe_expert_num
                ),
                "shared_l1_weights_sf_ep_arrays": None,
                "shared_l2_weights_sf_ep_arrays": None,
                "shared_w1_npu_ep_arrays": None,
                "shared_w2_npu_ep_arrays": None,
                "weight1_scale_standalone_ep_arrays": None,
                "w1_npu_standalone_ep_arrays": None,
                "shared_l1_weights_sf_standalone_ep_arrays": None,
                "shared_l1_weights_standalone_ep_arrays": None,
            })
            if self.shared_expert_num > 0:
                weight_storage.update({
                    "shared_l1_weights_sf_ep_arrays": self.allocate_ep_arrays(
                        weight_scale1_npu, self.shared_expert_num
                    ),
                    "shared_l2_weights_sf_ep_arrays": self.allocate_ep_arrays(
                        weight_scale2_npu, self.shared_expert_num
                    ),
                    "shared_w1_npu_ep_arrays": self.allocate_ep_arrays(
                        w1_npu, self.shared_expert_num
                    ),
                    "shared_w2_npu_ep_arrays": self.allocate_ep_arrays(
                        w2_npu, self.shared_expert_num
                    ),
                })
            if self.is_interleaved:
                weight_storage.update({
                    "weight1_scale_standalone_ep_arrays": self.allocate_ep_arrays(
                        weight_scale1_npu_standalone,
                        self.local_moe_expert_num,
                    ),
                    "w1_npu_standalone_ep_arrays": self.allocate_ep_arrays(
                        w1_npu_standalone, self.local_moe_expert_num
                    ),
                })
                if self.shared_expert_num > 0:
                    weight_storage.update({
                        "shared_l1_weights_sf_standalone_ep_arrays":
                            self.allocate_ep_arrays(
                                weight_scale1_npu_standalone,
                                self.shared_expert_num,
                            ),
                        "shared_l1_weights_standalone_ep_arrays":
                            self.allocate_ep_arrays(
                                w1_npu_standalone,
                                self.shared_expert_num,
                            ),
                    })
            weight_storage["initialized"] = True

    def _store_retained_weights(
        self, weight_storage, global_expert_idx, retained_weights
    ):
        self._initialize_parallel_weight_storage(
            weight_storage, retained_weights
        )
        # Expert IDs map to disjoint destination slices, so workers can copy
        # directly without serializing the large weights through the main thread.
        (
            weight_scale1_npu,
            weight_scale2_npu,
            w1_npu,
            w2_npu,
            weight_scale1_npu_standalone,
            w1_npu_standalone,
        ) = retained_weights
        self.write_chunk_to_ep_arrays(
            weight_storage["weight1_scale_ep_arrays"],
            weight_scale1_npu,
            global_expert_idx,
            self.shared_expert_num,
        )
        self.write_chunk_to_ep_arrays(
            weight_storage["weight2_scale_ep_arrays"],
            weight_scale2_npu,
            global_expert_idx,
            self.shared_expert_num,
        )
        self.write_chunk_to_ep_arrays(
            weight_storage["w1_npu_ep_arrays"],
            w1_npu,
            global_expert_idx,
            self.shared_expert_num,
        )
        self.write_chunk_to_ep_arrays(
            weight_storage["w2_npu_ep_arrays"],
            w2_npu,
            global_expert_idx,
            self.shared_expert_num,
        )
        if self.is_interleaved:
            self.write_chunk_to_ep_arrays(
                weight_storage["weight1_scale_standalone_ep_arrays"],
                weight_scale1_npu_standalone,
                global_expert_idx,
                self.shared_expert_num,
            )
            self.write_chunk_to_ep_arrays(
                weight_storage["w1_npu_standalone_ep_arrays"],
                w1_npu_standalone,
                global_expert_idx,
                self.shared_expert_num,
            )
        if self.shared_expert_num > 0:
            self.write_chunk_to_ep_arrays(
                weight_storage["shared_l1_weights_sf_ep_arrays"],
                weight_scale1_npu,
                global_expert_idx,
            )
            self.write_chunk_to_ep_arrays(
                weight_storage["shared_l2_weights_sf_ep_arrays"],
                weight_scale2_npu,
                global_expert_idx,
            )
            self.write_chunk_to_ep_arrays(
                weight_storage["shared_w1_npu_ep_arrays"],
                w1_npu,
                global_expert_idx,
            )
            self.write_chunk_to_ep_arrays(
                weight_storage["shared_w2_npu_ep_arrays"],
                w2_npu,
                global_expert_idx,
            )
            if self.is_interleaved:
                self.write_chunk_to_ep_arrays(
                    weight_storage[
                        "shared_l1_weights_sf_standalone_ep_arrays"
                    ],
                    weight_scale1_npu_standalone,
                    global_expert_idx,
                )
                self.write_chunk_to_ep_arrays(
                    weight_storage[
                        "shared_l1_weights_standalone_ep_arrays"
                    ],
                    w1_npu_standalone,
                    global_expert_idx,
                )

    def _gen_ffn_golden_parallel(self, chunk_size, expert_scales, expert_ids):
        expert_num = self.total_expert_num
        configured_chunk_size = max(1, min(chunk_size, expert_num))
        chunk_size = calculate_parallel_chunk_size(
            configured_chunk_size,
            expert_num,
            self.golden_expert_workers,
        )
        if chunk_size != configured_chunk_size:
            print(
                "[golden-profile] 并行调度chunk调整: "
                f"{configured_chunk_size} -> {chunk_size}, "
                f"worker={self.golden_expert_workers}"
            )
        output = []
        weight1_scale_ep_arrays = None
        weight2_scale_ep_arrays = None
        shared_l1_weights_sf_ep_arrays = None
        shared_l2_weights_sf_ep_arrays = None
        w1_npu_ep_arrays = None
        w2_npu_ep_arrays = None
        shared_w1_npu_ep_arrays = None
        shared_w2_npu_ep_arrays = None
        weight1_scale_standalone_ep_arrays = None
        w1_npu_standalone_ep_arrays = None
        shared_l1_weights_sf_standalone_ep_arrays = None
        shared_l1_weights_standalone_ep_arrays = None
        ffn_start_time = time.perf_counter()
        cpu_capacity = get_process_cpu_capacity()
        profile_stage_names = (
            "weight_generation",
            "gmm1",
            "swiglu_quant",
            "gmm2",
            "retain_weights",
            "worker_total",
        )
        profile_window_start = 0
        profile_window_wall = 0.0
        profile_window_cpu = 0.0
        profile_window_active = 0
        profile_window_experts = 0
        profile_window_combine = 0.0
        profile_window_store = 0.0
        profile_window_gc = 0.0
        profile_window_stages = {name: 0.0 for name in profile_stage_names}
        weight_storage = {
            "lock": threading.Lock(),
            "initialized": False,
        }
        profile_log_enabled = (
            self.golden_log_interval is not None
            and self.golden_log_interval > 0
        )
        next_profile_log = (
            self.golden_log_interval if profile_log_enabled else None
        )

        for start in range(0, expert_num, chunk_size):
            chunk_start_time = time.perf_counter()
            chunk_cpu_start = time.process_time()
            chunk_active = 0
            chunk_combine = 0.0
            chunk_store = 0.0
            chunk_stages = {name: 0.0 for name in profile_stage_names}
            end = min(start + chunk_size, expert_num)
            log_chunk = profile_log_enabled and (
                start == 0
                or end == expert_num
                or end >= next_profile_log
            )
            expert_indices = range(start, end)
            results = self._golden_executor.map(
                lambda expert_idx: self._calculate_ffn_expert(
                    expert_idx,
                    expert_scales,
                    weight_storage if self.retain_npu_weights else None,
                ),
                expert_indices,
            )

            for global_expert_idx, output2, retained_weights, stage_times in results:
                chunk_active += stage_times["active_expert"]
                for stage_name in profile_stage_names:
                    chunk_stages[stage_name] += stage_times[stage_name]

                combine_start = time.perf_counter()
                if expert_scales is None or expert_ids is None:
                    output.append(output2)
                else:
                    self.accumulate_combine_golden(
                        [output2], global_expert_idx, expert_scales, expert_ids
                    )
                chunk_combine += time.perf_counter() - combine_start

                if retained_weights is not None:
                    raise RuntimeError(
                        "parallel Golden weight storage must run on worker threads"
                    )

            gc_start = time.perf_counter()
            gc.collect()
            chunk_gc = time.perf_counter() - gc_start
            chunk_elapsed_time = time.perf_counter() - chunk_start_time
            chunk_cpu_time = time.process_time() - chunk_cpu_start
            profile_window_wall += chunk_elapsed_time
            profile_window_cpu += chunk_cpu_time
            profile_window_active += chunk_active
            profile_window_experts += end - start
            profile_window_combine += chunk_combine
            profile_window_store += chunk_store
            profile_window_gc += chunk_gc
            for stage_name in profile_stage_names:
                profile_window_stages[stage_name] += chunk_stages[stage_name]

            if log_chunk:
                ffn_elapsed_time = time.perf_counter() - ffn_start_time
                average_cores, cpu_utilization = calculate_process_cpu_profile(
                    profile_window_cpu,
                    profile_window_wall,
                    cpu_capacity,
                )
                worker_parallelism = (
                    profile_window_stages["worker_total"] / profile_window_wall
                    if profile_window_wall > 0
                    else 0.0
                )
                print(
                    f"[golden] experts {start}:{end}/{expert_num} done, "
                    f"chunk_time={chunk_elapsed_time:.3f}s, elapsed={ffn_elapsed_time:.3f}s"
                )
                print(
                    f"[golden-profile] experts {profile_window_start}:{end}, "
                    f"活跃={profile_window_active}, "
                    f"空专家={profile_window_experts - profile_window_active}, "
                    f"CPU平均={average_cores:.2f}/{cpu_capacity}核 "
                    f"({cpu_utilization:.2f}%), "
                    f"worker并行度={worker_parallelism:.2f}"
                )
                print(
                    "[golden-profile] 线程累计耗时(s): "
                    f"权重生成={profile_window_stages['weight_generation']:.3f}, "
                    f"GMM1={profile_window_stages['gmm1']:.3f}, "
                    f"SwiGLU量化={profile_window_stages['swiglu_quant']:.3f}, "
                    f"GMM2={profile_window_stages['gmm2']:.3f}, "
                    f"权重保留/并行写入={profile_window_stages['retain_weights']:.3f}; "
                    "主线程耗时(s): "
                    f"Combine={profile_window_combine:.3f}, "
                    f"权重写入={profile_window_store:.3f}, "
                    f"GC={profile_window_gc:.3f}, "
                    f"wall={profile_window_wall:.3f}"
                )
                profile_window_start = end
                profile_window_wall = 0.0
                profile_window_cpu = 0.0
                profile_window_active = 0
                profile_window_experts = 0
                profile_window_combine = 0.0
                profile_window_store = 0.0
                profile_window_gc = 0.0
                profile_window_stages = {name: 0.0 for name in profile_stage_names}
                while (
                    next_profile_log is not None
                    and next_profile_log <= end
                ):
                    next_profile_log += self.golden_log_interval

        if self.retain_npu_weights:
            if not weight_storage["initialized"]:
                raise RuntimeError("parallel Golden weight storage was not initialized")
            weight1_scale_ep_arrays = weight_storage["weight1_scale_ep_arrays"]
            weight2_scale_ep_arrays = weight_storage["weight2_scale_ep_arrays"]
            shared_l1_weights_sf_ep_arrays = weight_storage[
                "shared_l1_weights_sf_ep_arrays"
            ]
            shared_l2_weights_sf_ep_arrays = weight_storage[
                "shared_l2_weights_sf_ep_arrays"
            ]
            w1_npu_ep_arrays = weight_storage["w1_npu_ep_arrays"]
            w2_npu_ep_arrays = weight_storage["w2_npu_ep_arrays"]
            shared_w1_npu_ep_arrays = weight_storage[
                "shared_w1_npu_ep_arrays"
            ]
            shared_w2_npu_ep_arrays = weight_storage[
                "shared_w2_npu_ep_arrays"
            ]
            weight1_scale_standalone_ep_arrays = weight_storage[
                "weight1_scale_standalone_ep_arrays"
            ]
            w1_npu_standalone_ep_arrays = weight_storage[
                "w1_npu_standalone_ep_arrays"
            ]
            shared_l1_weights_sf_standalone_ep_arrays = weight_storage[
                "shared_l1_weights_sf_standalone_ep_arrays"
            ]
            shared_l1_weights_standalone_ep_arrays = weight_storage[
                "shared_l1_weights_standalone_ep_arrays"
            ]
        if not self.is_interleaved:
            weight1_scale_standalone_ep_arrays = weight1_scale_ep_arrays
            w1_npu_standalone_ep_arrays = w1_npu_ep_arrays
            shared_l1_weights_sf_standalone_ep_arrays = shared_l1_weights_sf_ep_arrays
            shared_l1_weights_standalone_ep_arrays = shared_w1_npu_ep_arrays
        return (
            output,
            None,
            None,
            weight1_scale_ep_arrays,
            weight2_scale_ep_arrays,
            shared_l1_weights_sf_ep_arrays,
            shared_l2_weights_sf_ep_arrays,
            None,
            None,
            w1_npu_ep_arrays,
            w2_npu_ep_arrays,
            shared_w1_npu_ep_arrays,
            shared_w2_npu_ep_arrays,
            w1_npu_standalone_ep_arrays,
            weight1_scale_standalone_ep_arrays,
            shared_l1_weights_standalone_ep_arrays,
            shared_l1_weights_sf_standalone_ep_arrays,
        )

    def gen_ffn_golden(self, chunk_size=16, expert_scales=None, expert_ids=None):
        self.is_interleaved = bool(getattr(self, "is_interleaved", False))
        if getattr(self, "_golden_executor", None) is not None and not self.golden_batch_matmul:
            return self._gen_ffn_golden_parallel(chunk_size, expert_scales, expert_ids)

        w_dtype = self.weight_dtype
        expert_num = self.total_expert_num
        chunk_size = max(1, min(chunk_size, expert_num))

        output = []
        weight1_scale_ep_arrays = None
        weight2_scale_ep_arrays = None
        shared_l1_weights_sf_ep_arrays = None
        shared_l2_weights_sf_ep_arrays = None
        w1_npu_ep_arrays = None
        w2_npu_ep_arrays = None
        shared_w1_npu_ep_arrays = None
        shared_w2_npu_ep_arrays = None
        weight1_scale_standalone_ep_arrays = None
        w1_npu_standalone_ep_arrays = None
        shared_l1_weights_sf_standalone_ep_arrays = None
        shared_l1_weights_standalone_ep_arrays = None
        ffn_start_time = time.perf_counter()

        for start in range(0, expert_num, chunk_size):
            chunk_start_time = time.perf_counter()
            end = min(start + chunk_size, expert_num)
            cur_expert_num = end - start
            log_chunk = (
                self.golden_log_interval is not None
                and self.golden_log_interval > 0
                and (start == 0 or end == expert_num or start % self.golden_log_interval == 0)
            )
            (
                weight1,
                weight2,
                weight_scale1,
                weight_scale2,
                weight_scale1_npu_cpu,
                weight_scale2_npu_cpu,
                w1_npu,
                w2_npu,
                use_fast_random_weight,
            ) = generate_ffn_weight_chunk(
                start,
                cur_expert_num,
                self.random_seed,
                self.h,
                self.n,
                w_dtype,
                self.transpose_weight,
                self.fast_random_weight,
            )

            # CPU golden and the cascade GMM1 keep the original [gate | up]
            # layout.  Only the fused MegaMoe GMM1 input is interleaved.
            weight_scale1_npu_cpu_standalone = weight_scale1_npu_cpu
            w1_npu_standalone = w1_npu
            if self.is_interleaved:
                w_n_axis = 1 if self.transpose_weight == 1 else 2
                w1_npu = interleave_gate_up_along_axis(w1_npu, n_axis=w_n_axis)
                weight_scale1_npu_cpu = interleave_gate_up_along_axis(
                    weight_scale1_npu_cpu, n_axis=w_n_axis
                )

            weight1_list = list(weight1)
            weight2_list = list(weight2)
            weight1_scale_list = list(weight_scale1)
            weight2_scale_list = list(weight_scale2)
            x_list = self.a_reordered_list[start:end]
            x_scale_list = self.a_scale_list[start:end]
            debug_expert_zero = start == 0 and len(x_list) > 0 and x_list[0].shape[0] > 0
            output1 = qbmm_mx_golden_calculate(
                x_list, weight1_list, weight1_scale_list, x_scale_list,
                batch_matmul=self.golden_batch_matmul,
                batch_matmul_min_m=self.golden_batch_matmul_min_m,
                unit_weight_scale=use_fast_random_weight,
                expert_workers=self.golden_expert_workers,
                expert_executor=getattr(self, "_golden_executor", None),
            )
            output1 = golden_output_process(output1, 2, 0, log_stats=log_chunk and self.golden_log_stats)

            split_sizes = [t.shape[0] for t in x_list]
            split_indices = np.cumsum([0] + split_sizes)

            if output1[0].shape[0] == 0:
                output2 = [torch.empty(0, self.h, dtype=torch.float32) for _ in range(cur_expert_num)]
            else:
                output1[0] = output1[0].to(torch.bfloat16).to(torch.float32)
                swiglu_out = apply_golden_activation(
                    output1[0],
                    self.activation_type,
                    self.clamp_limit,
                    self.swiglu_alpha,
                    self.swiglu_beta,
                    self.situ_beta,
                    self.situ_linear_beta,
                )
                # swiglu_out = swiglu_out.to(torch.bfloat16)
                if self.topk_weights_type:
                    chunk_topk_weights = []
                    for global_expert_idx in range(start, end):
                        row_count = len(self.token_indices_list[global_expert_idx])
                        if self.is_shared_expert_slot(global_expert_idx):
                            weights = torch.ones(row_count, dtype=swiglu_out.dtype)
                        else:
                            token_indices = self.token_indices_list[global_expert_idx]
                            expert_positions = self.expert_pos_list[global_expert_idx]
                            weights = expert_scales[token_indices, expert_positions].to(swiglu_out.dtype)
                        chunk_topk_weights.append(weights)
                    topk_weights = torch.cat(chunk_topk_weights).unsqueeze(1)
                    swiglu_out = swiglu_out * topk_weights
                swiglu_out = swiglu_out.to(torch.bfloat16).to(torch.float32)
                quant_output, quant_scale_output = mx_quantize(
                    swiglu_out.numpy(), mx_ele_dtype=self.swiglu_quant_dtype, axis=-1
                )
                x_transpose_flag = 0
                perTokenScale_input = np.concatenate(
                    [quant_scale_output], axis=-2 if x_transpose_flag == 1 else -1
                )
                perTokenScale_input = perTokenScale_input.reshape(
                    perTokenScale_input.shape[0], perTokenScale_input.shape[1] * 2
                )
                perTokenScale_input = transpose_last_two_dim(perTokenScale_input, x_transpose_flag)
                del output1, swiglu_out
                quant_output = [quant_output[split_indices[i]:split_indices[i + 1]] for i in range(cur_expert_num)]
                quant_scale_output = [
                    perTokenScale_input[split_indices[i]:split_indices[i + 1]] for i in range(cur_expert_num)
                ]
                output2 = qbmm_mx_golden_calculate(
                    quant_output, weight2_list, weight2_scale_list, quant_scale_output,
                    batch_matmul=self.golden_batch_matmul,
                    batch_matmul_min_m=self.golden_batch_matmul_min_m,
                    unit_weight_scale=use_fast_random_weight,
                    expert_workers=self.golden_expert_workers,
                    expert_executor=getattr(self, "_golden_executor", None),
                )
                if debug_expert_zero:
                    self.print_golden_debug_values(
                        910004, "gmm2_output", output2[0][0].to(torch.bfloat16), 20
                    )

            if expert_scales is None or expert_ids is None:
                output.extend(output2)
            else:
                self.accumulate_combine_golden(output2, start, expert_scales, expert_ids)
            if self.retain_npu_weights and weight1_scale_ep_arrays is None:
                weight1_scale_ep_arrays = self.allocate_ep_arrays(
                    weight_scale1_npu_cpu, self.local_moe_expert_num
                )
                weight2_scale_ep_arrays = self.allocate_ep_arrays(
                    weight_scale2_npu_cpu, self.local_moe_expert_num
                )
                w1_npu_ep_arrays = self.allocate_ep_arrays(w1_npu, self.local_moe_expert_num)
                w2_npu_ep_arrays = self.allocate_ep_arrays(w2_npu, self.local_moe_expert_num)
                if self.shared_expert_num > 0:
                    shared_l1_weights_sf_ep_arrays = self.allocate_ep_arrays(
                        weight_scale1_npu_cpu, self.shared_expert_num
                    )
                    shared_l2_weights_sf_ep_arrays = self.allocate_ep_arrays(
                        weight_scale2_npu_cpu, self.shared_expert_num
                    )
                    shared_w1_npu_ep_arrays = self.allocate_ep_arrays(w1_npu, self.shared_expert_num)
                    shared_w2_npu_ep_arrays = self.allocate_ep_arrays(w2_npu, self.shared_expert_num)
                if self.is_interleaved:
                    weight1_scale_standalone_ep_arrays = self.allocate_ep_arrays(
                        weight_scale1_npu_cpu_standalone, self.local_moe_expert_num
                    )
                    w1_npu_standalone_ep_arrays = self.allocate_ep_arrays(
                        w1_npu_standalone, self.local_moe_expert_num
                    )
                    if self.shared_expert_num > 0:
                        shared_l1_weights_sf_standalone_ep_arrays = self.allocate_ep_arrays(
                            weight_scale1_npu_cpu_standalone, self.shared_expert_num
                        )
                        shared_l1_weights_standalone_ep_arrays = self.allocate_ep_arrays(
                            w1_npu_standalone, self.shared_expert_num
                        )
            if self.retain_npu_weights:
                self.write_chunk_to_ep_arrays(
                    weight1_scale_ep_arrays, weight_scale1_npu_cpu, start, self.shared_expert_num
                )
                self.write_chunk_to_ep_arrays(
                    weight2_scale_ep_arrays, weight_scale2_npu_cpu, start, self.shared_expert_num
                )
                self.write_chunk_to_ep_arrays(w1_npu_ep_arrays, w1_npu, start, self.shared_expert_num)
                self.write_chunk_to_ep_arrays(w2_npu_ep_arrays, w2_npu, start, self.shared_expert_num)
                if self.is_interleaved:
                    self.write_chunk_to_ep_arrays(
                        weight1_scale_standalone_ep_arrays,
                        weight_scale1_npu_cpu_standalone,
                        start,
                        self.shared_expert_num,
                    )
                    self.write_chunk_to_ep_arrays(
                        w1_npu_standalone_ep_arrays,
                        w1_npu_standalone,
                        start,
                        self.shared_expert_num,
                    )
                if self.shared_expert_num > 0:
                    self.write_chunk_to_ep_arrays(
                        shared_l1_weights_sf_ep_arrays, weight_scale1_npu_cpu, start
                    )
                    self.write_chunk_to_ep_arrays(
                        shared_l2_weights_sf_ep_arrays, weight_scale2_npu_cpu, start
                    )
                    self.write_chunk_to_ep_arrays(shared_w1_npu_ep_arrays, w1_npu, start)
                    self.write_chunk_to_ep_arrays(shared_w2_npu_ep_arrays, w2_npu, start)
                    if self.is_interleaved:
                        self.write_chunk_to_ep_arrays(
                            shared_l1_weights_sf_standalone_ep_arrays,
                            weight_scale1_npu_cpu_standalone,
                            start,
                        )
                        self.write_chunk_to_ep_arrays(
                            shared_l1_weights_standalone_ep_arrays,
                            w1_npu_standalone,
                            start,
                        )
            del weight1_list, weight2_list, weight1_scale_list, weight2_scale_list, x_list, x_scale_list
            if 'output1' in locals():
                del output1
            if 'quant_output' in locals():
                del quant_output, quant_scale_output, perTokenScale_input
            del output2, weight1, weight2, weight_scale1, weight_scale2, weight_scale1_npu_cpu, weight_scale2_npu_cpu, w1_npu, w2_npu
            del weight_scale1_npu_cpu_standalone, w1_npu_standalone
            gc.collect()
            if log_chunk:
                chunk_elapsed_time = time.perf_counter() - chunk_start_time
                ffn_elapsed_time = time.perf_counter() - ffn_start_time
                print(
                    f"[golden] experts {start}:{end}/{expert_num} done, "
                    f"chunk_time={chunk_elapsed_time:.3f}s, elapsed={ffn_elapsed_time:.3f}s"
                )

        weight1_list = None
        weight2_list = None
        weight1_scale_list = weight1_scale_ep_arrays
        weight2_scale_list = weight2_scale_ep_arrays
        weight1_scale_list_npu = None
        weight2_scale_list_npu = None
        w1_npu = w1_npu_ep_arrays
        w2_npu = w2_npu_ep_arrays
        if not self.is_interleaved:
            weight1_scale_standalone_ep_arrays = weight1_scale_ep_arrays
            w1_npu_standalone_ep_arrays = w1_npu_ep_arrays
            shared_l1_weights_sf_standalone_ep_arrays = shared_l1_weights_sf_ep_arrays
            shared_l1_weights_standalone_ep_arrays = shared_w1_npu_ep_arrays
        return (
            output,
            weight1_list,
            weight2_list,
            weight1_scale_list,
            weight2_scale_list,
            shared_l1_weights_sf_ep_arrays,
            shared_l2_weights_sf_ep_arrays,
            weight1_scale_list_npu,
            weight2_scale_list_npu,
            w1_npu,
            w2_npu,
            shared_w1_npu_ep_arrays,
            shared_w2_npu_ep_arrays,
            w1_npu_standalone_ep_arrays,
            weight1_scale_standalone_ep_arrays,
            shared_l1_weights_standalone_ep_arrays,
            shared_l1_weights_sf_standalone_ep_arrays,
        )

    def allocate_ep_arrays(self, sample, experts_per_rank=None):
        if experts_per_rank is None:
            experts_per_rank = self.expert_per_rank
        shape = (experts_per_rank,) + sample.shape[1:]
        return [np.empty(shape, dtype=sample.dtype) for _ in range(self.ep_world_size)]

    def write_chunk_to_ep_arrays(self, destinations, chunk, chunk_start, partition_offset=0):
        chunk_end = chunk_start + chunk.shape[0]
        for rank, destination in enumerate(destinations):
            rank_start = rank * self.expert_per_rank + partition_offset
            rank_end = rank_start + destination.shape[0]
            overlap_start = max(chunk_start, rank_start)
            overlap_end = min(chunk_end, rank_end)
            if overlap_start >= overlap_end:
                continue
            source_start = overlap_start - chunk_start
            source_end = overlap_end - chunk_start
            destination_start = overlap_start - rank_start
            destination_end = overlap_end - rank_start
            destination[destination_start:destination_end] = chunk[source_start:source_end]

    def apply_comm_quant_golden(self, rows):
        if self.comm_quant_mode != 3 and self.comm_quant_mode != 4:
            return rows
        rows = rows.to(torch.bfloat16).to(torch.float32)
        w_dtype = "float8_e4m3fn" if self.comm_quant_mode == 4 else "float8_e5m2"
        c2_quant, c2_scale = dispatch_mx_quantize(
            rows.numpy(),
            mx_ele_dtype=w_dtype,
            axis=-1,
            block_size=32,
            round_mode="rint"
        )
        c2_scale_broadcast = np.repeat(c2_scale, 32, axis=-1)
        c2_scale_broadcast = c2_scale_broadcast[..., :c2_quant.shape[-1]]
        return torch.from_numpy(c2_quant.astype(np.float32) * c2_scale_broadcast.astype(np.float32))

    def accumulate_combine_golden(self, output2, expert_start, expert_scales, expert_ids):
        for local_g, c2_rows in enumerate(output2):
            g = expert_start + local_g
            if len(self.token_indices_list[g]) == 0:
                continue
            indices = self.token_indices_list[g]
            if self.is_shared_expert_slot(g):
                self.combine_golden_accumulator.index_add_(0, indices, c2_rows)
            else:
                expert_pos = self.expert_pos_list[g]
                expert_scale_cur = expert_scales[indices, expert_pos].to(torch.float32).unsqueeze(1)
                if self.topk_weights_type:
                    c2_rows = self.apply_comm_quant_golden(c2_rows)
                    self.combine_golden_accumulator.index_add_(0, indices, c2_rows)
                else:
                    c2_rows = self.apply_comm_quant_golden(c2_rows)
                    self.combine_golden_accumulator.index_add_(0, indices, c2_rows * expert_scale_cur)

    def release_golden_intermediates(self):
        self.a_reordered_list = None
        self.a_scale_list = None
        self.token_indices_list = None
        self.expert_pos_list = None
        self.prob_list = None
        self.c2_fp32_list = None
        self.weight1 = None
        self.weight2 = None
        self.weight1_scale_npu = None
        self.weight2_scale_npu = None
        gc.collect()
    
    def gen_combine_golden(self, expert_scales, expert_ids):
        c2_final = torch.zeros(self.global_bs, self.h, dtype=torch.float32)
        for g in range(self.total_expert_num):
            if len(self.token_indices_list[g]) > 0:
                indices = self.token_indices_list[g]
                
                for i, idx in enumerate(indices):
                    c2_row = self.c2_fp32_list[g][i].unsqueeze(0)
                    if self.is_shared_expert_slot(g):
                        c2_final[idx] += c2_row.squeeze(0)
                    else:
                        expert_pos = self.expert_pos_list[g][i]
                        expert_scale_cur = expert_scales[idx][expert_pos]
                        if self.topk_weights_type:
                            c2_data = self.apply_comm_quant_golden(c2_row).squeeze(0)
                            c2_final[idx] += c2_data
                        else:
                            c2_data = self.apply_comm_quant_golden(c2_row).squeeze(0)
                            c2_final[idx] += c2_data * expert_scale_cur
        return self.split_token_tensor(c2_final)
