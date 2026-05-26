import os
import socket
from typing import List
import torch
import torch_npu
import csv
import numpy as np
from torch.multiprocessing import Process, Manager
import torch.distributed as dist
from torch.distributed import ReduceOp
import torch.multiprocessing as mp
import toolss
import time
import pandas as pd
from math import ceil
from enum import Enum
from quantize import torch_to_numpy_tensor, numpy_to_torch_tensor, get_dtype_int_value
from copy import deepcopy
from itertools import chain
import bisect
import traceback


# region 控制参数配置

# region 执行哪些测试

is_dispatch = 1
is_combine = 1
is_continue = 0

# endregion

# region 性能检测相关

is_performance = 0      # 是否进行性能检测
loop = 105              # 性能检测时的性能采集轮数
repeat = 1              # 性能采集分为几个批次
warmup = 5              # 每批次的热身轮数

# endregion

# region 特性使能
comm_alg = "fullmesh_v1"    # "fullmesh_v1" / "fullmesh_v2" / "ccu" / ""
dispatch_comm_alg = comm_alg
combine_comm_alg = "" if comm_alg in ["fullmesh_v1", "fullmesh_v2"] else comm_alg

is_ccu = (comm_alg == "ccu")
is_fullmesh_v2 = (comm_alg == "fullmesh_v2")
graph_type = 0              # 0为单算子，1为静态图模式，2为动态图模式（当前不支持）

# 0: dispatch算子非量化场景
# 1: 静态量化
# 2: pertoken动态量化
# 3: pergroup动态量化
# 4: MX量化
quant_mode = 0
comm_quant_mode = 0         # 0无量化，1则combine通信采用int12动态量化（不支持），2为int8动态量化

x_active_mask_dim = 0
is_2d_mask = 1 if x_active_mask_dim == 2 else 0
is_mask = 1 if x_active_mask_dim == 1 else 0

is_variable_bs = 1
is_variable_local_moe_expert_num = 0                           # 是否使能可变moe卡上专家数
is_shared_expert_x = 0
is_quant_scales = 0
is_full_core = 1                                                # 1时为全核模式，使用全部的核，0时使用部分核
expert_shard_type = 0                                           # 0 共享专家部署在最前面的卡上，仅支持 0
core_num = 16 if not is_full_core else 32                       # 非全核模式使用的核数（单个核上有一个aic核，一个aiv核）

is_quant = (quant_mode > 0)
is_unfixed_real_token = is_mask or is_variable_bs               # 每张卡上真实token数是否固定
aic_core_num = core_num                                         # 非全核模式使用的aic核数
aiv_core_num = core_num * 2                                     # 非全核模式使用的aiv核数

# 控制 dispatch 输出的 expert_token_nums 的格式：
# 0 表示每个专家收到 token 数量的前缀和；
# 1 表示每个专家收到的 token 数量（默认）
expert_token_nums_type = 0

# endregion

# region 通信域配置

server_num = 1                  # 单机或者多机模式
tp_world_size = 1               # 1或2, 2时dispatch算子在tp域执行allgather, combine算子执行reducescatter, 1时均不执行
shared_expert_num = 1
shared_expert_rank_num = 1
local_moe_expert_num = 8
is_cloud = 0                    # 是否在云上环境进行测试，使用云上环境时设置为1，使用物理机时设为0
if not is_cloud:
    server_index = 0 if server_num > 1 else 0                        # 多机每个脚本保持一致并递增index值
    master_ip = '71.20.45.123' if server_num > 1 else '127.0.0.1'    # 如果通信不通，查看一下master_ip是否设对
else:
    # 云上环境用下面方式获取server_index和master_ip
    server_index = int(os.environ.get("VC_TASK_INDEX"))
    host_url_list = os.environ.get("VC_WORKER_HOSTS").split(",")
    master_ip = socket.gethostbyname(host_url_list[0])

rank_per_dev = 2                                                        # 每个host有几个die
world_size = server_num * rank_per_dev                                  # 通信域总die数
ep_world_size = world_size // tp_world_size
rank_num_per_shared_expert = (shared_expert_rank_num // shared_expert_num 
                                if shared_expert_num > 0 
                                else 0)                                 # 每个共享专家占据的卡数
moe_rank_num = ep_world_size - shared_expert_rank_num                   # moe专家卡数
moe_expert_num = moe_rank_num * local_moe_expert_num                    # moe专家数

is_shared = (shared_expert_rank_num > 0)
server_ranks = slice(server_index * rank_per_dev, server_index * rank_per_dev + rank_per_dev)   # 本机各die索引

include_tp_info = tp_world_size >= 2     # 是否传入 group_tp 和 tp_rank_id，无 tp 域通信时，dispatch 和 combine 不需要指定这两个参数

# endregion

# region 输入输出维度相关

def cal_hccl_buffsize(bs, h, k, ep_world_size, local_moe_expert_num, shared_expert_num):
    MAX_OUT_DTYPE_SIZE = 2
    WIN_ADDR_ALIGN = 512
    FULL_MESH_DATA_ALIGN = 480
    UB_ALIGN = 32
    SCALE_EXPAND_IDX_BUFFER = 44
    DOUBLE_DATA_BUFFER = 2

    token_need_size_combine = ((h * MAX_OUT_DTYPE_SIZE + WIN_ADDR_ALIGN - 1) // WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN
    token_actual_len = ((h * MAX_OUT_DTYPE_SIZE + UB_ALIGN - 1) // UB_ALIGN) * UB_ALIGN + SCALE_EXPAND_IDX_BUFFER
    if is_fullmesh_v2:
        token_need_size_dispatch = ((token_actual_len + FULL_MESH_DATA_ALIGN - 1) // FULL_MESH_DATA_ALIGN) * WIN_ADDR_ALIGN
    else:
        token_need_size_dispatch = ((token_actual_len + WIN_ADDR_ALIGN - 1) // WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN
    actual_size = ((bs * token_need_size_dispatch * ep_world_size * local_moe_expert_num)
                   + (bs * token_need_size_combine * (k + shared_expert_num))) * DOUBLE_DATA_BUFFER

    hccl_buffsize = ceil(actual_size / 1024 / 1024) + 5 # 留出 5MB 空间余量
    return str(hccl_buffsize)

bs = 8
h = 7168
# k_max = 1
k = 8

# 计算 HCCL_BUFFSIZE 大小
os.environ['HCCL_BUFFSIZE'] = cal_hccl_buffsize(bs, h, k, ep_world_size, local_moe_expert_num, shared_expert_num)

global_bs = bs * ep_world_size
max_shared_group_num = ceil(ep_world_size / rank_num_per_shared_expert) if (rank_num_per_shared_expert > 0) else 0
A_shared = bs * max_shared_group_num
A_moe = global_bs * min(local_moe_expert_num, k)

# endregion

# region 产生什么样的输入

# 静态量化场景下，是不是所有专家共享scales
# 0：所有专家不共享scale
# 1：所有专家共享scales，且维度为（h，）
# 2：所有专家共享scales，且维度为（1，）
is_static_shared_scales = 0

x_dtype = 'fp16'
expandx_dtype = 'fp16'
is_topk_balance = False     # 是否生成均衡的topK
is_x_random = True          # 是否让x值随机
random_seed = 90

input_dtype = torch.bfloat16 if x_dtype == "bf16" else torch.float16 # dispatch输入及combine输出的数据类型
if expandx_dtype in ("fp16", "bf16"):
    combine_input_dtype = input_dtype
else:
    combine_input_dtype = torch.float16 if torch.rand(1) < 0.5 else torch.bfloat16

# endregion

# region 测试脚本执行相关

is_continue_golden_expand_x = True                              # 在级联模式下，是否将dispatch的golden_expand_x直接传给combine算子
is_save_tensor_to_bin = False
is_check_setting = True
current_case_name = 'MOE_Dis_Com_2_bs474_h7872_k14_xdtypefp16_quant0_scale0_continue0_tp1_localmoe828_shareRank1_moeExpert828_mte_ID0014'

# endregion

# endregion

# region 常用常量

# region 精度对比相关常量

class ComparedOutput(Enum):
    EP_RECV_COUNTS = "ep_recv_counts"
    TP_RECV_COUNTS = "tp_recv_counts"
    EXPAND_X = "expand_x"
    EXPAND_IDX = "expand_idx"
    DYNAMIC_SCALES = "dynamic_scales"
    EXPERT_TOKEN_NUMS = "expert_token_nums"
    X = "x"

# endregion

# region 性能处理相关常量

dispatch_profiling_path = './Profiling_Result_mc2_dispatch_16p'
combine_profiling_path = './Profiling_Result_mc2_combine_16p'
continue_profiling_path = './Profiling_Result_mc2_continue_16p'
dispatch_op_type = 'MoeDistributeDispatchV2'
combine_op_type = 'MoeDistributeCombineV2'
path_to_op_types = {
    dispatch_profiling_path: [dispatch_op_type],
    combine_profiling_path: [combine_op_type],
    continue_profiling_path: [dispatch_op_type, combine_op_type]
}

# endregion

# region 通信域相关常量

# 单机, tp = 2 ep_ranks_list = [[0,2,4,6,8,10,12,14], [1,3,5,7,9,11,13,15]]  ## 如果指定ep list,按照指定ep list划分ep域
ep_ranks_list = [list(range(tp_id, world_size, tp_world_size)) for tp_id in range(tp_world_size)]
# 单机, tp = 2 tp_ranks_list = [[0,1], [2,3], [4,5], [6,7], [8,9], [10,11], [12,13], [14,15]]  ## 如果指定tp list,按照指定tp list划分tp域
tp_ranks_list = [list(range(ep_id * tp_world_size, (ep_id + 1) * tp_world_size)) for ep_id in range(ep_world_size)]

# endregion

# endregion

# region 工具函数

# region 张量处理函数

def chunk_tensor(tensor: torch.Tensor, chunks: int) -> List[torch.Tensor]:
    return list(tensor.chunk(chunks))

def split_tensor(tensor: torch.Tensor, split_size_or_sections: int or List[int]) -> List[torch.Tensor]:
    return list(tensor.split(split_size_or_sections))

def concat_tensors(tensors, dim=0):
    """
    主要用于处理 float8_e4m3fn, float8_e5m2 这些类型的拼接，这些类型 torch 可能不支持
    """
    valid_index = 0
    while (valid_index < len(tensors) and tensors[valid_index].shape[0] == 0):
        valid_index += 1
    if (valid_index >= len(tensors)):
        return tensors[-1]
    dtype = tensors[valid_index].dtype
    dim = dim % tensors[valid_index].ndim

    for t in tensors:
        if t.shape[0] == 0:
            continue
        if t.dtype != dtype:
            raise ValueError(f"dtype 不一致: {t.dtype} != {dtype}")
        if any(t.shape[d] != tensors[0].shape[d] for d in range(t.ndim) if d != dim):
            raise ValueError(f"非拼接维度不匹配: {tensors[0].shape} vs {t.shape}")

    out_shape = list(tensors[0].shape)
    out_shape[dim] = sum(t.shape[dim] for t in tensors)
    out = torch.empty(out_shape, dtype=dtype)
    idx = 0
    for t in tensors:
        if t.shape[dim] > 0:
            slices = [slice(None)] * t.ndim
            slices[dim] = slice(idx, idx + t.shape[dim])
            out[tuple(slices)] = t
        idx += t.shape[dim]
    return out

def reshape_with_padding(input, tile_size):
    row, col = input.shape
    padded_col = ceil(col / tile_size) * tile_size
    input_padded = np.pad(input, 
                          ((0, 0), (0, padded_col - col)), 
                          mode='constant', 
                          constant_values=0)
    return input_padded.reshape(-1, tile_size)

@torch.compiler.disable
def _convert_hifloat8_to_target_type(expand_x, target_type):
    # torch 原生不支持的类型
    from en_dtypes import hifloat8
    # hifloat8 用的 torch.uint8 存储
    expand_x_numpy = expand_x.cpu().numpy().view(hifloat8).astype(np.float32)
    return torch.from_numpy(expand_x_numpy).to(device=expand_x.device, dtype=target_type)

def convert_expand_x_dtype(expand_x):
    """
    将 dispatch 输出的 expand_x 转为 fp16 或 bf16，用于 combine 输入
    """
    if expand_x.dtype in (torch.float16, torch.bfloat16):
        return expand_x
    
    from quantize import dtype_map
    if dtype_map.get(expandx_dtype, None) in ("float8_e4m3fn", "float8_e5m2", "int8", "float16", "bfloat16"):
        # torch 原生支持的类型
        return expand_x.to(combine_input_dtype)
    elif dtype_map.get(expandx_dtype, None) == "hifloat8":
        return _convert_hifloat8_to_target_type(expand_x, combine_input_dtype)
    else:
        raise ValueError(f"Unsupported expand_x dtype {expandx_dtype}")

# endregion

# region dataframe打印函数

def print_dataframe(df: pd.DataFrame):
    print('\n' + '-' * 60)
    print(df.to_string(index=False))
    print('-' * 60 + '\n')

# endregion

# region 文件处理函数

def write_row_to_csv(csv_name: str, row_content: list):
    if not os.path.exists("csv_result"):
        os.makedirs("csv_result")
    with open(f'csv_result/{csv_name}', 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(row_content)

def save_tensor_to_bin(tensor, bin_name):
    if not os.path.exists("data"):
        os.makedirs("data")
    tensor.cpu().numpy().tofile(f"data/{bin_name}.bin")

# endregion

# region 性能采集函数

def get_profiling(loop, repeat, warmup, path):
    active = loop // repeat - warmup

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1, 
        l2_cache=False, 
        data_simplification=False
    )
    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
        ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=warmup, active=active, repeat=repeat, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(path),
        profile_memory=True,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    )

    return prof

# endregion

# endregion

# region 测试执行前预备工作相关函数

def clean_previous_profiling_result():
    profiling_result = './Profiling_Result_'
    os.system(f'rm -rf {profiling_result}*')

def print_setting():
    print(f"{server_num=}")
    print(f"{rank_per_dev=}")
    print(f"{tp_world_size=}")
    print(f"{ep_world_size=}")
    print(f"{shared_expert_num=}")
    print(f"{shared_expert_rank_num=}")
    print(f"{moe_expert_num=}")
    print(f"{rank_num_per_shared_expert=}")
    print(f"{moe_rank_num=}")
    print(f"{local_moe_expert_num=}")
    print(f"{bs=}")
    print(f"{h=}")
    print(f"{k=}")
    print(f"{quant_mode=}")
    print(f"{comm_quant_mode=}")
    print(f"{graph_type=}")
    print(f"{comm_alg=}")

def check_setting():
    def error_log_and_exit(log_info, exit_code=1):
        print(f"ERROR: {log_info}")
        exit(exit_code)
    
    if bs < 1:
        error_log_and_exit("bs 应大于等于 1")
    
    if h < 1:
        error_log_and_exit("h 应大于等于 1")

    if k < 1:
        error_log_and_exit("k 应大于等于 1")

    if moe_expert_num < 1:
        error_log_and_exit("moe 专家数应大于等于 1")

    if tp_world_size == 2 and local_moe_expert_num > 1:
        error_log_and_exit("unSupported tp = 2 and local moe > 1")

    if k > moe_expert_num:
        error_log_and_exit("k 不能大于 moe_expert_num")

    if (server_index < 0) or (server_index > server_num):
        error_log_and_exit("server_index 取值范围为 [0, server_num)")

    if server_num * rank_per_dev != world_size:
        error_log_and_exit("server_num * rank_per_dev 必须等于 world_size")

    if tp_world_size * ep_world_size != world_size:
        error_log_and_exit("tp_world_size * ep_world_size 必须等于 world_size")

    if shared_expert_rank_num != shared_expert_num * rank_num_per_shared_expert:
        error_log_and_exit("shared_expert_rank_num 必须等于 shared_expert_num * rank_num_per_shared_expert")

    if shared_expert_rank_num >= ep_world_size:
        error_log_and_exit("shared_expert_rank_num 必须小于 ep_world_size")
    
    if moe_expert_num % moe_rank_num != 0:
        error_log_and_exit("moe_expert_num 必须是 moe_rank_num 的整数倍")

    if graph_type == 2:
        error_log_and_exit("当前不支持动态图模式, graph_type 合法值仅有 0 和 1")

    if comm_quant_mode == 1:
        error_log_and_exit("当前不支持 int12 通信量化, comm_quant_mode 合法值仅有 0 和 2")

    if tp_world_size == 2 and comm_quant_mode == 2:
        error_log_and_exit("仅 tp = 1 的场景支持 int8 通信量化")
    
    if (is_mask or is_2d_mask) and is_variable_bs:
        error_log_and_exit("掩膜 与 可变bs 特性不能同时使能")
    
    if is_shared_expert_x and shared_expert_rank_num > 0:
        error_log_and_exit("有共享专家卡时不应传入 shared_expert_x")
    
    if tp_world_size == 2 and is_variable_local_moe_expert_num:
        error_log_and_exit("tp = 2 时不支持可变 moe 专家数")

    if is_ccu and is_2d_mask:
        error_log_and_exit("CCU 通信方式不支持2d_mask")

# endregion

# region 性能数据处理函数

def get_performance(profiling_path: str):
    def get_execution_time(profiling_path: str):
        op_types = path_to_op_types[profiling_path]
        rank_types = ['shared', 'moe']
        performance_info = {(op_type, rank_type): [] for op_type in op_types for rank_type in rank_types}

        long_wait_ranks = set()
        for dirpath, _, filenames in os.walk(profiling_path):
            for filename in filenames:
                if filename.startswith("op_summary"):
                    op_summary = pd.read_csv(os.path.join(dirpath, filename))
                    local_rank_id = op_summary['Device_id'].iloc[0]
                    ep_rank_id = (rank_per_dev * server_index + local_rank_id) // tp_world_size
                    rank_type = 'shared' if ep_rank_id < shared_expert_rank_num else 'moe'
                    for op_type in op_types:
                        op_info = op_summary[op_summary['OP Type'] == op_type]

                        op_execute_time_series = op_info['Task Duration(us)'].tail(50)
                        performance_info[(op_type, rank_type)].append(op_execute_time_series)

                        mean_op_wait_time = op_info['Task Wait Time(us)'].tail(50).mean()
                        if mean_op_wait_time > 5:
                            long_wait_ranks.add(local_rank_id)

        print(f"Task Wait > 5 us local rank id is {sorted(long_wait_ranks)}")

        mean_performances = []
        min_performances = []
        for execute_time_series_list in performance_info.values():
            whole_execute_time_series = (
                pd.concat(execute_time_series_list) 
                if execute_time_series_list
                else pd.Series(dtype=float)
            )
            mean_performances.append(whole_execute_time_series.mean())
            min_performances.append(whole_execute_time_series.min())
        
        # 顺序：平均，最小；dispatch, combine; 共享，moe
        return mean_performances + min_performances

    def get_memory_usage(profiling_path: str):
        columns = ['rank_id', 'memory_usage']
        memory_usage_per_rank = pd.DataFrame(columns=columns)

        for dirpath, _, filenames in os.walk(profiling_path):
            for filename in filenames:
                if filename.startswith("memory_record"):
                    memory_record = pd.read_csv(os.path.join(dirpath, filename))

                    local_rank_id = int(memory_record['Device Type'][0].split(":")[1])  # memory_record['Device Type']值形如 "NPU:6"
                    rank_id = rank_per_dev * server_index + local_rank_id

                    rank_max_allocated_memory = memory_record['Total Allocated(MB)'].max()
                    rank_min_allocated_memory = memory_record['Total Allocated(MB)'].min()
                    rank_memory_usage = rank_max_allocated_memory - rank_min_allocated_memory

                    memory_usage_per_rank.loc[memory_usage_per_rank.shape[0]] = [rank_id, rank_memory_usage]

        memory_usage_per_rank['rank_id'] = memory_usage_per_rank['rank_id'].astype('int32')
        memory_usage_per_rank.sort_values(by='rank_id', inplace=True)
        memory_usage_per_rank.reset_index(drop=True, inplace=True)
        return memory_usage_per_rank

    return get_execution_time(profiling_path), get_memory_usage(profiling_path)

def get_expected_memory_usage(test_mode: str, rank_id: int):
    def get_dispatch_expected_memory_usage(ep_rank_id: int):
        is_shared_card = (ep_rank_id < shared_expert_rank_num)
        A = A_shared if is_shared_card else A_moe
        local_expert_num = 1 if is_shared_card else local_moe_expert_num

        dispatch_output_memory_usage = (
            A * tp_world_size * h * expand_x_dtype_bytes
            + A * tp_world_size * float_bytes
            + A * expand_idx_per_token * int32_bytes
            + local_expert_num * int64_bytes
            + ep_world_size * local_expert_num * int32_bytes
            + tp_world_size * int32_bytes
        )
        workspace = 16 + (512 * (aiv_core_num ** 2)) / MB_to_bytes
        dispatch_expected_memory_usage = (dispatch_output_memory_usage + workspace * MB_to_bytes) / MB_to_bytes

        return dispatch_expected_memory_usage

    def get_combine_expected_memory_usage(ep_rank_id: int, tp_rank_id: int):
        ep_x_active_mask_per_card = chunk_tensor(x_active_mask_list[tp_rank_id], ep_world_size)
        if is_2d_mask:
            real_bs = len(ep_x_active_mask_per_card[ep_rank_id])
        else:
            real_bs = ep_x_active_mask_per_card[ep_rank_id].sum().item()
        card_bs = real_bs if is_variable_bs else bs 

        combine_output_memory_usage = card_bs * h * out_dtype_bytes
        workspace = 16
        combine_expected_memory_usage = (combine_output_memory_usage + workspace * MB_to_bytes) / MB_to_bytes

        return combine_expected_memory_usage

    expand_idx_per_token = 128

    expand_x_dtype_bytes = 1 if is_quant else 2
    out_dtype_bytes = 2
    float_bytes = 4
    int32_bytes = 4
    int64_bytes = 8

    MB_to_bytes = 1024 ** 2

    ep_rank_id = rank_id // tp_world_size
    tp_rank_id = rank_id % tp_world_size

    if test_mode == 'dispatch':
        return get_dispatch_expected_memory_usage(ep_rank_id)

    if test_mode == 'combine':
        return get_combine_expected_memory_usage(ep_rank_id, tp_rank_id)

    if test_mode == 'continue':
        return max(get_dispatch_expected_memory_usage((ep_rank_id)), get_combine_expected_memory_usage(ep_rank_id, tp_rank_id))

def print_dispatch_performance(dispatch_performance):
    (mean_share_performance, mean_moe_performance, min_share_performance, min_moe_performance), memory_usage = dispatch_performance

    print(f'[INFO] MC2融合算子Average性能为: dispatch共享:{mean_share_performance} us, dispatch Moe:{mean_moe_performance} us\n')
    print(f'[INFO] MC2融合算子Min性能为: dispatch共享:{min_share_performance} us, dispatch Moe:{min_moe_performance} us\n')
    print(f'[INFO] Dispatch算子内存占用:')
    print_dataframe(memory_usage)

def print_combine_performance(combine_performance):
    (mean_share_performance, mean_moe_performance, min_share_performance, min_moe_performance), memory_usage = combine_performance

    print(f'[INFO] MC2融合算子Average性能为: combine共享:{mean_share_performance} us, combine Moe:{mean_moe_performance} us\n')
    print(f'[INFO] MC2融合算子Min性能为: combine共享:{min_share_performance} us, combine Moe:{min_moe_performance} us\n')
    print(f'[INFO] Combine算子内存占用:')
    print_dataframe(memory_usage)

def print_continue_performance(continue_performance):
    (
        dispatch_share_performance, dispatch_moe_performance, combine_share_performance, combine_moe_performance, 
        min_dispatch_share_performance, min_dispatch_moe_performance, min_combine_share_performance, min_combine_moe_performance
    ), memory_usage = continue_performance

    print(
        f'[INFO] MC2融合算子Average性能为:'
        f'dispatch共享: {dispatch_share_performance} us, dispatch Moe: {dispatch_moe_performance} us, '
        f'combine共享: {combine_share_performance} us, combine Moe: {combine_moe_performance} us\n'
    )
    print(
        f'[INFO] MC2融合算子Min性能为:'
        f'dispatch共享:{min_dispatch_share_performance} us, dispatch Moe:{min_dispatch_moe_performance} us, '
        f'combine共享:{min_combine_share_performance} us, combine Moe:{min_combine_moe_performance} us\n'
    )
    print(f'[INFO] Dispatch与Combine算子总内存占用:')
    print_dataframe(memory_usage)

def save_performance_result(test_mode, case_name, op_types, op_performances, op_memory_usage_info):
    def save_execution_time_result(test_mode, case_name, op_types, op_performances):
        csv_name = f'{test_mode}_performance_result.csv'
        title = [
            'op_type', 'case_name',
            'mc2_shared_average', 'mc2_moe_average', 'mc2_shared_min', 'mc2_moe_min',
            'recordtime'
        ]

        for op_type, op_performance in zip(op_types, op_performances):
            row = [op_type, case_name] + op_performance + [time.strftime("%Y-%m-%d %H:%M:%S")]    
            if not os.path.exists(f"csv_result/{csv_name}"):
                write_row_to_csv(csv_name, title)
            write_row_to_csv(csv_name, row)

    def save_memory_usage_result(test_mode: str, case_name: str, op_types: list, op_memory_usage_info: pd.DataFrame):
        tolerance_coefficient = 1.05

        csv_name = 'memory_usage_result.csv'
        title = [
            'op_types', 'case_name', 'rank_id',
            'expected_memory_usage', 'real_memory_usage', 'expected_memory_usage_with_tolerance',
            'result',
            'recordtime'
        ]

        df = pd.DataFrame(index=op_memory_usage_info.index, columns=title)
        df['op_types'] = ', '.join(op_types)
        df['case_name'] = case_name
        df['rank_id'] = op_memory_usage_info['rank_id']
        df['expected_memory_usage'] = op_memory_usage_info['rank_id'].apply(
            lambda rank_id: get_expected_memory_usage(test_mode, rank_id)
        )
        df['real_memory_usage'] = op_memory_usage_info['memory_usage']
        df['expected_memory_usage_with_tolerance'] = (df['expected_memory_usage'] * tolerance_coefficient)
        df['result'] = (df['real_memory_usage'] < df['expected_memory_usage_with_tolerance']).apply(
            lambda is_less: 'PASS' if is_less else 'FAIL'
        )
        df['recordtime'] = time.strftime("%Y-%m-%d %H:%M:%S")

        if not os.path.exists("csv_result"):
            os.mkdir("csv_result")
        set_header = (not os.path.exists(f"csv_result/{csv_name}"))
        df.to_csv(f"csv_result/{csv_name}", header=set_header, index=False, mode='a')

    save_execution_time_result(test_mode, case_name, op_types, op_performances)
    save_memory_usage_result(test_mode, case_name, op_types, op_memory_usage_info)

def get_dispatch_performance():
    dispatch_performance = get_performance(dispatch_profiling_path)

    print_dispatch_performance(dispatch_performance)

    op_types = path_to_op_types[dispatch_profiling_path]
    save_performance_result('dispatch', current_case_name, op_types, [dispatch_performance[0]], dispatch_performance[1])

    return dispatch_performance

def get_combine_performance():
    combine_performance = get_performance(combine_profiling_path)

    print_combine_performance(combine_performance)

    op_types = path_to_op_types[combine_profiling_path]
    save_performance_result('combine', current_case_name, op_types, [combine_performance[0]], combine_performance[1])

    return combine_performance

def get_continue_performance():
    continue_performance, memory_usage = get_performance(continue_profiling_path)

    print_continue_performance([continue_performance, memory_usage])

    dispatch_performance = continue_performance[:2] + continue_performance[4:6]
    combine_performance = continue_performance[2:4] + continue_performance[6:]
    op_types = path_to_op_types[continue_profiling_path]
    save_performance_result('continue', current_case_name, op_types, [dispatch_performance, combine_performance], memory_usage)

    return [continue_performance, memory_usage]

# endregion

# region 图模式相关类和函数

def compile_model(model):
    if graph_type == 0:
        print('单算子模式下不应调用compile_model函数')
        exit(1)

    print(f"------------------graph_type: {graph_type}------------------", )
    import torchair

    from torchair.configs.compiler_config import CompilerConfig
    if is_full_core:
        compiler_config = None
    else:
        # 使用一半的核
        compiler_config = CompilerConfig()
        compiler_config.ge_config.aicore_num = f"{aic_core_num}|{aiv_core_num}"
        print(f"当前图模式仅使用: {aic_core_num} aic核, {aiv_core_num} aiv核")
    npu_backend = torchair.get_npu_backend(compiler_config=compiler_config)

    if graph_type == 1:  
        # 传统入图模式，静态shape+在线编译场景
        print('graph_type=1, 测试场景: 传统入图模式，静态shape+在线编译场景')
        compiled_model = torch.compile(model, backend=npu_backend, dynamic=False)
    elif graph_type == 2:  
        # ACLNN入图模式，动态shape+二进制
        print('graph_type=2, 测试场景: ACLNN入图模式，动态shape+二进制场景')
        compiled_model = torch.compile(model, backend=npu_backend, dynamic=True)

    return compiled_model

class MOE_DISTRIBUTE_DISPATCH_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, x, expert_ids, x_active_mask, scales, group_ep, ep_rank_id, ep_world_size, tp_world_size,
        expert_shard_type, shared_expert_num, shared_expert_rank_num, moe_expert_num, quant_mode, global_bs, y_dtype,
        expert_token_nums_type, comm_alg, group_tp=None, tp_rank_id=None
    ):
        if include_tp_info:
            output = torch_npu.npu_moe_distribute_dispatch_v2(
                x=x,
                expert_ids=expert_ids,
                x_active_mask=x_active_mask,
                group_ep=group_ep,
                group_tp=group_tp,
                ep_rank_id=ep_rank_id,
                tp_rank_id=tp_rank_id,
                ep_world_size=ep_world_size,
                tp_world_size=tp_world_size,
                expert_shard_type=expert_shard_type,
                shared_expert_num=shared_expert_num,
                shared_expert_rank_num=shared_expert_rank_num,
                moe_expert_num=moe_expert_num,
                scales=scales,
                quant_mode=quant_mode,
                global_bs=global_bs,
                y_dtype=y_dtype,
                expert_token_nums_type=expert_token_nums_type,
                comm_alg=comm_alg
            )
        else:
            output = torch_npu.npu_moe_distribute_dispatch_v2(
                x=x,
                expert_ids=expert_ids,
                x_active_mask=x_active_mask,
                group_ep=group_ep,
                ep_rank_id=ep_rank_id,
                ep_world_size=ep_world_size,
                tp_world_size=tp_world_size,
                expert_shard_type=expert_shard_type,
                shared_expert_num=shared_expert_num,
                shared_expert_rank_num=shared_expert_rank_num,
                moe_expert_num=moe_expert_num,
                scales=scales,
                quant_mode=quant_mode,
                global_bs=global_bs,
                y_dtype=y_dtype,
                expert_token_nums_type=expert_token_nums_type,
                comm_alg=comm_alg
            )
        return output

class MOE_DISTRIBUTE_COMBINE_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(
        self, expand_x, expert_ids, assist_info_for_combine, ep_send_counts, tp_send_counts, expert_scales,
        x_active_mask, shared_expert_x, group_ep, ep_rank_id, ep_world_size, tp_world_size,
        expert_shard_type, shared_expert_num, shared_expert_rank_num, moe_expert_num, comm_quant_mode, global_bs,
        comm_alg, group_tp=None, tp_rank_id=None
    ):
        if include_tp_info:
            output = torch_npu.npu_moe_distribute_combine_v2(
                expand_x=expand_x,
                expert_ids=expert_ids,
                assist_info_for_combine=assist_info_for_combine,
                ep_send_counts=ep_send_counts,
                tp_send_counts=tp_send_counts,
                x_active_mask=x_active_mask,
                expert_scales=expert_scales,
                shared_expert_x=shared_expert_x,
                group_ep=group_ep,
                group_tp=group_tp,
                ep_rank_id=ep_rank_id,
                tp_rank_id=tp_rank_id,
                ep_world_size=ep_world_size,
                tp_world_size=tp_world_size,
                expert_shard_type=expert_shard_type,
                shared_expert_num=shared_expert_num,
                shared_expert_rank_num=shared_expert_rank_num,
                moe_expert_num=moe_expert_num,
                comm_quant_mode=comm_quant_mode,
                global_bs=global_bs,
                comm_alg=comm_alg
            )
        else:
            output = torch_npu.npu_moe_distribute_combine_v2(
                expand_x=expand_x,
                expert_ids=expert_ids,
                assist_info_for_combine=assist_info_for_combine,
                ep_send_counts=ep_send_counts,
                tp_send_counts=tp_send_counts,
                x_active_mask=x_active_mask,
                expert_scales=expert_scales,
                shared_expert_x=shared_expert_x,
                group_ep=group_ep,
                ep_rank_id=ep_rank_id,
                ep_world_size=ep_world_size,
                tp_world_size=tp_world_size,
                expert_shard_type=expert_shard_type,
                shared_expert_num=shared_expert_num,
                shared_expert_rank_num=shared_expert_rank_num,
                moe_expert_num=moe_expert_num,
                comm_quant_mode=comm_quant_mode,
                global_bs=global_bs,
                comm_alg=comm_alg
            )
        return output

class MOE_DISTRIBUTE_CONTINUE_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, x, expert_ids, expert_scales, x_active_mask, scales, shared_expert_x,
        group_ep, ep_rank_id, ep_world_size, tp_world_size,
        expert_shard_type, shared_expert_num, shared_expert_rank_num, moe_expert_num,
        quant_mode, comm_quant_mode, global_bs, y_dtype, golden_expand_x,
        expert_token_nums_type, dispatch_comm_alg, combine_comm_alg, group_tp=None, tp_rank_id=None
    ):
        x_combine_res = None
        local_loop = loop if is_performance else 1
        for _ in range(local_loop):
            if include_tp_info:
                output_dispatch_npu = torch_npu.npu_moe_distribute_dispatch_v2(
                    x=x,
                    expert_ids=expert_ids,
                    x_active_mask=x_active_mask,
                    group_ep=group_ep,
                    group_tp=group_tp,
                    ep_rank_id=ep_rank_id,
                    tp_rank_id=tp_rank_id,
                    ep_world_size=ep_world_size,
                    tp_world_size=tp_world_size,
                    expert_shard_type=expert_shard_type,
                    shared_expert_num=shared_expert_num,
                    shared_expert_rank_num=shared_expert_rank_num,
                    moe_expert_num=moe_expert_num,
                    scales=scales,
                    quant_mode=quant_mode,
                    global_bs=global_bs,
                    y_dtype=y_dtype,
                    expert_token_nums_type=expert_token_nums_type,
                    comm_alg=dispatch_comm_alg
                )
            else:
                output_dispatch_npu = torch_npu.npu_moe_distribute_dispatch_v2(
                    x=x,
                    expert_ids=expert_ids,
                    x_active_mask=x_active_mask,
                    group_ep=group_ep,
                    ep_rank_id=ep_rank_id,
                    ep_world_size=ep_world_size,
                    tp_world_size=tp_world_size,
                    expert_shard_type=expert_shard_type,
                    shared_expert_num=shared_expert_num,
                    shared_expert_rank_num=shared_expert_rank_num,
                    moe_expert_num=moe_expert_num,
                    scales=scales,
                    quant_mode=quant_mode,
                    global_bs=global_bs,
                    y_dtype=y_dtype,
                    expert_token_nums_type=expert_token_nums_type,
                    comm_alg=dispatch_comm_alg
                )
            
            expand_x_npu, _, expand_idx_npu, expert_token_nums_npu, ep_recv_counts_npu, tp_recv_counts_npu, _ = output_dispatch_npu  
            
            if is_continue_golden_expand_x:
                expand_x_npu = golden_expand_x
            else:
                if expand_x_npu.dtype not in (torch.float16, torch.bfloat16):
                    expand_x_npu = convert_expand_x_dtype(expand_x_npu)
                real_rank_token_num = expert_token_nums_npu.sum() \
                        if expert_token_nums_type == 1 \
                        else expert_token_nums_npu[-1]
                L = expand_x_npu.shape[0]
                mask = torch.arange(L, device=expand_x_npu.device) >= real_rank_token_num
                expand_x_npu = expand_x_npu.masked_fill_(mask.unsqueeze(-1), 0)
            
            if include_tp_info:
                output_combine_npu = torch_npu.npu_moe_distribute_combine_v2(
                    expand_x=expand_x_npu,
                    expert_ids=expert_ids,
                    assist_info_for_combine=expand_idx_npu,
                    ep_send_counts=ep_recv_counts_npu,
                    tp_send_counts=tp_recv_counts_npu,
                    expert_scales=expert_scales,
                    x_active_mask=x_active_mask,
                    shared_expert_x=shared_expert_x,
                    group_ep=group_ep,
                    group_tp=group_tp,
                    ep_rank_id=ep_rank_id,
                    tp_rank_id=tp_rank_id,
                    ep_world_size=ep_world_size,
                    tp_world_size=tp_world_size,
                    expert_shard_type=expert_shard_type,
                    shared_expert_num=shared_expert_num,
                    shared_expert_rank_num=shared_expert_rank_num,
                    moe_expert_num=moe_expert_num,
                    comm_quant_mode=comm_quant_mode,
                    global_bs=global_bs,
                    comm_alg=combine_comm_alg
                )
            else:
                output_combine_npu = torch_npu.npu_moe_distribute_combine_v2(
                    expand_x=expand_x_npu,
                    expert_ids=expert_ids,
                    assist_info_for_combine=expand_idx_npu,
                    ep_send_counts=ep_recv_counts_npu,
                    tp_send_counts=tp_recv_counts_npu,
                    expert_scales=expert_scales,
                    x_active_mask=x_active_mask,
                    shared_expert_x=shared_expert_x,
                    group_ep=group_ep,
                    ep_rank_id=ep_rank_id,
                    ep_world_size=ep_world_size,
                    tp_world_size=tp_world_size,
                    expert_shard_type=expert_shard_type,
                    shared_expert_num=shared_expert_num,
                    shared_expert_rank_num=shared_expert_rank_num,
                    moe_expert_num=moe_expert_num,
                    comm_quant_mode=comm_quant_mode,
                    global_bs=global_bs,
                    comm_alg=combine_comm_alg
                )
                                                       
            x = output_combine_npu
            # x_combine_res取combine算子第一次的输出
            if x_combine_res == None:
                x_combine_res = output_combine_npu
        
        return [x_combine_res, output_combine_npu]

# endregion

# region npu执行算子相关函数

def set_device(rank):
    torch_npu.npu.set_device(rank % rank_per_dev)
    print(f"current device set: {torch_npu.npu.current_device()}")

def init_hccl_comm(rank):
    print(f'[INFO] device_{rank} 创建HCCL通信链路')
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:50001')
    print(f"device_{rank} init_process_group success")
    
    print(f"device {rank} 初始化EP域")
    for ep_ranks in ep_ranks_list:
        tmp_group = dist.new_group(backend="hccl", ranks=ep_ranks)
        if rank in ep_ranks:
            ep_group = tmp_group

    print(f"device {rank} 初始化TP域")
    for tp_ranks in tp_ranks_list:
        tmp_group = dist.new_group(backend="hccl", ranks=tp_ranks)
        if rank in tp_ranks:
            tp_group = tmp_group

    ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    tp_hcomm_info = tp_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    
    return ep_hcomm_info, tp_hcomm_info, ep_group, tp_group

def get_dispatch_kwargs(
    x, expert_ids, x_active_mask, scales,
    group_ep, group_tp, ep_rank_id, tp_rank_id
):
    x = x.to(input_dtype).npu()
    expert_ids = expert_ids.to(torch.int32).npu()
    x_active_mask = x_active_mask.to(torch.bool).npu() if is_mask or is_2d_mask else None
    if is_quant and is_quant_scales:
        scales = scales.to(torch.float32).npu()
    
    return {
        'x': x,
        'expert_ids': expert_ids,
        'x_active_mask': x_active_mask,
        'group_ep': group_ep,
        'group_tp': group_tp,
        'ep_rank_id': ep_rank_id,
        'tp_rank_id': tp_rank_id,
        'ep_world_size': ep_world_size,
        'tp_world_size': tp_world_size,
        'expert_shard_type': expert_shard_type,
        'shared_expert_num': shared_expert_num,
        'shared_expert_rank_num': shared_expert_rank_num,
        'moe_expert_num': moe_expert_num,
        'scales': scales,
        'quant_mode': quant_mode,
        'global_bs': global_bs,
        'y_dtype': get_dtype_int_value(expandx_dtype),
        'expert_token_nums_type': expert_token_nums_type,
        'comm_alg': dispatch_comm_alg
    }

def get_combine_kwargs(
    expand_x, expert_ids, expand_idx, ep_send_counts, tp_send_counts, expert_scales,
    x_active_mask, shared_expert_x, group_ep, group_tp, ep_rank_id, tp_rank_id,
):
    expand_x = expand_x.to(input_dtype).npu()
    expert_ids = expert_ids.to(torch.int32).npu()
    expand_idx = expand_idx.to(torch.int32).npu()
    ep_send_counts = ep_send_counts.to(torch.int32).npu()
    tp_send_counts = tp_send_counts.to(torch.int32).npu()
    expert_scales = expert_scales.to(torch.float32).npu()
    x_active_mask = x_active_mask.to(torch.bool).npu() if is_mask or is_2d_mask else None
    shared_expert_x = shared_expert_x.to(input_dtype).npu() if is_shared_expert_x else None

    return {
        'expand_x': expand_x,
        'expert_ids': expert_ids,
        'assist_info_for_combine': expand_idx,
        'ep_send_counts': ep_send_counts,
        'tp_send_counts': tp_send_counts,
        'x_active_mask': x_active_mask,
        'expert_scales': expert_scales,
        'shared_expert_x': shared_expert_x,
        'group_ep': group_ep,
        'group_tp': group_tp,
        'ep_rank_id': ep_rank_id,
        'tp_rank_id': tp_rank_id,
        'ep_world_size': ep_world_size,
        'tp_world_size': tp_world_size,
        'expert_shard_type': expert_shard_type,
        'shared_expert_num': shared_expert_num,
        'shared_expert_rank_num': shared_expert_rank_num,
        'moe_expert_num': moe_expert_num,
        'comm_quant_mode': comm_quant_mode,
        'global_bs': global_bs,
        'comm_alg': combine_comm_alg
    }

def run_dispatch_npu(queue, rank, x, expert_ids, x_active_mask, scales):
    try:
        set_device(rank)
        ep_hcomm_info, tp_hcomm_info, _, _ = init_hccl_comm(rank)
        if is_save_tensor_to_bin:
            save_tensor_to_bin(x.to(torch.float32), "x_in_" + str(rank))
            save_tensor_to_bin(expert_ids.to(torch.int32), "expertIds_" + str(rank))
            if is_quant and is_quant_scales:
                save_tensor_to_bin(scales, "scales_" + str(rank))
        
        print(f'[INFO] device_{rank} 构造dispatch算子输入数据')
        dispatch_kwargs = get_dispatch_kwargs(
            x=x,
            expert_ids=expert_ids,
            x_active_mask=x_active_mask,
            scales=scales,
            group_ep=ep_hcomm_info,
            group_tp=tp_hcomm_info,
            ep_rank_id=rank//tp_world_size,
            tp_rank_id=rank%tp_world_size,
        )

        if not include_tp_info:
            del dispatch_kwargs['group_tp']
            del dispatch_kwargs['tp_rank_id']

        if graph_type == 0:
            if not is_performance:
                (
                    expand_x, dynamic_scales, expand_idx, 
                    expert_token_nums, ep_recv_counts, tp_recv_counts, _
                ) = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_kwargs)
            else:
                tensor = torch.ones(10000, 10000).to(torch.float16).npu()
                with get_profiling(loop, repeat, warmup, dispatch_profiling_path) as prof:
                    for i in range(loop):
                        torch.npu.synchronize()

                        (
                            expand_x, dynamic_scales, expand_idx, 
                            expert_token_nums, ep_recv_counts, tp_recv_counts, _
                        ) = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_kwargs)

                        if i == 1:
                            for _ in range(100):
                                torch.exp(tensor)

                        prof.step()

        else:
            model = compile_model(MOE_DISTRIBUTE_DISPATCH_GRAPH_Model())
            (
                expand_x, dynamic_scales, expand_idx,
                expert_token_nums, ep_recv_counts, tp_recv_counts, _
            ) = model(**dispatch_kwargs)
        
        torch.npu.synchronize()
        dist.destroy_process_group()
        print(f'rank {rank} epid {rank//tp_world_size} tpid {rank%tp_world_size} npu finished! \n')
        if is_save_tensor_to_bin:
            save_tensor_to_bin(expand_x.to(torch.float32), "expand_x_out_" + str(rank))
            save_tensor_to_bin(expand_idx, "expand_idx_" + str(rank))
            save_tensor_to_bin(expert_token_nums, "expert_token_nums_" + str(rank))
            save_tensor_to_bin(ep_recv_counts, "ep_recv_counts_" + str(rank))
            save_tensor_to_bin(tp_recv_counts, "tp_recv_counts_" + str(rank))
            if is_quant:
                save_tensor_to_bin(dynamic_scales, "dynamic_scales_" + str(rank))

        # 获得expand_x, expand_idx和dynamic_scales的真实值，并放入队列
        # 注意, expand_idx无法用相同方式获取真实值，因为real_rank_token_num是all gather之后的token数
        real_rank_token_num = expert_token_nums.cpu().sum().item() \
                            if expert_token_nums_type == 1 \
                            else expert_token_nums.cpu()[-1].item()
        real_origin_token_num = (
            tp_recv_counts[rank%tp_world_size].cpu().item()
            if tp_world_size > 1 
            else real_rank_token_num
        )
        expand_x = expand_x[:real_rank_token_num, :]
        if is_ccu:
            expand_idx = expand_idx[:torch.sum(x_active_mask) * k]   # A5 dispatch_v2 用的不是三元组
        else:   # MTE
            expand_idx = expand_idx[:real_origin_token_num * 3]
        dynamic_scales = dynamic_scales[:real_rank_token_num]
        print("============================Queue.put============================")
        queue.put([
                    rank, 
                    [
                        expand_x.cpu(), dynamic_scales.cpu(), expand_idx.cpu(),
                        expert_token_nums.cpu(), ep_recv_counts.cpu(), tp_recv_counts.cpu()
                    ]
                ])
    except Exception as e:
        print("============================Exception catch============================")
        error_msg = f"Rank {rank} failed with error:\n{traceback.format_exc()}"
        print(error_msg)
        queue.put([rank, None])
        print("============================Print err============================")

def run_combine_npu(
    queue, rank, expand_x, expert_ids, expand_idx, ep_send_counts, tp_send_counts, x_active_mask, expert_scales, shared_expert_x
):
    try:
        set_device(rank)
        ep_hcomm_info, tp_hcomm_info, ep_group, _ = init_hccl_comm(rank)
        if is_save_tensor_to_bin:
            save_tensor_to_bin(expand_x.to(torch.float32), "expand_x_"+str(rank))
            save_tensor_to_bin(expert_ids.to(torch.int32), "expertIds_"+str(rank))
            save_tensor_to_bin(expand_idx.to(torch.int32), "expandIdx_"+str(rank))
            save_tensor_to_bin(ep_send_counts.to(torch.int32), "epSendCounts_"+str(rank))
            save_tensor_to_bin(tp_send_counts.to(torch.int32), "tpSendCounts_"+str(rank))
            save_tensor_to_bin(expert_scales.to(torch.float32), "expertScales_"+str(rank))

        print(f'[INFO] device_{rank} 构造combine算子输入数据')
        combine_kwargs = get_combine_kwargs(
            expand_x=expand_x,
            expert_ids=expert_ids,
            expand_idx=expand_idx,
            ep_send_counts=ep_send_counts,
            tp_send_counts=tp_send_counts,
            expert_scales=expert_scales,
            x_active_mask=x_active_mask,
            shared_expert_x=shared_expert_x,
            group_ep=ep_hcomm_info,
            group_tp=tp_hcomm_info,
            ep_rank_id=rank//tp_world_size,
            tp_rank_id=rank%tp_world_size,
        )

        if not include_tp_info:
            del combine_kwargs['group_tp']
            del combine_kwargs['tp_rank_id']

        if graph_type == 0:
            if not is_performance:
                x = torch_npu.npu_moe_distribute_combine_v2(**combine_kwargs)
            else:
                x = torch_npu.npu_moe_distribute_combine_v2(**combine_kwargs)

                tensor = torch.ones(10000, 10000).to(torch.float16).npu()
                with get_profiling(loop, repeat, warmup, combine_profiling_path) as prof:
                    for i in range(loop):
                        torch.npu.synchronize()

                        # 做全卡同步，因为combine只会同步一部分卡
                        dist.all_reduce(combine_kwargs['expand_x'][0,0], op=ReduceOp.SUM, async_op=False, group=ep_group)

                        _ = torch_npu.npu_moe_distribute_combine_v2(**combine_kwargs)

                        if i == 1:
                            for _ in range(100):
                                torch.exp(tensor)
                
                        prof.step()

        else :
            model = compile_model(MOE_DISTRIBUTE_COMBINE_GRAPH_Model())
            x = model(**combine_kwargs)

        torch.npu.synchronize()
        dist.destroy_process_group()
        print(f'rank {rank} epid {rank//tp_world_size} tpid {rank%tp_world_size} npu finished! \n')
        
        # 获得x中的有效token
        if is_mask and not is_2d_mask:
            real_tokens = x_active_mask.cpu().sum().item()
            x = x[:real_tokens, :]
        queue.put([
            rank, 
            [
                x.cpu()
            ]
        ])
    except Exception as e:
        print("============================Exception catch============================")
        error_msg = f"Rank {rank} failed with error:\n{traceback.format_exc()}"
        print(error_msg)
        queue.put([rank, None])
        print("============================Print err============================")

def run_continue_npu(queue, rank, x, expert_ids, x_active_mask, scales, expert_scales, shared_expert_x, golden_expand_x):
    try:
        set_device(rank)
        ep_hcomm_info, tp_hcomm_info, _, _ = init_hccl_comm(rank)
        
        print(f'[INFO] device_{rank} 构造两算子输入数据')
        expert_scales = expert_scales.to(torch.float32).npu() # 图模式下传参需要
        shared_expert_x = shared_expert_x.to(input_dtype).npu() if is_shared_expert_x else None # 图模式下传参需要
        golden_expand_x = golden_expand_x.to(input_dtype).npu()

        dispatch_kwargs = get_dispatch_kwargs(
            x=x,
            expert_ids=expert_ids,
            x_active_mask=x_active_mask,
            scales=scales,
            group_ep=ep_hcomm_info,
            group_tp=tp_hcomm_info,
            ep_rank_id=rank//tp_world_size,
            tp_rank_id=rank%tp_world_size,
        )

        combine_kwargs = get_combine_kwargs(
            expand_x=torch.empty(0),
            expert_ids=expert_ids,
            expand_idx=torch.empty(0),
            ep_send_counts=torch.empty(0),
            tp_send_counts=torch.empty(0),
            expert_scales=expert_scales,
            x_active_mask=x_active_mask,
            shared_expert_x=shared_expert_x,
            group_ep=ep_hcomm_info,
            group_tp=tp_hcomm_info,
            ep_rank_id=rank//tp_world_size,
            tp_rank_id=rank%tp_world_size,
        )

        if not include_tp_info:
            del dispatch_kwargs['group_tp']
            del dispatch_kwargs['tp_rank_id']
            del combine_kwargs['group_tp']
            del combine_kwargs['tp_rank_id']

        if graph_type == 0:
            if not is_performance:
                expand_x, _, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts, _ = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_kwargs)

                if is_continue_golden_expand_x:
                    expand_x = golden_expand_x
                else:
                    if is_quant:
                        expand_x = convert_expand_x_dtype(expand_x)
                    real_rank_token_num = expert_token_nums.sum().item() \
                            if expert_token_nums_type == 1 \
                            else expert_token_nums[-1].item()
                    expand_x[real_rank_token_num:, :] = 0
                
                combine_kwargs.update(expand_x=expand_x, assist_info_for_combine=expand_idx, ep_send_counts=ep_recv_counts, tp_send_counts=tp_recv_counts)
                x = torch_npu.npu_moe_distribute_combine_v2(**combine_kwargs)

            else:
                tensor = torch.ones(10000, 10000).to(torch.float16).npu()
                with get_profiling(loop, repeat, warmup, continue_profiling_path) as prof:
                    for i in range(loop):
                        torch.npu.synchronize()

                        expand_x, _, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts, _ = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_kwargs)
                        
                        if is_continue_golden_expand_x:
                            expand_x = golden_expand_x
                        else:
                            if is_quant:
                                expand_x = convert_expand_x_dtype(expand_x)
                            real_rank_token_num = expert_token_nums.sum().item() \
                                    if expert_token_nums_type == 1 \
                                    else expert_token_nums[-1].item()
                            expand_x[real_rank_token_num:, :] = 0
                        combine_kwargs.update(expand_x=expand_x, assist_info_for_combine=expand_idx, ep_send_counts=ep_recv_counts, tp_send_counts=tp_recv_counts)

                        x = torch_npu.npu_moe_distribute_combine_v2(**combine_kwargs)
                        
                        if i == 1:
                            for _ in range(100):
                                torch.exp(tensor)
                    
                        prof.step()
            
        else :
            continue_graph_kwargs = {
                **dispatch_kwargs,
                "expert_scales": expert_scales,
                "shared_expert_x": shared_expert_x,
                "golden_expand_x": golden_expand_x,
                'comm_quant_mode': comm_quant_mode,
            }
            del continue_graph_kwargs["comm_alg"]
            continue_graph_kwargs["dispatch_comm_alg"] = dispatch_comm_alg
            continue_graph_kwargs["combine_comm_alg"] = combine_comm_alg

            model = compile_model(MOE_DISTRIBUTE_CONTINUE_GRAPH_Model())
            if not is_performance:
                x, _ = model(**continue_graph_kwargs)
            else:
                loop_time = 2
                repeat_time = 1
                warm_time = 1
                prof = get_profiling(loop_time, repeat_time, warm_time, continue_profiling_path)
                prof.start()
            
                for _ in range(loop_time):
                    x, _ = model(**continue_graph_kwargs)
                    prof.step()
            
                prof.stop()

        torch.npu.synchronize()
        dist.destroy_process_group()
        print(f'rank {rank} epid {rank//tp_world_size} tpid {rank%tp_world_size} npu finished! \n')
        
        # 获得x中的有效token
        if is_mask and not is_2d_mask:
            real_tokens = x_active_mask.cpu().sum().item()
            x = x[:real_tokens, :]
        queue.put([
            rank,
            [
                x.cpu()
            ]
        ])
    except Exception as e:
        print("============================Exception catch============================")
        error_msg = f"Rank {rank} failed with error:\n{traceback.format_exc()}"
        print(error_msg)
        queue.put([rank, None])
        print("============================Print err============================")

def gen_npu(target_func, **server_kwargs):
    def parse_rank_input(target_func, result_queue, rank, server_kwargs):
        def pad_expand_x(expand_x: torch.Tensor, ep_id: int):
            A = A_shared if ep_id < shared_expert_rank_num else A_moe
            pad_for_expand_x = torch.empty(size=[tp_world_size*A - expand_x.size(0), h], dtype=expand_x.dtype)
            return torch.cat([expand_x, pad_for_expand_x])

        def pad_expand_idx(expand_idx: torch.Tensor, ep_id: int):
            if is_ccu:
                return expand_idx
            A = A_shared if ep_id < shared_expert_rank_num else A_moe
            pad_for_expand_idx = torch.empty(size=[A*128 - expand_idx.size(0),], dtype=expand_idx.dtype)
            return torch.cat([expand_idx, pad_for_expand_idx])
        
        def pad_shared_expert_x(shared_expert_x: torch.Tensor):
            if (not is_shared_expert_x) or (not is_mask):
                return shared_expert_x
            
            pad_for_shared_expert_x = torch.empty(size=[bs - shared_expert_x.size(0), h], dtype=shared_expert_x.dtype)
            return torch.cat([shared_expert_x, pad_for_shared_expert_x])
        
        def filter_tensor(tensor: torch.Tensor, mask: torch.Tensor):
            return tensor[mask] if is_variable_bs else tensor
        
        ep_id = rank // tp_world_size
        tp_id = rank % tp_world_size

        mask = server_kwargs["x_active_mask_list"][tp_id][ep_id]        
        
        if target_func == run_dispatch_npu:
            return {
                "queue": result_queue,
                "rank": rank,
                "x": filter_tensor(server_kwargs["x_list"][tp_id][ep_id], mask),
                "expert_ids": filter_tensor(server_kwargs["expert_ids_list"][tp_id][ep_id], mask),
                "x_active_mask": server_kwargs["x_active_mask_list"][tp_id][ep_id],
                "scales": server_kwargs["scales_list"][tp_id],
            }

        if target_func == run_combine_npu:
            return {
                "queue": result_queue,
                "rank": rank,
                "expand_x": pad_expand_x(server_kwargs["expand_x_list"][tp_id][ep_id], ep_id),
                "expert_ids": filter_tensor(server_kwargs["expert_ids_list"][tp_id][ep_id], mask),
                "expand_idx": pad_expand_idx(server_kwargs["expand_idx_list"][rank], ep_id),
                "ep_send_counts": server_kwargs["ep_send_count_list"][rank],
                "tp_send_counts": server_kwargs["tp_send_count_list"][rank],
                "x_active_mask": server_kwargs["x_active_mask_list"][tp_id][ep_id],
                "expert_scales": filter_tensor(server_kwargs["expert_scales_list"][tp_id][ep_id], mask),
                "shared_expert_x": pad_shared_expert_x(server_kwargs["shared_expert_x_list"][tp_id][ep_id]),
            }
        
        if target_func == run_continue_npu:
            return {
                "queue": result_queue,
                "rank": rank,
                "x": filter_tensor(server_kwargs["x_list"][tp_id][ep_id], mask),
                "expert_ids": filter_tensor(server_kwargs["expert_ids_list"][tp_id][ep_id], mask),
                "x_active_mask": server_kwargs["x_active_mask_list"][tp_id][ep_id],
                "scales": server_kwargs["scales_list"][tp_id],
                "expert_scales": filter_tensor(server_kwargs["expert_scales_list"][tp_id][ep_id], mask),
                "shared_expert_x": pad_shared_expert_x(server_kwargs["shared_expert_x_list"][tp_id][ep_id]),
                "golden_expand_x": pad_expand_x(server_kwargs["golden_expand_x_list"][tp_id][ep_id], ep_id),
            }
    
    if target_func not in [run_dispatch_npu, run_combine_npu, run_continue_npu]:
        print("target_func not supported!")
        exit(1)
    
    if int(server_num) > 1:
        print(f"multi_server scene!!!!! {server_num=}, {world_size=}, {server_index=}. Only support running case one by one!!!!!!!!!!!!!")
        rank_list = list(range(server_index * rank_per_dev, (server_index + 1) * rank_per_dev))
    else:
        print("single_server scene!!!!!")
        rank_list = list(range(world_size))
    print(f"rank list is: {rank_list}")

    # 多进程，每个进程调用一张卡执行目标函数
    proc_list = []
    manager = Manager()
    result_queue = manager.Queue()
    mp.set_start_method("forkserver", force=True)
    for rank in rank_list:
        rank_kwargs = parse_rank_input(target_func, result_queue, rank, server_kwargs)
        proc = Process(target=target_func, kwargs=rank_kwargs)
        print("============================Start Proc============================")
        proc.start()
        proc_list.append(proc)


    rank_outputs = [None] * rank_per_dev
    for proc in proc_list:
        print("============================Begin To Get Proc============================")
        rank_id, rank_output = result_queue.get()
        local_rank_id = rank_id - server_index * rank_per_dev
        rank_outputs[local_rank_id] = rank_output
        print("============================Out To Proc============================")


    for proc in proc_list:
        proc.join()

    if None in rank_outputs:
        print("[ERROR] Task failed! Please check the detailed error logs printed by the subprocesses.")
        exit(1)
    
    # 将各类输出放入同一个列表中，category_outputs存储各类输出的列表
    category_outputs = []
    category_num = len(rank_outputs[0])
    for category_id in range(category_num):
        specific_category_output = [rank_output[category_id] for rank_output in rank_outputs]
        category_outputs.append(specific_category_output)

    return category_outputs

# endregion

# region 输入构造函数

def gen_local_moe_expert_num():
    min_local_moe_expert_num = 1 if is_variable_local_moe_expert_num else local_moe_expert_num
    return [
        torch.randint(min_local_moe_expert_num, local_moe_expert_num + 1, size=[moe_rank_num])
        for _ in range(tp_world_size)
    ]

def gen_x():
    x_list = []
    for _ in range(tp_world_size):
        if is_x_random:
            cur_x = torch.empty(size=[global_bs, h], dtype=input_dtype).uniform_(-5, 5)
        else:
            cur_x = torch.arange(global_bs, dtype=input_dtype).view(-1, 1).repeat(1, h)
        
        x_list.append(cur_x)

    return x_list

def gen_expert_ids():
    def gen_random_token_expert_ids(length, from_, to_):
        return (torch.randperm(to_ - from_, dtype=torch.int32) + from_)[:length]
    
    def gen_balanced_expert_ids_element(round_index):
        # 例如k = 8, moe = 15 [0, 1, .... 14], [1, 2 ... 14, 0]...
        return torch.arange(round_index, round_index + moe_expert_num, dtype=torch.int32) % moe_expert_num
    
    ep_expert_ids_list = []
    for _ in range(tp_world_size):
        if is_topk_balance:
            ep_expert_ids = [gen_balanced_expert_ids_element(round_index) for round_index in range((global_bs * k) // moe_expert_num + 1)]
        else:
            ep_expert_ids = [gen_random_token_expert_ids(k, 0, moe_expert_num) for _ in range(global_bs)]
        
        ep_expert_ids_list.append(torch.cat(ep_expert_ids)[:global_bs * k].view([global_bs, k]))

    return ep_expert_ids_list

def gen_x_active_mask():
    def gen_card_active_mask():
        min_real_token_num = 0 if is_mask else 1
        real_token_num = torch.randint(min_real_token_num, bs + 1, (1,)).item() if is_unfixed_real_token else bs
        return torch.cat(
            (torch.ones(real_token_num), torch.zeros(bs - real_token_num))
        ).to(torch.bool)
    
    x_active_mask_list = []
    for _ in range(tp_world_size):
        x_active_mask = [gen_card_active_mask() for _ in range(ep_world_size)]
        if is_variable_bs:
            upper_bs_card_ep_id = torch.randint(0, ep_world_size, (1,)).item()
            x_active_mask[upper_bs_card_ep_id] = torch.full((bs,), 1).to(torch.bool)

        if is_2d_mask:
            for j in range(ep_world_size):
                out_mask = torch.zeros(bs, k, dtype=torch.bool)
                true_indices = x_active_mask[j].nonzero().squeeze(-1)
                if true_indices.numel() > 0:
                    random_vals = torch.rand(len(true_indices), k) > 0.5
                    out_mask[true_indices] = random_vals
                x_active_mask[j] = out_mask
        x_active_mask_list.append(torch.cat(x_active_mask))

    return x_active_mask_list

def gen_scales(moe_expert_num_list):
    scales_list = []
    padded_scales_list = []
    for tp_id in range(tp_world_size):
        
        if is_quant and is_quant_scales:
            if quant_mode == 9:
                scales = torch.empty(size=[1, 1], dtype=torch.float32).uniform_(-1, 1)
                padded_scales = scales.repeat(shared_expert_rank_num + moe_expert_num_list[tp_id], 1)
            else:
                if quant_mode == 1 and is_static_shared_scales == 1 and expandx_dtype == "int8":
                    scales = torch.empty(size=[h], dtype=torch.float32).uniform_(-1, 1)
                    padded_scales = scales.repeat(shared_expert_rank_num + moe_expert_num_list[tp_id], 1)
                elif quant_mode == 1 and is_static_shared_scales == 2 and expandx_dtype == "int8":
                    scales = torch.empty(size=[1], dtype=torch.float32).uniform_(-1, 1)
                    padded_scales = scales.repeat(shared_expert_rank_num + moe_expert_num_list[tp_id], 1)
                elif quant_mode == 1 and is_static_shared_scales == 2 and expandx_dtype == "hif8":
                    scales = torch.empty(size=[1], dtype=torch.float32).uniform_(-1, 1)
                    padded_scales = scales.repeat(shared_expert_rank_num + moe_expert_num_list[tp_id], 1)
                else:
                    # 暂未支持 scales type 为 fp8_e8m0
                    actual_shared_expert_num = 0 \
                                                if shared_expert_num == 0 or shared_expert_rank_num == 0 \
                                                else shared_expert_num
                    scales = torch.empty(size=[actual_shared_expert_num + moe_expert_num_list[tp_id], h], dtype=torch.float32).uniform_(-1, 1)
                    padded_scales = torch.cat((
                        scales[:actual_shared_expert_num].repeat_interleave(repeats=rank_num_per_shared_expert, dim=0), scales[actual_shared_expert_num:]
                    ))
        else:
            # 使用空张量torch.empty(0)会被dispatch算子的tiling检查，只能用None，无法保证数据类型一致
            scales = None
            padded_scales = None
        scales_list.append(scales)
        padded_scales_list.append(padded_scales)

    return scales_list, padded_scales_list

def gen_expert_scales():
    return [torch.empty(size=[global_bs, k], dtype=torch.float32).uniform_(-1, 1) for _ in range(tp_world_size)]

def gen_shared_expert_x(x_active_mask_list: List[List[torch.Tensor]], from_=0, to=1):
    if not is_shared_expert_x:
        return [[torch.empty(0) for _ in range(ep_world_size)] for _ in range(tp_world_size)]
    
    x_active_mask_per_card = [chunk_tensor(mask_ep, ep_world_size) for mask_ep in x_active_mask_list]

    shared_expert_x_list = []
    for tp_id in range(tp_world_size):
        shared_expert_x_list.append([
            torch.empty(size=[bs, h] if is_2d_mask else[x_active_mask_per_card[tp_id][ep_id].sum().item(), h], dtype=input_dtype).uniform_(from_, to)
            for ep_id in range(ep_world_size)
        ])
    
    return shared_expert_x_list

# endregion

# region golden值生成函数

def gen_dispatch_golden(
    local_moe_expert_num_list,
    x_list, expert_ids_list, x_active_mask_list, padded_scales_list
):
    def moe_dispatch(
        ep_local_moe_expert_num: torch.Tensor, 
        x: torch.Tensor, expert_ids: torch.Tensor, x_active_mask: torch.Tensor
    ):
        '''将每张卡上的原始数据分发给moe专家。

        根据分发信息expert_ids，分发输入的张量x得到输出expanded_x，
        并生成expanded_x每行对应的（x, k）中元素的索引sorted_col_idx，
        以及分到各moe专家的数据数量golden_moe_token
        '''

        # 全局的x_mask掩掉x和expert_ids对应false的行，得到实际要发送的token的集合x及相应的expert_ids
        if not is_2d_mask:
            x = x[x_active_mask]
            expert_ids = expert_ids[x_active_mask]

        # expanded_x从前往后，放入x相应的行
        sorted_col_idx = torch.argsort(expert_ids.flatten(), stable=True)
        dst_row_to_src_row = sorted_col_idx // expert_ids.size(1)
        expanded_x = torch.index_select(input=x, dim=0, index=dst_row_to_src_row)
        if is_2d_mask:
            expanded_mask = torch.index_select(input=x_active_mask.flatten(), dim=0, index=sorted_col_idx)

         # 统计每个专家出现次数，即为接收到的token数
        expert_ids_sorted, _ = torch.sort(expert_ids.flatten())
        golden_moe_token = torch.bincount(expert_ids_sorted[expanded_mask] if is_2d_mask else expert_ids.flatten(),
                                             minlength=ep_local_moe_expert_num.sum().item())
        return expanded_x[expanded_mask] if is_2d_mask else expanded_x, sorted_col_idx[
            expanded_mask] if is_2d_mask else sorted_col_idx, golden_moe_token
    
    def shared_dispatch(x: torch.Tensor, x_active_mask: torch.Tensor):
        '''将每张卡上的原始数据分发给共享专家。

        将 ep 域内所有卡分组 (ep group)，每组 rank_num_per_shared_expert 张卡，
        将共享专家卡按卡上专家分组 (shared group)，每组 rank_num_per_shared_expert 张卡，
        ep group 各组组内编号为 i 的卡将数据分发给 shared group 各组组内第 i 张共享专家卡
        '''

        shared_list = [torch.empty(0) for _ in range(shared_expert_rank_num)]
        shared_tokens = torch.empty((shared_expert_rank_num,), dtype=torch.int64)


        if is_2d_mask:
            real_x_per_card = [
                local_x[local_mask.any(dim=1)]  # 只保留 mask 至少有一个 True 的行
                for local_x, local_mask in zip(
                    chunk_tensor(x, ep_world_size),
                    chunk_tensor(x_active_mask, ep_world_size)
                )
            ]
        else:
            real_x_per_card = [
                local_x[local_mask]
                for local_x, local_mask in
                zip(chunk_tensor(x, ep_world_size), chunk_tensor(x_active_mask, ep_world_size))
            ]

        for shared_card_id in range(shared_expert_rank_num):
            card_id_in_shared_group = shared_card_id % rank_num_per_shared_expert
            send_card_ids = slice(card_id_in_shared_group, ep_world_size, rank_num_per_shared_expert)
            local_expand_x = torch.cat(real_x_per_card[send_card_ids])
            shared_list[shared_card_id] = local_expand_x
            shared_tokens[shared_card_id] = local_expand_x.size(0)

        return shared_list, shared_tokens
    
    def quant_process(
        ep_local_moe_expert_num: torch.Tensor, 
        shared_list: list, moe_x_segments: list, scales
    ):
        '''对各专家接收到的数据进行量化，并生成量化场景输出的scales'''
        
        QUANT_MODE_DTYPE = {
            "fp16": "float16",
            "bf16": "bfloat16",
            "int8": "int8",
            "fp8_e5m2": "float8_e5m2",
            "fp8_e4m3fn": "float8_e4m3fn",
            "hif8": "hifloat8"
        }
        def cal_quant(y: torch.Tensor, scales: torch.Tensor):
            from quantize import static_quantize, dynamic_quantize, mx_quantize, QUANT_PER_TOKEN
            # target_type = QUANT_MODE_DTYPE[quant_mode]
            target_type = QUANT_MODE_DTYPE[expandx_dtype]
            y = torch_to_numpy_tensor(y)
            scales = torch_to_numpy_tensor(scales)
            # if quant_mode in [1, 9]:
            if quant_mode == 1:
                y = static_quantize(y, scales, dst_type_str=target_type)
                dynamic_scales = np.repeat(scales, (y.shape[0],), axis=0)
                dynamic_scales = dynamic_scales.reshape(y.shape[0], -1)
            # if quant_mode in [2, 5, 6]:
            if quant_mode == 2:
                y, dynamic_scales = dynamic_quantize(y, dst_type_str=target_type, smooth_scales=scales, quant_mode=QUANT_PER_TOKEN)
            # elif quant_mode in [3, 4]:
            elif quant_mode == 4:
                tile_size = 32
                dynamic_scales, y = mx_quantize(y, target_type, axis=-1, block_size=tile_size, round_mode="rint")
            # elif quant_mode in [7, 8]:
            elif quant_mode == 3:
                tile_size = 128
                ori_h = y.shape[-1]
                ori_bs = y.shape[0]
                y = reshape_with_padding(y, tile_size)
                if is_quant_scales:
                    scales = scales.reshape(1, h).repeat((ori_bs,), axis=0)
                    scales = reshape_with_padding(scales, tile_size)
                y, dynamic_scales = dynamic_quantize(y, dst_type_str=target_type, smooth_scales=scales, quant_mode=QUANT_PER_TOKEN)
                y = y.reshape((ori_bs, -1))[:, :ori_h]
                dynamic_scales = dynamic_scales.reshape(-1, ceil(ori_h / tile_size))
            return numpy_to_torch_tensor(y), numpy_to_torch_tensor(dynamic_scales)

        #每个专家的dynamic_scales，一个token一个scales
        shared_dynamic_scales = [torch.empty(0) for _ in range(len(shared_list))]
        moe_dynamic_scales = [torch.empty(0) for _ in range(len(moe_x_segments))]

        for i in range(shared_expert_rank_num):
            # shared_dynamic_scales[i] 为长度为 shared_list[i] 行数的一维张量，长度即每张共享专家卡上的token数
            shared_list[i], shared_dynamic_scales[i] = cal_quant(shared_list[i], scales[i,:] if is_quant_scales else torch.ones((1,)))
        
        ### 对每个专家的moe_x进行量化（列表moe_x_segments中元素为各moe专家的expand_x）
        for moe_expert_id in range(len(moe_x_segments)):
            # moe_dynamic_scales[i] 长度为 golden_moe_token[i] 大小，即每个moe专家接收到的token数
            if moe_x_segments[moe_expert_id].size(0) > 0:
                moe_x_segments[moe_expert_id], moe_dynamic_scales[moe_expert_id] = cal_quant(
                    moe_x_segments[moe_expert_id],
                    scales[shared_expert_rank_num+moe_expert_id,:] if is_quant_scales else torch.ones((1,))
                )
        
        # 得到各moe卡的dynamic scales
        rank_expert_id_bounds = torch.cat(
            [torch.zeros(1, dtype=torch.int32), ep_local_moe_expert_num.cumsum(dim=0)]     # 每个数字, 既是前一张卡的尾，又是后一张卡的头 
        ).tolist()
        moe_dynamic_scales_per_card = [
            torch.cat(moe_dynamic_scales[rank_expert_id_bounds[moe_rank_id] : rank_expert_id_bounds[moe_rank_id+1]])
            for moe_rank_id in range(moe_rank_num)
        ]

        golden_dynamic_scales = shared_dynamic_scales + moe_dynamic_scales_per_card
        
        return shared_list, moe_x_segments, golden_dynamic_scales

    def gen_triplets_expand_idx(
        ep_local_moe_expert_num: torch.Tensor,
        expert_ids: torch.Tensor, x_active_mask: torch.Tensor
    ):
        """ 输出expand_x中各token对应的(rank_id, token_id, k_id)"""

        def gen_shared_expand_idx(x_active_mask: torch.Tensor):
            shared_expand_idx = [[] for _ in range(shared_expert_rank_num)]       #存储每个共享专家收到的(r,t,k)
            if is_2d_mask:
                real_token_num_list = [bs for i in range(ep_world_size)]
                x_active_mask_per_card = x_active_mask.chunk(ep_world_size)

            else:
                real_token_num_list = x_active_mask.view(-1, bs).sum(dim=1).tolist()

            for shared_card_id in range(shared_expert_rank_num):
                shared_expert_index = shared_card_id // rank_num_per_shared_expert
                card_id_in_shared_group = shared_card_id % rank_num_per_shared_expert
                for send_card_id in range(card_id_in_shared_group, ep_world_size, rank_num_per_shared_expert):
                    if not is_2d_mask:
                        rtk_list = [
                            ele
                            for token_id in range(real_token_num_list[send_card_id])
                            for ele in (send_card_id, token_id, k + shared_expert_index)
                        ]
                        shared_expand_idx[shared_card_id].extend(rtk_list)

                    else:

                        for token_id in range(bs):
                            if x_active_mask_per_card[send_card_id][token_id].any():
                                rtk_list = [send_card_id, token_id,k + shared_expert_index]

                                shared_expand_idx[shared_card_id].extend(rtk_list)
            
            return shared_expand_idx

        def gen_moe_expand_idx(
            ep_local_moe_expert_num: torch.Tensor,
            expert_ids: torch.Tensor, x_active_mask: torch.Tensor
        ):
            moe_expand_idx = [[] for _ in range(moe_expert_num)]       #存储每个moe专家收到的(r,t,k)

            x_active_mask_per_card = x_active_mask.chunk(ep_world_size)
            expert_ids_segments = expert_ids.chunk(ep_world_size)
            # 生成moe卡的expand_idx
            for send_rank_id in range(ep_world_size):
                segment = expert_ids_segments[send_rank_id]
                for token_id in range(bs):
                    if is_2d_mask:
                        for k_id in range(k):
                            if x_active_mask_per_card[send_rank_id][token_id][k_id]:
                                received_moe_expert = segment[token_id, k_id].item()
                                moe_expand_idx[received_moe_expert].extend([send_rank_id, token_id, k_id])
                    else:
                        if x_active_mask_per_card[send_rank_id][token_id]:
                            for k_id in range(k):
                                received_moe_expert = segment[token_id, k_id].item()
                                moe_expand_idx[received_moe_expert].extend([send_rank_id, token_id, k_id])
                   
            # 转换，让moe_expand_idx存储每张moe卡收到的(r,t,k)
            rank_expert_id_bounds = torch.cat(
                [torch.zeros(1, dtype=torch.int32), ep_local_moe_expert_num.cumsum(dim=0)]     # 每个数字, 既是前一张卡的尾，又是后一张卡的头 
            ).tolist()  
            moe_expand_idx = [
                sum(moe_expand_idx[rank_expert_id_bounds[moe_rank_id] : rank_expert_id_bounds[moe_rank_id+1]], []) 
                for moe_rank_id in range(moe_rank_num)
            ]
            
            return moe_expand_idx
        
        expand_idx = (
            gen_shared_expand_idx(x_active_mask)
            + gen_moe_expand_idx(ep_local_moe_expert_num, expert_ids, x_active_mask)
        )
        expand_idx = [torch.tensor(local_expand_idx, dtype=torch.int32) for local_expand_idx in expand_idx] # 格式转换，以便后续处理
        return expand_idx

    def gen_expand_idx(expert_ids: torch.Tensor, x_active_mask: torch.Tensor):
        moe_expand_idx_list = []
        moe_expand_idx_masked_list = []
        expert_ids_segments = expert_ids.chunk(ep_world_size)
        x_active_mask_segments = x_active_mask.chunk(ep_world_size)
        for send_rank_id in range(ep_world_size):
            # 获取本卡发出的 expert_ids
            if is_ccu:
                segment = expert_ids_segments[send_rank_id][x_active_mask_segments[send_rank_id]]
            else:
                segment = expert_ids_segments[send_rank_id]
            # 获取各个专家接收到 token 数量的前缀和
            cumsum_pad = torch.cat([torch.zeros(1, dtype=segment.dtype),
                                    torch.cumsum(torch.bincount(segment.flatten(), minlength=moe_expert_num), dim=0)])
            moe_expand_idx = torch.zeros((bs, k), dtype=torch.int32)
            indices = torch.arange(cumsum_pad.size(0) - 1)
            rank_based_indices = (indices // local_moe_expert_num) * local_moe_expert_num
            # expert_cnts 内的初始值表示本专家在 rank 内的基础偏移
            expert_cnts = cumsum_pad[indices] - cumsum_pad[rank_based_indices]
            for token_id in range(len(segment)):
                for k_id in range(k):
                    expert = segment[token_id, k_id].item()
                    moe_expand_idx[token_id, k_id] = expert_cnts[expert]
                    expert_cnts[expert] += 1
            moe_expand_idx_list.append(moe_expand_idx.flatten())
            moe_expand_idx_masked_list.append(moe_expand_idx[:len(segment)].flatten())
        return moe_expand_idx_list, moe_expand_idx_masked_list

    def gen_ep_recv_count(
        ep_local_moe_expert_num: torch.Tensor, 
        expert_ids: torch.Tensor, x_active_mask: torch.Tensor
    ):
        """生成各专家的recv_count"""

        def gen_shared_expert_recv_count(x_active_mask: torch.Tensor):
            shared_expert_recv_count = torch.zeros(shared_expert_rank_num, ep_world_size, dtype=torch.int32)
            if is_2d_mask:
                # real_token_num_tensor = torch.tensor([bs] * ep_world_size)
                real_token_num_tensor = x_active_mask.any(dim=1).view(ep_world_size, bs).sum(dim=1)
            else:
                real_token_num_tensor = x_active_mask.view(-1, bs).sum(dim=1)

            for shared_card_id in range(shared_expert_rank_num):
                card_id_in_shared_group = shared_card_id % rank_num_per_shared_expert
                send_card_ids = slice(card_id_in_shared_group, ep_world_size, rank_num_per_shared_expert)
                shared_expert_recv_count[shared_card_id][send_card_ids] = real_token_num_tensor[send_card_ids]

            # 对每张共享专家卡上的recv_count求累加和，确定发回给每张卡的数据偏移和数据量
            shared_expert_recv_count = shared_expert_recv_count.cumsum(dim=1, dtype=torch.int32)

            shared_expert_recv_count_per_card = [local_ep_recv_count for local_ep_recv_count in shared_expert_recv_count]

            return shared_expert_recv_count_per_card

        def gen_moe_expert_recv_count(
            ep_local_moe_expert_num: torch.Tensor, 
            expert_ids: torch.Tensor, x_active_mask: torch.Tensor
        ):
            x_active_mask_per_card = x_active_mask.chunk(ep_world_size)

            # 将全局的expert_ids切分为ep_world_size份，放入元组expert_ids_segments中。每份为一张卡上的expert_ids
            original_expert_ids_segments = expert_ids.chunk(ep_world_size)
            expert_ids_segments =  [
                original_expert_ids_segments[rank_id][x_active_mask_per_card[rank_id]] 
                for rank_id in range(ep_world_size)
            ]
            # 初始化一个二维张量，每行存储一个moe专家从ep域各卡收到的token数，大小为(moe_expert_num, ep_world_size)
            moe_expert_recv_count = torch.zeros(moe_expert_num, ep_world_size, dtype=torch.int32)
            # 遍历各卡
            for i in range(ep_world_size):
                # 获取当前卡数据
                segment = expert_ids_segments[i].flatten()
                # 对当前卡发送的每个专家进行计数，并更新到result_tensor中
                counts = torch.bincount(segment, minlength=moe_expert_num)
                moe_expert_recv_count[:, i] = counts
            
            # 得到各张卡的moe_expert_recv_count
            moe_expert_recv_count_per_card = split_tensor(moe_expert_recv_count, ep_local_moe_expert_num.tolist())
            # 对每张moe卡上的recv_count求累加和，确定发回给每张卡的数据偏移和数据量
            moe_expert_recv_count_per_card = [
                local_moe_expert_recv_count.flatten().cumsum(dim=0, dtype=torch.int32)
                for local_moe_expert_recv_count in moe_expert_recv_count_per_card
            ]

            return moe_expert_recv_count_per_card
        
        # 拼接共享专家和moe专家收到的token数
        shared_expert_recv_count_per_card = gen_shared_expert_recv_count(x_active_mask)
        moe_expert_recv_count_per_card = gen_moe_expert_recv_count(ep_local_moe_expert_num, expert_ids, x_active_mask)
        ep_recv_count = shared_expert_recv_count_per_card + moe_expert_recv_count_per_card

        return ep_recv_count

    golden_expand_x_list, golden_expand_idx_list, golden_expand_idx_masked_list, golden_tokens_list, \
    golden_ep_recv_count_list, golden_dynamic_scales_list, middle_idx_list, golden_triplets_expand_idx_list = \
        [], [], [], [], [], [], [], []
    for tp_id in range(tp_world_size):
        ep_local_moe_expert_num, x, expert_ids, scales, x_active_mask,  = (
            local_moe_expert_num_list[tp_id],
            x_list[tp_id], expert_ids_list[tp_id], 
            padded_scales_list[tp_id], x_active_mask_list[tp_id]
        )

        # region ep域dispatch

        expanded_x, sorted_col_idx, golden_moe_token = moe_dispatch(ep_local_moe_expert_num, x, expert_ids, x_active_mask)
    
        shared_list, shared_tokens = shared_dispatch(x, x_active_mask)
        
        golden_tokens = torch.cat((shared_tokens, golden_moe_token)) # 分到所有卡上的golden值

        moe_x_segments = split_tensor(expanded_x, golden_moe_token.tolist())
        golden_dynamic_scales = []
        if is_quant:
            shared_list, moe_x_segments, golden_dynamic_scales = quant_process(
                ep_local_moe_expert_num, 
                shared_list, moe_x_segments, scales
            )
        rank_expert_id_bounds = torch.cat(
            [torch.zeros(1, dtype=torch.int32), ep_local_moe_expert_num.cumsum(dim=0)]     # 每个数字, 既是前一张卡的尾，又是后一张卡的头 
        ).tolist()
        moe_list = [
            concat_tensors(moe_x_segments[rank_expert_id_bounds[moe_rank_id] : rank_expert_id_bounds[moe_rank_id+1]])
            for moe_rank_id in range(moe_rank_num)
        ]
        golden_expand_x = shared_list + moe_list

        golden_triplets_expand_idx = gen_triplets_expand_idx(ep_local_moe_expert_num, expert_ids, x_active_mask)
        golden_expand_idx, golden_expand_idx_masked = gen_expand_idx(expert_ids, x_active_mask)
        golden_ep_recv_count = gen_ep_recv_count(ep_local_moe_expert_num, expert_ids, x_active_mask)

        # endregion

        golden_expand_x_list.append(golden_expand_x)
        golden_expand_idx_list.append(golden_expand_idx)
        golden_expand_idx_masked_list.append(golden_expand_idx_masked)
        golden_tokens_list.append(golden_tokens)
        golden_ep_recv_count_list.append(golden_ep_recv_count)
        golden_dynamic_scales_list.append(golden_dynamic_scales)
        golden_triplets_expand_idx_list.append(golden_triplets_expand_idx)
        middle_idx_list.append(sorted_col_idx)
    
    return golden_expand_x_list, golden_expand_idx_list, golden_expand_idx_masked_list, \
        golden_tokens_list, golden_ep_recv_count_list, \
        golden_dynamic_scales_list, middle_idx_list, golden_triplets_expand_idx_list

def gen_all_gather_golden(
    golden_expand_x_list, golden_tokens_list,
    golden_dynamic_scales_list, golden_ep_recv_count_list,
    tp_world_size
):
    tp_recv_counts = torch.stack([golden_tokens_list[0], golden_tokens_list[1]] * 2, dim = 1).view(-1).to(torch.int32)

    golden_tokens = torch.repeat_interleave(golden_tokens_list[0] + golden_tokens_list[1], repeats = 2, dim = 0)

    gathered_golden_expand_x_list = [[] for _ in range(tp_world_size)]
    expand_x_list = []
    ep_recv_count_list = []
    dynamic_scales_list = []
    for ep_id in range(ep_world_size):
        for tp_id in range(tp_world_size):
            peer_tp_id = (tp_id + 1) % tp_world_size
            gathered_local_expand_x = torch.cat((golden_expand_x_list[tp_id][ep_id], golden_expand_x_list[peer_tp_id][ep_id]))

            gathered_golden_expand_x_list[tp_id].append(gathered_local_expand_x)
            expand_x_list.append(gathered_local_expand_x)
            ep_recv_count_list.append(
                torch.cat((golden_ep_recv_count_list[tp_id][ep_id], golden_ep_recv_count_list[peer_tp_id][ep_id]))
            )
            if is_quant:
                dynamic_scales_list.append(
                    torch.cat((golden_dynamic_scales_list[tp_id][ep_id], golden_dynamic_scales_list[peer_tp_id][ep_id]))
                )

    return gathered_golden_expand_x_list, expand_x_list, ep_recv_count_list, tp_recv_counts, golden_tokens, dynamic_scales_list

def gen_reduce_scatter_golden(golden_expand_x_list, golden_tp_recv_count):
    shared_x_list = [[] for _ in range(tp_world_size)]
    moe_x_list = [[] for _ in range(tp_world_size)]

    for tp_id in range(tp_world_size):
        peer_tp_id = (tp_id + 1) % tp_world_size

        local_golden_expand_x = golden_expand_x_list[tp_id]
        peer_golden_expand_x = golden_expand_x_list[peer_tp_id]

        for ep_id in range(ep_world_size):
            local_real_token_num = golden_tp_recv_count[2*ep_id][tp_id]
            peer_real_token_num = golden_tp_recv_count[2*ep_id][peer_tp_id]
            tp_real_token_num = local_real_token_num + peer_real_token_num
            local_real_expand_x = (
                local_golden_expand_x[ep_id][:local_real_token_num, :]
                + peer_golden_expand_x[ep_id][peer_real_token_num:tp_real_token_num, :]
            )

            if ep_id < shared_expert_rank_num:
                shared_x_list[tp_id].append(local_real_expand_x)
            else:
                moe_x_list[tp_id].append(local_real_expand_x)

    shared_x_list = [
        torch.cat(shared_x_list[tp_id]) if is_shared else torch.empty(0) 
        for tp_id in range(tp_world_size)
    ]
    moe_x_list = [torch.cat(moe_x_list[tp_id]) for tp_id in range(tp_world_size)]

    return shared_x_list, moe_x_list

def gen_combine_golden(
    middle_shared_x_list, middle_moe_x_list, middle_idx_list, expert_scales_list,
    golden_tokens_list, x_active_mask_list, shared_expert_x_list
):
    def comm_quant_round_trip(tensor: torch.Tensor):
        """ 将张量x做comm_quant量化，并对量化的结果进行逆运算

            comm_quant_mode=2时，对输入x进行comm_quant操作并返回结果
            comm_quant_mode=0时，直接返回输入x
        """

        def comm_quant(tensor: torch.Tensor):
            quant_dtype = torch.int8 if comm_quant_mode == 2 else torch.int16
            quant_dtype_max = 127.0 if comm_quant_mode == 2 else 2047.0

            blocks = tensor.view(-1, element_per_block).to(torch.float32)
            blocks_max, _ = blocks.abs().max(dim=-1, keepdim=True)
            blocks_scales = blocks_max / quant_dtype_max
            blocks_scales[blocks_scales==0.0] = 1.0 # block_scales = 0 的时候改成 1，避免除 0
            blocks_quant = (blocks / blocks_scales).to(torch.float16).round().to(quant_dtype)

            return blocks_quant, blocks_scales.to(input_dtype)

        def comm_dequant(blocks_quant: torch.Tensor, blocks_scales: torch.Tensor):
            blocks_scales = blocks_scales.to(torch.float32)
            blocks_quant = blocks_quant.to(torch.float16).to(torch.float32)
            blocks_dequant = (blocks_quant * blocks_scales).to(input_dtype)
            tensor_dequant = blocks_dequant.view(-1, h + token_pad_length)

            return tensor_dequant
        
        if comm_quant_mode == 0:
            return tensor

        block_size = 32  # 核函数侧，一个 block 有 32 字节
        computation_dtype = torch.float32

        computation_dtype_size = torch.finfo(computation_dtype).bits // 8   # 用于量化计算的数据类型的字节数
        element_per_block =  block_size // computation_dtype_size # 8

        token_pad_length = (element_per_block - (tensor.size(1) % element_per_block)) % element_per_block
        padded_tensor = torch.cat([tensor, tensor.new_zeros(tensor.size(0), token_pad_length)], dim=1)
        padded_tensor_after_round_trip = comm_dequant(*comm_quant(padded_tensor))

        return padded_tensor_after_round_trip[:, :tensor.size(1)]

    def combine_x_for_shared(
        middle_shared_x: List[torch.Tensor], golden_tokens: torch.Tensor, 
        x_active_mask: torch.Tensor, shared_scales_list=[1 for _ in range(shared_expert_num)]
    ):
        returned_real_x = [[torch.empty(0) for _ in range(ep_world_size)] for _ in range(shared_expert_num)]
        if is_2d_mask:
            real_token_num_list = x_active_mask.view(ep_world_size, bs, -1).any(dim=-1).sum(dim=1).tolist()
        else:
            real_token_num_list = x_active_mask.view(-1, bs).sum(dim=1).tolist()
        middle_shared_x_per_card = middle_shared_x.split(golden_tokens[:shared_expert_rank_num].tolist())

        for shared_card_id in range(shared_expert_rank_num):
            shared_expert_index = shared_card_id // rank_num_per_shared_expert

            card_id_in_shared_group = shared_card_id % rank_num_per_shared_expert
            send_card_ids = slice(card_id_in_shared_group, ep_world_size, rank_num_per_shared_expert)

            local_middle_shared_x = middle_shared_x_per_card[shared_card_id]
            token_num_from_origin_card = real_token_num_list[send_card_ids]

            returned_real_x[shared_expert_index][send_card_ids] = split_tensor(local_middle_shared_x, token_num_from_origin_card)

        returned_tensor_after_round_trip_list = [
            comm_quant_round_trip(torch.cat(returned_real_x_from_shared_expert))
            for returned_real_x_from_shared_expert in returned_real_x
        ]
        if is_2d_mask:
            j = 0
            scaled_real_x_from_shared_expert = [torch.zeros(bs * ep_world_size, h) for _ in range(shared_expert_num)]
            # scaled_real_x_from_shared_expert
            x_active_counts = x_active_mask.sum(dim=1)
            for i, k_num in enumerate(x_active_counts):
                if k_num.item() != 0:  # 如果一行全为0，或者没有共享专家
                    for share_num in range(shared_expert_num):
                        scaled_real_x_from_shared_expert[share_num][i] = \
                        returned_tensor_after_round_trip_list[share_num][j].to(torch.float32)
                    j += 1
            # scaled_x = torch.stack(scaled_x, dim=0)
        else:
            scaled_real_x_from_shared_expert = [
                returned_tensor.to(torch.float32) for returned_tensor in returned_tensor_after_round_trip_list
            ]

        return scaled_real_x_from_shared_expert

    def combine_x_for_moe(
        middle_moe_x: torch.Tensor, 
        middle_idx: torch.Tensor, 
        expert_scales: torch.Tensor,
        x_active_mask: torch.Tensor
    ):
        '''将真实的expand_x还回去，得到真实的x'''

        middle_moe_x = middle_moe_x.view(len(middle_idx), h)

        # 因为middle_idx的值为0到len(middle_idx) - 1，排第i的值（即为值i）对应的索引即为值i对应的索引
        expandx_row_to_x_row = torch.argsort(middle_idx)
        # x_k即为x每一行变为k行的结果，上文expandx_row_to_x_row中的x即为x_k，为书写简便上文记为x
        x_k = middle_moe_x.index_select(dim=0, index=expandx_row_to_x_row)
        x_k_after_round_trip = comm_quant_round_trip(x_k)

        # 得到有效的token的expert_scales
        real_expert_scales = expert_scales[x_active_mask].view(-1, 1)
        # 每个还回去的token乘上对应的scale，并将同token求和
        scaled_x_k = x_k_after_round_trip.to(torch.float32) * real_expert_scales
        no_token = []
        if is_2d_mask:
            j = 0
            scaled_x = [torch.zeros(h) for _ in range(bs * ep_world_size)]

            x_active_counts = x_active_mask.sum(dim=1)
            for i, k_num in enumerate(x_active_counts):
                if k_num.item() == 0:  # 如果一行全为0，或者没有共享专家
                    no_token.append(i)
                for k_id in range(k_num.item()):
                    scaled_x[i] += scaled_x_k[j]  # 三元组的顺序
                    j += 1
            scaled_x = torch.stack(scaled_x, dim=0)
        else:
            # 每个原始token对应k个处理后的token，将这些处理后的token按照k_id从前往后相加，确保顺序和核函数求和顺序一致
            scaled_x_k_by_token = scaled_x_k.view(-1, k, scaled_x_k.size(1))
            scaled_x = sum([scaled_x_k_by_token[:, k_id, :] for k_id in range(k)])

        return scaled_x, no_token
    
    golden_x_list = []
    for tp_id in range(tp_world_size):
        middle_shared_x, middle_moe_x, middle_idx, expert_scales, golden_tokens, x_active_mask, shared_expert_x_per_card = (
            middle_shared_x_list[tp_id],
            middle_moe_x_list[tp_id],
            middle_idx_list[tp_id],
            expert_scales_list[tp_id],
            golden_tokens_list[tp_id],
            x_active_mask_list[tp_id],
            shared_expert_x_list[tp_id]
        )
        
        golden_x_from_shared = (
            combine_x_for_shared(middle_shared_x, golden_tokens, x_active_mask)
            if is_shared else torch.empty(0)
        )
        golden_x_from_moe, no_token= combine_x_for_moe(
            middle_moe_x, 
            middle_idx,
            expert_scales,
            x_active_mask
        )
        golden_x_from_moe = golden_x_from_moe.to(torch.float32)

        golden_x_ep = golden_x_from_moe
        if is_shared:
            golden_x_ep = sum(golden_x_from_shared, start=golden_x_ep)
        if is_shared_expert_x:
            golden_x_ep += torch.cat(shared_expert_x_per_card).to(torch.float32)
        golden_x_ep = golden_x_ep.to(input_dtype)
        if is_2d_mask:
            real_token_num_list = [bs] * ep_world_size
        else:
            real_token_num_list = x_active_mask.view(-1, bs).sum(dim=1).tolist()
        golden_x_list.append(split_tensor(golden_x_ep, real_token_num_list))

    # golden_x为列表，存储每张卡上真实的x值
    golden_x = []
    for ep_id in range(ep_world_size):
        for tp_id in range(tp_world_size):
            golden_x.append(golden_x_list[tp_id][ep_id])
    
    return golden_x

# endregion

# region 精度对比

def compare_output(compared_output: ComparedOutput, golden_per_card, npu_per_card, compare_dtype, compare_per_rank):
    def convert_to_numpy(tensor, dtype):
        tensor = tensor.cpu().contiguous()
        if dtype == "bfloat16":
            from ml_dtypes import bfloat16
            return tensor.view(torch.int16).numpy().view(bfloat16)
        elif dtype == "float8_e4m3fn":
            from ml_dtypes import float8_e4m3fn
            return tensor.view(torch.uint8).numpy().view(float8_e4m3fn)
        elif dtype == "float8_e5m2":
            from ml_dtypes import float8_e5m2
            return tensor.view(torch.uint8).numpy().view(float8_e5m2)
        elif dtype == "float8_e8m0":
            from en_dtypes import float8_e8m0
            return tensor.view(torch.uint8).numpy().view(float8_e8m0)
        elif dtype == "hifloat8":
            from en_dtypes import hifloat8
            return tensor.view(torch.uint8).numpy().view(hifloat8)
        else:
            try:
                torch_dtype = getattr(torch, dtype)
                return tensor.to(torch_dtype).numpy()
            except Exception:
                raise ValueError(f"Data type '{dtype}' is currently not supported.")

    def compare(golden, npu, dtype, tensor_name, diffThd=0.001, pctThd=0.001):
        from quantize import dtype_map

        if dtype not in dtype_map:
            raise ValueError(f"Unknown dtype: {dtype}.")

        dtype = dtype_map[dtype]
        print("dtype: ", dtype)
        np_output_npu = convert_to_numpy(npu, dtype)
        np_output_golden = convert_to_numpy(golden, dtype)

        max_diff_hd = 0.0
        if dtype in ("int8", "float32", "float8_e5m2", "float8_e4m3fn", "float8_e8m0", "hifloat8"):
            max_diff_hd = 1.0
        res_output = toolss.data_compare_np(np_output_npu, np_output_golden, diff_thd=diffThd, pct_thd=pctThd, max_diff_hd=max_diff_hd)
        print(f'[INFO] tensor {tensor_name} output测试的精度结果为：{res_output}')
        write_row_to_csv('moe_distribute.csv', [current_case_name, tensor_name, res_output])
        return res_output[0] == 'PASS'
    
    mark_str = compared_output.value.upper().replace('_', ' ')
    result = {}

    if not compare_per_rank:
        print(f"##################Compare {mark_str}##################")
        res = compare(
            torch.cat(golden_per_card),
            torch.cat(npu_per_card),
            compare_dtype,
            compared_output.value
        )
        result[compared_output.value] = res
    else:
        for i in range(server_index*rank_per_dev, (server_index+1)*rank_per_dev):
            print(f"##################Compare {mark_str} in rank",i,"epId",i//tp_world_size,"tpId",i%tp_world_size,"##################")
            local_rank_id = i - server_index*rank_per_dev

            res = compare(
                golden_per_card[local_rank_id],
                npu_per_card[local_rank_id],
                compare_dtype, 
                f"{compared_output.value}_{i}"
            )
            result[f"{compared_output.value}_{i}"] = res

    return result

# endregion

if __name__ == "__main__":
    # region 测试前准备工作

    clean_previous_profiling_result()

    print_setting()

    if is_check_setting:
        check_setting()
    
    # endregion

    # 设置随机种子
    torch.manual_seed(random_seed)

    # 生成各ep域的moe专家分布
    local_moe_expert_num_list = gen_local_moe_expert_num()
    # 生成各ep域的x
    x_list = gen_x()
    # 生成各ep域的expert_ids
    expert_ids_list = gen_expert_ids()
    # 生成各ep域的x_active_mask及各ep域有效token的个数
    x_active_mask_list = gen_x_active_mask()

    x_active_mask_list_ela_ori = deepcopy(x_active_mask_list)
    for i in x_active_mask_list_ela_ori:
        if not torch.any(i):
            print("当前不支持x_active_mask全为false!")
            exit(1)

    if is_2d_mask:
        mask_tp = [[local_mask.any(dim=1) for local_mask in chunk_tensor(x_active_mask, ep_world_size)] for
                   x_active_mask in (x_active_mask_list_ela_ori if len(x_active_mask_list_ela_ori[0].shape) != 1 else [
                original_tensor.unsqueeze(1).repeat(1, k) for original_tensor in x_active_mask_list_ela_ori])]
        mask_per_card = list(chain.from_iterable(zip(*mask_tp)))
    
    # 生成各ep域的scales
    scales_list, padded_scales_list = gen_scales(
        [
            ep_local_moe_expert_num.sum().item()
            for ep_local_moe_expert_num in local_moe_expert_num_list
        ]
    )

    # region 生成dispatch算子的golden值

    (
        golden_expand_x_list, golden_expand_idx_list, golden_expand_idx_masked_list, golden_tokens_list,
        golden_ep_recv_count_list, golden_dynamic_scales_list, middle_idx_list, golden_triplets_expand_idx_list
    ) = gen_dispatch_golden(
        local_moe_expert_num_list, 
        x_list, expert_ids_list, x_active_mask_list, padded_scales_list
    )

    if tp_world_size != 1:
        golden_expand_x_list, golden_expand_x, golden_ep_recv_count, golden_tp_recv_count, golden_tokens, golden_dynamic_scales = gen_all_gather_golden(
            golden_expand_x_list, golden_tokens_list,
            golden_dynamic_scales_list, golden_ep_recv_count_list,
            tp_world_size,
        )

        # 将golden_expand_idx的元素变为按卡号顺序排列
        golden_expand_idx = [
            golden_expand_idx_list[tp_id][ep_id]
            for ep_id in range(ep_world_size)
            for tp_id in range(tp_world_size)
        ]
        golden_triplets_expand_idx = [
            golden_triplets_expand_idx_list[tp_id][ep_id]
            for ep_id in range(ep_world_size)
            for tp_id in range(tp_world_size)
        ]
        golden_expand_idx_masked = [
            golden_expand_idx_masked_list[tp_id][ep_id]
            for ep_id in range(ep_world_size)
            for tp_id in range(tp_world_size)
        ]
        # 使golden_tokens和golden_tp_recv_count以每张卡为单位
        golden_tokens = chunk_tensor(golden_tokens, world_size)
        golden_tp_recv_count = chunk_tensor(golden_tp_recv_count, world_size)
    else:
        # 生成tp=1时的同语义变量
        golden_expand_x, golden_ep_recv_count, golden_tokens, golden_dynamic_scales, golden_expand_idx, golden_expand_idx_masked, golden_triplets_expand_idx = (
            golden_expand_x_list[0],
            golden_ep_recv_count_list[0],
            golden_tokens_list[0],
            golden_dynamic_scales_list[0],
            golden_expand_idx_list[0],
            golden_expand_idx_masked_list[0],
            golden_triplets_expand_idx_list[0]
        )
        expert_num_list = [1] * shared_expert_rank_num + local_moe_expert_num_list[0].tolist()
        golden_tokens = split_tensor(golden_tokens, expert_num_list)
        golden_tp_recv_count = [torch.empty(size=(tp_world_size,), dtype=torch.int32) for _ in range(world_size)]
    
    if expert_token_nums_type == 0:
        for i in range(len(golden_tokens)):
            golden_tokens[i] = torch.cumsum(golden_tokens[i], dim=0)
    # endregion

    # 生成各ep域的expert_scales
    expert_scales_list = gen_expert_scales()
    # 生成各ep域的shared_expert_x
    shared_expert_x_list = gen_shared_expert_x(x_active_mask_list)

    # region 生成combine算子golden值

    if is_quant:
        for tp_id in range(tp_world_size):
            golden_expand_x_list[tp_id] = [convert_expand_x_dtype(tensor) for tensor in golden_expand_x_list[tp_id]]

    if tp_world_size == 2:
        middle_shared_x_list, middle_moe_x_list = gen_reduce_scatter_golden(golden_expand_x_list, golden_tp_recv_count)
    else:
        expand_x = golden_expand_x_list[0]
        middle_shared_x_list, middle_moe_x_list = (
            [torch.cat(expand_x[:shared_expert_rank_num]) if is_shared else torch.empty(0)], 
            [torch.cat(expand_x[shared_expert_rank_num:])],
        )

    golden_x = gen_combine_golden(
        middle_shared_x_list, middle_moe_x_list, middle_idx_list, expert_scales_list,
        golden_tokens_list, x_active_mask_list, shared_expert_x_list
    )

    # endregion
    
    print("Generate golden success.")

    if is_dispatch:
        [npu_expand_x, npu_dynamic_scales, npu_expand_idx, npu_tokens, npu_eprecv_count, npu_tprecv_count] = gen_npu(
            run_dispatch_npu,
            x_list=[chunk_tensor(x, ep_world_size) for x in x_list],
            expert_ids_list=[chunk_tensor(expert_ids, ep_world_size) for expert_ids in expert_ids_list],
            x_active_mask_list=[chunk_tensor(x_active_mask, ep_world_size) for x_active_mask in x_active_mask_list],
            scales_list=scales_list,
        )
        print("Generate dispatch npu success.")

        if is_performance and graph_type == 0:
            dispatch_performance = get_dispatch_performance()

        # 精度对比
        print("##################Start Dispatch Compare##################")

        result_dispatch = {}
        
        # 比较 ep recv count
        res = compare_output(
            ComparedOutput.EP_RECV_COUNTS,
            golden_ep_recv_count[server_ranks],
            npu_eprecv_count,
            "int32",
            False,
        )
        result_dispatch.update(res)

        # 比较 tp recv count
        if tp_world_size != 1:
            res = compare_output(
                ComparedOutput.TP_RECV_COUNTS,
                golden_tp_recv_count[server_ranks],
                npu_tprecv_count,
                "int32",
                False,
            )
            result_dispatch.update(res)
        
        # 比较 expand x
        res = compare_output(
            ComparedOutput.EXPAND_X,
            golden_expand_x[server_ranks],
            npu_expand_x,
            expandx_dtype,
            True,
        )
        result_dispatch.update(res)
        
        # 比较 dynamic scales
        if is_quant and quant_mode != 1:
            if quant_mode == 4 and npu_dynamic_scales[0].dtype != torch.uint8:
                npu_dynamic_scales = [npu_dynamic_scales[i].view(torch.uint8) for i in range(len(npu_dynamic_scales))]
            res = compare_output(
                ComparedOutput.DYNAMIC_SCALES,
                golden_dynamic_scales[server_ranks],
                npu_dynamic_scales,
                "float32",
                False
            )
            result_dispatch.update(res)
        
        # 比较 expand idx
        res = compare_output(
            ComparedOutput.EXPAND_IDX,
            golden_expand_idx_masked[server_ranks] if is_ccu else golden_triplets_expand_idx[server_ranks],
            npu_expand_idx,
            "int32",
            True
        )
        result_dispatch.update(res)

        # 比较 expert token nums
        res = compare_output(
            ComparedOutput.EXPERT_TOKEN_NUMS,
            golden_tokens[server_ranks],
            npu_tokens,
            "int64",
            False
        )
        result_dispatch.update(res)

    if is_2d_mask:
        mask_per_card_server = deepcopy(mask_per_card)

    if is_combine:
        print(f"golden_triplets_expand_idx.shape = {len(golden_triplets_expand_idx)} {len(golden_triplets_expand_idx[0])}")
        [npu_x] = gen_npu(
            run_combine_npu,
            expand_x_list=golden_expand_x_list,
            expert_ids_list=[chunk_tensor(expert_ids, ep_world_size) for expert_ids in expert_ids_list],
            expand_idx_list=golden_expand_idx if is_ccu else golden_triplets_expand_idx,
            ep_send_count_list=golden_ep_recv_count,
            tp_send_count_list=golden_tp_recv_count,
            x_active_mask_list=[chunk_tensor(x_active_mask, ep_world_size) for x_active_mask in x_active_mask_list],
            expert_scales_list=[chunk_tensor(expert_scales, ep_world_size) for expert_scales in expert_scales_list],
            shared_expert_x_list=shared_expert_x_list,
        )

        print("Generate combine npu success.")

        if is_performance and graph_type == 0:
            combine_performance = get_combine_performance()

        print("##################Start Combine Compare##################")

        result_combine = {}

        if is_2d_mask:
                npu_x = [npu_x_per_card[mask_all_per_card] for npu_x_per_card, mask_all_per_card in zip(npu_x,
                                                                                                        mask_per_card if server_num == 1 else
                                                                                                        mask_per_card_server[work_server_ranks(server_index)])]
                golden_x_rank = [golden_x_per_card[mask_all_per_card] for golden_x_per_card, mask_all_per_card in
                                 zip(golden_x, mask_per_card)]
        # 比较x
        res = compare_output(
            ComparedOutput.X,
            golden_x_rank[server_ranks] if is_2d_mask else golden_x[server_ranks],
            npu_x,
            "float32",
            True
        )
        result_combine.update(res)

    if is_continue:
        [npu_x] = gen_npu(
            run_continue_npu,
            x_list=[chunk_tensor(x, ep_world_size) for x in x_list],
            expert_ids_list=[chunk_tensor(expert_ids, ep_world_size) for expert_ids in expert_ids_list],
            x_active_mask_list=[chunk_tensor(x_active_mask, ep_world_size) for x_active_mask in x_active_mask_list],
            scales_list=scales_list,
            expert_scales_list=[chunk_tensor(expert_scales, ep_world_size) for expert_scales in expert_scales_list], 
            golden_expand_x_list=golden_expand_x_list, 
            shared_expert_x_list=shared_expert_x_list,
        )

        if is_performance:
            continue_performance = get_continue_performance()

        print("Generate continue npu success.")

        print("##################Start Continue Compare##################")

        result_continue = {}
        if is_2d_mask:
                npu_x = [npu_x_per_card[mask_all_per_card] for npu_x_per_card, mask_all_per_card in zip(npu_x,
                                                                                                        mask_per_card if server_num == 1 else
                                                                                                        mask_per_card_server[
                                                                                                            work_server_ranks(
                                                                                                                server_index)])]
                golden_x_rank = [golden_x_per_card[mask_all_per_card] for golden_x_per_card, mask_all_per_card in
                                 zip(golden_x, mask_per_card)]
        # 比较 x
        res = compare_output(
            ComparedOutput.X,
            golden_x_rank[server_ranks] if is_2d_mask else golden_x[server_ranks],
            npu_x,
            "float32",
            True
        )
        result_continue.update(res)
    
    if is_dispatch:
        print("\n##################dispatch result##################")
        print("result_dispatch: ", result_dispatch)
        all_result = all(result_dispatch.values())
        if all_result:
            print("dispatch test Success.")
        else:
            print("dispatch test Failed.")
        
        if is_performance and graph_type == 0:
            print_dispatch_performance(dispatch_performance)
    
    if is_combine:
        print("\n##################combine result##################")
        print("result: ", result_combine)
        all_result = all(result_combine.values())
        if all_result:
            print("combine test Success.")
        else:
            print("combine test Failed.")
        
        if is_performance and graph_type == 0:
            print_combine_performance(combine_performance)
    
    if is_continue:
        print("\n##################continue result##################")
        print("result: ", result_continue)
        all_result = all(result_continue.values())
        if all_result:
            print("continue test Success.")
        else:
            print("continue test Failed.")
        
        if is_performance:
            print_continue_performance(continue_performance)
