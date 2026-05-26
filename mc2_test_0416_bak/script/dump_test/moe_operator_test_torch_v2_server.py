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
from itertools import chain
import random
from copy import deepcopy
import bisect
import torch.nn.functional as F

# region 控制参数配置

# region 执行哪些测试

x_active_mask_dim = 1   # mask的维度，当前支持0,1,2

is_dispatch = 1        # 是否单跑dispatch
is_combine = 0         # 是否单跑combine
is_cascade = 0         # 是否跑dispatch和combine的级联

is_elastic = 1          # 是否动态缩容      ############ 多机环境仅支持缩容单机，不支持调节下列选项
elastic_shared_ranks = 0    # 共享专家缩容卡数
elastic_moe_ranks = 1   # moe专家卡缩容卡数
elastic_random_ranks = 0    # 随机缩容卡数
elastic_shared = (elastic_shared_ranks > 0)      # 是否缩共享专家卡
elastic_moe = (elastic_moe_ranks > 0)            # 是否缩moe专家
elastic_random = (elastic_random_ranks > 0)      # 是否随机缩容

zero_expert_num = 0
copy_expert_num = 0
const_expert_num = 0
is_zero_compute_expert = zero_expert_num + copy_expert_num + const_expert_num != 0

graph_type = 0              # 0为单算子，1为静态图模式，2为动态图模式
quant_mode = 2              # 0为dispatch算子非量化场景，1为静态量化（当前不支持），2为动态量化
comm_quant_mode = 2         # 0无量化，1则combine通信采用int12动态量化（不支持），2为int8动态量化

# endregion

# region 性能检测相关

is_performance = 0      # 是否进行性能检测
loop = 105              # 性能检测时的性能采集轮数
repeat = 1              # 性能采集分为几个批次
warmup = 0              # 每批次的热身轮数

# endregion

# region 特性使能
is_variable_bs = 0                                             # 是否使能可变bs
is_variable_local_moe_expert_num = 0                           # 是否使能可变moe卡上专家数
is_shared_expert_x = 0      # 是否传入并处理shared_expert_x
is_dispatch_scales = 0      # 动态量化场景下，是否给dispatch算子传入scales
is_full_core = True                                             # True时为全核模式，使用全部的核，False时使用部分核
expert_shard_type = 0                                           # 0 共享专家部署在最前面的卡上，仅支持 0
core_num = 12 if not is_full_core else 24                       # 非全核模式使用的核数（单个核上有一个aic核，一个aiv核）

is_quant = (quant_mode > 0)

aic_core_num = core_num                                         # 非全核模式使用的aic核数
aiv_core_num = core_num * 2                                     # 非全核模式使用的aiv核数

# endregion

# region 通信域配置

server_num = 1                  # 单机或者多机模式
tp_world_size = 1               # 1或2, 2时dispatch算子在tp域执行allgather, combine算子执行reducescatter, 1时均不执行
shared_expert_num = 1
rank_num_per_shared_expert = 4            # 每个共享专家占据的卡数
local_moe_expert_num = 2              # moe卡上的专家数
is_cloud = False                      # 是否在云上环境进行测试，使用云上环境时设置为True，使用物理机时设为False
if not is_cloud:
    server_index = 0 if server_num > 1 else 0                        # 多机每个脚本保持一致并递增index值
    master_ip = '71.20.45.123' if server_num > 1 else '127.0.0.1'    # 如果通信不通，查看一下master_ip是否设对
else:
    # 云上环境用下面方式获取server_index和master_ip
    server_index = int(os.environ.get("VC_TASK_INDEX"))
    host_url_list = os.environ.get("VC_WORKER_HOSTS").split(",")
    master_ip = socket.gethostbyname(host_url_list[0])

rank_per_dev = 16                                                       # 每个host有几个die
is_2d_mask = 1 if x_active_mask_dim == 2 else 0
is_mask = 1 if x_active_mask_dim == 1 else 0
is_unfixed_real_token = is_mask or is_variable_bs               # 每张卡上真实token数是否固定
world_size = server_num * rank_per_dev                                  # 通信域总die数
ep_world_size = world_size // tp_world_size
shared_expert_rank_num = shared_expert_num * rank_num_per_shared_expert      # 共享专家卡数
moe_rank_num = ep_world_size - shared_expert_rank_num                   # moe专家卡数
moe_expert_num = moe_rank_num * local_moe_expert_num                    # moe专家数

is_shared = (shared_expert_rank_num > 0)



# endregion

# region 输入输出维度相关

bs = 376                          # token数量
h = 7168                        # 每个token的长度
# k = moe_expert_num if moe_expert_num < 8 else 8    # 每个token发给多少个moe专家
k = 15
if bs > 16:
    os.environ['HCCL_BUFFSIZE'] = '6000'

global_bs = bs * ep_world_size
max_shared_group_num = ceil(ep_world_size / rank_num_per_shared_expert) if (rank_num_per_shared_expert > 0) else 0
A_shared = bs * max_shared_group_num
A_moe = global_bs * min(local_moe_expert_num, k)
if is_elastic:
    A_shared = A_moe = max(A_shared, A_moe)

# endregion

# region 产生什么样的输入

out_dtype = 0               # 控制dispatch输入及combine输出的数据类型， 0为bfloat16，1为float16
is_topk_balance = False     # 是否生成均衡的topK
is_x_random = False          # 是否让x值随机
random_seed = 0             # 随机种子，用于固定各随机输入变量的产生，如x,expert_ids,expert_scales等

input_dtype = torch.bfloat16 if out_dtype == 0 else torch.float16 # dispatch输入及combine输出的数据类型

# endregion

# region 测试脚本执行相关

is_cascade_golden_expand_x = True                              # 在级联模式下，是否将dispatch的golden_expand_x直接传给combine算子
is_save_tensor_to_bin = False
is_check_setting = True
current_case_name = 'MOE_Dis_Com_16p_bs12_k6_xdtype0_quant2_graph0_scale1_cascade0_tp1_localmoe3_share1_moe21_ID034'

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
cascade_profiling_path = './Profiling_Result_mc2_cascade_16p'
dispatch_op_type = 'MoeDistributeDispatchV2'
combine_op_type = 'MoeDistributeCombineV2'
path_to_op_types = {
    dispatch_profiling_path: [dispatch_op_type],
    combine_profiling_path: [combine_op_type],
    cascade_profiling_path: [dispatch_op_type, combine_op_type]
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
def work_server_ranks(server_index_elastic):
    return slice(server_index_elastic * rank_per_dev, server_index_elastic * rank_per_dev + rank_per_dev)

def chunk_tensor(tensor: torch.Tensor, chunks: int) -> List[torch.Tensor]:
    return list(tensor.chunk(chunks))

def split_tensor(tensor: torch.Tensor, split_size_or_sections: int or List[int]) -> List[torch.Tensor]:
    return list(tensor.split(split_size_or_sections))

def pad_tensor(tensor_list):
    padded_list = []
    for tensor in tensor_list:
        n, h = tensor.size()
        padded = torch.zeros(bs, h, dtype=tensor.dtype)
        padded[:n, :] = tensor
        padded_list.append(padded)
    return padded_list


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

    if shared_expert_num > 0 and rank_num_per_shared_expert == 0:
        error_log_and_exit("shared_expert_num > 0 时, rank_num_per_shared_expert 必须大于 0")

    if shared_expert_rank_num != shared_expert_num * rank_num_per_shared_expert:
        error_log_and_exit("shared_expert_rank_num 必须等于 shared_expert_num * rank_num_per_shared_expert")

    if shared_expert_rank_num >= ep_world_size:
        error_log_and_exit("shared_expert_rank_num 不能大于 ep_world_size")
    
    if moe_expert_num % moe_rank_num != 0:
        error_log_and_exit("moe_expert_num 必须是 moe_rank_num 的整数倍")
    
    if quant_mode == 1:
        error_log_and_exit("当前不支持静态量化, quant_mode 合法值仅有 0 和 2")

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

    if tp_world_size == 2 and is_elastic:
        error_log_and_exit("tp = 2 时不支持动态缩容")

    if elastic_shared_ranks > shared_expert_rank_num:
        error_log_and_exit("elastic_shared_ranks不得超过共享专家卡数")

    if elastic_moe_ranks >= moe_rank_num:
        error_log_and_exit("moe卡不支持全部缩容")

    if ep_world_size - elastic_random_ranks < shared_expert_rank_num + k//local_moe_expert_num:
        error_log_and_exit("elastic_random_ranks不得超过ep_world_size-shared_expert_rank_num-k//local_moe_expert_num")

    if elastic_random and (elastic_moe or elastic_shared):
        error_log_and_exit("随机缩容不和moe缩容、shared缩容同时开")

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
        combine_expected_memory_usage = (combine_output_memory_usage + workspace * MB_to_bytes) / MB_to_bytes + aic_core_num * 10 / 1024

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

    if test_mode == 'cascade':
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

def print_cascade_performance(cascade_performance):
    (
        dispatch_share_performance, dispatch_moe_performance, combine_share_performance, combine_moe_performance, 
        min_dispatch_share_performance, min_dispatch_moe_performance, min_combine_share_performance, min_combine_moe_performance
    ), memory_usage = cascade_performance

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

def get_cascade_performance():
    cascade_performance, memory_usage = get_performance(cascade_profiling_path)

    print_cascade_performance([cascade_performance, memory_usage])

    dispatch_performance = cascade_performance[:2] + cascade_performance[4:6]
    combine_performance = cascade_performance[2:4] + cascade_performance[6:]
    op_types = path_to_op_types[cascade_profiling_path]
    save_performance_result('cascade', current_case_name, op_types, [dispatch_performance, combine_performance], memory_usage)

    return [cascade_performance, memory_usage]

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
        self, x, expert_ids, x_active_mask, scales, group_ep, group_tp, ep_rank_id, tp_rank_id, ep_world_size, tp_world_size,
        expert_shard_type, shared_expert_num, shared_expert_rank_num, moe_expert_num, quant_mode, global_bs, elastic_info,
        zero_expert_num, copy_expert_num, const_expert_num,
    ):
        output = torch_npu.npu_moe_distribute_dispatch_v2(
            x=x,
            expert_ids=expert_ids,
            x_active_mask=x_active_mask,
            scales=scales,
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
            quant_mode=quant_mode,
            global_bs=global_bs,
            elastic_info=elastic_info,
            zero_expert_num=zero_expert_num,
            copy_expert_num=copy_expert_num,
            const_expert_num=const_expert_num,
        )
        return output

class MOE_DISTRIBUTE_COMBINE_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(
        self, expand_x, expert_ids, assist_info_for_combine, ep_send_counts, tp_send_counts, expert_scales,
        x_active_mask, shared_expert_x, ori_x, const_expert_alpha_1, const_expert_alpha_2, const_expert_v,
        group_ep, group_tp, ep_rank_id, tp_rank_id, ep_world_size, tp_world_size, expert_shard_type, shared_expert_num,
        shared_expert_rank_num, moe_expert_num, comm_quant_mode, global_bs, elastic_info, zero_expert_num, copy_expert_num, const_expert_num,
    ):
        output = torch_npu.npu_moe_distribute_combine_v2(
            expand_x=expand_x,
            expert_ids=expert_ids,
            assist_info_for_combine=assist_info_for_combine,
            ep_send_counts=ep_send_counts,
            tp_send_counts=tp_send_counts,
            expert_scales=expert_scales,
            x_active_mask=x_active_mask,
            shared_expert_x=shared_expert_x,
            ori_x=ori_x,
            const_expert_alpha_1=const_expert_alpha_1,
            const_expert_alpha_2=const_expert_alpha_2,
            const_expert_v=const_expert_v,
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
            elastic_info=elastic_info,
            zero_expert_num=zero_expert_num,
            copy_expert_num=copy_expert_num,
            const_expert_num=const_expert_num,
        )
        return output

class MOE_DISTRIBUTE_CASCADE_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, x, expert_ids, expert_scales, x_active_mask, scales, shared_expert_x,
        ori_x, const_expert_alpha_1, const_expert_alpha_2, const_expert_v,
        group_ep, group_tp, ep_rank_id, tp_rank_id, ep_world_size, tp_world_size, 
        expert_shard_type, shared_expert_num, shared_expert_rank_num, moe_expert_num,
        quant_mode, comm_quant_mode, global_bs, golden_expand_x, elastic_info, zero_expert_num, copy_expert_num, const_expert_num,
    ):
        x_combine_res = None
        local_loop = loop if is_performance else 1
        for _ in range(local_loop):
            output_dispatch_npu = torch_npu.npu_moe_distribute_dispatch_v2(
                x=x,
                expert_ids=expert_ids,
                x_active_mask=x_active_mask,
                scales=scales,
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
                quant_mode=quant_mode,
                global_bs=global_bs,
                elastic_info=elastic_info,
                zero_expert_num=zero_expert_num,
                copy_expert_num=copy_expert_num,
                const_expert_num=const_expert_num,
            )
            
            expand_x_npu, _, expand_idx_npu, _, ep_recv_counts_npu, tp_recv_counts_npu, _ = output_dispatch_npu  
            if expand_x_npu.dtype == torch.int8:
                expand_x_npu = expand_x_npu.to(input_dtype)
            if is_cascade_golden_expand_x:
                expand_x_npu = golden_expand_x
            
            output_combine_npu = torch_npu.npu_moe_distribute_combine_v2(
                expand_x=expand_x_npu,
                expert_ids=expert_ids,
                assist_info_for_combine=expand_idx_npu,
                ep_send_counts=ep_recv_counts_npu,
                tp_send_counts=tp_recv_counts_npu,
                expert_scales=expert_scales,
                x_active_mask=x_active_mask,
                shared_expert_x=shared_expert_x,
                ori_x=ori_x,
                const_expert_alpha_1=const_expert_alpha_1,
                const_expert_alpha_2=const_expert_alpha_2,
                const_expert_v=const_expert_v,
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
                elastic_info=elastic_info,
                zero_expert_num=zero_expert_num,
                copy_expert_num=copy_expert_num,
                const_expert_num=const_expert_num,
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

def warm_up_dispatch(rank, group_ep, group_tp):
    x_warm_up = torch.empty(size=[1, h], dtype=input_dtype).uniform_(-1024, 1024).to(input_dtype).npu()
    expert_ids_warm_up = torch.arange(0, k, dtype=torch.int32).unsqueeze(0).npu()

    dispatch_kwargs_before = get_dispatch_kwargs_warmup(
        x=x_warm_up,
        expert_ids=expert_ids_warm_up,
        group_ep=group_ep,
        group_tp=group_tp,
        ep_rank_id=rank//tp_world_size,
        tp_rank_id=rank%tp_world_size,
    )

    (
        expand_x, dynamic_scales, expand_idx, 
        expert_token_nums, ep_recv_counts, tp_recv_counts, _
    ) = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_kwargs_before)
    

    return expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts

def get_dispatch_kwargs(
    x, expert_ids, x_active_mask, scales,
    group_ep, group_tp, ep_rank_id, tp_rank_id, elastic_info
):
    x = x.to(input_dtype).npu()
    expert_ids = expert_ids.to(torch.int32).npu()
    x_active_mask = x_active_mask.to(torch.bool).npu() if is_mask or is_2d_mask else None
    if elastic_info != None:
        elastic_info = elastic_info.to(torch.int32).npu() if is_elastic else None
    if is_quant and is_dispatch_scales:
        scales = scales.to(torch.float32).npu()
    
    return {
        'x': x,
        'expert_ids': expert_ids,
        'x_active_mask': x_active_mask,
        'group_ep': group_ep,
        'group_tp': group_tp,
        'ep_rank_id': 15 if ep_rank_id != 15 else 14,
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
        'elastic_info': elastic_info, 
        'zero_expert_num': zero_expert_num,
        'copy_expert_num': copy_expert_num,
        'const_expert_num': const_expert_num,
    }

def get_dispatch_kwargs_warmup(
    x, expert_ids, group_ep, group_tp, ep_rank_id, tp_rank_id, 
):
    x = x.to(input_dtype).npu()
    expert_ids = expert_ids.to(torch.int32).npu()
    
    return {
        'x': x,
        'expert_ids': expert_ids,
        'x_active_mask': None,
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
        'scales': None,
        'quant_mode': 2,
        'global_bs': ep_world_size,
        # 'zero_expert_num': zero_expert_num,
        # 'copy_expert_num': copy_expert_num,
        # 'const_expert_num': const_expert_num,
    }
    

def get_combine_kwargs(
    expand_x, expert_ids, expand_idx, ep_send_counts, tp_send_counts, expert_scales,
    x_active_mask, shared_expert_x, group_ep, group_tp, ep_rank_id, tp_rank_id, elastic_info,
    ori_x, const_expert_alpha_1, const_expert_alpha_2, const_expert_v,
):
    expand_x = expand_x.to(input_dtype).npu()
    expert_ids = expert_ids.to(torch.int32).npu()
    expand_idx = expand_idx.to(torch.int32).npu()
    ep_send_counts = ep_send_counts.to(torch.int32).npu()
    tp_send_counts = tp_send_counts.to(torch.int32).npu()
    expert_scales = expert_scales.to(torch.float32).npu()
    x_active_mask = x_active_mask.to(torch.bool).npu() if is_mask and is_2d_mask else None
    shared_expert_x = shared_expert_x.to(input_dtype).npu() if is_shared_expert_x else None
    if elastic_info != None:
        elastic_info = elastic_info.to(torch.int32).npu() if is_elastic else None
    ori_x = ori_x.to(input_dtype).npu() if copy_expert_num or const_expert_num else None
    const_expert_alpha_1 = const_expert_alpha_1.to(input_dtype).npu() if const_expert_num else None
    const_expert_alpha_2 = const_expert_alpha_2.to(input_dtype).npu() if const_expert_num else None
    const_expert_v = const_expert_v.to(input_dtype).npu() if const_expert_num else None

    return {
        'expand_x': expand_x,
        'expert_ids': expert_ids,
        'assist_info_for_combine': expand_idx,
        'ep_send_counts': ep_send_counts,
        'tp_send_counts': tp_send_counts,
        'expert_scales': expert_scales,
        'x_active_mask': x_active_mask,
        'shared_expert_x': shared_expert_x,
        'ori_x': ori_x,
        'const_expert_alpha_1': const_expert_alpha_1,
        'const_expert_alpha_2': const_expert_alpha_2,
        'const_expert_v': const_expert_v,
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
        'elastic_info': elastic_info,
        'zero_expert_num': zero_expert_num,
        'copy_expert_num': copy_expert_num,
        'const_expert_num': const_expert_num,
    }

def run_dispatch_npu(queue, rank, x, expert_ids, x_active_mask, scales, available_ranks, elastic_info):
    set_device(rank)
    ep_hcomm_info, tp_hcomm_info, _, _ = init_hccl_comm(rank)
    if is_save_tensor_to_bin:
        save_tensor_to_bin(x.to(torch.float32), "x_in_" + str(rank))
        save_tensor_to_bin(expert_ids.to(torch.int32), "expertIds_" + str(rank))
        if is_quant and is_dispatch_scales:
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
        elastic_info=elastic_info,
    )
    
    if graph_type == 0:
        if not is_performance:
            if is_elastic:
                expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts = warm_up_dispatch(rank, ep_hcomm_info, tp_hcomm_info)
                torch.npu.synchronize()
                print("====================warm_up_over=====================")
                #if rank in available_ranks:
                (
                    expand_x, dynamic_scales, expand_idx, 
                    expert_token_nums, ep_recv_counts, tp_recv_counts, _
                ) = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_kwargs)
                time.sleep(10)
            else:
                (
                    expand_x, dynamic_scales, expand_idx, 
                    expert_token_nums, ep_recv_counts, tp_recv_counts, _
                ) = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_kwargs)


        else:
            if is_elastic:
                expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts = warm_up_dispatch(rank, ep_hcomm_info, tp_hcomm_info)
                torch.npu.synchronize()
                print("====================warm_up_over=====================")
                if rank in available_ranks:
                    tensor = torch.ones(10000, 10000).to(torch.float16).npu()
                    with get_profiling(loop, repeat, warmup, dispatch_profiling_path) as prof:
                        for i in range(loop):
                            (
                                expand_x, dynamic_scales, expand_idx, 
                                expert_token_nums, ep_recv_counts, tp_recv_counts, _
                            ) = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_kwargs)

                            if i == 1:
                                for _ in range(100):
                                    torch.exp(tensor)

                            prof.step()
                time.sleep(10)

            else:
                tensor = torch.ones(10000, 10000).to(torch.float16).npu()
                with get_profiling(loop, repeat, warmup, dispatch_profiling_path) as prof:
                    for i in range(loop):

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
        if is_elastic:
            expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts = warm_up_dispatch(rank, ep_hcomm_info, tp_hcomm_info)
            torch.npu.synchronize()
            print("====================warm_up_over=====================")
            if rank in available_ranks:
                
                (
                    expand_x, dynamic_scales, expand_idx,
                    expert_token_nums, ep_recv_counts, tp_recv_counts, _
                ) = model(**dispatch_kwargs)
            time.sleep(10)
        else:
            (
                expand_x, dynamic_scales, expand_idx,
                expert_token_nums, ep_recv_counts, tp_recv_counts, _
            ) = model(**dispatch_kwargs)
    torch.npu.synchronize()
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
    if is_elastic and rank < available_ranks[shared_expert_rank_num]:
        real_rank_token_num = expert_token_nums.cpu()[0].item()
    else:
        real_rank_token_num = expert_token_nums.cpu().sum().item()
    real_origin_token_num = (
        tp_recv_counts[rank%tp_world_size].cpu().item()
        if tp_world_size > 1 
        else real_rank_token_num
    )
    expand_x = expand_x[:real_rank_token_num, :]
    expand_idx = expand_idx[:real_origin_token_num * 3]
    dynamic_scales = dynamic_scales[:real_rank_token_num]
    queue.put([
        rank, 
        [
            expand_x.cpu(), dynamic_scales.cpu(), expand_idx.cpu(),
            expert_token_nums.cpu(), ep_recv_counts.cpu(), tp_recv_counts.cpu()
        ]
    ])
    

def run_combine_npu(
    queue, x, scales, rank, expand_x, expert_ids, expand_idx, ep_send_counts, tp_send_counts, x_active_mask, expert_scales, shared_expert_x, available_ranks, elastic_info,
    ori_x, const_expert_alpha_1, const_expert_alpha_2, const_expert_v,
    ):
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
        ori_x = ori_x, 
        const_expert_alpha_1 = const_expert_alpha_1, 
        const_expert_alpha_2 = const_expert_alpha_2, 
        const_expert_v = const_expert_v,
        group_ep=ep_hcomm_info,
        group_tp=tp_hcomm_info,
        ep_rank_id=rank//tp_world_size,
        tp_rank_id=rank%tp_world_size,
        elastic_info=None,
    )
    dispatch_kwargs = get_dispatch_kwargs(
        x=x,
        expert_ids=expert_ids,
        x_active_mask=x_active_mask,
        scales=scales,
        group_ep=ep_hcomm_info,
        group_tp=tp_hcomm_info,
        ep_rank_id=rank//tp_world_size,
        tp_rank_id=rank%tp_world_size,
        elastic_info=elastic_info,
    )

    
    if graph_type == 0:
        if not is_performance:
            if is_elastic:
                x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts = warm_up_dispatch(rank, ep_hcomm_info, tp_hcomm_info)
                torch.npu.synchronize()
                print("====================warm_up_over=====================")
                if rank in available_ranks:
                    x = torch_npu.npu_moe_distribute_combine_v2(**combine_kwargs)
                time.sleep(10)
            else:
                x = torch_npu.npu_moe_distribute_combine_v2(**combine_kwargs)
        else:
            if is_elastic:
                x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts = warm_up_dispatch(rank, ep_hcomm_info, tp_hcomm_info)
                torch.npu.synchronize()
                print("====================warm_up_over=====================")
                if rank in available_ranks:
                    tensor = torch.ones(10000, 10000).to(torch.float16).npu()
                    with get_profiling(loop, repeat, warmup, combine_profiling_path) as prof:
                        for i in range(loop):
                            _ = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_kwargs)
                            x = torch_npu.npu_moe_distribute_combine_v2(**combine_kwargs)

                            if i == 1:
                                for _ in range(100):
                                    torch.exp(tensor)
                    
                            prof.step()
                time.sleep(100)



            else:
                x = torch_npu.npu_moe_distribute_combine_v2(**combine_kwargs)

                tensor = torch.ones(10000, 10000).to(torch.float16).npu()
                with get_profiling(loop, repeat, warmup, combine_profiling_path) as prof:
                    for i in range(loop):

                        # 做全卡同步，因为combine只会同步一部分卡
                        dist.all_reduce(combine_kwargs['expand_x'][0,0], op=ReduceOp.SUM, async_op=False, group=ep_group)
                        _ = torch_npu.npu_moe_distribute_combine_v2(**combine_kwargs)

                        if i == 1:
                            for _ in range(100):
                                torch.exp(tensor)
                
                        prof.step()



    else :
        model = compile_model(MOE_DISTRIBUTE_COMBINE_GRAPH_Model())
        dispatch_model = compile_model(MOE_DISTRIBUTE_DISPATCH_GRAPH_Model())
        if is_elastic:
            x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts = warm_up_dispatch(rank, ep_hcomm_info, tp_hcomm_info)
            torch.npu.synchronize()
            print("====================warm_up_over=====================")
            if rank in available_ranks:
                x = model(**combine_kwargs)
            time.sleep(10)
        else:
            x = model(**combine_kwargs)

    torch.npu.synchronize()
    print(f'rank {rank} epid {rank//tp_world_size} tpid {rank%tp_world_size} npu finished! \n')
    # 获得x中的有效token
    if is_mask and not is_2d_mask and not is_zero_compute_expert:
        real_tokens = x_active_mask.cpu().sum().item()
        x = x[:real_tokens, :]
    queue.put([
        rank, 
        [
            x.cpu()
        ]
    ])

def run_cascade_npu(queue, rank, x, expert_ids, x_active_mask, scales, expert_scales, shared_expert_x, golden_expand_x, available_ranks, elastic_info,
                    ori_x, const_expert_alpha_1, const_expert_alpha_2, const_expert_v,):
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
        elastic_info=elastic_info,
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
        ori_x = ori_x, 
        const_expert_alpha_1 = const_expert_alpha_1, 
        const_expert_alpha_2 = const_expert_alpha_2, 
        const_expert_v = const_expert_v,
        group_ep=ep_hcomm_info,
        group_tp=tp_hcomm_info,
        ep_rank_id=rank//tp_world_size,
        tp_rank_id=rank%tp_world_size,
        elastic_info=elastic_info,
    )

    

    if graph_type == 0:
        if not is_performance:
            if is_elastic:
                x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts = warm_up_dispatch(rank, ep_hcomm_info, tp_hcomm_info)
                torch.npu.synchronize()
                print("====================warm_up_over=====================")
                if rank in available_ranks:
                    expand_x, _, expand_idx, _, ep_recv_counts, tp_recv_counts, _ = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_kwargs)
                    
                    if is_quant:
                        expand_x = expand_x.to(input_dtype)
                    if is_cascade_golden_expand_x:
                        expand_x = golden_expand_x
                    
                    combine_kwargs.update(expand_x=expand_x, assist_info_for_combine=expand_idx, ep_send_counts=ep_recv_counts, tp_send_counts=tp_recv_counts)
                    x = torch_npu.npu_moe_distribute_combine_v2(**combine_kwargs)
                time.sleep(10)
            else:
                expand_x, _, expand_idx, _, ep_recv_counts, tp_recv_counts, _ = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_kwargs)
                    
                if is_quant:
                    expand_x = expand_x.to(input_dtype)
                if is_cascade_golden_expand_x:
                    expand_x = golden_expand_x
                
                combine_kwargs.update(expand_x=expand_x, assist_info_for_combine=expand_idx, ep_send_counts=ep_recv_counts, tp_send_counts=tp_recv_counts)
                x = torch_npu.npu_moe_distribute_combine_v2(**combine_kwargs)

        else:
            if is_elastic:
                x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts = warm_up_dispatch(rank, ep_hcomm_info, tp_hcomm_info)
                torch.npu.synchronize()
                print("====================warm_up_over=====================")
                if rank in available_ranks:
                    tensor = torch.ones(10000, 10000).to(torch.float16).npu()
                    with get_profiling(loop, repeat, warmup, cascade_profiling_path) as prof:
                        for i in range(loop):

                            expand_x, _, expand_idx, _, ep_recv_counts, tp_recv_counts, _ = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_kwargs)
                            
                            if is_quant:
                                expand_x = expand_x.to(input_dtype)
                            if is_cascade_golden_expand_x:
                                expand_x = golden_expand_x
                            combine_kwargs.update(expand_x=expand_x, assist_info_for_combine=expand_idx, ep_send_counts=ep_recv_counts, tp_send_counts=tp_recv_counts)

                            x = torch_npu.npu_moe_distribute_combine_v2(**combine_kwargs)
                            
                            if i == 1:
                                for _ in range(100):
                                    torch.exp(tensor)
                        
                            prof.step()
                time.sleep(20)
            else:
                tensor = torch.ones(10000, 10000).to(torch.float16).npu()
                with get_profiling(loop, repeat, warmup, cascade_profiling_path) as prof:
                    for i in range(loop):

                        expand_x, _, expand_idx, _, ep_recv_counts, tp_recv_counts, _ = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_kwargs)
                        
                        if is_quant:
                            expand_x = expand_x.to(input_dtype)
                        if is_cascade_golden_expand_x:
                            expand_x = golden_expand_x
                        combine_kwargs.update(expand_x=expand_x, assist_info_for_combine=expand_idx, ep_send_counts=ep_recv_counts, tp_send_counts=tp_recv_counts)

                        x = torch_npu.npu_moe_distribute_combine_v2(**combine_kwargs)
                        
                        if i == 1:
                            for _ in range(100):
                                torch.exp(tensor)
                    
                        prof.step()
        
    else :
        cascade_graph_kwargs = {
            **dispatch_kwargs,
            "expert_scales": expert_scales,
            "shared_expert_x": shared_expert_x,
            "golden_expand_x": golden_expand_x,
            'comm_quant_mode': comm_quant_mode,
            'ori_x': combine_kwargs["ori_x"],
            'const_expert_alpha_1': combine_kwargs["const_expert_alpha_1"],
            'const_expert_alpha_2': combine_kwargs["const_expert_alpha_2"],
            'const_expert_v': combine_kwargs["const_expert_v"],
        }

        model = compile_model(MOE_DISTRIBUTE_CASCADE_GRAPH_Model())
        if not is_performance:
            if is_elastic:
                x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts = warm_up_dispatch(rank, ep_hcomm_info, tp_hcomm_info)
                torch.npu.synchronize()
                print("====================warm_up_over=====================")
                if rank in available_ranks:
                    x, _ = model(**cascade_graph_kwargs)
                time.sleep(10)
            else:
                x, _ = model(**cascade_graph_kwargs)
        else:
            loop_time = 2
            repeat_time = 1
            warm_time = 1
            if is_elastic:
                x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts = warm_up_dispatch(rank, ep_hcomm_info, tp_hcomm_info)
                torch.npu.synchronize()
                print("====================warm_up_over=====================")
                if rank in available_ranks:
                    prof = get_profiling(loop_time, repeat_time, warm_time, cascade_profiling_path)
                    prof.start()
                
                    for i in range(loop_time):
                        x, _ = model(**cascade_graph_kwargs)
                        prof.step()
                
                    prof.stop()
                time.sleep(100)
            else:
                prof = get_profiling(loop_time, repeat_time, warm_time, cascade_profiling_path)
                prof.start()
            
                for _ in range(loop_time):
                    x, _ = model(**cascade_graph_kwargs)
                    prof.step()
            
                prof.stop()

    torch.npu.synchronize()
    print(f'rank {rank} epid {rank//tp_world_size} tpid {rank%tp_world_size} npu finished! \n')
    
    # 获得x中的有效token
    if is_mask and not is_2d_mask and not is_zero_compute_expert:
        real_tokens = x_active_mask.cpu().sum().item()
        x = x[:real_tokens, :]
    queue.put([
        rank,
        [
            x.cpu()
        ]
    ])

def gen_npu(target_func, available_ranks, **server_kwargs):
    def parse_rank_input(target_func, result_queue, rank, server_kwargs):
        def pad_expand_x(expand_x: torch.Tensor, ep_id: int):
            A = A_shared if ep_id < shared_expert_rank_num else A_moe
            pad_for_expand_x = torch.zeros(size=[tp_world_size*A - expand_x.size(0), h], dtype=expand_x.dtype)
            return torch.cat([expand_x, pad_for_expand_x])

        def pad_expand_idx(expand_idx: torch.Tensor, ep_id: int):
            A = A_shared if ep_id < shared_expert_rank_num else A_moe
            pad_for_expand_idx = torch.zeros(size=[A*128 - expand_idx.size(0),], dtype=expand_idx.dtype)
            return torch.cat([expand_idx, pad_for_expand_idx])
        
        def pad_shared_expert_x(shared_expert_x: torch.Tensor):
            if (not is_shared_expert_x) or (not is_mask):
                return shared_expert_x
            
            pad_for_shared_expert_x = torch.zeros(size=[bs - shared_expert_x.size(0), h], dtype=shared_expert_x.dtype)
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
                "available_ranks": available_ranks,
                "elastic_info": server_kwargs['elastic_info'],
            }

        if target_func == run_combine_npu:
            return {
                "queue": result_queue,
                "rank": rank,
                "x": filter_tensor(server_kwargs["x_list"][tp_id][ep_id], mask),
                "scales": server_kwargs["scales_list"][tp_id],
                "expand_x": pad_expand_x(server_kwargs["expand_x_list"][tp_id][ep_id], ep_id),
                "expert_ids": filter_tensor(server_kwargs["expert_ids_list"][tp_id][ep_id], mask),
                "expand_idx": pad_expand_idx(server_kwargs["expand_idx_list"][rank], ep_id),
                "ep_send_counts": server_kwargs["ep_send_count_list"][rank],
                "tp_send_counts": server_kwargs["tp_send_count_list"][rank],
                "x_active_mask": server_kwargs["x_active_mask_list"][tp_id][ep_id],
                "expert_scales": filter_tensor(server_kwargs["expert_scales_list"][tp_id][ep_id], mask),
                "shared_expert_x": pad_shared_expert_x(server_kwargs["shared_expert_x_list"][tp_id][ep_id]),
                "available_ranks": available_ranks,
                "elastic_info": server_kwargs['elastic_info'],
                "ori_x": filter_tensor(server_kwargs["ori_x_list"][tp_id][ep_id], mask),
                "const_expert_alpha_1": server_kwargs["const_expert_alpha_1"], 
                "const_expert_alpha_2": server_kwargs["const_expert_alpha_2"], 
                "const_expert_v": server_kwargs["const_expert_v"],
            }
        
        if target_func == run_cascade_npu:
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
                "available_ranks": available_ranks,
                "elastic_info": server_kwargs['elastic_info'],
                "ori_x": filter_tensor(server_kwargs["ori_x_list"][tp_id][ep_id], mask),
                "const_expert_alpha_1": server_kwargs["const_expert_alpha_1"], 
                "const_expert_alpha_2": server_kwargs["const_expert_alpha_2"], 
                "const_expert_v": server_kwargs["const_expert_v"],
            }
    
    if target_func not in [run_dispatch_npu, run_combine_npu, run_cascade_npu]:
        print("target_func not supported!")
        exit(1)
    
    if int(server_num) > 1:
        print("multi_server scene!!!!! only support running case one by one!!!!!!!!!!!!!")
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
        proc.start()
        proc_list.append(proc)
    for proc in proc_list:
        proc.join()
    
    rank_outputs = [None] * rank_per_dev
    for proc in proc_list:
        rank_id, rank_output = result_queue.get()
        local_rank_id = rank_id - server_index * rank_per_dev
        rank_outputs[local_rank_id] = rank_output
    
    # 将各类输出放入同一个列表中，category_outputs存储各类输出的列表
    category_outputs = []
    category_num = len(rank_outputs[0])
    for category_id in range(category_num):
        specific_category_output = [rank_output[category_id] for rank_output in rank_outputs]
        category_outputs.append(specific_category_output)

    return category_outputs

# endregion

# region 输入构造函数

def gen_avalible_list():
    shared_list = list(range(shared_expert_rank_num))
    moe_list = list(range(shared_expert_rank_num, ep_world_size))
    all_list = list(range(ep_world_size))
    server_elastic_index = 1000
    if server_num == 1:
        if elastic_shared:
            random.seed(random_seed)
            elastic_rank_list = random.sample(shared_list, elastic_shared_ranks)
            for rank in elastic_rank_list:
                shared_list.remove(rank)
        if elastic_moe:
            random.seed(random_seed)
            elastic_rank_list = random.sample(moe_list, elastic_moe_ranks)
            for rank in elastic_rank_list:
                moe_list.remove(rank)
        avalible_list = shared_list + moe_list
        if elastic_random:
            random.seed(random_seed)
            avalible_list = random.sample(range(ep_world_size), ep_world_size-elastic_random_ranks)
    else:
        random.seed(random_seed)
        server_elastic_index = random.randint(0, server_num)
        original_list = list(range(server_num * rank_per_dev))
        start = server_elastic_index * rank_per_dev
        end = start + rank_per_dev
        avalible_list = [x for x in original_list if not (start <= x < end)]
    print(f'{avalible_list=}')
    return sorted(avalible_list) if is_elastic else list(range(ep_world_size)), server_elastic_index

def gen_elastic_info(avalible_list):
    elastic_info = torch.zeros(4 + 2 * ep_world_size, dtype=torch.int32)# 初始化张量
    # 设置前四个值
    elastic_info[0] = is_elastic # 是否缩容 (1表示是)
    elastic_info[1] = len(avalible_list)  # 缩容后实际rank数
    elastic_info[2] = shared_expert_rank_num  # 共享专家使用rank数
    elastic_info[3] = (len(avalible_list)-shared_expert_rank_num)*local_moe_expert_num  # moe专家个数
    table1 = [-1] * ep_world_size  # Table1[epRankId] = localEpRankId或-1
    table2 = [-1] * ep_world_size  # Table2[localEpRankId] = epRankId
    for local_rank_id, ep_rank_id in enumerate(avalible_list):
        if ep_rank_id < ep_world_size:  # 确保ep_rank_id在有效范围内
            table1[ep_rank_id] = local_rank_id #local_rank_id表示缩容后的ep_rank_id
            table2[local_rank_id] = ep_rank_id
    for i in range(ep_world_size):
        elastic_info[4 + i] = table1[i]  
    for i in range(ep_world_size):
        elastic_info[4 + ep_world_size + i] = table2[i]
    
    return elastic_info, table1, table2

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
            cur_x = torch.empty(size=[global_bs, h], dtype=input_dtype).uniform_(-1024, 1024)
        else:
            cur_x = torch.arange(1, global_bs+1, dtype=input_dtype).view(-1, 1).repeat(1, h)
        
        x_list.append(cur_x)

    return x_list

def gen_expert_ids(moe_expert_num_re):
    def gen_random_token_expert_ids(length, from_, to_):
        return (torch.randperm(to_ - from_, dtype=torch.int32) + from_)[:length]
    
    def gen_balanced_expert_ids_element(round_index):
        return torch.arange(round_index, round_index + moe_expert_num_re + zero_expert_num + copy_expert_num + const_expert_num, \
                            dtype=torch.int32) % (moe_expert_num_re + zero_expert_num + copy_expert_num + const_expert_num)
    
    ep_expert_ids_list = []
    for _ in range(tp_world_size):
        if is_topk_balance:
            ep_expert_ids = [gen_balanced_expert_ids_element(round_index) for round_index in range((global_bs * k) // \
                                (moe_expert_num_re + zero_expert_num + copy_expert_num + const_expert_num) + 1)]
        else:
            ep_expert_ids = [gen_random_token_expert_ids(k, 0, moe_expert_num_re + zero_expert_num + copy_expert_num + const_expert_num) \
                                for _ in range(global_bs)]
        
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
    for i in range(tp_world_size):
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
        
        if is_quant and is_dispatch_scales:
            scales = torch.empty(size=[shared_expert_num + moe_expert_num_list[tp_id], h], dtype=torch.float32).uniform_(-1, 1)
            padded_scales = torch.cat((
                scales[:shared_expert_num, :].repeat_interleave(repeats=rank_num_per_shared_expert, dim=0),
                scales[shared_expert_num:, :]
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
            torch.empty(size=[bs, h] if is_2d_mask else [x_active_mask_per_card[tp_id][ep_id].sum().item(), h], dtype=input_dtype).uniform_(from_, to)
            for ep_id in range(ep_world_size)
        ])
    
    return shared_expert_x_list

def gen_const_expert_alpha_1():
    const_expert_alpha_1 = torch.empty(size=[const_expert_num], dtype=input_dtype).uniform_(-1, 1)
    return const_expert_alpha_1

def gen_const_expert_alpha_2():
    const_expert_alpha_2 = torch.empty(size=[const_expert_num], dtype=input_dtype).uniform_(-1, 1)
    return const_expert_alpha_2

def gen_const_expert_v():
    const_expert_v = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
    return const_expert_v

def apply_mapping(tensor_list, map_list):
    mapped_tensors = []
    for t in tensor_list:
        length = t.shape[0]
        mask = (torch.arange(length) % 3) == 0
        map_tensor = torch.tensor(map_list, dtype=t.dtype)
        indices = t[mask].int()
        replacements = map_tensor[indices]
        new_t = t.clone()
        new_t[mask] = replacements
        mapped_tensors.append(new_t)

    return mapped_tensors
# endregion

# region golden值生成函数

def gen_dispatch_golden(
    local_moe_expert_num_list,
    x_list, expert_ids_list, x_active_mask_list, x_active_mask_list_ela_ori, padded_scales_list, table1_elastic, available_ranks, table2_elastic, elastic_info
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
            expanded_mask = torch.index_select(input=x_active_mask.flatten(),dim=0,index=sorted_col_idx)

         # 统计每个专家出现次数，即为接收到的token数
        expert_ids_sorted, _ = torch.sort(expert_ids.flatten())
        golden_moe_token = torch.bincount(expert_ids_sorted[expanded_mask] if is_2d_mask else expert_ids.flatten(), minlength=ep_local_moe_expert_num.sum().item())
        return expanded_x[expanded_mask] if is_2d_mask else expanded_x, sorted_col_idx[expanded_mask] if is_2d_mask else sorted_col_idx, golden_moe_token
    
    def shared_dispatch(x: torch.Tensor, x_active_mask: torch.Tensor, available_ranks, table2_elastic):
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
        if not is_elastic:
            for shared_card_id in range(shared_expert_rank_num):
                card_id_in_shared_group = shared_card_id % rank_num_per_shared_expert
                send_card_ids = slice(card_id_in_shared_group, ep_world_size, rank_num_per_shared_expert)
                local_expand_x = torch.cat(real_x_per_card[send_card_ids])
                shared_list[shared_card_id] = local_expand_x
                shared_tokens[shared_card_id] = local_expand_x.size(0)
        
        else:
            for shared_card_id in range(shared_expert_rank_num):
                card_id_in_shared_group = shared_card_id % rank_num_per_shared_expert
                send_card_ids_elastic = list(range(card_id_in_shared_group, len(available_ranks), rank_num_per_shared_expert))
                send_card_ids = [table2_elastic[i] for i in send_card_ids_elastic]
                local_expand_x = torch.cat([real_x_per_card[i] for i in send_card_ids])
                shared_list[shared_card_id] = local_expand_x
                shared_tokens[shared_card_id] = local_expand_x.size(0)



        return shared_list, shared_tokens
    
    def quant_process(
        ep_local_moe_expert_num: torch.Tensor, 
        shared_list: list, moe_x_segments: list, scales
    ):
        '''对各专家接收到的数据进行量化，并生成量化场景输出的scales'''
        def cal_quant(y: torch.Tensor, scales):
            '''量化各条数据，并计算每条数据对应的scale

            根据各专家接收到的数据y和scales计算得到量化后的数据和量化场景的dynamic_scales
            '''
            # dynamic_scales一维，长度等于y的行数
            y = y.to(torch.float32) * scales

            y_max, _ = y.abs().max(dim=-1, keepdim=True)
            dynamic_scales = y_max / 127.0
            dynamic_scales[dynamic_scales==0] = 1.0 # scales = 0的时候改成1，避免除0
            # y_int32 = (y / dynamic_scales).round().to(torch.int32)
            # y_half = y_int32.to(torch.float16)
            # y_int8 = y_half.trunc().to(torch.int8)
            y_int8 = (y / dynamic_scales).round().to(torch.int8)

            return y_int8, dynamic_scales.flatten()

        #每个专家的dynamic_scales，一个token一个scales
        shared_dynamic_scales = [torch.empty(0) for _ in range(len(shared_list))]
        moe_dynamic_scales = [torch.empty(0) for _ in range(len(moe_x_segments))]

        for i in range(shared_expert_rank_num):
            # shared_dynamic_scales[i] 为长度为 shared_list[i] 行数的一维张量，长度即每张共享专家卡上的token数
            shared_list[i], shared_dynamic_scales[i] = cal_quant(shared_list[i], scales[i,:] if is_dispatch_scales else 1)
        
        ### 对每个专家的moe_x进行量化（列表moe_x_segments中元素为各moe专家的expand_x）
        for moe_expert_id in range(len(moe_x_segments)):
            # moe_dynamic_scales[i] 长度为 golden_moe_token[i] 大小，即每个moe专家接收到的token数
            if moe_x_segments[moe_expert_id].size(0) > 0:
                moe_x_segments[moe_expert_id], moe_dynamic_scales[moe_expert_id] = cal_quant(
                    moe_x_segments[moe_expert_id],
                    scales[shared_expert_rank_num+moe_expert_id,:] if is_dispatch_scales else 1
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

    def gen_expand_idx(
        ep_local_moe_expert_num: torch.Tensor,
        expert_ids: torch.Tensor, x_active_mask: torch.Tensor, x_active_mask_ela_ori, available_ranks, table2_elastic
    ):
        """ 输出expand_x中各token对应的(rank_id, token_id, k_id)"""

        def gen_shared_expand_idx(x_active_mask: torch.Tensor, available_ranks, table2_elastic):
            shared_expand_idx = [[] for _ in range(shared_expert_rank_num)]       #存储每个共享专家收到的(r,t,k)
            if is_2d_mask:
                real_token_num_list = [bs for i in range(ep_world_size)]
                x_active_mask_per_card = x_active_mask.chunk(ep_world_size)

            else:
                real_token_num_list = x_active_mask.view(-1, bs).sum(dim=1).tolist()

            for shared_card_id in range(shared_expert_rank_num):
                shared_expert_index = shared_card_id // rank_num_per_shared_expert
                card_id_in_shared_group = shared_card_id % rank_num_per_shared_expert
                shared_idx_range = list(range(card_id_in_shared_group, len(available_ranks), rank_num_per_shared_expert)) \
                    if is_elastic else list(range(card_id_in_shared_group, ep_world_size, rank_num_per_shared_expert))
                for send_card_id in shared_idx_range:
                    if not is_2d_mask:
                        rtk_list = [
                            ele
                            for token_id in range(real_token_num_list[send_card_id if not is_elastic else table2_elastic[send_card_id]])
                            for ele in (send_card_id if not is_elastic else table2_elastic[send_card_id], token_id, k + shared_expert_index)
                        ]
                        shared_expand_idx[shared_card_id].extend(rtk_list)
                    else:

                        for token_id in range(bs):
                            if x_active_mask_per_card[send_card_id if not is_elastic else table2_elastic[send_card_id]][token_id].any():
                                rtk_list = [send_card_id if not is_elastic else table2_elastic[send_card_id], token_id, k + shared_expert_index]

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
            gen_shared_expand_idx(x_active_mask_ela_ori, available_ranks, table2_elastic)
            + gen_moe_expand_idx(ep_local_moe_expert_num, expert_ids, x_active_mask)
        )
        expand_idx = [torch.tensor(local_expand_idx) for local_expand_idx in expand_idx] # 格式转换，以便后续处理
        return expand_idx

    def gen_ep_recv_count(
        ep_local_moe_expert_num: torch.Tensor, 
        expert_ids: torch.Tensor, x_active_mask: torch.Tensor, x_active_mask_ela_ori, table1_elastic, available_ranks, table2_elastic, elastic_info
    ):
        """生成各专家的recv_count"""

        def gen_shared_expert_recv_count(x_active_mask: torch.Tensor, table1_elastic, available_ranks, table2_elastic):
            shared_expert_recv_count = torch.zeros(shared_expert_rank_num, ep_world_size, dtype=torch.int32)
            if is_2d_mask:
                # real_token_num_tensor = torch.tensor([bs] * ep_world_size)
                real_token_num_tensor = x_active_mask.any(dim=1).view(ep_world_size, bs).sum(dim=1)
            else:
                real_token_num_tensor = x_active_mask.view(-1, bs).sum(dim=1)

            
            if not is_elastic:
                for shared_card_id in range(shared_expert_rank_num):
                    card_id_in_shared_group = shared_card_id % rank_num_per_shared_expert
                    send_card_ids = slice(card_id_in_shared_group, ep_world_size, rank_num_per_shared_expert)
                    shared_expert_recv_count[shared_card_id][send_card_ids] = real_token_num_tensor[send_card_ids]
            else:
                for shared_card_id in range(shared_expert_rank_num):
                    card_id_in_shared_group = shared_card_id % rank_num_per_shared_expert
                    send_card_ids_elastic = list(range(card_id_in_shared_group, len(available_ranks), rank_num_per_shared_expert))
                    send_card_ids = [table2_elastic[i] for i in send_card_ids_elastic]
                    shared_expert_recv_count[shared_card_id][send_card_ids_elastic] = real_token_num_tensor[send_card_ids].to(torch.int32)

            # 对每张共享专家卡上的recv_count求累加和，确定发回给每张卡的数据偏移和数据量
            shared_expert_recv_count = shared_expert_recv_count.cumsum(dim = 1)

            shared_expert_recv_count_per_card = [local_ep_recv_count for local_ep_recv_count in shared_expert_recv_count]

            return shared_expert_recv_count_per_card

        def gen_moe_expert_recv_count(
            ep_local_moe_expert_num: torch.Tensor, 
            expert_ids: torch.Tensor, x_active_mask: torch.Tensor, available_ranks, table2_elastic, elastic_info
        ):
            x_active_mask_per_card = x_active_mask.chunk(ep_world_size)

            # 将全局的expert_ids切分为ep_world_size份，放入元组expert_ids_segments中。每份为一张卡上的expert_ids
            original_expert_ids_segments = expert_ids.chunk(ep_world_size)
            expert_ids_segments =  [
                original_expert_ids_segments[rank_id][x_active_mask_per_card[rank_id]] 
                for rank_id in range(ep_world_size)
            ]
            # 初始化一个二维张量，每行存储一个moe专家从ep域各卡收到的token数，大小为(moe_expert_num, ep_world_size)
            if not is_elastic:
                moe_expert_recv_count = torch.zeros(moe_expert_num, ep_world_size, dtype=torch.int32)
            else:
                moe_expert_recv_count = torch.zeros(moe_expert_num, len(available_ranks), dtype=torch.int32)
            # 遍历各卡
            if not is_elastic:
                for i in range(ep_world_size):
                    # 获取当前卡数据
                    segment = expert_ids_segments[i].flatten()
                    # 对当前卡发送的每个专家进行计数，并更新到result_tensor中
                    counts = torch.bincount(segment, minlength=moe_expert_num)
                    moe_expert_recv_count[:, i] = counts
            else:
                for i in range(len(available_ranks)):
                    # 获取当前卡数据
                    segment = expert_ids_segments[table2_elastic[i]].flatten()
                    # 对当前卡发送的每个专家进行计数，并更新到result_tensor中
                    counts = torch.bincount(segment, minlength=moe_expert_num)
                    moe_expert_recv_count[:, i] = counts
            
            # 得到各张卡的moe_expert_recv_count
            moe_expert_recv_count_per_card = split_tensor(moe_expert_recv_count, ep_local_moe_expert_num.tolist())
            # 对每张moe卡上的recv_count求累加和，确定发回给每张卡的数据偏移和数据量
            moe_expert_recv_count_per_card = [
                local_moe_expert_recv_count.flatten().cumsum(dim=0)
                for local_moe_expert_recv_count in moe_expert_recv_count_per_card
            ]

            return moe_expert_recv_count_per_card
        
        # 拼接共享专家和moe专家收到的token数
        shared_expert_recv_count_per_card = gen_shared_expert_recv_count(x_active_mask_ela_ori, table1_elastic, available_ranks, table2_elastic)
        moe_expert_recv_count_per_card = gen_moe_expert_recv_count(ep_local_moe_expert_num, expert_ids, x_active_mask, available_ranks, table2_elastic, elastic_info)
        ep_recv_count = shared_expert_recv_count_per_card + moe_expert_recv_count_per_card

        return ep_recv_count

    if is_zero_compute_expert:
        if len(x_active_mask_list_ela_ori[0].shape) == 1:
            x_active_mask_list_ela_ori = [original_tensor.unsqueeze(1).repeat(1, k) for original_tensor in x_active_mask_list_ela_ori]
    golden_expand_x_list, golden_expand_idx_list, golden_tokens_list, \
    golden_ep_recv_count_list, golden_dynamic_scales_list, middle_idx_list = \
        [], [], [], [], [], []
    for tp_id in range(tp_world_size):
        ep_local_moe_expert_num, x, expert_ids, scales, x_active_mask, x_active_mask_ela_ori = (
            local_moe_expert_num_list[tp_id],
            x_list[tp_id], expert_ids_list[tp_id], 
            padded_scales_list[tp_id], x_active_mask_list[tp_id], x_active_mask_list_ela_ori[tp_id]
        )

        # region ep域dispatch

        expanded_x, sorted_col_idx, golden_moe_token = moe_dispatch(ep_local_moe_expert_num, x, expert_ids, x_active_mask)
    
        shared_list, shared_tokens = shared_dispatch(x, x_active_mask_ela_ori, available_ranks, table2_elastic)
        
        golden_tokens = torch.cat((shared_tokens, golden_moe_token)) # 分到所有卡上的golden值

        moe_x_segments = split_tensor(expanded_x, golden_moe_token.tolist())
        golden_dynamic_scales = []
        if is_quant:
            shared_list, moe_x_segments, golden_dynamic_scales = quant_process(
                ep_local_moe_expert_num, 
                shared_list, moe_x_segments, scales,
            )
        rank_expert_id_bounds = torch.cat(
            [torch.zeros(1, dtype=torch.int32), ep_local_moe_expert_num.cumsum(dim=0)]     # 每个数字, 既是前一张卡的尾，又是后一张卡的头 
        ).tolist()
        moe_list = [
            torch.cat(moe_x_segments[rank_expert_id_bounds[moe_rank_id] : rank_expert_id_bounds[moe_rank_id+1]])
            for moe_rank_id in range(moe_rank_num)
        ]
        golden_expand_x = shared_list + moe_list

        golden_expand_idx = gen_expand_idx(ep_local_moe_expert_num, expert_ids, x_active_mask, x_active_mask_ela_ori, available_ranks, table2_elastic)
        golden_ep_recv_count = gen_ep_recv_count(ep_local_moe_expert_num, expert_ids, x_active_mask, x_active_mask_ela_ori, table1_elastic, available_ranks, table2_elastic, elastic_info)

        # endregion

        golden_expand_x_list.append(golden_expand_x)
        golden_expand_idx_list.append(golden_expand_idx)
        golden_tokens_list.append(golden_tokens)
        golden_ep_recv_count_list.append(golden_ep_recv_count)
        golden_dynamic_scales_list.append(golden_dynamic_scales)
        middle_idx_list.append(sorted_col_idx)
    
    return golden_expand_x_list, golden_expand_idx_list, golden_tokens_list, golden_ep_recv_count_list, golden_dynamic_scales_list, middle_idx_list

def gen_all_gather_golden(
    golden_expand_x_list, golden_tokens_list,
    golden_dynamic_scales_list, golden_ep_recv_count_list,
    tp_world_size
):
    tp_recv_counts = torch.stack([golden_tokens_list[0], golden_tokens_list[1]] * 2, dim = 1).view(-1)

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
    computation_dtype = torch.float32 if input_dtype == torch.bfloat16 else input_dtype
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
                local_golden_expand_x[ep_id][:local_real_token_num, :].to(computation_dtype)
                + peer_golden_expand_x[ep_id][peer_real_token_num:tp_real_token_num, :].to(computation_dtype)
            ).to(input_dtype)

            if ep_id < shared_expert_rank_num:
                shared_x_list[tp_id].append(local_real_expand_x)
            else:
                moe_x_list[tp_id].append(local_real_expand_x)

    moe_x_list = [torch.cat(moe_x_list[tp_id]) for tp_id in range(tp_world_size)]

    return shared_x_list, moe_x_list

def gen_combine_golden(
    middle_shared_x_list, middle_moe_x_list, middle_idx_list,
    expert_scales_list, x_active_mask_list, x_active_mask_list_ela_ori, shared_expert_x_list,
    ori_x_list, expert_ids_list,
    const_expert_alpha_1, const_expert_alpha_2, const_expert_v,
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
        middle_shared_x_per_card: List[torch.Tensor], x_active_mask: torch.Tensor,
        shared_scales_list=[1 for _ in range(shared_expert_num)]
    ):
        returned_real_x = [[torch.empty(0) for _ in range(ep_world_size)] for _ in range(shared_expert_num)]
        if is_2d_mask:
            real_token_num_list = x_active_mask.view(ep_world_size, bs, -1).any(dim=-1).sum(dim=1).tolist()
        else:
            real_token_num_list = x_active_mask.view(-1, bs).sum(dim=1).tolist()

        for shared_card_id in range(shared_expert_rank_num):
            shared_expert_index = shared_card_id // rank_num_per_shared_expert

            card_id_in_shared_group = shared_card_id % rank_num_per_shared_expert
            if not is_elastic:
                send_card_ids = slice(card_id_in_shared_group, ep_world_size, rank_num_per_shared_expert)

                local_middle_shared_x = middle_shared_x_per_card[shared_card_id]
                token_num_from_origin_card = real_token_num_list[send_card_ids]

                returned_real_x[shared_expert_index][send_card_ids] = split_tensor(local_middle_shared_x, token_num_from_origin_card)
            else:
                send_card_ids_elastic = list(range(card_id_in_shared_group, len(available_ranks), rank_num_per_shared_expert))
                send_card_ids = [table2_elastic[i] for i in send_card_ids_elastic]
                local_middle_shared_x = middle_shared_x_per_card[shared_card_id]
                token_num_from_origin_card = [real_token_num_list[i] for i in send_card_ids]
                split_result = split_tensor(local_middle_shared_x, token_num_from_origin_card)
                for idx, value in zip(send_card_ids, split_result):
                    returned_real_x[shared_expert_index][idx] = value
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
                if k_num.item() != 0:        # 如果一行全为0，或者没有共享专家
                    for share_num in range(shared_expert_num):
                        scaled_real_x_from_shared_expert[share_num][i] = returned_tensor_after_round_trip_list[share_num][j].to(torch.float32)
                    j += 1
            # scaled_x = torch.stack(scaled_x, dim=0)
        else:
            scaled_real_x_from_shared_expert = [
                returned_tensor.to(torch.float32) for returned_tensor in returned_tensor_after_round_trip_list
            ]

        return scaled_real_x_from_shared_expert

    def combine_x_for_super_moe(
        middle_moe_x: torch.Tensor, 
        middle_idx: torch.Tensor, 
        expert_scales: torch.Tensor,
        expert_ids,
        x_active_mask: torch.Tensor,
        ori_x,
        const_expert_alpha_1, 
        const_expert_alpha_2, 
        const_expert_v,
    ):
        '''将真实的expand_x还回去，得到真实的x'''

        middle_moe_x = middle_moe_x.view(len(middle_idx), h)

        # 因为middle_idx的值为0到len(middle_idx) - 1，排第i的值（即为值i）对应的索引即为值i对应的索引
        expandx_row_to_x_row = torch.argsort(middle_idx)
        # x_k即为x每一行变为k行的结果，上文expandx_row_to_x_row中的x即为x_k，为书写简便上文记为x
        x_k = middle_moe_x.index_select(dim=0, index=expandx_row_to_x_row)
        x_k_after_round_trip = comm_quant_round_trip(x_k)
        x_k_after_round_trip = x_k_after_round_trip.to(torch.float32)
        ori_x_32 = ori_x.to(torch.float32)
        const_expert_alpha_1_32 = const_expert_alpha_1.to(torch.float32)
        const_expert_alpha_2_32 = const_expert_alpha_2.to(torch.float32)
        const_expert_v_32 = const_expert_v.to(torch.float32)


        expand_indx = 0
        scaled_x = [torch.zeros(h, dtype=torch.float32) for _ in range(bs * ep_world_size)]
        for token_id_global in range(bs * ep_world_size):
            for top_id in range(k):
                if x_active_mask.dim() == 2:
                    flag_mask = x_active_mask[token_id_global][top_id]
                else:
                    flag_mask = x_active_mask[token_id_global]
                
                if not flag_mask:
                    continue

                if expert_ids[token_id_global][top_id] < moe_expert_num:
                    scaled_x[token_id_global] += x_k_after_round_trip[expand_indx] * expert_scales[token_id_global][top_id]
                    expand_indx += 1

                elif expert_ids[token_id_global][top_id] < moe_expert_num + zero_expert_num:
                    continue
                elif expert_ids[token_id_global][top_id] < moe_expert_num + zero_expert_num + copy_expert_num:
                    scaled_x[token_id_global] += ori_x_32[token_id_global] * expert_scales[token_id_global][top_id]
                else:
                    scaled_x[token_id_global] += (const_expert_alpha_1_32[expert_ids[token_id_global][top_id].item()-moe_expert_num-zero_expert_num-copy_expert_num] * ori_x_32[token_id_global] \
                        + const_expert_alpha_2_32[expert_ids[token_id_global][top_id].item()-moe_expert_num-zero_expert_num-copy_expert_num] * const_expert_v_32[expert_ids[token_id_global][top_id].item()\
                        -moe_expert_num-zero_expert_num-copy_expert_num]) * expert_scales[token_id_global][top_id]

        scaled_x = torch.stack(scaled_x, dim=0)

        return scaled_x[x_active_mask] if x_active_mask.dim() == 1 else scaled_x

    def combine_x_for_copy(ori_x, expert_ids, expert_scales, x_active_mask):
        ori_x_copy = ori_x.to(torch.float32)
        valid_mask = (expert_ids >= moe_expert_num + zero_expert_num) & (expert_ids < moe_expert_num + zero_expert_num + copy_expert_num) & x_active_mask

        ori_copy = ori_x_copy.unsqueeze(1) * expert_scales.unsqueeze(-1)


        sum_scales = ori_copy * valid_mask.unsqueeze(-1).expand(-1, -1, 7168)
        sum_scales = torch.sum(sum_scales, dim=1)
        return sum_scales
    

    def combine_x_for_const(ori_x, expert_ids, expert_scales, const_expert_alpha_1, const_expert_alpha_2, const_expert_v, x_active_mask):
        ori_x_const = ori_x.to(torch.float32)
        const_expert_alpha_1_fp = const_expert_alpha_1.to(torch.float32)
        const_expert_alpha_2_fp = const_expert_alpha_2.to(torch.float32)
        const_expert_v_fp = const_expert_v.to(torch.float32)
        expert_idx = expert_ids - moe_expert_num - zero_expert_num - copy_expert_num
        valid_mask = (expert_ids >= moe_expert_num + zero_expert_num + copy_expert_num) & x_active_mask
        expert_idx = torch.where(valid_mask, expert_idx, torch.zeros_like(expert_idx))

        
        alpha1 = const_expert_alpha_1_fp[expert_idx]
        alpha2 = const_expert_alpha_2_fp[expert_idx]
        v = const_expert_v_fp[expert_idx]
        
        ori_x_const = ori_x_const.unsqueeze(1)*alpha1.unsqueeze(-1)
        ori_x_const_2 = v*alpha2.unsqueeze(-1)
        ori_x_const = ori_x_const + ori_x_const_2
        

        ori_x_const = ori_x_const * expert_scales.unsqueeze(-1)
        
        ori_x_const = ori_x_const * valid_mask.unsqueeze(-1)
        
        ori_x_const = torch.sum(ori_x_const, dim=1)
        
        return ori_x_const

    if is_zero_compute_expert:
        if len(x_active_mask_list_ela_ori[0].shape) == 1:
            x_active_mask_list_ela_ori = [original_tensor.unsqueeze(1).repeat(1, k) for original_tensor in x_active_mask_list_ela_ori]
    golden_x_list = []
    for tp_id in range(tp_world_size):
        middle_shared_x_per_card, middle_moe_x, middle_idx, expert_scales, x_active_mask, x_active_mask_ela_ori, shared_expert_x_per_card, ori_x, \
        expert_ids, = (
            middle_shared_x_list[tp_id],
            middle_moe_x_list[tp_id],
            middle_idx_list[tp_id],
            expert_scales_list[tp_id],
            x_active_mask_list[tp_id],
            x_active_mask_list_ela_ori[tp_id],
            shared_expert_x_list[tp_id],
            ori_x_list[tp_id],
            expert_ids_list[tp_id],
        )
        
        golden_x_from_shared = (
            combine_x_for_shared(middle_shared_x_per_card, x_active_mask_ela_ori)
            if is_shared else []
        )
        golden_x_ep = combine_x_for_super_moe(
            middle_moe_x, 
            middle_idx,
            expert_scales,
            expert_ids,
            x_active_mask_ela_ori,
            ori_x,
            const_expert_alpha_1, 
            const_expert_alpha_2, 
            const_expert_v,
        ).to(torch.float32)

        if is_shared:
            golden_x_ep = sum(golden_x_from_shared, start=golden_x_ep)
        if is_shared_expert_x:
            shared_expert_x_per_card_vb = []
            if (is_variable_bs or is_mask) and is_2d_mask:
                for i in range(len(shared_expert_x_per_card)):
                    shared_expert_x_per_card_vb.append(F.pad(shared_expert_x_per_card[i], (0, 0, 0, bs - shared_expert_x_per_card[i].size(0))))
            if is_elastic and not is_2d_mask:
                shared_expert_x_per_card = [shared_expert_x_per[mask_shared_x[:len(shared_expert_x_per)]] for shared_expert_x_per, mask_shared_x in zip(shared_expert_x_per_card, chunk_tensor(x_active_mask_ela_ori, ep_world_size))]
            golden_x_ep += torch.cat(shared_expert_x_per_card if not shared_expert_x_per_card_vb else shared_expert_x_per_card_vb).to(torch.float32)

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

def compare_output(compared_output: ComparedOutput, golden_per_card, npu_per_card, compare_dtype, compare_per_rank, available_ranks):
    def compare(golden, npu, dtype, tensor_name, diffThd=0.001, pctThd=0.001):        
        np_output_npu = np.array(golden.cpu())
        np_output_golden = np.array(npu.cpu())

        max_diff_hd = 0.0
        if dtype == torch.int8:
            max_diff_hd = 1.0
        res_output = toolss.data_compare_np(np_output_golden, np_output_npu, diff_thd=diffThd, pct_thd=pctThd, max_diff_hd=max_diff_hd)
        print(f'[INFO] tensor {tensor_name} output测试的精度结果为：{res_output}')
        write_row_to_csv('moe_distribute.csv', [current_case_name, tensor_name, res_output])
        return res_output[0] == 'PASS'
    
    mark_str = compared_output.value.upper().replace('_', ' ')
    result = {}

    if not compare_per_rank:
        print(f"##################Compare {mark_str}##################")
        res = compare(
            torch.cat(golden_per_card).to(compare_dtype), 
            torch.cat(npu_per_card).to(compare_dtype), 
            compare_dtype,
            compared_output.value
        )
        result[compared_output.value] = res
    else:
        for i in (range(len(available_ranks)) if (is_elastic and server_num == 1) else range(server_index*rank_per_dev, (server_index+1)*rank_per_dev)):
            print(f"##################Compare {mark_str} in rank",i,"epId",i//tp_world_size,"tpId",i%tp_world_size,"##################")
            local_rank_id = i - server_index*rank_per_dev
            print(f'local_rank_id2374:{local_rank_id}')

            res = compare(
                golden_per_card[local_rank_id].to(compare_dtype), 
                npu_per_card[local_rank_id].to(compare_dtype), 
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

    available_ranks, server_elastic_index = gen_avalible_list()
    
    elastic_info, table1_elastic, table2_elastic = gen_elastic_info(available_ranks)
    elastic_rank = elastic_info[1]
    elastic_shared_rank = elastic_info[2]
    elastic_moe = elastic_info[3]

    print("=======================elastic_info=========================")
    print(f"{is_elastic=}")
    print(f"{elastic_rank=}")
    print(f"{elastic_shared_rank=}")
    print(f"{elastic_moe=}")

    # 生成各ep域的moe专家分布
    local_moe_expert_num_list = gen_local_moe_expert_num()
    # 生成各ep域的x
    x_list = gen_x()
    # 生成各ep域的expert_ids
    expert_ids_list = gen_expert_ids(elastic_moe)
    if is_elastic:
        expert_ids_list[0][expert_ids_list[0] >= elastic_moe] += (moe_expert_num - elastic_moe)
    # 生成各ep域的x_active_mask及各ep域有效token的个数
    x_active_mask_list = gen_x_active_mask()
    # 生成各ep域的shared_expert_x
    shared_expert_x_list = gen_shared_expert_x(x_active_mask_list)
    x_active_mask_list_ori = deepcopy(x_active_mask_list)
    if is_elastic:
        for i in [x for x in list(range(ep_world_size)) if x not in available_ranks]:
            x_active_mask_list[0][i*bs:(i+1)*bs] = False
    x_active_mask_list_ela_ori = deepcopy(x_active_mask_list)
    if is_zero_compute_expert:
        if not is_2d_mask:
            is_2d_mask = 1
            x_active_mask_list = [original_tensor.unsqueeze(1).repeat(1, k) for original_tensor in x_active_mask_list]
        for x_active_mask, expert_ids in zip(x_active_mask_list, expert_ids_list):
            x_active_mask[expert_ids>=moe_expert_num] = False
    


    const_expert_alpha_1, const_expert_alpha_2, const_expert_v = gen_const_expert_alpha_1(), gen_const_expert_alpha_2(), gen_const_expert_v()

    for i in x_active_mask_list:
        if not torch.any(i):
            print("当前不支持x_active_mask全为false!")
            exit(1)

    if is_2d_mask:
        mask_tp = [[local_mask.any(dim=1) for local_mask in chunk_tensor(x_active_mask, ep_world_size)] for x_active_mask in (x_active_mask_list_ela_ori if len(x_active_mask_list_ela_ori[0].shape)!=1 else [original_tensor.unsqueeze(1).repeat(1, k) for original_tensor in x_active_mask_list_ela_ori])]
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
        golden_expand_x_list, golden_expand_idx_list, golden_tokens_list,
        golden_ep_recv_count_list, golden_dynamic_scales_list, middle_idx_list
    ) = gen_dispatch_golden(
        local_moe_expert_num_list, 
        x_list, expert_ids_list, x_active_mask_list, x_active_mask_list_ela_ori, padded_scales_list, table1_elastic, available_ranks, table2_elastic, elastic_info
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
        # 使golden_tokens和golden_tp_recv_count以每张卡为单位
        golden_tokens = chunk_tensor(golden_tokens, world_size)
        golden_tp_recv_count = chunk_tensor(golden_tp_recv_count, world_size)
    else:
        # 生成tp=1时的同语义变量
        golden_expand_x, golden_ep_recv_count, golden_tokens, golden_dynamic_scales, golden_expand_idx = (
            golden_expand_x_list[0],
            golden_ep_recv_count_list[0],
            golden_tokens_list[0],
            golden_dynamic_scales_list[0],
            golden_expand_idx_list[0],
        )
        expert_num_list = [1] * shared_expert_rank_num + local_moe_expert_num_list[0].tolist()
        golden_tokens = split_tensor(golden_tokens, expert_num_list)
        golden_tp_recv_count = [torch.empty(size=(tp_world_size,), dtype=torch.int32) for _ in range(world_size)]
    
    # endregion

    # 生成各ep域的expert_scales
    expert_scales_list = gen_expert_scales()
    

    # region 生成combine算子golden值

    if is_quant:
        for tp_id in range(tp_world_size):
            golden_expand_x_list[tp_id] = [tensor.to(input_dtype) for tensor in golden_expand_x_list[tp_id]]

    if tp_world_size == 2:
        middle_shared_x_list, middle_moe_x_list = gen_reduce_scatter_golden(golden_expand_x_list, golden_tp_recv_count)
    else:
        expand_x = golden_expand_x_list[0]
        middle_shared_x_list, middle_moe_x_list = (
            [expand_x[:shared_expert_rank_num]], 
            [torch.cat(expand_x[shared_expert_rank_num:])],
        )

    golden_x = gen_combine_golden(
        middle_shared_x_list, middle_moe_x_list, middle_idx_list,
        expert_scales_list, x_active_mask_list, x_active_mask_list_ela_ori, shared_expert_x_list,
        x_list, expert_ids_list, const_expert_alpha_1, const_expert_alpha_2, const_expert_v,
    )
    golden_x_ori = deepcopy(golden_x)

    if is_elastic:
        no_elastic_shared_ranks_list = list(range(len(available_ranks)))
        if is_quant:
            src_dynamic_scales = [golden_dynamic_scales[i].clone() for i in no_elastic_shared_ranks_list]
        src_expand_idx = [golden_expand_idx[i] for i in no_elastic_shared_ranks_list]
        src_expand_x = [golden_expand_x[i] for i in no_elastic_shared_ranks_list]
        src_ep_recv = [golden_ep_recv_count[i] for i in no_elastic_shared_ranks_list]
        src_tokens = [golden_tokens[i].clone() for i in no_elastic_shared_ranks_list]
        

        for i in range(len(available_ranks)):
            dst_idx = available_ranks[i]
            if is_quant:
                golden_dynamic_scales[dst_idx] = src_dynamic_scales[i]
            golden_ep_recv_count[dst_idx] = src_ep_recv[i]
            golden_expand_idx[dst_idx] = src_expand_idx[i]
            golden_expand_x[dst_idx] = src_expand_x[i]
            golden_tokens[dst_idx] = src_tokens[i]
        
        golden_expand_idx = apply_mapping(golden_expand_idx, table1_elastic)
        for i, tensor in enumerate(golden_ep_recv_count):
            if tensor.size(0) < ep_world_size * local_moe_expert_num:
                pad_size = ep_world_size * local_moe_expert_num - tensor.size(0)
                tensor = F.pad(tensor, (0, pad_size), "constant", 0)
                golden_ep_recv_count[i] = tensor
        golden_expand_x_list = [golden_expand_x]
        

    # exit(0)

            
    # endregion
    
    print("Generate golden success.")
    # exit(0)
    server_index_el = server_index if not is_elastic or (server_index<server_elastic_index) else (server_index-1)
    server_ranks = work_server_ranks(server_index)
    server_ranks_dis = work_server_ranks(server_index_el)
    if is_dispatch:
        [npu_expand_x, npu_dynamic_scales, npu_expand_idx, npu_tokens, npu_eprecv_count, npu_tprecv_count] = gen_npu(
            run_dispatch_npu, available_ranks,
            x_list=[chunk_tensor(x, ep_world_size) for x in x_list],
            expert_ids_list=[chunk_tensor(expert_ids, ep_world_size) for expert_ids in expert_ids_list],
            x_active_mask_list=[chunk_tensor(x_active_mask, ep_world_size) for x_active_mask in (x_active_mask_list if not (is_elastic or is_zero_compute_expert) else x_active_mask_list_ori)],
            scales_list=scales_list,
            elastic_info=elastic_info,
        )
        golden_tokens_el = deepcopy(golden_tokens)
        golden_ep_recv_count_el = deepcopy(golden_ep_recv_count)
        golden_expand_x_el = deepcopy(golden_expand_x)
        golden_dynamic_scales_el = deepcopy(golden_dynamic_scales)
        golden_expand_idx_el = deepcopy(golden_expand_idx)
        if is_elastic:
            golden_tokens_el = [golden_tokens_el[i] for i in available_ranks]
            golden_ep_recv_count_el = [golden_ep_recv_count_el[i] for i in available_ranks]
            golden_expand_x_el = [golden_expand_x_el[i] for i in available_ranks]
            if is_quant:
                golden_dynamic_scales_el = [golden_dynamic_scales_el[i] for i in available_ranks]
            golden_expand_idx_el = [golden_expand_idx_el[i] for i in available_ranks]
            npu_tokens = [npu_tokens[i] for i in available_ranks] if server_num == 1 else npu_tokens
            npu_eprecv_count = [npu_eprecv_count[i] for i in available_ranks] if server_num == 1 else npu_eprecv_count
            npu_expand_x = [npu_expand_x[i] for i in available_ranks] if server_num == 1 else npu_expand_x
            if is_quant:
                npu_dynamic_scales = [npu_dynamic_scales[i] for i in available_ranks] if server_num == 1 else npu_dynamic_scales
            npu_expand_idx = [npu_expand_idx[i] for i in available_ranks] if server_num == 1 else npu_expand_idx
            for idx in range(shared_expert_rank_num):
                golden_ep_recv_count_el[idx] = golden_ep_recv_count_el[idx][:len(available_ranks)]
                if server_num == 1:
                    npu_eprecv_count[idx] = npu_eprecv_count[idx][:len(available_ranks)]
                    npu_tokens[idx] = npu_tokens[idx][0:1]
                
            for idx in range(shared_expert_rank_num, len(available_ranks)):
                golden_ep_recv_count_el[idx] = golden_ep_recv_count_el[idx][:len(available_ranks) * local_moe_expert_num]
                if server_num == 1:
                    npu_eprecv_count[idx] = npu_eprecv_count[idx][:len(available_ranks) * local_moe_expert_num]
            if server_num > 1:
                if server_index_el < shared_expert_rank_num // rank_per_dev:
                    npu_eprecv_count = [tensor_recv[:len(available_ranks)] for tensor_recv in npu_eprecv_count]
                    npu_tokens = [t[0:1] for t in npu_tokens]
                else:
                    npu_eprecv_count = [tensor_recv[:len(available_ranks) * local_moe_expert_num] for tensor_recv in npu_eprecv_count]
        print("Generate dispatch npu success.")

        if is_performance and graph_type == 0:
            dispatch_performance = get_dispatch_performance()

        # 精度对比
        print("##################Start Dispatch Compare##################")
        if is_elastic and server_num > 1 and server_index == server_elastic_index:
            time.sleep(10)
            pass
        else:
            result_dispatch = {}
            
            # 比较 ep recv count
            res = compare_output(
                ComparedOutput.EP_RECV_COUNTS,
                golden_ep_recv_count_el[server_ranks_dis],
                npu_eprecv_count,
                torch.int32,
                False,
                available_ranks,
            )
            result_dispatch.update(res)

            # 比较 tp recv count
            if tp_world_size != 1:
                res = compare_output(
                    ComparedOutput.TP_RECV_COUNTS,
                    golden_tp_recv_count[server_ranks_dis],
                    npu_tprecv_count,
                    torch.int32,
                    False,
                    available_ranks,
                )
                result_dispatch.update(res)
            
            # 比较 expand x
            res = compare_output(
                ComparedOutput.EXPAND_X,
                golden_expand_x_el[server_ranks_dis],
                npu_expand_x,
                torch.int8 if is_quant else torch.float32,
                True,
                available_ranks,
            )
            result_dispatch.update(res)
            
            # 比较 dynamic scales
            if is_quant:
                res = compare_output(
                    ComparedOutput.DYNAMIC_SCALES,
                    golden_dynamic_scales_el[server_ranks_dis],
                    npu_dynamic_scales,
                    torch.float32,
                    False,
                    available_ranks,
                )
                result_dispatch.update(res)
            
            # 比较 expand idx
            res = compare_output(
                ComparedOutput.EXPAND_IDX,
                golden_expand_idx_el[server_ranks_dis],
                npu_expand_idx,
                torch.int32,
                True,
                available_ranks,
            )
            result_dispatch.update(res)

            # 比较 expert token nums
            res = compare_output(
                ComparedOutput.EXPERT_TOKEN_NUMS,
                golden_tokens_el[server_ranks_dis],
                npu_tokens,
                torch.int64,
                False,
                available_ranks,
            )
            result_dispatch.update(res)
    if is_2d_mask:
        mask_per_card_server = deepcopy(mask_per_card)
        if is_elastic:
            mask_per_card = [mask_per_card[i] for i in available_ranks] if server_num == 1 else mask_per_card
        
    if is_combine:
        [npu_x] = gen_npu(
            run_combine_npu, available_ranks,
            x_list=[chunk_tensor(x, ep_world_size) for x in x_list],
            scales_list=scales_list,
            expand_x_list=golden_expand_x_list,
            expert_ids_list=[chunk_tensor(expert_ids, ep_world_size) for expert_ids in expert_ids_list],
            expand_idx_list=golden_expand_idx,
            ep_send_count_list=golden_ep_recv_count,
            tp_send_count_list=golden_tp_recv_count,
            x_active_mask_list=[chunk_tensor(x_active_mask, ep_world_size) for x_active_mask in (x_active_mask_list if not (is_elastic or is_zero_compute_expert) else x_active_mask_list_ori)],
            expert_scales_list=[chunk_tensor(expert_scales, ep_world_size) for expert_scales in expert_scales_list],
            shared_expert_x_list=shared_expert_x_list,
            elastic_info=elastic_info,
            ori_x_list = [chunk_tensor(x, ep_world_size) for x in x_list], 
            const_expert_alpha_1 = const_expert_alpha_1, 
            const_expert_alpha_2 = const_expert_alpha_2, 
            const_expert_v = const_expert_v,
        )

        print("Generate combine npu success.")
        
        if is_performance and graph_type == 0:
            combine_performance = get_combine_performance()

        print("##################Start Combine Compare##################")
        if is_elastic and server_num > 1 and server_index == server_elastic_index:
            pass
        else:
            result_combine = {}
            # 比较x
            
            if is_elastic:
                golden_x = golden_x_ori
                if server_num == 1:
                    npu_x = [npu_x[i] for i in available_ranks]
                    golden_x = [golden_x[i] for i in available_ranks]
            if is_variable_bs and is_zero_compute_expert:
                npu_x = pad_tensor(npu_x)
            if is_2d_mask:
                npu_x = [npu_x_per_card[mask_all_per_card] for npu_x_per_card, mask_all_per_card in zip(npu_x, mask_per_card if server_num == 1 else mask_per_card_server[work_server_ranks(server_index)])]
                golden_x_rank = [golden_x_per_card[mask_all_per_card] for golden_x_per_card, mask_all_per_card in zip(golden_x, mask_per_card)]
            

            
            res = compare_output(
                ComparedOutput.X,
                golden_x_rank[server_ranks] if is_2d_mask else golden_x[server_ranks],
                npu_x,
                torch.float32,
                True,
                available_ranks
            )
            result_combine.update(res)

    if is_cascade:
        [npu_x] = gen_npu(
            run_cascade_npu, available_ranks,
            x_list=[chunk_tensor(x, ep_world_size) for x in x_list],
            expert_ids_list=[chunk_tensor(expert_ids, ep_world_size) for expert_ids in expert_ids_list],
            x_active_mask_list=[chunk_tensor(x_active_mask, ep_world_size) for x_active_mask in (x_active_mask_list if not (is_elastic or is_zero_compute_expert) else x_active_mask_list_ori)],
            scales_list=scales_list,
            expert_scales_list=[chunk_tensor(expert_scales, ep_world_size) for expert_scales in expert_scales_list], 
            golden_expand_x_list=golden_expand_x_list, 
            shared_expert_x_list=shared_expert_x_list,
            elastic_info=elastic_info,
            ori_x_list = [chunk_tensor(x, ep_world_size) for x in x_list], 
            const_expert_alpha_1 = const_expert_alpha_1, 
            const_expert_alpha_2 = const_expert_alpha_2, 
            const_expert_v = const_expert_v,
        )

        

        if is_performance:
            cascade_performance = get_cascade_performance()

        print("Generate cascade npu success.")

        print("##################Start Cascade Compare##################")
        if is_elastic and server_num > 1 and server_index == server_elastic_index:
            pass
        else:
            result_cascade = {}
            # 比较 x
            if is_elastic:
                golden_x = golden_x_ori
                if server_num == 1:
                    npu_x = [npu_x[i] for i in available_ranks]
                    golden_x = [golden_x[i] for i in available_ranks]
            if is_variable_bs and is_zero_compute_expert:
                npu_x = pad_tensor(npu_x)
            
            if is_2d_mask:
                npu_x = [npu_x_per_card[mask_all_per_card] for npu_x_per_card, mask_all_per_card in zip(npu_x, mask_per_card if server_num == 1 else mask_per_card_server[work_server_ranks(server_index)])]
                golden_x_rank = [golden_x_per_card[mask_all_per_card] for golden_x_per_card, mask_all_per_card in zip(golden_x, mask_per_card)]
            res = compare_output(
                ComparedOutput.X,
                golden_x_rank[server_ranks] if is_2d_mask else golden_x[server_ranks],
                npu_x,
                torch.float32,
                True,
                available_ranks
            )
            result_cascade.update(res)

    if is_elastic and server_num > 1 and server_index == server_elastic_index:
        pass
    else:
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
        
        if is_cascade:
            print("\n##################cascade result##################")
            print("result: ", result_cascade)
            all_result = all(result_cascade.values())
            if all_result:
                print("cascade test Success.")
            else:
                print("cascade test Failed.")
            
            if is_performance:
                print_cascade_performance(cascade_performance)
